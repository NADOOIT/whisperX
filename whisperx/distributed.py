"""Distributed processing system for WhisperX."""

import os
import torch
import numpy as np
import socket
import json
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import zmq
from pathlib import Path
import logging
from .optimizations import OptimizationConfig, MemoryManager

@dataclass
class NodeConfig:
    """Configuration for a distributed processing node."""
    node_id: str
    host: str
    port: int
    device: str = "mps"  # or "cuda" for NVIDIA GPUs
    compute_type: str = "float16"
    model_size: str = "base"
    chunk_size: int = 30
    batch_size: int = 8
    priority: int = 1  # Higher number = higher priority

class DistributedNode:
    """Node in the distributed processing network."""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.context = zmq.Context()
        self.memory_manager = MemoryManager(self.device)
        self.running = False
        self.task_queue = queue.PriorityQueue()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the node."""
        log_dir = Path.home() / ".whisperx" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"node_{self.config.node_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"Node-{self.config.node_id}")
    
    def start(self):
        """Start the processing node."""
        self.running = True
        
        # Start main processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()
        
        # Start communication threads
        self.receiver_thread = threading.Thread(target=self._receive_loop)
        self.sender_thread = threading.Thread(target=self._send_loop)
        self.receiver_thread.start()
        self.sender_thread.start()
        
        self.logger.info(f"Node {self.config.node_id} started on {self.config.host}:{self.config.port}")
    
    def stop(self):
        """Stop the processing node."""
        self.running = False
        self.process_thread.join()
        self.receiver_thread.join()
        self.sender_thread.join()
        self.context.term()
        self.logger.info(f"Node {self.config.node_id} stopped")
    
    def _process_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get next task from queue
                priority, task = self.task_queue.get(timeout=1)
                
                # Process task
                with self.memory_manager.track_memory():
                    result = self._process_task(task)
                
                # Send result back
                self._send_result(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing task: {str(e)}")
                self.memory_manager.cleanup()
    
    def _receive_loop(self):
        """Receive incoming tasks."""
        socket = self.context.socket(zmq.PULL)
        socket.bind(f"tcp://{self.config.host}:{self.config.port}")
        
        while self.running:
            try:
                # Receive task
                message = socket.recv_json()
                task_type = message.get("type")
                
                if task_type == "process":
                    # Add task to queue with priority
                    priority = message.get("priority", 0)
                    self.task_queue.put((priority, message))
                elif task_type == "status":
                    # Respond with node status
                    self._send_status()
                
            except Exception as e:
                self.logger.error(f"Error receiving task: {str(e)}")
    
    def _send_loop(self):
        """Send results back to coordinator."""
        socket = self.context.socket(zmq.PUSH)
        socket.connect(f"tcp://{self.config.host}:{self.config.port + 1}")
        
        while self.running:
            try:
                # Send any pending results
                result = self.result_queue.get(timeout=1)
                socket.send_json(result)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error sending result: {str(e)}")

class DistributedCoordinator:
    """Coordinates distributed processing across nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, NodeConfig] = {}
        self.context = zmq.Context()
        self.result_queue = queue.Queue()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the coordinator."""
        log_dir = Path.home() / ".whisperx" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "coordinator.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("Coordinator")
    
    def add_node(self, config: NodeConfig):
        """Add a processing node."""
        self.nodes[config.node_id] = config
        self.logger.info(f"Added node {config.node_id} at {config.host}:{config.port}")
    
    def remove_node(self, node_id: str):
        """Remove a processing node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Removed node {node_id}")
    
    def process_audio(self, audio: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Process audio using distributed nodes."""
        try:
            # Split audio into chunks
            chunks = self._split_audio(audio)
            
            # Distribute chunks to nodes
            futures = []
            for i, chunk in enumerate(chunks):
                # Select node for this chunk
                node = self._select_node()
                
                # Create task
                task = {
                    "type": "process",
                    "chunk_id": i,
                    "audio": chunk.tolist(),
                    "priority": kwargs.get("priority", 1),
                    "params": kwargs
                }
                
                # Send task to node
                future = self._send_task(node, task)
                futures.append(future)
            
            # Collect and merge results
            results = self._collect_results(futures)
            return self._merge_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in distributed processing: {str(e)}")
            raise
    
    def _split_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into chunks for distributed processing."""
        chunk_size = 30 * 16000  # 30 seconds at 16kHz
        overlap = int(chunk_size * 0.1)  # 10% overlap
        
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _select_node(self) -> NodeConfig:
        """Select the best node for processing."""
        # Get node status
        node_status = self._get_nodes_status()
        
        # Select node based on:
        # 1. Available memory
        # 2. Current load
        # 3. Priority
        best_node = None
        best_score = float('-inf')
        
        for node_id, status in node_status.items():
            node = self.nodes[node_id]
            
            # Calculate node score
            memory_score = status.get("available_memory", 0) / status.get("total_memory", 1)
            load_score = 1 - (status.get("current_tasks", 0) / status.get("max_tasks", 1))
            priority_score = node.priority / max(n.priority for n in self.nodes.values())
            
            score = memory_score * 0.4 + load_score * 0.4 + priority_score * 0.2
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _send_task(self, node: NodeConfig, task: Dict[str, Any]) -> Any:
        """Send task to a node."""
        socket = self.context.socket(zmq.PUSH)
        socket.connect(f"tcp://{node.host}:{node.port}")
        socket.send_json(task)
        return task["chunk_id"]
    
    def _collect_results(self, futures: List[Any]) -> List[Dict[str, Any]]:
        """Collect results from all nodes."""
        results = [None] * len(futures)
        socket = self.context.socket(zmq.PULL)
        socket.bind("tcp://*:5558")  # Result collection port
        
        # Collect results
        for _ in range(len(futures)):
            result = socket.recv_json()
            chunk_id = result["chunk_id"]
            results[chunk_id] = result
        
        return results
    
    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from all chunks."""
        merged = {
            "text": "",
            "segments": []
        }
        
        # Merge text and segments
        time_offset = 0
        for result in results:
            # Add text
            merged["text"] += " " + result.get("text", "")
            
            # Adjust and add segments
            segments = result.get("segments", [])
            for segment in segments:
                segment["start"] += time_offset
                segment["end"] += time_offset
                merged["segments"].append(segment)
            
            # Update time offset
            if segments:
                time_offset = segments[-1]["end"]
        
        merged["text"] = merged["text"].strip()
        return merged
    
    def _get_nodes_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all nodes."""
        status = {}
        for node_id, node in self.nodes.items():
            try:
                socket = self.context.socket(zmq.REQ)
                socket.connect(f"tcp://{node.host}:{node.port}")
                socket.send_json({"type": "status"})
                status[node_id] = socket.recv_json()
            except Exception as e:
                self.logger.warning(f"Failed to get status from node {node_id}: {str(e)}")
                status[node_id] = {"status": "unavailable"}
        return status

def discover_nodes(network: str = "192.168.1.0/24", port_range: Tuple[int, int] = (5555, 5565)) -> List[NodeConfig]:
    """Discover available processing nodes on the network."""
    discovered = []
    
    # Scan network
    for ip in _scan_network(network):
        # Check ports
        for port in range(port_range[0], port_range[1]):
            try:
                # Try to connect
                socket = zmq.Context().socket(zmq.REQ)
                socket.connect(f"tcp://{ip}:{port}")
                socket.send_json({"type": "info"})
                info = socket.recv_json()
                
                # Create node config
                config = NodeConfig(
                    node_id=info["node_id"],
                    host=ip,
                    port=port,
                    device=info["device"],
                    compute_type=info["compute_type"],
                    model_size=info["model_size"],
                    priority=info["priority"]
                )
                discovered.append(config)
                
            except Exception:
                continue
    
    return discovered

def _scan_network(network: str) -> List[str]:
    """Scan network for available hosts."""
    import ipaddress
    network = ipaddress.ip_network(network)
    
    available = []
    for ip in network.hosts():
        try:
            if os.system(f"ping -c 1 -W 1 {ip} >/dev/null 2>&1") == 0:
                available.append(str(ip))
        except:
            continue
    
    return available
