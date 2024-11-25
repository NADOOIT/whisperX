"""Auto-discovery module for WhisperX distributed processing."""

import socket
import json
import platform
import torch
import threading
import time
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import logging

@dataclass
class DeviceCapabilities:
    """Device capabilities information."""
    platform: str  # 'windows', 'darwin', 'linux'
    device_type: str  # 'cuda', 'mps', 'cpu'
    compute_units: int
    memory_available: int
    cuda_version: Optional[str] = None
    metal_version: Optional[str] = None

@dataclass
class NodeInfo:
    """Information about a processing node."""
    node_id: str
    host: str
    port: int
    capabilities: DeviceCapabilities
    status: str = "available"  # available, busy, offline
    last_heartbeat: float = 0.0

class WhisperXDiscovery:
    """Auto-discovery service for WhisperX nodes."""
    
    SERVICE_TYPE = "_whisperx._tcp.local."
    
    def __init__(self, node_port: int = 5555):
        self.zeroconf = Zeroconf()
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_port = node_port
        self.local_info = self._get_local_info()
        self.browser = None
        self.heartbeat_thread = None
        self.running = False
        
    def _get_local_info(self) -> NodeInfo:
        """Get local device capabilities."""
        platform_name = platform.system().lower()
        
        # Detect available compute devices
        if torch.cuda.is_available():
            device_type = "cuda"
            compute_units = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            metal_version = None
        elif platform_name == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            compute_units = 1  # Metal GPU
            cuda_version = None
            metal_version = "2.0"  # Example version
        else:
            device_type = "cpu"
            compute_units = os.cpu_count()
            cuda_version = None
            metal_version = None
        
        capabilities = DeviceCapabilities(
            platform=platform_name,
            device_type=device_type,
            compute_units=compute_units,
            memory_available=torch.cuda.get_device_properties(0).total_memory if device_type == "cuda" else 0,
            cuda_version=cuda_version,
            metal_version=metal_version
        )
        
        return NodeInfo(
            node_id=f"{platform.node()}_{device_type}",
            host=socket.gethostbyname(socket.gethostname()),
            port=self.node_port,
            capabilities=capabilities
        )
    
    def start_advertising(self):
        """Start advertising this node's services."""
        info = ServiceInfo(
            self.SERVICE_TYPE,
            f"{self.local_info.node_id}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(self.local_info.host)],
            port=self.local_info.port,
            properties=asdict(self.local_info.capabilities)
        )
        self.zeroconf.register_service(info)
        self.running = True
        self._start_heartbeat()
        
    def start_discovery(self):
        """Start discovering other nodes."""
        self.browser = ServiceBrowser(self.zeroconf, self.SERVICE_TYPE, self)
        
    def remove_service(self, zeroconf: Zeroconf, type_: str, name: str):
        """Handle removed services."""
        node_id = name.replace(f".{self.SERVICE_TYPE}", "")
        if node_id in self.nodes:
            self.nodes[node_id].status = "offline"
            logging.info(f"Node {node_id} went offline")
    
    def add_service(self, zeroconf: Zeroconf, type_: str, name: str):
        """Handle discovered services."""
        info = zeroconf.get_service_info(type_, name)
        if info:
            node_id = name.replace(f".{self.SERVICE_TYPE}", "")
            host = socket.inet_ntoa(info.addresses[0])
            capabilities = DeviceCapabilities(**info.properties)
            
            node = NodeInfo(
                node_id=node_id,
                host=host,
                port=info.port,
                capabilities=capabilities
            )
            
            self.nodes[node_id] = node
            logging.info(f"Discovered node: {node_id} ({host})")
    
    def _start_heartbeat(self):
        """Start heartbeat thread."""
        def heartbeat():
            while self.running:
                self.local_info.last_heartbeat = time.time()
                time.sleep(1)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def stop(self):
        """Stop discovery and advertising."""
        self.running = False
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()
        
    def get_available_nodes(self) -> List[NodeInfo]:
        """Get list of available nodes."""
        return [node for node in self.nodes.values() if node.status == "available"]
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def update_node_status(self, node_id: str, status: str):
        """Update node status."""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            self.nodes[node_id].last_heartbeat = time.time()
