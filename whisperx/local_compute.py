"""Local multi-compute processing for WhisperX."""

import torch
import coremltools as ct
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import threading
import queue
import logging
from enum import Enum
import psutil
import os

class LocalComputeType(Enum):
    CPU = "cpu"
    METAL = "metal"
    NEURAL_ENGINE = "neural_engine"
    ANE = "ane"  # Apple Neural Engine alias

@dataclass
class LocalDevice:
    """Represents a local compute device."""
    type: LocalComputeType
    name: str
    memory: int
    compute_units: int
    supported_ops: List[str]
    current_load: float = 0.0
    batch_size: int = 1
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score for task allocation."""
        base_score = {
            LocalComputeType.NEURAL_ENGINE: 100,  # Best for inference
            LocalComputeType.ANE: 100,           # Alias for Neural Engine
            LocalComputeType.METAL: 85,          # Great for parallel processing
            LocalComputeType.CPU: 60             # Good for preprocessing
        }[self.type]
        
        # Adjust for current device load
        load_penalty = self.current_load / 100
        return base_score * (1 - load_penalty)

class LocalProcessor:
    """Manages processing on a specific local device."""
    
    def __init__(self, device: LocalDevice):
        self.device = device
        self.model = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize device-specific optimizations."""
        if self.device.type in [LocalComputeType.NEURAL_ENGINE, LocalComputeType.ANE]:
            self._setup_neural_engine()
        elif self.device.type == LocalComputeType.METAL:
            self._setup_metal()
        else:
            self._setup_cpu()
    
    def _setup_neural_engine(self):
        """Setup Neural Engine with CoreML optimization."""
        try:
            # Convert model for Neural Engine
            mlmodel = ct.convert(
                "whisper-small",
                source="pytorch",
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.NEURAL_ENGINE_ONLY,
                minimum_deployment_target=ct.target.iOS15
            )
            
            # Enable ANE-specific optimizations
            spec = mlmodel.get_spec()
            for nn_spec in spec.neuralNetwork:
                nn_spec.computeUnits = ct.proto.ComputeUnit.NEURAL_ENGINE_ONLY
            
            self.model = ct.models.MLModel(spec)
            logging.info("Neural Engine setup complete")
        except Exception as e:
            logging.error(f"Neural Engine setup failed: {e}")
            self.device.type = LocalComputeType.METAL  # Fallback to Metal
            self._setup_metal()
    
    def _setup_metal(self):
        """Setup Metal with MPS optimization."""
        try:
            if torch.backends.mps.is_available():
                self.device_obj = torch.device("mps")
                # Load and optimize model for Metal
                self.model = torch.jit.load("path/to/whisper_model.pt")
                self.model = self.model.to(self.device_obj)
                
                # Enable Metal Performance Shaders
                torch.mps.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory
                logging.info("Metal setup complete")
            else:
                raise RuntimeError("Metal not available")
        except Exception as e:
            logging.error(f"Metal setup failed: {e}")
            self.device.type = LocalComputeType.CPU  # Fallback to CPU
            self._setup_cpu()
    
    def _setup_cpu(self):
        """Setup CPU with threading optimization."""
        try:
            # Optimize CPU threads
            torch.set_num_threads(self.device.compute_units)
            torch.set_num_interop_threads(min(4, self.device.compute_units))
            
            # Load model for CPU
            self.model = torch.jit.load("path/to/whisper_model.pt")
            self.model = self.model.to("cpu")
            
            # Enable MKL optimizations if available
            os.environ["MKL_NUM_THREADS"] = str(self.device.compute_units)
            logging.info("CPU setup complete")
        except Exception as e:
            logging.error(f"CPU setup failed: {e}")
    
    def start(self):
        """Start processing thread."""
        self.running = True
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

class LocalComputeManager:
    """Manages local compute resources for optimal processing."""
    
    def __init__(self):
        self.devices: List[LocalDevice] = []
        self.processors: Dict[LocalComputeType, LocalProcessor] = {}
        self._discover_local_devices()
        self._initialize_processors()
        
    def _discover_local_devices(self):
        """Discover available local compute devices."""
        # Check for Neural Engine
        try:
            if ct.ComputeUnit.NEURAL_ENGINE_ONLY:
                self.devices.append(LocalDevice(
                    type=LocalComputeType.NEURAL_ENGINE,
                    name="Apple Neural Engine",
                    memory=0,  # Managed by system
                    compute_units=1,
                    supported_ops=["inference"],
                    batch_size=4  # ANE works well with small batches
                ))
        except Exception:
            logging.info("Neural Engine not available")
        
        # Check for Metal
        if torch.backends.mps.is_available():
            self.devices.append(LocalDevice(
                type=LocalComputeType.METAL,
                name="Metal GPU",
                memory=int(torch.mps.current_allocated_memory()),
                compute_units=1,
                supported_ops=["inference", "preprocessing"],
                batch_size=8  # Metal handles larger batches well
            ))
        
        # Add CPU
        cpu_count = psutil.cpu_count(logical=True)
        self.devices.append(LocalDevice(
            type=LocalComputeType.CPU,
            name=f"CPU ({cpu_count} threads)",
            memory=psutil.virtual_memory().total,
            compute_units=cpu_count,
            supported_ops=["preprocessing", "postprocessing"],
            batch_size=2  # CPU works better with smaller batches
        ))
    
    def _initialize_processors(self):
        """Initialize processors for each device."""
        for device in self.devices:
            processor = LocalProcessor(device)
            self.processors[device.type] = processor
            processor.start()
    
    def process_audio(self, audio: np.ndarray, chunk_duration: float = 30.0) -> Dict:
        """Process audio using all available local compute resources."""
        chunks = self._split_audio(audio, chunk_duration)
        results = []
        
        # Distribute chunks based on device efficiency
        for chunk in chunks:
            best_device = self._get_best_device(chunk)
            processor = self.processors[best_device.type]
            processor.input_queue.put(chunk)
        
        # Collect and merge results
        for _ in range(len(chunks)):
            for processor in self.processors.values():
                if not processor.output_queue.empty():
                    result = processor.output_queue.get()
                    results.append(result)
        
        return self._merge_results(results)
    
    def _split_audio(self, audio: np.ndarray, chunk_duration: float) -> List[np.ndarray]:
        """Split audio into optimal chunks for processing."""
        samples_per_chunk = int(chunk_duration * 16000)  # Assuming 16kHz sample rate
        return np.array_split(audio, max(1, len(audio) // samples_per_chunk))
    
    def _get_best_device(self, chunk: np.ndarray) -> LocalComputeType:
        """Get best device for processing based on current efficiency."""
        scores = [(device, device.get_efficiency_score()) for device in self.devices]
        return max(scores, key=lambda x: x[1])[0].type
    
    def _merge_results(self, results: List[Dict]) -> Dict:
        """Merge results from different devices."""
        merged = {
            "text": "",
            "segments": [],
            "device_usage": {}
        }
        
        # Sort by timestamp and merge
        sorted_results = sorted(results, key=lambda x: x.get("start_time", 0))
        for result in sorted_results:
            merged["text"] += result.get("text", "") + " "
            if "segments" in result:
                merged["segments"].extend(result["segments"])
        
        # Add device usage statistics
        for device in self.devices:
            merged["device_usage"][device.name] = {
                "load": device.current_load,
                "processed_chunks": len([r for r in results if r.get("device") == device.type.value])
            }
        
        return merged
    
    def shutdown(self):
        """Cleanup and shutdown all processors."""
        for processor in self.processors.values():
            processor.stop()
        
        # Clear GPU memory if used
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
