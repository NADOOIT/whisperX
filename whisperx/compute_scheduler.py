"""Multi-compute scheduler for WhisperX."""

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

class ComputeType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NEURAL_ENGINE = "neural_engine"

@dataclass
class ComputeDevice:
    """Represents a compute device with its capabilities."""
    type: ComputeType
    name: str
    memory: int
    compute_units: int
    supported_precisions: List[str]
    current_load: float = 0.0
    
    def get_performance_score(self) -> float:
        """Calculate performance score based on device capabilities and current load."""
        base_score = {
            ComputeType.NEURAL_ENGINE: 100,  # Neural Engine is fastest for supported ops
            ComputeType.GPU: 80,            # GPU is generally fast
            ComputeType.CPU: 50             # CPU is slowest but most flexible
        }[self.type]
        
        # Adjust score based on load
        load_factor = 1 - (self.current_load / 100)
        return base_score * load_factor

class ChunkProcessor:
    """Processes audio chunks using specified compute device."""
    
    def __init__(self, device: ComputeDevice):
        self.device = device
        self.model = None
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.running = False
        self._setup_device()
    
    def _setup_device(self):
        """Setup compute device and optimize model for it."""
        if self.device.type == ComputeType.NEURAL_ENGINE:
            # Convert model to Core ML for Neural Engine
            self.model = self._prepare_coreml_model()
        elif self.device.type == ComputeType.GPU:
            # Use Metal Performance Shaders for GPU
            if torch.backends.mps.is_available():
                self.model = self._prepare_mps_model()
        else:
            # CPU model with threading
            self.model = self._prepare_cpu_model()
    
    def _prepare_coreml_model(self):
        """Prepare Core ML model optimized for Neural Engine."""
        try:
            # Convert PyTorch model to Core ML
            model = ct.convert(
                "whisper-small",
                source="pytorch",
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL  # Use Neural Engine when possible
            )
            return model
        except Exception as e:
            logging.error(f"Failed to prepare Core ML model: {e}")
            return None
    
    def _prepare_mps_model(self):
        """Prepare model for Metal Performance Shaders."""
        try:
            device = torch.device("mps")
            model = torch.load("path/to/whisper_model.pt")
            model = model.to(device)
            return model
        except Exception as e:
            logging.error(f"Failed to prepare MPS model: {e}")
            return None
    
    def _prepare_cpu_model(self):
        """Prepare CPU-optimized model with threading."""
        try:
            model = torch.load("path/to/whisper_model.pt")
            model = model.to("cpu")
            # Enable Intel MKL optimizations if available
            torch.set_num_threads(self.device.compute_units)
            return model
        except Exception as e:
            logging.error(f"Failed to prepare CPU model: {e}")
            return None

    def start(self):
        """Start processing thread."""
        self.running = True
        threading.Thread(target=self._process_queue, daemon=True).start()
    
    def stop(self):
        """Stop processing thread."""
        self.running = False
    
    def _process_queue(self):
        """Process audio chunks from queue."""
        while self.running:
            try:
                chunk = self.processing_queue.get(timeout=1)
                if chunk is None:
                    continue
                
                result = self._process_chunk(chunk)
                self.results_queue.put(result)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
    
    def _process_chunk(self, chunk: np.ndarray) -> Dict:
        """Process a single audio chunk."""
        if self.device.type == ComputeType.NEURAL_ENGINE:
            return self._process_neural_engine(chunk)
        elif self.device.type == ComputeType.GPU:
            return self._process_gpu(chunk)
        else:
            return self._process_cpu(chunk)
    
    def _process_neural_engine(self, chunk: np.ndarray) -> Dict:
        """Process chunk using Neural Engine."""
        try:
            # Prepare input for Core ML model
            input_dict = {"audio": chunk}
            prediction = self.model.predict(input_dict)
            return {"text": prediction["text"], "device": "neural_engine"}
        except Exception as e:
            logging.error(f"Neural Engine processing error: {e}")
            return {"error": str(e)}
    
    def _process_gpu(self, chunk: np.ndarray) -> Dict:
        """Process chunk using GPU (Metal)."""
        try:
            device = torch.device("mps")
            chunk_tensor = torch.from_numpy(chunk).to(device)
            with torch.no_grad():
                result = self.model(chunk_tensor)
            return {"text": result["text"], "device": "gpu"}
        except Exception as e:
            logging.error(f"GPU processing error: {e}")
            return {"error": str(e)}
    
    def _process_cpu(self, chunk: np.ndarray) -> Dict:
        """Process chunk using CPU."""
        try:
            chunk_tensor = torch.from_numpy(chunk)
            with torch.no_grad():
                result = self.model(chunk_tensor)
            return {"text": result["text"], "device": "cpu"}
        except Exception as e:
            logging.error(f"CPU processing error: {e}")
            return {"error": str(e)}

class MultiComputeScheduler:
    """Scheduler that distributes work across all available compute devices."""
    
    def __init__(self):
        self.devices: List[ComputeDevice] = []
        self.processors: Dict[ComputeType, ChunkProcessor] = {}
        self._discover_devices()
        self._setup_processors()
    
    def _discover_devices(self):
        """Discover available compute devices."""
        # Check for Neural Engine
        try:
            ct_devices = ct.ComputeUnit.ALL
            self.devices.append(ComputeDevice(
                type=ComputeType.NEURAL_ENGINE,
                name="Apple Neural Engine",
                memory=0,  # Neural Engine memory is managed by the system
                compute_units=1,
                supported_precisions=["float16"]
            ))
        except Exception:
            logging.info("Neural Engine not available")
        
        # Check for GPU (Metal)
        if torch.backends.mps.is_available():
            self.devices.append(ComputeDevice(
                type=ComputeType.GPU,
                name="Metal GPU",
                memory=0,  # Metal memory is managed by the system
                compute_units=1,
                supported_precisions=["float16", "float32"]
            ))
        
        # Add CPU
        cpu_count = psutil.cpu_count(logical=True)
        self.devices.append(ComputeDevice(
            type=ComputeType.CPU,
            name=f"CPU ({cpu_count} threads)",
            memory=psutil.virtual_memory().total,
            compute_units=cpu_count,
            supported_precisions=["float32"]
        ))
    
    def _setup_processors(self):
        """Setup processors for each device."""
        for device in self.devices:
            processor = ChunkProcessor(device)
            self.processors[device.type] = processor
            processor.start()
    
    def process_audio(self, audio: np.ndarray, chunk_size: int = 30) -> List[Dict]:
        """Process audio using all available compute devices."""
        chunks = self._split_audio(audio, chunk_size)
        results = []
        
        # Distribute chunks across devices based on their performance scores
        for i, chunk in enumerate(chunks):
            best_device = self._get_best_device()
            processor = self.processors[best_device.type]
            processor.processing_queue.put(chunk)
        
        # Collect results
        for _ in range(len(chunks)):
            for processor in self.processors.values():
                if not processor.results_queue.empty():
                    result = processor.results_queue.get()
                    results.append(result)
        
        return self._merge_results(results)
    
    def _split_audio(self, audio: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """Split audio into chunks."""
        return np.array_split(audio, max(1, len(audio) // chunk_size))
    
    def _get_best_device(self) -> ComputeDevice:
        """Get best device for next chunk based on performance scores."""
        return max(self.devices, key=lambda d: d.get_performance_score())
    
    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """Merge results from different devices."""
        # Sort results by timestamp if available
        sorted_results = sorted(results, key=lambda x: x.get("timestamp", 0))
        return sorted_results
    
    def shutdown(self):
        """Shutdown all processors."""
        for processor in self.processors.values():
            processor.stop()
