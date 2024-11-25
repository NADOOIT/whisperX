"""Optimization utilities for WhisperX pipeline."""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
from queue import Queue
import concurrent.futures
from contextlib import contextmanager

@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    chunk_size: int = 30  # seconds
    overlap: float = 0.5  # overlap between chunks
    batch_size: int = 8
    num_threads: int = 4
    max_memory: float = 0.9  # maximum memory usage (90% of available)
    use_cache: bool = True
    precision: torch.dtype = torch.float16
    stream_buffer_size: int = 1024 * 1024  # 1MB buffer for streaming
    prefetch_factor: int = 2

class MemoryManager:
    """Manages memory allocation and monitoring for Metal device."""
    
    def __init__(self, device: torch.device, max_memory_fraction: float = 0.9):
        self.device = device
        self.max_memory_fraction = max_memory_fraction
        self._lock = threading.Lock()
        
    @contextmanager
    def track_memory(self):
        """Context manager to track memory usage."""
        if self.device.type == "mps":
            initial = torch.mps.current_allocated_memory()
            try:
                yield
            finally:
                current = torch.mps.current_allocated_memory()
                if current > initial:
                    # If memory increased significantly, trigger cleanup
                    self.cleanup()
    
    def cleanup(self):
        """Force memory cleanup."""
        with self._lock:
            if self.device.type == "mps":
                torch.mps.empty_cache()
                # Additional Metal-specific optimizations
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
    
    def is_memory_available(self, required_bytes: int) -> bool:
        """Check if enough memory is available."""
        if self.device.type == "mps":
            current = torch.mps.current_allocated_memory()
            # Get total memory from Metal device
            # Note: This is an approximation as Metal doesn't provide direct memory info
            total = 1024 * 1024 * 1024 * 4  # Assume 4GB for Metal GPU
            available = total - current
            return available > required_bytes
        return True

class StreamingProcessor:
    """Handles streaming audio processing for real-time transcription."""
    
    def __init__(self, config: OptimizationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.buffer = Queue(maxsize=config.stream_buffer_size)
        self.memory_manager = MemoryManager(device)
        self._stop_event = threading.Event()
    
    def start_streaming(self):
        """Start the streaming processor."""
        self._stop_event.clear()
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.start()
    
    def stop_streaming(self):
        """Stop the streaming processor."""
        self._stop_event.set()
        self.processing_thread.join()
    
    def _process_stream(self):
        """Process the audio stream in real-time."""
        while not self._stop_event.is_set():
            chunks = []
            # Collect chunks until we have enough for a batch
            while len(chunks) < self.config.batch_size:
                try:
                    chunk = self.buffer.get(timeout=0.1)
                    chunks.append(chunk)
                except:
                    if self._stop_event.is_set():
                        break
            
            if chunks:
                with self.memory_manager.track_memory():
                    self._process_batch(chunks)
    
    def _process_batch(self, chunks: List[torch.Tensor]):
        """Process a batch of audio chunks."""
        batch = torch.stack(chunks).to(self.device)
        # Process batch here
        # Implementation depends on the specific model

class ParallelProcessor:
    """Handles parallel processing of audio segments."""
    
    def __init__(self, config: OptimizationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.memory_manager = MemoryManager(device)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.num_threads
        )
    
    def process_parallel(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Process audio segments in parallel."""
        chunks = self._split_audio(audio)
        futures = []
        
        for chunk in chunks:
            future = self.thread_pool.submit(self._process_chunk, chunk)
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        return self._merge_results(results)
    
    def _split_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into overlapping chunks."""
        chunk_samples = int(self.config.chunk_size * audio.shape[0])
        overlap_samples = int(self.config.overlap * chunk_samples)
        
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            chunks.append(chunk)
            start = end - overlap_samples
        
        return chunks
    
    def _process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single audio chunk."""
        with self.memory_manager.track_memory():
            # Convert to tensor and move to device
            tensor = torch.from_numpy(chunk).to(
                self.device, 
                dtype=self.config.precision
            )
            # Process chunk here
            # Implementation depends on the specific model
            return {"chunk": tensor}
    
    def _merge_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge results from parallel processing."""
        # Implement result merging logic
        return results

class CacheManager:
    """Manages caching for processed audio segments."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        self._lock = threading.Lock()
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if available."""
        with self._lock:
            return self.cache.get(key)
    
    def cache_result(self, key: str, result: Any):
        """Cache processing result."""
        with self._lock:
            self.cache[key] = result
    
    def clear_cache(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()

class MetalOptimizer:
    """Optimizes processing for Metal device."""
    
    def __init__(self, config: OptimizationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.memory_manager = MemoryManager(device)
    
    @contextmanager
    def optimize_for_metal(self):
        """Context manager for Metal-specific optimizations."""
        if self.device.type == "mps":
            # Set optimal Metal configurations
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.config.precision)
            
            try:
                yield
            finally:
                # Restore original configurations
                torch.set_default_dtype(original_dtype)
                self.memory_manager.cleanup()
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for Metal execution."""
        if self.device.type == "mps":
            # Convert model to optimal precision
            model = model.to(dtype=self.config.precision)
            
            # Optimize model architecture if possible
            if hasattr(model, 'optimize_for_metal'):
                model = model.optimize_for_metal()
            
            # Move model to Metal device
            model = model.to(self.device)
            
            # Enable Metal-specific optimizations
            if hasattr(model, 'prepare_for_metal'):
                model.prepare_for_metal()
        
        return model
    
    def optimize_inference(self, model: torch.nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        """Optimize inference process for Metal."""
        with self.optimize_for_metal():
            # Ensure input is in correct precision
            input_data = input_data.to(dtype=self.config.precision)
            
            # Run inference with optimal settings
            with torch.no_grad():
                output = model(input_data)
            
            return output
