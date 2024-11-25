import time
import functools
from typing import Optional, Callable, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMetric:
    def __init__(self, name: str, device: str, start_time: float):
        self.name = name
        self.device = device
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        
    def stop(self):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def __str__(self) -> str:
        if self.duration is None:
            return f"{self.name} on {self.device} (running...)"
        return f"{self.name} on {self.device}: {self.duration:.2f}s"

def track_performance(name: Optional[str] = None) -> Callable:
    """
    Decorator to track performance metrics of functions.
    
    Args:
        name: Optional name for the performance metric. If None, uses function name.
        
    Returns:
        Decorated function that tracks performance.
    
    Example:
        @track_performance("transcription")
        def transcribe_audio(audio_path: str) -> str:
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get device from kwargs or default to "cpu"
            device = kwargs.get("device", "cpu")
            metric_name = name or func.__name__
            
            # Create performance metric
            metric = PerformanceMetric(metric_name, device, time.time())
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop timing and log results
                metric.stop()
                logger.info(str(metric))
                
        return wrapper
    return decorator

# Global performance metrics storage
_performance_metrics = []

def get_performance_summary() -> str:
    """Get a summary of all tracked performance metrics."""
    if not _performance_metrics:
        return "No performance metrics available."
    
    summary = ["Performance Summary:"]
    for metric in _performance_metrics:
        summary.append(str(metric))
    return "\n".join(summary)
