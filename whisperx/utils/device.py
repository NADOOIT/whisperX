"""Device selection and management utilities."""
import torch
import warnings
from typing import Union, Any, Dict

def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    if not hasattr(torch.backends, "mps"):
        return False
    return torch.backends.mps.is_available()

def get_device_from_name(device: Union[str, torch.device, None] = None) -> torch.device:
    """Get the appropriate torch device from a device name or object."""
    if device is None:
        device = get_optimal_device()
    elif isinstance(device, str):
        if device == "mps" and not is_mps_available():
            warnings.warn("MPS requested but not available. Falling back to CPU.")
            device = "cpu"
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise ValueError(f"Unsupported device type: {type(device)}")
    return device

def get_optimal_device() -> torch.device:
    """Get the optimal available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif is_mps_available():
        # Check if we're running on Apple Silicon
        try:
            import platform
            if platform.processor() == 'arm':
                return torch.device("mps")
        except:
            pass
    return torch.device("cpu")

def get_compute_type(device: torch.device) -> str:
    """Get the optimal compute type for the given device."""
    if device.type == "mps":
        return "float32"  # MPS works best with float32
    elif device.type == "cuda":
        return "float16"  # CUDA works well with float16
    return "float32"  # CPU defaults to float32

def optimize_device_settings(device: torch.device) -> dict:
    """Get optimized settings for the given device."""
    settings = {
        'compute_type': get_compute_type(device),
        'cpu_threads': 1,
        'batch_size': 16,
    }
    
    if device.type == "mps":
        settings.update({
            'cpu_threads': 4,  # Use more CPU threads when CTranslate2 falls back to CPU
            'batch_size': 8,   # Smaller batches work better on MPS
        })
    elif device.type == "cpu":
        settings.update({
            'cpu_threads': 4,
        })
        
    return settings

def move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move an object and all its parameters to the specified device."""
    if hasattr(obj, "to"):
        # Handle torch.Tensor and nn.Module objects
        return obj.to(device)
    elif isinstance(obj, dict):
        # Handle dictionaries
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Handle lists and tuples
        moved = [move_to_device(item, device) for item in obj]
        return type(obj)(moved)
    elif isinstance(obj, (int, float, str, bool)):
        # Handle primitive types
        return obj
    
    # For other types, try to move their attributes if they exist
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if hasattr(value, "to") or isinstance(value, (dict, list, tuple)):
                setattr(obj, key, move_to_device(value, device))
    return obj

def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    return {
        'cuda': torch.cuda.is_available(),
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps': is_mps_available(),
        'cpu_threads': torch.get_num_threads(),
        'optimal_device': str(get_optimal_device()),
        'torch_version': torch.__version__,
    }
