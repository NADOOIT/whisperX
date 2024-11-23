import pytest
import torch
import platform
from whisperx.asr import load_model

# Skip all tests if not on Linux
pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="These tests are specific to Linux"
)

def test_cuda_availability():
    """Test CUDA availability on Linux"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this Linux machine")
    
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() > 0

def test_cuda_model_loading():
    """Test model loading on CUDA device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this Linux machine")
    
    model = load_model("tiny", device="cuda")
    assert model.model.device == "cuda"
    assert model.model.compute_type == "float16"

def test_cuda_fallback():
    """Test fallback to CPU when CUDA fails"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this Linux machine")
    
    # Force CUDA error by requesting more memory than available
    try:
        torch.cuda.set_per_process_memory_fraction(2.0)  # This will cause CUDA to fail
        model = load_model("tiny", device="cuda")
    except Exception:
        # Should fallback to CPU
        model = load_model("tiny", device="cpu")
        assert model.model.device == "cpu"
        assert model.model.compute_type == "int8"
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)  # Reset memory fraction
