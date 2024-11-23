import pytest
import torch
import os
import platform
from whisperx.asr import load_model

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="These tests are specific to macOS"
)

def test_mps_availability():
    """Test MPS availability on macOS"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this Mac")
    
    assert torch.backends.mps.is_available()
    assert torch.backends.mps.is_built()

def test_mps_model_loading():
    """Test model loading on MPS device"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this Mac")
    
    model = load_model("tiny", device="mps")
    assert model.model.device == "mps"
    assert model.model.compute_type == "float32"

def test_mps_fallback():
    """Test fallback to CPU when MPS fails"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this Mac")
    
    # Force MPS error
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    
    try:
        model = load_model("tiny", device="mps")
    except Exception:
        # Should fallback to CPU
        model = load_model("tiny", device="cpu")
        assert model.model.device == "cpu"
        assert model.model.compute_type == "int8"
    finally:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
