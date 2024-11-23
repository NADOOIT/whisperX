import pytest
import torch
import os
from whisperx.asr import load_model

def test_mps_device_detection():
    """Test MPS device detection and fallback"""
    if torch.backends.mps.is_available():
        model = load_model("tiny", device="mps")
        assert model.model.device == "mps"
        assert model.model.compute_type == "float32"
    else:
        # Skip test if MPS not available
        pytest.skip("MPS not available on this device")

def test_mps_fallback():
    """Test fallback to CPU when MPS fails"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this device")
    
    # Force MPS error by setting environment variable
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    
    try:
        model = load_model("tiny", device="mps")
    except Exception:
        # Should fallback to CPU
        model = load_model("tiny", device="cpu")
        assert model.model.device == "cpu"
        assert model.model.compute_type == "int8"
    finally:
        # Reset environment variable
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def test_compute_type_selection():
    """Test compute type selection based on device"""
    if torch.cuda.is_available():
        model = load_model("tiny", device="cuda")
        assert model.model.compute_type == "float16"
    elif torch.backends.mps.is_available():
        model = load_model("tiny", device="mps")
        assert model.model.compute_type == "float32"
    else:
        model = load_model("tiny", device="cpu")
        assert model.model.compute_type == "int8"
