import os
import pytest
import torch
from whisperx.asr import load_model, get_optimal_device

def test_optimal_device_selection():
    """Test optimal device selection logic"""
    device, compute_type = get_optimal_device()
    
    if torch.cuda.is_available():
        assert device == "cuda"
        assert compute_type == "float16"
    elif torch.backends.mps.is_available():
        assert device == "cpu"  # Currently using CPU for MPS systems
        assert compute_type == "float32"
    else:
        assert device == "cpu"
        assert compute_type == "int8"

def test_model_loading_with_auto_device():
    """Test model loading with auto device selection"""
    model = load_model("tiny", device="auto")
    assert model is not None

def test_mps_fallback_behavior():
    """Test MPS fallback behavior"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this device")
    
    # Force MPS to test fallback
    model = load_model("tiny", device="mps")
    assert model is not None
    # Verify we're actually using CPU with appropriate settings
    assert model.model.device == "cpu"
    assert model.model.compute_type == "float32"

def test_compute_type_override():
    """Test compute type override behavior"""
    model = load_model("tiny", device="cpu", compute_type="int8")
    assert model is not None
    # CTranslate2 uses int8_float32 mode for int8 quantization
    assert model.model.compute_type in ["int8", "int8_float32"]
