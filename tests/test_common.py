import pytest
import torch
from whisperx.asr import load_model

def test_cpu_availability():
    """Test basic CPU functionality - should work on all platforms"""
    model = load_model("tiny", device="cpu", compute_type="int8")
    assert model is not None

def test_model_basic_operations():
    """Test basic model operations that should work on all platforms"""
    model = load_model("tiny", device="cpu", compute_type="int8")
    
    # Test model properties
    assert hasattr(model, 'model')
    assert model.model is not None

@pytest.mark.parametrize("compute_type", ["int8", "float32"])
def test_compute_types(compute_type):
    """Test different compute types on CPU"""
    model = load_model("tiny", device="cpu", compute_type=compute_type)
    assert model is not None
