import pytest
import torch
from whisperx.asr import load_model

def test_cpu_availability():
    """Test basic CPU functionality - should work on all platforms"""
    model = load_model("tiny", device="cpu")
    assert model.model.device == "cpu"
    assert model.model.compute_type == "int8"

def test_model_basic_operations():
    """Test basic model operations that should work on all platforms"""
    model = load_model("tiny", device="cpu")
    
    # Test model properties
    assert hasattr(model, 'model')
    assert hasattr(model, 'options')
    
    # Test model configuration
    assert model.model.is_multilingual
    assert model.options is not None

@pytest.mark.parametrize("compute_type", ["int8", "float32"])
def test_compute_types(compute_type):
    """Test different compute types on CPU"""
    model = load_model("tiny", device="cpu", compute_type=compute_type)
    assert model.model.compute_type == compute_type
