import pytest
import torch
import platform
from whisperx.asr import load_model

# Skip all tests if not on Android
pytestmark = pytest.mark.skipif(
    not platform.system().startswith("Linux") or not any("Android" in line for line in open("/proc/version").readlines()),
    reason="These tests are specific to Android"
)

def test_cpu_model_loading():
    """Test model loading on CPU for Android"""
    model = load_model("tiny", device="cpu")
    assert model.model.device == "cpu"
    assert model.model.compute_type == "int8"

def test_memory_efficient_loading():
    """Test memory-efficient model loading"""
    # Android devices typically have limited memory
    try:
        model = load_model("tiny", device="cpu", compute_type="int8")
        assert model.model.compute_type == "int8"
        
        # Test memory usage is reasonable
        import psutil
        process = psutil.Process()
        memory_use = process.memory_info().rss / 1024 / 1024  # Convert to MB
        assert memory_use < 1000  # Should use less than 1GB
    except Exception as e:
        pytest.fail(f"Failed to load model efficiently: {str(e)}")
