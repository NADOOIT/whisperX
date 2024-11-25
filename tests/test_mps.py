"""Tests for MPS (Metal Performance Shaders) support."""
import pytest
import torch
import numpy as np
from whisperx.utils.device import (
    get_device_from_name,
    get_optimal_device,
    is_mps_available,
    get_device_info
)
from whisperx.asr import ASRModel


@pytest.mark.skipif(not is_mps_available(), reason="MPS not available")
class TestMPSSupport:
    """Test suite for MPS support in WhisperX."""
    
    def test_device_detection(self):
        """Test MPS device detection."""
        device_info = get_device_info()
        assert device_info['mps'] is True
        assert isinstance(device_info['mps'], bool)
    
    def test_device_selection(self):
        """Test MPS device selection."""
        device = get_device_from_name('mps')
        assert device.type == 'mps'
        assert isinstance(device, torch.device)
    
    def test_optimal_device(self):
        """Test optimal device selection on Apple Silicon."""
        device = get_optimal_device()
        # On Apple Silicon without CUDA, MPS should be selected
        if not torch.cuda.is_available():
            assert device.type == 'mps'
    
    @pytest.mark.parametrize("compute_type", ["float16", "int8"])
    def test_model_initialization(self, compute_type):
        """Test model initialization with MPS."""
        model = ASRModel("tiny", device="mps", compute_type=compute_type)
        assert model.device.type == 'mps'
        # Both int8 and float16 should fall back to float32 on MPS
        assert model.compute_type == "float32"
        # CTranslate2 should use CPU backend
        assert model._ct2_device == "cpu"
    
    def test_transcription(self):
        """Test basic transcription with MPS."""
        # Create a simple test audio signal
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        model = ASRModel("tiny", device="mps")
        result = model.transcribe(audio)
        
        assert isinstance(result, dict)
        assert "segments" in result
    
    def test_device_info(self):
        """Test device info retrieval."""
        model = ASRModel("tiny", device="mps")
        info = model.get_device_info()
        
        assert info['device_type'] == 'mps'
        assert info['mps_available'] is True
        assert info['compute_type'] == "float32"
        assert info['ct2_device'] == "cpu"
        assert info['cpu_threads'] > 0
    
    def test_batch_size_adjustment(self):
        """Test batch size adjustment for MPS."""
        model = ASRModel("tiny", device="mps")
        # Create dummy audio
        audio = np.zeros(16000)  # 1 second of silence
        
        # Test with different batch sizes
        large_batch = model.transcribe(audio, batch_size=16)
        small_batch = model.transcribe(audio, batch_size=4)
        
        # Both should work, and large batch should be automatically reduced
        assert isinstance(large_batch, dict)
        assert isinstance(small_batch, dict)


@pytest.mark.skipif(is_mps_available(), reason="MPS is available")
class TestMPSFallback:
    """Test MPS fallback behavior when MPS is not available."""
    
    def test_fallback_to_cpu(self):
        """Test fallback to CPU when MPS is requested but not available."""
        model = ASRModel("tiny", device="mps")
        assert model.device.type == 'cpu'
        assert model._ct2_device == 'cpu'
    
    def test_device_info_without_mps(self):
        """Test device info when MPS is not available."""
        device_info = get_device_info()
        assert device_info['mps'] is False
