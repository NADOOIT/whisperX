"""Tests for speaker diarization module."""

import os
import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from whisperx.diarize import DiarizationPipeline
from whisperx.audio import SAMPLE_RATE

# Test fixtures
@pytest.fixture
def test_audio_dir():
    """Create test audio directory."""
    test_dir = Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    return test_dir

@pytest.fixture
def sample_audio_file(test_audio_dir):
    """Create a sample audio file with multiple speakers."""
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # Create two distinct "speakers" with different frequencies
    speaker1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz
    speaker2 = np.sin(2 * np.pi * 880 * t)  # 880 Hz
    
    # Alternate between speakers
    audio = np.zeros_like(t)
    segment_length = len(t) // 4
    
    audio[:segment_length] = speaker1[:segment_length]  # Speaker 1
    audio[segment_length:2*segment_length] = speaker2[segment_length:2*segment_length]  # Speaker 2
    audio[2*segment_length:3*segment_length] = speaker1[2*segment_length:3*segment_length]  # Speaker 1
    audio[3*segment_length:] = speaker2[3*segment_length:]  # Speaker 2
    
    file_path = test_audio_dir / "multi_speaker.wav"
    import soundfile as sf
    sf.write(file_path, audio, SAMPLE_RATE)
    
    yield str(file_path)
    
    # Cleanup
    if file_path.exists():
        file_path.unlink()

@pytest.fixture
def diarizer(device):
    """Create DiarizationPipeline instance."""
    return DiarizationPipeline(device=device)

# Test DiarizationPipeline
class TestDiarizationPipeline:
    """Test suite for DiarizationPipeline class."""
    
    def test_init(self, device):
        """Test pipeline initialization."""
        diarizer = DiarizationPipeline(device=device)
        assert diarizer.device == device
        assert diarizer.model is not None
    
    def test_diarization(self, diarizer, sample_audio_file):
        """Test basic diarization."""
        # Load audio
        import soundfile as sf
        audio, _ = sf.read(sample_audio_file)
        
        # Run diarization
        result = diarizer(audio)
        
        # Check result format
        assert isinstance(result, dict)
        assert "segments" in result
        assert "embeddings" in result
        
        # Check segments
        segments = result["segments"]
        assert len(segments) > 0
        for segment in segments:
            assert "start" in segment
            assert "end" in segment
            assert "speaker" in segment
            assert segment["end"] > segment["start"]
        
        # Check embeddings
        embeddings = result["embeddings"]
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2  # (num_segments, embedding_dim)
    
    def test_num_speakers(self, diarizer, sample_audio_file):
        """Test diarization with specified number of speakers."""
        audio, _ = sf.read(sample_audio_file)
        
        # Test with 2 speakers (known ground truth)
        result = diarizer(audio, num_speakers=2)
        speakers = set(s["speaker"] for s in result["segments"])
        assert len(speakers) == 2
        
        # Test with more speakers than actually present
        result = diarizer(audio, num_speakers=4)
        speakers = set(s["speaker"] for s in result["segments"])
        assert len(speakers) <= 4  # Should not find more speakers than actually present
    
    def test_min_max_speakers(self, diarizer, sample_audio_file):
        """Test diarization with speaker count bounds."""
        audio, _ = sf.read(sample_audio_file)
        
        # Test with valid bounds
        result = diarizer(audio, min_speakers=1, max_speakers=3)
        speakers = set(s["speaker"] for s in result["segments"])
        assert 1 <= len(speakers) <= 3
        
        # Test with invalid bounds
        with pytest.raises(ValueError):
            diarizer(audio, min_speakers=3, max_speakers=2)
    
    def test_short_audio(self, diarizer):
        """Test handling of very short audio."""
        # Create 0.1 second audio
        duration = 0.1
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        result = diarizer(audio)
        assert isinstance(result, dict)
        assert "segments" in result
        # May or may not find segments in very short audio
    
    def test_silent_audio(self, diarizer):
        """Test handling of silent audio."""
        audio = np.zeros(SAMPLE_RATE * 2)  # 2 seconds of silence
        
        result = diarizer(audio)
        assert isinstance(result, dict)
        assert "segments" in result
        assert len(result["segments"]) == 0  # Should find no speakers
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory(self, diarizer, sample_audio_file, check_memory_usage):
        """Test GPU memory usage."""
        audio, _ = sf.read(sample_audio_file)
        
        # Run diarization and check memory
        _ = diarizer(audio)
        max_memory = check_memory_usage()
        
        # Memory usage should be reasonable (less than 2GB)
        assert max_memory < 2 * 1024 * 1024 * 1024  # 2GB in bytes

# Test error handling
def test_invalid_audio(diarizer):
    """Test handling of invalid audio input."""
    # None input
    with pytest.raises(ValueError):
        diarizer(None)
    
    # Empty audio
    with pytest.raises(ValueError):
        diarizer(np.array([]))
    
    # Wrong type
    with pytest.raises(TypeError):
        diarizer("not_an_array")
    
    # Wrong shape
    with pytest.raises(ValueError):
        diarizer(np.random.rand(10, 2))  # 2D array not supported

def test_invalid_parameters(diarizer, sample_audio_file):
    """Test handling of invalid parameters."""
    audio, _ = sf.read(sample_audio_file)
    
    # Invalid number of speakers
    with pytest.raises(ValueError):
        diarizer(audio, num_speakers=0)
    
    with pytest.raises(ValueError):
        diarizer(audio, num_speakers=-1)
    
    # Invalid speaker bounds
    with pytest.raises(ValueError):
        diarizer(audio, min_speakers=0)
    
    with pytest.raises(ValueError):
        diarizer(audio, max_speakers=0)
    
    with pytest.raises(ValueError):
        diarizer(audio, min_speakers=2, max_speakers=1)
