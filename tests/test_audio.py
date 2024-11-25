"""Tests for audio processing module."""

import os
import pytest
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from whisperx.audio import AudioProcessor, load_audio, SAMPLE_RATE

# Test fixtures
@pytest.fixture
def test_audio_dir():
    """Create test audio directory."""
    test_dir = Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    return test_dir

@pytest.fixture
def sample_audio_file(test_audio_dir):
    """Create a sample audio file for testing."""
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    file_path = test_audio_dir / "test_audio.wav"
    sf.write(file_path, audio, SAMPLE_RATE)
    
    yield str(file_path)
    
    # Cleanup
    if file_path.exists():
        file_path.unlink()

@pytest.fixture
def noisy_audio_file(test_audio_dir):
    """Create a noisy audio file for testing."""
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    noise = np.random.normal(0, 0.1, len(t))
    audio = signal + noise
    
    file_path = test_audio_dir / "noisy_audio.wav"
    sf.write(file_path, audio, SAMPLE_RATE)
    
    yield str(file_path)
    
    # Cleanup
    if file_path.exists():
        file_path.unlink()

@pytest.fixture
def stereo_audio_file(test_audio_dir):
    """Create a stereo audio file for testing."""
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    audio = np.vstack([left, right])
    
    file_path = test_audio_dir / "stereo_audio.wav"
    sf.write(file_path, audio.T, SAMPLE_RATE)
    
    yield str(file_path)
    
    # Cleanup
    if file_path.exists():
        file_path.unlink()

# Test AudioProcessor class
class TestAudioProcessor:
    """Test suite for AudioProcessor class."""
    
    def test_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor()
        assert processor.sample_rate == SAMPLE_RATE
        
        custom_sr = 44100
        processor = AudioProcessor(sample_rate=custom_sr)
        assert processor.sample_rate == custom_sr
    
    def test_load_mono(self, sample_audio_file):
        """Test loading mono audio file."""
        processor = AudioProcessor()
        audio = processor.load(sample_audio_file)
        
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert len(audio) == SAMPLE_RATE * 2  # 2 seconds
    
    def test_load_stereo(self, stereo_audio_file):
        """Test loading stereo audio file."""
        processor = AudioProcessor()
        audio = processor.load(stereo_audio_file)
        
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1  # Should be converted to mono
    
    def test_load_with_trim(self, sample_audio_file):
        """Test loading audio with trimming."""
        processor = AudioProcessor()
        
        # Test start time
        audio = processor.load(sample_audio_file, start=1.0)
        assert len(audio) == SAMPLE_RATE  # 1 second remaining
        
        # Test duration
        audio = processor.load(sample_audio_file, duration=1.0)
        assert len(audio) == SAMPLE_RATE  # 1 second
        
        # Test both
        audio = processor.load(sample_audio_file, start=0.5, duration=1.0)
        assert len(audio) == SAMPLE_RATE  # 1 second
    
    def test_enhance(self, noisy_audio_file):
        """Test audio enhancement."""
        processor = AudioProcessor()
        noisy_audio = processor.load(noisy_audio_file)
        
        # Test with default parameters
        enhanced = processor.enhance(noisy_audio)
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == noisy_audio.shape
        
        # Test with specific parameters
        enhanced = processor.enhance(
            noisy_audio,
            noise_reduce=True,
            normalize=True,
            trim_silence=True,
            noise_params={'threshold': 0.2}
        )
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape <= noisy_audio.shape  # May be shorter due to silence trimming
    
    def test_reduce_noise(self, noisy_audio_file):
        """Test noise reduction."""
        processor = AudioProcessor()
        noisy_audio = processor.load(noisy_audio_file)
        
        # Test with default parameters
        cleaned = processor._reduce_noise(noisy_audio)
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.shape == noisy_audio.shape
        
        # Test with custom parameters
        cleaned = processor._reduce_noise(
            noisy_audio,
            frame_length=4096,
            hop_length=1024,
            threshold=0.2
        )
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.shape == noisy_audio.shape

# Test convenience functions
def test_load_audio(sample_audio_file):
    """Test load_audio convenience function."""
    audio = load_audio(sample_audio_file)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE * 2  # 2 seconds

def test_invalid_file():
    """Test handling of invalid audio file."""
    with pytest.raises(RuntimeError):
        load_audio("nonexistent_file.wav")

def test_invalid_parameters(sample_audio_file):
    """Test handling of invalid parameters."""
    processor = AudioProcessor()
    
    # Invalid start time
    with pytest.raises(RuntimeError):
        processor.load(sample_audio_file, start=-1.0)
    
    # Invalid duration
    with pytest.raises(RuntimeError):
        processor.load(sample_audio_file, duration=-1.0)
    
    # Start time beyond file length
    with pytest.raises(RuntimeError):
        processor.load(sample_audio_file, start=10.0)  # File is only 2 seconds
