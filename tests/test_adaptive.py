"""Tests for adaptive learning module."""

import os
import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from whisperx.adaptive import AdaptiveProcessor, VoiceProfile
from whisperx.audio import SAMPLE_RATE

# Test fixtures
@pytest.fixture
def test_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def test_audio_dir():
    """Create test audio directory."""
    test_dir = Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    return test_dir

@pytest.fixture
def sample_audio_files(test_audio_dir):
    """Create sample audio files for testing."""
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    files = []
    for i, freq in enumerate([440, 880]):
        audio = np.sin(2 * np.pi * freq * t)
        file_path = test_audio_dir / f"sample_{i}.wav"
        
        import soundfile as sf
        sf.write(file_path, audio, SAMPLE_RATE)
        files.append(str(file_path))
    
    yield files
    
    # Cleanup
    for file_path in files:
        if os.path.exists(file_path):
            os.unlink(file_path)

@pytest.fixture
def processor(test_cache_dir):
    """Create AdaptiveProcessor instance."""
    return AdaptiveProcessor(cache_dir=test_cache_dir)

# Test VoiceProfile
class TestVoiceProfile:
    """Test suite for VoiceProfile class."""
    
    def test_creation(self):
        """Test VoiceProfile creation."""
        profile = VoiceProfile(
            speaker_id="test_speaker",
            name="Test Speaker",
            embeddings=[np.random.rand(512)],
            audio_samples=["sample.wav"],
            enhancement_params={},
            adaptation_params={}
        )
        
        assert profile.speaker_id == "test_speaker"
        assert profile.name == "Test Speaker"
        assert len(profile.embeddings) == 1
        assert len(profile.audio_samples) == 1
        assert isinstance(profile.enhancement_params, dict)
        assert isinstance(profile.adaptation_params, dict)

# Test AdaptiveProcessor
class TestAdaptiveProcessor:
    """Test suite for AdaptiveProcessor class."""
    
    def test_init(self, test_cache_dir):
        """Test processor initialization."""
        processor = AdaptiveProcessor(cache_dir=test_cache_dir)
        
        assert processor.device in ["cuda", "cpu"]
        assert processor.cache_dir == Path(test_cache_dir)
        assert processor.profiles_dir == Path(test_cache_dir) / "profiles"
        assert processor.profiles_dir.exists()
    
    def test_create_profile(self, processor, sample_audio_files):
        """Test profile creation."""
        speaker_id = processor.create_profile(
            name="Test Speaker",
            audio_files=sample_audio_files
        )
        
        assert speaker_id in processor.profiles
        profile = processor.profiles[speaker_id]
        
        assert profile.name == "Test Speaker"
        assert len(profile.embeddings) > 0
        assert len(profile.audio_samples) > 0
        
        # Check if profile was saved
        profile_path = processor.profiles_dir / f"{speaker_id}.pt"
        assert profile_path.exists()
    
    def test_create_profile_with_params(self, processor, sample_audio_files):
        """Test profile creation with parameters."""
        enhancement_params = {
            "noise_reduce": True,
            "normalize": True
        }
        
        adaptation_params = {
            "num_steps": 100,
            "learning_rate": 0.001
        }
        
        speaker_id = processor.create_profile(
            name="Test Speaker",
            audio_files=sample_audio_files,
            enhancement_params=enhancement_params,
            adaptation_params=adaptation_params
        )
        
        profile = processor.profiles[speaker_id]
        assert profile.enhancement_params == enhancement_params
        assert profile.adaptation_params == adaptation_params
    
    def test_delete_profile(self, processor, sample_audio_files):
        """Test profile deletion."""
        speaker_id = processor.create_profile(
            name="Test Speaker",
            audio_files=sample_audio_files
        )
        
        profile_path = processor.profiles_dir / f"{speaker_id}.pt"
        assert profile_path.exists()
        
        processor.delete_profile(speaker_id)
        assert speaker_id not in processor.profiles
        assert not profile_path.exists()
    
    def test_get_profiles(self, processor, sample_audio_files):
        """Test getting profile list."""
        # Create multiple profiles
        speaker_ids = []
        for i in range(3):
            speaker_id = processor.create_profile(
                name=f"Speaker {i}",
                audio_files=sample_audio_files
            )
            speaker_ids.append(speaker_id)
        
        profiles = processor.get_profiles()
        assert isinstance(profiles, dict)
        assert len(profiles) == 3
        
        for speaker_id in speaker_ids:
            assert speaker_id in profiles
            assert profiles[speaker_id] == f"Speaker {speaker_ids.index(speaker_id)}"
    
    def test_process_audio(self, processor, sample_audio_files):
        """Test audio processing."""
        # Create profile
        speaker_id = processor.create_profile(
            name="Test Speaker",
            audio_files=sample_audio_files,
            enhancement_params={"noise_reduce": True}
        )
        
        # Process without speaker
        result = processor.process_audio(sample_audio_files[0])
        assert isinstance(result, dict)
        assert "audio" in result
        assert "sample_rate" in result
        assert "enhanced" in result
        assert not result["enhanced"]
        
        # Process with speaker and enhancement
        result = processor.process_audio(
            sample_audio_files[0],
            speaker_id=speaker_id,
            enhance_audio=True
        )
        assert result["enhanced"]
        assert isinstance(result["audio"], np.ndarray)
        assert result["sample_rate"] == SAMPLE_RATE

# Test error handling
def test_invalid_profile_creation(processor):
    """Test handling of invalid profile creation."""
    with pytest.raises(ValueError):
        processor.create_profile(
            name="Invalid",
            audio_files=[]
        )
    
    with pytest.raises(RuntimeError):
        processor.create_profile(
            name="Invalid",
            audio_files=["nonexistent.wav"]
        )

def test_invalid_profile_deletion(processor):
    """Test handling of invalid profile deletion."""
    with pytest.raises(ValueError):
        processor.delete_profile("nonexistent_id")

def test_invalid_audio_processing(processor, sample_audio_files):
    """Test handling of invalid audio processing."""
    with pytest.raises(ValueError):
        processor.process_audio(
            sample_audio_files[0],
            speaker_id="nonexistent_id"
        )
    
    with pytest.raises(RuntimeError):
        processor.process_audio("nonexistent.wav")
