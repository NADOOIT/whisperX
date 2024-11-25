"""Tests for the WhisperX pipeline."""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from whisperx.pipeline import WhisperXPipeline
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
    """Create a sample audio file with speech."""
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # Create simple speech-like audio
    audio = np.sin(2 * np.pi * 440 * t) * 0.5  # Base frequency
    audio += np.sin(2 * np.pi * 880 * t) * 0.3  # Harmonics
    audio *= np.exp(-t)  # Decay
    
    file_path = test_audio_dir / "speech.wav"
    import soundfile as sf
    sf.write(file_path, audio, SAMPLE_RATE)
    
    yield str(file_path)
    
    # Cleanup
    if file_path.exists():
        file_path.unlink()

@pytest.fixture
def pipeline(device):
    """Create WhisperXPipeline instance."""
    return WhisperXPipeline(device=device)

# Test WhisperXPipeline
class TestWhisperXPipeline:
    """Test suite for WhisperXPipeline class."""
    
    def test_init(self, device):
        """Test pipeline initialization."""
        pipeline = WhisperXPipeline(device=device)
        assert pipeline.device == device
        assert pipeline.model is not None
        assert pipeline.diarizer is not None
    
    def test_transcribe(self, pipeline, sample_audio_file):
        """Test basic transcription."""
        # Load audio
        import soundfile as sf
        audio, _ = sf.read(sample_audio_file)
        
        # Run transcription
        result = pipeline.transcribe(audio)
        
        # Check result format
        assert isinstance(result, dict)
        assert "segments" in result
        assert "text" in result
        
        # Check segments
        segments = result["segments"]
        assert len(segments) > 0
        for segment in segments:
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert segment["end"] > segment["start"]
    
    def test_transcribe_with_diarization(self, pipeline, sample_audio_file):
        """Test transcription with speaker diarization."""
        audio, _ = sf.read(sample_audio_file)
        
        # Run transcription with diarization
        result = pipeline.transcribe(audio, diarize=True)
        
        # Check diarization results
        assert "speaker_segments" in result
        speaker_segments = result["speaker_segments"]
        assert len(speaker_segments) > 0
        for segment in speaker_segments:
            assert "speaker" in segment
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
    
    def test_language_detection(self, pipeline, sample_audio_file):
        """Test language detection."""
        audio, _ = sf.read(sample_audio_file)
        
        # Run transcription with language detection
        result = pipeline.transcribe(audio, detect_language=True)
        
        # Check language detection
        assert "detected_language" in result
        assert isinstance(result["detected_language"], str)
        assert len(result["detected_language"]) == 2  # ISO language code
    
    def test_batch_processing(self, pipeline, test_audio_dir):
        """Test batch processing of multiple audio files."""
        # Create multiple test files
        files = []
        for i in range(3):
            duration = 2.0
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            file_path = test_audio_dir / f"test_{i}.wav"
            import soundfile as sf
            sf.write(file_path, audio, SAMPLE_RATE)
            files.append(str(file_path))
        
        # Process batch
        results = pipeline.batch_transcribe(files)
        
        # Check results
        assert len(results) == len(files)
        for result in results:
            assert isinstance(result, dict)
            assert "text" in result
            assert "segments" in result
        
        # Cleanup
        for file_path in files:
            Path(file_path).unlink()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory(self, pipeline, sample_audio_file, check_memory_usage):
        """Test GPU memory usage."""
        audio, _ = sf.read(sample_audio_file)
        
        # Run transcription and check memory
        _ = pipeline.transcribe(audio)
        max_memory = check_memory_usage()
        
        # Memory usage should be reasonable
        assert max_memory < 4 * 1024 * 1024 * 1024  # 4GB in bytes
    
    def test_error_handling(self, pipeline):
        """Test error handling in pipeline."""
        # Test with None input
        with pytest.raises(ValueError):
            pipeline.transcribe(None)
        
        # Test with empty audio
        with pytest.raises(ValueError):
            pipeline.transcribe(np.array([]))
        
        # Test with invalid audio shape
        with pytest.raises(ValueError):
            pipeline.transcribe(np.random.rand(10, 2))
        
        # Test with invalid sample rate
        with pytest.raises(ValueError):
            pipeline.transcribe(np.random.rand(16000), sample_rate=8000)
    
    def test_timestamp_alignment(self, pipeline, sample_audio_file):
        """Test timestamp alignment in transcription."""
        audio, _ = sf.read(sample_audio_file)
        
        # Run transcription
        result = pipeline.transcribe(audio)
        
        # Check timestamp alignment
        segments = result["segments"]
        for i in range(len(segments) - 1):
            # Check segments are sequential
            assert segments[i]["end"] <= segments[i + 1]["start"]
            # Check segment duration is reasonable
            duration = segments[i]["end"] - segments[i]["start"]
            assert 0 < duration < 30  # Reasonable segment duration
    
    def test_model_configuration(self, device):
        """Test different model configurations."""
        # Test with different model sizes
        for model_size in ["tiny", "base", "small"]:
            pipeline = WhisperXPipeline(
                device=device,
                model_size=model_size
            )
            assert pipeline.model is not None
            assert pipeline.model_size == model_size
        
        # Test with invalid model size
        with pytest.raises(ValueError):
            WhisperXPipeline(device=device, model_size="invalid")
    
    def test_output_formats(self, pipeline, sample_audio_file):
        """Test different output formats."""
        audio, _ = sf.read(sample_audio_file)
        
        # Test JSON output
        result = pipeline.transcribe(audio, output_format="json")
        assert isinstance(result, dict)
        
        # Test text output
        result = pipeline.transcribe(audio, output_format="text")
        assert isinstance(result, str)
        
        # Test SRT output
        result = pipeline.transcribe(audio, output_format="srt")
        assert isinstance(result, str)
        assert "-->" in result  # SRT timestamp format
        
        # Test invalid format
        with pytest.raises(ValueError):
            pipeline.transcribe(audio, output_format="invalid")
