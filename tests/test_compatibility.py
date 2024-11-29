import os
import pytest
import numpy as np
import torch
from whisperx.asr import WhisperModel, load_model
from whisperx.audio import load_audio

@pytest.fixture
def test_audio_path():
    # You'll need to provide a short test audio file
    return os.path.join(os.path.dirname(__file__), "assets", "test_audio.wav")

@pytest.fixture
def model():
    return load_model("tiny", device="cpu", compute_type="float32")

def test_model_initialization(model):
    assert isinstance(model, WhisperModel)
    assert model.device == "cpu"

def test_basic_transcription(model, test_audio_path):
    # Test basic transcription functionality
    audio = load_audio(test_audio_path)
    result = model.transcribe(audio)
    assert isinstance(result, dict)
    assert "segments" in result
    assert len(result["segments"]) > 0

def test_batch_processing(model, test_audio_path):
    # Test batch processing capabilities
    audio = load_audio(test_audio_path)
    # Create a batch by repeating the same audio
    batch = np.stack([audio, audio])
    result = model.transcribe(batch, batch_size=2)
    assert isinstance(result, list)
    assert len(result) == 2

def test_different_model_sizes():
    # Test with different model sizes
    model_sizes = ["tiny", "base"]
    for size in model_sizes:
        model = load_model(size, device="cpu", compute_type="float32")
        assert isinstance(model, WhisperModel)

def test_multilingual(model, test_audio_path):
    # Test transcription with different languages
    audio = load_audio(test_audio_path)
    languages = ["en", "fr", "de"]
    for lang in languages:
        result = model.transcribe(audio, language=lang)
        assert isinstance(result, dict)
        assert "segments" in result

def test_timestamp_generation(model, test_audio_path):
    # Test timestamp generation
    audio = load_audio(test_audio_path)
    result = model.transcribe(audio)
    for segment in result["segments"]:
        assert "start" in segment
        assert "end" in segment
        assert segment["end"] > segment["start"]

def test_token_generation(model, test_audio_path):
    # Test the token generation process specifically
    audio = load_audio(test_audio_path)
    mel = model._get_mel(audio)
    options = model._get_transcribe_options(language="en")
    tokens = model.generate_segment_batched(mel, model.tokenizer, options)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
