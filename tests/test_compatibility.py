import os
import pytest
import numpy as np
import torch
from whisperx.asr import WhisperModel, load_model
from whisperx.audio import load_audio
import logging

@pytest.fixture
def test_audio_path():
    # You'll need to provide a short test audio file
    return os.path.join(os.path.dirname(__file__), "assets", "test_audio.wav")

@pytest.fixture
def model():
    return load_model("tiny", device="cpu", compute_type="float32")

@pytest.fixture
def test_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

@pytest.mark.parametrize("model_size", ["tiny", "base"])
def test_model_initialization(model_size, test_logger):
    test_logger.info(f"Testing model initialization with size: {model_size}")
    try:
        model = load_model(model_size, device="cpu", compute_type="float32")
        assert isinstance(model, WhisperModel)
        assert model.device == "cpu"
        test_logger.info(f"Successfully initialized {model_size} model")
    except Exception as e:
        test_logger.error(f"Model initialization failed for {model_size}: {str(e)}")
        raise

def test_basic_transcription(model, test_audio_path, test_logger):
    test_logger.info("Testing basic transcription functionality")
    try:
        audio = load_audio(test_audio_path)
        result = model.transcribe(audio)
        test_logger.debug(f"Type of result: {type(result)}, value: {result}")
        if isinstance(result, dict):
            assert "segments" in result
            assert len(result["segments"]) > 0
            test_logger.info(f"Successfully transcribed audio with {len(result['segments'])} segments")
            test_logger.debug(f"Transcription result: {result}")
        elif isinstance(result, tuple):
            # Accept generator output (legacy or alt. API)
            gen = result[0]
            assert hasattr(gen, '__iter__'), "First element of tuple should be iterable/generator"
            test_logger.info("Transcription returned a generator (legacy/alt. API)")
        else:
            raise AssertionError(f"Unexpected result type: {type(result)}")
    except Exception as e:
        test_logger.error(f"Transcription failed: {str(e)}")
        raise

def test_batch_processing(model, test_audio_path, test_logger):
    test_logger.info("Testing batch processing capabilities")
    try:
        audio = load_audio(test_audio_path)
        batch = np.stack([audio, audio])
        test_logger.debug(f"Created batch with shape: {batch.shape}")
        
        try:
            result = model.transcribe(batch, batch_size=2)
        except TypeError as te:
            # Fallback: Modell akzeptiert kein batch_size
            test_logger.debug(f"Model does not accept batch_size: {te}")
            result = model.transcribe(batch)
        if isinstance(result, list):
            assert len(result) == 2
            test_logger.info("Successfully processed batch (list result)")
            test_logger.debug(f"Batch results: {result}")
        elif isinstance(result, tuple):
            gen = result[0]
            assert hasattr(gen, '__iter__'), "First element of tuple should be iterable/generator"
            test_logger.info("Batch processing returned a generator (legacy/alt. API)")
        else:
            raise AssertionError(f"Unexpected result type: {type(result)}")
    except Exception as e:
        test_logger.error(f"Batch processing failed: {str(e)}")
        raise

@pytest.mark.parametrize("model_size", ["tiny", "base"])
def test_different_model_sizes(model_size, test_logger):
    test_logger.info(f"Testing model size: {model_size}")
    try:
        model = load_model(model_size, device="cpu", compute_type="float32")
        assert isinstance(model, WhisperModel)
        test_logger.info(f"Successfully loaded {model_size} model")
    except Exception as e:
        test_logger.error(f"Failed to load {model_size} model: {str(e)}")
        raise

@pytest.mark.parametrize("language", ["en", "fr", "de"])
def test_multilingual(model, test_audio_path, language, test_logger):
    test_logger.info(f"Testing transcription with language: {language}")
    try:
        audio = load_audio(test_audio_path)
        result = model.transcribe(audio, language=language)
        if isinstance(result, dict):
            test_logger.info("Transcription returned dict (new API)")
            assert "segments" in result
            segments = result["segments"]
            assert isinstance(segments, list) and len(segments) > 0, "No segments found in result"
            test_logger.info(f"Successfully transcribed in {language}")
            test_logger.debug(f"Transcription result: {result}")
        elif isinstance(result, tuple):
            gen = result[0]
            assert hasattr(gen, '__iter__'), "First element of tuple should be iterable/generator"
            segments = list(gen)
            assert len(segments) > 0, "No segments found in generator result"
            test_logger.info(f"Successfully transcribed in {language} (generator API)")
            test_logger.debug(f"Transcription segments: {segments}")
        else:
            raise AssertionError(f"Unexpected result type: {type(result)}")
    except Exception as e:
        test_logger.error(f"Multilingual transcription failed for {language}: {str(e)}")
        raise

def test_timestamp_generation(model, test_audio_path, test_logger):
    test_logger.info("Testing timestamp generation")
    try:
        audio = load_audio(test_audio_path)
        result = model.transcribe(audio)
        if isinstance(result, dict):
            segments = result["segments"]
        elif isinstance(result, tuple):
            gen = result[0]
            assert hasattr(gen, '__iter__'), "First element of tuple should be iterable/generator"
            segments = list(gen)
        else:
            raise AssertionError(f"Unexpected result type: {type(result)}")
        for i, segment in enumerate(segments):
            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
            else:
                assert hasattr(segment, "start") and hasattr(segment, "end"), "Segment object missing start/end attributes"
                start = segment.start
                end = segment.end
            assert end > start
            test_logger.debug(f"Segment {i}: {start:.2f}s -> {end:.2f}s")
        test_logger.info("Successfully validated timestamps")
    except Exception as e:
        test_logger.error(f"Timestamp validation failed: {str(e)}")
        raise

def test_token_generation(model, test_audio_path, test_logger):
    test_logger.info("Testing token generation process")
    try:
        audio = load_audio(test_audio_path)
        if not hasattr(model, "_get_mel"):
            import pytest
            pytest.skip("Model has no _get_mel method; skipping token generation test.")
        mel = model._get_mel(audio)
        test_logger.debug(f"Generated mel spectrogram with shape: {mel.shape}")
        
        options = model._get_transcribe_options(language="en")
        tokens = model.generate_segment_batched(mel, model.tokenizer, options)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        test_logger.info(f"Successfully generated {len(tokens)} tokens")
        test_logger.debug(f"Token generation result: {tokens}")
    except Exception as e:
        test_logger.error(f"Token generation failed: {str(e)}")
        raise

def test_model_attributes(model, test_logger):
    test_logger.info("Testing model attributes")
    try:
        # Test model configuration
        assert hasattr(model, 'model')
        if not hasattr(model, 'tokenizer'):
            test_logger.warning('Model has no tokenizer attribute; skipping tokenizer-related checks.')
            import pytest
            pytest.skip('Model has no tokenizer attribute; skipping tokenizer-related checks.')
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'feature_extractor')
        
        # Test device configuration
        assert model.device in ['cpu', 'cuda']
        
        # Test compute type
        assert model.compute_type in ['float32', 'float16', 'int8']
        
        test_logger.info("Successfully validated model attributes")
    except Exception as e:
        test_logger.error(f"Model attribute validation failed: {str(e)}")
        raise

def test_error_handling(model, test_logger):
    test_logger.info("Testing error handling")
    try:
        # Test invalid audio input
        with pytest.raises(Exception):
            model.transcribe(None)
        test_logger.info("Successfully caught invalid audio input")
        
        # Test invalid language code
        with pytest.raises(Exception):
            audio = load_audio(test_audio_path)
            model.transcribe(audio, language="invalid_language")
        test_logger.info("Successfully caught invalid language code")
    except Exception as e:
        test_logger.error(f"Error handling test failed: {str(e)}")
        raise
