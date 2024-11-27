import pytest
from whisperx.asr import load_model


def test_load_model_valid_arguments():
    # Test with valid arguments
    try:
        model = load_model(name="small", device="cpu", compute_type="float32")
        assert model is not None
    except Exception as e:
        pytest.fail(f"load_model raised an exception with valid arguments: {e}")


def test_load_model_invalid_argument():
    # Test with an invalid argument to ensure it raises a TypeError
    with pytest.raises(TypeError):
        load_model(name="small", device="cpu", invalid_arg="value")


def test_load_model_with_download_root_argument():
    # Test with the download_root argument to ensure it raises a TypeError
    with pytest.raises(TypeError):
        load_model(name="small", device="cpu", download_root="/some/path")


def test_load_model_with_vad_params():
    # Test to ensure the FasterWhisperPipeline is initialized with vad_params
    try:
        model = load_model(name="small", device="cpu", compute_type="float32")
        assert model is not None
        assert hasattr(model, '_vad_params'), "FasterWhisperPipeline instance is missing '_vad_params' attribute"
    except Exception as e:
        pytest.fail(f"load_model raised an exception with vad_params: {e}")


def test_load_model_unexpected_kwargs():
    # Test to ensure TypeError is raised when unexpected kwargs are passed
    with pytest.raises(TypeError, match="unexpected keyword argument 'download_root'"):
        load_model(name="small", device="cpu", compute_type="float32", download_root="/some/path")
