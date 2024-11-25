"""Tests for utility functions."""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from whisperx.utils import (
    get_device,
    setup_logger,
    ensure_dir,
    load_config,
    save_config
)

# Test device utilities
def test_get_device():
    """Test device selection."""
    device = get_device()
    assert isinstance(device, str)
    assert device in ["cuda", "cpu", "mps"]
    
    # Test with specific device
    device = get_device("cpu")
    assert device == "cpu"
    
    # Test with unavailable device
    with pytest.raises(ValueError):
        get_device("invalid_device")

# Test logging utilities
def test_setup_logger(tmp_path):
    """Test logger setup."""
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_logger", log_file)
    
    # Test logger configuration
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0
    
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    
    # Verify log file
    assert log_file.exists()
    with open(log_file) as f:
        log_content = f.read()
        assert test_message in log_content

# Test directory utilities
def test_ensure_dir(tmp_path):
    """Test directory creation."""
    test_dir = tmp_path / "test_dir" / "subdir"
    
    # Create directory
    ensure_dir(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()
    
    # Test with existing directory
    ensure_dir(test_dir)  # Should not raise error
    
    # Test with file path
    test_file = tmp_path / "test.txt"
    test_file.touch()
    with pytest.raises(ValueError):
        ensure_dir(test_file)

# Test configuration utilities
def test_config_io(tmp_path):
    """Test configuration I/O."""
    config_file = tmp_path / "config.json"
    
    # Test configuration
    config = {
        "model": "base",
        "device": "cpu",
        "batch_size": 16,
        "sample_rate": 16000,
        "nested": {
            "param1": 1,
            "param2": "value"
        }
    }
    
    # Save configuration
    save_config(config, config_file)
    assert config_file.exists()
    
    # Load configuration
    loaded_config = load_config(config_file)
    assert loaded_config == config
    
    # Test with invalid file
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.json")
    
    # Test with invalid JSON
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("invalid json")
    with pytest.raises(ValueError):
        load_config(invalid_file)

# Test error handling
def test_error_handling():
    """Test error handling utilities."""
    # Test device error
    with pytest.raises(ValueError, match="Invalid device"):
        get_device("invalid")
    
    # Test directory error
    with pytest.raises(ValueError, match="Path exists and is not a directory"):
        test_file = Path("test.txt")
        test_file.touch()
        ensure_dir(test_file)
        test_file.unlink()
    
    # Test config error
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_config(Path("nonexistent.json"))

# Test performance utilities
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_memory_tracking():
    """Test GPU memory tracking."""
    # Create a large tensor
    tensor = torch.rand(1000, 1000).cuda()
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    assert allocated > 0
    assert reserved >= allocated
    
    # Clean up
    del tensor
    torch.cuda.empty_cache()
    
    # Check memory was freed
    new_allocated = torch.cuda.memory_allocated()
    assert new_allocated < allocated

# Test data type conversions
def test_data_conversions():
    """Test data type conversion utilities."""
    # Create test data
    audio = np.random.rand(16000)  # 1 second of audio
    
    # Convert to tensor
    tensor = torch.from_numpy(audio)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == audio.shape
    
    # Convert back to numpy
    array = tensor.numpy()
    assert isinstance(array, np.ndarray)
    assert array.shape == audio.shape
    np.testing.assert_array_almost_equal(array, audio)

# Test path handling
def test_path_handling(tmp_path):
    """Test path handling utilities."""
    # Test relative to absolute path
    rel_path = "test/path"
    abs_path = os.path.abspath(rel_path)
    assert os.path.isabs(abs_path)
    
    # Test path joining
    path1 = tmp_path / "dir1"
    path2 = "dir2"
    joined_path = os.path.join(str(path1), path2)
    assert isinstance(joined_path, str)
    
    # Test path existence
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert test_dir.exists()
    assert test_dir.is_dir()
    
    # Test file path
    test_file = test_dir / "test.txt"
    test_file.touch()
    assert test_file.exists()
    assert test_file.is_file()
