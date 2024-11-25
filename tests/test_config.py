"""Test configuration and shared fixtures."""

import os
import pytest
import torch
import logging
from pathlib import Path

# Configure logging for tests
@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for all tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Device configuration
@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# Base directories
@pytest.fixture
def base_dir():
    """Get base directory for tests."""
    return Path(__file__).parent

@pytest.fixture
def data_dir(base_dir):
    """Get data directory for tests."""
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture
def cache_dir(base_dir):
    """Get cache directory for tests."""
    cache_dir = base_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

# Environment setup
@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment variables for testing."""
    # Store original environment
    old_env = dict(os.environ)
    
    # Set test environment variables
    os.environ.update({
        "WHISPERX_TEST": "1",
        "CUDA_VISIBLE_DEVICES": "0" if torch.cuda.is_available() else "",
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(old_env)

# Error checking
@pytest.fixture
def assert_logs():
    """Assert that specific log messages were emitted."""
    class LogChecker:
        def __init__(self):
            self.handler = None
            self.logs = []
        
        def __enter__(self):
            self.handler = logging.StreamHandler()
            self.handler.formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            
            logger = logging.getLogger()
            logger.addHandler(self.handler)
            self.handler.stream = type(
                'Stream',
                (),
                {'write': self.logs.append, 'flush': lambda: None}
            )()
            
            return self.logs
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            logging.getLogger().removeHandler(self.handler)
    
    return LogChecker()

# Performance monitoring
@pytest.fixture
def check_memory_usage():
    """Monitor memory usage during tests."""
    if not torch.cuda.is_available():
        yield lambda: None
        return
    
    torch.cuda.reset_peak_memory_stats()
    yield lambda: torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
