import pytest
import platform
import os
from logging_config import setup_logging

# Global logger for all tests
logger = None

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "platform(name): mark test to run only on named platform"
    )
    # Initialize global logger
    global logger
    logger = setup_logging('whisperx_tests')

def pytest_runtest_setup(item):
    """Skip tests that are not for the current platform"""
    platforms = [mark.args[0] for mark in item.iter_markers(name="platform")]
    if platforms:
        current_platform = platform.system().lower()
        if current_platform not in platforms:
            pytest.skip(f"Test requires platform in {platforms}")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on platform"""
    # Always run common tests
    platform_tests = []
    common_tests = []
    
    current_platform = platform.system().lower()
    
    for item in items:
        if "platforms" in str(item.fspath):
            # Only run platform-specific tests for current platform
            if current_platform in str(item.fspath).lower():
                platform_tests.append(item)
        else:
            common_tests.append(item)
    
    items[:] = common_tests + platform_tests

@pytest.fixture(scope="session")
def test_logger():
    """Provide the logger to test functions."""
    global logger
    return logger

@pytest.fixture(scope="session")
def test_audio_path():
    """Provide path to test audio file."""
    audio_path = os.path.join(os.path.dirname(__file__), "assets", "test_audio.wav")
    if not os.path.exists(audio_path):
        pytest.skip("Test audio file not found")
    return audio_path

@pytest.fixture(scope="session")
def model_config():
    """Provide model configuration for tests."""
    return {
        "model_sizes": ["tiny", "base"],
        "compute_type": "float32",
        "device": "cpu"
    }

@pytest.fixture(scope="session")
def test_languages():
    """Provide test languages for multilingual testing."""
    return ["en", "fr", "de"]
