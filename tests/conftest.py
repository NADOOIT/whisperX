import pytest
import platform

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "platform(name): mark test to run only on named platform"
    )

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
