# WhisperX Test Suite

This test suite is designed to validate WhisperX functionality across different platforms and devices.

## Test Structure

```
tests/
├── platforms/              # Platform-specific tests
│   ├── test_macos.py      # Apple Silicon (MPS) tests
│   ├── test_windows.py    # Windows CUDA tests
│   ├── test_linux.py      # Linux CUDA tests
│   └── test_android.py    # Android CPU tests
├── test_common.py         # Common tests for all platforms
└── conftest.py           # pytest configuration
```

## Running Tests

### Local Testing
Tests automatically run only for your current platform:

```bash
# Run all tests for current platform
pytest

# Run with verbose output
pytest -v

# Run specific platform tests
pytest tests/platforms/test_macos.py  # Only on macOS
pytest tests/platforms/test_windows.py # Only on Windows
pytest tests/platforms/test_linux.py   # Only on Linux
pytest tests/platforms/test_android.py # Only on Android
```

### Remote Testing
Remote testing is handled by CI/CD pipelines for each platform:
- macOS (Apple Silicon)
- Windows (CUDA)
- Linux (CUDA)
- Android (CPU)
- iOS (Coming soon)

## Test Categories

1. **Common Tests** (`test_common.py`)
   - Basic CPU functionality
   - Model operations
   - Compute type validation

2. **Platform-Specific Tests**
   - **macOS**
     * MPS availability
     * Model loading on MPS
     * Fallback behavior
   - **Windows/Linux**
     * CUDA availability
     * Model loading on CUDA
     * Fallback behavior
   - **Android**
     * CPU model loading
     * Memory efficiency

## Adding New Tests

1. **Platform-Specific Tests**
   - Add new tests to appropriate file in `platforms/`
   - Use `pytestmark` for platform skipping
   - Include proper error handling

2. **Common Tests**
   - Add to `test_common.py`
   - Ensure compatibility across all platforms
   - Use appropriate compute types

## Test Guidelines

1. **Device Detection**
   - Always check device availability
   - Implement proper skipping
   - Handle fallback cases

2. **Memory Management**
   - Clean up resources after tests
   - Monitor memory usage
   - Test memory-constrained scenarios

3. **Error Handling**
   - Test failure scenarios
   - Verify fallback behavior
   - Provide clear error messages
