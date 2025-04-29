# WhisperX Error Tracking

This document tracks errors encountered during the development and testing of WhisperX, along with their solutions.

## Error History

### Error 1: ModuleNotFoundError - ctranslate2
**Date:** Current
**Error Message:**
```python
ModuleNotFoundError: No module named 'ctranslate2'
```
**Context:** Encountered while running pytest on test_compatibility.py
**Solution:**
1. Cloned CTranslate2 from NADOOIT fork
2. Checked out surfer branch
3. Initialized submodules
4. Installed package in editable mode
```bash
git clone https://github.com/NADOOIT/CTranslate2.git
cd CTranslate2
git checkout surfer
git submodule update --init --recursive
cd python
pip install -e .
```

### Error 2: ImportError - N_SAMPLES
**Date:** Current
**Error Message:**
```python
ImportError: cannot import name 'N_SAMPLES' from 'whisperx.audio'
```
**Context:** After fixing ctranslate2 dependency, encountered while running pytest
**Solution:**
1. Added N_SAMPLES constant to whisperx/audio.py
2. Added missing log_mel_spectrogram function
3. Constants and functions added:
```python
N_SAMPLES = 480000  # 30 seconds of audio at 16kHz

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    # Function implementation...
```

### Error 3: Metal Support Error
**Date:** Current
**Error Message:**
```
ValueError: This CTranslate2 package was not compiled with Metal support
```
**Context:** Encountered when attempting to use Metal for model loading in CTranslate2 during transcription.
**Solution:**
1. Reverted device configuration changes in NADOO-Launchpad to investigate Metal support in WhisperX and CTranslate2.
2. Compiled CTranslate2 with Metal support enabled:
   ```bash
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP_RUNTIME=NONE -DBUILD_SHARED_LIBS=ON -DENABLE_METAL=ON ..
   make
   ```
3. Updated WhisperX configuration to utilize Metal support.

**Next Steps:**
- Verify WhisperX configuration for Metal support.
- Run tests to validate Metal support functionality.
- Document changes and update issue.

## Current Status
- ✅ Fixed ctranslate2 dependency issue
- ✅ Fixed N_SAMPLES import error
- **Issue #6**: Metal support error is being addressed in the `error/6-metal-support` branch.
- **Progress**: CTranslate2 compiled with Metal support, WhisperX configuration under review, and Metal support error documented.
- ⏳ Running tests to identify any remaining issues
