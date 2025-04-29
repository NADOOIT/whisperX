# Issue #6: Metal Support Error in CTranslate2 and WhisperX

## Error Description
- **Error Type**: ValueError
- **Error Message**: This CTranslate2 package was not compiled with Metal support
- **Location**: whisperx/asr.py and related files
- **Test**: test_get_text_for_wav_audio_file in NADOO-Launchpad

## Current State
- [ ] Branch created: error/6-metal-support
- [ ] Initial test written
- [ ] Debug information added

## Debug Information
The error occurs when attempting to use Metal for model loading in CTranslate2, which is not supported by the current package build.

### Current Implementation
The device selection logic in WhisperX assumes Metal support if MPS is available, but this is not the case with the current CTranslate2 package.

### Expected Behavior
- CTranslate2 should be compiled with Metal support
- WhisperX should utilize Metal support when available

## Progress
### Attempted Solutions
1. Reverted device configuration changes in NADOO-Launchpad to investigate Metal support in WhisperX and CTranslate2.

### Next Steps
- [ ] Create branch error/6-metal-support
- [ ] Verify CTranslate2 compilation options
- [ ] Ensure WhisperX configuration supports Metal
- [ ] Run tests to validate Metal support
- [ ] Document changes

## Related Issues
- Issue #5 in NADOO-Launchpad: Metal support error
