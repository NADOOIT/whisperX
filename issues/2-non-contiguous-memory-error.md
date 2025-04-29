# Issue #2: Non-Contiguous Memory Error in StorageView

## Error Description
- **Error Type**: ValueError
- **Error Message**: StorageView does not support arrays with non contiguous memory
- **Location**: whisperx/asr.py:90
- **Test**: test_basic_transcription, test_timestamp_generation

## Current State
- [x] Branch created: error/2-non-contiguous-memory
- [x] Initial test written
- [x] Debug information added

## Debug Information
The error occurs when trying to convert numpy arrays to CTranslate2's StorageView format. This happens in the encode method when processing audio features.

### Current Implementation
```python
def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
    # When the model is running on multiple GPUs, the encoder output should be moved
    # to the CPU since we don't know which GPU will handle the next job.
    to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
    # unsqueeze if batch size = 1
    if len(features.shape) == 2:
        features = np.expand_dims(features, 0)

    if not features.flags['C_CONTIGUOUS']:
        features = np.ascontiguousarray(features)

    features = ctranslate2.StorageView.from_array(features)
```

### Expected Behavior
The array should be converted to a contiguous memory layout before being passed to StorageView.

## Progress
### Attempted Solutions
1. Added memory contiguity check and conversion in encode method:
   - Check array contiguity with features.flags['C_CONTIGUOUS']
   - Convert non-contiguous arrays using np.ascontiguousarray()
   - This ensures arrays meet StorageView's memory requirements

### Status
- [x] Issue identified
- [x] Fix implemented
- [x] Tests passing
- [x] Changes documented

## Resolution
The issue was resolved by ensuring input arrays are contiguous in memory before conversion to StorageView. This was implemented by adding a contiguity check and conversion step in the encode method.

### Code Changes
- Location: whisperx/asr.py
- Function: encode()
- Change: Added memory contiguity handling before StorageView conversion

### Impact
- Fixes ValueError about non-contiguous memory
- Maintains compatibility with CTranslate2's requirements
- No performance impact for already-contiguous arrays
- Small overhead for non-contiguous arrays that need conversion

## Related Issues
- Issue #1: Device attribute error (resolved)
