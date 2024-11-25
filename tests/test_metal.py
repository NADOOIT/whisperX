"""Tests for Metal device functionality in WhisperX."""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from whisperx.pipeline import WhisperXPipeline
from whisperx.audio import SAMPLE_RATE, AudioProcessor
from whisperx.utils import get_device

# Skip all tests if Metal is not available
metal_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
requires_metal = pytest.mark.skipif(not metal_available, reason="Metal not available")

@pytest.fixture
def metal_device():
    """Get Metal device if available."""
    if not metal_available:
        pytest.skip("Metal not available")
    return torch.device("mps")

@pytest.fixture
def test_audio_dir():
    """Create test audio directory."""
    test_dir = Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    return test_dir

@pytest.fixture
def sample_audio_file(test_audio_dir):
    """Create a sample audio file for testing."""
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # Create complex audio with multiple frequencies
    audio = np.sin(2 * np.pi * 440 * t) * 0.4  # A4 note
    audio += np.sin(2 * np.pi * 880 * t) * 0.3  # A5 note
    audio += np.sin(2 * np.pi * 1760 * t) * 0.2  # A6 note
    audio *= np.exp(-t/2)  # Apply decay
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(t))
    audio += noise
    
    file_path = test_audio_dir / "complex_audio.wav"
    import soundfile as sf
    sf.write(file_path, audio, SAMPLE_RATE)
    
    yield str(file_path)
    
    if file_path.exists():
        file_path.unlink()

@requires_metal
class TestMetalDevice:
    """Test suite for Metal device functionality."""
    
    def test_device_availability(self):
        """Test Metal device availability and properties."""
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
        assert device.type == "mps"
        
        # Test basic tensor operations on Metal
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = x @ y
        assert z.device.type == "mps"
    
    def test_memory_management(self, metal_device):
        """Test Metal memory management."""
        # Create large tensors to test memory allocation
        tensors = []
        initial_memory = torch.mps.current_allocated_memory()
        
        # Allocate multiple tensors
        for _ in range(5):
            tensor = torch.randn(1000, 1000, device=metal_device)
            tensors.append(tensor)
        
        # Check memory increased
        assert torch.mps.current_allocated_memory() > initial_memory
        
        # Delete tensors and check memory is freed
        del tensors
        torch.mps.empty_cache()
        final_memory = torch.mps.current_allocated_memory()
        assert final_memory < initial_memory * 2  # Memory should be significantly reduced
    
    def test_data_transfer(self, metal_device):
        """Test data transfer between CPU and Metal device."""
        # CPU to Metal
        cpu_tensor = torch.randn(100, 100)
        metal_tensor = cpu_tensor.to(metal_device)
        assert metal_tensor.device.type == "mps"
        assert torch.allclose(cpu_tensor, metal_tensor.cpu(), rtol=1e-3)
        
        # Metal to CPU
        back_to_cpu = metal_tensor.cpu()
        assert back_to_cpu.device.type == "cpu"
        assert torch.allclose(cpu_tensor, back_to_cpu, rtol=1e-3)
        
        # Test with different data types
        dtypes = [torch.float32, torch.int64, torch.bool]
        for dtype in dtypes:
            cpu_tensor = torch.randn(50, 50).to(dtype)
            metal_tensor = cpu_tensor.to(metal_device)
            assert metal_tensor.dtype == dtype
            assert torch.allclose(cpu_tensor.float(), metal_tensor.cpu().float(), rtol=1e-3)

@requires_metal
class TestMetalAudioProcessing:
    """Test suite for audio processing on Metal."""
    
    def test_audio_processor_metal(self, metal_device, sample_audio_file):
        """Test AudioProcessor with Metal device."""
        processor = AudioProcessor(device=metal_device)
        
        # Load and process audio
        import soundfile as sf
        audio, _ = sf.read(sample_audio_file)
        
        # Test basic processing
        processed = processor.process(audio)
        assert isinstance(processed, torch.Tensor)
        assert processed.device.type == "mps"
        
        # Test batch processing
        batch = torch.stack([torch.from_numpy(audio)] * 3)
        batch = batch.to(metal_device)
        processed_batch = processor.process_batch(batch)
        assert processed_batch.device.type == "mps"
        assert processed_batch.shape[0] == 3
    
    def test_audio_augmentation_metal(self, metal_device, sample_audio_file):
        """Test audio augmentation on Metal."""
        processor = AudioProcessor(device=metal_device)
        audio, _ = sf.read(sample_audio_file)
        
        # Test various augmentations
        # Time stretching
        stretched = processor.time_stretch(audio, rate=1.2)
        assert isinstance(stretched, torch.Tensor)
        assert stretched.device.type == "mps"
        
        # Pitch shifting
        shifted = processor.pitch_shift(audio, n_steps=2)
        assert isinstance(shifted, torch.Tensor)
        assert shifted.device.type == "mps"
        
        # Test combined augmentations
        augmented = processor.augment(audio, 
                                    time_stretch_rate=1.1,
                                    pitch_shift_steps=1)
        assert isinstance(augmented, torch.Tensor)
        assert augmented.device.type == "mps"

@requires_metal
class TestMetalPipeline:
    """Test suite for WhisperX pipeline on Metal."""
    
    def test_pipeline_metal(self, metal_device, sample_audio_file):
        """Test full pipeline on Metal."""
        pipeline = WhisperXPipeline(device="mps")
        
        # Test transcription
        audio, _ = sf.read(sample_audio_file)
        result = pipeline.transcribe(audio)
        
        assert isinstance(result, dict)
        assert "text" in result
        assert len(result["text"]) > 0
    
    def test_parallel_processing_metal(self, metal_device, test_audio_dir):
        """Test parallel processing capabilities on Metal."""
        # Create multiple test files
        files = []
        for i in range(3):
            duration = 2.0
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            file_path = test_audio_dir / f"test_metal_{i}.wav"
            sf.write(file_path, audio, SAMPLE_RATE)
            files.append(str(file_path))
        
        pipeline = WhisperXPipeline(device="mps")
        
        # Test batch processing
        results = pipeline.batch_transcribe(files)
        assert len(results) == len(files)
        
        # Cleanup
        for file_path in files:
            Path(file_path).unlink()
    
    def test_model_operations_metal(self, metal_device):
        """Test model operations on Metal."""
        pipeline = WhisperXPipeline(device="mps")
        
        # Test model parameters are on Metal
        for param in pipeline.model.parameters():
            assert param.device.type == "mps"
        
        # Test forward pass
        dummy_input = torch.randn(1, 80, 3000, device=metal_device)
        with torch.no_grad():
            output = pipeline.model(dummy_input)
        assert output.device.type == "mps"
    
    def test_metal_performance(self, metal_device, sample_audio_file):
        """Test performance metrics on Metal."""
        pipeline = WhisperXPipeline(device="mps")
        audio, _ = sf.read(sample_audio_file)
        
        # Warm-up run
        _ = pipeline.transcribe(audio)
        
        # Measure processing time
        import time
        start_time = time.time()
        result = pipeline.transcribe(audio)
        processing_time = time.time() - start_time
        
        # Processing time should be reasonable
        assert processing_time < 30  # Adjust threshold based on your requirements
        
        # Test memory efficiency
        initial_memory = torch.mps.current_allocated_memory()
        _ = pipeline.transcribe(audio)
        final_memory = torch.mps.current_allocated_memory()
        
        # Memory usage should be stable
        assert final_memory < initial_memory * 2

@requires_metal
def test_metal_error_handling(metal_device):
    """Test error handling specific to Metal device."""
    
    # Test invalid tensor operations
    with pytest.raises(RuntimeError):
        # Try to perform invalid operation
        invalid_shape = torch.randn(2, 3, device=metal_device)
        invalid_shape @ torch.randn(4, 5, device=metal_device)
    
    # Test device mismatch
    cpu_tensor = torch.randn(2, 2)
    metal_tensor = torch.randn(2, 2, device=metal_device)
    
    with pytest.raises(RuntimeError):
        # Try to perform operation between CPU and Metal tensors without explicit transfer
        _ = cpu_tensor + metal_tensor
    
    # Test out of memory simulation
    try:
        # Try to allocate a very large tensor
        huge_tensor = torch.randn(1000000, 1000000, device=metal_device)
    except RuntimeError as e:
        assert "out of memory" in str(e).lower()

@requires_metal
def test_metal_precision(metal_device):
    """Test numerical precision on Metal device."""
    
    # Test different precisions
    for dtype in [torch.float32, torch.float16]:
        x = torch.randn(100, 100, dtype=dtype, device=metal_device)
        y = torch.randn(100, 100, dtype=dtype, device=metal_device)
        
        # Test basic operations
        z = x @ y
        assert z.dtype == dtype
        
        # Compare with CPU results
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        z_cpu = x_cpu @ y_cpu
        
        # Check results are close within tolerance
        rtol = 1e-3 if dtype == torch.float32 else 1e-2
        assert torch.allclose(z.cpu(), z_cpu, rtol=rtol)

@requires_metal
def test_metal_device_switching(metal_device):
    """Test device switching behavior with Metal."""
    
    # Create tensors on different devices
    cpu_tensor = torch.randn(10, 10)
    metal_tensor = torch.randn(10, 10, device=metal_device)
    
    # Test switching between devices
    assert cpu_tensor.to(metal_device).device.type == "mps"
    assert metal_tensor.cpu().device.type == "cpu"
    
    # Test operation results after device switching
    result = cpu_tensor.to(metal_device) @ metal_tensor
    assert result.device.type == "mps"
    
    # Test copying vs moving
    metal_copy = cpu_tensor.to(metal_device)
    assert metal_copy.device.type == "mps"
    assert cpu_tensor.device.type == "cpu"  # Original should remain on CPU
