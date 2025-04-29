"""Audio processing utilities for WhisperX."""

import numpy as np
import torch
import torchaudio
import librosa
from typing import Optional, Dict, Any, Union

SAMPLE_RATE = 16000
N_SAMPLES = 480000  # 30 seconds of audio at 16kHz

class AudioProcessor:
    """Audio processing utilities."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
    def load(
        self,
        file_path: str,
        start: Optional[float] = None,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """Load audio file with optional trimming.
        
        Args:
            file_path: Path to audio file
            start: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio as numpy array
        """
        if (start is not None and start < 0) or (duration is not None and duration < 0):
            raise RuntimeError("start and duration must be non-negative")
        try:
            # Load audio
            audio, sr = torchaudio.load(file_path)
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Convert to numpy
            audio = audio.squeeze().numpy()
            
            # Trim if requested
            if start is not None or duration is not None:
                audio_len = len(audio)
                if start is not None and duration is not None:
                    start_sample = int(start * self.sample_rate)
                    end_sample = int((start + duration) * self.sample_rate)
                    if start_sample >= audio_len or end_sample > audio_len:
                        raise RuntimeError("start or duration out of bounds")
                    audio = audio[start_sample:end_sample]
                elif start is not None:
                    start_sample = int(start * self.sample_rate)
                    if start_sample >= audio_len:
                        raise RuntimeError("start out of bounds")
                    audio = audio[start_sample:]
                elif duration is not None:
                    end_sample = int(duration * self.sample_rate)
                    if end_sample > audio_len:
                        raise RuntimeError("duration out of bounds")
                    audio = audio[:end_sample]
            
            return audio
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")
    
    def enhance(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        noise_reduce: bool = True,
        normalize: bool = True,
        trim_silence: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Enhance audio quality.
        
        Args:
            audio: Input audio
            noise_reduce: Apply noise reduction
            normalize: Normalize audio volume
            trim_silence: Remove silence
            **kwargs: Additional enhancement parameters
            
        Returns:
            Enhanced audio as numpy array
        """
        try:
            # Convert to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            
            # Make copy to avoid modifying input
            audio = audio.copy()
            
            # Apply enhancements
            if trim_silence:
                audio = librosa.effects.trim(audio, **kwargs.get('trim_params', {}))[0]
                
            if noise_reduce:
                # Apply noise reduction using spectral gating
                audio = self._reduce_noise(audio, **kwargs.get('noise_params', {}))
                
            if normalize:
                audio = librosa.util.normalize(audio)
            
            return audio
            
        except Exception as e:
            raise RuntimeError(f"Failed to enhance audio: {str(e)}")
    
    def _reduce_noise(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
        threshold: float = 0.1
    ) -> np.ndarray:
        """Reduce noise using spectral gating.
        
        Args:
            audio: Input audio
            frame_length: Length of the FFT window
            hop_length: Number of samples between successive frames
            threshold: Noise threshold
            
        Returns:
            Noise-reduced audio
        """
        try:
            # Compute spectrogram
            D = librosa.stft(
                audio,
                n_fft=frame_length,
                hop_length=hop_length
            )
            
            # Compute magnitude spectrogram
            mag = np.abs(D)
            
            # Estimate noise floor
            noise_floor = np.mean(
                np.sort(mag, axis=1)[:, :int(mag.shape[1] * 0.1)],
                axis=1, keepdims=True
            )
            
            # Create mask
            mask = (mag > noise_floor * threshold).astype(float)
            
            # Apply mask
            D_cleaned = D * mask
            
            # Inverse STFT
            audio_cleaned = librosa.istft(
                D_cleaned,
                hop_length=hop_length,
                length=len(audio)
            )
            
            return audio_cleaned
            
        except Exception as e:
            raise RuntimeError(f"Failed to reduce noise: {str(e)}")


def load_audio(
    file_path: str,
    start: Optional[float] = None,
    duration: Optional[float] = None
) -> np.ndarray:
    """Convenience function to load audio file."""
    processor = AudioProcessor()
    return processor.load(file_path, start, duration)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Create a log-mel spectrogram from an audio file or waveform.
    
    Args:
        audio: Path to audio file or audio waveform
        n_mels: Number of mel filterbanks
        padding: Number of samples to pad
        device: Device to use for processing
        
    Returns:
        Log-mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
        
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    if padding > 0:
        audio = torch.nn.functional.pad(audio, (0, padding))
    
    if device is not None:
        audio = audio.to(device)
    
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2
    
    mel_filters = torch.from_numpy(
        librosa.filters.mel(sr=SAMPLE_RATE, n_fft=400, n_mels=n_mels)
    ).to(audio.device)
    
    mel_spec = mel_filters @ magnitudes
    
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec
