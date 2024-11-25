"""Adaptive learning module for WhisperX."""

import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from .audio import load_audio, AudioProcessor
from .diarize import DiarizationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    """Voice profile for speaker adaptation."""
    speaker_id: str
    name: str
    embeddings: List[np.ndarray]
    audio_samples: List[str]
    enhancement_params: Dict[str, Any]
    adaptation_params: Dict[str, Any]

class AdaptiveProcessor:
    """Processor for adaptive transcription."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize the adaptive processor.
        
        Args:
            cache_dir: Directory to store voice profiles and models
            device: Device to run models on ('cuda', 'cpu', etc.)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/whisperx"))
        self.profiles_dir = self.cache_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
        
        # Initialize components
        try:
            logger.info("Initializing diarization pipeline...")
            self.diarizer = DiarizationPipeline(device=self.device)
            
            logger.info("Initializing audio processor...")
            self.audio_processor = AudioProcessor()
            
            logger.info("Loading voice profiles...")
            self.profiles: Dict[str, VoiceProfile] = self._load_profiles()
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise RuntimeError(f"Failed to initialize AdaptiveProcessor: {str(e)}")
    
    def _load_profiles(self) -> Dict[str, VoiceProfile]:
        """Load voice profiles from cache directory."""
        profiles = {}
        if not self.profiles_dir.exists():
            return profiles
            
        try:
            for profile_file in self.profiles_dir.glob("*.pt"):
                try:
                    profile_data = torch.load(profile_file)
                    profile = VoiceProfile(**profile_data)
                    profiles[profile.speaker_id] = profile
                    logger.info(f"Loaded profile: {profile.name} ({profile.speaker_id})")
                except Exception as e:
                    logger.warning(f"Failed to load profile {profile_file}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading profiles: {str(e)}")
            
        return profiles
    
    def create_profile(
        self,
        name: str,
        audio_files: List[str],
        enhancement_params: Optional[Dict[str, Any]] = None,
        adaptation_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new voice profile.
        
        Args:
            name: Name of the speaker
            audio_files: List of paths to audio samples
            enhancement_params: Audio enhancement parameters
            adaptation_params: Model adaptation parameters
            
        Returns:
            speaker_id: Unique ID for the created profile
        """
        try:
            # Generate unique ID
            speaker_id = f"speaker_{len(self.profiles)}"
            
            # Process audio samples
            embeddings = []
            valid_samples = []
            
            for audio_file in audio_files:
                try:
                    # Load and process audio
                    audio = load_audio(audio_file)
                    
                    # Extract embedding
                    diarization = self.diarizer(audio)
                    if len(diarization) > 0:
                        embedding = diarization['embedding'].mean(0)
                        embeddings.append(embedding)
                        valid_samples.append(audio_file)
                        logger.info(f"Processed sample: {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to process {audio_file}: {str(e)}")
            
            if not embeddings:
                raise ValueError("No valid audio samples could be processed")
            
            # Create profile
            profile = VoiceProfile(
                speaker_id=speaker_id,
                name=name,
                embeddings=embeddings,
                audio_samples=valid_samples,
                enhancement_params=enhancement_params or {},
                adaptation_params=adaptation_params or {}
            )
            
            # Save profile
            self._save_profile(profile)
            self.profiles[speaker_id] = profile
            
            logger.info(f"Created profile: {name} ({speaker_id})")
            return speaker_id
            
        except Exception as e:
            logger.error(f"Failed to create profile: {str(e)}")
            raise
    
    def _save_profile(self, profile: VoiceProfile):
        """Save voice profile to disk."""
        try:
            profile_path = self.profiles_dir / f"{profile.speaker_id}.pt"
            torch.save(profile.__dict__, profile_path)
            logger.info(f"Saved profile to {profile_path}")
        except Exception as e:
            logger.error(f"Failed to save profile: {str(e)}")
            raise
    
    def delete_profile(self, speaker_id: str):
        """Delete a voice profile."""
        try:
            if speaker_id not in self.profiles:
                raise ValueError(f"Profile not found: {speaker_id}")
                
            profile_path = self.profiles_dir / f"{speaker_id}.pt"
            if profile_path.exists():
                profile_path.unlink()
                
            del self.profiles[speaker_id]
            logger.info(f"Deleted profile: {speaker_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete profile: {str(e)}")
            raise
    
    def get_profiles(self) -> Dict[str, str]:
        """Get a dictionary of available profiles."""
        return {id: profile.name for id, profile in self.profiles.items()}
    
    def process_audio(
        self,
        audio_path: str,
        speaker_id: Optional[str] = None,
        enhance_audio: bool = False
    ) -> Dict[str, Any]:
        """Process audio with optional speaker adaptation.
        
        Args:
            audio_path: Path to audio file
            speaker_id: Optional speaker profile to use
            enhance_audio: Whether to apply audio enhancement
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        try:
            # Load audio
            audio = load_audio(audio_path)
            
            # Apply speaker-specific processing if requested
            if speaker_id:
                if speaker_id not in self.profiles:
                    raise ValueError(f"Profile not found: {speaker_id}")
                    
                profile = self.profiles[speaker_id]
                
                # Apply audio enhancement if enabled
                if enhance_audio and profile.enhancement_params:
                    audio = self.audio_processor.enhance(
                        audio,
                        **profile.enhancement_params
                    )
                    logger.info("Applied audio enhancement")
                
                # Apply speaker adaptation
                if profile.adaptation_params:
                    # TODO: Implement LoRA adaptation
                    pass
            
            return {
                'audio': audio,
                'sample_rate': self.audio_processor.sample_rate,
                'enhanced': enhance_audio
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio: {str(e)}")
            raise
