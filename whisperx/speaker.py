"""Speaker verification and embedding extraction for WhisperX."""

import torch
import torchaudio
from typing import Optional
from speechbrain.pretrained import EncoderClassifier

class SpeakerVerification:
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None
    ):
        """Initialize speaker verification model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name}",
            run_opts={"device": device}
        )

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure correct sample rate
        if hasattr(self.model, 'audio_normalizer'):
            target_sr = self.model.audio_normalizer.sample_rate
            current_sr = 16000  # Default WhisperX sample rate
            
            if current_sr != target_sr:
                waveform = torchaudio.transforms.Resample(
                    current_sr, target_sr
                )(waveform)

        # Extract embedding
        with torch.no_grad():
            embeddings = self.model.encode_batch(waveform)
            embedding = torch.mean(embeddings, dim=1)
            
        return embedding

    def verify_speaker(
        self,
        reference_emb: torch.Tensor,
        test_emb: torch.Tensor,
        threshold: float = 0.75
    ) -> bool:
        """Verify if two embeddings are from the same speaker."""
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            reference_emb, test_emb
        )
        
        return similarity.item() > threshold

    def compare_speakers(
        self,
        reference_audio: torch.Tensor,
        test_audio: torch.Tensor,
        threshold: float = 0.75
    ) -> bool:
        """Compare two audio segments to determine if they're the same speaker."""
        ref_emb = self.get_embedding(reference_audio)
        test_emb = self.get_embedding(test_audio)
        
        return self.verify_speaker(ref_emb, test_emb, threshold)
