"""Enhanced WhisperX pipeline with optimizations."""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .audio import AudioProcessor
from .diarize import DiarizationPipeline
from .optimizations import (
    OptimizationConfig,
    MemoryManager,
    StreamingProcessor,
    ParallelProcessor,
    CacheManager,
    MetalOptimizer
)

@dataclass
class PipelineConfig:
    """Configuration for the WhisperX pipeline."""
    model_size: str = "base"
    device: str = "mps"
    compute_type: str = "float16"
    batch_size: int = 8
    chunk_size: int = 30
    beam_size: int = 5
    language: Optional[str] = None
    task: str = "transcribe"
    optimize_metal: bool = True

class WhisperXPipeline:
    """Enhanced WhisperX pipeline with Metal optimizations."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize optimization components
        self.opt_config = OptimizationConfig(
            chunk_size=self.config.chunk_size,
            batch_size=self.config.batch_size,
            precision=torch.float16 if self.config.compute_type == "float16" else torch.float32
        )
        self.memory_manager = MemoryManager(self.device)
        self.metal_optimizer = MetalOptimizer(self.opt_config, self.device)
        self.cache_manager = CacheManager(self.opt_config)
        
        # Initialize processors
        self.streaming_processor = StreamingProcessor(self.opt_config, self.device)
        self.parallel_processor = ParallelProcessor(self.opt_config, self.device)
        
        # Initialize model and processors
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize model and processing components."""
        # Initialize audio processor
        self.audio_processor = AudioProcessor(device=self.device)
        
        # Initialize diarization pipeline if needed
        self.diarizer = DiarizationPipeline(device=self.device)
        
        # Load and optimize model
        self.model = self._load_model()
        if self.config.optimize_metal and self.device.type == "mps":
            self.model = self.metal_optimizer.optimize_model(self.model)
    
    def _load_model(self) -> torch.nn.Module:
        """Load and prepare the Whisper model."""
        import whisper
        model = whisper.load_model(self.config.model_size)
        
        # Optimize model loading for Metal
        if self.device.type == "mps":
            model = model.to(self.device)
            # Enable memory efficient optimizations
            if hasattr(model, "enable_metal_optimizations"):
                model.enable_metal_optimizations()
        
        return model
    
    def transcribe(self, 
                  audio: np.ndarray,
                  diarize: bool = False,
                  stream: bool = False,
                  **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio with optimized processing.
        
        Args:
            audio: Input audio array
            diarize: Whether to perform speaker diarization
            stream: Whether to use streaming mode
            **kwargs: Additional arguments for transcription
        """
        # Check if result is cached
        cache_key = self._get_cache_key(audio)
        if self.opt_config.use_cache:
            cached_result = self.cache_manager.get_cached(cache_key)
            if cached_result is not None:
                return cached_result
        
        try:
            # Process audio based on mode
            if stream:
                result = self._process_streaming(audio)
            else:
                result = self._process_parallel(audio)
            
            # Add diarization if requested
            if diarize:
                result = self._add_diarization(result, audio)
            
            # Cache result
            if self.opt_config.use_cache:
                self.cache_manager.cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            # Handle errors and cleanup
            self.memory_manager.cleanup()
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _process_streaming(self, audio: np.ndarray) -> Dict[str, Any]:
        """Process audio in streaming mode."""
        self.streaming_processor.start_streaming()
        
        try:
            # Split audio into chunks for streaming
            chunks = self._prepare_streaming_chunks(audio)
            
            # Process chunks
            results = []
            for chunk in chunks:
                # Add chunk to streaming buffer
                self.streaming_processor.buffer.put(chunk)
                
                # Get processed result
                result = self._process_chunk(chunk)
                results.append(result)
            
            # Merge streaming results
            return self._merge_streaming_results(results)
            
        finally:
            self.streaming_processor.stop_streaming()
    
    def _process_parallel(self, audio: np.ndarray) -> Dict[str, Any]:
        """Process audio in parallel mode."""
        return self.parallel_processor.process_parallel(audio)
    
    def _add_diarization(self, result: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
        """Add speaker diarization to transcription result."""
        diarization = self.diarizer(audio)
        
        # Merge transcription with diarization
        segments = result["segments"]
        for segment in segments:
            # Find corresponding speaker
            start, end = segment["start"], segment["end"]
            speaker = self._find_speaker(start, end, diarization)
            segment["speaker"] = speaker
        
        result["speaker_segments"] = diarization
        return result
    
    def _prepare_streaming_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """Prepare audio chunks for streaming."""
        chunk_size = int(self.opt_config.chunk_size * self.audio_processor.sample_rate)
        overlap = int(chunk_size * self.opt_config.overlap)
        
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single audio chunk."""
        with self.memory_manager.track_memory():
            # Convert to tensor and optimize for Metal
            tensor = torch.from_numpy(chunk).to(self.device)
            if self.device.type == "mps":
                tensor = tensor.to(dtype=self.opt_config.precision)
            
            # Run inference with optimizations
            with torch.no_grad():
                output = self.metal_optimizer.optimize_inference(self.model, tensor)
            
            return self._process_output(output)
    
    def _merge_streaming_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from streaming processing."""
        # Implement custom merging logic for streaming results
        merged = {
            "text": " ".join(r.get("text", "") for r in results),
            "segments": []
        }
        
        # Merge segments with proper timing
        offset = 0
        for result in results:
            segments = result.get("segments", [])
            for segment in segments:
                segment["start"] += offset
                segment["end"] += offset
                merged["segments"].append(segment)
            offset += self.opt_config.chunk_size - self.opt_config.overlap
        
        return merged
    
    def _find_speaker(self, start: float, end: float, diarization: List[Dict[str, Any]]) -> str:
        """Find the speaker for a given time segment."""
        # Find the speaker with maximum overlap
        max_overlap = 0
        speaker = "UNKNOWN"
        
        for entry in diarization:
            overlap_start = max(start, entry["start"])
            overlap_end = min(end, entry["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                speaker = entry["speaker"]
        
        return speaker
    
    def _get_cache_key(self, audio: np.ndarray) -> str:
        """Generate cache key for audio input."""
        import hashlib
        # Generate hash of audio data and configuration
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        config_str = f"{self.config.model_size}_{self.config.compute_type}"
        return f"{audio_hash}_{config_str}"
    
    def _process_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Process model output into final result."""
        # Implement output processing logic
        return {"output": output}
