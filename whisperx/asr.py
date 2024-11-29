import os
import warnings
from typing import List, Union, Optional, NamedTuple
import ctranslate2
import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment
from .utils.performance import track_performance
from .utils.device import get_device_from_name, move_to_device, is_mps_available
import logging

logger = logging.getLogger(__name__)

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    This version is compatible with faster-whisper 1.1.0.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        if encoder_output is None:
            encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        # Updated for faster-whisper 1.1.0 API
        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
        )

        # Handle the updated result format in 1.1.0
        tokens_batch = [x.sequences_ids[0] for x in result]
        text = tokenizer.tokenizer.decode_batch([tokens for tokens in tokens_batch])
        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        
        features = ctranslate2.StorageView.from_array(features)
        return self.model.encode(features, to_cpu=to_cpu)

class ASRModel:
    def __init__(self, model_name: str, device: str = None, compute_type: str = "float16", 
                 download_root: str = None, local_files_only: bool = False):
        """Initialize ASR model with device support."""
        self.device = get_device_from_name(device)
        
        # Get optimized settings for the device
        settings = optimize_device_settings(self.device)
        
        # Override compute_type if specified
        if compute_type != "default":
            settings['compute_type'] = compute_type
        
        # Handle MPS-specific configurations
        if self.device.type == 'mps':
            if settings['compute_type'] in ["int8", "float16"]:
                warnings.warn(f"{settings['compute_type']} not supported on MPS. Using float32.")
                settings['compute_type'] = "float32"
            # For MPS, we need to use CPU for CTranslate2 but keep tensors on MPS
            self._ct2_device = "cpu"
            print(f"Using CPU ({settings['cpu_threads']} threads) for CTranslate2 backend with tensors on MPS")
        else:
            self._ct2_device = self.device.type
        
        self.compute_type = settings['compute_type']
        self.model = self._load_model(
            model_name, 
            settings['compute_type'],
            download_root, 
            local_files_only,
            cpu_threads=settings['cpu_threads']
        )

    def _load_model(self, model_name: str, compute_type: str, 
                   download_root: str, local_files_only: bool,
                   cpu_threads: int = 4) -> "WhisperModel":
        """Load the model with appropriate device settings."""
        model = faster_whisper.WhisperModel(
            model_name,
            device=self._ct2_device,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
            cpu_threads=cpu_threads,
            num_workers=cpu_threads if self._ct2_device == "cpu" else 1,
        )
        
        # If using MPS, move the model's tensors to MPS after loading
        if self.device.type == 'mps':
            model = move_to_device(model, self.device)
        
        return model
    
    def transcribe(self, audio: Union[str, np.ndarray], batch_size: int = None, **kwargs) -> dict:
        """Transcribe audio with device-specific optimizations."""
        # Use device-optimized batch size if not specified
        if batch_size is None:
            settings = optimize_device_settings(self.device)
            batch_size = settings['batch_size']
        
        # Ensure audio tensor is on the correct device
        if isinstance(audio, torch.Tensor):
            audio = audio.to(self.device)
        
        # Adjust batch size for MPS
        if self.device.type == 'mps':
            batch_size = min(batch_size, 8)  # MPS performs better with smaller batches
        
        # Transcribe with optimized settings
        return self.model.transcribe(audio, batch_size=batch_size, **kwargs)
    
    @property
    def is_multilingual(self) -> bool:
        """Check if the model is multilingual."""
        return self.model.is_multilingual
    
    def get_device_info(self) -> dict:
        """Get information about the current device configuration."""
        return {
            'device_type': self.device.type,
            'device_index': self.device.index if self.device.type == 'cuda' else None,
            'mps_available': is_mps_available(),
            'compute_type': self.compute_type,
            'ct2_device': self._ct2_device,
            'cpu_threads': torch.get_num_threads(),
        }

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            vad,
            vad_params: dict,
            options : NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            language : Optional[str] = None,
            suppress_numerals: bool = False,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, task=None, chunk_size=30, print_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=language)
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=language)
                
        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            print(f"Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}


    def detect_language(self, audio: np.ndarray):
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language

def get_optimal_device():
    """Determine the optimal device and compute type for the current system."""
    if torch.cuda.is_available():
        return "cuda", "float16"
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) support
        # Currently, CTranslate2 doesn't support MPS directly
        # We'll use CPU with optimized settings for now
        return "cpu", "float32"
    else:
        return "cpu", "int8"

def optimize_device_settings(device):
    if device.type == 'cuda':
        return {
            'compute_type': 'float16',
            'batch_size': 32,
            'cpu_threads': 4
        }
    elif device.type == 'mps':
        return {
            'compute_type': 'float32',
            'batch_size': 8,
            'cpu_threads': 4
        }
    else:
        return {
            'compute_type': 'int8',
            'batch_size': 16,
            'cpu_threads': 4
        }

@track_performance("load_model")
def load_model(
    model_size_or_path: str,
    device: str = "auto",
    compute_type: str = "default",
    download_root: str = None,
    local_files_only: bool = False,
) -> "WhisperModel":
    """Load a Whisper model for inference.
    
    Args:
        model_size_or_path: Size or path of the model to load
        device: Device to use (auto, cpu, cuda, mps)
        compute_type: Type of computation (default, float16, float32, int8)
        download_root: Directory to download model to
        local_files_only: Only use local files
        
    Returns:
        WhisperModel: Loaded model
    """
    if device == "auto":
        device, default_compute = get_optimal_device()
        if compute_type == "default":
            compute_type = default_compute
    
    try:
        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
        )
        return model
    except ValueError as e:
        if "unsupported device mps" in str(e):
            logger.warning("MPS device not supported by CTranslate2, falling back to CPU")
            return load_model(
                model_size_or_path,
                device="cpu",
                compute_type="float32",
                download_root=download_root,
                local_files_only=local_files_only,
            )
        raise
