import whisperx
import gc
import sys
from dotenv import load_dotenv
import os

# Ensure the site-packages path is in sys.path
site_packages_path = "/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages"
if site_packages_path not in sys.path:
    sys.path.append(site_packages_path)

# Load environment variables
load_dotenv()

# Get the Hugging Face token from the .env file
hf_token = os.getenv("HF_TOKEN")

# Configuration
device = "cuda"
audio_file = "audio.mp3"
batch_size = 16
compute_type = "float16"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])  # before alignment

gc.collect()  # Clear resources

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print(result["segments"])  # after alignment

gc.collect()  # Clear resources

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"])  # segments are now assigned speaker IDs
