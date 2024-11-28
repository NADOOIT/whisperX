import os
import whisperx
import torch
from pathlib import Path

# Input directory path
input_dir = "/Users/christophbackhaus/nadoo_launchpad/dir/converter/in"

# Find the first WAV file in the input directory
wav_files = list(Path(input_dir).glob("*.wav"))
if not wav_files:
    print("No WAV files found in the input directory")
    exit(1)

wav_file = str(wav_files[0])
print(f"Found WAV file: {wav_file}")

# Determine device
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
else:
    device = "cpu"
    compute_type = "int8"

print(f"Using device: {device}")

try:
    # Load model
    print("Loading model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # Load and transcribe audio
    print("Loading audio...")
    audio = whisperx.load_audio(wav_file)
    
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=8 if device == "cpu" else 32)

    print("\nTranscription result:")
    print("Language detected:", result["language"])
    print("\nSegments:")
    for segment in result["segments"]:
        print(f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    raise
