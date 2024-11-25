# WhisperX Adaptive Learning Features

## Overview
WhisperX now supports adaptive learning and speaker-specific optimization to improve transcription accuracy over time. This system includes:

- **Voice Profiles**: Create and manage speaker-specific profiles
- **Adaptive Learning**: Fine-tune the model for specific speakers
- **Audio Enhancement**: Optimize audio processing for known voices
- **Continuous Improvement**: Learn from corrections and feedback

## Installation

Install additional dependencies for adaptive features:

```bash
pip install -r requirements_adaptive.txt
```

## Usage

### Command Line Interface

Basic usage with speaker adaptation:
```bash
whisperx audio.mp3 --speaker_id "speaker1" --adapt_model --enhance_audio
```

Options:
- `--speaker_id`: Unique identifier for the speaker
- `--adapt_model`: Enable LoRA model adaptation
- `--enhance_audio`: Apply speaker-specific audio enhancement

### NADOO Launchpad Integration

The adaptive features are integrated into NADOO Launchpad through the Speaker Management menu:

1. **Speaker Selection**
   - Click "Speakers" in the menubar
   - Select "Manage Speakers" to open the speaker management window
   - Create new speaker profiles or select existing ones

2. **Profile Management**
   - Create profiles by providing sample audio
   - View and edit existing profiles
   - Delete unused profiles

3. **Adaptation Settings**
   - Enable/disable model adaptation
   - Configure adaptation parameters
   - Set audio enhancement options

4. **Using Speaker Profiles**
   - Select a speaker from the dropdown in the main window
   - Enable "Adapt to Speaker" for better accuracy
   - Use "Enhance Audio" for cleaner input

### Python API

```python
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor

# Initialize
processor = AdaptiveProcessor()

# Create profile
profile = processor.create_voice_profile(
    audio_path="sample.mp3",
    speaker_id="speaker1",
    language="en"
)

# Transcribe with adaptation
model = load_model("base")
processor.adapt_to_speaker(profile, model)

# Enhanced transcription
result = model.transcribe(
    "speech.mp3",
    speaker_id="speaker1",
    enhance_audio=True
)
```

## Technical Details

### Voice Profiles
Profiles are stored in `~/.cache/whisperx/voice_profiles` and contain:
- Speaker embeddings
- Audio samples
- Language preferences
- Adaptation state

### Model Adaptation
Uses LoRA (Low-Rank Adaptation) to efficiently adapt the model:
- Small trainable matrices
- Quick adaptation
- Minimal memory footprint

### Audio Enhancement
Speaker-aware processing pipeline:
- Voice activity detection
- Speaker verification
- Adaptive noise reduction
- Volume normalization

## Contributing

To contribute to the adaptive learning features:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
Same as WhisperX main license
