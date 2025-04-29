# 🐍 Python API

Hier findest du eine Übersicht und Beispiele für die Nutzung der WhisperX Python API – inklusive adaptiver Features.

---

## ⚡ Schnellstart (API)

```python
from whisperx import load_model
model = load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])
```

> 💡 **Tipp:** Für fortgeschrittene Nutzung und adaptive Features siehe unten!

---

## 🧠 Adaptive Features per API

### Profil anlegen
```python
from whisperx.adaptive import AdaptiveProcessor
proc = AdaptiveProcessor()
profile = proc.create_voice_profile("sample.wav", "speaker1", language="de")
```

processor = AdaptiveProcessor()
profile = processor.create_voice_profile("sample.wav", "speaker1", language="de")
model = load_model("base")
processor.adapt_to_speaker(profile, model)
result = model.transcribe("audio.wav", speaker_id="speaker1", enhance_audio=True)
print(result["text"])
```

## Wichtige Klassen & Methoden

### AdaptiveProcessor
- `create_voice_profile(audio_path, speaker_id, language)`: Erstellt ein Sprecherprofil
- `adapt_to_speaker(profile, model)`: Passt das Modell an

### Model
- `transcribe(audio_path, speaker_id, enhance_audio)`: Transkribiert Audio unter Berücksichtigung des Profils

## Erweiterte Nutzung
- Profile exportieren/importieren
- Mehrsprachige Profile
- Automatisierte Workflows

Weitere Beispiele und Details findest du in den anderen Doku-Kapiteln.
