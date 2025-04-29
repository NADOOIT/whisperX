# 🐍 Python API

This page provides an overview and examples for using the WhisperX Python API – including adaptive features.

---

## ⚡ Quickstart (API)

```python
from whisperx import load_model
model = load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])
```

> 💡 **Tip:** For advanced usage and adaptive features, see below!

---

## 🧠 Adaptive Features via API

### Create a Speaker Profile
```python
from whisperx.adaptive import AdaptiveProcessor
proc = AdaptiveProcessor()
profile = proc.create_voice_profile("sample.wav", "speaker1", language="en")
```

### Adapt the Model
```python
from whisperx import load_model
model = load_model("base")
proc.adapt_to_speaker(profile, model)
```

### Transcribe with Adaptation
```python
result = model.transcribe("audio.wav", speaker_id="speaker1", enhance_audio=True)
print(result["text"])
```

---

## 🆘 Tips & Best Practices

- Regularly update profiles with new speech samples
- For debugging: use `verbose=True` in your calls
- More on errors: [Troubleshooting](./troubleshooting_en.md)
