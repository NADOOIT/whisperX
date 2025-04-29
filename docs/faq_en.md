# ❓ FAQ – Frequently Asked Questions

Find answers to the most important questions about WhisperX Adaptive Learning.

---

## 🚀 Installation & Getting Started

**How do I install WhisperX?**

```bash
pip install whisperx
```

> 💡 **Tip:** See [Installation](./installation_en.md) for details and troubleshooting!

---

## 🧑‍💻 Usage

**How do I start a transcription?**

```bash
whisperx audio.mp3 --speaker_id "user1" --adapt_model
```

**How do I create a speaker profile?**

```python
from whisperx.adaptive import AdaptiveProcessor
proc = AdaptiveProcessor()
profile = proc.create_voice_profile("audio.wav", "user1")
```

---

## 🛠️ Problems & Solutions

**The model does not recognize the speaker correctly – what can I do?**
> Check if the profile is up to date and contains enough speech samples.

**Error on startup?**
> See [Troubleshooting](./troubleshooting_en.md) and check dependencies.

---

## 📞 Support

More questions? Open an [issue on GitHub](https://github.com/NADOOIT/whisperX/issues)!

## 🤔 General Questions

### How many speaker profiles can I create?
There is no hard limit, but many profiles may affect system performance.

### Can I transfer profiles between systems?
Yes, copy the profile folder (`~/.cache/whisperx/voice_profiles`) to the target system.

### What if transcription quality is poor?
> Check audio quality
> Collect more speech samples
> Use the feedback loop to improve profiles
- Check audio quality
- Collect more speech samples
- Use the feedback loop to improve profiles

## Are my data stored securely?
See [Security & Data Protection](./security_en.md)

## How do I report bugs or give feedback?
Open a [GitHub Issue](https://github.com/NADOOIT/whisperX/issues) or email support@nadoo.de

## Does WhisperX support multiple languages?
Yes, specify the language when creating a profile (e.g., `language="en"`).

## Can I integrate WhisperX into my own tools?
Yes, via the Python API and CLI.
