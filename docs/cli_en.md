# 💻 Command Line Interface (CLI)

This page documents all commands and options for using WhisperX from the command line.

---

## ⚡ Quickstart (CLI)

```bash
whisperx audio.mp3 --speaker_id "speaker1" --adapt_model --enhance_audio
```

> 💡 **Tip:** The most important parameters are shown in the example above. See below for more options!

---

## 📝 Key Options

- `--speaker_id <ID>`: Unique identifier for the speaker
- `--adapt_model`: Enables model adaptation (Adaptive Learning)
- `--enhance_audio`: Apply audio enhancement
- `--language <code>`: Specify language (e.g., `en`, `de`)

---

## 📚 Examples

**Transcription with adaptation:**
```bash
whisperx meeting.wav --speaker_id "boss" --adapt_model --enhance_audio
```

**Audio enhancement only:**
```bash
whisperx call.wav --speaker_id "customer" --enhance_audio
```

**Batch processing:**
```bash
for f in *.mp3; do whisperx "$f" --speaker_id "user1" --adapt_model; done
```

---

## 🆘 Troubleshooting & Tips

- For profile issues: use `--reset_profiles`
- For debug logs: add `--verbose`

> ℹ️ **More help:** See [FAQ](./faq_en.md) and [Troubleshooting](./troubleshooting_en.md)
