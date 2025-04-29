<p align="right"><b>Language/Sprache: <a href="./index_en.md">English</a> | <a href="./index.md">Deutsch</a></b></p>

# 📖 WhisperX Adaptive Learning – Documentation

Welcome to the comprehensive documentation for WhisperX Adaptive Learning features!

---

## 🗂️ Table of Contents

- [🚀 Introduction](#introduction)
- [⚙️ Installation & Requirements](installation_en.md) – System requirements and installation instructions
- [⚡ Quickstart](#quickstart)
- [🧠 Adaptive Learning & Speaker Profiles](../README_ADAPTIVE_en.md) – Adaptive features and profile management
- [💻 Command Line Interface (CLI)](cli_en.md) – All CLI commands and examples
- [🐍 Python API](api_en.md) – API reference and examples
- [🔄 Workflows & Best Practices](workflows_en.md) – Practical examples and recommendations
- [🛠️ Troubleshooting & Error Messages](troubleshooting_en.md) – Error analysis and solutions
- [❓ FAQ](faq_en.md) – Frequently asked questions
- [📚 Glossary](glossar_en.md) – Key terms in ASR, AI, and WhisperX
- [🔒 Security & Data Protection](security_en.md) – Best practices and GDPR compliance
- [📝 Changelog](changelog_en.md) – Version history and updates

---

> 🧑‍💻 **Interactive Tutorial:**
> See the [Demo Jupyter Notebook](./demo_notebook.ipynb) for hands-on examples of Adaptive Learning!

<img src="./img/adaptive_learning_flow_en.svg" alt="Adaptive Learning Flow" width="600"/>

---

For details on adaptive learning features and speaker profile management, see [README_ADAPTIVE_en.md](../README_ADAPTIVE_en.md).

## 🚀 Introduction

WhisperX brings adaptive learning, speaker-specific profiles, and continuous improvement to automatic speech recognition. This documentation is for users, developers, and integrators.

---

## Installation & Requirements

See [installation.md](installation.md) for detailed instructions.

---

## Quickstart

1. Create a speaker profile
2. Adapt the model
3. Transcribe

```python
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor

processor = AdaptiveProcessor()
profile = processor.create_voice_profile("sample.wav", "speaker1", language="en")
model = load_model("base")
processor.adapt_to_speaker(profile, model)
result = model.transcribe("audio.wav", speaker_id="speaker1", enhance_audio=True)
print(result["text"])
```

---

## Command Line Interface (CLI)

See [cli.md](cli.md) for all commands and examples.

---

## Python API

See [api.md](api.md) for reference and advanced examples.

---

## Workflows & Best Practices

See [workflows.md](workflows.md) for practical examples and recommendations.

---

## Troubleshooting & Error Messages

See [troubleshooting.md](troubleshooting.md) for error analysis and solutions.

---

## FAQ

See [faq.md](faq.md) for frequently asked questions.

---

## Glossary

See [glossar.md](glossar.md) for definitions of key terms in ASR, AI, and WhisperX.

---

## Security & Data Protection

See [security.md](security.md) for best practices and GDPR compliance.

---

## Changelog

See [changelog.md](changelog.md) for version history and updates.

---
