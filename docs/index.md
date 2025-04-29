<p align="right"><b>Sprache/Language: <a href="./index.md">Deutsch</a> | <a href="./index_en.md">English</a></b></p>

# 📖 WhisperX Adaptive Learning – Dokumentation

Willkommen zur umfassenden Dokumentation der adaptiven Lernfunktionen von WhisperX!

---

## 🗂️ Inhaltsverzeichnis

- [🚀 Einleitung](#einleitung)
- [⚙️ Installation & Voraussetzungen](installation.md) – Systemvoraussetzungen und Installationsanleitung
- [⚡ Schnellstart](#schnellstart)
- [💻 Kommandozeile (CLI)](cli.md) – Alle CLI-Befehle und Beispiele
- [🐍 Python API](api.md) – API-Referenz und Beispiele
- [🧑‍💻 Profile & Anpassung](#profile--anpassung)
- [🔄 Workflows & Best Practices](workflows.md) – Praxisbeispiele und empfohlene Abläufe
- [🛠️ Troubleshooting & Fehlermeldungen](troubleshooting.md) – Fehleranalyse und Lösungen
- [❓ FAQ](faq.md) – Häufige Fragen und Antworten
- [📚 Glossar](glossar.md) – Begriffe rund um ASR, KI und WhisperX
- [🔒 Sicherheit & Datenschutz](security.md) – Hinweise für sicheren und DSGVO-konformen Einsatz
- [💬 Kontakt & Support](#kontakt--support)
- [📝 Changelog](changelog.md) – Versionshistorie und Neuerungen

---

> 🧑‍💻 **Interaktives Tutorial:**
> Siehe das [Demo Jupyter Notebook](./demo_notebook.ipynb) für praktische Beispiele zu Adaptive Learning!

<img src="./img/adaptive_learning_flow_de.svg" alt="Adaptive Learning Ablauf" width="600"/>

---

## 🚀 Einleitung

WhisperX erweitert die Transkription um adaptive Lernfunktionen, sprecherspezifische Profile und kontinuierliche Verbesserung. Diese Dokumentation richtet sich an Anwender, Entwickler und Integratoren.

---

## ⚙️ Installation & Voraussetzungen

- Siehe [requirements_adaptive.txt](../requirements_adaptive.txt)
- Empfohlen: Python >= 3.8, Linux/macOS/Windows, optional CUDA-fähige GPU

```bash
pip install -r requirements_adaptive.txt
```

---

## Schnellstart

1. Profil erstellen
2. Modell anpassen
3. Transkribieren

```python
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor

processor = AdaptiveProcessor()
profile = processor.create_voice_profile("sample.wav", "speaker1", language="de")
model = load_model("base")
processor.adapt_to_speaker(profile, model)
result = model.transcribe("audio.wav", speaker_id="speaker1", enhance_audio=True)
print(result["text"])
```

---

## Kommandozeile (CLI)

```bash
whisperx audio.mp3 --speaker_id "speaker1" --adapt_model --enhance_audio
```

**Optionen:**
- `--speaker_id`: Eindeutige Kennung für den Sprecher
- `--adapt_model`: Aktiviert die Modellanpassung
- `--enhance_audio`: Führt Audio-Optimierung durch

---

## Python API

... (ausführliche API-Doku mit Beispielen, siehe README_ADAPTIVE.md)

---

## Profile & Anpassung

- Wie werden Profile erstellt, gespeichert und verwaltet?
- Tipps für optimale Sprachproben
- LoRA-Adaptation erklärt

---

## Workflows & Best Practices

- Multi-Sprecher-Projekte
- Batch-Transkription
- Feedback-Loop
- Integration in Pipelines

---

## Troubleshooting & Fehlermeldungen

| Fehlermeldung | Ursache | Lösung |
|---|---|---|
| Profile not found | Profil fehlt | Profilnamen prüfen, neu anlegen |
| CUDA device not found | Keine GPU | CUDA/Treiber prüfen, CPU-Modus |
| ... | ... | ... |

---

## FAQ

- Wie viele Profile?
- Kann ich Profile übertragen?
- ...

---

## Glossar

- LoRA, Embedding, VAD, ...

---

## Sicherheit & Datenschutz

- Empfehlungen für sicheren Umgang mit Daten

---

## Kontakt & Support

- [GitHub Issues](https://github.com/NADOOIT/whisperX/issues)
- support@nadoo.de

---

## Changelog

- Siehe [Changelog-Abschnitt in README_ADAPTIVE.md](../README_ADAPTIVE.md)

---
