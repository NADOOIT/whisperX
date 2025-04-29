# WhisperX Adaptive Learning Features

👉 **Die vollständige, aktuelle Dokumentation findest du jetzt im Verzeichnis [`docs/`](./docs/index.md).**


## Inhaltsverzeichnis

- [Überblick](#overview)
- [Installation](#installation)
- [Anwendung](#anwendung)
- [NADOO Launchpad Integration](#nadoo-launchpad-integration)
- [Python API](#python-api)
- [Technische Details](#technische-details)
- [Häufige Fragen (FAQ)](#häufige-fragen-faq)
- [Beitrag leisten](#beitrag-leisten)
- [Lizenz](#lizenz)
- [Systemvoraussetzungen](#systemvoraussetzungen)
- [Bekannte Probleme & Workarounds](#bekannte-probleme--workarounds)
- [Kontakt & Support](#kontakt--support)
- [Changelog (Adaptive Features)](#changelog-adaptive-features)
- [Komplettes Beispielprojekt](#komplettes-beispielprojekt)
- [Datenschutz & Sicherheit](#datenschutz--sicherheit)
- [Mehrsprachige Nutzung](#mehrsprachige-nutzung)
- [Weiterführende Ressourcen](#weiterführende-ressourcen)
- [API-Referenz (Auszug)](#api-referenz-auszug)
- [Best Practices für die Modellanpassung](#best-practices-für-die-modellanpassung)
- [Adaptive vs. Standard-Transkription](#adaptive-vs-standard-transkription)
- [Feedback & Verbesserungsvorschläge](#feedback--verbesserungsvorschläge)
- [Getting Started](#getting-started)
- [Erweiterte Workflows](#erweiterte-workflows)
- [Fehlermeldungen & Lösungen](#fehlermeldungen--lösungen)
- [Sicherheit im Produktivbetrieb](#sicherheit-im-produktivbetrieb)
- [Glossar (erweitert)](#glossar-erweitert)


## Overview
WhisperX bietet jetzt adaptive Lernfunktionen und sprecherspezifische Optimierungen, um die Transkriptionsgenauigkeit kontinuierlich zu verbessern. Das System umfasst:

- **Voice Profiles**: Erstellung und Verwaltung von sprecherspezifischen Profilen
- **Adaptive Learning**: Feinabstimmung des Modells für einzelne Sprecher
- **Audio Enhancement**: Optimierte Audioverarbeitung für bekannte Stimmen
- **Continuous Improvement**: Fortlaufendes Lernen aus Korrekturen und Nutzerfeedback

## Installation

Für die adaptiven Funktionen müssen zusätzliche Abhängigkeiten installiert werden:

```bash
pip install -r requirements_adaptive.txt
```

## Anwendung

### Automatische Qualitätskontrolle & Testdaten

Das System unterstützt jetzt einen kontinuierlichen Qualitäts-Workflow:

- **Testdaten-Upload:** Im Webinterface können Sie kurze, gelabelte Audiodateien (mit Transkript) als Testdaten hochladen.
- **Automatische Evaluierung:** Nach jedem Trainingsprozess wird das aktuelle Modell automatisch auf alle Testdaten angewendet. Die generierten Transkripte werden mit den Ziel-Transkripten verglichen (z. B. WER/Accuracy).
- **Live-Metriken:** Die Ergebnisse (z. B. Word Error Rate vor/nach Training) werden direkt im Webinterface angezeigt. Verbesserungen oder Verschlechterungen sind sofort sichtbar.
- **Automatisches Entfernen schlechter Trainingsdaten:** Wird die Qualität nach Training schlechter, entfernt das System automatisch die problematische Datei aus den Trainingsdaten und markiert sie zur späteren Korrektur.
- **Iterative Verbesserung:** Schlechte Audios können später korrigiert und erneut als hochwertige Trainingsdaten verwendet werden.

**Hinweis:** Diese Funktionen sind vorrangig im Webinterface verfügbar und sorgen für kontinuierliche Verbesserung und Transparenz im Speaker-Training.

### Kommandozeile (CLI)

Beispiel für die Nutzung mit Sprecheranpassung:
```bash
whisperx audio.mp3 --speaker_id "speaker1" --adapt_model --enhance_audio
```

**Optionen:**
- `--speaker_id`: Eindeutige Kennung für den Sprecher (z.B. "speaker1")
- `--adapt_model`: Aktiviert die LoRA-Modellanpassung für den ausgewählten Sprecher
- `--enhance_audio`: Wendet eine sprecherspezifische Audio-Optimierung an

**Best Practice:**
- Verwenden Sie für jeden Sprecher eine eigene ID und sammeln Sie möglichst klare Sprachbeispiele für optimale Ergebnisse.
- Die Modellanpassung ist besonders effektiv bei wiederkehrenden Sprechern oder längeren Projekten.
- Feedback und Korrekturen helfen dem System, sich weiter zu verbessern.

### NADOO Launchpad Integration

Die adaptiven Funktionen sind im NADOO Launchpad über das Menü "Speaker Management" integriert:

1. **Sprecherauswahl**
   - Klicken Sie auf "Speakers" in der Menüleiste
   - Wählen Sie "Manage Speakers", um das Management-Fenster zu öffnen
   - Erstellen Sie neue Sprecherprofile oder wählen Sie bestehende aus

2. **Profilverwaltung**
   - Profile können durch Bereitstellung von Beispiel-Audio erstellt werden
   - Bestehende Profile lassen sich anzeigen und bearbeiten
   - Nicht mehr benötigte Profile können gelöscht werden

3. **Anpassungseinstellungen**
   - Modellanpassung aktivieren/deaktivieren
   - Anpassungsparameter konfigurieren
   - Audio-Optimierung einstellen

4. **Verwendung von Sprecherprofilen**
   - Wählen Sie einen Sprecher im Hauptfenster aus
   - Aktivieren Sie "Adapt to Speaker" für höhere Genauigkeit
   - Nutzen Sie "Enhance Audio" für klarere Eingaben

### Python API

```python
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor

# Initialisierung
processor = AdaptiveProcessor()

# Profil erstellen
profile = processor.create_voice_profile(
    audio_path="sample.mp3",
    speaker_id="speaker1",
    language="en"
)

# Modell an Sprecher anpassen
model = load_model("base")
processor.adapt_to_speaker(profile, model)

# Verbesserte Transkription
result = model.transcribe(
    "speech.mp3",
    speaker_id="speaker1",
    enhance_audio=True
)
```

**Hinweise:**
- Die API erlaubt eine flexible Integration in eigene Workflows.
- Für beste Ergebnisse sollten Sprachproben möglichst rauscharm sein.
- Die Methoden sind dokumentiert; weitere Details finden sich im Quellcode.

## Technische Details

### Voice Profiles
Profile werden unter `~/.cache/whisperx/voice_profiles` gespeichert und enthalten:
- Sprecher-Embeddings
- Audiodateien
- Spracheinstellungen
- Anpassungsstatus

### Modellanpassung (LoRA)
Zur effizienten Anpassung wird LoRA (Low-Rank Adaptation) eingesetzt:
- Kleine, trainierbare Matrizen
- Schnelle Anpassung
- Geringer Speicherbedarf

### Audio Enhancement
Sprecherbezogene Audiopipeline:
- Spracherkennung (Voice Activity Detection)
- Sprecherverifikation
- Adaptive Rauschunterdrückung
- Lautstärkenormalisierung

## Häufige Fragen (FAQ)

**Wie viele Sprecherprofile kann ich anlegen?**
- Es gibt keine feste Begrenzung, aber die Systemleistung kann bei sehr vielen Profilen beeinträchtigt werden.

**Kann ich Profile zwischen Systemen übertragen?**
- Ja, kopieren Sie den Profilordner auf das Zielsystem.

**Was tun bei schlechter Transkriptionsqualität?**
- Prüfen Sie die Audioqualität und sammeln Sie ggf. weitere Sprachbeispiele.
- Nutzen Sie die Feedback-Funktion, um Korrekturen einzureichen.

## Beitrag leisten

So können Sie zu den adaptiven Funktionen beitragen:

1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch
3. Reichen Sie einen Pull Request ein

## Lizenz
Es gilt die Hauptlizenz von WhisperX.

---

## Weitere Anwendungsbeispiele

### Mehrere Sprecher in einem Projekt
Wenn Sie mehrere Sprecherprofile in einem Projekt verwenden, können Sie für jede Audiodatei das passende Profil auswählen:

```bash
whisperx audio1.mp3 --speaker_id "sprecherA" --adapt_model
whisperx audio2.mp3 --speaker_id "sprecherB" --adapt_model
```

### Batch-Transkription mit Anpassung
Für die Verarbeitung vieler Dateien mit jeweils eigenem Sprecherprofil empfiehlt sich ein Skript:

```python
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor

files_and_speakers = [
    ("audio1.mp3", "sprecherA"),
    ("audio2.mp3", "sprecherB"),
]

model = load_model("base")
processor = AdaptiveProcessor()

for audio, speaker in files_and_speakers:
    profile = processor.create_voice_profile(audio, speaker)
    processor.adapt_to_speaker(profile, model)
    result = model.transcribe(audio, speaker_id=speaker, enhance_audio=True)
    print(result["text"])
```

### Feedback-Loop für kontinuierliche Verbesserung
Nutzen Sie Korrekturen aus dem Alltag, um Profile weiter zu optimieren. Sammeln Sie korrigierte Transkripte und führen Sie diese regelmäßig dem System zu.

---

## Troubleshooting

### Profilerstellung schlägt fehl
- **Mögliche Ursache:** Audiodatei ist zu kurz oder von schlechter Qualität
- **Lösung:** Längere oder klarere Sprachproben verwenden

### Modellanpassung funktioniert nicht
- **Mögliche Ursache:** Fehlende Abhängigkeiten oder falsche Modellversion
- **Lösung:** Prüfen Sie die Installation und Kompatibilität der Pakete

### Transkriptionsergebnis ist ungenau
- **Mögliche Ursache:** Sprecherprofil nicht optimal oder Audio verrauscht
- **Lösung:** Mehr Sprachproben sammeln und ggf. Audioqualität verbessern

### Fehlermeldung: "Profile not found"
- **Lösung:** Überprüfen Sie den Pfad zu den Profilen und die Schreibrechte im Profilverzeichnis

---

## Integrationstipps

- **API-Rate-Limits:** Bei paralleler Verarbeitung vieler Dateien auf API-Limits achten
- **Performance:** Für große Projekte empfiehlt sich die Nutzung von GPU-Beschleunigung
- **Profil-Backup:** Sprecherprofile regelmäßig sichern, insbesondere bei produktivem Einsatz
- **Automatisierung:** Die Python-API lässt sich gut in bestehende Datenpipelines integrieren

---

## Glossar

- **LoRA (Low-Rank Adaptation):** Methode zur effizienten Anpassung neuronaler Netze mit geringem Speicherbedarf
- **Speaker Embedding:** Vektor, der charakteristische Merkmale einer Stimme repräsentiert
- **Voice Activity Detection (VAD):** Algorithmus zur Erkennung von Sprachabschnitten in Audiodaten
- **Speaker Verification:** Überprüfung, ob eine Stimme zu einem bestimmten Profil passt
- **Adaptive Noise Reduction:** Dynamische Rauschunterdrückung, die sich an den Sprecher anpasst
- **Enhance Audio:** Funktion zur Verbesserung der Audioqualität für bessere Transkriptionsergebnisse

---


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

---

## Systemvoraussetzungen

- **Betriebssysteme:** Linux, macOS, Windows (getestet auf aktuellen Versionen)
- **Python-Version:** 3.8 oder neuer empfohlen
- **Hardware:**
  - CPU: Mindestens 4 Kerne empfohlen
  - GPU: Für große Modelle und schnelle Anpassung empfohlen (CUDA-kompatibel)
- **Abhängigkeiten:** Siehe `requirements_adaptive.txt`
- **Speicher:** Je nach Modellgröße und Anzahl der Profile ausreichend RAM einplanen

---

## Bekannte Probleme & Workarounds

- **Problem:** Profilerstellung schlägt bei sehr kurzen Audiodateien fehl
  - **Workaround:** Mindestens 10 Sekunden klare Sprachaufnahme verwenden

- **Problem:** GPU wird nicht erkannt
  - **Workaround:** Prüfen Sie CUDA-Installation und GPU-Treiber; ggf. auf CPU-Modus ausweichen

- **Problem:** Lange Ladezeiten bei vielen Profilen
  - **Workaround:** Nicht benötigte Profile regelmäßig löschen oder archivieren

- **Problem:** Fehler "Permission denied" beim Zugriff auf Profilordner
  - **Workaround:** Schreibrechte für das Verzeichnis `~/.cache/whisperx/voice_profiles` prüfen

Bitte melden Sie weitere Probleme über GitHub-Issues (siehe unten).

---

## Kontakt & Support

- **GitHub-Issues:** [https://github.com/NADOOIT/whisperX/issues](https://github.com/NADOOIT/whisperX/issues)
- **E-Mail:** support@nadoo.de (für kommerzielle Anfragen)
- **Community:** Diskussionen und Erfahrungsaustausch im GitHub-Forum oder über die Projektseite

Bitte geben Sie bei Supportanfragen möglichst genaue Informationen zu System, WhisperX-Version und Fehlermeldung an.

---

## Changelog (Adaptive Features)

- **v1.2.0** (2025-04):
  - Verbesserte Dokumentation, neue Troubleshooting- und FAQ-Abschnitte
  - Batch-Transkription und Feedback-Loop-Beispiele ergänzt
- **v1.1.0** (2025-03):
  - NADOO Launchpad Integration
  - Erweiterte Python-API für Profilmanagement
- **v1.0.0** (2025-02):
  - Einführung der adaptiven Lernfunktionen und Sprecherprofile

---

Same as WhisperX main license

---

## Komplettes Beispielprojekt

Ein vollständiger Workflow für ein adaptives Transkriptionsprojekt:

```python
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor

# Schritt 1: Sprecherprofil erstellen
profile = AdaptiveProcessor().create_voice_profile(
    audio_path="samples/speaker1_intro.wav",
    speaker_id="speaker1",
    language="de"
)

# Schritt 2: Modell laden und anpassen
model = load_model("base")
AdaptiveProcessor().adapt_to_speaker(profile, model)

# Schritt 3: Transkription mit Anpassung
result = model.transcribe(
    "samples/speaker1_meeting.wav",
    speaker_id="speaker1",
    enhance_audio=True
)
print(result["text"])
```

---

## Datenschutz & Sicherheit

- Audiodaten und Sprecherprofile enthalten personenbezogene Informationen. Bewahren Sie diese sicher auf und geben Sie sie nicht an Unbefugte weiter.
- Nutzen Sie verschlüsselte Speichermedien, wenn sensible Daten verarbeitet werden.
- Löschen Sie nicht mehr benötigte Profile und Audiodateien regelmäßig.
- Beachten Sie ggf. geltende Datenschutzgesetze (z.B. DSGVO).

---

## Mehrsprachige Nutzung

- WhisperX unterstützt viele Sprachen. Geben Sie beim Erstellen eines Profils die korrekte Sprache an (`language="en"`, `language="de"` etc.).
- Für mehrsprachige Sprecher können mehrere Profile (pro Sprache) angelegt werden.
- Die automatische Spracherkennung kann bei gemischten Audiodaten hilfreich sein, ist aber bei klaren Sprachangaben meist präziser.

---

## Weiterführende Ressourcen

- [WhisperX GitHub](https://github.com/NADOOIT/whisperX)
- [Original Whisper (OpenAI)](https://github.com/openai/whisper)
- [LoRA Paper (arXiv)](https://arxiv.org/abs/2106.09685)
- [DSGVO Informationen](https://www.bfdi.bund.de/DE/Datenschutz/datenschutz-node.html)
- [Audio-Datenschutz in der Praxis (externer Leitfaden)](https://www.datenschutz.org/audioaufnahmen/)
