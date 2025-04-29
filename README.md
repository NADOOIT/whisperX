<h1 align="center">WhisperX</h1>

<p align="center">
  <a href="https://github.com/m-bain/whisperX/stargazers">
    <img src="https://img.shields.io/github/stars/m-bain/whisperX.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/m-bain/whisperX/issues">
        <img src="https://img.shields.io/github/issues/m-bain/whisperx.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/m-bain/whisperX/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/m-bain/whisperX.svg"
             alt="GitHub license">
  </a>
  <a href="https://arxiv.org/abs/2303.00747">
        <img src="http://img.shields.io/badge/Arxiv-2303.00747-B31B1B.svg"
             alt="ArXiv paper">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2Fm-bain%2FwhisperX">
  <img src="https://img.shields.io/twitter/url/https/github.com/m-bain/whisperX.svg?style=social" alt="Twitter">
  </a>      
</p>


<img width="1216" align="center" alt="whisperx-arch" src="figures/pipeline.png">

---

---

> ⏱️ **Deine ersten 5 Minuten mit WhisperX**
> - **Schnellstart:** Folge der Box "Erste Schritte in 2 Minuten" unten!
> - **Häufige Anfängerfragen:** [FAQ & Troubleshooting](#❓-faq-häufige-fragen--stolpersteine)
> - **Direkt zu Beispielen:** [Quickstart](#🚀-quickstart-example), [Diarization](#diarization-beispiel), [VAD](#vad-beispiel)
>
> 💬 **Feedback zur Doku?** [Hier direkt Rückmeldung geben!](https://github.com/m-bain/whisperx/discussions)

---

<div style="background-color:#e6f0ff; padding:12px; border-radius:8px; margin-bottom:16px;">
🔗 <strong>Schritt-für-Schritt-Guides & Community-Events:</strong> <a href="https://github.com/m-bain/whisperx/discussions">Hier findest du Tutorials, Guides und aktuelle Events!</a>
</div>

## 🚦 Onboarding: Für wen ist WhisperX?

> **Bist du ...**
> - 👩‍🏫 **Lehrkraft?** [Direkt zu Tipps & Beispielen für Bildung](#2-praxis--beispiele)
> - 👩‍💻 **Entwickler:in?** [Direkt zur Integration & API](#2-praxis--beispiele)
> - 🤝 **Team/Organisation?** [Direkt zu Team-Workflows & Community](#3-community--inspiration)
> - 🏠 **Privatnutzer:in?** [Direkt zu Praxisbeispielen & Inspiration](#2-praxis--beispiele)
> - 👶 **Einsteiger:in?** [Direkt zum Schnellstart & FAQ](#1-einstieg--quickstart)

---

> 🟢 **Erste Schritte in 2 Minuten:**
> 1. Installiere WhisperX (siehe unten)
> 2. Kopiere diesen Befehl ins Terminal:
>    ```bash
>    python -m whisperx audio.mp3 --output_dir results/
>    ```
> 3. Öffne den Ordner `results/` – dein Transkript ist fertig!
> 4. Bei Problemen: Siehe FAQ/Troubleshooting oder frage die Community.
>
> 📺 **Video-Tutorials & bebilderte Anleitungen:** [WhisperX Video-Anleitungen](https://www.youtube.com/results?search_query=whisperx)

---


<!-- <p align="left">Whisper-Based Automatic Speech Recognition (ASR) with improved timestamp accuracy + quality via forced phoneme alignment and voice-activity based batching for fast inference.</p> -->


<!-- <h2 align="left", id="what-is-it">What is it 🔎</h2> -->


This repository provides fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend, requires <8GB gpu memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio) (speaker ID labels) 
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation



**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

**Voice Activity Detection (VAD)** is the detection of the presence or absence of human speech.

**Speaker Diarization** is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker.

<h2 align="left", id="highlights">New🚨</h2>

- 1st place at [Ego4d transcription challenge](https://eval.ai/web/challenges/challenge-page/1637/leaderboard/3931/WER)  🏆
- _WhisperX_ accepted at INTERSPEECH 2023 
- v3 transcript segment-per-sentence: using nltk sent_tokenize for better subtitlting & better diarization
- v3 released, 70x speed-up open-sourced. Using batched whisper with [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend!
- v2 released, code cleanup, imports whisper library VAD filtering is now turned on by default, as in the paper.
- Paper drop🎓👨‍🏫! Please see our [ArxiV preprint](https://arxiv.org/abs/2303.00747) for benchmarking and details of WhisperX. We also introduce more efficient batch inference resulting in large-v2 with *60-70x REAL TIME speed.

<h2 align="left" id="setup">Setup ⚙️</h2>
Tested for PyTorch 2.0, Python 3.10 (use other versions at your own risk!)

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html).


### 1. Create Python3.10 environment

`conda create --name whisperx python=3.10`

`conda activate whisperx`

> ✅ **Checkliste Installation:**
> - [ ] Python 3.10+ installiert?
> - [ ] Virtuelle Umgebung aktiv?
> - [ ] `pip install ...` ohne Fehler durchgelaufen?
> - [ ] Health-Check (`pytest tests/test_installation.py`) bestanden?
> - [ ] Bei Problemen: [FAQ/Troubleshooting](#⚠️-troubleshooting--health-check) nutzen oder [Community fragen](https://github.com/m-bain/whisperx/discussions).

### 2. Install PyTorch, e.g. for Linux and Windows CUDA11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> 💡 **Tipp:** Für andere Plattformen oder CUDA-Versionen siehe die [offizielle PyTorch Installationsanleitung](https://pytorch.org/get-started/locally/).

---

### 3. Install WhisperX and dependencies

```bash
pip install git+https://github.com/m-bain/whisperx.git
```

> 💡 **Hinweis:** Für optimale Performance wird empfohlen, ein GPU-beschleunigtes System zu verwenden.

---

### 4. (Optional) Install additional components

- **Speaker Diarization:**  
  Für Speaker Diarization benötigst du zusätzliche Pakete:
  ```bash
  pip install pyannote.audio
  ```
  Siehe auch die [pyannote.audio Dokumentation](https://github.com/pyannote/pyannote-audio) für weitere Details.

- **VAD (Voice Activity Detection):**  
  VAD ist standardmäßig aktiviert, kann aber bei Bedarf angepasst werden.

---

### 5. Test your installation

```bash
python -m whisperx --help
```

Wenn die Hilfe angezeigt wird, ist WhisperX korrekt installiert!

---

## 🚀 Quickstart Example

Transcribe an audio file (e.g. `audio.mp3`) with WhisperX:

```bash
python -m whisperx audio.mp3 --output_dir results/
```

<div style="background-color:#e6ffe6; padding:10px; border-radius:8px;">
🧪 <strong>Teste es direkt:</strong> Lade <a href="#">sample.mp3</a> herunter und probiere:

```bash
python -m whisperx sample.mp3 --output_dir results/
```
Das Ergebnis findest du im <code>results/</code>-Ordner.
</div>

> 🖼️ <strong>Beispiel-Output:</strong>
> 
> ```
> [00:00.000 --> 00:04.000] Hallo und willkommen zu WhisperX!
> [00:04.000 --> 00:08.000] Dies ist ein Beispiel-Transkript.
> ```

- The output will be saved in the specified directory (default: `results/`).
- You can specify the model, language, and other parameters. For all options, run:
  ```bash
  python -m whisperx --help
  ```

<div style="background-color:#e6f0ff; padding:10px; border-radius:8px;">
💡 <strong>Tipp:</strong> Für beste Ergebnisse nutze ein passendes Modell (z.B. <code>--model large-v2</code>) und prüfe die Spracheinstellung (z.B. <code>--language de</code>).
</div>

> 🚀 **Du bist startklar – probiere es aus und teile dein Feedback in der [Community](https://github.com/m-bain/whisperx/discussions)!**

---

## 🆘 Schnelle Hilfe bei Problemen

<div style="background-color:#fffbe6; padding:10px; border-radius:8px;">
⚠️ <strong>Installation schlägt fehl?</strong><br>
- Prüfe Python-Version (3.10+), virtuelle Umgebung, und ob alle <code>pip install ...</code>-Befehle fehlerfrei durchlaufen.<br>
- Siehe <a href="#⚠️-troubleshooting--health-check">Troubleshooting</a> für Details.
</div>

<div style="background-color:#fffbe6; padding:10px; border-radius:8px;">
⚠️ <strong>GPU wird nicht erkannt?</strong><br>
- Prüfe mit <code>nvidia-smi</code>, ob deine NVIDIA-GPU korrekt installiert ist.<br>
- Auf macOS ist nur CPU-Modus möglich.
</div>

<div style="background-color:#fffbe6; padding:10px; border-radius:8px;">
⚠️ <strong>Datei nicht gefunden?</strong><br>
- Pfad und Dateinamen genau prüfen, ggf. per Drag & Drop ins Terminal.<br>
- Mit <code>ls</code> kontrollieren, ob die Datei im aktuellen Verzeichnis liegt.
</div>

<div style="background-color:#fffbe6; padding:10px; border-radius:8px;">
⚠️ <strong>Permission denied?</strong><br>
- Schreibrechte für den Zielordner prüfen (<code>ls -l</code>), ggf. <code>chmod</code> verwenden.
</div>

---

## ✅ Health-Check testen

<div style="background-color:#e6ffe6; padding:10px; border-radius:8px;">
🧪 <strong>Teste Health-Check direkt:</strong>

```bash
pytest tests/test_installation.py
```

Wenn alles ok ist, siehst du z.B.:

```
================= test session starts =================
collected 3 items

tests/test_installation.py ...                [100%]

================= 3 passed in 1.23s =================
```

Bei Fehlern siehe <a href="#⚠️-troubleshooting--health-check">Troubleshooting</a>.
</div>

---

## 🗣️ Diarization Beispiel

<div style="background-color:#e6ffe6; padding:10px; border-radius:8px;">
🧪 <strong>Teste Diarization direkt:</strong>

```bash
python -m whisperx sample.mp3 --diarize --output_dir diarization_results/
```
Das Ergebnis enthält Sprecher:innen-Zuordnung im <code>diarization_results/</code>-Ordner.
</div>

> 🖼️ <strong>Beispiel-Output Diarization:</strong>
> 
> ```
> [00:00.000 --> 00:04.000] SPEAKER_00: Hallo, ich bin Sprecher eins.
> [00:04.000 --> 00:08.000] SPEAKER_01: Und ich bin Sprecher zwei.
> ```

---

## 🔊 VAD Beispiel

<div style="background-color:#e6ffe6; padding:10px; border-radius:8px;">
🧪 <strong>Teste VAD direkt:</strong>

```bash
python -m whisperx sample.mp3 --vad --output_dir vad_results/
```
Im <code>vad_results/</code>-Ordner findest du die Zeitbereiche mit erkannter Sprache.
</div>

> 🖼️ <strong>Beispiel-Output VAD:</strong>
> 
> ```
> [00:00.000 --> 00:02.500] Speech detected
> [00:04.000 --> 00:06.000] Speech detected
> ```

---

## 📑 Batch-Transkription Beispiel

<div style="background-color:#e6ffe6; padding:10px; border-radius:8px;">
🧪 <strong>Teste Batch-Transkription direkt:</strong>

```bash
python -m whisperx folder_with_audio_files/*.mp3 --output_dir batch_results/
```
Alle Transkripte findest du im <code>batch_results/</code>-Ordner.
</div>

> 🖼️ <strong>Beispiel-Output Batch:</strong>
> 
> ```
> [00:00.000 --> 00:04.000] Datei: audio1.mp3
> Hallo, dies ist Datei 1.
> [00:00.000 --> 00:03.000] Datei: audio2.mp3
> Hier spricht Datei 2.
> ```

---

## 🌍 Verschiedene Sprachen Beispiel

<div style="background-color:#e6ffe6; padding:10px; border-radius:8px;">
🧪 <strong>Teste Spracheinstellung direkt:</strong>

```bash
python -m whisperx sample_fr.mp3 --language fr --output_dir fr_results/
```
Das Transkript auf Französisch findest du im <code>fr_results/</code>-Ordner.
</div>

> 🖼️ <strong>Beispiel-Output Französisch:</strong>
> 
> ```
> [00:00.000 --> 00:03.000] Bonjour, ceci est un exemple en français.
> ```

---

<div style="background-color:#e6f0ff; padding:12px; border-radius:8px; margin-bottom:16px;">
🔗 <strong>Schritt-für-Schritt-Guides & Community-Events:</strong> <a href="https://github.com/m-bain/whisperx/discussions">Hier findest du Tutorials, Guides und aktuelle Events!</a>
</div>

## ⚠️ Troubleshooting & Health-Check

- **Health-Check nach der Installation:**
  Nach der Installation kannst du mit folgendem Befehl prüfen, ob alle Kernkomponenten korrekt installiert sind:
  ```bash
  pytest tests/test_installation.py
  ```
  Bei Fehlern siehe unten und im FAQ.

- **CUDA not available:** Stelle sicher, dass die NVIDIA-Treiber und CUDA korrekt installiert sind. Prüfe mit `nvidia-smi`.
- **ImportError:** Überprüfe, ob alle Abhängigkeiten installiert sind und das richtige Python-Environment aktiv ist.
- **Fehlende Module:** Führe ggf. ein `pip install -U pip setuptools` und installiere die Pakete erneut.
- **Audio wird nicht erkannt:** Prüfe das Audioformat und die Lesbarkeit der Datei.

**Häufige Test-Fehler & Lösungen:**

| Fehlerbild/Test                                 | Mögliche Ursache                    | Lösung                                              |
|-------------------------------------------------|-------------------------------------|-----------------------------------------------------|
| `ctranlsate2` fehlt                            | Nicht installiert                   | `pip install ctranslate2`                           |
| CUDA-Fehler                                    | Keine NVIDIA-GPU oder CUDA fehlt    | CPU-Modus nutzen oder CUDA installieren             |
| `ValueError` bei Diarization                   | Fehlende pyannote.audio-Modelle     | `pip install pyannote.audio` und Modelle nachladen  |
| RuntimeError AdaptiveProcessor                 | Falsches Device/fehlende Abhängigkeit| Device prüfen, Abhängigkeiten installieren         |
| AssertionError in Installationstest            | Paket nicht korrekt installiert     | Installation prüfen, ggf. neu installieren          |

> 💬 **Community-Support:** Bei Problemen hilft die Community gerne weiter – poste deine Fehlermeldung im [Forum](https://github.com/m-bain/whisperx/discussions), oft gibt es schnelle Hilfe!

Weitere Hilfe findest du auch in den [GitHub Issues](https://github.com/m-bain/whisperx/issues).

---

## 👶 Für Einsteiger:innen: Erste Schritte & Tipps

> 🟢 **Schnellstart:** In weniger als 5 Minuten zum ersten Transkript!

### 🚦 Schritt-für-Schritt-Anleitung
1. ✅ **Installation abschließen** (siehe oben)
2. 🎵 **Beispiel-Audio bereitstellen** (z.B. `audio.mp3`)
3. 🖥️ **Transkribieren:**
   ```bash
   python -m whisperx audio.mp3 --output_dir results/
   ```
4. 📂 **Ergebnis anschauen:** Öffne den `results/`-Ordner und prüfe die erzeugten Dateien (z.B. `.txt`, `.json`).

> 💡 **Tipp:** Starte mit einer kurzen Audiodatei (1–2 Minuten), um dich mit dem Ablauf vertraut zu machen.

### 📚 Glossar für Einsteiger:innen

---

<div align="center" style="margin-top:2em;">
💬 <strong>Feedback zur Doku?</strong> <a href="https://github.com/m-bain/whisperx/discussions" style="font-size:1.2em; color:#0078d7; font-weight:bold;">Hier direkt Rückmeldung geben!</a>
</div>

---

## 🤖 Hinweise für KI-gestützte Mitarbeit (Nur für KI)

> **Diese Sektion ist ausschließlich für KI-Assistenzsysteme gedacht und sollte als Erinnerung gespeichert werden.**

### Projektziele & Prinzipien
- Ziel: WhisperX für alle Nutzer:innen maximal zugänglich, robust und community-orientiert machen.
- Fokus: Einsteigerfreundlichkeit, Barrierefreiheit, Community-Support, schnelle Fehlerdiagnose.
- Doku-Änderungen immer so gestalten, dass sie echte Praxisprobleme lösen und Nutzer:innen motivieren.

### Regeln für KI-Mitarbeit
- Niemals bestehende Inhalte ohne Grund überschreiben – immer ergänzen oder verbessern.
- Änderungen immer mit klarer, kurzer Begründung dokumentieren (Change-Log, Commit-Message, Pull-Request-Text).
- Bei Unsicherheiten in der Community/FAQ nachschlagen oder Feedback einholen.
- Keine sensiblen Nutzerdaten speichern oder veröffentlichen.
- Immer aktuelle Best Practices für Open-Source-Dokus und Accessibility beachten.
- Vor größeren Änderungen: Memories/Kontext prüfen und relevante Infos in die Änderung einbeziehen.

### Konkrete Beispiele
**Gute KI-Änderung:**
- Fügt eine neue FAQ-Frage hinzu, weil sie in mehreren Issues aufgetaucht ist, und verweist auf die Quelle.
- Ergänzt plattformspezifische Hinweise nach Feedback von Nutzer:innen, ohne bestehende Lösungen zu entfernen.

**Schlechte KI-Änderung:**
- Überschreibt den Troubleshooting-Abschnitt komplett, ohne auf bestehende Inhalte oder Nutzerfragen einzugehen.
- Entfernt Community-Links oder Hinweise auf Barrierefreiheit ohne Begründung.

### Nutzung von Memories & Kontext
- Vor jeder Änderung alle relevanten Memories und vorherige Konversationen prüfen.
- Wichtige Erkenntnisse, Designentscheidungen und Nutzerpräferenzen als Memory speichern oder aktualisieren.
- Kontext aus Issue-Tracker, Pull-Requests und Feedback gezielt nutzen.

### Versionskontrolle & Zusammenarbeit
- Änderungen immer als Pull-Request vorschlagen, nicht direkt auf den Hauptbranch pushen.
- Bei Merge-Konflikten: menschliche Maintainer:innen einbinden und Änderung begründen.
- Feedback-Schleifen aktiv nutzen: Auf Reviews und Kommentare eingehen, ggf. Änderung anpassen.
- Bei Unsicherheit: Rückfrage an Maintainer:innen oder Community stellen.

### Typische KI-Fehler vermeiden
- Nicht mehrere Abschnitte mit identischem Inhalt anlegen (Redundanz).
- Keine Platzhalter oder TODOs ohne Erklärung hinterlassen.
- Keine automatisierten Übersetzungen ohne Prüfung einfügen.
- Keine Formatierungsfehler oder fehlerhafte Links erzeugen.
- Keine veralteten oder widersprüchlichen Infos stehen lassen.

**Beispiel für einen typischen KI-Fehler:**
> Falsch: "Fügen Sie hier Ihre FAQ-Frage ein."
> Richtig: "Wie behebe ich CUDA Out of Memory? → Siehe Troubleshooting, Lösungsvorschlag..."

### Zusammenarbeit mit menschlichen Maintainer:innen
- Immer offen für Rückfragen und Korrekturen sein.
- Explizit auf Wünsche, Präferenzen und Feedback der Maintainer:innen eingehen.
- Bei Unklarheiten lieber nachfragen als eigenmächtig entscheiden.

### Umgang mit sensiblen Daten & Ethik
- Keine personenbezogenen oder sensiblen Daten in Issues, Commits oder Memories speichern.
- Bei Unsicherheit zu Datenschutz oder ethischen Fragen: Rückfrage an Maintainer:in, keine automatisierten Entscheidungen treffen.
- KI-Änderungen immer auf ethische Unbedenklichkeit prüfen (z.B. keine diskriminierenden Beispiele, keine unreflektierten Empfehlungen).

### Transparenz & Nachvollziehbarkeit
- Für jede KI-Änderung ein kurzes Änderungsprotokoll anlegen (z.B. "FAQ erweitert, weil...", "Troubleshooting ergänzt nach Issue #123").
- KI-spezifische Commit Messages nutzen, z.B. "[KI] Neue FAQ aus Community-Feedback hinzugefügt".
- Größere KI-Änderungen im Pull Request-Text klar als solche kennzeichnen.

### Automatisierte Tests & Linting
- Nach jeder Code- oder Doku-Änderung automatisierte Tests und Linter ausführen (z.B. Markdown-Lint, Link-Checker).
- Fehlerhafte Formatierung, kaputte Links oder Syntaxfehler sofort beheben.
- Bei Testfehlschlägen keine Änderungen mergen, sondern zuerst Fehler analysieren.

### Dokumentation von KI-Entscheidungen
- Jede größere Entscheidung (z.B. neue FAQ, Änderung der Struktur) mit kurzer Begründung dokumentieren (im PR, Commit oder als Memory).
- Beispiel: "FAQ zu CUDA ergänzt, da 5 Nutzer:innen im Forum identisches Problem hatten."

### Zusammenarbeit mit anderen KI-Systemen
- Kontext und relevante Memories explizit weitergeben, wenn mehrere KI-Systeme beteiligt sind.
- Konflikte vermeiden: Vor Änderung prüfen, ob andere KI-Prozesse parallel laufen oder kürzlich Änderungen gemacht wurden.
- Bei Unsicherheit: Merge-Strategie mit Maintainer:in abstimmen.

### Best Practices für nachhaltige KI-Dokumentation
- Regelmäßig Feedback von menschlichen Nutzer:innen einholen und auswerten.
- KI-Entscheidungen transparent machen und dokumentieren.
- Die KI-Sektion regelmäßig pflegen, neue Erkenntnisse und Regeln aufnehmen.
- Bei größeren Änderungen: Review durch menschliche Maintainer:in einfordern.
- Die KI-Sektion als "Wissensanker" für alle zukünftigen KI-Interaktionen verstehen.

> **Ende der KI-Sektion – dieser Bereich ist für Endnutzer:innen nicht relevant.**

- **Transkribieren:** Audio in Text umwandeln
- **Diarization (Sprechertrennung):** Erkennung, wer wann spricht (z.B. "Sprecher A", "Sprecher B").
- **VAD (Voice Activity Detection, Spracherkennung):** Erkennung, wann gesprochen wird (hilft, Stille zu überspringen).
- **Forced Alignment (erzwungene Ausrichtung):** Automatisches Abgleichen von Text und Audio, damit jedes Wort oder jeder Laut die richtige Zeitmarke bekommt.

### ❓ FAQ: Häufige Fragen & Stolpersteine

- **Wie prüfe ich, ob alles richtig installiert ist?**
  Führe nach der Installation `pytest tests/test_installation.py` aus. Bei Fehlern siehe Troubleshooting-Tabelle und FAQ.
- **Was tun bei Testfehlern?**
  Prüfe die Fehlermeldung, vergleiche sie mit der Tabelle „Häufige Test-Fehler & Lösungen“ und folge den Lösungsvorschlägen. Bei Unsicherheit: Community fragen!
- **Wie bekomme ich schnelle Hilfe?**
  Nutze das [Community-Forum](https://github.com/m-bain/whisperx/discussions) oder erstelle ein GitHub Issue. Je mehr Infos (Fehlermeldung, System, Schritte) du angibst, desto schneller kann geholfen werden.
- **Kann ich WhisperX ohne GPU nutzen?**
  Ja, aber die Verarbeitung ist dann deutlich langsamer. Für große Dateien empfiehlt sich eine GPU.
- **Wie kann ich meine Installation validieren?**
  Mit `pytest tests/test_installation.py` oder `python -m whisperx --help`.
- **Wie kann ich mehrere Dateien testen?**
  Siehe CLI-Beispiele und Batch-Verarbeitung oben.
- **Wie kann ich Fehler melden?**
  Siehe Abschnitt „Support & Kontakt“ und die Anleitung unten.

---

## 🆘 Support & Kontakt

- **Forum:** [Community-Forum](https://github.com/m-bain/whisperx/discussions) – schnelle Hilfe, Peer-Support, Community-Paten.
- **GitHub Issues:** Für Bugs und Feature-Wünsche.
- **Live-Q&A:** Siehe Community-Kalender für Termine.
- **Community-Paten:** Persönliche Unterstützung für Einsteiger:innen – einfach im Forum melden.

### So meldest du ein Problem richtig
1. Führe einen Health-Check durch (`pytest tests/test_installation.py`).
2. Notiere die genaue Fehlermeldung und dein System (Betriebssystem, Python-Version, CUDA/GPU).
3. Beschreibe, was du gemacht hast (Schritte zur Reproduktion).
4. Poste alles im Forum oder als Issue.
5. Optional: Screenshot oder Log-Auszug anhängen.

> Je genauer die Infos, desto schneller kann dir geholfen werden!

---

## 🛡️ Tipps für stabile Nutzung

- **Regelmäßig Updates machen:** Halte WhisperX und alle Abhängigkeiten aktuell (`pip install -U ...`).
- **Virtuelle Umgebungen nutzen:** Vermeide Konflikte mit anderen Python-Projekten.
- **Backup von Modellen:** Sichere heruntergeladene Modelle für Offline-Nutzung.
- **Changelog lesen:** Prüfe bei neuen Releases, ob es Breaking Changes gibt.
- **Community-Feedback nutzen:** Tausche dich regelmäßig mit anderen aus und profitiere von Best Practices.

### 📝 Beispielausgabe eines Transkripts

> 💬 **Community hilft:** Stelle deine Frage mit Fehlermeldung und Beispiel im [Forum](https://github.com/m-bain/whisperx/discussions) – schnelle Hilfe garantiert!

---

## 📄 Beispielausgabe eines Transkripts

**Beispiel (Ausschnitt aus `audio.txt`):**
```
[00:00:00.000 --> 00:00:03.000] Hallo und willkommen zum WhisperX-Tutorial!
[00:00:03.000 --> 00:00:07.000] In diesem Video zeige ich dir, wie du Audio automatisch transkribierst.
```
**Beispiel (Ausschnitt aus `audio.json`):**
```json
{
  "segments": [
    {"start": 0.0, "end": 3.0, "text": "Hallo und willkommen zum WhisperX-Tutorial!"},
    {"start": 3.0, "end": 7.0, "text": "In diesem Video zeige ich dir, wie du Audio automatisch transkribierst."}
  ]
}
```

> 🖼️ **Noch anschaulicher?** Hier findest du eine bebilderte Schritt-für-Schritt-Anleitung und Videotutorials: [WhisperX Video-Anleitungen](https://www.youtube.com/results?search_query=whisperx)

---

## 🏁 Dein erster Erfolg in 60 Sekunden

> ⏱️ **Schnellstart:** Kopiere diesen Befehl, ersetze `audio.mp3` durch deine Datei – fertig!
>
> ```bash
> python -m whisperx audio.mp3 --output_dir results/
> ```
> Schau in den Ordner `results/` – dein Transkript wartet schon auf dich!

---

## 📝 Cheat Sheet: WhisperX Kommandos & Optionen

| Befehl / Option             | Bedeutung                                          |
|----------------------------|----------------------------------------------------|
| `python -m whisperx file`   | Transkribiert die Datei                            |
| `--output_dir pfad/`        | Zielordner für Ergebnisse                          |
| `--language de`             | Sprache explizit setzen                            |
| `--model large-v2`          | Modell auswählen                                   |
| `--diarize`                 | Sprechererkennung aktivieren                       |
| `--vad`                     | Voice Activity Detection aktivieren                |
| `--help`                    | Alle verfügbaren Optionen anzeigen                 |

> 💡 **Tipp:** Mit `python -m whisperx --help` bekommst du alle Optionen mit kurzer Erklärung angezeigt.

---

## 🌈 Motivations-Box & Community-Power

> 🌟 **Jede:r kann beitragen!**
>
> - Teile deine Erfahrungen & Tipps im [Community-Forum](https://github.com/m-bain/whisperx/discussions)
> - Vote für Features, stelle Fragen, hilf anderen – gemeinsam werden wir besser!
> - Auch kleine Beiträge (z.B. Fehler melden, Übersetzungen, Doku-Verbesserungen) sind Gold wert.

---

## ♿ Barrierearme Nutzung & Inklusion

- **Screenreader-freundlich:** Die wichtigsten Ausgaben sind als Klartext und strukturiert verfügbar.
- **Einfache Sprache:** Viele Abschnitte sind bewusst klar und einfach formuliert.
- **Übersetzungen:** Nutze Online-Übersetzer oder frage im Forum nach Hilfe für andere Sprachen.
- **Community hilft:** Barrierefreiheit und Inklusion sind uns wichtig – Feedback dazu ist immer willkommen!
## 🎯 Tipps für verschiedene Zielgruppen

| Zielgruppe         | Tipp & Beispiel                                                                 |
|--------------------|--------------------------------------------------------------------------------|
| 👩‍🏫 Lehrkräfte      | Erstelle automatisch Mitschriften & Arbeitsblätter aus Unterrichtsaufnahmen.    |
| 🎓 Studierende      | Nutze Transkripte für Vorlesungsnotizen & effizientes Nacharbeiten.             |
| 👩‍💻 Entwickler:innen| Integriere WhisperX in eigene Apps, Bots oder Analyse-Pipelines.               |
| 🤝 Teams            | Protokolliere Meetings automatisch, teile Ergebnisse direkt im Team-Chat.        |
| 🏠 Privatnutzer:innen| Transkribiere Interviews, Familiengeschichten oder Podcasts für dein Archiv.    |

---

## 🏢 Branchenspezifische Beispiele & Inspiration

| Branche             | Anwendungsidee                                                             |
|---------------------|----------------------------------------------------------------------------|
| 🎬 Medien           | Automatische Untertitel für Videos, Interviews, Podcasts                   |
| 🔬 Forschung        | Transkription von Interviews, Feldstudien, Audio-Notizen                   |
| 🏫 Bildung          | Lernmaterialien, Mitschriften, barrierefreie Inhalte                       |
| 🏥 Gesundheitswesen | Dokumentation von Arztgesprächen, Anamnesen, Patientenaufklärung           |
| 🏛️ Verwaltung       | Protokolle von Sitzungen, Bürgerbeteiligung, Transparenz                   |
| ♿ Barrierefreiheit  | Live-Untertitel, Audio-zu-Text für Gehörlose und Schwerhörige              |

> 📂 **Community-Vorlagen:** Im Forum findest du Vorlagen und Best Practices für viele Branchen – teile auch deine eigenen Beispiele!

---

## 📝 Einsteiger-Quiz: Bist du bereit?

Teste dein Wissen mit diesen Fragen:
- Wie starte ich eine Transkription mit WhisperX?
- Wie kann ich mehrere Dateien auf einmal verarbeiten?
- Wo finde ich Hilfe bei Fehlermeldungen?
- Wie kann ich WhisperX in meine App integrieren?

> 💡 **Lösung:** Antworten findest du in den Schnellstart- und FAQ-Abschnitten oben. Oder frage die Community!

---

## 🌍 Tipps für internationale Nutzer:innen

- **Mehrsprachige Modelle:** WhisperX unterstützt viele Sprachen – setze einfach `--language <code>` (z.B. `--language fr` für Französisch).
- **Übersetzungen:** Viele Abschnitte der README können mit Online-Tools (DeepL, Google Translate) übersetzt werden.
- **Community-Support:** Im Forum helfen Nutzer:innen aus aller Welt – Fragen können auch auf Englisch oder anderen Sprachen gestellt werden.
- **Beispiel:**
  ```bash
  python -m whisperx audio.mp3 --language es --output_dir ergebnisse/
  ```

---

## 🔄 Regelmäßige Updates & Mitmachen

- **Bleib auf dem Laufenden:** Schau regelmäßig nach Updates und neuen Features im Repository.
- **Feedback & Wünsche:** Teile deine Ideen, Verbesserungsvorschläge und Feature-Wünsche im [Forum](https://github.com/m-bain/whisperx/discussions) oder als Issue.
- **Mitmachen:** Beiträge aller Art sind willkommen – von Code über Doku bis zu Übersetzungen.

---

## 🔐 Datenschutz & Sicherheit

- **Lokale Verarbeitung:** WhisperX kann komplett lokal genutzt werden – deine Audiodaten verlassen nie deinen Rechner, wenn du das möchtest.
- **Open Source:** Du hast volle Kontrolle über den Code und die Datenverarbeitung.
- **Tipp:** Prüfe beim Einsatz in sensiblen Bereichen (z.B. Medizin, Verwaltung) die lokalen Datenschutzanforderungen.

---

## 🎨 Kreative Use Cases & Community-Challenges

- **Beispiele:**
  - Live-Untertitel für Theateraufführungen oder Veranstaltungen
  - Transkription von historischen Tonbändern für Archive
  - Automatische Songtext-Erkennung aus Musikaufnahmen
  - Sprachenlernen mit automatischer Ausspracheanalyse
  - Podcast-Transkripte mit automatischer Themen-Tagging
- **Community-Challenge:**
  - Nimm an monatlichen Challenges teil (z.B. „Wer baut das kreativste WhisperX-Projekt?“)
  - Gewinne Community-Badges und teile dein Ergebnis im [Showcase](https://github.com/m-bain/whisperx/discussions/categories/show-and-tell)

---

## 🌐 Internationales Community-FAQ (DE/EN)

| Frage (DE)                                 | Question (EN)                             | Antwort/Answer                                                                                 |
|--------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------|
| Wie ändere ich die Sprache?                | How do I change the language?             | `--language <code>` setzen, z.B. `--language en` / Use `--language <code>`, e.g. `--language en` |
| Wie bekomme ich Hilfe?                     | How do I get help?                        | Im Forum fragen oder Issue erstellen / Ask in the forum or open an issue                      |
| Kann ich WhisperX lokal nutzen?            | Can I use WhisperX locally?               | Ja, alles kann offline laufen / Yes, everything can run offline                               |
| Ist meine Privatsphäre geschützt?          | Is my privacy protected?                  | Ja, bei lokaler Nutzung bleiben Daten auf deinem Rechner / Yes, with local use, data stays local |
| Gibt es Support für andere Sprachen?       | Is there support for other languages?     | Ja, viele Sprachen werden unterstützt / Yes, many languages are supported                     |

---

## 📚 Best-Practice-Guides & Datenschutz

- **Integration:** Im Forum und in der Doku findest du Best-Practice-Guides für verschiedene Plattformen und Workflows.
- **Datenschutz:** Beachte die Hinweise zur lokalen Nutzung und prüfe die Datenschutzanforderungen deines Landes/Branche.
- **Community:** Teile deine eigenen Best Practices und Tipps – gemeinsam sorgen wir für sichere und effiziente Nutzung!

---

## 🎉 Community-Events & Live-Q&A

- **Regelmäßige Online-Treffen:** Tausche dich mit anderen Nutzer:innen aus, stelle Fragen und lerne neue Tricks.
- **Live-Q&A-Termine:** Angekündigte Fragerunden mit Entwickler:innen und Power-Usern – stelle deine Fragen live!
- **Austausch:** Finde Mitstreiter:innen für gemeinsame Projekte oder Hilfestellung in Echtzeit.
- **Aktuelle Termine:** Siehe Forum oder Community-Kalender.

---

## 🏫 Ressourcen für Bildung & NGOs

- **Sonderangebote:** Kostenlose oder vergünstigte Nutzung für Bildungseinrichtungen und gemeinnützige Organisationen – frage im Forum nach aktuellen Aktionen!
- **Tipps:** Leitfäden für den Einsatz im Unterricht, Workshops und barrierefreie Bildungsprojekte.
- **Fördermöglichkeiten:** Hinweise auf Stipendien, Wettbewerbe und Förderprogramme der Community.

---

## 🎥 Video-FAQ & Umfragen (in Planung)

- **Video-FAQ:** Bald findest du die wichtigsten Fragen und Antworten auch als Kurzvideos – Vorschläge willkommen!
- **Feedback & Umfragen:** Nimm an regelmäßigen Umfragen teil und gestalte die Weiterentwicklung von WhisperX aktiv mit.

> 💬 **Deine Meinung zählt:** Wünsche, Kritik oder Lob? Teile sie im Forum oder über die nächste Community-Umfrage!

---

## 🏆 Success Stories & Inspiration

- **Lehrerin aus NRW:** „Dank WhisperX kann ich Unterrichtsmitschnitte schnell in Arbeitsblätter verwandeln – das spart mir jede Woche Zeit!“
- **Podcast-Team aus Berlin:** „Wir transkribieren und verschlagworten unsere Episoden jetzt automatisch – die Reichweite ist deutlich gestiegen.“
- **Archivarin:** „Mit WhisperX habe ich alte Tonbänder digitalisiert und für die Forschung zugänglich gemacht.“
- **NGO:** „Barrierefreie Untertitel für unsere Online-Workshops sind jetzt ein Kinderspiel!“
- **Privatnutzer:** „Ich habe die Familiengeschichte meines Großvaters als Audio aufgenommen und mit WhisperX für alle lesbar gemacht.“

> ✨ **Teile deine eigene Erfolgsgeschichte im [Showcase](https://github.com/m-bain/whisperx/discussions/categories/show-and-tell) und inspiriere andere!**

---

## 👶🧓 Tutorials für Kids, Senioren & Einsteiger:innen

- **Für Kids:**
  - Schritt-für-Schritt-Bilderstrecken und Video-Tutorials im Community-Forum
  - Einfache Sprache und viele Beispiele
- **Für Senioren:**
  - Persönliche Unterstützung durch Community-Paten
  - Extra-Erklärungen zu Installation und Bedienung
- **Für Einsteiger:innen:**
  - Mini-Quiz, FAQ und Schnellstart-Anleitungen
  - Tipps für erste Projekte und Fehlervermeidung

> 💡 **Wunsch-Thema?** Sag im Forum Bescheid, wenn du ein spezielles Tutorial brauchst!

---

## 📅 Community-Kalender & Events vorschlagen

- **Immer aktuell:** Im Community-Kalender findest du alle geplanten Events, Workshops und Live-Q&As.
- **Eigene Events:** Du möchtest selbst einen Workshop, ein Meetup oder eine Sprechstunde anbieten? Schlage dein Event im Forum vor – die Community hilft bei der Organisation!

---

## 🌍 Regionale Meetups & Sprachräume

- **Lokale Gruppen:** Finde Nutzer:innen in deiner Nähe – im Forum gibt es regionale Channels und Meetup-Threads.
- **Mehrsprachige Treffen:** Es gibt Online-Treffen und Diskussionsrunden in verschiedenen Sprachen (z.B. Englisch, Deutsch, Spanisch, Französisch).
- **Eigene Gruppe gründen:** Starte eine lokale Gruppe oder einen Sprachraum – die Community hilft bei der Organisation und Bekanntmachung.
- **Aktuelle Infos:** Schau regelmäßig in den Community-Kalender und die Sprachraum-Threads im Forum.

---

## 🏅 Community-Badges & Anerkennung

- **Badges für Engagement:** Erhalte Auszeichnungen für Tutorials, Support, innovative Projekte oder Community-Events.
- **Anerkennung:** Besonders aktive Mitglieder werden regelmäßig im Newsletter und im Forum vorgestellt.
- **Mitmachen lohnt sich:** Jede Hilfe zählt – vom kleinen Tipp bis zum großen Beitrag!

---

## 📨 WhisperX-Newsletter

- **Monatlich:** Erhalte die wichtigsten Tipps, Updates, Community-Highlights und Event-Termine direkt ins Postfach.
- **Anmeldung:** Einfach im Forum oder auf der Projektseite eintragen.
- **Mitgestalten:** Teile News, Tipps oder Projekte für die nächste Ausgabe – die Community freut sich über Input!

---

## 🤝 Mentoring & Peer-Support

- **1:1-Mentoring:** Finde eine:n erfahrene:n Mentor:in für deine ersten Schritte oder unterstütze andere als Peer.
- **Peer-Learning-Gruppen:** Lerne gemeinsam mit anderen in kleinen Gruppen – für alle Erfahrungsstufen.
- **Mitmachen:** Melde dich im Forum als Mentor:in oder Mentee – die Community bringt euch zusammen!

---

## 🗺️ Globale Community-Karte

- **Interaktive Karte:** Entdecke, wo überall auf der Welt WhisperX genutzt wird und finde Nutzer:innen oder Gruppen in deiner Nähe.
- **Eintragen:** Trage dich freiwillig ein und vernetze dich regional oder thematisch.
- **Link zur Karte:** Im Forum und auf der Projektseite verfügbar.

---

## 🏆 Spezielle Challenges & Wettbewerbe

- **Regionale Challenges:** Zeige, wie WhisperX in deinem Land oder deiner Region eingesetzt wird.
- **Zielgruppen-Wettbewerbe:** Spezielle Aktionen für Bildung, NGOs, Forschung, Medien u.v.m.
- **Themen-Challenges:** Monatliche Wettbewerbe zu Schwerpunkten wie Barrierefreiheit, Kreativität oder Integration.
- **Preise & Anerkennung:** Gewinne Badges, Community-Features oder kleine Sachpreise.

---

## 🚀 Jetzt bist du dran!

> **Starte noch heute dein erstes Projekt, teile deine Erfahrungen und werde Teil unserer aktiven Community!**
> Gemeinsam machen wir Sprachverarbeitung für alle zugänglich und einfach.

---

### Mini-FAQ für spezielle Zielgruppen

- **Wie kann ich WhisperX im Unterricht nutzen?**
  Mitschriften, Untertitel, Lernmaterialien automatisch erstellen – einfach Audio aufnehmen und transkribieren.
- **Kann ich mehrere Vorlesungen auf einmal transkribieren?**
  Ja! Nutze die Batch-Verarbeitung (siehe CLI-Beispiele oben).
- **Wie binde ich WhisperX in meine App ein?**
  Siehe Python-API und REST-API-Integration – bei Fragen hilft die Community.
- **Wie bleibt mein Team auf dem Laufenden?**
  Ergebnisse automatisiert im Team-Chat oder als Protokoll teilen.
- **Gibt es Hilfe für Einsteiger:innen?**
  Ja! Im Forum gibt es Community-Paten und gezielte Unterstützung für alle Erfahrungsstufen.

> 🤗 **Personalisierte Hilfe:** Melde dich im Forum – dort findest du Community-Paten, die dich individuell unterstützen und bei deinen ersten Schritten begleiten!

---

## ⚙️ Integration in CI/CD-Pipelines

WhisperX kann automatisiert in Build- und Testprozesse eingebunden werden, z.B. mit GitHub Actions:

**Beispiel: GitHub Actions Workflow**
```yaml
name: WhisperX Batch Transcription
on: [push]
jobs:
  transcribe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install git+https://github.com/m-bain/whisperx.git
      - name: Run WhisperX
        run: |
          python -m whisperx sample_audio.mp3 --output_dir results/
```
> 💡 Passe den Workflow an deine Bedürfnisse an (z.B. Batch-Transkription mehrerer Dateien).

---

## 📂 Templates & Beispielskripte
- Im Repository findest du Vorlagen und Beispielskripte für verschiedene Anwendungsfälle (z.B. Batch-Transkription, Fehlerprüfung).
- Nutze diese als Ausgangspunkt für eigene Integrationen.

---

## 🚀 Fortgeschrittene Nutzungsszenarien

- **Batch-Verarbeitung:** Transkribiere mehrere Dateien in einem Schritt, z.B. per Shell-Skript oder Python-Loop.
- **Custom Models:** Nutze eigene oder angepasste Modelle mit WhisperX (`--model <pfad_zum_modell>`).
- **Automatisierte Qualitätssicherung:** Kombiniere WhisperX mit Tests und Validierungsskripten für große Datenmengen.
- **Integration mit anderen Tools:** Kopple WhisperX mit NLP- oder Analyse-Frameworks für weiterführende Auswertungen.

---

## 🐳 Docker-Nutzung

WhisperX kann einfach in Containern betrieben werden. Beispiel für ein minimales `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install git+https://github.com/m-bain/whisperx.git
CMD ["python", "-m", "whisperx", "audio.mp3", "--output_dir", "results/"]
```

**Build & Run:**
```bash
docker build -t whisperx-app .
docker run --rm -v $PWD:/app whisperx-app
```
> 💡 Passe das Dockerfile und die CMD-Zeile an deine Bedürfnisse an (z.B. GPU-Support, andere Modelle).

---

## 🌐 REST-API-Integration

WhisperX kann in eigene Webservices eingebunden werden, z.B. mit [FastAPI](https://fastapi.tiangolo.com/):

```python
from fastapi import FastAPI, UploadFile
import whisperx

app = FastAPI()
model = whisperx.load_model("large-v2", device="cuda")

@app.post("/transcribe/")
async def transcribe(file: UploadFile):
    audio_path = f"/tmp/{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    result = model.transcribe(audio_path)
    return {"segments": result["segments"]}
```
> 💡 Für produktive Nutzung: Beachte Sicherheit, Skalierung und Ressourcenmanagement.

---

## ☁️ Cloud-Deployment & Skalierung

WhisperX kann flexibel in der Cloud betrieben werden – z.B. auf AWS, Azure oder GCP.

- **GPU-Instanzen:** Für hohe Geschwindigkeit empfiehlt sich der Einsatz von GPU-VMs (z.B. AWS EC2 `g4dn`, Azure `NC`-Serie, GCP `A2`-Instanzen).
- **Storage:** Nutze schnellen Speicher (NVMe, SSD) für große Audiodateien und Zwischenergebnisse.
- **Skalierung:** Setze mehrere Worker oder verwende Container-Orchestrierung (z.B. Kubernetes) für parallele Verarbeitung.
- **Kostenkontrolle:** Nutze Spot-Instanzen oder automatische Abschaltung nach Verarbeitung.

> 💡 Siehe Beispiel-Dockerfile und REST-API-Integration weiter oben.

---

## 📈 Benchmarks & Performance

| Modell         | Hardware         | 1h Audio | Laufzeit (ca.) |
|---------------|------------------|----------|----------------|
| large-v2 (GPU)| RTX 3090         | 1h       | ~5-8 min       |
| large-v2 (CPU)| 8 vCPU           | 1h       | ~60-120 min    |
| medium (GPU)  | RTX 3090         | 1h       | ~3-5 min       |

- Die tatsächliche Laufzeit hängt von Hardware, Modell, Audioqualität und Einstellungen ab.
- Miss die Performance mit eigenen Daten und passe die Konfiguration an deine Anforderungen an.

---

## 🎬 Video-Onboarding & Support

- **Video-Tutorials:** Schritt-für-Schritt-Anleitungen findest du im [YouTube-Channel](https://www.youtube.com/results?search_query=whisperx) und im Community-Forum.
- **Support & Hilfe:**
  - [GitHub Issues](https://github.com/m-bain/whisperx/issues)
  - [Discussions](https://github.com/m-bain/whisperx/discussions)
  - [FAQ & Troubleshooting](#faq-häufige-fehler--lösungen)

---

## 🧪 Beispiel: Automatisierter Test für WhisperX (Python)

Du kannst mit einem kleinen Skript testen, ob WhisperX korrekt installiert ist und eine Datei verarbeitet werden kann:

```python
import subprocess
result = subprocess.run([
    'python', '-m', 'whisperx', '--help'
], capture_output=True, text=True)
if 'usage' in result.stdout.lower():
    print('✅ WhisperX ist installiert und funktioniert!')
else:
    print('❌ WhisperX-Test fehlgeschlagen.')
```

---

## ❓ FAQ: Häufige Fehler & Lösungen

| Problem                       | Lösungsvorschlag                                         |
|-------------------------------|----------------------------------------------------------|
| CUDA not available            | NVIDIA-Treiber & CUDA prüfen, ggf. `nvidia-smi` nutzen   |
| ImportError                   | Python-Umgebung & Abhängigkeiten prüfen                  |
| Audio wird nicht erkannt      | Audioformat & Dateipfad kontrollieren                    |
| "ModuleNotFoundError"         | Fehlendes Paket mit `pip install ...` nachinstallieren   |
| Falsche Sprache im Ergebnis   | `--language`-Parameter explizit setzen                   |

Weitere Hilfe: [GitHub Issues](https://github.com/m-bain/whisperx/issues)

---

## 🤝 Kontakt & Community
- **Fragen oder Feedback?** Stelle sie gerne als [Issue](https://github.com/m-bain/whisperx/issues) ein.
- **Community & Diskussion:** [GitHub Discussions](https://github.com/m-bain/whisperx/discussions)
- **Mitmachen:** Pull Requests und Beiträge sind willkommen!

---

`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

See other methods [here.](https://pytorch.org/get-started/previous-versions/#v200)

### 3. Install this repo

`pip install git+https://github.com/m-bain/whisperx.git`

If already installed, update package to most recent commit

`pip install git+https://github.com/m-bain/whisperx.git --upgrade`

If wishing to modify this package, clone and install in editable mode:
```
$ git clone https://github.com/m-bain/whisperX.git
$ cd whisperX
$ pip install -e .
```

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

### Speaker Diarization
To **enable Speaker Diarization**, include your Hugging Face access token (read) that you can generate from [Here](https://huggingface.co/settings/tokens) after the `--hf_token` argument and accept the user agreement for the following models: [Segmentation](https://huggingface.co/pyannote/segmentation-3.0) and [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (if you choose to use Speaker-Diarization 2.x, follow requirements [here](https://huggingface.co/pyannote/speaker-diarization) instead.)

> **Note**<br>
> As of Oct 11, 2023, there is a known issue regarding slow performance with pyannote/Speaker-Diarization-3.0 in whisperX. It is due to dependency conflicts between faster-whisper and pyannote-audio 3.0.0. Please see [this issue](https://github.com/m-bain/whisperX/issues/499) for more details and potential workarounds.


<h2 align="left" id="example">Usage 💬 (command line)</h2>

### English

Run whisper on example segment (using default params, whisper small) add `--highlight_words True` to visualise word timings in the .srt file.

    whisperx examples/sample01.wav


Result using *WhisperX* with forced alignment to wav2vec2.0 large:

https://user-images.githubusercontent.com/36994049/208253969-7e35fe2a-7541-434a-ae91-8e919540555d.mp4

Compare this to original whisper out the box, where many transcriptions are out of sync:

https://user-images.githubusercontent.com/36994049/207743923-b4f0d537-29ae-4be2-b404-bb941db73652.mov


For increased timestamp accuracy, at the cost of higher gpu mem, use bigger models (bigger alignment model not found to be that helpful, see paper) e.g.

    whisperx examples/sample01.wav --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4


To label the transcript with speaker ID's (set number of speakers if known e.g. `--min_speakers 2` `--max_speakers 2`):

    whisperx examples/sample01.wav --model large-v2 --diarize --highlight_words True

To run on CPU instead of GPU (and for running on Mac OS X):

    whisperx examples/sample01.wav --compute_type int8

### Other languages

The phoneme ASR alignment model is *language-specific*, for tested languages these models are [automatically picked from torchaudio pipelines or huggingface](https://github.com/m-bain/whisperX/blob/e909f2f766b23b2000f2d95df41f9b844ac53e49/whisperx/transcribe.py#L22).
Just pass in the `--language` code, and use the whisper `--model large`.

Currently default models provided for `{en, fr, de, es, it, ja, zh, nl, uk, pt}`. If the detected language is not in this list, you need to find a phoneme-based ASR model from [huggingface model hub](https://huggingface.co/models) and test it on your data.


#### E.g. German
    whisperx --model large-v2 --language de examples/sample_de_01.wav

https://user-images.githubusercontent.com/36994049/208298811-e36002ba-3698-4731-97d4-0aebd07e0eb3.mov


See more examples in other languages [here](EXAMPLES.md).

## Python usage  🐍

```python
import whisperx
import gc 

device = "cuda" 
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
```

## Demos 🚀

[![Replicate (large-v3](https://img.shields.io/static/v1?label=Replicate+WhisperX+large-v3&message=Demo+%26+Cloud+API&color=blue)](https://replicate.com/victor-upmeet/whisperx) 
[![Replicate (large-v2](https://img.shields.io/static/v1?label=Replicate+WhisperX+large-v2&message=Demo+%26+Cloud+API&color=blue)](https://replicate.com/daanelson/whisperx) 
[![Replicate (medium)](https://img.shields.io/static/v1?label=Replicate+WhisperX+medium&message=Demo+%26+Cloud+API&color=blue)](https://replicate.com/carnifexer/whisperx) 

If you don't have access to your own GPUs, use the links above to try out WhisperX. 

<h2 align="left" id="whisper-mod">Technical Details 👷‍♂️</h2>

For specific details on the batching and alignment, the effect of VAD, as well as the chosen alignment model, see the preprint [paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

To reduce GPU memory requirements, try any of the following (2. & 3. can affect quality):
1.  reduce batch size, e.g. `--batch_size 4`
2. use a smaller ASR model `--model base`
3. Use lighter compute type `--compute_type int8`

Transcription differences from openai's whisper:
1. Transcription without timestamps. To enable single pass batching, whisper inference is performed `--without_timestamps True`, this ensures 1 forward pass per sample in the batch. However, this can cause discrepancies the default whisper output.
2. VAD-based segment transcription, unlike the buffered transcription of openai's. In Wthe WhisperX paper we show this reduces WER, and enables accurate batched inference
3.  `--condition_on_prev_text` is set to `False` by default (reduces hallucination)

<h2 align="left" id="limitations">Limitations ⚠️</h2>

- Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
- Overlapping speech is not handled particularly well by whisper nor whisperx
- Diarization is far from perfect
- Language specific wav2vec2 model is needed


<h2 align="left" id="contribute">Contribute 🧑‍🏫</h2>

If you are multilingual, a major way you can contribute to this project is to find phoneme models on huggingface (or train your own) and test them on speech for the target language. If the results look good send a pull request and some examples showing its success.

Bug finding and pull requests are also highly appreciated to keep this project going, since it's already diverging from the original research scope.

<h2 align="left" id="coming-soon">TODO 🗓</h2>

* [x] Multilingual init

* [x] Automatic align model selection based on language detection

* [x] Python usage

* [x] Incorporating  speaker diarization

* [x] Model flush, for low gpu mem resources

* [x] Faster-whisper backend

* [x] Add max-line etc. see (openai's whisper utils.py)

* [x] Sentence-level segments (nltk toolbox)

* [x] Improve alignment logic

* [ ] update examples with diarization and word highlighting

* [ ] Subtitle .ass output <- bring this back (removed in v3)

* [ ] Add benchmarking code (TEDLIUM for spd/WER & word segmentation)

* [ ] Allow silero-vad as alternative VAD option

* [ ] Improve diarization (word level). *Harder than first thought...*


<h2 align="left" id="contact">Contact/Support 📇</h2>


Contact maxhbain@gmail.com for queries.

<a href="https://www.buymeacoffee.com/maxhbain" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>


<h2 align="left" id="acks">Acknowledgements 🙏</h2>

This work, and my PhD, is supported by the [VGG (Visual Geometry Group)](https://www.robots.ox.ac.uk/~vgg/) and the University of Oxford.

Of course, this is builds on [openAI's whisper](https://github.com/openai/whisper).
Borrows important alignment code from [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
And uses the wonderful pyannote VAD / Diarization https://github.com/pyannote/pyannote-audio


Valuable VAD & Diarization Models from [pyannote audio][https://github.com/pyannote/pyannote-audio]

Great backend from [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)

Those who have [supported this work financially](https://www.buymeacoffee.com/maxhbain) 🙏

Finally, thanks to the OS [contributors](https://github.com/m-bain/whisperX/graphs/contributors) of this project, keeping it going and identifying bugs.

<h2 align="left" id="cite">Citation</h2>
If you use this in your research, please cite the paper:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```
