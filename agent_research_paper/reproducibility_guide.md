# Reproducibility Guide

This guide describes how to fully reproduce all experiments and results presented in the paper.

## 1. Prerequisites
- Python 3.8 or later
- Git
- Internet access for dataset download

## 2. Setup
```bash
# Clone repository
git clone https://github.com/NADOOIT/whisperX.git
cd whisperX/agent_research_paper

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# Install web and GUI dependencies
pip install uvicorn[standard] jinja2 fastapi tk
``` 

## 3. Data Preparation
1. Download and extract VoxCeleb1 into `data/vox1`
2. Place user-collected audio in `data/user` (organized by speaker ID)
3. (Optional) Generate or collect Deepfake/replay samples in `data/attacks`

## 4. Configuration
- Review and adjust parameters in `default_config.yaml`:
  - `dataset.vox1_dir`, `dataset.user_data_dir`
  - `training.lora_rank`, `pruning.epsilon`, etc.

## 5. Running Experiments
A single script orchestrates training, pruning cycles, evaluation, and plotting:
```bash
bash run_experiments.sh
```
This will produce:
- `results/metrics.csv` (WER, model size, inference time per cycle)
- `results/attack.csv` (FAR, ASR per attack type)
- `results/recovery.csv` (WER & ID-Accuracy recovery)
- Plots in `results/plots/`

## 6. Launch Web Interface (FastAPI)
From the `webui` folder, start the server and open the browser:
```bash
cd ../webui
uvicorn main:app --reload --port 8000
# then visit http://localhost:8000 in your browser
```

## 7. Launch Tkinter GUI (Launchpad)
From the `launchpad` folder, run the Launchpad app:
```bash
cd ../launchpad
python app.py
```

## 8. Verification
- Compare `results/*.csv` and `results/plots/*` with figures and tables in the paper.
- Ensure metrics match within tolerance (±0.2 WER, ±1% accuracy).

## 9. Data Management via Web UI
While experiments are running, use the FastAPI interface:
- Open http://localhost:8000 in a browser
- **Upload**: Add new audio/transcript samples
- **Test Data**: Flag, correct, or batch-remove problematic samples
- **Training**: Trigger training runs with optional notes
- **Metrics**: Download CSV of WER history and flagged data

## 10. Speaker Training via Launchpad GUI
Use the Tkinter Launchpad for offline management:
- Launch with `python app.py`
- Create/Delete speaker profiles, add samples via drag & drop
- Monitor training progress and WER history in real time
- Export LoRA adapter packages for on-device use

## 11. Logging and Proof-of-Execution
- Scripts write logs to `results/logs/`: `training.log`, `attack.log`, `recovery.log`
- Web UI actions are logged to `logs/webui.log`
- Launchpad GUI actions are logged to `logs/launchpad.log`
- Archive logs alongside results as proof of reproducibility.

## 12. Notes
- All processing occurs locally; no audio or embeddings leave the machine.
- For reproducibility, fix random seeds in your training scripts (e.g., `torch.manual_seed(42)`).

## 13. Evaluation Metrics & Protocol
- **Word Error Rate (WER):** (S + D + I) / N × 100% (S=substitutions, D=deletions, I=insertions, N=ref.words)
- **Model Size (MB):** Dateigröße der Modelle/Adapter auf der Festplatte
- **Inference Time (ms):** Durchschnittliche Latenz pro Utterance über 100 Durchläufe auf Zielhardware
- **Speaker-ID Accuracy:** (korrekte Zuordnungen / Gesamt) × 100%
- **False Acceptance Rate (FAR):** Anteil falscher Akzeptanzen von Angreifern
- **Attack Success Rate (ASR):** Anteil erfolgreicher Replay/Deepfake/Adversarial-Versuche
- **Dataset Split:** 80% Training, 10% Validation, 10% Test (pro Sprecher)
- Alle Messwerte werden in `results/metrics.csv` und `results/logs/` protokolliert

## 14. Troubleshooting & FAQs
- **uvicorn-Fehler:** Stelle sicher, dass alle Dependencies installiert sind (`pip install -r requirements.txt`)
- **Tkinter nicht gefunden:** Installiere Tcl/Tk (z.B. `brew install tcl-tk`) und `pip install tk`
- **Experimentskript hängt:** Prüfe Log-Dateien in `results/logs/` auf Fehler
- **Uploads im Web UI fehlgeschlagen:** Überprüfe Schreibrechte in `uploads/`
- **Weitere Fragen:** Bitte öffne ein Issue im GitHub-Repository unter `https://github.com/NADOOIT/whisperX/issues`

## 15. Appendix

### A. Directory Structure
```bash
whisperX/
├── agent_research_paper/      # Skripte, Guide, Ergebnisse
├── launchpad/                 # Tkinter-GUI (app.py)
├── webui/                     # FastAPI-Server (main.py, templates, static)
├── whisperx/                  # Core-Python-Pakete
├── uploads/                   # Hochgeladene Audio/Transcripts
├── testdata/                  # Testdaten und Flags
└── requirements.txt           # Abhängigkeiten
```

### B. Config File Reference (`default_config.yaml`)
- **dataset:**
  - `path`: Pfad zu Audiodateien
  - `sample_rate`: Audio-Sampling-Rate
- **training:**
  - `epochs`: Anzahl der Zyklen
  - `lora_rank`: Anfangsrang für LoRA
  - `learning_rate`: Lernrate
- **pruning:**
  - `strategy`: z.B. `magnitude`, `structured`
  - `target_sparsity`: Ziel-Dichte
- **evaluation:**
  - `attack_types`: Liste (z.B. `['replay','deepfake']`)
  - `threshold`: $sigma_{thr}$-Wert für Speaker-Verification

### C. Citation & Contact
- Repo: [github.com/NADOOIT/whisperX](https://github.com/NADOOIT/whisperX)
- Bitte zitieren:
  > C. Backhaus, A. Team. "Dynamic LoRA Optimization for Secure Speaker Verification in WhisperX". 2025.
- Issues & Fragen: https://github.com/NADOOIT/whisperX/issues

## 16. Containerization
- Build Docker image:
```bash
docker build -t whisperx:latest .
```
- Run full experiments:
```bash
docker run --rm -v $(pwd):/app whisperx:latest
```
- Run Web UI:
```bash
docker run --rm -p 8000:8000 -v $(pwd)/uploads:/app/uploads whisperx:latest uvicorn webui.main:app --host 0.0.0.0 --port 8000
```
- (Optional) Launch GUI with X11:
```bash
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix whisperx:latest python launchpad/app.py
