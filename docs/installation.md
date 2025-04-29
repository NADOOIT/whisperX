# ⚙️ Installation & Voraussetzungen

Hier erfährst du, wie du WhisperX und die adaptiven Features korrekt installierst.

---

## 🛠️ Systemvoraussetzungen

- **Python:** 3.8 oder neuer
- **Betriebssystem:** Linux, macOS oder Windows
- **Optional:** CUDA-fähige GPU für schnellere Verarbeitung
- **Empfohlen:** 4+ CPU-Kerne, ausreichend RAM

---

## 🚀 Installation Schritt für Schritt

1. **Repository klonen:**
   ```bash
   git clone https://github.com/NADOOIT/whisperX.git
   cd whisperX
   ```
2. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements_adaptive.txt
   ```
3. **(Optional) CUDA/GPU-Unterstützung:**
   - NVIDIA-Treiber und CUDA installieren
   - `torch` mit GPU-Support installieren

> 💡 **Tipp:** Für CPU-only-Betrieb reicht die Standardinstallation.

---

## 🔄 Aktualisierung

```bash
git pull
pip install -r requirements_adaptive.txt --upgrade
```

---

## 🆘 Häufige Installationsprobleme & Lösungen

| Fehlermeldung                | Ursache                            | Lösung                                       |
|-----------------------------|------------------------------------|----------------------------------------------|
| Profile not found            | Profilname falsch/Profil fehlt      | Profilnamen prüfen, Profil neu anlegen       |
| Permission denied            | Keine Schreibrechte im Profilordner | Rechte prüfen, ggf. als Admin ausführen      |
| CUDA device not found        | Keine/nicht erkannte GPU            | CUDA & Treiber prüfen, ggf. CPU-Modus nutzen |
| Audio too short              | Sprachprobe zu kurz                 | Längere, klarere Probe aufnehmen             |
| ImportError/ModuleNotFound   | Abhängigkeit fehlt                  | `pip install -r requirements_adaptive.txt`   |
| AssertionError in Test       | Paket nicht korrekt installiert     | Installation prüfen, ggf. neu installieren   |

> ℹ️ **Weitere Hilfe:**
> - [FAQ](./faq.md)
> - [GitHub Issues](https://github.com/NADOOIT/whisperX/issues)
