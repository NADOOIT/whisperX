# 🛠️ Troubleshooting & Fehlerbehebung

Hier findest du eine Übersicht häufiger Fehler, deren Ursachen und Lösungen rund um WhisperX und die adaptiven Features.

---

## 🚫 Häufige Fehlermeldungen

| Fehlermeldung                | Ursache                            | Lösung                                       |
|-----------------------------|------------------------------------|----------------------------------------------|
| Profile not found            | Profilname falsch/Profil fehlt      | > **Lösung:** Profilnamen prüfen, Profil neu anlegen       |
| Permission denied            | Keine Schreibrechte im Profilordner | > **Lösung:** Rechte prüfen, ggf. als Admin ausführen      |
| CUDA device not found        | Keine/nicht erkannte GPU            | > **Lösung:** CUDA & Treiber prüfen, ggf. CPU-Modus nutzen |
| Audio too short              | Sprachprobe zu kurz                 | > **Lösung:** Längere, klarere Probe aufnehmen             |
| ImportError/ModuleNotFound   | Abhängigkeit fehlt                  | > **Lösung:** `pip install -r requirements_adaptive.txt`   |
| AssertionError in Test       | Paket nicht korrekt installiert     | > **Lösung:** Installation prüfen, ggf. neu installieren   |
| Permission denied            | Keine Schreibrechte im Profilordner | Rechte prüfen, ggf. als Admin ausführen      |
| CUDA device not found        | Keine/nicht erkannte GPU            | CUDA & Treiber prüfen, ggf. CPU-Modus nutzen |
| Audio too short              | Sprachprobe zu kurz                 | Längere, klarere Probe aufnehmen             |
| ImportError/ModuleNotFound   | Abhängigkeit fehlt                  | `pip install -r requirements_adaptive.txt`   |
| AssertionError in Test       | Paket nicht korrekt installiert     | Installation prüfen, ggf. neu installieren   |

## Tipps zur Fehleranalyse
- Log-Ausgaben mit `--verbose` aktivieren
- Python-Tracebacks genau lesen
- Community/Forum nutzen, wenn du nicht weiterkommst

## Weitere Hilfestellungen
- [FAQ](./faq.md)
- [GitHub Issues](https://github.com/NADOOIT/whisperX/issues)
