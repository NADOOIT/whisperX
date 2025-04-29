# FAQ – Häufige Fragen

## Wie viele Sprecherprofile kann ich anlegen?
Es gibt keine feste Begrenzung, aber viele Profile können die Systemleistung beeinflussen.

## Kann ich Profile zwischen Systemen übertragen?
Ja, kopiere den Profilordner (`~/.cache/whisperx/voice_profiles`) auf das Zielsystem.

## Was tun bei schlechter Transkriptionsqualität?
- Audioqualität prüfen
- Mehr Sprachproben sammeln
- Feedback-Loop nutzen

---

## 🧑‍💻 Nutzung

**Was tun bei schlechter Transkriptionsqualität?**
> Prüfe die Audioqualität, sammle mehr Sprachproben und nutze den Feedback-Loop.

**Unterstützt WhisperX mehrere Sprachen?**
Ja, gib beim Profil die Sprache an (z.B. `language="en"`).

```python
from whisperx.adaptive import AdaptiveProcessor
proc = AdaptiveProcessor()
profile = proc.create_voice_profile("audio.wav", "user1", language="en")
```

**Kann ich WhisperX in eigene Tools integrieren?**
Ja, über die Python API und CLI.

```bash
whisperx audio.mp3 --speaker_id "user1" --adapt_model
```

---

## 🛠️ Probleme & Lösungen

**Das Modell erkennt den Sprecher nicht richtig – was tun?**
> Prüfe, ob das Profil aktuell ist und genügend Sprachproben enthält.

**Fehler beim Start?**
> Siehe [Troubleshooting](./troubleshooting.md) und prüfe die Abhängigkeiten.

---

## 📞 Support

**Wie kann ich Fehler melden oder Feedback geben?**
Eröffne ein [GitHub Issue](https://github.com/NADOOIT/whisperX/issues) oder schreibe an support@nadoo.de

**Werden meine Daten sicher gespeichert?**
Siehe [Sicherheit & Datenschutz](./security.md)
