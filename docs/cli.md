# 💻 CLI – Kommandozeile

Hier findest du alle Befehle und Optionen für die Nutzung von WhisperX über die Kommandozeile.

---

## ⚡ Schnellstart (CLI)

```bash
whisperx audio.mp3 --speaker_id "speaker1" --adapt_model --enhance_audio
```

### Wichtige Optionen
- `--speaker_id <ID>`: Eindeutige Kennung für den Sprecher
- `--adapt_model`: Aktiviert die Modellanpassung
- `--enhance_audio`: Führt Audio-Optimierung durch
- `--language <code>`: Sprache explizit angeben (z.B. `de`, `en`)

## Beispiele

**Transkription mit Anpassung:**
```bash
whisperx meeting.wav --speaker_id "chef" --adapt_model --enhance_audio
```

**Nur Audio-Optimierung:**
```bash
whisperx call.wav --speaker_id "kunde" --enhance_audio
```

**Batch-Verarbeitung:**
```bash
for f in *.mp3; do whisperx "$f" --speaker_id "user1" --adapt_model; done
```

## Fehlerbehandlung
- Bei Problemen mit Profilen: `--reset_profiles` nutzen
- Für Debug-Logs: `--verbose` angeben
