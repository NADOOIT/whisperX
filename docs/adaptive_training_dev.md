# 🛠️ Adaptive Speaker Model Training – Entwickler-Dokumentation (LoRA/PEFT)

## Ziel
Kontinuierliches, lokales Fine-Tuning von Sprecher-Modellen in WhisperX – **jetzt mit LoRA/PEFT** für maximale Effizienz. Für jeden Sprecher wird ein Adapter-Modell erzeugt und mit neuen Sprachproben weitertrainiert.

---

## Architektur (LoRA-basiert)
- **Profilverwaltung:**
  - Profile speichern Sprachproben, Embeddings und den Pfad zum individuellen LoRA-Adapter.
- **Trainer-Modul:**
  - `SpeakerModelTrainer` übernimmt das Fine-Tuning via LoRA/PEFT auf Basis neuer Sprachproben.
- **Ablage:**
  - Adapter-Modelle werden lokal im Profilordner gespeichert (`~/.cache/whisperx/speaker_models`).
- **Trigger:**
  - Training kann manuell oder automatisch nach n neuen Samples ausgelöst werden.
- **Integration:**
  - AdaptiveProcessor erkennt und lädt den passenden Adapter beim Transkribieren.

---

## LoRA/PEFT: Parameter-Efficient Fine-Tuning (Empfohlen)

**Was ist LoRA?**
- LoRA (Low-Rank Adaptation) und PEFT (Parameter-Efficient Fine-Tuning) ermöglichen es, große Sprachmodelle extrem effizient und ressourcenschonend auf individuelle Sprecher anzupassen.
- Nur kleine Zusatzmodule (Adapter) werden trainiert und gespeichert – das Grundmodell bleibt unverändert und wird von allen Nutzern gemeinsam verwendet.

**Vorteile:**
- **Extrem geringer Speicherbedarf** pro Sprecher (nur wenige MB statt GB!)
- **Schnelles Training**, ideal für lokale und häufige Updates
- Viele Profile parallel möglich (Skalierbarkeit)
- Geringeres Overfitting-Risiko bei wenig Daten

**Vergleich:**

| Methode                  | Speicherbedarf | Geschwindigkeit | Flexibilität | Ideal für ...                |
|--------------------------|---------------|-----------------|--------------|------------------------------|
| Klassisches Fine-Tuning  | Hoch          | Langsam         | Hoch         | Einzelmodell, viel Daten     |
| LoRA/PEFT (Adapter)      | Sehr gering   | Schnell         | Gut          | Viele Nutzer, wenig Daten    |

---

## API-Design (Python)
```python
from whisperx.adaptive import AdaptiveProcessor
from whisperx.adaptive_training import SpeakerModelTrainer

proc = AdaptiveProcessor()
profile = proc.profiles["speaker1"]
trainer = SpeakerModelTrainer()
trainer.train(profile, ["sample1.wav", "sample2.wav"])
```
- `train(profile, new_samples, base_model=None)`: Führt LoRA-Fine-Tuning durch und speichert Adapter.
- `_load_base_model()`: Lädt WhisperX-Basismodell (TODO: Implementierung).

---

## Beispiel: Training mit LoRA (Pseudocode)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq

# Basis-Modell laden
base_model = AutoModelForSpeechSeq2Seq.from_pretrained("whisper-base")

# LoRA-Konfiguration
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# Trainingsdaten vorbereiten (Audio-Features extrahieren)
# ...

# Training durchführen (PyTorch-Loop, nur Adapter-Parameter)
# ...

# Adapter speichern
model.save_pretrained("~/.cache/whisperx/speaker_models/speaker1_lora/")
```

---

## Integration & Best Practices
- LoRA-Adapter nach Training im Profil speichern (Pfad merken)
- Beim Transkribieren Adapter dynamisch zum Basismodell laden
- Training möglichst im Hintergrund laufen lassen
- Nur Adapter-Parameter speichern/übertragen
- Bei wenig Daten: Overfitting durch Early Stopping vermeiden
- **Tipp:** Adapter können zwischen Systemen geteilt werden, das Grundmodell bleibt identisch

---

## TODO
- Trainingslogik für echtes LoRA-Fine-Tuning implementieren (Feature-Extraktion, Trainingsloop, Adapter speichern)
- Integration in AdaptiveProcessor und CLI
- Dokumentation und Beispiele aktuell halten

Fragen oder Vorschläge? [GitHub Issue Tracker](https://github.com/NADOOIT/whisperX/issues)
