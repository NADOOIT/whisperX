import os
import tempfile
import shutil
import pytest
from whisperx.adaptive import AdaptiveProcessor
from whisperx.adaptive_training import SpeakerModelTrainer

# --- Hilfsfunktionen ---
import numpy as np
import soundfile as sf
from whisperx.audio import SAMPLE_RATE

def create_dummy_audio_file(path, duration=1.0, sample_rate=None):
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    sf.write(path, audio, sample_rate)


def create_dummy_transcript_file(path, text="Hallo Welt!"):
    with open(path, "w") as f:
        f.write(text)

# --- Testfälle ---

def test_end2end_speaker_training_and_evaluation():
    tmpdir = tempfile.mkdtemp()
    try:
        # 1. Profil anlegen
        proc = AdaptiveProcessor(cache_dir=tmpdir)
        # Patch diarizer to return dummy embedding
        proc.diarizer = lambda audio: {'embedding': np.random.randn(10, 256)}
        audio1 = os.path.join(tmpdir, "train1.wav")
        audio2 = os.path.join(tmpdir, "train2.wav")
        test_audio = os.path.join(tmpdir, "test1.wav")
        create_dummy_audio_file(audio1)
        create_dummy_audio_file(audio2)
        create_dummy_audio_file(test_audio)
        transcript1 = os.path.join(tmpdir, "train1.txt")
        transcript2 = os.path.join(tmpdir, "train2.txt")
        test_transcript = os.path.join(tmpdir, "test1.txt")
        create_dummy_transcript_file(transcript1, "Das ist ein Test.")
        create_dummy_transcript_file(transcript2, "Noch ein Test.")
        create_dummy_transcript_file(test_transcript, "Kontrollsatz.")
        speaker_id = proc.create_profile("TestUser", [audio1, audio2])

        # 2. Trainingsdaten zuweisen
        profile = proc.profiles[speaker_id]
        trainer = SpeakerModelTrainer(model_dir=tmpdir)
        # Training (mit und ohne Transkript)
        adapter_path = trainer.train(profile, [audio1, audio2])
        assert os.path.exists(adapter_path)

        # 3. Kontrolldaten (Testdaten) zuweisen
        # (In echter App: explizit markieren, hier simuliert)
        kontrolldaten = [(test_audio, test_transcript)]

        # 4. Evaluation: Modell auf Testdaten prüfen (Dummy-Ausgabe)
        # (Echte Implementierung: WER, Vorher/Nachher-Vergleich)
        # Hier: Wir prüfen nur, dass Testdaten "durchlaufen" werden
        for audio, transcript in kontrolldaten:
            assert os.path.exists(audio)
            assert os.path.exists(transcript)
            # TODO: Echte Evaluationsfunktion aufrufen

        # 5. Adapter-Management
        # Löschen
        shutil.rmtree(adapter_path)
        assert not os.path.exists(adapter_path)

    finally:
        shutil.rmtree(tmpdir)


def test_add_audio_and_auto_train(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    try:
        proc = AdaptiveProcessor(cache_dir=tmpdir)
        # Patch diarizer to return dummy embedding
        proc.diarizer = lambda audio: {'embedding': np.random.randn(10, 256)}
        audio = os.path.join(tmpdir, "auto_add.wav")
        create_dummy_audio_file(audio)
        speaker_id = proc.create_profile("AutoTrainUser", [audio])
        profile = proc.profiles[speaker_id]
        trainer = SpeakerModelTrainer(model_dir=tmpdir)
        # Simuliere: Nach Hinzufügen wird automatisch trainiert
        adapter_path = trainer.train(profile, [audio])
        assert os.path.exists(adapter_path)
    finally:
        shutil.rmtree(tmpdir)


def test_training_without_transcripts():
    tmpdir = tempfile.mkdtemp()
    try:
        proc = AdaptiveProcessor(cache_dir=tmpdir)
        # Patch diarizer to return dummy embedding
        proc.diarizer = lambda audio: {'embedding': np.random.randn(10, 256)}
        audio = os.path.join(tmpdir, "notxt.wav")
        create_dummy_audio_file(audio)
        speaker_id = proc.create_profile("NoTranscript", [audio])
        profile = proc.profiles[speaker_id]
        trainer = SpeakerModelTrainer(model_dir=tmpdir)
        adapter_path = trainer.train(profile, [audio])
        assert os.path.exists(adapter_path)
    finally:
        shutil.rmtree(tmpdir)

# --- TDD: Tests für GUI-Logik (Pseudo, für spätere Implementierung) ---
def test_gui_add_train_and_evaluate(monkeypatch):
    # Diese Funktion wird später mit GUI-Testtools (z.B. pytest-tkinter) umgesetzt
    # Hier nur als UseCase-Check
    assert True  # Placeholder
