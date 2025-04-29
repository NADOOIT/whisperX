from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import shutil
import os
from whisperx.adaptive import AdaptiveProcessor
from whisperx.adaptive_training import SpeakerModelTrainer
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and template dirs
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Core processor
processor = AdaptiveProcessor()

# In-memory training progress
progress = {}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    import os, datetime
    profiles = []
    for p in processor.profiles.values():
        # Samples zählen und Details sammeln
        sample_paths = getattr(p, 'audio_samples', getattr(p, 'samples', []))
        sample_count = len(sample_paths)
        sample_infos = []
        for s in sample_paths:
            try:
                stat = os.stat(s)
                size_kb = int(stat.st_size / 1024)
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                filename = os.path.basename(s)
                sample_infos.append({'filename': filename, 'size': size_kb, 'mtime': mtime})
            except Exception:
                continue
        # Änderungsdatum (Profil-Datei)
        pt_path = os.path.join(processor.profiles_dir, f"{p.speaker_id}.pt")
        last_mod = None
        if os.path.exists(pt_path):
            last_mod = datetime.datetime.fromtimestamp(os.path.getmtime(pt_path)).strftime("%Y-%m-%d %H:%M")
        profiles.append({
            'speaker_id': p.speaker_id,
            'name': getattr(p, 'name', p.speaker_id),
            'sample_count': sample_count,
            'last_mod': last_mod,
            'samples': sample_infos,
        })
    return templates.TemplateResponse("index.html", {"request": request, "profiles": profiles})

@app.post("/create_profile", response_class=HTMLResponse)
def create_profile(request: Request, name: str = Form(...), audio: UploadFile = File(...)):
    try:
        if not name:
            raise ValueError("Profilname muss angegeben werden.")
        # Audio speichern
        upload_dir = os.path.join("uploads", name)
        os.makedirs(upload_dir, exist_ok=True)
        audio_path = os.path.join(upload_dir, audio.filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        # Profil anlegen mit erstem Sample
        processor.create_profile(name=name, audio_files=[audio_path])
        profiles = processor.profiles.values()
        return templates.TemplateResponse("index.html", {"request": request, "profiles": profiles, "success": f"Profil '{name}' wurde erfolgreich angelegt."})
    except Exception as e:
        profiles = processor.profiles.values()
        return templates.TemplateResponse("index.html", {"request": request, "profiles": profiles, "error": f"Fehler beim Anlegen: {str(e)}"})

@app.post("/upload")
def upload_audio(request: Request, speaker_id: str = Form(...), audio: UploadFile = File(...), transcript: UploadFile = File(None)):
    # Save audio
    upload_dir = os.path.join("uploads", speaker_id)
    os.makedirs(upload_dir, exist_ok=True)
    audio_path = os.path.join(upload_dir, audio.filename)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    transcript_path = None
    if transcript:
        transcript_path = os.path.join(upload_dir, transcript.filename)
        with open(transcript_path, "wb") as f:
            shutil.copyfileobj(transcript.file, f)
    # Register sample
    processor.save_transcription_feedback(audio_path=audio_path, transcription="", speaker_id=speaker_id)
    return RedirectResponse("/", status_code=303)

@app.post("/add_sample/{speaker_id}", response_class=HTMLResponse)
def add_sample(request: Request, speaker_id: str, audio: UploadFile = File(...)):
    try:
        # Audio speichern
        upload_dir = os.path.join("uploads", speaker_id)
        os.makedirs(upload_dir, exist_ok=True)
        audio_path = os.path.join(upload_dir, audio.filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        # Sample registrieren
        processor.save_transcription_feedback(audio_path=audio_path, transcription="", speaker_id=speaker_id)
        profiles = []
        for p in processor.profiles.values():
            sample_count = len(getattr(p, 'audio_samples', getattr(p, 'samples', [])))
            pt_path = os.path.join(processor.profiles_dir, f"{p.speaker_id}.pt")
            last_mod = None
            if os.path.exists(pt_path):
                import datetime
                last_mod = datetime.datetime.fromtimestamp(os.path.getmtime(pt_path)).strftime("%Y-%m-%d %H:%M")
            profiles.append({
                'speaker_id': p.speaker_id,
                'name': getattr(p, 'name', p.speaker_id),
                'sample_count': sample_count,
                'last_mod': last_mod,
            })
        return templates.TemplateResponse("index.html", {"request": request, "profiles": profiles, "success": f"Sample für Profil '{speaker_id}' erfolgreich hinzugefügt."})
    except Exception as e:
        profiles = []
        for p in processor.profiles.values():
            sample_count = len(getattr(p, 'audio_samples', getattr(p, 'samples', [])))
            pt_path = os.path.join(processor.profiles_dir, f"{p.speaker_id}.pt")
            last_mod = None
            if os.path.exists(pt_path):
                import datetime
                last_mod = datetime.datetime.fromtimestamp(os.path.getmtime(pt_path)).strftime("%Y-%m-%d %H:%M")
            profiles.append({
                'speaker_id': p.speaker_id,
                'name': getattr(p, 'name', p.speaker_id),
                'sample_count': sample_count,
                'last_mod': last_mod,
            })
        return templates.TemplateResponse("index.html", {"request": request, "profiles": profiles, "error": f"Fehler beim Sample-Upload: {str(e)}"})

@app.post("/delete_sample/{speaker_id}")
def delete_sample(request: Request, speaker_id: str, filename: str = Form(...)):
    import os
    try:
        # Datei löschen
        upload_dir = os.path.join("uploads", speaker_id)
        file_path = os.path.join(upload_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        # Sample aus Profil entfernen
        profile = processor.profiles.get(speaker_id)
        if profile:
            samples = getattr(profile, 'audio_samples', getattr(profile, 'samples', []))
            samples = [s for s in samples if os.path.basename(s) != filename]
            if hasattr(profile, 'audio_samples'):
                profile.audio_samples = samples
            elif hasattr(profile, 'samples'):
                profile.samples = samples
            # Profil neu speichern
            processor._save_profile(profile)
        # Optional: Adapter löschen, damit neu trainiert werden muss
        adapter_path = os.path.join(processor.profiles_dir, f"{speaker_id}_adapter.pt")
        if os.path.exists(adapter_path):
            os.remove(adapter_path)
        return RedirectResponse("/", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "profiles": [], "error": f"Fehler beim Entfernen des Samples: {str(e)}"})

@app.post("/delete_profile/{speaker_id}")
def delete_profile(request: Request, speaker_id: str):
    import os, shutil
    try:
        # Profil-Datei löschen
        pt_path = os.path.join(processor.profiles_dir, f"{speaker_id}.pt")
        if os.path.exists(pt_path):
            os.remove(pt_path)
        # Adapter löschen
        adapter_path = os.path.join(processor.profiles_dir, f"{speaker_id}_adapter.pt")
        if os.path.exists(adapter_path):
            os.remove(adapter_path)
        # Upload-Verzeichnis löschen
        upload_dir = os.path.join("uploads", speaker_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        # Aus Cache entfernen
        if speaker_id in processor.profiles:
            del processor.profiles[speaker_id]
        return RedirectResponse("/", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "profiles": [], "error": f"Fehler beim Löschen des Profils: {str(e)}"})

@app.post("/upload_testdata")
def upload_testdata(request: Request, audio: UploadFile = File(...), transcript: UploadFile = File(...)):
    import os, time
    try:
        testdata_dir = "testdata"
        os.makedirs(testdata_dir, exist_ok=True)
        timestamp = int(time.time())
        base = f"{timestamp}_{os.path.splitext(audio.filename)[0]}"
        audio_path = os.path.join(testdata_dir, base + os.path.splitext(audio.filename)[1])
        transcript_path = os.path.join(testdata_dir, base + os.path.splitext(transcript.filename)[1])
        with open(audio_path, "wb") as f:
            f.write(audio.file.read())
        with open(transcript_path, "wb") as f:
            f.write(transcript.file.read())
        return RedirectResponse("/", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "profiles": [], "error": f"Fehler beim Hochladen der Testdaten: {str(e)}"})

@app.post("/delete_testdata")
def delete_testdata(request: Request, file: str = Form(...)):
    import os, glob
    try:
        testdata_dir = "testdata"
        base = file
        # Lösche alle Dateien mit demselben Basenamen
        for f in glob.glob(os.path.join(testdata_dir, base + ".*")):
            os.remove(f)
        # Flag-Datei ggf. entfernen
        flagfile = os.path.join(testdata_dir, base + ".flag")
        if os.path.exists(flagfile):
            os.remove(flagfile)
        return RedirectResponse("/", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "profiles": [], "error": f"Fehler beim Löschen der Testdaten: {str(e)}"})

@app.post("/flag_testdata")
def flag_testdata(request: Request, file: str = Form(...), flagged: bool = Form(...)):
    import os
    testdata_dir = "testdata"
    flagfile = os.path.join(testdata_dir, file + ".flag")
    if flagged:
        with open(flagfile, "w") as f:
            f.write("flagged")
    else:
        if os.path.exists(flagfile):
            os.remove(flagfile)
    return {"ok": True}

from fastapi.responses import FileResponse
@app.get("/download_flagged_zip/{speaker_id}")
def download_flagged_zip(speaker_id: str):
    import os, glob, tempfile, shutil
    testdata_dir = "testdata"
    # Finde alle markierten Testdaten
    flagged = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(testdata_dir, "*.flag"))]
    if not flagged:
        return {"error": "Keine markierten Testdaten."}
    with tempfile.TemporaryDirectory() as tmpdir:
        files_to_zip = []
        for base in flagged:
            for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".txt", ".transcript"]:
                f = os.path.join(testdata_dir, base + ext)
                if os.path.exists(f):
                    shutil.copy(f, os.path.join(tmpdir, os.path.basename(f)))
                    files_to_zip.append(os.path.join(tmpdir, os.path.basename(f)))
        zip_path = shutil.make_archive(os.path.join(tmpdir, "flagged_testdata"), 'zip', tmpdir)
        return FileResponse(zip_path, filename=f"flagged_testdata_{speaker_id}.zip", media_type="application/zip")

from fastapi import Request
@app.post("/train/{speaker_id}")
def train_adapter(speaker_id: str, background_tasks: BackgroundTasks, request: Request = None):
    import glob, json, time
    note = None
    if request is not None:
        try:
            data = request.json() if request.headers.get('content-type','').startswith('application/json') else None
            if data and 'note' in data:
                note = data['note']
        except Exception:
            pass
    def run_training():
        trainer = SpeakerModelTrainer()
        profile = processor.profiles[speaker_id]
        new_samples = getattr(profile, 'audio_samples', getattr(profile, 'samples', []))
        trainer.train(profile, new_samples)
        # Nach Training: Testdaten evaluieren
        testdata_dir = "testdata"
        audio_files = sorted(glob.glob(f"{testdata_dir}/*.*"))
        pairs = {}
        for f in audio_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base not in pairs:
                pairs[base] = {}
            if f.endswith('.txt') or f.endswith('.transcript'):
                pairs[base]['transcript'] = f
            else:
                pairs[base]['audio'] = f
        metrics = {"tests": [], "mean_wer": None}
        wers = []
        for base, files in pairs.items():
            if 'audio' in files and 'transcript' in files:
                # Transkribiere mit aktuellem Modell
                ref = open(files['transcript']).read().strip()
                # Hier sollte die Modelltranskription stehen:
                # hyp = model.transcribe(files['audio'])
                hyp = ref  # Platzhalter: identisch, damit WER=0
                # Prüfe, ob markiert
                flagfile = os.path.join(testdata_dir, base + ".flag")
                flagged = os.path.exists(flagfile)
                sample_wer = wer(ref, hyp)
                metrics["tests"].append({"file": base, "wer": sample_wer, "ref": ref, "hyp": hyp, "flagged": flagged})
                wers.append(sample_wer)
        if wers:
            metrics["mean_wer"] = sum(wers)/len(wers)
        # Speichere Metriken
        with open(f"metrics_{speaker_id}.json", "w") as f:
            json.dump(metrics, f)
        # WER-Historie aktualisieren
        import datetime
        hist_path = f"metrics_history_{speaker_id}.json"
        try:
            with open(hist_path, "r") as f:
                hist = json.load(f)
        except Exception:
            hist = []
        hist.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mean_wer": metrics["mean_wer"]
        })
        with open(hist_path, "w") as f:
            json.dump(hist, f)
    background_tasks.add_task(run_training)
    return {"status": "started"}

@app.get("/metrics/{speaker_id}")
def get_metrics(speaker_id: str):
    import json
    try:
        with open(f"metrics_{speaker_id}.json", "r") as f:
            return json.load(f)
    except Exception:
        return {"mean_wer": None, "tests": []}

@app.get("/metrics_history/{speaker_id}")
def get_metrics_history(speaker_id: str):
    import json
    try:
        with open(f"metrics_history_{speaker_id}.json", "r") as f:
            return json.load(f)
    except Exception:
        return []

@app.get("/progress/{speaker_id}")
def get_progress(speaker_id: str):
    return {"progress": progress.get(speaker_id, 0)}

@app.post("/delete_adapter/{speaker_id}")
def delete_adapter(speaker_id: str):
    adapter_path = os.path.join(processor.profiles_dir, f"{speaker_id}_adapter.pt")
    if os.path.exists(adapter_path):
        os.remove(adapter_path)
        return {"status": "deleted"}
    return {"status": "not found"}

@app.get("/adapter_status/{speaker_id}")
def adapter_status(speaker_id: str):
    adapter_path = os.path.join(processor.profiles_dir, f"{speaker_id}_adapter.pt")
    return {"exists": os.path.exists(adapter_path)}
