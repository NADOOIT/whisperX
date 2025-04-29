"""Speaker management window for NADOO Launchpad."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any
from whisperx.adaptive import AdaptiveProcessor, VoiceProfile

class SpeakerManagementWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Speaker Management")
        self.geometry("700x450")

        # Initialize adaptive processor
        self.processor = AdaptiveProcessor()

        # Notebook mit zwei Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Speaker Profiles (bestehende GUI)
        self.tab_profiles = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_profiles, text="Speaker Profiles")
        self._create_profile_widgets(self.tab_profiles)
        self._load_profiles()

        # Tab 2: Speaker Training (neu)
        self.tab_training = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_training, text="Speaker Training")
        self._create_training_widgets(self.tab_training)

    def _create_profile_widgets(self, parent):
        """Create the widgets for the profiles tab."""
        # Profile List
        list_frame = ttk.LabelFrame(parent, text="Speaker Profiles")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.profile_list = ttk.Treeview(
            list_frame,
            columns=("ID", "Language", "Samples"),
            show="headings"
        )
        self.profile_list.heading("ID", text="Speaker ID")
        self.profile_list.heading("Language", text="Language")
        self.profile_list.heading("Samples", text="Samples")
        self.profile_list.pack(fill=tk.BOTH, expand=True)

        # Buttons Frame
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(
            btn_frame,
            text="New Profile",
            command=self._create_profile
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            btn_frame,
            text="Add Sample",
            command=self._add_sample
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            btn_frame,
            text="Delete Profile",
            command=self._delete_profile
        ).pack(fill=tk.X, pady=2)

        # Settings Frame
        settings_frame = ttk.LabelFrame(btn_frame, text="Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        self.adapt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Enable Adaptation",
            variable=self.adapt_var
        ).pack(fill=tk.X)
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Enable Enhancement",
            variable=self.enhance_var
        ).pack(fill=tk.X)

    def _create_training_widgets(self, parent):
        """Create the widgets for the speaker training tab."""
        import sys
        # Profil-Auswahl
        profile_frame = ttk.LabelFrame(parent, text="Select Speaker Profile")
        profile_frame.pack(fill=tk.X, padx=5, pady=5)
        self.training_profile_var = tk.StringVar()
        self.training_profile_combo = ttk.Combobox(profile_frame, textvariable=self.training_profile_var, state="readonly")
        self.training_profile_combo.pack(fill=tk.X, padx=5, pady=5)
        self._refresh_training_profiles()

        # Upload Audio mit Drag & Drop
        upload_frame = ttk.LabelFrame(parent, text="Upload Training Audio")
        upload_frame.pack(fill=tk.X, padx=5, pady=5)
        self.audio_path_var = tk.StringVar()
        self.audio_entry = ttk.Entry(upload_frame, textvariable=self.audio_path_var, state="readonly")
        self.audio_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(upload_frame, text="Browse", command=self._browse_audio_training).pack(side=tk.LEFT, padx=5)
        # Drag & Drop für Audio
        try:
            import tkdnd
            self.tk.call('package', 'require', 'tkdnd')
            self.audio_entry.drop_target_register('DND_Files')
            self.audio_entry.dnd_bind('<<Drop>>', self._on_audio_drop)
        except Exception:
            pass  # Fällt bei fehlender tkdnd elegant zurück

        # Optional Transkript mit Drag & Drop
        transcript_frame = ttk.LabelFrame(parent, text="Optional: Upload Transcript")
        transcript_frame.pack(fill=tk.X, padx=5, pady=5)
        self.transcript_path_var = tk.StringVar()
        self.transcript_entry = ttk.Entry(transcript_frame, textvariable=self.transcript_path_var, state="readonly")
        self.transcript_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(transcript_frame, text="Browse", command=self._browse_transcript_training).pack(side=tk.LEFT, padx=5)
        try:
            import tkdnd
            self.tk.call('package', 'require', 'tkdnd')
            self.transcript_entry.drop_target_register('DND_Files')
            self.transcript_entry.dnd_bind('<<Drop>>', self._on_transcript_drop)
        except Exception:
            pass

        # Trainings-Button
        train_btn = ttk.Button(parent, text="Start Training", command=self._start_training)
        train_btn.pack(fill=tk.X, padx=10, pady=10)

        # Fortschrittsanzeige
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, padx=10, pady=2)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progressbar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progressbar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.training_status_var = tk.StringVar(value="Status: Ready")
        self.training_status_label = ttk.Label(progress_frame, textvariable=self.training_status_var)
        self.training_status_label.pack(side=tk.LEFT, padx=10)

        # Adapter-Management
        adapter_frame = ttk.LabelFrame(parent, text="Adapter Management")
        adapter_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(adapter_frame, text="Delete Adapter", command=self._delete_adapter).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(adapter_frame, text="Retrain Adapter", command=self._retrain_adapter).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(adapter_frame, text="Show Adapter Status", command=self._show_adapter_status).pack(side=tk.LEFT, padx=5, pady=5)

    def _on_audio_drop(self, event):
        files = event.data.strip().split()
        if files:
            self.audio_path_var.set(files[0])

    def _on_transcript_drop(self, event):
        files = event.data.strip().split()
        if files:
            self.transcript_path_var.set(files[0])

    def _start_training(self):
        import threading, time
        profile_id = self.training_profile_var.get()
        audio_path = self.audio_path_var.get()
        transcript_path = self.transcript_path_var.get()
        if not profile_id or not audio_path:
            self.training_status_var.set("Status: Please select profile and audio file.")
            return
        def run_training():
            try:
                self.progress_var.set(10)
                self.training_status_var.set("Status: Adding sample...")
                self.processor.save_transcription_feedback(audio_path=audio_path, transcription="", speaker_id=profile_id)
                self.progress_var.set(30)
                self.training_status_var.set("Status: Training...")
                from whisperx.adaptive_training import SpeakerModelTrainer
                profile = self.processor.voice_profiles[profile_id]
                trainer = SpeakerModelTrainer(model_dir=self.processor.profiles_dir)
                # TODO: Trainingsdaten sammeln, optional Transkript übergeben
                adapter_path = trainer.train(profile, [audio_path])
                self.progress_var.set(100)
                self.training_status_var.set(f"Status: Training complete. Adapter saved at {adapter_path}")
                self.audio_path_var.set("")
                self.transcript_path_var.set("")
                self._refresh_training_profiles()
                time.sleep(0.5)
                self.progress_var.set(0)
            except Exception as e:
                self.training_status_var.set(f"Status: Training failed: {str(e)}")
                self.progress_var.set(0)
        threading.Thread(target=run_training, daemon=True).start()

    def _delete_adapter(self):
        import os
        profile_id = self.training_profile_var.get()
        if not profile_id:
            self.training_status_var.set("Status: Please select profile.")
            return
        try:
            adapter_path = os.path.join(self.processor.profiles_dir, f"{profile_id}_adapter.pt")
            if os.path.exists(adapter_path):
                os.remove(adapter_path)
                self.training_status_var.set(f"Status: Adapter deleted for {profile_id}.")
            else:
                self.training_status_var.set("Status: No adapter found to delete.")
        except Exception as e:
            self.training_status_var.set(f"Status: Delete failed: {str(e)}")

    def _retrain_adapter(self):
        # Ruft einfach _start_training erneut auf (kann angepasst werden)
        self._start_training()

    def _show_adapter_status(self):
        import os
        profile_id = self.training_profile_var.get()
        if not profile_id:
            self.training_status_var.set("Status: Please select profile.")
            return
        adapter_path = os.path.join(self.processor.profiles_dir, f"{profile_id}_adapter.pt")
        if os.path.exists(adapter_path):
            self.training_status_var.set(f"Status: Adapter exists at {adapter_path}")
        else:
            self.training_status_var.set("Status: No adapter found.")

    def _refresh_training_profiles(self):
        """Refresh the profile list for training tab."""
        profiles = list(self.processor.voice_profiles.keys())
        self.training_profile_combo['values'] = profiles
        if profiles:
            self.training_profile_combo.current(0)

    def _browse_audio_training(self):
        path = filedialog.askopenfilename(title="Select Training Audio", filetypes=[("Audio Files", "*.mp3 *.wav *.m4a"), ("All Files", "*.*")])
        if path:
            self.audio_path_var.set(path)

    def _browse_transcript_training(self):
        path = filedialog.askopenfilename(title="Select Transcript", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if path:
            self.transcript_path_var.set(path)

    def _start_training(self):
        profile_id = self.training_profile_var.get()
        audio_path = self.audio_path_var.get()
        transcript_path = self.transcript_path_var.get()
        if not profile_id or not audio_path:
            self.training_status_var.set("Status: Please select profile and audio file.")
            return
        try:
            # TODO: Optional Transkript nutzen
            profile = self.processor.voice_profiles[profile_id]
            # Minimal: Sample hinzufügen
            self.processor.save_transcription_feedback(audio_path=audio_path, transcription="", speaker_id=profile_id)
            # Training auslösen (Dummy/Platzhalter)
            from whisperx.adaptive_training import SpeakerModelTrainer
            trainer = SpeakerModelTrainer(model_dir=self.processor.profiles_dir)
            # TODO: Trainingsdaten sammeln, optional Transkript übergeben
            adapter_path = trainer.train(profile, [audio_path])
            self.training_status_var.set(f"Status: Training complete. Adapter saved at {adapter_path}")
            self._refresh_training_profiles()
        except Exception as e:
            self.training_status_var.set(f"Status: Training failed: {str(e)}")

    def _delete_adapter(self):
        # TODO: Adapter für gewähltes Profil löschen
        self.training_status_var.set("Status: Delete Adapter (not yet implemented)")

    def _retrain_adapter(self):
        # TODO: Adapter für gewähltes Profil neu trainieren
        self.training_status_var.set("Status: Retrain Adapter (not yet implemented)")

    def _show_adapter_status(self):
        # TODO: Adapter-Status anzeigen
        self.training_status_var.set("Status: Show Adapter Status (not yet implemented)")

    def _create_widgets(self):
        """Create the window widgets."""
        # Profile List
        list_frame = ttk.LabelFrame(self, text="Speaker Profiles")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.profile_list = ttk.Treeview(
            list_frame,
            columns=("ID", "Language", "Samples"),
            show="headings"
        )
        self.profile_list.heading("ID", text="Speaker ID")
        self.profile_list.heading("Language", text="Language")
        self.profile_list.heading("Samples", text="Samples")
        self.profile_list.pack(fill=tk.BOTH, expand=True)
        
        # Buttons Frame
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Button(
            btn_frame,
            text="New Profile",
            command=self._create_profile
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text="Add Sample",
            command=self._add_sample
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text="Delete Profile",
            command=self._delete_profile
        ).pack(fill=tk.X, pady=2)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(btn_frame, text="Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        
        self.adapt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Enable Adaptation",
            variable=self.adapt_var
        ).pack(fill=tk.X)
        
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Enable Enhancement",
            variable=self.enhance_var
        ).pack(fill=tk.X)

    def _load_profiles(self):
        """Load existing speaker profiles."""
        self.profile_list.delete(*self.profile_list.get_children())
        
        for speaker_id, profile in self.processor.voice_profiles.items():
            self.profile_list.insert(
                "",
                tk.END,
                values=(
                    speaker_id,
                    profile.language,
                    len(profile.samples)
                )
            )

    def _create_profile(self):
        """Create a new speaker profile."""
        dialog = NewProfileDialog(self)
        if dialog.result:
            speaker_id, language, audio_path = dialog.result
            try:
                profile = self.processor.create_voice_profile(
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    language=language
                )
                self._load_profiles()
                messagebox.showinfo(
                    "Success",
                    f"Created profile for speaker {speaker_id}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to create profile: {str(e)}"
                )

    def _add_sample(self):
        """Add a sample to existing profile."""
        selection = self.profile_list.selection()
        if not selection:
            messagebox.showwarning(
                "Warning",
                "Please select a profile first"
            )
            return
            
        speaker_id = self.profile_list.item(selection[0])["values"][0]
        audio_path = filedialog.askopenfilename(
            title="Select Audio Sample",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a"),
                ("All Files", "*.*")
            ]
        )
        
        if audio_path:
            try:
                self.processor.save_transcription_feedback(
                    audio_path=audio_path,
                    transcription="",  # No transcription needed
                    speaker_id=speaker_id
                )
                self._load_profiles()
                messagebox.showinfo(
                    "Success",
                    "Added sample to profile"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to add sample: {str(e)}"
                )

    def _delete_profile(self):
        """Delete selected profile."""
        selection = self.profile_list.selection()
        if not selection:
            messagebox.showwarning(
                "Warning",
                "Please select a profile to delete"
            )
            return
            
        speaker_id = self.profile_list.item(selection[0])["values"][0]
        if messagebox.askyesno(
            "Confirm Delete",
            f"Delete profile for {speaker_id}?"
        ):
            try:
                profile_path = os.path.join(
                    self.processor.profiles_dir,
                    f"{speaker_id}.pt"
                )
                os.remove(profile_path)
                del self.processor.voice_profiles[speaker_id]
                self._load_profiles()
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to delete profile: {str(e)}"
                )

class NewProfileDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("New Speaker Profile")
        self.result = None
        
        # Speaker ID
        ttk.Label(self, text="Speaker ID:").grid(row=0, column=0, padx=5, pady=5)
        self.id_entry = ttk.Entry(self)
        self.id_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Language
        ttk.Label(self, text="Language:").grid(row=1, column=0, padx=5, pady=5)
        self.lang_entry = ttk.Entry(self)
        self.lang_entry.insert(0, "en")
        self.lang_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Audio Sample
        ttk.Label(self, text="Audio Sample:").grid(row=2, column=0, padx=5, pady=5)
        self.path_var = tk.StringVar()
        ttk.Entry(
            self,
            textvariable=self.path_var,
            state="readonly"
        ).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Button(
            self,
            text="Browse",
            command=self._browse_audio
        ).grid(row=2, column=2, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Create",
            command=self._create
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side=tk.LEFT, padx=5)

    def _browse_audio(self):
        """Browse for audio sample."""
        path = filedialog.askopenfilename(
            title="Select Audio Sample",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.path_var.set(path)

    def _create(self):
        """Create new profile."""
        speaker_id = self.id_entry.get().strip()
        language = self.lang_entry.get().strip()
        audio_path = self.path_var.get()
        
        if not all([speaker_id, language, audio_path]):
            messagebox.showwarning(
                "Warning",
                "Please fill all fields"
            )
            return
            
        self.result = (speaker_id, language, audio_path)
        self.destroy()
