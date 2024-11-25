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
        self.geometry("600x400")
        
        # Initialize adaptive processor
        self.processor = AdaptiveProcessor()
        
        self._create_widgets()
        self._load_profiles()

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
