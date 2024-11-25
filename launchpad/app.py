"""NADOO Launchpad - WhisperX GUI Application."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any
from whisperx import load_model
from whisperx.adaptive import AdaptiveProcessor
from .speaker_management import SpeakerManagementWindow
from .recorder import RecorderWindow

class LaunchpadApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("NADOO Launchpad - WhisperX")
        self.geometry("800x600")
        
        # Check for HF token
        if not self._check_hf_token():
            return
        
        # Initialize components
        try:
            self.processor = AdaptiveProcessor()
            self.current_model = None
            self.current_speaker = None
            
            self._create_menu()
            self._create_widgets()
            self._load_profiles()
        except Exception as e:
            messagebox.showerror(
                "Initialization Error",
                str(e)
            )
            self.quit()

    def _check_hf_token(self) -> bool:
        """Check for Hugging Face token."""
        token = os.getenv("HF_TOKEN")
        if not token:
            # Ask for token
            dialog = HFTokenDialog(self)
            self.wait_window(dialog)
            
            if not dialog.token:
                messagebox.showerror(
                    "Error",
                    "Hugging Face token is required to use this application."
                )
                self.quit()
                return False
                
            # Save token to environment
            os.environ["HF_TOKEN"] = dialog.token
            
        return True

    def _create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio", command=self._open_audio)
        file_menu.add_command(label="Record Audio", command=self._open_recorder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Model Menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Load Model", command=self._load_model)
        model_menu.add_command(label="Clear Cache", command=self._clear_cache)
        
        # Speaker Menu
        speaker_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Speakers", menu=speaker_menu)
        speaker_menu.add_command(
            label="Manage Speakers",
            command=self._open_speaker_management
        )

    def _create_widgets(self):
        """Create the main window widgets."""
        # Top Frame
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Speaker Selection
        ttk.Label(
            top_frame,
            text="Speaker:"
        ).pack(side=tk.LEFT, padx=5)
        
        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(
            top_frame,
            textvariable=self.speaker_var,
            state="readonly"
        )
        self.speaker_combo.pack(side=tk.LEFT, padx=5)
        
        # Adaptation Options
        self.adapt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_frame,
            text="Adapt to Speaker",
            variable=self.adapt_var
        ).pack(side=tk.LEFT, padx=10)
        
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_frame,
            text="Enhance Audio",
            variable=self.enhance_var
        ).pack(side=tk.LEFT, padx=10)
        
        # Main Frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Audio List
        list_frame = ttk.LabelFrame(main_frame, text="Audio Files")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.audio_list = ttk.Treeview(
            list_frame,
            columns=("Path", "Duration"),
            show="headings"
        )
        self.audio_list.heading("Path", text="File")
        self.audio_list.heading("Duration", text="Duration")
        self.audio_list.pack(fill=tk.BOTH, expand=True)
        
        # Buttons Frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Add Files",
            command=self._add_files
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text="Remove",
            command=self._remove_file
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text="Transcribe",
            command=self._transcribe
        ).pack(fill=tk.X, pady=2)

    def _load_profiles(self):
        """Load speaker profiles into combo box."""
        profiles = list(self.processor.voice_profiles.keys())
        self.speaker_combo["values"] = [""] + profiles
        self.speaker_combo.set("")

    def _open_speaker_management(self):
        """Open the speaker management window."""
        window = SpeakerManagementWindow(self)
        self.wait_window(window)
        self._load_profiles()

    def _open_audio(self):
        """Open audio file(s)."""
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a"),
                ("All Files", "*.*")
            ]
        )
        if files:
            self._add_files(files)

    def _add_files(self, files=None):
        """Add audio files to the list."""
        if files is None:
            files = filedialog.askopenfilenames(
                title="Select Audio Files",
                filetypes=[
                    ("Audio Files", "*.mp3 *.wav *.m4a"),
                    ("All Files", "*.*")
                ]
            )
        
        if files:
            for file in files:
                # TODO: Get actual duration
                self.audio_list.insert(
                    "",
                    tk.END,
                    values=(file, "00:00")
                )

    def _remove_file(self):
        """Remove selected file from list."""
        selection = self.audio_list.selection()
        if selection:
            self.audio_list.delete(selection)

    def _load_model(self):
        """Load WhisperX model."""
        # TODO: Add model selection dialog
        try:
            self.current_model = load_model("base")
            messagebox.showinfo(
                "Success",
                "Model loaded successfully"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load model: {str(e)}"
            )

    def _clear_cache(self):
        """Clear model cache."""
        if messagebox.askyesno(
            "Confirm",
            "Clear model cache?"
        ):
            # TODO: Implement cache clearing
            pass

    def _transcribe(self):
        """Transcribe selected audio files."""
        selection = self.audio_list.selection()
        if not selection:
            messagebox.showwarning(
                "Warning",
                "Please select files to transcribe"
            )
            return
            
        if not self.current_model:
            if not messagebox.askyesno(
                "No Model",
                "No model loaded. Load default model?"
            ):
                return
            self._load_model()
            
        speaker_id = self.speaker_var.get()
        adapt_model = self.adapt_var.get()
        enhance_audio = self.enhance_var.get()
        
        try:
            for item in selection:
                audio_path = self.audio_list.item(item)["values"][0]
                
                # Transcribe with adaptation if enabled
                if speaker_id and adapt_model:
                    profile = self.processor.voice_profiles[speaker_id]
                    self.processor.adapt_to_speaker(profile, self.current_model)
                
                result = self.current_model.transcribe(
                    audio_path,
                    speaker_id=speaker_id if speaker_id else None,
                    enhance_audio=enhance_audio
                )
                
                # Save result
                output_path = os.path.splitext(audio_path)[0] + ".txt"
                with open(output_path, "w") as f:
                    f.write(result["text"])
            
            messagebox.showinfo(
                "Success",
                "Transcription complete"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Transcription failed: {str(e)}"
            )

    def _open_recorder(self):
        """Open the audio recorder window."""
        window = RecorderWindow(self, self._on_recording_complete)
        window.protocol("WM_DELETE_WINDOW", window.on_closing)
        
    def _on_recording_complete(self, filename: str):
        """Handle completed recording."""
        self.audio_list.insert(
            "",
            tk.END,
            values=(filename, "New Recording")
        )

class HFTokenDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Hugging Face Token")
        self.geometry("400x200")
        
        self.token = None
        self._create_widgets()
        
    def _create_widgets(self):
        """Create dialog widgets."""
        # Instructions
        ttk.Label(
            self,
            text="Please provide your Hugging Face token.\n"
                 "Visit https://hf.co/settings/tokens to create one.",
            wraplength=350,
            justify="center"
        ).pack(pady=20)
        
        # Token Entry
        token_frame = ttk.Frame(self)
        token_frame.pack(fill=tk.X, padx=20)
        
        ttk.Label(
            token_frame,
            text="Token:"
        ).pack(side=tk.LEFT)
        
        self.token_var = tk.StringVar()
        ttk.Entry(
            token_frame,
            textvariable=self.token_var,
            width=40
        ).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=20)
        
        ttk.Button(
            btn_frame,
            text="OK",
            command=self._save_token
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side=tk.LEFT, padx=5)
        
    def _save_token(self):
        """Save the token and close."""
        token = self.token_var.get().strip()
        if token:
            self.token = token
            self.destroy()
        else:
            messagebox.showwarning(
                "Warning",
                "Please enter a valid token"
            )

if __name__ == "__main__":
    app = LaunchpadApp()
    app.mainloop()
