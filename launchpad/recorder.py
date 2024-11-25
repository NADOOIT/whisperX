"""Audio recorder for NADOO Launchpad."""

import os
import wave
import pyaudio
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.thread = None
        
        # Recording settings
        self.channels = 1
        self.rate = 16000  # Match WhisperX default
        self.chunk = 1024
        self.format = pyaudio.paFloat32
        
    def start_recording(self):
        """Start recording audio."""
        self.frames = []
        self.is_recording = True
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        
    def stop_recording(self) -> str:
        """Stop recording and save the audio file."""
        self.is_recording = False
        if self.thread:
            self.thread.join()
            
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.expanduser(f"~/Downloads/recording_{timestamp}.wav")
        
        # Save recording
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            
        return filename
        
    def _record(self):
        """Record audio in a separate thread."""
        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            
    def __del__(self):
        """Clean up resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()

class RecorderWindow(tk.Toplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Audio Recorder")
        self.geometry("300x150")
        
        self.callback = callback
        self.recorder = AudioRecorder()
        self.recording = False
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the recorder window widgets."""
        # Status Label
        self.status_label = ttk.Label(
            self,
            text="Ready to Record",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=20)
        
        # Record Button
        self.record_btn = ttk.Button(
            self,
            text="Start Recording",
            command=self._toggle_recording
        )
        self.record_btn.pack(pady=10)
        
    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.recording:
            # Start recording
            self.recorder.start_recording()
            self.recording = True
            self.status_label.config(text="Recording...")
            self.record_btn.config(text="Stop Recording")
        else:
            # Stop recording
            filename = self.recorder.stop_recording()
            self.recording = False
            self.status_label.config(text="Recording Saved!")
            self.record_btn.config(text="Start Recording")
            
            # Send filename back to main window
            self.callback(filename)
            
    def on_closing(self):
        """Handle window closing."""
        if self.recording:
            self.recorder.stop_recording()
        self.destroy()
