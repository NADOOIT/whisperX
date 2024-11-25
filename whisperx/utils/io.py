"""I/O utilities for WhisperX."""

import json
import os
import subprocess
from typing import TextIO, Iterator, Tuple

def get_writer(output_format: str, output_dir: str) -> TextIO:
    """Get a writer for the specified output format."""
    os.makedirs(output_dir, exist_ok=True)

    if output_format == "txt":
        return open(os.path.join(output_dir, "transcription.txt"), "w", encoding="utf-8")
    elif output_format == "vtt":
        return open(os.path.join(output_dir, "transcription.vtt"), "w", encoding="utf-8")
    elif output_format == "srt":
        return open(os.path.join(output_dir, "transcription.srt"), "w", encoding="utf-8")
    elif output_format == "tsv":
        return open(os.path.join(output_dir, "transcription.tsv"), "w", encoding="utf-8")
    elif output_format == "json":
        return open(os.path.join(output_dir, "transcription.json"), "w", encoding="utf-8")
    else:
        raise ValueError(f"Unknown output format: {output_format}")

def optional_int(string: str) -> int:
    """Convert string to optional int."""
    return None if string == "None" else int(string)

def optional_float(string: str) -> float:
    """Convert string to optional float."""
    return None if string == "None" else float(string)

def str2bool(string: str) -> bool:
    """Convert string to bool."""
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")
