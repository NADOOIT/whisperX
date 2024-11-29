import os
import platform

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="whisperx",
    py_modules=["whisperx"],
    version="3.1.1",
    description="Time-Accurate Automatic Speech Recognition using Whisper.",
    readme="README.md",
    python_requires=">=3.8",
    author="Max Bain",
    url="https://github.com/m-bain/whisperx",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "ctranslate2 @ git+https://github.com/NADOOIT/CTranslate2.git@surfer",
        "faster-whisper==1.1.0",
        "torch>=2",
        "torchaudio>=2",
        "transformers",
        "pandas",
        "setuptools>=65",
        "pyannote.audio==3.1.1"
    ],
    entry_points={
        "console_scripts": ["whisperx=whisperx.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
