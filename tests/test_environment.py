import sys
import os

# Add the whisperX directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Log the current Python path
print("Python Path:", sys.path)

# Check if whisperx module is available
try:
    import whisperx
    print("whisperx module is available")
except ModuleNotFoundError:
    print("whisperx module is NOT available")

# Check if ctranslate2 module is available
try:
    import ctranslate2
    print("ctranslate2 module is available")
except ModuleNotFoundError:
    print("ctranslate2 module is NOT available")
