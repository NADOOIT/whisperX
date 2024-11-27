import whisperx
import sys
import os

# Log the current Python path
print("Python Path:", sys.path)

# Check the installation directory for whisperx
installed_packages = os.popen('pip show whisperx').read()
print("Whisperx Installation Details:\n", installed_packages)


def test_load_model():
    try:
        # Attempt to load a model with whisperx
        model = whisperx.load_model(name="small", device="cpu", compute_type="float32")
        assert model is not None
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")


if __name__ == "__main__":
    test_load_model()
