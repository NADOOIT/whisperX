import unittest
import sys
import importlib.util
import platform
from pathlib import Path

class TestInstallation(unittest.TestCase):
    def test_ctranslate2_installation(self):
        """Test that CTranslate2 is properly installed from requirements.txt"""
        try:
            import ctranslate2
            self.assertIsNotNone(ctranslate2.__file__, "CTranslate2 module exists but has no file location")
            
            # Verify it's the correct version
            self.assertTrue(hasattr(ctranslate2, '__version__'), "CTranslate2 has no version attribute")
            
            # Check if the C++ library is properly installed
            lib_path = Path(sys.prefix) / "lib" / "libctranslate2.dylib"
            self.assertTrue(lib_path.exists(), f"CTranslate2 C++ library not found at {lib_path}")
            
        except ImportError as e:
            self.fail(f"Failed to import ctranslate2: {str(e)}")
    
    def test_whisperx_imports(self):
        """Test that WhisperX can import all necessary dependencies"""
        required_modules = [
            'whisperx',
            'torch',
            'torchaudio',
            'faster_whisper',
            'transformers',
            'pandas',
            'nltk',
            'cmake',  # Required for building
            'pybind11'  # Required for building
        ]
        
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                self.fail(f"Failed to import {module}: {str(e)}")
    
    def test_model_loading(self):
        """Test that WhisperX can load a model"""
        try:
            import whisperx
            model = whisperx.load_model('tiny', device='mps')
            self.assertIsNotNone(model, "Model failed to load")
        except Exception as e:
            self.fail(f"Failed to load WhisperX model: {str(e)}")
    
    def test_system_requirements(self):
        """Test that system requirements are met"""
        # Check OS
        self.assertEqual(platform.system(), "Darwin", "This package only supports macOS")
        
        # Check for required system libraries
        libomp_path = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
        self.assertTrue(libomp_path.exists(), "OpenMP library not found. Please install with: brew install libomp")

if __name__ == '__main__':
    unittest.main()
