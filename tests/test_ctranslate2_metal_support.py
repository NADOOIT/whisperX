import unittest
import ctranslate2

class TestCTranslate2MetalSupport(unittest.TestCase):
    def test_metal_support(self):
        """Test to check if ctranslate2 supports Metal (MPS)."""
        supported_devices = ctranslate2.get_supported_devices()
        self.assertIn("mps", supported_devices, "CTranslate2 should support Metal (MPS) on this system.")
