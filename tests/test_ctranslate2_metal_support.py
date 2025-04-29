import unittest
import ctranslate2

class TestCTranslate2MetalSupport(unittest.TestCase):
    def test_metal_support(self):
        """Test to check if ctranslate2 supports Metal (MPS)."""
        if not hasattr(ctranslate2, "get_supported_devices"):
            import unittest
            self.skipTest("ctranslate2.get_supported_devices does not exist; skipping Metal support test.")
        supported_devices = ctranslate2.get_supported_devices()
        self.assertIn("mps", supported_devices, "CTranslate2 should support Metal (MPS) on this system.")
