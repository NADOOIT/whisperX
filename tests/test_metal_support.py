import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import unittest
from whisperx.asr import load_model

class TestMetalSupport(unittest.TestCase):
    def test_metal_device_available(self):
        """Test to ensure Metal device is available and used."""
        if torch.backends.mps.is_available():
            pipeline = load_model(name="large-v2", device="mps", compute_type="float32")
            self.assertEqual(pipeline.device, "mps", "Metal device should be used when available.")
            self.assertEqual(pipeline.compute_type, "float32", "Compute type should be float32 for Metal.")
        else:
            self.skipTest("Metal device not available on this system.")
