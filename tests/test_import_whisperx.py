def test_import_whisperx():
    try:
        import whisperx
    except ImportError as e:
        assert False, f"Importing whisperx failed: {e}"
