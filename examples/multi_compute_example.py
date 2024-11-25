"""Example of using multiple compute devices simultaneously."""

import numpy as np
import soundfile as sf
import logging
from whisperx.compute_scheduler import MultiComputeScheduler

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize multi-compute scheduler
    scheduler = MultiComputeScheduler()
    
    # Log discovered devices
    logger.info("Discovered compute devices:")
    for device in scheduler.devices:
        logger.info(f"  - {device.name} ({device.type.value})")
        logger.info(f"    Compute Units: {device.compute_units}")
        logger.info(f"    Supported Precisions: {', '.join(device.supported_precisions)}")
    
    # Load audio file
    audio_file = "path/to/audio.wav"
    try:
        audio, sample_rate = sf.read(audio_file)
        logger.info(f"Loaded audio file: {len(audio)} samples, {sample_rate}Hz")
        
        # Process audio using all available devices
        results = scheduler.process_audio(
            audio,
            chunk_size=30  # 30 second chunks
        )
        
        # Print results
        print("\nTranscription Results:")
        print("-" * 50)
        
        for result in results:
            device = result.get("device", "unknown")
            text = result.get("text", "")
            print(f"[{device.upper()}] {text}")
        
        # Print device usage statistics
        print("\nDevice Usage Statistics:")
        print("-" * 50)
        for device in scheduler.devices:
            print(f"{device.name}: {device.current_load:.1f}% load")
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
    finally:
        # Cleanup
        scheduler.shutdown()

if __name__ == "__main__":
    main()
