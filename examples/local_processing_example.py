"""Example of local multi-device processing with WhisperX."""

import logging
import soundfile as sf
import time
from pathlib import Path
from whisperx.local_compute import LocalComputeManager

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize local compute manager
    manager = LocalComputeManager()
    
    # Print discovered devices
    logger.info("Available Local Compute Devices:")
    print("-" * 50)
    for device in manager.devices:
        print(f"Device: {device.name}")
        print(f"  Type: {device.type.value}")
        print(f"  Compute Units: {device.compute_units}")
        print(f"  Supported Operations: {', '.join(device.supported_ops)}")
        print(f"  Batch Size: {device.batch_size}")
        print("-" * 50)
    
    # Process audio file
    audio_file = "path/to/audio.wav"
    try:
        # Load audio
        logger.info(f"Loading audio file: {audio_file}")
        audio, sample_rate = sf.read(audio_file)
        
        # Start processing timer
        start_time = time.time()
        
        # Process audio using all local devices
        logger.info("Starting audio processing...")
        result = manager.process_audio(
            audio,
            chunk_duration=30.0  # 30 seconds per chunk
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Print results
        print("\nTranscription Results:")
        print("=" * 50)
        print(result["text"])
        
        print("\nSegments:")
        print("=" * 50)
        for segment in result["segments"]:
            start = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            end = time.strftime('%H:%M:%S', time.gmtime(segment['end']))
            print(f"[{start} -> {end}] {segment['text']}")
        
        print("\nPerformance Statistics:")
        print("=" * 50)
        print(f"Total Processing Time: {processing_time:.2f} seconds")
        print(f"Audio Duration: {len(audio)/sample_rate:.2f} seconds")
        print(f"Real-time Factor: {(len(audio)/sample_rate)/processing_time:.2f}x")
        
        print("\nDevice Usage:")
        print("=" * 50)
        for device_name, stats in result["device_usage"].items():
            print(f"{device_name}:")
            print(f"  Load: {stats['load']:.1f}%")
            print(f"  Processed Chunks: {stats['processed_chunks']}")
        
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_file}")
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
    finally:
        # Cleanup
        manager.shutdown()

if __name__ == "__main__":
    main()
