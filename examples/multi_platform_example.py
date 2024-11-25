"""Example of cross-platform distributed processing with WhisperX."""

import time
import logging
from pathlib import Path
from whisperx.auto_discovery import WhisperXDiscovery
from whisperx.distributed import DistributedCoordinator

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize discovery service
    discovery = WhisperXDiscovery()
    
    # Start advertising this node
    discovery.start_advertising()
    logger.info("Started advertising local node")
    
    # Start discovering other nodes
    discovery.start_discovery()
    logger.info("Started node discovery")
    
    # Wait for nodes to be discovered (in practice, you might want to implement a better waiting strategy)
    time.sleep(5)
    
    # Get available nodes
    available_nodes = discovery.get_available_nodes()
    logger.info(f"Discovered {len(available_nodes)} nodes:")
    for node in available_nodes:
        logger.info(f"  - {node.node_id} ({node.capabilities.platform}, {node.capabilities.device_type})")
    
    # Initialize coordinator with discovered nodes
    coordinator = DistributedCoordinator()
    for node in available_nodes:
        coordinator.add_node(node)
    
    # Example: Process audio file
    audio_file = Path("path/to/audio.wav")
    if audio_file.exists():
        # Process audio using all available nodes
        result = coordinator.process_audio(
            audio_file,
            diarize=True,
            language="en"
        )
        
        # Print results
        print("\nTranscription Results:")
        print("-" * 50)
        print(result["text"])
        
        print("\nSegments:")
        print("-" * 50)
        for segment in result["segments"]:
            start = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            end = time.strftime('%H:%M:%S', time.gmtime(segment['end']))
            print(f"[{start} -> {end}] {segment['text']}")
            if "speaker" in segment:
                print(f"Speaker: {segment['speaker']}")
    else:
        logger.error(f"Audio file not found: {audio_file}")
    
    # Cleanup
    discovery.stop()
    coordinator.shutdown()

if __name__ == "__main__":
    main()
