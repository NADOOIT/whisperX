"""Example of distributed processing with WhisperX."""

import os
import torch
import numpy as np
from pathlib import Path
from whisperx.distributed import (
    NodeConfig,
    DistributedNode,
    DistributedCoordinator,
    discover_nodes
)

def setup_mac_node():
    """Setup Mac as primary node."""
    config = NodeConfig(
        node_id="mac_primary",
        host="localhost",
        port=5555,
        device="mps",
        compute_type="float16",
        model_size="base",
        priority=2  # Higher priority
    )
    
    node = DistributedNode(config)
    node.start()
    return node

def setup_laptop_node(laptop_ip: str):
    """Setup laptop as secondary node."""
    config = NodeConfig(
        node_id="laptop_secondary",
        host=laptop_ip,
        port=5556,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16",
        model_size="base",
        priority=1  # Lower priority
    )
    
    node = DistributedNode(config)
    node.start()
    return node

def main():
    # Initialize coordinator
    coordinator = DistributedCoordinator()
    
    # Setup Mac node
    mac_node = setup_mac_node()
    coordinator.add_node(mac_node.config)
    
    # Discover and setup laptop node
    laptop_nodes = discover_nodes()
    for node_config in laptop_nodes:
        coordinator.add_node(node_config)
    
    # Example: Process a long audio file
    audio_file = "path/to/long_audio.wav"
    import soundfile as sf
    audio, _ = sf.read(audio_file)
    
    # Process audio using distributed nodes
    result = coordinator.process_audio(
        audio,
        priority=1,
        diarize=True
    )
    
    # Print results
    print("Transcription:", result["text"])
    print("\nSegments:")
    for segment in result["segments"]:
        print(f"{segment['start']:.2f}s -> {segment['end']:.2f}s: {segment['text']}")
        if "speaker" in segment:
            print(f"Speaker: {segment['speaker']}")
    
    # Cleanup
    mac_node.stop()
    for node_config in laptop_nodes:
        coordinator.remove_node(node_config.node_id)

if __name__ == "__main__":
    main()
