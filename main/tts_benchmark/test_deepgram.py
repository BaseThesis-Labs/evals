#!/usr/bin/env python3
"""Quick test of Deepgram TTS."""
import os
import sys

# Check API key
api_key = os.getenv("DEEPGRAM_API_KEY")
if not api_key:
    print("❌ DEEPGRAM_API_KEY not set!")
    print("\nSet it with:")
    print("export DEEPGRAM_API_KEY='your_key_here'")
    sys.exit(1)

print(f"✓ API key found: {api_key[:10]}...")

# Test import
try:
    from deepgram import DeepgramClient
    print("✓ Deepgram SDK imported")
except ImportError as e:
    print(f"❌ Cannot import Deepgram: {e}")
    print("\nInstall with:")
    print("pip install deepgram-sdk==3.2.7")
    sys.exit(1)

# Test generation
print("\n▶ Testing TTS generation...")
try:
    from models.deepgram_client import DeepgramClient as DGClient

    client = DGClient()
    print(f"✓ Client created: {client.name}")

    result = client.generate("Hello, this is a test.", utterance_id="test_001")

    print(f"✓ Generated audio!")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Sample rate: {result.sample_rate}Hz")
    print(f"  Inference time: {result.inference_time_ms:.0f}ms")
    print(f"  Audio shape: {result.audio.shape}")

    # Save test file
    import soundfile as sf
    test_path = "test_deepgram.wav"
    sf.write(test_path, result.audio, result.sample_rate)
    print(f"\n✓ Saved to: {test_path}")
    print("\n✅ Deepgram TTS works perfectly!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
