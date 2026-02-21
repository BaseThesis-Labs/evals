#!/usr/bin/env python3
"""Generate TTS audio for all models."""
import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm
import soundfile as sf
import librosa
import time
import random as _random


def load_model_config(config_path: Path) -> Dict:
    """Load model configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load utterance manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def get_client(model_config: Dict):
    """Instantiate TTS client from config."""
    class_path = model_config['class_path']
    module_name, class_name = class_path.rsplit('.', 1)

    module = importlib.import_module(module_name)
    client_class = getattr(module, class_name)

    config = model_config.get('config', {})
    return client_class(**config)


def resample_audio(audio, orig_sr, target_sr=24000):
    """Resample audio to target sample rate."""
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio, target_sr


def _call_with_retry(client, entry: Dict, max_retries: int = 4) -> object:
    """Call client.generate() with exponential backoff on 429 / rate-limit errors."""
    utt_id = entry['id']
    delay = 2.0  # initial backoff in seconds

    for attempt in range(max_retries + 1):
        try:
            return client.generate(
                text=entry['text'],
                reference_audio_path=entry.get('reference_audio_path'),
                speaker_id=entry.get('speaker_id'),
                utterance_id=utt_id,
            )
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                '429' in str(e) or
                'too many requests' in err_str or
                'rate limit' in err_str or
                'ratelimit' in err_str
            )

            if is_rate_limit and attempt < max_retries:
                # Exponential backoff with jitter: 2s, 4s, 8s, 16s
                jitter = _random.uniform(0, delay * 0.25)
                wait = delay + jitter
                print(f"\n  ⏳ Rate limited on {utt_id} "
                      f"(attempt {attempt + 1}/{max_retries}). "
                      f"Waiting {wait:.1f}s...")
                time.sleep(wait)
                delay *= 2  # double for next retry
            else:
                raise  # non-429 error or out of retries


def generate_for_model(
    model_name: str,
    model_config: Dict,
    manifest: List[Dict],
    output_dir: Path,
    skip_existing: bool = True,
):
    """Generate audio for one model."""
    print(f"\n▶ Generating audio for {model_name}")

    # Create output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    try:
        client = get_client(model_config)
    except Exception as e:
        print(f"✗ Failed to initialize {model_name}: {e}")
        return

    # Metadata for timing info
    metadata_path = model_output_dir / "gen_meta.jsonl"
    metadata_file = open(metadata_path, 'a')

    # Rate limiting — minimum gap between API calls
    rate_limit = model_config.get('rate_limit_rps', 0)
    min_interval = 1.0 / rate_limit if rate_limit > 0 else 0
    last_request_time = 0.0  # tracks when we last hit the API

    # Generate for each utterance
    for entry in tqdm(manifest, desc=f"{model_name}"):
        utt_id = entry['id']
        output_path = model_output_dir / f"{utt_id}.wav"

        # Skip if exists — but still enforce rate limit so a burst of skips
        # followed by new requests doesn't immediately 429
        if skip_existing and output_path.exists():
            continue

        # Enforce minimum interval since last API call
        if min_interval > 0:
            elapsed = time.time() - last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        try:
            result = _call_with_retry(client, entry)
            last_request_time = time.time()

            # Resample to 24kHz if needed
            audio, sr = resample_audio(
                result.audio,
                result.sample_rate,
                target_sr=24000,
            )

            # Save audio
            sf.write(output_path, audio, sr, subtype='PCM_16')

            # Save metadata
            meta = {
                'id': utt_id,
                'inference_time_ms': result.inference_time_ms,
                'ttfa_ms': result.ttfa_ms,
                'duration_s': result.duration_seconds,
                'rtf': result.inference_time_ms / 1000 / result.duration_seconds,
                'error': None,
            }
            metadata_file.write(json.dumps(meta) + '\n')
            metadata_file.flush()

        except Exception as e:
            last_request_time = time.time()
            print(f"✗ Error generating {utt_id}: {e}")
            meta = {
                'id': utt_id,
                'inference_time_ms': None,
                'ttfa_ms': None,
                'duration_s': None,
                'rtf': None,
                'error': str(e),
            }
            metadata_file.write(json.dumps(meta) + '\n')
            metadata_file.flush()

    metadata_file.close()

    # Cleanup
    try:
        client.cleanup()
    except:
        pass

    print(f"✓ Completed {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Generate TTS audio')
    parser.add_argument('--config', type=str, default='configs/models.yaml',
                       help='Model config file')
    parser.add_argument('--manifest', type=str, default='datasets/manifest.json',
                       help='Utterance manifest')
    parser.add_argument('--output', type=str, default='generated_audio',
                       help='Output directory')
    parser.add_argument('--model', type=str, default=None,
                       help='Generate for specific model only')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip existing audio files')
    args = parser.parse_args()

    # Load config and manifest
    config = load_model_config(Path(args.config))
    manifest = load_manifest(Path(args.manifest))

    print(f"✓ Loaded {len(manifest)} utterances")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each enabled model
    models = config['models']

    if args.model:
        # Single model
        if args.model not in models:
            print(f"✗ Model {args.model} not found in config")
            return

        if not models[args.model].get('enabled', True):
            print(f"✗ Model {args.model} is disabled")
            return

        generate_for_model(
            args.model,
            models[args.model],
            manifest,
            output_dir,
            args.skip_existing
        )
    else:
        # All enabled models
        for model_name, model_config in models.items():
            if not model_config.get('enabled', True):
                print(f"⊘ Skipping disabled model: {model_name}")
                continue

            generate_for_model(
                model_name,
                model_config,
                manifest,
                output_dir,
                args.skip_existing
            )

    print("\n✓ Generation complete!")
    print(f"✓ Output directory: {output_dir}")


if __name__ == '__main__':
    main()
