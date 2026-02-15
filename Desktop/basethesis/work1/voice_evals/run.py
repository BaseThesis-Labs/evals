#!/usr/bin/env python3
"""
Main entry point for voice evaluation with diarization
Quick and easy command-line interface
"""

from core.pipeline import VoiceEvaluationPipeline
import sys
from pathlib import Path


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║        Voice Evaluation Pipeline with Speaker Diarization        ║
╚══════════════════════════════════════════════════════════════════╝

Usage: python run.py <audio_file> [options]

Required:
  audio_file            Path to audio file (WAV, MP3, etc.)

Optional:
  transcript_file       Ground truth transcript for accuracy metrics
  --hf-token TOKEN     HuggingFace token for speaker diarization
  --num-speakers N     Number of speakers (default: auto-detect)
  --no-diarization     Skip speaker diarization
  --per-speaker        Run detailed evaluation per speaker (slower)

Environment Variables:
  HF_TOKEN             HuggingFace token (alternative to --hf-token)

Examples:
  # Basic evaluation with diarization
  python run.py audio.wav

  # With ground truth transcript
  python run.py audio.wav transcript.txt

  # With HuggingFace token for diarization
  python run.py audio.wav --hf-token hf_xxxxx

  # Fixed number of speakers
  python run.py audio.wav --num-speakers 3

  # Without diarization (faster, no speaker separation)
  python run.py audio.wav --no-diarization

  # Full per-speaker analysis
  python run.py audio.wav --per-speaker --hf-token hf_xxxxx

Get HuggingFace Token:
  1. Create account: https://huggingface.co
  2. Accept model terms: https://huggingface.co/pyannote/speaker-diarization
  3. Generate token: https://huggingface.co/settings/tokens
  4. Set environment: export HF_TOKEN="your-token-here"
        """)
        sys.exit(0)

    audio_path = sys.argv[1]
    transcript_path = None
    hf_token = None
    num_speakers = None
    enable_diarization = True
    per_speaker = False

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == '--hf-token' and i + 1 < len(sys.argv):
            hf_token = sys.argv[i + 1]
            i += 2
        elif arg == '--num-speakers' and i + 1 < len(sys.argv):
            num_speakers = int(sys.argv[i + 1])
            i += 2
        elif arg == '--no-diarization':
            enable_diarization = False
            i += 1
        elif arg == '--per-speaker':
            per_speaker = True
            i += 1
        elif not arg.startswith('--') and transcript_path is None:
            if Path(arg).exists():
                transcript_path = arg
            i += 1
        else:
            i += 1

    # Validate audio file
    if not Path(audio_path).exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Create pipeline
    try:
        pipeline = VoiceEvaluationPipeline(
            audio_path=audio_path,
            transcript_path=transcript_path,
            hf_token=hf_token,
            enable_diarization=enable_diarization,
            num_speakers=num_speakers,
            per_speaker_evaluation=per_speaker
        )

        # Run evaluation
        metrics = pipeline.evaluate()

        # Generate report
        report = pipeline.generate_report(metrics)
        print(report)

        # Save results
        pipeline.save_results(metrics)

        print("\n✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
