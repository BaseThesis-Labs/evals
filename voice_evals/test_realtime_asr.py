#!/usr/bin/env python3
"""
Test Real-time ASR Evaluation Metrics
Demonstrates UPWR, streaming analysis, chunk-size trade-offs, and noise robustness
"""

from metrics.realtime_asr import RealtimeASREvaluator, format_realtime_report
import json


def main():
    """Run real-time ASR evaluation on the sample audio"""

    audio_path = "/Users/riapicardo/Desktop/basethesis/work/evals/audio.wav"
    transcript_path = "/Users/riapicardo/Desktop/basethesis/work/evals/transcrpit.txt"

    # Load ground truth
    with open(transcript_path, 'r') as f:
        reference = f.read().strip()

    print("="*70)
    print("REAL-TIME ASR EVALUATION TEST")
    print("="*70)
    print(f"\nAudio: {audio_path}")
    print(f"Ground truth length: {len(reference)} characters")
    print(f"Reference: {reference[:100]}...")

    # Initialize evaluator
    print("\nInitializing Real-time ASR Evaluator...")
    evaluator = RealtimeASREvaluator(model_id="openai/whisper-base")

    # Run full evaluation
    print("\nRunning comprehensive real-time ASR evaluation...")
    print("This will test:")
    print("  1. UPWR (Unstable Partial Word Ratio)")
    print("  2. Streaming vs Batch comparison")
    print("  3. Chunk-size trade-offs")
    print("  4. Noise robustness")
    print()

    metrics = evaluator.evaluate_all(
        audio_path=audio_path,
        reference=reference,
        chunk_duration=1.0,  # 1 second chunks for UPWR
        chunk_sizes=[0.16, 0.32, 0.56, 1.0, 2.0],  # Test various chunk sizes
        noise_levels=[0.01, 0.05, 0.1]  # Test noise robustness
    )

    # Print formatted report
    report = format_realtime_report(metrics)
    print(report)

    # Save results
    output_path = "realtime_asr_evaluation.json"
    with open(output_path, 'w') as f:
        from dataclasses import asdict
        json.dump(asdict(metrics), f, indent=2)

    print(f"\n💾 Full results saved to: {output_path}")

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print(f"\n📊 UPWR: {metrics.upwr_metrics.upwr:.2%} of words change during streaming")
    if metrics.upwr_metrics.upwr < 0.15:
        print("   ✅ Excellent stability!")
    elif metrics.upwr_metrics.upwr < 0.30:
        print("   ✓ Good stability")
    else:
        print("   ⚠️  High instability - consider larger chunks")

    print(f"\n🔄 Streaming degrades WER by {metrics.streaming_degradation:+.1f} percentage points")
    if abs(metrics.streaming_degradation) < 10:
        print("   ✅ Better than typical (10-17pp degradation)")
    elif abs(metrics.streaming_degradation) < 17:
        print("   ✓ Within typical range")
    else:
        print("   ⚠️  Higher than typical degradation")

    print(f"\n⚖️  Optimal chunk size: {metrics.optimal_chunk_size}s")
    print(f"   Achieves {min(metrics.chunk_analyses, key=lambda x: x.wer).wer:.2%} WER")

    print(f"\n🔊 Noise increases WER by {metrics.noise_robustness.degradation_factor:.1f}x")
    if metrics.noise_robustness.degradation_factor < 1.5:
        print("   ✅ Robust to noise!")
    elif metrics.noise_robustness.degradation_factor < 2.0:
        print("   ✓ Moderate noise robustness")
    else:
        print("   ⚠️  Sensitive to noise")

    print("\n" + "="*70)
    print("✅ Real-time ASR evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
