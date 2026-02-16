"""
Real-time ASR Evaluation Metrics
Includes UPWR, streaming analysis, chunk-size trade-offs, and noise robustness
"""

import numpy as np
import librosa
import torch
import torchaudio
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class StreamingSegment:
    """Represents a streaming transcription segment"""
    chunk_id: int
    start_time: float
    end_time: float
    partial_text: str
    final_text: str
    changed_words: List[str]
    stable_words: List[str]
    latency_ms: float


@dataclass
class UPWRMetrics:
    """Unstable Partial Word Ratio metrics"""
    upwr: float  # Overall UPWR score
    total_partial_words: int
    changed_words: int
    stable_words: int
    word_stability_ratio: float
    chunks_analyzed: int
    avg_words_per_chunk: float


@dataclass
class ChunkAnalysis:
    """Chunk-size trade-off analysis"""
    chunk_duration: float
    wer: float
    latency_ms: float
    throughput: float
    num_chunks: int


@dataclass
class NoiseRobustnessMetrics:
    """Noise robustness evaluation metrics"""
    clean_wer: float
    noisy_wer: float
    normalized_wer: float  # (noisy_wer - clean_wer) / clean_wer
    wer_variance: float  # Stability across different noise levels
    degradation_factor: float  # How much performance degrades
    snr_correlation: float  # Correlation between SNR and WER


@dataclass
class RealtimeASRMetrics:
    """Complete real-time ASR evaluation metrics"""
    upwr_metrics: UPWRMetrics
    streaming_wer: float
    batch_wer: float
    streaming_degradation: float  # percentage points difference
    chunk_analyses: List[ChunkAnalysis]
    optimal_chunk_size: float
    noise_robustness: NoiseRobustnessMetrics
    total_latency_ms: float
    avg_chunk_latency_ms: float


class RealtimeASREvaluator:
    """
    Real-time ASR evaluation with streaming metrics

    Evaluates:
    - UPWR: Unstable Partial Word Ratio
    - Streaming vs Batch comparison
    - Chunk-size trade-offs
    - Noise robustness
    """

    def __init__(self, model_id: str = "openai/whisper-base"):
        """
        Initialize real-time ASR evaluator

        Args:
            model_id: HuggingFace model ID for ASR
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_models()

    def _init_models(self):
        """Initialize ASR models"""
        print(f"Loading ASR model: {self.model_id}...")

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id).to(self.device)
            processor = AutoProcessor.from_pretrained(self.model_id)

            # Batch pipeline
            self.batch_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=self.device,
                chunk_length_s=30,
                return_timestamps=True
            )

            # Streaming pipeline with smaller chunks
            self.streaming_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=self.device,
                chunk_length_s=5,  # Smaller chunks for streaming
                return_timestamps=True
            )

            print(f"✓ ASR models loaded on {self.device}")

        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def calculate_upwr(self,
                      audio_path: str,
                      chunk_duration: float = 1.0,
                      overlap: float = 0.5) -> UPWRMetrics:
        """
        Calculate Unstable Partial Word Ratio (UPWR)

        Simulates streaming ASR by processing overlapping chunks and tracking
        how words change as more context becomes available.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0-1)

        Returns:
            UPWRMetrics with UPWR score and details
        """
        print(f"\n{'='*70}")
        print("CALCULATING UPWR (Unstable Partial Word Ratio)")
        print(f"{'='*70}")
        print(f"Chunk duration: {chunk_duration}s, Overlap: {overlap*100}%")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - overlap))

        # Track word changes across chunks
        segments: List[StreamingSegment] = []
        previous_text = ""
        total_changed_words = 0
        total_partial_words = 0

        chunk_id = 0
        position = 0

        print(f"\nProcessing {int(duration/chunk_duration/(1-overlap))} chunks...")

        while position < len(audio):
            # Extract chunk
            chunk = audio[position:position + chunk_samples]

            if len(chunk) < sr * 0.1:  # Skip very short chunks
                break

            start_time = position / sr
            end_time = (position + len(chunk)) / sr

            # Transcribe chunk
            try:
                t0 = time.time()
                result = self.streaming_pipeline(chunk)
                latency = (time.time() - t0) * 1000

                current_text = result['text'].strip() if isinstance(result, dict) else str(result).strip()

                # Compare with previous transcription
                prev_words = set(previous_text.lower().split())
                curr_words = set(current_text.lower().split())

                # Words that changed from previous chunk
                changed = prev_words.symmetric_difference(curr_words)
                stable = prev_words.intersection(curr_words)

                total_changed_words += len(changed)
                total_partial_words += len(curr_words)

                segments.append(StreamingSegment(
                    chunk_id=chunk_id,
                    start_time=start_time,
                    end_time=end_time,
                    partial_text=current_text,
                    final_text="",  # Will be filled after final pass
                    changed_words=list(changed),
                    stable_words=list(stable),
                    latency_ms=latency
                ))

                previous_text = current_text

                if (chunk_id + 1) % 10 == 0:
                    print(f"  Processed {chunk_id + 1} chunks...")

            except Exception as e:
                print(f"  Warning: Error processing chunk {chunk_id}: {e}")

            position += hop_samples
            chunk_id += 1

        # Calculate UPWR
        upwr = total_changed_words / total_partial_words if total_partial_words > 0 else 0.0
        word_stability_ratio = 1.0 - upwr

        metrics = UPWRMetrics(
            upwr=upwr,
            total_partial_words=total_partial_words,
            changed_words=total_changed_words,
            stable_words=total_partial_words - total_changed_words,
            word_stability_ratio=word_stability_ratio,
            chunks_analyzed=len(segments),
            avg_words_per_chunk=total_partial_words / len(segments) if segments else 0
        )

        print(f"\n✓ UPWR Analysis Complete:")
        print(f"  UPWR: {upwr:.4f} ({upwr*100:.2f}% words changed)")
        print(f"  Stability: {word_stability_ratio:.4f} ({word_stability_ratio*100:.2f}% stable)")
        print(f"  Chunks analyzed: {len(segments)}")
        print(f"  Avg words/chunk: {metrics.avg_words_per_chunk:.1f}")

        return metrics

    def compare_streaming_vs_batch(self,
                                   audio_path: str,
                                   reference: str) -> Tuple[float, float, float]:
        """
        Compare streaming vs batch transcription

        Args:
            audio_path: Path to audio file
            reference: Ground truth transcript

        Returns:
            (streaming_wer, batch_wer, degradation_percentage_points)
        """
        print(f"\n{'='*70}")
        print("STREAMING vs BATCH COMPARISON")
        print(f"{'='*70}")

        # Batch transcription
        print("Running batch transcription...")
        t0 = time.time()
        batch_result = self.batch_pipeline(audio_path)
        batch_time = time.time() - t0
        batch_text = batch_result['text'] if isinstance(batch_result, dict) else str(batch_result)

        # Streaming transcription
        print("Running streaming transcription...")
        t0 = time.time()
        streaming_result = self.streaming_pipeline(audio_path)
        streaming_time = time.time() - t0
        streaming_text = streaming_result['text'] if isinstance(streaming_result, dict) else str(streaming_result)

        # Calculate WER for both
        # from ..metrics.enhanced_metrics import AlignmentCounts
        from .enhanced_metrics import AlignmentCounts

        def calculate_wer(ref: str, hyp: str) -> float:
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            if not ref_words:
                return 0.0 if not hyp_words else 1.0

            # Simple Levenshtein for WER
            len_r, len_h = len(ref_words), len(hyp_words)
            d = np.zeros((len_r + 1, len_h + 1), dtype=np.int32)
            for i in range(len_r + 1): d[i][0] = i
            for j in range(len_h + 1): d[0][j] = j

            for i in range(1, len_r + 1):
                for j in range(1, len_h + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = 1 + min(d[i-1][j-1], d[i][j-1], d[i-1][j])

            return d[len_r][len_h] / len_r

        streaming_wer = calculate_wer(reference, streaming_text)
        batch_wer = calculate_wer(reference, batch_text)
        degradation = (streaming_wer - batch_wer) * 100  # percentage points

        print(f"\n✓ Comparison Complete:")
        print(f"  Batch WER: {batch_wer:.4f} ({batch_wer*100:.2f}%)")
        print(f"  Streaming WER: {streaming_wer:.4f} ({streaming_wer*100:.2f}%)")
        print(f"  Degradation: {degradation:+.2f} percentage points")
        print(f"  Batch time: {batch_time:.2f}s")
        print(f"  Streaming time: {streaming_time:.2f}s")

        return streaming_wer, batch_wer, degradation

    def analyze_chunk_size_tradeoff(self,
                                    audio_path: str,
                                    reference: str,
                                    chunk_sizes: List[float] = [0.16, 0.32, 0.56, 1.0, 2.0]
                                    ) -> List[ChunkAnalysis]:
        """
        Analyze WER vs latency trade-off for different chunk sizes

        Args:
            audio_path: Path to audio file
            reference: Ground truth transcript
            chunk_sizes: List of chunk durations to test

        Returns:
            List of ChunkAnalysis results
        """
        print(f"\n{'='*70}")
        print("CHUNK-SIZE TRADE-OFF ANALYSIS")
        print(f"{'='*70}")
        print(f"Testing chunk sizes: {chunk_sizes}")

        results = []

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)

        for chunk_duration in chunk_sizes:
            print(f"\nTesting chunk size: {chunk_duration}s")

            # Create pipeline for this chunk size
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id).to(self.device)
                processor = AutoProcessor.from_pretrained(self.model_id)

                pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=self.device,
                    chunk_length_s=chunk_duration,
                    return_timestamps=True
                )

                # Measure latency and transcribe
                t0 = time.time()
                result = pipe(audio_path)
                total_time = time.time() - t0

                text = result['text'] if isinstance(result, dict) else str(result)

                # Calculate WER
                def calculate_wer(ref: str, hyp: str) -> float:
                    ref_words = ref.lower().split()
                    hyp_words = hyp.lower().split()
                    if not ref_words:
                        return 0.0

                    len_r, len_h = len(ref_words), len(hyp_words)
                    d = np.zeros((len_r + 1, len_h + 1), dtype=np.int32)
                    for i in range(len_r + 1): d[i][0] = i
                    for j in range(len_h + 1): d[0][j] = j

                    for i in range(1, len_r + 1):
                        for j in range(1, len_h + 1):
                            if ref_words[i-1] == hyp_words[j-1]:
                                d[i][j] = d[i-1][j-1]
                            else:
                                d[i][j] = 1 + min(d[i-1][j-1], d[i][j-1], d[i-1][j])

                    return d[len_r][len_h] / len_r

                wer = calculate_wer(reference, text)

                # Calculate metrics
                num_chunks = int(np.ceil(duration / chunk_duration))
                avg_latency = (total_time / num_chunks) * 1000  # ms
                throughput = duration / total_time

                analysis = ChunkAnalysis(
                    chunk_duration=chunk_duration,
                    wer=wer,
                    latency_ms=avg_latency,
                    throughput=throughput,
                    num_chunks=num_chunks
                )

                results.append(analysis)

                print(f"  WER: {wer:.4f} ({wer*100:.2f}%)")
                print(f"  Avg latency: {avg_latency:.2f}ms")
                print(f"  Throughput: {throughput:.2f}x")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        print(f"\n✓ Chunk-size analysis complete")
        print(f"\nOptimal chunk size (lowest WER): {min(results, key=lambda x: x.wer).chunk_duration}s")

        return results

    def evaluate_noise_robustness(self,
                                  audio_path: str,
                                  reference: str,
                                  noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
                                  ) -> NoiseRobustnessMetrics:
        """
        Evaluate noise robustness using multiple noise levels

        Args:
            audio_path: Path to clean audio
            reference: Ground truth transcript
            noise_levels: List of noise amplitudes to test

        Returns:
            NoiseRobustnessMetrics
        """
        print(f"\n{'='*70}")
        print("NOISE ROBUSTNESS EVALUATION")
        print(f"{'='*70}")
        print(f"Testing noise levels: {noise_levels}")

        # Load clean audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Transcribe clean audio
        print("\nTranscribing clean audio...")
        clean_result = self.batch_pipeline(audio_path)
        clean_text = clean_result['text'] if isinstance(clean_result, dict) else str(clean_result)

        def calculate_wer(ref: str, hyp: str) -> float:
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            if not ref_words:
                return 0.0

            len_r, len_h = len(ref_words), len(hyp_words)
            d = np.zeros((len_r + 1, len_h + 1), dtype=np.int32)
            for i in range(len_r + 1): d[i][0] = i
            for j in range(len_h + 1): d[0][j] = j

            for i in range(1, len_r + 1):
                for j in range(1, len_h + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = 1 + min(d[i-1][j-1], d[i][j-1], d[i-1][j])

            return d[len_r][len_h] / len_r

        clean_wer = calculate_wer(reference, clean_text)
        print(f"  Clean WER: {clean_wer:.4f} ({clean_wer*100:.2f}%)")

        # Test with different noise levels
        noisy_wers = []
        snrs = []

        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, len(audio))
            noisy_audio = audio + noise

            # Calculate SNR
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
            snrs.append(snr)

            # Transcribe noisy audio
            try:
                noisy_result = self.batch_pipeline(noisy_audio)
                noisy_text = noisy_result['text'] if isinstance(noisy_result, dict) else str(noisy_result)
                noisy_wer = calculate_wer(reference, noisy_text)
                noisy_wers.append(noisy_wer)

                print(f"\n  Noise level {noise_level:.3f} (SNR: {snr:.2f}dB):")
                print(f"    WER: {noisy_wer:.4f} ({noisy_wer*100:.2f}%)")
                print(f"    Degradation: {(noisy_wer - clean_wer)*100:+.2f} pp")

            except Exception as e:
                print(f"  Error at noise level {noise_level}: {e}")
                continue

        # Calculate metrics
        avg_noisy_wer = np.mean(noisy_wers) if noisy_wers else clean_wer
        normalized_wer = (avg_noisy_wer - clean_wer) / clean_wer if clean_wer > 0 else 0.0
        wer_variance = np.var(noisy_wers) if len(noisy_wers) > 1 else 0.0
        degradation_factor = avg_noisy_wer / clean_wer if clean_wer > 0 else 1.0

        # SNR-WER correlation
        if len(snrs) > 1 and len(noisy_wers) > 1:
            snr_correlation = np.corrcoef(snrs, noisy_wers)[0, 1]
        else:
            snr_correlation = 0.0

        metrics = NoiseRobustnessMetrics(
            clean_wer=clean_wer,
            noisy_wer=avg_noisy_wer,
            normalized_wer=normalized_wer,
            wer_variance=wer_variance,
            degradation_factor=degradation_factor,
            snr_correlation=snr_correlation
        )

        print(f"\n✓ Noise Robustness Analysis Complete:")
        print(f"  Normalized WER: {normalized_wer:.4f}")
        print(f"  WER Variance: {wer_variance:.6f}")
        print(f"  Degradation Factor: {degradation_factor:.2f}x")
        print(f"  SNR-WER Correlation: {snr_correlation:.4f}")

        return metrics

    def evaluate_all(self,
                    audio_path: str,
                    reference: str,
                    chunk_duration: float = 1.0,
                    chunk_sizes: List[float] = [0.16, 0.32, 0.56, 1.0],
                    noise_levels: List[float] = [0.01, 0.05, 0.1]
                    ) -> RealtimeASRMetrics:
        """
        Run complete real-time ASR evaluation

        Args:
            audio_path: Path to audio file
            reference: Ground truth transcript
            chunk_duration: Duration for UPWR analysis
            chunk_sizes: Chunk sizes for trade-off analysis
            noise_levels: Noise levels for robustness testing

        Returns:
            RealtimeASRMetrics with all results
        """
        print(f"\n{'='*70}")
        print("REAL-TIME ASR EVALUATION")
        print(f"{'='*70}\n")

        # 1. UPWR
        upwr_metrics = self.calculate_upwr(audio_path, chunk_duration)

        # 2. Streaming vs Batch
        streaming_wer, batch_wer, degradation = self.compare_streaming_vs_batch(audio_path, reference)

        # 3. Chunk-size trade-off
        chunk_analyses = self.analyze_chunk_size_tradeoff(audio_path, reference, chunk_sizes)
        optimal_chunk = min(chunk_analyses, key=lambda x: x.wer).chunk_duration if chunk_analyses else 1.0

        # 4. Noise robustness
        noise_robustness = self.evaluate_noise_robustness(audio_path, reference, noise_levels)

        # Calculate overall latency metrics
        total_latency = sum(a.latency_ms for a in chunk_analyses) / len(chunk_analyses) if chunk_analyses else 0

        metrics = RealtimeASRMetrics(
            upwr_metrics=upwr_metrics,
            streaming_wer=streaming_wer,
            batch_wer=batch_wer,
            streaming_degradation=degradation,
            chunk_analyses=chunk_analyses,
            optimal_chunk_size=optimal_chunk,
            noise_robustness=noise_robustness,
            total_latency_ms=total_latency,
            avg_chunk_latency_ms=total_latency
        )

        print(f"\n{'='*70}")
        print("✓ REAL-TIME ASR EVALUATION COMPLETE")
        print(f"{'='*70}\n")

        return metrics


def format_realtime_report(metrics: RealtimeASRMetrics) -> str:
    """Format real-time ASR metrics into readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           REAL-TIME ASR EVALUATION REPORT                         ║
╚══════════════════════════════════════════════════════════════════╝

📊 UPWR (Unstable Partial Word Ratio)
──────────────────────────────────────────────────────────────────
  UPWR Score:               {metrics.upwr_metrics.upwr:.4f} ({metrics.upwr_metrics.upwr*100:.2f}%)
  Word Stability:           {metrics.upwr_metrics.word_stability_ratio:.4f} ({metrics.upwr_metrics.word_stability_ratio*100:.2f}%)
  Changed Words:            {metrics.upwr_metrics.changed_words} / {metrics.upwr_metrics.total_partial_words}
  Chunks Analyzed:          {metrics.upwr_metrics.chunks_analyzed}
  Avg Words/Chunk:          {metrics.upwr_metrics.avg_words_per_chunk:.1f}

🔄 STREAMING vs BATCH TRANSCRIPTION
──────────────────────────────────────────────────────────────────
  Batch WER:                {metrics.batch_wer:.4f} ({metrics.batch_wer*100:.2f}%)
  Streaming WER:            {metrics.streaming_wer:.4f} ({metrics.streaming_wer*100:.2f}%)
  Degradation:              {metrics.streaming_degradation:+.2f} percentage points

⚖️  CHUNK-SIZE TRADE-OFF ANALYSIS
──────────────────────────────────────────────────────────────────
  Optimal Chunk Size:       {metrics.optimal_chunk_size:.2f}s

  Chunk Size    WER        Latency    Throughput
  ─────────────────────────────────────────────────────────────────"""

    for analysis in metrics.chunk_analyses:
        report += f"\n  {analysis.chunk_duration:>6.2f}s     {analysis.wer:.4f}    {analysis.latency_ms:>6.1f}ms    {analysis.throughput:.2f}x"

    report += f"""

🔊 NOISE ROBUSTNESS
──────────────────────────────────────────────────────────────────
  Clean WER:                {metrics.noise_robustness.clean_wer:.4f} ({metrics.noise_robustness.clean_wer*100:.2f}%)
  Noisy WER (avg):          {metrics.noise_robustness.noisy_wer:.4f} ({metrics.noise_robustness.noisy_wer*100:.2f}%)
  Normalized WER:           {metrics.noise_robustness.normalized_wer:.4f}
  WER Variance:             {metrics.noise_robustness.wer_variance:.6f}
  Degradation Factor:       {metrics.noise_robustness.degradation_factor:.2f}x
  SNR-WER Correlation:      {metrics.noise_robustness.snr_correlation:.4f}

⏱️  LATENCY METRICS
──────────────────────────────────────────────────────────────────
  Average Chunk Latency:    {metrics.avg_chunk_latency_ms:.2f}ms
  Total Processing Latency: {metrics.total_latency_ms:.2f}ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 METRICS GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UPWR (Unstable Partial Word Ratio)
  Fraction of interim words that change in subsequent updates
  Lower UPWR = more stable streaming transcription
  Typical: 0.10-0.30 for good streaming ASR

Streaming Degradation
  Typical: 10-17 percentage points worse than batch
  Due to limited future context in streaming mode

Chunk-Size Trade-off
  Smaller chunks = lower latency, higher WER
  Larger chunks = higher latency, lower WER
  Optimal: balance latency and accuracy for use case

Noise Robustness
  Normalized WER: relative degradation under noise
  WER Variance: stability across noise conditions
  Lower variance = more robust to noise
"""

    return report


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python realtime_asr.py <audio_file> <transcript_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    with open(sys.argv[2], 'r') as f:
        reference = f.read().strip()

    evaluator = RealtimeASREvaluator()
    metrics = evaluator.evaluate_all(audio_path, reference)

    print(format_realtime_report(metrics))

    # Save results
    import json
    output_path = "realtime_asr_results.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
