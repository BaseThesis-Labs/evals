"""
Main Voice Evaluation Pipeline with Speaker Diarization
Integrates all evaluation metrics with per-speaker analysis
"""

import os
import sys
import json
import time
import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import from pipeline2
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline2 import EnhancedVoiceEvaluator, EnhancedVoiceMetrics, format_report

from .diarization import SpeakerDiarizer, DiarizationResult


@dataclass
class SpeakerMetrics:
    """Metrics for individual speaker"""
    speaker_id: str
    speaking_time: float
    speaking_percentage: float
    num_turns: int
    avg_turn_duration: float
    words_per_minute: float
    transcript: str
    word_count: int


@dataclass
class DiarizedVoiceMetrics:
    """Complete metrics with speaker diarization"""
    # Overall metrics (from EnhancedVoiceMetrics)
    overall_metrics: Dict

    # Diarization results
    num_speakers: int
    speaker_metrics: List[SpeakerMetrics]
    diarization_timeline: str

    # Per-speaker detailed metrics (optional)
    per_speaker_detailed: Optional[Dict[str, Dict]] = None


class VoiceEvaluationPipeline:
    """
    Complete voice evaluation pipeline with speaker diarization

    Workflow:
    1. Perform speaker diarization to identify speakers
    2. Extract audio for each speaker
    3. Run full evaluation pipeline on overall audio
    4. Optionally run per-speaker evaluations
    """

    def __init__(self,
                 audio_path: str,
                 transcript_path: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 enable_diarization: bool = True,
                 num_speakers: Optional[int] = None,
                 utmos_model_dir: Optional[str] = None,
                 saer_lambda: float = 0.5,
                 saer_lang: str = "mixed",
                 per_speaker_evaluation: bool = False):
        """
        Initialize the evaluation pipeline

        Args:
            audio_path: Path to audio file
            transcript_path: Optional ground truth transcript
            hf_token: HuggingFace token for diarization
            enable_diarization: Whether to perform speaker diarization
            num_speakers: Fixed number of speakers (None = auto-detect)
            utmos_model_dir: Directory for UTMOS model
            saer_lambda: Lambda weight for SAER metric
            saer_lang: Language code for SAER
            per_speaker_evaluation: Run full eval on each speaker (slower)
        """
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.enable_diarization = enable_diarization
        self.num_speakers = num_speakers
        self.per_speaker_evaluation = per_speaker_evaluation

        print(f"\n{'='*70}")
        print("VOICE EVALUATION PIPELINE WITH SPEAKER DIARIZATION")
        print(f"{'='*70}")
        print(f"Audio: {audio_path}")
        print(f"Diarization: {'Enabled' if enable_diarization else 'Disabled'}")
        print(f"{'='*70}\n")

        # Load audio
        print("Loading audio...")
        self.audio, self.sr = librosa.load(audio_path, sr=None, mono=False)
        self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
        print(f"✓ Audio loaded: {self.duration:.2f}s @ {self.sr}Hz")

        # Initialize components
        self.diarizer = None
        self.diarization_result = None

        if enable_diarization:
            try:
                self.diarizer = SpeakerDiarizer(
                    hf_token=hf_token,
                    min_speakers=num_speakers or 2,
                    max_speakers=num_speakers or 2
                )
            except Exception as e:
                print(f"⚠️  Diarization disabled: {e}")
                self.enable_diarization = False

        # Initialize main evaluator
        self.evaluator = EnhancedVoiceEvaluator(
            audio_path=audio_path,
            transcript_path=transcript_path,
            utmos_model_dir=utmos_model_dir,
            saer_lambda=saer_lambda,
            saer_lang=saer_lang
        )

    def run_diarization(self) -> Optional[DiarizationResult]:
        """Perform speaker diarization"""
        if not self.enable_diarization or not self.diarizer:
            return None

        try:
            self.diarization_result = self.diarizer.diarize(
                self.audio_path,
                num_speakers=self.num_speakers
            )

            # Visualize
            self.diarizer.visualize_diarization(self.diarization_result)

            return self.diarization_result

        except Exception as e:
            print(f"⚠️  Diarization failed: {e}")
            return None

    def extract_speaker_metrics(self, diarization: DiarizationResult) -> List[SpeakerMetrics]:
        """Extract basic metrics for each speaker"""
        speaker_metrics = []

        for speaker_id in sorted(diarization.speaker_stats.keys()):
            stats = diarization.speaker_stats[speaker_id]

            # Extract speaker audio for transcription (simplified)
            speaker_segments = [s for s in diarization.segments if s.speaker_id == speaker_id]

            # Estimate word count based on speaking time (rough estimate: 150 wpm average)
            estimated_words = int(stats['total_time'] / 60 * 150)

            metrics = SpeakerMetrics(
                speaker_id=speaker_id,
                speaking_time=stats['total_time'],
                speaking_percentage=stats['percentage'],
                num_turns=stats['num_segments'],
                avg_turn_duration=stats['avg_segment_duration'],
                words_per_minute=0.0,  # Will be calculated if transcription available
                transcript="",  # Will be filled if per-speaker eval enabled
                word_count=estimated_words
            )

            speaker_metrics.append(metrics)

        return speaker_metrics

    def create_timeline_visualization(self, diarization: DiarizationResult) -> str:
        """Create a text-based timeline of speaker turns"""
        timeline_parts = []
        timeline_parts.append("="*70)
        timeline_parts.append("SPEAKER TIMELINE")
        timeline_parts.append("="*70)

        timeline_length = 60
        timeline_scale = diarization.total_duration / timeline_length

        for speaker_id in sorted(diarization.speaker_stats.keys()):
            speaker_segs = [s for s in diarization.segments if s.speaker_id == speaker_id]

            line = [' '] * timeline_length
            for seg in speaker_segs:
                start_pos = int(seg.start_time / timeline_scale)
                end_pos = int(seg.end_time / timeline_scale)
                for i in range(start_pos, min(end_pos + 1, timeline_length)):
                    line[i] = '█'

            timeline_parts.append(f"{speaker_id}: {''.join(line)}")

        return "\n".join(timeline_parts)

    def evaluate(self) -> DiarizedVoiceMetrics:
        """
        Run complete evaluation pipeline

        Returns:
            DiarizedVoiceMetrics with overall and per-speaker results
        """
        print(f"\n{'='*70}")
        print("STARTING COMPREHENSIVE EVALUATION")
        print(f"{'='*70}\n")

        # Step 1: Run diarization
        diarization = None
        speaker_metrics = []
        timeline = ""

        if self.enable_diarization:
            diarization = self.run_diarization()

            if diarization:
                # Extract speaker metrics
                speaker_metrics = self.extract_speaker_metrics(diarization)

                # Create timeline
                timeline = self.create_timeline_visualization(diarization)

        # Step 2: Run overall evaluation
        print(f"\n{'='*70}")
        print("OVERALL AUDIO EVALUATION")
        print(f"{'='*70}")

        overall_metrics_obj = self.evaluator.evaluate_all()
        overall_metrics = asdict(overall_metrics_obj)

        # Step 3: Per-speaker detailed evaluation (optional)
        per_speaker_detailed = None
        if self.per_speaker_evaluation and diarization:
            print(f"\n{'='*70}")
            print("PER-SPEAKER DETAILED EVALUATION")
            print(f"{'='*70}")
            per_speaker_detailed = {}

            # Note: This would require extracting audio per speaker and re-running evaluation
            # Skipping for now to keep the implementation focused
            print("⚠️  Per-speaker detailed evaluation not yet implemented")
            print("    (would require speaker-separated audio re-evaluation)")

        # Compile results
        result = DiarizedVoiceMetrics(
            overall_metrics=overall_metrics,
            num_speakers=diarization.num_speakers if diarization else 0,
            speaker_metrics=speaker_metrics,
            diarization_timeline=timeline,
            per_speaker_detailed=per_speaker_detailed
        )

        print(f"\n{'='*70}")
        print("✓ EVALUATION COMPLETE")
        print(f"{'='*70}\n")

        return result

    def generate_report(self, metrics: DiarizedVoiceMetrics) -> str:
        """Generate comprehensive report"""
        report_parts = []

        # Overall metrics report
        report_parts.append(format_report(
            EnhancedVoiceMetrics(**metrics.overall_metrics)
        ))

        # Diarization section
        if metrics.num_speakers > 0:
            report_parts.append("\n" + "="*70)
            report_parts.append("SPEAKER DIARIZATION ANALYSIS")
            report_parts.append("="*70)
            report_parts.append(f"\n📊 Detected {metrics.num_speakers} speakers\n")

            for sm in metrics.speaker_metrics:
                report_parts.append(f"\n{sm.speaker_id}:")
                report_parts.append(f"  Speaking time:    {sm.speaking_time:.2f}s ({sm.speaking_percentage:.1f}%)")
                report_parts.append(f"  Number of turns:  {sm.num_turns}")
                report_parts.append(f"  Avg turn length:  {sm.avg_turn_duration:.2f}s")
                report_parts.append(f"  Est. word count:  {sm.word_count}")

            report_parts.append(f"\n{metrics.diarization_timeline}\n")

        return "\n".join(report_parts)

    def save_results(self, metrics: DiarizedVoiceMetrics, output_path: Optional[str] = None):
        """Save results to JSON"""
        if output_path is None:
            output_path = Path(self.audio_path).stem + "_diarized_eval.json"

        def to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                return to_serializable(obj.__dict__)
            return obj

        metrics_dict = asdict(metrics)
        metrics_dict = to_serializable(metrics_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {output_path}")

        # Also save diarization RTTM if available
        if self.diarization_result and self.diarizer:
            rttm_path = Path(self.audio_path).stem + "_diarization.rttm"
            self.diarizer.save_diarization_rttm(self.diarization_result, rttm_path)


def main():
    """CLI entry point"""
    import sys

    if len(sys.argv) < 2:
        print("""
Usage: python pipeline.py <audio_file> [options]

Options:
  transcript_file       Ground truth transcript (optional)
  --hf-token TOKEN     HuggingFace token for diarization
  --num-speakers N     Fixed number of speakers (default: auto-detect 2)
  --no-diarization     Disable speaker diarization
  --per-speaker        Enable per-speaker detailed evaluation

Examples:
  python pipeline.py audio.wav
  python pipeline.py audio.wav transcript.txt --hf-token hf_xxx
  python pipeline.py audio.wav --num-speakers 3
  python pipeline.py audio.wav --no-diarization
        """)
        sys.exit(1)

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
            transcript_path = arg
            i += 1
        else:
            i += 1

    # Run pipeline
    pipeline = VoiceEvaluationPipeline(
        audio_path=audio_path,
        transcript_path=transcript_path,
        hf_token=hf_token,
        enable_diarization=enable_diarization,
        num_speakers=num_speakers,
        per_speaker_evaluation=per_speaker
    )

    # Evaluate
    metrics = pipeline.evaluate()

    # Generate and print report
    report = pipeline.generate_report(metrics)
    print(report)

    # Save results
    pipeline.save_results(metrics)


if __name__ == "__main__":
    main()
