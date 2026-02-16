"""
Speaker Diarization using Pyannote.audio
Identifies and separates speakers in audio recordings
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")


@dataclass
class SpeakerSegment:
    """Represents a speaker segment in the audio"""
    speaker_id: str
    start_time: float
    end_time: float
    duration: float

    def __repr__(self):
        return f"Speaker({self.speaker_id}, {self.start_time:.2f}s-{self.end_time:.2f}s)"


@dataclass
class DiarizationResult:
    """Complete diarization result for an audio file"""
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_stats: Dict[str, Dict[str, float]]  # speaker_id -> {total_time, percentage, num_segments}
    total_duration: float

    def get_speaker_audio_mask(self, speaker_id: str, sample_rate: int) -> np.ndarray:
        """Generate a boolean mask for samples belonging to a specific speaker"""
        total_samples = int(self.total_duration * sample_rate)
        mask = np.zeros(total_samples, dtype=bool)

        for seg in self.segments:
            if seg.speaker_id == speaker_id:
                start_sample = int(seg.start_time * sample_rate)
                end_sample = int(seg.end_time * sample_rate)
                mask[start_sample:end_sample] = True

        return mask

    def get_speaker_segments_text(self, speaker_id: str) -> str:
        """Get formatted text showing all segments for a speaker"""
        speaker_segs = [s for s in self.segments if s.speaker_id == speaker_id]
        return "\n".join([f"  {s.start_time:.2f}s - {s.end_time:.2f}s ({s.duration:.2f}s)"
                         for s in speaker_segs])


class SpeakerDiarizer:
    """
    Speaker diarization using Pyannote.audio pipeline

    Separates audio into speaker segments and provides per-speaker analysis
    """

    def __init__(self,
                 hf_token: Optional[str] = None,
                 min_speakers: int = 2,
                 max_speakers: int = 2,
                 use_auth_token: bool = True):
        """
        Initialize the diarization pipeline

        Args:
            hf_token: HuggingFace API token for pyannote models
                     (get from: https://huggingface.co/settings/tokens)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            use_auth_token: Whether to use authentication token
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is required for speaker diarization.\n"
                "Install with: pip install pyannote.audio\n"
                "You'll also need a HuggingFace token: "
                "https://huggingface.co/pyannote/speaker-diarization"
            )

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Get token from environment or parameter
        self.hf_token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

        if not self.hf_token and use_auth_token:
            print("""
⚠️  Warning: No HuggingFace token found!

To use speaker diarization, you need to:
1. Create a HuggingFace account: https://huggingface.co
2. Accept the terms for pyannote models:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
3. Generate a token: https://huggingface.co/settings/tokens
4. Set it as an environment variable:
   export HF_TOKEN="your-token-here"

Or pass it directly: SpeakerDiarizer(hf_token="your-token")
""")
            raise ValueError("HuggingFace token required for speaker diarization")

        print("Loading Pyannote speaker diarization pipeline...")
        try:
            # Use 'token' parameter for newer versions of pyannote.audio
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token if use_auth_token else None
            )

            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)

            print(f"✓ Diarization pipeline loaded on {device}")

        except Exception as e:
            print(f"Error loading diarization pipeline: {e}")
            raise

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to audio file
            num_speakers: Optional fixed number of speakers (overrides min/max)

        Returns:
            DiarizationResult with speaker segments and statistics
        """
        print(f"\n{'='*70}")
        print(f"SPEAKER DIARIZATION")
        print(f"{'='*70}")
        print(f"Audio file: {audio_path}")

        # Run diarization
        if num_speakers:
            print(f"Running diarization with {num_speakers} speakers...")
            diarization = self.pipeline(audio_path, num_speakers=num_speakers)
        else:
            print(f"Running diarization (detecting {self.min_speakers}-{self.max_speakers} speakers)...")
            diarization = self.pipeline(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )

        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            seg = SpeakerSegment(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
                duration=turn.end - turn.start
            )
            segments.append(seg)

        # Calculate statistics
        speaker_ids = set(seg.speaker_id for seg in segments)
        num_speakers = len(speaker_ids)

        # Get total duration from diarization
        total_duration = max(seg.end_time for seg in segments) if segments else 0.0

        speaker_stats = {}
        for speaker_id in speaker_ids:
            speaker_segments = [s for s in segments if s.speaker_id == speaker_id]
            total_time = sum(s.duration for s in speaker_segments)

            speaker_stats[speaker_id] = {
                'total_time': total_time,
                'percentage': (total_time / total_duration * 100) if total_duration > 0 else 0,
                'num_segments': len(speaker_segments),
                'avg_segment_duration': total_time / len(speaker_segments) if speaker_segments else 0
            }

        result = DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            speaker_stats=speaker_stats,
            total_duration=total_duration
        )

        # Print summary
        print(f"\n✓ Diarization complete!")
        print(f"  Detected {num_speakers} speakers")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total segments: {len(segments)}")

        print(f"\n📊 Speaker Statistics:")
        for speaker_id in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker_id]
            print(f"\n  {speaker_id}:")
            print(f"    Total speaking time: {stats['total_time']:.2f}s ({stats['percentage']:.1f}%)")
            print(f"    Number of turns: {stats['num_segments']}")
            print(f"    Avg turn duration: {stats['avg_segment_duration']:.2f}s")

        return result

    def extract_speaker_audio(self,
                            audio: np.ndarray,
                            sample_rate: int,
                            diarization: DiarizationResult,
                            speaker_id: str) -> np.ndarray:
        """
        Extract audio samples for a specific speaker

        Args:
            audio: Full audio array
            sample_rate: Sample rate
            diarization: DiarizationResult object
            speaker_id: Speaker ID to extract

        Returns:
            Audio array containing only the specified speaker's segments
        """
        mask = diarization.get_speaker_audio_mask(speaker_id, sample_rate)

        # Handle stereo audio
        if len(audio.shape) == 2:
            return audio[:, mask]
        else:
            return audio[mask]

    def save_diarization_rttm(self, diarization: DiarizationResult, output_path: str):
        """
        Save diarization result in RTTM format

        RTTM is the standard format for speaker diarization annotations
        """
        with open(output_path, 'w') as f:
            for seg in diarization.segments:
                # RTTM format: SPEAKER <file> <channel> <start> <duration> <na> <na> <speaker> <conf> <na>
                f.write(f"SPEAKER file 1 {seg.start_time:.3f} {seg.duration:.3f} <NA> <NA> {seg.speaker_id} <NA> <NA>\n")

        print(f"✓ Diarization saved to: {output_path}")

    def visualize_diarization(self, diarization: DiarizationResult, output_path: Optional[str] = None):
        """
        Create a text-based visualization of speaker turns
        """
        print(f"\n{'='*70}")
        print("SPEAKER TIMELINE")
        print(f"{'='*70}")

        # Group segments by speaker
        speaker_colors = {
            speaker: i for i, speaker in enumerate(sorted(diarization.speaker_stats.keys()))
        }

        # Create timeline
        timeline_length = 80
        timeline_scale = diarization.total_duration / timeline_length

        for speaker_id in sorted(speaker_colors.keys()):
            speaker_segs = [s for s in diarization.segments if s.speaker_id == speaker_id]

            # Create visual line
            line = [' '] * timeline_length
            for seg in speaker_segs:
                start_pos = int(seg.start_time / timeline_scale)
                end_pos = int(seg.end_time / timeline_scale)
                for i in range(start_pos, min(end_pos + 1, timeline_length)):
                    line[i] = '█'

            print(f"{speaker_id}: {''.join(line)}")

        # Time markers
        markers = []
        for i in range(0, timeline_length, 10):
            time_at_pos = i * timeline_scale
            markers.append(f"{time_at_pos:.0f}s")

        print(f"Time: {' ' * 10}".join(markers))
        print(f"{'='*70}\n")

        if output_path:
            with open(output_path, 'w') as f:
                f.write("Speaker Timeline\n")
                f.write("="*70 + "\n")
                for speaker_id in sorted(speaker_colors.keys()):
                    stats = diarization.speaker_stats[speaker_id]
                    f.write(f"\n{speaker_id}:\n")
                    f.write(diarization.get_speaker_segments_text(speaker_id))
                    f.write(f"\n\nTotal: {stats['total_time']:.2f}s ({stats['percentage']:.1f}%)\n")


def main():
    """Example usage of speaker diarization"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diarization.py <audio_file> [hf_token] [num_speakers]")
        print("\nExample:")
        print("  python diarization.py audio.wav")
        print("  python diarization.py audio.wav hf_xxx 2")
        sys.exit(1)

    audio_path = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Initialize diarizer
    diarizer = SpeakerDiarizer(hf_token=hf_token)

    # Perform diarization
    result = diarizer.diarize(audio_path, num_speakers=num_speakers)

    # Visualize
    diarizer.visualize_diarization(result)

    # Save RTTM
    output_rttm = Path(audio_path).stem + "_diarization.rttm"
    diarizer.save_diarization_rttm(result, output_rttm)


if __name__ == "__main__":
    main()
