"""
Simple Speaker Diarization for Stereo Audio
Works directly with stereo channels without needing HuggingFace models
Perfect for recordings where speakers are already on separate channels
"""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SpeakerSegment:
    """Speaker segment with timing"""
    speaker_id: str
    start_time: float
    end_time: float
    duration: float

    def __repr__(self):
        return f"Speaker({self.speaker_id}, {self.start_time:.2f}s-{self.end_time:.2f}s)"


@dataclass
class DiarizationResult:
    """Complete diarization result"""
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_stats: Dict[str, Dict[str, float]]
    total_duration: float

    def get_speaker_audio_mask(self, speaker_id: str, sample_rate: int) -> np.ndarray:
        """Generate boolean mask for speaker's audio samples"""
        total_samples = int(self.total_duration * sample_rate)
        mask = np.zeros(total_samples, dtype=bool)

        for seg in self.segments:
            if seg.speaker_id == speaker_id:
                start_sample = int(seg.start_time * sample_rate)
                end_sample = int(seg.end_time * sample_rate)
                mask[start_sample:min(end_sample, total_samples)] = True

        return mask


class StereoDiarizer:
    """
    Simple diarization for stereo audio
    Assumes: Left channel = Speaker 0, Right channel = Speaker 1
    Uses energy-based VAD to detect when each speaker is active
    """

    def __init__(self,
                 min_speech_duration: float = 0.3,
                 energy_percentile: int = 30):
        """
        Initialize stereo diarizer

        Args:
            min_speech_duration: Minimum speech segment duration (seconds)
            energy_percentile: Percentile for energy threshold (lower = more sensitive)
        """
        self.min_speech_duration = min_speech_duration
        self.energy_percentile = energy_percentile

    def detect_speech_segments(self,
                               audio: np.ndarray,
                               sample_rate: int,
                               speaker_id: str) -> List[SpeakerSegment]:
        """
        Detect speech segments in a single channel using energy-based VAD

        Args:
            audio: Audio samples (1D array)
            sample_rate: Sample rate
            speaker_id: Speaker identifier

        Returns:
            List of speech segments
        """
        # Calculate frame energy
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop

        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Adaptive threshold based on percentile
        threshold = np.percentile(energy, self.energy_percentile)

        # Find speech frames
        speech_frames = energy > threshold

        # Convert to time segments
        segments = []
        in_speech = False
        start_frame = 0

        for i, is_speech in enumerate(speech_frames):
            time = librosa.frames_to_time(i, sr=sample_rate, hop_length=hop_length)

            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=hop_length)
                end_time = time
                duration = end_time - start_time

                if duration >= self.min_speech_duration:
                    segments.append(SpeakerSegment(
                        speaker_id=speaker_id,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration
                    ))
                in_speech = False

        # Handle case where speech continues to end
        if in_speech:
            start_time = librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=hop_length)
            end_time = librosa.get_duration(y=audio, sr=sample_rate)
            duration = end_time - start_time

            if duration >= self.min_speech_duration:
                segments.append(SpeakerSegment(
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration
                ))

        return segments

    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Perform speaker diarization on stereo audio

        Args:
            audio_path: Path to stereo audio file

        Returns:
            DiarizationResult with speaker segments
        """
        print(f"\n{'='*70}")
        print("STEREO SPEAKER DIARIZATION")
        print(f"{'='*70}")
        print(f"Audio: {audio_path}")
        print(f"{'='*70}\n")

        # Load audio
        print("Loading audio...")
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # Check if stereo
        if len(audio.shape) != 2 or audio.shape[0] != 2:
            raise ValueError(
                f"Expected stereo audio (2 channels), got shape: {audio.shape}\n"
                "This diarizer requires stereo audio with speakers on separate channels."
            )

        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"✓ Stereo audio loaded: {duration:.2f}s @ {sr}Hz")

        # Extract channels
        left_channel = audio[0]   # Speaker 0
        right_channel = audio[1]  # Speaker 1

        # Detect speech for each speaker
        print(f"\nDetecting speech segments...")
        print(f"  Analyzing left channel (SPEAKER_00)...")
        speaker_0_segments = self.detect_speech_segments(left_channel, sr, "SPEAKER_00")
        print(f"  ✓ Found {len(speaker_0_segments)} segments for SPEAKER_00")

        print(f"  Analyzing right channel (SPEAKER_01)...")
        speaker_1_segments = self.detect_speech_segments(right_channel, sr, "SPEAKER_01")
        print(f"  ✓ Found {len(speaker_1_segments)} segments for SPEAKER_01")

        # Combine all segments
        all_segments = speaker_0_segments + speaker_1_segments

        # Sort by start time
        all_segments.sort(key=lambda x: x.start_time)

        # Calculate statistics
        speaker_stats = {}
        for speaker_id in ["SPEAKER_00", "SPEAKER_01"]:
            speaker_segs = [s for s in all_segments if s.speaker_id == speaker_id]
            total_time = sum(s.duration for s in speaker_segs)

            speaker_stats[speaker_id] = {
                'total_time': total_time,
                'percentage': (total_time / duration * 100) if duration > 0 else 0,
                'num_segments': len(speaker_segs),
                'avg_segment_duration': total_time / len(speaker_segs) if speaker_segs else 0
            }

        result = DiarizationResult(
            segments=all_segments,
            num_speakers=2,
            speaker_stats=speaker_stats,
            total_duration=duration
        )

        # Print summary
        print(f"\n{'='*70}")
        print("DIARIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Detected 2 speakers (stereo channels)")
        print(f"✓ Total segments: {len(all_segments)}")
        print(f"✓ Total duration: {duration:.2f}s")

        print(f"\n📊 Speaker Statistics:")
        for speaker_id in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker_id]
            print(f"\n  {speaker_id} ({'Left' if speaker_id == 'SPEAKER_00' else 'Right'} channel):")
            print(f"    Speaking time: {stats['total_time']:.2f}s ({stats['percentage']:.1f}%)")
            print(f"    Turns: {stats['num_segments']}")
            print(f"    Avg turn: {stats['avg_segment_duration']:.2f}s")

        # Visual timeline
        self.visualize_timeline(result)

        return result

    def visualize_timeline(self, result: DiarizationResult):
        """Create text-based timeline visualization"""
        print(f"\n{'='*70}")
        print("SPEAKER TIMELINE")
        print(f"{'='*70}")

        timeline_length = 70
        timeline_scale = result.total_duration / timeline_length

        for speaker_id in ["SPEAKER_00", "SPEAKER_01"]:
            speaker_segs = [s for s in result.segments if s.speaker_id == speaker_id]

            line = [' '] * timeline_length
            for seg in speaker_segs:
                start_pos = int(seg.start_time / timeline_scale)
                end_pos = int(seg.end_time / timeline_scale)
                for i in range(start_pos, min(end_pos + 1, timeline_length)):
                    line[i] = '█'

            channel = 'L' if speaker_id == 'SPEAKER_00' else 'R'
            print(f"{speaker_id} ({channel}): {''.join(line)}")

        # Time markers
        print(f"Time (s):    ", end="")
        for i in range(0, timeline_length, 10):
            time_at_pos = i * timeline_scale
            print(f"{time_at_pos:>5.0f}     ", end="")
        print()
        print(f"{'='*70}\n")

    def save_rttm(self, result: DiarizationResult, output_path: str):
        """Save diarization in RTTM format"""
        with open(output_path, 'w') as f:
            for seg in result.segments:
                f.write(f"SPEAKER file 1 {seg.start_time:.3f} {seg.duration:.3f} "
                       f"<NA> <NA> {seg.speaker_id} <NA> <NA>\n")
        print(f"✓ RTTM saved to: {output_path}")


def main():
    """Test stereo diarization"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diarization_stereo.py <stereo_audio_file>")
        print("\nNote: Audio file must be stereo (2 channels)")
        print("      Left channel = Speaker 0, Right channel = Speaker 1")
        sys.exit(1)

    audio_path = sys.argv[1]

    diarizer = StereoDiarizer()
    result = diarizer.diarize(audio_path)

    # Save results
    output_rttm = Path(audio_path).stem + "_stereo_diarization.rttm"
    diarizer.save_rttm(result, output_rttm)

    print(f"\n✅ Diarization complete!")
    print(f"   Segments: {len(result.segments)}")
    print(f"   Speakers: {result.num_speakers}")
    print(f"   Duration: {result.total_duration:.2f}s")


if __name__ == "__main__":
    main()
