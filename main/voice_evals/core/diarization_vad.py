"""
Enhanced Speaker Diarization with VAD Chunking
Uses pyannote.audio with Voice Activity Detection for better speaker separation
"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.core import Segment, Annotation
    import torch.nn.functional as F
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class VADSegment:
    """Voice Activity Detection segment"""
    start: float
    end: float
    duration: float
    speaker: Optional[str] = None

    def __repr__(self):
        return f"VAD({self.start:.2f}s-{self.end:.2f}s, speaker={self.speaker})"


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
    vad_segments: List[VADSegment]
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


class VADDiarizer:
    """
    Voice Activity Detection + Speaker Diarization

    Two-stage approach:
    1. VAD: Detect speech segments (when anyone is speaking)
    2. Clustering: Group speech segments by speaker using embeddings
    """

    def __init__(self,
                 hf_token: Optional[str] = None,
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.3):
        """
        Initialize VAD-based diarizer

        Args:
            hf_token: HuggingFace token (optional for some models)
            min_speech_duration: Minimum speech segment duration (seconds)
            min_silence_duration: Minimum silence duration between segments
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio required. Install with: pip install pyannote.audio"
            )

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn"
            )

        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize VAD and speaker embedding models"""
        try:
            print("Loading VAD model...")
            # Use segmentation model for VAD (use 'token' parameter for newer API)
            self.vad_model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                token=self.hf_token
            )
            self.vad_model.to(self.device)
            print("✓ VAD model loaded")

            print("Loading speaker embedding model...")
            # Use pre-trained speaker embedding model
            self.embedding_model = PretrainedSpeakerEmbedding(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                device=self.device
            )
            print("✓ Speaker embedding model loaded")

        except Exception as e:
            print(f"Error loading models: {e}")
            print("\nTrying alternative approach without authentication...")

            # Fallback: use simpler models that don't require authentication
            try:
                print("Loading alternative VAD model...")
                self.vad_model = Model.from_pretrained("pyannote/segmentation")
                self.vad_model.to(self.device)

                print("Loading alternative embedding model...")
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=self.device
                )
                print("✓ Alternative models loaded successfully")
            except Exception as e2:
                raise RuntimeError(f"Failed to load models: {e2}")

    def detect_speech_vad(self, audio_path: str) -> List[VADSegment]:
        """
        Detect speech segments using VAD

        Args:
            audio_path: Path to audio file

        Returns:
            List of VAD segments (speech regions)
        """
        print(f"\n{'='*70}")
        print("VOICE ACTIVITY DETECTION")
        print(f"{'='*70}")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        waveform = waveform.to(self.device)

        # Run VAD model
        with torch.no_grad():
            # Process in chunks for long audio
            chunk_duration = 30  # seconds
            chunk_samples = chunk_duration * sample_rate

            vad_output = []
            num_chunks = int(np.ceil(waveform.shape[1] / chunk_samples))

            for i in range(num_chunks):
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, waveform.shape[1])
                chunk = waveform[:, start_sample:end_sample]

                output = self.vad_model({"waveform": chunk, "sample_rate": sample_rate})
                vad_output.append(output)

        # Merge VAD outputs and extract speech segments
        segments = []

        # Simple thresholding approach
        for chunk_idx, output in enumerate(vad_output):
            chunk_offset = chunk_idx * chunk_duration

            # Get speech probabilities
            if hasattr(output, 'data'):
                probs = output.data
            else:
                probs = output

            # Find speech regions (threshold at 0.5)
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()

            # Simple peak detection
            is_speech = probs > 0.5 if len(probs.shape) == 1 else probs[:, 0] > 0.5

            # Convert to time segments
            in_speech = False
            start_time = 0

            for i, speech in enumerate(is_speech):
                time = chunk_offset + (i / len(is_speech)) * chunk_duration

                if speech and not in_speech:
                    start_time = time
                    in_speech = True
                elif not speech and in_speech:
                    duration = time - start_time
                    if duration >= self.min_speech_duration:
                        segments.append(VADSegment(
                            start=start_time,
                            end=time,
                            duration=duration
                        ))
                    in_speech = False

        print(f"\n✓ Detected {len(segments)} speech segments")

        # Print summary
        total_speech = sum(s.duration for s in segments)
        print(f"  Total speech time: {total_speech:.2f}s")

        return segments

    def extract_speaker_embeddings(self,
                                   audio_path: str,
                                   vad_segments: List[VADSegment]) -> np.ndarray:
        """
        Extract speaker embeddings for each VAD segment

        Args:
            audio_path: Path to audio file
            vad_segments: List of speech segments

        Returns:
            Array of embeddings (num_segments, embedding_dim)
        """
        print(f"\n{'='*70}")
        print("EXTRACTING SPEAKER EMBEDDINGS")
        print(f"{'='*70}")

        embeddings = []

        for i, segment in enumerate(vad_segments):
            # Extract audio for this segment
            waveform, sample_rate = torchaudio.load(
                audio_path,
                frame_offset=int(segment.start * sample_rate),
                num_frames=int(segment.duration * sample_rate)
            )

            # Get embedding
            emb = self.embedding_model(waveform)
            embeddings.append(emb.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(vad_segments)} segments...")

        embeddings = np.vstack(embeddings)
        print(f"✓ Extracted {len(embeddings)} embeddings")

        return embeddings

    def cluster_speakers(self,
                        embeddings: np.ndarray,
                        num_speakers: Optional[int] = None) -> np.ndarray:
        """
        Cluster embeddings to identify speakers

        Args:
            embeddings: Speaker embeddings
            num_speakers: Fixed number of speakers (None = auto-detect)

        Returns:
            Cluster labels (speaker IDs)
        """
        print(f"\n{'='*70}")
        print("CLUSTERING SPEAKERS")
        print(f"{'='*70}")

        if num_speakers is None:
            # Auto-detect number of speakers (2-5 range)
            num_speakers = min(5, max(2, len(embeddings) // 10))

        print(f"Clustering into {num_speakers} speakers...")

        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Agglomerative clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric='cosine',
            linkage='average'
        )

        labels = clustering.fit_predict(embeddings_norm)

        print(f"✓ Identified {len(set(labels))} unique speakers")

        return labels

    def diarize(self,
                audio_path: str,
                num_speakers: Optional[int] = None,
                enable_vad_chunking: bool = True) -> DiarizationResult:
        """
        Perform complete speaker diarization with VAD

        Args:
            audio_path: Path to audio file
            num_speakers: Fixed number of speakers (None = auto-detect)
            enable_vad_chunking: Use VAD for chunking

        Returns:
            DiarizationResult with speaker segments
        """
        print(f"\n{'='*70}")
        print("SPEAKER DIARIZATION WITH VAD")
        print(f"{'='*70}")
        print(f"Audio: {audio_path}")
        print(f"VAD Chunking: {'Enabled' if enable_vad_chunking else 'Disabled'}")
        print(f"{'='*70}\n")

        # Step 1: VAD - detect speech segments
        vad_segments = self.detect_speech_vad(audio_path)

        # Step 2: Extract speaker embeddings
        embeddings = self.extract_speaker_embeddings(audio_path, vad_segments)

        # Step 3: Cluster speakers
        speaker_labels = self.cluster_speakers(embeddings, num_speakers)

        # Step 4: Assign speakers to segments
        speaker_segments = []
        for segment, label in zip(vad_segments, speaker_labels):
            segment.speaker = f"SPEAKER_{label:02d}"
            speaker_segments.append(SpeakerSegment(
                speaker_id=f"SPEAKER_{label:02d}",
                start_time=segment.start,
                end_time=segment.end,
                duration=segment.duration
            ))

        # Calculate statistics
        total_duration = max(s.end_time for s in speaker_segments)
        speaker_ids = set(s.speaker_id for s in speaker_segments)

        speaker_stats = {}
        for speaker_id in sorted(speaker_ids):
            speaker_segs = [s for s in speaker_segments if s.speaker_id == speaker_id]
            total_time = sum(s.duration for s in speaker_segs)

            speaker_stats[speaker_id] = {
                'total_time': total_time,
                'percentage': (total_time / total_duration * 100) if total_duration > 0 else 0,
                'num_segments': len(speaker_segs),
                'avg_segment_duration': total_time / len(speaker_segs) if speaker_segs else 0
            }

        result = DiarizationResult(
            segments=speaker_segments,
            vad_segments=vad_segments,
            num_speakers=len(speaker_ids),
            speaker_stats=speaker_stats,
            total_duration=total_duration
        )

        # Print summary
        print(f"\n{'='*70}")
        print("DIARIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Detected {len(speaker_ids)} speakers")
        print(f"✓ Total segments: {len(speaker_segments)}")
        print(f"✓ Total duration: {total_duration:.2f}s")

        print(f"\n📊 Speaker Statistics:")
        for speaker_id in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker_id]
            print(f"\n  {speaker_id}:")
            print(f"    Speaking time: {stats['total_time']:.2f}s ({stats['percentage']:.1f}%)")
            print(f"    Turns: {stats['num_segments']}")
            print(f"    Avg turn: {stats['avg_segment_duration']:.2f}s")

        return result

    def save_rttm(self, result: DiarizationResult, output_path: str):
        """Save diarization in RTTM format"""
        with open(output_path, 'w') as f:
            for seg in result.segments:
                f.write(f"SPEAKER file 1 {seg.start_time:.3f} {seg.duration:.3f} "
                       f"<NA> <NA> {seg.speaker_id} <NA> <NA>\n")
        print(f"✓ RTTM saved to: {output_path}")


def main():
    """Test VAD diarization"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diarization_vad.py <audio_file> [num_speakers]")
        sys.exit(1)

    audio_path = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    diarizer = VADDiarizer()
    result = diarizer.diarize(audio_path, num_speakers=num_speakers)

    # Save results
    output_rttm = Path(audio_path).stem + "_vad_diarization.rttm"
    diarizer.save_rttm(result, output_rttm)


if __name__ == "__main__":
    main()
