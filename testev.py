"""
Enhanced Voice Evaluation Pipeline
Adds: Normalized WER, Word Accuracy, RTFx, proper WER (not CER),
      Match Error Rate (MER), Word Information Preserved (WIP),
      Word Information Lost (WIL), SeMaScore, SAER, and
      Aligned Semantic Distance (ASD).
"""

import os
import json
import librosa
import numpy as np
import torch
import torchaudio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import warnings
import time
import re
warnings.filterwarnings('ignore')

# Optional imports with graceful fallbacks
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

try:
    from funasr import AutoModel as FunASRModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("Warning: funasr not available for emotion. Install with: pip install funasr")

# ── NEW: semantic metric dependencies ──────────────────────────────────────
try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: BERT models not available for SeMaScore/ASD.")

try:
    from sentence_transformers import SentenceTransformer
    LABSE_AVAILABLE = True
except ImportError:
    LABSE_AVAILABLE = False
    print("Warning: sentence-transformers not available for SAER/LaBSE. "
          "Install with: pip install sentence-transformers")
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class AlignmentCounts:
    """
    Raw counts from Levenshtein alignment between reference and hypothesis.
    All four values are needed to derive WER, MER, WIP, and WIL without
    running the expensive alignment twice.

      H  = correct hits (reference word == hypothesis word)
      S  = substitutions
      D  = deletions   (in reference, missing from hypothesis)
      I  = insertions  (in hypothesis, absent from reference)
    """
    H: int
    S: int
    D: int
    I: int


class BasicTextNormalizer:
    """Basic text normalizer for computing normalized WER"""

    def __call__(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        fillers = ['um', 'uh', 'ah', 'hmm', 'mm', 'mhm', 'uhuh']
        words = [w for w in text.split() if w not in fillers]
        return re.sub(r'\s+', ' ', ' '.join(words)).strip()


# ══════════════════════════════════════════════════════════════════════════
#  NEW: SeMaScore helper
# ══════════════════════════════════════════════════════════════════════════

class SeMaScoreCalculator:
    """
    SeMaScore (Sasindran et al., Interspeech 2024).

    Four-phase pipeline:
      1. Segment mapping  – character-level Levenshtein aligns ref ↔ hyp words
      2. BERTScore cosine similarity per segment, penalised by MER
      3. Importance weighting relative to full-sentence BERT embedding
      4. Weighted aggregation → final score in [0, 1]

    The MER penalty corrects the BERTScore failure mode where phonetically
    similar but semantically different words (e.g. "Smoking" / "Something")
    receive raw cosine ≈ 0.98; SeMaScore maps these to ≈ 0.30.
    """

    # BERT model used for embeddings (same as original paper)
    _MODEL_NAME = "bert-base-uncased"

    def __init__(self):
        if not BERT_AVAILABLE:
            raise ImportError("transformers is required for SeMaScore.")
        print("Loading BERT model for SeMaScore …")
        self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        self._model = AutoModel.from_pretrained(self._MODEL_NAME)
        self._model.eval()
        print("BERT model loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _char_edit_distance(a: str, b: str) -> int:
        """Standard character-level Levenshtein distance."""
        la, lb = len(a), len(b)
        dp = np.arange(lb + 1, dtype=np.int32)
        for i in range(1, la + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, lb + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return int(dp[lb])

    def _align_words(self, ref_words: List[str], hyp_words: List[str]
                     ) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Phase 1 – character-level Levenshtein word alignment.
        Returns a list of (ref_word | None, hyp_word | None) pairs.
        """
        lr, lh = len(ref_words), len(hyp_words)
        # Build cost matrix using char edit distance for substitutions
        dp = np.full((lr + 1, lh + 1), np.inf)
        dp[0, :] = np.arange(lh + 1)
        dp[:, 0] = np.arange(lr + 1)

        for i in range(1, lr + 1):
            for j in range(1, lh + 1):
                sub_cost = self._char_edit_distance(
                    ref_words[i - 1].lower(), hyp_words[j - 1].lower()
                ) / max(len(ref_words[i - 1]), len(hyp_words[j - 1]), 1)
                dp[i, j] = min(
                    dp[i - 1, j - 1] + sub_cost,   # substitution / match
                    dp[i - 1, j] + 1.0,              # deletion
                    dp[i, j - 1] + 1.0,              # insertion
                )

        # Traceback
        pairs: List[Tuple[Optional[str], Optional[str]]] = []
        i, j = lr, lh
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub_cost = self._char_edit_distance(
                    ref_words[i - 1].lower(), hyp_words[j - 1].lower()
                ) / max(len(ref_words[i - 1]), len(hyp_words[j - 1]), 1)
                if dp[i, j] == dp[i - 1, j - 1] + sub_cost:
                    pairs.append((ref_words[i - 1], hyp_words[j - 1]))
                    i -= 1; j -= 1
                    continue
            if i > 0 and dp[i, j] == dp[i - 1, j] + 1.0:
                pairs.append((ref_words[i - 1], None))   # deletion
                i -= 1
            else:
                pairs.append((None, hyp_words[j - 1]))   # insertion
                j -= 1

        return list(reversed(pairs))

    def _get_word_embeddings(self, words: List[str]) -> torch.Tensor:
        """Return a (len(words), hidden) tensor of contextualised embeddings."""
        if not words:
            return torch.zeros(0, 768)
        text = " ".join(words)
        inputs = self._tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Last-hidden-state: shape (1, seq_len, hidden)
        token_embs = outputs.last_hidden_state[0]   # (seq_len, hidden)
        # Map sub-word tokens back to words (simple: average per word)
        word_ids = inputs.word_ids(batch_index=0)
        word_embs = []
        for wid in range(len(words)):
            mask = [i for i, w in enumerate(word_ids) if w == wid]
            if mask:
                word_embs.append(token_embs[mask].mean(0))
            else:
                word_embs.append(torch.zeros(token_embs.shape[-1]))
        return torch.stack(word_embs)   # (len(words), hidden)

    def _sentence_embedding(self, text: str) -> torch.Tensor:
        """CLS-token embedding for importance weighting."""
        inputs = self._tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=512)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.last_hidden_state[0, 0]   # (hidden,)

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    @staticmethod
    def _mer_penalty(ref_word: str, hyp_word: str) -> float:
        """
        MER-based penalty that collapses BERTScore's over-confidence on
        phonetically similar but semantically distinct word pairs.

        Maps the character-edit-distance ratio to a [0, 1] penalty via a
        sigmoid-like curve so that identical words → 0 penalty and very
        different words → penalty close to 1.
        """
        dist = sum(a != b for a, b in zip(ref_word.lower(), hyp_word.lower()))
        dist += abs(len(ref_word) - len(hyp_word))
        max_len = max(len(ref_word), len(hyp_word), 1)
        ratio = dist / max_len
        # Smooth penalty: 0 when ratio=0, approaches 1 when ratio→1
        return 1 - (1 - ratio) ** 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, reference: str, hypothesis: str) -> float:
        """
        Compute SeMaScore ∈ [0, 1].

        Returns 1.0 for identical strings, approaches 0 for completely
        different sequences.
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 1.0 if not hyp_words else 0.0

        # Phase 1: word alignment
        pairs = self._align_words(ref_words, hyp_words)

        # Phase 2: per-segment BERT cosine + MER penalty
        ref_embs = self._get_word_embeddings(ref_words)
        hyp_embs = self._get_word_embeddings(hyp_words)

        ref_idx = hyp_idx = 0
        segment_scores: List[float] = []
        segment_ref_words: List[str] = []

        for (rw, hw) in pairs:
            if rw is None:              # insertion – penalise fully
                segment_scores.append(0.0)
                segment_ref_words.append(hw or "")
                hyp_idx += 1
                continue

            segment_ref_words.append(rw)

            if hw is None:              # deletion – zero similarity
                segment_scores.append(0.0)
                ref_idx += 1
                continue

            # Get safe embedding indices
            ri = min(ref_idx, len(ref_embs) - 1)
            hi = min(hyp_idx, len(hyp_embs) - 1)
            cosine = self._cosine(ref_embs[ri], hyp_embs[hi])

            # Apply MER penalty
            penalty = self._mer_penalty(rw, hw)
            adjusted = cosine * (1 - penalty)
            segment_scores.append(float(np.clip(adjusted, 0.0, 1.0)))

            ref_idx += 1
            hyp_idx += 1

        if not segment_scores:
            return 0.0

        # Phase 3: importance weighting via sentence-level similarity
        sentence_emb = self._sentence_embedding(reference)
        weights: List[float] = []
        for w in segment_ref_words:
            if not w:
                weights.append(1.0)
                continue
            word_tok = self._tokenizer(w, return_tensors="pt")
            with torch.no_grad():
                word_out = self._model(**word_tok)
            word_emb = word_out.last_hidden_state[0, 0]
            sim = max(0.0, self._cosine(word_emb, sentence_emb))
            weights.append(sim if sim > 0 else 1e-6)

        total_weight = sum(weights)
        if total_weight == 0:
            return float(np.mean(segment_scores))

        # Phase 4: weighted aggregation
        semascore = sum(s * w for s, w in zip(segment_scores, weights)) / total_weight
        return float(np.clip(semascore, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
#  NEW: SAER helper
# ══════════════════════════════════════════════════════════════════════════

class SAERCalculator:
    """
    Semantic-Aware Error Rate (SAER) — SwitchLingua (2025).

        SAER = λ × F_form + (1 − λ) × ε_sem

    where:
      F_form  — language-appropriate form error
                  CER for logographic scripts (zh, ja, ko)
                  WER for alphabetic scripts
      ε_sem   — semantic dissimilarity via LaBSE multilingual embeddings
                  = 1 − cosine_similarity(emb_ref, emb_hyp)
      λ       — balance weight (default 0.5)

    For code-switching utterances supply `lang="mixed"` (default);
    for monolingual utterances supply the ISO-639-1 code (e.g. "en", "zh").
    """

    _LOGOGRAPHIC = {"zh", "ja", "ko"}   # scripts using CER

    def __init__(self, lambda_weight: float = 0.5):
        if not LABSE_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SAER. "
                "Install with: pip install sentence-transformers"
            )
        if not (0.0 <= lambda_weight <= 1.0):
            raise ValueError("lambda_weight must be in [0, 1].")
        self.lambda_weight = lambda_weight
        print("Loading LaBSE model for SAER …")
        self._labse = SentenceTransformer("sentence-transformers/LaBSE")
        print("LaBSE loaded.")

    # ------------------------------------------------------------------
    # Form-error helpers (reuse Levenshtein from the main evaluator)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise_wer(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _tokenise_cer(text: str) -> List[str]:
        return [c for c in text.lower() if not c.isspace()]

    @staticmethod
    def _levenshtein(a: List[str], b: List[str]) -> int:
        la, lb = len(a), len(b)
        dp = np.arange(lb + 1, dtype=np.int32)
        for i in range(1, la + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, lb + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return int(dp[lb])

    def _form_error(self, reference: str, hypothesis: str, lang: str) -> float:
        """Returns F_form ∈ [0, ∞) — WER or CER depending on script."""
        if lang in self._LOGOGRAPHIC:
            tokens_r = self._tokenise_cer(reference)
            tokens_h = self._tokenise_cer(hypothesis)
        else:
            tokens_r = self._tokenise_wer(reference)
            tokens_h = self._tokenise_wer(hypothesis)

        if not tokens_r:
            return 0.0 if not tokens_h else 1.0
        edits = self._levenshtein(tokens_r, tokens_h)
        return edits / len(tokens_r)

    def _semantic_dissimilarity(self, reference: str, hypothesis: str) -> float:
        """ε_sem = 1 − cosine(LaBSE(ref), LaBSE(hyp))  ∈ [0, 1]."""
        embs = self._labse.encode([reference, hypothesis],
                                  convert_to_tensor=True, normalize_embeddings=True)
        cosine = float(torch.dot(embs[0], embs[1]).item())
        return float(np.clip(1.0 - cosine, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, reference: str, hypothesis: str,
              lang: str = "mixed") -> Dict[str, float]:
        """
        Compute SAER and its components.

        Args:
            reference:  Ground-truth transcript (may contain multiple languages).
            hypothesis: ASR hypothesis.
            lang:       Language code or "mixed" for code-switching.
                        Logographic codes ("zh", "ja", "ko") → CER form error.
                        Everything else (including "mixed") → WER form error.

        Returns a dict with keys: saer, f_form, epsilon_sem.
        """
        # For code-switching: fall back to WER (alphabetic) as a safe default
        effective_lang = lang if lang != "mixed" else "en"

        f_form = self._form_error(reference, hypothesis, effective_lang)
        eps_sem = self._semantic_dissimilarity(reference, hypothesis)

        lam = self.lambda_weight
        saer = lam * f_form + (1 - lam) * eps_sem

        return {
            "saer": float(np.clip(saer, 0.0, None)),
            "f_form": f_form,
            "epsilon_sem": eps_sem,
        }


# ══════════════════════════════════════════════════════════════════════════
#  NEW: Aligned Semantic Distance (ASD)
# ══════════════════════════════════════════════════════════════════════════

class ASDCalculator:
    """
    Aligned Semantic Distance (ASD).

    Instead of comparing sentence-averaged embeddings (which conflate word
    order and meaning), ASD:

      1. Keeps per-token BERT embeddings (no averaging to sentence level).
      2. Uses dynamic programming to find the *optimal alignment* between
         the reference and hypothesis token sequences — the same edit-path
         logic as Levenshtein, but costs are cosine distances rather than
         binary 0/1 substitutions.
      3. For each matched pair along the optimal path, records the cosine
         distance between the two token embeddings.
      4. Returns the mean aligned distance, normalised by the number of
         reference tokens.

    Range: [0, 1]  — lower is better; 0.0 = semantically perfect match.

    This is strictly more fine-grained than sentence-BERT cosine similarity
    because it is sensitive to *which* words were wrong, not just how similar
    the overall meaning vectors are.
    """

    _MODEL_NAME = "bert-base-uncased"

    def __init__(self, bert_calculator: Optional["SeMaScoreCalculator"] = None):
        """
        Optionally reuse an already-loaded SeMaScoreCalculator to avoid
        loading BERT twice.
        """
        if bert_calculator is not None:
            self._tokenizer = bert_calculator._tokenizer
            self._model = bert_calculator._model
        else:
            if not BERT_AVAILABLE:
                raise ImportError("transformers is required for ASD.")
            print("Loading BERT model for ASD …")
            self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
            self._model = AutoModel.from_pretrained(self._MODEL_NAME)
            self._model.eval()
            print("BERT model loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_embeddings(self, words: List[str]) -> torch.Tensor:
        """
        Return contextualised embedding per *word* (averaged over sub-word
        tokens that belong to each word).  Shape: (len(words), hidden).
        """
        if not words:
            return torch.zeros(0, 768)
        text = " ".join(words)
        inputs = self._tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        token_embs = outputs.last_hidden_state[0]   # (seq, hidden)
        word_ids = inputs.word_ids(batch_index=0)
        embs = []
        for wid in range(len(words)):
            mask = [i for i, w in enumerate(word_ids) if w == wid]
            embs.append(token_embs[mask].mean(0) if mask else
                        torch.zeros(token_embs.shape[-1]))
        return torch.stack(embs)   # (len(words), hidden)

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine *distance* ∈ [0, 1]:  1 − cosine_similarity."""
        cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        return float(np.clip(1.0 - cos, 0.0, 1.0))

    # ------------------------------------------------------------------
    # DP alignment with semantic costs
    # ------------------------------------------------------------------

    def _dp_align(self, ref_embs: torch.Tensor, hyp_embs: torch.Tensor
                  ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Find the minimum-cost alignment between ref and hyp token sequences
        using dynamic programming.

        Edit costs:
          - Substitution  : cosine_distance(ref[i], hyp[j])
          - Deletion      : 1.0  (ref word with no hyp counterpart)
          - Insertion     : 0.0  (extra hyp word; does not affect ref coverage)

        Insertions are penalised at 0 in the DP so the alignment focuses on
        how well the reference words are covered.

        Returns (total_cost, list of (ref_idx, hyp_idx) matched pairs).
        """
        lr, lh = len(ref_embs), len(hyp_embs)

        # dp[i][j] = min cost to align ref[:i] with hyp[:j]
        INF = 1e9
        dp = np.full((lr + 1, lh + 1), INF)
        dp[0, 0] = 0.0
        for i in range(1, lr + 1):
            dp[i, 0] = dp[i - 1, 0] + 1.0   # deletions
        for j in range(1, lh + 1):
            dp[0, j] = dp[0, j - 1] + 0.0   # free insertions

        for i in range(1, lr + 1):
            for j in range(1, lh + 1):
                sub_cost = self._cosine_distance(ref_embs[i - 1], hyp_embs[j - 1])
                dp[i, j] = min(
                    dp[i - 1, j - 1] + sub_cost,   # substitution / match
                    dp[i - 1, j] + 1.0,              # deletion
                    dp[i, j - 1] + 0.0,              # insertion
                )

        # Traceback to recover matched pairs
        pairs: List[Tuple[int, int]] = []
        i, j = lr, lh
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub_cost = self._cosine_distance(ref_embs[i - 1], hyp_embs[j - 1])
                if abs(dp[i, j] - (dp[i - 1, j - 1] + sub_cost)) < 1e-9:
                    pairs.append((i - 1, j - 1))
                    i -= 1; j -= 1
                    continue
            if i > 0 and abs(dp[i, j] - (dp[i - 1, j] + 1.0)) < 1e-9:
                i -= 1   # deletion — no pair recorded
            else:
                j -= 1   # insertion — skip

        return float(dp[lr, lh]), list(reversed(pairs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ASD and its components.

        Returns a dict with:
          asd             — mean aligned cosine distance ∈ [0, 1]; lower = better
          asd_similarity  — 1 − asd; higher = better
          num_ref_tokens  — number of reference words evaluated
          num_matched     — number of ref–hyp word pairs on the optimal path
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return {"asd": 0.0, "asd_similarity": 1.0,
                    "num_ref_tokens": 0, "num_matched": 0}

        ref_embs = self._get_token_embeddings(ref_words)
        hyp_embs = self._get_token_embeddings(hyp_words)

        if len(hyp_words) == 0:
            # All ref words deleted → maximum distance
            return {
                "asd": 1.0,
                "asd_similarity": 0.0,
                "num_ref_tokens": len(ref_words),
                "num_matched": 0,
            }

        _, pairs = self._dp_align(ref_embs, hyp_embs)

        if not pairs:
            return {"asd": 1.0, "asd_similarity": 0.0,
                    "num_ref_tokens": len(ref_words), "num_matched": 0}

        # Mean cosine distance over matched pairs, normalised by ref length
        matched_distances = [
            self._cosine_distance(ref_embs[ri], hyp_embs[hi])
            for ri, hi in pairs
        ]
        # Unmatched ref words (deletions) contribute distance = 1.0
        num_deleted = len(ref_words) - len(pairs)
        total_distance = sum(matched_distances) + num_deleted * 1.0
        asd = total_distance / len(ref_words)
        asd = float(np.clip(asd, 0.0, 1.0))

        return {
            "asd": asd,
            "asd_similarity": 1.0 - asd,
            "num_ref_tokens": len(ref_words),
            "num_matched": len(pairs),
        }


# ══════════════════════════════════════════════════════════════════════════
#  Enhanced dataclass — adds semantic metric fields
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EnhancedVoiceMetrics:
    """Enhanced voice evaluation metrics"""
    # Basic metrics
    snr_db: float
    word_count: int
    token_count: int
    transcript_text: str

    # WER metrics (word-level, not character-level)
    wer_score: float
    wer_percentage: float
    word_accuracy: float
    normalized_wer_score: float
    normalized_wer_percentage: float
    normalized_word_accuracy: float
    cer_score: float
    cer_percentage: float
    mer_score: float
    mer_percentage: float
    wip_score: float
    wil_score: float
    ground_truth_text: str

    # ── NEW: semantic metrics ──────────────────────────────────────────
    # SeMaScore ∈ [0, 1] — higher is better
    semascore: float
    # SAER components — lower is better
    saer: float
    saer_f_form: float
    saer_epsilon_sem: float
    saer_lambda: float
    # ASD components
    asd: float                  # mean aligned cosine distance ∈ [0, 1]; lower = better
    asd_similarity: float       # 1 − asd; higher = better
    asd_num_matched: int        # alignment coverage count
    # ──────────────────────────────────────────────────────────────────

    # RTFx metric
    processing_time_seconds: float
    rtfx: float

    # Timing metrics
    average_latency_ms: float
    total_duration_seconds: float
    ai_speaking_time_seconds: float
    user_speaking_time_seconds: float
    talk_ratio: float
    words_per_minute: float

    # Behavioral flags
    user_interrupted_ai: bool
    early_termination: bool

    # Emotion analysis
    dominant_emotion: str
    dominant_emotion_score: float
    all_emotions: Dict[str, float]

    # Speech quality
    speech_quality_score: float

    # Pitch analysis
    pitch_std_hz: float
    monotone_score: float
    pace_std: float
    pace_score: float
    intonation_score: float
    overall_prosody_score: float

    # Raw data
    raw_data: Dict


# ══════════════════════════════════════════════════════════════════════════
#  Main evaluator
# ══════════════════════════════════════════════════════════════════════════

class EnhancedVoiceEvaluator:
    """Enhanced voice evaluation pipeline with WER, Semantic metrics, and RTFx"""

    def __init__(self, audio_path: str, transcript_path: Optional[str] = None,
                 utmos_model_dir: Optional[str] = None,
                 saer_lambda: float = 0.5, saer_lang: str = "mixed"):
        """
        Args:
            audio_path:      Path to audio file (stereo preferred).
            transcript_path: Optional ground-truth transcript file.
            utmos_model_dir: Optional UTMOS model directory.
            saer_lambda:     λ weight for SAER (default 0.5).
            saer_lang:       Language hint for SAER form-error selection.
                             Use ISO-639-1 code or "mixed" for code-switching.
        """
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.utmos_model_dir = utmos_model_dir
        self.saer_lambda = saer_lambda
        self.saer_lang = saer_lang

        print(f"Loading audio from {audio_path}...")
        self.audio, self.sr = librosa.load(audio_path, sr=None, mono=False)
        self.waveform_torch, self.sr_torch = torchaudio.load(audio_path)

        if len(self.audio.shape) == 2:
            self.ai_channel = self.audio[0]
            self.user_channel = self.audio[1]
            self.is_stereo = True
            print("Stereo audio detected - using channel separation")
        else:
            self.ai_channel = self.audio
            self.user_channel = self.audio
            self.is_stereo = False
            print("Mono audio detected - some metrics may be less accurate")

        self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
        print(f"Audio loaded: {self.duration:.2f}s, {self.sr}Hz")

        self.ground_truth = None
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                self.ground_truth = f.read().strip()
            print(f"Ground truth transcript loaded: {len(self.ground_truth)} characters")

        # Core models
        self.transcription_pipeline = None
        self.emotion_model = None
        self.utmos_predictor = None
        self.tokenizer = None
        self.normalizer = BasicTextNormalizer()

        # Semantic metric calculators (initialised lazily if BERT available)
        self._semascore_calc: Optional[SeMaScoreCalculator] = None
        self._saer_calc: Optional[SAERCalculator] = None
        self._asd_calc: Optional[ASDCalculator] = None

        if TRANSFORMERS_AVAILABLE:
            self._init_whisper()

        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                print("Warning: Failed to initialize tiktoken tokenizer")

        if FUNASR_AVAILABLE:
            self._init_emotion_model()

        if utmos_model_dir and os.path.exists(utmos_model_dir):
            self._init_utmos_model(utmos_model_dir)

        # Try to initialise semantic calculators
        self._init_semantic_calculators()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_whisper(self):
        try:
            print("Loading Whisper model...")
            model_id = "openai/whisper-base"
            device = "cpu"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            self.transcription_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
                chunk_length_s=30,
                return_timestamps=True,
            )
            print("Whisper model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load Whisper model: {e}")

    def _init_emotion_model(self):
        try:
            print("Loading emotion2vec model...")
            self.emotion_model = FunASRModel(
                model="iic/emotion2vec_plus_seed", device="cpu", hub="huggingface"
            )
            print("Emotion model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load emotion model: {e}")

    def _init_utmos_model(self, model_dir):
        try:
            print("Loading UTMOS model...")
            os.environ['TORCH_HOME'] = model_dir
            self.utmos_predictor = torch.hub.load(
                model_dir, 'utmos22_strong', source='local', trust_repo=True
            ).cpu().float()
            print("UTMOS model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load UTMOS model: {e}")

    def _init_semantic_calculators(self):
        """Initialise SeMaScore, SAER, and ASD calculators if dependencies exist."""
        if BERT_AVAILABLE:
            try:
                self._semascore_calc = SeMaScoreCalculator()
                # ASD reuses the same BERT instance to save memory
                self._asd_calc = ASDCalculator(bert_calculator=self._semascore_calc)
            except Exception as e:
                print(f"Warning: Could not initialise SeMaScore/ASD calculators: {e}")

        if LABSE_AVAILABLE:
            try:
                self._saer_calc = SAERCalculator(lambda_weight=self.saer_lambda)
            except Exception as e:
                print(f"Warning: Could not initialise SAER calculator: {e}")

    # ------------------------------------------------------------------
    # Core alignment primitive
    # ------------------------------------------------------------------

    @staticmethod
    def _align(ref_tokens: List[str], hyp_tokens: List[str]) -> AlignmentCounts:
        len_r, len_h = len(ref_tokens), len(hyp_tokens)
        d = np.zeros((len_r + 1, len_h + 1), dtype=np.int32)
        for i in range(len_r + 1): d[i][0] = i
        for j in range(len_h + 1): d[0][j] = j

        for i in range(1, len_r + 1):
            for j in range(1, len_h + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j-1], d[i][j-1], d[i-1][j])

        H = S = D = I = 0
        i, j = len_r, len_h
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
                H += 1; i -= 1; j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
                S += 1; i -= 1; j -= 1
            elif j > 0 and d[i][j] == d[i][j-1] + 1:
                I += 1; j -= 1
            else:
                D += 1; i -= 1

        return AlignmentCounts(H=H, S=S, D=D, I=I)

    # ------------------------------------------------------------------
    # WER / MER / WIP / WIL / CER — unchanged from original
    # ------------------------------------------------------------------

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        ac = self._align(ref_words, hyp_words)
        return (ac.S + ac.D + ac.I) / len(ref_words)

    def calculate_mer(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words and not hyp_words:
            return 0.0
        ac = self._align(ref_words, hyp_words)
        total = ac.H + ac.S + ac.D + ac.I
        return (ac.S + ac.D + ac.I) / total if total > 0 else 0.0

    def calculate_wip(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words and not hyp_words:
            return 1.0
        if not ref_words or not hyp_words:
            return 0.0
        ac = self._align(ref_words, hyp_words)
        denom = (ac.H + ac.S + ac.D) * (ac.H + ac.S + ac.I)
        return (ac.H ** 2) / denom if denom > 0 else 0.0

    def calculate_wil(self, reference: str, hypothesis: str) -> float:
        return 1.0 - self.calculate_wip(reference, hypothesis)

    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        ref_chars = [c for c in reference.lower() if c != ' ']
        hyp_chars = [c for c in hypothesis.lower() if c != ' ']
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0
        ac = self._align(ref_chars, hyp_chars)
        return (ac.S + ac.D + ac.I) / len(ref_chars)

    # ------------------------------------------------------------------
    # NEW: semantic metric wrappers
    # ------------------------------------------------------------------

    def calculate_semascore(self, reference: str, hypothesis: str) -> float:
        """Compute SeMaScore ∈ [0, 1]; returns -1 if unavailable."""
        if self._semascore_calc is None:
            return -1.0
        try:
            return self._semascore_calc.score(reference, hypothesis)
        except Exception as e:
            print(f"Warning: SeMaScore failed: {e}")
            return -1.0

    def calculate_saer(self, reference: str, hypothesis: str
                       ) -> Dict[str, float]:
        """Compute SAER; returns sentinel dict if unavailable."""
        if self._saer_calc is None:
            return {"saer": -1.0, "f_form": -1.0, "epsilon_sem": -1.0}
        try:
            return self._saer_calc.score(reference, hypothesis, lang=self.saer_lang)
        except Exception as e:
            print(f"Warning: SAER failed: {e}")
            return {"saer": -1.0, "f_form": -1.0, "epsilon_sem": -1.0}

    def calculate_asd(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ASD; returns sentinel dict if unavailable."""
        if self._asd_calc is None:
            return {"asd": -1.0, "asd_similarity": -1.0,
                    "num_ref_tokens": 0, "num_matched": 0}
        try:
            return self._asd_calc.score(reference, hypothesis)
        except Exception as e:
            print(f"Warning: ASD failed: {e}")
            return {"asd": -1.0, "asd_similarity": -1.0,
                    "num_ref_tokens": 0, "num_matched": 0}

    # ------------------------------------------------------------------
    # All remaining methods identical to original
    # ------------------------------------------------------------------

    def detect_speech_segments(self, audio: np.ndarray,
                                min_silence_duration: float = 0.3
                                ) -> List[Tuple[float, float]]:
        frame_length = int(self.sr * 0.025)
        hop_length = int(self.sr * 0.010)
        energy = librosa.feature.rms(y=audio, frame_length=frame_length,
                                     hop_length=hop_length)[0]
        threshold = np.percentile(energy, 30)
        speech_frames = energy > threshold
        segments, in_speech, start_frame = [], False, 0
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i; in_speech = True
            elif not is_speech and in_speech:
                s = librosa.frames_to_time(start_frame, sr=self.sr, hop_length=hop_length)
                e = librosa.frames_to_time(i, sr=self.sr, hop_length=hop_length)
                if e - s > 0.1:
                    segments.append((s, e))
                in_speech = False
        return segments

    def calculate_snr(self) -> float:
        ai_segments = self.detect_speech_segments(self.ai_channel)
        if not ai_segments:
            return 0.0
        ai_speech = []
        for start, end in ai_segments:
            ai_speech.extend(self.ai_channel[int(start*self.sr):int(end*self.sr)])
        ai_speech = np.array(ai_speech)
        noise_samples = []
        for i in range(len(ai_segments) - 1):
            ss = int(ai_segments[i][1] * self.sr)
            se = int(ai_segments[i+1][0] * self.sr)
            if se - ss > self.sr * 0.1:
                noise_samples.extend(self.ai_channel[ss:se])
        if not noise_samples:
            return 40.0
        noise = np.array(noise_samples)
        sp = np.mean(ai_speech ** 2)
        np_ = np.mean(noise ** 2)
        return max(0, 10 * np.log10(sp / np_)) if np_ > 0 else 40.0

    def transcribe_audio(self) -> Tuple[str, float]:
        if not self.transcription_pipeline:
            return "", 0.0
        try:
            print("Transcribing audio...")
            t0 = time.time()
            result = self.transcription_pipeline(self.audio_path)
            processing_time = time.time() - t0
            if isinstance(result, dict) and "text" in result:
                transcript = result["text"].strip()
            elif isinstance(result, dict) and "chunks" in result:
                transcript = " ".join(c["text"] for c in result["chunks"]).strip()
            else:
                transcript = str(result).strip()
            return transcript, processing_time
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return "", 0.0

    def analyze_emotion(self) -> Tuple[str, float, Dict[str, float]]:
        if not self.emotion_model:
            return "unknown", 0.0, {}
        try:
            print("Analyzing emotion...")
            waveform = self.waveform_torch
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            max_samples = 30 * self.sr_torch
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            if self.sr_torch != 16000:
                waveform = torchaudio.transforms.Resample(self.sr_torch, 16000)(waveform)
            rec = self.emotion_model.generate(
                waveform, output_dir=None, granularity="utterance",
                extract_embedding=False
            )[0]
            idx = rec['scores'].index(max(rec['scores']))
            lbl = rec['labels'][idx]
            dominant = lbl.split('/')[-1] if '/' in lbl else lbl
            all_em = {(l.split('/')[-1] if '/' in l else l): float(s)
                      for l, s in zip(rec['labels'], rec['scores'])}
            return dominant, float(rec['scores'][idx]), all_em
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return "unknown", 0.0, {}

    def calculate_speech_quality(self) -> float:
        if not self.utmos_predictor:
            return 0.0
        try:
            print("Calculating speech quality...")
            wave, sr = librosa.load(self.audio_path, sr=None)
            wave = wave.astype(np.float32)
            score = self.utmos_predictor(torch.from_numpy(wave).unsqueeze(0).cpu(), sr)
            return float(score.item())
        except Exception as e:
            print(f"Error calculating speech quality: {e}")
            return 0.0

    def analyze_pitch(self) -> Dict[str, float]:
        try:
            print("Analyzing pitch and prosody...")
            y, sr = self.ai_channel, self.sr

            def clamp01(x): return max(0.0, min(1.0, x))
            def norm(v, lo, hi): return clamp01((v - lo) / (hi - lo)) if hi != lo else 0.0

            pitches, mags = librosa.piptrack(y=y, sr=sr)
            voiced = pitches[mags > np.median(mags)]
            pitch_std = float(np.std(voiced)) if len(voiced) > 0 else 0.0
            monotone_score = norm(pitch_std, 20, 120)

            onsets = librosa.onset.onset_detect(y=y, sr=sr)
            pace_std = float(np.std(1 / np.diff(onsets / sr))) if len(onsets) > 1 else 0.0
            pace_score = norm(pace_std, 0.2, 1.5)

            f0 = librosa.yin(y, fmin=75, fmax=300, sr=sr)
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                q = max(1, len(f0) // 5)
                delta = float(np.mean(f0[-q:]) - np.mean(f0[:q]))
            else:
                delta = 0.0
            intonation_score = norm(delta, -10, 20)

            overall = 0.4 * monotone_score + 0.3 * pace_score + 0.3 * intonation_score
            return {
                'pitch_std_hz': round(pitch_std, 2),
                'monotone_score': round(monotone_score, 3),
                'pace_std': round(pace_std, 3),
                'pace_score': round(pace_score, 3),
                'intonation_score': round(intonation_score, 3),
                'overall_prosody_score': round(overall, 3),
            }
        except Exception as e:
            print(f"Error analyzing pitch: {e}")
            return {k: 0.0 for k in ('pitch_std_hz', 'monotone_score', 'pace_std',
                                     'pace_score', 'intonation_score', 'overall_prosody_score')}

    def count_words(self, text: str) -> int:
        return len(text.split())

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return int(len(text.split()) * 1.3)
        try:
            return len(self.tokenizer.encode(text))
        except:
            return int(len(text.split()) * 1.3)

    def calculate_latency(self) -> float:
        if not self.is_stereo:
            return 0.0
        user_segs = self.detect_speech_segments(self.user_channel)
        ai_segs = self.detect_speech_segments(self.ai_channel)
        latencies = []
        for _, ue in user_segs:
            nxt = [s for s in ai_segs if s[0] > ue]
            if nxt:
                lat = (nxt[0][0] - ue) * 1000
                if 0 < lat < 5000:
                    latencies.append(lat)
        return float(np.mean(latencies)) if latencies else 0.0

    def calculate_speaking_times(self) -> Tuple[float, float]:
        ai_segs = self.detect_speech_segments(self.ai_channel)
        ai_time = sum(e - s for s, e in ai_segs)
        if self.is_stereo:
            user_segs = self.detect_speech_segments(self.user_channel)
            user_time = sum(e - s for s, e in user_segs)
        else:
            user_time = self.duration - ai_time
        return float(ai_time), float(user_time)

    def detect_interruptions(self) -> bool:
        if not self.is_stereo:
            return False
        ai_segs = self.detect_speech_segments(self.ai_channel)
        user_segs = self.detect_speech_segments(self.user_channel)
        for as_, ae in ai_segs:
            for us, ue in user_segs:
                if as_ < us < ae and min(ae, ue) - us > 0.2:
                    return True
        return False

    def detect_early_termination(self) -> bool:
        if not self.is_stereo:
            return False
        user_segs = self.detect_speech_segments(self.user_channel)
        if len(user_segs) < 2:
            return False
        ls = user_segs[-1]
        ai_segs = self.detect_speech_segments(self.ai_channel)
        ai_at_end = any(s < ls[1] < e for s, e in ai_segs)
        return (ls[1] - ls[0] < 1.0 and
                self.duration - ls[1] < 0.5 and
                not ai_at_end)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def evaluate_all(self) -> EnhancedVoiceMetrics:
        print("\n" + "="*70)
        print("ENHANCED VOICE EVALUATION PIPELINE")
        print("="*70)

        print("\n[1/17] Calculating SNR...")
        snr = self.calculate_snr()
        print(f"   ✓ SNR: {snr:.2f} dB")

        print("\n[2/17] Transcribing audio...")
        transcript, processing_time = self.transcribe_audio()
        print(f"   ✓ Transcript: {len(transcript)} characters")

        print("\n[3/17] Calculating RTFx...")
        rtfx = self.duration / processing_time if processing_time > 0 else 0.0
        print(f"   ✓ RTFx: {rtfx:.2f}x")

        print("\n[4/17] Counting words...")
        word_count = self.count_words(transcript)
        print(f"   ✓ Word count: {word_count}")

        print("\n[5/17] Counting tokens...")
        token_count = self.count_tokens(transcript)
        print(f"   ✓ Token count: {token_count}")

        print("\n[6/17] Calculating WER metrics...")
        if self.ground_truth:
            wer = self.calculate_wer(self.ground_truth, transcript)
            word_accuracy = max(0.0, 1.0 - wer)
            norm_ref = self.normalizer(self.ground_truth)
            norm_hyp = self.normalizer(transcript)
            normalized_wer = self.calculate_wer(norm_ref, norm_hyp) if norm_ref else 0.0
            normalized_word_accuracy = max(0.0, 1.0 - normalized_wer)
            cer = self.calculate_cer(self.ground_truth, transcript)
            mer = self.calculate_mer(self.ground_truth, transcript)
            wip = self.calculate_wip(self.ground_truth, transcript)
            wil = self.calculate_wil(self.ground_truth, transcript)
            print(f"   ✓ WER: {wer:.4f} | Word Accuracy: {word_accuracy:.4f}")
            print(f"   ✓ Norm WER: {normalized_wer:.4f} | CER: {cer:.4f}")
            print(f"   ✓ MER: {mer:.4f} | WIP: {wip:.4f} | WIL: {wil:.4f}")
        else:
            wer = word_accuracy = normalized_wer = normalized_word_accuracy = 0.0
            cer = mer = wip = wil = 0.0
            print("   ⚠ No ground truth — all error metrics = 0")

        # ── NEW: Semantic metrics ─────────────────────────────────────
        print("\n[7/17] Calculating SeMaScore...")
        if self.ground_truth and self._semascore_calc:
            semascore = self.calculate_semascore(self.ground_truth, transcript)
            print(f"   ✓ SeMaScore: {semascore:.4f}")
        else:
            semascore = -1.0
            print("   ⚠ SeMaScore unavailable (need ground truth + BERT)")

        print("\n[8/17] Calculating SAER...")
        if self.ground_truth and self._saer_calc:
            saer_result = self.calculate_saer(self.ground_truth, transcript)
            print(f"   ✓ SAER: {saer_result['saer']:.4f}  "
                  f"(F_form={saer_result['f_form']:.4f}, "
                  f"ε_sem={saer_result['epsilon_sem']:.4f})")
        else:
            saer_result = {"saer": -1.0, "f_form": -1.0, "epsilon_sem": -1.0}
            print("   ⚠ SAER unavailable (need ground truth + LaBSE)")

        print("\n[9/17] Calculating ASD...")
        if self.ground_truth and self._asd_calc:
            asd_result = self.calculate_asd(self.ground_truth, transcript)
            print(f"   ✓ ASD: {asd_result['asd']:.4f}  "
                  f"(similarity={asd_result['asd_similarity']:.4f}, "
                  f"matched={asd_result['num_matched']}/{asd_result['num_ref_tokens']})")
        else:
            asd_result = {"asd": -1.0, "asd_similarity": -1.0,
                          "num_ref_tokens": 0, "num_matched": 0}
            print("   ⚠ ASD unavailable (need ground truth + BERT)")
        # ─────────────────────────────────────────────────────────────

        print("\n[10/17] Calculating latency...")
        avg_latency = self.calculate_latency()
        print(f"   ✓ Average latency: {avg_latency:.2f} ms")

        print("\n[11/17] Calculating speaking times...")
        ai_time, user_time = self.calculate_speaking_times()
        talk_ratio = ai_time / user_time if user_time > 0 else 0.0
        print(f"   ✓ AI: {ai_time:.2f}s | User: {user_time:.2f}s | Ratio: {talk_ratio:.2f}")

        print("\n[12/17] Detecting interruptions...")
        interruptions = self.detect_interruptions()
        print(f"   ✓ Interrupted: {interruptions}")

        print("\n[13/17] Checking early termination...")
        early_term = self.detect_early_termination()
        print(f"   ✓ Early termination: {early_term}")

        print("\n[14/17] Calculating WPM...")
        wpm = (word_count / ai_time) * 60 if ai_time > 0 else 0.0
        print(f"   ✓ WPM: {wpm:.2f}")

        print("\n[15/17] Analyzing emotion...")
        dominant_emotion, emotion_score, all_emotions = self.analyze_emotion()
        print(f"   ✓ {dominant_emotion} ({emotion_score*100:.2f}%)")

        print("\n[16/17] Calculating speech quality...")
        quality_score = self.calculate_speech_quality()
        print(f"   ✓ Quality: {quality_score:.4f}")

        print("\n[17/17] Analyzing pitch and prosody...")
        pitch_results = self.analyze_pitch()
        print(f"   ✓ Pitch std: {pitch_results['pitch_std_hz']:.2f} Hz | "
              f"Prosody: {pitch_results['overall_prosody_score']:.3f}")

        raw_data = {
            "audio_path": self.audio_path,
            "transcript_path": self.transcript_path,
            "is_stereo": self.is_stereo,
            "sample_rate": int(self.sr),
        }

        return EnhancedVoiceMetrics(
            snr_db=snr,
            word_count=word_count,
            token_count=token_count,
            transcript_text=transcript,
            wer_score=wer,
            wer_percentage=wer * 100,
            word_accuracy=word_accuracy,
            normalized_wer_score=normalized_wer,
            normalized_wer_percentage=normalized_wer * 100,
            normalized_word_accuracy=normalized_word_accuracy,
            cer_score=cer,
            cer_percentage=cer * 100,
            mer_score=mer,
            mer_percentage=mer * 100,
            wip_score=wip,
            wil_score=wil,
            ground_truth_text=self.ground_truth or "",
            # semantic
            semascore=semascore,
            saer=saer_result["saer"],
            saer_f_form=saer_result["f_form"],
            saer_epsilon_sem=saer_result["epsilon_sem"],
            saer_lambda=self.saer_lambda,
            asd=asd_result["asd"],
            asd_similarity=asd_result["asd_similarity"],
            asd_num_matched=asd_result["num_matched"],
            # performance
            processing_time_seconds=processing_time,
            rtfx=rtfx,
            average_latency_ms=avg_latency,
            total_duration_seconds=float(self.duration),
            ai_speaking_time_seconds=ai_time,
            user_speaking_time_seconds=user_time,
            talk_ratio=talk_ratio,
            words_per_minute=wpm,
            user_interrupted_ai=interruptions,
            early_termination=early_term,
            dominant_emotion=dominant_emotion,
            dominant_emotion_score=emotion_score,
            all_emotions=all_emotions,
            speech_quality_score=quality_score,
            pitch_std_hz=pitch_results['pitch_std_hz'],
            monotone_score=pitch_results['monotone_score'],
            pace_std=pitch_results['pace_std'],
            pace_score=pitch_results['pace_score'],
            intonation_score=pitch_results['intonation_score'],
            overall_prosody_score=pitch_results['overall_prosody_score'],
            raw_data=raw_data,
        )


# ══════════════════════════════════════════════════════════════════════════
#  Report formatter — updated with semantic section
# ══════════════════════════════════════════════════════════════════════════

def _fmt(val: float, decimals: int = 4, unavailable_sentinel: float = -1.0) -> str:
    if abs(val - unavailable_sentinel) < 1e-9:
        return "N/A (missing dependency)"
    return f"{val:.{decimals}f}"


def format_report(metrics: EnhancedVoiceMetrics) -> str:
    if metrics.rtfx > 1.0:
        rtfx_status = f"✓ {metrics.rtfx:.2f}x faster than real-time"
    elif metrics.rtfx == 1.0:
        rtfx_status = "= Real-time processing"
    elif metrics.rtfx > 0:
        rtfx_status = f"⚠ {1/metrics.rtfx:.2f}x slower than real-time"
    else:
        rtfx_status = "N/A"

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           ENHANCED VOICE EVALUATION REPORT                        ║
╚══════════════════════════════════════════════════════════════════╝

📊 AUDIO QUALITY
──────────────────────────────────────────────────────────────────
  Signal to Noise Ratio:    {metrics.snr_db:.2f} dB
  Speech Quality Score:     {metrics.speech_quality_score:.4f}

⚡ TRANSCRIPTION PERFORMANCE
──────────────────────────────────────────────────────────────────
  Processing Time:          {metrics.processing_time_seconds:.2f}s
  Audio Duration:           {metrics.total_duration_seconds:.2f}s
  RTFx (Speed Factor):      {metrics.rtfx:.2f}x
  Status:                   {rtfx_status}

📝 TRANSCRIPTION ACCURACY  (string-level)
──────────────────────────────────────────────────────────────────
  Word Error Rate (WER):         {metrics.wer_score:.4f} ({metrics.wer_percentage:.2f}%)
  Word Accuracy:                 {metrics.word_accuracy:.4f} ({metrics.word_accuracy*100:.2f}%)

  Normalized WER:                {metrics.normalized_wer_score:.4f} ({metrics.normalized_wer_percentage:.2f}%)
  Normalized Word Accuracy:      {metrics.normalized_word_accuracy:.4f} ({metrics.normalized_word_accuracy*100:.2f}%)

  Character Error Rate (CER):    {metrics.cer_score:.4f} ({metrics.cer_percentage:.2f}%)
  Match Error Rate (MER):        {metrics.mer_score:.4f} ({metrics.mer_percentage:.2f}%)

  Word Info Preserved (WIP):     {metrics.wip_score:.4f}  ↑ higher = better
  Word Info Lost (WIL):          {metrics.wil_score:.4f}  ↓ lower  = better

  Word Count:                    {metrics.word_count}
  Token Count:                   {metrics.token_count}

🧠 SEMANTIC ACCURACY  (meaning-level)
──────────────────────────────────────────────────────────────────
  SeMaScore:                     {_fmt(metrics.semascore)}  ↑ higher = better
    (BERT cosine + MER penalty + importance weighting)

  SAER:                          {_fmt(metrics.saer)}  ↓ lower = better
    λ weight:                    {metrics.saer_lambda}
    F_form (WER/CER):            {_fmt(metrics.saer_f_form)}
    ε_sem  (LaBSE dissimilarity):{_fmt(metrics.saer_epsilon_sem)}

  ASD (Aligned Semantic Distance): {_fmt(metrics.asd)}  ↓ lower = better
    ASD Similarity  (1 − ASD):   {_fmt(metrics.asd_similarity)}  ↑ higher = better
    Matched tokens:               {metrics.asd_num_matched}

⏱️  TIMING METRICS
──────────────────────────────────────────────────────────────────
  Total Duration:           {metrics.total_duration_seconds:.2f}s
  AI Speaking Time:         {metrics.ai_speaking_time_seconds:.2f}s
  User Speaking Time:       {metrics.user_speaking_time_seconds:.2f}s
  Talk Ratio (AI/User):     {metrics.talk_ratio:.2f}
  Average Latency:          {metrics.average_latency_ms:.2f} ms
  Words Per Minute:         {metrics.words_per_minute:.2f} WPM

🎭 EMOTION ANALYSIS
──────────────────────────────────────────────────────────────────
  Dominant Emotion:         {metrics.dominant_emotion}
  Confidence:               {metrics.dominant_emotion_score*100:.2f}%

  Top Emotions:"""

    for emotion, score in sorted(metrics.all_emotions.items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
        if score > 0.01:
            report += f"\n    {emotion:15s}: {score*100:.2f}%"

    report += f"""

🎵 PITCH & PROSODY ANALYSIS
──────────────────────────────────────────────────────────────────
  Pitch Std Dev:            {metrics.pitch_std_hz:.2f} Hz
  Monotone Score:           {metrics.monotone_score:.3f}
  Pace Std Dev:             {metrics.pace_std:.3f}
  Pace Score:               {metrics.pace_score:.3f}
  Intonation Score:         {metrics.intonation_score:.3f}
  Overall Prosody:          {metrics.overall_prosody_score:.3f}

⚠️  BEHAVIORAL FLAGS
──────────────────────────────────────────────────────────────────
  User Interrupted AI:      {'Yes ⚠️' if metrics.user_interrupted_ai else 'No ✓'}
  Early Termination:        {'Yes ⚠️' if metrics.early_termination else 'No ✓'}

📄 TRANSCRIPT
──────────────────────────────────────────────────────────────────
{metrics.transcript_text[:500]}{'...' if len(metrics.transcript_text) > 500 else ''}
"""

    if metrics.ground_truth_text:
        report += f"""
📌 GROUND TRUTH
──────────────────────────────────────────────────────────────────
{metrics.ground_truth_text[:500]}{'...' if len(metrics.ground_truth_text) > 500 else ''}
"""

    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 METRICS INTERPRETATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRING-LEVEL METRICS
──────────────────────────────────────────────────────────────────
WER / CER / MER  — lower is better (0 = perfect)
  WER : (S+D+I) / N_ref  — standard word-level accuracy
  CER : character-level analogue of WER
  MER : (S+D+I) / (H+S+D+I) — bounded [0,1], D/I symmetric

WIP / WIL — information-theoretic word overlap
  WIP = H² / ((H+S+D)(H+S+I)) — higher is better
  WIL = 1 − WIP                — lower  is better

RTFx  — RTFx > 1.0 = faster than real-time ✓

SEMANTIC METRICS  (require ground truth + optional dependencies)
──────────────────────────────────────────────────────────────────
SeMaScore ∈ [0,1]  — higher = better meaning preservation
  Phase 1 : word alignment via char-level Levenshtein
  Phase 2 : BERT cosine similarity per segment, with MER penalty
            that corrects BERTScore's failure mode
            ("Smoking" vs "Something": BERT=0.98 → SeMaScore=0.30)
  Phase 3 : importance-weighting by sentence-level context
  Phase 4 : weighted aggregation
  Requires: transformers (BERT)

SAER  — lower is better; SAER = λ×F_form + (1−λ)×ε_sem
  F_form  : WER for alphabetic scripts, CER for logographic (CJK)
  ε_sem   : 1 − LaBSE cosine — language-agnostic meaning distance
  λ       : tunable form/semantic balance (default 0.5)
  Designed for code-switching evaluation; set saer_lang accordingly
  Requires: sentence-transformers (LaBSE)

ASD (Aligned Semantic Distance) ∈ [0,1]  — lower = better
  Unlike sentence-BERT averaging, ASD keeps token embeddings separate:
  1. Token embeddings  : per-word BERT contextual embeddings (no pooling)
  2. Optimal alignment : DP minimises total cosine distance across paths
                         (substitution = cosine dist; deletion = 1.0;
                          insertion = 0.0 so reference coverage drives score)
  3. Aligned distance  : mean cosine dist of matched pairs +
                         deletion penalty / N_ref
  ASD Similarity = 1 − ASD (higher = better)
  Sensitive to WHICH words were wrong, not just overall sentence similarity
  Requires: transformers (BERT)
"""

    return report


# ══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python enhanced_voice_eval.py <audio_file> "
              "[transcript_file] [utmos_model_dir] [saer_lambda] [saer_lang]")
        print("\nExamples:")
        print("  python enhanced_voice_eval.py audio.wav")
        print("  python enhanced_voice_eval.py audio.wav transcript.txt")
        print("  python enhanced_voice_eval.py audio.wav transcript.txt '' 0.5 en")
        print("  python enhanced_voice_eval.py audio.wav transcript.txt '' 0.5 zh")
        sys.exit(1)

    audio_path = sys.argv[1]
    transcript_path = sys.argv[2] if len(sys.argv) > 2 else None
    utmos_dir      = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    saer_lambda    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    saer_lang      = sys.argv[5] if len(sys.argv) > 5 else "mixed"

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}"); sys.exit(1)
    if transcript_path and not os.path.exists(transcript_path):
        print(f"Warning: Transcript file not found: {transcript_path}")
        transcript_path = None

    evaluator = EnhancedVoiceEvaluator(
        audio_path=audio_path,
        transcript_path=transcript_path,
        utmos_model_dir=utmos_dir,
        saer_lambda=saer_lambda,
        saer_lang=saer_lang,
    )
    metrics = evaluator.evaluate_all()
    print(format_report(metrics))

    output_path = Path(audio_path).stem + "_enhanced_eval.json"

    def to_serial(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_):       return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serial(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serial(v) for v in obj]
        return obj

    metrics_dict = to_serial(asdict(metrics))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()