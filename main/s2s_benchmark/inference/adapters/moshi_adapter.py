"""
Moshi full-duplex Speech LM adapter.

Moshi (Kyutai, 2024): 7B-param full-duplex model that listens and speaks
simultaneously. Uses dual-stream Helium backbone + Mimi neural audio codec.
~200ms natural latency. License: CC-BY 4.0.

HuggingFace: kyutai/moshika-pytorch-bf16  (female voice)
             kyutai/moshiko-pytorch-bf16  (male voice)

Requirements:
    pip install moshi sentencepiece huggingface-hub

Hardware:
    GPU strongly recommended (~16GB VRAM for bfloat16).
    CPU works but is ~10-30x slower than real-time.

Config keys:
    hf_repo       (str): HuggingFace repo (default "kyutai/moshika-pytorch-bf16")
    device        (str): "cuda" | "mps" | "cpu"  (default "cuda" if available)
    max_tokens    (int): max generated audio tokens (default 500, ~10s at 12.5 tok/s)
    sample_rate   (int): output sample rate (Moshi always outputs 24000)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from inference.adapters.base import BaseS2SAdapter, S2SResult

_MODELS: dict = {}


def _get_device(preferred: str) -> str:
    try:
        import torch
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_moshi(hf_repo: str, device: str):
    if hf_repo in _MODELS:
        return _MODELS[hf_repo]

    print(f"  [moshi] Loading {hf_repo} on {device} …")
    import torch
    from moshi.models import loaders  # type: ignore

    # Load Mimi (audio codec) and Moshi LM
    mimi_weight, moshi_weight = loaders.get_moshi_lm(hf_repo=hf_repo)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    moshi_lm = loaders.get_lm(moshi_weight, device=device)

    _MODELS[hf_repo] = (mimi, moshi_lm, device)
    print(f"  [moshi] Loaded.")
    return mimi, moshi_lm, device


class MoshiAdapter(BaseS2SAdapter):
    """Moshi full-duplex Speech LM adapter.

    Processes audio input in offline batch mode: feeds the entire utterance
    as user audio, generates the system response stream, and saves the output.
    """

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.hf_repo = config.get("hf_repo", "kyutai/moshika-pytorch-bf16")
        preferred_device = config.get("device", "cuda")
        self.device = _get_device(preferred_device)
        self.max_tokens = int(config.get("max_tokens", 500))   # ~10s at 12.5 tok/s
        self.output_sr = 24_000   # Moshi always outputs 24kHz via Mimi

    def process(
        self,
        audio_in_path: str,
        reference_text: str,
        utterance_id: str,
        output_dir: str,
    ) -> S2SResult:
        import torch
        import soundfile as sf  # type: ignore
        import librosa           # type: ignore

        t_start = time.perf_counter()

        try:
            mimi, moshi_lm, device = _load_moshi(self.hf_repo, self.device)

            # ── Load and resample input audio to 24kHz (Mimi's rate) ─────────
            audio_np, sr = sf.read(audio_in_path, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=-1)
            if sr != self.output_sr:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.output_sr)

            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)

            # ── Encode user audio into Mimi tokens ────────────────────────────
            with torch.no_grad():
                user_codes = mimi.encode(audio_tensor)   # (1, K, T_enc)

            # ── Run Moshi LM in offline inference mode ────────────────────────
            from moshi.models import LMGen  # type: ignore

            lm_gen = LMGen(
                moshi_lm,
                temp=0.8,
                temp_text=0.7,
                do_sample=True,
            )

            # Feed user codes and generate system response
            out_audio_codes = []
            t_first_token = None

            with torch.no_grad():
                lm_gen.reset_streaming()
                n_user_steps = user_codes.shape[-1]

                # Feed user audio tokens step-by-step
                for step in range(n_user_steps):
                    user_step = user_codes[:, :, step : step + 1]   # (1, K, 1)
                    out_codes = lm_gen.step(user_step)
                    if out_codes is not None:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        out_audio_codes.append(out_codes)

                # Continue generating response after user audio ends
                silence = torch.zeros(1, user_codes.shape[1], 1, device=device, dtype=user_codes.dtype)
                for _ in range(self.max_tokens):
                    out_codes = lm_gen.step(silence)
                    if out_codes is None:
                        break
                    out_audio_codes.append(out_codes)

            if not out_audio_codes:
                return S2SResult(
                    audio_out_path="",
                    model_name=self.model_name,
                    utterance_id=utterance_id,
                    e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                    error="moshi_no_output",
                )

            # ── Decode Mimi tokens back to audio ──────────────────────────────
            all_codes = torch.cat(out_audio_codes, dim=-1)   # (1, K, T_out)
            with torch.no_grad():
                audio_out = mimi.decode(all_codes)            # (1, 1, T_audio)
            audio_np_out = audio_out[0, 0].cpu().float().numpy()

            # ── Save ──────────────────────────────────────────────────────────
            out_path = self._make_output_path(output_dir, utterance_id)
            sf.write(out_path, audio_np_out, self.output_sr, subtype="PCM_16")

            e2e_latency_ms = (time.perf_counter() - t_start) * 1_000
            ttfb_ms = ((t_first_token or t_start) - t_start) * 1_000
            duration_s = len(audio_np_out) / self.output_sr
            rtf = (e2e_latency_ms / 1_000) / max(duration_s, 1e-6)

            return S2SResult(
                audio_out_path=out_path,
                sample_rate=self.output_sr,
                asr_transcript=None,    # Moshi doesn't output text by default
                ttfb_ms=ttfb_ms,
                e2e_latency_ms=e2e_latency_ms,
                rtf=rtf,
                model_name=self.model_name,
                utterance_id=utterance_id,
            )

        except ImportError as exc:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                error=f"moshi not installed: pip install moshi — {exc}",
            )
        except Exception as exc:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                error=str(exc),
            )

    def cleanup(self) -> None:
        pass
