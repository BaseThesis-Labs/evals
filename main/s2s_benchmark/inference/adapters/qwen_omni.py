"""
Qwen2.5-Omni end-to-end Speech-to-Speech adapter.

Architecture: Thinker-Talker with TMRoPE for time-aligned multimodal processing.
Supports 3B and 7B sizes. Ranked #1 open-source on MMSU + MMAU benchmarks.

HuggingFace: Qwen/Qwen2.5-Omni-7B  or  Qwen/Qwen2.5-Omni-3B

Requirements:
    pip install transformers>=4.51 accelerate soundfile
    pip install qwen-omni-utils   # for process_mm_info helper

Hardware:
    7B: ~16GB VRAM (bfloat16) or ~8GB (4-bit), or Apple MPS (slow)
    3B: ~6GB VRAM or ~4GB (4-bit)

Config keys:
    model_id      (str): HuggingFace model ID (default "Qwen/Qwen2.5-Omni-7B")
    device        (str): "auto" | "cuda" | "mps" | "cpu"  (default "auto")
    dtype         (str): "bfloat16" | "float16" | "float32"  (default "bfloat16")
    load_in_4bit  (bool): enable 4-bit quantization via bitsandbytes (default false)
    system_prompt (str): override default system message
    max_new_tokens(int): max response tokens (default 512)
    sample_rate   (int): output sample rate Hz (default 24000)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from inference.adapters.base import BaseS2SAdapter, S2SResult

_DEFAULT_SYSTEM = (
    "You are a helpful voice assistant. "
    "Respond naturally and concisely in 1-3 sentences as if speaking aloud."
)

# Model-level cache (shared across calls)
_MODELS: dict = {}


def _load_model(model_id: str, device: str, dtype_str: str, load_in_4bit: bool):
    key = (model_id, device, dtype_str, load_in_4bit)
    if key in _MODELS:
        return _MODELS[key]

    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"  [qwen_omni] Loading {model_id} (dtype={dtype_str}, device={device}) …")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    load_kwargs: dict = {"torch_dtype": torch_dtype, "device_map": device}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # type: ignore
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs.pop("torch_dtype", None)  # incompatible with 4-bit

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()

    _MODELS[key] = (processor, model)
    print(f"  [qwen_omni] Model loaded.")
    return processor, model


class Qwen25OmniAdapter(BaseS2SAdapter):
    """Qwen2.5-Omni end-to-end speech-to-speech adapter."""

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.model_id = config.get("model_id", "Qwen/Qwen2.5-Omni-7B")
        self.device = config.get("device", "auto")
        self.dtype_str = config.get("dtype", "bfloat16")
        self.load_in_4bit = bool(config.get("load_in_4bit", False))
        self.system_prompt = config.get("system_prompt", _DEFAULT_SYSTEM)
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.output_sr = int(config.get("sample_rate", 24000))

    def process(
        self,
        audio_in_path: str,
        reference_text: str,
        utterance_id: str,
        output_dir: str,
    ) -> S2SResult:
        import torch
        import soundfile as sf  # type: ignore

        t_start = time.perf_counter()

        try:
            processor, model = _load_model(
                self.model_id, self.device, self.dtype_str, self.load_in_4bit
            )

            # Build conversation with audio input
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio_url": audio_in_path}],
                },
            ]

            # Apply chat template
            text_input = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process multimodal inputs
            try:
                from qwen_omni_utils import process_mm_info  # type: ignore
                audios, images, videos = process_mm_info(
                    conversation, use_audio_in_video=False
                )
                inputs = processor(
                    text=[text_input],
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=16_000,
                )
            except ImportError:
                # Fallback: load audio manually and pass directly
                import librosa  # type: ignore
                audio_array, sr = sf.read(audio_in_path, dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=-1)
                if sr != 16_000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16_000)
                inputs = processor(
                    text=[text_input],
                    audio=[audio_array],
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=16_000,
                )

            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            t_first_token = None

            # Generate response (text + audio)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_audio=True,
                )

            t_first_token = time.perf_counter()

            # Decode outputs
            # output is either a tuple (text_ids, audio_waveform) or a GenerateOutput object
            if isinstance(output, tuple) and len(output) == 2:
                text_ids, audio_data = output
            elif hasattr(output, "sequences") and hasattr(output, "audio_values"):
                text_ids = output.sequences
                audio_data = output.audio_values
            else:
                text_ids = output
                audio_data = None

            # Decode text
            input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
            new_ids = text_ids[:, input_len:]
            asr_response = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip()

            # Save audio output
            out_path = self._make_output_path(output_dir, utterance_id)
            if audio_data is not None:
                if hasattr(audio_data, "cpu"):
                    audio_np = audio_data[0].cpu().float().numpy()
                else:
                    audio_np = np.array(audio_data[0], dtype=np.float32)
                # Normalize if needed
                if audio_np.max() > 1.0:
                    audio_np = audio_np / 32768.0
                sf.write(out_path, audio_np, self.output_sr, subtype="PCM_16")
                output_sr = self.output_sr
            else:
                # No audio output — generate via CPU TTS fallback (rare)
                print(f"  [qwen_omni] No audio output for {utterance_id}; model may not support return_audio.")
                return S2SResult(
                    audio_out_path="",
                    model_name=self.model_name,
                    utterance_id=utterance_id,
                    asr_transcript=asr_response,
                    e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                    error="no_audio_output",
                )

            e2e_latency_ms = (time.perf_counter() - t_start) * 1_000
            ttfb_ms = ((t_first_token or t_start) - t_start) * 1_000

            duration_s = len(audio_np) / max(output_sr, 1)
            rtf = (e2e_latency_ms / 1_000) / max(duration_s, 1e-6)

            return S2SResult(
                audio_out_path=out_path,
                sample_rate=output_sr,
                asr_transcript=asr_response,
                ttfb_ms=ttfb_ms,
                e2e_latency_ms=e2e_latency_ms,
                rtf=rtf,
                model_name=self.model_name,
                utterance_id=utterance_id,
                extra={"text_response": asr_response},
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
