#!/usr/bin/env python3
"""
S2S Evaluation Pipeline — master evaluator.

Loads the manifest + inference outputs, runs all metric modules per utterance,
saves {model}_metrics.json, calls aggregate.py, saves results/leaderboard.json.

Usage:
    python pipeline.py \
        --manifest datasets/manifests/s2s_manifest.json \
        --outputs  s2s_outputs \
        --results  results \
        [--model   cascaded_elevenlabs] \
        [--config  config/eval_config.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Ensure s2s_benchmark root is importable ───────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── stt_benchmark on sys.path (for enriched metrics) ─────────────────────────
_STT_BENCH = _ROOT.parent / "stt_benchmark"
if _STT_BENCH.is_dir() and str(_STT_BENCH) not in sys.path:
    sys.path.insert(0, str(_STT_BENCH))

from metrics.content import (  # noqa: E402
    compute_wer_cer,
    compute_asr_details,
    compute_stt_enriched,
    compute_bert_score,
    compute_sem_dist,
    compute_rouge_l,
)
from metrics.quality import compute_utmos, compute_dnsmos, compute_nisqa, compute_pesq, compute_mcd
from metrics.speaker import compute_secs, compute_sim_wavlm, compute_sim_ecapa, compute_eer
from metrics.prosody import compute_all_prosody
from metrics.emotion import compute_all_emotion
from metrics.latency import extract_latency_from_result, summarize_latency
from metrics.interaction import compute_all_interaction
from metrics.judge import compute_all_judge
from scoring.aggregate import aggregate_model, aggregate_multiturn_sessions, build_leaderboard


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(path: str) -> List[Dict]:
    with open(path) as f:
        first_char = f.read(1)
    with open(path) as f:
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def load_gen_meta(model_dir: Path) -> Dict[str, Dict]:
    meta_path = model_dir / "gen_meta.jsonl"
    if not meta_path.exists():
        return {}
    meta = {}
    with open(meta_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                meta[rec["id"]] = rec
            except json.JSONDecodeError:
                pass
    return meta


def _load_audio(path: str):
    """Load audio as (np.ndarray float32, sample_rate)."""
    import numpy as np
    import soundfile as sf  # type: ignore
    if not path or not Path(path).exists():
        return None, None
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio, sr
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_utterance(
    entry: Dict,
    gen_meta: Dict,
    model_dir: Path,
    cfg_metrics: Dict,
) -> Dict:
    """Compute all metrics for a single utterance.

    Args:
        entry:     manifest entry
        gen_meta:  gen_meta.jsonl record for this utterance (may be empty)
        model_dir: directory containing {utt_id}.wav
        cfg_metrics: metrics section of eval_config.yaml

    Returns:
        dict of metric_name → value, plus id / error fields.
    """
    utt_id: str = entry["id"]
    ref_text: str = entry.get("reference_text", "")
    ref_audio_path: Optional[str] = entry.get("audio_in_path")  # input audio = reference
    emotion_label: Optional[str] = entry.get("emotion")
    # model_type: "echo" (output should match input text) vs "generative" (AI response)
    # Echo metrics (WER/CER/BERTScore/ROUGE) are only valid for echo models
    is_echo: bool = gen_meta.get("model_type", "echo") == "echo"

    result: Dict = {"id": utt_id, "model_type": gen_meta.get("model_type", "echo"), "error": None}

    # ── Output audio path ─────────────────────────────────────────────────────
    hyp_audio_path_from_meta = gen_meta.get("audio_out_path", "")
    hyp_audio_path = str(model_dir / f"{utt_id}.wav")
    if not Path(hyp_audio_path).exists() and hyp_audio_path_from_meta:
        hyp_audio_path = hyp_audio_path_from_meta

    if not Path(hyp_audio_path).exists():
        result["error"] = "output audio not found"
        return result

    # ── Latency ───────────────────────────────────────────────────────────────
    latency = extract_latency_from_result(gen_meta)
    result.update(latency)

    # ── ASR transcript ────────────────────────────────────────────────────────
    # Prefer transcript saved during inference
    asr_transcript: Optional[str] = gen_meta.get("asr_transcript")

    # If no saved transcript, run ASR on output audio
    if not asr_transcript and cfg_metrics.get("compute_wer", True):
        asr_transcript = _run_asr_on_output(hyp_audio_path)

    hyp_text = asr_transcript or ""

    # ── Content metrics ───────────────────────────────────────────────────────
    # WER/CER/BERTScore/ROUGE only make sense for echo models (output ≈ input text).
    # For generative models the output is an AI response, not a repetition of input.
    if ref_text and hyp_text:
        ref_norm = ref_text.strip().lower()
        hyp_norm = hyp_text.strip().lower()

        if is_echo:
            if cfg_metrics.get("compute_wer", True):
                result.update(compute_wer_cer(ref_norm, hyp_norm))

            if cfg_metrics.get("compute_asr_details", True):
                result.update(compute_asr_details(ref_norm, hyp_norm))

            if cfg_metrics.get("use_stt_benchmark_metrics", True):
                result.update(compute_stt_enriched(ref_norm, hyp_norm))

            if cfg_metrics.get("compute_rouge_l", True):
                result["rouge_l"] = compute_rouge_l(ref_norm, hyp_norm)

        # Semantic distance is valid for both: echo measures preservation,
        # generative measures relevance of response to input
        if cfg_metrics.get("compute_sem_dist", True):
            dists = compute_sem_dist([ref_norm], [hyp_norm])
            result["sem_dist"] = dists[0] if dists else None

    # ── BERTScore — echo only ─────────────────────────────────────────────────
    if is_echo and ref_text and hyp_text and cfg_metrics.get("compute_bert_score", True):
        bs = compute_bert_score([ref_text], [hyp_text])
        result["bert_score_f1"] = bs.get("bert_score_f1")
        result["bert_score_precision"] = bs.get("bert_score_precision")
        result["bert_score_recall"] = bs.get("bert_score_recall")

    # ── Audio quality ─────────────────────────────────────────────────────────
    hyp_audio, hyp_sr = _load_audio(hyp_audio_path)
    if hyp_audio is not None and hyp_sr is not None:
        if cfg_metrics.get("compute_utmos", True):
            result["utmos"] = compute_utmos(hyp_audio, hyp_sr)

        if cfg_metrics.get("compute_dnsmos", True):
            dnsmos = compute_dnsmos(hyp_audio, hyp_sr)
            result.update(dnsmos)

        if cfg_metrics.get("compute_nisqa", False):
            nisqa = compute_nisqa(hyp_audio, hyp_sr)
            result.update(nisqa)

        if is_echo and cfg_metrics.get("compute_mcd", True) and ref_audio_path and Path(ref_audio_path).exists():
            ref_audio, ref_sr = _load_audio(ref_audio_path)
            if ref_audio is not None:
                result["mcd"] = compute_mcd(ref_audio, ref_sr, hyp_audio, hyp_sr)

    # ── PESQ (needs reference audio — echo only) ─────────────────────────────
    if (
        is_echo
        and cfg_metrics.get("compute_pesq", True)
        and ref_audio_path
        and Path(ref_audio_path).exists()
        and hyp_audio is not None
    ):
        ref_audio, ref_sr = _load_audio(ref_audio_path)
        if ref_audio is not None:
            result["pesq"] = compute_pesq(ref_audio, ref_sr, hyp_audio, hyp_sr)

    # ── Speaker similarity ────────────────────────────────────────────────────
    if ref_audio_path and Path(ref_audio_path).exists():
        use_large = cfg_metrics.get("use_secs_large", False)
        if cfg_metrics.get("compute_sim_wavlm", True):
            if use_large:
                result["secs"] = compute_secs(ref_audio_path, hyp_audio_path, use_large=True)
                result["sim_wavlm"] = compute_secs(ref_audio_path, hyp_audio_path, use_large=False)
            else:
                # CPU mode: only compute base model, assign to sim_wavlm only
                result["sim_wavlm"] = compute_secs(ref_audio_path, hyp_audio_path, use_large=False)

        if cfg_metrics.get("include_ecapa", False):
            result["sim_ecapa"] = compute_sim_ecapa(ref_audio_path, hyp_audio_path)

    # ── Prosody ───────────────────────────────────────────────────────────────
    if ref_audio_path and Path(ref_audio_path).exists() and cfg_metrics.get("compute_f0_rmse", True):
        include_dswed = cfg_metrics.get("include_dswed", False)
        prosody = compute_all_prosody(
            ref_audio_path,
            hyp_audio_path,
            include_dswed=include_dswed,
            hyp_audio=hyp_audio,
            hyp_sr=hyp_sr,
            hyp_text=hyp_text or None,
            is_echo=is_echo,
        )
        result.update(prosody)
        # Record source of speaking_rate text for transparency
        if result.get("speaking_rate") is not None:
            result["speaking_rate_source"] = "reference" if is_echo else "asr_hypothesis"
        # Echo-mode: compute speaking_rate_ratio = hyp_rate / ref_rate
        if is_echo and ref_text and hyp_text:
            try:
                from metrics.prosody import compute_speaking_rate
                ref_audio_arr, ref_sr_val = _load_audio(ref_audio_path)
                if ref_audio_arr is not None:
                    ref_rate = compute_speaking_rate(ref_audio_arr, ref_sr_val, ref_text)
                    hyp_rate = result.get("speaking_rate")
                    if ref_rate and hyp_rate and ref_rate > 0:
                        result["speaking_rate_ratio"] = hyp_rate / ref_rate
            except Exception:
                pass  # non-critical; skip if speaking rate computation fails
    elif hyp_audio is not None and hyp_sr is not None:
        # No reference audio — still compute single-signal prosody metrics
        from metrics.prosody import compute_speaking_rate, compute_pause_ratio
        result["pause_ratio"] = compute_pause_ratio(hyp_audio, hyp_sr)
        if hyp_text:
            result["speaking_rate"] = compute_speaking_rate(hyp_audio, hyp_sr, hyp_text)

    # ── Emotion ───────────────────────────────────────────────────────────────
    if ref_audio_path and Path(ref_audio_path).exists():
        use_e2v = cfg_metrics.get("use_emotion2vec", True)
        include_esim = cfg_metrics.get("include_esim", True)
        if emotion_label is not None or use_e2v:
            emotion = compute_all_emotion(
                ref_audio_path, hyp_audio_path,
                use_emotion2vec=use_e2v,
                compute_esim_metric=include_esim,
            )
            result.update(emotion)

    # ── LLM-as-judge + instruction following + safety refusal ────────────────
    # Skip judge metrics entirely for echo models — they are semantically
    # meaningless (echo models repeat input, not generate responses).
    use_llm_judge = cfg_metrics.get("use_llm_judge", False)
    compute_if = cfg_metrics.get("compute_instruction_follow", True)
    compute_sr = cfg_metrics.get("compute_safety_refusal", True)

    if not is_echo and hyp_text and (use_llm_judge or compute_if or compute_sr):
        judge_entry = {**entry}
        # Add flags so compute_all_judge can skip LLM call if not requested
        judge_results = compute_all_judge(judge_entry, hyp_text, use_llm_judge=use_llm_judge)
        # Selectively keep results based on flags
        if use_llm_judge:
            for k in ("judge_coherence", "judge_relevance", "judge_helpfulness",
                      "judge_safety", "judge_naturalness", "judge_overall", "judge_reasoning"):
                result[k] = judge_results.get(k)
            result["judge_model"] = judge_results.get("judge_model")
        if compute_if:
            result["instruction_follow"] = judge_results.get("instruction_follow")
        if compute_sr:
            result["safety_refusal"] = judge_results.get("safety_refusal")

    # ── Interaction / TOR ─────────────────────────────────────────────────────
    if cfg_metrics.get("compute_tor", True):
        interaction = compute_all_interaction(gen_meta, ref_audio_path=ref_audio_path)
    else:
        interaction = compute_all_interaction(gen_meta, ref_audio_path=None)
    result.update(interaction)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ASR helper (on output audio — when transcript not in gen_meta)
# ─────────────────────────────────────────────────────────────────────────────

_WHISPER_MODEL = None


def _run_asr_on_output(audio_path: str) -> Optional[str]:
    global _WHISPER_MODEL
    try:
        import whisper  # type: ignore
        if _WHISPER_MODEL is None:
            print("  [pipeline] Loading Whisper for output ASR …")
            _WHISPER_MODEL = whisper.load_model("base", device="cpu")
        result = _WHISPER_MODEL.transcribe(audio_path, language="en", fp16=False)
        return result.get("text", "").strip()
    except Exception as exc:
        print(f"  [pipeline] ASR on output failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    manifest: List[Dict],
    outputs_root: Path,
    results_dir: Path,
    cfg_metrics: Dict,
    skip_existing: bool = True,
) -> Optional[Dict]:
    model_dir = outputs_root / model_name
    if not model_dir.exists():
        print(f"  ⊘ No output directory for {model_name}: {model_dir}")
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_metrics.json"

    # Load gen_meta
    gen_meta_all = load_gen_meta(model_dir)

    # Evaluate each utterance
    utterance_results = []
    for entry in tqdm(manifest, desc=f"eval {model_name}"):
        utt_id = entry["id"]
        gen_meta_rec = gen_meta_all.get(utt_id, {})

        if gen_meta_rec.get("error") and not gen_meta_rec.get("audio_out_path"):
            utterance_results.append({"id": utt_id, "error": gen_meta_rec["error"]})
            continue

        metrics = evaluate_utterance(entry, gen_meta_rec, model_dir, cfg_metrics)
        utterance_results.append(metrics)

    # Latency summary
    latency_summary = summarize_latency(utterance_results)

    # ── Dataset-level EER (optional, needs ≥ 100 pairs for meaningful results) ─
    if cfg_metrics.get("compute_eer", False):
        ref_paths = []
        hyp_paths = []
        for entry in manifest:
            utt_id = entry["id"]
            rp = entry.get("audio_in_path", "")
            hp = str(model_dir / f"{utt_id}.wav")
            if rp and Path(rp).exists() and Path(hp).exists():
                ref_paths.append(rp)
                hyp_paths.append(hp)
        if len(ref_paths) >= 5:
            use_large = cfg_metrics.get("use_secs_large", False)
            eer_val = compute_eer(ref_paths, hyp_paths, use_large=use_large)
            print(f"  EER ({model_name}): {eer_val:.2f}%" if eer_val is not None else "  EER: N/A")
            # Inject EER into each utterance result so aggregate.py can average it
            for ur in utterance_results:
                if not ur.get("error"):
                    ur["eer"] = eer_val

    # Aggregate — detect model_type from first valid utterance
    _mt = "generative"
    for ur in utterance_results:
        if ur.get("model_type"):
            _mt = ur["model_type"]
            break
    agg = aggregate_model(utterance_results, model_type=_mt)
    agg["latency_percentiles"] = latency_summary

    # Persist per-model results
    output = {
        "model": model_name,
        "utterances": utterance_results,
        "aggregate": agg,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ {model_name} — saved {out_path}")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multiturn(
    model_name: str,
    model_cfg: Dict,
    config: Dict,
    results_dir: Path,
) -> Optional[Dict]:
    """Run multi-turn scenario evaluation for a single model.

    Returns aggregate dict compatible with aggregate_model() output.
    """
    from datasets.multiturn.scenario_builder import load_all_scenarios
    from inference.multiturn_runner import run_all_sessions, SessionResult
    from metrics.multiturn.session_quality import compute_session_quality

    mt_cfg = config.get("multiturn", {})
    scenarios_dir = Path(mt_cfg.get("scenarios_dir", "datasets/multiturn/scenarios"))
    if not scenarios_dir.exists():
        print(f"  ⊘ No scenarios directory: {scenarios_dir}")
        return None

    scenarios = load_all_scenarios(str(scenarios_dir))
    if not scenarios:
        print(f"  ⊘ No scenarios found in {scenarios_dir}")
        return None

    # Load adapter
    import importlib
    class_path = model_cfg["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    adapter = cls(model_name=model_name, config=model_cfg.get("config", {}))

    print(f"  Running {len(scenarios)} multi-turn scenarios for {model_name} …")
    session_results: List[SessionResult] = run_all_sessions(adapter, scenarios, config)

    # Compute session-level metrics for each result
    session_metrics_list = []
    for sr, scenario in zip(session_results, scenarios):
        try:
            sm = compute_session_quality(sr, scenario)
            sm["scenario_id"] = sr.scenario_id
            sm["model_name"] = model_name
            session_metrics_list.append(sm)
        except Exception as exc:
            print(f"    ⊘ session_quality error for {sr.scenario_id}: {exc}")
            session_metrics_list.append({
                "scenario_id": sr.scenario_id,
                "model_name": model_name,
                "error": str(exc),
            })

    # Aggregate across sessions
    agg = aggregate_multiturn_sessions(session_metrics_list, model_type="generative")

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_multiturn.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "sessions": session_metrics_list,
            "aggregate": agg,
        }, f, indent=2)
    print(f"  ✓ {model_name} multi-turn — saved {out_path}")

    try:
        adapter.cleanup()
    except Exception:
        pass

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="S2S evaluation pipeline")
    parser.add_argument(
        "--manifest",
        default="datasets/manifests/s2s_manifest.json",
    )
    parser.add_argument("--outputs", default="s2s_outputs", help="Inference output root")
    parser.add_argument("--results", default="results", help="Results output directory")
    parser.add_argument("--config", default="config/eval_config.yaml")
    parser.add_argument("--model", default=None, help="Evaluate single model only")
    parser.add_argument(
        "--mode",
        choices=["single", "multiturn", "both"],
        default="single",
        help="Evaluation mode: single-turn, multiturn agent, or both",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-evaluate even if results JSON already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_metrics = cfg.get("metrics", {})
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = cfg.get("models", {})

    # Discover models
    if args.model:
        model_names = [args.model]
    else:
        model_names = [
            name for name, mcfg in models_cfg.items()
            if mcfg.get("enabled", True)
        ]

    # ── Single-turn evaluation ────────────────────────────────────────────
    model_aggregates: Dict[str, Dict] = {}
    if args.mode in ("single", "both"):
        manifest = load_manifest(args.manifest)
        print(f"✓ Loaded {len(manifest)} utterances from {args.manifest}")

        outputs_root = Path(args.outputs)
        # Also scan outputs_root for extra directories
        if outputs_root.exists():
            for p in outputs_root.iterdir():
                if p.is_dir() and p.name not in model_names:
                    model_names.append(p.name)

        for mname in model_names:
            agg = evaluate_model(
                mname,
                manifest,
                outputs_root,
                results_dir,
                cfg_metrics,
                skip_existing=not args.no_skip,
            )
            if agg is not None:
                model_aggregates[mname] = agg

        if model_aggregates:
            for use_case in ["balanced", "conversational", "audiobook", "voice_cloning", "expressive"]:
                leaderboard = build_leaderboard(model_aggregates, use_case=use_case)
                lb_path = results_dir / f"leaderboard_{use_case}.json"
                with open(lb_path, "w") as f:
                    json.dump(leaderboard, f, indent=2)
                print(f"  ✓ Leaderboard ({use_case}) → {lb_path}")

            lb_balanced = build_leaderboard(model_aggregates, "balanced")
            with open(results_dir / "leaderboard.json", "w") as f:
                json.dump(lb_balanced, f, indent=2)

    # ── Multi-turn evaluation ─────────────────────────────────────────────
    mt_aggregates: Dict[str, Dict] = {}
    if args.mode in ("multiturn", "both"):
        print("\n── Multi-turn agent evaluation ──")
        for mname in model_names:
            if mname not in models_cfg:
                continue
            mcfg = models_cfg[mname]
            agg = evaluate_multiturn(mname, mcfg, cfg, results_dir)
            if agg is not None:
                mt_aggregates[mname] = agg

        if mt_aggregates:
            lb_agent = build_leaderboard(mt_aggregates, use_case="agent")
            lb_path = results_dir / "leaderboard_agent.json"
            with open(lb_path, "w") as f:
                json.dump(lb_agent, f, indent=2)
            print(f"  ✓ Leaderboard (agent) → {lb_path}")

    if not model_aggregates and not mt_aggregates:
        print("No model results to aggregate.")
        return

    print("\n✓ Evaluation complete!")
    print(f"  Results: {results_dir}")


if __name__ == "__main__":
    main()
