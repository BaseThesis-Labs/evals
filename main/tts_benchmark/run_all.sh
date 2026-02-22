#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║   TTS BENCHMARK — Complete Pipeline     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "✗ Virtual environment not found"
    echo "  Please run: bash setup.sh"
    exit 1
fi

source .venv/bin/activate

# Check API keys
if [ -z "${DEEPGRAM_API_KEY:-}" ] && [ -z "${ASSEMBLYAI_API_KEY:-}" ]; then
    echo "⚠️  Warning: No ASR API key set"
    echo "  Set DEEPGRAM_API_KEY or ASSEMBLYAI_API_KEY"
    echo ""
fi

if [ -z "${CARTESIA_API_KEY:-}" ]; then
    echo "⚠️  Warning: CARTESIA_API_KEY not set"
    echo "  Cartesia model will be skipped"
    echo ""
fi

# Determine ASR provider
ASR_PROVIDER="assemblyai"
if [ -n "${DEEPGRAM_API_KEY:-}" ]; then
    ASR_PROVIDER="deepgram"
fi

START_TIME=$(date +%s)

# Step 1: Download dataset
echo "▶ [1/9] Downloading Seed-TTS-Eval & sampling 200 utterances..."
python datasets/download.py --n-samples 200 --seed 42
echo ""

# Step 2: Generate audio
echo "▶ [2/9] Generating audio (5 models × 200 utterances)..."
python generate.py --config configs/models.yaml --manifest datasets/manifest.json
echo ""

# Step 3: Evaluate
echo "▶ [3/9] Computing 35 metrics..."
python evaluate.py \
    --gen-dir generated_audio \
    --manifest datasets/manifest.json \
    --asr $ASR_PROVIDER \
    --device cpu
echo ""

# Step 4: Aggregate
echo "▶ [4/9] Aggregating scores & computing rankings..."
python aggregate.py --results-dir results --output analysis/leaderboard.json
echo ""

# Step 5: Visualize
echo "▶ [5/9] Generating 6 charts..."
python visualize.py \
    --input analysis/leaderboard.json \
    --output analysis/charts \
    --results-dir results
echo ""

# ── TTSDS Benchmark ────────────────────────────────────────────────────────
echo "▶ [6/9] TTSDS: Downloading dataset & reference audio..."
python datasets/download.py --dataset ttsds --n-samples 50 --seed 42
echo ""

echo "▶ [7/9] TTSDS: Generating audio..."
python generate.py \
    --config configs/models.yaml \
    --manifest datasets/ttsds_manifest.json \
    --output ttsds_gen_audio
echo ""

echo "▶ [8/9] TTSDS: Evaluating distributional scores..."
python evaluate_ttsds.py \
    --gen-dir ttsds_gen_audio \
    --reference-dir datasets/ttsds_reference \
    --output ttsds_results/ttsds_scores.json
echo ""

echo "▶ [9/9] TTSDS: Merging into leaderboard..."
python aggregate.py \
    --results-dir results \
    --ttsds-results ttsds_results/ttsds_scores.json \
    --output analysis/leaderboard.json
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "════════════════════════════════════════════"
echo "  ✅ BENCHMARK COMPLETE!"
echo ""
echo "  Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "  📊 Results:"
echo "     Leaderboard: analysis/leaderboard.json"
echo "     Charts:      analysis/charts/"
echo ""
echo "  📁 Generated:"
echo "     Audio:       generated_audio/ (1,000 files)"
echo "     Metrics:     results/"
echo "════════════════════════════════════════════"
