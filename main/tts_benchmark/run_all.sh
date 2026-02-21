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
echo "▶ [1/5] Downloading Seed-TTS-Eval & sampling 200 utterances..."
python datasets/download.py --n-samples 200 --seed 42
echo ""

# Step 2: Generate audio
echo "▶ [2/5] Generating audio (5 models × 200 utterances)..."
python generate.py --config configs/models.yaml --manifest datasets/manifest.json
echo ""

# Step 3: Evaluate
echo "▶ [3/5] Computing 35 metrics..."
python evaluate.py \
    --gen-dir generated_audio \
    --manifest datasets/manifest.json \
    --asr $ASR_PROVIDER \
    --device cpu
echo ""

# Step 4: Aggregate
echo "▶ [4/5] Aggregating scores & computing rankings..."
python aggregate.py --results-dir results --output analysis/leaderboard.json
echo ""

# Step 5: Visualize
echo "▶ [5/5] Generating 6 charts..."
python visualize.py \
    --input analysis/leaderboard.json \
    --output analysis/charts \
    --results-dir results
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
