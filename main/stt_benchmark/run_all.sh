#!/usr/bin/env bash
# run_all.sh — End-to-end STT benchmark pipeline
#
# Usage:
#   bash run_all.sh                                          # test-clean, all models
#   bash run_all.sh --subset test-other                     # LibriSpeech test-other
#   bash run_all.sh --subset both                           # test-clean AND test-other
#   bash run_all.sh --dataset tedlium                       # TED-LIUM 3 test
#   bash run_all.sh --dataset voxpopuli                    # VoxPopuli EN test
#   bash run_all.sh --dataset kincaid --kincaid-dir /path  # Kincaid46 (local)
#   bash run_all.sh --dataset commonvoice                  # Common Voice EN test
#   bash run_all.sh --dataset commonvoice --cv-lang de     # Common Voice German
#   bash run_all.sh --dataset earnings22                   # Earnings22 financial
#   bash run_all.sh --quick                                  # 50 samples per subset
#   bash run_all.sh --max-samples 100
set -euo pipefail

LIBRI_SUBSET="test-clean"
DATASET="librispeech"      # librispeech | tedlium | voxpopuli | kincaid | commonvoice | earnings22
MODELS="all"
MAX_SAMPLES=""
QUICK=false
KINCAID_DIR=""
CV_LANG="en"

while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)         QUICK=true;                    shift ;;
    --subset)        LIBRI_SUBSET="$2";             shift 2 ;;
    --dataset)       DATASET="$2";                  shift 2 ;;
    --models)        MODELS="$2";                   shift 2 ;;
    --max-samples)   MAX_SAMPLES="--max-samples $2"; shift 2 ;;
    --kincaid-dir)   KINCAID_DIR="$2";              shift 2 ;;
    --cv-lang)       CV_LANG="$2";                  shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if $QUICK; then MAX_SAMPLES="--max-samples 50"; fi

# Build list of (name, manifest_path) pairs to evaluate
declare -a EVAL_NAMES=()
declare -a EVAL_MANIFESTS=()

if [[ "$DATASET" == "librispeech" ]]; then
  if [[ "$LIBRI_SUBSET" == "both" ]]; then
    SUBSETS=("test-clean" "test-other")
  else
    SUBSETS=("$LIBRI_SUBSET")
  fi
  for S in "${SUBSETS[@]}"; do
    EVAL_NAMES+=("$S")
    EVAL_MANIFESTS+=("datasets/librispeech/${S}_manifest.jsonl")
  done

elif [[ "$DATASET" == "tedlium" ]]; then
  EVAL_NAMES+=("tedlium_test")
  EVAL_MANIFESTS+=("datasets/tedlium/tedlium_test_manifest.jsonl")

elif [[ "$DATASET" == "voxpopuli" ]]; then
  EVAL_NAMES+=("voxpopuli_test")
  EVAL_MANIFESTS+=("datasets/voxpopuli/voxpopuli_test_manifest.jsonl")

elif [[ "$DATASET" == "kincaid" ]]; then
  EVAL_NAMES+=("kincaid46")
  EVAL_MANIFESTS+=("datasets/kincaid46/kincaid46_manifest.jsonl")

elif [[ "$DATASET" == "commonvoice" ]]; then
  EVAL_NAMES+=("commonvoice_${CV_LANG}_test")
  EVAL_MANIFESTS+=("datasets/commonvoice/commonvoice_${CV_LANG}_test_manifest.jsonl")

elif [[ "$DATASET" == "earnings22" ]]; then
  EVAL_NAMES+=("earnings22_test")
  EVAL_MANIFESTS+=("datasets/earnings22/earnings22_test_manifest.jsonl")

else
  echo "Unknown --dataset: $DATASET"; exit 1
fi

echo "======================================================"
echo " STT Benchmark Pipeline"
echo " Dataset : $DATASET"
echo " Models  : $MODELS"
echo "======================================================"

# ── Setup checks ───────────────────────────────────────────────────────────────
echo ""
echo "[setup] Checking spaCy model (en_core_web_sm)..."
python - <<'PYEOF'
import sys
try:
    import spacy
    spacy.load("en_core_web_sm")
    print("  spaCy en_core_web_sm: OK")
except OSError:
    import subprocess
    print("  en_core_web_sm not found — downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    print("  spaCy en_core_web_sm: downloaded OK")
except ImportError:
    print("  WARNING: spaCy not installed — entity metrics will be skipped")
PYEOF

# Step 0 — Generate silence/noise once
echo ""
echo "[0] Generating silence/noise files..."
python generate.py --source silence

# ── Download / prepare datasets ────────────────────────────────────────────────

if [[ "$DATASET" == "librispeech" ]]; then
  for i in "${!EVAL_NAMES[@]}"; do
    MANIFEST="${EVAL_MANIFESTS[$i]}"
    S="${EVAL_NAMES[$i]}"
    if [[ ! -f "$MANIFEST" ]]; then
      echo ""
      echo "Downloading LibriSpeech ${S}..."
      python generate.py --source librispeech --subsets "$S" $MAX_SAMPLES
    else
      echo "Manifest exists: $MANIFEST"
    fi
  done

elif [[ "$DATASET" == "tedlium" ]]; then
  MANIFEST="${EVAL_MANIFESTS[0]}"
  if [[ ! -f "$MANIFEST" ]]; then
    echo ""
    echo "Downloading TED-LIUM 3 test..."
    python generate.py --source tedlium $MAX_SAMPLES
  fi

elif [[ "$DATASET" == "voxpopuli" ]]; then
  MANIFEST="${EVAL_MANIFESTS[0]}"
  if [[ ! -f "$MANIFEST" ]]; then
    echo ""
    echo "Downloading VoxPopuli EN test..."
    python generate.py --source voxpopuli $MAX_SAMPLES
  fi

elif [[ "$DATASET" == "kincaid" ]]; then
  MANIFEST="${EVAL_MANIFESTS[0]}"
  if [[ ! -f "$MANIFEST" ]]; then
    if [[ -z "$KINCAID_DIR" ]]; then
      echo "ERROR: --kincaid-dir is required for --dataset kincaid"; exit 1
    fi
    echo ""
    echo "Building Kincaid46 manifest from $KINCAID_DIR..."
    python generate.py --source kincaid --kincaid-dir "$KINCAID_DIR" $MAX_SAMPLES
  fi

elif [[ "$DATASET" == "commonvoice" ]]; then
  MANIFEST="${EVAL_MANIFESTS[0]}"
  if [[ ! -f "$MANIFEST" ]]; then
    echo ""
    echo "Downloading Common Voice ${CV_LANG} test..."
    echo "NOTE: Requires HuggingFace login and dataset terms acceptance."
    python generate.py --source commonvoice --commonvoice-lang "$CV_LANG" $MAX_SAMPLES
  fi

elif [[ "$DATASET" == "earnings22" ]]; then
  MANIFEST="${EVAL_MANIFESTS[0]}"
  if [[ ! -f "$MANIFEST" ]]; then
    echo ""
    echo "Downloading Earnings22 test..."
    python generate.py --source earnings22 $MAX_SAMPLES
  fi
fi

# ── Evaluate each dataset/subset ──────────────────────────────────────────────

for i in "${!EVAL_NAMES[@]}"; do
  NAME="${EVAL_NAMES[$i]}"
  MANIFEST="${EVAL_MANIFESTS[$i]}"

  RESULTS_DIR="results/${DATASET}/${NAME}"
  ANALYSIS_DIR="analysis/${DATASET}/${NAME}"
  CHARTS_DIR="${ANALYSIS_DIR}/charts"

  echo ""
  echo "======================================================"
  echo " Evaluating: ${NAME}"
  echo " Manifest  → ${MANIFEST}"
  echo " Results   → ${RESULTS_DIR}/"
  echo " Analysis  → ${ANALYSIS_DIR}/"
  echo "======================================================"

  # Step 1 — Evaluate all models
  echo ""
  echo "[1/3] Evaluating STT models..."
  MODELS_ARG=""
  if [[ "$MODELS" != "all" ]]; then MODELS_ARG="--models $MODELS"; fi
  python evaluate.py \
    --dataset "$MANIFEST" \
    $MODELS_ARG \
    --output-dir "$RESULTS_DIR" \
    $MAX_SAMPLES

  # Step 1b — Hallucination test
  echo ""
  echo "  [1b] Hallucination test..."
  python evaluate.py \
    --dataset datasets/silence_noise \
    $MODELS_ARG \
    --output-dir "$RESULTS_DIR" \
    --hallucination-only || true

  # Step 2 — Aggregate scores
  echo ""
  echo "[2/3] Aggregating scores..."
  python aggregate.py \
    --metrics-dir "${RESULTS_DIR}/metrics" \
    --output-dir  "$ANALYSIS_DIR" \
    --case-study  balanced

  # Step 3 — Visualize
  echo ""
  echo "[3/3] Generating charts..."
  python visualize.py \
    --leaderboard "${ANALYSIS_DIR}/leaderboard.json" \
    --metrics-dir "${RESULTS_DIR}/metrics" \
    --output      "$CHARTS_DIR"

  echo ""
  echo "  Leaderboard : ${ANALYSIS_DIR}/leaderboard.json"
  echo "  Charts      : ${CHARTS_DIR}/"

done

# ── Cross-dataset comparison (if multiple subsets were evaluated) ──────────────
if [[ ${#EVAL_NAMES[@]} -gt 1 ]]; then
  ANALYSIS_DIRS=""
  for NAME in "${EVAL_NAMES[@]}"; do
    ANALYSIS_DIRS+="analysis/${DATASET}/${NAME},"
  done
  ANALYSIS_DIRS="${ANALYSIS_DIRS%,}"   # strip trailing comma

  echo ""
  echo "======================================================"
  echo " Cross-dataset comparison..."
  echo "======================================================"
  python compare_datasets.py \
    --dirs "$ANALYSIS_DIRS" \
    --output "comparison_charts/${DATASET}"
  echo "  Comparison charts → comparison_charts/${DATASET}/"
fi

echo ""
echo "======================================================"
echo " All done!"
echo "======================================================"
