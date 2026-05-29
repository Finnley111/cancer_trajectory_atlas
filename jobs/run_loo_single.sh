#!/bin/bash
# Per-task LOO runner. Called by submit_loo_array.sh via SLURM array.
# Do NOT submit this directly — use submit_loo_array.sh.
#
# Expects these environment variables set by the array job:
#   HELD_OUT     — stem of the held-out slide (e.g. 6027-4L-2M-1_x5)
#   FULL_RUN     — path to the full 16-slide reference run (for in-manifold PT)
#   CACHE_DIR    — path to feature cache directory
#   MAX_PATCHES  — per-slide patch cap (optional; omit or leave empty for no cap)
#   SAMPLE_SEED  — base seed for patch subsampling (default: 42)

set -euo pipefail

if [[ -z "${HELD_OUT:-}" ]]; then
    echo "ERROR: HELD_OUT is not set"
    exit 1
fi

MAX_PATCHES=${MAX_PATCHES:-}
SAMPLE_SEED=${SAMPLE_SEED:-42}
LOO_SUFFIX=${LOO_SUFFIX:-}

LOO_OUT="$SCRATCH/results/loo_${HELD_OUT}${LOO_SUFFIX}"
mkdir -p "$LOO_OUT"

echo "=== LOO run: held-out = $HELD_OUT ==="
echo "Output: $LOO_OUT"
echo "Reference run: ${FULL_RUN:-$SCRATCH/results/atlas_none_harmony}"
echo ""

# Build comma-separated list of all 15 training slides (exclude held-out)
SLIDE_LIST="$HOME/cancer_trajectory_atlas/data/loo_slides.txt"
if [[ ! -f "$SLIDE_LIST" ]]; then
    echo "ERROR: loo_slides.txt not found at $SLIDE_LIST"
    exit 1
fi

TRAINING_SLIDES=$(grep -v "^${HELD_OUT}$" "$SLIDE_LIST" | paste -sd,)
echo "Training slides (15): $TRAINING_SLIDES"
echo ""

cd ~

# Phase A — run pipeline on 15 training slides (features loaded from cache)
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir             $SCRATCH/data/MCF7_x5_cropped \
    --annotation-dir      ~/cancer_trajectory_atlas/data/annotations_ratio \
    --output-dir          "$LOO_OUT" \
    --stain-method        none \
    --model               phikon \
    --patch-size          112 \
    --stride              96 \
    --clustering-method   leiden \
    --leiden-resolution   0.5 \
    --harmony \
    --harmony-key         section_number \
    --n-permutations      200 \
    --slides              "$TRAINING_SLIDES" \
    --features-cache-dir  "${CACHE_DIR:-$SCRATCH/data/features_cache}" \
    ${MAX_PATCHES:+--max-patches-per-slide "$MAX_PATCHES"} \
    --patch-sample-seed   "$SAMPLE_SEED"

echo ""
echo "=== Phase B: projecting held-out slide ==="

FULL_RUN_DIR="${FULL_RUN:-$SCRATCH/results/atlas_none_harmony}"

# Phase B requires the reference full-run results.csv.
# If it doesn't exist yet (reference run still pending), skip Phase B with a clear message.
if [[ ! -f "$FULL_RUN_DIR/results.csv" ]]; then
    echo ""
    echo "WARNING: Phase B skipped — reference run results.csv not found at:"
    echo "  $FULL_RUN_DIR/results.csv"
    echo ""
    echo "Phase A projector saved to: $LOO_OUT/projector"
    echo "Re-run Phase B once the reference run finishes:"
    echo "  python -m cancer_trajectory_atlas.analysis.loo_project \\"
    echo "      --projector-dir  $LOO_OUT/projector \\"
    echo "      --held-out-slide \"$HELD_OUT\" \\"
    echo "      --cache-dir      \"${CACHE_DIR:-\$SCRATCH/data/features_cache}\" \\"
    echo "      --full-run-dir   \"$FULL_RUN_DIR\" \\"
    echo "      --output-dir     \"$LOO_OUT\" \\"
    if [[ -n "$MAX_PATCHES" ]]; then
        echo "      --max-patches-per-slide $MAX_PATCHES \\"
    fi
    echo "      --patch-sample-seed $SAMPLE_SEED"
    exit 0
fi

python -m cancer_trajectory_atlas.analysis.loo_project \
    --projector-dir  "$LOO_OUT/projector" \
    --held-out-slide "$HELD_OUT" \
    --cache-dir      "${CACHE_DIR:-$SCRATCH/data/features_cache}" \
    --full-run-dir   "$FULL_RUN_DIR" \
    --output-dir     "$LOO_OUT" \
    ${MAX_PATCHES:+--max-patches-per-slide "$MAX_PATCHES"} \
    --patch-sample-seed "$SAMPLE_SEED"

echo ""
echo "=== Done: $HELD_OUT ==="
echo "Results in: $LOO_OUT"
