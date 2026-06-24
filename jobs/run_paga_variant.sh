#!/bin/bash
# Parameterized PAGA experiment variant: one slide-subset x one harmony
# setting. Reused for all 6 variants (3 subsets x 2 harmony settings) via
# submit_paga_runs.sh rather than duplicated into 6 near-identical files.
#
# Stain normalization is fixed to 'none' for every variant by design (see
# PROJECT_STATE.md — no-norm + Harmony is the canonical pipeline config),
# which is also what makes the shared feature cache safe across all variants.
#
# Within one job this does, in order:
#   1. Full-cohort run over the chosen slide subset (with/without Harmony).
#   2. Post-processing (interactive overlay + patch export) on that run.
#   3. A full leave-one-out suite sized to the subset (16-fold for 'all',
#      8-fold for 'section1'/'section2').
#   4. LOO summary aggregation.
#
# Prerequisite: jobs/run_cache_prepop.sh must have completed successfully
# (all 16 slides' features cached at $SCRATCH/data/features_cache) BEFORE
# this job starts, so every run_all.py call below is a cache-hit-only,
# read-only pass — no GPU needed, and no write races with the other 5
# variants running in parallel. submit_paga_runs.sh enforces this via
# --dependency=afterok; do not submit this script standalone unless you have
# already confirmed the cache is fully populated.
#
# Usage (normally only called by submit_paga_runs.sh):
#   sbatch jobs/run_paga_variant.sh <all|section1|section2> <harmony|noharmony> <output_subdir_name>
# Example:
#   sbatch jobs/run_paga_variant.sh section1 harmony section1_harmony

#SBATCH --account=def-lmarti46
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_paga_variant
#SBATCH --output=logs/paga_variant-%j.out

set -euo pipefail

mkdir -p logs

SUBSET="${1:-}"
HARMONY_MODE="${2:-}"
VARIANT_NAME="${3:-}"

if [[ "$SUBSET" != "all" && "$SUBSET" != "section1" && "$SUBSET" != "section2" ]]; then
    echo "ERROR: first argument must be 'all', 'section1', or 'section2' (got: '$SUBSET')"
    exit 1
fi
if [[ "$HARMONY_MODE" != "harmony" && "$HARMONY_MODE" != "noharmony" ]]; then
    echo "ERROR: second argument must be 'harmony' or 'noharmony' (got: '$HARMONY_MODE')"
    exit 1
fi
if [[ -z "$VARIANT_NAME" ]]; then
    echo "ERROR: third argument (output subdir name) is required"
    exit 1
fi

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
CACHE_DIR="$SCRATCH/data/features_cache"

# ── Resolve the slide list for this variant ──────────────────────────────
ALL_SLIDES_LIST=(
    6027-4L-2M-1_x5
    6027-4L-2M-2_x5
    6027-4R-2M-1_x5
    6027-4R-2M-2_x5
    6028-4L-2M-1_x5
    6028-4L-2M-2_x5
    6028-4R-2M-1_x5
    6028-4R-2M-2_x5
    6029-4L-2M-1_x5
    6029-4L-2M-2_x5
    6029-4R-2M-1_x5
    6029-4R-2M-2_x5
    6031-4L-2M-1_x5
    6031-4L-2M-2_x5
    6031-4R-2M-1_x5
    6031-4R-2M-2_x5
)

case "$SUBSET" in
    all)
        SLIDES=("${ALL_SLIDES_LIST[@]}")
        SLIDE_LIST_FILE=""
        ;;
    section1)
        SLIDE_LIST_FILE="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
        mapfile -t SLIDES < "$SLIDE_LIST_FILE"
        ;;
    section2)
        SLIDE_LIST_FILE="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"
        mapfile -t SLIDES < "$SLIDE_LIST_FILE"
        ;;
esac

SLIDES_CSV=$(IFS=,; echo "${SLIDES[*]}")

# ── Resolve Harmony args ──────────────────────────────────────────────────
HARMONY_ARGS=()
if [[ "$HARMONY_MODE" == "harmony" ]]; then
    HARMONY_ARGS=(--harmony --harmony-key section_number)
fi

VARIANT_DIR="$SCRATCH/results/runs_paga/$VARIANT_NAME"
FULL_DIR="$VARIANT_DIR/full"
LOO_DIR="$VARIANT_DIR/loo"
mkdir -p "$FULL_DIR" "$LOO_DIR"

echo "============================================"
echo "  PAGA experiment variant: $VARIANT_NAME"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Subset:     $SUBSET (${#SLIDES[@]} slides)"
echo "  Harmony:    $HARMONY_MODE"
echo "  Stain:      none"
echo "  Full run:   $FULL_DIR"
echo "  LOO dir:    $LOO_DIR"
echo "============================================"

cd ~

# ── Step 1: full-cohort run ───────────────────────────────────────────────
echo ""
echo "=== Full-cohort run (${#SLIDES[@]} slides) ==="
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$FULL_DIR" \
    --slides                "$SLIDES_CSV" \
    --stain-method          none \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.5 \
    "${HARMONY_ARGS[@]}" \
    --n-permutations        1000 \
    --cap-strategy          median \
    --n-roots               20 \
    --features-cache-dir    "$CACHE_DIR"

echo "Full-cohort results.csv:"
ls -lh "$FULL_DIR/results.csv"

# ── Step 2: post-processing (overlays + patch export), automatic ────────
echo ""
echo "=== Post-processing: overlays and patch export ==="
python -m cancer_trajectory_atlas.visualize.interactive_overlay \
    --results-csv  "$FULL_DIR/results.csv" \
    --png-dir      "$PNG_DIR" \
    --output-dir   "$FULL_DIR/overlays" \
    --patch-size   112

python -m cancer_trajectory_atlas.visualize.export_patches \
    --results-csv  "$FULL_DIR/results.csv" \
    --png-dir      "$PNG_DIR" \
    --output-dir   "$FULL_DIR/patch_export" \
    --patch-size   112 \
    --n-per-bin    50

# ── Step 3: LOO loop over this variant's own slide list ──────────────────
BASELINE_CAP=$(cat "$FULL_DIR/active_cap.txt")
echo ""
echo "=== LOO loop (${#SLIDES[@]} folds, cap=$BASELINE_CAP) ==="

for HELD_OUT in "${SLIDES[@]}"; do
    echo ""
    echo "--- LOO: held-out = $HELD_OUT ---"

    LOO_OUT="$LOO_DIR/loo_${HELD_OUT}"
    mkdir -p "$LOO_OUT"

    TRAINING_SLIDES=$(printf '%s\n' "${SLIDES[@]}" | grep -v "^${HELD_OUT}$" | paste -sd,)

    # Phase A — atlas on the variant's slides minus the held-out one.
    python -m cancer_trajectory_atlas.run_all \
        --run \
        --png-dir               "$PNG_DIR" \
        --annotation-dir        "$ANN_DIR" \
        --output-dir            "$LOO_OUT" \
        --slides                "$TRAINING_SLIDES" \
        --stain-method          none \
        --model                 phikon \
        --patch-size            112 \
        --stride                96 \
        --clustering-method     leiden \
        --leiden-resolution     0.5 \
        "${HARMONY_ARGS[@]}" \
        --n-permutations        200 \
        --cap-strategy          median \
        --n-roots               20 \
        --features-cache-dir    "$CACHE_DIR"

    # Phase B — project the held-out slide onto the training manifold.
    # --max-patches-per-slide must equal BASELINE_CAP so patch counts align
    # with the full-cohort run (paired Spearman rho comparison).
    python -m cancer_trajectory_atlas.analysis.loo_project \
        --projector-dir         "$LOO_OUT/projector" \
        --held-out-slide        "$HELD_OUT" \
        --cache-dir             "$CACHE_DIR" \
        --full-run-dir          "$FULL_DIR" \
        --output-dir            "$LOO_OUT" \
        --max-patches-per-slide "$BASELINE_CAP" \
        --patch-sample-seed     42

    echo "Done: $HELD_OUT"
done

# ── Step 4: aggregate LOO results ─────────────────────────────────────────
echo ""
echo "=== LOO summary ==="
python -m cancer_trajectory_atlas.analysis.loo_summary \
    --loo-dirs   "$LOO_DIR"/loo_* \
    --output-dir "$LOO_DIR/summary"

echo ""
echo "============================================"
echo "  VARIANT COMPLETE: $VARIANT_NAME"
echo "============================================"
echo "  Full run:    $FULL_DIR"
echo "  Overlays:    $FULL_DIR/overlays/"
echo "  Patches:     $FULL_DIR/patch_export/"
echo "  LOO results: $LOO_DIR/loo_*/loo_result_*.json"
echo "  LOO summary: $LOO_DIR/summary/loo_summary.csv"
