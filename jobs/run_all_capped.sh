#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas: Macenko + Harmony + per-slide patch cap
#
# Identical to submit_harmony_macenko.sh except patches are capped at MAX_PATCHES per slide
# (default 1900, the cohort median) to prevent over-represented slides from dominating the
# pooled manifold.  Uses a separate feature cache directory to avoid contaminating the
# full-run cache.
#
# Usage:
#   sbatch run_all_capped.sh                  # 1900-patch cap, section_number key
#   sbatch run_all_capped.sh 1500             # custom cap
#   sbatch run_all_capped.sh 1900 slide_id    # custom harmony key

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_macenko_harmony_capped
#SBATCH --output=logs/atlas_macenko_harmony_capped-%j.out

set -euo pipefail

MAX_PATCHES=${1:-1900}
HARMONY_KEY=${2:-section_number}
SAMPLE_SEED=42

OUT_NAME="atlas_macenko_harmony_cap${MAX_PATCHES}"
CACHE_DIR=$SCRATCH/data/features_cache_cap${MAX_PATCHES}

PNG_DIR=$SCRATCH/data/MCF7_x5_cropped
ANN_DIR=~/cancer_trajectory_atlas/data/annotations
OUT_DIR=$SCRATCH/results/$OUT_NAME

mkdir -p logs
mkdir -p "$OUT_DIR"
mkdir -p "$CACHE_DIR"

echo "========================================"
echo "Atlas Harmony Run (capped)"
echo "  Stain method:        macenko"
echo "  Harmony key:         $HARMONY_KEY"
echo "  Max patches/slide:   $MAX_PATCHES"
echo "  Sample seed:         $SAMPLE_SEED"
echo "  Feature cache dir:   $CACHE_DIR"
echo "  Output dir:          $OUT_DIR"
echo "========================================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -c "import staintools, spams; print('staintools + spams OK')" || {
    echo "ERROR: staintools or spams not importable"; exit 1
}
python -c "import harmonypy; print('harmonypy OK')" || {
    echo "ERROR: harmonypy not importable — run: pip install harmonypy"
    exit 1
}

cd ~

python -m cancer_trajectory_atlas.run_all --run    \
    --png-dir                "$PNG_DIR"             \
    --annotation-dir         "$ANN_DIR"             \
    --output-dir             "$OUT_DIR"             \
    --stain-method           macenko                \
    --harmony                                       \
    --harmony-key            "$HARMONY_KEY"         \
    --model                  phikon                 \
    --patch-size             112                    \
    --stride                 96                     \
    --clustering-method      leiden                 \
    --leiden-resolution      0.5                    \
    --n-permutations         1000                   \
    --features-cache-dir     "$CACHE_DIR"           \
    --max-patches-per-slide  "$MAX_PATCHES"         \
    --patch-sample-seed      "$SAMPLE_SEED"

echo ""
echo "=== Post-processing: overlays and patch exports ==="

python -m cancer_trajectory_atlas.visualize.interactive_overlay \
    --results-csv  "$OUT_DIR/results.csv" \
    --png-dir      "$PNG_DIR" \
    --output-dir   "$OUT_DIR/overlays" \
    --patch-size   112

python -m cancer_trajectory_atlas.visualize.export_patches \
    --results-csv  "$OUT_DIR/results.csv" \
    --png-dir      "$PNG_DIR" \
    --output-dir   "$OUT_DIR/patch_export" \
    --patch-size   112 \
    --n-per-bin    50

echo ""
echo "Done. Results in $OUT_DIR"
echo "  Sampling log → $OUT_DIR/patch_sampling_log.csv"
echo "  Sampling idx → $OUT_DIR/sampling/"
echo "  Overlays     → $OUT_DIR/overlays/"
echo "  Patches      → $OUT_DIR/patch_export/"
echo ""
echo "Run LOO with this cap:"
echo "  MAX_PATCHES=$MAX_PATCHES FULL_RUN=$OUT_DIR CACHE_DIR=$CACHE_DIR sbatch jobs/submit_loo_array.sh"
