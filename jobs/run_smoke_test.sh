#!/bin/bash
# End-to-end smoke test: 2 slides, 50-patch cap, no Harmony.
# Prereqs: feature cache populated for SLIDE1 and SLIDE2.
#
# Run interactively (not sbatch):
#   salloc --account=def-lmarti46 --gres=gpu:1 --mem=16G --time=0:30:00
#   bash ~/cancer_trajectory_atlas/jobs/run_smoke_test.sh
set -euo pipefail

source ~/envs/atlas/bin/activate
cd ~

SLIDE1="6027-4L-2M-1_x5"
SLIDE2="6028-4L-2M-1_x5"
SMOKE_DIR="$SCRATCH/results/smoke_$(date +%Y%m%d_%H%M%S)"
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
CAP=50

echo "=== SMOKE TEST START: $SMOKE_DIR ==="

# ── Step 1: Full 2-slide reference run ───────────────────────────────────────
# Exercises: patch extraction, Phikon (cache hit), PCA, UMAP, Leiden,
#            diffusion pseudotime, morphological validation, projector save.
echo "--- Step 1: Reference run (2 slides) ---"
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$SMOKE_DIR/ref" \
    --slides                "$SLIDE1,$SLIDE2" \
    --stain-method          none \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.3 \
    --max-patches-per-slide $CAP \
    --n-permutations        10 \
    --diffmap-neighbors     10 \
    --diffmap-comps         5 \
    --features-cache-dir    "$CACHE_DIR"

# ── Step 2: LOO training run (train on SLIDE1 only) ──────────────────────────
# Exercises: single-slide atlas path, projector serialisation.
echo "--- Step 2: LOO training (1 slide) ---"
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$SMOKE_DIR/loo_train" \
    --slides                "$SLIDE1" \
    --stain-method          none \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.3 \
    --max-patches-per-slide $CAP \
    --n-permutations        10 \
    --diffmap-neighbors     10 \
    --diffmap-comps         5 \
    --features-cache-dir    "$CACHE_DIR"

# ── Step 3: LOO Phase B — project SLIDE2 onto 1-slide manifold ───────────────
# Exercises: AtlasProjector.load(), project(method="knn"),
#            Spearman rho against ref results.csv, plot output.
# Cap must match Step 1 so patch counts align in loo_result comparison.
echo "--- Step 3: LOO projection (knn) ---"
python -m cancer_trajectory_atlas.analysis.loo_project \
    --projector-dir         "$SMOKE_DIR/loo_train/projector" \
    --held-out-slide        "$SLIDE2" \
    --cache-dir             "$CACHE_DIR" \
    --full-run-dir          "$SMOKE_DIR/ref" \
    --output-dir            "$SMOKE_DIR/loo_train" \
    --max-patches-per-slide $CAP \
    --patch-sample-seed     42

echo ""
echo "=== SMOKE TEST PASSED ==="
echo "Outputs at: $SMOKE_DIR"
echo ""
echo "Verify:"
echo "  ls $SMOKE_DIR/ref/adata_full.h5ad"
echo "  ls $SMOKE_DIR/loo_train/projector/"
echo "  cat $SMOKE_DIR/loo_train/loo_result_${SLIDE2}.json"
