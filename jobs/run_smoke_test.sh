#!/bin/bash
# SLURM job script — end-to-end smoke test (2 slides, 50-patch cap, no Harmony).
# Verifies the full code path: patch extraction → atlas build → LOO projection.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_smoke_test.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_smoke_test
#SBATCH --output=logs/smoke_test-%j.out

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

SLIDE1="6027-4L-2M-1_x5"
SLIDE2="6028-4L-2M-1_x5"
SMOKE_DIR="$SCRATCH/results/smoke_${SLURM_JOB_ID}"
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
CAP=50

echo "=== SMOKE TEST START ==="
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Output:    $SMOKE_DIR"
echo "  Slides:    $SLIDE1  +  $SLIDE2"
echo "  Cap:       $CAP patches/slide"
echo "  Cache:     $CACHE_DIR"
echo "========================="

cd ~

# ── Step 1: Full 2-slide reference run ───────────────────────────────────────
# Exercises: patch extraction, Phikon (cache hit), PCA, UMAP, Leiden,
#            diffusion pseudotime, morphological validation, projector save.
echo ""
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
    --features-cache-dir    /scratch/finnley1/data/smoke_test_cache

# ── Step 2: LOO training run (train on SLIDE1 only) ──────────────────────────
# Exercises: single-slide atlas path, projector serialisation.
echo ""
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
    --features-cache-dir    /scratch/finnley1/data/smoke_test_cache

# ── Step 3: LOO Phase B — project SLIDE2 onto 1-slide manifold ───────────────
# Exercises: AtlasProjector.load(), project(method="knn"),
#            Spearman rho against ref results.csv, plot output.
# Cap must match Step 1 so patch counts align in the loo_result comparison.
echo ""
echo "--- Step 3: LOO projection (knn) ---"
python -m cancer_trajectory_atlas.analysis.loo_project \
    --projector-dir         "$SMOKE_DIR/loo_train/projector" \
    --held-out-slide        "$SLIDE2" \
    --cache-dir             "/scratch/finnley1/data/smoke_test_cache" \
    --full-run-dir          "$SMOKE_DIR/ref" \
    --output-dir            "$SMOKE_DIR/loo_train" \
    --max-patches-per-slide $CAP \
    --patch-sample-seed     42

echo ""
echo "=== SMOKE TEST PASSED ==="
echo ""
echo "Outputs at: $SMOKE_DIR"
echo "  Step 1 atlas:     $SMOKE_DIR/ref/adata_full.h5ad"
echo "  Step 2 projector: $SMOKE_DIR/loo_train/projector/"
echo "  Step 3 result:    $SMOKE_DIR/loo_train/loo_result_${SLIDE2}.json"
