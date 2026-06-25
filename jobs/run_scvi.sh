#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas: scVI batch correction run
#
# Escalation from Harmony to scVI for the section_number batch effect.
# Harmony k-NN batch purity was 0.9425 vs a 0.502 chance baseline (see
# PROJECT_STATE.md Working Log, 2026-06-24) — only ~11% of the segregation
# gap closed. This run tests whether scVI's nonlinear correction does
# meaningfully better while preserving the Phase 6 morphological signal.
#
# Reuses the existing unsampled no-stain features_cache via
# --features-cache-dir — NO feature rebuild, no GPU inference needed for
# Phikon. GPU is requested here for scVI's own VAE training.
#
# PREREQUISITE: scvi-tools must be installed in ~/envs/atlas
# (pip install scvi-tools on a login node — needs internet, compute nodes
# don't have it). Confirm `python -c "import scvi"` succeeds with a
# CUDA-capable torch build before submitting this job.
#
# Usage:
#   sbatch run_scvi.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_scvi
#SBATCH --output=logs/atlas_scvi-%j.out

set -euo pipefail

OUT_NAME="atlas_none_scvi"
CACHE_DIR=$SCRATCH/data/features_cache

PNG_DIR=$SCRATCH/data/MCF7_x5_cropped
ANN_DIR=~/cancer_trajectory_atlas/data/annotations_ratio
OUT_DIR=$SCRATCH/results/$OUT_NAME

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Atlas scVI Run"
echo "  Stain method:        none"
echo "  Batch method:        scvi"
echo "  Feature cache dir:   $CACHE_DIR  (reusing existing unsampled cache)"
echo "  Output dir:          $OUT_DIR"
echo "========================================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -c "import scvi; print('scvi-tools OK, version', scvi.__version__)" || {
    echo "ERROR: scvi-tools not importable — run: pip install scvi-tools"
    exit 1
}

cd ~

python -m cancer_trajectory_atlas.run_all --run    \
    --png-dir                "$PNG_DIR"             \
    --annotation-dir         "$ANN_DIR"             \
    --output-dir             "$OUT_DIR"             \
    --stain-method           none                   \
    --batch-method           scvi                   \
    --model                  phikon                 \
    --patch-size              112                   \
    --stride                  96                    \
    --clustering-method      leiden                 \
    --leiden-resolution      0.5                    \
    --n-permutations         1000                   \
    --features-cache-dir     "$CACHE_DIR"

echo ""
echo "Done. Results in $OUT_DIR"
echo ""
echo "Run the batch-mixing diagnostic on this result with:"
echo "  python -m cancer_trajectory_atlas.analysis.run_batch_mixing $OUT_DIR"
echo "Run the section UMAP diagnostic on this result with:"
echo "  python ~/cancer_trajectory_atlas/analysis/plot_umap_by_section.py $OUT_DIR"
