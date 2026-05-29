#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_cache
#SBATCH --output=logs/atlas_cache-%j.out

# Full 16-slide run with feature caching enabled.
# Run this ONCE before submit_loo_array.sh — it populates
# $SCRATCH/data/features_cache/ so the 16 LOO tasks can skip GPU inference.
# This also produces the reference full-run output used by loo_project.py.

mkdir -p logs
mkdir -p $SCRATCH/results/atlas_none_harmony
mkdir -p $SCRATCH/data/features_cache

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Pre-run check ==="
echo "PNG dir:"; ls $SCRATCH/data/MCF7_x5_cropped/*.png | head
echo "Annotations:"; ls ~/cancer_trajectory_atlas/data/annotations_ratio/ | head
echo "===================="

cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir             $SCRATCH/data/MCF7_x5_cropped \
    --annotation-dir      ~/cancer_trajectory_atlas/data/annotations_ratio \
    --output-dir          $SCRATCH/results/atlas_none_harmony \
    --stain-method        none \
    --model               phikon \
    --patch-size          112 \
    --stride              96 \
    --clustering-method   leiden \
    --leiden-resolution   0.5 \
    --harmony \
    --harmony-key         section_number \
    --n-permutations      1000 \
    --features-cache-dir  $SCRATCH/data/features_cache

echo ""
echo "Feature cache populated:"
ls -lh $SCRATCH/data/features_cache/*.npy | head -20
echo ""
echo "Full-run output size:"
du -sh $SCRATCH/results/atlas_none_harmony 2>/dev/null
