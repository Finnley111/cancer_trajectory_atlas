#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_loo_scvi
#SBATCH --output=logs/loo_scvi-%A_%a.out
# Array size is set below by the --array flag.
#
# scVI variant of submit_loo_array.sh (which is unchanged). Adds --gres=gpu:1
# since each fold now trains a scVI VAE in Phase A, not just Harmony (CPU-only).
#
# RECOMMENDED: run array index 0 alone first to get real per-fold timing
# before committing to all 16 GPU allocations — we don't yet have a
# completed scVI run's wall-clock time (the first attempt crashed at epoch 1
# on a NaN bug, since fixed in analysis/scvi_integration.py).
#   sbatch --array=0 jobs/submit_loo_array_scvi.sh
# Then, once happy with timing:
#   sbatch --array=0-15 jobs/submit_loo_array_scvi.sh
#
# Prerequisites:
#   1. $SCRATCH/results/atlas_none_scvi/results.csv must exist (the full
#      16-slide scVI reference run) for Phase B in-manifold comparison.
#   2. data/loo_slides.txt must contain all 16 slide stems (one per line) —
#      same file the Harmony LOO array job uses.
#   3. $SCRATCH/data/features_cache/ must contain one .npy per slide.
#
# After all 16 tasks complete, aggregate results:
#   python -m cancer_trajectory_atlas.analysis.loo_summary_scvi \
#       --loo-dirs $SCRATCH/results/loo_*_scvi \
#       --full-run-dir $SCRATCH/results/atlas_none_scvi \
#       --output-dir $SCRATCH/results/atlas_none_scvi/loo_scvi

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -c "import scvi; print('scvi-tools OK, version', scvi.__version__)" || {
    echo "ERROR: scvi-tools not importable — run: pip install scvi-tools"
    exit 1
}

# Read slide list and pick this task's held-out slide by array index
SLIDES_FILE="$HOME/cancer_trajectory_atlas/data/loo_slides.txt"
if [[ ! -f "$SLIDES_FILE" ]]; then
    echo "ERROR: $SLIDES_FILE not found."
    echo "Create it with one slide stem per line, matching files in $SCRATCH/data/MCF7_x5_cropped/"
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set."
    echo "Submit with: sbatch --array=0-15 jobs/submit_loo_array_scvi.sh"
    exit 1
fi

mapfile -t SLIDES < "$SLIDES_FILE"

if [[ $SLURM_ARRAY_TASK_ID -ge ${#SLIDES[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#SLIDES[@]} slides."
    exit 1
fi

export HELD_OUT="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
export FULL_RUN="${FULL_RUN:-$SCRATCH/results/atlas_none_scvi}"
export CACHE_DIR="${CACHE_DIR:-$SCRATCH/data/features_cache}"
export LOO_SUFFIX="${LOO_SUFFIX:-_scvi}"

echo "Task $SLURM_ARRAY_TASK_ID / $((${#SLIDES[@]}-1)) — held-out: $HELD_OUT"

cd ~
bash ~/cancer_trajectory_atlas/jobs/run_loo_single_scvi.sh
