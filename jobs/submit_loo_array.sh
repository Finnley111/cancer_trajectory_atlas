#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=atlas_loo
#SBATCH --output=logs/loo-%A_%a.out
# Array size is set below by the --array flag.
# Submit with: sbatch --array=0-15 submit_loo_array.sh
#
# Prerequisites:
#   1. Run jobs/run_cache_population.sh and wait for it to finish.
#   2. Populate data/loo_slides.txt with all 16 slide stems (one per line).
#   3. Confirm $SCRATCH/data/features_cache/ contains one .npy per slide.
#
# After all 16 tasks complete, aggregate results:
#   python -m cancer_trajectory_atlas.analysis.loo_summary \
#       --loo-dirs $SCRATCH/results/loo_* \
#       --output-dir $SCRATCH/results/loo_summary

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Read slide list and pick this task's held-out slide by array index
SLIDES_FILE="$HOME/cancer_trajectory_atlas/data/loo_slides.txt"
if [[ ! -f "$SLIDES_FILE" ]]; then
    echo "ERROR: $SLIDES_FILE not found."
    echo "Create it with one slide stem per line, matching files in $SCRATCH/data/MCF7_x5_cropped/"
    exit 1
fi

mapfile -t SLIDES < "$SLIDES_FILE"

if [[ $SLURM_ARRAY_TASK_ID -ge ${#SLIDES[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#SLIDES[@]} slides."
    exit 1
fi

export HELD_OUT="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
export FULL_RUN="${FULL_RUN:-$SCRATCH/results/atlas_none_harmony}"
export CACHE_DIR="${CACHE_DIR:-$SCRATCH/data/features_cache}"

echo "Task $SLURM_ARRAY_TASK_ID / $((${#SLIDES[@]}-1)) — held-out: $HELD_OUT"

cd ~
bash ~/cancer_trajectory_atlas/jobs/run_loo_single.sh
