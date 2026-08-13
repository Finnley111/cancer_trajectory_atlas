#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas QC diagnostics (steps 1-4)
#
# Usage:
#   sbatch cancer_trajectory_atlas/jobs/submit_qc.sh atlas_full_reinhard reinhard
#   sbatch cancer_trajectory_atlas/jobs/submit_qc.sh atlas_full_macenko  macenko
#
# Arguments:
#   $1 = run name (directory under $SCRATCH/results/)
#   $2 = stain method: reinhard | macenko | none  (default: reinhard)

#SBATCH --account=def-lmarti46
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=atlas_qc
#SBATCH --output=logs/atlas_qc-%j.out

set -euo pipefail

RUN_NAME=${1:-atlas_full_reinhard}
STAIN_METHOD=${2:-reinhard}

RUN_DIR=$SCRATCH/results/$RUN_NAME
SLIDES_DIR=$SCRATCH/data/MCF7_x5_cropped

mkdir -p logs
mkdir -p "$RUN_DIR/qc"

echo "========================================"
echo "Atlas QC — $RUN_NAME  (stain: $STAIN_METHOD)"
echo "Run dir:    $RUN_DIR"
echo "Slides dir: $SLIDES_DIR"
echo "========================================"

# Standard atlas environment (same modules every pipeline job loads)
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

# HF offline (in case QC code imports anything that loads phikon at import time)
export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Sanity check the brittle deps before doing real work
python -c "import staintools, spams; print('staintools + spams OK')" || {
    echo "ERROR: staintools or spams not importable"
    exit 1
}

cd ~

# Run all four QC steps
python -m cancer_trajectory_atlas.qc.run_qc \
    --run-dir          "$RUN_DIR"      \
    --slides-dir       "$SLIDES_DIR"   \
    --stain-method     "$STAIN_METHOD" \
    --steps            1234            \
    --patch-size       112             \
    --n-contact-patches 25

echo "Done. QC outputs in $RUN_DIR/qc/"