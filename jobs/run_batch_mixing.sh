#!/bin/bash
# SLURM job script — batch-mixing diagnostic (pure post-hoc analysis, no pipeline rerun)
#
# Loads an existing run's adata_full.h5ad and computes a quantitative kNN
# section-purity score on obsm['X_pca_original'] (raw PCA, pre-Harmony) vs.
# obsm['X_pca_harmony'] (post-Harmony) — the same representation that feeds
# the Leiden neighbor graph. Writes batch_mixing.json to the run directory.
#
# Usage:
#   sbatch run_batch_mixing.sh [run_dir]
#
# $1 = run directory containing adata_full.h5ad
#      (default: $SCRATCH/results/runs_paga/all_harmony/full)
#
# Can also be run directly on a login node without sbatch (from ~):
#   python -m cancer_trajectory_atlas.analysis.run_batch_mixing <run_dir>

#SBATCH --account=def-lmarti46
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=batch_mixing
#SBATCH --output=logs/batch_mixing-%j.out

set -euo pipefail

RUN_DIR=${1:-$SCRATCH/results/runs_paga/all_harmony/full}

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.run_batch_mixing "$RUN_DIR"
