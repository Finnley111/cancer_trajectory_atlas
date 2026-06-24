#!/bin/bash
# SLURM job script — UMAP-by-section diagnostic (pure plotting, no pipeline rerun)
#
# Loads an existing run's adata.h5ad and saves a UMAP scatter colored by
# section_number, to check whether Harmony actually interspersed 2M-1/2M-2
# or whether the two sections are still spatially segregated.
#
# Usage:
#   sbatch run_umap_by_section.sh [run_dir]
#
# $1 = run directory containing adata.h5ad
#      (default: $SCRATCH/results/runs_paga/all_harmony/full)
#
# Can also be run directly on a login node without sbatch:
#   python ~/cancer_trajectory_atlas/analysis/plot_umap_by_section.py <run_dir>

#SBATCH --account=def-lmarti46
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=umap_by_section
#SBATCH --output=logs/umap_by_section-%j.out

set -euo pipefail

RUN_DIR=${1:-$SCRATCH/results/runs_paga/all_harmony/full}

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

python ~/cancer_trajectory_atlas/analysis/plot_umap_by_section.py "$RUN_DIR"
