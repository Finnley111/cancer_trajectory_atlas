#!/bin/bash
# Post-processing visualizations for completed baseline runs.
# Generates interactive overlays and patch exports — no GPU required.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_post_processing.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_postproc
#SBATCH --output=logs/post_processing-%j.out

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"

BASELINE_DIRS=(
    "$SCRATCH/results/baseline/atlas_none_harmony"
    "$SCRATCH/results/baseline/atlas_macenko_harmony"
)

echo "============================================"
echo "  Atlas post-processing"
echo "  Job ID: $SLURM_JOB_ID"
echo "============================================"

cd ~

for OUT_DIR in "${BASELINE_DIRS[@]}"; do
    echo ""
    echo "=== Processing: $OUT_DIR ==="

    python -m cancer_trajectory_atlas.visualize.interactive_overlay \
        --results-csv   "$OUT_DIR/results.csv" \
        --png-dir       "$PNG_DIR" \
        --output-dir    "$OUT_DIR/overlays" \
        --patch-size    112

    python -m cancer_trajectory_atlas.visualize.export_patches \
        --results-csv   "$OUT_DIR/results.csv" \
        --png-dir       "$PNG_DIR" \
        --output-dir    "$OUT_DIR/patch_export" \
        --patch-size    112 \
        --n-per-bin     50

    echo "Done: $OUT_DIR"
done

echo ""
echo "============================================"
echo "  POST-PROCESSING COMPLETE"
echo "============================================"
echo ""
echo "Outputs:"
for OUT_DIR in "${BASELINE_DIRS[@]}"; do
    echo "  $OUT_DIR/overlays/"
    echo "  $OUT_DIR/patch_export/"
done
