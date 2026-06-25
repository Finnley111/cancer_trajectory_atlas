#!/bin/bash
# Post-processing for the scVI atlas run.
# Runs interactive pseudotime overlays, patch exports, and the scVI-specific
# post-processing analysis (section-mixing figures, morphology table, leiden_lowres).
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_post_processing_scvi.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_postproc_scvi
#SBATCH --output=logs/post_processing_scvi-%j.out

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
SCVI_RUN_DIR="$SCRATCH/results/atlas_none_scvi"
N_PER_BIN=50
LEIDEN_RES=0.4

BASELINE_DIRS=(
    "$SCVI_RUN_DIR"
)

echo "============================================"
echo "  Atlas scVI post-processing"
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
        --n-per-bin     "$N_PER_BIN"

    echo "Done: $OUT_DIR"
done

echo ""
echo "=== scVI-specific post-processing ==="

python -m cancer_trajectory_atlas.visualize.scvi_postprocess \
    --run-dir           "$SCVI_RUN_DIR" \
    --output-dir        "$SCVI_RUN_DIR/postprocess" \
    --leiden-resolution "$LEIDEN_RES"

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
echo "  $SCVI_RUN_DIR/postprocess/"
