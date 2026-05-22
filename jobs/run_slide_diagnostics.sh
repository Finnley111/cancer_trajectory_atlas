#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas: slide diagnostic report
#
# Post-hoc diagnostic for an outlier slide in the LOO projection experiment.
# Runs entirely on CPU; no GPU required.
#
# Usage:
#   sbatch run_slide_diagnostics.sh                             # default target + macenko_harmony run
#   sbatch run_slide_diagnostics.sh 6028-4L-2M-2_x5            # explicit target slide
#   sbatch run_slide_diagnostics.sh 6028-4L-2M-2_x5 atlas_none_harmony  # different atlas run

#SBATCH --account=def-lmarti46
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=slide_diagnostics
#SBATCH --output=logs/slide_diagnostics-%j.out

set -euo pipefail

TARGET_SLIDE=${1:-6028-4L-2M-2_x5}
RUN_NAME=${2:-atlas_macenko_harmony}
LOO_DIR=${3:-$SCRATCH/results/loo_summary}

ATLAS_DIR=$SCRATCH/results/$RUN_NAME
ANN_DIR=$SCRATCH/data/annotations
FEATURES_DIR=$SCRATCH/data/features_cache
OUT_DIR=$SCRATCH/results/slide_diagnostics_${TARGET_SLIDE}

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Slide Diagnostic Report"
echo "  Target slide: $TARGET_SLIDE"
echo "  Atlas run:    $ATLAS_DIR"
echo "  LOO dir:      $LOO_DIR"
echo "  Output:       $OUT_DIR"
echo "========================================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Pre-run checks ==="
echo "Atlas adata:";    ls -lh "$ATLAS_DIR/adata_full.h5ad"   2>/dev/null || echo "  NOT FOUND: $ATLAS_DIR/adata_full.h5ad"
echo "Atlas results:";  ls -lh "$ATLAS_DIR/results.csv"        2>/dev/null || echo "  NOT FOUND: $ATLAS_DIR/results.csv"
echo "LOO dir:";        ls "$LOO_DIR/"                  2>/dev/null | head || echo "  NOT FOUND: $LOO_DIR"
echo "Features cache:"; ls "$FEATURES_DIR/"             2>/dev/null | head || echo "  NOT FOUND: $FEATURES_DIR (H3 will be skipped)"
echo "Annotations:";    ls "$ANN_DIR/"                  2>/dev/null | head || echo "  NOT FOUND: $ANN_DIR (H5 will be skipped)"
echo "===================="

cd ~

python -m cancer_trajectory_atlas.analysis.slide_diagnostics \
    --adata-path         "$ATLAS_DIR/adata_full.h5ad" \
    --results-csv        "$ATLAS_DIR/results.csv" \
    --loo-dir            "$LOO_DIR" \
    --features-cache-dir "$FEATURES_DIR" \
    --annotation-dir     "$ANN_DIR" \
    --target-slide       "$TARGET_SLIDE" \
    --output-dir         "$OUT_DIR"

echo ""
echo "Done. Report:"
echo "  $OUT_DIR/slide_diagnostics_report.md"
echo "Figures:"
ls "$OUT_DIR"/*.png 2>/dev/null || echo "  (no PNGs found — check log for errors)"
