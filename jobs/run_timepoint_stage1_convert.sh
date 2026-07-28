#!/bin/bash
# Timepoint projection, Stage 1: convert + inventory (CPU).
#
# 8 additional annotated slides exist at timepoints (4W, 8W) different from the 16
# single-timepoint pipeline slides, from different mice, never run through the
# pipeline. This is the FIRST of a staged, gated pipeline
# (analysis/timepoint_inventory.py docstring has full detail):
#   Stage 1 (this job)      -> convert + inventory
#   Stage 2 (HARD GATE)     -> stain batch check -- STOP and report before Stage 3
#   Stage 3 (GPU, deferred) -> feature extraction + projection onto the existing
#                              2M-1 manifold via the saved AtlasProjector
#   Stage 4 (deferred)      -> timepoint analysis (pre-specified primary test)
#
# PROJECT, DO NOT RETRAIN: these slides are never added to any training cohort.
# This job does not touch any existing pipeline output, manifold, feature cache, or
# results.csv -- it only converts NDPI -> PNG into a brand-new directory (with its
# own, separate slide_dimensions.json sidecar) and writes an inventory report.
#
# Reads:
#   $SCRATCH/data/timepoint_ndpi/*.ndpi           (the 8 new raw NDPI files --
#                                                   PLACE THESE HERE before running)
#   ~/cancer_trajectory_atlas/data/annotations/*.geojson  (existing, read-only --
#                                                          none expected for these
#                                                          8 slides as of writing)
#
# Writes (NEW, separate from the existing pipeline's MCF7_x5_cropped):
#   $SCRATCH/data/timepoint_x5_cropped/*.png
#   $SCRATCH/data/timepoint_x5_cropped/slide_dimensions.json   (separate sidecar)
#   $SCRATCH/results/timepoint_projection/stage1_convert/stage1_inventory.json
#   $SCRATCH/results/timepoint_projection/stage1_convert/stage1_inventory.md
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stage1_convert.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=timepoint_stage1_convert
#SBATCH --output=logs/timepoint_stage1_convert-%j.out

set -euo pipefail
mkdir -p logs

# ── Parameters -- confirm/edit before submitting ─────────────────────────────
NDPI_DIR="$SCRATCH/data/timepoint_ndpi"
PNG_DIR="$SCRATCH/data/timepoint_x5_cropped"
SLIDE_DIMENSIONS="$PNG_DIR/slide_dimensions.json"
ANNOTATION_DIR="$HOME/cancer_trajectory_atlas/data/annotations"
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_timepoint.txt"
NDPI_SCALE=1.0
NDPI_LEVEL=0

OUTPUT_DIR="$SCRATCH/results/timepoint_projection/stage1_convert"

echo "========================================================"
echo "  Timepoint projection — Stage 1: convert + inventory"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  NDPI dir     : $NDPI_DIR"
echo "  PNG dir (NEW): $PNG_DIR"
echo "  Annotations  : $ANNOTATION_DIR"
echo "  Slide list   : $SLIDE_LIST"
echo "  ndpi_scale   : $NDPI_SCALE"
echo "  Output dir   : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo "Expected NDPI files:"
while read -r stem; do
  [ -z "$stem" ] && continue
  echo -n "  $stem.ndpi : "
  ls -lh "$NDPI_DIR/$stem.ndpi" 2>/dev/null || echo "NOT FOUND"
done < "$SLIDE_LIST"
echo "======================"
echo ""

mkdir -p "$PNG_DIR"

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

echo "=== Converting NDPI -> PNG (existing, unmodified run_all.py --convert) ==="
python -m cancer_trajectory_atlas.run_all \
    --convert \
    --ndpi-dir   "$NDPI_DIR" \
    --png-dir    "$PNG_DIR" \
    --ndpi-level "$NDPI_LEVEL" \
    --ndpi-scale "$NDPI_SCALE"

echo ""
echo "=== Stage 1 inventory ==="
python -m cancer_trajectory_atlas.analysis.timepoint_inventory \
    --slide-list       "$SLIDE_LIST" \
    --ndpi-dir         "$NDPI_DIR" \
    --png-dir          "$PNG_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --ndpi-scale       "$NDPI_SCALE" \
    --output-dir       "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE 1 COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $PNG_DIR/*.png, slide_dimensions.json (NEW, separate from MCF7_x5_cropped)"
echo "  $OUTPUT_DIR/stage1_inventory.json"
echo "  $OUTPUT_DIR/stage1_inventory.md"
echo ""
echo "Review stage1_inventory.md before proceeding to Stage 2. Any slide flagged"
echo "for manual review (left-crop assumption) or with an ndpi_scale mismatch"
echo "should be resolved before its converted PNG is trusted."
echo ""
