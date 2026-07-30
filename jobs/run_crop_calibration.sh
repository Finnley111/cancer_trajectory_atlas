#!/bin/bash
# Crop calibration: read-only diagnostic phase, hard gate before re-conversion.
#
# Stage 1's left/right HSV tissue-fraction diagnostic flagged all 7 converted
# timepoint slides for manual review (ratios 0.30-1.05, not near-zero). That
# diagnostic has never been run on the ORIGINAL 16 pipeline slides, which are
# documented as using the exact same "two copies side by side" NDPI layout --
# so there's no baseline to say whether 0.30-1.05 is actually anomalous. This
# job establishes that baseline (Task A) and adds an independent view via
# embedded macro/label images (Task B).
#
# HARD GATE: this job does NOT convert any NDPI, does NOT modify run_all.py,
# and does NOT touch any existing PNG or slide_dimensions.json. It only reads
# NDPI coarsest-pyramid-level data and small embedded associated images, and
# writes a report. The decision about whether/how to re-convert any slide is
# made AFTER reviewing this job's output, not by this job.
#
# Reads (all read-only):
#   $SCRATCH/data/MCF7_x5/*.ndpi                      (16 original slides --
#     NOTE: paths.json's "raw_ndpi" says ~/scratch/data/ndpi, which does NOT
#     match this, the actual proven-working convention already used by
#     jobs/convert_ndpi.sh -- same discrepancy class as the documented
#     annotations/annotations_ratio split in PROJECT_STATE.md Issue 5)
#   ~/cancer_trajectory_atlas/jobs/slides_section1.txt, slides_section2.txt
#   $SCRATCH/data/timepoint_ndpi/*.ndpi                (7 converted timepoint slides)
#   $SCRATCH/data/timepoint_ndpi_deferred/*.ndpi       (6069-4R-4W, set aside)
#   ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt
#   $SCRATCH/results/timepoint_projection/stage1_convert/stage1_inventory.json
#
# Writes (NEW directory only):
#   $SCRATCH/results/crop_calibration/crop_calibration_report.md
#   $SCRATCH/results/crop_calibration/crop_calibration.json
#   $SCRATCH/results/crop_calibration/macro_images/{original,timepoint}/*.png
#   $SCRATCH/results/crop_calibration/contact_sheet_{original,timepoint}.png
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_crop_calibration.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=crop_calibration
#SBATCH --output=logs/crop_calibration-%j.out

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
ORIGINAL_NDPI_DIR="$SCRATCH/data/MCF7_x5"
ORIGINAL_SLIDE_LIST_1="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
ORIGINAL_SLIDE_LIST_2="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"

TIMEPOINT_NDPI_DIR="$SCRATCH/data/timepoint_ndpi"
TIMEPOINT_NDPI_DEFERRED_DIR="$SCRATCH/data/timepoint_ndpi_deferred"
TIMEPOINT_SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_timepoint.txt"

STAGE1_INVENTORY_JSON="$SCRATCH/results/timepoint_projection/stage1_convert/stage1_inventory.json"
OUTPUT_DIR="$SCRATCH/results/crop_calibration"

echo "========================================================"
echo "  Crop calibration — read-only diagnostic, hard gate"
echo "  Job ID              : ${SLURM_JOB_ID:-local}"
echo "  Original NDPI dir    : $ORIGINAL_NDPI_DIR"
echo "  Original slide lists : $ORIGINAL_SLIDE_LIST_1, $ORIGINAL_SLIDE_LIST_2"
echo "  Timepoint NDPI dir   : $TIMEPOINT_NDPI_DIR (+ deferred: $TIMEPOINT_NDPI_DEFERRED_DIR)"
echo "  Stage 1 inventory    : $STAGE1_INVENTORY_JSON"
echo "  Output dir           : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Original NDPI count (expect 16)   : "; ls "$ORIGINAL_NDPI_DIR"/*.ndpi 2>/dev/null | wc -l
echo -n "Timepoint NDPI count (expect 7)   : "; ls "$TIMEPOINT_NDPI_DIR"/*.ndpi 2>/dev/null | wc -l
echo -n "Timepoint deferred count (expect 1, 6069-4R-4W): "; ls "$TIMEPOINT_NDPI_DEFERRED_DIR"/6069-4R-4W.ndpi 2>/dev/null | wc -l
echo -n "Stage 1 inventory JSON            : "; ls -lh "$STAGE1_INVENTORY_JSON" 2>/dev/null || echo "NOT FOUND -- run Stage 1 first"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.crop_calibration \
    --original-ndpi-dir           "$ORIGINAL_NDPI_DIR" \
    --original-slide-lists        "$ORIGINAL_SLIDE_LIST_1" "$ORIGINAL_SLIDE_LIST_2" \
    --timepoint-ndpi-dir          "$TIMEPOINT_NDPI_DIR" \
    --timepoint-ndpi-deferred-dir "$TIMEPOINT_NDPI_DEFERRED_DIR" \
    --timepoint-slide-list        "$TIMEPOINT_SLIDE_LIST" \
    --stage1-inventory-json       "$STAGE1_INVENTORY_JSON" \
    --output-dir                  "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  CROP CALIBRATION COMPLETE — HARD GATE, STOP HERE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/crop_calibration_report.md"
echo "  $OUTPUT_DIR/crop_calibration.json"
echo "  $OUTPUT_DIR/macro_images/original/*.png"
echo "  $OUTPUT_DIR/macro_images/timepoint/*.png"
echo "  $OUTPUT_DIR/contact_sheet_original.png"
echo "  $OUTPUT_DIR/contact_sheet_timepoint.png"
echo ""
echo "*** STOP HERE. No NDPI has been converted, no PNG or sidecar modified. ***"
echo "Review crop_calibration_report.md's Task A verdict before deciding"
echo "whether/how to re-convert any slide."
echo ""
