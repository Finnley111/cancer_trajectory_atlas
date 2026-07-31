#!/bin/bash
# Timepoint cohort: Stage B -- within-cohort stain homogeneity gate (HARD GATE).
#
# Tests whether staining is homogeneous ACROSS TIMEPOINT GROUPS (4W/7W/8W/12W)
# within the timepoint cohort itself -- NOT vs. the original 2M-1 cohort (that
# cross-cohort comparison is CANCELLED; see timepoint_stage2_stain_check.py). If
# sub-batches within this cohort align with timepoint, the within-cohort design
# fails for the same reason, and the project stops here.
#
# Runs on the COARSEST NDPI pyramid level (cheap), not full-resolution PNGs, to
# avoid committing to converting ~30 slides before this gate passes. First
# validates that shortcut against the 7 slides that already have a full-width
# PNG (timepoint_x5_full) -- if the coarse-level proxy doesn't agree with the
# PNG-based stats, this job HALTS (non-zero exit) rather than trusting an
# unvalidated proxy for the remaining slides.
#
# Depends on:
#   - run_timepoint_cohort_inventory.sh (Stage A) having already run
#   - jobs/run_stage2_reference_threshold.sh's output (reference_rank_biserial)
#   - the 7 already-converted PNGs at $SCRATCH/data/timepoint_x5_full
#
# Reads (read-only):
#   $SCRATCH/results/timepoint_cohort/stageA_inventory/stageA_inventory.json
#   $SCRATCH/data/timepoint_x5_full/*.png
#   raw .ndpi files referenced by Stage A's inventory (both timepoint dirs)
#   $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json
#
# Writes (NEW directory):
#   $SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity/stageB_stain_homogeneity.json
#   $SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity/stageB_stain_homogeneity.md
#
# STOP AFTER THIS JOB. Report stageB_stain_homogeneity.md and await explicit
# confirmation before Stage C (conversion of the remaining slides) is written,
# let alone submitted.
#
# Usage (after Stage A and run_stage2_reference_threshold.sh have both
# completed and been reviewed):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stain_homogeneity.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=timepoint_stain_homogeneity
#SBATCH --output=logs/timepoint_stain_homogeneity-%j.out

# NOTE on --mem=64G: coarse-level NDPI reads are cheap, but
# _deconvolve_hematoxylin (skimage rgb2hed) still upcasts its input to float64
# internally per slide -- mirrors run_stage2_reference_threshold.sh's
# known-safe 64G setting for the same underlying computation.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
STAGEA_INVENTORY_JSON="$SCRATCH/results/timepoint_cohort/stageA_inventory/stageA_inventory.json"
CONVERTED_PNG_DIR="$SCRATCH/data/timepoint_x5_full"
REFERENCE_THRESHOLD_JSON="$SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity"

echo "========================================================"
echo "  Timepoint cohort — Stage B: within-cohort stain homogeneity gate"
echo "  Job ID                   : ${SLURM_JOB_ID:-local}"
echo "  Stage A inventory JSON    : $STAGEA_INVENTORY_JSON"
echo "  Converted PNG dir (7 slides) : $CONVERTED_PNG_DIR"
echo "  Reference threshold JSON  : $REFERENCE_THRESHOLD_JSON"
echo "  Output dir                : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Stage A inventory JSON : "; ls -lh "$STAGEA_INVENTORY_JSON" 2>/dev/null || echo "NOT FOUND — run jobs/run_timepoint_cohort_inventory.sh first"
echo -n "Converted PNG dir      : "; ls -d "$CONVERTED_PNG_DIR" 2>/dev/null || echo "NOT FOUND"
echo -n "Reference threshold JSON : "; ls -lh "$REFERENCE_THRESHOLD_JSON" 2>/dev/null || echo "NOT FOUND — run jobs/run_stage2_reference_threshold.sh first"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_stain_homogeneity \
    --stageA-inventory-json    "$STAGEA_INVENTORY_JSON" \
    --converted-png-dir        "$CONVERTED_PNG_DIR" \
    --reference-threshold-json "$REFERENCE_THRESHOLD_JSON" \
    --output-dir                "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE B STAIN HOMOGENEITY GATE COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageB_stain_homogeneity.json"
echo "  $OUTPUT_DIR/stageB_stain_homogeneity.md"
echo ""
echo "*** STOP HERE. Do not write or submit any Stage C job. ***"
echo "Report stageB_stain_homogeneity.md and await explicit confirmation before"
echo "Stage C (conversion of the remaining slides) is designed or run."
echo ""
