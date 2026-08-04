#!/bin/bash
# Timepoint cohort: Stage A v2 -- corrected cohort inventory.
#
# Re-runs Stage A's inventory logic (imported, not reimplemented, from
# analysis/timepoint_cohort_inventory.py, which is NOT modified) against the
# TRUE set of usable slides now that Stage C conversion has completed: 22 of
# 24 remaining slides converted successfully, joining the 7 already-converted
# for 29 total. Two data issues Stage C surfaced are reconciled here:
#   - 6041-4L-12W: NEW corrupted slide (OpenSlideError "Restart marker not
#     found" at read_region, same failure mode as 6069-4R-4W). Both excluded.
#   - 60997-4L-4W-2: confirmed (level0 dims + file size match exactly)
#     duplicate/rescan of 6097-4L-4W -- excluded as duplicate, not counted as
#     new data. The determination is verified fresh against this run's real
#     data, not assumed from the earlier partial inventory.
#
# Descriptive only -- does NOT gate PASS/FAIL. Stage B v2
# (run_timepoint_stain_homogeneity_v2.sh) is the hard gate that consumes this
# job's usable_slides output.
#
# NEW, STANDALONE SCRIPT -- distinct from run_timepoint_cohort_inventory.sh
# (Stage A v1, not modified).
#
# Reads (read-only):
#   $SCRATCH/data/timepoint_ndpi/*.ndpi
#   $SCRATCH/data/timepoint_ndpi_deferred/*.ndpi
#   $SCRATCH/data/timepoint_x5_full/*.png            (existence check only)
#
# Writes (NEW directory):
#   $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json
#   $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.md
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_cohort_inventory_v2.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=timepoint_cohort_inventory_v2
#SBATCH --output=logs/timepoint_cohort_inventory_v2-%j.out

# NOTE on --time/--mem: same class of job as Stage A v1 (level-0 dimensions +
# properties only, no read_region, no pixel decode, across ~32 raw NDPIs) plus
# a cheap PNG-existence check (os.path.exists, not a decode) against
# timepoint_x5_full. 1 hour / 32G is a generous margin, not a real estimate
# of need -- unchanged from Stage A v1.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
NDPI_DIR_MAIN="$SCRATCH/data/timepoint_ndpi"
NDPI_DIR_DEFERRED="$SCRATCH/data/timepoint_ndpi_deferred"
CONVERTED_PNG_DIR="$SCRATCH/data/timepoint_x5_full"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageA_inventory_v2"

echo "========================================================"
echo "  Timepoint cohort — Stage A v2: corrected cohort inventory"
echo "  Job ID              : ${SLURM_JOB_ID:-local}"
echo "  NDPI dir (main)      : $NDPI_DIR_MAIN"
echo "  NDPI dir (deferred)  : $NDPI_DIR_DEFERRED"
echo "  Converted PNG dir    : $CONVERTED_PNG_DIR"
echo "  Output dir           : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "NDPI dir (main)     : "; ls -d "$NDPI_DIR_MAIN" 2>/dev/null || echo "NOT FOUND"
if [ -d "$NDPI_DIR_MAIN" ]; then
  echo "  .ndpi file count  : $(find "$NDPI_DIR_MAIN" -maxdepth 1 -iname '*.ndpi' | wc -l)"
fi
echo -n "NDPI dir (deferred) : "; ls -d "$NDPI_DIR_DEFERRED" 2>/dev/null || echo "NOT FOUND"
if [ -d "$NDPI_DIR_DEFERRED" ]; then
  echo "  .ndpi file count  : $(find "$NDPI_DIR_DEFERRED" -maxdepth 1 -iname '*.ndpi' | wc -l)"
fi
echo -n "Converted PNG dir   : "; ls -d "$CONVERTED_PNG_DIR" 2>/dev/null || echo "NOT FOUND — run jobs/run_timepoint_convert_stageC.sh first"
if [ -d "$CONVERTED_PNG_DIR" ]; then
  echo "  .png file count   : $(find "$CONVERTED_PNG_DIR" -maxdepth 1 -iname '*.png' | wc -l) (expect 29)"
fi
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_cohort_inventory_v2 \
    --ndpi-dir           "$NDPI_DIR_MAIN" \
    --ndpi-dir           "$NDPI_DIR_DEFERRED" \
    --converted-png-dir  "$CONVERTED_PNG_DIR" \
    --output-dir         "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE A v2 COHORT INVENTORY COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageA_inventory_v2.json"
echo "  $OUTPUT_DIR/stageA_inventory_v2.md"
echo ""
echo "Review before running Stage B v2: the 60997-4L-4W-2 duplicate"
echo "determination, any UNEXPLAINED gap slides, the per-(mouse, timepoint)"
echo "coverage check (any group dropped to zero?), and the corrected"
echo "counts-by-timepoint / scan-date confound vs the v1 numbers (n=19 mice,"
echo "rho=-0.126, not significant)."
echo ""
