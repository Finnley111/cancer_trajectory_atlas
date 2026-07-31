#!/bin/bash
# Timepoint cohort: Stage A -- cohort inventory.
#
# The cross-cohort experiment (timepoint slides projected onto 2M-1, compared
# against 2M-1) is CANCELLED -- its own hard gate found the timepoint batch
# confounded with staining relative to 2M-1, not identifiable by any
# correction. Replacement design: a WITHIN-timepoint-cohort comparison across
# four timepoints (4W/7W/8W/12W) drawn from ~30 slides across both timepoint
# NDPI directories. This job builds the FIRST real inventory of that cohort --
# no manifest for it exists anywhere in this repo; the raw NDPI files live only
# on this cluster's $SCRATCH.
#
# Descriptive only -- does NOT gate PASS/FAIL. Stage B
# (run_timepoint_stain_homogeneity.sh) is the hard gate that consumes this
# job's output.
#
# NEW, STANDALONE SCRIPT -- distinct from run_timepoint_convert_nocrop.sh and
# from the (different, already-superseded) analysis/timepoint_inventory.py
# module belonging to the cancelled cross-cohort experiment's own Stage 1.
#
# Reads (read-only):
#   $SCRATCH/data/timepoint_ndpi/*.ndpi
#   $SCRATCH/data/timepoint_ndpi_deferred/*.ndpi
#
# Writes (NEW directory):
#   $SCRATCH/results/timepoint_cohort/stageA_inventory/stageA_inventory.json
#   $SCRATCH/results/timepoint_cohort/stageA_inventory/stageA_inventory.md
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_cohort_inventory.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=timepoint_cohort_inventory
#SBATCH --output=logs/timepoint_cohort_inventory-%j.out

# NOTE on --time/--mem: this reads level-0 dimensions + properties only (no
# read_region, no pixel decode) across ~30 slides -- the same class of job as
# the existing MPP pre-flight check in timepoint_convert_nocrop.py, which runs
# in seconds per slide. 1 hour / 32G is a generous margin, not a real estimate
# of need.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
NDPI_DIR_MAIN="$SCRATCH/data/timepoint_ndpi"
NDPI_DIR_DEFERRED="$SCRATCH/data/timepoint_ndpi_deferred"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageA_inventory"

echo "========================================================"
echo "  Timepoint cohort — Stage A: cohort inventory"
echo "  Job ID          : ${SLURM_JOB_ID:-local}"
echo "  NDPI dir (main)  : $NDPI_DIR_MAIN"
echo "  NDPI dir (deferred) : $NDPI_DIR_DEFERRED"
echo "  Output dir        : $OUTPUT_DIR"
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
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_cohort_inventory \
    --ndpi-dir    "$NDPI_DIR_MAIN" \
    --ndpi-dir    "$NDPI_DIR_DEFERRED" \
    --output-dir  "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE A COHORT INVENTORY COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageA_inventory.json"
echo "  $OUTPUT_DIR/stageA_inventory.md"
echo ""
echo "This stage is descriptive only. Review the counts-by-timepoint, the mouse"
echo "6072 dual-timepoint check, the scan-date confound check, and any parse or"
echo "OpenSlide failures in the per-slide table before running Stage B"
echo "(run_timepoint_stain_homogeneity.sh), which reads this output."
echo ""
