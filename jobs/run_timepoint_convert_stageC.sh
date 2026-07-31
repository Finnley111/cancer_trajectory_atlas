#!/bin/bash
# Timepoint cohort: Stage C -- conversion of the remaining timepoint slides.
#
# Reuses analysis/timepoint_convert_nocrop.py COMPLETELY UNMODIFIED (same
# module already used for the first 7 slides) -- this job only supplies a
# different --slide-list and --ndpi-dir. No new Python code was needed.
#
# Per Stage A's cohort inventory (stageA_inventory.json), the 32-slide
# timepoint cohort splits as:
#   - 7 slides already converted (jobs/slides_timepoint.txt), all from
#     $SCRATCH/data/timepoint_ndpi -- untouched by this job.
#   - 24 remaining slides (jobs/slides_timepoint_remaining.txt), ALL of which
#     live in $SCRATCH/data/timepoint_ndpi_deferred -- this job converts
#     exactly those 24, into the SAME output directory as the first 7
#     (timepoint_x5_full is keyed by stem and idempotent/skip-if-exists per
#     slide, so this is purely additive -- nothing from the first 7 is
#     touched or re-encoded).
#   - 1 slide ("60997-4L-4W-2") deliberately EXCLUDED from this run: Stage A
#     found it fails filename parsing, but its file size, level-0 dimensions,
#     and embedded scan timestamp are IDENTICAL to 6097-4L-4W -- almost
#     certainly the same file under a corrupted/duplicate filename, not new
#     data. Confirm its identity before ever including it in a slide list.
#
# On 6069-4R-4W: an earlier attempt (this session, different location:
# $SCRATCH/data/timepoint_ndpi) found this stem CORRUPTED at the OpenSlide
# read_region level ("Restart marker not found"). Stage A's inventory now
# finds a stem with the same name living in timepoint_ndpi_deferred, which
# opens fine at the METADATA level -- but Stage A never calls read_region, so
# that does not prove the pixel data is intact. It is included in this run
# deliberately: timepoint_convert_nocrop.py already wraps each slide's
# conversion in its own try/except (confirmed in that module), so if this
# copy is also corrupted, it fails gracefully, is logged in
# conversion_summary.json, and does not abort the rest of the batch.
#
# Reads (read-only):
#   $SCRATCH/data/timepoint_ndpi_deferred/*.ndpi  (24 slides -- all must be present)
#   $SCRATCH/data/ndpi                             (3-slide sample, MPP comparison only)
#   ~/cancer_trajectory_atlas/jobs/slides_timepoint_remaining.txt
#
# Writes (SAME directory as the first 7 -- additive, nothing overwritten):
#   $SCRATCH/data/timepoint_x5_full/*.png
#   $SCRATCH/data/timepoint_x5_full/slide_dimensions.json  (sidecar, merged/updated)
#   $SCRATCH/data/timepoint_x5_full/conversion_summary.json (OVERWRITTEN with this
#     run's 24-slide summary -- the first 7's summary was already reviewed; if you
#     need it again it's in this job's own SLURM log)
#
# Writes (NEW file, separate from the first run's mpp_verification.json so that
# record is never touched):
#   $SCRATCH/results/timepoint_cohort/stageC_convert/mpp_verification.json
#
# NOTE: this job does NOT wait for or depend on Stage B
# (run_timepoint_stain_homogeneity.sh) -- conversion itself doesn't depend on
# the stain-homogeneity result. It CAN be run in parallel with Stage B. But if
# Stage B comes back FAIL, the within-cohort design is confounded and this
# conversion's output would not be used for the timepoint analysis as
# designed -- that's a real possibility, weighed and accepted deliberately here
# given the alternative was idle GPU/CPU time either way.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_convert_stageC.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=timepoint_convert_stageC
#SBATCH --output=logs/timepoint_convert_stageC-%j.out

# NOTE on --time=48:00:00: this is a GUESS, not a measured estimate -- the only
# real precedent (an 8-slide run) was bumped from 3h to 24h as a safety margin,
# and the most recent 7-slide run had all 7 already converted (pure SKIP, no
# real conversion timing data). 24 slides is ~3x the prior batch, so 48h is a
# generous scale-up, not a tight estimate. If your account's QOS caps walltime
# below this, sbatch will reject the submission immediately -- lower --time and
# resubmit if so.
#
# NOTE on --mem=128G: unchanged from the first run -- peak memory is
# per-slide (one NDPI decoded into memory at a time, sequentially), not
# cumulative across the batch, so a larger slide count doesn't need more
# memory, just more wall time.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_timepoint_remaining.txt"
NDPI_DIR="$SCRATCH/data/timepoint_ndpi_deferred"
PNG_DIR="$SCRATCH/data/timepoint_x5_full"
NDPI_LEVEL=0
ORIGINAL_NDPI_DIR="$SCRATCH/data/ndpi"
ORIGINAL_SAMPLE_STEMS="6027-4L-2M-1_x5 6028-4L-2M-1_x5 6029-4L-2M-1_x5"
MPP_TOLERANCE_PCT=2.0
MPP_OUTPUT_JSON="$SCRATCH/results/timepoint_cohort/stageC_convert/mpp_verification.json"

echo "========================================================"
echo "  Timepoint cohort — Stage C: conversion of remaining slides"
echo "  Job ID              : ${SLURM_JOB_ID:-local}"
echo "  Slide list           : $SLIDE_LIST (24 slides)"
echo "  NDPI dir              : $NDPI_DIR"
echo "  PNG dir (additive)    : $PNG_DIR"
echo "  ndpi_level             : $NDPI_LEVEL"
echo "  ndpi_scale             : determined dynamically by MPP pre-flight check (see below)"
echo "  Original NDPI dir      : $ORIGINAL_NDPI_DIR"
echo "  Original sample stems  : $ORIGINAL_SAMPLE_STEMS"
echo "  MPP tolerance          : ${MPP_TOLERANCE_PCT}%"
echo "  MPP output JSON        : $MPP_OUTPUT_JSON"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo "Expected NDPI files (exactly 24):"
MISSING=0
while read -r stem; do
  [ -z "$stem" ] && continue
  if [ -f "$NDPI_DIR/$stem.ndpi" ]; then
    echo -n "  $stem.ndpi  : "; ls -lh "$NDPI_DIR/$stem.ndpi"
  else
    echo "  $stem.ndpi  : NOT FOUND"
    MISSING=1
  fi
done < "$SLIDE_LIST"

EXPECTED_COUNT=$(grep -c . "$SLIDE_LIST")
echo ""
echo "Expected slide count: $EXPECTED_COUNT (should be 24)"
if [ "$MISSING" -ne 0 ]; then
  echo "*** HARD FAIL: one or more expected NDPI files are missing from $NDPI_DIR."
  echo "*** Check jobs/slides_timepoint_remaining.txt against Stage A's inventory."
  exit 1
fi
echo "======================"
echo ""

mkdir -p "$PNG_DIR"
mkdir -p "$(dirname "$MPP_OUTPUT_JSON")"

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_convert_nocrop \
    --slide-list             "$SLIDE_LIST" \
    --ndpi-dir               "$NDPI_DIR" \
    --png-dir                "$PNG_DIR" \
    --ndpi-level             "$NDPI_LEVEL" \
    --original-ndpi-dir      "$ORIGINAL_NDPI_DIR" \
    --original-sample-stems  $ORIGINAL_SAMPLE_STEMS \
    --mpp-tolerance-pct      "$MPP_TOLERANCE_PCT" \
    --mpp-output-json        "$MPP_OUTPUT_JSON"

echo ""
echo "========================================================"
echo "  STAGE C CONVERSION COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $PNG_DIR/*.png (24 new + the existing 7, untouched)"
echo "  $PNG_DIR/slide_dimensions.json"
echo "  $PNG_DIR/conversion_summary.json"
echo "  $MPP_OUTPUT_JSON"
echo ""
echo "Check conversion_summary.json for any per-slide errors -- in particular,"
echo "6069-4R-4W was included despite a prior corruption finding elsewhere; if it"
echo "failed again here, that's expected and does not indicate a problem with"
echo "this job."
echo ""
