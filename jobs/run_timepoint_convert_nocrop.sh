#!/bin/bash
# Timepoint projection: no-crop conversion (full-width PNG).
#
# Stage 1's original conversion (run_all.py --convert, reused unmodified) applies
# an unconditional left-half crop -- correct for the 16 original pipeline slides
# (annotation-confirmed duplicate right half) but never verified for these
# timepoint slides, which have no annotations at all. Per-slide visual inspection
# found the batch is NOT uniform (some duplicate, some not), so no single
# crop-or-don't-crop rule is safe without per-slide domain knowledge this
# project doesn't have yet.
#
# This job instead converts WITHOUT cropping -- full image width kept -- so the
# decision about whether to restrict any specific slide to its left half can be
# made LATER, post-hoc, from patch coordinates already in the pipeline's normal
# output, rather than needing to guess correctly before ever converting a slide.
#
# NEW, STANDALONE SCRIPT -- run_all.py is NOT modified and is not called here.
# Writes to a NEW output directory (timepoint_x5_full) with its own sidecar --
# the existing (left-cropped) timepoint_x5_cropped directory and its sidecar
# are never touched, so nothing is lost if this approach needs revisiting.
#
# Slide count: exactly 7. 6069-4R-4W is PERMANENTLY EXCLUDED -- confirmed
# corrupted at the OpenSlide level ("Restart marker not found", not a transfer
# issue -- SHA256 matched the local copy). It has been removed from
# jobs/slides_timepoint.txt; there is no deferred/recovery directory or fallback
# path for it in this job or in timepoint_convert_nocrop.py.
#
# Scale: NDPI_SCALE is NOT a static input to this job -- it is determined by an
# automated MPP pre-flight check (Step 0 inside timepoint_convert_nocrop.py) that
# runs BEFORE any conversion. It reads level-0 dimensions + openslide.mpp-x/-y
# (metadata only, no pixel decode -- fast) for all 7 timepoint NDPIs and for a
# fixed 3-slide sample of the original 16 NDPIs, and compares them. If MPP values
# match within tolerance, it proceeds using ndpi_scale=1.0 (the original
# pipeline's actual conversion scale -- confirmed against jobs/convert_ndpi.sh and
# pipeline_config.py's default, NOT the previously-assumed 0.5). If they don't
# match, or metadata is missing for either batch, the job HARD STOPS before
# writing any PNG and reports the scale that would be needed instead. The full
# comparison is always written to mpp_verification.json, independent of this
# job's SLURM log.
#
# Reads (read-only):
#   $SCRATCH/data/timepoint_ndpi/*.ndpi   (exactly 7 slides -- all must be present)
#   $SCRATCH/data/MCF7_x5/*.ndpi          (3-slide sample, MPP comparison only)
#   ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt
#
# Writes (NEW directory -- existing timepoint_x5_cropped is never touched):
#   $SCRATCH/data/timepoint_x5_full/*.png
#   $SCRATCH/data/timepoint_x5_full/slide_dimensions.json
#   $SCRATCH/data/timepoint_x5_full/conversion_summary.json
#   $SCRATCH/results/timepoint_projection/mpp_verification.json
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_convert_nocrop.sh
#
# After this completes, re-run Stage 1's inventory against the new directory to
# refresh the per-slide report:
#   python -m cancer_trajectory_atlas.analysis.timepoint_inventory \
#       --slide-list       ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \
#       --ndpi-dir         $SCRATCH/data/timepoint_ndpi \
#       --png-dir          $SCRATCH/data/timepoint_x5_full \
#       --slide-dimensions $SCRATCH/data/timepoint_x5_full/slide_dimensions.json \
#       --annotation-dir   ~/cancer_trajectory_atlas/data/annotations \
#       --ndpi-scale       1.0 \
#       --output-dir       $SCRATCH/results/timepoint_projection/stage1_convert_nocrop

#SBATCH --account=def-lmarti46
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=timepoint_convert_nocrop
#SBATCH --output=logs/timepoint_convert_nocrop-%j.out

# NOTE on --mem=128G: this reads the SAME full-resolution NDPI data Stage 1's
# original conversion did (peak memory is reached before any crop/no-crop
# decision -- the crop only affects what gets SAVED, not what gets decoded into
# memory first) -- reuses the memory setting that already fixed Stage 1's OOM.
#
# NOTE on --time=24:00:00: the previous attempt (8 slides, ndpi_scale=1.0) timed
# out on a 3-hour walltime and had to be resubmitted at 24 hours. This run has
# ONE FEWER slide (7, not 8 -- 6069-4R-4W excluded) and ndpi_scale will likely
# resolve to 1.0 again (possibly less, if the MPP check finds a mismatch), so it
# should if anything run FASTER and use somewhat LESS peak memory per slide than
# the previous attempt. 24 hours here is a safety margin, not an expectation of
# needing that long.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_timepoint.txt"
NDPI_DIR="$SCRATCH/data/timepoint_ndpi"
PNG_DIR="$SCRATCH/data/timepoint_x5_full"
NDPI_LEVEL=0
ORIGINAL_NDPI_DIR="$SCRATCH/data/MCF7_x5"
ORIGINAL_SAMPLE_STEMS="6027-4L-2M-1_x5 6028-4L-2M-1_x5 6029-4L-2M-1_x5"
MPP_TOLERANCE_PCT=2.0
MPP_OUTPUT_JSON="$SCRATCH/results/timepoint_projection/mpp_verification.json"

echo "========================================================"
echo "  Timepoint projection — no-crop conversion (full-width PNG)"
echo "  Job ID              : ${SLURM_JOB_ID:-local}"
echo "  Slide list           : $SLIDE_LIST"
echo "  NDPI dir              : $NDPI_DIR"
echo "  PNG dir (NEW)         : $PNG_DIR"
echo "  ndpi_level             : $NDPI_LEVEL"
echo "  ndpi_scale             : determined dynamically by MPP pre-flight check (see below)"
echo "  Original NDPI dir      : $ORIGINAL_NDPI_DIR"
echo "  Original sample stems  : $ORIGINAL_SAMPLE_STEMS"
echo "  MPP tolerance          : ${MPP_TOLERANCE_PCT}%"
echo "  MPP output JSON        : $MPP_OUTPUT_JSON"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo "Expected NDPI files (exactly 7 -- 6069-4R-4W permanently excluded, confirmed"
echo "corrupted at the OpenSlide level):"
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
echo "Expected slide count: $EXPECTED_COUNT (should be 7)"
if [ "$MISSING" -ne 0 ]; then
  echo "*** HARD FAIL: one or more expected NDPI files are missing from $NDPI_DIR."
  echo "*** There is no fallback/deferred directory -- place the missing file(s)"
  echo "*** there and resubmit."
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
echo "  NO-CROP CONVERSION COMPLETE"
echo "========================================================"
echo ""
echo "Outputs (NEW directory -- timepoint_x5_cropped untouched):"
echo "  $PNG_DIR/*.png"
echo "  $PNG_DIR/slide_dimensions.json"
echo "  $PNG_DIR/conversion_summary.json"
echo "  $MPP_OUTPUT_JSON"
echo ""
echo "Next: re-run analysis.timepoint_inventory against $PNG_DIR to refresh"
echo "the per-slide report (see the header comment in this script for the exact command)."
echo ""
