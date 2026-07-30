#!/bin/bash
# Timepoint projection: no-crop conversion (full-width PNG).
#
# Stage 1's original conversion (run_all.py --convert, reused unmodified) applies
# an unconditional left-half crop -- correct for the 16 original pipeline slides
# (annotation-confirmed duplicate right half) but never verified for these 8
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
# Reads (read-only):
#   $SCRATCH/data/timepoint_ndpi/*.ndpi            (7 of 8 slides)
#   $SCRATCH/data/timepoint_ndpi_deferred/*.ndpi   (6069-4R-4W, still set aside
#     pending its QuPath OME-TIFF re-export -- if not yet recovered, this job
#     will report it as an error and continue with the other 7, same as Stage 1)
#   ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt
#
# Writes (NEW directory -- existing timepoint_x5_cropped is never touched):
#   $SCRATCH/data/timepoint_x5_full/*.png
#   $SCRATCH/data/timepoint_x5_full/slide_dimensions.json
#   $SCRATCH/data/timepoint_x5_full/conversion_summary.json
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
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=timepoint_convert_nocrop
#SBATCH --output=logs/timepoint_convert_nocrop-%j.out

# NOTE on --mem=128G: this reads the SAME full-resolution NDPI data Stage 1's
# original conversion did (peak memory is reached before any crop/no-crop
# decision -- the crop only affects what gets SAVED, not what gets decoded into
# memory first) -- reuses the memory setting that already fixed Stage 1's OOM.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_timepoint.txt"
NDPI_DIR="$SCRATCH/data/timepoint_ndpi"
NDPI_DEFERRED_DIR="$SCRATCH/data/timepoint_ndpi_deferred"
PNG_DIR="$SCRATCH/data/timepoint_x5_full"
NDPI_LEVEL=0
NDPI_SCALE=1.0

echo "========================================================"
echo "  Timepoint projection — no-crop conversion (full-width PNG)"
echo "  Job ID              : ${SLURM_JOB_ID:-local}"
echo "  Slide list           : $SLIDE_LIST"
echo "  NDPI dir              : $NDPI_DIR"
echo "  NDPI deferred dir     : $NDPI_DEFERRED_DIR"
echo "  PNG dir (NEW)         : $PNG_DIR"
echo "  ndpi_level / ndpi_scale: $NDPI_LEVEL / $NDPI_SCALE"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo "Expected NDPI files (primary, then deferred if not found in primary):"
while read -r stem; do
  [ -z "$stem" ] && continue
  if [ -f "$NDPI_DIR/$stem.ndpi" ]; then
    echo -n "  $stem.ndpi (primary)  : "; ls -lh "$NDPI_DIR/$stem.ndpi"
  elif [ -f "$NDPI_DEFERRED_DIR/$stem.ndpi" ]; then
    echo -n "  $stem.ndpi (deferred) : "; ls -lh "$NDPI_DEFERRED_DIR/$stem.ndpi"
  else
    echo "  $stem.ndpi : NOT FOUND in either directory"
  fi
done < "$SLIDE_LIST"
echo "======================"
echo ""

mkdir -p "$PNG_DIR"

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_convert_nocrop \
    --slide-list         "$SLIDE_LIST" \
    --ndpi-dir           "$NDPI_DIR" \
    --ndpi-deferred-dir  "$NDPI_DEFERRED_DIR" \
    --png-dir            "$PNG_DIR" \
    --ndpi-level         "$NDPI_LEVEL" \
    --ndpi-scale         "$NDPI_SCALE"

echo ""
echo "========================================================"
echo "  NO-CROP CONVERSION COMPLETE"
echo "========================================================"
echo ""
echo "Outputs (NEW directory -- timepoint_x5_cropped untouched):"
echo "  $PNG_DIR/*.png"
echo "  $PNG_DIR/slide_dimensions.json"
echo "  $PNG_DIR/conversion_summary.json"
echo ""
echo "Next: re-run analysis.timepoint_inventory against $PNG_DIR to refresh"
echo "the per-slide report (see the header comment in this script for the exact command)."
echo ""
