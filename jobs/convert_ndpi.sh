#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_convert
#SBATCH --output=logs/atlas_convert-%j.out

# NDPI -> left-half PNG conversion. No GPU needed, pure image I/O.
#
# ── CORRECTED 2026-08-25. Two flags were wrong. ─────────────────────────────
#
# This script did not reproduce the cohort the pipeline actually ran on. Both
# errors are fixed below; the evidence is in reports/codebase_inventory.md §3.1.
#
#   was: --ndpi-dir $SCRATCH/data/MCF7_x5      that directory does not exist
#   now: --ndpi-dir $SCRATCH/data/ndpi         matches paths.json "raw_ndpi"
#
#   was: --ndpi-scale 1.0                      gives TWICE the linear resolution
#   now: --ndpi-scale 0.5                      reproduces the reference exactly
#
# HOW THE SCALE WAS ESTABLISHED. jobs/verify_conversion_smoke.sh converted
# 6027-4L-2M-1 and 6027-4L-2M-2 at level 0 / scale 0.5 (job 1648162) and
# reproduced the existing PNGs BIT-IDENTICALLY: all 1,520,640,000 and
# 1,589,575,680 channel values equal, and patch counts matching the feature
# cache exactly at 616 and 1228. PNG is lossless, so bit-identity is proof.
#
# --ndpi-level 1 was tested first (job 1647619) and REJECTED. It yields the same
# DIMENSIONS, because level 1 of these pyramids is a factor-2 downsample, but
# different PIXELS: mean |diff| 1.708 and 1.972, and 617/1252 patches against the
# cached 616/1228. A scanner pyramid level is not a LANCZOS resize of level 0.
#
# WHAT THE OLD FLAGS WOULD HAVE DONE. At scale 1.0 every slide comes out at
# twice the linear resolution, so roughly 4x the patch count, and every absolute
# number downstream differs. Worse, run_all --convert SKIPS existing PNGs but
# REWRITES slide_dimensions.json unconditionally (run_all.py:129-133). Running
# the old version against the live directory would have left the PNGs alone
# while silently replacing the sidecar with wrong dimensions, and every
# ratio-to-pixel annotation transform reads that sidecar.
#
# ── Regenerating the cohort ─────────────────────────────────────────────────
# Existing PNGs are skipped, so a re-run is normally a no-op that only refreshes
# the sidecar. To force a genuine rebuild, move the old PNGs aside first rather
# than deleting them.
#
# To convert somewhere else without touching the reference data:
#   sbatch --export=ALL,PNG_DIR=$SCRATCH/data/test_convert jobs/convert_ndpi.sh
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/convert_ndpi.sh

set -euo pipefail

# Revision of THIS FILE, printed below so a job log says which version ran.
SCRIPT_REV="2026-08-25b"

# paths.json "raw_ndpi" is authoritative for the NDPI source.
NDPI_DIR="${NDPI_DIR:-$SCRATCH/data/ndpi}"
PNG_DIR="${PNG_DIR:-$SCRATCH/data/MCF7_x5_cropped}"

# MEASURED, not assumed. See the header. Do not change without re-running
# jobs/verify_conversion_smoke.sh, which is what establishes these.
NDPI_LEVEL="${NDPI_LEVEL:-0}"
NDPI_SCALE="${NDPI_SCALE:-0.5}"

mkdir -p logs
mkdir -p "$PNG_DIR"

echo "============================================================"
echo "  NDPI -> left-half PNG conversion"
echo "  Script rev: $SCRIPT_REV"
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  NDPI in   : $NDPI_DIR"
echo "  PNG out   : $PNG_DIR"
echo "  Resolution: --ndpi-level $NDPI_LEVEL --ndpi-scale $NDPI_SCALE"
echo "============================================================"

if [ ! -d "$NDPI_DIR" ]; then
    echo "ERROR: NDPI directory not found: $NDPI_DIR"
    echo "  paths.json says raw_ndpi = ~/scratch/data/ndpi. Locate the slides with:"
    echo "    find \$SCRATCH -maxdepth 3 -name '*.ndpi' | head"
    exit 1
fi

# Counted with find, not `ls ... | wc -l`. Under `set -o pipefail` a glob that
# matches nothing makes ls fail, the pipeline fails, and `set -e` kills the job
# after printing a bare "0" and no explanation. find returns success on an empty
# result, so the count is followed by a real error message.
N_NDPI=$(find "$NDPI_DIR" -maxdepth 1 -name '*.ndpi' | wc -l)
echo "NDPI files: $N_NDPI"
if [ "$N_NDPI" -eq 0 ]; then
    echo "ERROR: no .ndpi files in $NDPI_DIR"
    echo "  Locate them with: find \$SCRATCH -maxdepth 3 -name '*.ndpi' | head"
    exit 1
fi

# Existing PNGs are skipped by --convert, but slide_dimensions.json is rewritten
# every time. Say so, because that file is what every ratio-coordinate
# annotation transform depends on.
if [ -f "$PNG_DIR/slide_dimensions.json" ]; then
    echo ""
    echo "NOTE: $PNG_DIR/slide_dimensions.json exists and WILL BE REWRITTEN."
    echo "      Existing PNGs are skipped, so this run refreshes the sidecar only."
    echo "      With the flags above it rewrites the same values. With different"
    echo "      --ndpi-level/--ndpi-scale it would not, and every ratio-to-pixel"
    echo "      annotation transform reads this file."
fi

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.run_all \
    --convert \
    --ndpi-dir   "$NDPI_DIR" \
    --png-dir    "$PNG_DIR" \
    --ndpi-level "$NDPI_LEVEL" \
    --ndpi-scale "$NDPI_SCALE"

echo ""
echo "Done. PNG count: $(find "$PNG_DIR" -maxdepth 1 -name '*.png' | wc -l)"
echo "Output size:   $(du -sh "$PNG_DIR" 2>/dev/null | cut -f1)"

echo ""
echo "Verify the conversion reproduces the reference:"
echo "  sbatch ~/cancer_trajectory_atlas/jobs/verify_conversion_smoke.sh"
