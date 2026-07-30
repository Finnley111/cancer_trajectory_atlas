#!/bin/bash
# Stage 2 support (Task A): slide-level 2M-1 vs 2M-2 reference threshold.
#
# Stage 2's stain batch check (timepoint_stage2_stain_check.py) gates GPU work on
# whether timepoint slides are confounded with staining relative to the existing
# 2M-1 slides. Its threshold previously came from
# diagnostics/audit_feature_diagnostics.py's D3 check, computed on PER-PATCH
# h_intensity values -- but Stage 2 compares PER-SLIDE summaries. Patch-level and
# slide-level rank-biserial are not the same quantity, so a patch-level threshold
# was not a valid gate for a slide-level comparison.
#
# This job recomputes the project's known cross-section confound (2M-1 vs 2M-2)
# at SLIDE level, using tissue-masked stain features (same method Stage 2 uses),
# and reports a slide-level reference_rank_biserial for Stage 2 to consume via
# --reference-threshold-json, replacing the old patch-level 0.71.
#
# READ-ONLY, no dependency on the in-progress no-crop timepoint conversion --
# reads only the EXISTING 16 original slides' already-converted, left-cropped
# PNGs. Can run now, independently of that conversion job.
#
# Reads (read-only):
#   $SCRATCH/data/MCF7_x5_cropped/*.png
#   ~/cancer_trajectory_atlas/jobs/slides_section1.txt (8 2M-1 slides)
#   ~/cancer_trajectory_atlas/jobs/slides_section2.txt (8 2M-2 slides)
#
# Writes (NEW directory):
#   $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json
#   $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.md
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_stage2_reference_threshold.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=stage2_reference_threshold
#SBATCH --output=logs/stage2_reference_threshold-%j.out

# NOTE on --mem=32G: this reads 16 ALREADY-CONVERTED, left-cropped PNGs at full
# resolution (no NDPI decode at all) -- far lighter than any conversion job.
# 32G is defensive headroom, not a measured requirement.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
SECTION1_SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
SECTION2_SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
REFERENCE_MEASURE="h_intensity_mean_masked"
PATCH_LEVEL_REFERENCE_R=0.71

OUTPUT_DIR="$SCRATCH/results/timepoint_projection/stage2_reference_threshold"

echo "========================================================"
echo "  Stage 2 reference threshold — slide-level 2M-1 vs 2M-2 (Task A)"
echo "  Job ID                  : ${SLURM_JOB_ID:-local}"
echo "  Section 1 (2M-1) slides : $SECTION1_SLIDE_LIST"
echo "  Section 2 (2M-2) slides : $SECTION2_SLIDE_LIST"
echo "  PNG dir (read-only)     : $PNG_DIR"
echo "  Reference measure       : $REFERENCE_MEASURE"
echo "  Patch-level reference r : $PATCH_LEVEL_REFERENCE_R (old D3 number, comparison only)"
echo "  Output dir               : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Section 1 slide list (expect 8) : "; ls -lh "$SECTION1_SLIDE_LIST" 2>/dev/null || echo "NOT FOUND"
echo -n "Section 2 slide list (expect 8) : "; ls -lh "$SECTION2_SLIDE_LIST" 2>/dev/null || echo "NOT FOUND"
echo -n "PNG dir                          : "; ls -d "$PNG_DIR" 2>/dev/null || echo "NOT FOUND"
echo "======================"
echo ""

mkdir -p "$OUTPUT_DIR"

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.stage2_reference_threshold \
    --section1-slide-list      "$SECTION1_SLIDE_LIST" \
    --section2-slide-list      "$SECTION2_SLIDE_LIST" \
    --png-dir                  "$PNG_DIR" \
    --reference-measure        "$REFERENCE_MEASURE" \
    --patch-level-reference-r  "$PATCH_LEVEL_REFERENCE_R" \
    --output-dir                "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE 2 REFERENCE THRESHOLD COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stage2_reference_threshold.json"
echo "  $OUTPUT_DIR/stage2_reference_threshold.md"
echo ""
echo "Next: jobs/run_timepoint_stage2_stain_check.sh reads this JSON's"
echo "reference_rank_biserial by default via --reference-threshold-json -- no"
echo "further action needed here beyond reviewing the .md report."
echo ""
