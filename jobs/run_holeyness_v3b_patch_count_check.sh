#!/bin/bash
# Holeyness v3b: patch-count-per-duct discriminant + within-slide undersampling check.
#
# v3's per-slide investigation compared the 3 flagged low-signal slides
# (6028-4R-2M-1_x5, 6029-4L-2M-1_x5, 6031-4L-2M-1_x5) against the other 5 on median
# DUCT AREA and found no clear difference. It never directly tested PATCH COUNT PER
# DUCT, even though all 3 flagged slides sit at median_n_patches = 2 (every other
# slide is 3 or 4). This job:
#   0. Resolves the ~0.000118 mismatch in v3's own consistency check (CSV-rounding
#      precision loss vs. a fresh full-precision recompute) before trusting anything else.
#   1. Tests median_n_patches_per_duct / frac_single_patch_ducts as the discriminant.
#   2. Within each flagged slide, tests whether restricting to ducts with >=3 patches
#      changes that slide's area-adjusted partial correlation (undersampling test).
#
# Reads (all read-only — v1/v2/v3 outputs are never modified):
#   $SCRATCH/results/holeyness/<SECTION>/holeyness_per_duct.csv                         (v1 output)
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/holeyness_validation_v2.json  (v2 output)
#   $SCRATCH/results/holeyness/<SECTION>/v3_significance/holeyness_validation_v3.json   (v3 output)
#   $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt   (raw input, Check 0 only)
#   $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
#   $SCRATCH/results/per_section/atlas_<SECTION>/results.csv
#   ~/cancer_trajectory_atlas/data/annotations_ratio/*.json
#
# Writes (NEW versioned subdirectory — v1/v2/v3 outputs are never touched):
#   $SCRATCH/results/holeyness/<SECTION>/v3b_patch_count_check/holeyness_validation_v3b.json
#   $SCRATCH/results/holeyness/<SECTION>/v3b_patch_count_check/holeyness_validation_v3b.md
#   $SCRATCH/results/holeyness/<SECTION>/v3b_patch_count_check/v3b_median_n_patches_bar.{pdf,png}
#
# Usage (section 2M-1):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_v3b_patch_count_check.sh
#
# To run 2M-2 later (adjust the variables below, same pattern as v1/v2/v3):
#   SECTION="2M-2"
#   RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-2/results.csv"
#   SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"
#   V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-2"

#SBATCH --account=def-lmarti46
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=holeyness_v3b_patch_count_check
#SBATCH --output=logs/holeyness_v3b_patch_count_check-%j.out

set -euo pipefail
mkdir -p logs

# ── Section-specific parameters (edit these for 2M-2) ────────────────────────
SECTION="2M-1"
RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-1"
V2_OUTPUT_DIR="$V1_OUTPUT_DIR/v2_area_adjusted"
V3_OUTPUT_DIR="$V1_OUTPUT_DIR/v3_significance"

# ── v3b output goes to a NEW versioned subdirectory — v1/v2/v3 outputs untouched ──
OUTPUT_DIR="$V1_OUTPUT_DIR/v3b_patch_count_check"
V1_PER_DUCT_CSV="$V1_OUTPUT_DIR/holeyness_per_duct.csv"
V2_JSON="$V2_OUTPUT_DIR/holeyness_validation_v2.json"
V3_JSON="$V3_OUTPUT_DIR/holeyness_validation_v3.json"

# ── Fixed raw-input paths (same as v1/v2/v3 — used only for Check 0's full-precision re-derivation) ──
EXPORT="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
ANNOTATION_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
SLIDE_DIMENSIONS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"

echo "========================================================"
echo "  Holeyness v3b patch-count check — section ${SECTION}"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  v1 per-duct  : $V1_PER_DUCT_CSV"
echo "  v2 JSON      : $V2_JSON"
echo "  v3 JSON      : $V3_JSON"
echo "  Export       : $EXPORT"
echo "  Annotations  : $ANNOTATION_DIR"
echo "  Slide dims   : $SLIDE_DIMENSIONS"
echo "  Results      : $RESULTS_CSV"
echo "  Slide list   : $SLIDE_LIST"
echo "  Output dir   : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "v1 per-duct  : "; ls -lh "$V1_PER_DUCT_CSV"   2>/dev/null || echo "NOT FOUND — run run_holeyness_validation.sh first"
echo -n "v2 JSON      : "; ls -lh "$V2_JSON"           2>/dev/null || echo "NOT FOUND — run run_holeyness_validation_v2.sh first"
echo -n "v3 JSON      : "; ls -lh "$V3_JSON"           2>/dev/null || echo "NOT FOUND — run run_holeyness_v3_significance.sh first"
echo -n "Export       : "; ls -lh "$EXPORT"            2>/dev/null || echo "NOT FOUND"
echo -n "Slide dims   : "; ls -lh "$SLIDE_DIMENSIONS"   2>/dev/null || echo "NOT FOUND"
echo -n "Results CSV  : "; ls -lh "$RESULTS_CSV"        2>/dev/null || echo "NOT FOUND"
echo -n "Slide list   : "; ls -lh "$SLIDE_LIST"         2>/dev/null || echo "NOT FOUND"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.holeyness_v3b_patch_count_check \
    --section          "$SECTION" \
    --v1-per-duct-csv  "$V1_PER_DUCT_CSV" \
    --v2-json          "$V2_JSON" \
    --v3-json          "$V3_JSON" \
    --export           "$EXPORT" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --results          "$RESULTS_CSV" \
    --slide-list       "$SLIDE_LIST" \
    --output-dir       "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  HOLEYNESS V3B PATCH-COUNT CHECK COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/holeyness_validation_v3b.json"
echo "  $OUTPUT_DIR/holeyness_validation_v3b.md"
echo "  $OUTPUT_DIR/v3b_median_n_patches_bar.{pdf,png}"
echo ""
echo "v1/v2/v3 outputs (unchanged):"
echo "  $V1_OUTPUT_DIR/holeyness_per_duct.csv"
echo "  $V2_OUTPUT_DIR/holeyness_validation_v2.json"
echo "  $V3_OUTPUT_DIR/holeyness_validation_v3.json"
echo ""
