#!/bin/bash
# Holeyness v2: extended, area-adjusted duct-level validation.
#
# Extends the v1 holeyness validation (run_holeyness_validation.sh) with checks
# for a duct-area confound found post-hoc: duct area correlates with both
# pseudotime (rho +0.43) and hole_pct (rho +0.39), and the v1 independence check
# (nuclear_density only) did not catch this. Also adds an exclusion-bias check
# on zero-patch ducts, a within-slide/nested permutation test, and aggregation
# sensitivity checks.
#
# Reads (same raw inputs as v1 — no pipeline stage is rerun):
#   $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt
#   $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
#   $SCRATCH/results/per_section/atlas_<SECTION>/results.csv
#   ~/cancer_trajectory_atlas/data/annotations_ratio/*.json
#   $SCRATCH/results/holeyness/<SECTION>/holeyness_per_duct.csv   (v1 output; read-only, for the consistency check)
#
# Writes (NEW versioned subdirectory — v1 outputs are never touched):
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/holeyness_validation_v2.json
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/holeyness_validation_v2.md
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/duct_table_full.csv
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/v2_scatter_pt_vs_hole_pct_by_area.{pdf,png}
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/v2_scatter_pt_vs_area.{pdf,png}
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/v2_small_multiples_per_slide.{pdf,png}
#
# Does NOT modify holeyness_per_duct.csv, holeyness_validation.json, or the two
# v1 scatter figures at $SCRATCH/results/holeyness/<SECTION>/.
#
# Usage (section 2M-1):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_validation_v2.sh
#
# To run 2M-2 later (adjust the variables below, same pattern as v1):
#   SECTION="2M-2"
#   RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-2/results.csv"
#   SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"
#   V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-2"

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=holeyness_validation_v2
#SBATCH --output=logs/holeyness_validation_v2-%j.out

set -euo pipefail
mkdir -p logs

# ── Section-specific parameters (edit these for 2M-2) ────────────────────────
SECTION="2M-1"
RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-1"

# ── v2 output goes to a NEW versioned subdirectory — v1 outputs are untouched ──
OUTPUT_DIR="$V1_OUTPUT_DIR/v2_area_adjusted"
V1_PER_DUCT_CSV="$V1_OUTPUT_DIR/holeyness_per_duct.csv"

# ── Fixed input paths (same as v1) ────────────────────────────────────────────
EXPORT="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
ANNOTATION_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
SLIDE_DIMENSIONS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"

echo "========================================================"
echo "  Holeyness validation v2 (area-adjusted) — section ${SECTION}"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Export       : $EXPORT"
echo "  Annotations  : $ANNOTATION_DIR"
echo "  Slide dims   : $SLIDE_DIMENSIONS"
echo "  Results      : $RESULTS_CSV"
echo "  Slide list   : $SLIDE_LIST"
echo "  v1 per-duct  : $V1_PER_DUCT_CSV"
echo "  Output dir   : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Export       : "; ls -lh "$EXPORT"           2>/dev/null || echo "NOT FOUND"
echo -n "Slide dims   : "; ls -lh "$SLIDE_DIMENSIONS"  2>/dev/null || echo "NOT FOUND"
echo -n "Results CSV  : "; ls -lh "$RESULTS_CSV"       2>/dev/null || echo "NOT FOUND"
echo -n "Slide list   : "; ls -lh "$SLIDE_LIST"        2>/dev/null || echo "NOT FOUND"
echo -n "v1 per-duct  : "; ls -lh "$V1_PER_DUCT_CSV"   2>/dev/null || echo "NOT FOUND — run run_holeyness_validation.sh first"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.holeyness \
    --section          "$SECTION" \
    --export           "$EXPORT" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --results          "$RESULTS_CSV" \
    --output-dir       "$OUTPUT_DIR" \
    --slide-list       "$SLIDE_LIST" \
    --v2 \
    --v1-per-duct-csv  "$V1_PER_DUCT_CSV" \
    --n-permutations   1000

echo ""
echo "========================================================"
echo "  HOLEYNESS VALIDATION v2 COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/holeyness_validation_v2.json"
echo "  $OUTPUT_DIR/holeyness_validation_v2.md"
echo "  $OUTPUT_DIR/duct_table_full.csv"
echo "  $OUTPUT_DIR/v2_scatter_pt_vs_hole_pct_by_area.{pdf,png}"
echo "  $OUTPUT_DIR/v2_scatter_pt_vs_area.{pdf,png}"
echo "  $OUTPUT_DIR/v2_small_multiples_per_slide.{pdf,png}"
echo ""
echo "v1 outputs (unchanged):"
echo "  $V1_OUTPUT_DIR/holeyness_per_duct.csv"
echo "  $V1_OUTPUT_DIR/holeyness_validation.json"
echo ""
