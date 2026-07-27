#!/bin/bash
# Holeyness v3: significance test on the area-adjusted partial correlation.
#
# v2 found rho(pseudotime, hole_pct) = 0.276 drops to a partial rho of 0.131 after
# controlling for duct area (0.158 controlling for area + nuclear_density), but v2
# never ran a permutation test ON the partial (its permutation tests were on the raw
# correlation only), and its aggregation-sensitivity sweep was likewise raw-only.
# This job closes both gaps and investigates three slides with a near-zero/negative
# area-adjusted partial (6028-4R-2M-1_x5, 6029-4L-2M-1_x5, 6031-4L-2M-1_x5).
#
# Reads:
#   $SCRATCH/results/holeyness/<SECTION>/holeyness_per_duct.csv                    (v1 output, read-only)
#   $SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/holeyness_validation_v2.json  (v2 output, read-only)
#   $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt                  (raw input, for mean-agg re-derivation only)
#   $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
#   $SCRATCH/results/per_section/atlas_<SECTION>/results.csv
#   ~/cancer_trajectory_atlas/data/annotations_ratio/*.json
#
# Writes (NEW versioned subdirectory — v1 and v2 outputs are never touched):
#   $SCRATCH/results/holeyness/<SECTION>/v3_significance/holeyness_validation_v3.json
#   $SCRATCH/results/holeyness/<SECTION>/v3_significance/holeyness_validation_v3.md
#   $SCRATCH/results/holeyness/<SECTION>/v3_significance/v3_per_slide_partial_bar.{pdf,png}
#
# Usage (section 2M-1):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_v3_significance.sh
#
# To run 2M-2 later (adjust the variables below, same pattern as v1/v2):
#   SECTION="2M-2"
#   RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-2/results.csv"
#   SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"
#   V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-2"
#   V2_OUTPUT_DIR="$V1_OUTPUT_DIR/v2_area_adjusted"

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=holeyness_v3_significance
#SBATCH --output=logs/holeyness_v3_significance-%j.out

set -euo pipefail
mkdir -p logs

# ── Section-specific parameters (edit these for 2M-2) ────────────────────────
SECTION="2M-1"
RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-1"
V2_OUTPUT_DIR="$V1_OUTPUT_DIR/v2_area_adjusted"

# ── v3 output goes to a NEW versioned subdirectory — v1/v2 outputs untouched ───
OUTPUT_DIR="$V1_OUTPUT_DIR/v3_significance"
V1_PER_DUCT_CSV="$V1_OUTPUT_DIR/holeyness_per_duct.csv"
V2_JSON="$V2_OUTPUT_DIR/holeyness_validation_v2.json"

# ── Fixed raw-input paths (same as v1/v2 — used only for mean-agg re-derivation) ──
EXPORT="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
ANNOTATION_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
SLIDE_DIMENSIONS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"

echo "========================================================"
echo "  Holeyness v3 significance — section ${SECTION}"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  v1 per-duct  : $V1_PER_DUCT_CSV"
echo "  v2 JSON      : $V2_JSON"
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
echo -n "Export       : "; ls -lh "$EXPORT"            2>/dev/null || echo "NOT FOUND"
echo -n "Slide dims   : "; ls -lh "$SLIDE_DIMENSIONS"   2>/dev/null || echo "NOT FOUND"
echo -n "Results CSV  : "; ls -lh "$RESULTS_CSV"        2>/dev/null || echo "NOT FOUND"
echo -n "Slide list   : "; ls -lh "$SLIDE_LIST"         2>/dev/null || echo "NOT FOUND"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.holeyness_v3_significance \
    --section          "$SECTION" \
    --v1-per-duct-csv  "$V1_PER_DUCT_CSV" \
    --v2-json          "$V2_JSON" \
    --export           "$EXPORT" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --results          "$RESULTS_CSV" \
    --slide-list       "$SLIDE_LIST" \
    --output-dir       "$OUTPUT_DIR" \
    --n-permutations   1000

echo ""
echo "========================================================"
echo "  HOLEYNESS V3 SIGNIFICANCE COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/holeyness_validation_v3.json"
echo "  $OUTPUT_DIR/holeyness_validation_v3.md"
echo "  $OUTPUT_DIR/v3_per_slide_partial_bar.{pdf,png}"
echo ""
echo "v1/v2 outputs (unchanged):"
echo "  $V1_OUTPUT_DIR/holeyness_per_duct.csv"
echo "  $V2_OUTPUT_DIR/holeyness_validation_v2.json"
echo ""
