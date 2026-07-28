#!/bin/bash
# Holeyness final consolidation: one authoritative report over v1-v3b for section 2M-1.
#
# Corrects four methodological problems accumulated across v1-v3b (see
# analysis/holeyness_final.py docstring for full detail):
#   1. Replaces v3/v3b's CIRCULAR flagged-vs-other slide comparison with a
#      non-circular check across all 8 slides (no subsetting).
#   2. Labels every reported quantity PRIMARY or EXPLORATORY and counts the
#      exploratory correlations computed across v1-v3b (no correction was ever
#      applied to them).
#   3. States the ESTIMAND precisely (retained, >=1-patch ducts only — the 571
#      excluded ducts are systematically smaller/less holey, per v2).
#   4. Corrects v3b's sign/magnitude-conflated "3/3 strengthened" verdict with a
#      direction-preserving re-interpretation.
# Also reports raw rho and area-adjusted partial rho as CO-PRIMARY (neither is "the"
# result — the choice depends on an unresolved mediator-vs-confound question), and
# optionally (Task F) tests an overlap-based patch-to-duct assignment rule as a
# direct fix for the exclusion bias, gated on the raw-input paths below being
# supplied — omit them to skip Task F automatically.
#
# Reads (ALL READ-ONLY — v1/v2/v3/v3b outputs are never modified or rerun):
#   $SCRATCH/results/holeyness/2M-1/holeyness_per_duct.csv
#   $SCRATCH/results/holeyness/2M-1/holeyness_validation.json
#   $SCRATCH/results/holeyness/2M-1/v2_area_adjusted/holeyness_validation_v2.json
#   $SCRATCH/results/holeyness/2M-1/v3_significance/holeyness_validation_v3.json
#   $SCRATCH/results/holeyness/2M-1/v3b_patch_count_check/holeyness_validation_v3b.json
#   (Task F only, optional) combined_matched_measurements.txt, slide_dimensions.json,
#   per-section results.csv, annotations_ratio/*.json, slide list
#
# Writes (NEW directory — v1/v2/v3/v3b outputs are never touched):
#   $SCRATCH/results/holeyness/2M-1/final/holeyness_final_report.md
#   $SCRATCH/results/holeyness/2M-1/final/holeyness_final.json
#   $SCRATCH/results/holeyness/2M-1/final/final_scatter_partial_rho_vs_median_n_patches.{pdf,png}
#   $SCRATCH/results/holeyness/2M-1/final/final_scatter_pt_vs_hole_pct_by_area.{pdf,png}
#
# Usage (section 2M-1, after v1/v2/v3/v3b have all already run):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_final.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=holeyness_final
#SBATCH --output=logs/holeyness_final-%j.out

set -euo pipefail
mkdir -p logs

# ── Section-specific parameters ───────────────────────────────────────────────
SECTION="2M-1"
V1_OUTPUT_DIR="$SCRATCH/results/holeyness/2M-1"
V2_OUTPUT_DIR="$V1_OUTPUT_DIR/v2_area_adjusted"
V3_OUTPUT_DIR="$V1_OUTPUT_DIR/v3_significance"
V3B_OUTPUT_DIR="$V1_OUTPUT_DIR/v3b_patch_count_check"

# ── final output goes to a NEW directory — v1/v2/v3/v3b outputs untouched ────
OUTPUT_DIR="$V1_OUTPUT_DIR/final"
V1_PER_DUCT_CSV="$V1_OUTPUT_DIR/holeyness_per_duct.csv"
V1_JSON="$V1_OUTPUT_DIR/holeyness_validation.json"
V2_JSON="$V2_OUTPUT_DIR/holeyness_validation_v2.json"
V3_JSON="$V3_OUTPUT_DIR/holeyness_validation_v3.json"
V3B_JSON="$V3B_OUTPUT_DIR/holeyness_validation_v3b.json"

# ── Optional raw-input paths (Task F ONLY — omit/leave unset to skip Task F) ──
RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
EXPORT="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
ANNOTATION_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
SLIDE_DIMENSIONS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"

echo "========================================================"
echo "  Holeyness final consolidation — section ${SECTION}"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  v1 per-duct  : $V1_PER_DUCT_CSV"
echo "  v1 JSON      : $V1_JSON"
echo "  v2 JSON      : $V2_JSON"
echo "  v3 JSON      : $V3_JSON"
echo "  v3b JSON     : $V3B_JSON"
echo "  Output dir   : $OUTPUT_DIR"
echo "  (Task F, optional) Export/Annotations/SlideDims/Results/SlideList below"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "v1 per-duct  : "; ls -lh "$V1_PER_DUCT_CSV" 2>/dev/null || echo "NOT FOUND — run run_holeyness_validation.sh first"
echo -n "v1 JSON      : "; ls -lh "$V1_JSON"         2>/dev/null || echo "NOT FOUND — run run_holeyness_validation.sh first"
echo -n "v2 JSON      : "; ls -lh "$V2_JSON"         2>/dev/null || echo "NOT FOUND — run run_holeyness_validation_v2.sh first"
echo -n "v3 JSON      : "; ls -lh "$V3_JSON"         2>/dev/null || echo "NOT FOUND — run run_holeyness_v3_significance.sh first"
echo -n "v3b JSON     : "; ls -lh "$V3B_JSON"        2>/dev/null || echo "NOT FOUND — run run_holeyness_v3b_patch_count_check.sh first"
echo -n "(F) Export   : "; ls -lh "$EXPORT"          2>/dev/null || echo "not found — Task F will be skipped"
echo -n "(F) SlideDim : "; ls -lh "$SLIDE_DIMENSIONS" 2>/dev/null || echo "not found — Task F will be skipped"
echo -n "(F) Results  : "; ls -lh "$RESULTS_CSV"      2>/dev/null || echo "not found — Task F will be skipped"
echo -n "(F) SlideList: "; ls -lh "$SLIDE_LIST"       2>/dev/null || echo "not found — Task F will be skipped"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.holeyness_final \
    --section          "$SECTION" \
    --v1-per-duct-csv  "$V1_PER_DUCT_CSV" \
    --v1-json          "$V1_JSON" \
    --v2-json          "$V2_JSON" \
    --v3-json          "$V3_JSON" \
    --v3b-json         "$V3B_JSON" \
    --output-dir       "$OUTPUT_DIR" \
    --export           "$EXPORT" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --results          "$RESULTS_CSV" \
    --slide-list       "$SLIDE_LIST"

echo ""
echo "========================================================"
echo "  HOLEYNESS FINAL CONSOLIDATION COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/holeyness_final_report.md"
echo "  $OUTPUT_DIR/holeyness_final.json"
echo "  $OUTPUT_DIR/final_scatter_partial_rho_vs_median_n_patches.{pdf,png}"
echo "  $OUTPUT_DIR/final_scatter_pt_vs_hole_pct_by_area.{pdf,png}"
echo ""
echo "v1/v2/v3/v3b outputs (unchanged, retained as provenance):"
echo "  $V1_OUTPUT_DIR/holeyness_per_duct.csv, holeyness_validation.json"
echo "  $V2_OUTPUT_DIR/holeyness_validation_v2.json"
echo "  $V3_OUTPUT_DIR/holeyness_validation_v3.json"
echo "  $V3B_OUTPUT_DIR/holeyness_validation_v3b.json"
echo ""
