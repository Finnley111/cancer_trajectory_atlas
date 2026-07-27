#!/bin/bash
# Duct-level holeyness validation against pseudotime.
#
# Assignment strategy: cross-file UUID join.
#   data/annotations_ratio/<slide>.json  (polygon geometry, ratio coords)
#   combined_matched_measurements.txt    (hole %, keyed by QuPath UUID)
#
# Reads:
#   $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt
#   $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
#   $SCRATCH/results/per_section/atlas_<SECTION>/results.csv
#   ~/cancer_trajectory_atlas/data/annotations_ratio/*.json
#
# Writes:
#   $SCRATCH/results/holeyness/<SECTION>/holeyness_per_duct.csv
#   $SCRATCH/results/holeyness/<SECTION>/holeyness_validation.json
#   $SCRATCH/results/holeyness/<SECTION>/scatter_pt_vs_hole_pct.{pdf,png}
#   $SCRATCH/results/holeyness/<SECTION>/scatter_hole_pct_vs_nd.{pdf,png}
#
# Does NOT modify any existing pipeline output.
#
# Usage (section 2M-1):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_validation.sh
#
# To run 2M-2 later (adjust the four variables below):
#   SECTION="2M-2"
#   RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-2/results.csv"
#   SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section2.txt"
#   OUTPUT_DIR="$SCRATCH/results/holeyness/2M-2"

#SBATCH --account=def-lmarti46
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=holeyness_validation
#SBATCH --output=logs/holeyness_validation-%j.out

set -euo pipefail
mkdir -p logs

# ── Section-specific parameters (edit these for 2M-2) ────────────────────────
SECTION="2M-1"
RESULTS_CSV="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
OUTPUT_DIR="$SCRATCH/results/holeyness/2M-1"

# ── Fixed input paths ─────────────────────────────────────────────────────────
EXPORT="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
ANNOTATION_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
SLIDE_DIMENSIONS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"

echo "========================================================"
echo "  Holeyness validation — section ${SECTION}"
echo "  Job ID      : ${SLURM_JOB_ID:-local}"
echo "  Export      : $EXPORT"
echo "  Annotations : $ANNOTATION_DIR"
echo "  Slide dims  : $SLIDE_DIMENSIONS"
echo "  Results     : $RESULTS_CSV"
echo "  Slide list  : $SLIDE_LIST"
echo "  Output dir  : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Export      : "; ls -lh "$EXPORT"      2>/dev/null || echo "NOT FOUND"
echo -n "Slide dims  : "; ls -lh "$SLIDE_DIMENSIONS" 2>/dev/null || echo "NOT FOUND"
echo -n "Results CSV : "; ls -lh "$RESULTS_CSV" 2>/dev/null || echo "NOT FOUND"
echo -n "Slide list  : "; ls -lh "$SLIDE_LIST"  2>/dev/null || echo "NOT FOUND"
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
    --aggregation      median \
    --n-permutations   1000

echo ""
echo "========================================================"
echo "  HOLEYNESS VALIDATION COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/holeyness_per_duct.csv"
echo "  $OUTPUT_DIR/holeyness_validation.json"
echo "  $OUTPUT_DIR/scatter_pt_vs_hole_pct.{pdf,png}"
echo "  $OUTPUT_DIR/scatter_hole_pct_vs_nd.{pdf,png}"
echo ""
