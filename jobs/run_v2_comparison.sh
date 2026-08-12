#!/bin/bash
# TASK 5 — baseline vs v2 comparison.
#
# Compares $SCRATCH/results/per_section/ (BASELINE, pre-fix) against
# $SCRATCH/results/per_section_v2/ (post-fix Task 1). Both are READ-ONLY.
#
# Reports:
#   - failure count and rate per section (new info; the pre-fix code silently
#     wrote 0.0, so there is no baseline equivalent)
#   - all six feature correlations, baseline vs v2, with absolute difference;
#     h_intensity reported BOTH ways so Fix 1c is isolable from 1a and 1b
#   - cellularity confound survivors/collapses in both runs, and whether any
#     verdict changed. Recomputed inline rather than via
#     analyze_run_nuclear_density, which writes into the run directory and would
#     therefore MODIFY THE BASELINE.
#   - Spearman(baseline pseudotime, v2 pseudotime) per section — the headline.
#     >= 0.98 means existing LOO results carry over and LOO need not re-run.
#   - whether the DPT root set changed, and by how many of the 20. v2 persists
#     dpt_root_candidates; the baseline does not, so the baseline set must be
#     reconstructed — and the module refuses to report it unless the
#     reconstruction first reproduces v2's STORED roots exactly.
#   - whether the cross-section directionality mismatch persists in v2.
#
# WRITES (NEW directory only): $SCRATCH/results/v2_comparison/
#
# CPU only, minutes.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_v2_comparison.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=v2_compare
#SBATCH --output=logs/v2_compare-%j.out

set -euo pipefail
mkdir -p logs

SECTIONS=("2M-1" "2M-2")
BASE="$SCRATCH/results/per_section"
V2="$SCRATCH/results/per_section_v2"
BASELINE_DIRS=("$BASE/atlas_2M-1" "$BASE/atlas_2M-2")
V2_DIRS=("$V2/atlas_2M-1" "$V2/atlas_2M-2")
OUTPUT_DIR="$SCRATCH/results/v2_comparison${OUT_SUFFIX:-}"

echo "============================================================"
echo "  TASK 5 — baseline vs v2"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Baseline   : $BASE   (read-only)"
echo "  v2         : $V2     (read-only)"
echo "  Output dir : $OUTPUT_DIR"
echo "============================================================"

if [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUTPUT_DIR already exists and is not empty."
    echo "       Set OUT_SUFFIX=... to write elsewhere, or FORCE=1 to overwrite."
    exit 1
fi

MISSING=0
for D in "${BASELINE_DIRS[@]}" "${V2_DIRS[@]}"; do
    echo -n "  $D/adata_full.h5ad : "
    ls -lh "$D/adata_full.h5ad" 2>/dev/null || { echo "NOT FOUND"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: both runs must exist. Submit jobs/run_per_section_v2.sh first."
    exit 1
}

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUTPUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.v2_comparison \
    --sections       "${SECTIONS[@]}" \
    --baseline-dirs  "${BASELINE_DIRS[@]}" \
    --v2-dirs        "${V2_DIRS[@]}" \
    --output-dir     "$OUTPUT_DIR"

echo ""
echo "  $OUTPUT_DIR/v2_comparison_report.md   <- read this first"
echo "  $OUTPUT_DIR/v2_comparison.json"
echo "  $OUTPUT_DIR/v2_feature_correlations.png"
echo ""
echo "Decision gate: per_section.<sec>.pseudotime.loo_carries_over"
echo "  true  -> existing LOO results stand; do NOT re-run LOO"
echo "  false -> LOO was measured on a different axis; re-run it against v2"
echo ""
echo "Baseline is unchanged:"
for D in "${BASELINE_DIRS[@]}"; do ls -l "$D/adata_full.h5ad"; done
