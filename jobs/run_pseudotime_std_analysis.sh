#!/bin/bash
# TASK 2 — per-patch disagreement across the 20 DPT roots.
#
# WHY:
#   pseudotime_std = std(pt_matrix, axis=0) over the 20 per-root DPT runs
#   (analysis/diffusion.py). It has never been analysed. root_sensitivity showed
#   the AGGREGATE ordering is stable — random 20-root sets reproduce production
#   pseudotime at |rho| 0.78-0.89 — but that says nothing about whether any
#   individual patch has a well-determined position on the axis.
#
# SCALE WARNING:
#   pseudotime is min-max normalised to [0,1] (diffusion.py:193); pseudotime_std
#   is stored RAW and pre-normalisation (diffusion.py:186). The two are NOT
#   comparable. The conversion needs the raw pre-normalisation range, which
#   compute_dpt_multi_root PRINTS but never STORES. The module searches nearby
#   logs for it; pass more with --raw-range. If it cannot be found the module
#   says so and reports raw values with an explicit non-comparability statement.
#   It never fabricates a normalisation.
#
# READS (READ-ONLY): <run_dir>/results.csv for each section. Baseline untouched.
# WRITES (NEW directory): $SCRATCH/results/pseudotime_std_analysis/
#
# CPU only, no GPU, no model, seconds of work.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_pseudotime_std_analysis.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=pt_std
#SBATCH --output=logs/pt_std-%j.out

set -euo pipefail
mkdir -p logs

SECTIONS=("2M-1" "2M-2")
PER_SECTION_BASE="$SCRATCH/results/per_section"
RUN_DIRS=("$PER_SECTION_BASE/atlas_2M-1" "$PER_SECTION_BASE/atlas_2M-2")
OUTPUT_DIR="$SCRATCH/results/pseudotime_std_analysis"

echo "============================================================"
echo "  TASK 2 — pseudotime_std per-patch root disagreement"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  Output dir : $OUTPUT_DIR  (NEW — baseline untouched)"
echo "============================================================"

MISSING=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo -n "  $RUN_DIR/results.csv : "
    ls -lh "$RUN_DIR/results.csv" 2>/dev/null || { echo "NOT FOUND"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing baseline results.csv"; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUTPUT_DIR"
cd ~

# Pass any surviving per-section run logs so the raw pseudotime range can be
# recovered; harmless if the globs match nothing.
python -m cancer_trajectory_atlas.analysis.pseudotime_std_analysis \
    --sections   "${SECTIONS[@]}" \
    --run-dirs   "${RUN_DIRS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --raw-range  $(ls ~/cancer_trajectory_atlas/logs/*per_section* 2>/dev/null || true)

echo ""
echo "  $OUTPUT_DIR/pseudotime_std_report.md   <- read this first"
echo "  $OUTPUT_DIR/pseudotime_std_analysis.json"
for S in "${SECTIONS[@]}"; do
    echo "  $OUTPUT_DIR/pseudotime_std_hist_${S}.png"
    echo "  $OUTPUT_DIR/pseudotime_vs_std_${S}.png"
done
