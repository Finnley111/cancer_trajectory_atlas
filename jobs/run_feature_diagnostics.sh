#!/bin/bash
# Morphological feature diagnostics for the cancer trajectory atlas.
#
# Reads existing per-section results.csv files from $SCRATCH.
# Does NOT modify any pipeline output, rerun any pipeline stage,
# or overwrite any existing results.csv, validation.json,
# or cellularity_confound.json.
#
# Output (on the login node):
#   ~/cancer_trajectory_atlas/reports/morphological_features_diagnostics_results.md
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_feature_diagnostics.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=feature_diagnostics
#SBATCH --output=logs/feature_diagnostics-%j.out

set -euo pipefail

mkdir -p logs

# ── Input paths (absolute, per-section pipeline outputs on $SCRATCH) ──────────
SECTION1_CSV="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
SECTION2_CSV="$SCRATCH/results/per_section/atlas_2M-2/results.csv"

# ── Report written to the local repo checkout ─────────────────────────────────
REPORT_PATH="$HOME/cancer_trajectory_atlas/reports/morphological_features_diagnostics_results.md"

echo "========================================================"
echo "  Cancer Trajectory Atlas — Feature Diagnostics"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  2M-1 CSV : $SECTION1_CSV"
echo "  2M-2 CSV : $SECTION2_CSV"
echo "  Report   : $REPORT_PATH"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Section 2M-1: "
ls -lh "$SECTION1_CSV" 2>/dev/null || echo "NOT FOUND — $SECTION1_CSV"
echo -n "Section 2M-2: "
ls -lh "$SECTION2_CSV" 2>/dev/null || echo "NOT FOUND — $SECTION2_CSV"
echo "========================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.diagnostics.audit_feature_diagnostics \
    --section1-results "$SECTION1_CSV" \
    --section2-results "$SECTION2_CSV" \
    --output-report    "$REPORT_PATH"

echo ""
echo "========================================================"
echo "  FEATURE DIAGNOSTICS COMPLETE"
echo "========================================================"
echo ""
echo "Report written to:"
echo "  $REPORT_PATH"
echo ""
