#!/bin/bash
# v3 comparison report — each config against v2, plus v3a vs v3c.
#
# REPORTS NUMBERS. ADJUDICATES NOTHING. No configuration is declared better; a
# reduced pseudotime_std and a sign-flipped nuclear_density correlation are both
# explicitly marked as NON-validating in the output.
#
# ASSESSMENT CRITERION, FIXED BEFORE THE RUN
#   Random 20-root sets reproduce v2 pseudotime at |rho| 0.78-0.89, so a root-rule
#   change is EXPECTED to alter orientation and root membership, not ordering.
#   Config A landing in that band is the PREDICTION. |rho| < 0.7 is flagged
#   prominently as CONTRADICTING the random-root finding.
#
# ⚠ TWO GAPS THE REPORT DECLARES RATHER THAN FILLS
#   1. The raw pre-normalisation pseudotime range (pt_max - pt_min) is PRINTED by
#      compute_dpt_multi_root but never persisted. It is parsed from the SLURM
#      logs listed in LOGS below when available, and reported UNRECOVERABLE
#      otherwise. It is never fabricated. Edit LOGS to point at the real .out
#      files once the jobs have run — otherwise the "std as % of raw range"
#      column will legitimately read n/a.
#   2. v2's cellularity-confound verdicts are READ from its existing
#      cellularity_confound/cellularity_confound.json. They are NOT regenerated,
#      because analyze_run_nuclear_density writes into the tree it analyses and
#      per_section_v2 must not be modified by this experiment.
#
# READ-ONLY with respect to every run tree, including v2.
# WRITES (NEW ONLY): $SCRATCH/results/v3_root_experiment/compare
#
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3_compare.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=v3_compare
#SBATCH --output=logs/v3_compare-%j.out

set -euo pipefail
mkdir -p logs
# NOT $(dirname "$0"): sbatch copies the script to a spool dir, so $0 is not
# the repo. SLURM_SUBMIT_DIR is unset for an interactive run, hence the fallback.
V3_JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$V3_JOBS_DIR/_v3_common.sh" ] || V3_JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$V3_JOBS_DIR/_v3_common.sh"

OUT_DIR="$V3_COMPARE/compare"

# Point these at the real .out files after the runs finish. Globs that match
# nothing are dropped below rather than passed through literally.
LOGS_GLOB=(
    "$HOME/cancer_trajectory_atlas/logs/per_section_v2-"*.out
    "$HOME/cancer_trajectory_atlas/logs/v3a_holeyroot-"*.out
    "$HOME/cancer_trajectory_atlas/logs/v3b_relaxed-"*.out
    "$HOME/cancer_trajectory_atlas/logs/v3c_both-"*.out
)

echo "============================================================"
echo "  v3 comparison report"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Output : $OUT_DIR   (NEW)"
echo "  v2 reference (READ-ONLY): $V2_BASE/atlas_${SECTION}"
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"

if [ ! -f "$V2_BASE/atlas_${SECTION}/results.csv" ]; then
    echo "ERROR: v2 reference not found at $V2_BASE/atlas_${SECTION}/results.csv"
    exit 1
fi

LABELS=(v3a v3b v3c)
DIRS=(
    "$V3A_BASE/atlas_${SECTION}"
    "$V3B_BASE/atlas_${SECTION}"
    "$V3C_BASE/atlas_${SECTION}"
)
KEEP_L=(); KEEP_D=()
for i in "${!LABELS[@]}"; do
    if [ -f "${DIRS[$i]}/results.csv" ]; then
        KEEP_L+=("${LABELS[$i]}"); KEEP_D+=("${DIRS[$i]}")
        echo "  include ${LABELS[$i]}"
    else
        echo "  SKIP    ${LABELS[$i]} — no results.csv at ${DIRS[$i]}"
    fi
done
[ "${#KEEP_L[@]}" -gt 0 ] || { echo "ERROR: no config produced results.csv."; exit 1; }

LOGS=()
for g in "${LOGS_GLOB[@]}"; do
    [ -e "$g" ] && LOGS+=("$g")
done
if [ "${#LOGS[@]}" -eq 0 ]; then
    echo ""
    echo "  NOTE: no SLURM logs matched. The raw pseudotime range is not stored in"
    echo "        any artifact, so 'pseudotime_std as % of raw range' will be"
    echo "        reported UNRECOVERABLE rather than invented. That is intended"
    echo "        behaviour, not a failure."
else
    echo "  ${#LOGS[@]} log file(s) available for raw-range recovery."
fi
echo "============================================"

v3_load_env
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.v3_root_experiment_compare \
    --v2-dir        "$V2_BASE/atlas_${SECTION}" \
    --config-labels "${KEEP_L[@]}" \
    --config-dirs   "${KEEP_D[@]}" \
    --output-dir    "$OUT_DIR" \
    ${LOGS[@]+--logs "${LOGS[@]}"}

echo ""
echo "============================================================"
echo "  COMPARISON COMPLETE"
echo "============================================================"
echo "  $OUT_DIR/v3_comparison.md    <- read this"
echo "  $OUT_DIR/v3_comparison.json"
echo ""
echo "Read it alongside $V3_COMPARE/root_sheets/. The report states numbers;"
echo "it does not decide which configuration is better."
