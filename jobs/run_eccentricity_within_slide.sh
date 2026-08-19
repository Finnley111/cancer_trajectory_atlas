#!/bin/bash
# Is the late-pseudotime structure biology, or is it one slide?
#
# eccentricity_check checked Task A's directionality WITHIN slides but ran Task
# B's enrichment and late subclustering on the GLOBAL top decile, which on the
# holeyroot axis is 55% one slide in 2M-1 and 43% in 2M-2 against 12.5% uniform.
# Its own report calls that a "SEPARATE AND OVERRIDING CONCERN" and then issues
# the Task B verdicts anyway.
#
# It matters most in 2M-1, where the two Task B verdicts CONTRADICT:
#     Verdict 2  0/6 bidirectional, 5 unidirectional        -> trajectory-like
#     Verdict 3  2/6 opposing late subclusters              -> eccentricity-like
#                (texture_entropy, h_intensity)
# and Verdict 4 reports "survives both tests" by counting Tasks A and B while
# dropping the subclustering test 2M-1 fails.
#
# TASK 1 is the decisive cheap test: the late SUBCLUSTER x SLIDE contingency
# table with Cramer's V. run_late_subclustering already computes the per-cluster
# slide breakdown; it is simply never surfaced. High V => the late subclusters
# ARE slides and Verdict 3 is a batch split. TASK 2 and TASK 3 redo the
# enrichment and the subclustering inside each slide.
#
# POINT THIS AT THE SAME RUNS eccentricity_check WAS POINTED AT. Default below is
# the holeyroot tree, which is what produced the report this responds to. It can
# also be pointed at an axis exported by export_anchor_axis:
#   RUN_BASE=$SCRATCH/results/holeyroot_experiment/anchor_axes/area_stratified \
#   OUT_SUFFIX=_area_stratified sbatch <this script>
#
# CPU only, no DPT, no re-embedding. Minutes.
#
# READS (READ-ONLY): the run tree's adata_full.h5ad
# WRITES (NEW ONLY): $SCRATCH/results/eccentricity_within_slide<OUT_SUFFIX>
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_eccentricity_within_slide.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=ecc_within
#SBATCH --output=logs/ecc_within-%j.out

set -euo pipefail
mkdir -p logs

SECTIONS=("2M-1" "2M-2")
PER_SECTION_BASE="${RUN_BASE:-$SCRATCH/results/per_section_holeyroot}"
RUN_DIRS=(
    "$PER_SECTION_BASE/atlas_2M-1"
    "$PER_SECTION_BASE/atlas_2M-2"
)
OUT_DIR="$SCRATCH/results/eccentricity_within_slide${OUT_SUFFIX:-}"

TAIL="${TAIL:-0.10}"
MIN_PATCHES="${MIN_PATCHES:-200}"
MIN_LATE="${MIN_LATE:-60}"

echo "============================================================================"
echo "  Late-tail structure: biology or one slide?"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Runs   : $PER_SECTION_BASE"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

# Refuse to silently overwrite: a holeyroot pass and an area-stratified pass are
# the whole point of this analysis and must not be confusable after the fact.
if [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUT_DIR already exists and is not empty."
    echo "       Set OUT_SUFFIX to write elsewhere, or FORCE=1 to overwrite."
    exit 1
fi

MISSING=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo -n "  $RUN_DIR/adata_full.h5ad : "
    if [ -e "$RUN_DIR/adata_full.h5ad" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.eccentricity_within_slide \
    --sections   "${SECTIONS[@]}" \
    --run-dirs   "${RUN_DIRS[@]}" \
    --output-dir "$OUT_DIR" \
    --tail-fraction         "$TAIL" \
    --min-patches-per-slide "$MIN_PATCHES" \
    --min-late-per-slide    "$MIN_LATE"

echo ""
echo "============================================================================"
echo "  COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/eccentricity_within_slide.md    <- read this"
echo "  $OUT_DIR/eccentricity_within_slide.json"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. task1 subcluster_by_slide.cramers_v. High (>=0.5) means the cohort"
echo "      late subclusters ARE slides, so eccentricity_check's Verdict 3 is a"
echo "      batch split and 2M-1's contradiction dissolves. Low (<0.3) means the"
echo "      split is phenotypic and Verdict 4 is overstated."
echo "   2. task2 per_feature_across_slides. A feature bidirectional cohort-wide"
echo "      but in no slide was showing a slide contrast."
echo "   3. task3 per_feature_across_slides, and the skipped-slide list — a thin"
echo "      late tail is reported, never fitted."
echo ""
echo "  Within-slide thresholds are computed inside each slide, so these numbers"
echo "  are NOT on the cohort scale. Compare PATTERNS, never values."
