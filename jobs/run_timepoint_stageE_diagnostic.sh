#!/bin/bash
# Timepoint cohort: Stage E -- diagnostic analysis (CPU-only).
#
# NOT A VALIDATION CLAIM. This runs despite a FAILED stain gate whose driver is
# specifically HEMATOXYLIN INTENSITY (not broad staining differences), an effect
# consistent with either a reagent-side confound or genuine cellularity change
# with tumor age. It reports, side by side: the RAW pseudotime-vs-weeks
# correlation, the PARTIAL correlation controlling for each hematoxylin measure,
# and hematoxylin's own correlation with weeks on the identical mouse set --
# with and without the 3 ambiguous-provenance suffix slides. Its report is
# required to open by stating what it is and is not.
#
# Hematoxylin values are READ from Stage B v2's output, never recomputed.
#
# Reads (read-only):
#   $SCRATCH/results/timepoint_cohort/stageD_projection/stageD_projection.json
#   $SCRATCH/results/timepoint_cohort/stageB_v2_fullres/stageB_v2_fullres.json
#
# Writes:
#   $SCRATCH/results/timepoint_cohort/stageE_diagnostic/stageE_diagnostic.{json,md}
#
# Usage (after Stage D completes):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stageE_diagnostic.sh
# or chained automatically via submit_timepoint_stageDE.sh.

#SBATCH --account=def-lmarti46
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=timepoint_stageE
#SBATCH --output=logs/timepoint_stageE-%j.out

# NOTE on resources: this reads two small JSON files and runs 1000 permutations
# over at most 20 mouse-level rows. Seconds of work; 1h/16G is a generous margin,
# not an estimate of need. No GPU.

set -euo pipefail
mkdir -p logs

STAGED_JSON="$SCRATCH/results/timepoint_cohort/stageD_projection/stageD_projection.json"
STAGEB_JSON="$SCRATCH/results/timepoint_cohort/stageB_v2_fullres/stageB_v2_fullres.json"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageE_diagnostic"
N_PERMUTATIONS=1000

echo "========================================================"
echo "  Timepoint cohort — Stage E: diagnostic analysis"
echo "  Job ID          : ${SLURM_JOB_ID:-local}"
echo "  Stage D JSON     : $STAGED_JSON"
echo "  Stage B v2 JSON  : $STAGEB_JSON"
echo "  Output dir       : $OUTPUT_DIR"
echo "  Permutations     : $N_PERMUTATIONS"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
FAIL=0
echo -n "Stage D JSON    : "; ls -lh "$STAGED_JSON" 2>/dev/null || {
  echo "NOT FOUND — run jobs/run_timepoint_stageD_projection.sh first"; FAIL=1; }
echo -n "Stage B v2 JSON : "; ls -lh "$STAGEB_JSON" 2>/dev/null || {
  echo "NOT FOUND — run jobs/run_timepoint_stain_homogeneity_v2.sh first"; FAIL=1; }
if [ "$FAIL" -ne 0 ]; then
  echo "*** HARD FAIL: required inputs missing. Not starting."
  exit 1
fi
echo "======================"
echo ""

mkdir -p "$OUTPUT_DIR"

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_diagnostic \
    --stageD-json    "$STAGED_JSON" \
    --stageB-json    "$STAGEB_JSON" \
    --output-dir     "$OUTPUT_DIR" \
    --n-permutations "$N_PERMUTATIONS"

echo ""
echo "========================================================"
echo "  STAGE E COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageE_diagnostic.json"
echo "  $OUTPUT_DIR/stageE_diagnostic.md"
echo ""
echo "*** STOP HERE. ***"
echo "Report the RAW and PARTIAL correlations TOGETHER with the projection-validity"
echo "section — never the correlations alone. This is a diagnostic, not a validated"
echo "timepoint result, and must not be presented as one. Do not build any"
echo "pathologist-facing figure or summary from it in this pass."
echo ""
