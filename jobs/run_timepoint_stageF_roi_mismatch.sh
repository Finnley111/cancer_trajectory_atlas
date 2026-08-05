#!/bin/bash
# Timepoint cohort: Stage F -- ROI / patch-composition mismatch check (CPU-only).
#
# WHY: Stage D returned frac_beyond_training_p99 = 1.0 for EVERY one of the 29
# slides -- total extrapolation with no discrimination between slides. Stage E's
# pseudotime-vs-weeks correlation was therefore computed entirely outside the
# training manifold's support, which makes it uninterpretable as it stands.
#
# HYPOTHESIS UNDER TEST: the 29 timepoint slides have no annotations, so Stage D
# patched them WHOLE-SLIDE, while the manifold was built EXCLUSIVELY from
# annotated tumor ROI patches. Stroma/necrosis/non-tumor tissue is therefore in
# the projection set but was never in training -- a patch-COMPOSITION mismatch,
# distinct from the hematoxylin confound Stage B found.
#
# This job does NOT validate the timepoint hypothesis and does NOT conclude
# whether timepoint projection is valid. It reports which of three pre-specified
# outcomes the evidence matches, and stops.
#
# NO GPU NEEDED -- confirmed, not assumed: projection is never re-run. Stage D
# already saved per-patch pseudotime and nearest-neighbour distances
# (per_slide/*.npy), so Tasks B and C are pure MASKING of those arrays. The only
# new computation is regenerating patch pixels (Stage D cached 768-dim vectors,
# not images) and running the six morphological features on a per-slide sample.
#
# The training cohort is NEVER re-extracted: its per-patch morphology is read
# from the manifold run's own results.csv (run_all.py writes all six features
# there alongside x/y/slide/cluster/pseudotime).
#
# Reads (read-only):
#   $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json
#   $SCRATCH/results/timepoint_cohort/stageD_projection/stageD_projection.json
#   $SCRATCH/results/timepoint_cohort/stageD_projection/per_slide/*.npy
#   $SCRATCH/results/timepoint_cohort/stageE_diagnostic/stageE_diagnostic.json
#   $SCRATCH/results/baseline/atlas_none_harmony_median/results.csv
#   $SCRATCH/data/timepoint_x5_full/*.png
#
# Writes:
#   $SCRATCH/results/timepoint_cohort/stageF_roi_mismatch_check/stageF_roi_mismatch.{json,md}
#   $SCRATCH/results/timepoint_cohort/stageF_roi_mismatch_check/per_slide_morph/*.npz
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stageF_roi_mismatch.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=timepoint_stageF
#SBATCH --output=logs/timepoint_stageF-%j.out

# NOTE on --mem=128G: same reason as Stage D -- these are full-width no-crop
# PNGs (~110000 x 45000 decodes to ~15 GB as an RGB array), plus up to ~10 GB
# for the patch array during np.array(). Not the pipeline's usual 64G.
#
# NOTE on --time=12:00:00: an HONEST OVER-ESTIMATE, not a measurement. Cost is
# dominated by re-scanning the 29 slides' patch grids (~15.6M positions), the
# same step that dominated Stage D; the morphology itself runs on only ~3000
# patches per slide. RESUMABLE: each slide's morphology is cached to
# per_slide_morph/{stem}_morph.npz as it completes, so a timeout resumes from
# the next unprocessed slide -- resubmit rather than restarting from scratch.
#
# If Stage D's own measured runtime (seff <stageD_jobid>) is now known, use it
# to size this better: the re-scan here is Stage D's extraction step WITHOUT the
# GPU inference, so Stage D's elapsed time is a generous upper bound.

set -euo pipefail
mkdir -p logs

STAGEA_JSON="$SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json"
STAGED_DIR="$SCRATCH/results/timepoint_cohort/stageD_projection"
STAGED_JSON="$STAGED_DIR/stageD_projection.json"
STAGED_PER_SLIDE="$STAGED_DIR/per_slide"
STAGEE_JSON="$SCRATCH/results/timepoint_cohort/stageE_diagnostic/stageE_diagnostic.json"
TRAINING_CSV="$SCRATCH/results/baseline/atlas_none_harmony_median/results.csv"
PNG_DIR="$SCRATCH/data/timepoint_x5_full"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageF_roi_mismatch_check"
MORPH_SAMPLE=3000
MIN_FILTERED_PATCHES=50
N_PERMUTATIONS=1000

echo "========================================================"
echo "  Timepoint cohort — Stage F: ROI/composition mismatch check"
echo "  Job ID            : ${SLURM_JOB_ID:-local}"
echo "  Stage A inventory  : $STAGEA_JSON"
echo "  Stage D JSON       : $STAGED_JSON"
echo "  Stage D per-slide  : $STAGED_PER_SLIDE"
echo "  Stage E JSON       : $STAGEE_JSON"
echo "  Training results   : $TRAINING_CSV"
echo "  PNG dir            : $PNG_DIR"
echo "  Output dir         : $OUTPUT_DIR"
echo "  Morph sample/slide : $MORPH_SAMPLE  (0 = all patches, far slower)"
echo "  Min filtered patch : $MIN_FILTERED_PATCHES"
echo "========================================================"

echo ""
echo "=== Pre-run checks (fail fast) ==="
FAIL=0
check() {  # $1=label  $2=path
  printf '%-22s: ' "$1"
  ls -d "$2" >/dev/null 2>&1 && ls -lhd "$2" | awk '{print $5, $9}' || { echo "NOT FOUND — $3"; FAIL=1; }
}
check "Stage A inventory"  "$STAGEA_JSON"      "run jobs/run_timepoint_cohort_inventory_v2.sh"
check "Stage D JSON"       "$STAGED_JSON"      "run jobs/run_timepoint_stageD_projection.sh"
check "Stage D per_slide"  "$STAGED_PER_SLIDE" "Stage D must have written per-patch .npy arrays"
check "Stage E JSON"       "$STAGEE_JSON"      "run jobs/run_timepoint_stageE_diagnostic.sh"
check "Training results"   "$TRAINING_CSV"     "must be the SAME run as the projector Stage D used"
check "PNG dir"            "$PNG_DIR"          "run jobs/run_timepoint_convert_stageC.sh"

if [ -d "$STAGED_PER_SLIDE" ]; then
  echo "  per-slide arrays  : $(find "$STAGED_PER_SLIDE" -name 'projected_pt_*.npy' | wc -l) pseudotime, \
$(find "$STAGED_PER_SLIDE" -name 'nn_mean_knn_*.npy' | wc -l) nn-distance (expect 29 each)"
fi
if [ "$FAIL" -ne 0 ]; then
  echo "*** HARD FAIL: required inputs missing (see above). Not starting."
  exit 1
fi
echo "======================"
echo ""

mkdir -p "$OUTPUT_DIR"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_roi_mismatch \
    --stageA-inventory-json  "$STAGEA_JSON" \
    --stageD-json            "$STAGED_JSON" \
    --stageD-per-slide-dir   "$STAGED_PER_SLIDE" \
    --stageE-json            "$STAGEE_JSON" \
    --training-results-csv   "$TRAINING_CSV" \
    --png-dir                "$PNG_DIR" \
    --output-dir             "$OUTPUT_DIR" \
    --morph-sample-per-slide "$MORPH_SAMPLE" \
    --min-filtered-patches   "$MIN_FILTERED_PATCHES" \
    --n-permutations         "$N_PERMUTATIONS"

echo ""
echo "========================================================"
echo "  STAGE F COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageF_roi_mismatch.json"
echo "  $OUTPUT_DIR/stageF_roi_mismatch.md"
echo "  $OUTPUT_DIR/per_slide_morph/   (cached morphology; makes re-runs cheap)"
echo ""
echo "*** Read the report IN ORDER: Task A (what tissue is extrapolating), then"
echo "*** Task B (does filtering fix it), then Task C (does the correlation"
echo "*** survive). Never quote Task C's correlations without A and B alongside."
echo ""
echo "The report states which of the three pre-specified outcomes the data match."
echo "It deliberately does NOT conclude whether timepoint projection is valid —"
echo "that interpretation is left to you."
echo ""
