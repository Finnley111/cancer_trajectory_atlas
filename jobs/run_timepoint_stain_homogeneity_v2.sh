#!/bin/bash
# Timepoint cohort: Stage B v2 -- within-cohort stain homogeneity gate, FULL
# RESOLUTION (HARD GATE).
#
# Re-runs Stage B's gate logic (imported, not reimplemented, from
# analysis/timepoint_stain_homogeneity.py, which is NOT modified) against the
# corrected usable-slide list from Stage A v2, computing gate measures from
# each slide's full-resolution converted PNG instead of the coarse NDPI
# pyramid proxy Stage B v1 used. This is a full re-verification at full
# resolution and larger n (~29 usable slides vs Stage B v1's coarse-for-all/
# PNG-validated-for-7), NOT the same coarse-level run repeated. The output
# report includes a required comparison-to-v1 section (verdict, headline
# hematoxylin measures, and the 4Wvs12W pairwise comparison specifically).
#
# Depends on:
#   - run_timepoint_cohort_inventory_v2.sh (Stage A v2) having already run
#     and been reviewed
#   - jobs/run_stage2_reference_threshold.sh's output (reference_rank_biserial)
#   - Stage B v1's own output (for the required before/after comparison)
#   - the (now ~29) converted PNGs at $SCRATCH/data/timepoint_x5_full
#
# Reads (read-only):
#   $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json
#   $SCRATCH/data/timepoint_x5_full/*.png
#   raw .ndpi files referenced by Stage A v2's inventory (informational
#     coarse-vs-PNG validation only -- does not gate this run)
#   $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json
#   $SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity/stageB_stain_homogeneity.json  (v1, comparison only)
#
# Writes (NEW directory):
#   $SCRATCH/results/timepoint_cohort/stageB_v2_fullres/stageB_v2_fullres.json
#   $SCRATCH/results/timepoint_cohort/stageB_v2_fullres/stageB_v2_fullres.md
#
# STOP AFTER THIS JOB regardless of verdict -- no Stage D / projection work in
# this pass, and no compartment-stratified stain comparison either (separate,
# not-yet-planned analysis).
#
# Usage (after Stage A v2 has completed and been reviewed):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stain_homogeneity_v2.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=timepoint_stain_homogeneity_v2
#SBATCH --output=logs/timepoint_stain_homogeneity_v2-%j.out

# NOTE on --time=06:00:00: this is a GUESS, not a measured estimate. Stage B
# v1 ran the same rgb2hed/_deconvolve_hematoxylin computation on ~29 coarse
# NDPI reads (cheap) plus 7 full-res PNGs (the validation subset) in its
# 3-hour budget. This run computes full-res PNG features (compute_slide_
# stain_features, downsample_factor=8) for ALL ~29 usable slides -- roughly
# 4x the full-res decode work of Stage B v1's 7-slide validation step. The
# informational coarse-vs-PNG comparison (validate_coarse_vs_precomputed_
# fullres) reuses those already-computed full-res features and only adds
# ~29 cheap coarse NDPI reads on top -- it does NOT read each PNG a second
# time (an earlier draft of this module did; fixed before first submission,
# see the module docstring). So the real added cost vs the 3h v1 budget is
# roughly "4x Stage B v1's full-res work, once", not "8x". 6h is still a
# generous margin on top of that, not a tight estimate -- there is no
# per-slide timing data from this cluster to derive a tighter number from.
# If you want a real number before trusting future runs, check this job's
# own `seff <jobid>` / `sacct -j <jobid> --format=Elapsed` after it finishes.
#
# NOTE on --mem=64G: unchanged from Stage B v1 -- rgb2hed still upcasts its
# (downsampled) input to float64 internally per slide; peak memory is
# per-slide, not cumulative across the batch, so more slides doesn't need
# more memory, just more wall time.

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
STAGEA_INVENTORY_JSON="$SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json"
CONVERTED_PNG_DIR="$SCRATCH/data/timepoint_x5_full"
REFERENCE_THRESHOLD_JSON="$SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json"
PRIOR_STAGEB_JSON="$SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity/stageB_stain_homogeneity.json"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageB_v2_fullres"

echo "========================================================"
echo "  Timepoint cohort — Stage B v2: within-cohort stain homogeneity gate (FULL RES)"
echo "  Job ID                    : ${SLURM_JOB_ID:-local}"
echo "  Stage A v2 inventory JSON : $STAGEA_INVENTORY_JSON"
echo "  Converted PNG dir (~29)   : $CONVERTED_PNG_DIR"
echo "  Reference threshold JSON  : $REFERENCE_THRESHOLD_JSON"
echo "  Prior (v1) Stage B JSON   : $PRIOR_STAGEB_JSON"
echo "  Output dir                : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "Stage A v2 inventory JSON : "; ls -lh "$STAGEA_INVENTORY_JSON" 2>/dev/null || echo "NOT FOUND — run jobs/run_timepoint_cohort_inventory_v2.sh first"
echo -n "Converted PNG dir         : "; ls -d "$CONVERTED_PNG_DIR" 2>/dev/null || echo "NOT FOUND"
echo -n "Reference threshold JSON  : "; ls -lh "$REFERENCE_THRESHOLD_JSON" 2>/dev/null || echo "NOT FOUND — run jobs/run_stage2_reference_threshold.sh first"
echo -n "Prior (v1) Stage B JSON   : "; ls -lh "$PRIOR_STAGEB_JSON" 2>/dev/null || echo "NOT FOUND — run jobs/run_timepoint_stain_homogeneity.sh first (needed for the v1 comparison)"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_stain_homogeneity_v2 \
    --stageA-inventory-json    "$STAGEA_INVENTORY_JSON" \
    --converted-png-dir        "$CONVERTED_PNG_DIR" \
    --reference-threshold-json "$REFERENCE_THRESHOLD_JSON" \
    --prior-stageB-json        "$PRIOR_STAGEB_JSON" \
    --output-dir                "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE B v2 STAIN HOMOGENEITY GATE (FULL RES) COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageB_v2_fullres.json"
echo "  $OUTPUT_DIR/stageB_v2_fullres.md"
echo ""
echo "*** STOP HERE regardless of verdict. ***"
echo "Report stageB_v2_fullres.md INCLUDING the comparison-to-v1 section"
echo "(verdict change, h_intensity_mean/median_masked, 4Wvs12W pairwise) —"
echo "do not report a bare new PASS/FAIL. No Stage D / projection work and no"
echo "compartment-stratified stain comparison in this pass."
echo ""
