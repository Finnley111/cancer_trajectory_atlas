#!/bin/bash
# PASS 1 of 3 — per-slide pseudotime with the ARBITRARY default root.
#
# ============================================================================
# NON-COMPARABILITY CONSTRAINT
# PSEUDOTIME FROM ONE SLIDE IS NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO ANY
# PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT. run_individual.py
# fits a separate PCA basis per slide, no cap, no cache, no batch correction,
# single-root cluster-anchored DPT. Says nothing about differences BETWEEN
# slides or timepoints. Does not address the 100%-extrapolation projection
# finding or the staining differences vs the 2M cohort.
# ============================================================================
#
# ⚠ PASS 1 OUTPUT IS NOT THE DELIVERABLE.
#   Its DPT root is run_individual.py's default: the LOWEST-NUMBERED Leiden
#   cluster. Leiden IDs are arbitrary labels, so that origin is arbitrary. This
#   pass exists only to produce the cluster labels Pass 2 needs in order to pick
#   a principled root. Pass 3 is the run that is kept.
#
#   Kept on disk as an audit trail, so the effect of the root choice is visible.
#
# WHY THREE PASSES. Choosing a root by lowest-median-nuclear-density needs
# cluster labels, which only exist after a full GPU run. run_individual.py saves
# no adata and computes no morphological features, so nothing can be re-anchored
# post-hoc. run_individual.py is neither modified nor bypassed.
#
# WHY --stain-method none (not the reinhard default):
#   1. run_individual's stain reference is slides[0] AFTER filtering, so a
#      per-slide invocation normalises each slide to ITSELF — a silent
#      per-slide-reference confound.
#   2. Pass 2 crops RAW pixels to compute nuclear density; under a normalizer the
#      densities would not correspond to the clusters they are attributed to.
#
# WHOLE-SLIDE PATCHING IS CONFIRMED SUPPORTED. These slides have no annotations;
#   discover_slides sets annotation=None and run_one_slide takes the same
#   roi_polygons=None path run_all.py uses. Verified by reading the code, not
#   assumed.
#
# READS (READ-ONLY): $SCRATCH/data/timepoint_x5_full
# WRITES (NEW ONLY): $SCRATCH/results/individual_timepoint/pass1_arbitrary_root
#
# WALLTIME/MEMORY ARE REQUESTS, NOT MEASUREMENTS — see _individual_timepoint_common.sh.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_individual_timepoint_pass1.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=indiv_tp_p1
#SBATCH --output=logs/indiv_tp_p1-%j.out

set -euo pipefail
mkdir -p logs
JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$JOBS_DIR/_individual_timepoint_common.sh" ] || JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$JOBS_DIR/_individual_timepoint_common.sh"

banner
echo "  PASS 1 of 3 — arbitrary default root. NOT the deliverable."
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Output : $PASS1_DIR   (NEW)"
echo "============================================================================"

assert_not_clobbering "$PASS1_DIR"

echo ""
echo "=== Resolving slides ==="
[ -d "$PNG_DIR" ] || { echo "ERROR: PNG dir not found: $PNG_DIR"; exit 1; }
resolve_slides

load_env
mkdir -p "$PASS1_DIR" "$NO_ANN_DIR"
cd ~

# Guard the whole-slide assumption rather than trusting it: if anything ever
# lands in this directory, discover_slides would attach it as an ROI and the run
# would silently stop being whole-slide.
if [ -n "$(ls -A "$NO_ANN_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $NO_ANN_DIR is not empty. It must be, so patching stays"
    echo "       whole-slide. Remove its contents or point NO_ANN_DIR elsewhere."
    exit 1
fi

OK=0; FAILED=0; FAILED_LIST=()
for i in "${!SLIDES[@]}"; do
    STEM="${SLIDES[$i]}"
    echo ""
    echo "--- [$((i+1))/${#SLIDES[@]}] $STEM ---"

    if ! assert_unambiguous "$STEM"; then
        FAILED=$((FAILED+1)); FAILED_LIST+=("$STEM (ambiguous --slide filter)"); continue
    fi

    # Per-slide failure must not abort the batch.
    if python -m cancer_trajectory_atlas.run_individual \
            --slide             "$STEM" \
            --png-dir           "$PNG_DIR" \
            --annotation-dir    "$NO_ANN_DIR" \
            --output-dir        "$PASS1_DIR" \
            --stain-method      "$STAIN_METHOD" \
            --patch-size        "$PATCH_SIZE" \
            --stride            "$STRIDE" \
            --leiden-resolution "$LEIDEN_RES" \
            --ndpi-scale        1.0; then
        OK=$((OK+1))
    else
        echo "    FAILED (continuing to next slide)"
        FAILED=$((FAILED+1)); FAILED_LIST+=("$STEM")
    fi
done

echo ""
echo "============================================================================"
echo "  PASS 1 COMPLETE — $OK ok, $FAILED failed, of ${#SLIDES[@]}"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failures:"
    for f in "${FAILED_LIST[@]}"; do echo "    - $f"; done
fi
echo ""
echo "  ⚠ These roots are ARBITRARY (lowest-numbered Leiden cluster). Do not read"
echo "    direction into Pass 1. Run Pass 2 next."
banner
