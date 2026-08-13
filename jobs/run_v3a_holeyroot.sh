#!/bin/bash
# CONFIG A — HOLEYNESS ROOTS, PRODUCTION FILTERS.  Section 2M-1 only.
#
# Identical to jobs/run_per_section_v2.sh in EVERY respect except root selection:
# same production Phikon cache, --stain-method none, no batch correction,
# --cap-strategy median, --n-roots 20, Leiden k=15 cosine, diffusion k=30
# euclidean / 10 comps, same per-root inf-clamping, median aggregation and
# min-max normalisation.
#
# WHAT CHANGES
#   Roots become the 20 patches drawn from the LOWEST-holeyness ducts, where
#   holeyness is Miranda's expert-annotated per-duct `holes_carnoys: hole %`.
#   That measure comes from HAND ANNOTATION, not from the pipeline's pixels, so
#   it REMOVES the circularity of anchoring on nuclear_density — which is
#   simultaneously the root selector, one of the six validation features, and the
#   covariate partialled out in the cellularity confound analysis. Under this
#   anchor all six morphological features become independent validators.
#   Direction: expert judgment is that duct diameter increases with progression
#   and holes increase with diameter, so LOW holeyness = EARLY.
#
# WHAT TO EXPECT — STATE THIS BEFORE READING THE RESULT
#   Uniformly random 20-root sets already reproduce the v2 pseudotime at
#   |rho| 0.78-0.89. The manifold fixes the ORDERING; roots fix only which end is
#   zero. So this run is EXPECTED to change the axis ORIENTATION and the root set
#   and NOT the ordering. A rho in 0.78-0.89 is the PREDICTION, not a finding.
#   |rho| < 0.7 would CONTRADICT the random-root result and must be explained
#   before anything downstream is trusted.
#
# COST. Cheapest of the three configs: the production feature cache is reused, so
# no GPU and no Phikon inference. The morphological feature pass still decodes
# every slide and re-runs segmentation, which dominates the runtime.
#
# READS (READ-ONLY): $SCRATCH/data/features_cache   <- never written; job aborts
#                    if any entry is missing rather than regenerating it
#                    $SCRATCH/data/MCF7_x5_cropped, data/annotations_ratio,
#                    $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt
# WRITES (NEW ONLY): $SCRATCH/results/per_section_v3a_holeyroot/atlas_2M-1
#
# Submit in parallel with the prepop job; it shares nothing with B and C.
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3a_holeyroot.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=v3a_holeyroot
#SBATCH --output=logs/v3a_holeyroot-%j.out

set -euo pipefail
mkdir -p logs
# NOT $(dirname "$0"): sbatch copies the script to a spool dir, so $0 is not
# the repo. SLURM_SUBMIT_DIR is unset for an interactive run, hence the fallback.
V3_JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$V3_JOBS_DIR/_v3_common.sh" ] || V3_JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$V3_JOBS_DIR/_v3_common.sh"

OUT_DIR="$V3A_BASE/atlas_${SECTION}"

echo "============================================================"
echo "  CONFIG A — holeyness roots, PRODUCTION filters"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Section : $SECTION"
echo "  Output  : $OUT_DIR   (NEW)"
echo "  Cache   : $PROD_CACHE   (READ-ONLY, must be complete)"
echo "  Protected: $V2_BASE, $BASELINE_BASE"
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"

echo ""
echo "=== Pre-run checks ==="
v3_assert_inputs_exist "$PNG_DIR" "$ANN_DIR" "$SLIDE_DIMS" "$HOLEYNESS_EXPORT" "$PROD_CACHE"
v3_assert_prod_cache_complete

# Snapshot the cache so an unexpected write is detectable after the fact.
CACHE_BEFORE=$(find "$PROD_CACHE" -name '*_features.npy' -newermt '1970-01-01' \
               -printf '%f %s\n' | sort | md5sum | cut -d' ' -f1)
echo "  Production cache fingerprint (pre-run): $CACHE_BEFORE"
echo "============================================"

v3_load_env
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir              "$PNG_DIR" \
    --annotation-dir       "$ANN_DIR" \
    --output-dir           "$OUT_DIR" \
    --stain-method         none \
    --batch-method         none \
    --model                phikon \
    --patch-size           "$PATCH_SIZE" \
    --stride               "$STRIDE" \
    --clustering-method    leiden \
    --leiden-resolution    "$LEIDEN_RES" \
    --n-roots              "$N_ROOTS" \
    --n-permutations       "$N_PERMUTATIONS" \
    --features-cache-dir   "$PROD_CACHE" \
    --cap-strategy         median \
    --slides               "$SLIDES_CSV" \
    --root-source          holeyness \
    --holeyness-export     "$HOLEYNESS_EXPORT" \
    --holeyness-slide-dims "$SLIDE_DIMS" \
    --holeyness-percentile 10 \
    --holeyness-min-patches 1

CACHE_AFTER=$(find "$PROD_CACHE" -name '*_features.npy' -newermt '1970-01-01' \
              -printf '%f %s\n' | sort | md5sum | cut -d' ' -f1)
echo ""
echo "  Production cache fingerprint (post-run): $CACHE_AFTER"
if [ "$CACHE_BEFORE" != "$CACHE_AFTER" ]; then
    echo "  WARNING: the production cache CHANGED during this run. It should not"
    echo "           have — every slide was present beforehand. Investigate before"
    echo "           trusting this run or any other that shares the cache."
else
    echo "  Production cache unchanged, as expected."
fi

echo ""
echo "  --- holeyness root selection, section $SECTION ---"
cat "$OUT_DIR/holeyness_roots.json" 2>/dev/null | python -c "
import json,sys
d=json.load(sys.stdin); c=d['counts']
print('  patches in no duct   :', c['n_patches_no_duct'], '/', c['n_patches_total'],
      f\"({100*c['frac_patches_no_duct']:.1f}%)\")
print('  ducts in table       :', c['n_ducts_in_table'])
print('  ducts with 0 patches :', c['n_ducts_with_zero_patches'])
print('  candidate pool       :', c['n_ducts_in_candidate_pool'], 'ducts')
print('  hole% threshold      :', round(c['hole_pct_threshold'],4))
print('  degenerate pool      :', c['pool_is_degenerate_all_at_threshold'])
" || echo "  (holeyness_roots.json not readable)"

echo ""
echo "  --- extraction failures ---"
cat "$OUT_DIR/feature_failures.json" 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
print('  quick n_failed =', d['nuclear_density_quick']['n_failed'])
print('  full  n_failed =', d['morphological_features']['n_failed'])
" || echo "  (feature_failures.json not readable)"

echo ""
echo "============================================================"
echo "  CONFIG A COMPLETE — $OUT_DIR"
echo "  Next: run_v3_root_inspection.sh, then run_v3_compare.sh"
echo "============================================================"
