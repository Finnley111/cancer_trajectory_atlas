#!/bin/bash
# CONFIG C — HOLEYNESS ROOTS *AND* RELAXED TISSUE FILTERS.  Section 2M-1 only.
#
# Both changes together. This config exists precisely to expose the INTERACTION:
# A and B each change one thing, C changes both, so comparing A vs C isolates the
# filter effect at a FIXED (holeyness) root rule, and B vs C isolates the root
# effect at a FIXED (relaxed) filter setting.
#
# WHY THE INTERACTION IS WORTH A RUN RATHER THAN AN INFERENCE
#   The two changes are not obviously independent. Under relaxed filters the
#   patch set gains background, which shifts which patches fall inside a duct
#   polygon and therefore changes the holeyness candidate pool — so the root rule
#   does not necessarily do the same thing in C that it did in A. That is exactly
#   what a two-factor design cannot be talked into predicting.
#
# ⚠ NOT COMPARABLE TO v2 ON ABSOLUTE VALUES — different patch set, different PCA
#   basis, different everything downstream. Structure only.
#
# DEPENDENCY. Shares the relaxed cache with Config B. Submit as:
#     PREPOP=$(sbatch --parsable jobs/run_v3_relaxed_cache_prepop.sh)
#     sbatch --dependency=afterok:$PREPOP jobs/run_v3b_relaxed.sh
#     sbatch --dependency=afterok:$PREPOP jobs/run_v3c_both.sh
#   B and C run in parallel; both are pure cache hits, CPU-only.
#
# READS (READ-ONLY): $SCRATCH/data/features_cache_v3relaxed, $PNG_DIR, $ANN_DIR,
#                    $SLIDE_DIMS, the holeyness measurement export
#   The PRODUCTION cache is never opened by this job.
# WRITES (NEW ONLY): $SCRATCH/results/per_section_v3c_both/atlas_2M-1
#
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3c_both.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=v3c_both
#SBATCH --output=logs/v3c_both-%j.out

set -euo pipefail
mkdir -p logs
# shellcheck disable=SC1091
source "$(dirname "$0")/_v3_common.sh"

OUT_DIR="$V3C_BASE/atlas_${SECTION}"

echo "============================================================"
echo "  CONFIG C — holeyness roots + RELAXED tissue filters"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Output  : $OUT_DIR   (NEW)"
echo "  Cache   : $RELAXED_CACHE   (shared with Config B, read-only here)"
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"
v3_assert_cache_safe  "$RELAXED_CACHE"

echo ""
echo "=== Pre-run checks ==="
v3_assert_inputs_exist "$PNG_DIR" "$ANN_DIR" "$SLIDE_DIMS" "$HOLEYNESS_EXPORT" "$RELAXED_CACHE"

MISSING=0
for SLIDE in "${SLIDES_2M_1[@]}"; do
    if [ ! -f "$RELAXED_CACHE/${SLIDE}_features.npy" ]; then
        echo "  ERROR: relaxed cache miss for $SLIDE"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "ERROR: the relaxed cache is incomplete. This job is CPU-only and cannot"
    echo "       generate the missing entries. Run"
    echo "       jobs/run_v3_relaxed_cache_prepop.sh first."
    exit 1
fi
echo "  Relaxed cache complete for all ${#SLIDES_2M_1[@]} slides."
echo "============================================"

v3_load_env
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$OUT_DIR" \
    --stain-method          none \
    --batch-method          none \
    --model                 phikon \
    --patch-size            "$PATCH_SIZE" \
    --stride                "$STRIDE" \
    --clustering-method     leiden \
    --leiden-resolution     "$LEIDEN_RES" \
    --n-roots               "$N_ROOTS" \
    --n-permutations        "$N_PERMUTATIONS" \
    --features-cache-dir    "$RELAXED_CACHE" \
    --cap-strategy          median \
    --slides                "$SLIDES_CSV" \
    --relaxed-tissue-filters \
    --root-source           holeyness \
    --holeyness-export      "$HOLEYNESS_EXPORT" \
    --holeyness-slide-dims  "$SLIDE_DIMS" \
    --holeyness-percentile  10 \
    --holeyness-min-patches 1

echo ""
echo "  --- holeyness root selection under RELAXED filters ---"
echo "  Compare these counts against Config A's: the relaxed patch set changes"
echo "  which patches fall inside a duct, hence the candidate pool itself."
cat "$OUT_DIR/holeyness_roots.json" 2>/dev/null | python -c "
import json,sys
d=json.load(sys.stdin); c=d['counts']
print('  patches in no duct   :', c['n_patches_no_duct'], '/', c['n_patches_total'],
      f\"({100*c['frac_patches_no_duct']:.1f}%)\")
print('  ducts with 0 patches :', c['n_ducts_with_zero_patches'], '/', c['n_ducts_in_table'])
print('  candidate pool       :', c['n_ducts_in_candidate_pool'], 'ducts')
print('  hole% threshold      :', round(c['hole_pct_threshold'],4))
print('  degenerate pool      :', c['pool_is_degenerate_all_at_threshold'])
" || echo "  (holeyness_roots.json not readable)"

echo ""
echo "============================================================"
echo "  CONFIG C COMPLETE — $OUT_DIR"
echo "  A vs C isolates the filter effect at a fixed root rule."
echo "============================================================"
