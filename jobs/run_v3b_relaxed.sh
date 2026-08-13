#!/bin/bash
# CONFIG B — PRODUCTION ROOTS, RELAXED TISSUE FILTERS.  Section 2M-1 only.
#
# Identical to jobs/run_per_section_v2.sh except that the two patch-level tissue
# filters in features/patching.py are DISABLED:
#     - white rejection  (>70% of pixels with all RGB > 220)
#     - HSV tissue check (>=50% of pixels with S > 15 and V < 230)
# ROI-polygon inclusion and exclusion filtering stay ON. Root rule is unchanged:
# the 20 patches with the lowest MEASURED nuclear density.
#
# ⚠ RISK THIS CONFIG DELIBERATELY EXPOSES
#   With the tissue filters off, background and slide-edge patches enter the
#   embedding. compute_nuclear_density_quick returns ~0 for background, and the
#   production root rule takes the LOWEST-density patches — so BACKGROUND IS A
#   STRONG CANDIDATE TO BECOME A ROOT HERE. That is not a bug in this script; it
#   is the thing the config is built to reveal. Read the root inspection sheets
#   (jobs/run_v3_root_inspection.sh) before interpreting anything else, and note
#   what the roots actually are in the report.
#
# ⚠ NOT COMPARABLE TO v2 ON ABSOLUTE VALUES
#   A different patch set means a different PCA basis and therefore a different
#   value for every downstream number. v2 comparison is on STRUCTURE only, and
#   the comparison module reports the shared-patch subset size rather than
#   pretending an element-wise comparison is defined.
#
# DEPENDENCY. Requires the relaxed cache from run_v3_relaxed_cache_prepop.sh.
#   Submit as:
#     PREPOP=$(sbatch --parsable jobs/run_v3_relaxed_cache_prepop.sh)
#     sbatch --dependency=afterok:$PREPOP jobs/run_v3b_relaxed.sh
#     sbatch --dependency=afterok:$PREPOP jobs/run_v3c_both.sh
#   B and C then run in parallel, CPU-only, as pure cache hits.
#
# READS (READ-ONLY): $SCRATCH/data/features_cache_v3relaxed, $PNG_DIR, $ANN_DIR
#   The PRODUCTION cache is never opened by this job.
# WRITES (NEW ONLY): $SCRATCH/results/per_section_v3b_relaxed/atlas_2M-1
#
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3b_relaxed.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=v3b_relaxed
#SBATCH --output=logs/v3b_relaxed-%j.out

set -euo pipefail
mkdir -p logs
# shellcheck disable=SC1091
source "$(dirname "$0")/_v3_common.sh"

OUT_DIR="$V3B_BASE/atlas_${SECTION}"

echo "============================================================"
echo "  CONFIG B — production roots, RELAXED tissue filters"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Output  : $OUT_DIR   (NEW)"
echo "  Cache   : $RELAXED_CACHE   (relaxed; production cache NOT used)"
echo ""
echo "  NOTE: walltime 12h and mem 96G are RAISED from v2's 8h/64G because the"
echo "        relaxed patch set is substantially larger. Both are guesses, not"
echo "        measurements — see the note in _v3_common.sh."
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"
v3_assert_cache_safe  "$RELAXED_CACHE"

echo ""
echo "=== Pre-run checks ==="
v3_assert_inputs_exist "$PNG_DIR" "$ANN_DIR" "$RELAXED_CACHE"

MISSING=0
for SLIDE in "${SLIDES_2M_1[@]}"; do
    if [ ! -f "$RELAXED_CACHE/${SLIDE}_features.npy" ]; then
        echo "  ERROR: relaxed cache miss for $SLIDE"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "ERROR: the relaxed cache is incomplete. This job is CPU-only and has no"
    echo "       GPU allocated, so it cannot generate the missing entries. Run"
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
    --features-cache-dir   "$RELAXED_CACHE" \
    --cap-strategy         median \
    --slides               "$SLIDES_CSV" \
    --relaxed-tissue-filters

echo ""
echo "  --- WHAT ARE THE ROOTS? (the question this config exists to answer) ---"
python - "$OUT_DIR" <<'PY'
import sys, numpy as np, pandas as pd
try:
    import anndata as ad
except ImportError:
    sys.exit("  anndata unavailable")
d = sys.argv[1]
a = ad.read_h5ad(f"{d}/adata_full.h5ad", backed="r")
r = [int(i) for i in np.asarray(a.uns["dpt_root_candidates"]).ravel()]
df = pd.read_csv(f"{d}/results.csv")
sub = df.iloc[r]
print("  root source :", a.uns.get("dpt_root_source", "nuclear_density"))
print("  n roots     :", len(r))
for c in ("nuclear_density", "nucleus_count"):
    if c in sub.columns:
        v = sub[c].values.astype(float); v = v[np.isfinite(v)]
        if v.size:
            print(f"  {c:16s}: median {np.median(v):.4g}  range [{v.min():.4g}, {v.max():.4g}]")
print("  slides      :", sorted(set(sub['slide_name'])))
print("  LOOK AT THE CONTACT SHEETS before interpreting anything else.")
PY

echo ""
echo "============================================================"
echo "  CONFIG B COMPLETE — $OUT_DIR"
echo "  Absolute values are NOT comparable to v2; structure only."
echo "============================================================"
