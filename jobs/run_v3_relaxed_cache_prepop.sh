#!/bin/bash
# PREPOP — populate the RELAXED-FILTER Phikon cache shared by Configs B and C.
#
# WHY THIS JOB EXISTS
#   Configs B and C use IDENTICAL patch extraction: same relaxed filters, same
#   ROI polygons, same 112/96, same slides. They differ only in root rule, which
#   is applied after embedding. The feature cache stores UNCAPPED features and the
#   cap is seeded by slide name, so B and C see byte-identical patch sets.
#
#   Running B and C truly in parallel from a cold cache would therefore make BOTH
#   do full GPU inference into the SAME .npy paths — a write race on every slide.
#   This job does that inference once; B and C then run in parallel, CPU-only,
#   as pure cache hits.
#
#   Precedent: jobs/run_cache_prepop.sh does the same for the PAGA variant suite.
#
# WHAT "RELAXED" MEANS
#   --relaxed-tissue-filters disables BOTH patch-level tissue filters in
#   features/patching.py: the white rejection (>70% of pixels with all RGB > 220)
#   and the HSV tissue check (>=50% of pixels with S > 15 and V < 230). ROI
#   inclusion and exclusion filtering stay ON. Background and slide-edge patches
#   therefore enter the embedding. Expect a LARGE increase in patch count.
#
# ⚠ THIS IS A DIFFERENT CACHE, ON PURPOSE
#   It writes to $SCRATCH/data/features_cache_v3relaxed, never to the production
#   cache. run_all's cache guard compares N and would correctly REJECT the
#   production cache for a relaxed run — but the guard is a backstop, not the
#   plan. _v3_common.sh asserts the target is not the production cache before
#   anything runs. Nothing is ever deleted.
#
# ⚠ THIS JOB WRITES A THROWAWAY RESULTS TREE
#   run_all has no "extract and cache only" mode, so this necessarily completes a
#   full pipeline run. Its output goes to <V3B_BASE>/_prepop_discard, which exists
#   ONLY to give the cache-population pass somewhere legal to write. Do not read
#   results from it: its root rule is the production default, so it is neither
#   Config B nor Config C. Config B is the run that reproduces it deliberately.
#
# WALLTIME / MEMORY: NOT MEASURED. The GPU request mirrors
#   jobs/run_cache_population.sh (a100, 8 cpu, 64G, 6h) which populated the cache
#   for 16 slides under PRODUCTION filters. This job has 8 slides but many more
#   patches per slide, so that figure is an informed guess, not a measurement.
#   Substitute the real numbers after the first run:
#     sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State --name=v3_prepop
#
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3_relaxed_cache_prepop.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=v3_prepop
#SBATCH --output=logs/v3_prepop-%j.out

set -euo pipefail
mkdir -p logs
# NOT $(dirname "$0"): sbatch copies the script to a spool dir, so $0 is not
# the repo. SLURM_SUBMIT_DIR is unset for an interactive run, hence the fallback.
V3_JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$V3_JOBS_DIR/_v3_common.sh" ] || V3_JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$V3_JOBS_DIR/_v3_common.sh"

OUT_DIR="$V3B_BASE/_prepop_discard"

echo "============================================================"
echo "  PREPOP — relaxed-filter feature cache for Configs B and C"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Cache   : $RELAXED_CACHE   (NEW — production cache untouched)"
echo "  Scratch output (DISCARD): $OUT_DIR"
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"
v3_assert_cache_safe  "$RELAXED_CACHE"

echo ""
echo "=== Pre-run checks ==="
v3_assert_inputs_exist "$PNG_DIR" "$ANN_DIR"

if [ -d "$RELAXED_CACHE" ]; then
    N_EXISTING=$(find "$RELAXED_CACHE" -name '*_features.npy' | wc -l)
    echo "  Relaxed cache already exists with $N_EXISTING entries."
    echo "  Existing entries are REUSED, never deleted or overwritten — run_all"
    echo "  only writes on a cache miss. Resubmitting this job is safe and resumable."
else
    echo "  Relaxed cache does not exist yet; it will be created."
fi
echo "============================================"

v3_load_env
mkdir -p "$RELAXED_CACHE" "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir                 "$PNG_DIR" \
    --annotation-dir          "$ANN_DIR" \
    --output-dir              "$OUT_DIR" \
    --stain-method            none \
    --batch-method            none \
    --model                   phikon \
    --patch-size              "$PATCH_SIZE" \
    --stride                  "$STRIDE" \
    --clustering-method       leiden \
    --leiden-resolution       "$LEIDEN_RES" \
    --n-roots                 "$N_ROOTS" \
    --n-permutations          "$N_PERMUTATIONS" \
    --features-cache-dir      "$RELAXED_CACHE" \
    --cap-strategy            median \
    --slides                  "$SLIDES_CSV" \
    --relaxed-tissue-filters

echo ""
echo "=== Relaxed cache populated ==="
ls -lh "$RELAXED_CACHE"/*.npy | head -20
python - <<'PY'
import os, glob, numpy as np
d = os.path.join(os.environ["SCRATCH"], "data", "features_cache_v3relaxed")
tot = 0
for f in sorted(glob.glob(os.path.join(d, "*_features.npy"))):
    a = np.load(f, mmap_mode="r")
    tot += a.shape[0]
    print(f"  {os.path.basename(f):45s} {a.shape[0]:7d} x {a.shape[1]}")
print(f"  TOTAL uncapped patches under relaxed filters: {tot}")
PY

echo ""
echo "============================================================"
echo "  PREPOP COMPLETE"
echo "  $OUT_DIR is a THROWAWAY — do not read results from it."
echo "  Now submit Configs B and C; they will run CPU-only on cache hits."
echo "============================================================"
