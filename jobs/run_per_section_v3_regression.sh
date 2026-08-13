#!/bin/bash
# PHASE 8 — CLEANUP REGRESSION RUN.  DO NOT SUBMIT WITHOUT REVIEW.
#
# ── WHAT THIS IS ─────────────────────────────────────────────────────────────
# This is a REGRESSION TEST, NOT A NEW SCIENTIFIC RESULT.
#
# jobs/run_per_section_v2.sh already established that the Task 1 feature fixes
# left the numbers bit-identical to the pre-fix baseline (Spearman 1.000000, max
# diff 0.000e+00, DPT root sets 20/20 unchanged). That question is settled and is
# not being re-asked.
#
# The ONLY question here is whether the 2026-08 codebase cleanup (Phases 1-7:
# docstrings, comments, archived dead code, archived superseded job scripts)
# changed pipeline behaviour. The expected answer is that nothing moved at all.
#
# Any non-identical value is a REGRESSION to be bisected to a cleanup phase, not
# a finding to interpret. Do not write it up. Do not adjust a threshold to make
# it pass. Find the phase that broke it.
#
# ── CONFIGURATION: BYTE-FOR-BYTE THE SAME AS run_per_section_v2.sh ───────────
#   --stain-method none   --batch-method none   --model phikon
#   --patch-size 112      --stride 96           --clustering-method leiden
#   --leiden-resolution 0.5   --n-roots 20      --n-permutations 1000
#   --cap-strategy median     --features-cache-dir $SCRATCH/data/features_cache
#   Same 8 slides per section, same order.
#
# Everything not listed takes the argparse default, which the cleanup did not
# touch: diffmap-neighbors=30, diffmap-comps=10, patch-sample-seed=42,
# min-roi-coverage=None, use-stardist=off, target-total=3200.
#
# ── THE CACHE MUST NOT BE REBUILT ────────────────────────────────────────────
# v2 read its Phikon features from $SCRATCH/data/features_cache and this run must
# read the SAME cache files. Cached-vs-cached comparison is exact; cached-vs-fresh
# is not, because a different final-batch size can select a different cuDNN kernel
# and shift the last ULP (see features/extractors.py). If the cache has been
# rebuilt or repopulated since the v2 run, STOP — a pseudotime delta would then be
# uninterpretable, and you would be bisecting a kernel change, not a cleanup bug.
#
# Morphological features ARE recomputed from patch images (that is the dominant
# cost here), but they are deterministic given identical pixels.
#
# ── WALLTIME / MEMORY: NOT RECOVERED ─────────────────────────────────────────
# The SBATCH values below are INHERITED from jobs/run_per_section_v2.sh. They are
# NOT a measurement of what v2 actually used. The real figures live in the v2
# job's sacct record or in logs/per_section_v2-*.out, neither of which is
# reachable from the machine this script was written on.
#
# Recover them on Narval BEFORE submitting, and paste them here:
#     sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#           --name=atlas_per_section_v2
#
# If v2's record has aged out of the sacct retention window, say so here rather
# than substituting a guess. This job does strictly less work than
# run_per_section.sh (no LOO, no batch mixing, no overlays, no patch export, no
# cross-section comparison), so 08:00:00 / 64G is an upper bound carried over from
# a larger job, not a right-sized request.
#
# ── PATHS ────────────────────────────────────────────────────────────────────
# READS  (READ-ONLY): $SCRATCH/data/features_cache
#                     $SCRATCH/data/MCF7_x5_cropped
#                     data/annotations_ratio
#                     $SCRATCH/results/per_section_v2   (comparison target)
# WRITES (NEW ONLY) : $SCRATCH/results/per_section_v3_regression/atlas_2M-1
#                     $SCRATCH/results/per_section_v3_regression/atlas_2M-2
#
# The two sections share no state. To run them as independent parallel jobs:
#       sbatch --export=ONLY_SECTION=2M-1 jobs/run_per_section_v3_regression.sh
#       sbatch --export=ONLY_SECTION=2M-2 jobs/run_per_section_v3_regression.sh
#
# ── AFTERWARDS ───────────────────────────────────────────────────────────────
# This script only produces the run. It asserts nothing. Run the comparison:
#
#     python -m cancer_trajectory_atlas.analysis.v3_regression_check \
#         --sections 2M-1 2M-2 \
#         --v2-base  $SCRATCH/results/per_section_v2 \
#         --v3-base  $SCRATCH/results/per_section_v3_regression \
#         --output-dir $SCRATCH/results/v3_regression_check
#
# It exits 0 only if every check is identical.
#
# One caveat: its check 6 (cellularity confound verdicts) is the only part that
# writes. It creates <run_dir>/cellularity_confound/ in BOTH trees, v2 included,
# and would overwrite an existing confound output there. Back that up first, or
# pass --skip-confound. Checks 0-5 are strictly read-only.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_per_section_v3_regression.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_per_section_v3_reg
#SBATCH --output=logs/per_section_v3_regression-%j.out

set -euo pipefail

mkdir -p logs

# ── Constants — identical to run_per_section_v2.sh ───────────────────────────
LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
BASELINE_BASE="$SCRATCH/results/per_section"
V2_BASE="$SCRATCH/results/per_section_v2"
V3_BASE="$SCRATCH/results/per_section_v3_regression"

SLIDES_2M_1=(
    6027-4L-2M-1_x5  6027-4R-2M-1_x5
    6028-4L-2M-1_x5  6028-4R-2M-1_x5
    6029-4L-2M-1_x5  6029-4R-2M-1_x5
    6031-4L-2M-1_x5  6031-4R-2M-1_x5
)
SLIDES_2M_2=(
    6027-4L-2M-2_x5  6027-4R-2M-2_x5
    6028-4L-2M-2_x5  6028-4R-2M-2_x5
    6029-4L-2M-2_x5  6029-4R-2M-2_x5
    6031-4L-2M-2_x5  6031-4R-2M-2_x5
)

if [ -n "${ONLY_SECTION:-}" ]; then
    SECTIONS=("$ONLY_SECTION")
else
    SECTIONS=("2M-1" "2M-2")
fi

echo "============================================================"
echo "  Per-section v3 — CLEANUP REGRESSION RUN"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  Output base: $V3_BASE"
echo "  Compare vs : $V2_BASE   (MUST NOT BE WRITTEN)"
echo "  Baseline   : $BASELINE_BASE  (MUST NOT BE WRITTEN)"
echo "============================================================"

# ── Guards: never write into the baseline or into v2 ─────────────────────────
for PROTECTED in "$BASELINE_BASE" "$V2_BASE"; do
    case "$V3_BASE" in
        "$PROTECTED"|"$PROTECTED"/*)
            echo "ERROR: v3 output path is inside a protected tree ($PROTECTED)."
            echo "       Refusing to run."
            exit 1;;
    esac
done

if [ ! -d "$V2_BASE" ]; then
    echo "ERROR: $V2_BASE not found. There is nothing to regress against, and"
    echo "       this run would be a new result rather than a regression test."
    exit 1
fi

# ── Pre-run checks ───────────────────────────────────────────────────────────
echo ""
echo "=== Pre-run checks ==="
MISSING=0
for D in "$CACHE_DIR" "$PNG_DIR" "$ANN_DIR"; do
    echo -n "  $D : "
    if [ -d "$D" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }

for SECTION in "${SECTIONS[@]}"; do
    if [ "$SECTION" = "2M-1" ]; then S=("${SLIDES_2M_1[@]}"); else S=("${SLIDES_2M_2[@]}"); fi
    for SLIDE in "${S[@]}"; do
        if [ ! -f "$CACHE_DIR/${SLIDE}_features.npy" ]; then
            echo "  ERROR: feature cache miss for $SLIDE"
            MISSING=1
        fi
    done
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: incomplete Phikon cache — this job runs CPU-only and will not"
    echo "       fall back to inference. Populate it with run_cache_population.sh."
    echo "       NOTE: if you have to repopulate, this stops being a valid"
    echo "       regression test against v2 — see the cache note in the header."
    exit 1
}
echo "  Phikon cache complete for all requested slides."

# ── Cache-staleness warning (advisory, not a hard gate) ──────────────────────
# If any cache file is NEWER than v2's outputs, the cache changed after v2 ran
# and a cached-vs-cached comparison can no longer be assumed exact.
V2_STAMP="$V2_BASE/atlas_${SECTIONS[0]}/adata_full.h5ad"
if [ -f "$V2_STAMP" ]; then
    NEWER=$(find "$CACHE_DIR" -name '*_features.npy' -newer "$V2_STAMP" 2>/dev/null | wc -l)
    if [ "$NEWER" -gt 0 ]; then
        echo ""
        echo "  WARNING: $NEWER cache file(s) are newer than the v2 output."
        echo "           The cache may have been rebuilt since v2 ran. If so, any"
        echo "           pseudotime delta is NOT attributable to the cleanup."
        echo "           Investigate before trusting a FAIL from the comparison."
    else
        echo "  Cache files all predate the v2 output — cached-vs-cached is valid."
    fi
fi
echo "============================================"

# ── Environment ──────────────────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$V3_BASE"
cd ~

# Record what code produced this run, so a regression can be bisected later.
{
    echo "commit   : $(git -C ~/cancer_trajectory_atlas rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty    : $(git -C ~/cancer_trajectory_atlas status --porcelain 2>/dev/null | wc -l) modified file(s)"
    echo "job_id   : ${SLURM_JOB_ID:-local}"
    echo "date_utc : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$V3_BASE/provenance.txt"
echo ""
echo "Provenance written to $V3_BASE/provenance.txt:"
cat "$V3_BASE/provenance.txt"

for SECTION in "${SECTIONS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Section: $SECTION"
    echo "============================================"

    if [ "$SECTION" = "2M-1" ]; then
        SECTION_SLIDES=("${SLIDES_2M_1[@]}")
    else
        SECTION_SLIDES=("${SLIDES_2M_2[@]}")
    fi
    SLIDES_CSV=$(IFS=,; echo "${SECTION_SLIDES[*]}")

    OUT_DIR="$V3_BASE/atlas_${SECTION}"
    mkdir -p "$OUT_DIR"

    python -m cancer_trajectory_atlas.run_all \
        --run \
        --png-dir             "$PNG_DIR" \
        --annotation-dir      "$ANN_DIR" \
        --output-dir          "$OUT_DIR" \
        --stain-method        none \
        --batch-method        none \
        --model               phikon \
        --patch-size          112 \
        --stride              96 \
        --clustering-method   leiden \
        --leiden-resolution   "$LEIDEN_RES" \
        --n-roots             "$N_ROOTS" \
        --n-permutations      "$N_PERMUTATIONS" \
        --features-cache-dir  "$CACHE_DIR" \
        --cap-strategy        median \
        --slides              "$SLIDES_CSV"

    echo ""
    echo "  --- extraction failures, section $SECTION ---"
    cat "$OUT_DIR/feature_failures.json" 2>/dev/null \
        | python -c "import json,sys; d=json.load(sys.stdin); \
print('  quick n_failed =', d['nuclear_density_quick']['n_failed']); \
print('  full  n_failed =', d['morphological_features']['n_failed']); \
print('  nan per feature =', d['morphological_features']['nan_counts_per_feature'])" \
        || echo "  (feature_failures.json not readable)"
done

echo ""
echo "============================================================"
echo "  V3 REGRESSION RUN COMPLETE — NOTHING HAS BEEN ASSERTED YET"
echo "============================================================"
echo ""
echo "v2 (unchanged): $V2_BASE"
echo "v3 outputs    : $V3_BASE"
echo ""
echo "Now run the comparison. It exits 0 only if every check is identical:"
echo ""
echo "  python -m cancer_trajectory_atlas.analysis.v3_regression_check \\"
echo "      --sections 2M-1 2M-2 \\"
echo "      --v2-base  $V2_BASE \\"
echo "      --v3-base  $V3_BASE \\"
echo "      --output-dir \$SCRATCH/results/v3_regression_check"
echo ""
echo "Any non-identical value is a regression to bisect against the Phase 1-7"
echo "commits, not a result to interpret."
