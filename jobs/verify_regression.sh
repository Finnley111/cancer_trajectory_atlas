#!/bin/bash
# TIER 1 — regression re-run of the per-section pipeline.
#
# PURPOSE
#   Re-runs the exact configuration that produced $SCRATCH/results/per_section_v2
#   into a NEW directory, so jobs/verify_compare.sh can assert bit-identity
#   against it. This is the gate that says a refactor changed no number.
#
#   Everything here is downstream of the Phikon feature cache, which is REUSED,
#   not rebuilt. That is what makes exact identity a reasonable expectation:
#   the same cached embeddings enter the same deterministic code path. Tier 2
#   (jobs/verify_conversion_smoke.sh) covers the stages upstream of the cache,
#   where exact identity is NOT a reasonable expectation.
#
# CONFIGURATION — identical to jobs/run_per_section_v2.sh, verbatim
#   --stain-method none   --batch-method none   --model phikon
#   --patch-size 112      --stride 96           --clustering-method leiden
#   --leiden-resolution 0.5   --n-roots 20      --n-permutations 1000
#   --cap-strategy median     --features-cache-dir $SCRATCH/data/features_cache
#   Same 8 slides per section.
#
#   If you change ANY flag here, this script stops being a regression test and
#   becomes a new experiment. The comparison in verify_compare.sh will then fail
#   for a reason that has nothing to do with the refactor being tested.
#
# WHY THE FEATURE PASS STILL COSTS AN HOUR PER SECTION DESPITE THE CACHE
#   The Phikon cache removes model inference. It does NOT remove morphological
#   feature extraction, which is computed from the PATCH IMAGES, so every slide
#   is still decoded and re-segmented. That is the dominant cost.
#
# WALLTIME / MEMORY — NOT MEASURED
#   No sacct record is recoverable and logs/ is empty on the machine this was
#   written on, so nothing below is a measurement.
#
#   The only basis available is jobs/run_per_section.sh's own estimate comment,
#   which puts "main pipeline x2 (~8 slides, CPU, from cache)" at 2 x 60 min.
#   That gives ~1 h per section. The 4 h request below is that estimate with a
#   4x margin. Memory is inherited from run_per_section_v2.sh's 64 G, itself an
#   upper bound carried over from a larger job, not a measurement.
#
#   AFTER THE FIRST RUN, replace these with real numbers:
#       sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#             --name=verify_regression
#
# READS  (READ-ONLY): $SCRATCH/data/features_cache
#                     $SCRATCH/data/MCF7_x5_cropped
#                     data/annotations_ratio
# WRITES (NEW ONLY) : $SCRATCH/results/verify_regression/<TAG>/atlas_2M-{1,2}
#
# The two sections share no state. To run them as independent parallel jobs:
#       sbatch --export=ALL,ONLY_SECTION=2M-1 jobs/verify_regression.sh
#       sbatch --export=ALL,ONLY_SECTION=2M-2 jobs/verify_regression.sh
#   Use --export=ALL,... not --export=ONLY_SECTION=..., or $SCRATCH is unset
#   inside the job.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/verify_regression.sh
#   sbatch --export=ALL,VERIFY_TAG=post_phase2 ~/.../jobs/verify_regression.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=verify_regression
#SBATCH --output=logs/verify_regression-%j.out

set -euo pipefail

mkdir -p logs

# ── Constants — must match run_per_section_v2.sh exactly ─────────────────────
LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"

# Trees that hold published results. NEVER written by this script.
V2_BASE="$SCRATCH/results/per_section_v2"
BASELINE_BASE="$SCRATCH/results/per_section"

# A tag keeps repeat verifications from overwriting each other. Default is a
# timestamp so two runs on the same day do not collide.
VERIFY_TAG="${VERIFY_TAG:-$(date +%Y%m%d_%H%M%S)}"
VERIFY_BASE="$SCRATCH/results/verify_regression/$VERIFY_TAG"

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
echo "  TIER 1 — regression re-run"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  Output     : $VERIFY_BASE   (NEW)"
echo "  Reference  : $V2_BASE       (READ-ONLY, compared later)"
echo "============================================================"

# ── Guard: never write into a published tree ─────────────────────────────────
# Checked rather than assumed. A typo in VERIFY_TAG that resolved to an empty
# string would otherwise put the output one level above the results root.
case "$VERIFY_BASE" in
    "$V2_BASE"|"$V2_BASE"/*|"$BASELINE_BASE"|"$BASELINE_BASE"/*)
        echo "ERROR: output path is inside a published results tree. Refusing."
        exit 1;;
esac
if [ -z "${VERIFY_TAG// }" ]; then
    echo "ERROR: VERIFY_TAG is empty. Refusing to run."
    exit 1
fi
if [ -d "$VERIFY_BASE" ]; then
    echo "ERROR: $VERIFY_BASE already exists. Pick a new VERIFY_TAG rather than"
    echo "       overwriting a previous verification run."
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

echo -n "  $V2_BASE : "
if [ -d "$V2_BASE" ]; then
    echo "ok"
else
    echo "NOT FOUND"
    echo "    WARNING: the reference tree is absent. This job will still run and"
    echo "    produce output, but verify_compare.sh will have nothing to compare"
    echo "    against. That is a broken verification, not a passing one."
fi
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
    echo "ERROR: incomplete Phikon cache. This job runs CPU-only and will NOT"
    echo "       fall back to inference. A cache miss here would silently make"
    echo "       the run non-comparable. Populate it with run_cache_population.sh."
    exit 1
}
echo "  Phikon cache complete for all requested slides."
echo "============================================"

# ── Environment — identical module set to run_per_section_v2.sh ──────────────
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$VERIFY_BASE"
cd ~

# Record what this verification was run against, so a stale comparison is
# detectable later. Nothing else in the pipeline writes a provenance file.
cat > "$VERIFY_BASE/verify_provenance.txt" <<PROV
verify_tag        = $VERIFY_TAG
slurm_job_id      = ${SLURM_JOB_ID:-local}
date_utc          = $(date -u +%Y-%m-%dT%H:%M:%SZ)
host              = $(hostname)
sections          = ${SECTIONS[*]}
reference_tree    = $V2_BASE
features_cache    = $CACHE_DIR
png_dir           = $PNG_DIR
annotation_dir    = $ANN_DIR
git_describe      = $(cd ~/cancer_trajectory_atlas 2>/dev/null && git describe --always --dirty 2>/dev/null || echo "unavailable")
PROV
echo "  Provenance: $VERIFY_BASE/verify_provenance.txt"

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

    OUT_DIR="$VERIFY_BASE/atlas_${SECTION}"
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
print('  full  n_failed =', d['morphological_features']['n_failed'])" \
        || echo "  (feature_failures.json not readable)"
done

echo ""
echo "============================================================"
echo "  TIER 1 RE-RUN COMPLETE"
echo "============================================================"
echo ""
echo "Output: $VERIFY_BASE"
echo ""
echo "NOTHING IS VERIFIED YET. This job only produced output. Run the gate:"
echo ""
echo "    sbatch --export=ALL,VERIFY_TAG=$VERIFY_TAG \\"
echo "        ~/cancer_trajectory_atlas/jobs/verify_compare.sh"
echo ""
