#!/bin/bash
# TASK 4 — per-section re-run with the Task 1 feature fixes.  DO NOT SUBMIT
#          WITHOUT REVIEW.
#
# Reproduces the baseline per-section pipeline EXACTLY, changing nothing but the
# feature-extraction code, and writes to per_section_v2 so the baseline at
# $SCRATCH/results/per_section/ is never touched.
#
# ── Configuration: identical to jobs/run_per_section.sh ──────────────────────
#   --stain-method none   --batch-method none   --model phikon
#   --patch-size 112      --stride 96           --clustering-method leiden
#   --leiden-resolution 0.5   --n-roots 20      --n-permutations 1000
#   --cap-strategy median     --features-cache-dir $SCRATCH/data/features_cache
#   Same 8 slides per section.
#
# ── What changed underneath (Task 1) ─────────────────────────────────────────
#   1a  Extraction failures return nan, not 0.0; counted, indexed, and written to
#       <out>/feature_failures.json. Root selection now EXCLUDES non-finite
#       densities explicitly (analysis/diffusion.py) instead of relying on
#       numpy's incidental NaN-last argsort ordering.
#   1b  compute_texture_entropy averages over 4 angles x 3 distances (was 1 angle
#       x 3 distances). Entropy is computed per (distance, angle) pair and the 12
#       scalars averaged — NOT one entropy over a pooled GLCM, which would be
#       biased upward by Jensen's inequality.
#   1c  h_intensity is now masked to segmented nuclei; the legacy whole-patch
#       value is retained as the separate feature h_intensity_wholepatch. BOTH
#       appear in obs and in results.csv.
#   1d  packing_irregularity returns nan (not 0.0) when <3 nuclei are segmented.
#   Also: DPT root indices are now persisted to adata.uns['dpt_root_candidates'],
#   which the baseline run did not store — this is what makes the Task 5 root-set
#   comparison possible at all.
#
# ── WHY THE FEATURE PASS MUST RE-RUN DESPITE THE CACHE ───────────────────────
#   The Phikon feature cache is reused (--features-cache-dir), so no GPU and no
#   model inference. But MORPHOLOGICAL features are computed from the PATCH
#   IMAGES, not from the embeddings, so all four fixes require decoding every
#   slide and re-running segmentation. That is the dominant cost of this job.
#
# ── WALLTIME / MEMORY ────────────────────────────────────────────────────────
#   NOT RECOVERED. The actual resource use of the baseline runs would come from
#   `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,ReqMem` or from
#   logs/per_section-*.out, neither of which is available from the machine this
#   script was written on. The values below are inherited from
#   jobs/run_per_section.sh (14 h / 64 G / 8 cpus), which covered a superset of
#   this work — that script also ran 32 LOO jobs, batch mixing, overlays, patch
#   export and the cross-section comparison, none of which run here. So this is
#   an UPPER BOUND carried over from a larger job, not a measurement.
#
#   Before submitting, run this on Narval and put the real numbers here:
#       sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#             --name=atlas_per_section
#
# READS  (READ-ONLY): $SCRATCH/data/features_cache, $SCRATCH/data/MCF7_x5_cropped,
#                     data/annotations_ratio
# WRITES (NEW ONLY) : $SCRATCH/results/per_section_v2/atlas_2M-1
#                     $SCRATCH/results/per_section_v2/atlas_2M-2
#
# The two sections share no state. To run them as independent parallel jobs:
#       sbatch --export=ONLY_SECTION=2M-1 jobs/run_per_section_v2.sh
#       sbatch --export=ONLY_SECTION=2M-2 jobs/run_per_section_v2.sh
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_per_section_v2.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_per_section_v2
#SBATCH --output=logs/per_section_v2-%j.out

set -euo pipefail

mkdir -p logs

# ── Constants — identical to the baseline run ────────────────────────────────
LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
BASELINE_BASE="$SCRATCH/results/per_section"
V2_BASE="$SCRATCH/results/per_section_v2"

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
echo "  Per-section re-run v2 — Task 1 feature fixes"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  Output base: $V2_BASE"
echo "  Baseline   : $BASELINE_BASE  (MUST NOT BE WRITTEN)"
echo "============================================================"

# ── Guard: never write into the baseline ─────────────────────────────────────
case "$V2_BASE" in
    "$BASELINE_BASE"|"$BASELINE_BASE"/*)
        echo "ERROR: v2 output path is inside the baseline tree. Refusing to run."
        exit 1;;
esac

if [ ! -d "$BASELINE_BASE" ]; then
    echo "WARNING: baseline $BASELINE_BASE not found — the Task 5 comparison will"
    echo "         have nothing to compare against."
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
    exit 1
}
echo "  Phikon cache complete for all requested slides."
echo "============================================"

# ── Environment ──────────────────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$V2_BASE"
cd ~

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

    OUT_DIR="$V2_BASE/atlas_${SECTION}"
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
echo "  V2 RE-RUN COMPLETE"
echo "============================================================"
echo ""
echo "Baseline (unchanged): $BASELINE_BASE"
echo "V2 outputs          : $V2_BASE"
echo ""
echo "Next: Task 5 comparison, then re-run Tasks 2 and 3 against v2 into"
echo "      _v2-suffixed output directories."
