#!/bin/bash
# Cache pre-population gate for the runs_paga experiment suite.
#
# Populates $SCRATCH/data/features_cache/ with all 16 slides' Phikon features
# (stain-method=none, patch_size=112, stride=96 — must match every variant in
# run_paga_variant.sh exactly, since those are the only settings that
# invalidate a cached .npy). Idempotent: does NOT delete existing cache files
# first, so if the cache is already fully populated (e.g. from a prior
# run_cache_population.sh / run_full_experiments.sh run on Narval) this job
# is a fast cache-hit-only pass.
#
# This MUST complete successfully (exit 0) before any run_paga_variant.sh job
# starts — submit_paga_runs.sh enforces that via --dependency=afterok. Do not
# submit this job concurrently with any run_paga_variant.sh job: concurrent
# first-time writes to the same {slide}_features.npy file are a real race
# condition this script exists specifically to avoid.
#
# Usage (normally only called by submit_paga_runs.sh, but can run standalone):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_cache_prepop.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_paga_cache_prepop
#SBATCH --output=logs/paga_cache_prepop-%j.out

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
CACHE_DIR="$SCRATCH/data/features_cache"
PREPOP_OUT="$SCRATCH/results/runs_paga/_cache_prepop_reference"

mkdir -p "$CACHE_DIR" "$PREPOP_OUT"

ALL_SLIDES=(
    6027-4L-2M-1_x5
    6027-4L-2M-2_x5
    6027-4R-2M-1_x5
    6027-4R-2M-2_x5
    6028-4L-2M-1_x5
    6028-4L-2M-2_x5
    6028-4R-2M-1_x5
    6028-4R-2M-2_x5
    6029-4L-2M-1_x5
    6029-4L-2M-2_x5
    6029-4R-2M-1_x5
    6029-4R-2M-2_x5
    6031-4L-2M-1_x5
    6031-4L-2M-2_x5
    6031-4R-2M-1_x5
    6031-4R-2M-2_x5
)
ALL_SLIDES_CSV=$(IFS=,; echo "${ALL_SLIDES[*]}")

echo "============================================"
echo "  PAGA suite — feature cache pre-population"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Cache dir:  $CACHE_DIR"
echo "  Slides:     ${#ALL_SLIDES[@]}"
echo "============================================"

cd ~

# A full 16-slide run also gives us a harmless reference output; the actual
# point of this job is the feature cache it populates as a side effect via
# --features-cache-dir.
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$PREPOP_OUT" \
    --slides                "$ALL_SLIDES_CSV" \
    --stain-method          none \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.5 \
    --harmony \
    --harmony-key           section_number \
    --n-permutations        1000 \
    --cap-strategy          median \
    --n-roots               20 \
    --features-cache-dir    "$CACHE_DIR"

echo ""
echo "=== Cache integrity check ==="
N_CACHED=$(ls "$CACHE_DIR"/*_features.npy 2>/dev/null | wc -l)
echo "Cached feature files: $N_CACHED / ${#ALL_SLIDES[@]}"

MISSING=0
for SLIDE in "${ALL_SLIDES[@]}"; do
    if [[ ! -f "$CACHE_DIR/${SLIDE}_features.npy" ]]; then
        echo "  MISSING: ${SLIDE}_features.npy"
        MISSING=1
    fi
done

if [[ "$MISSING" -ne 0 || "$N_CACHED" -ne ${#ALL_SLIDES[@]} ]]; then
    echo ""
    echo "ERROR: feature cache is incomplete. Refusing to exit 0 — the 6"
    echo "downstream run_paga_variant.sh jobs depend on this job's success"
    echo "(--dependency=afterok) and must never start against an incomplete"
    echo "or concurrently-written cache."
    exit 1
fi

echo ""
echo "=== CACHE PRE-POPULATION COMPLETE — all ${#ALL_SLIDES[@]} slides cached ==="
echo "Cache dir: $CACHE_DIR"
