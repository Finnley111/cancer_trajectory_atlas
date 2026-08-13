#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas: Harmony batch correction run
#
# Runs the full pipeline with Harmony correction and saves results to a new
# directory alongside atlas_full_macenko / atlas_full_reinhard so nothing
# from those runs is touched.
#
# Usage:
#   sbatch submit_harmony.sh macenko section_number
#   sbatch submit_harmony.sh reinhard section_number
#   sbatch submit_harmony.sh macenko slide_id        # ablation: 16-group correction
#
# Arguments:
#   $1 = stain method: macenko | reinhard | none  (default: macenko)
#   $2 = harmony batch key: section_number | slide_id | mouse_id  (default: section_number)
#
# Output dirs created under $SCRATCH/results/:
#   atlas_macenko_harmony           (default stain + default key)
#   atlas_reinhard_harmony
#   atlas_macenko_harmony_slideid   (non-default key gets a suffix)

#SBATCH --account=def-lmarti46
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_harmony
#SBATCH --output=logs/atlas_harmony-%j.out

set -euo pipefail

STAIN_METHOD=${1:-macenko}
HARMONY_KEY=${2:-section_number}

# Name the output dir after the stain method + harmony key.
# Only append the key suffix when it differs from the default so the most
# common case stays readable.
if [ "$HARMONY_KEY" = "section_number" ]; then
    OUT_NAME="atlas_${STAIN_METHOD}_harmony"
else
    OUT_NAME="atlas_${STAIN_METHOD}_harmony_${HARMONY_KEY}"
fi

PNG_DIR=$SCRATCH/data/MCF7_x5_cropped
ANN_DIR=~/cancer_trajectory_atlas/data/annotations
OUT_DIR=$SCRATCH/results/$OUT_NAME

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Atlas Harmony Run"
echo "  Stain method: $STAIN_METHOD"
echo "  Harmony key:  $HARMONY_KEY"
echo "  Output dir:   $OUT_DIR"
echo "========================================"

# Match env setup from run_all_macenko.sh / submit_qc.sh
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

# HF offline — phikon weights must already be cached on Narval from a prior run
export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Sanity-check brittle deps before doing real work
python -c "import staintools, spams; print('staintools + spams OK')" || {
    echo "ERROR: staintools or spams not importable"; exit 1
}
python -c "import harmonypy; print('harmonypy OK')" || {
    echo "ERROR: harmonypy not importable — run: pip install harmonypy"
    exit 1
}

cd ~

python -m cancer_trajectory_atlas.run_all --run    \
    --png-dir           "$PNG_DIR"                 \
    --annotation-dir    "$ANN_DIR"                 \
    --output-dir        "$OUT_DIR"                 \
    --stain-method      "$STAIN_METHOD"            \
    --harmony                                      \
    --harmony-key       "$HARMONY_KEY"             \
    --model             phikon                     \
    --patch-size        112                        \
    --stride            96                         \
    --clustering-method leiden                     \
    --leiden-resolution 0.5                        \
    --n-permutations    1000

echo ""
echo "Done. Results in $OUT_DIR"
echo ""
echo "Run QC on these results with:"
echo "  sbatch cancer_trajectory_atlas/jobs/submit_qc.sh $OUT_NAME $STAIN_METHOD"
