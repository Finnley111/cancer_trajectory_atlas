#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas: no stain normalization + Harmony
#
# Usage:
#   sbatch submit_harmony_none.sh                  # default: section_number key
#   sbatch submit_harmony_none.sh slide_id         # ablation: 16-group correction
#   sbatch submit_harmony_none.sh mouse_id
#
# $1 = harmony batch key: section_number | slide_id | mouse_id  (default: section_number)

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_none_harmony
#SBATCH --output=logs/atlas_none_harmony-%j.out

set -euo pipefail

HARMONY_KEY=${1:-section_number}

if [ "$HARMONY_KEY" = "section_number" ]; then
    OUT_NAME="atlas_none_harmony"
else
    OUT_NAME="atlas_none_harmony_${HARMONY_KEY}"
fi

PNG_DIR=$SCRATCH/data/MCF7_x5_cropped
ANN_DIR=~/cancer_trajectory_atlas/data/annotations
OUT_DIR=$SCRATCH/results/$OUT_NAME

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Atlas Harmony Run"
echo "  Stain method: none"
echo "  Harmony key:  $HARMONY_KEY"
echo "  Output dir:   $OUT_DIR"
echo "========================================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

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
    --stain-method      none                       \
    --harmony                                      \
    --harmony-key       "$HARMONY_KEY"             \
    --model             phikon                     \
    --patch-size        112                        \
    --stride            96                         \
    --clustering-method leiden                     \
    --leiden-resolution 0.5                        \
    --n-permutations    1000

echo ""
echo "=== Post-processing: overlays and patch exports ==="

python -m cancer_trajectory_atlas.visualize.interactive_overlay \
    --results-csv  "$OUT_DIR/results.csv" \
    --png-dir      "$PNG_DIR" \
    --output-dir   "$OUT_DIR/overlays" \
    --patch-size   112

python -m cancer_trajectory_atlas.visualize.export_patches \
    --results-csv  "$OUT_DIR/results.csv" \
    --png-dir      "$PNG_DIR" \
    --output-dir   "$OUT_DIR/patch_export" \
    --patch-size   112 \
    --n-per-bin    50

echo ""
echo "Done. Results in $OUT_DIR"
echo "  Overlays   → $OUT_DIR/overlays/"
echo "  Patches    → $OUT_DIR/patch_export/"
echo ""
echo "Run QC on these results with:"
echo "  sbatch cancer_trajectory_atlas/jobs/submit_qc.sh $OUT_NAME none"
