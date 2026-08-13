#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas: no stain normalization, single section
#
# Runs the pipeline on 8 slides from one section only, avoiding the cross-section
# batch effect that produces two-island UMAPs when all 16 slides are pooled.
#
# Usage:
#   sbatch run_all_none_section.sh 1   # 2M-1 slides only
#   sbatch run_all_none_section.sh 2   # 2M-2 slides only

#SBATCH --account=def-lmarti46
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_none_section
#SBATCH --output=logs/atlas_none_section%a-%j.out

set -euo pipefail

SECTION=${1:-}
if [ "$SECTION" != "1" ] && [ "$SECTION" != "2" ]; then
    echo "ERROR: pass section number as first argument: 1 or 2"
    echo "  sbatch run_all_none_section.sh 1"
    echo "  sbatch run_all_none_section.sh 2"
    exit 1
fi

SLIDES_FILE=~/cancer_trajectory_atlas/jobs/slides_section${SECTION}.txt

OUT_DIR=$SCRATCH/results/atlas_none_section${SECTION}

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Atlas Run — Section ${SECTION} only (no stain normalization)"
echo "  Slides file: $SLIDES_FILE"
echo "  Output dir:  $OUT_DIR"
echo "========================================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Pre-run check ==="
echo "PNG dir:"; ls $SCRATCH/data/MCF7_x5_cropped/*.png | head
echo "Dimensions sidecar:"; ls -lh $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
echo "Annotations:"; ls ~/cancer_trajectory_atlas/data/annotations/ | head
echo "Slides:"; cat "$SLIDES_FILE"
echo "===================="

cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir           $SCRATCH/data/MCF7_x5_cropped \
    --annotation-dir    ~/cancer_trajectory_atlas/data/annotations \
    --output-dir        "$OUT_DIR" \
    --stain-method      none \
    --slides-from-file  "$SLIDES_FILE" \
    --model             phikon \
    --patch-size        112 \
    --stride            96 \
    --clustering-method leiden \
    --leiden-resolution 0.5 \
    --n-permutations    1000

echo ""
echo "=== Post-processing: overlays and patch exports ==="

PNG_DIR=$SCRATCH/data/MCF7_x5_cropped

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
echo "Done. Output size:"
du -sh "$OUT_DIR" 2>/dev/null
echo "  Overlays   → $OUT_DIR/overlays/"
echo "  Patches    → $OUT_DIR/patch_export/"
