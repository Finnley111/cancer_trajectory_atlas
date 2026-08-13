#!/bin/bash
# Per-slide pseudotime via run_individual.py. Qualitative only — see that
# module's header: no patch cap, no feature cache, per-slide PCA basis, and
# single-root DPT, so these pseudotimes are NOT comparable with atlas runs.
#
# ⚠ STALE ANNOTATION PATH — --annotation-dir below still points at
#   data/annotations, which held ratio-coordinate JSON when this script was
#   written (2026-05-13) but holds QuPath GeoJSON since commit f050e4a. Running
#   it as-is feeds absolute pixel coords to a loader that assumes ratios, which
#   places every ROI off-canvas and yields few or no in-ROI patches.
#
#   To actually run this, change --annotation-dir to:
#       ~/cancer_trajectory_atlas/data/annotations_ratio
#   Left unchanged here because that alters what the script does; see
#   NOTES.md -> Annotation directory -> History.
#
#SBATCH --account=def-lmarti46
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_individual
#SBATCH --output=logs/atlas_individual-%j.out

mkdir -p logs
mkdir -p $SCRATCH/results/individual_pseudotime_runs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "PNG dir:"; ls $SCRATCH/data/MCF7_x5_cropped/ | head
echo "Dimensions sidecar:"; ls -lh $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json

cd ~

python -m cancer_trajectory_atlas.run_individual \
    --png-dir        $SCRATCH/data/MCF7_x5_cropped \
    --annotation-dir ~/cancer_trajectory_atlas/data/annotations \
    --output-dir     $SCRATCH/results/individual_pseudotime_runs \
    --ndpi-scale     1.0

echo ""
echo "=== Post-processing: overlays and patch exports (per slide) ==="

INDIV_ROOT=$SCRATCH/results/individual_pseudotime_runs
PNG_DIR=$SCRATCH/data/MCF7_x5_cropped

for SLIDE_DIR in "$INDIV_ROOT"/*/; do
    SLIDE_CSV="$SLIDE_DIR/results.csv"
    [[ -f "$SLIDE_CSV" ]] || continue
    SLIDE_NAME=$(basename "$SLIDE_DIR")
    echo "  Processing: $SLIDE_NAME"

    python -m cancer_trajectory_atlas.visualize.interactive_overlay \
        --results-csv  "$SLIDE_CSV" \
        --png-dir      "$PNG_DIR" \
        --output-dir   "$SLIDE_DIR/overlays" \
        --patch-size   112

    python -m cancer_trajectory_atlas.visualize.export_patches \
        --results-csv  "$SLIDE_CSV" \
        --png-dir      "$PNG_DIR" \
        --output-dir   "$SLIDE_DIR/patch_export" \
        --patch-size   112 \
        --n-per-bin    50
done

echo ""
echo "Done. Output size:"
du -sh $SCRATCH/results/individual_pseudotime_runs 2>/dev/null
