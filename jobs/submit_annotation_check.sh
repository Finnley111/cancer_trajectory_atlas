#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=annot_check
#SBATCH --output=logs/annot_check-%j.out

# Converts new-format GeoJSON annotations to ratio-coord JSON, then draws
# polygon outlines over slide thumbnails for visual QC.
#
# Prerequisites:
#   1. New .geojson files are in ~/cancer_trajectory_atlas/data/annotations/
#      (rsync them from local: rsync -av data/annotations/ narval:~/cancer_trajectory_atlas/data/annotations/)
#   2. Cropped PNGs are at $SCRATCH/data/MCF7_x5_cropped/
#
# Output:
#   Converted annotations: ~/cancer_trajectory_atlas/data/annotations_ratio/
#   Overlay thumbnails:    $SCRATCH/annotation_check/
#
# Submit:
#   sbatch jobs/submit_annotation_check.sh

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

cd ~

echo "========================================"
echo "Step 1: Convert annotations to ratio coords"
echo "========================================"

python -m cancer_trajectory_atlas.converters.geojson_to_ratio_json \
    --input-dir   ~/cancer_trajectory_atlas/data/annotations \
    --output-dir  ~/cancer_trajectory_atlas/data/annotations_ratio \
    --dims-json   $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json

echo ""
echo "========================================"
echo "Step 2: Draw annotation overlays"
echo "========================================"

python ~/cancer_trajectory_atlas/jobs/check_annotations.py \
    --png-dir    $SCRATCH/data/MCF7_x5_cropped \
    --ann-dir    ~/cancer_trajectory_atlas/data/annotations_ratio \
    --dims-json  $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json \
    --output-dir $SCRATCH/annotation_check

echo ""
echo "Done. Thumbnails saved to: $SCRATCH/annotation_check/"
echo "Copy back locally with:"
echo "  rsync -av narval:$SCRATCH/annotation_check/ annotation_check/"
