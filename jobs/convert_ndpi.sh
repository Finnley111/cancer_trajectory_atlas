#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_convert
#SBATCH --output=logs/atlas_convert-%j.out

# NDPI → left-half PNG conversion. No GPU needed — pure image I/O.

mkdir -p logs
mkdir -p $SCRATCH/data/MCF7_x5_cropped

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

echo "NDPI files:"; ls $SCRATCH/data/MCF7_x5/*.ndpi | wc -l

cd ~

python -m cancer_trajectory_atlas.run_all \
    --convert \
    --ndpi-dir   $SCRATCH/data/MCF7_x5 \
    --png-dir    $SCRATCH/data/MCF7_x5_cropped \
    --ndpi-level 0 \
    --ndpi-scale 1.0

echo ""
echo "Done. PNG count:"
ls $SCRATCH/data/MCF7_x5_cropped/*.png | wc -l
echo "Output size:"
du -sh $SCRATCH/data/MCF7_x5_cropped 2>/dev/null
