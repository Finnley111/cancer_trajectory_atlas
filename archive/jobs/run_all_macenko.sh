#!/bin/bash
#SBATCH --account=def-lmarti46
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_macenko
#SBATCH --output=logs/atlas_macenko-%j.out

mkdir -p logs
mkdir -p $SCRATCH/results/atlas_full_macenko

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -c "import staintools, spams; print('staintools + spams OK')" || {
    echo "ERROR: staintools or spams not importable"
    exit 1
}

echo "=== Pre-run check ==="
echo "PNG dir:"; ls $SCRATCH/data/MCF7_x5_cropped/*.png | head
echo "Dimensions sidecar:"; ls -lh $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
echo "Annotations:"; ls ~/cancer_trajectory_atlas/data/annotations/ | head
echo "===================="

cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir           $SCRATCH/data/MCF7_x5_cropped \
    --annotation-dir    ~/cancer_trajectory_atlas/data/annotations \
    --output-dir        $SCRATCH/results/atlas_full_macenko \
    --stain-method      macenko \
    --model             phikon \
    --patch-size        112 \
    --stride            96 \
    --clustering-method leiden \
    --leiden-resolution 0.5 \
    --n-permutations    1000

echo ""
echo "Done. Output size:"
du -sh $SCRATCH/results/atlas_full_macenko 2>/dev/null
