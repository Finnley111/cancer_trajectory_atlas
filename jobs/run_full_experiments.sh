#!/bin/bash
# Full Cancer Trajectory Atlas experiment pipeline — sequential, single GPU job.
# Runs: cache reset → cache population → Baseline A → Baseline B → 16-slide LOO.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_full_experiments.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=atlas_full
#SBATCH --output=logs/full_experiments-%j.out

set -euo pipefail

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
CACHE_DIR_NONE="$SCRATCH/data/features_cache"
CACHE_DIR_MACENKO="$SCRATCH/data/features_cache_macenko"
BASELINE_DIR="$SCRATCH/results/baseline"
LOO_DIR="$SCRATCH/results/loo"

rm -f "$CACHE_DIR_NONE"/*.npy
rm -f "$CACHE_DIR_MACENKO"/*.npy
mkdir -p "$CACHE_DIR_NONE" "$CACHE_DIR_MACENKO" "$BASELINE_DIR" "$LOO_DIR"

SLIDES=(
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

ALL_SLIDES=$(IFS=,; echo "${SLIDES[*]}")

echo "============================================"
echo "  Atlas full experiment suite"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Cache (none):  $CACHE_DIR_NONE"
echo "  Cache (mack):  $CACHE_DIR_MACENKO"
echo "  Baseline:      $BASELINE_DIR"
echo "  LOO:           $LOO_DIR"
echo "============================================"

cd ~

# ── Cache population: none stain (all 16 slides, uncapped) ───────────────────
echo ""
echo "=== Cache population — none stain (16 slides, uncapped) ==="
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$SCRATCH/results/cache_pop_none_${SLURM_JOB_ID}" \
    --slides                "$ALL_SLIDES" \
    --stain-method          none \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.5 \
    --harmony \
    --harmony-key           section_number \
    --n-permutations        1000 \
    --features-cache-dir    "$CACHE_DIR_NONE"

echo "Cache (none) contents (${#SLIDES[@]} .npy files expected):"
ls -lh "$CACHE_DIR_NONE"/*.npy

# ── Baseline A: none + Harmony (canonical reference for LOO Phase B) ─────────
echo ""
echo "=== Baseline A: none + Harmony ==="
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$BASELINE_DIR/atlas_none_harmony" \
    --slides                "$ALL_SLIDES" \
    --stain-method          none \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.5 \
    --harmony \
    --harmony-key           section_number \
    --n-permutations        1000 \
    --features-cache-dir    "$CACHE_DIR_NONE"

echo "Baseline A results.csv:"
ls -lh "$BASELINE_DIR/atlas_none_harmony/results.csv"

# ── Baseline B: Macenko + Harmony ────────────────────────────────────────────
# Uses a separate cache dir — Macenko-normalized patches produce different
# Phikon embeddings than raw patches; sharing the none cache is incorrect.
echo ""
echo "=== Baseline B: Macenko + Harmony ==="
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir               "$PNG_DIR" \
    --annotation-dir        "$ANN_DIR" \
    --output-dir            "$BASELINE_DIR/atlas_macenko_harmony" \
    --slides                "$ALL_SLIDES" \
    --stain-method          macenko \
    --model                 phikon \
    --patch-size            112 \
    --stride                96 \
    --clustering-method     leiden \
    --leiden-resolution     0.5 \
    --harmony \
    --harmony-key           section_number \
    --n-permutations        1000 \
    --features-cache-dir    "$CACHE_DIR_MACENKO"

echo "Baseline B results.csv:"
ls -lh "$BASELINE_DIR/atlas_macenko_harmony/results.csv"

# ── LOO loop: 16 held-out slides ─────────────────────────────────────────────
echo ""
echo "=== LOO loop (${#SLIDES[@]} slides) ==="

FULL_RUN_DIR="$BASELINE_DIR/atlas_none_harmony"

for HELD_OUT in "${SLIDES[@]}"; do
    echo ""
    echo "--- LOO: held-out = $HELD_OUT ---"

    LOO_OUT="$LOO_DIR/loo_${HELD_OUT}"
    mkdir -p "$LOO_OUT"

    # Comma-separated list of the other 15 slides.
    TRAINING_SLIDES=$(printf '%s\n' "${SLIDES[@]}" | grep -v "^${HELD_OUT}$" | paste -sd,)

    # Phase A — build atlas on 15 training slides (cache hits from CACHE_DIR_NONE).
    python -m cancer_trajectory_atlas.run_all \
        --run \
        --png-dir               "$PNG_DIR" \
        --annotation-dir        "$ANN_DIR" \
        --output-dir            "$LOO_OUT" \
        --slides                "$TRAINING_SLIDES" \
        --stain-method          none \
        --model                 phikon \
        --patch-size            112 \
        --stride                96 \
        --clustering-method     leiden \
        --leiden-resolution     0.5 \
        --harmony \
        --harmony-key           section_number \
        --n-permutations        200 \
        --features-cache-dir    "$CACHE_DIR_NONE"

    # Phase B — project held-out slide onto the 15-slide manifold.
    python -m cancer_trajectory_atlas.analysis.loo_project \
        --projector-dir         "$LOO_OUT/projector" \
        --held-out-slide        "$HELD_OUT" \
        --cache-dir             "$CACHE_DIR_NONE" \
        --full-run-dir          "$FULL_RUN_DIR" \
        --output-dir            "$LOO_OUT" \
        --patch-sample-seed     42

    echo "Done: $HELD_OUT"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo ""
echo "Baseline A:  $BASELINE_DIR/atlas_none_harmony"
echo "Baseline B:  $BASELINE_DIR/atlas_macenko_harmony"
echo "LOO results: $LOO_DIR/loo_*/loo_result_*.json"
echo ""
echo "Aggregate LOO results with:"
echo "  python -m cancer_trajectory_atlas.analysis.loo_summary \\"
echo "      --loo-dirs \$SCRATCH/results/loo/loo_* \\"
echo "      --output-dir \$SCRATCH/results/loo_summary"
