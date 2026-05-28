#!/bin/bash
# Parameterised SLURM job for new-annotation pipeline runs.
# Do NOT submit directly — use submit_new_annotations_all.sh.
#
# Expects RUN_TYPE env var:
#   none_harmony    — all 16 slides, no stain, Harmony (section_number key)
#   macenko_harmony — all 16 slides, Macenko, Harmony (section_number key)
#   none_section1   — 2M-1 slides only, no stain, no Harmony
#   none_section2   — 2M-2 slides only, no stain, no Harmony
#   individual      — per-slide pseudotime via run_individual

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=new_ann
#SBATCH --output=logs/new_ann_%x-%j.out

set -euo pipefail

if [[ -z "${RUN_TYPE:-}" ]]; then
    echo "ERROR: RUN_TYPE is not set"
    echo "  Valid values: none_harmony macenko_harmony none_section1 none_section2 individual"
    exit 1
fi

mkdir -p logs

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PNG_DIR=$SCRATCH/data/MCF7_x5_cropped
ANN_DIR=~/cancer_trajectory_atlas/data/annotations_ratio
SUPER_DIR=$SCRATCH/results/new_annotations
OUT_DIR=$SUPER_DIR/$RUN_TYPE

mkdir -p "$OUT_DIR"

echo "========================================"
echo "New-annotation run: $RUN_TYPE"
echo "  Annotation dir: $ANN_DIR"
echo "  Output dir:     $OUT_DIR"
echo "========================================"

cd ~

_postprocess() {
    local dir=$1
    python -m cancer_trajectory_atlas.visualize.interactive_overlay \
        --results-csv  "$dir/results.csv" \
        --png-dir      "$PNG_DIR" \
        --output-dir   "$dir/overlays" \
        --patch-size   112

    python -m cancer_trajectory_atlas.visualize.export_patches \
        --results-csv  "$dir/results.csv" \
        --png-dir      "$PNG_DIR" \
        --output-dir   "$dir/patch_export" \
        --patch-size   112 \
        --n-per-bin    50
}

case "$RUN_TYPE" in

  none_harmony)
    python -c "import harmonypy; print('harmonypy OK')" || { echo "ERROR: harmonypy missing"; exit 1; }

    python -m cancer_trajectory_atlas.run_all --run \
        --png-dir           "$PNG_DIR"              \
        --annotation-dir    "$ANN_DIR"              \
        --output-dir        "$OUT_DIR"              \
        --stain-method      none                    \
        --harmony                                   \
        --harmony-key       section_number          \
        --model             phikon                  \
        --patch-size        112                     \
        --stride            96                      \
        --clustering-method leiden                  \
        --leiden-resolution 0.5                     \
        --n-permutations    1000                    \
        --min-roi-coverage  0.75

    _postprocess "$OUT_DIR"
    ;;

  macenko_harmony)
    python -c "import staintools, spams; print('staintools + spams OK')" || { echo "ERROR: staintools/spams missing"; exit 1; }
    python -c "import harmonypy; print('harmonypy OK')"                  || { echo "ERROR: harmonypy missing"; exit 1; }

    python -m cancer_trajectory_atlas.run_all --run \
        --png-dir           "$PNG_DIR"              \
        --annotation-dir    "$ANN_DIR"              \
        --output-dir        "$OUT_DIR"              \
        --stain-method      macenko                 \
        --harmony                                   \
        --harmony-key       section_number          \
        --model             phikon                  \
        --patch-size        112                     \
        --stride            96                      \
        --clustering-method leiden                  \
        --leiden-resolution 0.5                     \
        --n-permutations    1000                    \
        --min-roi-coverage  0.75

    _postprocess "$OUT_DIR"
    ;;

  none_section1 | none_section2)
    SECTION=${RUN_TYPE##*section}
    SLIDES_FILE=~/cancer_trajectory_atlas/jobs/slides_section${SECTION}.txt

    if [[ ! -f "$SLIDES_FILE" ]]; then
        echo "ERROR: slides file not found: $SLIDES_FILE"
        exit 1
    fi

    python -m cancer_trajectory_atlas.run_all --run \
        --png-dir           "$PNG_DIR"              \
        --annotation-dir    "$ANN_DIR"              \
        --output-dir        "$OUT_DIR"              \
        --stain-method      none                    \
        --slides-from-file  "$SLIDES_FILE"          \
        --model             phikon                  \
        --patch-size        112                     \
        --stride            96                      \
        --clustering-method leiden                  \
        --leiden-resolution 0.5                     \
        --n-permutations    1000                    \
        --min-roi-coverage  0.75

    _postprocess "$OUT_DIR"
    ;;

  individual)
    python -m cancer_trajectory_atlas.run_individual \
        --png-dir           "$PNG_DIR"               \
        --annotation-dir    "$ANN_DIR"               \
        --output-dir        "$OUT_DIR"               \
        --ndpi-scale        1.0                      \
        --min-roi-coverage  0.75

    for SLIDE_DIR in "$OUT_DIR"/*/; do
        SLIDE_CSV="$SLIDE_DIR/results.csv"
        [[ -f "$SLIDE_CSV" ]] || continue
        echo "  Post-processing: $(basename "$SLIDE_DIR")"
        python -m cancer_trajectory_atlas.visualize.interactive_overlay \
            --results-csv  "$SLIDE_CSV" \
            --png-dir      "$PNG_DIR"   \
            --output-dir   "$SLIDE_DIR/overlays" \
            --patch-size   112
        python -m cancer_trajectory_atlas.visualize.export_patches \
            --results-csv  "$SLIDE_CSV" \
            --png-dir      "$PNG_DIR"   \
            --output-dir   "$SLIDE_DIR/patch_export" \
            --patch-size   112 \
            --n-per-bin    50
    done
    ;;

  *)
    echo "ERROR: unknown RUN_TYPE '$RUN_TYPE'"
    echo "  Valid: none_harmony macenko_harmony none_section1 none_section2 individual"
    exit 1
    ;;
esac

echo ""
echo "Done: $RUN_TYPE"
echo "  Results in: $OUT_DIR"
du -sh "$OUT_DIR" 2>/dev/null
