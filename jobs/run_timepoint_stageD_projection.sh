#!/bin/bash
# Timepoint cohort: Stage D -- feature extraction + projection (GPU).
#
# DIAGNOSTIC RUN DESPITE A FAILED STAIN GATE. Stage B v2 found timepoint groups
# separating specifically on HEMATOXYLIN INTENSITY (the RGB channels are mostly
# negligible-to-small) -- consistent with either a reagent-side confound or
# genuine cellularity change with tumor age. No correction is applied and the
# gate still stands. This stage exists to feed Stage E's diagnostic question
# (does pseudotime add anything beyond that one channel?), NOT to produce a
# timepoint result. Do not describe the Stage B finding as "broad staining
# differences" -- it is specifically hematoxylin.
#
# Extracts Phikon features for the 29 usable timepoint slides into a SEPARATE
# cache (never the existing $SCRATCH/data/features_cache) and projects them onto
# the EXISTING saved manifold via AtlasProjector. Nothing is retrained, refitted,
# or written back: run_all.py, the manifold, its Harmony correction, and every
# projector artifact are read-only here.
#
# Reads (read-only):
#   $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json
#   $SCRATCH/data/timepoint_x5_full/*.png
#   $SCRATCH/results/baseline/atlas_none_harmony_median/projector/
#
# Writes:
#   $SCRATCH/data/timepoint_features_cache/*.npy            (NEW cache dir)
#   $SCRATCH/results/timepoint_cohort/stageD_projection/stageD_projection.{json,md}
#   $SCRATCH/results/timepoint_cohort/stageD_projection/per_slide/*.npy
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stageD_projection.sh
# or, to chain Stage E automatically:
#   bash ~/cancer_trajectory_atlas/jobs/submit_timepoint_stageDE.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=timepoint_stageD
#SBATCH --output=logs/timepoint_stageD-%j.out

# NOTE on --mem=128G (the pipeline's usual runs use 64G): these are FULL-WIDTH
# no-crop PNGs. At ~110000 x 45000 an RGB array is ~15 GB, versus ~5.8 GB for the
# left-cropped originals the 64G jobs were sized against.
#
# NOTE on --time=24:00:00: an HONEST GUESS, not a measurement. The 16 cropped
# slides fit in 6h at ~3.3M grid positions total; these 29 full-width slides are
# ~15.6M positions, and patch extraction (per-patch PIL/HSV on CPU) dominates,
# not the GPU. IMPORTANT: this job is RESUMABLE -- each slide's features are
# written to the cache and its projection to per_slide/ as soon as that slide
# finishes, and a cache hit skips all decode + GPU work. If it hits the walltime,
# just resubmit and it continues from the next unprocessed slide; nothing is
# lost. Only if it repeatedly fails to converge should you reach for
# --max-patches-per-slide (no paired comparison here requires a cap, so capping
# is purely a cost lever, not a correctness one).

set -euo pipefail
mkdir -p logs

STAGEA_JSON="$SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json"
PNG_DIR="$SCRATCH/data/timepoint_x5_full"
PROJECTOR_DIR="$SCRATCH/results/baseline/atlas_none_harmony_median/projector"
FEATURES_CACHE_DIR="$SCRATCH/data/timepoint_features_cache"
OUTPUT_DIR="$SCRATCH/results/timepoint_cohort/stageD_projection"

echo "========================================================"
echo "  Timepoint cohort — Stage D: extraction + projection"
echo "  Job ID           : ${SLURM_JOB_ID:-local}"
echo "  Stage A inventory : $STAGEA_JSON"
echo "  PNG dir           : $PNG_DIR"
echo "  Projector dir     : $PROJECTOR_DIR"
echo "  Feature cache     : $FEATURES_CACHE_DIR  (SEPARATE from features_cache)"
echo "  Output dir        : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks (fail fast, before any GPU work) ==="
FAIL=0
echo -n "Stage A inventory : "; ls -lh "$STAGEA_JSON" 2>/dev/null || { echo "NOT FOUND"; FAIL=1; }
echo -n "PNG dir           : "; ls -d "$PNG_DIR" 2>/dev/null || { echo "NOT FOUND"; FAIL=1; }
if [ -d "$PNG_DIR" ]; then
  echo "  .png count      : $(find "$PNG_DIR" -maxdepth 1 -iname '*.png' | wc -l) (expect 29)"
fi
echo -n "Projector dir     : "; ls -d "$PROJECTOR_DIR" 2>/dev/null || {
  echo "NOT FOUND"
  echo "  *** PROJECT_STATE.md marks baseline/atlas_none_harmony_median as 'pending'."
  echo "  *** Confirm that run completed and wrote its projector/ subdirectory."
  echo "  *** There is deliberately NO fallback to another manifold: projecting onto"
  echo "  *** a different run would silently change every pseudotime value."
  FAIL=1
}
if [ -d "$PROJECTOR_DIR" ]; then
  for f in scaler.pkl pca.pkl knn_pseudotime.pkl; do
    if [ ! -f "$PROJECTOR_DIR/$f" ]; then echo "  MISSING: $f"; FAIL=1; fi
  done
  echo -n "  adata_train.h5ad: "
  ls -lh "$PROJECTOR_DIR/adata_train.h5ad" 2>/dev/null || \
    echo "absent — provenance check will report 'unknown' (KNN is still already-fitted)"
fi
if [ "$FAIL" -ne 0 ]; then
  echo "*** HARD FAIL: required inputs missing (see above). Not starting."
  exit 1
fi
echo "======================"
echo ""

mkdir -p "$FEATURES_CACHE_DIR" "$OUTPUT_DIR"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_projection \
    --stageA-inventory-json "$STAGEA_JSON" \
    --png-dir               "$PNG_DIR" \
    --projector-dir         "$PROJECTOR_DIR" \
    --features-cache-dir    "$FEATURES_CACHE_DIR" \
    --output-dir            "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE D COMPLETE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stageD_projection.json"
echo "  $OUTPUT_DIR/stageD_projection.md"
echo "  $OUTPUT_DIR/per_slide/     (per-slide projected pseudotime + NN distances)"
echo "  $FEATURES_CACHE_DIR/       (Phikon cache; makes any re-run cheap)"
echo ""
echo "Review BEFORE trusting Stage E: the projector-provenance block (which space"
echo "the KNN was fitted in) and the PROJECTION VALIDITY section (how many slides"
echo "are substantially extrapolated, and in which timepoint groups)."
echo ""
