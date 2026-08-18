#!/bin/bash
# PHASE 2 root inspection — 4 root sets: v2 and holeyroot, both sections.
#
# Reuses diagnostics/inspect_roots_v3.py UNMODIFIED. That module reads the roots
# each run ACTUALLY used, from adata.uns['dpt_root_candidates'], rather than
# re-deriving them from nuclear_density — which would be simply wrong for the
# holeyness-rooted runs and unreliable even for the density-rooted ones.
#
# Produces, per root set:
#   roots_<label>_patches.{png,pdf}   20 native 112x112 crops, labelled 4x5 sheet
#   roots_<label>_context.{png,pdf}   1500x1500 px windows, patch outlined,
#                                     edge-clamped panels labelled
#   roots_<label>/                    individual crops at native resolution
#   roots_<label>.json                per-root measures
# Labels carry root index, slide, (x, y), nuclear_density, nucleus count, and for
# the holeyness runs the duct's hole % and duct ID. pdf + png at 300 dpi.
#
# PRESENTED NEUTRALLY. No early/late judgement and no tissue-vs-artifact call is
# printed anywhere — that is deliberate, so the images can be read without being
# led.
#
# CPU only. Cost is decoding 16 whole-slide PNGs.
#
# READS (READ-ONLY): the four run trees and the source slide PNGs.
# WRITES (NEW ONLY): $SCRATCH/results/holeyroot_experiment/root_sheets
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyroot_root_inspection.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=holeyroot_sheets
#SBATCH --output=logs/holeyroot_sheets-%j.out

set -euo pipefail
mkdir -p logs

V2_BASE="$SCRATCH/results/per_section_v2"
HR_BASE="$SCRATCH/results/per_section_holeyroot"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
OUT_DIR="$SCRATCH/results/holeyroot_experiment/root_sheets"
CONTEXT_PX=1500
PATCH_SIZE=112

LABELS=(v2_2M-1 holeyroot_2M-1 v2_2M-2 holeyroot_2M-2)
DIRS=(
    "$V2_BASE/atlas_2M-1"
    "$HR_BASE/atlas_2M-1"
    "$V2_BASE/atlas_2M-2"
    "$HR_BASE/atlas_2M-2"
)

echo "============================================================================"
echo "  PHASE 2 root inspection — v2 and holeyroot, both sections"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Context : ${CONTEXT_PX}x${CONTEXT_PX} px, 300 dpi, pdf + png"
echo "  Output  : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$V2_BASE"|"$V2_BASE"/*) echo "ERROR: output inside protected v2 tree."; exit 1;;
esac

KEEP_L=(); KEEP_D=()
for i in "${!LABELS[@]}"; do
    if [ -f "${DIRS[$i]}/adata_full.h5ad" ]; then
        KEEP_L+=("${LABELS[$i]}"); KEEP_D+=("${DIRS[$i]}")
        echo "  include ${LABELS[$i]}"
    else
        echo "  SKIP    ${LABELS[$i]} : no adata_full.h5ad at ${DIRS[$i]}"
    fi
done
[ "${#KEEP_L[@]}" -gt 0 ] || { echo "ERROR: no run has an adata_full.h5ad."; exit 1; }
[ -d "$PNG_DIR" ] || { echo "ERROR: PNG dir not found: $PNG_DIR"; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.diagnostics.inspect_roots_v3 \
    --labels         "${KEEP_L[@]}" \
    --run-dirs       "${KEEP_D[@]}" \
    --png-dir        "$PNG_DIR" \
    --output-dir     "$OUT_DIR" \
    --patch-size     "$PATCH_SIZE" \
    --context-window "$CONTEXT_PX"

echo ""
echo "============================================================================"
echo "  ROOT SHEETS COMPLETE"
echo "============================================================================"
ls -1 "$OUT_DIR"/*.png 2>/dev/null
echo ""
echo "  Compare v2 vs holeyroot within each section, then across sections."
echo "  2M-2's v2 roots are the ones to look at first: all 20 reportedly had"
echo "  nuclear_density exactly 0.0, which earlier inspection flagged as likely"
echo "  background or segmentation failure rather than genuinely acellular tissue."
