#!/bin/bash
# Look at the DPT root patches — are they tissue, or artifact?
#
# WHY:
#   The pseudotime is rooted at the 20 lowest-nuclear_density patches
#   (analysis/diffusion.py:165). eccentricity_check Task 0 found that in 2M-2 ALL
#   twenty roots have nuclear_density EXACTLY 0.0, drawn from 21 such patches, so
#   which 20 get used is arbitrary tie-ordering.
#
#   compute_nuclear_density_quick returns 0.0 in two very different situations:
#   genuinely acellular tissue, AND any patch whose segmentation raised — the
#   handler at validation/morphological_features.py:177-178 is a bare
#   `except: pass`. The stored array cannot distinguish them. The images can.
#
# WHAT IT DECIDES:
#   root_sensitivity established that the roots do NOT set the pseudotime
#   ORDERING (uniformly random 20-root sets reproduce it at |rho| 0.78-0.89).
#   They set only which END is called "early". So if the roots are artifacts,
#   the ordering survives but the early->late DIRECTION is uninterpretable —
#   which would also explain why 2M-1 and 2M-2 point in OPPOSITE morphological
#   directions on every feature directional in both.
#
# READS (READ-ONLY): <run_dir>/results.csv  and the source slide PNGs.
# WRITES (NEW directory only): $SCRATCH/results/root_patches/
#   root_patches_<section>.{png,pdf}    the 20 roots, annotated
#   root_context_<section>.{png,pdf}    same patches with surrounding tissue
#   control_patches_<section>.{png,pdf} 20 MEDIAN-density patches, for reference
#   root_patches_<section>/             individual full-resolution crops
#   root_patch_report.{md,json}
#
# CPU only, no GPU, no model, no re-extraction. Cost is dominated by decoding a
# handful of whole-slide PNGs — minutes.
#
# IMPORTANT: --patch-size must match the value the per-section run used
# (run_all.py --patch-size, default 112). A mismatch silently crops the wrong
# window. Check the run's log if unsure.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_inspect_root_patches.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=root_patches
#SBATCH --output=logs/root_patches-%j.out

set -euo pipefail

mkdir -p logs

SECTIONS=("2M-1" "2M-2")
PER_SECTION_BASE="$SCRATCH/results/per_section"
RUN_DIRS=(
    "$PER_SECTION_BASE/atlas_2M-1"
    "$PER_SECTION_BASE/atlas_2M-2"
)
PNG_DIR="${PNG_DIR:-$SCRATCH/data/png}"
OUTPUT_DIR="$SCRATCH/results/root_patches"
N_ROOTS=20
PATCH_SIZE=112
CONTEXT=7

echo "============================================================"
echo "  DPT root patch inspection — is the origin real tissue?"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  PNG dir    : $PNG_DIR"
echo "  Patch size : $PATCH_SIZE px   Context: ${CONTEXT}x"
echo "  Output dir : $OUTPUT_DIR  (NEW — existing runs untouched)"
echo "============================================================"

echo ""
echo "=== Pre-run checks (all inputs read-only) ==="
MISSING=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo -n "  $RUN_DIR/results.csv : "
    ls -lh "$RUN_DIR/results.csv" 2>/dev/null \
        || { echo "NOT FOUND — run run_per_section.sh first"; MISSING=1; }
done

echo -n "  $PNG_DIR : "
if [ -d "$PNG_DIR" ]; then
    echo "$(ls -1 "$PNG_DIR"/*.png 2>/dev/null | wc -l) png file(s)"
else
    echo "NOT A DIRECTORY — set PNG_DIR=/path/to/pngs before sbatch"
    MISSING=1
fi

if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "ERROR: missing inputs. This diagnostic reuses an existing per-section"
    echo "       run and the original slide PNGs; it regenerates neither."
    exit 1
fi
echo "============================================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

python -c "import PIL, pandas, matplotlib; print('pillow', PIL.__version__, '/ pandas', pandas.__version__)" || {
    echo "ERROR: pillow/pandas/matplotlib not importable in ~/envs/atlas"; exit 1
}

mkdir -p "$OUTPUT_DIR"
cd ~

python -m cancer_trajectory_atlas.diagnostics.inspect_root_patches \
    --sections    "${SECTIONS[@]}" \
    --run-dirs    "${RUN_DIRS[@]}" \
    --png-dir     "$PNG_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --n-roots     "$N_ROOTS" \
    --patch-size  "$PATCH_SIZE" \
    --context     "$CONTEXT"

echo ""
echo "============================================================"
echo "  ROOT PATCH INSPECTION COMPLETE"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/root_patch_report.md   <- read this first"
echo "  $OUTPUT_DIR/root_patch_report.json"
for SECTION in "${SECTIONS[@]}"; do
    echo "  $OUTPUT_DIR/root_patches_${SECTION}.png     (the 20 roots)"
    echo "  $OUTPUT_DIR/root_context_${SECTION}.png     (roots in context)"
    echo "  $OUTPUT_DIR/control_patches_${SECTION}.png  (median-density control)"
done
echo ""
echo "Copy the three PNGs per section to your laptop and look at them side by"
echo "side. The control panel is the reference — 'is this background?' is not"
echo "answerable from the root patches alone."
echo ""
