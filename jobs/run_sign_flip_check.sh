#!/bin/bash
# TASK 3 — are the two sections' axes opposed, or merely oriented oppositely?
#
# WHY:
#   Every feature directional in both sections has the OPPOSITE sign
#   (nuclear_density +0.445 vs -0.150; nc_ratio +0.328 vs -0.401;
#   packing_irregularity -0.222 vs +0.168). Root choice sets which end is
#   labelled zero, and root_sensitivity showed the roots do NOT set the ordering
#   (random 20-root sets reproduce it at |rho| 0.78-0.89) — only the orientation.
#   So the two axes may be one axis read in opposite directions.
#
# INTERPRETIVE CONSTRAINT (enforced in the output, not merely documented):
#   Negating 2M-2's pseudotime NECESSARILY flips every correlation's sign. That
#   is arithmetic, not evidence, and the module never reports sign agreement
#   alone as a finding. It reports (i) per-feature ABSOLUTE DIFFERENCE between
#   2M-1 and flipped 2M-2, so magnitude agreement is visible, and (ii) whether a
#   feature's directional / non-directional status matches across sections —
#   which no reorientation can change, making a mismatch evidence AGAINST a
#   shared axis regardless of orientation.
#
# READS (READ-ONLY): <run_dir>/adata_full.h5ad. Baseline untouched.
# WRITES (NEW directory): $SCRATCH/results/sign_flip_check/
#
# CPU only, seconds of work.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_sign_flip_check.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=sign_flip
#SBATCH --output=logs/sign_flip-%j.out

set -euo pipefail
mkdir -p logs

# Retarget without editing this file:
#   RUN_BASE=$SCRATCH/results/per_section_v2 OUT_SUFFIX=_v2 sbatch <this script>
PER_SECTION_BASE="${RUN_BASE:-$SCRATCH/results/per_section}"
RUN_A="$PER_SECTION_BASE/atlas_2M-1"
RUN_B="$PER_SECTION_BASE/atlas_2M-2"
OUTPUT_DIR="$SCRATCH/results/sign_flip_check${OUT_SUFFIX:-}"

# Refuse to write over an existing result set — see run_pseudotime_std_analysis.sh.
if [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUTPUT_DIR already exists and is not empty."
    echo "       Set OUT_SUFFIX=_v2 to write elsewhere, or FORCE=1 to overwrite."
    exit 1
fi

echo "============================================================"
echo "  TASK 3 — sign-flip check"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Reference  : 2M-1  ($RUN_A)"
echo "  Flipped    : 2M-2  ($RUN_B)"
echo "  Output dir : $OUTPUT_DIR  (NEW — baseline untouched)"
echo "============================================================"

MISSING=0
for RUN_DIR in "$RUN_A" "$RUN_B"; do
    echo -n "  $RUN_DIR/adata_full.h5ad : "
    ls -lh "$RUN_DIR/adata_full.h5ad" 2>/dev/null || { echo "NOT FOUND"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing baseline adata_full.h5ad"; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUTPUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.sign_flip_check \
    --run-dir-a  "$RUN_A" \
    --run-dir-b  "$RUN_B" \
    --label-a    "2M-1" \
    --label-b    "2M-2" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "  $OUTPUT_DIR/sign_flip_report.md   <- read this first"
echo "  $OUTPUT_DIR/sign_flip_check.json"
echo "  $OUTPUT_DIR/sign_flip_scatter.png"
echo ""
echo "Read the |diff| column, NOT the signs. Every sign was forced to match by"
echo "the flip; only the magnitudes carry information."
