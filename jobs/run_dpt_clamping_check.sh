#!/bin/bash
# Is the diffusion graph connected, and is the DPT non-finite clamp firing?
#
# WHY (run this BEFORE the v2 re-run):
#   Task 2 found that in 2M-2, pseudotime_std is an affine function of pseudotime
#   (std ~ a*(1-pt), matching at every quantile to well under 1%). That is not
#   what per-patch uncertainty looks like. It is what this clamp produces:
#
#       finite_mask = np.isfinite(pt)
#       pt[~finite_mask] = pt[finite_mask].max()      # diffusion.py
#
#   scanpy's DPT returns inf for any patch unreachable from the root. If the
#   neighbour graph has more than one connected component, patches outside the
#   root's component are pinned to that run's MAXIMUM — silently relabelled
#   "maximally late" rather than "unmeasurable".
#
#   If that is happening:
#     - the TOP of the pseudotime axis is partly an artifact, not advanced tissue;
#     - the late tail's single-slide concentration (49.6% in 2M-1, 43.4% in 2M-2)
#       follows directly, since a disconnected component is one slide's tissue;
#     - NONE of the Task 1 feature fixes touch it, so the 8-hour v2 re-run would
#       reproduce it unchanged.
#
#   Hence: run this first. It costs minutes and can change what v2 needs to do.
#
# READS (READ-ONLY): <run_dir>/adata_full.h5ad, <run_dir>/results.csv
# WRITES (NEW directory): $SCRATCH/results/dpt_clamping_check/
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_dpt_clamping_check.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=dpt_clamp
#SBATCH --output=logs/dpt_clamp-%j.out

set -euo pipefail
mkdir -p logs

SECTIONS=("2M-1" "2M-2")
PER_SECTION_BASE="$SCRATCH/results/per_section"
RUN_DIRS=("$PER_SECTION_BASE/atlas_2M-1" "$PER_SECTION_BASE/atlas_2M-2")
OUTPUT_DIR="$SCRATCH/results/dpt_clamping_check"

echo "============================================================"
echo "  DPT clamp / graph connectivity check"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  Output dir : $OUTPUT_DIR  (NEW — baseline untouched)"
echo "============================================================"

MISSING=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo -n "  $RUN_DIR/adata_full.h5ad : "
    ls -lh "$RUN_DIR/adata_full.h5ad" 2>/dev/null || { echo "NOT FOUND"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing baseline adata_full.h5ad"; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUTPUT_DIR"
cd ~

python -m cancer_trajectory_atlas.diagnostics.dpt_clamping_check \
    --sections   "${SECTIONS[@]}" \
    --run-dirs   "${RUN_DIRS[@]}" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "  $OUTPUT_DIR/dpt_clamping_report.md   <- read this first"
echo "  $OUTPUT_DIR/dpt_clamping_check.json"
echo ""
echo "Key field: graph.n_connected_components. If > 1, the clamp is firing and the"
echo "top of the pseudotime axis is partly unreachable patches, not late tissue."
