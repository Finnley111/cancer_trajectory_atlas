#!/bin/bash
# Root-patch contact sheets for v2 and all three v3 configs.
#
# WHY
#   Each config anchors pseudotime somewhere. This shows WHERE, as images, so the
#   anchor can be judged by eye rather than by a density number that cannot
#   distinguish acellular tissue from a segmentation failure. For Config B in
#   particular the roots may well be background — see the risk note in
#   jobs/run_v3b_relaxed.sh — and only the images settle that.
#
# WHAT IT READS
#   adata.uns['dpt_root_candidates'] — the roots each run ACTUALLY used. It never
#   re-derives them from nuclear_density, which would be wrong for the holeyness
#   configs and unreliable even for the density ones. Runs predating root
#   persistence will error rather than guess.
#
# OUTPUT per run: native 112px sheet, 1500x1500px context sheet (patch outlined,
#   edge-clamped panels labelled), individual crops, and a JSON of the measures.
#   pdf + png at 300 dpi. Presented NEUTRALLY — no early/late or tissue/artifact
#   judgement is printed anywhere.
#
# CPU only, no GPU, no re-extraction. Cost is decoding 8 whole-slide PNGs.
#
# Run AFTER all three configs finish. Any config not yet present can be dropped
# from LABELS/RUN_DIRS; the script does not require all four.
#
# READS (READ-ONLY): each run's adata_full.h5ad, results.csv, holeyness_roots.json
#                    and the source slide PNGs.
# WRITES (NEW ONLY): $SCRATCH/results/v3_root_experiment/root_sheets
#
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3_root_inspection.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=v3_root_sheets
#SBATCH --output=logs/v3_root_sheets-%j.out

set -euo pipefail
mkdir -p logs
# NOT $(dirname "$0"): sbatch copies the script to a spool dir, so $0 is not
# the repo. SLURM_SUBMIT_DIR is unset for an interactive run, hence the fallback.
V3_JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$V3_JOBS_DIR/_v3_common.sh" ] || V3_JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$V3_JOBS_DIR/_v3_common.sh"

OUT_DIR="$V3_COMPARE/root_sheets"
CONTEXT_PX=1500

LABELS=(v2 v3a v3b v3c)
RUN_DIRS=(
    "$V2_BASE/atlas_${SECTION}"
    "$V3A_BASE/atlas_${SECTION}"
    "$V3B_BASE/atlas_${SECTION}"
    "$V3C_BASE/atlas_${SECTION}"
)

echo "============================================================"
echo "  v3 root inspection — v2 + three configs"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Context : ${CONTEXT_PX}x${CONTEXT_PX} px"
echo "  Output  : $OUT_DIR   (NEW)"
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"

# Drop any run that has not been produced yet, rather than failing the whole job.
KEEP_L=(); KEEP_D=()
for i in "${!LABELS[@]}"; do
    if [ -f "${RUN_DIRS[$i]}/adata_full.h5ad" ]; then
        KEEP_L+=("${LABELS[$i]}"); KEEP_D+=("${RUN_DIRS[$i]}")
        echo "  include ${LABELS[$i]} : ${RUN_DIRS[$i]}"
    else
        echo "  SKIP    ${LABELS[$i]} : no adata_full.h5ad at ${RUN_DIRS[$i]}"
    fi
done
if [ "${#KEEP_L[@]}" -eq 0 ]; then
    echo "ERROR: no run has an adata_full.h5ad. Nothing to inspect."
    exit 1
fi

v3_assert_inputs_exist "$PNG_DIR"
echo "============================================"

v3_load_env
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
echo "============================================================"
echo "  ROOT SHEETS COMPLETE"
echo "============================================================"
ls -1 "$OUT_DIR"/*.png 2>/dev/null
echo ""
echo "Copy the sheets to your laptop and compare the four anchors side by side."
echo "Config B's roots are the ones to look at first."
