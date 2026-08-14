#!/bin/bash
# PASS 2 of 3 — choose each slide's DPT root cluster. CPU ONLY, no GPU.
#
# ============================================================================
# NON-COMPARABILITY CONSTRAINT
# PSEUDOTIME FROM ONE SLIDE IS NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO ANY
# PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT. Cluster IDs are
# per-slide labels: the root cluster chosen here for one slide has no
# relationship to the root cluster chosen for another. Says nothing about
# differences BETWEEN slides or timepoints. Does not address the
# 100%-extrapolation projection finding or the staining differences vs 2M.
# ============================================================================
#
# THE RULE, fixed in advance and applied identically to every slide:
#   root = the Leiden cluster with the LOWEST MEDIAN nuclear_density, computed by
#   validation/morphological_features.compute_nuclear_density_quick — the same
#   function the atlas pipeline uses to rank its own DPT roots. Ties break to the
#   lowest cluster ID (arbitrary, but reproducible and independent of every
#   downstream quantity).
#
#   This replaces run_individual.py's default of the LOWEST-NUMBERED Leiden
#   cluster, which is an arbitrary label and therefore an arbitrary origin.
#
#   EVERY cluster's median density is written to root_choices.json, not just the
#   winner, so the choice is auditable rather than asserted.
#
# Patches are cropped from the PNG at the (x, y) already in Pass 1's results.csv,
# so no determinism assumption about re-running patch extraction is needed. This
# is exact only because Pass 1 ran with --stain-method none; the module REFUSES
# to run otherwise rather than measuring the wrong pixels.
#
# READS (READ-ONLY): Pass 1 tree, $SCRATCH/data/timepoint_x5_full
# WRITES (NEW ONLY): $SCRATCH/results/individual_timepoint/root_choices.json
#
# Cost: decoding ~29 whole-slide PNGs and Otsu-segmenting a bounded sample of
# patches per cluster. Minutes to a couple of hours. No GPU, no model.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_individual_timepoint_pass2.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=indiv_tp_p2
#SBATCH --output=logs/indiv_tp_p2-%j.out

set -euo pipefail
mkdir -p logs
JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$JOBS_DIR/_individual_timepoint_common.sh" ] || JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$JOBS_DIR/_individual_timepoint_common.sh"

banner
echo "  PASS 2 of 3 — root cluster choice (CPU only)"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Output : $ROOT_CHOICES   (NEW)"
echo "============================================================================"

if [ ! -d "$PASS1_DIR" ]; then
    echo "ERROR: Pass 1 output not found at $PASS1_DIR. Run pass 1 first."
    exit 1
fi
if [ -e "$ROOT_CHOICES" ]; then
    echo "ERROR: $ROOT_CHOICES already exists. Refusing to overwrite an existing"
    echo "       root-choice record; move it aside first."
    exit 1
fi

load_env
cd ~

python -m cancer_trajectory_atlas.analysis.individual_root_choice \
    --pass1-dir              "$PASS1_DIR" \
    --png-dir                "$PNG_DIR" \
    --output                 "$ROOT_CHOICES" \
    --patch-size             "$PATCH_SIZE" \
    --max-patches-per-cluster 2000 \
    --stain-method-was       "$STAIN_METHOD"

echo ""
echo "  --- root choices ---"
python - "$ROOT_CHOICES" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  resolved: {d['n_slides_resolved']}   failed: {d['n_slides_failed']}")
for name, r in sorted(d["choices"].items()):
    tie = "  [TIE -> lowest ID]" if r["tie_broken_by_lowest_cluster_id"] else ""
    print(f"    {name:34s} root={r['root_cluster']:<3d} of {r['n_clusters']:<3d} "
          f"median_nd={r['root_median_nuclear_density']:.5g}{tie}")
for name, err in d["failures"].items():
    print(f"    {name:34s} FAILED: {err}")
print("\n  Cluster IDs are PER-SLIDE labels — root cluster 2 on one slide has no")
print("  relationship to root cluster 2 on another.")
PY

echo ""
echo "============================================================================"
echo "  PASS 2 COMPLETE — $ROOT_CHOICES"
echo "  Next: pass 3 re-runs each slide with its chosen root."
banner
