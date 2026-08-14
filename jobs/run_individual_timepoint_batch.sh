#!/bin/bash
# Submit the whole per-slide timepoint run: pass 1 -> pass 2 -> pass 3, chained
# with --dependency=afterok. NOT an sbatch script — run it on the login node.
#
# ============================================================================
# NON-COMPARABILITY CONSTRAINT
# PSEUDOTIME FROM ONE SLIDE IS NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO ANY
# PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT. run_individual.py
# fits a separate PCA basis per slide: no patch cap, no feature cache, no batch
# correction, single-root cluster-anchored DPT. This answers "what does the
# internal morphological ordering look like within this one slide" and NOTHING
# about differences BETWEEN slides or timepoints. It does not address the
# 100%-extrapolation projection finding or the staining differences vs the 2M
# cohort.
# ============================================================================
#
# WHY THREE PASSES, not one. Choosing a principled DPT root per slide (lowest
# median nuclear density among that slide's Leiden clusters) requires cluster
# labels, which only exist after a full GPU run. run_individual.py saves no adata
# and computes no morphological features, so nothing can be re-anchored
# afterwards. Pass 1 produces the labels, Pass 2 picks the roots on CPU, Pass 3
# is the run that is kept. run_individual.py is never modified or bypassed.
#
# Cost: 2 GPU passes. That is the price of leaving run_individual.py untouched.
#
# WALLTIME/MEMORY IN THE PASS SCRIPTS ARE REQUESTS, NOT MEASUREMENTS. No sacct
# record for run_individual.py on this cohort was recoverable. Reference points,
# neither a measurement of this workload:
#   * jobs/run_individual_pseudotime.sh requested 6h/64G/a100 for 16 CROPPED
#     slides in one job.
#   * Stage D used 24h/128G for these same 29 FULL-WIDTH slides — and that had a
#     feature cache, which run_individual.py does not.
# After the first run, substitute real numbers:
#   sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#         --name=indiv_tp_p1,indiv_tp_p2,indiv_tp_p3
#
# Usage:
#   bash ~/cancer_trajectory_atlas/jobs/run_individual_timepoint_batch.sh

set -euo pipefail
JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$JOBS_DIR/_individual_timepoint_common.sh"

banner
echo "  Submitting pass 1 -> 2 -> 3"
echo "============================================================================"

[ -d "$PNG_DIR" ] || { echo "ERROR: PNG dir not found: $PNG_DIR"; exit 1; }

echo ""
echo "=== Slides that will be processed ==="
resolve_slides
for s in "${SLIDES[@]}"; do echo "    $s"; done

echo ""
echo "=== Output paths (all NEW; nothing existing is overwritten) ==="
echo "  pass 1 : $PASS1_DIR"
echo "  pass 2 : $ROOT_CHOICES"
echo "  pass 3 : $FINAL_DIR"
echo "  index  : $INDEX_MD"

for p in "$PASS1_DIR" "$FINAL_DIR"; do
    if [ -d "$p" ] && [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
        echo ""
        echo "ERROR: $p already exists and is non-empty. Refusing to overwrite."
        echo "       Move it aside or edit OUT_ROOT in _individual_timepoint_common.sh."
        exit 1
    fi
done
if [ -e "$ROOT_CHOICES" ]; then
    echo ""
    echo "ERROR: $ROOT_CHOICES exists. Refusing to overwrite."
    exit 1
fi

cd "$HOME/cancer_trajectory_atlas"

P1=$(sbatch --parsable jobs/run_individual_timepoint_pass1.sh)
echo ""
echo "  pass 1 submitted: $P1  (GPU, arbitrary root — NOT the deliverable)"

P2=$(sbatch --parsable --dependency=afterok:"$P1" jobs/run_individual_timepoint_pass2.sh)
echo "  pass 2 submitted: $P2  (CPU, root choice)      after $P1"

P3=$(sbatch --parsable --dependency=afterok:"$P2" jobs/run_individual_timepoint_pass3.sh)
echo "  pass 3 submitted: $P3  (GPU, deliverable)      after $P2"

echo ""
echo "  Watch:   squeue -u \$USER"
echo "  Then read: $INDEX_MD"
banner
