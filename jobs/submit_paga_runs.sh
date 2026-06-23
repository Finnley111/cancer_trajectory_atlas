#!/bin/bash
# Launcher for the PAGA 6-variant experiment suite. This is a PLAIN script —
# run it directly on the login node, do NOT submit it with sbatch:
#
#   bash ~/cancer_trajectory_atlas/jobs/submit_paga_runs.sh
#
# It submits jobs/run_cache_prepop.sh, then submits all 6
# jobs/run_paga_variant.sh jobs with --dependency=afterok on the prepop job,
# so none of the 6 (which run in parallel and share one feature cache) can
# start until the cache is guaranteed fully populated and the prepop job has
# stopped writing to it.
#
# Prerequisite: make sure this checkout includes the PAGA gate commit
# (analysis/diffusion.py, run_all.py, utils/viz.py) — `git pull` first if
# unsure, otherwise these jobs will silently run the pre-PAGA pipeline code.

set -euo pipefail

JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"

echo "============================================"
echo "  Submitting PAGA cache pre-population job"
echo "============================================"
CACHE_JOBID=$(sbatch --parsable "$JOBS_DIR/run_cache_prepop.sh")
echo "  Cache prepop job ID: $CACHE_JOBID"

declare -A VARIANT_JOBIDS

VARIANTS=(
    "all       harmony    all_harmony"
    "all       noharmony  all_noharmony"
    "section1  harmony    section1_harmony"
    "section1  noharmony  section1_noharmony"
    "section2  harmony    section2_harmony"
    "section2  noharmony  section2_noharmony"
)

echo ""
echo "============================================"
echo "  Submitting 6 variant jobs (dependency: afterok:$CACHE_JOBID)"
echo "============================================"

for VARIANT_SPEC in "${VARIANTS[@]}"; do
    read -r SUBSET HARMONY_MODE NAME <<< "$VARIANT_SPEC"
    JOBID=$(sbatch --parsable \
        --dependency=afterok:"$CACHE_JOBID" \
        --job-name="atlas_paga_${NAME}" \
        "$JOBS_DIR/run_paga_variant.sh" "$SUBSET" "$HARMONY_MODE" "$NAME")
    VARIANT_JOBIDS["$NAME"]="$JOBID"
    echo "  $NAME  ->  job $JOBID  (subset=$SUBSET, harmony=$HARMONY_MODE)"
done

echo ""
echo "============================================"
echo "  SUBMISSION SUMMARY"
echo "============================================"
echo "  Cache prepop:        $CACHE_JOBID"
for NAME in "${!VARIANT_JOBIDS[@]}"; do
    printf "  %-20s %s\n" "$NAME:" "${VARIANT_JOBIDS[$NAME]}"
done
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Once the cache prepop job finishes (sacct -j $CACHE_JOBID --format=JobID,State,ExitCode),"
echo "the 6 variant jobs above will start automatically. They run in parallel"
echo "and require no GPU (cache-hit-only)."
echo ""
echo "All outputs will land under: \$SCRATCH/results/runs_paga/"
