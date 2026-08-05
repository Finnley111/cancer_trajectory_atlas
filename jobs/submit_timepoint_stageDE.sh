#!/bin/bash
# Submit Stage D (GPU) and Stage E (CPU) together, with E gated on D succeeding.
#
# Stage E is submitted with --dependency=afterok:<stageD_jobid>, so it starts
# only if Stage D exits 0. If Stage D fails or hits its walltime, Stage E never
# runs and SLURM leaves it in the queue as DependencyNeverSatisfied -- cancel it
# with `scancel <stageE_jobid>` before resubmitting Stage D.
#
# NOTE: Stage D is resumable. If it times out, resubmit it -- cached per-slide
# features let it continue from where it stopped. Use this wrapper again for the
# resubmission so a fresh Stage E is chained to the new attempt.
#
# Usage:
#   bash ~/cancer_trajectory_atlas/jobs/submit_timepoint_stageDE.sh

set -euo pipefail
cd "$(dirname "$0")/.."
JOBS_DIR="$(pwd)/jobs"

echo "Submitting Stage D (GPU: extraction + projection)..."
D_OUT=$(sbatch --parsable "$JOBS_DIR/run_timepoint_stageD_projection.sh")
D_ID="${D_OUT%%;*}"        # --parsable may return "jobid;cluster"
echo "  Stage D job ID: $D_ID"

echo "Submitting Stage E (CPU: diagnostic), gated on Stage D success..."
E_OUT=$(sbatch --parsable --dependency="afterok:${D_ID}" \
        "$JOBS_DIR/run_timepoint_stageE_diagnostic.sh")
E_ID="${E_OUT%%;*}"
echo "  Stage E job ID: $E_ID  (waits for afterok:${D_ID})"

echo ""
echo "Both submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/timepoint_stageD-${D_ID}.out"
echo ""
echo "When Stage D finishes, capture a REAL walltime measurement (this project"
echo "has only guesses so far) with:"
echo "  seff ${D_ID}"
echo "  sacct -j ${D_ID} --format=JobID,Elapsed,MaxRSS,State"
echo ""
echo "Results land in:"
echo "  \$SCRATCH/results/timepoint_cohort/stageD_projection/"
echo "  \$SCRATCH/results/timepoint_cohort/stageE_diagnostic/"
