#!/bin/bash
# Submit all new-annotation pipeline runs as parallel SLURM jobs.
#
# Each run uses annotations_ratio/ and writes to:
#   $SCRATCH/results/new_annotations/<run_type>/
#
# Prerequisites:
#   - annotations_ratio/ exists (run submit_annotation_check.sh first)
#   - Phikon model cached at $SCRATCH/huggingface_cache
#
# Usage:
#   bash jobs/submit_new_annotations_all.sh

set -euo pipefail

mkdir -p logs

SCRIPT=~/cancer_trajectory_atlas/jobs/run_new_annotations.sh

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT not found"
    exit 1
fi

RUNS=(none_harmony macenko_harmony none_section1 none_section2 individual)

echo "Submitting ${#RUNS[@]} jobs..."
echo ""

for RUN_TYPE in "${RUNS[@]}"; do
    JOB_ID=$(RUN_TYPE="$RUN_TYPE" sbatch \
        --job-name="new_ann_${RUN_TYPE}" \
        --output="logs/new_ann_${RUN_TYPE}-%j.out" \
        --export=ALL,RUN_TYPE="$RUN_TYPE" \
        "$SCRIPT" | awk '{print $NF}')
    echo "  $RUN_TYPE -> job $JOB_ID  (logs/new_ann_${RUN_TYPE}-${JOB_ID}.out)"
done

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Results will be in: \$SCRATCH/results/new_annotations/"
