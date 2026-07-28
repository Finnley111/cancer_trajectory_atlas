#!/bin/bash
# Timepoint projection, Stage 2: stain batch check (CPU) -- HARD GATE.
#
# Before any GPU work (Stage 3), tests whether the 8 new timepoint slides (4W/8W)
# differ in staining from the 8 existing pipeline 2M-1 slides, by an amount
# comparable to the project's own known cross-section h_intensity confound
# (rank-biserial r = 0.71). If so, timepoint would be confounded with staining (not
# just with mouse) and the projection-based timepoint comparison in Stage 3/4 would
# be uninterpretable.
#
# STOP HERE after this job. Report stage2_stain_check.md/.json and await explicit
# confirmation before Stage 3 (GPU feature extraction + projection) is even written,
# let alone submitted -- per the task's hard gate.
#
# Reads (existing 2M-1 PNGs are READ-ONLY, never modified):
#   $SCRATCH/data/timepoint_x5_cropped/*.png          (Stage 1 output)
#   $SCRATCH/data/MCF7_x5_cropped/*.png                (existing pipeline data)
#   ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt
#   ~/cancer_trajectory_atlas/jobs/slides_section1.txt (existing, read-only)
#
# Writes (NEW directory):
#   $SCRATCH/results/timepoint_projection/stage2_stain_check/stage2_stain_check.json
#   $SCRATCH/results/timepoint_projection/stage2_stain_check/stage2_stain_check.md
#
# Usage (after Stage 1 has completed and been reviewed):
#   sbatch ~/cancer_trajectory_atlas/jobs/run_timepoint_stage2_stain_check.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=timepoint_stage2_stain_check
#SBATCH --output=logs/timepoint_stage2_stain_check-%j.out

set -euo pipefail
mkdir -p logs

# ── Parameters ────────────────────────────────────────────────────────────────
NEW_SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_timepoint.txt"
NEW_PNG_DIR="$SCRATCH/data/timepoint_x5_cropped"
EXISTING_SLIDE_LIST="$HOME/cancer_trajectory_atlas/jobs/slides_section1.txt"
EXISTING_PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
KNOWN_CONFOUND_R=0.71

OUTPUT_DIR="$SCRATCH/results/timepoint_projection/stage2_stain_check"

echo "========================================================"
echo "  Timepoint projection — Stage 2: stain batch check"
echo "  Job ID           : ${SLURM_JOB_ID:-local}"
echo "  New slides       : $NEW_SLIDE_LIST ($NEW_PNG_DIR)"
echo "  Existing 2M-1    : $EXISTING_SLIDE_LIST ($EXISTING_PNG_DIR)"
echo "  Known confound r : $KNOWN_CONFOUND_R"
echo "  Output dir       : $OUTPUT_DIR"
echo "========================================================"

echo ""
echo "=== Pre-run checks ==="
echo -n "New slide list      : "; ls -lh "$NEW_SLIDE_LIST"      2>/dev/null || echo "NOT FOUND"
echo -n "New PNG dir         : "; ls -d  "$NEW_PNG_DIR"          2>/dev/null || echo "NOT FOUND — run Stage 1 first"
echo -n "Existing slide list : "; ls -lh "$EXISTING_SLIDE_LIST" 2>/dev/null || echo "NOT FOUND"
echo -n "Existing PNG dir    : "; ls -d  "$EXISTING_PNG_DIR"     2>/dev/null || echo "NOT FOUND"
echo "======================"
echo ""

module load StdEnv/2023 python/3.11 gcc opencv openslide
source ~/envs/atlas/bin/activate

cd ~

python -m cancer_trajectory_atlas.analysis.timepoint_stage2_stain_check \
    --new-slide-list      "$NEW_SLIDE_LIST" \
    --new-png-dir         "$NEW_PNG_DIR" \
    --existing-slide-list "$EXISTING_SLIDE_LIST" \
    --existing-png-dir    "$EXISTING_PNG_DIR" \
    --known-confound-r    "$KNOWN_CONFOUND_R" \
    --output-dir          "$OUTPUT_DIR"

echo ""
echo "========================================================"
echo "  STAGE 2 STAIN BATCH CHECK COMPLETE — HARD GATE"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/stage2_stain_check.json"
echo "  $OUTPUT_DIR/stage2_stain_check.md"
echo ""
echo "*** STOP HERE. Do not submit any Stage 3 job. ***"
echo "Report stage2_stain_check.md and await explicit confirmation before Stage 3"
echo "(GPU feature extraction + projection) is designed or run."
echo ""
