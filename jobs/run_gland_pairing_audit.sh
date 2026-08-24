#!/bin/bash
# TASK 1 — gland pairing audit. Establishes the matched-pair design empirically.
#
# The 16 slides are 8 MATCHED PAIRS, not 16 independent samples: each mouse-flank
# (gland) contributes one slide to 2M-1 (Carnoy's) and one to 2M-2 (PFA). Every
# analysis to date has treated them as independent, which makes the C(16,8) = 12,870
# exact between-section test mis-specified — its null admits between-gland and
# between-mouse variation the paired design already controls.
#
# This job does NOT fix anything. It parses mouse / flank / section out of the
# per-duct tables, prints the 8x2 pairing table, gates on the design actually being
# balanced, and reports per-gland marginals. If the design is not balanced it exits
# non-zero, because a sign-flip null assumes exactly one value per gland per section.
#
# READ-ONLY. Reads two CSVs, recomputes nothing, reruns no pipeline stage, and does
# not touch results/holeyness_section_comparison/ (the unpaired result on record).
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_gland_pairing_audit.sh
#    or: bash   ~/cancer_trajectory_atlas/jobs/run_gland_pairing_audit.sh   (it is
#        small enough to run on a login node)

#SBATCH --account=def-lmarti46
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=gland_pairing
#SBATCH --output=logs/gland_pairing-%j.out

set -euo pipefail
mkdir -p logs

HOLEY="$SCRATCH/results/holeyness"
CSV_2M1="$HOLEY/2M-1/holeyness_per_duct.csv"
CSV_2M2="$HOLEY/2M-2/holeyness_per_duct.csv"
OUT_DIR="$SCRATCH/results/holeyness_section_comparison_paired"

# The unpaired result on record. Never written to by this job or by Task 2.
UNPAIRED="$SCRATCH/results/holeyness_section_comparison"

echo "============================================================================"
echo "  TASK 1 — gland pairing audit"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$UNPAIRED"|"$UNPAIRED"/*|"$HOLEY"|"$HOLEY"/*)
        echo "ERROR: output would land inside a protected results tree."; exit 1;;
esac

MISSING=0
for p in "$CSV_2M1" "$CSV_2M2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: a per-duct table is missing. This audit reads the tables the"
    echo "       holeyness validation wrote and must not recompute them."
    exit 1; }

module load StdEnv/2023 python/3.11 gcc openblas hdf5
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.gland_pairing_audit \
    --sections      2M-1 2M-2 \
    --per-duct-csvs "$CSV_2M1" "$CSV_2M2" \
    --output-dir    "$OUT_DIR"

echo ""
echo "============================================================================"
echo "  TASK 1 COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/gland_pairing_audit.md    <- read this"
echo "  $OUT_DIR/gland_pairing_audit.json"
echo ""
echo "  Unpaired result, untouched: $UNPAIRED"
echo ""
echo "  Read (b) FIRST. If the design is not balanced 8x2, Task 2 must not run:"
echo "  a sign-flip null assumes exactly one value per gland per section."
