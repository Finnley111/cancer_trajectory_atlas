#!/bin/bash
# TASKS 2 + 3 — between-section comparison, corrected for the matched-pair design.
#
# THE CORRECTION. The 16 slides are 8 MATCHED PAIRS: each mouse-flank gland
# contributes one slide to 2M-1 (Carnoy's) and one to 2M-2 (PFA). The comparison on
# record permuted all C(16,8) = 12,870 splits of 16 slides, a null that admits
# between-gland and between-mouse variation the paired design already controls. Its
# minimum detectable differences were therefore inflated (0.250 raw, 0.133 on the
# best-powered variant).
#
# This recomputes it as an exact 2^8 = 256 sign-flip test on the 8 within-gland
# differences, enumerated exhaustively, for all four estimands the unpaired test
# covered. It then reports both designs side by side.
#
# THE P-VALUE FLOOR. With 256 permutations the smallest attainable two-sided p is
# 2/256 = 0.0078. A paired p at that floor is NOT weaker evidence than the unpaired
# test's 1.554e-4 -- it is the same evidence at coarser resolution. The unpaired test
# could resolve further only by assuming an independence that is false here. The
# module reports "< 0.0078" rather than 0.
#
# BALANCE GATE. The module refuses to compute anything unless the design really is a
# balanced 8x2, because a sign-flip null assumes exactly one value per gland per
# section. Run jobs/run_gland_pairing_audit.sh (Task 1) first and read its verdict.
#
# READ-ONLY. Reads two per-duct CSVs and the unpaired result's JSON. Recomputes no
# pseudotime, no duct assignment and no duct table; reruns no pipeline stage; and
# NEVER writes to results/holeyness_section_comparison/, the unpaired result on
# record.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_paired_comparison.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=holey_paired
#SBATCH --output=logs/holey_paired-%j.out

set -euo pipefail
mkdir -p logs

HOLEY="$SCRATCH/results/holeyness"
CSV_2M1="$HOLEY/2M-1/holeyness_per_duct.csv"
CSV_2M2="$HOLEY/2M-2/holeyness_per_duct.csv"

# The unpaired result on record. READ ONLY -- never written to.
UNPAIRED_DIR="$SCRATCH/results/holeyness_section_comparison"
UNPAIRED_JSON="$UNPAIRED_DIR/holeyness_section_comparison.json"

OUT_DIR="$SCRATCH/results/holeyness_section_comparison_paired"

echo "============================================================================"
echo "  TASKS 2-3 — paired between-section comparison"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Design : 8 matched pairs, exact 2^8 = 256 sign-flip null"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$UNPAIRED_DIR"|"$UNPAIRED_DIR"/*|"$HOLEY"|"$HOLEY"/*)
        echo "ERROR: output would land inside a protected results tree."; exit 1;;
esac

MISSING=0
for p in "$CSV_2M1" "$CSV_2M2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: a per-duct table is missing. This analysis reads the tables the"
    echo "       holeyness validation wrote and must not recompute them."
    exit 1; }

# The unpaired JSON is optional: without it the paired half still runs, but the
# side-by-side comparison that Task 3 asks for cannot be produced.
UNPAIRED_ARG=()
echo -n "  $UNPAIRED_JSON : "
if [ -e "$UNPAIRED_JSON" ]; then
    echo "ok"
    UNPAIRED_ARG=(--unpaired-json "$UNPAIRED_JSON")
else
    echo "NOT FOUND"
    echo "  NOTE: the unpaired result is absent, so Task 3's side-by-side will be"
    echo "        PARTIAL -- the paired half is reported alone. Nothing is invented"
    echo "        to fill the gap."
fi

module load StdEnv/2023 python/3.11 gcc openblas hdf5
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.holeyness_paired_comparison \
    --sections      2M-1 2M-2 \
    --per-duct-csvs "$CSV_2M1" "$CSV_2M2" \
    --output-dir    "$OUT_DIR" \
    "${UNPAIRED_ARG[@]+"${UNPAIRED_ARG[@]}"}"

echo ""
echo "============================================================================"
echo "  COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/holeyness_paired_comparison.md    <- read this"
echo "  $OUT_DIR/holeyness_paired_comparison.json"
echo "  $OUT_DIR/paired_by_gland.{png,pdf}"
echo "  $OUT_DIR/paired_null_distributions.{png,pdf}"
echo ""
echo "  Unpaired result, untouched: $UNPAIRED_DIR"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. The balance gate. If it did not report BALANCED the run stopped, and"
echo "      nothing below exists."
echo "   2. The per-gland tables. Eight rows each -- you can see the paired"
echo "      structure directly, including which glands disagree in sign."
echo "   3. The side-by-side MDD column. That is what the correct design buys."
echo "   4. Framing, per estimand: 'same conclusion, tighter bounds' is the"
echo "      expected outcome and is NOT a new discovery. Only a row flagged"
echo "      CONCLUSION CHANGES is a change in what the data support."
echo ""
echo "  A paired p of '< 0.0078' is the design's floor, not weak evidence."
