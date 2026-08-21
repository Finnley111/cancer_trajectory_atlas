#!/bin/bash
# Do the two sections' holey-ness correlations actually differ?
#
# A prior diagnostic corrected a circulated error: 2M-2's "0.020" was
# rho(pseudotime, nuclear_density), not rho(pseudotime, hole_pct). Recomputed
# correctly the validation is POSITIVE IN BOTH sections -- 0.276 (2M-1, CI
# [0.217, 0.347], 8/8 slides) and 0.1906 (2M-2, CI [0.086, 0.294], 7/8 slides) --
# and both candidate explanations for an asymmetry were ruled out.
#
#   TASK 1  Complete the correlation table. Only 2M-1's area-adjusted value
#           (0.131) is on record. Compute raw, |area, and |area+nuclear_density
#           for BOTH sections, each with a slide-clustered bootstrap interval and
#           a within-slide-shuffled permutation p-value.
#
#   TASK 2  Test the between-section difference EXACTLY: enumerate all
#           C(16,8) = 12,870 slide-level relabellings, for the raw statistic, the
#           area-adjusted partial, rho(area, pseudotime), and a within-slide
#           normalised variant. No subsampling.
#
# WHAT IT CANNOT ESTABLISH. Fixation is perfectly collinear with section -- every
# Carnoy's slide is 2M-1, every PFA slide is 2M-2 -- so neither a difference nor
# its absence can be attributed to fixation chemistry as opposed to anatomical
# region. Bridge samples would be required. It also does not adjudicate whether
# duct area is a mediator or a confounder; both estimands are reported.
#
# AN EXCHANGEABILITY PROBLEM, MEASURED NOT ASSUMED AWAY. The observed statistic
# comes from the ONE split that is section-pure; almost all 12,870 splits are
# section-MIXED, and the sections differ in their marginal distributions of
# hole_pct, area and pseudotime. A mixed group's pooled Spearman can therefore
# pick up between-section contrast a pure group never sees. The module reports
# the test as specified, PLUS a purity-stratified null (is the null homogeneous
# across compositions?) and a within-slide-normalised variant whose null IS
# exchangeable with its observed value. Where they disagree, prefer the latter.
#
# READ-ONLY. Reads the per-duct tables as the holeyness validation wrote them.
# Recomputes no pseudotime, no duct assignment, no duct table; reruns no pipeline
# stage; modifies no module; writes only to its own new output directory.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_section_comparison.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=holey_seccmp
#SBATCH --output=logs/holey_seccmp-%j.out

set -euo pipefail
mkdir -p logs

HOLEY_BASE="$SCRATCH/results/holeyness"
OUT_DIR="$SCRATCH/results/holeyness_section_comparison"

CSV_2M1="$HOLEY_BASE/2M-1/holeyness_per_duct.csv"
CSV_2M2="$HOLEY_BASE/2M-2/holeyness_per_duct.csv"

N_BOOT="${N_BOOT:-2000}"
N_PERM="${N_PERM:-5000}"

echo "============================================================================"
echo "  Holey-ness section comparison"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Boot   : $N_BOOT slide-clustered resamples"
echo "  Perm   : $N_PERM within-slide shuffles (Task 1)"
echo "  Exact  : all C(16,8) = 12,870 slide relabellings x 4 statistics (Task 2)"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

# Never write into the holeyness tree this reads, and never silently replace a
# previous run of this comparison.
case "$OUT_DIR" in
    "$HOLEY_BASE"|"$HOLEY_BASE"/*|"$SCRATCH/results/per_section"*|"$SCRATCH/results/per_section_v2"*)
        echo "ERROR: output is inside a protected results tree."; exit 1;;
esac
if [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUT_DIR already exists and is not empty."
    echo "       Set FORCE=1 to overwrite, or move the previous run aside."
    exit 1
fi

MISSING=0
for p in "$CSV_2M1" "$CSV_2M2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: a per-duct table is missing. This analysis reads the tables the"
    echo "       holeyness validation wrote and must not recompute them. Run"
    echo "       jobs/run_holeyness_validation.sh first."
    exit 1; }

module load StdEnv/2023 python/3.11 gcc openblas hdf5
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.holeyness_section_comparison \
    --sections      2M-1 2M-2 \
    --per-duct-csvs "$CSV_2M1" "$CSV_2M2" \
    --output-dir    "$OUT_DIR" \
    --n-boot        "$N_BOOT" \
    --n-perm        "$N_PERM"

echo ""
echo "============================================================================"
echo "  COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/holeyness_section_comparison.md    <- read this"
echo "  $OUT_DIR/holeyness_section_comparison.json"
echo "  $OUT_DIR/forest_correlations.{png,pdf}"
echo "  $OUT_DIR/exact_null_distributions.{png,pdf}"
echo "  $OUT_DIR/null_by_split_purity.{png,pdf}"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. Step 0's consistency gate. 2M-1's |area value must reproduce the"
echo "      on-record 0.131, and the rank-residual and algebraic partial"
echo "      implementations must agree. If either fails, stop there."
echo "   2. Task 1's table, then the line on whether the ADJUSTED values agree"
echo "      more closely between sections than the RAW values do."
echo "   3. Task 2's POWER STATEMENT before any p-value. With 8 slides per"
echo "      section the minimum detectable difference is large; a"
echo "      non-significant result is NO EVIDENCE OF A DIFFERENCE, never"
echo "      evidence of equivalence."
echo "   4. The within-slide-normalised row of Task 2. Its null is the one"
echo "      exchangeable with its observed value; if it disagrees with the raw"
echo "      row, prefer it and say so."
echo ""
echo "  Fixation is collinear with section, so nothing here attributes any"
echo "  difference to fixation chemistry. Bridge samples would be required."
