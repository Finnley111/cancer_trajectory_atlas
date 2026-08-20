#!/bin/bash
# Why does the expert holey-ness validation succeed in 2M-1 and fail in 2M-2?
#
# The validation has been run on both sections against per_section_v2 pseudotime
# and the outcome is strongly asymmetric. Duct retention is comparable between
# sections, so differential exclusion is unlikely to be the whole explanation and
# the cause is unknown. This diagnostic discriminates between:
#
#   (A) MECHANICAL   -- 2M-2 has less usable spread in hole_pct, so there is
#                       little signal available to correlate with anything.
#   (B) GROUND TRUTH -- the duct-diameter-to-holes relationship the pathologist
#                       describes holds under one fixation and breaks under the
#                       other, making the null a property of the annotation.
#   (C) AXIS         -- the 2M-2 pseudotime axis is itself degenerate. Documented
#                       with existing evidence; NOT tested here, since testing it
#                       needs the h5ad and this diagnostic reads CSVs only.
#
# WHAT IT CANNOT ESTABLISH: that FIXATION causes anything. Fixation is perfectly
# collinear with section in this cohort -- every Carnoy's slide is 2M-1 and every
# PFA slide is 2M-2 -- so fixation and anatomical region are not separable.
# Bridge samples (serial sections from one block, both fixations, one staining
# run) would be required. The report states this in its opening paragraph.
#
# A NOTE ON CHECK 1. Spearman is invariant to monotone rescaling, so a difference
# in hole_pct's raw sd/iqr/cv CANNOT by itself produce a rank-correlation null.
# The module therefore reports those statistics as requested AND a tie/granularity
# block, and decides explanation (A) quantitatively: it compares the maximum
# attenuation the tie structure can produce against the size of the drop in rho
# that actually needs explaining. Ties move slowly -- three equal levels only
# reach rank_sd_ratio ~0.94 -- so this matters.
#
# READ-ONLY. Reads the existing per-duct tables as the holey-ness validation
# wrote them. Recomputes no duct table, reruns no pipeline stage, modifies no
# module, and writes only to its own new output directory.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_asymmetry.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=holey_asym
#SBATCH --output=logs/holey_asym-%j.out

set -euo pipefail
mkdir -p logs

HOLEY_BASE="$SCRATCH/results/holeyness"
OUT_DIR="$SCRATCH/results/holeyness_asymmetry_diagnostic"

CSV_2M1="$HOLEY_BASE/2M-1/holeyness_per_duct.csv"
CSV_2M2="$HOLEY_BASE/2M-2/holeyness_per_duct.csv"
V2_2M1="$HOLEY_BASE/2M-1/v2_area_adjusted"
V2_2M2="$HOLEY_BASE/2M-2/v2_area_adjusted"

N_PERM="${N_PERM:-5000}"
# rho(pt, hole_pct) as quoted in the diagnostic request, carried so the report
# can reconcile them against what the tables actually contain.
QUOTED_2M1="${QUOTED_2M1:-0.276}"
QUOTED_2M2="${QUOTED_2M2:-0.020}"

echo "============================================================================"
echo "  Holey-ness validation asymmetry diagnostic"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Perms  : $N_PERM (within-slide shuffle)"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

# Guard the inputs: this diagnostic must never write into the holeyness tree it
# reads, and must never overwrite an existing diagnostic run.
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
    echo "ERROR: a per-duct table is missing. This diagnostic reads the tables the"
    echo "       holeyness validation wrote and must not recompute them -- a"
    echo "       recomputed table could differ from the one whose result is being"
    echo "       explained. Run jobs/run_holeyness_validation.sh first."
    exit 1; }

# v2 supplies the EXCLUDED-duct population (ducts with zero assigned patches).
# v1's per-duct CSV contains retained ducts only, so without v2 that half of
# Check 1 is reported as skipped rather than approximated.
V2_ARGS=()
if [ -f "$V2_2M1/duct_table_full.csv" ] && [ -f "$V2_2M2/duct_table_full.csv" ]; then
    V2_ARGS=(--v2-dirs "$V2_2M1" "$V2_2M2")
    echo "  v2 duct_table_full.csv found for both sections -- excluded-duct"
    echo "  comparison ENABLED."
else
    echo "  NOTE: v2 duct_table_full.csv not found for both sections."
    echo "        The excluded-duct half of Check 1 will be reported as SKIPPED,"
    echo "        not approximated. Everything else runs."
fi

module load StdEnv/2023 python/3.11 gcc openblas hdf5
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.holeyness_asymmetry \
    --sections       2M-1 2M-2 \
    --per-duct-csvs  "$CSV_2M1" "$CSV_2M2" \
    --output-dir     "$OUT_DIR" \
    --n-perm         "$N_PERM" \
    --quoted-rho     "$QUOTED_2M1" "$QUOTED_2M2" \
    "${V2_ARGS[@]+"${V2_ARGS[@]}"}"

echo ""
echo "============================================================================"
echo "  DIAGNOSTIC COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/holeyness_asymmetry.md          <- read this"
echo "  $OUT_DIR/holeyness_asymmetry.json"
echo "  $OUT_DIR/hole_pct_distribution.{png,pdf}"
echo "  $OUT_DIR/area_vs_hole_pct.{png,pdf}"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. Step 0's reconciliation table. If the recomputed rho(pt, hole_pct)"
echo "      disagrees with the quoted value, the quoted figure came from a"
echo "      different quantity, axis or run, and the framing of the asymmetry"
echo "      must be revisited before any verdict here is acted on."
echo "   2. Check 1's 'Can ties account for the drop?' block. That, not the"
echo "      sd/iqr/cv table, is what decides explanation (A)."
echo "   3. Check 2's rho(area, hole_area) sanity value. If it is not strongly"
echo "      positive in a section, something is wrong with that section's"
echo "      annotation and the rest of Check 2 should not be read."
echo "   4. Check 3's outcome, one of (i)-(iv). Outcome (iv) means neither"
echo "      candidate explains it -- that is a real result, not a failure."
echo ""
echo "  This cannot establish that fixation causes anything: fixation is"
echo "  collinear with section in this cohort. Bridge samples would be required."
