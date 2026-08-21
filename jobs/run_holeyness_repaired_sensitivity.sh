#!/bin/bash
# Repaired-axis SENSITIVITY check, reported beside the primary result.
#
# PRE-DECLARED, BEFORE ANY NUMBER WAS SEEN:
#   * This is a SENSITIVITY CHECK. The PRIMARY external validation remains the v2
#     density-rooted result -- rho(pt, hole_pct) = +0.2763 (2M-1) / +0.1906
#     (2M-2), with the exact slide-level permutation test showing no evidence the
#     sections differ (p = 0.313 on the within-slide normal-scores variant, MDD
#     0.133). The repaired axis does NOT replace those numbers.
#   * The hypothesis is MECHANISTIC: that the between-section divergence in
#     rho(duct area, pseudotime) (+0.4325 vs -0.0844, exact p = 1.55e-4) is an
#     artifact of 2M-2's degenerate root set rather than a property of the tissue.
#   * Predictions recorded in advance: (a) 2M-2's rho(area, pseudotime) moves
#     toward zero-or-positive; (b) 2M-2's pseudotime_std drops from 27.70% toward
#     ~3.40%; (c) the between-section difference in rho(area, pseudotime) falls
#     below the primary run's MDD of 0.353. Each is reported held or not held.
#   * 2M-1 had ZERO discordant roots, so it is a built-in NEGATIVE CONTROL.
#
# THREE STAGES, all writing to NEW directories:
#   1. holeyness.py on the repaired axis, both sections. Same module, same
#      arguments as the primary runs -- only --results and --output-dir change.
#   2. holeyness_section_comparison on the new per-duct tables. Same 2000
#      bootstrap resamples and 5000 permutations, so the runs are comparable.
#   3. holeyness_repaired_sensitivity: the side-by-side report.
#
# THE NEGATIVE CONTROL NEEDS A GATE. The two sections' PRIMARY validations read
# DIFFERENT run trees -- 2M-1 from per_section/, 2M-2 from per_section_v2/ --
# while the repaired axis derives from per_section_v2 for both. So the 2M-1 arm
# changes the baseline tree as well as the repair (a no-op there). Stage 3
# compares the two 2M-1 results.csv pseudotime columns directly and declares the
# control CONFOUNDED if they differ, rather than reporting a tree difference as
# an unexpected change.
#
# PROTECTED, NEVER WRITTEN TO: results/holeyness/2M-1, results/holeyness/2M-2,
# results/holeyness_section_comparison. Those hold the primary analysis.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_repaired_sensitivity.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=holey_repaired
#SBATCH --output=logs/holey_repaired-%j.out

set -euo pipefail
mkdir -p logs
REPO="$HOME/cancer_trajectory_atlas"

AXES="$SCRATCH/results/holeyroot_experiment/anchor_axes/v2_repaired"
HOLEY="$SCRATCH/results/holeyness"

# PRIMARY outputs -- read only, never written to.
PRIM_1="$HOLEY/2M-1"
PRIM_2="$HOLEY/2M-2"
PRIM_CMP="$SCRATCH/results/holeyness_section_comparison"

# NEW outputs.
OUT_1="$HOLEY/2M-1_repaired"
OUT_2="$HOLEY/2M-2_repaired"
OUT_CMP="$SCRATCH/results/holeyness_section_comparison_repaired"
OUT_REPORT="$SCRATCH/results/holeyness_repaired_sensitivity"

ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"
S1="$REPO/jobs/slides_section1.txt"
S2="$REPO/jobs/slides_section2.txt"

# For the negative-control gate: the two trees 2M-1's primary and repaired axes
# ultimately derive from.
BASE_1_PRIMARY="$SCRATCH/results/per_section/atlas_2M-1/results.csv"
BASE_1_V2="$SCRATCH/results/per_section_v2/atlas_2M-1/results.csv"

N_BOOT="${N_BOOT:-2000}"
N_PERM="${N_PERM:-5000}"

echo "============================================================================"
echo "  Repaired-axis SENSITIVITY check   (the primary v2 result is unchanged)"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Axis   : $AXES"
echo "  Out    : $OUT_1, $OUT_2, $OUT_CMP, $OUT_REPORT   (all NEW)"
echo "============================================================================"

# Refuse to write anywhere near the primary analysis.
for d in "$OUT_1" "$OUT_2" "$OUT_CMP" "$OUT_REPORT"; do
    case "$d" in
        "$PRIM_1"|"$PRIM_1"/*|"$PRIM_2"|"$PRIM_2"/*|"$PRIM_CMP"|"$PRIM_CMP"/*)
            echo "ERROR: $d resolves inside a PRIMARY output directory."; exit 1;;
    esac
    if [ -n "$(ls -A "$d" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "ERROR: $d already exists and is not empty. FORCE=1 to overwrite."
        exit 1
    fi
done

MISSING=0
for p in "$AXES/atlas_2M-1/results.csv" "$AXES/atlas_2M-2/results.csv" \
         "$AXES/atlas_2M-1/anchor_axis.json" "$AXES/atlas_2M-2/anchor_axis.json" \
         "$PRIM_1/holeyness_per_duct.csv" "$PRIM_2/holeyness_per_duct.csv" \
         "$PRIM_CMP/holeyness_section_comparison.json" \
         "$EXPORT_2M1" "$EXPORT_2M2" "$ANN_DIR" "$SLIDE_DIMS" "$S1" "$S2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }

# The gate inputs are optional: without them the report says the gate was NOT RUN
# rather than assuming the baseline trees agree.
GATE_ARGS=()
if [ -f "$BASE_1_PRIMARY" ] && [ -f "$BASE_1_V2" ]; then
    GATE_ARGS=(--baseline-csvs "$BASE_1_PRIMARY" "$BASE_1_V2")
    echo "  negative-control gate ENABLED (both 2M-1 baseline trees present)."
else
    echo "  NOTE: one of the 2M-1 baseline results.csv files is absent, so the"
    echo "        negative-control gate will be reported as NOT RUN, not assumed."
fi

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_1" "$OUT_2" "$OUT_CMP" "$OUT_REPORT"
cd ~

echo ""
echo "--- STAGE 1: holeyness validation on the repaired axis --------------------"
echo "  Identical to the primary invocations except --results and --output-dir."
echo ""
echo "  2M-1 command:"
echo "    holeyness --section 2M-1 --export $EXPORT_2M1 --annotation-dir $ANN_DIR"
echo "      --slide-dimensions $SLIDE_DIMS --results $AXES/atlas_2M-1/results.csv"
echo "      --output-dir $OUT_1 --slide-list $S1 --aggregation median"
echo "      --n-permutations 1000"
python -m cancer_trajectory_atlas.analysis.holeyness \
    --section          2M-1 \
    --export           "$EXPORT_2M1" \
    --annotation-dir   "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --results          "$AXES/atlas_2M-1/results.csv" \
    --output-dir       "$OUT_1" \
    --slide-list       "$S1" \
    --aggregation      median \
    --n-permutations   1000

echo ""
echo "  2M-2 command:"
echo "    holeyness --section 2M-2 --export $EXPORT_2M2 --annotation-dir $ANN_DIR"
echo "      --slide-dimensions $SLIDE_DIMS --results $AXES/atlas_2M-2/results.csv"
echo "      --output-dir $OUT_2 --slide-list $S2 --aggregation median"
echo "      --n-permutations 1000"
python -m cancer_trajectory_atlas.analysis.holeyness \
    --section          2M-2 \
    --export           "$EXPORT_2M2" \
    --annotation-dir   "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --results          "$AXES/atlas_2M-2/results.csv" \
    --output-dir       "$OUT_2" \
    --slide-list       "$S2" \
    --aggregation      median \
    --n-permutations   1000

# Duct assignment does not depend on pseudotime, so retention MUST match the
# primary runs. A difference means the two runs used different inputs and the
# comparison would not be like-for-like.
echo ""
echo "--- retention gate --------------------------------------------------------"
for pair in "2M-1:$PRIM_1:$OUT_1:1602" "2M-2:$PRIM_2:$OUT_2:1360"; do
    SEC="${pair%%:*}"; REST="${pair#*:}"
    P="${REST%%:*}"; REST="${REST#*:}"
    O="${REST%%:*}"; EXP="${REST##*:}"
    NP=$(( $(wc -l < "$P/holeyness_per_duct.csv") - 1 ))
    NO=$(( $(wc -l < "$O/holeyness_per_duct.csv") - 1 ))
    echo "  $SEC: primary $NP ducts, repaired $NO ducts, expected $EXP"
    if [ "$NP" -ne "$NO" ] || [ "$NO" -ne "$EXP" ]; then
        echo "  ERROR: retention differs. Duct assignment does not depend on"
        echo "         pseudotime, so this means the runs used different inputs."
        exit 1
    fi
done
echo "  retention gate PASSED"

echo ""
echo "--- STAGE 2: section comparison on the repaired per-duct tables -----------"
python -m cancer_trajectory_atlas.analysis.holeyness_section_comparison \
    --sections      2M-1 2M-2 \
    --per-duct-csvs "$OUT_1/holeyness_per_duct.csv" "$OUT_2/holeyness_per_duct.csv" \
    --output-dir    "$OUT_CMP" \
    --n-boot        "$N_BOOT" \
    --n-perm        "$N_PERM"

echo ""
echo "--- STAGE 3: side-by-side sensitivity report ------------------------------"
python -m cancer_trajectory_atlas.analysis.holeyness_repaired_sensitivity \
    --sections               2M-1 2M-2 \
    --primary-json           "$PRIM_CMP/holeyness_section_comparison.json" \
    --sensitivity-json       "$OUT_CMP/holeyness_section_comparison.json" \
    --primary-per-duct-csvs  "$PRIM_1/holeyness_per_duct.csv" \
                             "$PRIM_2/holeyness_per_duct.csv" \
    --repaired-per-duct-csvs "$OUT_1/holeyness_per_duct.csv" \
                             "$OUT_2/holeyness_per_duct.csv" \
    --anchor-axis-jsons      "$AXES/atlas_2M-1/anchor_axis.json" \
                             "$AXES/atlas_2M-2/anchor_axis.json" \
    --baseline-section       2M-1 \
    --output-dir             "$OUT_REPORT" \
    "${GATE_ARGS[@]+"${GATE_ARGS[@]}"}"

echo ""
echo "============================================================================"
echo "  COMPLETE"
echo "============================================================================"
echo "  $OUT_REPORT/holeyness_repaired_sensitivity.md    <- read this"
echo "  $OUT_REPORT/holeyness_repaired_sensitivity.json"
echo "  $OUT_1/  $OUT_2/                    repaired-axis per-duct tables"
echo "  $OUT_CMP/                           repaired-axis section comparison"
echo ""
echo "  PRIMARY outputs untouched:"
echo "    $PRIM_1  $PRIM_2  $PRIM_CMP"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. The negative control. If it says CONFOUNDED, the 2M-1 arm changed the"
echo "      baseline tree as well as the repair and nothing below is clean."
echo "   2. The three predictions, held or not held. A failed prediction is the"
echo "      result; do not reframe the hypothesis around what the data shows."
echo "   3. The side-by-side tables. The PRIMARY column is the headline; the"
echo "      repaired column is a sensitivity check and is never a replacement."
