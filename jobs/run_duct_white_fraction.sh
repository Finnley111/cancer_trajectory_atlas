#!/bin/bash
# Does the expert hole % annotation actually track white space in the duct?
#
# WHY
# ---
# The entire holey-ness anchor rests on hole_pct, a QuPath hand annotation.
# holeyroot_duct_checks Task 3 tested it against the pipeline's own pixels and
# the two sections disagreed in SIGN on three of four optical features:
#
#     rho(hole %, ...)          2M-1      2M-2
#     h_intensity_wholepatch   +0.080    -0.271
#     h_intensity              +0.202    -0.098
#     nuclear_density          +0.051    -0.120
#     texture_entropy          +0.377    +0.335
#
# Holes are white and white depresses haematoxylin, so 2M-2 has the physically
# expected sign and 2M-1 has the wrong one. 2M-1 is also the section whose
# trajectory verdict collapses to ECCENTRICITY IN EMBEDDING ONLY once the
# anchor's duct-size extremity is removed.
#
# But Task 3 measured through a PROXY and through the patch-to-duct assignment,
# which drops 571/2173 ducts in 2M-1 and 389/1749 in 2M-2 (systematically the
# smallest) and measures 112px windows rather than ducts.
#
# This measures it DIRECTLY: every Tumor polygon rasterised against its slide
# PNG, white = mean RGB > 220 (WHITE_THRESH imported from inspect_roots_v3, not
# restated), no patch assignment, every duct included -- including the zero-patch
# ones no earlier analysis could see.
#
# WHAT THE OUTCOMES MEAN
#   HIGH in both sections -> the annotation has a real pixel referent; Task 3's
#     weak 2M-1 coupling was an artifact of patch aggregation.
#   LOW in 2M-1           -> the 2M-1 holes_carnoys column does not denote
#     optical holes, and the anchor is invalid there regardless of everything
#     else -- which would retire Phase 2's 2M-1 re-anchoring and Phase 3's 2M-1
#     Task C along with it.
#
# Cost is decoding 16 whole-slide PNGs, one at a time, same as the root-sheet
# job. Memory is sized for that, not for the correlations.
#
# READS (READ-ONLY): the ratio annotations, the measurement exports, the slide
#                    PNGs, and (optionally) results.csv for the Task B split
# WRITES (NEW ONLY): $SCRATCH/results/holeyness/duct_white_fraction
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_duct_white_fraction.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=duct_white
#SBATCH --output=logs/duct_white-%j.out

set -euo pipefail
mkdir -p logs
REPO="$HOME/cancer_trajectory_atlas"

HR_BASE="$SCRATCH/results/per_section_holeyroot"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
OUT_DIR="$SCRATCH/results/holeyness/duct_white_fraction"

ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$PNG_DIR/slide_dimensions.json"
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"
S1="$REPO/jobs/slides_section1.txt"
S2="$REPO/jobs/slides_section2.txt"

N_BOOT="${N_BOOT:-2000}"
N_PERM="${N_PERM:-2000}"

echo "============================================================================"
echo "  Duct white fraction vs annotated hole %"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  PNGs   : $PNG_DIR"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$HR_BASE"|"$HR_BASE"/*|"$SCRATCH/results/per_section"/*|"$SCRATCH/results/per_section_v2"/*)
        echo "ERROR: output is inside a protected run tree."; exit 1;;
esac

MISSING=0
for p in "$EXPORT_2M1" "$EXPORT_2M2" "$ANN_DIR" "$SLIDE_DIMS" "$PNG_DIR" "$S1" "$S2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }

# Every slide named in the two lists must have a decodable PNG. The module
# refuses on a missing image rather than skipping it, because a skipped slide
# would silently drop its ducts from the correlation; check here so the failure
# arrives in the preflight instead of an hour in.
NPNG=0
while read -r s; do
    [ -n "$s" ] || continue
    if ls "$PNG_DIR/$s".* >/dev/null 2>&1; then
        NPNG=$((NPNG + 1))
    else
        echo "  MISSING PNG for slide: $s"; MISSING=1
    fi
done < <(cat "$S1" "$S2")
echo "  slide PNGs found: $NPNG"
[ "$MISSING" -eq 0 ] || { echo "ERROR: a slide has no image."; exit 1; }

# results.csv is OPTIONAL and only enables Task B (the exclusion-bias split).
CSVS=()
if [ -f "$HR_BASE/atlas_2M-1/results.csv" ] && [ -f "$HR_BASE/atlas_2M-2/results.csv" ]; then
    CSVS=(--results-csvs "$HR_BASE/atlas_2M-1/results.csv" "$HR_BASE/atlas_2M-2/results.csv")
    echo "  results.csv found for both sections — Task B enabled."
else
    echo "  NOTE: results.csv not found for both sections. Task B (does patch"
    echo "        assignment explain the earlier weakness?) will be SKIPPED, not"
    echo "        approximated. Everything else runs."
fi

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.duct_white_fraction \
    --sections         2M-1 2M-2 \
    --exports          "$EXPORT_2M1" "$EXPORT_2M2" \
    --slide-lists      "$S1"         "$S2" \
    --annotation-dir   "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --png-dir          "$PNG_DIR" \
    --output-dir       "$OUT_DIR" \
    --n-boot           "$N_BOOT" \
    --n-perm           "$N_PERM" \
    "${CSVS[@]+"${CSVS[@]}"}"

echo ""
echo "============================================================================"
echo "  COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/duct_white_fraction.md    <- read this"
echo "  $OUT_DIR/duct_white_fraction.json"
echo "  $OUT_DIR/duct_white_fraction_2M-<n>.csv   (per-duct measurements)"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. The headline table. 2M-1 is the section at issue: a low rho there"
echo "      invalidates the anchor in the one section where the trajectory"
echo "      verdict already collapses under de-sizing."
echo "   2. task_b. If zero-patch ducts correlate like assigned ones, patch"
echo "      assignment does NOT explain Task 3's weakness."
echo "   3. task_d. A conclusion that flips across the threshold sweep is a"
echo "      conclusion about the threshold, not about the annotation."
echo "   4. task_e partial given area, since bigger ducts hold more lumen."
echo ""
echo "  A STRONG correlation does not make the anchor good: the realised root set"
echo "  is still the bottom 1-3% of ducts by hole % and duct-size-extreme, and its"
echo "  headline correlation is still reproduced by size-matched anchors that"
echo "  ignore hole % entirely."
