#!/bin/bash
# Persist an alternative-anchor pseudotime as a run dir other tools read.
#
#   ANCHOR=area_stratified        (default) re-anchors the HOLEYROOT run
#   ANCHOR=area_matched_surrogate its size-matched control
#   ANCHOR=v2_repaired            repairs the DENSITY-rooted v2 run
#
# The source tree follows the anchor automatically; override with SOURCE_BASE.
#
# WHY v2_repaired
# ---------------
# The holey-ness validation can only be an EXTERNAL check on an axis that was not
# anchored on holey-ness -- on the holeyroot and area-stratified axes
# rho(pt, hole_pct) is circular by construction. So that validation has to live
# on the density-rooted axis, which is exactly the axis whose 2M-2 roots are
# degenerate: 20 patches at nuclear_density exactly 0.0, none inside any Tumor
# annotation, three ordering the manifold backwards, pseudotime_std at 27.7% of
# range against 5.0% in 2M-1.
#
# v2_repaired therefore REPAIRS rather than replaces: v2's own roots minus the
# discordant ones, so the anchor stays nuclear_density and hole_pct stays
# external. Gated against anchor_area_control Task E's recorded repair (drop
# count and resulting spread) rather than Task C's correlations.
#
# TWO CAVEATS. The drop rule was fixed in advance but applied AFTER the
# discordance was observed, so results on that axis are a sensitivity analysis
# unless pre-declared primary. And the rule keeps whichever orientation MOST
# roots share -- it removes minority disagreement, and cannot tell a majority of
# bad roots from a majority of good ones.
#
# WHY area_stratified
# -------------------
# anchor_area_control showed the holeyroot anchor is duct-size-extreme (20/20
# root ducts below the eligible median in both sections) and that its
# rho(pt, duct area) is entirely reproduced by size-matched anchors that know
# nothing about hole %. Its Task C built the fix — lowest hole % WITHIN each area
# stratum — which drove rho(pt, area) back to the random-anchor baseline
# (+0.2664 in 2M-1, +0.0205 in 2M-2) while leaving rho(pt, hole_pct) intact.
#
# But anchor_area_control DISCARDS that pseudotime vector, keeping only summary
# correlations. So eccentricity_check, eccentricity_within_slide and
# holeyroot_duct_checks can only ever see the production axis.
#
# That gap has a specific cost. eccentricity_check returned DIRECTIONAL IN
# MORPHOLOGY for the holeyroot axis, counting nuclear_density among 2M-2's
# directional, within-slide-surviving features. anchor_area_control found that
# same relationship to be a duct-size artifact of the anchor: under the
# area-stratified anchor it goes to -0.244 at duct level. eccentricity_check
# cannot test that — it reads whatever pseudotime it is handed. Re-running it
# against the area-stratified axis is the direct test, and needs that axis on
# disk.
#
# THE CONSISTENCY GATE. The area-stratified rule is deterministic, so the rebuild
# must reproduce Task C exactly. --expect-json points at anchor_area_control.json
# and the module REFUSES if the duct-level correlations differ beyond
# --tolerance. Without it, a silent divergence would hand every downstream tool a
# different axis under the same name.
#
# Cost is ~20 sc.tl.dpt calls per section on the frozen graph — minutes, because
# none of anchor_area_control's 50 null draws are rebuilt.
#
# AFTERWARDS:
#   RUN_BASE=$SCRATCH/results/holeyroot_experiment/anchor_axes/area_stratified \
#   OUT_SUFFIX=_area_stratified sbatch jobs/run_eccentricity_check.sh
#   (same two env vars work for jobs/run_eccentricity_within_slide.sh)
#   Ignore eccentricity_check's Task 0 on those dirs: it reconstructs roots as
#   argsort(nuclear_density)[:20] and will profile the wrong root set, exactly as
#   it does for a holeyroot run. Read uns['anchor_provenance'] instead.
#
# READS (READ-ONLY): per_section_holeyroot/, the exports, the ratio annotations,
#                    slide_dimensions.json, anchor_area_control.json
# WRITES (NEW ONLY): $SCRATCH/results/holeyroot_experiment/anchor_axes/<anchor>
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_export_anchor_axis.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=export_axis
#SBATCH --output=logs/export_axis-%j.out

set -euo pipefail
mkdir -p logs
REPO="$HOME/cancer_trajectory_atlas"

HR_BASE="$SCRATCH/results/per_section_holeyroot"
V2_BASE="$SCRATCH/results/per_section_v2"
ANCHOR="${ANCHOR:-area_stratified}"

# The source tree depends on the anchor. area_stratified and
# area_matched_surrogate re-anchor the HOLEYROOT run; v2_repaired instead repairs
# the DENSITY-rooted v2 run, because that is the only axis on which hole_pct is
# still an EXTERNAL validator -- on any holeyness-anchored axis the holeyness
# validation is circular by construction.
case "$ANCHOR" in
    v2_repaired) SRC_BASE="${SOURCE_BASE:-$V2_BASE}" ;;
    *)           SRC_BASE="${SOURCE_BASE:-$HR_BASE}" ;;
esac
OUT_DIR="$SCRATCH/results/holeyroot_experiment/anchor_axes/$ANCHOR"
EXPECT_JSON="$SCRATCH/results/holeyroot_experiment/anchor_area_control/anchor_area_control.json"

ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"
S1="$REPO/jobs/slides_section1.txt"
S2="$REPO/jobs/slides_section2.txt"

echo "============================================================================"
echo "  Export anchor axis : $ANCHOR"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Source : $SRC_BASE"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$HR_BASE"|"$HR_BASE"/*|"$V2_BASE"|"$V2_BASE"/*|"$SCRATCH/results/per_section"/*)
        echo "ERROR: output is inside a protected run tree."; exit 1;;
esac

if [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUT_DIR already exists and is not empty."
    echo "       A derived axis must not be silently replaced — downstream results"
    echo "       would then refer to an axis that no longer exists. FORCE=1 to overwrite."
    exit 1
fi

MISSING=0
for p in "$SRC_BASE/atlas_2M-1/adata_full.h5ad" "$SRC_BASE/atlas_2M-2/adata_full.h5ad" \
         "$SRC_BASE/atlas_2M-1/results.csv" "$SRC_BASE/atlas_2M-2/results.csv" \
         "$EXPORT_2M1" "$EXPORT_2M2" "$ANN_DIR" "$SLIDE_DIMS" "$S1" "$S2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }

# The gate is optional in the module but required here: exporting an unverified
# axis is the one failure mode that would corrupt every downstream analysis.
GATE=()
echo -n "  $EXPECT_JSON : "
if [ -e "$EXPECT_JSON" ]; then
    echo "ok"; GATE=(--expect-json "$EXPECT_JSON")
elif [ "${ALLOW_UNGATED:-0}" = "1" ]; then
    echo "NOT FOUND — proceeding UNGATED because ALLOW_UNGATED=1"
else
    echo "NOT FOUND"
    echo "ERROR: anchor_area_control.json is required so the rebuilt axis can be"
    echo "       checked against the Task C numbers it must reproduce. Run"
    echo "       jobs/run_anchor_area_control.sh first, or set ALLOW_UNGATED=1 to"
    echo "       export without that check and say so wherever the axis is used."
    exit 1
fi

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.export_anchor_axis \
    --sections         2M-1 2M-2 \
    --run-dirs         "$SRC_BASE/atlas_2M-1" "$SRC_BASE/atlas_2M-2" \
    --exports          "$EXPORT_2M1"         "$EXPORT_2M2" \
    --slide-lists      "$S1"                 "$S2" \
    --annotation-dir   "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --output-dir       "$OUT_DIR" \
    --anchor           "$ANCHOR" \
    "${GATE[@]+"${GATE[@]}"}"

echo ""
echo "============================================================================"
echo "  EXPORT COMPLETE"
echo "============================================================================"
ls -1 "$OUT_DIR"/atlas_*/ 2>/dev/null
echo ""
if [ "$ANCHOR" = "v2_repaired" ]; then
    echo "  This axis keeps the DENSITY anchor, so hole_pct is still an EXTERNAL"
    echo "  validator on it. The point of exporting it is to re-run the holey-ness"
    echo "  validation on an axis that is both sound and independently validatable:"
    echo ""
    echo "    python -m cancer_trajectory_atlas.analysis.holeyness \\"
    echo "        --section 2M-2 --results-csv $OUT_DIR/atlas_2M-2/results.csv \\"
    echo "        --output-dir \$SCRATCH/results/holeyness/2M-2_repaired  ..."
    echo ""
    echo "  then re-run jobs/run_holeyness_section_comparison.sh against the new"
    echo "  per-duct tables."
    echo ""
    echo "  DECIDE BEFORE RUNNING, not after: is the repaired-axis result your"
    echo "  PRIMARY analysis or a SENSITIVITY check? The drop rule was fixed in"
    echo "  advance but applied after the discordance was seen, and this is a"
    echo "  second look at a question already answered on the unrepaired axis."
    echo "  Read anchor_axis.json -> anchor_rule.post_hoc_caveat."
    echo ""
    echo "  Also check anchor_rule.identical_to_source. In 2M-1 no root was"
    echo "  discordant, so the repaired axis there IS the source axis; only 2M-2"
    echo "  actually changes."
else
    echo "  Next, run the trajectory tests against this axis:"
    echo "    RUN_BASE=$OUT_DIR OUT_SUFFIX=_${ANCHOR} sbatch jobs/run_eccentricity_check.sh"
    echo "    RUN_BASE=$OUT_DIR OUT_SUFFIX=_${ANCHOR} sbatch jobs/run_eccentricity_within_slide.sh"
    echo ""
    echo "  If DIRECTIONAL IN MORPHOLOGY does NOT survive on this axis, the trajectory"
    echo "  verdict was an artifact of a duct-size-extreme anchor. If it does survive,"
    echo "  the verdict is anchor-robust — which still says nothing about whether the"
    echo "  anchor is correct."
fi
