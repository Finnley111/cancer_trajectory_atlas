#!/bin/bash
# PHASE 2 report — holey-ness-rooted pseudotime vs v2, both sections.
#
# REPORTS NUMBERS. DECLARES NEITHER ANCHOR BETTER.
#
# ⚠ rho(pseudotime, hole_pct) rising is PARTLY CIRCULAR — the anchor IS
#   holey-ness — and is not evidence. The non-circular tests, neither used in
#   root selection, are rho(pt, duct AREA) and rho(pt, nuclear_density). Both are
#   recomputed at duct level against the NEW pseudotime by reusing holeyness.py's
#   own aggregation, so Phase 1's numbers and these come from identical code.
#
# ⚠ The two sections read DIFFERENT export files (2M-1 the original TSV, 2M-2 the
#   converted GeoJSON export whose header was renamed). Order matters below.
#
# READ-ONLY on per_section_v2/ and per_section_holeyroot/. Writes only its own
# output directory. v2's cellularity-confound verdicts are READ, never
# regenerated — analyze_run_nuclear_density writes into the tree it analyses.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyroot_compare.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=holeyroot_cmp
#SBATCH --output=logs/holeyroot_cmp-%j.out

set -euo pipefail
mkdir -p logs
REPO="$HOME/cancer_trajectory_atlas"

V2_BASE="$SCRATCH/results/per_section_v2"
HR_BASE="$SCRATCH/results/per_section_holeyroot"
OUT_DIR="$SCRATCH/results/holeyroot_experiment/compare"

ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"

echo "============================================================================"
echo "  PHASE 2 comparison — holeyroot vs v2, both sections"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Output : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$V2_BASE"|"$V2_BASE"/*|"$SCRATCH/results/per_section"/*)
        echo "ERROR: output is inside a protected tree."; exit 1;;
esac

MISSING=0
for p in "$V2_BASE/atlas_2M-1/results.csv" "$V2_BASE/atlas_2M-2/results.csv" \
         "$HR_BASE/atlas_2M-1/results.csv" "$HR_BASE/atlas_2M-2/results.csv" \
         "$EXPORT_2M1" "$EXPORT_2M2" "$ANN_DIR" "$SLIDE_DIMS"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs. Run Phase 2 first."; exit 1; }

LOGS=()
for g in "$REPO/logs/per_section_v2-"*.out "$REPO/logs/holeyroot-"*.out; do
    [ -e "$g" ] && LOGS+=("$g")
done
if [ "${#LOGS[@]}" -eq 0 ]; then
    echo ""
    echo "  NOTE: no SLURM logs matched. The raw pre-normalisation pseudotime range"
    echo "        is printed but never persisted, so 'pseudotime_std as % of range'"
    echo "        will read UNRECOVERABLE rather than being invented. Intended."
else
    echo "  ${#LOGS[@]} log file(s) available for raw-range recovery."
fi

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.holeyroot_compare \
    --sections        2M-1 2M-2 \
    --v2-dirs         "$V2_BASE/atlas_2M-1"  "$V2_BASE/atlas_2M-2" \
    --holeyroot-dirs  "$HR_BASE/atlas_2M-1"  "$HR_BASE/atlas_2M-2" \
    --exports         "$EXPORT_2M1"          "$EXPORT_2M2" \
    --slide-lists     "$REPO/jobs/slides_section1.txt" "$REPO/jobs/slides_section2.txt" \
    --annotation-dir  "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --output-dir      "$OUT_DIR" \
    --patch-size      112 \
    ${LOGS[@]+--logs "${LOGS[@]}"}

echo ""
echo "============================================================================"
echo "  COMPARISON COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/holeyroot_comparison.md    <- read this"
echo "  $OUT_DIR/holeyroot_comparison.json"
echo ""
echo "  Read the DUCT-LEVEL block first: rho(pt, duct area) and"
echo "  rho(pt, nuclear_density) are the non-circular tests. rho(pt, hole_pct)"
echo "  is circular by construction and proves nothing about the anchor."
