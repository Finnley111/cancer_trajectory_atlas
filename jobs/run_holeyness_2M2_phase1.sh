#!/bin/bash
# PHASE 1 — holey-ness validation for section 2M-2 (PFA-fixed).
#
# Runs, in order:
#   0. GeoJSON -> TSV conversion   (analysis/holeyness_geojson_export.py, NEW)
#   1. holeyness.py v1            -> $SCRATCH/results/holeyness/2M-2/
#   2. holeyness.py --v2          -> $SCRATCH/results/holeyness/2M-2/v2_area_adjusted/
#
# analysis/holeyness.py is NOT modified. 2M-2 flows through byte-identical
# analysis code to 2M-1 — that is the whole point of converting the input rather
# than teaching holeyness.py a second format.
#
# ── WHY TWO holeyness.py RUNS ────────────────────────────────────────────────
#   --v2 requires --v1-per-duct-csv (holeyness.py errors without it) and 2M-2 has
#   no v1 output yet. Same sequence 2M-1 went through.
#
# ── ⚠ COLUMN RENAME ─────────────────────────────────────────────────────────
#   2M-2's measurements are prefixed 'holes_pfa:'; holeyness.py hardcodes
#   'holes_carnoys:'. The converter renames the HEADER ONLY — values untouched —
#   and records the true source keys in a .provenance.json sidecar. The emitted
#   column name therefore misstates the fixative. Do not read the header as
#   evidence about how 2M-2 was fixed.
#
#   Carnoy's (2M-1) and PFA (2M-2) differ in shrinkage, so hole % distributions
#   may differ between sections for FIXATION reasons alone, independent of
#   biology. Irrelevant within a section; a live confound for any cross-section
#   claim in Phase 2.
#
# ── ESTIMAND ────────────────────────────────────────────────────────────────
#   Two exclusions apply BEFORE the zero-patch exclusion you already know about:
#     * 23/1776 (1.3%) Tumor ducts are MultiPolygon; load_duct_polygons accepts
#       Polygon only and counts them as 'bad geometry'. (2M-1: 12/2242 = 0.5%.)
#     * 4/1776 have a non-numeric hole % ("NaN" written by QuPath for sub-micron
#       polygons) and are dropped by build_duct_table's notna filter.
#   Plus 7 ducts below 100 um^2 — degenerate annotation artifacts, all in
#   6027-4L-2M-2 — which cannot receive a patch centre and fall out at the
#   zero-patch stage. The analysis population is therefore "single-polygon Tumor
#   ducts with a numeric hole % and >=1 assigned patch".
#
# ── PSEUDOTIME SOURCE ───────────────────────────────────────────────────────
#   per_section_v2 (as requested). The 2M-1 reference values were computed
#   against per_section (baseline). Pseudotime is bit-identical between the two
#   trees, so rho(pt, hole_pct) is EXACTLY comparable; only
#   rho(hole_pct, packing_irregularity) and the area+nuclear-density partial rest
#   on a different feature version. Footnoted in the report, not hidden.
#
# READS (READ-ONLY): holeyness_section_2/, data/annotations_ratio/,
#                    per_section_v2/atlas_2M-2/results.csv, MCF7_x5_cropped/
# WRITES (NEW ONLY): $SCRATCH/data/holeyness/2M-2_converted/
#                    $SCRATCH/results/holeyness/2M-2/
#   holeyness/2M-1/, per_section/, per_section_v2/ are NEVER written.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyness_2M2_phase1.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=holeyness_2M2_p1
#SBATCH --output=logs/holeyness_2M2_p1-%j.out

set -euo pipefail
mkdir -p logs

SECTION="2M-2"
REPO="$HOME/cancer_trajectory_atlas"

GEOJSON_DIR="$REPO/holeyness_section_2"
CONVERT_DIR="$SCRATCH/data/holeyness/2M-2_converted"
EXPORT_TSV="$CONVERT_DIR/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"

ANNOTATION_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMENSIONS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
RESULTS_CSV="$SCRATCH/results/per_section_v2/atlas_2M-2/results.csv"
SLIDE_LIST="$REPO/jobs/slides_section2.txt"

V1_DIR="$SCRATCH/results/holeyness/2M-2"
V2_DIR="$V1_DIR/v2_area_adjusted"
V1_PER_DUCT_CSV="$V1_DIR/holeyness_per_duct.csv"

# PROTECTED — must never be written by this job.
PROTECTED=(
    "$SCRATCH/results/holeyness/2M-1"
    "$SCRATCH/results/per_section"
    "$SCRATCH/results/per_section_v2"
)

echo "============================================================================"
echo "  PHASE 1 — holey-ness validation, section ${SECTION} (PFA-fixed)"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  GeoJSON in : $GEOJSON_DIR"
echo "  Converted  : $EXPORT_TSV"
echo "  Pseudotime : $RESULTS_CSV   (per_section_v2, as requested)"
echo "  v1 out     : $V1_DIR   (NEW)"
echo "  v2 out     : $V2_DIR   (NEW)"
echo "============================================================================"

# ── Guards ──────────────────────────────────────────────────────────────────
for p in "${PROTECTED[@]}"; do
    for out in "$V1_DIR" "$V2_DIR" "$CONVERT_DIR"; do
        case "$out" in
            "$p"|"$p"/*)
                echo "ERROR: output '$out' is inside protected tree '$p'. Refusing."
                exit 1;;
        esac
    done
done
for d in "$V1_DIR" "$V2_DIR"; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "ERROR: $d exists and is non-empty. Refusing to overwrite an existing"
        echo "       holeyness result. Move it aside or pick a new path."
        exit 1
    fi
done

echo ""
echo "=== Pre-run checks ==="
MISSING=0
for p in "$GEOJSON_DIR" "$ANNOTATION_DIR" "$SLIDE_DIMENSIONS" "$RESULTS_CSV" "$SLIDE_LIST"; do
    echo -n "  $p : "
    if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
N_GEO=$(find "$GEOJSON_DIR" -maxdepth 1 -name '*.geojson' 2>/dev/null | wc -l)
echo "  GeoJSON files: $N_GEO (expect 8)"
[ "$N_GEO" -eq 8 ] || { echo "  WARNING: expected 8 slides for section 2M-2"; }
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }
echo "======================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
cd ~

# ── Step 0: convert GeoJSON -> TSV ──────────────────────────────────────────
echo ""
echo "=== Step 0/2 — GeoJSON to TSV ==="
mkdir -p "$CONVERT_DIR"
if [ -f "$EXPORT_TSV" ]; then
    echo "  Converted TSV already exists; reusing it (converter refuses to overwrite)."
else
    python -m cancer_trajectory_atlas.analysis.holeyness_geojson_export \
        --geojson-dir "$GEOJSON_DIR" \
        --output      "$EXPORT_TSV"
fi

# ── Step 1: v1 ──────────────────────────────────────────────────────────────
echo ""
echo "=== Step 1/2 — holeyness.py v1 ==="
mkdir -p "$V1_DIR"
python -m cancer_trajectory_atlas.analysis.holeyness \
    --section          "$SECTION" \
    --export           "$EXPORT_TSV" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --results          "$RESULTS_CSV" \
    --output-dir       "$V1_DIR" \
    --slide-list       "$SLIDE_LIST"

[ -f "$V1_PER_DUCT_CSV" ] || {
    echo "ERROR: v1 did not produce $V1_PER_DUCT_CSV; cannot run v2."; exit 1; }

# ── Step 2: v2 ──────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2/2 — holeyness.py --v2 (area-adjusted) ==="
mkdir -p "$V2_DIR"
python -m cancer_trajectory_atlas.analysis.holeyness \
    --section          "$SECTION" \
    --export           "$EXPORT_TSV" \
    --annotation-dir   "$ANNOTATION_DIR" \
    --slide-dimensions "$SLIDE_DIMENSIONS" \
    --results          "$RESULTS_CSV" \
    --output-dir       "$V2_DIR" \
    --slide-list       "$SLIDE_LIST" \
    --v2 \
    --v1-per-duct-csv  "$V1_PER_DUCT_CSV" \
    --n-permutations   1000

echo ""
echo "============================================================================"
echo "  PHASE 1 COMPLETE"
echo "============================================================================"
echo "  $V1_DIR/holeyness_validation.json"
echo "  $V2_DIR/holeyness_validation_v2.md      <- read this"
echo "  $V2_DIR/holeyness_validation_v2.json"
echo "  ${EXPORT_TSV}.provenance.json           <- column rename + data quality"
echo ""
echo "  Then run the side-by-side vs 2M-1:"
echo "    python -m cancer_trajectory_atlas.analysis.holeyness_section_compare \\"
echo "        --v2-json-2M2 $V2_DIR/holeyness_validation_v2.json \\"
echo "        --output      $SCRATCH/results/holeyness/2M-2/SECTION_COMPARISON.md"
echo ""
echo "  REMINDER: raw rho is the PRIMARY estimate. Duct area is a MEDIATOR of"
echo "  progression per the pathologist, so the area-adjusted value is a"
echo "  deliberately OVER-ADJUSTED sensitivity analysis, not a correction."
