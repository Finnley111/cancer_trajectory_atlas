#!/bin/bash
# PHASE 3 companion — nesting, uncertainty, and what "hole %" actually measures.
#
# Phase 2 reported every duct-level correlation as one pooled point estimate over
# ~1,400-1,600 ducts from 8 slides, with no interval and no nesting check.
#
#   TASK 1  WITHIN-SLIDE vs BETWEEN-SLIDE. The v1-v3 holey-ness work already found
#           per-slide partials spanning -0.069 to +0.30 on this cohort. 2M-2's
#           headline move — rho(pt, area) from -0.084 to +0.249 — could be a shift
#           in slide MEDIANS rather than anything happening inside a slide.
#
#   TASK 2  CLUSTER BOOTSTRAP. With 8 slides the resampling unit is the SLIDE.
#           A duct-level bootstrap would treat 1,360 nested ducts as independent;
#           on synthetic data with the same structure it gave an interval ~80x
#           too narrow. Expect wide intervals. That width is the design, not the
#           method.
#
#   TASK 3  WHAT THE ANNOTATION MEASURES. The root sheets show the hand-annotated
#           LEAST holey ducts contain the WHITEST patches (frac_pixels_white
#           median ~0.28 for holeyroot roots vs ~0.15 for v2's), while
#           h_intensity_wholepatch — which white space mechanically depresses —
#           moved 0.039 -> 0.323 in 2M-2 and is counted as one of three features
#           now agreeing across sections. Both cannot be comfortable at once.
#           Also prints each section's hole % distribution: 2M-1's roots sit at
#           median 0.025, 2M-2's at 1.80 with a P10 threshold of 7.63, from
#           different export files. "Lowest holey-ness" is not the same anchor in
#           the two sections.
#
# CPU only, no h5ad, no scanpy — reads results.csv and the annotations. Cheap.
#
# READS (READ-ONLY): per_section_v2/, per_section_holeyroot/, the exports, the
#                    ratio annotations, slide_dimensions.json
# WRITES (NEW ONLY): $SCRATCH/results/holeyroot_experiment/duct_checks
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_holeyroot_duct_checks.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=duct_checks
#SBATCH --output=logs/duct_checks-%j.out

set -euo pipefail
mkdir -p logs
REPO="$HOME/cancer_trajectory_atlas"

V2_BASE="$SCRATCH/results/per_section_v2"
HR_BASE="$SCRATCH/results/per_section_holeyroot"
OUT_DIR="$SCRATCH/results/holeyroot_experiment/duct_checks"

ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"
S1="$REPO/jobs/slides_section1.txt"
S2="$REPO/jobs/slides_section2.txt"

N_BOOT="${N_BOOT:-2000}"

echo "============================================================================"
echo "  PHASE 3 companion — duct-level nesting, CIs, annotation-vs-optics"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Boot    : $N_BOOT cluster-bootstrap draws (slide is the resampling unit)"
echo "  Output  : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$V2_BASE"|"$V2_BASE"/*|"$HR_BASE"|"$HR_BASE"/*|"$SCRATCH/results/per_section"/*)
        echo "ERROR: output is inside a protected run tree."; exit 1;;
esac

MISSING=0
for p in "$V2_BASE/atlas_2M-1/results.csv" "$V2_BASE/atlas_2M-2/results.csv" \
         "$HR_BASE/atlas_2M-1/results.csv" "$HR_BASE/atlas_2M-2/results.csv" \
         "$EXPORT_2M1" "$EXPORT_2M2" "$ANN_DIR" "$SLIDE_DIMS" "$S1" "$S2"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs. Phase 2 must have run first."; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

# NOTE the ORDER: exports and slide lists are positional and paired to labels.
# 2M-1 reads the original TSV, 2M-2 the column-renamed GeoJSON-derived export.
python -m cancer_trajectory_atlas.analysis.holeyroot_duct_checks \
    --labels        v2_2M-1 holeyroot_2M-1 v2_2M-2 holeyroot_2M-2 \
    --results-csvs  "$V2_BASE/atlas_2M-1/results.csv" \
                    "$HR_BASE/atlas_2M-1/results.csv" \
                    "$V2_BASE/atlas_2M-2/results.csv" \
                    "$HR_BASE/atlas_2M-2/results.csv" \
    --exports       "$EXPORT_2M1" "$EXPORT_2M1" "$EXPORT_2M2" "$EXPORT_2M2" \
    --slide-lists   "$S1" "$S1" "$S2" "$S2" \
    --annotation-dir   "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --output-dir       "$OUT_DIR" \
    --n-boot           "$N_BOOT"

echo ""
echo "============================================================================"
echo "  DUCT CHECKS COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/holeyroot_duct_checks.md    <- read this"
echo "  $OUT_DIR/holeyroot_duct_checks.json"
echo ""
echo "  The three numbers that matter, for holeyroot_2M-2 rho(pt, area) = +0.249:"
echo "   1. task1 within_slide_summary.pt_area.n_same_sign_as_pooled — if only a"
echo "      minority of the 8 slides carry the positive sign, +0.249 is a"
echo "      slide-level effect, not a duct-level one."
echo "   2. task2 pt_area.ci95 — the first interval ever put on this number."
echo "   3. task3 rho_hole_vs.h_intensity_wholepatch — how much of the anchor"
echo "      variable the pipeline's own pixels already see, and therefore how"
echo "      independent that feature is as a validator."
