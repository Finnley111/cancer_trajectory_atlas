#!/bin/bash
# PHASE 3 — is the holey-ness anchor's rho(pt, duct AREA) a DUCT-SIZE artifact?
#
# Phase 2 called rho(pt, duct area) the sharpest NON-CIRCULAR discriminator
# between anchors, and reported it moving -0.084 -> +0.249 in 2M-2.
# holeyness_roots.json says that test is not non-circular:
#
#     root duct area, median      :  7,763 um^2
#     ALL eligible ducts, median  : 33,807 um^2
#     largest root duct           : 30,822 um^2   <- ALL 20 below the median
#
# Under a size-blind rule, 20/20 below the median has probability ~1e-6. This job
# asks how much of the +0.249 an anchor that knows NOTHING about hole %, but sits
# in ducts of the same SIZE, reproduces on its own.
#
# It re-runs ONLY sc.tl.dpt on each run's FROZEN graph and diffusion map, reusing
# root_sensitivity.py's build_dpt_adata / run_multi_root_dpt unmodified, so the
# aggregation is identical to production's compute_dpt_multi_root. Nothing is
# re-embedded and no run tree is written to.
#
# TASKS
#   A  where the 20 root ducts sit in the eligible area distribution
#   B  area-matched surrogate anchors (n draws)   <- the decisive control
#   C  lowest hole % WITHIN area strata           <- the complement
#   D  uniform random duct anchors (n draws)      <- reference null
#   E  v2 root repair: drop roots that order the manifold backwards vs their
#      peers, then re-test rho(v2, holeyroot) against the 0.78-0.89 band
#   F  eccentricity of every anchor, vs run 4's 0.808 / 0.802
#
# READS (READ-ONLY): per_section_v2/, per_section_holeyroot/, the exports, the
#                    ratio annotations, slide_dimensions.json
# WRITES (NEW ONLY): $SCRATCH/results/holeyroot_experiment/anchor_area_control
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_anchor_area_control.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=anchor_area
#SBATCH --output=logs/anchor_area-%j.out

set -euo pipefail
mkdir -p logs
REPO="$HOME/cancer_trajectory_atlas"

V2_BASE="$SCRATCH/results/per_section_v2"
HR_BASE="$SCRATCH/results/per_section_holeyroot"
OUT_DIR="$SCRATCH/results/holeyroot_experiment/anchor_area_control"

ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"

N_DRAWS="${N_DRAWS:-25}"
SEED="${SEED:-0}"

echo "============================================================================"
echo "  PHASE 3 — anchor area control"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Draws   : $N_DRAWS per null (Tasks B and D), seed $SEED"
echo "  Output  : $OUT_DIR   (NEW)"
echo "============================================================================"

case "$OUT_DIR" in
    "$V2_BASE"|"$V2_BASE"/*|"$HR_BASE"|"$HR_BASE"/*|"$SCRATCH/results/per_section"/*)
        echo "ERROR: output is inside a protected run tree."; exit 1;;
esac

MISSING=0
for p in "$V2_BASE/atlas_2M-1/adata_full.h5ad" "$V2_BASE/atlas_2M-2/adata_full.h5ad" \
         "$HR_BASE/atlas_2M-1/adata_full.h5ad" "$HR_BASE/atlas_2M-2/adata_full.h5ad" \
         "$EXPORT_2M1" "$EXPORT_2M2" "$ANN_DIR" "$SLIDE_DIMS" \
         "$REPO/jobs/slides_section1.txt" "$REPO/jobs/slides_section2.txt"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs. Phase 2 must have run first."; exit 1; }

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.analysis.anchor_area_control \
    --sections         2M-1 2M-2 \
    --holeyroot-dirs   "$HR_BASE/atlas_2M-1"  "$HR_BASE/atlas_2M-2" \
    --v2-dirs          "$V2_BASE/atlas_2M-1"  "$V2_BASE/atlas_2M-2" \
    --exports          "$EXPORT_2M1"          "$EXPORT_2M2" \
    --slide-lists      "$REPO/jobs/slides_section1.txt" "$REPO/jobs/slides_section2.txt" \
    --annotation-dir   "$ANN_DIR" \
    --slide-dimensions "$SLIDE_DIMS" \
    --output-dir       "$OUT_DIR" \
    --n-draws          "$N_DRAWS" \
    --seed             "$SEED"

echo ""
echo "============================================================================"
echo "  ANCHOR AREA CONTROL COMPLETE"
echo "============================================================================"
echo "  $OUT_DIR/anchor_area_control.md    <- read this"
echo "  $OUT_DIR/anchor_area_control.json"
echo ""
echo "  READ IN THIS ORDER:"
echo "   1. consistency_rerun_vs_stored_rho must be >= 0.999. If it is not, the"
echo "      frozen-graph re-run does not reproduce production's axis and every"
echo "      control below is measured against a different DPT. Stop there."
echo "   2. task_b_summary.rho_pt_area.observed_inside_null_range. TRUE means an"
echo "      anchor that knows nothing about hole % reaches the same value, and"
echo "      Phase 2's discriminator is uninformative."
echo "   3. task_c rho_pt_hole_pct vs rho_pt_area. Hole surviving while area"
echo "      collapses is the only pattern in which holey-ness is doing work that"
echo "      duct size is not."
echo "   4. task_e repaired.reenters_random_root_band. TRUE means Phase 2's"
echo "      sub-floor rho of 0.7105 was v2's defect, not the new anchor's."
echo ""
echo "  NONE of these outcomes makes an anchor correct. The candidate pool still"
echo "  excludes every duct with no assigned patch — 22.2% of 2M-2's ducts,"
echo "  systematically the smallest and least holey."
