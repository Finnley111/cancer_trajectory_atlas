#!/bin/bash
# Eccentricity check — is the MorphPT pseudotime a trajectory, or an atypicality score?
#
# WHY:
#   jobs/run_root_sensitivity.sh found that the per-section pseudotime tracks
#   distance from the diffusion-map centroid (rho = 0.808 in 2M-1, 0.802 in 2M-2)
#   far better than position along DC1 (0.543 / 0.467), and that 25 uniformly
#   random 20-root sets reproduce the production pseudotime at |rho| 0.78-0.89.
#   The axis is therefore fixed by the manifold, not by the root rule — which
#   raises a bigger question than root choice: does it measure how far ALONG a
#   patch is, or how UNUSUAL it is? An eccentricity measure is DIRECTIONLESS
#   ('late' = atypical in any direction), and a trajectory framing would not
#   survive it.
#
# THE TAUTOLOGY THIS IS BUILT AROUND:
#   DPT pseudotime IS a diffusion distance from its roots, so a high correlation
#   with diffusion-space eccentricity is PARTLY TRUE BY CONSTRUCTION. That figure
#   is reported but labelled DEFINITIONAL and excluded from every verdict. The
#   informative tests are in PCA space (which the diffusion map was built FROM)
#   and in morphological-feature space (what the paper's claims are about) —
#   DPT is defined in terms of neither.
#
# TASK A — which geometry is the pseudotime?
#   rho(PT, distance from centroid) in diffusion space (definitional), PCA space
#   (informative) and morphology (decisive), plus DC1, DC1-eccentricity and local
#   graph sparsity. The decisive contrast is rho(PT, mean |z|) vs
#   rho(PT, mean signed z) across the six features: if the UNSIGNED deviation
#   tracks pseudotime and the SIGNED one does not, late patches are extreme in
#   inconsistent directions. Repeated within each slide, so the result cannot be
#   a between-slide batch effect.
#
# TASK B — is late pseudotime heterogeneous, and in one direction or many?
#   Heterogeneity alone does NOT separate eccentricity from a branching
#   trajectory — both give diverse late states. Direction does. Primary statistic:
#   bidirectional enrichment — among top-decile pseudotime patches, are BOTH the
#   high and low tails of a feature enriched (eccentricity) or only one
#   (trajectory)? Plus per-decile dispersion, a subclustering of the late patches
#   with signed feature profiles, and the slide composition of each end.
#
# READS (ALL READ-ONLY — per-section runs are never modified or re-run):
#   $SCRATCH/results/per_section/atlas_2M-1/adata_full.h5ad
#   $SCRATCH/results/per_section/atlas_2M-2/adata_full.h5ad
#
# WRITES (NEW directory only):
#   $SCRATCH/results/eccentricity/eccentricity_report.md
#   $SCRATCH/results/eccentricity/eccentricity_check.json
#   $SCRATCH/results/eccentricity/eccentricity_<section>.{png,pdf}
#
# CPU only, no GPU, no feature extraction, no DPT. Correlations on stored arrays
# plus one small KMeans silhouette sweep over the late patches — minutes, not hours.
#
# NOTE: like run_root_sensitivity.sh, this does NOT append to PROJECT_STATE.md.
# Add the working-log entry by hand after reading the report.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_eccentricity_check.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eccentricity
#SBATCH --output=logs/eccentricity-%j.out

set -euo pipefail

mkdir -p logs

# ── Constants ─────────────────────────────────────────────────────────────────
SECTIONS=("2M-1" "2M-2")
PER_SECTION_BASE="$SCRATCH/results/per_section"
RUN_DIRS=(
    "$PER_SECTION_BASE/atlas_2M-1"
    "$PER_SECTION_BASE/atlas_2M-2"
)
OUTPUT_DIR="$SCRATCH/results/eccentricity"
N_BINS=10
TAIL_FRACTION=0.10
K_MIN=2
K_MAX=8

echo "============================================================"
echo "  Eccentricity check — trajectory or atypicality score?"
echo "  Job ID        : ${SLURM_JOB_ID:-local}"
echo "  Sections      : ${SECTIONS[*]}"
echo "  PT bins       : $N_BINS"
echo "  Tail fraction : $TAIL_FRACTION  (defines 'early'/'late' and feature extremes)"
echo "  Subcluster k  : $K_MIN..$K_MAX  (chosen by silhouette)"
echo "  Output dir    : $OUTPUT_DIR  (NEW — existing runs untouched)"
echo "============================================================"

# ── Pre-run checks ────────────────────────────────────────────────────────────
echo ""
echo "=== Pre-run checks (all inputs read-only) ==="
MISSING=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo -n "  $RUN_DIR/adata_full.h5ad : "
    ls -lh "$RUN_DIR/adata_full.h5ad" 2>/dev/null \
        || { echo "NOT FOUND — run run_per_section.sh first"; MISSING=1; }
done

if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "ERROR: at least one adata_full.h5ad is missing. This analysis reuses"
    echo "       existing per-section runs and must not regenerate them."
    exit 1
fi
echo "============================================"
echo ""

# ── Environment ───────────────────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

python -c "import scanpy, anndata, sklearn; print('scanpy', scanpy.__version__, '/ anndata', anndata.__version__)" || {
    echo "ERROR: scanpy/anndata/sklearn not importable in ~/envs/atlas"; exit 1
}

mkdir -p "$OUTPUT_DIR"

cd ~

python -m cancer_trajectory_atlas.analysis.eccentricity_check \
    --sections       "${SECTIONS[@]}" \
    --run-dirs       "${RUN_DIRS[@]}" \
    --output-dir     "$OUTPUT_DIR" \
    --n-bins         "$N_BINS" \
    --tail-fraction  "$TAIL_FRACTION" \
    --k-min          "$K_MIN" \
    --k-max          "$K_MAX"

echo ""
echo "============================================================"
echo "  ECCENTRICITY CHECK COMPLETE"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/eccentricity_report.md"
echo "  $OUTPUT_DIR/eccentricity_check.json"
for SECTION in "${SECTIONS[@]}"; do
    echo "  $OUTPUT_DIR/eccentricity_${SECTION}.png (+ .pdf)"
done
echo ""
echo "Inputs (unchanged, read-only):"
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "  $RUN_DIR/adata_full.h5ad"
done
echo ""
echo "Read in this order:"
echo "  1. verdicts.4_overall            — does the trajectory framing survive?"
echo "  2. task_a.<section>.measures      — DECISIVE rows first (morph |z| vs signed z),"
echo "                                      then INFORMATIVE (PCA). IGNORE the"
echo "                                      DEFINITIONAL diffusion-space rows as evidence."
echo "  3. task_b.<section>.bidirectional_enrichment — both tails enriched = eccentricity."
echo ""
