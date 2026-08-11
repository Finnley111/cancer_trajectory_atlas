#!/bin/bash
# Root-selection sensitivity — Checks A and C on the two completed per-section runs.
#
# WHY:
#   analysis/diffusion.py:compute_dpt_multi_root() selects its 20 DPT roots as
#   np.argsort(nuclear_density)[:20] — the 20 LOWEST-nuclear-density patches, via
#   compute_nuclear_density_quick(). nuclear_density is ALSO one of the six
#   morphological features pseudotime is validated against, AND the covariate
#   partialled out in analysis/cellularity_confound.py. The pseudotime axis is
#   therefore partly defined by a quantity it is later validated against, and
#   partly by the quantity used to adjust that validation.
#
# CHECK A (primary) — geometry-seeded roots:
#   Re-runs multi-root DPT per section changing ONLY the root rule: roots become
#   the 20 patches at an extreme of the first non-trivial diffusion component
#   (DC1) of the ALREADY-COMPUTED diffusion map. The PCA embedding, the stored
#   neighbour graphs (Leiden k=15 cosine / diffusion k=30 euclidean), the
#   diffusion map and its eigenvalues are all reused verbatim from
#   adata_full.h5ad — only sc.tl.dpt and the correlation suite are recomputed.
#   Same 20 roots, same per-root inf-clamping, same median-across-roots
#   aggregation, same min-max normalisation, same 1000-shuffle permutation null
#   (the production validation.correlations.permutation_test is imported, not
#   reimplemented). BOTH DC1 tails are run and reported. The direction-matched
#   tail is labelled from rho(DC1, nuclear_density) over ALL patches (the mean
#   nuclear_density of just 20 roots is too noisy to orient on); the 20-root mean
#   is kept as a secondary signal and any disagreement is flagged in the report.
#   Each tail also reports ROOT PROVENANCE — how many distinct slides the 20 roots
#   come from. A tail whose roots are >=80% from one slide is a local outlier lobe,
#   not a manifold endpoint, and its axis is unreliable.
#   Headline number: Spearman rho between the ORIGINAL and GEOMETRY-SEEDED
#   pseudotime vectors, per section, read against the RANDOM-ROOT NULL below.
#
# RANDOM-ROOT NULL (calibration, part of Check A):
#   N_RANDOM_DRAWS sets of 20 uniformly-random roots per section, same aggregation,
#   reporting rho vs the original pseudotime. Without this there is no scale for
#   the DC1 numbers: rho=0.52 means one thing if random roots give 0.1 and quite
#   another if they give 0.5. A DC1 tail that does not exceed the random ceiling
#   carries no information beyond what any arbitrary root set would give.
#
# CHECK C (robustness) — alternative confound covariate:
#   Re-runs the cellularity confound analysis on the EXISTING, UNCHANGED
#   per-section pseudotime, partialling out nc_ratio instead of nuclear_density.
#   Also reproduces the saved nuclear_density-controlled partials with the same
#   parameterised loop as a parity check, and reports rho(nuclear_density,
#   nc_ratio) per section so the reader can judge how independent the
#   substitution actually is.
#
# READS (ALL READ-ONLY — per-section runs are never modified or re-run):
#   $SCRATCH/results/per_section/atlas_2M-1/adata_full.h5ad
#   $SCRATCH/results/per_section/atlas_2M-1/validation.json
#   $SCRATCH/results/per_section/atlas_2M-1/cellularity_confound/cellularity_confound.json
#   ... and the same three for atlas_2M-2
#
# WRITES (NEW directory only — no existing results directory is touched):
#   $SCRATCH/results/root_sensitivity/root_sensitivity_report.md
#   $SCRATCH/results/root_sensitivity/root_sensitivity.json
#   $SCRATCH/results/root_sensitivity/check_a_root_sensitivity_<section>.{png,pdf}
#   $SCRATCH/results/root_sensitivity/check_c_covariate_<section>.{png,pdf}
#   $SCRATCH/results/root_sensitivity/pseudotime_geometry_seeded_<section>_<tail>.npy
#   $SCRATCH/results/root_sensitivity/pseudotime_std_geometry_seeded_<section>_<tail>.npy
#
# CPU only — no GPU. No feature extraction: the cached Phikon features were
# already consumed by the per-section runs, and this job reads only their saved
# PCA embedding out of adata_full.h5ad.
#
# NOTE: unlike run_per_section.sh / run_cellularity_confound.sh, this script does
# NOT append a working log to PROJECT_STATE.md — this re-analysis was specified
# as strictly non-modifying. Add the log entry by hand after reviewing the report.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_root_sensitivity.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=root_sensitivity
#SBATCH --output=logs/root_sensitivity-%j.out

set -euo pipefail

mkdir -p logs

# ── Constants ─────────────────────────────────────────────────────────────────
SECTIONS=("2M-1" "2M-2")
PER_SECTION_BASE="$SCRATCH/results/per_section"
RUN_DIRS=(
    "$PER_SECTION_BASE/atlas_2M-1"
    "$PER_SECTION_BASE/atlas_2M-2"
)
OUTPUT_DIR="$SCRATCH/results/root_sensitivity"
N_ROOTS=20
N_PERMUTATIONS=1000
N_RANDOM_DRAWS=5

echo "============================================================"
echo "  Root-selection sensitivity — Checks A and C"
echo "  Job ID        : ${SLURM_JOB_ID:-local}"
echo "  Sections      : ${SECTIONS[*]}"
echo "  Roots per run : $N_ROOTS"
echo "  Permutations  : $N_PERMUTATIONS"
echo "  Random draws  : $N_RANDOM_DRAWS  (null baseline for the DC1 numbers)"
echo "  Output dir    : $OUTPUT_DIR  (NEW — existing runs untouched)"
echo "============================================================"

# ── Pre-run checks ────────────────────────────────────────────────────────────
echo ""
echo "=== Pre-run checks (all inputs read-only) ==="
MISSING=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "  $RUN_DIR"
    echo -n "    adata_full.h5ad          : "
    ls -lh "$RUN_DIR/adata_full.h5ad" 2>/dev/null \
        || { echo "NOT FOUND — run run_per_section.sh first"; MISSING=1; }
    echo -n "    validation.json          : "
    ls -lh "$RUN_DIR/validation.json" 2>/dev/null \
        || echo "not found — original rho values will be reported as UNVERIFIED"
    echo -n "    cellularity_confound.json: "
    ls -lh "$RUN_DIR/cellularity_confound/cellularity_confound.json" 2>/dev/null \
        || echo "not found — Check C parity vs the saved analysis will be UNAVAILABLE"
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

python -c "import scanpy, anndata; print('scanpy', scanpy.__version__, '/ anndata', anndata.__version__)" || {
    echo "ERROR: scanpy/anndata not importable in ~/envs/atlas"; exit 1
}

mkdir -p "$OUTPUT_DIR"

cd ~

python -m cancer_trajectory_atlas.analysis.root_sensitivity \
    --sections        "${SECTIONS[@]}" \
    --run-dirs        "${RUN_DIRS[@]}" \
    --output-dir      "$OUTPUT_DIR" \
    --n-roots         "$N_ROOTS" \
    --n-permutations  "$N_PERMUTATIONS"     --n-random-draws  "$N_RANDOM_DRAWS"

echo ""
echo "============================================================"
echo "  ROOT SENSITIVITY COMPLETE"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/root_sensitivity_report.md"
echo "  $OUTPUT_DIR/root_sensitivity.json"
for SECTION in "${SECTIONS[@]}"; do
    echo "  $OUTPUT_DIR/check_a_root_sensitivity_${SECTION}.png (+ .pdf)"
    echo "  $OUTPUT_DIR/check_c_covariate_${SECTION}.png (+ .pdf)"
    for TAIL in low high; do
        echo "  $OUTPUT_DIR/pseudotime_geometry_seeded_${SECTION}_${TAIL}.npy"
        echo "  $OUTPUT_DIR/pseudotime_std_geometry_seeded_${SECTION}_${TAIL}.npy"
    done
done
echo ""
echo "Inputs (unchanged, read-only):"
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "  $RUN_DIR/adata_full.h5ad, validation.json, cellularity_confound/"
done
echo ""
echo "Headline numbers to read first — Check A, 'original vs geometry-seeded"
echo "pseudotime' table at the top of the report: near 1.0 closes the"
echo "circularity concern; divergence bounds how much root choice drives it."
echo ""
