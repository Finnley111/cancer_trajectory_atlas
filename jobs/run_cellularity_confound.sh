#!/bin/bash
# Cellularity confound test on the two completed per-section runs.
#
# For each of the 5 non-density morphological features, computes:
#   raw Spearman rho vs pseudotime
#   partial Spearman rho controlling for nuclear_density
#   delta = partial - raw
#   permutation p-value on the partial rho (1000 shuffles)
#
# Outputs per run dir (additive; never touches adata_full.h5ad or results.csv):
#   $RUN_DIR/cellularity_confound/cellularity_confound.json
#   $RUN_DIR/cellularity_confound/raw_vs_partial_rho.png
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_cellularity_confound.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=cellularity_confound
#SBATCH --output=logs/cellularity_confound-%j.out

set -euo pipefail

mkdir -p logs

RUN_DIRS=(
    "$SCRATCH/results/per_section/atlas_2M-1"
    "$SCRATCH/results/per_section/atlas_2M-2"
)
N_PERMUTATIONS=1000

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

echo "============================================"
echo "  Cellularity confound test"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Permutations: $N_PERMUTATIONS"
echo "============================================"

cd ~

for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo ""
    echo "=== $RUN_DIR ==="

    python -m cancer_trajectory_atlas.analysis.cellularity_confound \
        --mode            partial \
        --results-dirs    "$RUN_DIR" \
        --n-permutations  "$N_PERMUTATIONS"

    echo "  JSON:   $RUN_DIR/cellularity_confound/cellularity_confound.json"
    echo "  Figure: $RUN_DIR/cellularity_confound/raw_vs_partial_rho.png"
done

echo ""
echo "============================================"
echo "  DONE"
echo "============================================"

# ── Append working log to PROJECT_STATE.md ────────────────────────────────────
cat >> ~/cancer_trajectory_atlas/PROJECT_STATE.md << 'WORKLOG'

---

## Working Log — Cellularity Confound Test (2026-07-02)

**Job:** `jobs/run_cellularity_confound.sh`

### Test
For each per-section run (2M-1, 2M-2), computed partial Spearman rho between
pseudotime and each of the 5 non-density morphological features (mean_nuclear_area,
nc_ratio, texture_entropy, h_intensity, packing_irregularity), controlling for
nuclear_density. Compared to raw (zero-order) rho. 1000-permutation null on the
partial correlations. Threshold for survival: |partial_rho| >= 0.1.

See cellularity_confound.json and raw_vs_partial_rho.png per run dir:
  $SCRATCH/results/per_section/atlas_2M-1/cellularity_confound/
  $SCRATCH/results/per_section/atlas_2M-2/cellularity_confound/

### Results
**2M-1:** rho(PT, nuclear_density) = [fill]
  Survivors (|partial_rho| >= 0.1): [fill]
  Collapses:                         [fill]

**2M-2:** rho(PT, nuclear_density) = [fill]
  Survivors (|partial_rho| >= 0.1): [fill]
  Collapses:                         [fill]

### Interpretation
[Fill after inspecting output: does the pseudotime axis survive partialling for
nuclear_density (true independent signal) or does it collapse (cellularity axis)?]

WORKLOG

echo "PROJECT_STATE.md updated."
