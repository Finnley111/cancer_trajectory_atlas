#!/bin/bash
# Per-section atlas analysis: clean pipeline (no stain normalization, no batch correction)
# for sections 2M-1 and 2M-2 independently, with within-section LOO stability,
# post-processing overlays, batch purity, and a cross-section replication check.
#
# Features are loaded from the per-slide cache — no GPU requested.
# If any slide is missing from the cache the pipeline exits immediately with a
# cache-miss error rather than silently running slow CPU inference.
#
# Estimated walltime breakdown:
#   Main pipeline x2  (~8 slides, CPU, from cache)  : 2 x 60 min = 2 h
#   LOO Phase A x16   (~7 slides, 200 perms each)   : 16 x 30 min = 8 h
#   LOO Phase B x16                                  : 16 x  5 min = 1.5 h
#   Post-processing x2                               :              ~40 min
#   LOO summary, cross-section compare               :              ~10 min
#   Total                                            :             ~12.5 h
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/run_per_section.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=atlas_per_section
#SBATCH --output=logs/per_section-%j.out

set -euo pipefail

mkdir -p logs

# ── Constants — edit here, not inline ─────────────────────────────────────────
SECTIONS=("2M-1" "2M-2")
N_PER_BIN=50
LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
N_PERMUTATIONS_LOO=200
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
PER_SECTION_BASE="$SCRATCH/results/per_section"

# ── Slide lists (one bash array per section; no external file dependency) ─────
SLIDES_2M_1=(
    6027-4L-2M-1_x5  6027-4R-2M-1_x5
    6028-4L-2M-1_x5  6028-4R-2M-1_x5
    6029-4L-2M-1_x5  6029-4R-2M-1_x5
    6031-4L-2M-1_x5  6031-4R-2M-1_x5
)
SLIDES_2M_2=(
    6027-4L-2M-2_x5  6027-4R-2M-2_x5
    6028-4L-2M-2_x5  6028-4R-2M-2_x5
    6029-4L-2M-2_x5  6029-4R-2M-2_x5
    6031-4L-2M-2_x5  6031-4R-2M-2_x5
)

# ── Environment ───────────────────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "============================================"
echo "  Atlas per-section runs (no correction)"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "============================================"

mkdir -p "$PER_SECTION_BASE"
cd ~

# ── Per-section loop ──────────────────────────────────────────────────────────
for SECTION in "${SECTIONS[@]}"; do

    echo ""
    echo "============================================"
    echo "  Section: $SECTION"
    echo "============================================"

    # Select the slide array for this section
    if [ "$SECTION" = "2M-1" ]; then
        SECTION_SLIDES=("${SLIDES_2M_1[@]}")
    else
        SECTION_SLIDES=("${SLIDES_2M_2[@]}")
    fi
    SLIDES_CSV=$(IFS=,; echo "${SECTION_SLIDES[*]}")

    OUT_DIR="$PER_SECTION_BASE/atlas_${SECTION}"
    mkdir -p "$OUT_DIR"

    # ── Step 1: Main pipeline ─────────────────────────────────────────────────
    echo ""
    echo "=== [1/5] Main pipeline: section $SECTION ==="

    python -m cancer_trajectory_atlas.run_all \
        --run \
        --png-dir             "$PNG_DIR" \
        --annotation-dir      "$ANN_DIR" \
        --output-dir          "$OUT_DIR" \
        --stain-method        none \
        --batch-method        none \
        --model               phikon \
        --patch-size          112 \
        --stride              96 \
        --clustering-method   leiden \
        --leiden-resolution   "$LEIDEN_RES" \
        --n-roots             "$N_ROOTS" \
        --n-permutations      "$N_PERMUTATIONS" \
        --features-cache-dir  "$CACHE_DIR" \
        --cap-strategy        median \
        --slides              "$SLIDES_CSV"

    # ── Step 2: Within-section kNN batch purity ───────────────────────────────
    echo ""
    echo "=== [2/5] Batch purity: section $SECTION ==="
    # With --batch-method none there is no harmony/scvi embedding; only raw_pca
    # mixing is reported. Within one section this reflects biology, not batch.

    python -m cancer_trajectory_atlas.analysis.run_batch_mixing "$OUT_DIR"

    # ── Step 3: Post-processing overlays and patch exports ────────────────────
    echo ""
    echo "=== [3/5] Post-processing: section $SECTION ==="

    python -m cancer_trajectory_atlas.visualize.interactive_overlay \
        --results-csv  "$OUT_DIR/results.csv" \
        --png-dir      "$PNG_DIR" \
        --output-dir   "$OUT_DIR/overlays" \
        --patch-size   112

    python -m cancer_trajectory_atlas.visualize.export_patches \
        --results-csv  "$OUT_DIR/results.csv" \
        --png-dir      "$PNG_DIR" \
        --output-dir   "$OUT_DIR/patch_export" \
        --patch-size   112 \
        --n-per-bin    "$N_PER_BIN"

    # ── Step 4: Within-section LOO ────────────────────────────────────────────
    echo ""
    echo "=== [4/5] Within-section LOO: section $SECTION (${#SECTION_SLIDES[@]} folds) ==="

    LOO_DIRS=()

    for HELD_OUT in "${SECTION_SLIDES[@]}"; do
        echo ""
        echo "  --- LOO fold: held-out = $HELD_OUT ---"

        # Build 7-slide training list by excluding the held-out slide
        TRAINING_SLIDES=()
        for S in "${SECTION_SLIDES[@]}"; do
            [ "$S" != "$HELD_OUT" ] && TRAINING_SLIDES+=("$S")
        done
        TRAINING_CSV=$(IFS=,; echo "${TRAINING_SLIDES[*]}")

        LOO_OUT="$PER_SECTION_BASE/loo_${SECTION}_${HELD_OUT}"
        mkdir -p "$LOO_OUT"

        # Phase A — train on 7 slides
        python -m cancer_trajectory_atlas.run_all \
            --run \
            --png-dir             "$PNG_DIR" \
            --annotation-dir      "$ANN_DIR" \
            --output-dir          "$LOO_OUT" \
            --stain-method        none \
            --batch-method        none \
            --model               phikon \
            --patch-size          112 \
            --stride              96 \
            --clustering-method   leiden \
            --leiden-resolution   "$LEIDEN_RES" \
            --n-roots             "$N_ROOTS" \
            --n-permutations      "$N_PERMUTATIONS_LOO" \
            --features-cache-dir  "$CACHE_DIR" \
            --cap-strategy        median \
            --slides              "$TRAINING_CSV"

        # Phase B — project held-out slide onto the LOO manifold
        # Forward the active patch cap so projection matches training exactly
        ACTIVE_CAP=$(cat "$LOO_OUT/active_cap.txt")
        CAP_ARGS=()
        if [ "$ACTIVE_CAP" -gt 0 ]; then
            CAP_ARGS=(--max-patches-per-slide "$ACTIVE_CAP")
        fi

        python -m cancer_trajectory_atlas.analysis.loo_project \
            --projector-dir   "$LOO_OUT/projector" \
            --held-out-slide  "$HELD_OUT" \
            --cache-dir       "$CACHE_DIR" \
            --full-run-dir    "$OUT_DIR" \
            --output-dir      "$LOO_OUT" \
            "${CAP_ARGS[@]}"

        LOO_DIRS+=("$LOO_OUT")
    done

    # ── Step 5: LOO summary ───────────────────────────────────────────────────
    echo ""
    echo "=== [5/5] LOO summary: section $SECTION ==="

    python -m cancer_trajectory_atlas.analysis.loo_summary \
        --loo-dirs   "${LOO_DIRS[@]}" \
        --output-dir "$OUT_DIR/loo_summary"

done  # end section loop

# ── Cross-section morphology replication check ────────────────────────────────
echo ""
echo "=== Cross-section replication check ==="

python -m cancer_trajectory_atlas.analysis.cross_section_compare \
    --run-dir-2m1  "$PER_SECTION_BASE/atlas_2M-1" \
    --run-dir-2m2  "$PER_SECTION_BASE/atlas_2M-2" \
    --output-dir   "$PER_SECTION_BASE"

# ── Output summary ────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  PER-SECTION ANALYSIS COMPLETE"
echo "============================================"
echo ""
for SECTION in "${SECTIONS[@]}"; do
    OUT_DIR="$PER_SECTION_BASE/atlas_${SECTION}"
    echo "Section $SECTION:"
    echo "  AnnData:        $OUT_DIR/adata_full.h5ad"
    echo "  Validation:     $OUT_DIR/validation.json"
    echo "  PAGA:           $OUT_DIR/figures/qc_paga_topology.png"
    echo "  Phase-6 panel:  $OUT_DIR/figures/fig6_features_vs_pt.png"
    echo "  Batch purity:   $OUT_DIR/batch_mixing.json"
    echo "  Overlays:       $OUT_DIR/overlays/"
    echo "  Patch export:   $OUT_DIR/patch_export/"
    echo "  LOO summary:    $OUT_DIR/loo_summary/loo_summary.csv"
    echo "  LOO figure:     $OUT_DIR/loo_summary/loo_stability_figure.png"
    echo ""
done
echo "Cross-section: $PER_SECTION_BASE/cross_section_comparison.csv"
echo ""

# ── Append working log to PROJECT_STATE.md ────────────────────────────────────
cat >> ~/cancer_trajectory_atlas/PROJECT_STATE.md << 'WORKLOG'

---

## Working Log — Per-Section Analysis (2026-06-27)

**Job:** `jobs/run_per_section.sh`

### Runs
Two clean per-section pipeline runs with no stain normalization (`--stain-method none`)
and no batch correction (`--batch-method none`) on sections 2M-1 and 2M-2 independently.
Each run: 8 slides, Leiden resolution 0.5, 20-root DPT, 1000 permutations, features
loaded from `$SCRATCH/data/features_cache`, cap strategy = median.
Output dirs: `$SCRATCH/results/per_section/atlas_2M-1` and `atlas_2M-2`.

PAGA topology, Phase-6 morphology correlations, and permutation nulls are produced
automatically by `run_all` (no separate invocation needed).

### LOO stability (within-section)
8-fold leave-one-slide-out per section (200 permutations each fold).
Held-out slide projected onto 7-slide manifold via `loo_project.py`.
Primary metric: Spearman rho (projected PT vs in-manifold PT from the 8-slide run).
Results: `$OUT/loo_summary/loo_summary.csv`, `loo_stability_figure.png`.

Mean rho per section:
- 2M-1: [fill after job completes]
- 2M-2: [fill after job completes]

### Phase-6 morphology (per section)
Spearman rho between pseudotime and morphological features in `validation.json` /
`figures/fig6_features_vs_pt.png`. Fill after job completes.

### Cross-section replication
`cross_section_comparison.csv` (`$SCRATCH/results/per_section/`): for each
morphological feature, rho and p-value in 2M-1 vs 2M-2; `replicated = True` iff
same sign AND |rho| >= 0.1 in both sections.

Primary diagnostic: whether the within-section pseudotime axis is reproducible and
encodes real biology (replicating morphological gradients across two independent
sections) vs. a section-specific or cellularity-only artefact.

Replication result: [fill after job completes]

### Batch purity
Within-section kNN batch purity (`batch_mixing.json` per run, `raw_pca` field only —
no harmony/scvi embedding). Within one section this score reflects biological
clustering, not cross-section batch effect; expected to be non-random.

WORKLOG

echo "PROJECT_STATE.md updated."
