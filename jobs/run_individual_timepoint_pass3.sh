#!/bin/bash
# PASS 3 of 3 — the run that is KEPT. Explicit per-slide root, then Task 2
# post-processing (overlays + patch export) and the Task 3 index.
#
# ============================================================================
# NON-COMPARABILITY CONSTRAINT
# PSEUDOTIME FROM ONE SLIDE IS NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO ANY
# PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT. run_individual.py
# fits a separate PCA basis per slide: no patch cap, no feature cache, no batch
# correction, single-root cluster-anchored DPT. This answers "what does the
# internal morphological ordering look like within this one slide" and NOTHING
# about differences BETWEEN slides or timepoints. It does not address the
# 100%-extrapolation projection finding or the staining differences vs the 2M
# cohort. A DO_NOT_COMPARE.txt is written into every slide directory.
# ============================================================================
#
# Each slide is re-run with --root-cluster taken from Pass 2's root_choices.json
# (lowest median nuclear density among that slide's Leiden clusters), rather than
# run_individual.py's arbitrary lowest-numbered-cluster default.
#
# Post-processing matches the prior convention in this project exactly
# (jobs/run_individual_pseudotime.sh): --patch-size 112, --n-per-bin 50.
# Both visualize tools were confirmed to accept run_individual.py's single-slide
# output as-is: each reads results.csv and requires only {x, y, slide_name,
# pseudotime}, which that CSV provides. NO ADAPTATION WAS NEEDED.
#
# Patch bins are low/mid/high pseudotime WITHIN THAT SLIDE — the thresholds are
# fractions of that slide's own axis and mean nothing across slides.
#
# READS (READ-ONLY): Pass 1 tree, root_choices.json, $SCRATCH/data/timepoint_x5_full
# WRITES (NEW ONLY): $SCRATCH/results/individual_timepoint/final and INDEX.md
#
# WALLTIME/MEMORY ARE REQUESTS, NOT MEASUREMENTS — see _individual_timepoint_common.sh.
#
# Usage: sbatch ~/cancer_trajectory_atlas/jobs/run_individual_timepoint_pass3.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=indiv_tp_p3
#SBATCH --output=logs/indiv_tp_p3-%j.out

set -euo pipefail
mkdir -p logs
JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$JOBS_DIR/_individual_timepoint_common.sh" ] || JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$JOBS_DIR/_individual_timepoint_common.sh"

banner
echo "  PASS 3 of 3 — explicit per-slide root. THIS IS THE DELIVERABLE."
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Output : $FINAL_DIR   (NEW)"
echo "============================================================================"

[ -f "$ROOT_CHOICES" ] || { echo "ERROR: $ROOT_CHOICES not found. Run pass 2 first."; exit 1; }
assert_not_clobbering "$FINAL_DIR"

echo ""
echo "=== Resolving slides ==="
resolve_slides

load_env
mkdir -p "$FINAL_DIR" "$NO_ANN_DIR"
if [ -n "$(ls -A "$NO_ANN_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $NO_ANN_DIR must stay empty so patching remains whole-slide."
    exit 1
fi
cd ~

OK=0; FAILED=0; SKIPPED=0; FAILED_LIST=()
for i in "${!SLIDES[@]}"; do
    STEM="${SLIDES[$i]}"
    SLIDE_NAME="${STEM}"
    echo ""
    echo "--- [$((i+1))/${#SLIDES[@]}] $STEM ---"

    if ! assert_unambiguous "$STEM"; then
        FAILED=$((FAILED+1)); FAILED_LIST+=("$STEM (ambiguous --slide filter)"); continue
    fi

    # Pull this slide's root from Pass 2. Missing entry => Pass 2 could not choose
    # one => skip rather than silently falling back to the arbitrary default.
    ROOT=$(python - "$ROOT_CHOICES" "$SLIDE_NAME" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
want = sys.argv[2]
for name, r in d["choices"].items():
    if name == want or name == want + "_x5" or name.replace("_x5", "") == want:
        print(r["root_cluster"]); break
else:
    print("MISSING")
PY
)
    if [ "$ROOT" = "MISSING" ]; then
        echo "    SKIP: no root choice in Pass 2 for '$SLIDE_NAME'. Not falling back"
        echo "          to the arbitrary default — that is what this pass exists to avoid."
        SKIPPED=$((SKIPPED+1)); continue
    fi
    echo "    root cluster: $ROOT  (lowest median nuclear density; see root_choices.json)"

    if ! python -m cancer_trajectory_atlas.run_individual \
            --slide             "$STEM" \
            --png-dir           "$PNG_DIR" \
            --annotation-dir    "$NO_ANN_DIR" \
            --output-dir        "$FINAL_DIR" \
            --stain-method      "$STAIN_METHOD" \
            --patch-size        "$PATCH_SIZE" \
            --stride            "$STRIDE" \
            --leiden-resolution "$LEIDEN_RES" \
            --root-cluster      "$ROOT" \
            --ndpi-scale        1.0; then
        echo "    FAILED at pseudotime (continuing to next slide)"
        FAILED=$((FAILED+1)); FAILED_LIST+=("$STEM (run_individual)"); continue
    fi

    SLIDE_DIR=""
    for cand in "$FINAL_DIR/$STEM" "$FINAL_DIR/${STEM}_x5"; do
        [ -f "$cand/results.csv" ] && SLIDE_DIR="$cand" && break
    done
    if [ -z "$SLIDE_DIR" ]; then
        echo "    FAILED: no results.csv produced (slide likely below the patch or"
        echo "            cluster floor run_individual enforces). Continuing."
        FAILED=$((FAILED+1)); FAILED_LIST+=("$STEM (no results.csv)"); continue
    fi

    write_do_not_compare "$SLIDE_DIR"

    # ── Task 2: post-processing, matching jobs/run_individual_pseudotime.sh ──
    echo "    overlays ..."
    python -m cancer_trajectory_atlas.visualize.interactive_overlay \
        --results-csv "$SLIDE_DIR/results.csv" \
        --png-dir     "$PNG_DIR" \
        --output-dir  "$SLIDE_DIR/overlays" \
        --patch-size  "$PATCH_SIZE" \
        || echo "    WARNING: overlay failed for $STEM (continuing)"

    echo "    patch export ..."
    python -m cancer_trajectory_atlas.visualize.export_patches \
        --results-csv "$SLIDE_DIR/results.csv" \
        --png-dir     "$PNG_DIR" \
        --output-dir  "$SLIDE_DIR/patch_export" \
        --patch-size  "$PATCH_SIZE" \
        --n-per-bin   "$N_PER_BIN" \
        || echo "    WARNING: patch export failed for $STEM (continuing)"

    OK=$((OK+1))
done

echo ""
echo "=== Task 3: index ==="
python -m cancer_trajectory_atlas.analysis.individual_timepoint_index \
    --final-dir    "$FINAL_DIR" \
    --root-choices "$ROOT_CHOICES" \
    --output       "$INDEX_MD" \
    --min-patches  "$MIN_PATCHES" \
    || echo "  WARNING: index generation failed"

echo ""
echo "============================================================================"
echo "  PASS 3 COMPLETE — $OK ok, $FAILED failed, $SKIPPED skipped, of ${#SLIDES[@]}"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failures:"
    for f in "${FAILED_LIST[@]}"; do echo "    - $f"; done
fi
echo ""
echo "  Deliverable : $FINAL_DIR"
echo "  Index       : $INDEX_MD   <- read this first"
echo "  Root record : $ROOT_CHOICES"
banner
