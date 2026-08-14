#!/bin/bash
# Shared config + guards for the per-slide timepoint run. SOURCED, not submitted.
#
# ============================================================================
# NON-COMPARABILITY CONSTRAINT
# ============================================================================
# run_individual.py fits a SEPARATE PCA basis per slide, with no patch cap, no
# feature cache, no batch correction, and single-root cluster-anchored DPT.
# PSEUDOTIME VALUES FROM ONE SLIDE ARE NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO
# ANY PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT.
#
# This answers "what does the internal morphological ordering look like within
# this one slide" and nothing about differences BETWEEN slides or timepoints.
#
# It is also independent of, and does not resolve, the earlier finding that
# projecting these slides onto the trained manifold produced 100% extrapolation,
# or that they differ substantially in staining from the 2M cohort.
# ============================================================================

set -euo pipefail

PNG_DIR="$SCRATCH/data/timepoint_x5_full"
OUT_ROOT="$SCRATCH/results/individual_timepoint"
PASS1_DIR="$OUT_ROOT/pass1_arbitrary_root"
FINAL_DIR="$OUT_ROOT/final"
ROOT_CHOICES="$OUT_ROOT/root_choices.json"
INDEX_MD="$OUT_ROOT/INDEX.md"

# The timepoint slides have NO annotations, so patching must be whole-slide.
# run_individual's discover_slides sets annotation=None when no <stem>.json or
# <stem>.geojson is found, and run_one_slide then takes the same
# roi_polygons=None path run_all.py uses — verified by reading the code.
# Pointing --annotation-dir at a guaranteed-EMPTY directory makes that explicit
# and immune to a stray file appearing in a shared annotation directory.
NO_ANN_DIR="$OUT_ROOT/_no_annotations_intentionally_empty"

PATCH_SIZE=112
STRIDE=96
N_PER_BIN=50
LEIDEN_RES=0.5

# --stain-method none is REQUIRED, not stylistic:
#   1. run_individual's default is reinhard, and its reference slide is
#      slides[0] AFTER filtering — so a per-slide invocation normalises each
#      slide to ITSELF, a silent per-slide-reference confound.
#   2. Pass 2 crops RAW pixels from the PNG to compute nuclear density. Under any
#      normalizer the pipeline would have clustered different pixels, and the
#      densities would not correspond to the clusters they are attributed to.
STAIN_METHOD=none

# Confirmed corrupt (OpenSlideError "Restart marker not found") and the confirmed
# filename duplicate. Excluded by stem match, never deleted.
EXCLUDE_STEMS=(
    "6041-4L-12W"
    "6069-4R-4W"
    "60997-4L-4W-2"
)

# Slides below this are flagged UNSTABLE in the index. run_individual's own floor
# is 50 patches, far too low for a stable per-slide PCA/Leiden/DPT.
MIN_PATCHES=500

banner() {
    echo "============================================================================"
    echo "  NON-COMPARABILITY CONSTRAINT"
    echo "  run_individual.py fits a SEPARATE PCA basis per slide: no patch cap, no"
    echo "  feature cache, no batch correction, single-root cluster-anchored DPT."
    echo "  PSEUDOTIME FROM ONE SLIDE IS NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO"
    echo "  ANY PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT."
    echo "  Says nothing about differences BETWEEN slides or timepoints."
    echo "  Does not address the 100%-extrapolation projection finding or the"
    echo "  staining differences vs the 2M cohort."
    echo "============================================================================"
}

# Resolve the usable slide stems from the PNG directory, minus the exclusions.
# Derived at runtime rather than hardcoded, so the list cannot drift from what is
# actually on disk.
resolve_slides() {
    SLIDES=()
    local png stem skip ex
    for png in "$PNG_DIR"/*.png; do
        [ -e "$png" ] || continue
        stem=$(basename "$png" .png)
        [ "$stem" = "slide_dimensions" ] && continue
        skip=0
        for ex in "${EXCLUDE_STEMS[@]}"; do
            case "$stem" in "$ex"|"$ex"_x5) skip=1;; esac
        done
        if [ "$skip" -eq 1 ]; then
            echo "  EXCLUDED (known corrupt/duplicate): $stem"
            continue
        fi
        SLIDES+=("$stem")
    done
    if [ "${#SLIDES[@]}" -eq 0 ]; then
        echo "ERROR: no usable PNGs found in $PNG_DIR"
        exit 1
    fi
    echo "  Usable slides: ${#SLIDES[@]}"
}

# run_individual.py's --slide is a SUBSTRING filter, so an ambiguous stem would
# silently process several slides into one another's output. Verify 1:1.
assert_unambiguous() {
    local stem="$1" n
    n=$(find "$PNG_DIR" -maxdepth 1 -name '*.png' -printf '%f\n' \
        | sed 's/\.png$//' | grep -cF -- "$stem" || true)
    if [ "$n" -ne 1 ]; then
        echo "    ERROR: --slide '$stem' matches $n PNGs (needs exactly 1)."
        echo "           run_individual's --slide is a substring filter; an"
        echo "           ambiguous stem would process the wrong slides. Skipping."
        return 1
    fi
    return 0
}

assert_not_clobbering() {
    local d="$1"
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "ERROR: $d already exists and is non-empty."
        echo "       Refusing to overwrite an existing results directory. Move it"
        echo "       aside or pick a new OUT_ROOT."
        exit 1
    fi
}

load_env() {
    module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
    # shellcheck disable=SC1091
    source ~/envs/atlas/bin/activate
    export HF_HOME=$SCRATCH/huggingface_cache
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
}

# Dropped into every slide directory so the constraint travels with the data.
write_do_not_compare() {
    local d="$1"
    cat > "$d/DO_NOT_COMPARE.txt" <<'EOF'
PER-SLIDE RESULT — NOT COMPARABLE TO ANYTHING ELSE
==================================================
This directory was produced by run_individual.py, which fits a SEPARATE PCA
basis for this slide alone, with:
  - no patch cap
  - no feature cache
  - no batch correction
  - single-root, cluster-anchored DPT (not the atlas's 20-root median)

Consequences:
  * The pseudotime column is on THIS SLIDE'S OWN AXIS. A value of 0.8 here and
    0.8 in any other slide's results.csv mean nothing in common.
  * Cluster IDs are per-slide labels. Cluster 2 here is unrelated to cluster 2
    anywhere else.
  * These values are NOT comparable with per-section runs, the atlas run, or any
    projected pseudotime elsewhere in this project.

Valid use: inspecting the internal morphological ordering WITHIN this slide.
Invalid use: any comparison, pooling, trend, or test across slides or timepoints.

For a cross-slide comparison, use the main pipeline's shared-PCA logic or the
existing projection pathway. That is a different pipeline, not a summary of this
output.

This run also does not address the earlier finding that projecting these slides
onto the trained manifold produced 100% extrapolation, nor the staining
differences between this cohort and the 2M cohort.
EOF
}

# ── WALLTIME / MEMORY — REQUESTS, NOT MEASUREMENTS ──────────────────────────
# No sacct record for run_individual.py on this cohort was recoverable. Two
# reference points, neither a measurement of THIS workload:
#   * jobs/run_individual_pseudotime.sh REQUESTED 6h / 64G / 1x a100 for 16
#     CROPPED slides in a single job.
#   * The Stage D projection used 24h / 128G for these same 29 FULL-WIDTH slides,
#     and that had a feature cache. run_individual.py has none.
# The per-slide array sizing below is derived from those requests, not measured.
# After the first run, substitute real numbers:
#   sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State --name=indiv_tp_p1
