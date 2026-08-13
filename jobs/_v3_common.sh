#!/bin/bash
# Shared constants and safety guards for the v3 root/filter experiment.
# SOURCED by run_v3*.sh — not submittable on its own.
#
# WHY A SHARED FILE. The four v3 jobs must agree exactly on which tree is the
# protected reference and which cache belongs to which config. Duplicating those
# paths four times is how one of them ends up pointing at per_section_v2 after an
# edit. Everything here is read-only configuration plus assertions.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
HOLEYNESS_EXPORT="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"

# PROTECTED — never written by anything in this experiment.
PROD_CACHE="$SCRATCH/data/features_cache"
V2_BASE="$SCRATCH/results/per_section_v2"
BASELINE_BASE="$SCRATCH/results/per_section"

# NEW — created by this experiment.
RELAXED_CACHE="$SCRATCH/data/features_cache_v3relaxed"
V3A_BASE="$SCRATCH/results/per_section_v3a_holeyroot"
V3B_BASE="$SCRATCH/results/per_section_v3b_relaxed"
V3C_BASE="$SCRATCH/results/per_section_v3c_both"
V3_COMPARE="$SCRATCH/results/v3_root_experiment"

# ── Pipeline constants — identical to jobs/run_per_section_v2.sh ─────────────
LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
PATCH_SIZE=112
STRIDE=96

# 2M-1 only: holeyness exists for this section alone.
SECTION="2M-1"
SLIDES_2M_1=(
    6027-4L-2M-1_x5  6027-4R-2M-1_x5
    6028-4L-2M-1_x5  6028-4R-2M-1_x5
    6029-4L-2M-1_x5  6029-4R-2M-1_x5
    6031-4L-2M-1_x5  6031-4R-2M-1_x5
)
SLIDES_CSV=$(IFS=,; echo "${SLIDES_2M_1[*]}")

# ── Guards ───────────────────────────────────────────────────────────────────

# Refuse to write anywhere inside a protected results tree.
v3_assert_output_safe() {
    local out="$1"
    local prot
    for prot in "$V2_BASE" "$BASELINE_BASE"; do
        case "$out" in
            "$prot"|"$prot"/*)
                echo "ERROR: output path '$out' is inside the protected tree '$prot'."
                echo "       Refusing to run. v2 and the baseline are read-only here."
                exit 1;;
        esac
    done
}

# Refuse to let a relaxed-filter run touch the production feature cache.
v3_assert_cache_safe() {
    local cache="$1"
    case "$cache" in
        "$PROD_CACHE"|"$PROD_CACHE"/*)
            echo "ERROR: a relaxed-filter config was pointed at the PRODUCTION cache"
            echo "       '$PROD_CACHE'. Relaxed filters change the patch count, so"
            echo "       this would either abort on the shape guard or, worse, write"
            echo "       relaxed features into the production cache. Refusing to run."
            exit 1;;
    esac
}

# Production cache must be COMPLETE for Config A — which runs CPU-only and must
# never silently fall back to GPU inference and write new files into it.
v3_assert_prod_cache_complete() {
    local missing=0 slide
    for slide in "${SLIDES_2M_1[@]}"; do
        if [ ! -f "$PROD_CACHE/${slide}_features.npy" ]; then
            echo "  ERROR: production cache miss for $slide"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "ERROR: the production Phikon cache is incomplete. Config A is CPU-only"
        echo "       and must not regenerate cache entries. Populate it first with"
        echo "       jobs/run_cache_population.sh, then resubmit."
        exit 1
    fi
    echo "  Production cache complete for all ${#SLIDES_2M_1[@]} slides (READ-ONLY here)."
}

v3_assert_inputs_exist() {
    local missing=0 p
    for p in "$@"; do
        echo -n "  $p : "
        if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; missing=1; fi
    done
    [ "$missing" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }
}

v3_load_env() {
    module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
    # shellcheck disable=SC1091
    source ~/envs/atlas/bin/activate
    export HF_HOME=$SCRATCH/huggingface_cache
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
}

# ── WALLTIME / MEMORY — NOT MEASURED ─────────────────────────────────────────
# The values in the individual scripts are INHERITED from jobs/run_per_section_v2.sh
# (08:00:00 / 64G / 8 cpus), which itself documents them as an upper bound carried
# over from the larger jobs/run_per_section.sh rather than a measurement. No sacct
# record was recoverable from the machine these scripts were written on.
#
# Before relying on them, run on Narval and substitute the real numbers:
#     sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#           --name=atlas_per_section_v2
# The GPU sizing in run_v3_relaxed_cache_prepop.sh is likewise unverified; it
# mirrors jobs/run_cache_population.sh's request.
