#!/bin/bash
# END-TO-END HANDOFF RUN — raw NDPI through to final results, in one job.
#
# Purpose: prove the whole pipeline runs clean from raw data on a fresh
# namespace, so the project can be handed to someone with no context. Every
# stage that a newcomer would have to perform manually is performed here.
#
# ── EVERYTHING IS WRITTEN SOMEWHERE NEW. NOTHING EXISTING IS TOUCHED. ───────
# All output goes under ONE directory:
#
#     $RUN_DIR = $RUN_BASE/$RUN_ID       (default $SCRATCH/handoff/handoff_<date>_<jobid>)
#       ├── png/                  freshly converted left-half PNGs + slide_dimensions.json
#       ├── annotation_build/     isolated GeoJSON -> ratio-JSON conversion
#       │     └── data/annotations_ratio/    the annotations this run actually uses
#       ├── annotation_qc/        polygon-overlay thumbnails for eyeballing
#       ├── features_cache/       A BRAND NEW Phikon cache, built by this run
#       ├── results/atlas_2M-1/   per-section pipeline output
#       ├── results/atlas_2M-2/
#       ├── stages/               .done markers (see RESUMING below)
#       └── provenance.txt        commit, dirty state, job id, timestamps
#
# The script REFUSES TO START if $RUN_DIR already contains results, and refuses
# to run at all if any output path resolves inside a protected tree:
#     $SCRATCH/data/MCF7_x5_cropped        (existing PNGs)
#     $SCRATCH/data/features_cache         (existing Phikon cache)
#     $SCRATCH/results/per_section*        (baseline, v2, v3 regression)
#     ~/cancer_trajectory_atlas/data/annotations_ratio   (repo annotations)
#
# The repo's own data/annotations_ratio/ is READ but never written.
# converters/batch_convert.py hardcodes relative paths './data/annotations_ratio/',
# so it is run from a staging copy in $RUN_DIR rather than from the repo. That is
# the only reason annotation_build/ has a nested data/ + converters/ layout.
#
# ── DO NOT EXPECT BITWISE AGREEMENT WITH per_section_v2 ─────────────────────
# This run builds a NEW feature cache by fresh GPU inference. Cached-vs-cached
# comparison is exact; fresh-vs-cached is not, because a different final-batch
# size can select a different cuDNN kernel and move the last ULP. If the PNGs
# also differ by even one byte, patch pixels differ and so does everything after.
#
# So: pseudotime Spearman against v2 should be ~0.99+, NOT exactly 1.000000, and
# max_abs_diff will NOT be 0.000e+00. That is expected and is not a regression.
# Phase 8's jobs/run_per_section_v3_regression.sh is the exact-equality test;
# this script is the does-it-run-from-scratch test. Do not conflate them.
#
# ── RESUMING AFTER A TIMEOUT ────────────────────────────────────────────────
# Each stage drops a marker in $RUN_DIR/stages/. Re-submitting with the SAME
# RUN_ID skips completed stages:
#
#     sbatch --export=ALL,RUN_ID=handoff_20260812_1234567 jobs/run_full_pipeline_handoff.sh
#
# Stage 3 is additionally self-resuming: run_all.py reuses any cache file already
# in $RUN_DIR/features_cache, so a partially-built cache is not rebuilt.
#
# ── RESOURCES ──────────────────────────────────────────────────────────────
# 3 days requested as a safety margin, NOT a measurement. Proven figures for the
# individual stages, from their own job scripts:
#     conversion       4 h, 8 cpu, 64G, no GPU   (jobs/convert_ndpi.sh)
#     cache + pipeline 6 h, 8 cpu, 64G, A100     (jobs/run_cache_population.sh)
#     per-section x2   8 h, 8 cpu, 64G, no GPU   (jobs/run_per_section_v2.sh)
# Sum is well under 24 h; 3 days is headroom so a slow queue or a large NDPI
# cannot truncate the run. Memory is 128G — double the 64G proven for each stage
# separately. Peak is during conversion (level-0 NDPIs are ~4 gigapixels, and a
# single decoded left-half PNG is several GB before compression). Dropping to
# 64G would queue faster and has been shown to work stage-by-stage.
#
# The GPU sits idle during conversion and during morphological features, which is
# wasteful but is the price of a single self-contained job. If the GPU queue is
# long, split it: run jobs/convert_ndpi.sh first (no GPU), then this script with
# SKIP_CONVERT=1 pointing at those PNGs.
#
# ── USAGE ──────────────────────────────────────────────────────────────────
#     sbatch jobs/run_full_pipeline_handoff.sh
#
#     # custom run id / location
#     sbatch --export=ALL,RUN_ID=my_run,RUN_BASE=$SCRATCH/handoff \
#            jobs/run_full_pipeline_handoff.sh
#
#     # reuse already-converted PNGs instead of re-converting
#     sbatch --export=ALL,SKIP_CONVERT=1,PNG_DIR=$SCRATCH/data/MCF7_x5_cropped \
#            jobs/run_full_pipeline_handoff.sh
#
#     # NDPIs somewhere else
#     sbatch --export=ALL,NDPI_DIR=/path/to/ndpi jobs/run_full_pipeline_handoff.sh
#
# The NDPI filenames drive everything downstream: <stem>.ndpi becomes
# <stem>_x5.png, and the per-section slide lists below expect stems of the form
# 6027-4L-2M-1. A mismatch shows up as "0 slides discovered" in stage 3, not as
# an error in stage 1, so the stem check in stage 1 is worth reading.
#
# ALWAYS use --export=ALL,... — a bare --export=VAR=val drops $SCRATCH from the
# job environment and every path below resolves wrong.

#SBATCH --account=def-lmarti46
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=atlas_handoff_e2e
#SBATCH --output=logs/handoff_e2e-%j.out

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
REPO="${REPO:-$HOME/cancer_trajectory_atlas}"
# NDPI source. Matches paths.json's "raw_ndpi", which is authoritative and was
# confirmed correct on Narval 2026-08-12 (16 .ndpi present).
# NOTE: jobs/convert_ndpi.sh points at $SCRATCH/data/MCF7_x5 instead, which does
# NOT exist — that script is stale and would fail if submitted today.
NDPI_DIR="${NDPI_DIR:-$SCRATCH/data/ndpi}"
RUN_BASE="${RUN_BASE:-$SCRATCH/handoff}"
RUN_ID="${RUN_ID:-handoff_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}}"
RUN_DIR="$RUN_BASE/$RUN_ID"

SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_QC="${SKIP_QC:-0}"

# Pipeline parameters — identical to jobs/run_per_section_v2.sh, the canonical
# reference configuration. Do not change these without a reason you can write down.
LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
NDPI_LEVEL=0
NDPI_SCALE=1.0

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
SECTIONS=("2M-1" "2M-2")

# Derived paths — all inside RUN_DIR.
PNG_DIR="${PNG_DIR:-$RUN_DIR/png}"
ANN_BUILD="$RUN_DIR/annotation_build"
ANN_DIR="$ANN_BUILD/data/annotations_ratio"
QC_DIR="$RUN_DIR/annotation_qc"
CACHE_DIR="$RUN_DIR/features_cache"
RESULTS_DIR="$RUN_DIR/results"
STAGE_DIR="$RUN_DIR/stages"

mkdir -p logs

echo "============================================================"
echo "  END-TO-END HANDOFF RUN — NDPI to results"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Run ID  : $RUN_ID"
echo "  Run dir : $RUN_DIR"
echo "  Started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Guard 1: never write into a protected tree ──────────────────────────────
PROTECTED=(
    "$SCRATCH/data/MCF7_x5_cropped"
    "$SCRATCH/data/features_cache"
    "$SCRATCH/results/per_section"
    "$SCRATCH/results/per_section_v2"
    "$SCRATCH/results/per_section_v3_regression"
    "$REPO/data/annotations_ratio"
    "$REPO/data/annotations"
    "$NDPI_DIR"
)
for OUT in "$RUN_DIR" "$CACHE_DIR" "$RESULTS_DIR" "$ANN_BUILD" "$QC_DIR"; do
    for P in "${PROTECTED[@]}"; do
        case "$OUT" in
            "$P"|"$P"/*)
                echo "FATAL: output path '$OUT' is inside protected tree '$P'."
                echo "       Refusing to run. Change RUN_BASE / RUN_ID."
                exit 1;;
        esac
    done
done
# PNG_DIR may legitimately point at an existing tree when SKIP_CONVERT=1, but
# only then, and only for reading.
if [ "$SKIP_CONVERT" != "1" ]; then
    for P in "${PROTECTED[@]}"; do
        case "$PNG_DIR" in
            "$P"|"$P"/*)
                echo "FATAL: PNG_DIR '$PNG_DIR' is inside protected tree '$P' and"
                echo "       SKIP_CONVERT is not set, so this run would WRITE there."
                echo "       Set SKIP_CONVERT=1 to reuse existing PNGs read-only."
                exit 1;;
        esac
    done
fi
echo "  Guard 1 ok: no output path is inside a protected tree."

# ── Guard 2: refuse to clobber an existing completed run ────────────────────
if [ -d "$RESULTS_DIR" ] && [ -n "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ] \
   && [ ! -d "$STAGE_DIR" ]; then
    echo "FATAL: $RESULTS_DIR already has content but no stage markers exist."
    echo "       This looks like a different run. Refusing to overwrite."
    echo "       Pick a new RUN_ID."
    exit 1
fi
echo "  Guard 2 ok: $RUN_DIR is safe to write."

mkdir -p "$RUN_DIR" "$STAGE_DIR" "$RESULTS_DIR" "$CACHE_DIR"

stage_done() { [ -f "$STAGE_DIR/$1.done" ]; }
mark_done()  { touch "$STAGE_DIR/$1.done"; echo "  [stage:$1] marked complete"; }

# ── Environment ─────────────────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── Provenance ──────────────────────────────────────────────────────────────
{
    echo "run_id     : $RUN_ID"
    echo "job_id     : ${SLURM_JOB_ID:-local}"
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo       : $REPO"
    echo "commit     : $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty_files: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
    echo "ndpi_dir   : $NDPI_DIR"
    echo "png_dir    : $PNG_DIR"
    echo "ann_dir    : $ANN_DIR"
    echo "cache_dir  : $CACHE_DIR  (NEW — built by this run)"
    echo "results    : $RESULTS_DIR"
    echo "params     : leiden=$LEIDEN_RES n_roots=$N_ROOTS perms=$N_PERMUTATIONS"
    echo "             stain=none batch=none cap=median model=phikon p=112 s=96"
} > "$RUN_DIR/provenance.txt"
cat "$RUN_DIR/provenance.txt"

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — NDPI to left-half PNG
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  STAGE 1 — NDPI -> cropped PNG"
echo "============================================================"

if [ "$SKIP_CONVERT" = "1" ]; then
    echo "  SKIP_CONVERT=1 — reusing existing PNGs at $PNG_DIR (read-only)."
    [ -d "$PNG_DIR" ] || { echo "FATAL: $PNG_DIR not found."; exit 1; }
    [ -f "$PNG_DIR/slide_dimensions.json" ] || {
        echo "FATAL: $PNG_DIR/slide_dimensions.json missing. Ratio annotations"
        echo "       cannot be mapped without it."; exit 1; }
elif stage_done convert; then
    echo "  Already complete (stage marker present). Skipping."
else
    [ -d "$NDPI_DIR" ] || { echo "FATAL: NDPI dir $NDPI_DIR not found."; exit 1; }
    N_NDPI=$(ls "$NDPI_DIR"/*.ndpi 2>/dev/null | wc -l)
    echo "  NDPI files found: $N_NDPI"
    [ "$N_NDPI" -gt 0 ] || { echo "FATAL: no .ndpi files in $NDPI_DIR"; exit 1; }

    mkdir -p "$PNG_DIR"
    cd ~
    python -m cancer_trajectory_atlas.run_all \
        --convert \
        --ndpi-dir   "$NDPI_DIR" \
        --png-dir    "$PNG_DIR" \
        --ndpi-level "$NDPI_LEVEL" \
        --ndpi-scale "$NDPI_SCALE"

    mark_done convert
fi

echo "  PNG count : $(ls "$PNG_DIR"/*.png 2>/dev/null | wc -l)"
echo "  PNG size  : $(du -sh "$PNG_DIR" 2>/dev/null | cut -f1)"

# ── Every slide the section lists name must now exist as a PNG ──────────────
# Without this, a filename mismatch surfaces as "0 slides discovered" deep in
# stage 3, hours later, instead of here.
echo ""
echo "  Verifying all 16 expected slides are present as PNGs ..."
MISSING_PNG=0
for SLIDE in "${SLIDES_2M_1[@]}" "${SLIDES_2M_2[@]}"; do
    if [ ! -f "$PNG_DIR/${SLIDE}.png" ]; then
        echo "    MISSING: ${SLIDE}.png"
        MISSING_PNG=$((MISSING_PNG+1))
    fi
done
if [ "$MISSING_PNG" -ne 0 ]; then
    echo "  FATAL: $MISSING_PNG expected slide PNG(s) missing from $PNG_DIR"
    echo "         Source files present in $NDPI_DIR:"
    ls "$NDPI_DIR" 2>/dev/null | head -20
    echo "         Expected NDPI names are <stem>.ndpi where <stem> is e.g."
    echo "         6027-4L-2M-1 (the section lists in this script use <stem>_x5)."
    exit 1
fi
echo "    OK: all 16 slides present."

if [ ! -f "$PNG_DIR/slide_dimensions.json" ]; then
    echo "  FATAL: $PNG_DIR/slide_dimensions.json missing — ratio annotations"
    echo "         cannot be mapped to pixel space without it."
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — GeoJSON annotations to ratio-coordinate JSON (isolated)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  STAGE 2 — annotations: GeoJSON -> ratio JSON"
echo "============================================================"

if stage_done annotations; then
    echo "  Already complete (stage marker present). Skipping."
else
    # batch_convert.py hardcodes './data/annotations', './converters/img_dims.txt'
    # and './data/annotations_ratio'. Staging a copy keeps it away from the repo.
    mkdir -p "$ANN_BUILD/data/annotations" "$ANN_BUILD/converters"
    cp "$REPO"/data/annotations/*.geojson "$ANN_BUILD/data/annotations/"
    cp "$REPO/converters/img_dims.txt"    "$ANN_BUILD/converters/"

    N_GEO=$(ls "$ANN_BUILD"/data/annotations/*.geojson | wc -l)
    echo "  Staged $N_GEO GeoJSON file(s) into $ANN_BUILD"

    ( cd "$ANN_BUILD" && python "$REPO/converters/batch_convert.py" )

    N_RATIO=$(ls "$ANN_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Produced $N_RATIO ratio-coordinate JSON file(s)"
    [ "$N_RATIO" -eq "$N_GEO" ] || {
        echo "FATAL: converted $N_RATIO of $N_GEO annotations. Most likely a slide"
        echo "       is missing from converters/img_dims.txt."; exit 1; }

    mark_done annotations
fi

# ── CRITICAL INVARIANT: img_dims.txt must agree with slide_dimensions.json ──
# batch_convert DIVIDES polygon coords by img_dims.txt; load_roi_polygons
# MULTIPLIES them by original_full_width from slide_dimensions.json. If those two
# disagree, every ROI lands in the wrong place and the run is silently garbage.
echo ""
echo "  Checking img_dims.txt against freshly written slide_dimensions.json ..."
python - "$ANN_BUILD/converters/img_dims.txt" "$PNG_DIR/slide_dimensions.json" <<'PYEOF'
import json, re, sys
dims_txt, sidecar = sys.argv[1], sys.argv[2]

table = {}
for line in open(dims_txt):
    m = re.search(r"(.*?):\s*w=(\d+)\s*h=(\d+)", line.strip())
    if m:
        table[m.group(1).strip()] = (int(m.group(2)), int(m.group(3)))

side = json.load(open(sidecar))
bad, checked = [], 0
for png, d in side.items():
    base = png.replace("_x5.png", "").replace(".png", "")
    if base not in table:
        bad.append(f"{base}: present in slide_dimensions.json but NOT in img_dims.txt")
        continue
    checked += 1
    w, h = table[base]
    if (w, h) != (d["original_full_width"], d["original_full_height"]):
        bad.append(f"{base}: img_dims.txt={w}x{h} but sidecar="
                   f"{d['original_full_width']}x{d['original_full_height']}")

print(f"    compared {checked} slide(s)")
if bad:
    print("    FATAL: dimension tables disagree - every ROI would be misplaced:")
    for b in bad:
        print("      " + b)
    sys.exit(1)
print("    OK: annotation divisor matches pipeline multiplier for every slide.")
PYEOF

# ── Informational: does this match the repo's committed ratio annotations? ──
echo ""
echo "  Comparing generated annotations with the repo's committed copy ..."
DIFFS=0
for F in "$ANN_DIR"/*.json; do
    B=$(basename "$F")
    if [ -f "$REPO/data/annotations_ratio/$B" ]; then
        cmp -s "$F" "$REPO/data/annotations_ratio/$B" || { DIFFS=$((DIFFS+1)); echo "    differs: $B"; }
    else
        echo "    not in repo: $B"
    fi
done
if [ "$DIFFS" -eq 0 ]; then
    echo "    Identical to data/annotations_ratio/ — the committed annotations are reproducible."
else
    echo "    WARNING: $DIFFS file(s) differ from the committed annotations."
    echo "             Not fatal — this run uses the freshly generated ones — but it"
    echo "             means data/annotations_ratio/ is not reproducible from"
    echo "             data/annotations/ + img_dims.txt at the current commit."
fi

# ── Optional: visual QC overlays ────────────────────────────────────────────
if [ "$SKIP_QC" = "1" ]; then
    echo "  SKIP_QC=1 — skipping annotation overlay thumbnails."
elif stage_done annotation_qc; then
    echo "  Annotation QC already complete. Skipping."
else
    echo ""
    echo "  Rendering annotation overlay thumbnails ..."
    mkdir -p "$QC_DIR"
    python "$REPO/jobs/check_annotations.py" \
        --png-dir    "$PNG_DIR" \
        --ann-dir    "$ANN_DIR" \
        --dims-json  "$PNG_DIR/slide_dimensions.json" \
        --output-dir "$QC_DIR" \
      && mark_done annotation_qc \
      || echo "  WARNING: overlay rendering failed — not fatal, continuing."
    echo "  Overlays: $QC_DIR  (EYEBALL THESE before trusting the results)"
fi

# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — per-section pipeline, building a NEW feature cache
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  STAGE 3 — per-section pipeline (fresh Phikon cache)"
echo "============================================================"
echo "  Cache dir: $CACHE_DIR"
echo "  This is a NEW cache. The shared $SCRATCH/data/features_cache is NOT read"
echo "  and NOT written. First section pays the GPU inference cost."

cd ~
for SECTION in "${SECTIONS[@]}"; do
    if stage_done "pipeline_$SECTION"; then
        echo ""
        echo "  Section $SECTION already complete. Skipping."
        continue
    fi

    echo ""
    echo "  ------------------------------------------------------"
    echo "  Section $SECTION  ($(date -u +%H:%M:%SZ))"
    echo "  ------------------------------------------------------"

    if [ "$SECTION" = "2M-1" ]; then
        SECTION_SLIDES=("${SLIDES_2M_1[@]}")
    else
        SECTION_SLIDES=("${SLIDES_2M_2[@]}")
    fi
    SLIDES_CSV=$(IFS=,; echo "${SECTION_SLIDES[*]}")

    OUT_DIR="$RESULTS_DIR/atlas_${SECTION}"
    mkdir -p "$OUT_DIR"

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

    echo ""
    echo "  --- extraction failures, section $SECTION ---"
    python - "$OUT_DIR/feature_failures.json" <<'PYEOF' || echo "  (feature_failures.json not readable)"
import json, sys
d = json.load(open(sys.argv[1]))
q = d["nuclear_density_quick"]["n_failed"]
m = d["morphological_features"]["n_failed"]
print(f"  quick n_failed = {q}")
print(f"  full  n_failed = {m}")
print(f"  nan per feature = {d['morphological_features']['nan_counts_per_feature']}")
if q or m:
    print("  NOTE: non-zero failures. per_section_v2 had 0 and 0 - investigate.")
PYEOF

    mark_done "pipeline_$SECTION"
done

# ════════════════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════════════════
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/provenance.txt"

echo ""
echo "============================================================"
echo "  END-TO-END RUN COMPLETE"
echo "============================================================"
echo ""
echo "  Run dir : $RUN_DIR"
echo "  PNGs    : $PNG_DIR                ($(ls "$PNG_DIR"/*.png 2>/dev/null | wc -l) files)"
echo "  Annots  : $ANN_DIR"
echo "  Overlays: $QC_DIR"
echo "  Cache   : $CACHE_DIR              ($(ls "$CACHE_DIR"/*_features.npy 2>/dev/null | wc -l) slides)"
echo "  Results : $RESULTS_DIR"
echo ""
for SECTION in "${SECTIONS[@]}"; do
    V="$RESULTS_DIR/atlas_${SECTION}/validation.json"
    if [ -f "$V" ]; then
        echo "  --- $SECTION verdict ---"
        python - "$V" <<'PYEOF'
import json, sys
v = json.load(open(sys.argv[1]))
s = v.get("summary", {})
print("   ", s.get("verdict", "?"))
print(f"    n_strong={s.get('n_strong_correlations')} "
      f"n_significant={s.get('n_significant_permutations')}")
for f, d in (v.get("feature_correlations") or {}).items():
    r = d.get("rho")
    print(f"      {f:<26s} rho={r:+.4f}" if isinstance(r, float) else f"      {f:<26s} rho={r}")
PYEOF
    fi
done
echo ""
echo "  NEXT — sanity-check against the reference run:"
echo ""
echo "    python -m cancer_trajectory_atlas.analysis.v3_regression_check \\"
echo "        --sections 2M-1 2M-2 \\"
echo "        --v2-base  \$SCRATCH/results/per_section_v2 \\"
echo "        --v3-base  $RESULTS_DIR \\"
echo "        --output-dir $RUN_DIR/comparison_vs_v2 \\"
echo "        --skip-confound"
echo ""
echo "  READ THIS BEFORE INTERPRETING THAT OUTPUT:"
echo "    It will almost certainly report FAIL, and that is EXPECTED here."
echo "    That tool tests BITWISE equality and was built for the cached-vs-cached"
echo "    Phase 8 regression. This run built a NEW cache by fresh GPU inference,"
echo "    so last-ULP differences are normal."
echo ""
echo "    What actually matters for this run:"
echo "      check 0  alignment + PCA width   SHOULD still pass exactly"
echo "      check 1  pseudotime Spearman     expect ~0.99+, not 1.000000"
echo "      check 2  max_abs_diff            expect small but NON-ZERO"
echo "      check 3  feature rhos            expect agreement to ~2-3 decimals"
echo "      check 4  DPT root sets           expect high overlap, maybe not 20/20"
echo "      check 5  failure counts          SHOULD still be 0 and 0"
echo ""
echo "    A failing check 0, or non-zero extraction failures, IS a real problem."
echo "    Everything else, judge by magnitude."
echo ""
echo "  Also eyeball $QC_DIR before trusting anything downstream."
