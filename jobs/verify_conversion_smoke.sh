#!/bin/bash
# TIER 2 — conversion and patch-extraction smoke test. CPU-ONLY.
#
# PURPOSE
#   Confirms the stages UPSTREAM of the feature cache still work: NDPI decode,
#   left-half crop, dimension bookkeeping, and patch extraction geometry.
#
# WHY THIS TIER ASSERTS STRUCTURE, NOT BYTES
#   PNG encoding is not guaranteed byte-stable across zlib or Pillow versions,
#   and any pixel difference propagates through Phikon into different
#   embeddings. A byte-diff of freshly converted PNGs would therefore be
#   uninterpretable: a refactor bug and an encoder version bump look identical.
#
#   So this asserts the quantities that are determined by GEOMETRY AND THE
#   TISSUE FILTERS, which are reproducible even when the bytes are not:
#
#     1. PNG dimensions match the recorded slide_dimensions.json
#     2. cropped_width == original_full_width // 2, the documented invariant
#     3. patch count matches the row count of the slide's cached feature file
#
#   Assertion 3 is the meaningful one. Patch count is decided by the tiling
#   ranges, the ROI polygons, and the five tissue filters. If it still matches
#   the cache, extraction geometry and every filter threshold are unchanged.
#
#   The mean absolute per-pixel difference against the existing PNG is REPORTED
#   AND NOT ASSERTED ON, as diagnostic information only. A small non-zero value
#   is expected and is not a failure.
#
# WHY THIS NEEDS NO GPU
#   The patch-count assertion stops at extraction. Patches are counted, never
#   embedded, so Phikon never loads. The cached .npy is read only for its row
#   count via a header-only shape read, not fully loaded. CPU-only, no GPU
#   allocation, and no HuggingFace access.
#
# THE CAP MUST NOT BE APPLIED
#   The feature cache stores FULL, UNCAPPED features; the per-slide cap is
#   applied later, in run_all's Pass 2, after the cohort median is known. So the
#   comparison here is against the UNCAPPED patch count. Applying a cap would
#   make this test fail for a reason that is not a defect.
#
# LAST RESULT: PASS, job 1648162, 2026-08-25, at this revision.
#   Both slides bit-identical to the reference PNGs, patch counts 616 and 1228
#   matching the cache exactly. That run is what established the conversion
#   recipe as --ndpi-level 0 --ndpi-scale 0.5; see the NDPI_LEVEL note below.
#
# WALLTIME / MEMORY — NOT MEASURED
#   Job 1648162 completed but its elapsed and MaxRSS were never recorded here.
#   Recover them with the sacct line below before trusting the request. The only
#   other reference is jobs/convert_ndpi.sh, which requests 4 h and 64 G for all
#   16 slides. This converts 2, so time should be far under that.
#
#   Memory is the real constraint and is reasoned, not measured. A level-0 NDPI
#   here runs to roughly 96000 x 42240 px. The full RGB image is about 12 GB and
#   the left half about 6 GB, and openslide materialises the full image before
#   the crop.
#
#   The pixel-difference diagnostic decodes BOTH PNGs (about 6 GB each; PIL's
#   crop loads the whole image, so striping does not avoid that). Striping keeps
#   the int16 subtraction from adding two more full-size copies. Rough peak is
#   the 6 GB patch-extraction array, then about 12 GB for the two decoded PNGs.
#   128 G is generous for that and deliberately so, since the failure mode of
#   guessing low here is a dead job hours in.
#
#   After the first run:
#       sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#             --name=verify_conv_smoke
#
# READS (READ-ONLY): $SCRATCH/data/ndpi           (NDPI source, override NDPI_DIR)
#                    $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
#                    $SCRATCH/data/MCF7_x5_cropped/<slide>.png  (diagnostic only)
#                    $SCRATCH/data/features_cache/<slide>_features.npy (shape only)
#                    data/annotations_ratio
# WRITES (NEW ONLY): $SCRATCH/verify_conversion_smoke/<TAG>/
#
#   It does NOT write to MCF7_x5_cropped and does NOT write to the feature
#   cache. Both are guarded below.
#
# Usage:
#   sbatch ~/cancer_trajectory_atlas/jobs/verify_conversion_smoke.sh
#   sbatch --export=ALL,SMOKE_TAG=mytag ~/.../verify_conversion_smoke.sh
#     (the defaults are the measured-correct ones; no overrides needed)
#   sbatch --export=ALL,SMOKE_SLIDES="6027-4L-2M-1 6027-4L-2M-2" ~/.../verify_conversion_smoke.sh
#
#   NDPI_DIR, SMOKE_SLIDES and SMOKE_TAG are all overridable from the command
#   line. NDPI_DIR is the one worth checking before submitting.

#SBATCH --account=def-lmarti46
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=verify_conv_smoke
#SBATCH --output=logs/verify_conv_smoke-%j.out

set -euo pipefail

# Revision of THIS FILE. Printed in the banner below so a job log says which
# version ran. These scripts are edited off-cluster; if the banner does not
# match the revision you expect, the copy on the cluster is stale and the
# fix you are looking for is not in it. Bump on every change.
SCRIPT_REV="2026-08-25b"

mkdir -p logs

# NDPI source. paths.json's "raw_ndpi" is authoritative and says
# $SCRATCH/data/ndpi, confirmed on Narval 2026-08-12 with 16 .ndpi present.
#
# HISTORY: jobs/convert_ndpi.sh hardcoded $SCRATCH/data/MCF7_x5, which does not
# exist. This script inherited that wrong path and failed its own pre-flight
# check on job 1640510. Both are corrected as of 2026-08-25; the note is kept
# because the same mistake is easy to reintroduce by copying an old recipe.
NDPI_DIR="${NDPI_DIR:-$SCRATCH/data/ndpi}"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
CACHE_DIR="$SCRATCH/data/features_cache"
ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"

# One slide per section, so both fixations are exercised.
SMOKE_SLIDES="${SMOKE_SLIDES:-6027-4L-2M-1 6027-4L-2M-2}"

# Conversion resolution. MEASURED, not inherited.
#
# These defaults reproduce $SCRATCH/data/MCF7_x5_cropped BIT-IDENTICALLY, verified
# by job 1648162 on 2026-08-25: every channel value equal on both test slides,
# and patch counts matching the feature cache exactly (616 and 1228).
#
# Do NOT copy these from jobs/convert_ndpi.sh, which is wrong twice over: it
# points at a non-existent NDPI directory AND uses --ndpi-scale 1.0, which
# yields twice the linear resolution and roughly 4x the patch count.
#
# --ndpi-level 1 was tested and REJECTED (job 1647619). It gives identical
# DIMENSIONS, since level 1 is a factor-2 downsample, but different PIXELS:
# mean |diff| ~1.7-2.0, and 617/1252 patches against the cached 616/1228.
#
# Override only if verifying a differently-converted cohort.
NDPI_LEVEL="${NDPI_LEVEL:-0}"
NDPI_SCALE="${NDPI_SCALE:-0.5}"

SMOKE_TAG="${SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_BASE="$SCRATCH/verify_conversion_smoke/$SMOKE_TAG"
SMOKE_NDPI="$SMOKE_BASE/ndpi_subset"
SMOKE_PNG="$SMOKE_BASE/png"

echo "============================================================"
echo "  TIER 2 — conversion + extraction smoke test (CPU-only)"
echo "  Script rev: $SCRIPT_REV"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Slides   : $SMOKE_SLIDES"
echo "  Output   : $SMOKE_BASE   (NEW)"
echo "============================================================"

# ── Guards: never write into a data tree ─────────────────────────────────────
case "$SMOKE_BASE" in
    "$PNG_DIR"|"$PNG_DIR"/*|"$CACHE_DIR"|"$CACHE_DIR"/*|"$NDPI_DIR"|"$NDPI_DIR"/*)
        echo "ERROR: smoke output path is inside a data tree. Refusing."
        exit 1;;
esac
if [ -z "${SMOKE_TAG// }" ]; then
    echo "ERROR: SMOKE_TAG is empty. Refusing."
    exit 1
fi

MISSING=0
for D in "$NDPI_DIR" "$PNG_DIR" "$CACHE_DIR" "$ANN_DIR"; do
    echo -n "  $D : "
    if [ -d "$D" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done

# The NDPI source is the one input likely to be absent: the raw slides are large
# and are the obvious thing to have been cleared off scratch after conversion.
# Say so explicitly rather than leaving a bare NOT FOUND, because this is the
# only tier that needs them and it cannot be run without them.
if [ ! -d "$NDPI_DIR" ]; then
    echo ""
    echo "  The NDPI source directory is missing. This is the ONLY input Tier 2"
    echo "  needs that no other tier does, and Tier 2 cannot run without it."
    echo ""
    echo "  Look for the raw slides:"
    echo "    ls -d \$SCRATCH/data/*/ | head -20"
    echo "    find \$SCRATCH -maxdepth 3 -name '*.ndpi' 2>/dev/null | head"
    echo ""
    echo "  Then re-submit with the correct path, for example:"
    echo "    sbatch --export=ALL,NDPI_DIR=\$SCRATCH/data/ndpi \\"
    echo "        ~/cancer_trajectory_atlas/jobs/verify_conversion_smoke.sh"
    echo ""
    echo "  If the raw NDPIs are gone from scratch, Tier 2 cannot be run at all"
    echo "  until they are restored from wherever they are archived. That does"
    echo "  NOT invalidate Tier 1 or Tier 3, which do not touch them."
fi
DIMS_JSON="$PNG_DIR/slide_dimensions.json"
echo -n "  $DIMS_JSON : "
if [ -f "$DIMS_JSON" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
[ "$MISSING" -eq 0 ] || { echo "ERROR: missing inputs."; exit 1; }

# ── Build an NDPI subset directory ───────────────────────────────────────────
# run_all --convert globs every *.ndpi in --ndpi-dir and has NO slide filter,
# so restricting to two slides means pointing it at a directory that contains
# only those two. Symlinks, so no image data is copied.
mkdir -p "$SMOKE_NDPI" "$SMOKE_PNG"
for SLIDE in $SMOKE_SLIDES; do
    SRC="$NDPI_DIR/${SLIDE}.ndpi"
    if [ ! -f "$SRC" ]; then
        echo "ERROR: NDPI not found: $SRC"
        exit 1
    fi
    ln -sf "$SRC" "$SMOKE_NDPI/${SLIDE}.ndpi"
    echo "  linked: ${SLIDE}.ndpi"
done

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5
source ~/envs/atlas/bin/activate

# Python block-buffers stdout when it is a file rather than a TTY, so a
# long run's progress does not reach the log until the buffer flushes.
# That makes a running job look like a job that died at the last bash
# echo. Unbuffer so `tail -f` on the SLURM log shows real progress.
export PYTHONUNBUFFERED=1

cd ~

# ── Convert, with the same flags as jobs/convert_ndpi.sh ─────────────────────
# Resolution pre-check. Asks openslide what each pyramid level measures,
# reproduces run_all's dimension arithmetic for the requested (level, scale),
# and compares that against the recorded slide_dimensions.json. On a mismatch it
# searches the other levels and names one that WOULD reproduce the reference,
# so the level-1-versus-scale-0.5 question is settled by measurement.
#
# Costs a header read per slide. Fails in seconds rather than after a 40-minute
# conversion followed by four confusing assertion failures.
echo ""
echo "=== Resolution pre-check ==="
set +e
python - "$DIMS_JSON" "$SMOKE_NDPI" "$NDPI_LEVEL" "$NDPI_SCALE" $SMOKE_SLIDES <<'PRECHK'
import json
import sys

dims_path, ndpi_dir = sys.argv[1], sys.argv[2]
level, scale = int(sys.argv[3]), float(sys.argv[4])
slides = sys.argv[5:]

try:
    import openslide
except ImportError:
    print("  openslide-python not importable; cannot inspect the NDPI pyramid.")
    print("  Skipping the pre-check rather than guessing. The assertions after")
    print("  conversion remain the real test.")
    sys.exit(0)

recorded = json.load(open(dims_path))


def full_dims_for(level_dims, lvl, scl):
    """Reproduce run_all's dimension arithmetic exactly (run_all.py:119-120)."""
    if lvl >= len(level_dims):
        return None
    w, h = level_dims[lvl]
    if scl != 1.0:
        return int(w * scl), int(h * scl)
    return w, h


mismatch = False
suggestions = []

for slide in slides:
    path = "%s/%s.ndpi" % (ndpi_dir, slide)
    key = slide + "_x5.png"
    rec = recorded.get(key)
    print("  %s" % slide)
    if rec is None:
        print("    not in slide_dimensions.json; cannot check")
        continue

    try:
        osr = openslide.OpenSlide(path)
    except Exception as exc:
        print("    could not open %s (%s); skipping check" % (path, exc))
        continue

    level_dims = list(osr.level_dimensions)
    downs = list(osr.level_downsamples)
    osr.close()

    print("    openslide pyramid:")
    for i, ((w, h), d) in enumerate(zip(level_dims, downs)):
        print("      level %d: %6d x %-6d  (downsample %.2f)" % (i, w, h, d))

    want_w = rec["original_full_width"]
    want_h = rec["original_full_height"]
    got = full_dims_for(level_dims, level, scale)

    print("    recorded original_full : %d x %d" % (want_w, want_h))
    if got is None:
        print("    requested level %d does not exist in this pyramid" % level)
        mismatch = True
    else:
        print("    level %d, scale %.3f  -> %d x %d" % (level, scale, got[0], got[1]))
        if got == (want_w, want_h):
            print("    MATCH")
            continue
        mismatch = True

    # Search for a (level, scale) that does reproduce the recorded dimensions.
    found = []
    for lvl in range(len(level_dims)):
        for scl in (1.0, 0.5, 0.25):
            cand = full_dims_for(level_dims, lvl, scl)
            if cand == (want_w, want_h):
                found.append((lvl, scl))
    if found:
        print("    WOULD MATCH at: " + ", ".join(
            "--ndpi-level %d --ndpi-scale %g" % (l, sc) for l, sc in found))
        suggestions.extend(found)
    else:
        print("    NO (level, scale) in {0..%d} x {1.0, 0.5, 0.25} reproduces it."
              % (len(level_dims) - 1))
        print("    The reference PNGs may have been produced by a route this")
        print("    pipeline no longer contains. Tier 2 cannot verify them.")

if mismatch:
    print("")
    print("  The requested resolution does not reproduce the reference PNGs.")
    if suggestions:
        uniq = sorted(set(suggestions))
        l, sc = uniq[0]
        print("  Re-submit with:")
        print("    sbatch --export=ALL,NDPI_LEVEL=%d,NDPI_SCALE=%g \\" % (l, sc))
        print("        ~/cancer_trajectory_atlas/jobs/verify_conversion_smoke.sh")
        if len(uniq) > 1:
            print("  (Several candidates give identical dimensions: %s."
                  % ", ".join("L%d/s%g" % (a, b) for a, b in uniq))
            print("   They differ in pixel content, not size. Run one, then read")
            print("   the mean per-pixel difference in the report: the correct")
            print("   route is near zero, the others visibly larger.)")
    sys.exit(3)

print("")
print("  Resolution matches the reference. Proceeding to convert.")
PRECHK
PRE_RC=$?
set -e
if [ "$PRE_RC" -eq 3 ]; then
    echo ""
    echo "ABORTING before conversion. Only the symlink directory was created."
    exit 3
fi

echo ""
echo "=== Converting (--ndpi-level $NDPI_LEVEL --ndpi-scale $NDPI_SCALE) ==="
python -m cancer_trajectory_atlas.run_all \
    --convert \
    --ndpi-dir   "$SMOKE_NDPI" \
    --png-dir    "$SMOKE_PNG" \
    --ndpi-level "$NDPI_LEVEL" \
    --ndpi-scale "$NDPI_SCALE"

echo ""
echo "=== Assertions ==="

set +e
python - "$SMOKE_PNG" "$PNG_DIR" "$CACHE_DIR" "$ANN_DIR" "$SMOKE_BASE" $SMOKE_SLIDES <<'PYEOF'
"""Tier 2 assertions. Structural agreement, not byte identity.

Exit 0 = all assertions passed. 1 = at least one FAILED. 2 = at least one could
not be evaluated and none failed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

FRESH_PNG_DIR = Path(sys.argv[1])
EXISTING_PNG_DIR = Path(sys.argv[2])
CACHE_DIR = Path(sys.argv[3])
ANN_DIR = Path(sys.argv[4])
OUT_BASE = Path(sys.argv[5])
SLIDES = sys.argv[6:]

# Production extraction flags. These must match what populated the cache, which
# is jobs/run_cache_population.sh / run_per_section_v2.sh: patch 112, stride 96,
# min_roi_coverage unset, tissue filters ON. Everything else is a fn-default.
PATCH_SIZE = 112
STRIDE = 96
MIN_ROI_COVERAGE = None

# Rows per stripe for the pixel-difference diagnostic. Bounds peak memory: a
# stripe of 4096 rows at ~48000 px wide is about 590 MB as uint8 RGB, so two
# stripes plus overhead stay well inside the allocation.
STRIPE_ROWS = 4096

n_fail = 0
n_skip = 0
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def ok(label, detail=""):
    emit(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))


def fail(label, expected, actual, note=""):
    global n_fail
    n_fail += 1
    emit(f"  FAIL  {label}")
    emit(f"          expected = {expected}")
    emit(f"          actual   = {actual}")
    if note:
        emit(f"          {note}")


def skip(label, why):
    global n_skip
    n_skip += 1
    emit(f"  SKIP  {label}")
    emit(f"          {why}")


def pixel_difference_stats(path_a, path_b):
    """Compare two PNGs pixel by pixel, accumulated in row stripes.

    Returns a dict of statistics, or {"shapes": [...]} when the sizes differ and
    a per-pixel comparison is undefined.

    PNG is LOSSLESS, so any non-zero difference here proves the SOURCE pixels
    differ. It is never an encoder artifact. That makes this the decisive test
    for which downsampling route produced the reference PNGs: the correct route
    gives all-zero, and any other route gives a broad scatter of small
    differences.

    ``frac_differing`` is what separates the two failure modes the mean alone
    cannot. A few pixels differing a lot (rare decode edge cases) gives a tiny
    fraction; a different downsampling route gives a large one, because almost
    every textured pixel lands slightly differently.

    The stripe loop bounds the NUMPY working set, not total memory. PIL's
    ``crop`` calls ``load`` internally, so both PNGs are fully decoded and
    resident (roughly 6 GB each at these dimensions). Striping avoids a third
    and fourth full-size copy in int16 for the subtraction.
    """
    with Image.open(path_a) as ia, Image.open(path_b) as ib:
        if ia.size != ib.size:
            return {"shapes": [list(ia.size), list(ib.size)]}
        w, h = ia.size
        total = 0.0
        n_diff = 0
        count = 0
        max_diff = 0
        for top in range(0, h, STRIPE_ROWS):
            bot = min(top + STRIPE_ROWS, h)
            a = np.asarray(ia.crop((0, top, w, bot)).convert("RGB"), dtype=np.int16)
            b = np.asarray(ib.crop((0, top, w, bot)).convert("RGB"), dtype=np.int16)
            d = np.abs(a - b)
            total += float(d.sum())
            n_diff += int(np.count_nonzero(d))
            m = int(d.max()) if d.size else 0
            if m > max_diff:
                max_diff = m
            count += d.size
            del a, b, d
        mean_all = total / count if count else 0.0
        return {
            "n_channel_values": count,
            "mean_abs_diff": mean_all,
            "n_differing": n_diff,
            "frac_differing": (n_diff / count) if count else 0.0,
            "max_abs_diff": max_diff,
            "mean_over_differing": (total / n_diff) if n_diff else 0.0,
        }


emit("=" * 72)
emit("TIER 2 — conversion and extraction smoke test")
emit(f"fresh PNGs : {FRESH_PNG_DIR}")
emit(f"reference  : {EXISTING_PNG_DIR}")
emit("=" * 72)

# Recorded dimensions, written by whichever --convert populated MCF7_x5_cropped.
recorded = json.loads((EXISTING_PNG_DIR / "slide_dimensions.json").read_text())

# The sidecar this run just wrote. Keyed by PNG filename, same as the recorded one.
fresh_dims_path = FRESH_PNG_DIR / "slide_dimensions.json"
fresh = json.loads(fresh_dims_path.read_text()) if fresh_dims_path.exists() else {}
if not fresh:
    emit("")
    emit("  NOTE: the fresh conversion wrote no slide_dimensions.json. Dimension")
    emit("        assertions below will be evaluated against the PNG itself.")

diagnostics = {}

for slide in SLIDES:
    png_name = f"{slide}_x5.png"
    slide_stem = f"{slide}_x5"
    emit("")
    emit(f"--- {slide_stem} ---")

    fresh_png = FRESH_PNG_DIR / png_name
    if not fresh_png.exists():
        fail(f"[{slide}] conversion produced a PNG", png_name, "absent",
             "Conversion did not run or wrote a different name.")
        continue

    with Image.open(fresh_png) as im:
        fw, fh = im.size

    rec = recorded.get(png_name)
    if rec is None:
        skip(f"[{slide}] dimensions vs recorded",
             f"{png_name} absent from the recorded slide_dimensions.json")
    else:
        # 1. cropped PNG dimensions match the recorded cropped dimensions
        exp_w, exp_h = rec["cropped_width"], rec["cropped_height"]
        if (fw, fh) == (exp_w, exp_h):
            ok(f"[{slide}] cropped PNG dimensions", f"{fw} x {fh}")
        else:
            fail(f"[{slide}] cropped PNG dimensions",
                 f"{exp_w} x {exp_h}", f"{fw} x {fh}",
                 "Decode or crop geometry changed. Every ratio-to-pixel "
                 "transform downstream depends on these.")

        # 2. the documented invariant, checked on the recorded values
        ofw = rec["original_full_width"]
        if exp_w == ofw // 2:
            ok(f"[{slide}] cropped_width == original_full_width // 2",
               f"{exp_w} == {ofw} // 2")
        else:
            fail(f"[{slide}] cropped_width == original_full_width // 2",
                 f"{ofw // 2}", f"{exp_w}",
                 "The recorded sidecar violates its own invariant.")

        # and again on what this run just wrote, if it wrote anything
        f_rec = fresh.get(png_name)
        if f_rec is not None:
            if f_rec["cropped_width"] == f_rec["original_full_width"] // 2:
                ok(f"[{slide}] fresh sidecar invariant",
                   f"{f_rec['cropped_width']} == {f_rec['original_full_width']} // 2")
            else:
                fail(f"[{slide}] fresh sidecar invariant",
                     f_rec["original_full_width"] // 2, f_rec["cropped_width"])
            if f_rec["original_full_width"] != ofw:
                fail(f"[{slide}] original_full_width vs recorded",
                     ofw, f_rec["original_full_width"],
                     "openslide reports a different level-0 width than when the "
                     "reference was converted. That is a library-level change.")
            else:
                ok(f"[{slide}] original_full_width vs recorded", str(ofw))

    # 3. THE MEANINGFUL ASSERTION: patch count vs cached feature rows
    cache_file = CACHE_DIR / f"{slide_stem}_features.npy"
    if not cache_file.exists():
        skip(f"[{slide}] patch count vs cache", f"no cached features: {cache_file}")
    else:
        # Header-only read. np.load(mmap_mode="r") does not pull the array into
        # memory, so a multi-hundred-MB cache costs nothing here.
        cached_rows = int(np.load(cache_file, mmap_mode="r").shape[0])

        ann_path = None
        for cand in (ANN_DIR / f"{slide_stem}.json", ANN_DIR / f"{slide}.json",
                     ANN_DIR / f"{slide_stem}.geojson", ANN_DIR / f"{slide}.geojson"):
            if cand.exists():
                ann_path = cand
                break

        if ann_path is None:
            skip(f"[{slide}] patch count vs cache",
                 f"no annotation found in {ANN_DIR} for {slide_stem}. The cache "
                 "was built WITH ROI filtering, so an unfiltered count would not "
                 "be comparable.")
        else:
            # Ratio annotations are scaled by the FULL-NDPI width, not the
            # cropped width. Prefer the sidecar this run just wrote, since that
            # is what describes the PNG being patched; fall back to the recorded
            # one. With neither, load_roi_polygons raises ValueError, so check
            # here and skip with a message rather than dying mid-run.
            dims_src = fresh.get(png_name) or rec
            if dims_src is None or dims_src.get("original_full_width") is None:
                skip(f"[{slide}] patch count vs cache",
                     "no original_full_width available from either the fresh or "
                     "the recorded slide_dimensions.json, so ratio annotations "
                     "cannot be scaled to pixels")
                continue

            from cancer_trajectory_atlas.features.patching import (
                load_roi_polygons, get_patches_from_array,
            )
            with Image.open(fresh_png) as im:
                img_arr = np.array(im.convert("RGB"))

            roi_polys, exclude_polys = load_roi_polygons(
                str(ann_path),
                coordinate_space="ratio",
                original_full_width=dims_src["original_full_width"],
                original_full_height=dims_src.get("original_full_height"),
                cropped_w=img_arr.shape[1],
                cropped_h=img_arr.shape[0],
            )

            # No cap: the cache stores FULL uncapped features.
            patches, coords = get_patches_from_array(
                img_arr,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                image_name=slide_stem,
                roi_polygons=roi_polys,
                exclude_polygons=exclude_polys,
                min_roi_coverage=MIN_ROI_COVERAGE,
            )
            n_patches = int(len(patches))
            del img_arr, patches, coords

            if n_patches == cached_rows:
                ok(f"[{slide}] patch count vs cached rows", f"{n_patches}")
            else:
                delta = n_patches - cached_rows
                pct = 100.0 * abs(delta) / cached_rows if cached_rows else float("inf")
                if pct < 5.0:
                    note = (
                        "Difference is %+d patches (%.2f%%). A difference this "
                        "small is what BORDERLINE PATCHES FLIPPING across the "
                        "white_frac=0.70 or tissue_threshold=0.5 boundary looks "
                        "like, and that happens whenever the pixels differ at "
                        "all. Read the pixel diagnostic below before concluding "
                        "the code changed: if the PNGs are not identical, this "
                        "follows from that, and the fix is to convert by the "
                        "route that reproduces the reference exactly."
                        % (delta, pct))
                else:
                    note = (
                        "Difference is %+d patches (%.2f%%). That is far too "
                        "large for threshold-boundary flips. Extraction geometry "
                        "or a tissue filter really did change."
                        % (delta, pct))
                fail(f"[{slide}] patch count vs cached rows",
                     f"{cached_rows} (cached)", f"{n_patches} (fresh)", note)

    # 4. DIAGNOSTIC ONLY — never asserted on
    existing_png = EXISTING_PNG_DIR / png_name
    if not existing_png.exists():
        emit(f"  INFO  [{slide}] pixel diff: reference PNG absent, skipped")
    else:
        try:
            st = pixel_difference_stats(fresh_png, existing_png)
            diagnostics[slide_stem] = st
            if "shapes" in st:
                emit(f"  INFO  [{slide}] pixel diff: shapes differ {st['shapes']}, "
                     "not computed (the dimension assertion above is the real check)")
            elif st["n_differing"] == 0:
                emit(f"  INFO  [{slide}] pixel diff: IDENTICAL "
                     f"({st['n_channel_values']} channel values, all equal)")
                emit("        This is the conversion route that produced the "
                     "reference PNGs.")
            else:
                emit(f"  INFO  [{slide}] pixel diff: NOT identical")
                emit(f"        mean |diff| over all      = {st['mean_abs_diff']:.6f}")
                emit(f"        channel values differing  = {st['n_differing']} "
                     f"({100.0 * st['frac_differing']:.2f}%)")
                emit(f"        mean |diff| where differing = "
                     f"{st['mean_over_differing']:.3f}")
                emit(f"        max |diff|                = {st['max_abs_diff']}")
                emit("        PNG is lossless, so a non-zero difference means the")
                emit("        SOURCE pixels differ, not the encoder. If most values")
                emit("        differ slightly, this is a DIFFERENT DOWNSAMPLING")
                emit("        ROUTE, and the patch-count mismatch below follows from")
                emit("        it rather than from any change to the code.")
        except (OSError, ValueError) as exc:
            emit(f"  INFO  [{slide}] pixel diff not computed ({exc})")

emit("")
emit("=" * 72)
if n_fail:
    emit(f"RESULT: FAIL — {n_fail} assertion(s) failed.")
    emit("")
    emit("Read in this order:")
    emit("  1. Dimension assertions. If those fail, the resolution is wrong and")
    emit("     nothing below means anything.")
    emit("  2. The pixel diagnostic. PNG is lossless, so 'not identical' means")
    emit("     the conversion route differs from the one that made the reference.")
    emit("  3. Patch count. A sub-5% difference alongside a non-identical pixel")
    emit("     diagnostic is a consequence of 2, not evidence that the code")
    emit("     regressed. A large difference, or any difference when the pixels")
    emit("     ARE identical, is a genuine geometry or filter change.")
elif n_skip:
    emit(f"RESULT: INCOMPLETE — 0 failures, {n_skip} assertion(s) not evaluated.")
    emit("This is not a pass. Resolve the missing inputs above.")
else:
    emit("RESULT: PASS — dimensions, invariant, and patch counts all reproduce.")
emit("=" * 72)

OUT_BASE.mkdir(parents=True, exist_ok=True)
(OUT_BASE / "verify_conversion_smoke_report.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8")
(OUT_BASE / "pixel_diff_diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2), encoding="utf-8")
print(f"\nReport: {OUT_BASE / 'verify_conversion_smoke_report.txt'}")

sys.exit(1 if n_fail else (2 if n_skip else 0))
PYEOF

RC=$?
set -e

echo ""
case "$RC" in
    0) echo "verify_conversion_smoke: PASS (exit 0)";;
    1) echo "verify_conversion_smoke: FAIL (exit 1) — extraction geometry changed.";;
    2) echo "verify_conversion_smoke: INCOMPLETE (exit 2) — not a pass.";;
    *) echo "verify_conversion_smoke: unexpected exit $RC";;
esac

echo ""
echo "Scratch output kept at: $SMOKE_BASE"
echo "  The converted PNGs are large. Delete the tree once you have read the"
echo "  report, which is small and self-contained:"
echo "    rm -rf $SMOKE_BASE"
exit $RC
