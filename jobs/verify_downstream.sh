#!/bin/bash
# TIER 3 — re-run the analysis branches against the Tier 1 regression output
#          and assert they reproduce recorded values.
#
# PURPOSE
#   Tier 1 proves the pipeline produces identical pseudotime and features.
#   Tier 3 proves the ANALYSIS BRANCHES built on top of it still land on the
#   numbers in docs/ANCHOR_VALIDATION_RECORD.md. It is the difference between
#   "the pipeline is unchanged" and "the conclusions are unchanged".
#
#   Strictly speaking Tier 3 is redundant IF Tier 1 passes cleanly: identical
#   inputs into deterministic analysis give identical outputs. It is worth
#   running anyway because it exercises code Tier 1 does not touch, and because
#   a Tier 3 failure after a Tier 1 pass localises the bug to the analysis layer
#   immediately.
#
# WHAT IS ASSERTED, and against what
#   holeyness v1  rho(pt, hole_pct)          2M-1 +0.2763   2M-2 +0.1906
#   holeyness v2  partial rho(pt,hole|area)  2M-1 +0.1315   2M-2 +0.2379
#   paired section comparison                 rho(pt,hole) diff +0.0392, p 0.578
#                                             rho(area,pt) diff +0.5236, p <= 0.0078
#   cellularity confound                      verdicts, compared to per_section_v2
#
#   Source for every recorded value: docs/ANCHOR_VALIDATION_RECORD.md
#   sections 3.2 (table), 3.11 (paired table). They are quoted there to 4 dp, so
#   the tolerance below is 1e-4. Asserting tighter than the recorded precision
#   would fail on rounding rather than on a regression.
#
# WHAT CANNOT BE RE-RUN AUTOMATICALLY, and why — reported, not skipped silently
#
#   1. THE 16-SLIDE CROSS-SECTION LOO ARRAY (jobs/submit_loo_array.sh).
#      NOT re-runnable against this output, and not attempted. It is a different
#      configuration on a different cohort: 16 slides rather than 8, --harmony
#      with --harmony-key section_number rather than --batch-method none,
#      --n-permutations 200, no --cap-strategy median, and its reference is
#      $SCRATCH/results/atlas_none_harmony, not per_section_v2. Pointing it at
#      the Tier 1 output would not be a regression test of anything.
#      It also carries the matched-pair leakage recorded as KNOWN_ISSUES §1.3:
#      holding out 6027-4L-2M-1 leaves its partner 6027-4L-2M-2 in training, so
#      re-running it would faithfully reproduce a number already known to be
#      optimistically biased.
#
#   2. THE WITHIN-SECTION SLIDE LOO (8 folds per section).
#      RE-RUNNABLE and run below when LOO=1, because it uses exactly this
#      configuration. But there is NO RECORDED BASELINE to assert against:
#      per_section_v2 never ran LOO, so nothing was recorded for it. This script
#      therefore RUNS it and REPORTS the concordance values without asserting.
#      Off by default because it is the dominant cost, roughly 8 folds x 2
#      sections x ~35 min.
#
#   3. ROOT-LEAVE-ONE-OUT CONCORDANCE (2M-1 0.726, 2M-2 0.478).
#      These come from analysis/anchor_area_control.py, NOT from the slide LOO,
#      and the two are easy to confuse. That module needs BOTH a v2 tree and a
#      holeyroot tree (--holeyroot-dirs), and this job builds only the former.
#      Run separately with jobs/run_anchor_area_control.sh, pointing --v2-dirs
#      at the Tier 1 output, once the holeyroot tree is available.
#
# ORDER MATTERS
#   holeyness v1 writes holeyness_per_duct.csv, which BOTH v2 (as
#   --v1-per-duct-csv) and the paired comparison read. Running the paired
#   comparison first would find no per-duct tables.
#
# WALLTIME / MEMORY — NOT MEASURED
#   No sacct record exists for this script. The only basis is that the holeyness
#   and cellularity jobs each request 1000 permutations, and their own scripts
#   request modest resources. The request below is reasoned, not measured, and
#   is dominated by the optional LOO. Without LOO this should finish in well
#   under an hour; with it, allow the full 12 h.
#       sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#             --name=verify_downstream
#
# READS (READ-ONLY): $SCRATCH/results/verify_regression/<TAG>
#                    $SCRATCH/results/per_section_v2  (cellularity baseline only)
#                    $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt
#                    $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json
#                    data/annotations_ratio
# WRITES (NEW ONLY): $SCRATCH/results/verify_regression/<TAG>/downstream/
#                    and cellularity_confound/ inside each atlas_<section> dir
#
# Usage:
#   sbatch --export=ALL,VERIFY_TAG=<tag> ~/cancer_trajectory_atlas/jobs/verify_downstream.sh
#   sbatch --export=ALL,VERIFY_TAG=<tag>,LOO=1 ~/.../jobs/verify_downstream.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=verify_downstream
#SBATCH --output=logs/verify_downstream-%j.out

set -euo pipefail

mkdir -p logs

REPO="$HOME/cancer_trajectory_atlas"
V2_BASE="$SCRATCH/results/per_section_v2"
EXPORT_FILE="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json"
CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
N_PERMUTATIONS=1000

if [ -z "${VERIFY_TAG:-}" ]; then
    echo "ERROR: VERIFY_TAG is not set."
    echo "  Available verification runs:"
    ls -1 "$SCRATCH/results/verify_regression/" 2>/dev/null | sed 's/^/    /' \
        || echo "    (none found)"
    exit 2
fi

NEW_BASE="$SCRATCH/results/verify_regression/$VERIFY_TAG"
DOWN="$NEW_BASE/downstream"

echo "============================================================"
echo "  TIER 3 — downstream analysis reproduction"
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Candidate : $NEW_BASE"
echo "  Output    : $DOWN   (NEW)"
echo "  Slide LOO : ${LOO:-0}  (1 = run it, no baseline to assert against)"
echo "============================================================"

if [ ! -d "$NEW_BASE" ]; then
    echo "ERROR: candidate tree not found: $NEW_BASE"
    echo "  Run jobs/verify_regression.sh first."
    exit 2
fi

# ── Guard: never write into a published tree ─────────────────────────────────
case "$DOWN" in
    "$V2_BASE"|"$V2_BASE"/*)
        echo "ERROR: downstream output would land inside per_section_v2. Refusing."
        exit 1;;
esac

echo ""
echo "=== Pre-run checks ==="
MISSING=0
for P in "$EXPORT_FILE" "$SLIDE_DIMS"; do
    echo -n "  $P : "
    if [ -e "$P" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
for D in "$ANN_DIR" "$REPO/jobs"; do
    echo -n "  $D : "
    if [ -d "$D" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
for S in 2M-1 2M-2; do
    echo -n "  $NEW_BASE/atlas_$S/results.csv : "
    if [ -f "$NEW_BASE/atlas_$S/results.csv" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: missing inputs. Nothing was run."
    exit 2
}

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate

# Python block-buffers stdout when it is a file rather than a TTY, so a
# long run's progress does not reach the log until the buffer flushes.
# That makes a running job look like a job that died at the last bash
# echo. Unbuffer so `tail -f` on the SLURM log shows real progress.
export PYTHONUNBUFFERED=1

export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$DOWN"
cd ~

# ── Step 1: holeyness v1, both sections ──────────────────────────────────────
# Must precede v2 and the paired comparison: it writes holeyness_per_duct.csv,
# which both of them read.
for SECTION in 2M-1 2M-2; do
    echo ""
    echo "=== [1/4] Holeyness v1 — section $SECTION ==="
    if [ "$SECTION" = "2M-1" ]; then
        SLIDE_LIST="$REPO/jobs/slides_section1.txt"
    else
        SLIDE_LIST="$REPO/jobs/slides_section2.txt"
    fi

    python -m cancer_trajectory_atlas.analysis.holeyness \
        --section          "$SECTION" \
        --export           "$EXPORT_FILE" \
        --annotation-dir   "$ANN_DIR" \
        --slide-dimensions "$SLIDE_DIMS" \
        --results          "$NEW_BASE/atlas_$SECTION/results.csv" \
        --output-dir       "$DOWN/holeyness/$SECTION" \
        --slide-list       "$SLIDE_LIST" \
        --n-permutations   "$N_PERMUTATIONS"
done

# ── Step 2: holeyness v2, both sections ──────────────────────────────────────
for SECTION in 2M-1 2M-2; do
    echo ""
    echo "=== [2/4] Holeyness v2 (area-adjusted) — section $SECTION ==="
    if [ "$SECTION" = "2M-1" ]; then
        SLIDE_LIST="$REPO/jobs/slides_section1.txt"
    else
        SLIDE_LIST="$REPO/jobs/slides_section2.txt"
    fi

    python -m cancer_trajectory_atlas.analysis.holeyness \
        --section          "$SECTION" \
        --export           "$EXPORT_FILE" \
        --annotation-dir   "$ANN_DIR" \
        --slide-dimensions "$SLIDE_DIMS" \
        --results          "$NEW_BASE/atlas_$SECTION/results.csv" \
        --output-dir       "$DOWN/holeyness/$SECTION/v2_area_adjusted" \
        --slide-list       "$SLIDE_LIST" \
        --v2 \
        --v1-per-duct-csv  "$DOWN/holeyness/$SECTION/holeyness_per_duct.csv" \
        --n-permutations   "$N_PERMUTATIONS"
done

# ── Step 3: paired between-section comparison ────────────────────────────────
echo ""
echo "=== [3/4] Paired section comparison (8 pairs, exact 2^8 sign-flip) ==="
python -m cancer_trajectory_atlas.analysis.holeyness_paired_comparison \
    --sections      2M-1 2M-2 \
    --per-duct-csvs "$DOWN/holeyness/2M-1/holeyness_per_duct.csv" \
                    "$DOWN/holeyness/2M-2/holeyness_per_duct.csv" \
    --output-dir    "$DOWN/paired_comparison"

# ── Step 4: cellularity confound, both sections ──────────────────────────────
# Writes into each atlas_<section>/cellularity_confound/, which is where
# verify_compare.sh's check 6 looks for the candidate side.
for SECTION in 2M-1 2M-2; do
    echo ""
    echo "=== [4/4] Cellularity confound — section $SECTION ==="
    python -m cancer_trajectory_atlas.analysis.cellularity_confound \
        --mode            partial \
        --results-dirs    "$NEW_BASE/atlas_$SECTION" \
        --n-permutations  "$N_PERMUTATIONS"
done

# ── Optional: within-section slide LOO ───────────────────────────────────────
if [ "${LOO:-0}" = "1" ]; then
    echo ""
    echo "============================================================"
    echo "  Within-section slide LOO — RUN, BUT NOT ASSERTED"
    echo "  per_section_v2 never ran LOO, so there is no recorded"
    echo "  baseline. Values below are reported for inspection only."
    echo "============================================================"

    SLIDES_2M_1=(6027-4L-2M-1_x5 6027-4R-2M-1_x5 6028-4L-2M-1_x5 6028-4R-2M-1_x5
                 6029-4L-2M-1_x5 6029-4R-2M-1_x5 6031-4L-2M-1_x5 6031-4R-2M-1_x5)
    SLIDES_2M_2=(6027-4L-2M-2_x5 6027-4R-2M-2_x5 6028-4L-2M-2_x5 6028-4R-2M-2_x5
                 6029-4L-2M-2_x5 6029-4R-2M-2_x5 6031-4L-2M-2_x5 6031-4R-2M-2_x5)

    for SECTION in 2M-1 2M-2; do
        if [ "$SECTION" = "2M-1" ]; then S=("${SLIDES_2M_1[@]}"); else S=("${SLIDES_2M_2[@]}"); fi
        OUT_DIR="$NEW_BASE/atlas_$SECTION"

        # Phase B pairs projected pseudotime against results.csv patch by patch,
        # so the held-out slide must be loaded with the FULL run's cap, not the
        # LOO training cap. The training median is over 7 slides, not 8, so the
        # two differ.
        FULL_RUN_CAP=$(cat "$OUT_DIR/active_cap.txt")
        echo "  [$SECTION] full-run active cap: $FULL_RUN_CAP"

        LOO_DIRS=()
        for HELD_OUT in "${S[@]}"; do
            echo ""
            echo "  --- LOO fold: held-out = $HELD_OUT ---"
            TRAINING=()
            for X in "${S[@]}"; do [ "$X" != "$HELD_OUT" ] && TRAINING+=("$X"); done
            TRAINING_CSV=$(IFS=,; echo "${TRAINING[*]}")

            LOO_OUT="$DOWN/loo/${SECTION}_${HELD_OUT}"
            mkdir -p "$LOO_OUT"

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
                --leiden-resolution   0.5 \
                --n-roots             20 \
                --n-permutations      200 \
                --features-cache-dir  "$CACHE_DIR" \
                --cap-strategy        median \
                --slides              "$TRAINING_CSV"

            CAP_ARGS=()
            if [ "$FULL_RUN_CAP" -gt 0 ]; then
                CAP_ARGS=(--max-patches-per-slide "$FULL_RUN_CAP")
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

        python -m cancer_trajectory_atlas.analysis.loo_summary \
            --loo-dirs   "${LOO_DIRS[@]}" \
            --output-dir "$DOWN/loo/summary_$SECTION"
    done
else
    echo ""
    echo "  Within-section slide LOO: SKIPPED (set LOO=1 to run)."
    echo "  It has no recorded baseline to assert against; see the header."
fi

# ── Assertions ───────────────────────────────────────────────────────────────
echo ""
echo "=== Asserting against recorded values ==="

set +e
python - "$DOWN" "$NEW_BASE" "$V2_BASE" <<'PYEOF'
"""Assert the Tier 3 outputs reproduce the values in ANCHOR_VALIDATION_RECORD.md.

Exit 0 = all assertions passed. 1 = at least one FAILED. 2 = at least one could
not be evaluated and none failed.
"""
import json
import sys
from pathlib import Path

DOWN, NEW_BASE, V2_BASE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

# Recorded values are quoted to 4 dp in ANCHOR_VALIDATION_RECORD.md, so a
# tolerance tighter than that would fail on rounding rather than on a
# regression. 1e-4 is the precision of the record itself.
TOL = 1e-4

# ANCHOR_VALIDATION_RECORD.md section 3.2, primary table.
RECORDED_V1 = {"2M-1": 0.2763, "2M-2": 0.1906}
RECORDED_PARTIAL_AREA = {"2M-1": 0.1315, "2M-2": 0.2379}

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
    emit(f"          recorded = {expected}")
    emit(f"          actual   = {actual}")
    if note:
        emit(f"          {note}")


def skip(label, why):
    global n_skip
    n_skip += 1
    emit(f"  SKIP  {label}")
    emit(f"          {why}")


def dig(obj, *path):
    """Follow an explicit key path. Returns (value, None) or (None, reason).

    Explicit paths rather than a recursive key search, deliberately. The key
    `rho_pt_hole_pct` appears at half a dozen places in the v1 JSON (per-slide
    rows, the aggregation-sensitivity block, the within-slide block), so a
    depth-first search for it would silently return whichever one happened to
    serialise first and could assert against the wrong quantity.

    Verified against the writers as of 2026-08-24:
      v1     holeyness.py:966       result["correlations"]
      v2     holeyness.py:1184      result["area_covariate"]
      paired holeyness_paired_comparison.py:871  result["paired"][est]["test"]
    """
    cur = obj
    for i, k in enumerate(path):
        if not isinstance(cur, dict) or k not in cur:
            got = ", ".join(sorted(cur)) if isinstance(cur, dict) else type(cur).__name__
            return None, f"path {'.'.join(path[:i + 1])} not found (present: {got})"
        cur = cur[k]
    if cur is None:
        return None, f"path {'.'.join(path)} is null"
    if not isinstance(cur, (int, float)):
        return None, f"path {'.'.join(path)} is {type(cur).__name__}, not numeric"
    return float(cur), None


emit("=" * 72)
emit("TIER 3 — downstream reproduction against recorded values")
emit(f"tolerance: {TOL} (the precision the record quotes)")
emit("=" * 72)

# ── holeyness v1 ─────────────────────────────────────────────────────────────
emit("")
emit("--- holeyness v1: rho(pt, hole_pct) ---")
for section, expected in RECORDED_V1.items():
    p = DOWN / "holeyness" / section / "holeyness_validation.json"
    if not p.exists():
        skip(f"[{section}] holeyness v1 rho", f"not found: {p}")
        continue
    data = json.loads(p.read_text())
    got, why = dig(data, "correlations", "rho_pt_hole_pct")
    if got is None:
        skip(f"[{section}] holeyness v1 rho", f"{p.name}: {why}")
    elif abs(got - expected) <= TOL:
        ok(f"[{section}] holeyness v1 rho", f"{got:+.4f}")
    else:
        fail(f"[{section}] holeyness v1 rho", f"{expected:+.4f}", f"{got:+.4f}")

# ── holeyness v2 partial ─────────────────────────────────────────────────────
emit("")
emit("--- holeyness v2: partial rho(pt, hole | area) ---")
for section, expected in RECORDED_PARTIAL_AREA.items():
    p = DOWN / "holeyness" / section / "v2_area_adjusted" / "holeyness_validation_v2.json"
    if not p.exists():
        skip(f"[{section}] holeyness v2 partial rho", f"not found: {p}")
        continue
    data = json.loads(p.read_text())
    got, why = dig(data, "area_covariate", "partial_rho_pt_hole_given_area")
    if got is None:
        skip(f"[{section}] holeyness v2 partial rho", f"{p.name}: {why}")
    elif abs(got - expected) <= TOL:
        ok(f"[{section}] holeyness v2 partial rho", f"{got:+.4f}")
    else:
        fail(f"[{section}] holeyness v2 partial rho",
             f"{expected:+.4f}", f"{got:+.4f}")

# ── paired comparison ────────────────────────────────────────────────────────
emit("")
emit("--- paired section comparison ---")
p = DOWN / "paired_comparison" / "holeyness_paired_comparison.json"
if not p.exists():
    skip("paired comparison", f"not found: {p}")
else:
    data = json.loads(p.read_text())
    # Estimand keys come from ESTIMANDS in holeyness_paired_comparison.py:131.
    # They are NOT the column names used in the record's table.
    # ANCHOR_VALIDATION_RECORD.md section 3.11.
    for estimand, exp_diff in (("raw_rho_pt_hole", 0.0392),
                               ("rho_area_pseudotime", 0.5236)):
        got, why = dig(data, "paired", estimand, "test", "observed_mean_difference")
        if got is None:
            skip(f"paired {estimand} difference", why)
        elif abs(got - exp_diff) <= TOL:
            ok(f"paired {estimand} difference", f"{got:+.4f}")
        else:
            fail(f"paired {estimand} difference", f"{exp_diff:+.4f}", f"{got:+.4f}")

    # raw_rho_pt_hole is the non-significant one: recorded p = 0.578.
    got, why = dig(data, "paired", "raw_rho_pt_hole", "test", "exact_p_two_sided")
    if got is None:
        skip("paired raw_rho_pt_hole p-value", why)
    elif abs(got - 0.578) <= 1e-3:
        ok("paired raw_rho_pt_hole p-value", f"{got:.4f}")
    else:
        fail("paired raw_rho_pt_hole p-value", "0.578", f"{got:.4f}")

    # rho(area, pt) sits at the design floor, 2/256 = 0.0078. Asserted as an
    # inequality because the floor is the smallest value the test can return.
    got, why = dig(data, "paired", "rho_area_pseudotime", "test", "exact_p_two_sided")
    if got is None:
        skip("paired rho_area_pseudotime p-value", why)
    elif got <= 0.0078 + 1e-9:
        ok("paired rho_area_pseudotime p-value", f"{got:.4f} (at or below the floor)")
    else:
        fail("paired rho_area_pseudotime p-value", "<= 0.0078", f"{got:.4f}",
             "At n=8 the paired test is a sign test; 8/8 sign agreement gives "
             "exactly 0.0078. A larger p means the signs no longer all agree.")

# ── cellularity confound, candidate vs per_section_v2 ────────────────────────
emit("")
emit("--- cellularity confound verdicts vs per_section_v2 ---")
rel = Path("cellularity_confound") / "cellularity_confound.json"
for section in ("2M-1", "2M-2"):
    new_p = NEW_BASE / f"atlas_{section}" / rel
    ref_p = V2_BASE / f"atlas_{section}" / rel
    if not new_p.exists():
        skip(f"[{section}] cellularity verdicts", f"candidate absent: {new_p}")
        continue
    if not ref_p.exists():
        skip(f"[{section}] cellularity verdicts",
             f"reference absent: {ref_p}. Generate it by running "
             "jobs/run_cellularity_confound.sh against per_section_v2. Its "
             "absence says nothing about the refactor.")
        continue
    nj = json.loads(new_p.read_text()).get("summary", {})
    rj = json.loads(ref_p.read_text()).get("summary", {})
    for field in ("survivors", "collapses", "uncomputable"):
        a, b = sorted(rj.get(field, [])), sorted(nj.get(field, []))
        if a == b:
            ok(f"[{section}] cellularity {field}", ", ".join(b) or "none")
        else:
            fail(f"[{section}] cellularity {field}", a, b)

# ── LOO status, reported not asserted ────────────────────────────────────────
emit("")
emit("--- LOO ---")
loo_dir = DOWN / "loo"
if loo_dir.is_dir():
    emit("  RUN, NOT ASSERTED. Within-section slide LOO completed. There is no")
    emit("  recorded per_section_v2 LOO baseline, so no assertion is possible.")
    for section in ("2M-1", "2M-2"):
        s = loo_dir / f"summary_{section}"
        emit(f"    {section}: {s}" if s.is_dir() else f"    {section}: no summary written")
else:
    emit("  NOT RUN (LOO=1 not set). No recorded baseline exists for it anyway.")
emit("  The 16-slide cross-section LOO array is a DIFFERENT configuration and is")
emit("  deliberately not attempted here; see this script's header.")
emit("  Root-LOO concordance (2M-1 0.726 / 2M-2 0.478) comes from")
emit("  analysis/anchor_area_control.py, not from the slide LOO, and needs a")
emit("  holeyroot tree this job does not build.")

emit("")
emit("=" * 72)
if n_fail:
    emit(f"RESULT: FAIL — {n_fail} recorded value(s) not reproduced.")
    emit("If Tier 1 passed and Tier 3 failed, the regression is in the analysis")
    emit("layer, not in the pipeline.")
elif n_skip:
    emit(f"RESULT: INCOMPLETE — 0 failures, {n_skip} assertion(s) not evaluated.")
    emit("Not a pass. Each SKIP above names what was missing.")
else:
    emit("RESULT: PASS — every recorded value reproduced within tolerance.")
emit("=" * 72)

DOWN.mkdir(parents=True, exist_ok=True)
(DOWN / "verify_downstream_report.txt").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
print(f"\nReport: {DOWN / 'verify_downstream_report.txt'}")
sys.exit(1 if n_fail else (2 if n_skip else 0))
PYEOF

RC=$?
set -e

echo ""
case "$RC" in
    0) echo "verify_downstream: PASS (exit 0)";;
    1) echo "verify_downstream: FAIL (exit 1) — a recorded value moved.";;
    2) echo "verify_downstream: INCOMPLETE (exit 2) — not a pass.";;
    *) echo "verify_downstream: unexpected exit $RC";;
esac
exit $RC
