#!/bin/bash
# TIER 1 GATE — assert the regression re-run is identical to per_section_v2.
#
# PURPOSE
#   This is the script that decides whether a refactor changed a number. It
#   compares jobs/verify_regression.sh's output against the reference tree and
#   exits non-zero on any mismatch, naming the quantity and both values.
#
#   Read-only. It writes one report file into the verification tree and touches
#   nothing else.
#
# EXIT CODES — distinguish "the refactor broke something" from "I could not tell"
#   0  every comparison ran and passed
#   1  MISMATCH. At least one quantity differs. This is the failure the gate
#      exists to catch, and it means REVERT, not rationalise.
#   2  COULD NOT COMPARE. An input was missing, so at least one assertion was
#      not made. NOT a pass. A missing baseline is not evidence of identity.
#
#   The distinction matters because exit 2 usually means a setup problem on your
#   side, while exit 1 means a real regression in the code.
#
# WHAT IS ASSERTED
#   1. Spearman(v2, new) pseudotime, per section         expected exactly 1.000000
#   2. element-wise max |pseudotime difference|           expected 0.000e+00
#   3. six verdict features + h_intensity_wholepatch      identical to 6 dp
#   4. DPT root sets                                      identical, 20/20 per section
#   5. extraction failure counts                          identical, expected 0 and 0
#   6. cellularity confound verdicts                      identical
#   7. PAGA connected-component count                     identical
#   8. active_cap.txt                                     identical
#
# A NOTE ON CHECK 1 vs CHECK 2
#   They are not redundant, and check 2 is the strict one. Spearman is invariant
#   to any monotone transform, so a bug that rescaled every pseudotime value
#   identically would leave rho at exactly 1.000000 and still be a regression.
#   Check 2 catches that; check 1 catches reordering. Both must pass.
#
# A NOTE ON CHECK 6
#   The cellularity confound is NOT produced by run_all. It comes from
#   jobs/verify_downstream.sh (Tier 3) on the new side, and from whenever
#   run_cellularity_confound.sh was last run on the reference side. If either is
#   absent this check reports COULD NOT COMPARE (exit 2) rather than failing,
#   because a missing baseline says nothing about the refactor.
#
# A NOTE ON CHECK 7
#   The PAGA component count is computed, printed, and then DISCARDED by
#   run_all; it is not persisted anywhere. This script therefore recomputes it
#   from adata.uns['paga']['connectivities'], which IS stored in the h5ad, using
#   the same threshold=0.05 that analysis/diffusion.py:compute_paga_topology
#   applies. That threshold has never been exposed to the CLI.
#
# WALLTIME / MEMORY — NOT MEASURED
#   No sacct record exists for this script; it has never been run. The request
#   below is reasoned, not measured: the job loads four h5ad files and four
#   results.csv files. The per-section runs are on the order of 10^4 patches, so
#   the dominant cost is h5ad I/O, not computation. 30 min and 32 G is generous
#   for that and should be revised down after the first run:
#       sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State \
#             --name=verify_compare
#
# READS (READ-ONLY): $SCRATCH/results/per_section_v2
#                    $SCRATCH/results/verify_regression/<TAG>
# WRITES           : <TAG>/verify_compare_report.txt only
#
# Usage:
#   sbatch --export=ALL,VERIFY_TAG=<tag> ~/cancer_trajectory_atlas/jobs/verify_compare.sh
#
#   VERIFY_TAG must match the tag verify_regression.sh used. If you did not set
#   one, it was a timestamp; read it from the directory name under
#   $SCRATCH/results/verify_regression/.

#SBATCH --account=def-lmarti46
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=verify_compare
#SBATCH --output=logs/verify_compare-%j.out

set -euo pipefail

mkdir -p logs

V2_BASE="$SCRATCH/results/per_section_v2"

if [ -z "${VERIFY_TAG:-}" ]; then
    echo "ERROR: VERIFY_TAG is not set."
    echo "  Available verification runs:"
    ls -1 "$SCRATCH/results/verify_regression/" 2>/dev/null | sed 's/^/    /' \
        || echo "    (none found)"
    echo "  Submit with: sbatch --export=ALL,VERIFY_TAG=<tag> jobs/verify_compare.sh"
    exit 2
fi

NEW_BASE="$SCRATCH/results/verify_regression/$VERIFY_TAG"

echo "============================================================"
echo "  TIER 1 GATE — bit-identity assertion"
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Reference : $V2_BASE"
echo "  Candidate : $NEW_BASE"
echo "============================================================"

if [ ! -d "$V2_BASE" ]; then
    echo "COULD NOT COMPARE: reference tree not found: $V2_BASE"
    exit 2
fi
if [ ! -d "$NEW_BASE" ]; then
    echo "COULD NOT COMPARE: candidate tree not found: $NEW_BASE"
    echo "  Run jobs/verify_regression.sh first."
    exit 2
fi

module load StdEnv/2023 python/3.11 gcc openblas hdf5 igraph
source ~/envs/atlas/bin/activate

# Python block-buffers stdout when it is a file rather than a TTY, so a
# long run's progress does not reach the log until the buffer flushes.
# That makes a running job look like a job that died at the last bash
# echo. Unbuffer so `tail -f` on the SLURM log shows real progress.
export PYTHONUNBUFFERED=1

cd ~

# `set -e` would abort the script the instant python exits non-zero, before the
# exit code could be captured and reported. Disable it around the gate so a
# MISMATCH is reported as a MISMATCH rather than as an unexplained job failure.
set +e
python - "$V2_BASE" "$NEW_BASE" "$NEW_BASE/verify_compare_report.txt" <<'PYEOF'
"""Tier 1 gate. Asserts the candidate tree is identical to the reference.

Exit 0 = all checks passed. 1 = at least one MISMATCH. 2 = at least one check
could not be made and none mismatched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF_BASE, NEW_BASE, REPORT_PATH = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SECTIONS = ["2M-1", "2M-2"]

# The six features that vote on the verdict, plus the legacy definition of
# h_intensity, which is reported but does not vote. run_all.py passes exactly
# this list as verdict_features; h_intensity_wholepatch is added here because
# the brief asks for it and because it is the one feature that would reveal a
# regression in the 1c fix specifically.
VERDICT_FEATURES = [
    "nuclear_density", "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]
FEATURES_TO_CHECK = VERDICT_FEATURES + ["h_intensity_wholepatch"]

# Applied to PAGA connectivities before counting components. Must stay in step
# with analysis/diffusion.py:compute_paga_topology, which hardcodes it.
PAGA_THRESHOLD = 0.05

lines = []
n_mismatch = 0
n_uncomparable = 0


def emit(s=""):
    print(s)
    lines.append(s)


def ok(label, detail=""):
    emit(f"  PASS      {label}" + (f"   [{detail}]" if detail else ""))


def mismatch(label, ref_val, new_val, note=""):
    global n_mismatch
    n_mismatch += 1
    emit(f"  MISMATCH  {label}")
    emit(f"              reference = {ref_val}")
    emit(f"              candidate = {new_val}")
    if note:
        emit(f"              {note}")


def uncomparable(label, why):
    global n_uncomparable
    n_uncomparable += 1
    emit(f"  NO-COMPARE {label}")
    emit(f"              {why}")


def load_adata(p):
    import anndata as ad
    return ad.read_h5ad(p)


def paga_components(adata):
    """Recompute the PAGA component count from stored connectivities.

    run_all does not persist the count, only the connectivities, so this
    reproduces diffusion.py:compute_paga_topology's thresholded count.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    conn = adata.uns["paga"]["connectivities"]
    thresh = (conn > PAGA_THRESHOLD).astype(np.float32)
    n, _ = connected_components(csr_matrix(thresh), directed=False)
    return int(n)


emit("=" * 72)
emit("TIER 1 GATE — per-section bit-identity")
emit(f"reference: {REF_BASE}")
emit(f"candidate: {NEW_BASE}")
emit("=" * 72)

for section in SECTIONS:
    ref_dir, new_dir = REF_BASE / f"atlas_{section}", NEW_BASE / f"atlas_{section}"
    emit("")
    emit(f"--- section {section} ---")

    if not ref_dir.is_dir():
        uncomparable(f"[{section}] entire section", f"reference dir absent: {ref_dir}")
        continue
    if not new_dir.is_dir():
        uncomparable(f"[{section}] entire section",
                     f"candidate dir absent: {new_dir}. Was this section run? "
                     "verify_regression.sh honours ONLY_SECTION.")
        continue

    # ── results.csv: pseudotime and features ─────────────────────────────────
    ref_csv, new_csv = ref_dir / "results.csv", new_dir / "results.csv"
    if not (ref_csv.exists() and new_csv.exists()):
        uncomparable(f"[{section}] results.csv",
                     f"missing: ref={ref_csv.exists()} new={new_csv.exists()}")
    else:
        r = pd.read_csv(ref_csv)
        n = pd.read_csv(new_csv)

        if len(r) != len(n):
            # Everything below joins by row position, so a length difference
            # makes every subsequent check meaningless rather than merely failed.
            mismatch(f"[{section}] results.csv row count", len(r), len(n),
                     "Row counts differ, so per-patch comparisons below are skipped "
                     "for this section: they align by position and would compare "
                     "different patches.")
        else:
            # 1. Spearman, expected exactly 1.000000
            from scipy.stats import spearmanr
            rho = spearmanr(r["pseudotime"].values, n["pseudotime"].values).statistic
            if f"{rho:.6f}" == "1.000000":
                ok(f"[{section}] Spearman(v2, new) pseudotime", f"{rho:.6f}")
            else:
                mismatch(f"[{section}] Spearman(v2, new) pseudotime",
                         "1.000000", f"{rho:.6f}",
                         "Ordering changed. This is a hard regression.")

            # 2. max abs difference, expected exactly 0
            d = np.abs(r["pseudotime"].values - n["pseudotime"].values)
            dmax = float(np.nanmax(d)) if len(d) else 0.0
            if dmax == 0.0:
                ok(f"[{section}] max |pseudotime diff|", f"{dmax:.3e}")
            else:
                mismatch(f"[{section}] max |pseudotime diff|", "0.000e+00", f"{dmax:.3e}",
                         "Values moved even if the ordering did not. Spearman alone "
                         "would not have caught this.")

            # 3. features identical to 6 dp
            for feat in FEATURES_TO_CHECK:
                if feat not in r.columns or feat not in n.columns:
                    uncomparable(f"[{section}] feature {feat}",
                                 f"absent: ref={feat in r.columns} new={feat in n.columns}")
                    continue
                a = np.round(r[feat].values.astype(float), 6)
                b = np.round(n[feat].values.astype(float), 6)
                # nan == nan must count as agreement: nan is the documented
                # missing-value convention, not a failure to compare.
                both_nan = np.isnan(a) & np.isnan(b)
                differs = ~(both_nan | (a == b))
                n_diff = int(np.count_nonzero(differs))
                if n_diff == 0:
                    ok(f"[{section}] feature {feat}", "identical to 6 dp")
                else:
                    i = int(np.flatnonzero(differs)[0])
                    mismatch(f"[{section}] feature {feat} ({n_diff} rows differ)",
                             f"row {i} = {a[i]!r}", f"row {i} = {b[i]!r}")

    # ── adata: root set and PAGA ─────────────────────────────────────────────
    ref_h5, new_h5 = ref_dir / "adata_full.h5ad", new_dir / "adata_full.h5ad"
    if not (ref_h5.exists() and new_h5.exists()):
        uncomparable(f"[{section}] adata_full.h5ad",
                     f"missing: ref={ref_h5.exists()} new={new_h5.exists()}")
    else:
        ra, na = load_adata(ref_h5), load_adata(new_h5)

        # 4. DPT root set, identical, 20/20
        key = "dpt_root_candidates"
        if key not in ra.uns or key not in na.uns:
            uncomparable(f"[{section}] DPT root set",
                         f"uns['{key}'] absent: ref={key in ra.uns} new={key in na.uns}. "
                         "Runs before 2026-08 did not persist it.")
        else:
            rr = np.asarray(ra.uns[key]).ravel().astype(np.int64)
            nn = np.asarray(na.uns[key]).ravel().astype(np.int64)
            if rr.shape == nn.shape and np.array_equal(rr, nn):
                ok(f"[{section}] DPT root set", f"{len(nn)}/{len(rr)} identical")
            else:
                shared = len(set(rr.tolist()) & set(nn.tolist()))
                mismatch(f"[{section}] DPT root set",
                         f"n={len(rr)} {sorted(rr.tolist())}",
                         f"n={len(nn)} {sorted(nn.tolist())}",
                         f"{shared}/{len(rr)} indices shared. A changed root set "
                         "moves the pseudotime origin.")

        # 7. PAGA component count
        try:
            rc, nc = paga_components(ra), paga_components(na)
            if rc == nc:
                ok(f"[{section}] PAGA components", f"{nc} (threshold {PAGA_THRESHOLD})")
            else:
                mismatch(f"[{section}] PAGA components", rc, nc,
                         "Manifold connectivity changed, which changes whether DPT "
                         "is considered valid at all.")
        except KeyError as exc:
            uncomparable(f"[{section}] PAGA components",
                         f"uns['paga'] not usable ({exc}).")

    # ── 5. extraction failure counts ─────────────────────────────────────────
    ref_ff, new_ff = ref_dir / "feature_failures.json", new_dir / "feature_failures.json"
    if not (ref_ff.exists() and new_ff.exists()):
        uncomparable(f"[{section}] feature_failures.json",
                     f"missing: ref={ref_ff.exists()} new={new_ff.exists()}")
    else:
        rj, nj = json.loads(ref_ff.read_text()), json.loads(new_ff.read_text())
        for block in ("nuclear_density_quick", "morphological_features"):
            rv = rj.get(block, {}).get("n_failed")
            nv = nj.get(block, {}).get("n_failed")
            if rv is None or nv is None:
                uncomparable(f"[{section}] {block}.n_failed", "key absent in one tree")
            elif rv != nv:
                mismatch(f"[{section}] {block}.n_failed", rv, nv)
            elif nv != 0:
                # Identical but non-zero. Not a regression, but the reference
                # is documented as zero-failure, so say so rather than pass mute.
                ok(f"[{section}] {block}.n_failed",
                   f"{nv} — identical, but NOT the documented 0")
            else:
                ok(f"[{section}] {block}.n_failed", "0")

    # ── 8. active_cap.txt ────────────────────────────────────────────────────
    ref_cap, new_cap = ref_dir / "active_cap.txt", new_dir / "active_cap.txt"
    if not (ref_cap.exists() and new_cap.exists()):
        uncomparable(f"[{section}] active_cap.txt",
                     f"missing: ref={ref_cap.exists()} new={new_cap.exists()}")
    else:
        rv, nv = ref_cap.read_text().strip(), new_cap.read_text().strip()
        if rv == nv:
            ok(f"[{section}] active_cap.txt", rv)
        else:
            mismatch(f"[{section}] active_cap.txt", rv, nv,
                     "The cohort median patch count changed, so a different "
                     "subset of patches entered the PCA.")

    # ── 6. cellularity confound verdicts ─────────────────────────────────────
    rel = Path("cellularity_confound") / "cellularity_confound.json"
    ref_cc, new_cc = ref_dir / rel, new_dir / rel
    if not ref_cc.exists() or not new_cc.exists():
        uncomparable(
            f"[{section}] cellularity confound verdicts",
            f"missing: ref={ref_cc.exists()} new={new_cc.exists()}. "
            "Not produced by run_all. The candidate side comes from "
            "jobs/verify_downstream.sh; the reference side from "
            "jobs/run_cellularity_confound.sh pointed at per_section_v2.")
    else:
        rj, nj = json.loads(ref_cc.read_text()), json.loads(new_cc.read_text())
        rs, ns = rj.get("summary", {}), nj.get("summary", {})
        for field in ("survivors", "collapses", "uncomputable"):
            rv, nv = sorted(rs.get(field, [])), sorted(ns.get(field, []))
            if rv == nv:
                ok(f"[{section}] cellularity {field}", ", ".join(rv) or "none")
            else:
                mismatch(f"[{section}] cellularity {field}", rv, nv)

emit("")
emit("=" * 72)
if n_mismatch:
    emit(f"RESULT: FAIL — {n_mismatch} mismatch(es).")
    emit("")
    emit("A mismatch here is a REGRESSION TO BE REVERTED, not rationalised.")
    emit("The reference outputs are the specification.")
elif n_uncomparable:
    emit(f"RESULT: INCOMPLETE — 0 mismatches, but {n_uncomparable} check(s) "
         "could not be made.")
    emit("")
    emit("This is NOT a pass. A check that did not run is not a check that")
    emit("succeeded. Resolve the missing inputs listed above and re-run.")
else:
    emit("RESULT: PASS — every comparison ran and every one matched.")
emit("=" * 72)

try:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")
except OSError as exc:
    print(f"\nWARNING: could not write report ({exc}). Console output above stands.")

sys.exit(1 if n_mismatch else (2 if n_uncomparable else 0))
PYEOF

RC=$?
set -e

echo ""
case "$RC" in
    0) echo "verify_compare: PASS (exit 0)";;
    1) echo "verify_compare: MISMATCH (exit 1) — a number changed. Revert.";;
    2) echo "verify_compare: COULD NOT COMPARE (exit 2) — not a pass.";;
    *) echo "verify_compare: unexpected exit $RC — the gate itself failed to run.";;
esac
exit $RC
