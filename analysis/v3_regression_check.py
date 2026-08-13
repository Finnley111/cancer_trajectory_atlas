"""PHASE 8 — cleanup regression check: per_section_v3_regression vs per_section_v2.

THIS IS A REGRESSION TEST, NOT A NEW SCIENTIFIC RESULT.

jobs/run_per_section_v2.sh already established reproducibility against the pre-fix
baseline. That question is closed. The only question here is whether the 2026-08
codebase cleanup (Phases 1-7: docstrings, comments, archived dead code, archived
superseded job scripts) changed pipeline behaviour. Every check below is expected
to come back IDENTICAL.

A FAIL is a regression to bisect against the Phase 1-7 commits. It is not a
finding, and the fix is never to loosen a tolerance here.

Checks, in order (0 is a precondition — the rest are meaningless without it):

  0. PATCH ALIGNMENT   same (slide_name, x, y) in the same row order
     PCA WIDTH         same number of PCA components in adata.X
  1. PSEUDOTIME RHO    Spearman(v2, v3) per section        expect exactly 1.000000
  2. PSEUDOTIME DELTA  max |v2 - v3| element-wise          expect 0.000e+00
  3. FEATURE RHOS      6 verdict features + h_intensity_wholepatch,
                       identical to 6 decimal places
  4. DPT ROOT SETS     adata.uns['dpt_root_candidates']    expect identical, 20/20
  5. FAILURE COUNTS    feature_failures.json               expect identical (0 and 0)
  6. CONFOUND VERDICTS analyze_run_nuclear_density status per feature, identical

Exit code is 0 only if every check passes on every section.

WRITES — checks 0-5 are read-only, CHECK 6 IS NOT
    Checks 0 through 5 only read from the two run trees. Check 6 calls
    analysis/cellularity_confound.py:analyze_run_nuclear_density, which writes
    its own output into <run_dir>/cellularity_confound/ for BOTH trees — including
    the per_section_v2 reference tree.

    That write is additive and the function never touches adata_full.h5ad,
    results.csv or validation.json, so the quantities this script compares are
    not at risk. But it WILL overwrite an existing
    per_section_v2/atlas_<section>/cellularity_confound/cellularity_confound.json
    if one is already there from an earlier confound run.

    If the v2 confound output is something you need to preserve, back it up first
    or pass --skip-confound and compare the confound verdicts separately.

Usage:
    python -m cancer_trajectory_atlas.analysis.v3_regression_check \
        --sections 2M-1 2M-2 \
        --v2-base  $SCRATCH/results/per_section_v2 \
        --v3-base  $SCRATCH/results/per_section_v3_regression \
        --output-dir $SCRATCH/results/v3_regression_check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# The six features that vote on the verdict, plus the retained legacy definition.
# Order matches validation/correlations.py's verdict_features list in run_all.py.
VERDICT_FEATURES = [
    "nuclear_density", "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]
EXTRA_FEATURE = "h_intensity_wholepatch"
ALL_CHECKED_FEATURES = VERDICT_FEATURES + [EXTRA_FEATURE]

RHO_DECIMALS = 6          # "identical to 6 decimal places"
EXPECTED_N_ROOTS = 20


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_run(run_dir: Path):
    """Read one run directory. Returns (adata, results_df, validation, failures)."""
    import anndata as ad

    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(f"{h5ad} not found — did that run complete?")
    adata = ad.read_h5ad(h5ad)

    csv = run_dir / "results.csv"
    df = pd.read_csv(csv) if csv.exists() else None

    vj = run_dir / "validation.json"
    validation = json.loads(vj.read_text()) if vj.exists() else None

    fj = run_dir / "feature_failures.json"
    failures = json.loads(fj.read_text()) if fj.exists() else None

    return adata, df, validation, failures


def _same_to_decimals(a, b, decimals: int) -> bool:
    """Equality at a fixed number of decimals, treating nan == nan as equal."""
    a_nan, b_nan = (a is None or not np.isfinite(a)), (b is None or not np.isfinite(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return round(float(a), decimals) == round(float(b), decimals)


def _fmt(x) -> str:
    if x is None:
        return "None"
    try:
        return "nan" if not np.isfinite(x) else f"{float(x):+.6f}"
    except TypeError:
        return str(x)


# ── check 0: preconditions ───────────────────────────────────────────────────

def check_preconditions(a2, df2, a3, df3) -> dict:
    """Patch alignment and PCA width.

    Every later check is per-patch or per-component. If the two runs do not hold
    the same patches in the same order, an element-wise pseudotime delta compares
    unrelated patches and silently returns a meaningless number rather than an
    error. This must pass before anything else is believed.

    PCA width is checked because fit_pca uses a float variance target (0.95), so
    the component count is data-dependent rather than fixed.
    """
    out: dict = {}

    if df2 is None or df3 is None:
        out["alignment"] = {"pass": False, "reason": "results.csv missing from one run"}
    elif len(df2) != len(df3):
        out["alignment"] = {"pass": False,
                            "reason": f"row counts differ: v2={len(df2)} v3={len(df3)}"}
    else:
        keys = [c for c in ("slide_name", "x", "y")
                if c in df2.columns and c in df3.columns]
        if len(keys) < 3:
            out["alignment"] = {"pass": False,
                                "reason": f"cannot key on (slide_name,x,y); have {keys}"}
        else:
            mismatches = {k: int((df2[k].values != df3[k].values).sum()) for k in keys}
            ok = all(v == 0 for v in mismatches.values())
            out["alignment"] = {
                "pass": bool(ok),
                "n_patches": int(len(df2)),
                "key_mismatches": mismatches,
                "reason": "identical (slide_name, x, y) in identical order" if ok
                          else "patch content or row order differs",
            }

    w2 = int(a2.X.shape[1]) if a2.X is not None else -1
    w3 = int(a3.X.shape[1]) if a3.X is not None else -1
    out["pca_width"] = {
        "pass": bool(w2 == w3),
        "v2_n_components": w2,
        "v3_n_components": w3,
        "reason": "identical" if w2 == w3 else "PCA component count differs",
    }
    return out


# ── checks 1 & 2: pseudotime ─────────────────────────────────────────────────

def check_pseudotime(a2, a3) -> dict:
    pt2 = np.asarray(a2.obs["pseudotime"].values, dtype=float)
    pt3 = np.asarray(a3.obs["pseudotime"].values, dtype=float)

    if pt2.shape != pt3.shape:
        return {"pass": False, "reason": f"length differs: {pt2.shape} vs {pt3.shape}"}

    both = np.isfinite(pt2) & np.isfinite(pt3)
    rho = float(spearmanr(pt2[both], pt3[both]).statistic) if both.sum() >= 10 else float("nan")

    # Element-wise delta over positions finite in BOTH; non-finite mismatches are
    # reported separately rather than being silently skipped.
    max_abs = float(np.max(np.abs(pt2[both] - pt3[both]))) if both.any() else float("nan")
    nonfinite_mismatch = int((np.isfinite(pt2) != np.isfinite(pt3)).sum())

    rho_ok = np.isfinite(rho) and round(rho, 6) == 1.000000
    delta_ok = np.isfinite(max_abs) and max_abs == 0.0

    return {
        "pass": bool(rho_ok and delta_ok and nonfinite_mismatch == 0),
        "spearman_rho": rho,
        "spearman_rho_ok": bool(rho_ok),
        "max_abs_diff": max_abs,
        "max_abs_diff_ok": bool(delta_ok),
        "n_compared": int(both.sum()),
        "n_nonfinite_mismatch": nonfinite_mismatch,
        "expected": "rho == 1.000000 and max_abs_diff == 0.000e+00",
    }


# ── check 3: feature correlations ────────────────────────────────────────────

def check_feature_correlations(v2_val, v3_val) -> dict:
    if v2_val is None or v3_val is None:
        return {"pass": False, "reason": "validation.json missing from one run"}

    c2 = v2_val.get("feature_correlations", {})
    c3 = v3_val.get("feature_correlations", {})

    per_feature, all_ok = {}, True
    for feat in ALL_CHECKED_FEATURES:
        r2 = c2.get(feat, {}).get("rho")
        r3 = c3.get(feat, {}).get("rho")
        present = feat in c2 and feat in c3
        same = present and _same_to_decimals(r2, r3, RHO_DECIMALS)
        all_ok &= bool(same)
        per_feature[feat] = {
            "pass": bool(same),
            "present_in_both": bool(present),
            "v2_rho": r2,
            "v3_rho": r3,
            "abs_delta": (abs(float(r2) - float(r3))
                          if present and r2 is not None and r3 is not None
                          and np.isfinite(r2) and np.isfinite(r3) else None),
            "counts_toward_verdict": feat in VERDICT_FEATURES,
        }

    # The headline verdict string is compared too: validation/correlations.py's
    # verdict rule is non-monotonic, so a changed string is a loud signal even
    # when individual rhos look close.
    verdict2 = (v2_val.get("summary") or {}).get("verdict")
    verdict3 = (v3_val.get("summary") or {}).get("verdict")
    verdict_ok = verdict2 == verdict3

    return {
        "pass": bool(all_ok and verdict_ok),
        "decimals": RHO_DECIMALS,
        "per_feature": per_feature,
        "verdict_pass": bool(verdict_ok),
        "v2_verdict": verdict2,
        "v3_verdict": verdict3,
    }


# ── check 4: DPT root sets ───────────────────────────────────────────────────

def check_roots(a2, a3) -> dict:
    """Both runs persist adata.uns['dpt_root_candidates'] (added in v2), so this
    is a direct set comparison — no reconstruction from a rule is needed, and
    none should be substituted if a key is missing."""
    r2 = a2.uns.get("dpt_root_candidates")
    r3 = a3.uns.get("dpt_root_candidates")

    if r2 is None or r3 is None:
        return {
            "pass": False,
            "reason": ("dpt_root_candidates absent from "
                       f"{'v2' if r2 is None else ''}{' and ' if r2 is None and r3 is None else ''}"
                       f"{'v3' if r3 is None else ''} — persistence was removed or the run predates it"),
            "v2_present": r2 is not None,
            "v3_present": r3 is not None,
        }

    s2 = sorted(int(i) for i in np.asarray(r2).ravel())
    s3 = sorted(int(i) for i in np.asarray(r3).ravel())
    overlap = len(set(s2) & set(s3))

    # A changed root ORDER with identical membership still yields identical
    # pseudotime, because compute_dpt_multi_root median-aggregates. Membership is
    # what matters; order is reported for diagnosis only.
    return {
        "pass": bool(set(s2) == set(s3) and len(s2) == len(s3)),
        "n_roots_v2": len(s2),
        "n_roots_v3": len(s3),
        "overlap": overlap,
        "expected_n_roots": EXPECTED_N_ROOTS,
        "n_roots_as_expected": bool(len(s2) == EXPECTED_N_ROOTS == len(s3)),
        "order_identical": bool([int(i) for i in np.asarray(r2).ravel()]
                                == [int(i) for i in np.asarray(r3).ravel()]),
        "only_in_v2": sorted(set(s2) - set(s3)),
        "only_in_v3": sorted(set(s3) - set(s2)),
        "n_excluded_nonfinite_v2": int(a2.uns.get("dpt_n_roots_excluded_nonfinite", -1)),
        "n_excluded_nonfinite_v3": int(a3.uns.get("dpt_n_roots_excluded_nonfinite", -1)),
    }


# ── check 5: extraction failure counts ───────────────────────────────────────

def check_failures(f2, f3) -> dict:
    if f2 is None or f3 is None:
        return {"pass": False, "reason": "feature_failures.json missing from one run"}

    def counts(d):
        return (int(d["nuclear_density_quick"]["n_failed"]),
                int(d["morphological_features"]["n_failed"]))

    q2, m2 = counts(f2)
    q3, m3 = counts(f3)

    nan2 = f2["morphological_features"].get("nan_counts_per_feature", {})
    nan3 = f3["morphological_features"].get("nan_counts_per_feature", {})

    return {
        "pass": bool(q2 == q3 and m2 == m3 and nan2 == nan3),
        "quick_n_failed": {"v2": q2, "v3": q3, "match": bool(q2 == q3)},
        "morph_n_failed": {"v2": m2, "v3": m3, "match": bool(m2 == m3)},
        "nan_counts_per_feature_match": bool(nan2 == nan3),
        "v2_nan_counts": nan2,
        "v3_nan_counts": nan3,
        "expected": "0 and 0, identical in both runs",
        "both_zero_as_expected": bool(q2 == q3 == 0 and m2 == m3 == 0),
    }


# ── check 6: cellularity confound verdicts ───────────────────────────────────

def check_confound(v2_dir: Path, v3_dir: Path, n_permutations: int) -> dict:
    """Recompute the confound analysis on both runs and compare the verdicts.

    analyze_run_nuclear_density seeds its permutation null with
    np.random.default_rng(42), so for identical input it is deterministic and the
    per-feature status strings are directly comparable.

    NOT READ-ONLY. This writes <run_dir>/cellularity_confound/ into BOTH trees,
    v2 included, overwriting any confound output already there. See the module
    docstring; --skip-confound avoids it.

    Note the verdict strings compared here are the per-feature
    SURVIVES/collapses/UNCOMPUTABLE statuses, not _decision_gate's recommendation
    string — that one is print-only, never persisted, and compares a signed rho
    with no isfinite guard (see cellularity_confound._decision_gate).
    """
    from .cellularity_confound import analyze_run_nuclear_density

    try:
        out2 = analyze_run_nuclear_density(v2_dir, n_permutations=n_permutations)
        out3 = analyze_run_nuclear_density(v3_dir, n_permutations=n_permutations)
    except Exception as exc:  # noqa: BLE001 — surfaced as a FAIL, not swallowed
        return {"pass": False, "reason": f"confound analysis raised: {type(exc).__name__}: {exc}"}

    if not out2 or not out3:
        return {"pass": False, "reason": "confound analysis returned no result for one run"}

    fr2 = out2.get("features", out2.get("feature_results", {})) or {}
    fr3 = out3.get("features", out3.get("feature_results", {})) or {}

    feats = sorted(set(fr2) | set(fr3))
    per_feature, all_ok = {}, True
    for f in feats:
        s2 = (fr2.get(f) or {}).get("status")
        s3 = (fr3.get(f) or {}).get("status")
        p2 = (fr2.get(f) or {}).get("partial_rho")
        p3 = (fr3.get(f) or {}).get("partial_rho")
        status_ok = s2 == s3 and s2 is not None
        rho_ok = _same_to_decimals(p2, p3, RHO_DECIMALS)
        all_ok &= bool(status_ok and rho_ok)
        per_feature[f] = {
            "pass": bool(status_ok and rho_ok),
            "v2_status": s2, "v3_status": s3,
            "v2_partial_rho": p2, "v3_partial_rho": p3,
        }

    return {"pass": bool(all_ok and feats), "n_permutations": n_permutations,
            "per_feature": per_feature}


# ── driver ───────────────────────────────────────────────────────────────────

def compare_section(section: str, v2_dir: Path, v3_dir: Path,
                    n_permutations: int, skip_confound: bool) -> dict:
    print(f"\n{'='*70}\n  SECTION {section}\n{'='*70}")
    print(f"  v2: {v2_dir}\n  v3: {v3_dir}")

    a2, df2, val2, fail2 = _load_run(v2_dir)
    a3, df3, val3, fail3 = _load_run(v3_dir)

    res = {"section": section, "v2_dir": str(v2_dir), "v3_dir": str(v3_dir)}
    res["check0_preconditions"] = check_preconditions(a2, df2, a3, df3)
    res["check1_2_pseudotime"] = check_pseudotime(a2, a3)
    res["check3_feature_correlations"] = check_feature_correlations(val2, val3)
    res["check4_dpt_roots"] = check_roots(a2, a3)
    res["check5_failure_counts"] = check_failures(fail2, fail3)
    res["check6_confound_verdicts"] = (
        {"pass": None, "skipped": True, "reason": "--skip-confound"} if skip_confound
        else check_confound(v2_dir, v3_dir, n_permutations)
    )

    # ── report ───────────────────────────────────────────────────────────────
    pre = res["check0_preconditions"]
    print("\n  [0] PRECONDITIONS")
    print(f"      alignment : {'PASS' if pre['alignment']['pass'] else 'FAIL'}"
          f"  ({pre['alignment'].get('reason')})")
    print(f"      pca width : {'PASS' if pre['pca_width']['pass'] else 'FAIL'}"
          f"  v2={pre['pca_width']['v2_n_components']} v3={pre['pca_width']['v3_n_components']}")
    if not (pre["alignment"]["pass"] and pre["pca_width"]["pass"]):
        print("      >>> Preconditions failed. Every check below compares "
              "non-corresponding data and means nothing.")

    pt = res["check1_2_pseudotime"]
    print("\n  [1] PSEUDOTIME SPEARMAN  expect exactly 1.000000")
    print(f"      rho = {_fmt(pt.get('spearman_rho'))}   "
          f"{'PASS' if pt.get('spearman_rho_ok') else 'FAIL'}")
    print("  [2] PSEUDOTIME MAX |DIFF|  expect 0.000e+00")
    mad = pt.get("max_abs_diff")
    print(f"      max|d| = {mad:.3e}   {'PASS' if pt.get('max_abs_diff_ok') else 'FAIL'}"
          if mad is not None and np.isfinite(mad) else "      max|d| = nan   FAIL")
    if pt.get("n_nonfinite_mismatch"):
        print(f"      >>> {pt['n_nonfinite_mismatch']} patch(es) finite in one run "
              "and not the other")

    fc = res["check3_feature_correlations"]
    print(f"\n  [3] FEATURE CORRELATIONS  identical to {RHO_DECIMALS} dp")
    if "per_feature" in fc:
        for feat, d in fc["per_feature"].items():
            tag = "verdict" if d["counts_toward_verdict"] else "extra  "
            print(f"      {'PASS' if d['pass'] else 'FAIL'}  {tag}  {feat:<24s}"
                  f" v2={_fmt(d['v2_rho'])}  v3={_fmt(d['v3_rho'])}")
        print(f"      {'PASS' if fc['verdict_pass'] else 'FAIL'}  verdict string")
        if not fc["verdict_pass"]:
            print(f"          v2: {fc['v2_verdict']}")
            print(f"          v3: {fc['v3_verdict']}")
    else:
        print(f"      FAIL  ({fc.get('reason')})")

    rt = res["check4_dpt_roots"]
    print(f"\n  [4] DPT ROOT SETS  expect identical, {EXPECTED_N_ROOTS}/{EXPECTED_N_ROOTS}")
    if "overlap" in rt:
        print(f"      overlap {rt['overlap']}/{rt['n_roots_v2']}   "
              f"{'PASS' if rt['pass'] else 'FAIL'}"
              f"{'' if rt['order_identical'] else '   (order differs; membership is what matters)'}")
        if rt["only_in_v2"] or rt["only_in_v3"]:
            print(f"      only in v2: {rt['only_in_v2']}")
            print(f"      only in v3: {rt['only_in_v3']}")
    else:
        print(f"      FAIL  ({rt.get('reason')})")

    fl = res["check5_failure_counts"]
    print("\n  [5] EXTRACTION FAILURES  expect identical, 0 and 0")
    if "quick_n_failed" in fl:
        print(f"      quick: v2={fl['quick_n_failed']['v2']} v3={fl['quick_n_failed']['v3']}   "
              f"morph: v2={fl['morph_n_failed']['v2']} v3={fl['morph_n_failed']['v3']}   "
              f"{'PASS' if fl['pass'] else 'FAIL'}")
        if not fl["both_zero_as_expected"]:
            print("      >>> counts match but are not both zero — the v2 run had "
                  "0 and 0, so investigate before accepting this.")
    else:
        print(f"      FAIL  ({fl.get('reason')})")

    cf = res["check6_confound_verdicts"]
    print("\n  [6] CELLULARITY CONFOUND VERDICTS  expect identical")
    if cf.get("skipped"):
        print("      SKIPPED (--skip-confound)")
    elif "per_feature" in cf:
        for feat, d in cf["per_feature"].items():
            print(f"      {'PASS' if d['pass'] else 'FAIL'}  {feat:<24s}"
                  f" v2={d['v2_status']}  v3={d['v3_status']}")
    else:
        print(f"      FAIL  ({cf.get('reason')})")

    checks = [res["check0_preconditions"]["alignment"]["pass"],
              res["check0_preconditions"]["pca_width"]["pass"],
              pt.get("pass", False), fc.get("pass", False),
              rt.get("pass", False), fl.get("pass", False)]
    if not cf.get("skipped"):
        checks.append(cf.get("pass", False))

    res["section_pass"] = bool(all(checks))
    print(f"\n  SECTION {section}: {'PASS — identical' if res['section_pass'] else 'FAIL — REGRESSION'}")
    return res


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 8 cleanup regression check: v3 vs per_section_v2.")
    p.add_argument("--sections", nargs="+", default=["2M-1", "2M-2"])
    p.add_argument("--v2-base", type=Path, required=True,
                   help="e.g. $SCRATCH/results/per_section_v2")
    p.add_argument("--v3-base", type=Path, required=True,
                   help="e.g. $SCRATCH/results/per_section_v3_regression")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-permutations", type=int, default=1000,
                   help="Permutations for the confound recomputation (default: 1000, "
                        "matching the production setting).")
    p.add_argument("--skip-confound", action="store_true",
                   help="Skip check 6. It is the slowest check, and it is the only "
                        "one that WRITES: it creates <run_dir>/cellularity_confound/ "
                        "in both trees, overwriting any existing confound output in "
                        "the per_section_v2 reference tree.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PHASE 8 — CLEANUP REGRESSION CHECK")
    print("  This is a regression test, not a new scientific result.")
    print("  Every check is expected to be IDENTICAL. A FAIL is a bug to bisect")
    print("  against the Phase 1-7 commits, never a result to interpret.")
    print("=" * 70)

    results, missing = [], []
    for section in args.sections:
        v2_dir = args.v2_base / f"atlas_{section}"
        v3_dir = args.v3_base / f"atlas_{section}"
        if not v2_dir.exists() or not v3_dir.exists():
            missing.append({"section": section,
                            "v2_exists": v2_dir.exists(), "v3_exists": v3_dir.exists()})
            print(f"\n  ERROR: missing run directory for section {section}: "
                  f"v2={v2_dir.exists()} v3={v3_dir.exists()}")
            continue
        results.append(compare_section(section, v2_dir, v3_dir,
                                       args.n_permutations, args.skip_confound))

    overall = bool(results) and not missing and all(r["section_pass"] for r in results)

    report = {
        "what_this_is": ("Regression test of the 2026-08 codebase cleanup. Compares "
                         "per_section_v3_regression against per_section_v2. Any "
                         "non-identical value is a regression to bisect, not a finding."),
        "sections": args.sections,
        "v2_base": str(args.v2_base),
        "v3_base": str(args.v3_base),
        "n_permutations": args.n_permutations,
        "confound_skipped": bool(args.skip_confound),
        "missing_run_dirs": missing,
        "per_section": results,
        "overall_pass": overall,
    }
    out_json = args.output_dir / "v3_regression_check.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 70)
    print(f"  OVERALL: {'PASS — cleanup changed nothing' if overall else 'FAIL — REGRESSION DETECTED'}")
    print("=" * 70)
    print(f"  Report: {out_json}")
    if not overall:
        print("\n  Bisect against the Phase 1-7 commits. Do not loosen a tolerance")
        print("  here to make this pass, and do not interpret the difference.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
