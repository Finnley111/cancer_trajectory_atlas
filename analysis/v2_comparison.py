"""Task 5 — baseline vs v2, isolating the effect of each Task 1 fix.

Compares $SCRATCH/results/per_section/ (BASELINE, pre-fix) against
$SCRATCH/results/per_section_v2/ (post-fix). Both are read-only; every output
goes to a separate directory.

WHAT CHANGED BETWEEN THE RUNS
    1a  extraction failures -> nan instead of 0.0 (+ failure accounting)
    1b  texture_entropy averaged over 4 GLCM angles instead of 1
    1c  h_intensity masked to segmented nuclei; legacy value kept as
        h_intensity_wholepatch
    1d  packing_irregularity -> nan below 3 nuclei instead of 0.0
    plus: root selection masks non-finite densities explicitly, and
    compute_nuclear_density_quick now returns float64 rather than float32.

WHY THE DTYPE CHANGE MATTERS AND IS TESTED HERE
    Roots are argsort(nuclear_density)[:20]. float32 -> float64 is monotonic, so
    it cannot reorder distinct values — but it CAN split ties that float32 had
    collapsed, which changes which patches win the last root slots. With 21
    patches tied at exactly 0.0 in 2M-2 the root set is already ambiguous, so the
    root-set overlap is measured rather than assumed.

ROOT-SET COMPARISON IS SELF-VALIDATING
    v2 persists adata.uns['dpt_root_candidates']; the baseline does not, so the
    baseline set must be reconstructed with the production rule. That
    reconstruction is only trustworthy if it reproduces v2's STORED roots when
    applied to v2. This module checks that first and refuses to report a baseline
    root set if the check fails, rather than presenting a reconstruction as a
    measurement.

Usage:
    python -m cancer_trajectory_atlas.analysis.v2_comparison \
        --sections 2M-1 2M-2 \
        --baseline-dirs $SCRATCH/results/per_section/atlas_2M-1 \
                        $SCRATCH/results/per_section/atlas_2M-2 \
        --v2-dirs       $SCRATCH/results/per_section_v2/atlas_2M-1 \
                        $SCRATCH/results/per_section_v2/atlas_2M-2 \
        --output-dir    $SCRATCH/results/v2_comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .cellularity_confound import partial_spearman

FEATURES = ["nuclear_density", "mean_nuclear_area", "nc_ratio",
            "texture_entropy", "h_intensity", "packing_irregularity"]
V2_EXTRA = "h_intensity_wholepatch"

CORR_FLAG = 0.10        # |delta rho| above this is flagged prominently
PT_CARRYOVER = 0.98     # rho(baseline pt, v2 pt) at/above which LOO carries over
DIRECTIONAL_MIN = 0.15  # |rho| to call a feature directional (as in sign_flip_check)
PARTIAL_SURVIVE = 0.10  # |partial rho| at/above which a feature "survives"
N_ROOTS = 20


def _rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _load(run_dir: Path):
    import anndata as ad
    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(f"{h5ad} not found.")
    adata = ad.read_h5ad(h5ad)
    csv = run_dir / "results.csv"
    df = pd.read_csv(csv) if csv.exists() else None
    return adata, df


def check_alignment(df_b, df_v) -> dict:
    """Do the two runs contain the same patches in the same row order?

    Every downstream comparison is per-patch, so misalignment would silently
    produce meaningless numbers rather than an error. Compared on (slide_name, x, y)
    because those identify a patch independently of anything the fixes touched.
    """
    if df_b is None or df_v is None:
        return {"verified": False, "reason": "results.csv missing from one run"}
    if len(df_b) != len(df_v):
        return {"verified": False, "reason": f"row counts differ: {len(df_b)} vs {len(df_v)}"}
    keys = [c for c in ("slide_name", "x", "y") if c in df_b.columns and c in df_v.columns]
    if len(keys) < 3:
        return {"verified": False, "reason": f"cannot key on (slide_name,x,y); have {keys}"}
    same = all((df_b[k].values == df_v[k].values).all() for k in keys)
    n_diff = int(sum((df_b[k].values != df_v[k].values).sum() for k in keys))
    return {"verified": bool(same), "n_patches": int(len(df_b)),
            "n_key_mismatches": n_diff,
            "reason": "identical (slide_name, x, y) in identical order" if same
                      else "patch order or content differs between runs"}


def compare_roots(adata_b, adata_v) -> dict:
    """v2 stored its roots; the baseline did not. Validate the reconstruction rule
    against v2's stored set BEFORE using it on the baseline."""
    out: dict = {"n_roots": N_ROOTS}

    stored = adata_v.uns.get("dpt_root_candidates")
    out["v2_roots_stored"] = stored is not None
    if stored is None:
        out["verdict"] = ("v2 did not persist dpt_root_candidates, so no ground "
                          "truth exists and no comparison is possible.")
        return out
    stored = set(int(i) for i in np.asarray(stored).ravel())
    out["v2_stored_roots"] = sorted(stored)

    nd_v = adata_v.obs["nuclear_density"].values.astype(float)
    finite = np.isfinite(nd_v)
    idx = np.flatnonzero(finite)
    recon_v = set(int(i) for i in idx[np.argsort(nd_v[idx])][:N_ROOTS])
    overlap_v = len(stored & recon_v)
    out["reconstruction_reproduces_v2_stored"] = bool(overlap_v == N_ROOTS)
    out["reconstruction_overlap_with_stored"] = overlap_v

    # How ambiguous is the root set to begin with?
    thr = np.nanmax(nd_v[sorted(stored)]) if stored else np.nan
    n_tied = int(np.sum(nd_v <= thr))
    out["v2_patches_at_or_below_root_threshold"] = n_tied
    out["v2_tie_inflation"] = n_tied - N_ROOTS
    out["v2_root_density_max"] = float(thr)

    if overlap_v < N_ROOTS:
        out["verdict"] = (
            f"RECONSTRUCTION NOT VALID — applying argsort(nuclear_density)[:{N_ROOTS}] "
            f"to v2 recovers only {overlap_v}/{N_ROOTS} of its STORED roots, most "
            f"likely because {n_tied} patches tie at or below the threshold "
            f"({out['v2_tie_inflation']} more than needed). The baseline root set "
            "cannot be reconstructed reliably, so no baseline-vs-v2 root comparison "
            "is reported. Production selects on compute_nuclear_density_quick, which "
            "is never stored."
        )
        return out

    nd_b = adata_b.obs["nuclear_density"].values.astype(float)
    fb = np.isfinite(nd_b)
    ib = np.flatnonzero(fb)
    recon_b = set(int(i) for i in ib[np.argsort(nd_b[ib])][:N_ROOTS])
    out["baseline_reconstructed_roots"] = sorted(recon_b)

    shared = len(recon_b & stored)
    out["n_roots_shared"] = shared
    out["n_roots_differing"] = N_ROOTS - shared
    out["nuclear_density_arrays_identical"] = bool(
        len(nd_b) == len(nd_v) and np.allclose(nd_b, nd_v, equal_nan=True))
    out["verdict"] = (
        f"ROOT SET UNCHANGED — all {N_ROOTS} roots are the same in both runs."
        if shared == N_ROOTS else
        f"ROOT SET CHANGED — {N_ROOTS - shared} of {N_ROOTS} roots differ. With "
        f"{n_tied} patches tied at or below the threshold, the float32 -> float64 "
        "change in compute_nuclear_density_quick can split ties that were "
        "previously collapsed, reshuffling the last root slots."
    )
    return out


def compare_section(section: str, base_dir: Path, v2_dir: Path) -> dict:
    adata_b, df_b = _load(base_dir)
    adata_v, df_v = _load(v2_dir)

    res: dict = {"section": section,
                 "n_patches": {"baseline": int(adata_b.n_obs), "v2": int(adata_v.n_obs)}}

    # ── Failures (v2 only — the baseline had no accounting) ──────────────────
    ff = v2_dir / "feature_failures.json"
    if ff.exists():
        d = json.loads(ff.read_text())
        m, q = d["morphological_features"], d["nuclear_density_quick"]
        res["failures"] = {
            "baseline": "NO EQUIVALENT — the pre-fix code silently wrote 0.0",
            "v2_quick_n_failed": q["n_failed"],
            "v2_full_n_failed": m["n_failed"],
            "v2_failure_rate": m["failure_rate"],
            "v2_nan_h_intensity_empty_mask": m["n_nan_h_intensity_empty_mask"],
            "v2_nan_packing_irregularity_lt3": m["n_nan_packing_irregularity_lt3_nuclei"],
            "interpretation": (
                "ZERO extraction failures. Fix 1a therefore changed no value: it "
                "converts exceptions to nan, and no exception occurred. The "
                "zero-density DPT roots are REAL acellular patches, not crashes — "
                "which narrows the root problem to what those patches actually "
                "contain, not to broken code."
                if m["n_failed"] == 0 and q["n_failed"] == 0 else
                f"{m['n_failed']} extraction failure(s) in the full pass and "
                f"{q['n_failed']} in the quick pass; under the pre-fix code these "
                "were 0.0 and would have been promoted toward the root set."
            ),
        }

    res["alignment"] = check_alignment(df_b, df_v)

    # ── Headline: did the axis move? ─────────────────────────────────────────
    pt_b = adata_b.obs["pseudotime"].values.astype(float)
    pt_v = adata_v.obs["pseudotime"].values.astype(float)
    if res["alignment"]["verified"] and len(pt_b) == len(pt_v):
        r_pt = _rho(pt_b, pt_v)
        res["pseudotime"] = {
            "spearman_baseline_vs_v2": r_pt,
            "max_abs_difference": float(np.nanmax(np.abs(pt_b - pt_v))),
            "median_abs_difference": float(np.nanmedian(np.abs(pt_b - pt_v))),
            "identical": bool(np.allclose(pt_b, pt_v, equal_nan=True)),
            "loo_carries_over": bool(np.isfinite(r_pt) and r_pt >= PT_CARRYOVER),
            "verdict": (
                f"AXIS UNCHANGED (rho = {r_pt:.6f} >= {PT_CARRYOVER}). The fixes did "
                "not materially alter the pseudotime, so existing LOO results carry "
                "over and LOO does not need re-running."
                if np.isfinite(r_pt) and r_pt >= PT_CARRYOVER else
                f"AXIS MOVED (rho = {r_pt:.6f} < {PT_CARRYOVER}). LOO was measured on "
                "a different ordering and must be re-run against v2."
            ),
        }
    else:
        res["pseudotime"] = {
            "comparable": False,
            "reason": "patch alignment not verified — a per-patch correlation "
                      "between misaligned runs would be meaningless",
        }

    # ── Feature correlations ─────────────────────────────────────────────────
    feats = {}
    flagged = []
    for f in FEATURES:
        rb = _rho(pt_b, adata_b.obs[f].values.astype(float)) if f in adata_b.obs else float("nan")
        rv = _rho(pt_v, adata_v.obs[f].values.astype(float)) if f in adata_v.obs else float("nan")
        d = abs(rb - rv) if np.isfinite(rb) and np.isfinite(rv) else float("nan")
        entry = {"baseline_rho": rb, "v2_rho": rv, "abs_difference": d,
                 "directional_baseline": bool(np.isfinite(rb) and abs(rb) >= DIRECTIONAL_MIN),
                 "directional_v2": bool(np.isfinite(rv) and abs(rv) >= DIRECTIONAL_MIN)}
        entry["directionality_changed"] = (
            entry["directional_baseline"] != entry["directional_v2"])
        entry["flagged"] = bool(np.isfinite(d) and d > CORR_FLAG)
        if entry["flagged"]:
            flagged.append(f)
        feats[f] = entry

    # h_intensity both ways, so Fix 1c is isolable from 1a and 1b.
    if V2_EXTRA in adata_v.obs:
        rv_w = _rho(pt_v, adata_v.obs[V2_EXTRA].values.astype(float))
        rb_h = feats["h_intensity"]["baseline_rho"]
        feats[V2_EXTRA] = {
            "baseline_rho": rb_h,
            "v2_rho": rv_w,
            "abs_difference": abs(rb_h - rv_w) if np.isfinite(rb_h) and np.isfinite(rv_w) else float("nan"),
            "note": (
                "Legacy whole-patch definition, recomputed in v2. Its baseline "
                "column is the baseline h_intensity, which used the SAME "
                "definition — so this row isolates fixes 1a+1b (which should leave "
                "it unchanged), while the h_intensity row above carries 1a+1b+1c. "
                "The gap between the two rows is the effect of masking to nuclei."
            ),
        }
        res["fix_1c_isolated"] = {
            "h_intensity_masked_v2": feats["h_intensity"]["v2_rho"],
            "h_intensity_wholepatch_v2": rv_w,
            "difference_due_to_masking": (
                abs(feats["h_intensity"]["v2_rho"] - rv_w)
                if np.isfinite(feats["h_intensity"]["v2_rho"]) and np.isfinite(rv_w)
                else float("nan")),
        }

    res["feature_correlations"] = feats
    res["flagged_features"] = flagged

    # ── Cellularity confound, recomputed identically for both runs ───────────
    # NOT via analyze_run_nuclear_density, which writes into the run directory —
    # that would modify the baseline.
    conf = {}
    for label, adata, pt in (("baseline", adata_b, pt_b), ("v2", adata_v, pt_v)):
        nd = adata.obs["nuclear_density"].values.astype(float)
        per = {}
        surv, coll, unc = [], [], []
        for f in [x for x in FEATURES if x != "nuclear_density"]:
            if f not in adata.obs:
                continue
            v = adata.obs[f].values.astype(float)
            p = partial_spearman(pt, v, nd)
            if not np.isfinite(p):
                st = "UNCOMPUTABLE"; unc.append(f)
            elif abs(p) >= PARTIAL_SURVIVE:
                st = "SURVIVES"; surv.append(f)
            else:
                st = "collapses"; coll.append(f)
            per[f] = {"raw_rho": _rho(pt, v), "partial_rho": p, "status": st}
        conf[label] = {"per_feature": per, "survivors": surv,
                       "collapses": coll, "uncomputable": unc}

    changed = [f for f in conf["baseline"]["per_feature"]
               if f in conf["v2"]["per_feature"]
               and conf["baseline"]["per_feature"][f]["status"]
               != conf["v2"]["per_feature"][f]["status"]]
    conf["verdicts_changed"] = changed
    conf["verdict"] = (
        "NO VERDICT CHANGED — the same features survive and collapse in both runs."
        if not changed else
        f"VERDICT CHANGED for {len(changed)} feature(s): " + ", ".join(
            f"{f} ({conf['baseline']['per_feature'][f]['status']} -> "
            f"{conf['v2']['per_feature'][f]['status']})" for f in changed)
    )
    res["cellularity_confound"] = conf

    res["roots"] = compare_roots(adata_b, adata_v)
    return res


def cross_section_signs(results: dict) -> dict:
    """Does the cross-section directionality mismatch persist in v2?"""
    secs = list(results)
    if len(secs) != 2:
        return {"applicable": False, "reason": f"needs exactly 2 sections, got {len(secs)}"}
    a, b = secs
    rows, mismatch_b, mismatch_v = [], [], []
    for f in FEATURES:
        fa = results[a]["feature_correlations"].get(f, {})
        fb = results[b]["feature_correlations"].get(f, {})
        mb = fa.get("directional_baseline") != fb.get("directional_baseline")
        mv = fa.get("directional_v2") != fb.get("directional_v2")
        if mb:
            mismatch_b.append(f)
        if mv:
            mismatch_v.append(f)
        rows.append({
            "feature": f,
            f"{a}_baseline": fa.get("baseline_rho"), f"{a}_v2": fa.get("v2_rho"),
            f"{b}_baseline": fb.get("baseline_rho"), f"{b}_v2": fb.get("v2_rho"),
            "directionality_mismatch_baseline": bool(mb),
            "directionality_mismatch_v2": bool(mv),
        })
    return {
        "applicable": True, "sections": [a, b], "per_feature": rows,
        "mismatched_baseline": mismatch_b, "mismatched_v2": mismatch_v,
        "verdict": (
            f"PERSISTS — {len(mismatch_v)} feature(s) still directional in one "
            f"section only ({', '.join(mismatch_v)}). The fixes did not reconcile "
            "the two sections."
            if mismatch_v else
            "RESOLVED — no feature is directional in only one section under v2. "
            "The cross-section disagreement was an artifact of the old feature "
            "definitions."
        ) + (f" Baseline had {len(mismatch_b)} ({', '.join(mismatch_b) or 'none'})."),
    }


def write_report(out_dir: Path, results: dict, cross: dict) -> None:
    L = ["# Baseline vs v2 — effect of the Task 1 feature fixes", "",
         "Baseline = `per_section/` (pre-fix). v2 = `per_section_v2/` (post-fix).",
         "Both read-only; this report is the only output.", ""]

    for s, r in results.items():
        L += [f"## {s}", ""]
        fl = r.get("failures")
        if fl:
            L += ["### Extraction failures (new information — no baseline equivalent)", "",
                  f"- quick pass: **{fl['v2_quick_n_failed']}** failed",
                  f"- full pass: **{fl['v2_full_n_failed']}** failed "
                  f"({fl['v2_failure_rate']:.3%})",
                  f"- `h_intensity` nan from an empty nuclear mask: {fl['v2_nan_h_intensity_empty_mask']}",
                  f"- `packing_irregularity` nan from <3 nuclei: {fl['v2_nan_packing_irregularity_lt3']}",
                  "", fl["interpretation"], ""]

        p = r.get("pseudotime", {})
        if p.get("comparable") is False:
            L += ["### Pseudotime", "", f"NOT COMPARABLE — {p['reason']}", ""]
        else:
            L += ["### Pseudotime (headline)", "",
                  f"- Spearman(baseline, v2) = **{p['spearman_baseline_vs_v2']:.6f}**",
                  f"- identical arrays: {p['identical']}",
                  f"- max |difference|: {p['max_abs_difference']:.3e}", "",
                  f"**{p['verdict']}**", ""]

        L += ["### Feature correlations with pseudotime", "",
              "| Feature | baseline | v2 | \\|diff\\| | flag |", "|---|---|---|---|---|"]
        for f, e in r["feature_correlations"].items():
            flag = "**CHANGED >0.1**" if e.get("flagged") else (
                "directionality changed" if e.get("directionality_changed") else "")
            L.append(f"| `{f}` | {e['baseline_rho']:+.3f} | {e['v2_rho']:+.3f} | "
                     f"{e['abs_difference']:.3f} | {flag} |")
        if r.get("flagged_features"):
            L += ["", f"**Flagged (|delta| > {CORR_FLAG}): "
                      f"{', '.join(r['flagged_features'])}**"]
        if "fix_1c_isolated" in r:
            i = r["fix_1c_isolated"]
            L += ["", f"Fix 1c isolated: masked `h_intensity` = "
                      f"{i['h_intensity_masked_v2']:+.3f} vs whole-patch "
                      f"{i['h_intensity_wholepatch_v2']:+.3f} "
                      f"(difference {i['difference_due_to_masking']:.3f}), both from v2."]

        c = r["cellularity_confound"]
        L += ["", "### Cellularity confound", "",
              f"- baseline survivors: {c['baseline']['survivors'] or 'none'}",
              f"- v2 survivors: {c['v2']['survivors'] or 'none'}",
              "", f"**{c['verdict']}**", "",
              "### DPT roots", "", f"{r['roots']['verdict']}", ""]

    if cross.get("applicable"):
        a, b = cross["sections"]
        L += ["## Cross-section sign disagreement", "",
              f"| Feature | {a} base | {a} v2 | {b} base | {b} v2 | mismatch v2 |",
              "|---|---|---|---|---|---|"]
        for row in cross["per_feature"]:
            L.append(
                f"| `{row['feature']}` | {row[f'{a}_baseline']:+.3f} | {row[f'{a}_v2']:+.3f} | "
                f"{row[f'{b}_baseline']:+.3f} | {row[f'{b}_v2']:+.3f} | "
                f"{'YES' if row['directionality_mismatch_v2'] else '-'} |")
        L += ["", f"**{cross['verdict']}**", ""]

    (out_dir / "v2_comparison_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--baseline-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--v2-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not (len(args.sections) == len(args.baseline_dirs) == len(args.v2_dirs)):
        ap.error("--sections, --baseline-dirs and --v2-dirs must match in length and order")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Task 5 — baseline vs v2")
    print("=" * 64)

    results = {}
    for s, bd, vd in zip(args.sections, args.baseline_dirs, args.v2_dirs):
        print(f"\n  {s}\n    baseline: {bd}\n    v2:       {vd}")
        results[s] = compare_section(s, Path(bd), Path(vd))
        p = results[s]["pseudotime"]
        print(f"    {p.get('verdict', p.get('reason'))}")
        print(f"    {results[s]['roots']['verdict']}")
        print(f"    {results[s]['cellularity_confound']['verdict']}")

    cross = cross_section_signs(results)
    payload = {"analysis": "v2_comparison",
               "thresholds": {"correlation_flag": CORR_FLAG,
                              "pseudotime_loo_carryover": PT_CARRYOVER,
                              "directional_min_abs_rho": DIRECTIONAL_MIN,
                              "partial_survive_abs_rho": PARTIAL_SURVIVE},
               "per_section": results, "cross_section_signs": cross}

    with open(args.output_dir / "v2_comparison.json", "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: None if isinstance(o, float) else str(o))
    write_report(args.output_dir, results, cross)

    # Figure: baseline vs v2 correlation per feature, per section.
    try:
        fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 4.4),
                                 squeeze=False)
        for ax, (s, r) in zip(axes[0], results.items()):
            fs = [f for f in FEATURES if f in r["feature_correlations"]]
            x = np.arange(len(fs))
            w = 0.38
            ax.bar(x - w / 2, [r["feature_correlations"][f]["baseline_rho"] for f in fs],
                   w, label="baseline", color="#4878CF")
            ax.bar(x + w / 2, [r["feature_correlations"][f]["v2_rho"] for f in fs],
                   w, label="v2", color="#D65F5F")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([f.replace("_", "\n") for f in fs], fontsize=7)
            ax.set_ylabel("Spearman rho with pseudotime")
            ax.set_title(s, fontsize=10)
            ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(args.output_dir / f"v2_feature_correlations.{ext}", dpi=170)
        plt.close(fig)
    except Exception as exc:
        print(f"  WARNING: figure failed: {exc}")

    print(f"\n  JSON:     {args.output_dir / 'v2_comparison.json'}")
    print(f"  Markdown: {args.output_dir / 'v2_comparison_report.md'}")
    if cross.get("applicable"):
        print(f"\n  CROSS-SECTION: {cross['verdict']}")


if __name__ == "__main__":
    main()
