"""Phase 2 report: holey-ness-rooted pseudotime vs v2, both sections.

REPORTS NUMBERS. DECLARES NEITHER ANCHOR BETTER.

WHAT COUNTS AS EVIDENCE HERE, FIXED BEFORE THE NUMBERS
------------------------------------------------------
``rho(pseudotime, hole_pct)`` will rise under this anchor. **That is partly
circular** — the anchor IS holey-ness — and is NOT evidence the anchor is better.
It is reported for completeness and labelled circular wherever it appears.

The NON-CIRCULAR tests, neither of which is used in root selection:

  rho(pseudotime, duct AREA)        The lab's biology says area grows with
                                    progression. In v2 this was ~+0.41 in 2M-1
                                    but -0.084 in 2M-2 — the sections disagreed
                                    about the mediator. Whether re-anchoring
                                    moves 2M-2 toward positive is the sharpest
                                    single discriminator available.
  rho(pseudotime, nuclear_density)  Non-circular for the FIRST TIME under this
                                    anchor, because density no longer selects the
                                    roots. In v2: +0.445 (2M-1) / -0.150 (2M-2).

Both are recomputed here at duct level against the NEW pseudotime, by reusing
``holeyness.py``'s own aggregation pipeline unmodified — so the Phase 1 numbers
and these are produced by identical code.

EXPECTATION FOR THE ORDERING
----------------------------
Uniformly random 20-root sets reproduce production pseudotime at |rho| 0.78-0.89,
so a root-rule change is EXPECTED to alter orientation and root membership and to
leave the ordering largely intact. |rho| < 0.7 vs v2 contradicts that and is
flagged prominently. A value near -1 means the axis flipped end-for-end, which is
orientation-only and not a change in ordering.

NOT TO BE READ AS VALIDATION
----------------------------
  * A reduced pseudotime_std does not validate the anchor.
  * The two sections' feature directions reconciling does not validate it either;
    both could occur for reasons unrelated to the anchor being more biologically
    correct, and the sections differ in FIXATIVE (Carnoy's vs PFA), which can
    shift hole % independent of biology.

Read-only on every run tree. Writes only --output-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .v3_root_experiment_compare import parse_pt_range, FEATURES, KEY, DELTA_FLAG
from .holeyness import (
    load_slide_list, load_slide_dimensions, parse_measurement_export,
    load_duct_polygons, build_duct_table, assign_patches_to_ducts,
    aggregate_per_duct, _safe_spearman, _partial_spearman,
)

RHO_FLOOR = 0.7
DIRECTIONAL = 0.15   # |rho| above which a feature is called directional


def _f(v, nd=4):
    if v is None:
        return "n/a"
    try:
        if v != v:
            return "nan"
    except TypeError:
        return str(v)
    return f"{v:.{nd}f}"


def load_run(d: Path) -> dict:
    csv = d / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found.")
    rec = {"dir": d, "df": pd.read_csv(csv)}
    rec["roots"] = rec["root_source"] = rec["pca_width"] = None
    h5 = d / "adata_full.h5ad"
    if h5.exists():
        try:
            import anndata as ad
            a = ad.read_h5ad(h5, backed="r")
            if "dpt_root_candidates" in a.uns:
                rec["roots"] = [int(i) for i in np.asarray(a.uns["dpt_root_candidates"]).ravel()]
            rec["root_source"] = str(a.uns.get("dpt_root_source", "nuclear_density"))
            rec["pca_width"] = int(a.X.shape[1]) if a.X is not None else None
            nf = a.uns.get("dpt_n_nonfinite_per_root")
            rec["n_roots_clamped"] = int((np.asarray(nf) > 0).sum()) if nf is not None else None
        except Exception as e:                                  # noqa: BLE001
            rec["h5ad_error"] = f"{type(e).__name__}: {e}"
    hr = d / "holeyness_roots.json"
    rec["holeyness"] = json.loads(hr.read_text()) if hr.exists() else None
    cc = d / "cellularity_confound" / "cellularity_confound.json"
    rec["confound"] = json.loads(cc.read_text()) if cc.exists() else None
    return rec


def duct_level(results_csv: Path, export: Path, ann_dir: Path,
               dims: Path, slide_list: Path, patch_size: int) -> dict:
    """Duct-level correlations against whatever pseudotime results_csv holds.

    Reuses holeyness.py's pipeline unmodified, so Phase 1's numbers and these are
    produced by identical code and are directly comparable.
    """
    slides = load_slide_list(slide_list)
    meas = parse_measurement_export(export, slides)
    polys = load_duct_polygons(ann_dir, slides, load_slide_dimensions(dims))
    dt = build_duct_table(meas, polys)
    df = pd.read_csv(results_csv)
    assigned = assign_patches_to_ducts(df, dt, patch_size=patch_size)
    per_duct = aggregate_per_duct(assigned, dt, np.nanmedian, "median")

    pt = per_duct["pseudotime"].values
    hole = per_duct["hole_pct"].values
    area = per_duct["area_um2"].values
    nd = per_duct["nuclear_density"].values

    rho_hole, p_hole = _safe_spearman(pt, hole)
    rho_area, p_area = _safe_spearman(pt, area)
    rho_nd, p_nd = _safe_spearman(pt, nd)
    rho_hole_area, _ = _safe_spearman(hole, area)
    return {
        "n_ducts": int(len(per_duct)),
        "rho_pt_hole_pct": rho_hole, "p_pt_hole_pct": p_hole,
        "rho_pt_area": rho_area, "p_pt_area": p_area,
        "rho_pt_nuclear_density": rho_nd, "p_pt_nuclear_density": p_nd,
        "rho_hole_area": rho_hole_area,
        "partial_pt_hole_given_area": _partial_spearman(pt, hole, area),
    }


def compare_section(section: str, v2: dict, hr: dict, logs: list[Path],
                    duct_v2: dict | None, duct_hr: dict | None) -> dict:
    a, b = v2["df"], hr["df"]
    for d, nm in ((a, "v2"), (b, "holeyroot")):
        if d.duplicated(subset=KEY).any():
            raise ValueError(f"{section} {nm}: duplicate (slide_name, x, y) keys.")
    cols = KEY + [c for c in FEATURES + ["pseudotime", "pseudotime_std"]]
    m = a[[c for c in cols if c in a.columns]].merge(
        b[[c for c in cols if c in b.columns]], on=KEY, how="inner",
        suffixes=("_v2", "_hr"))
    identical = bool(len(a) == len(b) == len(m))

    pv, ph = m["pseudotime_v2"].values, m["pseudotime_hr"].values
    ok = np.isfinite(pv) & np.isfinite(ph)
    rho = float(spearmanr(pv[ok], ph[ok]).statistic) if ok.sum() >= 4 else None

    def frho(df, suf):
        out = {}
        pt = df[f"pseudotime{suf}"].values
        for f in FEATURES:
            c = f"{f}{suf}"
            if c not in df.columns:
                out[f] = None; continue
            v = df[c].values
            k = np.isfinite(pt) & np.isfinite(v)
            out[f] = float(spearmanr(pt[k], v[k]).statistic) if k.sum() >= 4 else None
        return out

    rv, rh = frho(m, "_v2"), frho(m, "_hr")
    feats = {}
    for f in FEATURES:
        x, y = rv.get(f), rh.get(f)
        delta = None if (x is None or y is None) else abs(y - x)
        feats[f] = {"v2": x, "holeyroot": y, "abs_delta": delta,
                    "flagged": bool(delta is not None and delta > DELTA_FLAG),
                    "sign_flip": bool(x is not None and y is not None and x * y < 0)}

    roots_v2 = set(v2["roots"] or [])
    roots_hr = set(hr["roots"] or [])
    n_diff = len(roots_hr - roots_v2) if (v2["roots"] and hr["roots"] and identical) else None

    def rootprops(rec):
        if not rec["roots"]:
            return {"unavailable": True}
        d = rec["df"]
        sub = d.iloc[[i for i in rec["roots"] if 0 <= i < len(d)]]
        p = {}
        for c in ("nuclear_density", "nucleus_count"):
            if c in sub.columns:
                v = sub[c].values.astype(float); v = v[np.isfinite(v)]
                p[c] = ({"median": float(np.median(v)), "min": float(v.min()),
                         "max": float(v.max())} if v.size else None)
        if rec["holeyness"]:
            hp = np.array([r["hole_pct"] for r in rec["holeyness"]["selected_roots"]], float)
            p["hole_pct"] = {"median": float(np.median(hp)),
                             "min": float(hp.min()), "max": float(hp.max())}
        return p

    def stdblock(rec):
        rng, src = parse_pt_range(logs, rec["dir"])
        s = rec["df"]["pseudotime_std"].values.astype(float) \
            if "pseudotime_std" in rec["df"].columns else np.array([])
        s = s[np.isfinite(s)]
        return {"median_raw": float(np.median(s)) if s.size else None,
                "p25": float(np.percentile(s, 25)) if s.size else None,
                "p75": float(np.percentile(s, 75)) if s.size else None,
                "raw_pt_range": rng, "raw_pt_range_source": src,
                "median_pct_of_range": float(100 * np.median(s) / rng)
                                       if (s.size and rng) else None,
                "n_roots_clamped": rec.get("n_roots_clamped")}

    return {
        "section": section,
        "identical_patch_sets": identical,
        "n_shared": int(len(m)),
        "spearman_v2_vs_holeyroot": rho,
        "sign_preserved": None if rho is None else bool(rho > 0),
        "contradicts_random_root_finding": bool(rho is not None and abs(rho) < RHO_FLOOR),
        "axis_flipped": bool(rho is not None and rho < -RHO_FLOOR),
        "n_roots_differing": n_diff,
        "root_props_v2": rootprops(v2),
        "root_props_holeyroot": rootprops(hr),
        "feature_correlations": feats,
        "n_flagged": sum(1 for v in feats.values() if v["flagged"]),
        "pca_width": {"v2": v2.get("pca_width"), "holeyroot": hr.get("pca_width")},
        "pseudotime_std": {"v2": stdblock(v2), "holeyroot": stdblock(hr)},
        "duct_level": {"v2": duct_v2, "holeyroot": duct_hr},
        "confound": {"v2": v2["confound"] or "not available (not regenerated: "
                            "analyze_run_nuclear_density writes into the tree it "
                            "analyses, and v2 must not be modified)",
                     "holeyroot": hr["confound"] or "not run"},
    }


def directionality(secs: list[dict]) -> dict:
    """Which features are directional in one section only, v2 vs holeyroot."""
    out = {}
    for rule in ("v2", "holeyroot"):
        rows, one_only, agree, disagree = {}, [], [], []
        for f in FEATURES:
            vals = {s["section"]: s["feature_correlations"][f][rule] for s in secs}
            dirs = {k: (None if v is None else (v > 0) if abs(v) >= DIRECTIONAL else None)
                    for k, v in vals.items()}
            rows[f] = {"values": vals, "directional": {k: v is not None for k, v in dirs.items()}}
            nd = [k for k, v in dirs.items() if v is not None]
            if len(nd) == 1:
                one_only.append(f)
            elif len(nd) == 2:
                (agree if dirs[nd[0]] == dirs[nd[1]] else disagree).append(f)
        out[rule] = {"per_feature": rows,
                     "directional_in_one_section_only": one_only,
                     "directional_both_AGREE": agree,
                     "directional_both_DISAGREE": disagree,
                     "threshold": DIRECTIONAL}
    return out


def write_report(res: dict, out: Path) -> None:
    L = ["# Phase 2 — holey-ness-rooted pseudotime vs v2", "",
         "**States numbers. Declares neither anchor better.**", "",
         "## What counts as evidence", "",
         "`rho(pseudotime, hole_pct)` rising is **partly circular** — the anchor "
         "IS holey-ness — and is **not** evidence the anchor is better. The "
         "non-circular tests, neither used in root selection, are:", "",
         "- **`rho(pseudotime, duct area)`** — in v2 this was ~+0.41 (2M-1) but "
         "**−0.084** (2M-2). The sections disagreed about the mediator the lab's "
         "biology rests on. Whether re-anchoring moves 2M-2 toward positive is "
         "the sharpest single discriminator here.",
         "- **`rho(pseudotime, nuclear_density)`** — non-circular for the first "
         "time, since density no longer selects the roots. v2: +0.445 / −0.150.", "",
         "Ordering expectation: random 20-root sets reproduce production "
         f"pseudotime at |rho| 0.78–0.89, so |rho| < {RHO_FLOOR} vs v2 would "
         "contradict that and is flagged. A value near −1 is an end-for-end flip "
         "— orientation only, ordering intact.", ""]

    for s in res["sections"]:
        sec = s["section"]
        L += [f"## {sec}", "",
              f"- patch sets: {'identical' if s['identical_patch_sets'] else 'DIFFER'}"
              f" ({s['n_shared']} shared)",
              f"- PCA width: v2 {s['pca_width']['v2']} → holeyroot {s['pca_width']['holeyroot']}",
              f"- **Spearman(v2, holeyroot) = {_f(s['spearman_v2_vs_holeyroot'])}**"
              + ("  — sign preserved" if s["sign_preserved"] else "  — SIGN REVERSED"),
              f"- roots differing from v2: "
              + (f"**{s['n_roots_differing']}/20**" if s["n_roots_differing"] is not None
                 else "n/a (patch sets differ)"), ""]
        if s["contradicts_random_root_finding"]:
            L += [f"> ⚠ **|rho| < {RHO_FLOOR} — CONTRADICTS the random-root finding "
                  "that the ordering is root-insensitive. Explain before trusting "
                  "anything below.**", ""]
        if s["axis_flipped"]:
            L += ["> The axis flipped end-for-end. That is an ORIENTATION change, "
                  "not a change in ordering — the two are different claims.", ""]

        dv, dh = s["duct_level"]["v2"], s["duct_level"]["holeyroot"]
        if dv or dh:
            L += ["### Duct-level correlations — the non-circular tests", "",
                  "| quantity | v2 | holeyroot | |", "|---|---|---|---|"]
            def row(lbl, k, note):
                x = dv.get(k) if dv else None
                y = dh.get(k) if dh else None
                return f"| {lbl} | {_f(x, 3)} | **{_f(y, 3)}** | {note} |"
            L += [row("rho(pt, hole_pct)", "rho_pt_hole_pct", "⚠ CIRCULAR — anchor is holeyness"),
                  row("**rho(pt, duct AREA)**", "rho_pt_area", "**non-circular**"),
                  row("**rho(pt, nuclear_density)**", "rho_pt_nuclear_density",
                      "**non-circular; first time**"),
                  row("rho(hole_pct, area)", "rho_hole_area", "anchor-independent"),
                  row("partial pt~hole | area", "partial_pt_hole_given_area", "—"),
                  ""]
            if dh and dv and dh.get("rho_pt_area") is not None and dv.get("rho_pt_area") is not None:
                d0, d1 = dv["rho_pt_area"], dh["rho_pt_area"]
                if d0 < 0 <= d1:
                    L += [f"> `rho(pt, duct area)` moved from {d0:+.3f} to {d1:+.3f} — "
                          "from contradicting the lab's stated biology to agreeing "
                          "with it. This is the pre-registered non-circular test. "
                          "It is **not** proof the anchor is correct; a root rule "
                          "tied to a duct-level quantity can align pseudotime with "
                          "other duct-level quantities for structural reasons.", ""]
                elif d0 * d1 > 0:
                    L += [f"> `rho(pt, duct area)` stayed the same sign "
                          f"({d0:+.3f} → {d1:+.3f}). Re-anchoring did not resolve "
                          "the section's relationship with the mediator.", ""]

        L += ["### Morphological feature correlations", "",
              "| feature | v2 | holeyroot | \\|delta\\| | |", "|---|---|---|---|---|"]
        for f, v in s["feature_correlations"].items():
            marks = []
            if v["flagged"]: marks.append("**>0.1**")
            if v["sign_flip"]: marks.append("**SIGN FLIP**")
            if f == "nuclear_density": marks.append("**NON-CIRCULAR NOW**")
            L.append(f"| `{f}` | {_f(v['v2'])} | {_f(v['holeyroot'])} | "
                     f"{_f(v['abs_delta'])} | {' '.join(marks)} |")
        L += ["", f"{s['n_flagged']} feature(s) moved by more than {DELTA_FLAG}.", "",
              "> **`nuclear_density` is non-circular for the first time.** Under v2 "
              "it selected the DPT roots, so its correlation with pseudotime was "
              "partly definitional. It no longer does. Whatever value it now takes "
              "is the first honest estimate — and a value that moved toward the "
              "'expected' sign is still not evidence the anchor is better.", ""]

        L += ["### pseudotime_std", "",
              "| | median (raw) | IQR | raw PT range | % of range | roots clamped |",
              "|---|---|---|---|---|---|"]
        for k in ("v2", "holeyroot"):
            b = s["pseudotime_std"][k]
            pct = ("n/a — range unrecoverable, NOT fabricated"
                   if b["median_pct_of_range"] is None else f"{b['median_pct_of_range']:.2f}%")
            L.append(f"| {k} | {_f(b['median_raw'])} | [{_f(b['p25'])}, {_f(b['p75'])}] | "
                     f"{_f(b['raw_pt_range'])} | {pct} | "
                     f"{b['n_roots_clamped'] if b['n_roots_clamped'] is not None else 'n/a'} |")
        L += ["", "A SMALLER pseudotime_std does **not** validate the anchor. If "
              "`roots clamped` is non-zero, some root could not reach every patch "
              "and contributed a near-constant vector to the median — check graph "
              "connectivity before reading anything into the spread.", ""]

    d = res["directionality"]
    L += ["## Cross-section directionality", "",
          f"A feature is called directional in a section when |rho| >= {DIRECTIONAL}.", "",
          "| | v2 | holeyroot |", "|---|---|---|",
          f"| directional in ONE section only | {len(d['v2']['directional_in_one_section_only'])} "
          f"| {len(d['holeyroot']['directional_in_one_section_only'])} |",
          f"| directional in both, AGREE | {len(d['v2']['directional_both_AGREE'])} "
          f"| {len(d['holeyroot']['directional_both_AGREE'])} |",
          f"| directional in both, DISAGREE | {len(d['v2']['directional_both_DISAGREE'])} "
          f"| {len(d['holeyroot']['directional_both_DISAGREE'])} |", ""]
    for rule in ("v2", "holeyroot"):
        L += [f"**{rule}** — one section only: "
              f"`{'`, `'.join(d[rule]['directional_in_one_section_only']) or 'none'}` · "
              f"agree: `{'`, `'.join(d[rule]['directional_both_AGREE']) or 'none'}` · "
              f"disagree: `{'`, `'.join(d[rule]['directional_both_DISAGREE']) or 'none'}`"]
    L += [""]
    if not d["holeyroot"]["directional_both_DISAGREE"] and d["v2"]["directional_both_DISAGREE"]:
        L += ["> The sections no longer disagree in direction on any feature they "
              "are both directional on. **State this plainly and do not overclaim**: "
              "the sections differ in FIXATIVE (Carnoy's vs PFA), which can shift "
              "hole % independent of biology, and both runs share the same "
              "patch-to-duct assignment rule and therefore the same exclusion bias.", ""]

    L += ["## Not to be read as validation", "",
          "- `rho(pt, hole_pct)` rising is circular by construction.",
          "- A reduced `pseudotime_std` does not validate the anchor.",
          "- Sections reconciling in direction does not validate it either — the "
          "fixative differs, and the exclusion bias is shared.",
          "- The excluded-duct bias measured in Phase 1 (smaller, less holey ducts "
          "dropped) applies to the root pool too: the anchor draws from a "
          "population that systematically under-represents the least holey ducts.", ""]
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=["2M-1", "2M-2"])
    ap.add_argument("--v2-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--holeyroot-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--exports", nargs="+", type=Path, required=True,
                    help="Holeyness export per section, SAME ORDER as --sections. "
                         "2M-1 and 2M-2 use DIFFERENT files.")
    ap.add_argument("--slide-lists", nargs="+", type=Path, required=True)
    ap.add_argument("--annotation-dir", type=Path, required=True)
    ap.add_argument("--slide-dimensions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=112)
    ap.add_argument("--logs", nargs="*", type=Path, default=[])
    ap.add_argument("--skip-duct-level", action="store_true",
                    help="Skip the duct-level recomputation (the non-circular "
                         "tests). Only for a quick structural check.")
    args = ap.parse_args()

    n = len(args.sections)
    for nm, v in (("--v2-dirs", args.v2_dirs), ("--holeyroot-dirs", args.holeyroot_dirs),
                  ("--exports", args.exports), ("--slide-lists", args.slide_lists)):
        if len(v) != n:
            ap.error(f"{nm} must have {n} entries to match --sections")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    secs = []
    for i, sec in enumerate(args.sections):
        print(f"\n{'='*70}\n  {sec}\n{'='*70}")
        v2 = load_run(args.v2_dirs[i])
        hr = load_run(args.holeyroot_dirs[i])
        dv = dh = None
        if not args.skip_duct_level:
            common = dict(export=args.exports[i], ann_dir=args.annotation_dir,
                          dims=args.slide_dimensions, slide_list=args.slide_lists[i],
                          patch_size=args.patch_size)
            print("\n--- duct level, v2 pseudotime ---")
            dv = duct_level(args.v2_dirs[i] / "results.csv", **common)
            print("\n--- duct level, holeyroot pseudotime ---")
            dh = duct_level(args.holeyroot_dirs[i] / "results.csv", **common)
        s = compare_section(sec, v2, hr, args.logs, dv, dh)
        secs.append(s)
        print(f"\n  rho(v2, holeyroot) = {_f(s['spearman_v2_vs_holeyroot'])}   "
              f"roots differing: {s['n_roots_differing']}   "
              f"flagged features: {s['n_flagged']}")
        if dh:
            print(f"  NON-CIRCULAR  rho(pt, area) = {_f(dh['rho_pt_area'], 3)}   "
                  f"rho(pt, nuclear_density) = {_f(dh['rho_pt_nuclear_density'], 3)}")

    res = {"sections": secs, "directionality": directionality(secs)}
    (args.output_dir / "holeyroot_comparison.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_report(res, args.output_dir / "holeyroot_comparison.md")
    print(f"\nWrote {args.output_dir}/holeyroot_comparison.{{json,md}}")
    print("Neither anchor is declared better. rho(pt, hole_pct) rising is circular.")


if __name__ == "__main__":
    main()
