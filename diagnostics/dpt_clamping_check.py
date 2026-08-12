"""Is the diffusion graph connected, and is the DPT clamp firing?

WHY THIS EXISTS
    Task 2 found that in 2M-2, pseudotime_std is an affine function of pseudotime
    (R^2 ~ 1). That is not what per-patch uncertainty looks like. It is what the
    non-finite clamp in compute_dpt_multi_root produces:

        finite_mask = np.isfinite(pt)
        pt[~finite_mask] = pt[finite_mask].max()

    scanpy's DPT returns inf for any patch not reachable from the root through
    the neighbour graph. If the graph has more than one connected component, then
    for a root in component A every patch in component B is inf and gets pinned to
    that run's MAXIMUM — i.e. silently relabelled "maximally late" rather than
    "unreachable". Across 20 roots split between components, each patch then sees
    a two-point distribution {true value, max}, which gives std exactly linear in
    (1 - pseudotime): the observed signature.

    Consequences if this is firing:
      - patches at the top of the axis may be UNREACHABLE, not advanced;
      - the late tail's slide concentration (49.6% / 43.4% one slide) would follow
        directly, since a disconnected component is one slide's tissue;
      - none of the Task 1 feature fixes touch it, so a v2 re-run would reproduce
        it unchanged.

    This check decides it from stored artifacts, in minutes.

READS (READ-ONLY): <run_dir>/adata_full.h5ad and <run_dir>/results.csv.
WRITES: a new directory only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R2_DETERMINISTIC = 0.98
TOP_OF_AXIS = 0.99


def analyse(section: str, run_dir: Path) -> dict:
    import anndata as ad
    from scipy.sparse.csgraph import connected_components

    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(f"{h5ad} not found — this check reuses an existing run.")
    adata = ad.read_h5ad(h5ad)
    obs = adata.obs

    out: dict = {"section": section, "n_patches": int(adata.n_obs)}

    # ── The decisive test: connectivity of the graph DPT actually walked ──────
    if "connectivities" not in adata.obsp:
        out["graph"] = {"available": False,
                        "reason": "adata.obsp['connectivities'] missing"}
    else:
        n_comp, labels = connected_components(adata.obsp["connectivities"],
                                              directed=False)
        sizes = np.bincount(labels)
        order = np.argsort(sizes)[::-1]
        giant = int(sizes[order[0]])
        comps = []
        for c in order[:10]:
            m = labels == c
            entry = {"component": int(c), "n_patches": int(m.sum()),
                     "fraction": float(m.sum() / adata.n_obs)}
            if "slide_id" in obs.columns:
                vc = obs.loc[m, "slide_id"].astype(str).value_counts()
                entry["slide_breakdown"] = {str(k): int(v) for k, v in vc.items()}
                entry["max_share_from_one_slide"] = float(vc.iloc[0] / vc.sum())
            if "pseudotime" in obs.columns:
                ptc = obs.loc[m, "pseudotime"].values.astype(float)
                entry["median_pseudotime"] = float(np.median(ptc))
                entry["frac_at_axis_top"] = float((ptc >= TOP_OF_AXIS).mean())
            comps.append(entry)

        out["graph"] = {
            "available": True,
            "n_connected_components": int(n_comp),
            "giant_component_n": giant,
            "giant_component_fraction": float(giant / adata.n_obs),
            "n_patches_outside_giant": int(adata.n_obs - giant),
            "components": comps,
            "clamp_fires": bool(n_comp > 1),
            "note": (
                "Every patch outside the giant component is unreachable from any "
                "root inside it, so DPT returns inf and the clamp pins it to that "
                "run's maximum — relabelling 'unreachable' as 'maximally late'."
                if n_comp > 1 else
                "Graph is fully connected, so DPT cannot return inf for graph "
                "reasons and the clamp should never fire. Any linear std-vs-"
                "pseudotime relationship needs a different explanation."
            ),
        }

    # ── The signature, recomputed here so the two are reported together ───────
    csv = run_dir / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        pt = df["pseudotime"].values.astype(float)
        sd = df["pseudotime_std"].values.astype(float)
        ok = np.isfinite(pt) & np.isfinite(sd)
        x, y = 1.0 - pt[ok], sd[ok]
        A = np.polyfit(x, y, 1)
        pred = np.polyval(A, x)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

        top = ok & (pt >= TOP_OF_AXIS)
        entry = {
            "std_vs_one_minus_pt_r_squared": float(r2),
            "slope": float(A[0]),
            "intercept": float(A[1]),
            "deterministic": bool(np.isfinite(r2) and r2 >= R2_DETERMINISTIC),
            "n_at_axis_top": int(top.sum()),
            "n_std_exactly_zero": int((ok & (sd == 0)).sum()),
        }
        if top.any() and "slide_name" in df.columns:
            vc = df.loc[top, "slide_name"].value_counts()
            entry["axis_top_slide_breakdown"] = {str(k): int(v) for k, v in vc.items()}
            entry["axis_top_max_share_from_one_slide"] = float(vc.iloc[0] / vc.sum())
        out["std_signature"] = entry
    else:
        out["std_signature"] = {"available": False, "reason": f"{csv} not found"}

    g = out.get("graph", {})
    s = out.get("std_signature", {})
    disconnected = bool(g.get("clamp_fires"))
    deterministic = bool(s.get("deterministic"))

    if disconnected and deterministic:
        verdict = (
            f"CONFIRMED — the graph has {g['n_connected_components']} connected "
            f"components ({g['n_patches_outside_giant']} patches outside the giant "
            f"one) AND std is affine in pseudotime (R^2 = {s['std_vs_one_minus_pt_r_squared']:.4f}). "
            "The clamp is relabelling unreachable patches as maximally late. "
            "pseudotime_std is not an uncertainty, and the top of the axis is "
            "partly an artifact. No Task 1 feature fix touches this — it needs a "
            "connectivity fix in compute_dpt_multi_root or in the neighbour graph."
        )
    elif disconnected:
        verdict = (
            f"GRAPH IS DISCONNECTED ({g['n_connected_components']} components, "
            f"{g['n_patches_outside_giant']} patches outside the giant one) but std "
            "is not a deterministic function of pseudotime. The clamp fires for "
            "some patches; quantify the affected set before trusting the top of "
            "the axis."
        )
    elif deterministic:
        verdict = (
            "UNEXPLAINED — the graph is connected, so the clamp should not fire, "
            f"yet std is affine in pseudotime (R^2 = {s['std_vs_one_minus_pt_r_squared']:.4f}). "
            "Something else is making the root runs agree up to a fixed multiple of "
            "(1 - pseudotime). Do not report std as an uncertainty until this is "
            "explained."
        )
    else:
        verdict = (
            "CLEAN — the graph is connected and std is not a deterministic function "
            "of pseudotime, so per-patch root disagreement is real information."
        )
    out["verdict"] = verdict
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if len(args.sections) != len(args.run_dirs):
        ap.error("--sections and --run-dirs must match in length and order")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  DPT clamp / graph connectivity check")
    print("=" * 64)

    results = {}
    for section, run_dir in zip(args.sections, args.run_dirs):
        print(f"\n  {section}  <-  {run_dir}")
        results[section] = analyse(section, Path(run_dir))
        g = results[section].get("graph", {})
        if g.get("available"):
            print(f"    components={g['n_connected_components']}  "
                  f"giant={g['giant_component_fraction']:.4%}  "
                  f"outside={g['n_patches_outside_giant']}")
        print(f"    {results[section]['verdict']}")

    with open(args.output_dir / "dpt_clamping_check.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: None if isinstance(o, float) else str(o))

    L = ["# DPT clamp / graph connectivity check", "",
         "`compute_dpt_multi_root` clamps non-finite DPT output to that run's",
         "maximum. scanpy returns inf for patches unreachable from the root, so if",
         "the neighbour graph is disconnected, unreachable patches are silently",
         "relabelled **maximally late** instead of **unmeasurable**.", ""]
    for s, r in results.items():
        g, sg = r.get("graph", {}), r.get("std_signature", {})
        L += [f"## {s}", "", f"- Patches: {r['n_patches']}"]
        if g.get("available"):
            L += [f"- Connected components: **{g['n_connected_components']}**",
                  f"- Giant component: {g['giant_component_n']} "
                  f"({g['giant_component_fraction']:.4%})",
                  f"- Patches outside it: **{g['n_patches_outside_giant']}**"]
        if sg.get("deterministic") is not None:
            L += [f"- std ~ a(1-pt)+b: R^2 = **{sg['std_vs_one_minus_pt_r_squared']:.4f}**, "
                  f"slope {sg['slope']:.4f}, intercept {sg['intercept']:.4f}",
                  f"- Patches at pseudotime >= {TOP_OF_AXIS}: {sg['n_at_axis_top']}"]
            if "axis_top_max_share_from_one_slide" in sg:
                L.append(f"- Largest single-slide share at the axis top: "
                         f"{sg['axis_top_max_share_from_one_slide']:.0%}")
        L += ["", f"**Verdict.** {r['verdict']}", ""]
    (args.output_dir / "dpt_clamping_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n  JSON:     {args.output_dir / 'dpt_clamping_check.json'}")
    print(f"  Markdown: {args.output_dir / 'dpt_clamping_report.md'}")


if __name__ == "__main__":
    main()
