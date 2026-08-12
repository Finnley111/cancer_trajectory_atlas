"""Task 2 — how much do the 20 DPT roots disagree, per patch?

`pseudotime_std` is std(pt_matrix, axis=0) across the 20 per-root DPT runs
(analysis/diffusion.py). It has never been analysed. root_sensitivity showed the
AGGREGATE ordering is stable — random 20-root sets reproduce production
pseudotime at |rho| 0.78-0.89 — but a stable ordering says nothing about whether
an individual patch sits at a well-determined position.

SCALE WARNING, load-bearing
    `pseudotime` is min-max normalised to [0, 1] (diffusion.py:193) but
    `pseudotime_std` is stored RAW, on the diffusion-distance scale, computed
    BEFORE that normalisation (diffusion.py:186). The two are therefore NOT
    comparable, and "std = 0.02" does not mean "2% of the axis".

    The conversion factor is 1 / (pt_max - pt_min) on the raw median-aggregated
    scale. compute_dpt_multi_root prints that range but does not store it, so it
    is recoverable ONLY from a run log. This module looks for it in the run
    directory and in any log file passed via --raw-range; if it cannot be found,
    it says so and reports raw values with an explicit non-comparability
    statement. It never invents a normalisation.

Read-only: consumes results.csv. Writes a new directory only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Verdict thresholds, expressed as median std relative to the raw pseudotime
# range. Only usable when that range is recoverable.
NEGLIGIBLE_REL = 0.02   # median std < 2% of the axis
MODERATE_REL = 0.10     # 2-10% moderate; >10% warrants a manuscript caveat


def _rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def find_raw_range(run_dir: Path, extra_logs: list[Path] | None = None):
    """Recover 'Pseudotime median range: [a, b]' from a run log, if one survives.

    diffusion.py:197 prints it; nothing stores it. Returns (lo, hi, source) or
    (None, None, reason).
    """
    pat = re.compile(r"Pseudotime median range:\s*\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]")
    candidates: list[Path] = []
    for d in (run_dir, run_dir.parent, run_dir / "logs"):
        if d.is_dir():
            candidates += sorted(d.glob("*.out")) + sorted(d.glob("*.log"))
    candidates += list(extra_logs or [])

    for p in candidates:
        try:
            m = pat.search(p.read_text(errors="ignore"))
        except Exception:
            continue
        if m:
            return float(m.group(1)), float(m.group(2)), str(p)
    return None, None, (
        f"not found — searched {len(candidates)} log file(s) near {run_dir}. "
        "compute_dpt_multi_root prints the raw range but does not persist it."
    )


def analyse(section: str, run_dir: Path, out_dir: Path,
            extra_logs: list[Path] | None) -> dict:
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found — this analysis reuses an existing run.")
    df = pd.read_csv(csv)
    for c in ("pseudotime", "pseudotime_std"):
        if c not in df.columns:
            raise KeyError(f"{csv} has no '{c}' column.")

    pt = df["pseudotime"].values.astype(float)
    sd = df["pseudotime_std"].values.astype(float)
    ok = np.isfinite(pt) & np.isfinite(sd)

    q = lambda a, p: float(np.percentile(a[ok], p))  # noqa: E731
    med = q(sd, 50)

    lo, hi, src = find_raw_range(run_dir, extra_logs)
    raw_range = (hi - lo) if (lo is not None and hi is not None) else None
    rel = {k: (v / raw_range if raw_range else None) for k, v in
           {"median": med, "p95": q(sd, 95), "max": float(np.nanmax(sd[ok]))}.items()}

    # Where do the high-uncertainty patches sit along the axis?
    def tail_profile(mult: float) -> dict:
        m = ok & (sd > mult * med)
        n = int(m.sum())
        if n == 0:
            return {"multiplier": mult, "n_patches": 0, "fraction": 0.0}
        pt_rank = np.argsort(np.argsort(pt)) / max(len(pt) - 1, 1)
        return {
            "multiplier": mult,
            "n_patches": n,
            "fraction": float(n / int(ok.sum())),
            "median_pseudotime": float(np.median(pt[m])),
            "median_pseudotime_percentile": float(np.median(pt_rank[m])),
            "frac_in_top_decile": float((pt[m] >= np.quantile(pt[ok], 0.9)).mean()),
            "frac_in_bottom_decile": float((pt[m] <= np.quantile(pt[ok], 0.1)).mean()),
        }

    per_slide = {}
    if "slide_name" in df.columns:
        for name, g in df.groupby("slide_name"):
            gp, gs = g["pseudotime"].values.astype(float), g["pseudotime_std"].values.astype(float)
            if len(g) < 30:
                continue
            per_slide[str(name)] = {
                "n": int(len(g)),
                "median_std": float(np.nanmedian(gs)),
                "rho_pt_vs_std": _rho(gp, gs),
            }

    rho_all = _rho(pt, sd)
    slide_rhos = [v["rho_pt_vs_std"] for v in per_slide.values() if np.isfinite(v["rho_pt_vs_std"])]

    if raw_range:
        r = rel["median"]
        level = ("NEGLIGIBLE" if r < NEGLIGIBLE_REL else
                 "MODERATE" if r < MODERATE_REL else "WARRANTS A MANUSCRIPT CAVEAT")
        verdict = (
            f"{level} — median per-patch root disagreement is {med:.4g} on the raw "
            f"diffusion-distance scale, which is {r:.1%} of the raw pseudotime range "
            f"[{lo:.4g}, {hi:.4g}] recovered from {src}. rho(pseudotime, std) = "
            f"{rho_all:+.3f}, so uncertainty "
            + ("concentrates at the late end" if rho_all > 0.2 else
               "concentrates at the early end" if rho_all < -0.2 else
               "is not concentrated at either end") + "."
        )
    else:
        verdict = (
            f"SCALE NOT RECOVERABLE — median per-patch std is {med:.4g} in RAW "
            "diffusion-distance units. pseudotime is min-max normalised to [0,1] but "
            "pseudotime_std is not, so this number CANNOT be read as a fraction of "
            f"the axis and no negligible/moderate call is made. Reason: {src}. "
            f"rho(pseudotime, std) = {rho_all:+.3f}, which is scale-free and "
            "therefore still interpretable: uncertainty "
            + ("concentrates at the late end" if rho_all > 0.2 else
               "concentrates at the early end" if rho_all < -0.2 else
               "is not concentrated at either end") + "."
        )

    # Figures
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sd[ok], bins=60, color="#4878CF")
    ax.axvline(med, color="#D65F5F", lw=1.5, label=f"median {med:.4g}")
    ax.set_xlabel("pseudotime_std (RAW diffusion-distance units)")
    ax.set_ylabel("patches")
    ax.set_title(f"{section}: per-patch disagreement across the 20 DPT roots")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pseudotime_std_hist_{section}.{ext}", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hexbin(pt[ok], sd[ok], gridsize=60, cmap="Blues", mincnt=1)
    ax.set_xlabel("pseudotime (normalised [0,1])")
    ax.set_ylabel("pseudotime_std (RAW units — different scale)")
    ax.set_title(f"{section}: rho = {rho_all:+.3f}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pseudotime_vs_std_{section}.{ext}", dpi=170)
    plt.close(fig)

    return {
        "section": section,
        "n_patches": int(len(df)),
        "n_finite": int(ok.sum()),
        "distribution_raw_units": {
            "min": float(np.nanmin(sd[ok])), "p25": q(sd, 25), "median": med,
            "p75": q(sd, 75), "p95": q(sd, 95), "max": float(np.nanmax(sd[ok])),
        },
        "raw_pseudotime_range": (
            {"min": lo, "max": hi, "range": raw_range, "source": src}
            if raw_range else {"recoverable": False, "reason": src}
        ),
        "std_relative_to_raw_range": rel if raw_range else None,
        "rho_pseudotime_vs_std": rho_all,
        "per_slide": per_slide,
        "per_slide_median_rho": float(np.median(slide_rhos)) if slide_rhos else float("nan"),
        "high_uncertainty_tails": [tail_profile(2.0), tail_profile(5.0)],
        "verdict": verdict,
        "scale_warning": (
            "pseudotime is min-max normalised to [0,1]; pseudotime_std is stored raw "
            "and pre-normalisation. Absolute std values are NOT fractions of the axis."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--raw-range", nargs="*", type=Path, default=None,
                    help="Extra log files to search for the printed raw pseudotime range.")
    args = ap.parse_args()
    if len(args.sections) != len(args.run_dirs):
        ap.error("--sections and --run-dirs must match in length and order")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  pseudotime_std — per-patch root disagreement")
    print("=" * 64)

    results = {}
    for section, run_dir in zip(args.sections, args.run_dirs):
        print(f"\n  {section}  <-  {run_dir}")
        results[section] = analyse(section, Path(run_dir), args.output_dir, args.raw_range)
        print(f"    {results[section]['verdict']}")

    with open(args.output_dir / "pseudotime_std_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None if isinstance(o, float) else str(o))

    L = ["# pseudotime_std — how much do the 20 DPT roots disagree per patch?", "",
         "`pseudotime` is min-max normalised to [0,1]; `pseudotime_std` is stored RAW",
         "and pre-normalisation. **Absolute std values are not fractions of the axis.**",
         "`rho(pseudotime, std)` is scale-free and interpretable regardless.", ""]
    for s, r in results.items():
        d = r["distribution_raw_units"]
        L += [f"## {s}", "",
              f"- Patches: {r['n_patches']} ({r['n_finite']} with finite std)",
              f"- Distribution (raw units): min {d['min']:.4g}, p25 {d['p25']:.4g}, "
              f"median {d['median']:.4g}, p75 {d['p75']:.4g}, p95 {d['p95']:.4g}, "
              f"max {d['max']:.4g}",
              f"- rho(pseudotime, std) = **{r['rho_pseudotime_vs_std']:+.3f}**; "
              f"per-slide median {r['per_slide_median_rho']:+.3f}", ""]
        for t in r["high_uncertainty_tails"]:
            if t["n_patches"]:
                L.append(f"- std > {t['multiplier']:.0f}x median: {t['n_patches']} patches "
                         f"({t['fraction']:.1%}), median pseudotime percentile "
                         f"{t['median_pseudotime_percentile']:.0%}, "
                         f"{t['frac_in_top_decile']:.0%} in the top decile")
            else:
                L.append(f"- std > {t['multiplier']:.0f}x median: none")
        L += ["", f"**Verdict.** {r['verdict']}", "",
              f"Figures: `pseudotime_std_hist_{s}.png`, `pseudotime_vs_std_{s}.png`", ""]
    (args.output_dir / "pseudotime_std_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n  JSON:     {args.output_dir / 'pseudotime_std_analysis.json'}")
    print(f"  Markdown: {args.output_dir / 'pseudotime_std_report.md'}")


if __name__ == "__main__":
    main()
