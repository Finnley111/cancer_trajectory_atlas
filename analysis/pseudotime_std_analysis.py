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


RANGE_PAT = re.compile(r"Pseudotime median range:\s*\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]")
SECTION_PAT = re.compile(r"Section:\s*(\S+)")


def find_raw_range(run_dir: Path, section: str, extra_logs: list[Path] | None = None):
    """Recover this SECTION's 'Pseudotime median range: [a, b]' from a run log.

    diffusion.py prints it; nothing stores it. Returns (lo, hi, source) or
    (None, None, reason).

    Must be matched per section. jobs/run_per_section.sh runs BOTH sections in a
    single SLURM job, so one log contains one range line per section. A plain
    .search() returns the first, which silently hands 2M-1's range to 2M-2 — and
    the range is the denominator for every "% of the axis" figure, so the error is
    invisible and wrong rather than absent. Each range is therefore attributed to
    the nearest PRECEDING "Section: X" banner, and a range with no identifiable
    section is refused rather than guessed.
    """
    candidates: list[Path] = []
    for d in (run_dir, run_dir.parent, run_dir / "logs"):
        if d.is_dir():
            candidates += sorted(d.glob("*.out")) + sorted(d.glob("*.log"))
    candidates += list(extra_logs or [])

    ambiguous = 0
    for p in candidates:
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue

        marks = [(m.start(), m.group(1)) for m in SECTION_PAT.finditer(text)]
        found_unlabelled = False
        for m in RANGE_PAT.finditer(text):
            owner = None
            for pos, name in marks:
                if pos < m.start():
                    owner = name
                else:
                    break
            if owner is None:
                found_unlabelled = True
                continue
            if owner == section:
                return float(m.group(1)), float(m.group(2)), f"{p} (section {owner})"
        if found_unlabelled:
            ambiguous += 1

    reason = (f"no range line attributable to section '{section}' in "
              f"{len(candidates)} log file(s) near {run_dir}. "
              "compute_dpt_multi_root prints the raw range but does not persist it.")
    if ambiguous:
        reason += (f" {ambiguous} file(s) contained a range with no preceding "
                   "'Section:' banner; those were REFUSED rather than assumed to "
                   "belong to this section.")
    return None, None, reason


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

    lo, hi, src = find_raw_range(run_dir, section, extra_logs)
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

    # ── Is std actually INDEPENDENT information, or a function of pseudotime? ──
    # compute_dpt_multi_root clamps non-finite DPT output to that run's max
    # (diffusion.py). If a subset of the 20 root runs assigns the clamped maximum
    # to (nearly) every patch — which happens when those roots sit in a different
    # connected component of the neighbour graph — then across roots each patch
    # sees a two-point distribution {t, max}, giving
    #     std = sqrt(f(1-f)) * (max - t)   i.e. std EXACTLY LINEAR in (1 - pt),
    # with intercept 0 at the top of the axis. Under that regime std is a
    # deterministic restatement of pseudotime and carries no per-patch
    # uncertainty at all, so no "% of the axis" caveat drawn from it is
    # meaningful. This test needs no knowledge of the raw range: it is internal
    # to the section.
    x_lin = 1.0 - pt[ok]
    y_lin = sd[ok]
    A = np.polyfit(x_lin, y_lin, 1)
    pred = np.polyval(A, x_lin)
    ss_res = float(((y_lin - pred) ** 2).sum())
    ss_tot = float(((y_lin - y_lin.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    max_rel_dev = float(np.max(np.abs(y_lin - pred)) / max(np.max(y_lin), 1e-12))

    linearity = {
        "model": "pseudotime_std ~ a*(1 - pseudotime) + b",
        "slope_a": float(A[0]),
        "intercept_b": float(A[1]),
        "r_squared": float(r2),
        "max_abs_residual_as_frac_of_max_std": max_rel_dev,
        # Only meaningful if the two-point model actually describes the data. The
        # inversion std = sqrt(f(1-f)) * span presumes every root run contributes
        # either the patch's true value or the clamped maximum; when R^2 says the
        # model does not hold, this is just an artefact of inverting the wrong
        # equation. Report None rather than a number that invites being quoted.
        "implied_fraction_of_roots_clamped": (
            float(0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - 4.0 * min(0.25, A[0] ** 2))))
            if (np.isfinite(A[0]) and np.isfinite(r2) and r2 >= 0.98) else None
        ),
        "deterministic": bool(np.isfinite(r2) and r2 >= 0.98),
        "interpretation": (
            "DETERMINISTIC — std is an affine function of pseudotime (R^2 >= 0.98), "
            "so it contains NO per-patch information. This is the signature of a "
            "fixed subset of root runs contributing the clamped maximum to every "
            "patch, not of genuine per-patch uncertainty. Do not quote it as an "
            "uncertainty, and do not read rho(pseudotime, std) as 'uncertainty "
            "concentrates at one end' — that correlation is forced by the same "
            "arithmetic."
            if np.isfinite(r2) and r2 >= 0.98 else
            "NOT deterministic — std has genuine scatter around any function of "
            "pseudotime, so it does carry per-patch information."
        ),
    }

    # Patches pinned at the very top of the axis are the ones the clamp would
    # produce if they are unreachable from the roots.
    top = ok & (pt >= 0.99)
    pinned = {
        "n_patches_pseudotime_ge_0.99": int(top.sum()),
        "fraction": float(top.sum() / max(int(ok.sum()), 1)),
        "median_std_among_them": float(np.median(sd[top])) if top.any() else float("nan"),
        "n_patches_std_exactly_zero": int((ok & (sd == 0)).sum()),
    }
    if top.any() and "slide_name" in df.columns:
        vc = df.loc[top, "slide_name"].value_counts()
        pinned["slide_breakdown"] = {str(k): int(v) for k, v in vc.items()}
        pinned["max_share_from_one_slide"] = float(vc.iloc[0] / vc.sum())

    if linearity["deterministic"]:
        # Overrides every other reading: if std is an affine function of
        # pseudotime there is no per-patch uncertainty to characterise, and the
        # negligible/moderate scale is not applicable.
        verdict = (
            f"NOT AN UNCERTAINTY — pseudotime_std is an affine function of "
            f"pseudotime (R^2 = {r2:.4f} against a*(1-pt)+b, max residual "
            f"{max_rel_dev:.2%} of max std). It therefore carries NO per-patch "
            "information and must not be reported as a confidence interval. This is "
            "the fingerprint of the non-finite clamp in compute_dpt_multi_root: a "
            "fixed subset of the 20 root runs assigns the clamped maximum to nearly "
            f"every patch. rho(pseudotime, std) = {rho_all:+.3f} is forced by the "
            "same arithmetic and is not evidence about where the axis is uncertain. "
            f"{pinned['n_patches_pseudotime_ge_0.99']} patch(es) sit at "
            "pseudotime >= 0.99, which is where unreachable patches would be pinned."
        )
    elif raw_range:
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
               "is not concentrated at either end")
            + f". std is NOT a deterministic function of pseudotime "
              f"(R^2 = {r2:.3f}), so it does carry per-patch information."
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
        "linearity_vs_pseudotime": linearity,
        "pinned_at_axis_top": pinned,
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
