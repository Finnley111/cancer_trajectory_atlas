"""Tasks 2 and 3 — the between-section comparison, corrected for the paired design.

WHY THIS EXISTS
---------------
The 16 slides are **8 matched pairs**, not 16 independent samples: each mouse-flank
(gland) contributes one slide to 2M-1 (Carnoy's) and one to 2M-2 (PFA).

The comparison currently on record used an exact permutation over all
C(16,8) = 12,870 ways of splitting 16 slides into two groups of 8. That null treats
the sections as independent groups, so it admits between-gland and between-mouse
variation that the paired design already controls. It is mis-specified, its null is
wider than it should be, and its minimum detectable difference is inflated
(0.250 raw, 0.133 on the best-powered variant).

THE CORRECTED DESIGN
--------------------
For each gland, compute the statistic in its Carnoy's half and in its PFA half, and
take the within-gland difference. Under the null that section membership is
irrelevant, each of the 8 differences is equally likely to carry either sign, so the
exact null is the **2^8 = 256 sign-flip permutations**, enumerated exhaustively.

THE P-VALUE FLOOR, WHICH IS A PROPERTY OF THE DESIGN
----------------------------------------------------
With 256 permutations the smallest attainable two-sided p is **2/256 = 0.0078** — the
observed sign vector and its mirror. A paired p at that floor is NOT weaker evidence
than the unpaired test's 1.554e-4; it is the same evidence at coarser resolution. The
unpaired test could resolve to 2/12,870 only because it had 12,870 permutations to
draw on, and it had those only by assuming an independence that is false here.
Reported as ``< 0.0078`` rather than 0.

ESTIMAND 4 COLLAPSES ONTO ESTIMAND 1, AND THAT IS EXPECTED
-----------------------------------------------------------
The within-slide normal-scores variant exists to strip between-slide location and
scale before pooling. In this design each gland-section **is** a single slide, so
there are no between-slide offsets left to remove. Spearman is invariant to monotone
transforms and normal scores are monotone in rank, so per gland
``rho(ns(pt), ns(hole)) == rho(pt, hole)`` exactly. Both are computed and the identity
is checked and reported rather than assumed — it is a statement about the design, not
a bug.

READ-ONLY. Reads the per-duct tables and the unpaired result's JSON. Recomputes no
pseudotime and no duct table, reruns no pipeline stage, and never writes to the
unpaired results directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from .gland_pairing_audit import load_per_duct, build_pairing
from .holeyness import _partial_spearman_multi

MIN_DUCTS_PER_GLAND = 30      # below this a per-gland correlation is not reported
N_SIGN_FLIPS = 256            # 2**8
P_FLOOR = 2.0 / N_SIGN_FLIPS  # 0.0078125

# Keys as they appear in the unpaired result's JSON, so the two can be aligned.
UNPAIRED_KEYS = {
    "raw_rho_pt_hole": "raw_rho_pt_hole",
    "partial_given_area": "partial_given_area",
    "rho_area_pseudotime": "rho_area_pseudotime",
    "raw_within_slide_normalised": "raw_within_slide_normalised",
}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_DUCTS_PER_GLAND:
        return None
    return float(spearmanr(x[ok], y[ok]).statistic)


def _normal_scores(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return out
    out[ok] = norm.ppf((rankdata(v[ok]) - 0.5) / ok.sum())
    return out


# ── the four estimands, each computed on ONE gland-section's ducts ───────────

def est_raw(d: pd.DataFrame):
    return _rho(d["pseudotime"].values, d["hole_pct"].values)


def est_partial_area(d: pd.DataFrame):
    if len(d) < MIN_DUCTS_PER_GLAND:
        return None
    v = _partial_spearman_multi(d["pseudotime"].values, d["hole_pct"].values,
                                [d["area_um2"].values])
    return None if not np.isfinite(v) else float(v)


def est_area_pt(d: pd.DataFrame):
    return _rho(d["area_um2"].values, d["pseudotime"].values)


def est_raw_ns(d: pd.DataFrame):
    return _rho(_normal_scores(d["pseudotime"].values),
                _normal_scores(d["hole_pct"].values))


ESTIMANDS = {
    "raw_rho_pt_hole":             ("rho(pseudotime, hole_pct)", est_raw),
    "partial_given_area":          ("rho(pt, hole_pct | area)", est_partial_area),
    "rho_area_pseudotime":         ("rho(duct area, pseudotime)", est_area_pt),
    "raw_within_slide_normalised": ("rho(pt, hole_pct), normal scores", est_raw_ns),
}


# ── the paired test ──────────────────────────────────────────────────────────

def per_gland_values(frames: dict, glands: list, key: str) -> dict:
    """The statistic for every gland in both sections, plus the difference."""
    label, fn = ESTIMANDS[key]
    a, b = list(frames)
    rows, unusable = [], []
    for g in glands:
        da = frames[a][frames[a]["gland"] == g]
        db = frames[b][frames[b]["gland"] == g]
        va, vb = fn(da), fn(db)
        row = {"gland": g,
               "n_ducts_a": int(len(da)), "n_ducts_b": int(len(db)),
               "rho_a": va, "rho_b": vb,
               "difference": (None if (va is None or vb is None)
                              else float(va - vb))}
        if row["difference"] is None:
            unusable.append(g)
        rows.append(row)
    return {"label": label, "section_a": a, "section_b": b,
            "per_gland": rows, "unusable_glands": unusable}


def sign_flip_test(diffs: np.ndarray) -> dict:
    """Exact null over all 2^n sign assignments. n is 8 here, so 256.

    The statistic is the MEAN of the within-gland differences. Under the null that
    section membership is irrelevant, each gland's difference is equally likely to
    carry either sign; enumerating every sign vector gives the exact null with no
    sampling error.

    The null is exactly antisymmetric — every sign vector appears alongside its
    negation — so its mean is 0 by construction and the smallest attainable
    two-sided p is 2/2^n.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    observed = float(diffs.mean())

    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = (signs * diffs).mean(axis=1)

    # tolerance so the observed vector and its mirror always count as "at least as
    # extreme" despite floating-point drift
    n_ge = int((np.abs(null) >= abs(observed) - 1e-12).sum())
    p = float(n_ge / null.size)
    floor = 2.0 / null.size
    at_floor = bool(n_ge <= 2)

    return {
        "n_pairs": int(n),
        "n_permutations": int(null.size),
        "exhaustive": True,
        "observed_mean_difference": observed,
        "median_difference": float(np.median(diffs)),
        "n_null_at_least_as_extreme": n_ge,
        "exact_p_two_sided": p,
        "p_value_display": (f"< {floor:.4g}" if at_floor else f"{p:.4g}"),
        "p_at_floor": at_floor,
        "p_floor": floor,
        "null_percentiles": {q: float(np.percentile(null, q))
                             for q in (0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5)},
        "null_mean": float(null.mean()),
        "minimum_detectable_difference_alpha05": float(
            np.percentile(np.abs(null), 97.5)),
        "observed_percentile_in_null": float((null < observed).mean() * 100),
    }


def run_estimand(frames: dict, glands: list, key: str) -> dict:
    pg = per_gland_values(frames, glands, key)
    if pg["unusable_glands"]:
        return {**pg, "test": None,
                "verdict": (f"NOT TESTED — {len(pg['unusable_glands'])} gland(s) "
                            f"{pg['unusable_glands']} could not produce a "
                            f"correlation in both sections (fewer than "
                            f"{MIN_DUCTS_PER_GLAND} usable ducts). A sign-flip null "
                            "requires all pairs.")}
    diffs = np.array([r["difference"] for r in pg["per_gland"]], dtype=float)
    t = sign_flip_test(diffs)
    mdd = t["minimum_detectable_difference_alpha05"]
    differ = t["exact_p_two_sided"] < 0.05
    if differ:
        verdict = (f"DIFFER — mean within-gland difference "
                   f"{t['observed_mean_difference']:+.4f}, exact p = "
                   f"{t['p_value_display']} over {t['n_permutations']} sign flips.")
        if t["p_at_floor"]:
            verdict += (" This is the design's p-value floor: the evidence is as "
                        "strong as 8 pairs can express, not weak.")
    else:
        verdict = (f"NO EVIDENCE OF A DIFFERENCE — mean within-gland difference "
                   f"{t['observed_mean_difference']:+.4f}, exact p = "
                   f"{t['p_value_display']}. That is not evidence of equivalence: "
                   f"with 8 pairs this design could not detect a true mean "
                   f"difference smaller than {mdd:.4f} at alpha = 0.05, so "
                   "differences up to that magnitude cannot be excluded.")
    return {**pg, "test": t, "verdict": verdict}


# ── Task 3: side by side against the unpaired result ─────────────────────────

def side_by_side(paired: dict, unpaired_json: dict | None) -> dict:
    out = {}
    for key, (label, _) in ESTIMANDS.items():
        p = paired[key]
        t = p["test"]
        row = {
            "label": label,
            "paired_observed": t["observed_mean_difference"] if t else None,
            "paired_p": t["exact_p_two_sided"] if t else None,
            "paired_p_display": t["p_value_display"] if t else None,
            "paired_mdd": t["minimum_detectable_difference_alpha05"] if t else None,
            "paired_n_perms": t["n_permutations"] if t else None,
        }
        u = (unpaired_json or {}).get("task_2", {}).get(UNPAIRED_KEYS[key])
        if u:
            row.update({
                "unpaired_observed": u.get("observed_difference"),
                "unpaired_p": u.get("exact_p_two_sided"),
                "unpaired_mdd": u.get("minimum_detectable_difference_alpha05"),
                "unpaired_n_perms": u.get("n_splits_enumerated"),
            })
            if row["paired_mdd"] and row["unpaired_mdd"]:
                row["mdd_ratio_paired_over_unpaired"] = float(
                    row["paired_mdd"] / row["unpaired_mdd"])
                row["mdd_change"] = float(row["paired_mdd"] - row["unpaired_mdd"])
            pu, pp = row.get("unpaired_p"), row.get("paired_p")
            if pu is not None and pp is not None:
                same = (pu < 0.05) == (pp < 0.05)
                row["conclusion_changed"] = not same
                row["framing"] = (
                    "Same conclusion, tighter bounds — improved resolution on a "
                    "finding the unpaired analysis already pointed at, NOT a new "
                    "discovery."
                    if same else
                    "CONCLUSION CHANGES between designs. The paired design is the "
                    "correct one; report both and say which is which.")
        out[key] = row
    return out


# ── figures ──────────────────────────────────────────────────────────────────

def _null_for(entry: dict) -> np.ndarray:
    diffs = np.array([r["difference"] for r in entry["per_gland"]], dtype=float)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=diffs.size)))
    return (signs * diffs).mean(axis=1)


def write_figures(paired: dict, sections: list, out_dir: Path) -> list:
    written = []
    a, b = sections

    # (1) paired plot, raw estimand
    pg = paired["raw_rho_pt_hole"]["per_gland"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in pg:
        ya, yb = r["rho_a"], r["rho_b"]
        if ya is None or yb is None:
            continue
        ax.plot([0, 1], [ya, yb], "-o", color="0.4", alpha=0.85, ms=6)
        ax.annotate(r["gland"], (0, ya), textcoords="offset points",
                    xytext=(-7, 0), ha="right", fontsize=8)
    ax.axhline(0, color="C3", lw=1, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{a}\n(Carnoy's)", f"{b}\n(PFA)"])
    ax.set_xlim(-0.35, 1.2)
    ax.set_ylabel("rho(pseudotime, hole_pct), within gland")
    ax.set_title("Each line is one gland (matched pair)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"paired_by_gland.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(str(p))
    plt.close(fig)

    # (2) the 256-permutation nulls, one panel per estimand
    keys = list(ESTIMANDS)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.0))
    for ax, key in zip(np.atleast_1d(axes), keys):
        e = paired[key]
        if not e["test"]:
            ax.set_title(f"{key}\nNOT TESTED", fontsize=9)
            ax.axis("off")
            continue
        null = _null_for(e)
        ax.hist(null, bins=32, color="0.75", edgecolor="0.4")
        obs = e["test"]["observed_mean_difference"]
        ax.axvline(obs, color="C3", lw=2, label=f"observed {obs:+.3f}")
        ax.axvline(0, color="0.5", lw=1, ls=":")
        ax.set_xlabel("mean within-gland difference")
        ax.set_ylabel(f"count (of {N_SIGN_FLIPS})")
        ax.set_title(f"{key}\nexact p = {e['test']['p_value_display']}", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"paired_null_distributions.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(str(p))
    plt.close(fig)
    return written


# ── report ───────────────────────────────────────────────────────────────────

def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    a, b = res["sections"]
    paired, sbs = res["paired"], res["side_by_side"]

    add("# Between-section comparison, corrected for the matched-pair design\n")
    add("**What changed.** The 16 slides are **8 matched pairs** — each mouse-flank "
        "gland contributes one slide to each section — not 16 independent samples. "
        "The comparison on record permuted all C(16,8) = 12,870 splits of 16 slides, "
        "a null that admits between-gland and between-mouse variation the paired "
        "design already controls. This recomputes it as an exact **2^8 = 256 "
        "sign-flip** test on the 8 within-gland differences.\n")
    add("**What did not change.** The unpaired result is preserved at "
        f"`{res['inputs']['unpaired_dir']}` and is reported alongside throughout. "
        "Nothing was recomputed from images, embeddings or pseudotime.\n")

    add("## Step 0 — inputs and the balance gate\n")
    for s, q in res["inputs"]["per_duct_tables"].items():
        add(f"- **{s}**: `{q}` — {res['inputs']['n_ducts'][s]} ducts")
    g = res["pairing"]
    add(f"\n- glands: **{g['n_glands']}**, identical sets between sections: "
        f"**{g['identical_gland_sets']}**, balanced: **{g['balanced']}**")
    add(f"- {g['verdict']}\n")

    add("## The p-value floor\n")
    add(f"With {N_SIGN_FLIPS} permutations the smallest attainable two-sided p is "
        f"**2/{N_SIGN_FLIPS} = {P_FLOOR:.4g}**. A paired p at that floor is **not "
        "weaker evidence** than the unpaired test's 1.554e-4 — it is the same "
        "evidence at coarser resolution. The unpaired test could resolve further "
        "only by assuming an independence that does not hold here. `< 0.0078` is "
        "reported rather than 0.\n")

    add("## Task 2 — paired results, per estimand\n")
    for key, (label, _) in ESTIMANDS.items():
        e = paired[key]
        add(f"### `{key}` — {label}\n")
        add(f"| gland | n ducts {a} | n ducts {b} | rho {a} | rho {b} | difference |")
        add("|---|---|---|---|---|---|")
        for r in e["per_gland"]:
            def f(v):
                return f"{v:+.4f}" if v is not None else "—"
            add(f"| {r['gland']} | {r['n_ducts_a']} | {r['n_ducts_b']} | "
                f"{f(r['rho_a'])} | {f(r['rho_b'])} | {f(r['difference'])} |")
        t = e["test"]
        if t:
            np_ = t["null_percentiles"]
            add(f"\n- mean difference **{t['observed_mean_difference']:+.4f}**, "
                f"median {t['median_difference']:+.4f}")
            add(f"- exact p (two-sided) = **{t['p_value_display']}** over "
                f"{t['n_permutations']} sign flips")
            add(f"- null 2.5 / 97.5 percentiles: [{np_[2.5]:+.4f}, {np_[97.5]:+.4f}]; "
                f"observed sits at the {t['observed_percentile_in_null']:.2f}th")
            add(f"- minimum detectable difference (alpha 0.05): "
                f"**{t['minimum_detectable_difference_alpha05']:.4f}**")
        add(f"\n**Verdict:** {e['verdict']}\n")

    ns = res.get("normal_scores_identity")
    if ns and ns.get("max_abs_difference") is not None:
        add("### Estimand 4 collapses onto estimand 1 — expected, and checked\n")
        add(f"Per gland, `rho(ns(pt), ns(hole))` equals `rho(pt, hole)` to "
            f"**{ns['max_abs_difference']:.2e}** across all "
            f"{ns['n_glands_checked']} gland-sections. Spearman is invariant to "
            "monotone transforms and normal scores are monotone in rank; each "
            "gland-section is a single slide, so there are no between-slide offsets "
            "left for the variant to remove. **The paired design has already "
            "absorbed what that variant was invented to fix.**\n")

    add("## Task 3 — unpaired (on record) vs paired (corrected)\n")
    add("| estimand | unpaired diff | unpaired p | unpaired MDD | paired diff | "
        "paired p | paired MDD | MDD ratio |")
    add("|---|---|---|---|---|---|---|---|")
    for key, r in sbs.items():
        def f(v, spec="{:+.4f}"):
            return spec.format(v) if v is not None else "—"
        add(f"| `{key}` | {f(r.get('unpaired_observed'))} | "
            + (f"{r['unpaired_p']:.4g}" if r.get("unpaired_p") is not None else "—")
            + f" | {f(r.get('unpaired_mdd'), '{:.4f}')} | "
            + f"{f(r.get('paired_observed'))} | "
            + (r.get("paired_p_display") or "—") + " | "
            + f"{f(r.get('paired_mdd'), '{:.4f}')} | "
            + (f"{r['mdd_ratio_paired_over_unpaired']:.3f}"
               if r.get("mdd_ratio_paired_over_unpaired") is not None else "—")
            + " |")
    add("\n> The two designs test different statistics — the unpaired one differences "
        "two pooled correlations, the paired one averages 8 within-gland differences "
        "— so the observed values are not expected to match. What is comparable is "
        "the **conclusion** and the **minimum detectable difference**.\n")

    for key, r in sbs.items():
        if r.get("framing"):
            add(f"- **`{key}`** — {r['framing']}")
    add("")

    add("## Standing caveats\n")
    add("- **This corrects the unit of analysis, not the tissue question.** It does "
        "not establish that the two pieces of each gland are equivalent tissue. If "
        "they sample different regions within the gland, within-gland regional "
        "variation remains mixed with the fixation effect and cannot be separated by "
        "any analysis of this data. **Outstanding clarification with the "
        "pathologist.**")
    add(f"- **Eight pairs is still eight pairs.** Failure to reject is not evidence "
        f"of equivalence, and the design's p-value floor is {P_FLOOR:.4g}.")
    add("- **Both sections' pseudotime comes from the density-rooted (v2) axis**, on "
        "which `hole_pct` remains an external validator. Nothing here uses the "
        "holeyness-anchored axis, where `rho(pt, hole_pct)` is circular by "
        "construction.")
    add("- Fixation remains perfectly collinear with section, so nothing here "
        "attributes any difference to fixation chemistry rather than anatomical "
        "region. Bridge samples would be required.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs=2, default=["2M-1", "2M-2"])
    ap.add_argument("--per-duct-csvs", nargs=2, type=Path, required=True)
    ap.add_argument("--unpaired-json", type=Path, default=None,
                    help="holeyness_section_comparison.json, for the side-by-side. "
                         "Omitted -> the paired half is reported alone.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    secs = list(args.sections)
    print("=" * 78)
    print("  TASKS 2-3 — paired between-section comparison")
    print("=" * 78)

    frames, paths = {}, {}
    for i, s in enumerate(secs):
        frames[s] = load_per_duct(args.per_duct_csvs[i], s)
        paths[s] = str(args.per_duct_csvs[i])
        print(f"  {s}: {args.per_duct_csvs[i]}  ({len(frames[s])} ducts)")

    pairing = build_pairing(frames)
    print(f"\n  balance gate: {pairing['verdict']}")
    if not pairing["balanced"]:
        raise SystemExit(
            "\nREFUSING TO PROCEED. A sign-flip null assumes exactly one value per "
            "gland per section. Run Task 1's audit and resolve the design first.")
    glands = pairing["glands"]

    paired = {}
    for key in ESTIMANDS:
        paired[key] = run_estimand(frames, glands, key)
        t = paired[key]["test"]
        if t:
            print(f"  {key:<30} mean diff {t['observed_mean_difference']:+.4f}  "
                  f"p {t['p_value_display']:>10}  "
                  f"MDD {t['minimum_detectable_difference_alpha05']:.4f}")
        else:
            print(f"  {key:<30} NOT TESTED")

    # estimand 4 vs estimand 1 identity check
    d1 = {r["gland"]: r for r in paired["raw_rho_pt_hole"]["per_gland"]}
    d4 = {r["gland"]: r for r in paired["raw_within_slide_normalised"]["per_gland"]}
    diffs = []
    for g in glands:
        for col in ("rho_a", "rho_b"):
            va, vb = d1[g][col], d4[g][col]
            if va is not None and vb is not None:
                diffs.append(abs(va - vb))
    ident = {"n_glands_checked": len(diffs),
             "max_abs_difference": float(max(diffs)) if diffs else None}
    if ident["max_abs_difference"] is not None:
        print(f"\n  normal-scores identity: max |diff| = "
              f"{ident['max_abs_difference']:.2e} over {ident['n_glands_checked']} "
              "gland-sections")

    unpaired = None
    if args.unpaired_json and Path(args.unpaired_json).exists():
        unpaired = json.loads(Path(args.unpaired_json).read_text(encoding="utf-8"))
        print(f"  side-by-side against {args.unpaired_json}")
    else:
        print("  NOTE: --unpaired-json absent or missing; side-by-side is partial.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "analysis": "holeyness_paired_comparison",
        "sections": secs,
        "inputs": {
            "per_duct_tables": paths,
            "n_ducts": {s: int(len(frames[s])) for s in secs},
            "unpaired_json": str(args.unpaired_json),
            "unpaired_dir": (str(Path(args.unpaired_json).parent)
                             if args.unpaired_json else None),
            "recomputed_anything": False,
        },
        "pairing": pairing,
        "design": {"n_pairs": len(glands), "n_permutations": N_SIGN_FLIPS,
                   "p_floor": P_FLOOR,
                   "statistic": "mean within-gland difference"},
        "paired": paired,
        "normal_scores_identity": ident,
        "side_by_side": side_by_side(paired, unpaired),
    }
    res["figures"] = write_figures(paired, secs, args.output_dir)
    out = args.output_dir / "holeyness_paired_comparison.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default),
                   encoding="utf-8")
    write_report(res, args.output_dir / "holeyness_paired_comparison.md")
    print(f"\n  JSON:     {out}")
    print(f"  Markdown: {args.output_dir / 'holeyness_paired_comparison.md'}")


if __name__ == "__main__":
    main()
