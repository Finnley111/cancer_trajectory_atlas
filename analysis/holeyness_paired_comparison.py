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
from math import comb as _comb
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


# -- Fixation shrinkage, isotropy, and the scale-invariance question ----------

def fixation_shrinkage(frames: dict, glands: list) -> dict:
    """Per-gland shrinkage of duct area, hole area and hole %, on matched tissue.

    This is the one measurement the cohort is uniquely positioned to make: the same
    gland split across two fixations, so a within-gland ratio isolates the fixation
    effect from every between-animal and between-gland source of variation.

    Carnoy's is coagulative and shrinks tissue; PFA cross-links and preserves volume.
    If that is what is happening, Carnoy's/PFA ratios should sit below 1 consistently.
    With 8 glands the exact sign test floors at 2/256 = 0.0078, the same floor as the
    paired permutation test and for the same reason.
    """
    a, b = list(frames)
    quantities = {"median_area_um2": "area_um2",
                  "median_hole_area_um2": "hole_area_um2",
                  "median_hole_pct": "hole_pct",
                  "n_ducts": None}
    rows = []
    for g in glands:
        da = frames[a][frames[a]["gland"] == g]
        db = frames[b][frames[b]["gland"] == g]
        e = {"gland": g}
        for qname, col in quantities.items():
            if col is None:
                va, vb = float(len(da)), float(len(db))
            else:
                va = float(np.nanmedian(da[col].values)) if col in da else float("nan")
                vb = float(np.nanmedian(db[col].values)) if col in db else float("nan")
            e[qname + "_a"] = va
            e[qname + "_b"] = vb
            e[qname + "_ratio"] = ((va / vb) if (np.isfinite(va) and np.isfinite(vb)
                                                 and vb != 0) else None)
        # log-area spread: a PURE multiplicative shrink leaves this unchanged
        for tag, d in (("a", da), ("b", db)):
            v = d["area_um2"].values.astype(float)
            v = v[np.isfinite(v) & (v > 0)]
            e["log_area_sd_" + tag] = float(np.std(np.log(v), ddof=1)) if v.size > 2 else None
        sa, sb = e.get("log_area_sd_a"), e.get("log_area_sd_b")
        e["log_area_sd_ratio"] = ((sa / sb) if (sa and sb) else None)
        rows.append(e)

    summary = {}
    for qname in list(quantities) + ["log_area_sd"]:
        key = qname + "_ratio"
        r = np.array([x[key] for x in rows if x.get(key) is not None], dtype=float)
        if r.size == 0:
            summary[qname] = None
            continue
        n_below, n = int((r < 1).sum()), int(r.size)
        tail = sum(_comb(n, k) for k in range(0, min(n_below, n - n_below) + 1))
        p_exact = min(1.0, 2.0 * tail / (2 ** n))
        summary[qname] = {
            "n_glands": n, "n_ratio_below_1": n_below,
            "median_ratio": float(np.median(r)),
            "min_ratio": float(r.min()), "max_ratio": float(r.max()),
            "fold_change_b_over_a": float(1.0 / np.median(r)) if np.median(r) else None,
            "sign_test_p_two_sided": float(p_exact),
            "consistent": bool(n_below == n or n_below == 0),
        }
    return {"section_a": a, "section_b": b, "per_gland": rows, "summary": summary}


def isotropy_check(shrink: dict) -> dict:
    """Do lumen and duct shrink by the SAME factor?

    If Carnoy's compression were isotropic, hole area and duct area would shrink by
    the same factor and hole_pct -- their ratio -- would be preserved. hole_pct is
    NOT preserved, so isotropy and that observation cannot both hold. This asks which
    one gives.

    It bears directly on the validation: if lumens collapse more than the duct as a
    whole, hole_pct is not a fixation-invariant measurement, and comparing it ACROSS
    sections compares two different quantities. Within a section it is unaffected.
    """
    rows = []
    for e in shrink["per_gland"]:
        ra = e.get("median_area_um2_ratio")
        rh = e.get("median_hole_area_um2_ratio")
        rows.append({
            "gland": e["gland"], "area_ratio": ra, "hole_area_ratio": rh,
            "hole_pct_ratio": e.get("median_hole_pct_ratio"),
            "hole_shrinks_more": (bool(rh < ra) if (ra is not None and rh is not None)
                                  else None),
            "anisotropy_hole_over_area": ((rh / ra) if (ra not in (None, 0)
                                                        and rh is not None) else None),
        })
    vals = [r["anisotropy_hole_over_area"] for r in rows
            if r["anisotropy_hole_over_area"] is not None]
    scored = [r for r in rows if r["hole_shrinks_more"] is not None]
    n = len(scored)
    n_more = sum(1 for r in scored if r["hole_shrinks_more"])
    tail = sum(_comb(n, k) for k in range(0, min(n_more, n - n_more) + 1)) if n else 0
    p_exact = (min(1.0, 2.0 * tail / (2 ** n)) if n else None)
    med = float(np.median(vals)) if vals else None
    isotropic = bool(med is not None and 0.9 <= med <= 1.1)
    return {
        "per_gland": rows, "n_glands": n,
        "n_glands_hole_shrinks_more": n_more,
        "median_anisotropy_hole_over_area": med,
        "sign_test_p_two_sided": p_exact,
        "isotropic": isotropic,
        "verdict": (
            "ISOTROPIC -- lumen and duct shrink by comparable factors, so hole_pct is "
            "approximately preserved across fixations and comparing it between "
            "sections is safe on this ground."
            if isotropic else
            "ANISOTROPIC -- the lumen shrinks by a different factor from the duct in "
            + str(n_more) + "/" + str(n) + " glands (median hole-area/duct-area "
            "shrink ratio " + (("%.3f" % med) if med is not None else "n/a") +
            ", sign-test p = " + (("%.4g" % p_exact) if p_exact is not None else "n/a")
            + "). **hole_pct is therefore NOT a fixation-invariant measurement.** The "
            "validation WITHIN each section is unaffected -- hole_pct still ranks "
            "ducts correctly there -- but any CROSS-SECTION comparison of hole_pct "
            "compares two different quantities and must carry this caveat."),
    }


def area_scale_invariance(frames: dict, glands: list) -> dict:
    """Can the shrinkage explain the rho(duct area, pseudotime) divergence?

    Two questions with different answers.

    PER GLAND: no, and it is settled by construction. Spearman is invariant to ANY
    monotone transform, so multiplying a gland's duct areas by 0.5, or raising them to
    a power, leaves its within-gland rho(area, pt) exactly unchanged. The paired
    test's per-gland correlations are therefore already immune to uniform compression
    however non-linear, and the 8/8 divergence they show cannot be a rescaling
    artifact. Demonstrated numerically below rather than asserted.

    POOLED ACROSS GLANDS: possibly, and worth measuring. The cohort-level values on
    record pool ducts from 8 glands whose area scales differ, so between-gland and
    between-section scale differences DO enter there. Recomputing the pooled
    correlation after rank-normalising area within each gland removes exactly that.

    WHAT NEITHER CAN RULE OUT: a NON-monotone distortion, i.e. compression that
    reorders ducts by size rather than rescaling them. That cannot be tested here at
    all -- the two halves of a gland are different physical slides with different
    ducts, so there is no duct-to-duct correspondence. The log-area spread ratio in
    the shrinkage block is the closest available proxy: a pure multiplicative shrink
    leaves it at 1.
    """
    a, b = list(frames)
    out = {"per_gland_demonstration": [], "pooled": {}}

    for g in glands:
        e = {"gland": g}
        for tag, s in (("a", a), ("b", b)):
            d = frames[s][frames[s]["gland"] == g]
            raw = _rho(d["area_um2"].values, d["pseudotime"].values)
            rn = _rho(rankdata(d["area_um2"].values), d["pseudotime"].values)
            e["rho_" + tag + "_raw"] = raw
            e["rho_" + tag + "_rank_normalised"] = rn
            e["abs_diff_" + tag] = (abs(raw - rn) if (raw is not None and rn is not None)
                                    else None)
        out["per_gland_demonstration"].append(e)
    diffs = [e["abs_diff_" + t] for e in out["per_gland_demonstration"]
             for t in ("a", "b") if e.get("abs_diff_" + t) is not None]
    out["per_gland_max_abs_difference"] = float(max(diffs)) if diffs else None

    for s in (a, b):
        df = frames[s]
        pooled_raw = _rho(df["area_um2"].values, df["pseudotime"].values)
        rn = np.full(len(df), np.nan)
        av = df["area_um2"].values.astype(float)
        gv = df["gland"].values
        for g in glands:
            m = (gv == g)
            if m.sum() >= 3:
                rn[m] = (rankdata(av[m]) - 0.5) / m.sum()
        pooled_rn = _rho(rn, df["pseudotime"].values)
        out["pooled"][s] = {
            "raw": pooled_raw,
            "within_gland_rank_normalised": pooled_rn,
            "change": ((pooled_rn - pooled_raw)
                       if (pooled_raw is not None and pooled_rn is not None) else None)}

    pa, pb = out["pooled"][a], out["pooled"][b]
    if all(v["raw"] is not None and v["within_gland_rank_normalised"] is not None
           for v in (pa, pb)):
        d_raw = pa["raw"] - pb["raw"]
        d_rn = (pa["within_gland_rank_normalised"]
                - pb["within_gland_rank_normalised"])
        out["pooled_difference_raw"] = float(d_raw)
        out["pooled_difference_rank_normalised"] = float(d_rn)
        pct = (100.0 * d_rn / d_raw) if d_raw != 0 else float("nan")
        out["fraction_of_pooled_difference_surviving"] = (
            float(d_rn / d_raw) if d_raw != 0 else None)
        survives = abs(d_rn) > 0.5 * abs(d_raw)
        out["verdict"] = (
            ("SURVIVES. Removing within-gland area scale leaves the pooled difference "
             "at %+.4f, %.0f%% of its original %+.4f. Combined with the per-gland "
             "invariance above -- where the correlations are immune to rescaling by "
             "construction -- the shrinkage does NOT explain the divergence."
             % (d_rn, pct, d_raw))
            if survives else
            ("LARGELY REMOVED. The pooled difference falls from %+.4f to %+.4f "
             "(%.0f%%) once within-gland area scale is taken out, so much of the "
             "POOLED value was scale-driven. This does not touch the per-gland "
             "result, which is invariant by construction and still shows the "
             "divergence." % (d_raw, d_rn, pct)))
    return out


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

    sh = res.get("fixation_shrinkage")
    if sh:
        add("## Fixation shrinkage, measured on matched tissue\n")
        add("This is the measurement the cohort is uniquely positioned to make. The "
            "same gland is split across two fixations, so a **within-gland ratio "
            "isolates the fixation effect** from every between-animal and "
            "between-gland source of variation. Carnoy's is coagulative and shrinks "
            "tissue; PFA cross-links and preserves volume.\n")
        add(f"| gland | median area {a} | median area {b} | ratio | median hole area {a} | median hole area {b} | ratio | hole % ratio |")
        add("|---|---|---|---|---|---|---|---|")
        for e in sh["per_gland"]:
            def g(k, spec="{:.0f}"):
                v = e.get(k)
                return spec.format(v) if v is not None and np.isfinite(v) else "—"
            add(f"| {e['gland']} | {g('median_area_um2_a')} | "
                f"{g('median_area_um2_b')} | {g('median_area_um2_ratio', '{:.3f}')} | "
                f"{g('median_hole_area_um2_a')} | {g('median_hole_area_um2_b')} | "
                f"{g('median_hole_area_um2_ratio', '{:.3f}')} | "
                f"{g('median_hole_pct_ratio', '{:.3f}')} |")
        add("")
        add("| quantity | " + a + " < " + b + " | median ratio | fold change " + b
            + "/" + a + " | ratio range | exact sign-test p |")
        add("|---|---|---|---|---|---|")
        for q, lbl in (("median_area_um2", "duct area"),
                       ("median_hole_area_um2", "hole area"),
                       ("median_hole_pct", "hole %"),
                       ("n_ducts", "n ducts"),
                       ("log_area_sd", "log-area SD")):
            e = sh["summary"].get(q)
            if not e:
                continue
            fold = (f"{e['fold_change_b_over_a']:.2f}x"
                    if e.get("fold_change_b_over_a") else "—")
            add(f"| {lbl} | **{e['n_ratio_below_1']}/{e['n_glands']}** | "
                f"{e['median_ratio']:.3f} | {fold} | "
                f"[{e['min_ratio']:.2f}, {e['max_ratio']:.2f}] | "
                f"**{e['sign_test_p_two_sided']:.4g}** |")
        add("\n> The sign test shares the paired permutation test's floor of "
            f"{P_FLOOR:.4g}, and for the same reason: with 8 glands, unanimity is the "
            "most extreme pattern available.\n")
        add("> **Reconciles with the earlier finding, it does not contradict it.** A "
            "previous diagnostic reported that the annotation behaves identically "
            "across sections, on the basis that `rho(area, hole_pct)` is +0.386 vs "
            "+0.361. That is about the *relationship* between the two variables, "
            "which a monotone rescaling preserves exactly. What shifts here is the "
            "*level* of both. Both statements are true simultaneously.\n")
        add("> **log-area SD** is the closest available check on whether the "
            "compression is a pure multiplicative shrink: a uniform scale factor "
            "leaves it at 1. It cannot be checked directly, because the two halves of "
            "a gland are different physical slides with different ducts and there is "
            "no duct-to-duct correspondence.\n")

    iso = res.get("isotropy")
    if iso:
        add("## Is the shrinkage isotropic? (bears on the validation itself)\n")
        add("If compression were isotropic, hole area and duct area would shrink by "
            "the same factor and `hole_pct` — their ratio — would be preserved. It is "
            "not preserved, so isotropy and that observation cannot both hold.\n")
        add("| gland | duct-area ratio | hole-area ratio | hole shrinks more? | anisotropy |")
        add("|---|---|---|---|---|")
        for r in iso["per_gland"]:
            def g(k):
                v = r.get(k)
                return f"{v:.3f}" if v is not None else "—"
            add(f"| {r['gland']} | {g('area_ratio')} | {g('hole_area_ratio')} | "
                + ("yes" if r["hole_shrinks_more"] else
                   ("no" if r["hole_shrinks_more"] is not None else "—"))
                + f" | {g('anisotropy_hole_over_area')} |")
        add(f"\n**Verdict:** {iso['verdict']}\n")

    sc = res.get("area_scale_invariance")
    if sc:
        add("## Can the shrinkage explain the rho(area, pseudotime) divergence?\n")
        add("**Per gland: no, by construction.** Spearman is invariant to any "
            "monotone transform, so compressing a gland's duct areas — by a constant "
            "factor or a power law — leaves its within-gland `rho(area, pt)` exactly "
            "unchanged. Verified on the real data: rank-normalising area within each "
            "gland changes the per-gland correlation by at most "
            f"**{sc['per_gland_max_abs_difference']:.2e}**. The paired test's 8/8 "
            "divergence is built from exactly these correlations, so it cannot be a "
            "rescaling artifact.\n")
        if "verdict" in sc:
            add("**Pooled across glands: measurable, and measured.** The cohort-level "
                "values pool ducts from 8 glands whose area scales differ, so scale "
                "differences do enter there.\n")
            add("| section | pooled rho(area, pt) | after within-gland rank-normalisation | change |")
            add("|---|---|---|---|")
            for sec in (a, b):
                e = sc["pooled"][sec]
                add(f"| {sec} | {e['raw']:+.4f} | "
                    f"{e['within_gland_rank_normalised']:+.4f} | "
                    f"{e['change']:+.4f} |")
            add(f"\n- pooled difference, raw: **{sc['pooled_difference_raw']:+.4f}**")
            add(f"- pooled difference, within-gland rank-normalised: "
                f"**{sc['pooled_difference_rank_normalised']:+.4f}**")
            add(f"\n**Verdict:** {sc['verdict']}\n")
        add("> **What neither test can rule out** is a NON-monotone distortion — "
            "compression that reorders ducts by size rather than rescaling them. "
            "There is no duct-to-duct correspondence between the two halves of a "
            "gland, so it cannot be tested from this data at all.\n")

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
    if res.get("isotropy") and not res["isotropy"]["isotropic"]:
        add("- **`hole_pct` is not fixation-invariant.** The lumen and the duct "
            "shrink by different factors, so `hole_pct` measured under Carnoy's and "
            "`hole_pct` measured under PFA are not the same quantity. The validation "
            "WITHIN each section is unaffected. Any CROSS-SECTION replication claim "
            "about `hole_pct` must carry this caveat.")
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

    shrink = fixation_shrinkage(frames, glands)
    iso = isotropy_check(shrink)
    scale = area_scale_invariance(frames, glands)
    a, b = secs
    print("\n  === fixation shrinkage, matched tissue ===")
    for q in ("median_area_um2", "median_hole_area_um2", "median_hole_pct",
              "log_area_sd"):
        e = shrink["summary"].get(q)
        if not e:
            continue
        print(f"  {q:<22} {a}<{b} in {e['n_ratio_below_1']}/{e['n_glands']} glands  "
              f"median ratio {e['median_ratio']:.3f}  sign-test p "
              f"{e['sign_test_p_two_sided']:.4g}")
    print(f"  isotropy: {iso['verdict'][:96]}...")
    print(f"  per-gland rho(area,pt) rank-normalisation max |diff| = "
          f"{scale['per_gland_max_abs_difference']:.2e}")
    if "verdict" in scale:
        print(f"  scale invariance: {scale['verdict'][:96]}...")

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
        "fixation_shrinkage": shrink,
        "isotropy": iso,
        "area_scale_invariance": scale,
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
