"""Do the two sections' holey-ness correlations actually differ?

WHAT THIS TESTS
---------------
A prior diagnostic corrected a circulated error: 2M-2's "0.020" was
rho(pseudotime, nuclear_density), not rho(pseudotime, hole_pct). Recomputed from
the per-duct tables the external validation is POSITIVE IN BOTH sections —
0.276 (2M-1, CI [0.217, 0.347], 8/8 slides) and 0.1906 (2M-2, CI [0.086, 0.294],
7/8 slides) — and both candidate explanations for an asymmetry were ruled out:
hole_pct is not rank-compressed in 2M-2 (rank_sd_ratio 1.0000 in both, >99%
distinct values) and the annotation behaves identically across sections
(rho(area, hole_pct) +0.386 vs +0.361, 8/8 slides positive in both).

Two things follow, and this module does both.

  TASK 1  Complete the correlation table. Only 2M-1's area-adjusted value
          (0.131) is on record. Compute raw, area-adjusted and
          area+nuclear_density-adjusted for BOTH sections, each with a
          slide-clustered bootstrap interval and a within-slide-shuffled
          permutation p-value, so raw and adjusted are directly comparable.

  TASK 2  Test the between-section difference exactly. With 8 slides per section
          a bootstrap of the difference is coarse, so instead enumerate ALL
          C(16,8) = 12,870 ways to relabel the 16 slides into two groups of 8 and
          build the exact null distribution of the difference in rho.

WHAT THIS CANNOT ESTABLISH
--------------------------
Fixation is perfectly collinear with section in this cohort — every Carnoy's
slide is 2M-1 and every PFA slide is 2M-2 — so no difference found here, and no
absence of one, can be attributed to fixation chemistry rather than anatomical
region. Bridge samples (serial sections from one block, split across both
fixations, stained in one run) would be required. It also cannot settle whether
duct area is a mediator or a confounder on the pseudotime-to-holey-ness path;
both estimands are reported and neither is designated correct.

WHICH PARTIAL-CORRELATION FUNCTION, AND WHY IT MATTERS
------------------------------------------------------
``holeyness.py`` carries two, and they are not the same computation:

  ``_partial_spearman``        the ALGEBRAIC three-correlation formula. This is
                               what produced the on-record 0.131.
  ``_partial_spearman_multi``  RANK-RESIDUAL: rank-transform, residualise by OLS,
                               Pearson r of the residuals. Handles one or many
                               controls uniformly.

The rank-residual form is used here as primary, per the analysis request. Its
docstring notes the two are algebraically equivalent with a single control; that
holds exactly only without ties, since the algebraic form uses tie-corrected
Spearman rhos while the residual form regresses on mid-ranks. hole_pct has
essentially no ties in either section, so they should agree — but both are
computed for the single-control case and any disagreement is flagged rather than
assumed away. 2M-1's area-adjusted value is additionally reconciled against the
on-record 0.131 as a consistency gate.

AN EXCHANGEABILITY PROBLEM IN TASK 2, MEASURED RATHER THAN ASSUMED AWAY
-----------------------------------------------------------------------
The exact test assumes the two groups of 8 slides are exchangeable under the
null. That is not obviously true here, because the sections differ substantially
in their MARGINAL distributions:

    hole_pct  median   4.25 (2M-1)  vs  17.45 (2M-2)
    area_um2  median  20653         vs  33807
    pseudotime median  0.0488       vs   0.0397

The observed statistic comes from the ONE split that is section-pure. Almost
every one of the other 12,870 splits is section-MIXED, and in a mixed group a
pooled Spearman picks up between-section contrast that a pure group never sees.
That can shift the null away from the kind of object the observed value is, and
the p-value inherits the bias in an unknown direction.

Three responses, all reported:

  1. The test exactly as specified, on the raw pooled statistic. PRIMARY.
  2. A PURITY-STRATIFIED null: the 12,870 splits grouped by how many 2M-1 slides
     landed in group A, with the mean and spread of the difference in each
     stratum. Note the null is exactly antisymmetric (a subset and its complement
     give equal and opposite differences), so stratum means are structurally
     +/- mirrored and the mean at 4/4 is exactly zero. What the stratification
     shows is whether the null's SPREAD is homogeneous across compositions; if it
     is not, the pooled null is a mixture and a single p-value drawn from it is
     hard to interpret. This is descriptive: it cannot by itself separate
     marginal contamination from a real graded effect, because both produce drift.
  3. A WITHIN-SLIDE-NORMALISED variant: pseudotime and hole_pct converted to
     within-slide normal scores before pooling. This removes between-slide (and
     therefore between-section) location and scale differences, so mixed and pure
     groups become comparable and the exchangeability assumption is restored.
     This is the contamination-free version of the test. Its observed statistic
     is a pooled WITHIN-slide correlation and will not equal the raw 0.276 /
     0.1906; that is expected, not an error.

READ-ONLY. Reads the existing per-duct tables only. Recomputes no pseudotime, no
duct assignment and no duct table; reruns no pipeline stage; modifies no module;
writes only to --output-dir.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from .holeyness import _partial_spearman, _partial_spearman_multi

N_BOOT_DEFAULT = 2000
N_PERM_DEFAULT = 5000
N_SLIDES_PER_SECTION = 8
# 2M-1's area-adjusted partial, as recorded by the v2 holeyness run.
ON_RECORD_2M1_PARTIAL_AREA = 0.131
ON_RECORD_TOL = 0.005
IMPL_AGREEMENT_TOL = 1e-6

REQUIRED_COLUMNS = [
    "object_id", "n_patches", "pseudotime", "nuclear_density",
    "packing_irregularity", "slide_name", "hole_pct", "hole_area_um2", "area_um2",
]


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r of mid-ranks — identical to scipy's spearmanr.statistic.

    Used in the 12,870-split enumeration where scipy's p-value machinery is pure
    overhead. Verified against scipy once at startup rather than trusted.
    """
    if x.size < 10:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _within_slide_normal_scores(values: np.ndarray, slide: np.ndarray) -> np.ndarray:
    """Convert to normal scores WITHIN each slide.

    Removes each slide's location and scale, so a pooled correlation over any mix
    of slides becomes an aggregate of within-slide correlations. This is what
    makes section-mixed and section-pure groups comparable in Task 2's null.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    for s in np.unique(slide):
        idx = np.flatnonzero(slide == s)
        v = values[idx]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        r = rankdata(v[ok])
        out[idx[ok]] = norm.ppf((r - 0.5) / ok.sum())
    return out


# ── estimands ────────────────────────────────────────────────────────────────

class Cols:
    """Column arrays for one pooled set of ducts, indexable by row positions."""

    def __init__(self, df: pd.DataFrame):
        self.pt = df["pseudotime"].values.astype(float)
        self.hole = df["hole_pct"].values.astype(float)
        self.area = df["area_um2"].values.astype(float)
        self.nd = df["nuclear_density"].values.astype(float)
        self.slide = df["slide_name"].astype(str).values
        self.section = df["section"].astype(str).values
        self.pt_ws = _within_slide_normal_scores(self.pt, self.slide)
        self.hole_ws = _within_slide_normal_scores(self.hole, self.slide)


def stat_raw(c: Cols, idx) -> float:
    return _fast_spearman(c.pt[idx], c.hole[idx])


def stat_partial_area(c: Cols, idx) -> float:
    return _partial_spearman_multi(c.pt[idx], c.hole[idx], [c.area[idx]])


def stat_partial_area_nd(c: Cols, idx) -> float:
    return _partial_spearman_multi(c.pt[idx], c.hole[idx], [c.area[idx], c.nd[idx]])


def stat_area_pt(c: Cols, idx) -> float:
    return _fast_spearman(c.area[idx], c.pt[idx])


def stat_raw_within_slide(c: Cols, idx) -> float:
    return _fast_spearman(c.pt_ws[idx], c.hole_ws[idx])


ESTIMANDS = {
    "raw_rho_pt_hole": ("rho(pseudotime, hole_pct)", stat_raw),
    "partial_given_area": ("rho(pt, hole_pct | area)", stat_partial_area),
    "partial_given_area_nd": ("rho(pt, hole_pct | area, nuclear_density)",
                              stat_partial_area_nd),
}


# ── TASK 1 ───────────────────────────────────────────────────────────────────

def slide_clustered_bootstrap(c: Cols, idx: np.ndarray, fn, n_boot: int,
                              seed: int) -> dict:
    """Resample SLIDES with replacement, then take all ducts of each drawn slide.

    Ducts are nested within slides, so the slide is the independent unit. A
    duct-level bootstrap would treat ~1,600 nested observations as independent
    and return an interval several times too narrow.
    """
    rng = np.random.default_rng(seed)
    slides = np.unique(c.slide[idx])
    by_slide = {s: idx[c.slide[idx] == s] for s in slides}
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        picked = rng.choice(len(slides), size=len(slides), replace=True)
        rows = np.concatenate([by_slide[slides[int(i)]] for i in picked])
        draws[b] = fn(c, rows)
    draws = draws[np.isfinite(draws)]
    return {
        "n_boot": int(draws.size),
        "n_slides": int(len(slides)),
        "ci95": ([float(np.percentile(draws, 2.5)),
                  float(np.percentile(draws, 97.5))] if draws.size else None),
        "ci80": ([float(np.percentile(draws, 10)),
                  float(np.percentile(draws, 90))] if draws.size else None),
        "resampling_unit": "slide, with replacement; all ducts of each drawn slide",
    }


def within_slide_permutation(c: Cols, idx: np.ndarray, fn, n_perm: int,
                             seed: int) -> dict:
    """Shuffle hole_pct WITHIN each slide, preserving the nesting."""
    rng = np.random.default_rng(seed)
    obs = fn(c, idx)
    slides = np.unique(c.slide[idx])
    pos_by_slide = [np.flatnonzero(c.slide[idx] == s) for s in slides]

    saved_hole, saved_hole_ws = c.hole.copy(), c.hole_ws.copy()
    null = np.full(n_perm, np.nan)
    try:
        for p in range(n_perm):
            h = saved_hole[idx].copy()
            hw = saved_hole_ws[idx].copy()
            for pos in pos_by_slide:
                perm = rng.permutation(pos.size)
                h[pos] = h[pos][perm]
                hw[pos] = hw[pos][perm]
            c.hole[idx] = h
            c.hole_ws[idx] = hw
            null[p] = fn(c, idx)
    finally:
        c.hole, c.hole_ws = saved_hole, saved_hole_ws

    null = null[np.isfinite(null)]
    n_ge = int((np.abs(null) >= abs(obs)).sum()) if null.size else 0
    if null.size == 0:
        p_str, p_val = "not computed", None
    elif n_ge == 0:
        p_str, p_val = f"< {1.0 / null.size:.3g}", float(1.0 / null.size)
    else:
        p_val = float(n_ge / null.size)
        p_str = f"{p_val:.4g}"
    return {
        "observed": obs, "n_perm": int(null.size),
        "n_null_at_least_as_extreme": n_ge,
        "p_value_two_sided": p_val, "p_value_display": p_str,
        "p_is_upper_bound": bool(n_ge == 0),
        "shuffle": "hole_pct permuted within slide",
    }


def task1(cols: dict, idx: dict, n_boot: int, n_perm: int, seed: int) -> dict:
    out = {"per_section": {}, "n_boot": n_boot, "n_perm": n_perm}
    for sec in cols:
        c, i = cols[sec], idx[sec]
        block = {"n_ducts": int(i.size), "n_slides": int(np.unique(c.slide[i]).size)}
        for key, (label, fn) in ESTIMANDS.items():
            point = fn(c, i)
            block[key] = {
                "label": label,
                "point_estimate": point,
                "bootstrap": slide_clustered_bootstrap(c, i, fn, n_boot, seed),
                "permutation": within_slide_permutation(c, i, fn, n_perm, seed),
            }
        # cross-check the two partial implementations on the single-control case
        alg = _partial_spearman(c.pt[i], c.hole[i], c.area[i])
        res = block["partial_given_area"]["point_estimate"]
        block["implementation_cross_check"] = {
            "rank_residual__partial_spearman_multi": res,
            "algebraic__partial_spearman": float(alg),
            "abs_difference": float(abs(alg - res)),
            "agree": bool(abs(alg - res) < IMPL_AGREEMENT_TOL),
            "tolerance": IMPL_AGREEMENT_TOL,
            "note": ("The two are algebraically equivalent with a single control "
                     "only in the absence of ties. Reported rather than assumed."),
        }
        out["per_section"][sec] = block

    a, b = list(cols)
    raw_gap = abs(out["per_section"][a]["raw_rho_pt_hole"]["point_estimate"]
                  - out["per_section"][b]["raw_rho_pt_hole"]["point_estimate"])
    adj_gap = abs(out["per_section"][a]["partial_given_area"]["point_estimate"]
                  - out["per_section"][b]["partial_given_area"]["point_estimate"])
    out["between_section_gap"] = {
        "raw_absolute_gap": float(raw_gap),
        "area_adjusted_absolute_gap": float(adj_gap),
        "adjusted_agree_more_closely": bool(adj_gap < raw_gap),
        "statement": (
            f"The area-adjusted estimates differ between sections by "
            f"{adj_gap:.4f}, the raw estimates by {raw_gap:.4f}. The adjusted "
            f"values therefore agree "
            + ("MORE" if adj_gap < raw_gap else "LESS")
            + " closely between sections than the raw values do."),
    }

    rec = out["per_section"].get(a, {}).get("partial_given_area", {}).get(
        "point_estimate")
    out["reconciliation_on_record"] = {
        "section": a,
        "on_record_value": ON_RECORD_2M1_PARTIAL_AREA,
        "recomputed": rec,
        "abs_difference": (float(abs(rec - ON_RECORD_2M1_PARTIAL_AREA))
                           if rec is not None else None),
        "agrees": (bool(abs(rec - ON_RECORD_2M1_PARTIAL_AREA) < ON_RECORD_TOL)
                   if rec is not None else None),
        "tolerance": ON_RECORD_TOL,
        "note": ("The on-record value came from the ALGEBRAIC implementation, so "
                 "compare it against the algebraic cross-check above if the "
                 "rank-residual value differs."),
    }
    out["mediator_or_confounder"] = (
        "OPEN QUESTION, NOT SETTLED HERE. The pathologist's account is that duct "
        "diameter increases with progression and hole count increases with "
        "diameter, which would make duct area a MEDIATOR and make adjusting for it "
        "an over-adjustment that removes real signal. But if rho(area, pseudotime) "
        "is substantially an anchor artifact — as size-matched resampling "
        "suggested, and as 2M-2's near-zero value is consistent with — then on the "
        "pseudotime-to-holey-ness path area is acting as a CONFOUNDER and "
        "adjustment is appropriate. Both estimands are reported for both sections. "
        "Which causal reading is correct is a biological question this analysis "
        "cannot settle.")
    return out


# ── TASK 2 ───────────────────────────────────────────────────────────────────

def exact_slide_permutation(c: Cols, all_idx: np.ndarray, fn, label: str,
                            section_of_slide: dict, section_a: str) -> dict:
    """Enumerate every C(16,8) relabelling of slides into two groups of 8.

    The null is exactly ANTISYMMETRIC: a subset and its complement yield equal
    and opposite differences. So it is symmetric with mean exactly zero by
    construction, and the smallest attainable two-sided p-value is 2/C(16,8) =
    1.55e-4 (the observed split and its mirror).
    """
    slides = sorted(np.unique(c.slide[all_idx]))
    if len(slides) != 2 * N_SLIDES_PER_SECTION:
        raise ValueError(
            f"Expected {2 * N_SLIDES_PER_SECTION} slides, found {len(slides)}. The "
            "exact enumeration is defined for a balanced 8/8 design.")
    rows_by_slide = {s: all_idx[c.slide[all_idx] == s] for s in slides}
    a_slides = {s for s in slides if section_of_slide[s] == section_a}

    diffs, purity = [], []
    observed = None
    for combo in combinations(range(len(slides)), N_SLIDES_PER_SECTION):
        pick = [slides[i] for i in combo]
        rest = [s for s in slides if s not in set(pick)]
        ia = np.concatenate([rows_by_slide[s] for s in pick])
        ib = np.concatenate([rows_by_slide[s] for s in rest])
        d = fn(c, ia) - fn(c, ib)
        k = sum(1 for s in pick if s in a_slides)
        diffs.append(d)
        purity.append(k)
        if k == N_SLIDES_PER_SECTION:
            observed = d
    diffs = np.asarray(diffs, dtype=float)
    purity = np.asarray(purity, dtype=int)
    finite = np.isfinite(diffs)

    if observed is None or not np.isfinite(observed):
        raise ValueError(f"{label}: the true section split produced no statistic.")

    n = int(finite.sum())
    n_ge = int((np.abs(diffs[finite]) >= abs(observed)).sum())
    p_exact = float(n_ge / n)

    strata = {}
    for k in range(N_SLIDES_PER_SECTION + 1):
        m = finite & (purity == k)
        if not m.any():
            continue
        v = diffs[m]
        strata[str(k)] = {
            "n_splits": int(v.size),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()),
        }

    mdd = float(np.percentile(np.abs(diffs[finite]), 97.5))
    return {
        "label": label,
        "observed_difference": float(observed),
        "n_splits_enumerated": n,
        "exhaustive": True,
        "exact_p_two_sided": p_exact,
        "n_null_at_least_as_extreme": n_ge,
        "min_attainable_p": float(2.0 / n),
        "null_percentiles": {p: float(np.percentile(diffs[finite], p))
                             for p in (0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5)},
        "null_mean": float(diffs[finite].mean()),
        "observed_percentile_in_null": float((diffs[finite] < observed).mean() * 100),
        "minimum_detectable_difference_alpha05": mdd,
        "purity_strata": strata,
        "purity_note": (
            "Stratified by how many 2M-1 slides fell in group A. The null is "
            "antisymmetric, so stratum means are structurally mirrored about 4/4 "
            "and the 4/4 mean is exactly zero; what to read here is whether the "
            "SPREAD is homogeneous across strata. If it is not, the pooled null is "
            "a mixture and one p-value drawn from it is hard to interpret. This "
            "cannot by itself separate marginal contamination from a real graded "
            "effect — both produce drift in the stratum means."),
        "verdict": (
            f"Observed difference {observed:+.4f}; exact two-sided p = "
            f"{p_exact:.4g} over {n} enumerated splits. "
            + ("There IS evidence the sections differ on this quantity."
               if p_exact < 0.05 else
               "There is NO EVIDENCE the sections differ on this quantity. That is "
               "not evidence of equivalence: with 8 slides per section this design "
               f"could not detect a true difference smaller than {mdd:.4f} at "
               "alpha = 0.05, so differences up to that magnitude cannot be "
               "excluded.")),
    }


def task2(c: Cols, all_idx: np.ndarray, section_of_slide: dict,
          section_a: str) -> dict:
    tests = {
        "raw_rho_pt_hole": ("difference in rho(pseudotime, hole_pct)", stat_raw),
        "partial_given_area": ("difference in rho(pt, hole_pct | area)",
                               stat_partial_area),
        "rho_area_pseudotime": ("difference in rho(duct area, pseudotime)",
                                stat_area_pt),
        "raw_within_slide_normalised": (
            "difference in rho(pt, hole_pct), within-slide normal scores "
            "[contamination-free variant]", stat_raw_within_slide),
    }
    out = {}
    for key, (label, fn) in tests.items():
        print(f"    enumerating: {label} ...")
        out[key] = exact_slide_permutation(c, all_idx, fn, label,
                                           section_of_slide, section_a)
    out["power_statement"] = (
        "POWER. Eight slides per section is a small design and this test is "
        "correspondingly weak. The minimum detectable difference at alpha = 0.05 "
        "two-sided, read off each exact null distribution's 97.5th percentile of "
        "|difference|, is reported per quantity above. A non-significant result "
        "here means NO EVIDENCE OF A DIFFERENCE — it is not evidence that the "
        "sections are equivalent, and differences up to the stated magnitude "
        "cannot be excluded by this data.")
    out["exchangeability_note"] = (
        "The observed statistic comes from the one split that is section-pure; "
        "almost all 12,870 splits are section-mixed. Because the sections differ "
        "in their marginal distributions of hole_pct, area and pseudotime, a mixed "
        "group's pooled Spearman can pick up between-section contrast that a pure "
        "group never sees. The within-slide-normalised variant above removes "
        "between-slide location and scale and is therefore the version whose null "
        "is exchangeable with its observed value; where the two variants disagree, "
        "prefer it and say so.")
    return out


# ── figures ──────────────────────────────────────────────────────────────────

def write_figures(t1: dict, t2: dict, sections: list, out_dir: Path) -> list:
    written = []

    rows = []
    for sec in sections:
        for key, tag in (("raw_rho_pt_hole", "raw"),
                         ("partial_given_area", "adjusted | area")):
            e = t1["per_section"][sec][key]
            ci = e["bootstrap"]["ci95"] or [np.nan, np.nan]
            rows.append((f"{sec}  {tag}", e["point_estimate"], ci[0], ci[1]))
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(rows) + 2))
    ypos = np.arange(len(rows))[::-1]
    for y, (lbl, pt, lo, hi) in zip(ypos, rows):
        ax.plot([lo, hi], [y, y], color="0.35", lw=2, solid_capstyle="round")
        ax.plot([pt], [y], "o", color="C0", ms=8, zorder=3)
    ax.axvline(0, color="C3", lw=1.2, ls="--")
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Spearman rho (slide-clustered 95% CI)")
    ax.set_title("Holey-ness correlation by section and estimand")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"forest_correlations.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight"); written.append(str(p))
    plt.close(fig)

    panels = ["raw_rho_pt_hole", "partial_given_area", "rho_area_pseudotime",
              "raw_within_slide_normalised"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2))
    for ax, key in zip(np.atleast_1d(axes), panels):
        r = t2[key]
        # the null is not stored in full; rebuild its shape from percentiles is
        # not honest, so plot the percentile ladder instead of a fake histogram
        pcts = r["null_percentiles"]
        xs = [pcts[p] for p in sorted(pcts)]
        ys = sorted(pcts)
        ax.plot(xs, ys, marker="o", color="0.4")
        ax.axvline(r["observed_difference"], color="C3", lw=2,
                   label=f"observed {r['observed_difference']:+.3f}")
        ax.axvline(0, color="0.7", lw=1, ls=":")
        ax.set_xlabel("difference in rho (group A - group B)")
        ax.set_ylabel("percentile of exact null")
        ax.set_title(f"{key}\nexact p = {r['exact_p_two_sided']:.4g}, "
                     f"n = {r['n_splits_enumerated']}", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"exact_null_distributions.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight"); written.append(str(p))
    plt.close(fig)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.0))
    for ax, key in zip(np.atleast_1d(axes), panels):
        st = t2[key]["purity_strata"]
        ks = sorted(int(k) for k in st)
        means = [st[str(k)]["mean"] for k in ks]
        sds = [st[str(k)]["sd"] for k in ks]
        ax.errorbar(ks, means, yerr=sds, marker="o", capsize=3, color="0.3")
        ax.axhline(t2[key]["observed_difference"], color="C3", lw=1.5, ls="--",
                   label="observed")
        ax.axhline(0, color="0.7", lw=1, ls=":")
        ax.set_xlabel("2M-1 slides in group A")
        ax.set_ylabel("difference in rho")
        ax.set_title(f"{key}\nnull by split purity (mean +/- sd)", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"null_by_split_purity.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight"); written.append(str(p))
    plt.close(fig)
    return written


# ── report ───────────────────────────────────────────────────────────────────

def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    a, b = res["sections"]
    t1, t2 = res["task_1"], res["task_2"]

    add("# Holey-ness correlations: 2M-1 vs 2M-2\n")
    add("**What this tests.** A prior diagnostic corrected a circulated error — "
        "2M-2's \"0.020\" was rho(pseudotime, nuclear_density), not "
        "rho(pseudotime, hole_pct) — and showed the external validation is "
        "positive in BOTH sections. This analysis completes the correlation table "
        "(raw and area-adjusted, both sections, with slide-clustered intervals and "
        "within-slide permutation p-values) and then tests whether the two "
        "sections' correlations differ, using an EXACT permutation over all "
        "C(16,8) = 12,870 slide-level relabellings rather than a coarse bootstrap "
        "of the difference. **What it cannot establish:** that fixation causes "
        "anything. Fixation is perfectly collinear with section here — every "
        "Carnoy's slide is 2M-1, every PFA slide is 2M-2 — so neither a difference "
        "nor its absence can be attributed to fixation chemistry as opposed to "
        "anatomical region; bridge samples would be required. It also cannot "
        "settle whether duct area is a mediator or a confounder; both estimands "
        "are reported and neither is called correct. Nothing was recomputed and no "
        "existing results directory was written to.\n")

    add("## Step 0 — paths and provenance\n")
    for sec, p in res["inputs"]["per_duct_tables"].items():
        add(f"- **{sec}**: `{p}` — {res['inputs']['n_rows'][sec]} ducts, "
            f"{res['inputs']['n_slides'][sec]} slides")
    add(f"\n- Partial correlation, primary: `{res['inputs']['partial_fn_primary']}` "
        "(rank-residual)")
    add(f"- Partial correlation, cross-check: `{res['inputs']['partial_fn_check']}` "
        "(algebraic; the source of the on-record 0.131)")
    add(f"- Fast Spearman verified against scipy: "
        f"max abs difference {res['inputs']['fast_spearman_max_error']:.3g}")
    add("\nNo duct table was recomputed; the per-duct tables are read as the "
        "holey-ness validation wrote them.\n")

    rc = t1["reconciliation_on_record"]
    add("### Consistency gate — reproducing the on-record value\n")
    add(f"- {rc['section']} `partial | area` on record: **{rc['on_record_value']}**; "
        f"recomputed: **{rc['recomputed']:+.4f}**; agrees within "
        f"{rc['tolerance']}: **{rc['agrees']}**")
    for sec in (a, b):
        x = t1["per_section"][sec]["implementation_cross_check"]
        add(f"- {sec} rank-residual {x['rank_residual__partial_spearman_multi']:+.6f} "
            f"vs algebraic {x['algebraic__partial_spearman']:+.6f} "
            f"(diff {x['abs_difference']:.2g}, agree: {x['agree']})")
    add("")

    add("## Task 1 — the correlation table\n")
    add("| section | estimand | rho | slide-clustered 95% CI | within-slide perm p |")
    add("|---|---|---|---|---|")
    for sec in (a, b):
        for key, (label, _) in ESTIMANDS.items():
            e = t1["per_section"][sec][key]
            ci = e["bootstrap"]["ci95"]
            add(f"| {sec} | `{label}` | **{e['point_estimate']:+.4f}** | "
                + (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—") + " | "
                + f"{e['permutation']['p_value_display']} |")
    add(f"\nBootstrap: {t1['n_boot']} resamples, unit = slide. Permutation: "
        f"{t1['n_perm']} within-slide shuffles; p reported as an upper bound when "
        "no permutation reached the observed value.\n")

    g = t1["between_section_gap"]
    add(f"**Do the adjusted values agree more closely than the raw ones?** "
        f"{g['statement']}\n")
    add(f"> {t1['mediator_or_confounder']}\n")

    add("## Task 2 — exact slide-level permutation on the between-section difference\n")
    add(f"**{t2['power_statement']}**\n")
    add("| quantity | observed diff | exact p | null 2.5/97.5 pct | min detectable diff | verdict |")
    add("|---|---|---|---|---|---|")
    for key in ("raw_rho_pt_hole", "partial_given_area", "rho_area_pseudotime",
                "raw_within_slide_normalised"):
        r = t2[key]
        np_ = r["null_percentiles"]
        add(f"| `{key}` | **{r['observed_difference']:+.4f}** | "
            f"{r['exact_p_two_sided']:.4g} | "
            f"[{np_[2.5]:+.4f}, {np_[97.5]:+.4f}] | "
            f"{r['minimum_detectable_difference_alpha05']:.4f} | "
            + ("**differ**" if r["exact_p_two_sided"] < 0.05
               else "no evidence of a difference") + " |")
    add(f"\nAll {t2['raw_rho_pt_hole']['n_splits_enumerated']} splits enumerated "
        "exhaustively; no subsampling. The null is exactly antisymmetric (a subset "
        "and its complement give equal and opposite differences), so its mean is "
        "zero by construction and the smallest attainable two-sided p-value is "
        f"{t2['raw_rho_pt_hole']['min_attainable_p']:.3g}.\n")

    for key in ("raw_rho_pt_hole", "partial_given_area", "rho_area_pseudotime",
                "raw_within_slide_normalised"):
        r = t2[key]
        add(f"**{r['label']}.** {r['verdict']} Observed sits at the "
            f"{r['observed_percentile_in_null']:.2f}th percentile of the null.\n")

    add("### Null by split purity\n")
    add("| quantity | " + " | ".join(f"{k}" for k in range(9)) + " |")
    add("|---" * 10 + "|")
    for key in ("raw_rho_pt_hole", "partial_given_area", "rho_area_pseudotime",
                "raw_within_slide_normalised"):
        st = t2[key]["purity_strata"]
        cells = []
        for k in range(9):
            e = st.get(str(k))
            cells.append(f"{e['mean']:+.3f}<br>±{e['sd']:.3f}" if e else "—")
        add(f"| `{key}` | " + " | ".join(cells) + " |")
    add(f"\n> {t2['raw_rho_pt_hole']['purity_note']}\n")
    add(f"> {t2['exchangeability_note']}\n")

    add("## Standing caveats\n")
    add("- **Fixation is perfectly collinear with section.** Every Carnoy's slide "
        "is 2M-1 and every PFA slide is 2M-2, so nothing here can attribute any "
        "difference — or any absence of one — to fixation chemistry as opposed to "
        "anatomical region. Bridge samples (serial sections from one block, split "
        "across both fixations, stained in one run) would be required.")
    add("- Eight slides per section. Every interval and every p-value inherits "
        "that, and the minimum detectable differences above are large.")
    add("- The mediator-versus-confounder question is not adjudicated. Both "
        "estimands are reported.")
    add("- The exchangeability of the exact null is not assumed: the "
        "within-slide-normalised variant is the version whose null is "
        "exchangeable with its observed value, and the purity strata show whether "
        "the pooled null is homogeneous.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs=2, default=["2M-1", "2M-2"])
    ap.add_argument("--per-duct-csvs", nargs=2, type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    secs = list(args.sections)
    print("=" * 78)
    print("  Holey-ness section comparison")
    print("=" * 78)

    frames, paths = {}, {}
    for i, sec in enumerate(secs):
        p = args.per_duct_csvs[i]
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. This analysis reads the per-duct table the "
                "holeyness validation wrote and must not recompute it.")
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"{p} is missing required columns {missing}.")
        df = df.copy()
        df["section"] = sec
        frames[sec] = df
        paths[sec] = str(p)
        print(f"  {sec}: {p}  ({len(df)} ducts, "
              f"{df['slide_name'].nunique()} slides)")

    pooled = pd.concat([frames[s] for s in secs], ignore_index=True)
    if pooled["slide_name"].nunique() != 2 * N_SLIDES_PER_SECTION:
        raise ValueError(
            f"Pooled data has {pooled['slide_name'].nunique()} distinct slides; the "
            f"exact enumeration requires {2 * N_SLIDES_PER_SECTION}.")
    c = Cols(pooled)
    idx = {s: np.flatnonzero(c.section == s) for s in secs}
    all_idx = np.arange(len(pooled))
    section_of_slide = dict(zip(pooled["slide_name"].astype(str),
                                pooled["section"].astype(str)))

    # verify the fast Spearman against scipy rather than trusting it
    errs = []
    for s in secs:
        i = idx[s]
        errs.append(abs(_fast_spearman(c.pt[i], c.hole[i])
                        - float(spearmanr(c.pt[i], c.hole[i]).statistic)))
    max_err = float(max(errs))
    print(f"\n  fast Spearman vs scipy: max abs difference {max_err:.3g}")
    if max_err > 1e-9:
        raise ValueError(
            f"_fast_spearman disagrees with scipy by {max_err:.3g}; the "
            "enumeration would not be computing the reported statistic.")

    print("\n  Task 1 ...")
    t1 = task1({s: c for s in secs}, idx, args.n_boot, args.n_perm, args.seed)
    for s in secs:
        e = t1["per_section"][s]
        print(f"    {s}: raw {e['raw_rho_pt_hole']['point_estimate']:+.4f}   "
              f"|area {e['partial_given_area']['point_estimate']:+.4f}   "
              f"|area+nd {e['partial_given_area_nd']['point_estimate']:+.4f}")
    print(f"    {t1['between_section_gap']['statement']}")

    print("\n  Task 2 — exact enumeration ...")
    t2 = task2(c, all_idx, section_of_slide, secs[0])
    for k in ("raw_rho_pt_hole", "partial_given_area", "rho_area_pseudotime",
              "raw_within_slide_normalised"):
        r = t2[k]
        print(f"    {k}: obs {r['observed_difference']:+.4f}, "
              f"exact p {r['exact_p_two_sided']:.4g}, "
              f"MDD {r['minimum_detectable_difference_alpha05']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "analysis": "holeyness_section_comparison",
        "sections": secs,
        "inputs": {
            "per_duct_tables": paths,
            "n_rows": {s: int(len(frames[s])) for s in secs},
            "n_slides": {s: int(frames[s]["slide_name"].nunique()) for s in secs},
            "partial_fn_primary": "holeyness._partial_spearman_multi",
            "partial_fn_check": "holeyness._partial_spearman",
            "fast_spearman_max_error": max_err,
            "recomputed_anything": False,
        },
        "config": {k: str(v) for k, v in vars(args).items()},
        "task_1": t1, "task_2": t2,
    }
    res["figures"] = write_figures(t1, t2, secs, args.output_dir)
    out = args.output_dir / "holeyness_section_comparison.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    write_report(res, args.output_dir / "holeyness_section_comparison.md")
    print(f"\n  JSON:     {out}")
    print(f"  Markdown: {args.output_dir / 'holeyness_section_comparison.md'}")


if __name__ == "__main__":
    main()
