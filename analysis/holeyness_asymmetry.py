"""Why does the holey-ness validation succeed in 2M-1 and fail in 2M-2?

WHAT THIS DIAGNOSTIC IS FOR
---------------------------
The expert ductal holey-ness validation has been run on both sections against the
per_section_v2 pseudotime, and the outcome is strongly asymmetric: 2M-1
(Carnoy's-fixed) gives a positive, slide-consistent, permutation-significant
rho(pseudotime, hole_pct), while 2M-2 (PFA-fixed) is effectively null. Duct
retention is comparable between sections, so differential exclusion is unlikely
to be the whole story. The cause is UNKNOWN. This diagnostic narrows it by
discriminating between two candidate explanations:

  (A) MECHANICAL. 2M-2 has less usable spread in hole_pct across ducts, so there
      is little signal available to correlate with anything.

  (B) THE GROUND TRUTH BEHAVES DIFFERENTLY. The collaborating pathologist's
      account is that duct diameter increases with progression and hole count
      increases with diameter. If that diameter-to-holes relationship holds under
      Carnoy's but breaks under PFA, the annotation is measuring something
      different in the two fixations, and the null is a property of the ground
      truth rather than of the pseudotime axis.

  (C) THE 2M-2 PSEUDOTIME AXIS IS ITSELF DEGENERATE. Added as a third candidate
      because evidence for it already exists and it would produce the same null
      without either (A) or (B) being true. Reported with its prior evidence and
      the duct-level pseudotime spread computable here; NOT recomputed, since
      that would need the h5ad and is outside this diagnostic's scope.

WHAT THIS DIAGNOSTIC CANNOT ESTABLISH
-------------------------------------
It cannot establish that FIXATION causes anything. Fixation is perfectly
collinear with section in this cohort — every Carnoy's slide is 2M-1 and every
PFA slide is 2M-2 — so fixation and anatomical region are not separable by any
analysis of this data. Bridge samples (serial sections from one block, both
fixations, stained in one run) would be required. Every statement below about
"the two fixations" is shorthand for "the two sections, which differ in fixation
among other things".

A NOTE ON WHAT CHECK 1 CAN AND CANNOT SHOW
------------------------------------------
Spearman correlation is invariant to any monotone rescaling of either variable.
So a difference in hole_pct's raw SD, IQR or CV between sections **cannot by
itself** produce a rank-correlation null — rescaling hole_pct by a factor of ten
would change every one of those statistics and leave rho untouched.

The version of explanation (A) that CAN produce a Spearman null is RANK
COMPRESSION: ties, coarse annotation granularity, or a large mass at one value,
all of which genuinely shrink the attainable |rho|. Check 1 therefore reports the
requested location/spread statistics AND a tie-and-granularity block, and the
verdict is driven by the rank-relevant measure. Reporting only the former would
let this diagnostic answer "spreads are comparable" while missing a real
mechanical cause, or answer "variance is compressed" and imply a consequence that
does not follow.

The rank-relevant statistic used is ``rank_sd_ratio``: the standard deviation of
the tie-corrected mid-ranks divided by the standard deviation of untied ranks
1..n. It is 1.0 with no ties and falls toward 0 as ties dominate, and it is
exactly the factor by which a tie structure attenuates Spearman.

READ-ONLY. Reads existing per-duct tables only. Recomputes no duct table, reruns
no pipeline stage, modifies no module, writes only to --output-dir.
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
from scipy.stats import levene, mannwhitneyu, rankdata, spearmanr

# The values quoted in the diagnostic request, carried so the report can
# reconcile them against what the tables actually contain rather than assuming
# they agree. Overridable from the CLI.
QUOTED = {"2M-1": 0.276, "2M-2": 0.020}

N_PERM_DEFAULT = 5000
MIN_DUCTS_FOR_SLIDE_RHO = 20
# Check 1 verdict thresholds, fixed here rather than chosen after seeing numbers.
RANK_COMPRESSION_RATIO = 0.90     # rank_sd_ratio below this share of the other
ABS_SPREAD_RATIO = 0.70           # IQR / CV below this share counts as compressed
ATTENUATION_TOLERANCE = 1.10      # slack when comparing tie ceiling to observed drop
# Check 2 verdict thresholds.
SIZE_HOLE_RHO_MIN = 0.15          # pooled |rho| for the relationship to "hold"
SIZE_HOLE_SLIDES_MIN = 6          # of 8 slides that must share the sign
SANITY_AREA_HOLEAREA_MIN = 0.50   # rho(area, hole_area) expected by construction

REQUIRED_COLUMNS = [
    "object_id", "n_patches", "pseudotime", "nuclear_density",
    "packing_irregularity", "slide_name", "hole_pct", "hole_area_um2",
    "area_um2", "centroid_x_um", "centroid_y_um",
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


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _rho(x, y, min_n: int = 10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return None
    return float(spearmanr(x[ok], y[ok]).statistic)


def describe(values, label: str) -> dict:
    """Location and spread, as requested, plus the rank-relevant statistics."""
    v = _finite(values)
    if v.size == 0:
        return {"label": label, "n": 0}
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    q = {p: float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)}
    iqr = q[75] - q[25]

    # ── rank-relevant block ──────────────────────────────────────────────
    # Spearman ignores monotone rescaling, so SD/IQR/CV above cannot by
    # themselves attenuate it. Ties can, and by exactly this factor.
    mid = rankdata(v)                       # average ranks -> tie-corrected
    untied_sd = float(np.std(np.arange(1, v.size + 1), ddof=1)) if v.size > 1 else 0.0
    rank_sd_ratio = (float(np.std(mid, ddof=1)) / untied_sd) if untied_sd > 0 else None
    vals, counts = np.unique(v, return_counts=True)
    # Fraction of PAIRS carrying no ordering information. More legible than
    # rank_sd_ratio and more sensitive to a large mass at one value.
    n_pairs = v.size * (v.size - 1) / 2.0
    tied_pairs = float((counts * (counts - 1) / 2.0).sum())
    return {
        "label": label,
        "n": int(v.size),
        "min": float(v.min()), "p5": q[5], "p25": q[25], "median": q[50],
        "p75": q[75], "p95": q[95], "max": float(v.max()),
        "mean": mean, "sd": sd, "iqr": iqr,
        "cv": float(sd / mean) if mean != 0 else None,
        "granularity": {
            "n_distinct_values": int(vals.size),
            "distinct_fraction": float(vals.size / v.size),
            "frac_exactly_zero": float((v == 0).mean()),
            "largest_tie_group_n": int(counts.max()),
            "largest_tie_group_frac": float(counts.max() / v.size),
            "modal_value": float(vals[int(counts.argmax())]),
            "rank_sd_ratio": rank_sd_ratio,
            "tied_pair_fraction": (float(tied_pairs / n_pairs) if n_pairs > 0 else None),
            "note": ("rank_sd_ratio is the factor by which this tie structure "
                     "attenuates Spearman: 1.0 with no ties, falling toward 0 as "
                     "ties dominate. This, not sd/iqr/cv, is what a rank "
                     "correlation is sensitive to. Note it moves SLOWLY: "
                     "quantising to three equal levels only takes it to ~0.94, and "
                     "even a binary split only reaches ~0.87, so ties have to be "
                     "near-total before they can explain a large drop in rho."),
        },
    }


def within_slide_permutation(df: pd.DataFrame, a: str, b: str, n_perm: int,
                             seed: int) -> dict:
    """Permutation p for rho(a, b), shuffling `b` WITHIN each slide.

    Ducts are nested within slides. A global shuffle would break both the
    duct-to-duct pairing and the slide structure, and would overstate
    significance by treating between-slide differences as exchangeable.
    """
    x = df[a].values.astype(float)
    y = df[b].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slides = df["slide_name"].astype(str).values[ok]
    obs = float(spearmanr(x, y).statistic)

    rng = np.random.default_rng(seed)
    idx_by_slide = [np.flatnonzero(slides == s) for s in np.unique(slides)]
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yy = y.copy()
        for idx in idx_by_slide:
            yy[idx] = rng.permutation(yy[idx])
        null[i] = spearmanr(x, yy).statistic
    null = null[np.isfinite(null)]
    return {
        "observed_rho": obs,
        "n_perm": int(null.size),
        "null_median": float(np.median(null)) if null.size else None,
        "null_p95_abs": float(np.percentile(np.abs(null), 95)) if null.size else None,
        "p_value_two_sided": (float((np.abs(null) >= abs(obs)).mean())
                              if null.size else None),
        "shuffle": "within slide",
    }


def per_slide_rho(df: pd.DataFrame, a: str, b: str) -> dict:
    """rho(a, b) inside each slide, reported DESCRIPTIVELY.

    With 8 slides no p-value on the across-slide summary would be honest, so
    none is produced; mean/median/range/count-positive only.
    """
    out = {}
    for s in sorted(df["slide_name"].astype(str).unique()):
        g = df[df["slide_name"].astype(str) == s]
        out[s] = {
            "n_ducts": int(len(g)),
            "rho": (_rho(g[a].values, g[b].values)
                    if len(g) >= MIN_DUCTS_FOR_SLIDE_RHO else None),
        }
    vals = np.array([v["rho"] for v in out.values() if v["rho"] is not None],
                    dtype=float)
    return {
        "per_slide": out,
        "n_slides_reported": int(vals.size),
        "n_slides_total": int(df["slide_name"].nunique()),
        "mean": float(vals.mean()) if vals.size else None,
        "median": float(np.median(vals)) if vals.size else None,
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
        "n_positive": int((vals > 0).sum()),
        "power_note": ("Descriptive only. Eight slides cannot support an inferential "
                       "statement about the across-slide distribution, so no p-value "
                       "is computed for it."),
    }


# ── loading ──────────────────────────────────────────────────────────────────

def load_per_duct(path: Path, section: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This diagnostic reads the per-duct table produced "
            "by the holeyness validation run and must not recompute it — a "
            "recomputed table could differ from the one whose result is being "
            "explained.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns {missing}.")
    df = df.copy()
    df["section"] = section
    return df


def load_excluded(v2_dir: Path, retained: pd.DataFrame, section: str):
    """Ducts with ZERO assigned patches, from v2's full duct table.

    v1's holeyness_per_duct.csv contains only ducts with >=1 assigned patch, so
    the excluded population is not recoverable from it. Returns (frame, note);
    the frame is None when v2 has not been run for this section, and the report
    says so rather than approximating.
    """
    path = v2_dir / "duct_table_full.csv"
    if not path.exists():
        return None, (f"{path} not found — v2 has not been run for {section}, so "
                      "the excluded-duct population cannot be described. v1's "
                      "per-duct CSV contains retained ducts only.")
    full = pd.read_csv(path)
    if "object_id" not in full.columns or "hole_pct" not in full.columns:
        return None, f"{path} lacks object_id/hole_pct; cannot identify excluded ducts."
    keep = set(retained["object_id"].astype(str))
    excl = full[~full["object_id"].astype(str).isin(keep)].copy()
    excl["section"] = section
    return excl, (f"{len(excl)} excluded ducts recovered from {path.name} "
                  f"({len(full)} total minus {len(keep)} retained).")


# ── CHECK 1 ──────────────────────────────────────────────────────────────────

def check1(frames: dict, excluded: dict, excl_notes: dict,
           rho_pt_hole: dict | None = None) -> dict:
    secs = list(frames)
    a, b = secs[0], secs[1]

    retained_stats, excluded_stats = {}, {}
    for s in secs:
        retained_stats[s] = {
            v: describe(frames[s][v].values, f"{s} retained {v}")
            for v in ("hole_pct", "hole_area_um2", "area_um2")
        }
        if excluded[s] is not None:
            excluded_stats[s] = {
                v: describe(excluded[s][v].values, f"{s} excluded {v}")
                for v in ("hole_pct", "hole_area_um2", "area_um2")
                if v in excluded[s].columns
            }
        else:
            excluded_stats[s] = {"unavailable": excl_notes[s]}

    ha = _finite(frames[a]["hole_pct"].values)
    hb = _finite(frames[b]["hole_pct"].values)

    u = mannwhitneyu(ha, hb, alternative="two-sided")
    # Rank-biserial: +1 when section `a` values dominate `b`, -1 when reversed.
    rb = float(2.0 * u.statistic / (ha.size * hb.size) - 1.0)

    # Brown-Forsythe (Levene centred on the MEDIAN) is primary: hole_pct is
    # right-skewed and mean-centred Levene is not robust to that. Mean-centred
    # Levene is reported alongside so the choice is visible, not hidden.
    bf = levene(ha, hb, center="median")
    lv = levene(ha, hb, center="mean")

    ra, rb_stats = retained_stats[a]["hole_pct"], retained_stats[b]["hole_pct"]
    ratios = {
        "iqr_ratio_b_over_a": (rb_stats["iqr"] / ra["iqr"]) if ra["iqr"] else None,
        "sd_ratio_b_over_a": (rb_stats["sd"] / ra["sd"]) if ra["sd"] else None,
        "cv_ratio_b_over_a": ((rb_stats["cv"] / ra["cv"])
                              if ra["cv"] not in (None, 0) and rb_stats["cv"] is not None
                              else None),
        "rank_sd_ratio_b_over_a": (
            rb_stats["granularity"]["rank_sd_ratio"]
            / ra["granularity"]["rank_sd_ratio"]
            if ra["granularity"]["rank_sd_ratio"] else None),
    }

    rank_compressed = (ratios["rank_sd_ratio_b_over_a"] is not None
                       and ratios["rank_sd_ratio_b_over_a"] < RANK_COMPRESSION_RATIO)
    abs_compressed = (ratios["iqr_ratio_b_over_a"] is not None
                      and ratios["iqr_ratio_b_over_a"] < ABS_SPREAD_RATIO)
    rel_compressed = (ratios["cv_ratio_b_over_a"] is not None
                      and ratios["cv_ratio_b_over_a"] < ABS_SPREAD_RATIO)

    # ── the quantitative form of the question ────────────────────────────
    # A threshold on rank_sd_ratio only says "ties differ". The question that
    # matters is whether the tie structure can account for the SIZE of the drop
    # in rho that needs explaining. Ties attenuate rho by at most the ratio of
    # the two rank_sd_ratios, so if that ceiling is far above the observed
    # ratio of rhos, ties cannot be the explanation however they test.
    budget = None
    if rho_pt_hole is not None:
        ra_rho, rb_rho = rho_pt_hole.get(a), rho_pt_hole.get(b)
        if ra_rho not in (None, 0) and rb_rho is not None:
            observed_ratio = float(rb_rho / ra_rho)
            ceiling = ratios["rank_sd_ratio_b_over_a"]
            budget = {
                "rho_pt_hole_a": ra_rho, "rho_pt_hole_b": rb_rho,
                "observed_rho_ratio_b_over_a": observed_ratio,
                "max_attenuation_from_ties": ceiling,
                # 10% tolerance: when ties ARE the mechanism the two quantities
                # are equal up to sampling noise, so an exact inequality would
                # flip on noise alone.
                "ties_can_account_for_the_drop": (
                    bool(ceiling is not None
                         and ceiling <= observed_ratio * ATTENUATION_TOLERANCE)),
                "tolerance": ATTENUATION_TOLERANCE,
                "shortfall": (float(ceiling - observed_ratio)
                              if ceiling is not None else None),
                "note": ("Ties can shrink rho by at most the ratio of the two "
                         "rank_sd_ratios. If that ceiling sits well ABOVE the "
                         "observed ratio of rhos, the tie structure cannot "
                         "account for the drop no matter how the threshold test "
                         "comes out."),
            }

    if budget is not None and not budget["ties_can_account_for_the_drop"]:
        verdict = (
            f"NOT SUPPORTED, quantitatively. {b}'s rho is "
            f"{budget['observed_rho_ratio_b_over_a']:.3f} of {a}'s, but its tie "
            f"structure can attenuate rho by at most a factor of "
            f"{budget['max_attenuation_from_ties']:.3f} — a shortfall of "
            f"{budget['shortfall']:.3f}. Rank compression cannot account for a drop "
            "of this size. (For scale: quantising a variable to three equal levels "
            "only moves rank_sd_ratio to ~0.94, and a binary split only to ~0.87.) "
            "Explanation (A) is ruled out as the mechanism, whatever the raw "
            "spread statistics show.")
    elif rank_compressed:
        verdict = (
            f"SUPPORTED. {b}'s hole_pct is rank-compressed relative to {a} "
            f"(rank_sd_ratio {ratios['rank_sd_ratio_b_over_a']:.3f} of {a}'s), which "
            "genuinely attenuates a Spearman correlation"
            + (", and that attenuation is large enough to account for the observed "
               "drop in rho" if budget is not None else "")
            + ". Explanation (A) is supported and the null is at least partly "
              "mechanical.")
    elif abs_compressed or rel_compressed:
        which = "absolute (IQR)" if abs_compressed else "relative (CV)"
        verdict = (
            f"NOT SUPPORTED as an explanation of the rank correlation, though "
            f"{b}'s {which} spread IS smaller. Spearman is invariant to monotone "
            "rescaling, so a difference in raw spread cannot by itself produce a "
            f"rank-correlation null; the rank-relevant measure is comparable "
            f"(rank_sd_ratio {ratios['rank_sd_ratio_b_over_a']:.3f} of {a}'s). "
            "Report the spread difference as a description, not as the cause.")
    else:
        verdict = (
            f"NOT SUPPORTED. {b}'s hole_pct is not materially compressed relative "
            f"to {a} on any measure — absolute (IQR ratio "
            f"{ratios['iqr_ratio_b_over_a']:.2f}), relative (CV ratio "
            f"{ratios['cv_ratio_b_over_a']:.2f}) or rank-relevant (rank_sd_ratio "
            f"ratio {ratios['rank_sd_ratio_b_over_a']:.3f}). Explanation (A) is "
            "ruled out: there is signal available to correlate with.")

    return {
        "sections": [a, b],
        "retained": retained_stats,
        "excluded": excluded_stats,
        "excluded_notes": excl_notes,
        "mann_whitney_hole_pct": {
            "U": float(u.statistic), "p_value": float(u.pvalue),
            "rank_biserial": rb,
            "convention": (f"positive means {a} hole_pct tends to exceed {b}; "
                           "r = 2U/(n1*n2) - 1"),
            "n_a": int(ha.size), "n_b": int(hb.size),
        },
        "variance_tests_hole_pct": {
            "brown_forsythe_median_centred": {"W": float(bf.statistic),
                                              "p_value": float(bf.pvalue),
                                              "role": "PRIMARY"},
            "levene_mean_centred": {"W": float(lv.statistic),
                                    "p_value": float(lv.pvalue),
                                    "role": "reported for comparison"},
            "why": ("hole_pct is right-skewed, so the median-centred "
                    "(Brown-Forsythe) form is the robust one and is designated "
                    "primary in advance."),
        },
        "spread_ratios": ratios,
        "attenuation_budget": budget,
        "thresholds": {"rank_compression_ratio": RANK_COMPRESSION_RATIO,
                       "abs_spread_ratio": ABS_SPREAD_RATIO},
        "verdict": verdict,
        "supported": bool(
            rank_compressed
            and (budget is None or budget["ties_can_account_for_the_drop"])),
    }


# ── CHECK 2 ──────────────────────────────────────────────────────────────────

def check2(frames: dict, n_perm: int, seed: int) -> dict:
    out = {"sections": list(frames), "per_section": {}}
    for s, df in frames.items():
        size_hole = within_slide_permutation(df, "area_um2", "hole_pct", n_perm, seed)
        size_hole_slides = per_slide_rho(df, "area_um2", "hole_pct")
        sanity = _rho(df["area_um2"].values, df["hole_area_um2"].values)
        out["per_section"][s] = {
            "n_ducts": int(len(df)),
            "rho_area_hole_pct": size_hole,
            "rho_area_hole_pct_per_slide": size_hole_slides,
            "rho_area_hole_area_um2": sanity,
            "rho_area_hole_area_sanity_ok": (sanity is not None
                                             and sanity >= SANITY_AREA_HOLEAREA_MIN),
            "rho_area_pseudotime": _rho(df["area_um2"].values, df["pseudotime"].values),
            "rho_pseudotime_hole_pct": _rho(df["pseudotime"].values,
                                            df["hole_pct"].values),
            "rho_pseudotime_hole_pct_per_slide": per_slide_rho(
                df, "pseudotime", "hole_pct"),
        }

    holds = {}
    for s, r in out["per_section"].items():
        rho = r["rho_area_hole_pct"]["observed_rho"]
        npos = r["rho_area_hole_pct_per_slide"]["n_positive"]
        nrep = r["rho_area_hole_pct_per_slide"]["n_slides_reported"]
        holds[s] = bool(rho is not None and abs(rho) >= SIZE_HOLE_RHO_MIN
                        and rho > 0 and npos >= SIZE_HOLE_SLIDES_MIN)
        r["relationship_holds"] = holds[s]
        r["holds_basis"] = (f"pooled rho {rho:+.4f} (threshold +{SIZE_HOLE_RHO_MIN}), "
                            f"{npos}/{nrep} slides positive (threshold "
                            f"{SIZE_HOLE_SLIDES_MIN})")

    a, b = list(frames)
    bad_sanity = [s for s, r in out["per_section"].items()
                  if not r["rho_area_hole_area_sanity_ok"]]
    if bad_sanity:
        out["sanity_flag"] = (
            f"rho(area, hole_area_um2) is below {SANITY_AREA_HOLEAREA_MIN} in "
            f"{bad_sanity}. Hole area should scale with duct area by construction, "
            "so this indicates something anomalous in that section's annotation and "
            "must be resolved before the rest of this check is read.")
    else:
        out["sanity_flag"] = None

    if holds[a] and holds[b]:
        verdict = (f"HOLDS IN BOTH. The duct-size-to-holey-ness relationship the "
                   f"pathologist describes is present in {a} and in {b}. Explanation "
                   "(B) is NOT supported: the annotation is not behaving differently "
                   "between the two sections on this axis.")
    elif holds[a] and not holds[b]:
        verdict = (f"BREAKS IN {b}. The relationship holds in {a} and fails in {b}, "
                   "which supports explanation (B): the annotation behaves "
                   "differently between sections, and the null reflects the ground "
                   "truth rather than the pseudotime axis.")
    elif holds[b] and not holds[a]:
        verdict = (f"BREAKS IN {a} — the reverse of the expected pattern. The "
                   "relationship holds in the section whose validation FAILED and "
                   "fails in the one that succeeded, which no version of "
                   "explanation (B) predicts and needs explaining on its own.")
    else:
        verdict = ("BREAKS IN BOTH. The size-to-holey-ness relationship is not "
                   "present in either section, so it cannot explain a difference "
                   "between them — and it also undercuts the stated biology.")

    out["relationship_holds"] = holds
    out["thresholds"] = {"pooled_rho_min": SIZE_HOLE_RHO_MIN,
                         "slides_positive_min": SIZE_HOLE_SLIDES_MIN,
                         "sanity_area_holearea_min": SANITY_AREA_HOLEAREA_MIN}
    out["verdict"] = verdict
    return out


# ── CHECK 3 ──────────────────────────────────────────────────────────────────

def check3(c1: dict, c2: dict, frames: dict, quoted: dict) -> dict:
    a, b = c1["sections"]
    compressed = c1["supported"]
    broken = not (c2["relationship_holds"][a] and c2["relationship_holds"][b])
    broken_in_b_only = (c2["relationship_holds"][a]
                        and not c2["relationship_holds"][b])

    if compressed and not broken:
        outcome, text = "(i)", (
            f"PRIMARILY MECHANICAL. {b}'s hole_pct is rank-compressed while the "
            "size-to-holey-ness relationship is intact in both sections. The null "
            "is a range restriction: there is little rank signal in the annotation "
            "to correlate with anything.")
    elif not compressed and broken_in_b_only:
        outcome, text = "(ii)", (
            f"GROUND TRUTH DIFFERS. {b}'s hole_pct spread is comparable to {a}'s, "
            "but the size-to-holey-ness relationship breaks there. The annotation "
            "is measuring something different in the two sections, and the null "
            "reflects the ground truth rather than the pseudotime axis.")
    elif compressed and broken_in_b_only:
        outcome, text = "(iii)", (
            "BOTH CONTRIBUTE. hole_pct is rank-compressed in "
            f"{b} AND the size-to-holey-ness relationship breaks there. Neither "
            "alone accounts for the null and both should be reported.")
    else:
        outcome, text = "(iv)", (
            "NEITHER CANDIDATE EXPLAINS IT. "
            + ("hole_pct is not rank-compressed in " + b
               if not compressed else "hole_pct IS rank-compressed in " + b)
            + " and the size-to-holey-ness relationship "
            + ("holds in both sections" if not broken else "does not hold as expected")
            + ". The asymmetry is not accounted for by explanation (A) or (B) and "
              "remains open. See the candidate (C) block below for the one "
              "alternative with existing evidence.")

    recon = {}
    for s, df in frames.items():
        rho = _rho(df["pseudotime"].values, df["hole_pct"].values)
        q = quoted.get(s)
        recon[s] = {
            "recomputed_rho_pt_hole_pct": rho,
            "quoted_in_request": q,
            "abs_difference": (abs(rho - q) if (rho is not None and q is not None)
                               else None),
            "agrees_within_0_01": (bool(abs(rho - q) < 0.01)
                                   if (rho is not None and q is not None) else None),
        }

    return {
        "outcome": outcome,
        "statement": text,
        "reconciliation_with_quoted_values": recon,
        "reconciliation_note": (
            "The request quoted rho(pseudotime, hole_pct) per section. Those values "
            "are reconciled against what the per-duct tables actually contain. A "
            "disagreement means the quoted figure came from a different quantity, a "
            "different axis or a different run, and the framing of the asymmetry "
            "should be revisited before any verdict here is acted on."),
        "hard_limitation": (
            "This analysis CANNOT establish that fixation causes anything. Fixation "
            "is perfectly collinear with section in this cohort — every Carnoy's "
            "slide is 2M-1 and every PFA slide is 2M-2 — so fixation and anatomical "
            "region are not separable. Establishing a fixation effect would require "
            "BRIDGE SAMPLES: serial sections from a single block, split across both "
            "fixations, processed and stained in one run."),
    }


def candidate_c(frames: dict) -> dict:
    """The third candidate: the 2M-2 pseudotime axis may itself be degenerate.

    Reported with prior evidence plus the duct-level pseudotime spread computable
    from these tables. Nothing here is recomputed from the h5ad — that is outside
    this diagnostic's read-only-CSV scope — so the prior evidence is cited as
    provenance, not re-derived.
    """
    spread = {}
    for s, df in frames.items():
        pt = _finite(df["pseudotime"].values)
        spread[s] = {
            "n": int(pt.size),
            "min": float(pt.min()), "max": float(pt.max()),
            "p5": float(np.percentile(pt, 5)), "p95": float(np.percentile(pt, 95)),
            "median": float(np.median(pt)), "sd": float(pt.std(ddof=1)),
            "iqr": float(np.percentile(pt, 75) - np.percentile(pt, 25)),
            "frac_of_unit_interval_covered_p5_p95": float(
                np.percentile(pt, 95) - np.percentile(pt, 5)),
        }
    return {
        "hypothesis": (
            "The 2M-2 per_section_v2 pseudotime axis is itself unstable, which "
            "would produce a null rho(pt, hole_pct) regardless of anything about "
            "the annotation."),
        "duct_level_pseudotime_spread": spread,
        "prior_evidence_not_recomputed_here": [
            "All 20 v2 roots in 2M-2 have nuclear_density exactly 0.0.",
            "None of those 20 roots lies inside any Tumor annotation (duct_id null).",
            "3 of 20 roots order the manifold backwards relative to the median of "
            "the other 19 (leave-one-out Spearman < 0).",
            "pseudotime_std is 27.70% of the axis range in 2M-2 versus 5.03% in "
            "2M-1; dropping the 3 discordant roots takes it to 3.40%.",
        ],
        "how_to_test": (
            "Recompute rho(pt, hole_pct) for 2M-2 against a repaired axis — v2 "
            "roots minus the discordant ones — and against the area-stratified "
            "anchor. Both require the stored h5ad and are outside this "
            "diagnostic's scope, which is deliberately limited to the per-duct "
            "tables."),
        "status": "DOCUMENTED ALTERNATIVE, NOT TESTED HERE",
    }


# ── figures ──────────────────────────────────────────────────────────────────

def write_figures(frames: dict, c2: dict, out_dir: Path) -> list:
    written = []
    secs = list(frames)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lo = min(_finite(frames[s]["hole_pct"].values).min() for s in secs)
    hi = max(np.percentile(_finite(frames[s]["hole_pct"].values), 99) for s in secs)
    bins = np.linspace(lo, hi, 60)
    for s in secs:
        axes[0].hist(_finite(frames[s]["hole_pct"].values), bins=bins, alpha=0.55,
                     label=f"{s} (n={len(frames[s])})", density=True)
    axes[0].set_xlabel("hole_pct"); axes[0].set_ylabel("density")
    axes[0].set_title("hole_pct distribution, retained ducts\n(shared axis, x clipped at p99)")
    axes[0].legend()
    axes[1].violinplot([_finite(frames[s]["hole_pct"].values) for s in secs],
                       showmedians=True)
    axes[1].set_xticks(range(1, len(secs) + 1)); axes[1].set_xticklabels(secs)
    axes[1].set_ylabel("hole_pct"); axes[1].set_title("hole_pct spread by section")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"hole_pct_distribution.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight"); written.append(str(p))
    plt.close(fig)

    fig, axes = plt.subplots(1, len(secs), figsize=(5.5 * len(secs), 4.6),
                             squeeze=False)
    for i, s in enumerate(secs):
        ax = axes[0][i]
        d = frames[s]
        ax.scatter(d["area_um2"], d["hole_pct"], s=6, alpha=0.35, edgecolors="none")
        ax.set_xscale("log")           # duct area is heavily right-skewed
        ax.set_xlabel("duct area (um^2, log scale)"); ax.set_ylabel("hole_pct")
        r = c2["per_section"][s]["rho_area_hole_pct"]
        p = r["p_value_two_sided"]
        ax.set_title(f"{s}   Spearman rho = {r['observed_rho']:+.3f}\n"
                     f"within-slide permutation p = "
                     + (f"{p:.4g}" if p is not None else "n/a")
                     + f"   n = {len(d)}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"area_vs_hole_pct.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight"); written.append(str(p))
    plt.close(fig)
    return written


# ── report ───────────────────────────────────────────────────────────────────

def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    a, b = res["check_1"]["sections"]

    add("# Holey-ness validation asymmetry — diagnostic\n")
    add("**What this is for.** The expert ductal holey-ness validation succeeds in "
        f"{a} and is effectively null in {b}, and duct retention is comparable, so "
        "differential exclusion is unlikely to be the whole explanation. This "
        "read-only diagnostic discriminates between two candidate causes: (A) that "
        f"{b} simply has less usable spread in `hole_pct`, and (B) that the "
        "duct-size-to-holey-ness relationship the pathologist describes holds under "
        "one fixation and breaks under the other, making the null a property of the "
        "ground truth rather than of the pseudotime axis. A third candidate (C) — "
        f"that the {b} pseudotime axis is itself degenerate — is documented with "
        "existing evidence but not tested here. **What it cannot establish:** that "
        "fixation causes anything. Fixation is perfectly collinear with section in "
        "this cohort, so fixation and anatomical region are not separable by any "
        "analysis of this data; bridge samples would be required. Nothing was "
        "recomputed and no existing results directory was written to.\n")

    add("## Step 0 — paths and provenance\n")
    for s, p in res["inputs"]["per_duct_tables"].items():
        add(f"- **{s}** retained ducts: `{p}`")
    for s, n in res["inputs"]["excluded_notes"].items():
        add(f"- **{s}** excluded ducts: {n}")
    add(f"\nRows read: " + ", ".join(f"{s} n={n}" for s, n
                                     in res["inputs"]["n_rows"].items()))
    add("\nNo duct table was recomputed. The per-duct tables are read exactly as the "
        "holey-ness validation wrote them.\n")

    add("### Reconciliation with the values quoted in the request\n")
    add("| section | quoted rho(pt, hole_pct) | recomputed | agrees |")
    add("|---|---|---|---|")
    for s, r in res["check_3"]["reconciliation_with_quoted_values"].items():
        rc = r["recomputed_rho_pt_hole_pct"]
        add(f"| {s} | {r['quoted_in_request']} | "
            + (f"{rc:+.4f}" if rc is not None else "—") + " | "
            + ("yes" if r["agrees_within_0_01"] else "**NO**") + " |")
    add(f"\n> {res['check_3']['reconciliation_note']}\n")

    c1 = res["check_1"]
    add("## Check 1 — hole_pct distribution comparison (tests explanation A)\n")
    add("### Retained ducts\n")
    add("| section | variable | n | min | p5 | p25 | median | p75 | p95 | max | mean | sd | iqr | cv |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in (a, b):
        for v in ("hole_pct", "hole_area_um2", "area_um2"):
            d = c1["retained"][s][v]
            add(f"| {s} | `{v}` | {d['n']} | {d['min']:.3f} | {d['p5']:.3f} | "
                f"{d['p25']:.3f} | {d['median']:.3f} | {d['p75']:.3f} | "
                f"{d['p95']:.3f} | {d['max']:.3f} | {d['mean']:.3f} | {d['sd']:.3f} | "
                f"{d['iqr']:.3f} | "
                + (f"{d['cv']:.3f}" if d["cv"] is not None else "—") + " |")

    add("\n### Rank-relevant statistics for `hole_pct`\n")
    add("Spearman is invariant to monotone rescaling, so the sd/iqr/cv above cannot "
        "by themselves attenuate a rank correlation. These can.\n")
    add("| section | distinct values | distinct frac | frac exactly 0 | largest tie group | rank_sd_ratio |")
    add("|---|---|---|---|---|---|")
    for s in (a, b):
        g = c1["retained"][s]["hole_pct"]["granularity"]
        add(f"| {s} | {g['n_distinct_values']} | {g['distinct_fraction']:.3f} | "
            f"{g['frac_exactly_zero']:.4f} | {g['largest_tie_group_frac']:.4f} | "
            + (f"**{g['rank_sd_ratio']:.4f}**" if g["rank_sd_ratio"] else "—") + " |")

    add("\n### Excluded ducts (zero assigned patches)\n")
    for s in (a, b):
        e = c1["excluded"][s]
        if "unavailable" in e:
            add(f"- **{s}**: {e['unavailable']}")
            continue
        for v, d in e.items():
            add(f"- **{s}** `{v}`: n={d['n']}, median {d['median']:.3f}, "
                f"iqr {d['iqr']:.3f}, mean {d['mean']:.3f}, sd {d['sd']:.3f}")

    mw = c1["mann_whitney_hole_pct"]
    vt = c1["variance_tests_hole_pct"]
    add(f"\n### Location and spread tests on `hole_pct`\n")
    add(f"- Mann-Whitney U = {mw['U']:.1f}, p = {mw['p_value']:.4g}, "
        f"rank-biserial = **{mw['rank_biserial']:+.4f}** ({mw['convention']})")
    add(f"- Brown-Forsythe (median-centred, PRIMARY): W = "
        f"{vt['brown_forsythe_median_centred']['W']:.4f}, "
        f"p = {vt['brown_forsythe_median_centred']['p_value']:.4g}")
    add(f"- Levene (mean-centred, for comparison): W = "
        f"{vt['levene_mean_centred']['W']:.4f}, "
        f"p = {vt['levene_mean_centred']['p_value']:.4g}")
    add(f"- {vt['why']}")
    ab = c1.get("attenuation_budget")
    if ab:
        add("\n### Can ties account for the drop?\n")
        add(f"- {b} rho is **{ab['observed_rho_ratio_b_over_a']:.3f}** of {a}'s "
            f"({ab['rho_pt_hole_b']:+.4f} vs {ab['rho_pt_hole_a']:+.4f})")
        add(f"- {b}'s tie structure can attenuate rho by at most a factor of "
            f"**{ab['max_attenuation_from_ties']:.3f}**")
        add(f"- ties can account for the drop: **{ab['ties_can_account_for_the_drop']}**"
            + (f" (shortfall {ab['shortfall']:+.3f})"
               if ab["shortfall"] is not None else ""))
        add(f"\n> {ab['note']}")
    r = c1["spread_ratios"]
    add(f"\nRatios ({b} / {a}): IQR "
        + (f"{r['iqr_ratio_b_over_a']:.3f}" if r['iqr_ratio_b_over_a'] else "—")
        + ", SD " + (f"{r['sd_ratio_b_over_a']:.3f}" if r['sd_ratio_b_over_a'] else "—")
        + ", CV " + (f"{r['cv_ratio_b_over_a']:.3f}" if r['cv_ratio_b_over_a'] else "—")
        + ", rank_sd_ratio "
        + (f"**{r['rank_sd_ratio_b_over_a']:.3f}**"
           if r['rank_sd_ratio_b_over_a'] else "—"))
    add(f"\n**Verdict:** {c1['verdict']}\n")

    c2 = res["check_2"]
    add("## Check 2 — duct area vs hole_pct, within each section (tests explanation B)\n")
    if c2["sanity_flag"]:
        add(f"> ⚠ **{c2['sanity_flag']}**\n")
    add("| section | n | rho(area, hole_pct) | within-slide perm p | slides positive | "
        "rho(area, hole_area) | rho(area, pseudotime) | rho(pt, hole_pct) |")
    add("|---|---|---|---|---|---|---|---|")
    for s in (a, b):
        r = c2["per_section"][s]
        sh, sl = r["rho_area_hole_pct"], r["rho_area_hole_pct_per_slide"]
        add(f"| {s} | {r['n_ducts']} | **{sh['observed_rho']:+.4f}** | "
            + (f"{sh['p_value_two_sided']:.4g}" if sh["p_value_two_sided"] is not None else "—")
            + f" | {sl['n_positive']}/{sl['n_slides_reported']} | "
            + (f"{r['rho_area_hole_area_um2']:+.4f}" if r["rho_area_hole_area_um2"] is not None else "—")
            + " | "
            + (f"{r['rho_area_pseudotime']:+.4f}" if r["rho_area_pseudotime"] is not None else "—")
            + " | "
            + (f"{r['rho_pseudotime_hole_pct']:+.4f}" if r["rho_pseudotime_hole_pct"] is not None else "—")
            + " |")

    add("\n### Per-slide `rho(area, hole_pct)`\n")
    for s in (a, b):
        sl = c2["per_section"][s]["rho_area_hole_pct_per_slide"]
        vals = ", ".join(
            f"{k}: " + (f"{v['rho']:+.3f}" if v["rho"] is not None else "n/a")
            for k, v in sl["per_slide"].items())
        add(f"- **{s}** — mean "
            + (f"{sl['mean']:+.4f}" if sl["mean"] is not None else "—")
            + ", median " + (f"{sl['median']:+.4f}" if sl["median"] is not None else "—")
            + ", range [" + (f"{sl['min']:+.3f}, {sl['max']:+.3f}"
                             if sl["min"] is not None else "—") + "], "
            + f"{sl['n_positive']}/{sl['n_slides_reported']} positive")
        add(f"  - {vals}")
    add(f"\n> {c2['per_section'][a]['rho_area_hole_pct_per_slide']['power_note']}")
    add(f"\n**Verdict:** {c2['verdict']}\n")

    c3 = res["check_3"]
    add("## Check 3 — joint verdict\n")
    add("Framework: (i) variance compressed AND relationship intact → mechanical; "
        "(ii) variance comparable AND relationship breaks → ground truth differs; "
        "(iii) both → both contribute; (iv) neither → open.\n")
    add(f"**Outcome {c3['outcome']}.** {c3['statement']}\n")
    add(f"> **Hard limitation.** {c3['hard_limitation']}\n")

    cc = res["candidate_c"]
    add("## Candidate (C) — is the 2M-2 pseudotime axis itself degenerate?\n")
    add(f"*{cc['status']}.* {cc['hypothesis']}\n")
    add("Existing evidence, cited as provenance and not re-derived here:\n")
    for e in cc["prior_evidence_not_recomputed_here"]:
        add(f"- {e}")
    add("\nDuct-level pseudotime spread in these tables:\n")
    add("| section | n | median | sd | iqr | p5 | p95 |")
    add("|---|---|---|---|---|---|---|")
    for s, d in cc["duct_level_pseudotime_spread"].items():
        add(f"| {s} | {d['n']} | {d['median']:.4f} | {d['sd']:.4f} | {d['iqr']:.4f} | "
            f"{d['p5']:.4f} | {d['p95']:.4f} |")
    add(f"\n> {cc['how_to_test']}\n")

    add("## Limitations\n")
    add("- Fixation is perfectly collinear with section; no causal reading is "
        "available from this cohort.")
    add("- Every across-slide summary rests on 8 slides and is reported "
        "descriptively, with no p-value on the across-slide distribution.")
    add("- The per-duct tables are read as written, including their 6-decimal "
        "float formatting, which sets the floor on the tie analysis.")
    add("- Check 1's location and spread statistics describe the annotation but "
        "cannot by themselves explain a rank-correlation null; only the "
        "rank-relevant block can.")
    add("- Candidate (C) is documented, not tested. Testing it needs the stored "
        "h5ad, which this diagnostic deliberately does not read.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs=2, default=["2M-1", "2M-2"])
    ap.add_argument("--per-duct-csvs", nargs=2, type=Path, required=True,
                    help="holeyness_per_duct.csv per section, SAME ORDER.")
    ap.add_argument("--v2-dirs", nargs=2, type=Path, default=None,
                    help="v2_area_adjusted dirs, for the excluded-duct population. "
                         "Omitted -> that half of Check 1 is reported as skipped.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quoted-rho", nargs=2, type=float, default=None,
                    help="rho(pt, hole_pct) as quoted per section, for reconciliation.")
    args = ap.parse_args()

    secs = list(args.sections)
    quoted = dict(zip(secs, args.quoted_rho)) if args.quoted_rho else \
        {s: QUOTED.get(s) for s in secs}

    print("=" * 78)
    print("  Holey-ness validation asymmetry diagnostic")
    print("=" * 78)
    frames, excluded, excl_notes, paths = {}, {}, {}, {}
    for i, s in enumerate(secs):
        p = args.per_duct_csvs[i]
        print(f"\n  {s} per-duct table: {p}")
        frames[s] = load_per_duct(p, s)
        paths[s] = str(p)
        print(f"    rows: {len(frames[s])}   slides: {frames[s]['slide_name'].nunique()}")
        if args.v2_dirs:
            excluded[s], excl_notes[s] = load_excluded(args.v2_dirs[i], frames[s], s)
        else:
            excluded[s], excl_notes[s] = None, (
                "--v2-dirs not supplied; the excluded-duct population is not "
                "described. v1's per-duct CSV contains retained ducts only.")
        print(f"    excluded ducts: {excl_notes[s]}")

    rho_pt_hole = {s: _rho(frames[s]["pseudotime"].values,
                           frames[s]["hole_pct"].values) for s in secs}
    print("\n  recomputed rho(pt, hole_pct): "
          + ", ".join(f"{s} " + (f"{v:+.4f}" if v is not None else "n/a")
                      for s, v in rho_pt_hole.items()))

    print("\n  Check 1 ...")
    c1 = check1(frames, excluded, excl_notes, rho_pt_hole)
    print(f"    {c1['verdict'][:110]}...")
    print("\n  Check 2 ...")
    c2 = check2(frames, args.n_perm, args.seed)
    print(f"    {c2['verdict'][:110]}...")
    c3 = check3(c1, c2, frames, quoted)
    print(f"\n  Check 3: outcome {c3['outcome']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figs = write_figures(frames, c2, args.output_dir)

    res = {
        "analysis": "holeyness_asymmetry_diagnostic",
        "inputs": {
            "per_duct_tables": paths,
            "excluded_notes": excl_notes,
            "n_rows": {s: int(len(frames[s])) for s in secs},
            "recomputed_anything": False,
        },
        "config": {k: str(v) for k, v in vars(args).items()},
        "check_1": c1, "check_2": c2, "check_3": c3,
        "candidate_c": candidate_c(frames),
        "figures": figs,
    }
    out = args.output_dir / "holeyness_asymmetry.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    write_report(res, args.output_dir / "holeyness_asymmetry.md")
    print(f"\n  JSON:     {out}")
    print(f"  Markdown: {args.output_dir / 'holeyness_asymmetry.md'}")


if __name__ == "__main__":
    main()
