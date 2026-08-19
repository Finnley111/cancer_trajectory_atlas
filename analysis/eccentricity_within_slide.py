"""Is the late-pseudotime structure biology, or is it one slide?

WHY THIS EXISTS
---------------
``eccentricity_check`` checked Task A's directionality WITHIN slides but ran
Task B — the bidirectional-enrichment test and the late subclustering — on the
GLOBAL top decile only. On the holeyroot axis that decile is 55% one slide in
2M-1 and 43% in 2M-2, against 12.5% under a uniform split across 8 slides. The
module's own report calls this a "SEPARATE AND OVERRIDING CONCERN" and then
issues Task B verdicts anyway.

That matters most for 2M-1, where Task B's two verdicts CONTRADICT each other:

    Verdict 2  0 of 6 features bidirectional, 5 unidirectional  -> trajectory-like
    Verdict 3  2 of 6 features have late subclusters on OPPOSITE sides
               (texture_entropy, h_intensity)                   -> eccentricity-like

and Verdict 4 reports "TRAJECTORY FRAMING SURVIVES both tests" by counting Tasks
A and B and silently dropping the subclustering test that 2M-1 fails. If the two
late subclusters are mostly two DIFFERENT SLIDES, Verdict 3 is a batch split
rather than two late phenotypes, and the contradiction dissolves. If they are
mixed across slides, Verdict 3 stands and Verdict 4 is overstated.

Note also that `h_intensity` is simultaneously one of 2M-1's four "directional
and within-slide-surviving" features AND one of the two features whose late
subclusters oppose (+1.26 vs -0.86). The same feature carries both verdicts.

WHAT THIS MODULE DOES
---------------------
  TASK 1  LATE-TAIL COMPOSITION, and the decisive cheap test. Which slide
          dominates the global late decile and by what fold over its own cohort
          share; then the late SUBCLUSTER x SLIDE contingency table with Cramer's
          V. ``eccentricity_check.run_late_subclustering`` already computes
          ``slide_breakdown`` per cluster — it is simply never surfaced. A high
          V means the late subclusters ARE slides, and Verdict 3 is a batch
          split.

  TASK 2  WITHIN-SLIDE BIDIRECTIONAL ENRICHMENT. Re-runs the unmodified
          ``run_bidirectional_enrichment`` on each slide separately, so both the
          pseudotime tail and the feature tails are defined inside one slide and
          no cross-slide stain or density offset can create or destroy an
          enrichment. Reported per feature as "in how many slides is this
          bidirectional / unidirectional", against the cohort answer.

  TASK 3  WITHIN-SLIDE LATE SUBCLUSTERING. Same, with
          ``run_late_subclustering``. A feature whose subclusters oppose inside
          MOST slides is showing a real split in the late state. One that opposes
          only cohort-wide is showing a slide contrast.

WHAT SUBSETTING CHANGES, STATED PLAINLY
---------------------------------------
Both reused functions z-score features and take quantiles over whatever AnnData
they are handed, so on a per-slide view every threshold becomes within-slide.
That is the correct normalisation for this question — it is what removes the
slide offset — but it means the within-slide numbers are NOT on the same scale
as the cohort ones and must not be differenced against them. They are compared
by PATTERN (bidirectional vs unidirectional; opposing vs not), never by value.

``k_range`` is lowered for the within-slide subclustering: a slide's late decile
is ~100-140 patches, and the cohort default of k up to 8 would fit 8 clusters to
~120 points. The default here is 2-4, and the realised silhouette scores are
reported so a thin fit is visible.

READ-ONLY on every run tree. Writes only --output-dir. ``eccentricity_check`` is
imported and never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from .root_sensitivity import MORPH_FEATURES, _json_default, load_run
from .eccentricity_check import (
    run_bidirectional_enrichment,
    run_late_subclustering,
    EXTREME_DECILE,
)

K_RANGE_WITHIN_DEFAULT = (2, 4)
K_RANGE_COHORT_DEFAULT = (2, 8)
MIN_PATCHES_PER_SLIDE = 200      # below this a slide is dropped entirely
MIN_LATE_PER_SLIDE = 60          # below this its late tail cannot be subclustered
FOLD_CONCERN = 3.0               # late-tail single-slide over-representation


def _slide_labels(obs) -> np.ndarray:
    """The slide each patch belongs to.

    ``slide_id`` is what eccentricity_check's own subclustering uses for its
    breakdown, so this module uses the same column and the two are directly
    comparable. It is stored as a string of the index into the run's slide list.
    """
    for col in ("slide_id", "slide_name"):
        if col in obs.columns:
            return obs[col].astype(str).values
    raise KeyError(
        "adata.obs has neither 'slide_id' nor 'slide_name', so patches cannot be "
        "grouped by slide. Refusing to run a within-slide analysis without slides."
    )


def _cramers_v(table: np.ndarray) -> dict:
    """Cramer's V for a subcluster x slide contingency table.

    V is reported with its chi-square p-value, but the p-value is close to
    meaningless here — n is ~800-1000 patches and almost any structure reaches
    significance — so the EFFECT SIZE is what to read. V near 0 means the late
    subclusters cut across slides (a phenotype split); V near 1 means they ARE
    slides (a batch split).
    """
    table = np.asarray(table, dtype=float)
    keep_r = table.sum(axis=1) > 0
    keep_c = table.sum(axis=0) > 0
    table = table[np.ix_(keep_r, keep_c)]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"cramers_v": None,
                "reason": "fewer than 2 non-empty subclusters or slides"}
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {
        "cramers_v": v,
        "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
        "n": int(n), "shape": list(table.shape),
        "interpretation": (
            "V near 1 means the late subclusters ARE slides — the opposing-feature "
            "verdict is then a batch split, not two late phenotypes. V near 0 means "
            "they cut across slides and the split is phenotypic. The p-value is not "
            "informative at this n; read V."
        ),
    }


# ── TASK 1: what the global late tail is made of ─────────────────────────────

def task1_late_tail_composition(adata, tail: float, k_range: tuple) -> dict:
    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)
    slides = _slide_labels(obs)

    late = pt >= np.quantile(pt, 1 - tail)
    early = pt <= np.quantile(pt, tail)

    def _shares(mask):
        uniq, cnt = np.unique(slides[mask], return_counts=True)
        order = np.argsort(-cnt)
        return {str(uniq[i]): int(cnt[i]) for i in order}

    cohort_u, cohort_c = np.unique(slides, return_counts=True)
    cohort_share = {str(u): float(c / len(slides)) for u, c in zip(cohort_u, cohort_c)}

    late_counts = _shares(late)
    n_late = int(late.sum())
    fold = {s: (c / n_late) / cohort_share[s] for s, c in late_counts.items()}
    top_slide = max(fold, key=fold.get)

    print("\n=== TASK 1 — late-tail composition ===")
    print(f"  late patches: {n_late}   slides: {len(cohort_u)}")
    print(f"  most over-represented slide: {top_slide}  "
          f"{100*late_counts[top_slide]/n_late:.1f}% of the late tail vs "
          f"{100*cohort_share[top_slide]:.1f}% of the cohort "
          f"({fold[top_slide]:.2f}x)")

    # The decisive test: are the late SUBCLUSTERS slides?
    sub = run_late_subclustering(adata, tail, k_range)
    contingency = None
    if sub.get("attempted"):
        cluster_ids = sorted(sub["clusters"])
        slide_ids = sorted({s for c in cluster_ids
                            for s in sub["clusters"][c].get("slide_breakdown", {})})
        table = np.array([
            [sub["clusters"][c].get("slide_breakdown", {}).get(s, 0) for s in slide_ids]
            for c in cluster_ids
        ], dtype=float)
        contingency = {
            "cluster_ids": cluster_ids,
            "slide_ids": slide_ids,
            "counts": table.tolist(),
            "per_cluster_max_slide_share": {
                c: sub["clusters"][c].get("max_share_from_one_slide")
                for c in cluster_ids
            },
            **_cramers_v(table),
        }
        v = contingency.get("cramers_v")
        print(f"  late subcluster x slide: Cramer's V = "
              + (f"{v:.3f}" if v is not None else "n/a"))
        for c in cluster_ids:
            ms = sub["clusters"][c].get("max_share_from_one_slide")
            print(f"    cluster {c}: n={sub['clusters'][c]['n']}, "
                  f"largest single slide = "
                  + (f"{100*ms:.1f}%" if ms is not None else "n/a"))

    return {
        "n_late": n_late, "n_early": int(early.sum()),
        "cohort_share_by_slide": cohort_share,
        "late_counts_by_slide": late_counts,
        "early_counts_by_slide": _shares(early),
        "late_fold_enrichment_by_slide": fold,
        "most_over_represented_slide": top_slide,
        "most_over_represented_fold": float(fold[top_slide]),
        "exceeds_fold_concern": bool(fold[top_slide] >= FOLD_CONCERN),
        "fold_concern_threshold": FOLD_CONCERN,
        "cohort_subclustering": sub,
        "subcluster_by_slide": contingency,
    }


# ── TASK 2 / 3: the same tests, inside each slide ────────────────────────────

def _slide_views(adata, min_patches: int):
    """(slide, sub-AnnData) for every slide with enough patches, plus skips."""
    slides = _slide_labels(adata.obs)
    views, skipped = [], {}
    for s in sorted(set(slides)):
        mask = slides == s
        n = int(mask.sum())
        if n < min_patches:
            skipped[str(s)] = n
            continue
        views.append((str(s), adata[mask].copy()))
    return views, skipped


def task2_within_slide_enrichment(views, tail: float) -> dict:
    print("\n=== TASK 2 — bidirectional enrichment, within each slide ===")
    per_slide = {}
    for s, sub in views:
        r = run_bidirectional_enrichment(sub, tail)
        per_slide[s] = r
        print(f"  slide {s}: n_late={r['n_late']}  "
              f"bidirectional {r['n_features_bidirectional']}/"
              f"{r['n_features_tested']}  "
              f"unidirectional {r['n_features_unidirectional']}")

    summary = {}
    for feat in MORPH_FEATURES:
        bi = sum(1 for r in per_slide.values()
                 if r["late"].get(feat) and r["late"][feat]["bidirectional"])
        uni = sum(1 for r in per_slide.values()
                  if r["late"].get(feat) and r["late"][feat]["unidirectional"])
        tested = sum(1 for r in per_slide.values() if r["late"].get(feat))
        summary[feat] = {
            "n_slides_bidirectional": bi,
            "n_slides_unidirectional": uni,
            "n_slides_tested": tested,
            "n_slides_neither": tested - bi - uni,
        }
    return {"per_slide": per_slide, "per_feature_across_slides": summary,
            "n_slides": len(per_slide)}


def task3_within_slide_subclustering(views, tail: float, k_range: tuple,
                                     min_late: int = MIN_LATE_PER_SLIDE) -> dict:
    """Within-slide late subclustering, with an explicit floor on the late tail.

    ``run_late_subclustering``'s own guard is ``len(idx) >= max(k_range) * 10``,
    which at k_range=(2,4) admits a 4-cluster fit to 40 patches. A slide's late
    decile is only ~10% of THAT SLIDE: 2M-2's smallest slide has 414 patches, so
    its late tail is 41 — over the inherited floor and far too thin to read. The
    floor is therefore raised here and applied BEFORE the call, so a thin slide is
    reported as skipped with its patch count rather than silently yielding a fit.
    """
    print("\n=== TASK 3 — late subclustering, within each slide ===")
    per_slide, attempted = {}, 0
    for s, sub in views:
        pt = sub.obs["pseudotime"].values.astype(float)
        n_late = int((pt >= np.quantile(pt, 1 - tail)).sum())
        if n_late < min_late:
            per_slide[s] = {
                "attempted": False,
                "n_late_patches": n_late,
                "reason": (f"late tail is {n_late} patches, below the "
                           f"{min_late}-patch floor for within-slide subclustering"),
            }
            print(f"  slide {s}: SKIPPED — {per_slide[s]['reason']}")
            continue
        r = run_late_subclustering(sub, tail, k_range)
        per_slide[s] = r
        if r.get("attempted"):
            attempted += 1
            print(f"  slide {s}: n_late={r['n_late_patches']}  k={r['best_k']}  "
                  f"opposing features {r['n_features_with_opposing_subclusters']}/"
                  f"{len(MORPH_FEATURES)}  "
                  f"silhouette={max(r['silhouette_scores'].values()):.3f}")
        else:
            print(f"  slide {s}: SKIPPED — {r.get('reason')}")

    summary = {}
    for feat in MORPH_FEATURES:
        opp = 0
        for r in per_slide.values():
            if not r.get("attempted"):
                continue
            o = r["opposing_directions"].get(feat)
            if o and o["spans_zero"] and o["opposing_magnitude"] >= 0.25:
                opp += 1
        summary[feat] = {"n_slides_opposing": opp, "n_slides_attempted": attempted}
    return {"per_slide": per_slide, "per_feature_across_slides": summary,
            "n_slides_attempted": attempted}


# ── verdicts ─────────────────────────────────────────────────────────────────

def build_verdict(t1: dict, t2: dict, t3: dict) -> dict:
    v = t1.get("subcluster_by_slide", {}) or {}
    cv = v.get("cramers_v")
    n_slides = t2["n_slides"]

    if cv is None:
        batch_call = "Cramer's V could not be computed; the subcluster/slide test is unavailable."
    elif cv >= 0.5:
        batch_call = (
            f"Cramer's V = {cv:.3f}. The cohort late subclusters largely ARE slides, so "
            "eccentricity_check's opposing-feature verdict (its Verdict 3) is a BATCH "
            "SPLIT and should not be read as two late phenotypes."
        )
    elif cv >= 0.3:
        batch_call = (
            f"Cramer's V = {cv:.3f}. The cohort late subclusters are partly aligned with "
            "slides. The opposing-feature verdict is contaminated but not wholly "
            "explained by slide; read the within-slide counts below."
        )
    else:
        batch_call = (
            f"Cramer's V = {cv:.3f}. The cohort late subclusters cut ACROSS slides, so "
            "the opposing-feature verdict is not a batch split and stands on its own."
        )

    bidir = {f: s for f, s in t2["per_feature_across_slides"].items()
             if s["n_slides_bidirectional"] > s["n_slides_unidirectional"]}
    opp = {f: s for f, s in t3["per_feature_across_slides"].items()
           if s["n_slides_attempted"] and
           s["n_slides_opposing"] >= max(2, s["n_slides_attempted"] / 2)}

    return {
        "late_tail_slide_concentration": {
            "slide": t1["most_over_represented_slide"],
            "fold": t1["most_over_represented_fold"],
            "exceeds_concern": t1["exceeds_fold_concern"],
        },
        "subcluster_is_slide": batch_call,
        "features_bidirectional_in_most_slides": sorted(bidir),
        "features_opposing_in_most_slides": sorted(opp),
        "overall": (
            ("WITHIN SLIDES, the eccentricity signature is ABSENT: no feature is "
             "bidirectional in a majority of slides."
             if not bidir else
             f"WITHIN SLIDES, {len(bidir)} feature(s) are bidirectional in a majority "
             f"of slides ({sorted(bidir)}) — the eccentricity signature survives "
             "de-confounding for those features.")
            + " "
            + ("No feature has opposing late subclusters inside a majority of slides, "
               "so any cohort-level opposition is between slides rather than within "
               "the late state."
               if not opp else
               f"{len(opp)} feature(s) have opposing late subclusters inside a "
               f"majority of slides ({sorted(opp)}), which a slide contrast cannot "
               "explain.")
            + f" Based on {n_slides} slide(s)."
        ),
        "not_evidence_of": (
            "None of this validates the pseudotime or the anchor. It only separates "
            "'the late state has structure' from 'the late tail is one slide'. The "
            "axis it describes is still anchored on a root set that anchor_area_control "
            "showed to be duct-size-extreme."
        ),
    }


# ── driver ───────────────────────────────────────────────────────────────────

def run_section(section: str, run_dir: Path, tail: float, k_cohort: tuple,
                k_within: tuple, min_patches: int, min_late: int) -> dict:
    print("\n" + "=" * 78)
    print(f"  SECTION {section}   ({run_dir})")
    print("=" * 78)
    adata, _, _ = load_run(run_dir)
    for col in ("pseudotime", *MORPH_FEATURES):
        if col not in adata.obs.columns:
            raise KeyError(
                f"adata.obs['{col}'] missing from {run_dir} — this analysis reads an "
                "existing run and cannot regenerate it.")

    t1 = task1_late_tail_composition(adata, tail, k_cohort)
    views, skipped = _slide_views(adata, min_patches)
    if not views:
        raise ValueError(
            f"No slide has >= {min_patches} patches, so no within-slide decile is "
            f"usable. Slide sizes: {skipped}")
    if skipped:
        print(f"\n  NOTE: {len(skipped)} slide(s) skipped for having "
              f"< {min_patches} patches: {skipped}")

    t2 = task2_within_slide_enrichment(views, tail)
    t3 = task3_within_slide_subclustering(views, tail, k_within, min_late)
    return {
        "section": section,
        "run_dir": str(run_dir),
        "n_patches": int(adata.n_obs),
        "slides_skipped_too_few_patches": skipped,
        "task1_late_tail_composition": t1,
        "task2_within_slide_enrichment": t2,
        "task3_within_slide_subclustering": t3,
        "verdict": build_verdict(t1, t2, t3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tail-fraction", type=float, default=EXTREME_DECILE)
    ap.add_argument("--k-min-cohort", type=int, default=K_RANGE_COHORT_DEFAULT[0])
    ap.add_argument("--k-max-cohort", type=int, default=K_RANGE_COHORT_DEFAULT[1])
    ap.add_argument("--k-min-within", type=int, default=K_RANGE_WITHIN_DEFAULT[0])
    ap.add_argument("--k-max-within", type=int, default=K_RANGE_WITHIN_DEFAULT[1])
    ap.add_argument("--min-patches-per-slide", type=int, default=MIN_PATCHES_PER_SLIDE,
                    help="Slides smaller than this are dropped from the analysis.")
    ap.add_argument("--min-late-per-slide", type=int, default=MIN_LATE_PER_SLIDE,
                    help="Late-tail floor for within-slide subclustering.")
    args = ap.parse_args()

    if len(args.sections) != len(args.run_dirs):
        raise SystemExit("--sections and --run-dirs must match in length and order.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {"analysis": "eccentricity_within_slide",
           "config": {k: str(v) for k, v in vars(args).items()},
           "sections": []}
    for sec, d in zip(args.sections, args.run_dirs):
        res["sections"].append(run_section(
            sec, d, args.tail_fraction,
            (args.k_min_cohort, args.k_max_cohort),
            (args.k_min_within, args.k_max_within),
            args.min_patches_per_slide, args.min_late_per_slide))

    out = args.output_dir / "eccentricity_within_slide.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    write_report(res, args.output_dir / "eccentricity_within_slide.md")
    print(f"\nWrote {out}")


def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    add("# Is the late-pseudotime structure biology, or is it one slide?\n")
    add("`eccentricity_check` checked Task A within slides but ran Task B's "
        "enrichment and subclustering on the GLOBAL top decile, which is majority "
        "one slide. This redoes both inside each slide, and asks directly whether "
        "the cohort late subclusters ARE slides.\n")

    for s in res["sections"]:
        t1, t2, t3, v = (s["task1_late_tail_composition"],
                         s["task2_within_slide_enrichment"],
                         s["task3_within_slide_subclustering"], s["verdict"])
        add(f"\n## {s['section']}\n")
        add(f"- {s['n_patches']} patches, {t2['n_slides']} slide(s) analysed")
        add(f"- late tail: **{t1['n_late']}** patches; most over-represented slide "
            f"**{t1['most_over_represented_slide']}** at "
            f"**{t1['most_over_represented_fold']:.2f}x** its cohort share"
            + ("  ⚠ exceeds the 3x concern threshold" if t1["exceeds_fold_concern"] else ""))

        add("\n### Task 1 — are the cohort late subclusters just slides?\n")
        c = t1.get("subcluster_by_slide")
        if not c:
            add("Cohort subclustering did not run, so this test is unavailable.")
        else:
            add(f"- Cramer's V (subcluster x slide) = "
                + (f"**{c['cramers_v']:.3f}**" if c["cramers_v"] is not None else "n/a"))
            for cid, share in (c.get("per_cluster_max_slide_share") or {}).items():
                n = t1["cohort_subclustering"]["clusters"][cid]["n"]
                add(f"- cluster {cid}: n={n}, largest single slide = "
                    + (f"{100*share:.1f}%" if share is not None else "n/a"))
            add(f"\n> {v['subcluster_is_slide']}")

        add("\n### Task 2 — bidirectional enrichment inside each slide\n")
        add("| feature | slides bidirectional | slides unidirectional | slides neither |")
        add("|---|---|---|---|")
        for f, r in t2["per_feature_across_slides"].items():
            add(f"| `{f}` | {r['n_slides_bidirectional']}/{r['n_slides_tested']} | "
                f"{r['n_slides_unidirectional']}/{r['n_slides_tested']} | "
                f"{r['n_slides_neither']}/{r['n_slides_tested']} |")

        add("\n### Task 3 — late subclustering inside each slide\n")
        thin = {k: r.get("n_late_patches") for k, r in t3["per_slide"].items()
                if not r.get("attempted")}
        if thin:
            add(f"- {len(thin)} slide(s) skipped for a late tail below the floor: "
                f"{thin}\n")
        add("| feature | slides with opposing subclusters |")
        add("|---|---|")
        for f, r in t3["per_feature_across_slides"].items():
            add(f"| `{f}` | {r['n_slides_opposing']}/{r['n_slides_attempted']} |")

        add(f"\n**Verdict.** {v['overall']}\n")
        add(f"> {v['not_evidence_of']}")

    add("\n## How to read this against eccentricity_check\n")
    add("- Within-slide thresholds are computed inside each slide, so the numbers "
        "are NOT on the cohort scale. Compare PATTERNS (bidirectional vs "
        "unidirectional, opposing vs not), never values.")
    add("- A feature that is bidirectional cohort-wide but in no slide was showing "
        "a slide contrast. One that is bidirectional in most slides is showing "
        "eccentricity that de-confounding does not remove.")
    add("- `eccentricity_check`'s Verdict 4 counts only Tasks A and B and omits the "
        "subclustering test. Where Task 1 below returns a high Cramer's V, that "
        "omission happens to be harmless; where it returns a low one, Verdict 4 is "
        "overstated and the subclustering result needs reporting alongside it.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
