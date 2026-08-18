"""Phase 3 companion: nesting, uncertainty, and what "hole %" actually measures.

Phase 2 reported every duct-level correlation as a single pooled point estimate
over ~1,400-1,600 ducts drawn from 8 slides. Three things were never checked.

  TASK 1  NESTING. Ducts are nested within slides, and the v1-v3 holey-ness work
          already found per-slide partial correlations spanning -0.069 to +0.30
          on the same cohort — i.e. a real Simpson's-paradox risk. 2M-2's
          headline move, rho(pt, duct area) from -0.084 to +0.249, could be a
          between-slide shift in slide medians rather than anything happening
          within a slide. This task recomputes every duct-level correlation
          WITHIN each slide, alongside a between-slide correlation over the 8
          slide medians, so the two can be told apart.

  TASK 2  UNCERTAINTY. No interval was ever put on any of these rhos. With 8
          slides, the honest resampling unit is the SLIDE, not the duct: a
          duct-level bootstrap would treat 1,360 ducts as 1,360 independent
          observations when they are 8 clusters. This task runs a cluster
          bootstrap — resample slides with replacement, then ducts within each
          drawn slide — and reports percentile intervals. Expect them to be much
          wider than a naive duct bootstrap would give; that width is the point.

  TASK 3  WHAT THE ANNOTATION MEASURES. The root inspection JSONs show the
          hand-annotated LEAST holey ducts contain the WHITEST patches
          (frac_pixels_white median ~0.28 for the holeyroot roots vs ~0.15 for
          v2's), which is backwards if "hole %" tracked optical white space. At
          the same time `h_intensity_wholepatch` — which white space mechanically
          depresses — moved 0.039 -> 0.323 in 2M-2 and is counted in Phase 2 as
          one of three features whose direction now agrees across sections.
          Both cannot be comfortable at once. This task correlates duct hole %
          against the pipeline's own patch-derived quantities at duct level, so
          the reader can see how much of the anchor variable the pipeline
          already sees, and therefore how independent `h_intensity_wholepatch`
          really is as a validator.

          It also reports the hole % distribution per section. 2M-1's roots sit
          at hole % median 0.025 while 2M-2's sit at 1.80 with a P10 threshold of
          7.63 — two orders of magnitude apart, from different export files
          (2M-2's is a column-renamed holes_pfa export). "Lowest holey-ness"
          does not denote the same thing in the two sections, which bears
          directly on whether their reconciliation in Phase 2 means anything.

Reads results.csv, the measurement exports and the annotations. Touches no run
tree. Writes only --output-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .holeyness import (
    PATCH_SIZE_DEFAULT,
    load_slide_list, load_slide_dimensions, parse_measurement_export,
    load_duct_polygons, build_duct_table, assign_patches_to_ducts,
    aggregate_per_duct, _partial_spearman,
)

# Patch features worth aggregating beyond what holeyness.aggregate_per_duct
# covers. h_intensity_wholepatch is the one Task 3 exists for.
OPTICS_FEATURES = ["h_intensity_wholepatch", "h_intensity", "nuclear_density",
                   "texture_entropy"]
MIN_DUCTS_PER_SLIDE = 20      # below this a per-slide rho is not reported
N_BOOT_DEFAULT = 2000


def _safe_rho(x, y, min_n: int = 10) -> float:
    """Spearman with an explicit minimum n.

    ``min_n`` is a parameter and not a constant because the between-slide
    correlation is computed over 8 SLIDE MEDIANS. The project-wide default of 10
    would silently return NaN for every between-slide value, which reads in the
    output as "not computed" rather than "n is small" — the two mean very
    different things and only one of them is true here. Callers that drop below
    10 must say so in the output; task1_nesting does.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_per_duct(results_csv: Path, export: Path, ann_dir: Path, dims: Path,
                   slide_list: Path, patch_size: int) -> pd.DataFrame:
    """Per-duct table, using holeyness.py's pipeline unmodified for the columns
    it covers and a matching nanmedian groupby for the extra optics features."""
    slides = load_slide_list(slide_list)
    meas = parse_measurement_export(export, slides)
    polys = load_duct_polygons(ann_dir, slides, load_slide_dimensions(dims))
    dt = build_duct_table(meas, polys)
    df = pd.read_csv(results_csv)
    assigned = assign_patches_to_ducts(df, dt, patch_size=patch_size)
    per_duct = aggregate_per_duct(assigned, dt, np.nanmedian, "median")

    extra = [f for f in OPTICS_FEATURES if f in assigned.columns
             and f not in per_duct.columns]
    if extra:
        sub = assigned[assigned["duct_id"].notna()]
        agg = sub.groupby("duct_id")[extra].median().reset_index()
        agg = agg.rename(columns={"duct_id": "object_id"})
        per_duct = per_duct.merge(agg, on="object_id", how="left")
    missing = [f for f in OPTICS_FEATURES if f not in per_duct.columns]
    if missing:
        print(f"  NOTE: {missing} absent from results.csv — Task 3 will skip them "
              "rather than substitute anything.")
    return per_duct


# ── TASK 1: within-slide vs between-slide ────────────────────────────────────

PAIRS = [
    ("pt_hole_pct", "pseudotime", "hole_pct"),
    ("pt_area", "pseudotime", "area_um2"),
    ("pt_nuclear_density", "pseudotime", "nuclear_density"),
    ("hole_area", "hole_pct", "area_um2"),
]


def task1_nesting(per_duct: pd.DataFrame) -> dict:
    """Pooled, within-slide and between-slide versions of every duct-level rho."""
    out: dict = {"pooled": {}, "per_slide": {}, "between_slide": {},
                 "within_slide_summary": {}}
    for label, a, b in PAIRS:
        out["pooled"][label] = _safe_rho(per_duct[a].values, per_duct[b].values)

    slides = sorted(per_duct["slide_name"].dropna().unique())
    for s in slides:
        g = per_duct[per_duct["slide_name"] == s]
        rec = {"n_ducts": int(len(g))}
        for label, a, b in PAIRS:
            rec[label] = (_safe_rho(g[a].values, g[b].values)
                          if len(g) >= MIN_DUCTS_PER_SLIDE else None)
        out["per_slide"][str(s)] = rec

    # Between-slide: one point per slide, so n = 8. Reported DESCRIPTIVELY —
    # a Spearman on 8 points has no useful precision and is here only to show
    # whether the pooled value is carried by slide-level structure.
    med = per_duct.groupby("slide_name").median(numeric_only=True)
    out["between_slide_n"] = int(len(med))
    out["between_slide_note"] = (
        f"Spearman over {len(med)} slide medians. Descriptive only: at this n the "
        "value has no useful precision, and no p-value or interval is reported for "
        "it. Its purpose is to show whether a pooled duct-level correlation is "
        "really a correlation between slides."
    )
    for label, a, b in PAIRS:
        out["between_slide"][label] = (
            _safe_rho(med[a].values, med[b].values, min_n=4)
            if len(med) >= 4 else None)

    for label, _, _ in PAIRS:
        vals = np.array([v[label] for v in out["per_slide"].values()
                         if v[label] is not None], dtype=float)
        vals = vals[np.isfinite(vals)]
        pooled = out["pooled"][label]
        out["within_slide_summary"][label] = {
            "n_slides_reported": int(vals.size),
            "median": float(np.median(vals)) if vals.size else None,
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "n_same_sign_as_pooled": int((np.sign(vals) == np.sign(pooled)).sum())
                                     if vals.size else 0,
            "pooled": pooled,
            "note": ("If the pooled value's sign is carried by only a minority of "
                     "slides, or if the between-slide value is much larger, the "
                     "pooled number is a slide-level effect, not a duct-level one."),
        }
    out["n_slides"] = len(slides)
    out["min_ducts_per_slide_for_reporting"] = MIN_DUCTS_PER_SLIDE
    return out


# ── TASK 2: cluster bootstrap ────────────────────────────────────────────────

def task2_cluster_bootstrap(per_duct: pd.DataFrame, n_boot: int,
                            seed: int) -> dict:
    """Percentile intervals with the SLIDE as the resampling unit.

    Resamples the 8 slides with replacement, then the ducts within each drawn
    slide with replacement. A duct-level bootstrap would treat ~1,400 nested
    observations as independent and report an interval several times too narrow.
    With only 8 clusters the interval is wide and the tails are coarse; that is a
    property of the design, not of the method, and is reported rather than
    smoothed over.
    """
    rng = np.random.default_rng(seed)
    slides = sorted(per_duct["slide_name"].dropna().unique())
    groups = {s: per_duct[per_duct["slide_name"] == s] for s in slides}

    draws = {label: np.full(n_boot, np.nan) for label, _, _ in PAIRS}
    for b in range(n_boot):
        picked = rng.choice(len(slides), size=len(slides), replace=True)
        parts = []
        for i in picked:
            g = groups[slides[int(i)]]
            idx = rng.integers(0, len(g), size=len(g))
            parts.append(g.iloc[idx])
        boot = pd.concat(parts, ignore_index=True)
        for label, a, b_ in PAIRS:
            draws[label][b] = _safe_rho(boot[a].values, boot[b_].values)

    out = {"n_boot": int(n_boot), "n_slides": len(slides),
           "resampling_unit": "slide, then ducts within slide"}
    for label, a, b_ in PAIRS:
        v = draws[label][np.isfinite(draws[label])]
        point = _safe_rho(per_duct[a].values, per_duct[b_].values)
        out[label] = {
            "point_estimate": point,
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    if v.size else None,
            "ci80": [float(np.percentile(v, 10)), float(np.percentile(v, 90))]
                    if v.size else None,
            "frac_draws_opposite_sign_to_point":
                float((np.sign(v) != np.sign(point)).mean()) if v.size else None,
            "n_usable_draws": int(v.size),
        }
    return out


# ── TASK 3: what does the annotation measure? ────────────────────────────────

def task3_annotation_vs_optics(per_duct: pd.DataFrame) -> dict:
    """How much of the anchor variable the pipeline's own pixels already see."""
    hole = per_duct["hole_pct"].values
    out = {
        "hole_pct_distribution": {
            "n_ducts": int(np.isfinite(hole).sum()),
            "median": float(np.nanmedian(hole)),
            "p10": float(np.nanpercentile(hole, 10)),
            "p90": float(np.nanpercentile(hole, 90)),
            "min": float(np.nanmin(hole)), "max": float(np.nanmax(hole)),
            "frac_exactly_zero": float(np.nanmean(hole == 0.0)),
        },
        "rho_hole_vs": {},
        "partial_given_area": {},
    }
    area = per_duct["area_um2"].values
    for f in OPTICS_FEATURES:
        if f not in per_duct.columns:
            continue
        v = per_duct[f].values
        out["rho_hole_vs"][f] = _safe_rho(hole, v)
        out["partial_given_area"][f] = _partial_spearman(hole, v, area)
    out["interpretation"] = (
        "A LARGE |rho| between duct hole % and a patch-derived optical feature "
        "means that feature is not an independent validator of an axis anchored "
        "on hole % — it is partly the anchor seen through the pipeline's own "
        "pixels. h_intensity_wholepatch is the specific case: white space "
        "mechanically depresses whole-patch haematoxylin intensity. A NEAR-ZERO "
        "|rho| is not reassuring either: it would mean the hand annotation and "
        "the pipeline disagree about what a hole is, and the anchor's stated "
        "biological direction (low hole % = early) then rests on a quantity the "
        "pipeline never sees."
    )
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def run_one(label: str, results_csv: Path, export: Path, ann_dir: Path,
            dims: Path, slide_list: Path, patch_size: int, n_boot: int,
            seed: int) -> dict:
    print("\n" + "=" * 78)
    print(f"  {label}")
    print("=" * 78)
    per_duct = build_per_duct(results_csv, export, ann_dir, dims, slide_list,
                              patch_size)
    t1 = task1_nesting(per_duct)
    print(f"\n  pooled rho(pt, area)   : {t1['pooled']['pt_area']:+.4f}")
    ws = t1["within_slide_summary"]["pt_area"]
    print(f"  within-slide median    : {ws['median']:+.4f}  "
          f"(same sign in {ws['n_same_sign_as_pooled']}/{ws['n_slides_reported']} slides)")
    print(f"  between-slide (n={t1['n_slides']})   : "
          f"{t1['between_slide']['pt_area']:+.4f}")
    t2 = task2_cluster_bootstrap(per_duct, n_boot, seed)
    print(f"  cluster-bootstrap 95% CI on rho(pt, area): {t2['pt_area']['ci95']}")
    return {
        "label": label,
        "n_ducts": int(len(per_duct)),
        "task1_nesting": t1,
        "task2_cluster_bootstrap": t2,
        "task3_annotation_vs_optics": task3_annotation_vs_optics(per_duct),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="e.g. v2_2M-1 holeyroot_2M-1 v2_2M-2 holeyroot_2M-2")
    ap.add_argument("--results-csvs", nargs="+", type=Path, required=True)
    ap.add_argument("--exports", nargs="+", type=Path, required=True,
                    help="One per label, SAME ORDER. 2M-1 and 2M-2 read different "
                         "export files.")
    ap.add_argument("--slide-lists", nargs="+", type=Path, required=True)
    ap.add_argument("--annotation-dir", type=Path, required=True)
    ap.add_argument("--slide-dimensions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = len(args.labels)
    for name, seq in (("--results-csvs", args.results_csvs),
                      ("--exports", args.exports),
                      ("--slide-lists", args.slide_lists)):
        if len(seq) != n:
            raise SystemExit(
                f"{name} has {len(seq)} entries but --labels has {n}. These are "
                "positional; a mismatch would analyse one section with another's "
                "annotations and still produce plausible numbers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {"config": {k: str(v) for k, v in vars(args).items()}, "runs": []}
    for i, lbl in enumerate(args.labels):
        res["runs"].append(run_one(
            lbl, args.results_csvs[i], args.exports[i], args.annotation_dir,
            args.slide_dimensions, args.slide_lists[i], args.patch_size,
            args.n_boot, args.seed))

    out = args.output_dir / "holeyroot_duct_checks.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    write_report(res, args.output_dir / "holeyroot_duct_checks.md")
    print(f"\nWrote {out}")


def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    add("# Phase 3 companion — nesting, uncertainty, and what hole % measures\n")

    add("## Task 1 — pooled vs within-slide vs between-slide\n")
    add("| run | quantity | pooled | within-slide median | same sign | between-slide |")
    add("|---|---|---|---|---|---|")
    for r in res["runs"]:
        t1 = r["task1_nesting"]
        for label, _, _ in PAIRS:
            w = t1["within_slide_summary"][label]
            bs = t1["between_slide"][label]
            add(f"| {r['label']} | `{label}` | {t1['pooled'][label]:+.4f} | "
                + (f"{w['median']:+.4f}" if w['median'] is not None else "—")
                + f" | {w['n_same_sign_as_pooled']}/{w['n_slides_reported']} | "
                + (f"{bs:+.4f}" if bs is not None else "—") + " |")
    add("\n> A pooled value whose sign only a minority of slides carry, or which "
        "the between-slide correlation dwarfs, is a slide-level effect.\n")

    add("## Task 2 — cluster bootstrap (slide is the resampling unit)\n")
    add("| run | quantity | point | 95% CI | frac draws opposite sign |")
    add("|---|---|---|---|---|")
    for r in res["runs"]:
        t2 = r["task2_cluster_bootstrap"]
        for label, _, _ in PAIRS:
            b = t2[label]
            ci = (f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
                  if b["ci95"] else "—")
            add(f"| {r['label']} | `{label}` | {b['point_estimate']:+.4f} | {ci} | "
                f"{b['frac_draws_opposite_sign_to_point']:.3f} |")
    add(f"\n> {res['runs'][0]['task2_cluster_bootstrap']['n_slides']} clusters only. "
        "The intervals are wide because the design has 8 slides, not because the "
        "method is conservative.\n")

    add("## Task 3 — duct hole % against the pipeline's own pixels\n")
    add("| run | feature | rho(hole %, feature) | partial given area |")
    add("|---|---|---|---|")
    for r in res["runs"]:
        t3 = r["task3_annotation_vs_optics"]
        for f, v in t3["rho_hole_vs"].items():
            p = t3["partial_given_area"].get(f)
            add(f"| {r['label']} | `{f}` | {v:+.4f} | "
                + (f"{p:+.4f}" if p is not None else "—") + " |")
    add("")
    for r in res["runs"]:
        d = r["task3_annotation_vs_optics"]["hole_pct_distribution"]
        add(f"- **{r['label']}** hole %: median {d['median']:.3f}, "
            f"P10 {d['p10']:.3f}, P90 {d['p90']:.3f}, "
            f"{100*d['frac_exactly_zero']:.1f}% exactly zero")
    add("\n> The two sections' hole % distributions come from different export "
        "files and are on different scales. 'Lowest holey-ness' is not the same "
        "anchor in 2M-1 and 2M-2, so their agreement in Phase 2 is agreement "
        "between two differently-defined anchors.\n")
    add(f"> {res['runs'][0]['task3_annotation_vs_optics']['interpretation']}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
