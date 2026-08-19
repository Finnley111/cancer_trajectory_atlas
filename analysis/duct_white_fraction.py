"""Does the expert hole % annotation actually track white space in the duct?

WHY THIS EXISTS
---------------
The whole holey-ness anchor rests on ``hole_pct``, a hand annotation in QuPath.
``holeyroot_duct_checks`` Task 3 tested it against the pipeline's own pixels and
got an uncomfortable answer — the two sections disagree in SIGN on three of four
optical features:

    rho(hole %, ...)          2M-1      2M-2
    h_intensity_wholepatch   +0.080    -0.271
    h_intensity              +0.202    -0.098
    nuclear_density          +0.051    -0.120
    texture_entropy          +0.377    +0.335

Holes are white, and white space mechanically depresses whole-patch haematoxylin
intensity, so 2M-2 has the physically expected sign and 2M-1 has the wrong one.
In 2M-1 the annotation is close to INVISIBLE to the pipeline. That is the section
whose trajectory verdict also collapses to ECCENTRICITY IN EMBEDDING ONLY once
the anchor's duct-size extremity is removed.

But Task 3 measured through a PROXY and through the patch-to-duct assignment. Its
per-duct values are medians over patches whose CENTRES fall inside the polygon,
which excludes 571/2173 ducts in 2M-1 and 389/1749 in 2M-2 — systematically the
smallest — and which measures 112 px windows, not the duct.

THIS MODULE MEASURES THE THING DIRECTLY. For every Tumor polygon it rasterises
the annotation against the slide PNG and computes the fraction of pixels inside
the duct that are white, then correlates that with the expert's ``hole_pct``.
No patch assignment, no proxy, and every duct included — including the zero-patch
ones no previous analysis could see.

THE TWO HYPOTHESES IT SEPARATES
-------------------------------
  HIGH correlation in both sections
      The annotation is a real measure of optical holes. Task 3's weak and
      sign-flipped 2M-1 coupling is then an artifact of patch aggregation and
      the exclusion bias, not of the annotation, and the anchor's stated
      direction (low hole % = early) is measuring what it claims.

  LOW correlation in 2M-1
      The 2M-1 ``holes_carnoys`` column does not denote optical holes. The anchor
      is then invalid in that section regardless of everything else, and every
      2M-1 result built on it — Phase 2's re-anchoring, Phase 3's Task C — is
      describing an ordering by a quantity with no pixel referent.

WHAT IS MEASURED, EXACTLY
-------------------------
``frac_pixels_white`` is reused byte-for-byte from
``diagnostics/inspect_roots_v3.py``: the fraction of pixels whose MEAN RGB
exceeds ``WHITE_THRESH`` (220). That constant is imported, not restated, so the
root sheets and this analysis can never drift apart. Note PIL's own "L"
conversion is ITU-R 601-2 luma and is NOT the same thing — the mean is computed
here explicitly.

OUTER RING ONLY. ``holeyness.load_duct_polygons`` builds each MplPath from
``geometry["coordinates"][0]``, so a duct's own lumen is not subtracted from its
mask. That is exactly right here: ``hole_pct`` is presumably hole area over duct
area, so white measured inside the OUTER boundary is the quantity it should be
compared against. It also matches what ``assign_patches_to_ducts`` does, so this
analysis and every earlier one describe the same regions.

COORDINATE SPACE. Polygons are scaled by ``original_full_width/height`` and
right-half copies are discarded, which puts left-half annotations in
[0, cropped_width] — the SAME space as the pipeline's patch coordinates and the
cropped PNG on disk (see features/patching.py:85). The crop is horizontal only,
so ``cropped_height == original_full_height``. Every mask is nonetheless checked
against the image bounds and out-of-bounds ducts are counted, not clipped
silently.

WHAT IT REPORTS BEYOND THE HEADLINE CORRELATION
-----------------------------------------------
  TASK A  the primary correlation, per slide as well as pooled, with a
          slide-clustered bootstrap interval and a within-slide permutation null.
          Eight slides is eight clusters; the interval is what it is.
  TASK B  the exclusion-bias test. The same correlation computed separately on
          ducts WITH assigned patches and ducts WITHOUT. If they agree, patch
          assignment does not explain Task 3's weakness; if the zero-patch ducts
          behave differently, the bias is real and measurable for the first time.
  TASK C  calibration, not just correlation. ``hole_pct`` and the measured white
          fraction are both percentages of duct area, so they can be compared on
          one scale. A high rho with a large systematic offset means the
          annotation ranks ducts correctly but does not measure what it names.
  TASK D  threshold sensitivity. 220 is a convention; rho is recomputed across
          200-240 so a conclusion that depends on the threshold is visible.
  TASK E  the duct-area confound, since bigger ducts may simply have more lumen.

READ-ONLY. Reads the annotations, the measurement exports and the slide PNGs.
Writes only --output-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .holeyness import (
    load_slide_list, load_slide_dimensions, parse_measurement_export,
    load_duct_polygons, build_duct_table, assign_patches_to_ducts,
    _partial_spearman, PATCH_SIZE_DEFAULT,
)
from .holeyroot_duct_checks import _safe_rho, _json_default
from ..diagnostics.inspect_roots_v3 import WHITE_THRESH

THRESHOLDS = [200, 210, 220, 230, 240]
N_BOOT_DEFAULT = 2000
N_PERM_DEFAULT = 2000
MIN_MASK_PIXELS = 100          # below this a duct's white fraction is noise
MIN_DUCTS_PER_SLIDE = 20
STRONG_RHO = 0.50              # what counts as the annotation tracking the pixels


# ── rasterisation ────────────────────────────────────────────────────────────

def _polygon_mask(vertices: np.ndarray, x0: int, y0: int, w: int, h: int):
    """Filled boolean mask of the polygon within the given bbox window.

    PIL's ImageDraw is used rather than MplPath.contains_points because the
    latter would be ~10^8 point-in-polygon tests over 2,000 ducts. Falls back to
    matplotlib if PIL is unavailable rather than approximating with the bbox,
    which would silently turn every duct into a rectangle.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:                                        # pragma: no cover
        from matplotlib.path import Path as MplPath
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
        pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
        return MplPath(vertices).contains_points(pts).reshape(h, w)

    img = Image.new("1", (w, h), 0)
    poly = [(float(vx) - x0, float(vy) - y0) for vx, vy in vertices]
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    return np.array(img, dtype=bool)


def measure_ducts(duct_table: pd.DataFrame, png_dir: Path,
                  thresholds: list[int]) -> pd.DataFrame:
    """White fraction inside every duct polygon, one slide decoded at a time."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None       # these are whole-slide PNGs, not a bomb

    rows = []
    for slide_name, grp in duct_table.groupby("slide_name"):
        path = png_dir / f"{slide_name}.png"
        if not path.exists():
            hits = list(png_dir.glob(f"{slide_name}.*"))
            path = hits[0] if hits else None
        if path is None:
            raise FileNotFoundError(
                f"No image for {slide_name} in {png_dir}. Every slide in the duct "
                "table must be decodable; skipping one would silently drop its "
                "ducts from the correlation.")

        rgb = np.asarray(Image.open(path).convert("RGB"))
        H, W = rgb.shape[:2]
        # mean RGB as uint16 sums, to match inspect_roots_v3's gray = rgb.mean(2)
        # without materialising a float array the size of a whole slide.
        gray_sum = (rgb[..., 0].astype(np.uint16) + rgb[..., 1] + rgb[..., 2])
        del rgb

        n_oob = n_tiny = 0
        for _, r in grp.iterrows():
            v = np.asarray(r["polygon"].vertices, dtype=float)
            x0, y0 = int(np.floor(v[:, 0].min())), int(np.floor(v[:, 1].min()))
            x1, y1 = int(np.ceil(v[:, 0].max())), int(np.ceil(v[:, 1].max()))
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(W, x1), min(H, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                n_oob += 1
                continue
            clipped = (cx0 != x0) or (cy0 != y0) or (cx1 != x1) or (cy1 != y1)

            mask = _polygon_mask(v, cx0, cy0, cx1 - cx0, cy1 - cy0)
            n_px = int(mask.sum())
            if n_px < MIN_MASK_PIXELS:
                n_tiny += 1
                continue

            vals = gray_sum[cy0:cy1, cx0:cx1][mask]
            row = {
                "object_id": r["object_id"],
                "slide_name": slide_name,
                "hole_pct": float(r["hole_pct"]),
                "area_um2": float(r["area_um2"]),
                "n_mask_pixels": n_px,
                "mean_intensity": float(vals.mean()) / 3.0,
                "bbox_clipped_at_image_edge": bool(clipped),
            }
            for t in thresholds:
                row[f"white_frac_{t}"] = float((vals > 3 * t).mean())
            rows.append(row)

        print(f"  {slide_name}: {len(grp)} ducts, "
              f"{sum(1 for x in rows if x['slide_name'] == slide_name)} measured"
              + (f", {n_oob} outside image" if n_oob else "")
              + (f", {n_tiny} under {MIN_MASK_PIXELS}px" if n_tiny else ""))
        del gray_sum

    if not rows:
        raise ValueError("No duct produced a usable mask.")
    return pd.DataFrame(rows)


# ── TASK A: the primary correlation ──────────────────────────────────────────

def task_a_primary(df: pd.DataFrame, col: str, n_boot: int, n_perm: int,
                   seed: int) -> dict:
    hole = df["hole_pct"].values
    white = df[col].values
    pooled = _safe_rho(hole, white)

    per_slide, slides = {}, sorted(df["slide_name"].unique())
    for s in slides:
        g = df[df["slide_name"] == s]
        per_slide[s] = {
            "n_ducts": int(len(g)),
            "rho": (_safe_rho(g["hole_pct"].values, g[col].values)
                    if len(g) >= MIN_DUCTS_PER_SLIDE else None),
        }
    vals = np.array([v["rho"] for v in per_slide.values() if v["rho"] is not None],
                    dtype=float)

    rng = np.random.default_rng(seed)
    groups = {s: df[df["slide_name"] == s] for s in slides}
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        parts = []
        for i in rng.choice(len(slides), size=len(slides), replace=True):
            g = groups[slides[int(i)]]
            parts.append(g.iloc[rng.integers(0, len(g), size=len(g))])
        bb = pd.concat(parts, ignore_index=True)
        boot[b] = _safe_rho(bb["hole_pct"].values, bb[col].values)
    boot = boot[np.isfinite(boot)]

    # Within-slide permutation: shuffle hole % inside each slide, so slide-level
    # brightness differences are preserved and only the duct-to-duct pairing is
    # broken. A pooled shuffle would break both and overstate significance.
    null = np.full(n_perm, np.nan)
    idx_by_slide = [np.flatnonzero((df["slide_name"] == s).values) for s in slides]
    for p in range(n_perm):
        h = hole.copy()
        for idx in idx_by_slide:
            h[idx] = rng.permutation(h[idx])
        null[p] = _safe_rho(h, white)
    null = null[np.isfinite(null)]

    return {
        "white_column": col,
        "n_ducts": int(len(df)),
        "pooled_rho": pooled,
        "per_slide": per_slide,
        "within_slide_median_rho": float(np.median(vals)) if vals.size else None,
        "n_slides_same_sign_as_pooled": (int((np.sign(vals) == np.sign(pooled)).sum())
                                         if vals.size else 0),
        "n_slides_reported": int(vals.size),
        "cluster_bootstrap": {
            "n_boot": int(boot.size),
            "ci95": [float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))] if boot.size else None,
            "frac_opposite_sign": (float((np.sign(boot) != np.sign(pooled)).mean())
                                   if boot.size else None),
            "resampling_unit": "slide, then ducts within slide",
        },
        "within_slide_permutation": {
            "n_perm": int(null.size),
            "null_median": float(np.median(null)) if null.size else None,
            "null_p95_abs": float(np.percentile(np.abs(null), 95)) if null.size else None,
            "p_value": (float((np.abs(null) >= abs(pooled)).mean())
                        if null.size else None),
            "note": ("hole % shuffled WITHIN each slide, so slide-level brightness "
                     "is preserved and only the duct-to-duct pairing is broken."),
        },
    }


# ── TASK B: does patch assignment explain Task 3's weakness? ─────────────────

def task_b_exclusion_bias(df: pd.DataFrame, col: str) -> dict:
    with_p = df[df["n_patches"] > 0]
    without = df[df["n_patches"] == 0]

    def _block(g):
        if len(g) < MIN_DUCTS_PER_SLIDE:
            return {"n_ducts": int(len(g)), "rho": None,
                    "reason": "too few ducts to report"}
        return {
            "n_ducts": int(len(g)),
            "rho": _safe_rho(g["hole_pct"].values, g[col].values),
            "median_area_um2": float(np.nanmedian(g["area_um2"].values)),
            "median_hole_pct": float(np.nanmedian(g["hole_pct"].values)),
            "median_white_frac": float(np.nanmedian(g[col].values)),
        }

    a, b = _block(with_p), _block(without)
    return {
        "ducts_with_assigned_patches": a,
        "ducts_with_zero_patches": b,
        "all_ducts_rho": _safe_rho(df["hole_pct"].values, df[col].values),
        "interpretation": (
            "The zero-patch ducts are the population every earlier analysis had to "
            "drop. If their correlation matches the assigned ducts', the "
            "centre-in-polygon exclusion does not bias this relationship and cannot "
            "explain holeyroot_duct_checks Task 3's weak 2M-1 coupling. If it "
            "differs, the bias is real and is being measured here for the first "
            "time."
        ),
    }


# ── TASK C: calibration, not just ranking ────────────────────────────────────

def task_c_calibration(df: pd.DataFrame, col: str) -> dict:
    hole = df["hole_pct"].values.astype(float)
    white_pct = df[col].values.astype(float) * 100.0
    ok = np.isfinite(hole) & np.isfinite(white_pct)
    diff = white_pct[ok] - hole[ok]
    return {
        "hole_pct": {"median": float(np.median(hole[ok])),
                     "p10": float(np.percentile(hole[ok], 10)),
                     "p90": float(np.percentile(hole[ok], 90))},
        "measured_white_pct": {"median": float(np.median(white_pct[ok])),
                               "p10": float(np.percentile(white_pct[ok], 10)),
                               "p90": float(np.percentile(white_pct[ok], 90))},
        "difference_measured_minus_annotated": {
            "median": float(np.median(diff)),
            "p10": float(np.percentile(diff, 10)),
            "p90": float(np.percentile(diff, 90)),
        },
        "interpretation": (
            "Both quantities are percentages of duct area, so they are directly "
            "comparable. A high rho with a large systematic offset means the "
            "annotation RANKS ducts by holeyness correctly but does not measure "
            "the quantity it is named after — which is enough for a root rule "
            "(it only needs the ranking) but not for any statement about how "
            "holey a duct IS."
        ),
    }


# ── TASKS D and E ────────────────────────────────────────────────────────────

def task_d_threshold_sensitivity(df: pd.DataFrame, thresholds: list[int]) -> dict:
    out = {str(t): _safe_rho(df["hole_pct"].values, df[f"white_frac_{t}"].values)
           for t in thresholds}
    vals = np.array(list(out.values()), dtype=float)
    finite = vals[np.isfinite(vals)]
    return {
        "rho_by_threshold": out,
        "range": ([float(finite.min()), float(finite.max())]
                  if finite.size else None),
        "sign_stable": (bool(np.all(np.sign(finite) == np.sign(finite[0])))
                        if finite.size else None),
        "default_threshold": int(WHITE_THRESH),
        "note": ("220 is the convention inherited from inspect_roots_v3. A "
                 "conclusion that changes across this sweep is a conclusion about "
                 "the threshold."),
    }


def task_e_area_confound(df: pd.DataFrame, col: str) -> dict:
    hole = df["hole_pct"].values
    white = df[col].values
    area = df["area_um2"].values
    return {
        "rho_hole_area": _safe_rho(hole, area),
        "rho_white_area": _safe_rho(white, area),
        "partial_rho_hole_white_given_area": _partial_spearman(hole, white, area),
        "note": ("Larger ducts may simply contain more lumen, which would produce a "
                 "hole/white correlation with no annotation quality involved. The "
                 "partial is the version that survives that."),
    }


# ── verdict ──────────────────────────────────────────────────────────────────

def build_verdict(section: str, a: dict, b: dict, e: dict) -> dict:
    rho = a["pooled_rho"]
    ci = a["cluster_bootstrap"]["ci95"]
    p = a["within_slide_permutation"]["p_value"]
    partial = e["partial_rho_hole_white_given_area"]
    same = a["n_slides_same_sign_as_pooled"]
    n = a["n_slides_reported"]

    if not np.isfinite(rho):
        call = "UNCOMPUTABLE"
    elif abs(rho) >= STRONG_RHO and (ci is None or ci[0] * ci[1] > 0):
        call = "ANNOTATION TRACKS THE PIXELS"
    elif abs(rho) >= 0.25:
        call = "PARTIAL — the annotation ranks holeyness weakly"
    else:
        call = "ANNOTATION DOES NOT TRACK THE PIXELS"

    return {
        "section": section,
        "call": call,
        "pooled_rho": rho,
        "ci95": ci,
        "within_slide_permutation_p": p,
        "partial_given_area": partial,
        "slides_same_sign": f"{same}/{n}",
        # NOTE the parentheses. Written without them, `a + b + c if p is not None
        # else ""` binds as `(a + b + c) if ... else ""` and blanks the ENTIRE
        # statement whenever the permutation could not be computed, rather than
        # just omitting the p-value.
        "statement": (
            f"{section}: rho(hole %, measured white fraction) = {rho:+.4f}"
            + (f", 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")
            + (f", within-slide permutation p = {p:.4g}" if p is not None else "")
        ),
        "what_this_licenses": (
            "A strong correlation means the anchor variable has a real pixel "
            "referent and the earlier proxy-based weakness was an artifact of "
            "patch aggregation. It does NOT make the anchor a good one: "
            "anchor_area_control showed the realised root set is the bottom 1-3% "
            "of ducts by hole % and duct-size-extreme, and that its headline "
            "correlation is reproduced by size-matched anchors that ignore hole % "
            "entirely. A weak correlation, by contrast, invalidates the anchor in "
            "that section outright."
        ),
    }


# ── driver ───────────────────────────────────────────────────────────────────

def run_section(section: str, export: Path, ann_dir: Path, dims: Path,
                slide_list: Path, png_dir: Path, results_csv: Path | None,
                patch_size: int, thresholds: list[int], n_boot: int,
                n_perm: int, seed: int) -> dict:
    print("\n" + "=" * 78)
    print(f"  SECTION {section}")
    print("=" * 78)

    slides = load_slide_list(slide_list)
    meas = parse_measurement_export(export, slides)
    polys = load_duct_polygons(ann_dir, slides, load_slide_dimensions(dims))
    duct_table = build_duct_table(meas, polys)
    if len(duct_table) == 0:
        raise ValueError(
            "Empty duct table — no measurement row joined to a Tumor polygon by "
            "UUID. Check the export matches these slides and that --annotation-dir "
            "is the RATIO directory.")

    print("\n=== Rasterising duct polygons against the slide PNGs ===")
    df = measure_ducts(duct_table, png_dir, thresholds)

    # How many patches each duct has, under the SAME centre-in-polygon rule every
    # earlier analysis used — so Task B's split is the exact population split
    # those analyses were subject to.
    if results_csv is not None and Path(results_csv).exists():
        res = pd.read_csv(results_csv)
        assigned = assign_patches_to_ducts(res, duct_table, patch_size=patch_size)
        counts = (assigned[assigned["duct_id"].notna()]
                  .groupby("duct_id").size().to_dict())
        df["n_patches"] = [int(counts.get(o, 0)) for o in df["object_id"]]
    else:
        df["n_patches"] = -1
        print("  NOTE: no results.csv supplied — Task B (exclusion bias) is skipped "
              "rather than guessed at.")

    col = f"white_frac_{WHITE_THRESH}"
    a = task_a_primary(df, col, n_boot, n_perm, seed)
    print(f"\n  rho(hole %, white) = {a['pooled_rho']:+.4f}   "
          f"within-slide median {a['within_slide_median_rho']}   "
          f"same sign {a['n_slides_same_sign_as_pooled']}/{a['n_slides_reported']}")
    print(f"  cluster-bootstrap 95% CI: {a['cluster_bootstrap']['ci95']}")
    print(f"  within-slide permutation p = "
          f"{a['within_slide_permutation']['p_value']}")

    b = (task_b_exclusion_bias(df, col) if (df["n_patches"] >= 0).all()
         else {"skipped": "no results.csv supplied"})
    c = task_c_calibration(df, col)
    d = task_d_threshold_sensitivity(df, thresholds)
    e = task_e_area_confound(df, col)
    print(f"  partial given duct area = {e['partial_rho_hole_white_given_area']:+.4f}")

    return {
        "section": section,
        "n_ducts_in_table": int(len(duct_table)),
        "n_ducts_measured": int(len(df)),
        "n_bbox_clipped": int(df["bbox_clipped_at_image_edge"].sum()),
        "white_threshold": int(WHITE_THRESH),
        "task_a_primary": a,
        "task_b_exclusion_bias": b,
        "task_c_calibration": c,
        "task_d_threshold_sensitivity": d,
        "task_e_area_confound": e,
        "verdict": build_verdict(section, a, b, e),
        "per_duct": df.drop(columns=[c for c in df.columns
                                     if c.startswith("white_frac_")
                                     and c != col]).to_dict(orient="list"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--exports", nargs="+", type=Path, required=True,
                    help="One per section, SAME ORDER. 2M-1 and 2M-2 differ.")
    ap.add_argument("--slide-lists", nargs="+", type=Path, required=True)
    ap.add_argument("--results-csvs", nargs="*", type=Path, default=[],
                    help="Optional, one per section; enables Task B.")
    ap.add_argument("--annotation-dir", type=Path, required=True)
    ap.add_argument("--slide-dimensions", type=Path, required=True)
    ap.add_argument("--png-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = len(args.sections)
    for name, seq in (("--exports", args.exports), ("--slide-lists", args.slide_lists)):
        if len(seq) != n:
            raise SystemExit(
                f"{name} has {len(seq)} entries but --sections has {n}. These are "
                "positional; a mismatch would measure one section's ducts against "
                "another section's annotations.")
    if args.results_csvs and len(args.results_csvs) != n:
        raise SystemExit("--results-csvs, if given, must have one entry per section.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {"analysis": "duct_white_fraction",
           "config": {k: str(v) for k, v in vars(args).items()}, "sections": []}
    for i, sec in enumerate(args.sections):
        res["sections"].append(run_section(
            sec, args.exports[i], args.annotation_dir, args.slide_dimensions,
            args.slide_lists[i], args.png_dir,
            args.results_csvs[i] if args.results_csvs else None,
            args.patch_size, THRESHOLDS, args.n_boot, args.n_perm, args.seed))

    # Per-duct rows go to CSV, not into the JSON: ~2,000 ducts x 2 sections would
    # bury the summary the JSON exists to carry. Pop BEFORE serialising.
    for s in res["sections"]:
        pd.DataFrame(s.pop("per_duct")).to_csv(
            args.output_dir / f"duct_white_fraction_{s['section']}.csv", index=False)
    out = args.output_dir / "duct_white_fraction.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    write_report(res, args.output_dir / "duct_white_fraction.md")
    print(f"\nWrote {out}")


def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    add("# Does the expert hole % annotation track white space in the duct?\n")
    add("`holeyroot_duct_checks` Task 3 tested this through patch-derived proxies "
        "and got sign-flipped answers between sections (`h_intensity_wholepatch` "
        "+0.080 in 2M-1 vs -0.271 in 2M-2). This measures it directly: every Tumor "
        "polygon rasterised against its slide PNG, no patch assignment, every duct "
        f"included. White = mean RGB > {WHITE_THRESH}.\n")

    add("| section | ducts measured | rho(hole %, white) | 95% CI | perm p | "
        "within-slide | partial given area |")
    add("|---|---|---|---|---|---|---|")
    for s in res["sections"]:
        a, e = s["task_a_primary"], s["task_e_area_confound"]
        ci = a["cluster_bootstrap"]["ci95"]
        pv = a["within_slide_permutation"]["p_value"]
        add(f"| **{s['section']}** | {s['n_ducts_measured']}/{s['n_ducts_in_table']} | "
            f"**{a['pooled_rho']:+.4f}** | "
            + (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—") + " | "
            + (f"{pv:.4g}" if pv is not None else "—") + " | "
            + (f"{a['within_slide_median_rho']:+.4f} "
               f"({a['n_slides_same_sign_as_pooled']}/{a['n_slides_reported']})"
               if a["within_slide_median_rho"] is not None else "—") + " | "
            + f"{e['partial_rho_hole_white_given_area']:+.4f} |")

    for s in res["sections"]:
        add(f"\n## {s['section']}\n")
        add(f"**{s['verdict']['call']}**\n")

        b = s["task_b_exclusion_bias"]
        add("### Task B — does patch assignment explain the earlier weakness?\n")
        if "skipped" in b:
            add(f"Skipped: {b['skipped']}")
        else:
            for lbl, k in (("ducts WITH assigned patches", "ducts_with_assigned_patches"),
                           ("ducts with ZERO patches", "ducts_with_zero_patches")):
                r = b[k]
                add(f"- {lbl}: n={r['n_ducts']}, rho = "
                    + (f"{r['rho']:+.4f}" if r.get("rho") is not None
                       else f"not reported ({r.get('reason')})")
                    + (f", median area {r['median_area_um2']:.0f} um^2"
                       if r.get("median_area_um2") is not None else ""))
            add(f"\n> {b['interpretation']}")

        c = s["task_c_calibration"]
        add("\n### Task C — calibration\n")
        add(f"- annotated hole %: median {c['hole_pct']['median']:.2f} "
            f"(P10 {c['hole_pct']['p10']:.2f}, P90 {c['hole_pct']['p90']:.2f})")
        add(f"- measured white %: median {c['measured_white_pct']['median']:.2f} "
            f"(P10 {c['measured_white_pct']['p10']:.2f}, "
            f"P90 {c['measured_white_pct']['p90']:.2f})")
        add(f"- measured minus annotated: median "
            f"{c['difference_measured_minus_annotated']['median']:+.2f} points")
        add(f"\n> {c['interpretation']}")

        d = s["task_d_threshold_sensitivity"]
        add("\n### Task D — threshold sensitivity\n")
        add("| white threshold | rho |")
        add("|---|---|")
        for t, v in d["rho_by_threshold"].items():
            add(f"| {t}{' (default)' if int(t) == WHITE_THRESH else ''} | {v:+.4f} |")
        add(f"\nSign stable across the sweep: **{d['sign_stable']}**.")

        add(f"\n> {s['verdict']['what_this_licenses']}")

    add("\n## Limitations\n")
    add("- Masks use the polygon's OUTER RING only, so a duct's own lumen is not "
        "subtracted. That is deliberate: hole % is a fraction of duct area, and "
        "every earlier analysis used the same outer-ring regions.")
    add("- White is a fixed grey-level threshold on mean RGB. It cannot separate "
        "lumen from fat, from tears, or from slide background inside a polygon.")
    add("- No stain normalisation is applied, so a systematically paler slide "
        "reads as holier. The within-slide permutation null and the per-slide "
        "breakdown are what control for that; the pooled number does not.")
    add("- Eight slides is eight bootstrap clusters. The intervals are wide "
        "because of the design.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
