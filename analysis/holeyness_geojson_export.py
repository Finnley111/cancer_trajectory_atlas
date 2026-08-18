"""Convert section-2M-2 holey-ness GeoJSON into the TSV ``holeyness.py`` expects.

WHY A CONVERTER RATHER THAN A NEW CODE PATH
-------------------------------------------
2M-1's holey-ness arrived as a single tab-separated QuPath export
(``combined_matched_measurements.txt``). 2M-2's arrived as eight per-slide QuPath
GeoJSON files, with the measurements inside ``properties.measurements``.

``analysis/holeyness.py:parse_measurement_export`` reads a TSV with hardcoded
column names and cannot read GeoJSON. The options were to teach it a second
input format, or to convert the new format into the one it already reads. This
module does the latter, deliberately: 2M-2 then flows through **byte-identical
analysis code** to 2M-1, which is exactly what a cross-section comparison
requires. ``holeyness.py`` is not modified.

⚠ COLUMN RENAME — READ THIS
---------------------------
2M-1 is Carnoy's-fixed and its measurements are prefixed ``holes_carnoys:``.
2M-2 is PFA-fixed and its measurements are prefixed ``holes_pfa:``. The prefix is
a hardcoded constant in ``holeyness.py`` (``COL_HOLE_PCT``), so this converter
must emit the ``holes_carnoys:`` spelling for the file to be readable.

**The emitted column name therefore misstates the fixative.** The numbers are
2M-2's PFA measurements; only the header is renamed. That is flagged four ways:
the output filename carries it, a ``.provenance.json`` sidecar records the true
source keys, this module prints a banner, and the Phase 1 report repeats it.

FIXATION IS A REAL CONFOUND, NOT JUST A HEADER
----------------------------------------------
Carnoy's and PFA differ in shrinkage behaviour, so ``hole %`` distributions may
differ systematically between the two sections for fixation reasons alone,
independent of biology. Irrelevant within a section; live for any cross-section
claim. Stated here so it travels with the data.

WHAT IS AND IS NOT TAKEN FROM THE GEOJSON
-----------------------------------------
Taken:      object UUID, ``Area µm^2``, ``holes_pfa: hole %``,
            ``holes_pfa: hole area µm^2``, and the geometry — the last used ONLY
            to derive centroids (see below).
NOT taken:  the duct polygons used for patch assignment. Those still come from
            ``data/annotations_ratio/`` via ``holeyness.load_duct_polygons``,
            exactly as for 2M-1. Verified: all 1776 Tumor UUIDs in the GeoJSONs
            match a Tumor polygon in ``annotations_ratio``.

CENTROIDS ARE DERIVED, NOT FABRICATED
-------------------------------------
The GeoJSON has no centroid columns, but ``parse_measurement_export`` requires
them. For each duct the polygon area in px² is computed from the geometry and
divided into the recorded ``Area µm^2``, giving a µm/px scale — measured at
0.441306 with CV 0.00% across a slide. The scale is derived PER SLIDE from the
data itself, never hardcoded, and the run aborts if it is not consistent.

Centroids are metadata only: they are written to the per-duct CSV and never
enter any correlation, partial correlation, or permutation test. Verified by
grepping every use of ``centroid_x_um`` / ``centroid_y_um`` in the holeyness
modules.

FAILS LOUDLY ON
---------------
  * any Tumor duct with no ``hole %`` measurement
  * duplicate object UUIDs within a slide
  * a µm/px scale whose coefficient of variation exceeds --scale-cv-tol
  * an empty result for any slide

Read-only on its inputs. Writes the TSV and its provenance sidecar only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The spelling holeyness.py hardcodes (analysis/holeyness.py COL_* constants).
OUT_COLS = ["Image", "Object ID", "Classification",
            "Centroid X µm", "Centroid Y µm", "Area µm^2",
            "holes_carnoys: hole %", "holes_carnoys: hole area µm^2"]

SRC_HOLE_PCT_DEFAULT = "holes_pfa: hole %"
SRC_HOLE_AREA_DEFAULT = "holes_pfa: hole area µm^2"
SRC_AREA = "Area µm^2"

BANNER = """
================================================================================
 COLUMN RENAME IN EFFECT
   source '{pct}'  ->  emitted 'holes_carnoys: hole %'
   source '{area}'  ->  emitted 'holes_carnoys: hole area um^2'

 The emitted header MISSTATES THE FIXATIVE. These are 2M-2 PFA measurements;
 only the column name is renamed, because holeyness.py hardcodes the Carnoy's
 spelling. Values are untouched.

 Carnoy's and PFA differ in shrinkage, so hole % distributions may differ
 between sections for fixation reasons alone. Irrelevant within a section;
 a live confound for any cross-section claim.
================================================================================
"""


def _ring_area(coords) -> float:
    a = np.asarray(coords, dtype=float)
    if a.ndim != 2 or a.shape[0] < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_area_px(geom: dict) -> float:
    """Shoelace area with inner rings subtracted. Handles Polygon and MultiPolygon."""
    t = geom.get("type")
    if t == "Polygon":
        rings = geom["coordinates"]
        return _ring_area(rings[0]) - sum(_ring_area(r) for r in rings[1:])
    if t == "MultiPolygon":
        total = 0.0
        for poly in geom["coordinates"]:
            total += _ring_area(poly[0]) - sum(_ring_area(r) for r in poly[1:])
        return total
    return 0.0


def _outer_coords(geom: dict) -> np.ndarray:
    t = geom.get("type")
    if t == "Polygon":
        return np.asarray(geom["coordinates"][0], dtype=float)
    if t == "MultiPolygon":
        # Largest constituent polygon's outer ring — the duct's main body.
        best, best_a = None, -1.0
        for poly in geom["coordinates"]:
            a = _ring_area(poly[0])
            if a > best_a:
                best_a, best = a, poly[0]
        return np.asarray(best, dtype=float)
    return np.empty((0, 2))


def convert_slide(path: Path, src_pct: str, src_hole_area: str,
                  scale_cv_tol: float,
                  degenerate_area_um2: float = 100.0) -> tuple[list[dict], dict]:
    """Return (rows, diagnostics) for one slide's GeoJSON.

    ``src_pct`` / ``src_hole_area`` are the HOLE measurement keys. The DUCT area
    key is the QuPath built-in ``Area µm^2`` (module constant ``SRC_AREA``) and
    is deliberately not configurable — conflating the two would silently derive
    the µm/px scale from hole area instead of duct area.
    """
    stem = path.stem
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"] if isinstance(data, dict) and "features" in data else data

    tumor = []
    for feat in feats:
        props = feat.get("properties", {}) or {}
        cls = props.get("classification") or {}
        name = cls.get("name") if isinstance(cls, dict) else str(cls)
        if name == "Tumor":
            tumor.append(feat)

    if not tumor:
        raise ValueError(f"{stem}: no Tumor features found.")

    missing = [f.get("id") for f in tumor
               if src_pct not in ((f.get("properties", {}) or {}).get("measurements") or {})]
    if missing:
        raise ValueError(
            f"{stem}: {len(missing)} Tumor duct(s) have no '{src_pct}' measurement "
            f"(e.g. {missing[:3]}). Refusing to emit a partial table — a duct with "
            "no holey-ness must not silently become a duct with zero holey-ness.")

    # QuPath writes the JSON string "NaN" when a measurement is undefined, which
    # happens for degenerate sub-micron polygons (stray clicks / stray vertices).
    # The key is present, so the check above does not catch it. These are carried
    # through unchanged and dropped downstream by build_duct_table's notna filter
    # — the point here is that the drop is COUNTED and reported, not silent.
    # A systematic failure still errors, via --max-nonnumeric-frac.
    nonnumeric = []
    for f in tumor:
        v = f["properties"]["measurements"].get(src_pct)
        if not isinstance(v, (int, float)) or v != v:
            nonnumeric.append({"object_id": f.get("id"), "value": repr(v),
                               "area_um2": f["properties"]["measurements"].get(SRC_AREA)})

    missing_area = [f.get("id") for f in tumor
                    if SRC_AREA not in ((f.get("properties", {}) or {}).get("measurements") or {})]
    if missing_area:
        raise ValueError(
            f"{stem}: {len(missing_area)} Tumor duct(s) lack '{SRC_AREA}'. Duct area "
            "is the mediator variable in this analysis and cannot be imputed.")

    # Degenerate annotations: a 112px patch is ~2440 um^2 at this resolution, so a
    # duct of a few um^2 is orders of magnitude smaller than a single patch and can
    # never receive one under centre-in-polygon. Reported, not filtered — filtering
    # here would silently change the analysis population; the existing zero-patch
    # exclusion already removes them, and that exclusion is itself reported.
    degenerate = [{"object_id": f.get("id"),
                   "area_um2": f["properties"]["measurements"].get(SRC_AREA),
                   "hole_pct": repr(f["properties"]["measurements"].get(src_pct))}
                  for f in tumor
                  if isinstance(f["properties"]["measurements"].get(SRC_AREA), (int, float))
                  and f["properties"]["measurements"][SRC_AREA] < degenerate_area_um2]

    # Per-slide um/px, derived from DUCT area (not hole area) rather than assumed.
    # Degenerate polygons are excluded from the estimate: their px areas are so
    # small that rounding dominates the ratio.
    scales = []
    for f in tumor:
        m = f["properties"]["measurements"]
        a_px = _polygon_area_px(f["geometry"])
        a_um = m.get(SRC_AREA, 0)
        if a_px > 0 and a_um and a_um >= degenerate_area_um2:
            scales.append((a_um / a_px) ** 0.5)
    if not scales:
        raise ValueError(f"{stem}: could not derive a um/px scale from any duct.")
    s = np.asarray(scales, dtype=float)
    cv = float(s.std() / s.mean()) if s.mean() else float("inf")
    if cv > scale_cv_tol:
        raise ValueError(
            f"{stem}: um/px scale is inconsistent across ducts (CV {cv:.4%} > "
            f"tolerance {scale_cv_tol:.4%}; range [{s.min():.6f}, {s.max():.6f}]). "
            "Centroids cannot be converted reliably. Investigate before proceeding.")
    scale = float(np.median(s))

    rows, seen = [], set()
    for f in tumor:
        uid = f.get("id")
        if uid is None:
            raise ValueError(f"{stem}: a Tumor feature has no 'id' (UUID); the join "
                             "to annotations_ratio would be impossible.")
        if uid in seen:
            raise ValueError(f"{stem}: duplicate object UUID {uid}.")
        seen.add(uid)
        m = f["properties"]["measurements"]
        ring = _outer_coords(f["geometry"])
        cx_px, cy_px = (ring[:, 0].mean(), ring[:, 1].mean()) if ring.size else (np.nan, np.nan)
        rows.append({
            "Image": f"{stem}.ndpi",          # parse_measurement_export strips .ndpi, adds _x5
            "Object ID": uid,
            "Classification": "Tumor",
            "Centroid X µm": cx_px * scale,
            "Centroid Y µm": cy_px * scale,
            "Area µm^2": m[SRC_AREA],
            "holes_carnoys: hole %": m[src_pct],
            "holes_carnoys: hole area µm^2": m.get(src_hole_area),
        })

    geom_types = {}
    for f in tumor:
        t = f["geometry"]["type"]
        geom_types[t] = geom_types.get(t, 0) + 1

    diag = {
        "slide": stem,
        "n_tumor": len(tumor),
        "n_emitted": len(rows),
        "um_per_px_median": scale,
        "um_per_px_cv": cv,
        "geometry_types": geom_types,
        "n_multipolygon": geom_types.get("MultiPolygon", 0),
        "n_nonnumeric_hole_pct": len(nonnumeric),
        "nonnumeric_hole_pct": nonnumeric,
        "n_degenerate_area": len(degenerate),
        "degenerate_ducts": degenerate,
    }
    return rows, diag


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geojson-dir", type=Path, required=True,
                    help="Directory of per-slide holey-ness GeoJSON (e.g. holeyness_section_2/).")
    ap.add_argument("--output", type=Path, required=True,
                    help="TSV to write. Give it a name that carries the rename warning.")
    # NB: argparse %%-formats help strings, and these defaults contain a literal
    # '%', so the default must not be interpolated into the help text.
    ap.add_argument("--src-hole-pct", default=SRC_HOLE_PCT_DEFAULT,
                    help="Source measurement key for hole percentage "
                         "(default: 'holes_pfa: hole %%').")
    ap.add_argument("--src-hole-area", default=SRC_HOLE_AREA_DEFAULT,
                    help="Source measurement key for hole area.")
    ap.add_argument("--scale-cv-tol", type=float, default=0.001,
                    help="Max coefficient of variation for the derived um/px scale "
                         "within a slide before aborting. (default: 0.001 = 0.1%%)")
    ap.add_argument("--degenerate-area-um2", type=float, default=100.0,
                    help="Ducts smaller than this are REPORTED as degenerate "
                         "annotation artifacts and excluded from the um/px scale "
                         "estimate. They are NOT filtered from the output: that "
                         "would silently change the analysis population, and the "
                         "existing zero-patch exclusion already removes them. For "
                         "scale, a 112px patch is ~2440 um^2. (default: 100.0)")
    ap.add_argument("--max-nonnumeric-frac", type=float, default=0.01,
                    help="Abort if more than this fraction of ducts have a "
                         "non-numeric hole %%. A handful of sub-micron artifacts is "
                         "expected; a systematic failure is not. (default: 0.01)")
    args = ap.parse_args()

    print(BANNER.format(pct=args.src_hole_pct, area=args.src_hole_area))

    if not args.geojson_dir.is_dir():
        sys.exit(f"ERROR: --geojson-dir not found: {args.geojson_dir}")
    if args.output.exists():
        sys.exit(f"ERROR: {args.output} exists; refusing to overwrite.")

    files = sorted(args.geojson_dir.glob("*.geojson"))
    if not files:
        sys.exit(f"ERROR: no .geojson files in {args.geojson_dir}")

    print(f"Converting {len(files)} slide(s) from {args.geojson_dir}\n")
    all_rows, diags = [], []
    for p in files:
        rows, diag = convert_slide(p, args.src_hole_pct, args.src_hole_area,
                                   args.scale_cv_tol, args.degenerate_area_um2)
        all_rows.extend(rows)
        diags.append(diag)
        print(f"  {diag['slide']:24s} {diag['n_emitted']:5d} Tumor ducts   "
              f"um/px {diag['um_per_px_median']:.6f} (CV {diag['um_per_px_cv']:.2%})"
              + (f"   [{diag['n_multipolygon']} MultiPolygon]" if diag["n_multipolygon"] else ""))

    df = pd.DataFrame(all_rows, columns=OUT_COLS)

    # ── Data-quality report: loud, counted, and non-fatal for a few artifacts ──
    n_nn = sum(d["n_nonnumeric_hole_pct"] for d in diags)
    n_dg = sum(d["n_degenerate_area"] for d in diags)
    if n_nn or n_dg:
        print("\n" + "-" * 78)
        print(" DATA QUALITY")
    if n_nn:
        frac = n_nn / len(df)
        print(f"  {n_nn}/{len(df)} duct(s) ({frac:.2%}) have a NON-NUMERIC hole % "
              "(QuPath wrote \"NaN\").")
        print("  They are carried through unchanged and dropped downstream by "
              "build_duct_table's\n  notna filter — counted here so the drop is not silent:")
        for d in diags:
            for r in d["nonnumeric_hole_pct"]:
                print(f"    {d['slide']:22s} {str(r['object_id'])[:8]}  "
                      f"hole%={r['value']:8s} area={r['area_um2']:.4f} um^2")
        if frac > args.max_nonnumeric_frac:
            sys.exit(f"\nERROR: non-numeric fraction {frac:.2%} exceeds "
                     f"--max-nonnumeric-frac {args.max_nonnumeric_frac:.2%}. That is a "
                     "systematic export problem, not a few stray annotations. Aborting.")
    if n_dg:
        print(f"\n  {n_dg}/{len(df)} duct(s) are below "
              f"{args.degenerate_area_um2:g} um^2 — degenerate annotation artifacts "
              "(a 112px patch\n  is ~2440 um^2, so these are orders of magnitude "
              "smaller than one patch and can\n  never receive a patch centre). NOT "
              "filtered here; the existing zero-patch\n  exclusion removes them and "
              "reports the count.")
        for d in diags:
            for r in d["degenerate_ducts"]:
                print(f"    {d['slide']:22s} {str(r['object_id'])[:8]}  "
                      f"area={r['area_um2']:10.4f} um^2  hole%={r['hole_pct']}")
    if n_nn or n_dg:
        print("-" * 78)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)

    prov = {
        "WARNING": ("The 'holes_carnoys:' columns in this file contain PFA "
                    "measurements from section 2M-2. The header was renamed so "
                    "analysis/holeyness.py, which hardcodes the Carnoy's spelling, "
                    "can read it. Values are unmodified."),
        "source_dir": str(args.geojson_dir),
        "source_files": [p.name for p in files],
        "column_rename": {
            args.src_hole_pct: "holes_carnoys: hole %",
            args.src_hole_area: "holes_carnoys: hole area µm^2",
        },
        "fixative_note": ("2M-1 = Carnoy's, 2M-2 = PFA. The two fixatives differ in "
                          "shrinkage behaviour, so hole %% distributions may differ "
                          "between sections for fixation reasons alone, independent "
                          "of biology. Irrelevant within a section; a live confound "
                          "for any cross-section claim."),
        "centroids": ("NOT present in the GeoJSON. Derived per slide as polygon "
                      "area_um2 / area_px2 -> um/px, then applied to the outer-ring "
                      "mean. Metadata only: never used in any correlation, partial "
                      "correlation, or permutation test."),
        "polygons_for_patch_assignment": ("NOT taken from these GeoJSONs. Still read "
                                          "from data/annotations_ratio/ by "
                                          "holeyness.load_duct_polygons, as for 2M-1."),
        "n_ducts_emitted": int(len(df)),
        "data_quality": {
            "n_nonnumeric_hole_pct": int(sum(d["n_nonnumeric_hole_pct"] for d in diags)),
            "n_degenerate_area": int(sum(d["n_degenerate_area"] for d in diags)),
            "degenerate_area_threshold_um2": args.degenerate_area_um2,
            "note": ("Non-numeric hole % ducts are dropped downstream by "
                     "build_duct_table's notna filter. Degenerate-area ducts are "
                     "kept in this file and removed by the existing zero-patch "
                     "exclusion, which reports its own count. Neither is filtered "
                     "here, so the analysis population is not silently changed."),
        },
        "per_slide": diags,
    }
    prov_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    n_mp = sum(d["n_multipolygon"] for d in diags)
    print(f"\nWrote {len(df)} duct rows -> {args.output}")
    print(f"Provenance             -> {prov_path}")
    if n_mp:
        print(f"\nNOTE: {n_mp} of {len(df)} ducts are MultiPolygon. "
              "holeyness.load_duct_polygons accepts Polygon only, so these will be "
              "counted as 'bad geometry' and excluded downstream. Expected and small; "
              "reported so the analysis population is stated accurately.")
    print(BANNER.format(pct=args.src_hole_pct, area=args.src_hole_area))


if __name__ == "__main__":
    main()
