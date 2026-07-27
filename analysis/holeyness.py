"""
Duct-level holeyness validation: correlate per-duct pseudotime against
QuPath-measured hole fraction.

Assignment strategy: cross-file UUID join.
  - data/annotations_ratio/<slide>.json  carries Tumor polygon geometry
    in ratio coords [0,1], each feature keyed by its QuPath UUID.
  - combined_matched_measurements.txt    carries Tumor-level hole % and
    hole area, also keyed by the same QuPath UUID (Object ID column).
  Joining on UUID gives polygon + hole measurements for each duct.
  Patch centre containment (matplotlib Path) assigns patches to ducts.

Reads: results.csv, combined measurement export, ratio annotation JSON,
       slide_dimensions.json.
Writes: holeyness_per_duct.csv, holeyness_validation.json, two scatter
        figures (pdf + png).  Does NOT modify any existing pipeline output.

CLI
---
  python -m cancer_trajectory_atlas.analysis.holeyness \\
      --section        2M-1 \\
      --export         $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt \\
      --annotation-dir ~/cancer_trajectory_atlas/data/annotations_ratio \\
      --slide-dimensions $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json \\
      --results        $SCRATCH/results/per_section/atlas_2M-1/results.csv \\
      --output-dir     $SCRATCH/results/holeyness/2M-1 \\
      --slide-list     ~/cancer_trajectory_atlas/jobs/slides_section1.txt

v2 (extended, area-adjusted validation): pass --v2 and --v1-per-duct-csv (pointing
at the v1 run's holeyness_per_duct.csv) to additionally run duct-area covariate
checks, within-slide/nested permutation tests, an exclusion-bias check on
zero-patch ducts, aggregation sensitivity, and a patch-sampling artifact check.
Writes holeyness_validation_v2.json / .md / duct_table_full.csv / 3 figures to
--output-dir, which should point at a NEW versioned subdirectory (e.g.
.../holeyness/2M-1/v2_area_adjusted/) so the v1 outputs are never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.stats import spearmanr, rankdata, mannwhitneyu


# ── Constants ────────────────────────────────────────────────────────────────

PATCH_SIZE_DEFAULT = 112
FEATURES_TO_AGGREGATE = ["pseudotime", "nuclear_density", "packing_irregularity"]

# Columns expected in results.csv
REQUIRED_RESULTS_COLS = ["x", "y", "slide_name", "pseudotime",
                          "nuclear_density", "packing_irregularity"]

# Column names in the measurement export (tab-separated)
COL_IMAGE      = "Image"
COL_OBJECT_ID  = "Object ID"
COL_CLASS      = "Classification"
COL_CENT_X     = "Centroid X µm"
COL_CENT_Y     = "Centroid Y µm"
COL_AREA       = "Area µm^2"
COL_HOLE_PCT   = "holes_carnoys: hole %"
COL_HOLE_AREA  = "holes_carnoys: hole area µm^2"

REQUIRED_MEAS_COLS = [COL_IMAGE, COL_OBJECT_ID, COL_CLASS,
                       COL_CENT_X, COL_CENT_Y, COL_AREA,
                       COL_HOLE_PCT, COL_HOLE_AREA]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_slide_list(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def load_slide_dimensions(path: Path) -> dict[str, dict]:
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for key, dims in raw.items():
        slide_name = Path(key).stem  # "6027-4L-2M-1_x5.png" → "6027-4L-2M-1_x5"
        result[slide_name] = dims
    return result


def parse_measurement_export(
    export_path: Path,
    pipeline_slides: list[str],
) -> pd.DataFrame:
    """
    Read the combined QuPath measurement export.
    Returns Tumor-only rows for pipeline slides, with renamed columns.
    """
    print("\n=== Measurement export ===")
    df = pd.read_csv(export_path, sep="\t", dtype=str, low_memory=False)

    missing = [c for c in REQUIRED_MEAS_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: measurement export missing columns: {missing}")

    # Drop any repeated header rows (in case sub-files were concatenated)
    df = df[df[COL_IMAGE] != COL_IMAGE].copy()

    n_raw = len(df)
    df = df[df[COL_CLASS] == "Tumor"].copy()
    print(f"  Total rows: {n_raw}  →  Tumor rows: {len(df)}")

    # Build slide_name: "6027-4L-2M-1.ndpi" → "6027-4L-2M-1_x5"
    df["slide_name"] = (
        df[COL_IMAGE].str.replace(r"\.ndpi$", "", regex=True) + "_x5"
    )

    pipeline_set = set(pipeline_slides)
    mask_in = df["slide_name"].isin(pipeline_set)
    dropped = df.loc[~mask_in, "slide_name"].unique().tolist()
    n_dropped = (~mask_in).sum()
    print(f"  Dropped {n_dropped} rows from {len(dropped)} non-pipeline slides:")
    for s in sorted(dropped):
        print(f"    {s}")
    df = df[mask_in].copy()
    print(f"  Pipeline Tumor rows: {len(df)} across {df['slide_name'].nunique()} slides")

    missing_slides = [s for s in pipeline_slides if s not in df["slide_name"].values]
    if missing_slides:
        print(f"  WARNING: {len(missing_slides)} pipeline slides have no measurement rows:")
        for s in missing_slides:
            print(f"    {s}")

    df = df.rename(columns={
        COL_OBJECT_ID: "object_id",
        COL_CENT_X:    "centroid_x_um",
        COL_CENT_Y:    "centroid_y_um",
        COL_AREA:      "area_um2",
        COL_HOLE_PCT:  "hole_pct",
        COL_HOLE_AREA: "hole_area_um2",
    })
    for col in ["centroid_x_um", "centroid_y_um", "area_um2", "hole_pct", "hole_area_um2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["slide_name", "object_id", "centroid_x_um", "centroid_y_um",
            "area_um2", "hole_pct", "hole_area_um2"]
    return df[keep].copy().reset_index(drop=True)


def load_duct_polygons(
    annotation_dir: Path,
    pipeline_slides: list[str],
    slide_dims: dict[str, dict],
) -> dict[str, dict]:
    """
    Load Tumor polygon geometry from ratio-coordinate annotation JSON files.
    Returns dict: object_uuid → {"polygon": MplPath, "slide_name": str}
    """
    print("\n=== Annotation polygons ===")
    polygons: dict[str, dict] = {}

    for slide_name in pipeline_slides:
        stem = slide_name.replace("_x5", "")
        json_path = annotation_dir / f"{stem}.json"
        if not json_path.exists():
            print(f"  WARNING: annotation file not found: {json_path}")
            continue

        dims = slide_dims.get(slide_name)
        if dims is None:
            print(f"  WARNING: no slide_dimensions entry for {slide_name}")
            continue

        full_w    = dims["original_full_width"]
        full_h    = dims["original_full_height"]
        cropped_w = dims["cropped_width"]

        with open(json_path) as f:
            data = json.load(f)

        features = (
            data["features"]
            if data.get("type") == "FeatureCollection"
            else (data if isinstance(data, list) else [data])
        )

        n_loaded = n_right = n_bad_geom = 0
        for feat in features:
            props   = feat.get("properties", {})
            cls_obj = props.get("classification", {})
            cls_name = (
                cls_obj.get("name", "")
                if isinstance(cls_obj, dict)
                else str(cls_obj)
            )
            if cls_name != "Tumor":
                continue

            uuid = feat.get("id")
            if uuid is None:
                continue

            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                n_bad_geom += 1
                continue

            coords = geom["coordinates"][0]  # outer ring
            arr    = np.asarray(coords, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 3:
                n_bad_geom += 1
                continue

            arr[:, 0] *= full_w  # ratio → pipeline x pixel
            arr[:, 1] *= full_h  # ratio → pipeline y pixel

            # Exclude right-half (duplicate slide copy)
            cx = arr[:, 0].mean()
            if cx > cropped_w:
                n_right += 1
                continue

            polygons[uuid] = {"polygon": MplPath(arr), "slide_name": slide_name}
            n_loaded += 1

        print(
            f"  {slide_name}: {n_loaded} Tumor polygons "
            f"({n_right} right-half excluded"
            + (f", {n_bad_geom} bad geometry" if n_bad_geom else "")
            + ")"
        )

    print(f"  Total Tumor polygons loaded: {len(polygons)}")
    return polygons


# ── Duct table construction ───────────────────────────────────────────────────

def build_duct_table(measurements: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """
    Join measurement rows with polygon geometry on object_id (UUID).
    Rows with no matching polygon are dropped with a logged count.
    """
    print("\n=== UUID join (measurement ↔ polygon) ===")
    rows = []
    n_no_poly = 0
    for _, row in measurements.iterrows():
        uid = row["object_id"]
        if uid not in polygons:
            n_no_poly += 1
            continue
        entry = {
            "object_id":    uid,
            "slide_name":   row["slide_name"],
            "hole_pct":     row["hole_pct"],
            "hole_area_um2": row["hole_area_um2"],
            "area_um2":     row["area_um2"],
            "centroid_x_um": row["centroid_x_um"],
            "centroid_y_um": row["centroid_y_um"],
            "polygon":      polygons[uid]["polygon"],
        }
        rows.append(entry)

    if n_no_poly:
        print(f"  WARNING: {n_no_poly} measurement rows have no matching polygon "
              f"(UUID absent from annotation JSON) — excluded")

    # Also log polygons with no measurement
    meas_uuids = set(measurements["object_id"])
    n_poly_no_meas = sum(1 for uid in polygons if uid not in meas_uuids)
    if n_poly_no_meas:
        print(f"  NOTE: {n_poly_no_meas} annotation polygons have no measurement row "
              f"(Ignore* or child objects — expected)")

    duct_table = pd.DataFrame(rows)
    duct_table = duct_table[pd.notna(duct_table["hole_pct"])].copy()
    print(f"  Ducts in final table: {len(duct_table)}")
    return duct_table


# ── Patch-to-duct assignment ──────────────────────────────────────────────────

def assign_patches_to_ducts(
    results_df: pd.DataFrame,
    duct_table: pd.DataFrame,
    patch_size: int = PATCH_SIZE_DEFAULT,
) -> pd.DataFrame:
    """
    For each patch, test whether its centre lies inside any duct polygon.
    Returns results_df with an added 'duct_id' column (None if unassigned).
    """
    print("\n=== Patch-to-duct assignment ===")
    results_df = results_df.copy()
    results_df["duct_id"] = None

    half = patch_size / 2.0

    for slide_name, slide_df in results_df.groupby("slide_name"):
        slide_ducts = duct_table[duct_table["slide_name"] == slide_name].reset_index(drop=True)
        if len(slide_ducts) == 0:
            print(f"  {slide_name}: 0 ducts — {len(slide_df)} patches left unassigned")
            continue

        cx = (slide_df["x"].values + half)
        cy = (slide_df["y"].values + half)
        points = np.column_stack([cx, cy])

        assigned = np.full(len(slide_df), None, dtype=object)
        for _, duct_row in slide_ducts.iterrows():
            inside = duct_row["polygon"].contains_points(points)
            assigned[inside] = duct_row["object_id"]

        results_df.loc[slide_df.index, "duct_id"] = assigned

        n_assigned = np.sum(assigned != None)  # noqa: E711
        print(
            f"  {slide_name}: {n_assigned}/{len(slide_df)} patches assigned "
            f"({100*n_assigned/len(slide_df):.1f}%)  |  {len(slide_ducts)} ducts"
        )

    total_assigned = results_df["duct_id"].notna().sum()
    total = len(results_df)
    print(
        f"  Overall: {total_assigned}/{total} patches assigned "
        f"({100*total_assigned/total:.1f}%)"
    )
    return results_df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_per_duct(
    results_df: pd.DataFrame,
    duct_table: pd.DataFrame,
    agg_fn,
    agg_label: str,
) -> pd.DataFrame:
    """
    Group assigned patches by duct_id; aggregate pseudotime, nuclear_density,
    packing_irregularity using agg_fn (e.g. np.nanmedian or np.nanmean).
    Returns per-duct DataFrame joined with duct metadata.
    """
    print(f"\n=== Per-duct aggregation (agg={agg_label}) ===")
    assigned = results_df[results_df["duct_id"].notna()].copy()

    agg_rows = []
    for duct_id, grp in assigned.groupby("duct_id"):
        row = {"object_id": duct_id, "n_patches": len(grp)}
        for feat in FEATURES_TO_AGGREGATE:
            row[feat] = float(agg_fn(grp[feat].values))
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    meta_cols = ["object_id", "slide_name", "hole_pct", "hole_area_um2",
                 "area_um2", "centroid_x_um", "centroid_y_um"]
    merged = agg_df.merge(
        duct_table[meta_cols],
        on="object_id",
        how="left",
    )

    n_total_ducts = len(duct_table)
    n_matched     = len(merged)
    n_zero        = n_total_ducts - n_matched
    print(f"  Ducts with ≥1 patch : {n_matched}")
    print(f"  Ducts with 0 patches: {n_zero}  (excluded from correlations)")
    print(f"  Median patches/duct : {merged['n_patches'].median():.0f}")
    return merged


# ── Statistics ────────────────────────────────────────────────────────────────

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    result = spearmanr(a[mask], b[mask])
    return float(result.statistic), float(result.pvalue)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Algebraic partial Spearman: rho(x, y | z). Same formula as cellularity_confound.py."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 10:
        return float("nan")
    x, y, z = x[valid], y[valid], z[valid]
    rho_xy = float(spearmanr(x, y).statistic)
    rho_xz = float(spearmanr(x, z).statistic)
    rho_yz = float(spearmanr(y, z).statistic)
    denom  = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
    if denom < 1e-10:
        return float("nan")
    return float((rho_xy - rho_xz * rho_yz) / denom)


def run_correlations(per_duct: pd.DataFrame) -> dict:
    """Compute all Spearman and partial Spearman values; no permutation here."""
    pt   = per_duct["pseudotime"].values
    hole = per_duct["hole_pct"].values
    hole_area = per_duct["hole_area_um2"].values
    nd   = per_duct["nuclear_density"].values
    pi   = per_duct["packing_irregularity"].values

    rho_pt_hole,      p_pt_hole      = _safe_spearman(pt, hole)
    rho_pt_hole_area, p_pt_hole_area = _safe_spearman(pt, hole_area)
    rho_nd_hole,      p_nd_hole      = _safe_spearman(nd, hole)
    rho_pi_hole,      p_pi_hole      = _safe_spearman(pi, hole)
    partial_rho = _partial_spearman(pt, hole, nd)

    return {
        "rho_pt_hole_pct":           rho_pt_hole,
        "p_pt_hole_pct_scipy":       p_pt_hole,
        "rho_pt_hole_area_um2":      rho_pt_hole_area,
        "p_pt_hole_area_um2_scipy":  p_pt_hole_area,
        "rho_nd_hole_pct":           rho_nd_hole,
        "p_nd_hole_pct_scipy":       p_nd_hole,
        "rho_pi_hole_pct":           rho_pi_hole,
        "p_pi_hole_pct_scipy":       p_pi_hole,
        "partial_rho_pt_hole_given_nd": partial_rho,
    }


def run_permutation_test(
    per_duct: pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """
    Shuffle duct-level pseudotime across ducts (1000×).
    Two-tailed p-value: fraction of |null| >= |observed|.
    """
    print(f"\n=== Permutation test ({n_permutations} shuffles) ===")
    pt   = per_duct["pseudotime"].values.copy()
    hole = per_duct["hole_pct"].values
    nd   = per_duct["nuclear_density"].values

    obs_rho     = float(spearmanr(pt, hole).statistic)
    obs_partial = _partial_spearman(pt, hole, nd)

    null_rho     = np.empty(n_permutations)
    null_partial = np.empty(n_permutations)

    for i in range(n_permutations):
        pt_perm = rng.permutation(pt)
        null_rho[i]     = abs(float(spearmanr(pt_perm, hole).statistic))
        null_partial[i] = abs(_partial_spearman(pt_perm, hole, nd))

    perm_p_rho     = float(np.mean(null_rho     >= abs(obs_rho)))
    perm_p_partial = float(np.mean(null_partial >= abs(obs_partial)))
    null95_rho     = float(np.percentile(null_rho, 95))
    null95_partial = float(np.percentile(null_partial, 95))

    print(f"  rho(PT, hole%) = {obs_rho:.4f}  perm_p = {perm_p_rho:.4f}  null95 = {null95_rho:.4f}")
    print(f"  partial_rho    = {obs_partial:.4f}  perm_p = {perm_p_partial:.4f}  null95 = {null95_partial:.4f}")

    return {
        "perm_p_rho_pt_hole_pct":              perm_p_rho,
        "null95_rho_pt_hole_pct":              null95_rho,
        "perm_p_partial_rho_pt_hole_given_nd": perm_p_partial,
        "null95_partial_rho_pt_hole_given_nd": null95_partial,
    }


# ── v2: extended, area-adjusted validation checks ────────────────────────────

WITHIN_SLIDE_MIN_DUCTS_DEFAULT = 10
N_PATCH_THRESHOLDS = (3, 5, 10, 20)


def _format_perm_p(p: float, n_permutations: int) -> str:
    """Report permutation p-values as '< 1/n' rather than 0.0 when no shuffle
    exceeded the observed statistic."""
    if not np.isfinite(p):
        return "nan"
    return f"< {1.0 / n_permutations:.4g}" if p == 0.0 else f"{p:.4f}"


def _partial_spearman_multi(x: np.ndarray, y: np.ndarray, controls: list) -> float:
    """Partial Spearman rho(x, y | controls) for an arbitrary number of controls.

    Rank-transforms x, y, and each control, residualizes the ranked x and y
    against the ranked controls via OLS, and returns the Pearson r of the
    residuals — the standard rank-based generalization of partial Spearman
    beyond one control. With a single control this is algebraically equivalent
    to _partial_spearman.
    """
    stacked = np.column_stack([x, y] + list(controls))
    valid = np.all(np.isfinite(stacked), axis=1)
    if valid.sum() < 10:
        return float("nan")
    stacked = stacked[valid]
    ranked = np.column_stack([rankdata(stacked[:, i]) for i in range(stacked.shape[1])])
    x_r, y_r = ranked[:, 0], ranked[:, 1]
    C = np.column_stack([np.ones(len(ranked)), ranked[:, 2:]])

    def _resid(v):
        coeffs, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ coeffs

    rx, ry = _resid(x_r), _resid(y_r)
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run_area_covariate_checks(per_duct: pd.DataFrame) -> dict:
    """Check 1: is rho(pseudotime, hole_pct) explained by duct area?"""
    pt   = per_duct["pseudotime"].values
    hole = per_duct["hole_pct"].values
    area = per_duct["area_um2"].values
    nd   = per_duct["nuclear_density"].values

    rho_area_hole, p_area_hole = _safe_spearman(area, hole)
    rho_area_pt,   p_area_pt   = _safe_spearman(area, pt)

    return {
        "partial_rho_pt_hole_given_area":        _partial_spearman(pt, hole, area),
        "partial_rho_pt_hole_given_area_and_nd": _partial_spearman_multi(pt, hole, [area, nd]),
        "partial_rho_pt_area_given_hole":        _partial_spearman(pt, area, hole),
        "rho_area_hole_pct":    rho_area_hole,
        "p_area_hole_pct":      p_area_hole,
        "rho_area_pseudotime":  rho_area_pt,
        "p_area_pseudotime":    p_area_pt,
    }


def run_within_slide_checks(per_duct: pd.DataFrame, min_ducts: int) -> dict:
    """Check 2: per-slide correlations (raw + area-adjusted), plus a
    between-slide (slide-level median) correlation to rule a Simpson's
    paradox in or out explicitly."""
    per_slide_rows = []
    for slide_name, grp in per_duct.groupby("slide_name"):
        if len(grp) < min_ducts:
            continue
        pt, hole, area = grp["pseudotime"].values, grp["hole_pct"].values, grp["area_um2"].values
        rho, p = _safe_spearman(pt, hole)
        partial = _partial_spearman(pt, hole, area)
        per_slide_rows.append({
            "slide_name": str(slide_name),
            "n_ducts": int(len(grp)),
            "rho_pt_hole_pct": rho,
            "p_pt_hole_pct": p,
            "partial_rho_pt_hole_given_area": partial,
            "median_pseudotime": float(np.nanmedian(pt)),
            "median_hole_pct": float(np.nanmedian(hole)),
        })

    if not per_slide_rows:
        return {
            "per_slide": [],
            "summary": {},
            "between_slide_median_correlation": {"rho": float("nan"), "p": float("nan")},
            "n_slides_qualifying": 0,
        }

    slide_df = pd.DataFrame(per_slide_rows)
    raw_vals = slide_df["rho_pt_hole_pct"].values
    partial_vals = slide_df["partial_rho_pt_hole_given_area"].values

    def _summarize(vals):
        return {
            "mean":   float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "n_positive": int(np.sum(vals > 0)),
            "n_slides": int(len(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }

    between_rho, between_p = _safe_spearman(
        slide_df["median_pseudotime"].values, slide_df["median_hole_pct"].values
    )

    return {
        "per_slide": per_slide_rows,
        "summary": {"raw": _summarize(raw_vals), "area_adjusted": _summarize(partial_vals)},
        "between_slide_median_correlation": {"rho": between_rho, "p": between_p},
        "n_slides_qualifying": int(len(slide_df)),
    }


def run_within_slide_permutation(
    per_duct: pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """Check 3: shuffle duct labels WITHIN each slide (preserves the nesting
    structure), contrasted with the existing global shuffle in
    run_permutation_test."""
    pt    = per_duct["pseudotime"].values
    hole  = per_duct["hole_pct"].values
    slide = per_duct["slide_name"].values

    obs_rho = float(spearmanr(pt, hole).statistic)
    slide_idx = [np.where(slide == s)[0] for s in np.unique(slide)]

    null_rho = np.empty(n_permutations)
    for i in range(n_permutations):
        pt_perm = pt.copy()
        for idxs in slide_idx:
            pt_perm[idxs] = rng.permutation(pt_perm[idxs])
        null_rho[i] = abs(float(spearmanr(pt_perm, hole).statistic))

    perm_p = float(np.mean(null_rho >= abs(obs_rho)))

    return {
        "obs_rho_pt_hole_pct": obs_rho,
        "perm_p": perm_p,
        "perm_p_display": _format_perm_p(perm_p, n_permutations),
        "null95": float(np.percentile(null_rho, 95)),
        "n_permutations": n_permutations,
    }


def run_exclusion_bias_check(duct_table: pd.DataFrame, per_duct: pd.DataFrame):
    """Check 4: do zero-patch (excluded) ducts systematically differ from
    retained ducts? Returns (summary dict, full duct table tagged with a
    'retained' bool column)."""
    retained_ids = set(per_duct["object_id"])
    full = duct_table[["object_id", "slide_name", "area_um2", "hole_pct"]].copy()
    full["retained"] = full["object_id"].isin(retained_ids)

    excluded = full.loc[~full["retained"]]
    retained = full.loc[full["retained"]]

    def _compare(col: str) -> dict:
        a = excluded[col].dropna().values
        b = retained[col].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            u_stat, p = mannwhitneyu(a, b, alternative="two-sided")
            u_stat, p = float(u_stat), float(p)
        else:
            u_stat, p = float("nan"), float("nan")
        return {
            "median_excluded": float(np.nanmedian(a)) if len(a) else float("nan"),
            "median_retained": float(np.nanmedian(b)) if len(b) else float("nan"),
            "mannwhitney_u": u_stat,
            "mannwhitney_p": p,
        }

    result = {
        "n_excluded": int(len(excluded)),
        "n_retained": int(len(retained)),
        "area_um2": _compare("area_um2"),
        "hole_pct": _compare("hole_pct"),
    }
    return result, full


def run_aggregation_sensitivity(results_df: pd.DataFrame, duct_table: pd.DataFrame) -> dict:
    """Check 5: median vs mean patch-to-duct aggregation, and sensitivity of
    the correlation to poorly-sampled (few-patch) ducts."""
    per_duct_median = aggregate_per_duct(results_df, duct_table, np.nanmedian, "median")
    per_duct_mean    = aggregate_per_duct(results_df, duct_table, np.nanmean,   "mean")

    rho_median, p_median = _safe_spearman(
        per_duct_median["pseudotime"].values, per_duct_median["hole_pct"].values
    )
    rho_mean, p_mean = _safe_spearman(
        per_duct_mean["pseudotime"].values, per_duct_mean["hole_pct"].values
    )

    by_threshold = {}
    for thresh in N_PATCH_THRESHOLDS:
        subset = per_duct_median[per_duct_median["n_patches"] >= thresh]
        rho, p = _safe_spearman(subset["pseudotime"].values, subset["hole_pct"].values)
        by_threshold[f"min_{thresh}_patches"] = {
            "rho_pt_hole_pct": rho, "p": p, "n_ducts": int(len(subset)),
        }

    return {
        "median_aggregation": {
            "rho_pt_hole_pct": rho_median, "p": p_median, "n_ducts": int(len(per_duct_median)),
        },
        "mean_aggregation": {
            "rho_pt_hole_pct": rho_mean, "p": p_mean, "n_ducts": int(len(per_duct_mean)),
        },
        "by_min_patches_threshold": by_threshold,
    }


def run_patch_sampling_artifact_check(per_duct: pd.DataFrame) -> dict:
    """Check 6: could small ducts' lower nuclear density reflect an edge-patch
    sampling artifact (1-2 patch ducts more likely to straddle the boundary)
    rather than real biology? Reported for awareness; not corrected here."""
    area      = per_duct["area_um2"].values
    nd        = per_duct["nuclear_density"].values
    n_patches = per_duct["n_patches"].values

    rho_area_nd,     p_area_nd     = _safe_spearman(area, nd)
    rho_npatches_nd, p_npatches_nd = _safe_spearman(n_patches, nd)

    return {
        "rho_area_nuclear_density":        rho_area_nd,
        "p_area_nuclear_density":          p_area_nd,
        "rho_n_patches_nuclear_density":   rho_npatches_nd,
        "p_n_patches_nuclear_density":     p_npatches_nd,
        "note": (
            "If small/poorly-sampled ducts show systematically lower nuclear "
            "density, this may reflect edge-patch artifacts (ducts with 1-2 "
            "patches are more likely to have those patches straddle the duct "
            "boundary) rather than real biology. Reported for awareness only; "
            "not corrected here."
        ),
    }


def check_consistency_with_v1(per_duct_recomputed: pd.DataFrame, v1_csv_path: Path) -> dict:
    """Sanity check: does the recomputed (median-aggregation) per-duct table
    match v1's saved holeyness_per_duct.csv? The extended v2 numbers should
    not be trusted if this fails."""
    v1 = pd.read_csv(v1_csv_path)
    merged = per_duct_recomputed.merge(
        v1[["object_id", "hole_pct", "pseudotime"]],
        on="object_id", suffixes=("_v2", "_v1"), how="inner",
    )
    n_v1 = len(v1)
    n_v2 = len(per_duct_recomputed)
    n_compared = len(merged)

    if n_compared == 0:
        return {
            "n_v1_ducts": n_v1, "n_v2_ducts": n_v2, "n_compared": 0,
            "all_match": False,
            "note": "No overlapping object_id between v1 CSV and the v2 recompute.",
        }

    max_diff_hole = float((merged["hole_pct_v2"] - merged["hole_pct_v1"]).abs().max())
    max_diff_pt   = float((merged["pseudotime_v2"] - merged["pseudotime_v1"]).abs().max())
    all_match = (n_compared == n_v1 == n_v2) and max_diff_hole < 1e-6 and max_diff_pt < 1e-6

    return {
        "n_v1_ducts": n_v1,
        "n_v2_ducts": n_v2,
        "n_compared": n_compared,
        "max_abs_diff_hole_pct": max_diff_hole,
        "max_abs_diff_pseudotime": max_diff_pt,
        "all_match": bool(all_match),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 150) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def write_scatter_pt_vs_hole(per_duct: pd.DataFrame, output_dir: Path, section: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(
        per_duct["hole_pct"], per_duct["pseudotime"],
        c=pd.Categorical(per_duct["slide_name"]).codes,
        cmap="tab10", alpha=0.7, s=20, linewidths=0,
    )
    ax.set_xlabel("Duct hole fraction (%)")
    ax.set_ylabel("Duct-level pseudotime (median)")
    ax.set_title(f"Section {section}: pseudotime vs hole fraction")
    _save_fig(fig, output_dir, "scatter_pt_vs_hole_pct")


def write_scatter_hole_vs_nd(per_duct: pd.DataFrame, output_dir: Path, section: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        per_duct["nuclear_density"], per_duct["hole_pct"],
        c=pd.Categorical(per_duct["slide_name"]).codes,
        cmap="tab10", alpha=0.7, s=20, linewidths=0,
    )
    ax.set_xlabel("Duct-level nuclear density (nuclei/area, median)")
    ax.set_ylabel("Duct hole fraction (%)")
    ax.set_title(f"Section {section}: hole fraction vs nuclear density")
    _save_fig(fig, output_dir, "scatter_hole_pct_vs_nd")


def write_scatter_pt_vs_hole_by_area(per_duct: pd.DataFrame, output_dir: Path, section: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(
        per_duct["hole_pct"], per_duct["pseudotime"],
        c=per_duct["area_um2"], cmap="viridis", alpha=0.75, s=20, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Duct area (µm²)")
    ax.set_xlabel("Duct hole fraction (%)")
    ax.set_ylabel("Duct-level pseudotime (median)")
    ax.set_title(f"Section {section}: pseudotime vs hole fraction, coloured by area")
    _save_fig(fig, output_dir, "v2_scatter_pt_vs_hole_pct_by_area", dpi=300)


def write_scatter_pt_vs_area(per_duct: pd.DataFrame, output_dir: Path, section: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        per_duct["area_um2"], per_duct["pseudotime"],
        c=pd.Categorical(per_duct["slide_name"]).codes,
        cmap="tab10", alpha=0.7, s=20, linewidths=0,
    )
    ax.set_xlabel("Duct area (µm²)")
    ax.set_ylabel("Duct-level pseudotime (median)")
    ax.set_title(f"Section {section}: pseudotime vs duct area")
    _save_fig(fig, output_dir, "v2_scatter_pt_vs_area", dpi=300)


def write_small_multiples_per_slide(per_duct: pd.DataFrame, output_dir: Path, section: str) -> None:
    slides = sorted(per_duct["slide_name"].unique())
    n = len(slides)
    ncols = min(4, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)

    for i, slide_name in enumerate(slides):
        ax = axes[i // ncols][i % ncols]
        grp = per_duct[per_duct["slide_name"] == slide_name]
        ax.scatter(grp["hole_pct"], grp["pseudotime"], alpha=0.6, s=12, linewidths=0, color="#4878CF")
        rho, _ = _safe_spearman(grp["pseudotime"].values, grp["hole_pct"].values)
        title = f"{slide_name}\nn={len(grp)}" + (f", ρ={rho:.2f}" if np.isfinite(rho) else "")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Section {section}: per-slide pseudotime vs hole fraction")
    fig.tight_layout()
    _save_fig(fig, output_dir, "v2_small_multiples_per_slide", dpi=300)


# ── Output writers ────────────────────────────────────────────────────────────

def write_outputs(
    per_duct: pd.DataFrame,
    corrs: dict,
    perm: dict,
    output_dir: Path,
    section: str,
    aggregation: str,
    n_permutations: int,
    n_patches_total: int,
    n_patches_assigned: int,
    n_ducts_measured: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "holeyness_per_duct.csv"
    out_df = per_duct.drop(columns=["polygon"] if "polygon" in per_duct.columns else [])
    out_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV: {csv_path}")

    result = {
        "section":             section,
        "aggregation":         aggregation,
        "n_permutations":      n_permutations,
        "n_patches_total":     n_patches_total,
        "n_patches_in_ducts":  n_patches_assigned,
        "n_ducts_with_measurements": n_ducts_measured,
        "n_ducts_with_patches": len(per_duct),
        "n_ducts_zero_patches": n_ducts_measured - len(per_duct),
        "correlations": {**corrs, **perm},
    }
    json_path = output_dir / "holeyness_validation.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON: {json_path}")


# ── v2: output writers ────────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v)


def write_v2_report(
    output_dir: Path,
    section: str,
    consistency: dict,
    primary: dict,
    area_cov: dict,
    within_slide: dict,
    perm_global: dict,
    perm_within: dict,
    exclusion: dict,
    agg_sens: dict,
    sampling_artifact: dict,
) -> None:
    lines = [f"# Holeyness v2 — area-adjusted validation — section {section}", ""]

    lines += ["## Consistency check (v2 recompute vs v1 saved output)", ""]
    lines.append(
        f"- Compared {consistency['n_compared']} ducts "
        f"(v1: {consistency['n_v1_ducts']}, v2: {consistency['n_v2_ducts']})"
    )
    if "max_abs_diff_hole_pct" in consistency:
        lines.append(
            f"- Max abs diff hole_pct: {_fmt(consistency['max_abs_diff_hole_pct'])}, "
            f"pseudotime: {_fmt(consistency['max_abs_diff_pseudotime'])}"
        )
    verdict = (
        "MATCH — v2 recompute agrees with v1."
        if consistency.get("all_match")
        else "MISMATCH — investigate before trusting any v2 numbers below."
    )
    lines.append(f"- **Verdict:** {verdict}")
    lines.append("")

    lines += ["## 1. Duct area as covariate", ""]
    lines.append(f"- raw rho(pseudotime, hole_pct) = {_fmt(primary['rho_pt_hole_pct'])}")
    lines.append(f"- partial rho(pseudotime, hole_pct | area) = {_fmt(area_cov['partial_rho_pt_hole_given_area'])}")
    lines.append(
        f"- partial rho(pseudotime, hole_pct | area, nuclear_density) = "
        f"{_fmt(area_cov['partial_rho_pt_hole_given_area_and_nd'])}"
    )
    lines.append(f"- partial rho(pseudotime, area | hole_pct) = {_fmt(area_cov['partial_rho_pt_area_given_hole'])}")
    lines.append(f"- raw rho(area, hole_pct) = {_fmt(area_cov['rho_area_hole_pct'])}")
    lines.append(f"- raw rho(area, pseudotime) = {_fmt(area_cov['rho_area_pseudotime'])}")
    raw_rho = primary["rho_pt_hole_pct"]
    partial_rho = area_cov["partial_rho_pt_hole_given_area"]
    drop_pct = (
        100 * (1 - partial_rho / raw_rho)
        if np.isfinite(raw_rho) and raw_rho != 0 and np.isfinite(partial_rho)
        else None
    )
    if drop_pct is not None and drop_pct > 20:
        verdict1 = f"CONFIRMED — area-adjusted partial rho drops {drop_pct:.0f}% from the raw correlation."
    else:
        verdict1 = "NOT CONFIRMED — area adjustment does not substantially change the correlation."
    lines.append(f"- **Verdict:** {verdict1}")
    lines.append("")

    lines += ["## 2. Within-slide correlations", ""]
    ws = within_slide["summary"]
    if ws:
        r, a = ws["raw"], ws["area_adjusted"]
        lines.append(
            f"- Raw: mean={_fmt(r['mean'])}, median={_fmt(r['median'])}, "
            f"n_positive={r['n_positive']}/{r['n_slides']}, range=[{_fmt(r['min'])}, {_fmt(r['max'])}]"
        )
        lines.append(
            f"- Area-adjusted: mean={_fmt(a['mean'])}, median={_fmt(a['median'])}, "
            f"n_positive={a['n_positive']}/{a['n_slides']}, range=[{_fmt(a['min'])}, {_fmt(a['max'])}]"
        )
        bsc = within_slide["between_slide_median_correlation"]
        lines.append(f"- Between-slide (slide-level medians) rho = {_fmt(bsc['rho'])}, p = {_fmt(bsc['p'])}")
        if np.isfinite(bsc["rho"]) and r["median"] != 0:
            same_sign = (r["median"] > 0) == (bsc["rho"] > 0)
        else:
            same_sign = True
        verdict2 = (
            "No Simpson's paradox — within- and between-slide correlations agree in sign."
            if same_sign
            else "POSSIBLE SIMPSON'S PARADOX — within- and between-slide correlations disagree in sign."
        )
    else:
        verdict2 = "No slides had >= min_ducts_per_slide ducts — check skipped."
    lines.append(f"- **Verdict:** {verdict2}")
    lines.append("")

    lines += ["## 3. Permutation test: global vs within-slide", ""]
    lines.append(f"- Global shuffle: perm_p {perm_global.get('perm_p_rho_pt_hole_pct_display', '?')}")
    lines.append(f"- Within-slide shuffle: perm_p {perm_within['perm_p_display']}")
    verdict3 = (
        "Within-slide (structure-preserving) permutation agrees with the global null."
        if abs(perm_global["perm_p_rho_pt_hole_pct"] - perm_within["perm_p"]) < 0.05
        else "Within-slide permutation gives a materially different p-value than the global null — "
             "the global test may have overstated significance by ignoring slide nesting."
    )
    lines.append(f"- **Verdict:** {verdict3}")
    lines.append("")

    lines += ["## 4. Exclusion bias (zero-patch ducts)", ""]
    lines.append(
        f"- Excluded: n={exclusion['n_excluded']}, "
        f"median area={_fmt(exclusion['area_um2']['median_excluded'])}, "
        f"median hole_pct={_fmt(exclusion['hole_pct']['median_excluded'])}"
    )
    lines.append(
        f"- Retained: n={exclusion['n_retained']}, "
        f"median area={_fmt(exclusion['area_um2']['median_retained'])}, "
        f"median hole_pct={_fmt(exclusion['hole_pct']['median_retained'])}"
    )
    lines.append(f"- Mann-Whitney area_um2: p = {_fmt(exclusion['area_um2']['mannwhitney_p'])}")
    lines.append(f"- Mann-Whitney hole_pct: p = {_fmt(exclusion['hole_pct']['mannwhitney_p'])}")
    p_area = exclusion["area_um2"]["mannwhitney_p"]
    p_hole = exclusion["hole_pct"]["mannwhitney_p"]
    sig = (np.isfinite(p_area) and p_area < 0.05) or (np.isfinite(p_hole) and p_hole < 0.05)
    verdict4 = (
        "Excluded ducts differ systematically from retained ducts — the zero-patch exclusion is NOT random."
        if sig
        else "No significant difference detected between excluded and retained ducts."
    )
    lines.append(f"- **Verdict:** {verdict4}")
    lines.append("")

    lines += ["## 5. Aggregation sensitivity", ""]
    lines.append(
        f"- Median aggregation: rho = {_fmt(agg_sens['median_aggregation']['rho_pt_hole_pct'])} "
        f"(n={agg_sens['median_aggregation']['n_ducts']})"
    )
    lines.append(
        f"- Mean aggregation: rho = {_fmt(agg_sens['mean_aggregation']['rho_pt_hole_pct'])} "
        f"(n={agg_sens['mean_aggregation']['n_ducts']})"
    )
    for k, v in agg_sens["by_min_patches_threshold"].items():
        lines.append(f"- {k}: rho = {_fmt(v['rho_pt_hole_pct'])} (n={v['n_ducts']})")
    lines.append(
        "- **Verdict:** compare the rows above — a large swing under stricter n_patches "
        "thresholds indicates the effect depends on poorly-sampled ducts."
    )
    lines.append("")

    lines += ["## 6. Patch sampling artifact check", ""]
    lines.append(f"- rho(area, nuclear_density) = {_fmt(sampling_artifact['rho_area_nuclear_density'])}")
    lines.append(f"- rho(n_patches, nuclear_density) = {_fmt(sampling_artifact['rho_n_patches_nuclear_density'])}")
    lines.append(f"- {sampling_artifact['note']}")
    lines.append("")

    (output_dir / "holeyness_validation_v2.md").write_text("\n".join(lines), encoding="utf-8")


def write_v2_outputs(
    output_dir: Path,
    section: str,
    consistency: dict,
    primary: dict,
    area_cov: dict,
    within_slide: dict,
    perm_global: dict,
    perm_within: dict,
    exclusion: dict,
    duct_table_full: pd.DataFrame,
    agg_sens: dict,
    sampling_artifact: dict,
    n_permutations: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "section": section,
        "n_permutations": n_permutations,
        "consistency_check": consistency,
        "primary_correlation": primary,
        "area_covariate": area_cov,
        "within_slide": within_slide,
        "permutation": {"global": perm_global, "within_slide": perm_within},
        "exclusion_bias": exclusion,
        "aggregation_sensitivity": agg_sens,
        "patch_sampling_artifact": sampling_artifact,
    }
    json_path = output_dir / "holeyness_validation_v2.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"  JSON: {json_path}")

    csv_path = output_dir / "duct_table_full.csv"
    duct_table_full.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  CSV: {csv_path}")

    write_v2_report(
        output_dir, section, consistency, primary, area_cov,
        within_slide, perm_global, perm_within, exclusion, agg_sens, sampling_artifact,
    )
    print(f"  Markdown report: {output_dir / 'holeyness_validation_v2.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Duct-level holeyness validation")
    parser.add_argument("--section",          required=True,
                        help="Section label, e.g. '2M-1'")
    parser.add_argument("--export",           required=True, type=Path,
                        help="Path to combined_matched_measurements.txt")
    parser.add_argument("--annotation-dir",   required=True, type=Path,
                        help="Directory containing <slide>.json ratio annotation files")
    parser.add_argument("--slide-dimensions", required=True, type=Path,
                        help="Path to slide_dimensions.json from MCF7_x5_cropped/")
    parser.add_argument("--results",          required=True, type=Path,
                        help="Path to this section's per-patch results.csv")
    parser.add_argument("--output-dir",       required=True, type=Path,
                        help="Output directory for CSV, JSON, figures")
    parser.add_argument("--slide-list",       required=True, type=Path,
                        help="Text file with one pipeline slide_name per line")
    parser.add_argument("--aggregation",      default="median",
                        choices=["median", "mean"],
                        help="Patch-to-duct aggregation function (default: median)")
    parser.add_argument("--n-permutations",   default=1000, type=int,
                        help="Permutation test iterations (default: 1000)")
    parser.add_argument("--patch-size",       default=PATCH_SIZE_DEFAULT, type=int,
                        help="Patch size in pixels (default: 112)")
    parser.add_argument("--seed",             default=42, type=int)
    parser.add_argument("--v2",               action="store_true",
                        help="Run the extended v2 area-adjusted validation instead of "
                             "the v1 pipeline. --output-dir should point at a NEW "
                             "versioned subdirectory; v1 outputs are never touched.")
    parser.add_argument("--v1-per-duct-csv",  default=None, type=Path,
                        help="Path to v1's holeyness_per_duct.csv (required with --v2; "
                             "used only for the consistency check).")
    parser.add_argument("--min-ducts-per-slide", default=WITHIN_SLIDE_MIN_DUCTS_DEFAULT, type=int,
                        help=f"Minimum ducts per slide for within-slide checks "
                             f"(default: {WITHIN_SLIDE_MIN_DUCTS_DEFAULT}).")
    args = parser.parse_args()

    if args.v2 and args.v1_per_duct_csv is None:
        parser.error("--v1-per-duct-csv is required when --v2 is set")

    rng = np.random.default_rng(args.seed)
    agg_fn    = np.nanmedian if args.aggregation == "median" else np.nanmean
    agg_label = args.aggregation

    print("=" * 60)
    print(f"  Holeyness validation — section {args.section}")
    print("=" * 60)

    # 1. Slide list
    pipeline_slides = load_slide_list(args.slide_list)
    print(f"\nPipeline slides ({len(pipeline_slides)}): {pipeline_slides}")

    # 2. Slide dimensions
    slide_dims = load_slide_dimensions(args.slide_dimensions)

    # 3. Measurement export
    measurements = parse_measurement_export(args.export, pipeline_slides)

    # 4. Annotation polygons
    polygons = load_duct_polygons(args.annotation_dir, pipeline_slides, slide_dims)

    # 5. Build duct table (join UUID → polygon + hole_pct)
    duct_table = build_duct_table(measurements, polygons)
    if len(duct_table) == 0:
        sys.exit("ERROR: no ducts remain after UUID join — check that annotation "
                 "files and measurement export share the same QuPath project UUIDs.")

    # 6. Load results.csv
    print("\n=== results.csv ===")
    results_df = pd.read_csv(args.results, low_memory=False)
    missing_cols = [c for c in REQUIRED_RESULTS_COLS if c not in results_df.columns]
    if missing_cols:
        sys.exit(f"ERROR: results.csv missing columns: {missing_cols}")
    results_df = results_df[REQUIRED_RESULTS_COLS].drop_duplicates()
    results_df = results_df[results_df["slide_name"].isin(pipeline_slides)].copy()
    n_patches_total = len(results_df)
    print(f"  Patches: {n_patches_total} across {results_df['slide_name'].nunique()} slides")

    # 7. Assign patches to ducts
    results_df = assign_patches_to_ducts(results_df, duct_table, args.patch_size)
    n_patches_assigned = int(results_df["duct_id"].notna().sum())

    # 8. Aggregate per duct
    per_duct = aggregate_per_duct(results_df, duct_table, agg_fn, agg_label)
    if len(per_duct) < 4:
        sys.exit("ERROR: fewer than 4 ducts with patches — cannot compute correlations.")

    n_ducts_measured = len(duct_table)

    if not args.v2:
        # ── v1 path: unchanged behavior ──────────────────────────────────────
        # 9. Correlations
        print("\n=== Correlations ===")
        corrs = run_correlations(per_duct)
        for k, v in corrs.items():
            print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

        # 10. Permutation test
        perm = run_permutation_test(per_duct, args.n_permutations, rng)

        # 11. Figures
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_scatter_pt_vs_hole(per_duct, args.output_dir, args.section)
        write_scatter_hole_vs_nd(per_duct, args.output_dir, args.section)

        # 12. Write outputs
        write_outputs(
            per_duct, corrs, perm, args.output_dir,
            section=args.section,
            aggregation=agg_label,
            n_permutations=args.n_permutations,
            n_patches_total=n_patches_total,
            n_patches_assigned=n_patches_assigned,
            n_ducts_measured=n_ducts_measured,
        )

        print("\n" + "=" * 60)
        print(f"  HOLEYNESS VALIDATION COMPLETE — section {args.section}")
        print("=" * 60)
        print(f"\n  rho(PT, hole%)  = {corrs['rho_pt_hole_pct']:.4f}"
              f"  perm_p = {perm['perm_p_rho_pt_hole_pct']:.4f}")
        print(f"  partial_rho     = {corrs['partial_rho_pt_hole_given_nd']:.4f}"
              f"  perm_p = {perm['perm_p_partial_rho_pt_hole_given_nd']:.4f}")
        print(f"  rho(ND, hole%)  = {corrs['rho_nd_hole_pct']:.4f}  (independence check)")
        print(f"  rho(PI, hole%)  = {corrs['rho_pi_hole_pct']:.4f}  (independence check)")
        print(f"\n  Output dir: {args.output_dir}")
        return

    # ── v2 path: extended, area-adjusted validation ─────────────────────────
    print("\n" + "=" * 60)
    print(f"  HOLEYNESS VALIDATION v2 (area-adjusted) — section {args.section}")
    print("=" * 60)

    # The primary v2 table is always median-aggregated, regardless of --aggregation.
    per_duct_v2 = aggregate_per_duct(results_df, duct_table, np.nanmedian, "median")
    if len(per_duct_v2) < 4:
        sys.exit("ERROR: fewer than 4 ducts with patches — cannot compute v2 correlations.")

    print("\n=== Consistency check vs v1 ===")
    consistency = check_consistency_with_v1(per_duct_v2, args.v1_per_duct_csv)
    if consistency.get("all_match"):
        print(f"  OK: v2 recompute matches v1 ({consistency['n_compared']} ducts).")
    else:
        print(f"  WARNING: v2 recompute does NOT match v1 saved output: {consistency}")

    print("\n=== Primary correlation (median aggregation) ===")
    primary = run_correlations(per_duct_v2)
    for k, v in primary.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    print("\n=== Check 1: duct area as covariate ===")
    area_cov = run_area_covariate_checks(per_duct_v2)

    print("\n=== Check 2: within-slide correlations ===")
    within_slide = run_within_slide_checks(per_duct_v2, args.min_ducts_per_slide)

    print("\n=== Check 3: permutation — global vs within-slide ===")
    perm_global = run_permutation_test(per_duct_v2, args.n_permutations, rng)
    perm_global = {
        **perm_global,
        "perm_p_rho_pt_hole_pct_display": _format_perm_p(
            perm_global["perm_p_rho_pt_hole_pct"], args.n_permutations
        ),
        "perm_p_partial_rho_pt_hole_given_nd_display": _format_perm_p(
            perm_global["perm_p_partial_rho_pt_hole_given_nd"], args.n_permutations
        ),
    }
    perm_within = run_within_slide_permutation(per_duct_v2, args.n_permutations, rng)

    print("\n=== Check 4: exclusion bias (zero-patch ducts) ===")
    exclusion, duct_table_full = run_exclusion_bias_check(duct_table, per_duct_v2)
    print(f"  Excluded: {exclusion['n_excluded']}  Retained: {exclusion['n_retained']}")

    print("\n=== Check 5: aggregation sensitivity ===")
    agg_sens = run_aggregation_sensitivity(results_df, duct_table)

    print("\n=== Check 6: patch sampling artifact check ===")
    sampling_artifact = run_patch_sampling_artifact_check(per_duct_v2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scatter_pt_vs_hole_by_area(per_duct_v2, args.output_dir, args.section)
    write_scatter_pt_vs_area(per_duct_v2, args.output_dir, args.section)
    write_small_multiples_per_slide(per_duct_v2, args.output_dir, args.section)

    write_v2_outputs(
        args.output_dir, args.section,
        consistency, primary, area_cov, within_slide,
        perm_global, perm_within, exclusion, duct_table_full,
        agg_sens, sampling_artifact, args.n_permutations,
    )

    print("\n" + "=" * 60)
    print(f"  HOLEYNESS VALIDATION v2 COMPLETE — section {args.section}")
    print("=" * 60)
    print(f"\n  raw rho(PT, hole%)               = {primary['rho_pt_hole_pct']:.4f}")
    print(f"  partial rho | area                = {area_cov['partial_rho_pt_hole_given_area']:.4f}")
    print(f"  partial rho | area, nuclear_dens  = {area_cov['partial_rho_pt_hole_given_area_and_nd']:.4f}")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
