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
from scipy.stats import spearmanr


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


# ── Figures ───────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
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
    args = parser.parse_args()

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
    results_df = results_df[REQUIRED_RESULTS_COLS + ["slide_name"]].drop_duplicates()
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


if __name__ == "__main__":
    main()
