"""Offline-capable interactive Plotly figures for a post-processed atlas run.

Produces two self-contained HTML files (plotly.js embedded inline):
  postprocess/interactive/umap_interactive.html   — 2D UMAP scatter
  postprocess/interactive/diffusion3d_interactive.html — 3D DC1/DC2/DC3 scatter

Usage:
    python -m cancer_trajectory_atlas.visualize.interactive_plotly \\
        --run-dir  $SCRATCH/results/atlas_none_scvi \\
        --color-by pseudotime \\
        --max-points-3d 12000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    print("ERROR: scanpy is required. Activate the atlas environment.")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly is required. pip install plotly.")
    sys.exit(1)

SECTION_COLORS = {"2M-1": "#1b9e77", "2M-2": "#d95f02"}

# tab20 hex codes (matplotlib tab20, indices 0–19)
_TAB20 = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]


def _load_adata(run_dir: Path) -> sc.AnnData:
    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        print(f"ERROR: adata_full.h5ad not found in {run_dir}")
        sys.exit(1)
    print(f"Loading {h5ad} ...")
    return sc.read_h5ad(h5ad)


def _load_slide_names(run_dir: Path, adata: sc.AnnData) -> np.ndarray:
    """Return per-patch slide name array; falls back to mouse_id + section_number."""
    csv = run_dir / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if "slide_id" in df.columns and "slide_name" in df.columns:
            mapping = {str(int(row.slide_id)): row.slide_name
                       for _, row in df[["slide_id", "slide_name"]].drop_duplicates().iterrows()}
            return np.array([mapping.get(str(sid), str(sid))
                             for sid in adata.obs["slide_id"].values])
    # fallback
    if "mouse_id" in adata.obs.columns and "section_number" in adata.obs.columns:
        return (adata.obs["mouse_id"].astype(str) + " "
                + adata.obs["section_number"].astype(str)).values
    return adata.obs["slide_id"].astype(str).values


def _build_customdata(adata: sc.AnnData, slide_names: np.ndarray) -> np.ndarray:
    pt = adata.obs["pseudotime"].values
    pt_std = adata.obs.get("pseudotime_std",
                           pd.Series(np.full(len(adata), np.nan), index=adata.obs.index)).values
    cluster = adata.obs.get("cluster",
                             pd.Series(["?" ] * len(adata), index=adata.obs.index)).values
    section = adata.obs.get("section_number",
                             pd.Series(["?"] * len(adata), index=adata.obs.index)).values
    return np.column_stack([
        slide_names,
        section,
        [f"{x:.3f}" for x in pt],
        [f"{x:.3f}" if not np.isnan(x) else "N/A" for x in pt_std],
        cluster.astype(str),
    ])


_HOVER = (
    "Slide: %{customdata[0]}<br>"
    "Section: %{customdata[1]}<br>"
    "Pseudotime: %{customdata[2]}<br>"
    "PT std: %{customdata[3]}<br>"
    "Cluster: %{customdata[4]}"
    "<extra></extra>"
)


def _color_args(color_by: str, adata: sc.AnnData, mask: np.ndarray):
    """Return a list of (label, idx_in_mask, marker_dict) tuples for the requested coloring."""
    pt = adata.obs["pseudotime"].values[mask]

    if color_by == "pseudotime":
        return [("Pseudotime", np.arange(len(pt)), dict(
            color=pt,
            colorscale="Plasma",
            colorbar=dict(title="Pseudotime", thickness=15),
            showscale=True,
        ))]

    if color_by == "leiden":
        labels = adata.obs.get(
            "leiden_lowres",
            adata.obs.get("cluster", pd.Series(["?" ] * len(adata), index=adata.obs.index))
        ).values[mask].astype(str)
        unique = sorted(set(labels), key=lambda x: (not x.lstrip("-").isdigit(), x))
        out = []
        for i, lbl in enumerate(unique):
            color = _TAB20[i % len(_TAB20)]
            out.append((f"Leiden {lbl}", np.where(labels == lbl)[0],
                        dict(color=color, showscale=False)))
        return out

    if color_by == "section":
        sections = adata.obs.get(
            "section_number", pd.Series(["?"] * len(adata), index=adata.obs.index)
        ).values[mask].astype(str)
        unique = sorted(set(sections))
        out = []
        for i, sec in enumerate(unique):
            color = SECTION_COLORS.get(sec, _TAB20[i % len(_TAB20)])
            out.append((sec, np.where(sections == sec)[0],
                        dict(color=color, showscale=False)))
        return out

    raise ValueError(f"Unknown --color-by value: {color_by!r}")


# ── 2D UMAP ──────────────────────────────────────────────────────────────────

def build_umap_figure(adata: sc.AnnData, custom: np.ndarray, color_by: str) -> go.Figure:
    coords = adata.obsm["X_umap"]
    mask = np.arange(len(adata))
    groups = _color_args(color_by, adata, mask)
    show_legend = color_by != "pseudotime"
    traces = []
    for name, idx, mkr in groups:
        traces.append(go.Scattergl(
            x=coords[idx, 0], y=coords[idx, 1],
            mode="markers",
            name=name,
            marker=dict(size=4, opacity=0.6, **mkr),
            customdata=custom[idx],
            hovertemplate=_HOVER,
            showlegend=show_legend,
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"UMAP — color by {color_by}",
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        plot_bgcolor="white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ── 3D Diffusion ──────────────────────────────────────────────────────────────

def _downsample_3d(adata: sc.AnnData, max_points: int, rng: np.random.Generator):
    """Return index array for 3D plot, retaining all top-20%-PT points."""
    pt = adata.obs["pseudotime"].values
    N = len(pt)
    if N <= max_points:
        print(f"3D: using all {N} points (≤ {max_points} cap)")
        return np.arange(N)

    thresh = np.percentile(pt, 80)
    high = np.where(pt >= thresh)[0]
    low  = np.where(pt <  thresh)[0]
    n_fill = max_points - len(high)
    if n_fill > 0:
        sampled_low = rng.choice(low, min(n_fill, len(low)), replace=False)
        idx = np.sort(np.concatenate([high, sampled_low]))
    else:
        idx = high
    print(
        f"3D: {len(idx)}/{N} pts "
        f"(all {len(high)} with PT ≥ {thresh:.3f} retained, "
        f"{len(idx) - len(high)} sampled from {len(low)} low-PT)"
    )
    return idx


def build_diffusion3d_figure(
    adata: sc.AnnData,
    custom: np.ndarray,
    color_by: str,
    max_points: int,
    rng: np.random.Generator,
) -> go.Figure:
    dc = adata.obsm["X_diffmap"][:, :3]
    idx = _downsample_3d(adata, max_points, rng)
    n_shown, N = len(idx), len(adata)
    sub_custom = custom[idx]

    groups = _color_args(color_by, adata, idx)
    show_legend = color_by != "pseudotime"
    traces = []
    for name, sub_idx, mkr in groups:
        traces.append(go.Scatter3d(
            x=dc[idx[sub_idx], 0],
            y=dc[idx[sub_idx], 1],
            z=dc[idx[sub_idx], 2],
            mode="markers",
            name=name,
            marker=dict(size=2, opacity=0.5, **mkr),
            customdata=sub_custom[sub_idx],
            hovertemplate=_HOVER,
            showlegend=show_legend,
        ))
    thresh_str = f"{np.percentile(adata.obs['pseudotime'].values, 80):.3f}" if N > max_points else "—"
    fig = go.Figure(traces)
    fig.update_layout(
        title=(
            f"Diffusion Manifold (DC1/DC2/DC3) — color by {color_by}<br>"
            f"<sup>{n_shown}/{N} pts shown"
            + (f"; all PT ≥ {thresh_str} retained" if N > max_points else "")
            + "</sup>"
        ),
        scene=dict(
            xaxis_title="DC1", yaxis_title="DC2", zaxis_title="DC3",
            camera=dict(eye=dict(x=1.25, y=1.25, z=0.7)),
            bgcolor="white",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate offline-capable interactive Plotly figures for an atlas run."
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Path to the atlas run directory (contains adata_full.h5ad).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: {run_dir}/postprocess/interactive).")
    parser.add_argument("--color-by", choices=["pseudotime", "leiden", "section"],
                        default="pseudotime",
                        help="How to color the scatter points (default: pseudotime).")
    parser.add_argument("--max-points-3d", type=int, default=12000,
                        help="Cap on 3D scatter points; high-PT points are always retained "
                             "(default: 12000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for 3D downsampling (default: 42).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or run_dir / "postprocess" / "interactive").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = _load_adata(run_dir)
    rng = np.random.default_rng(args.seed)

    # Validate required fields
    if "pseudotime" not in adata.obs.columns:
        print("ERROR: adata_full.h5ad is missing obs['pseudotime']. Run post-processing first.")
        sys.exit(1)

    slide_names = _load_slide_names(run_dir, adata)
    custom = _build_customdata(adata, slide_names)

    # 2D UMAP
    if "X_umap" not in adata.obsm:
        print("WARNING: X_umap not found in obsm — skipping 2D UMAP figure.")
    else:
        print("Building 2D UMAP figure ...")
        fig2d = build_umap_figure(adata, custom, args.color_by)
        out2d = out_dir / "umap_interactive.html"
        fig2d.write_html(str(out2d), include_plotlyjs=True)
        print(f"  Saved: {out2d}")

    # 3D Diffusion
    if "X_diffmap" not in adata.obsm or adata.obsm["X_diffmap"].shape[1] < 3:
        print("WARNING: X_diffmap with ≥3 components not found in obsm — skipping 3D figure.")
    else:
        print("Building 3D diffusion figure ...")
        fig3d = build_diffusion3d_figure(adata, custom, args.color_by, args.max_points_3d, rng)
        out3d = out_dir / "diffusion3d_interactive.html"
        fig3d.write_html(str(out3d), include_plotlyjs=True)
        print(f"  Saved: {out3d}")

    print("\nDone. Open HTML files in a browser — no internet connection required.")


if __name__ == "__main__":
    main()
