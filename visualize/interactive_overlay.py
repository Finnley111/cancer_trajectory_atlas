"""Interactive pseudotime overlay for whole-slide images.

Generates one standalone HTML file per slide: the PNG is embedded as a
background image and patch pseudotime values are rendered as a WebGL scatter
overlay.  The colorscale can be swapped interactively via a dropdown menu.

Usage:
    python -m cancer_trajectory_atlas.visualize.interactive_overlay \\
        --results-csv  $SCRATCH/results/atlas_none_harmony/results.csv \\
        --png-dir      $SCRATCH/data/MCF7_x5_cropped \\
        --output-dir   $SCRATCH/results/overlays \\
        --patch-size   112
"""

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Maximum pixel dimension of the embedded background image.
# Resizing only affects the embedded PNG; scatter coordinates stay in
# original pixel space and are scaled together with the axis range.
_MAX_EMBED_PX = 1200

_COLORSCALES = [
    ("Spectral (default)",    "Spectral_r"),
    ("RdYlBu",                "RdYlBu_r"),
    ("RdYlGn",                "RdYlGn_r"),
    ("Plasma",                "Plasma"),
    ("Viridis",               "Viridis"),
    ("Inferno",               "Inferno"),
    ("Turbo",                 "Turbo"),
    ("Hot",                   "Hot"),
    ("Jet",                   "Jet"),
]


def _png_to_base64(img_pil: Image.Image) -> str:
    """Return a data-URI string for embedding a PIL image."""
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resize_for_embed(img_pil: Image.Image, max_px: int = _MAX_EMBED_PX) -> Image.Image:
    """Downsample the image so its longest dimension is at most max_px."""
    w, h = img_pil.size
    if max(w, h) <= max_px:
        return img_pil
    scale = max_px / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img_pil.resize((new_w, new_h), Image.LANCZOS)


def build_slide_figure(
    slide_df: pd.DataFrame,
    png_path: Path,
    patch_size: int = 112,
    ndpi_scale: float = 0.5,
) -> "plotly.graph_objects.Figure":
    import plotly.graph_objects as go

    img_pil = Image.open(png_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    thumb = _resize_for_embed(img_pil)
    img_src = _png_to_base64(thumb)

    half = patch_size / 2.0
    xs = slide_df["x"].values + half
    ys = slide_df["y"].values + half
    pt = slide_df["pseudotime"].values.astype(float)

    # Convert patch-centre coordinates to NDPI level-0 pixel space.
    # The PNG was produced at ndpi_scale (default 0.5), so NDPI coords = PNG coords / ndpi_scale.
    ndpi_xs = (xs / ndpi_scale).astype(int)
    ndpi_ys = (ys / ndpi_scale).astype(int)

    default_cs = _COLORSCALES[8][1] # set default to "Jet"

    scatter = go.Scattergl(
        x=xs,
        y=ys,
        mode="markers",
        customdata=np.column_stack([ndpi_xs, ndpi_ys]),
        marker=dict(
            color=pt,
            colorscale=default_cs,
            reversescale=False,
            cmin=0.0,
            cmax=1.0,
            size=5,
            opacity=0.75,
            colorbar=dict(
                title="Pseudotime",
                thickness=15,
                len=0.6,
            ),
            showscale=True,
        ),
        hovertemplate=(
            "NDPI x=%{customdata[0]}, y=%{customdata[1]}<br>"
            "pseudotime=%{marker.color:.4f}"
            "<extra></extra>"
        ),
    )

    layout_image = dict(
        source=img_src,
        xref="x", yref="y",
        x=0, y=0,
        sizex=orig_w, sizey=orig_h,
        xanchor="left", yanchor="top",
        sizing="stretch",
        opacity=1.0,
        layer="below",
    )

    colorscale_buttons = [
        dict(
            label=label,
            method="restyle",
            args=[{"marker.colorscale": cs}, [0]],
        )
        for label, cs in _COLORSCALES
    ]

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        images=[layout_image],
        xaxis=dict(
            range=[0, orig_w],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[orig_h, 0],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="black",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=[
            dict(
                buttons=colorscale_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
                bgcolor="white",
                bordercolor="#aaa",
                font=dict(size=11),
            )
        ],
        annotations=[
            dict(
                text="Colorscale:",
                x=0.0, xref="paper",
                y=1.09, yref="paper",
                showarrow=False,
                font=dict(size=11),
            )
        ],
    )
    fig.update_layout(
        title=dict(
            text=f"{png_path.stem}  —  {len(slide_df):,} patches",
            font=dict(size=13),
        )
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Build interactive pseudotime overlay HTML files."
    )
    parser.add_argument("--results-csv", type=Path, required=True,
                        help="results.csv from a pipeline run")
    parser.add_argument("--png-dir", type=Path, required=True,
                        help="Directory containing slide PNG files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write per-slide HTML files")
    parser.add_argument("--patch-size", type=int, default=112,
                        help="Patch size used during extraction (pixels)")
    parser.add_argument("--slides", nargs="*",
                        help="Process only these slide stems (default: all)")
    parser.add_argument("--ndpi-scale", type=float, default=0.5,
                        help="Scale factor used during NDPI→PNG conversion (default: 0.5)")
    args = parser.parse_args()

    try:
        import plotly
    except ImportError:
        print("ERROR: plotly is required.  pip install plotly")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    required = {"x", "y", "slide_name", "pseudotime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {missing}")

    slide_names = sorted(df["slide_name"].unique())
    if args.slides:
        slide_names = [s for s in slide_names if s in args.slides]

    print(f"Building overlays for {len(slide_names)} slides...")

    for slide_name in slide_names:
        png_path = args.png_dir / f"{slide_name}.png"
        if not png_path.exists():
            print(f"  SKIP {slide_name}: PNG not found at {png_path}")
            continue

        slide_df = df[df["slide_name"] == slide_name].reset_index(drop=True)
        print(f"  {slide_name}: {len(slide_df)} patches", end="", flush=True)

        fig = build_slide_figure(slide_df, png_path, patch_size=args.patch_size, ndpi_scale=args.ndpi_scale)

        out_path = args.output_dir / f"overlay_{slide_name}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  →  {out_path.name}")

    print(f"\nDone. HTML files in: {args.output_dir}")


if __name__ == "__main__":
    main()
