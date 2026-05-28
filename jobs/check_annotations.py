"""Visual QC: draw annotation polygon outlines over slide thumbnails.

For each cropped slide PNG, renders all annotation polygons as coloured outlines
on a thumbnail so you can verify the converted annotations are correctly placed.

Colour key:
  Tumor      — red
  Ignore*    — grey
  Necrosis   — dark red
  Region*    — blue
  (other)    — orange

Usage (cluster):
    python ~/cancer_trajectory_atlas/jobs/check_annotations.py \
        --png-dir    $SCRATCH/data/MCF7_x5_cropped \
        --ann-dir    ~/cancer_trajectory_atlas/data/annotations_ratio \
        --dims-json  $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json \
        --output-dir $SCRATCH/annotation_check

Usage (local):
    python jobs/check_annotations.py \
        --png-dir    data/MCF7_x5_cropped \
        --ann-dir    data/annotations_ratio \
        --dims-json  data/MCF7_x5_cropped/slide_dimensions.json \
        --output-dir annotation_check
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

THUMB_WIDTH = 3000  # px — wide enough to see polygon detail

CLASS_COLORS = {
    "Tumor":     (220, 20,  20),
    "Ignore*":   (160, 160, 160),
    "Necrosis":  (100, 0,   0),
    "Region*":   (30,  80,  200),
}
DEFAULT_COLOR = (255, 140, 0)
LINE_WIDTH = 3


def _color_for(class_name: str) -> tuple:
    for key, color in CLASS_COLORS.items():
        if class_name.startswith(key.rstrip("*")):
            return color
    return DEFAULT_COLOR


def _draw_polygon(draw, vertices_thumb, color, line_width):
    """Draw a closed polygon outline as a polyline."""
    pts = [(x, y) for x, y in vertices_thumb]
    if len(pts) < 2:
        return
    pts_closed = pts + [pts[0]]
    draw.line(pts_closed, fill=color, width=line_width)


def _add_legend(img, present_classes):
    """Stamp a legend in the bottom-left corner."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except Exception:
        font = ImageFont.load_default()

    pad = 12
    swatch = 28
    line_h = swatch + 6
    legend_h = pad + len(present_classes) * line_h + pad
    legend_w = 260
    x0 = pad
    y0 = img.height - legend_h - pad

    draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h], fill=(20, 20, 20, 200))

    for i, (name, color) in enumerate(present_classes):
        sy = y0 + pad + i * line_h
        draw.rectangle([x0 + pad, sy, x0 + pad + swatch, sy + swatch], fill=color)
        draw.text((x0 + pad + swatch + 8, sy), name, fill=(255, 255, 255), font=font)


def process_slide(png_path: Path, ann_path: Path, orig_w: int, orig_h: int,
                  out_path: Path, thumb_width: int = THUMB_WIDTH):
    img = Image.open(png_path).convert("RGB")
    crop_w, crop_h = img.size

    scale_x = thumb_width / crop_w
    thumb_h  = int(crop_h * scale_x)
    thumb = img.resize((thumb_width, thumb_h), Image.LANCZOS)
    draw  = ImageDraw.Draw(thumb)

    # ratio → thumbnail pixel
    # full_px = ratio * orig_dim ; thumb_px = full_px * (thumb_size / crop_dim)
    sx = orig_w * scale_x   # = orig_w * (THUMB_WIDTH / crop_w)
    sy = orig_h * (thumb_h / crop_h)

    with open(ann_path) as f:
        data = json.load(f)

    features = data["features"] if data.get("type") == "FeatureCollection" else data

    present_classes: dict[str, tuple] = {}

    for feat in features:
        geom  = feat.get("geometry", {})
        props = feat.get("properties", {})
        cls   = props.get("classification") or {}
        name  = (cls.get("name") if isinstance(cls, dict) else cls) or "Tumor"
        color = _color_for(name)
        present_classes[name] = color

        geom_type = geom.get("type", "")
        if geom_type == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom_type == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            continue

        for ring in rings:
            verts = [(c[0] * sx, c[1] * sy) for c in ring]
            _draw_polygon(draw, verts, color, LINE_WIDTH)

    _add_legend(thumb, sorted(present_classes.items()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(out_path)
    return len(features), dict(present_classes)


def main():
    parser = argparse.ArgumentParser(
        description="Draw annotation outlines on slide thumbnails for visual QC."
    )
    parser.add_argument("--png-dir",    type=Path, required=True,
                        help="Directory with cropped slide PNGs (*_x5.png)")
    parser.add_argument("--ann-dir",    type=Path, required=True,
                        help="Directory with converted ratio-coord annotation JSONs")
    parser.add_argument("--dims-json",  type=Path, required=True,
                        help="slide_dimensions.json (original_full_width/height per slide)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write thumbnail PNGs")
    parser.add_argument("--thumb-width", type=int, default=THUMB_WIDTH,
                        help=f"Thumbnail width in pixels (default: {THUMB_WIDTH})")
    args = parser.parse_args()

    thumb_width = args.thumb_width

    with open(args.dims_json) as f:
        dims = json.load(f)

    png_files = sorted(args.png_dir.glob("*_x5.png"))
    if not png_files:
        print(f"ERROR: no *_x5.png files in {args.png_dir}")
        return

    print(f"Checking {len(png_files)} slides -> {args.output_dir}/\n")

    missing = []
    for png_path in png_files:
        stem      = png_path.stem.replace("_x5", "")   # e.g. 6027-4L-2M-1
        ann_path  = args.ann_dir / f"{stem}.json"
        dims_key  = png_path.name

        if not ann_path.exists():
            print(f"  SKIP {png_path.name}: no annotation at {ann_path}")
            missing.append(stem)
            continue

        if dims_key not in dims:
            print(f"  SKIP {png_path.name}: not in dims_json")
            missing.append(stem)
            continue

        d      = dims[dims_key]
        orig_w = d["original_full_width"]
        orig_h = d["original_full_height"]
        out    = args.output_dir / f"{stem}_annot_check.png"

        n_feats, classes = process_slide(png_path, ann_path, orig_w, orig_h, out, thumb_width)
        class_str = "  ".join(f"{k}={v}" for k, v in sorted(classes.items()))
        print(f"  {stem}: {n_feats} polygons  [{class_str}]  -> {out.name}")

    if missing:
        print(f"\nSkipped {len(missing)}: {missing}")
    print("\nDone.")


if __name__ == "__main__":
    main()
