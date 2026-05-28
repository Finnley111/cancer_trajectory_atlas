"""Convert new-format QuPath GeoJSON annotations (pixel coords at 10x) to the
ratio-coordinate JSON format expected by the pipeline.

New format (data/annotations/):
  - Pixel coordinates at 10x magnification (2x the 5x PNG scale)
  - No classification on some features (treated as Tumor)
  - Ignore* classification on artifact features

Old/pipeline format (data/old_annotations/):
  - Ratio coordinates in [0, 1] relative to the original full NDPI dims at 5x
  - classification.name present on every feature

Conversion:
  ratio_x = coord_x / (SCALE * original_full_width)
  ratio_y = coord_y / (SCALE * original_full_height)
  where SCALE = 2 (QuPath 10x vs 5x PNG).

Usage:
    python -m cancer_trajectory_atlas.converters.geojson_to_ratio_json \\
        --input-dir   data/annotations \\
        --output-dir  data/annotations_ratio \\
        --dims-json   data/MCF7_x5_cropped/slide_dimensions.json
"""
import argparse
import json
import sys
from pathlib import Path

SCALE = 2  # QuPath annotated at 10x; PNGs are 5x
DEFAULT_TUMOR_CLASS = {"name": "Tumor", "color": [200, 0, 0]}


def convert_file(src: Path, dst: Path, original_full_width: int, original_full_height: int) -> dict:
    """Convert one GeoJSON file to ratio-coordinate JSON.  Returns a summary dict."""
    with open(src) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features_in = data["features"]
    elif isinstance(data, list):
        features_in = data
    else:
        features_in = [data]

    denom_x = SCALE * original_full_width
    denom_y = SCALE * original_full_height

    features_out = []
    n_tumor = n_ignore = n_other = 0

    for feat in features_in:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})

        # Determine classification.
        cls = props.get("classification")
        if cls is None:
            cls = DEFAULT_TUMOR_CLASS
            n_tumor += 1
        else:
            name = cls.get("name", "") if isinstance(cls, dict) else cls
            if name == "Tumor":
                n_tumor += 1
            elif "Ignore" in name:
                n_ignore += 1
            else:
                n_other += 1

        # Convert coordinates.
        geom_type = geom.get("type", "")
        if geom_type == "Polygon":
            new_coords = [_convert_ring(ring, denom_x, denom_y)
                          for ring in geom["coordinates"]]
        elif geom_type == "MultiPolygon":
            new_coords = [[_convert_ring(ring, denom_x, denom_y) for ring in poly]
                          for poly in geom["coordinates"]]
        else:
            continue

        new_props = dict(props)
        new_props["classification"] = cls

        features_out.append({
            "type": "Feature",
            "id": feat.get("id", ""),
            "geometry": {"type": geom_type, "coordinates": new_coords},
            "properties": new_props,
        })

    output = {"type": "FeatureCollection", "features": features_out}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(output, f, indent=2)

    return {"tumor": n_tumor, "ignore": n_ignore, "other": n_other, "total": len(features_out)}


def _convert_ring(ring, denom_x, denom_y):
    return [[c[0] / denom_x, c[1] / denom_y] for c in ring]


def main():
    parser = argparse.ArgumentParser(description="Convert pixel-coord GeoJSON to ratio-coord JSON")
    parser.add_argument("--input-dir",  type=Path, required=True,
                        help="Directory containing *.geojson files (pixel coords at 10x)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination directory for converted *.json files")
    parser.add_argument("--dims-json",  type=Path, required=True,
                        help="slide_dimensions.json with original_full_width/height per slide")
    args = parser.parse_args()

    with open(args.dims_json) as f:
        dims = json.load(f)

    geojson_files = sorted(args.input_dir.glob("*.geojson"))
    if not geojson_files:
        print(f"ERROR: no .geojson files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(geojson_files)} files → {args.output_dir}/\n")
    missing_dims = []

    for src in geojson_files:
        stem = src.stem  # e.g. 6027-4L-2M-1
        dims_key = f"{stem}_x5.png"
        if dims_key not in dims:
            print(f"  SKIP {src.name}: no entry in dims_json for '{dims_key}'")
            missing_dims.append(stem)
            continue

        W = dims[dims_key]["original_full_width"]
        H = dims[dims_key]["original_full_height"]
        dst = args.output_dir / f"{stem}.json"

        summary = convert_file(src, dst, W, H)
        print(f"  {src.name} → {dst.name}  "
              f"(tumor={summary['tumor']}, ignore={summary['ignore']}, "
              f"other={summary['other']}, total={summary['total']})")

    if missing_dims:
        print(f"\nWARNING: {len(missing_dims)} slides skipped (no dimensions): {missing_dims}")

    print("\nDone.")
    print("NOTE: Ignore* polygons are preserved in the output.")
    print("      The pipeline silently skips them (not in label_order).")
    print("      Patches inside Ignore* regions will still appear in the UMAP.")
    print("      To actually exclude artifact patches, pipeline changes are needed.")


if __name__ == "__main__":
    main()
