"""Convert QuPath GeoJSON annotations to ratio-coordinate JSON.

This is the producer of ``data/annotations_ratio/``, which is what the pipeline
actually consumes. It is a manual, one-off tool: it takes no arguments and uses
hardcoded relative paths, so it MUST be run from the repository root.

    cd ~/cancer_trajectory_atlas && python converters/batch_convert.py

Inputs
------
``./data/annotations/*.geojson``
    QuPath exports, in absolute full-NDPI pixel coordinates.
``./converters/img_dims.txt``
    Per-slide full-NDPI level-0 dimensions, one ``<slide-id>: w=W h=H`` per line.

Output
------
``./data/annotations_ratio/*.json``
    Same FeatureCollection structure, every coordinate divided by (W, H) and
    rounded to 6 decimals, i.e. ratios in [0, 1] relative to the FULL NDPI width
    (which includes both side-by-side slide copies). Left-half annotations
    therefore have x in [0, 0.5]. See ``features/patching.py:load_roi_polygons``
    for how that is mapped back to cropped-PNG pixel space.

Round-trip invariant
--------------------
This script divides by ``converters/img_dims.txt``; the pipeline multiplies by
``original_full_width`` from ``slide_dimensions.json`` (written by ``run_all.py
--convert``) or, as a fallback, ``data/slide_registry.py:KNOWN_NDPI_DIMENSIONS``.
Verified 2026-08-12: ``img_dims.txt`` and ``KNOWN_NDPI_DIMENSIONS`` hold the same
16 keys with identical values, so the round trip is exact and the two sources are
interchangeable. If a slide is ever added, it must be added to BOTH.

Do not archive this file. It is the only in-repo record of how the ratio
annotations were generated, and it is part of the raw-data-to-results path.
"""

import json
import os
import re

def convert_coordinates(coords, width, height):
    """Recursively convert absolute coordinates to relative ratios."""
    if isinstance(coords[0], (int, float)):
        return [round(coords[0] / width, 6), round(coords[1] / height, 6)]
    else:
        return [convert_coordinates(c, width, height) for c in coords]

def main():
    dims_file = './converters/img_dims.txt'
    input_dir = './data/annotations/'
    output_dir = './data/annotations_ratio/'
    
    # Safely create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Parse the dimensions from the text file
    dimensions = {}
    try:
        with open(dims_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Look for the pattern: ID: w=XXXX h=YYYY
                match = re.search(r'(.*?):\s*w=(\d+)\s*h=(\d+)', line)
                if match:
                    img_name = match.group(1).strip()
                    w = float(match.group(2))
                    h = float(match.group(3))
                    dimensions[img_name] = {'w': w, 'h': h}
    except FileNotFoundError:
        print(f"Error: Could not find {dims_file}. Please ensure it is in the same folder.")
        return

    # 2. Process every GeoJSON file in the input directory
    processed_count = 0
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith('.geojson'):
                # Match the filename to the ID in the dimensions text file
                core_id = None
                for key in dimensions.keys():
                    if key in filename:
                        core_id = key
                        break
                
                if not core_id:
                    print(f"Skipping {filename}: Could not find matching dimensions in {dims_file}.")
                    continue
                
                w = dimensions[core_id]['w']
                h = dimensions[core_id]['h']
                
                input_path = os.path.join(input_dir, filename)
                output_filename = filename.replace('.geojson', '.json')
                output_path = os.path.join(output_dir, output_filename)
                
                # Load, convert, and save
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                if data.get("type") == "FeatureCollection":
                    for feature in data.get("features", []):
                        geom = feature.get("geometry", {})
                        if geom and "coordinates" in geom:
                            geom["coordinates"] = convert_coordinates(geom["coordinates"], w, h)
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Processed: {filename} (w={w}, h={h}) -> {output_filename}")
                processed_count += 1
                
    except FileNotFoundError:
        print(f"Error: Could not find the {input_dir} folder. Make sure your exported files are there.")
        return

    print(f"\nSuccess! Converted {processed_count} files and saved them to {output_dir}")

if __name__ == "__main__":
    main()