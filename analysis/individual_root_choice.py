"""Pass 2 of the per-slide timepoint run: choose each slide's DPT root cluster.

================================================================================
NON-COMPARABILITY CONSTRAINT — READ BEFORE USING ANY NUMBER THIS PRODUCES
================================================================================
``run_individual.py`` fits a SEPARATE PCA basis per slide, with no patch cap, no
feature cache, no batch correction, and single-root cluster-anchored DPT.
PSEUDOTIME VALUES FROM ONE SLIDE ARE NOT COMPARABLE TO ANY OTHER SLIDE, NOR TO
ANY PER-SECTION OR PROJECTED RESULT ELSEWHERE IN THIS PROJECT. The cluster
labels this module reads, and the root it picks, are likewise per-slide objects:
"cluster 3" on one slide has nothing to do with "cluster 3" on another.

This module answers only "which cluster within THIS slide should anchor THIS
slide's ordering". It says nothing about differences between slides or
timepoints, and does not address the 100%-extrapolation projection finding or
the staining differences between this cohort and the 2M cohort.
================================================================================

WHY THIS MODULE EXISTS
----------------------
``run_individual.py``'s ``--root-cluster`` defaults to the LOWEST-NUMBERED Leiden
cluster. Leiden IDs are arbitrary labels, so that default is an arbitrary origin.
Choosing a principled root needs cluster labels, which only exist after a full
GPU run — and ``run_individual.py`` saves no ``adata`` and computes no
morphological features, so nothing can be re-anchored after the fact. Hence three
passes:

    PASS 1 (GPU)  run_individual.py, arbitrary default root -> results.csv
    PASS 2 (CPU)  THIS MODULE -> root_choices.json
    PASS 3 (GPU)  run_individual.py --root-cluster N -> the run that is kept

``run_individual.py`` is neither modified nor bypassed at any point.

THE RULE, FIXED IN ADVANCE
--------------------------
Root = the Leiden cluster with the LOWEST MEDIAN ``nuclear_density``, computed by
``validation/morphological_features.compute_nuclear_density_quick`` — the same
function the atlas pipeline uses to rank its own DPT roots. Choosing the same
quantity keeps this consistent with the main pipeline instead of inventing a
per-slide rule. Ties break to the lowest cluster ID (arbitrary but reproducible,
and independent of every downstream quantity).

Every cluster's median density is written out, not just the winner, so the choice
is auditable rather than asserted.

WHY PATCHES ARE CROPPED, NOT RE-EXTRACTED
-----------------------------------------
Patches are cropped straight from the PNG at the (x, y) already stored in Pass 1's
results.csv. That sidesteps any assumption about ``get_patches_from_array`` being
re-run deterministically: the coordinates are read, not regenerated.

This is exact ONLY because Pass 1 runs with ``--stain-method none``. With any
normalizer the pipeline would have seen normalised pixels while this module reads
raw ones. The module therefore REFUSES to run unless ``--stain-method-was none``
is passed, rather than silently computing densities on the wrong pixels.

READ-ONLY with respect to Pass 1's output tree. Writes only --output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

NONCOMPARABILITY = (
    "PER-SLIDE RESULT. run_individual.py fits a separate PCA basis per slide, "
    "with no patch cap, no feature cache, no batch correction and single-root "
    "cluster-anchored DPT. Pseudotime values from one slide are NOT comparable "
    "to any other slide, nor to any per-section or projected result elsewhere "
    "in this project. Cluster IDs are per-slide labels and carry no meaning "
    "across slides."
)


def _png_for(png_dir: Path, slide_name: str) -> Path:
    """Pass 1's results.csv stores the PNG stem, e.g. '6054-4L-8W_x5'."""
    p = png_dir / f"{slide_name}.png"
    if p.exists():
        return p
    hits = sorted(png_dir.glob(f"{slide_name}.*"))
    if not hits:
        raise FileNotFoundError(
            f"No PNG for slide '{slide_name}' in {png_dir}. Pass 2 crops patches "
            "from the same image Pass 1 read; without it the root cannot be chosen."
        )
    return hits[0]


def choose_root_for_slide(
    slide_dir: Path,
    png_dir: Path,
    patch_size: int = 112,
    max_patches_per_cluster: int | None = None,
    seed: int = 42,
) -> dict:
    """Return the root-cluster decision for one Pass 1 slide directory."""
    csv = slide_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found — Pass 1 did not complete for this slide.")

    df = pd.read_csv(csv)
    for c in ("x", "y", "cluster", "slide_name"):
        if c not in df.columns:
            raise KeyError(f"{csv} lacks required column '{c}'.")

    slide_name = str(df["slide_name"].iloc[0])
    if df["slide_name"].nunique() != 1:
        raise ValueError(
            f"{csv} contains {df['slide_name'].nunique()} slide names. This module "
            "expects a single-slide run_individual.py output."
        )

    png = _png_for(png_dir, slide_name)
    img = np.array(Image.open(png).convert("RGB"))
    h, w = img.shape[:2]

    from ..validation.morphological_features import compute_nuclear_density_quick

    clusters = sorted(int(c) for c in df["cluster"].unique() if int(c) != -1)
    if len(clusters) < 2:
        raise ValueError(
            f"{slide_name}: only {len(clusters)} usable cluster(s). run_individual "
            "needs >=2 for diffusion pseudotime; nothing to anchor."
        )

    rng = np.random.default_rng(seed)
    per_cluster: dict[int, dict] = {}
    n_oob_total = 0

    for c in clusters:
        sub = df[df["cluster"] == c]
        idx = np.arange(len(sub))
        if max_patches_per_cluster and len(idx) > max_patches_per_cluster:
            # Subsampling only bounds runtime on very large clusters. The median is
            # what is compared, so a uniform subsample estimates it without bias.
            idx = rng.choice(idx, size=max_patches_per_cluster, replace=False)
            idx.sort()
        xs = sub["x"].values[idx].astype(int)
        ys = sub["y"].values[idx].astype(int)

        crops, n_oob = [], 0
        for x, y in zip(xs, ys):
            if x < 0 or y < 0 or x + patch_size > w or y + patch_size > h:
                n_oob += 1
                continue
            crops.append(img[y:y + patch_size, x:x + patch_size])
        n_oob_total += n_oob

        if not crops:
            per_cluster[c] = {"n_patches": int(len(sub)), "n_measured": 0,
                              "median_nuclear_density": None,
                              "note": "every sampled patch fell outside the image"}
            continue

        dens = compute_nuclear_density_quick(np.stack(crops))
        finite = dens[np.isfinite(dens)]
        per_cluster[c] = {
            "n_patches": int(len(sub)),
            "n_sampled": int(len(crops)),
            "n_measured": int(finite.size),
            "n_failed": int(len(crops) - finite.size),
            "median_nuclear_density": float(np.median(finite)) if finite.size else None,
            "mean_nuclear_density": float(finite.mean()) if finite.size else None,
        }

    usable = {c: v for c, v in per_cluster.items()
              if v.get("median_nuclear_density") is not None}
    if not usable:
        raise ValueError(
            f"{slide_name}: no cluster yielded a finite median nuclear density. "
            "Segmentation failed everywhere; refusing to guess a root."
        )

    # Lowest median density; ties -> lowest cluster ID (reproducible, and
    # independent of every downstream quantity).
    root = min(usable, key=lambda c: (usable[c]["median_nuclear_density"], c))
    ordered = sorted(usable, key=lambda c: (usable[c]["median_nuclear_density"], c))
    tied = [c for c in usable
            if usable[c]["median_nuclear_density"] == usable[root]["median_nuclear_density"]]

    if img.size:
        del img

    return {
        "slide_name": slide_name,
        "png": str(png),
        "png_width": int(w),
        "png_height": int(h),
        "n_patches": int(len(df)),
        "n_clusters": len(clusters),
        "clusters": {str(c): per_cluster[c] for c in clusters},
        "root_cluster": int(root),
        "root_median_nuclear_density": usable[root]["median_nuclear_density"],
        "cluster_order_by_density": [int(c) for c in ordered],
        "n_tied_at_minimum": len(tied),
        "tie_broken_by_lowest_cluster_id": len(tied) > 1,
        "n_patches_out_of_bounds": int(n_oob_total),
        "rule": ("root = Leiden cluster with the lowest median nuclear_density "
                 "(validation/morphological_features.compute_nuclear_density_quick, "
                 "the same function the atlas uses to rank its own DPT roots); "
                 "ties -> lowest cluster ID"),
        "noncomparability": NONCOMPARABILITY,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass1-dir", type=Path, required=True,
                    help="Pass 1 output root containing one subdirectory per slide.")
    ap.add_argument("--png-dir", type=Path, required=True,
                    help="Directory holding the slide PNGs Pass 1 read.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write root_choices.json (NEW file).")
    ap.add_argument("--patch-size", type=int, default=112,
                    help="MUST match Pass 1's --patch-size or the wrong window is "
                         "cropped, silently. (default: 112)")
    ap.add_argument("--max-patches-per-cluster", type=int, default=2000,
                    help="Cap patches sampled per cluster, to bound runtime. The "
                         "median is estimated without bias from a uniform "
                         "subsample. 0 = no cap. (default: 2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stain-method-was", type=str, required=True,
                    help="The --stain-method Pass 1 ran with. MUST be 'none'. This "
                         "module crops raw pixels from the PNG; under any normalizer "
                         "the pipeline saw different pixels and the densities would "
                         "not correspond to the clusters they are being attributed to.")
    args = ap.parse_args()

    print("=" * 74)
    print("PASS 2 — per-slide DPT root cluster choice")
    print("=" * 74)
    print(NONCOMPARABILITY)
    print("=" * 74)

    if args.stain_method_was != "none":
        sys.exit(
            f"\nERROR: --stain-method-was is '{args.stain_method_was}', not 'none'.\n"
            "  Pass 2 crops RAW pixels from the PNG. If Pass 1 normalised the slide "
            "first,\n  the pipeline clustered normalised pixels while this module "
            "would measure\n  raw ones, so the densities would not correspond to the "
            "clusters they are\n  attributed to. Re-run Pass 1 with --stain-method "
            "none, or extend this\n  module to apply the same normalizer.")

    if not args.pass1_dir.is_dir():
        sys.exit(f"ERROR: --pass1-dir not found: {args.pass1_dir}")
    if args.output.exists():
        sys.exit(f"ERROR: {args.output} already exists. Refusing to overwrite an "
                 "existing root-choice record; move it aside or pick a new path.")

    slide_dirs = sorted(d for d in args.pass1_dir.iterdir()
                        if d.is_dir() and (d / "results.csv").exists())
    if not slide_dirs:
        sys.exit(f"ERROR: no slide directories with results.csv under {args.pass1_dir}")

    print(f"\nSlides with Pass 1 output: {len(slide_dirs)}\n")
    cap = args.max_patches_per_cluster or None

    choices, failures = {}, {}
    for i, d in enumerate(slide_dirs, 1):
        print(f"[{i}/{len(slide_dirs)}] {d.name}")
        try:
            rec = choose_root_for_slide(d, args.png_dir, args.patch_size, cap, args.seed)
            choices[rec["slide_name"]] = rec
            dens = rec["root_median_nuclear_density"]
            print(f"    root cluster {rec['root_cluster']} of {rec['n_clusters']} "
                  f"(median nuclear density {dens:.5g})"
                  + ("  [TIE broken by lowest ID]" if rec["tie_broken_by_lowest_cluster_id"] else ""))
            if rec["n_patches_out_of_bounds"]:
                print(f"    NOTE: {rec['n_patches_out_of_bounds']} sampled patch(es) "
                      "fell outside the image and were skipped")
        except Exception as e:                                    # noqa: BLE001
            # Per-slide failure must not abort the batch.
            print(f"    FAILED: {type(e).__name__}: {e}")
            failures[d.name] = f"{type(e).__name__}: {e}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "noncomparability": NONCOMPARABILITY,
        "rule": ("root = Leiden cluster with the lowest median nuclear_density; "
                 "ties -> lowest cluster ID"),
        "pass1_dir": str(args.pass1_dir),
        "png_dir": str(args.png_dir),
        "patch_size": args.patch_size,
        "max_patches_per_cluster": cap,
        "seed": args.seed,
        "n_slides_resolved": len(choices),
        "n_slides_failed": len(failures),
        "failures": failures,
        "choices": choices,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"Resolved {len(choices)} slide(s); {len(failures)} failed.")
    print(f"Wrote {args.output}")
    print(NONCOMPARABILITY)
    print("=" * 74)


if __name__ == "__main__":
    main()
