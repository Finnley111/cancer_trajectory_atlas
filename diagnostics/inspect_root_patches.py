"""Visual + quantitative inspection of the DPT root patches.

WHY THIS EXISTS
    The production pseudotime is rooted at the 20 lowest-nuclear_density patches
    (analysis/diffusion.py:165). eccentricity_check Task 0 found that in 2M-2 ALL
    twenty roots have nuclear_density exactly 0.0, drawn from 21 such patches.

    compute_nuclear_density_quick returns 0.0 for two very different situations:
    genuinely acellular tissue, AND any patch whose segmentation raised — the
    handler at validation/morphological_features.py:177-178 is a bare
    `except: pass` that leaves the entry at its initialised 0.0. Those two cases
    are indistinguishable in the stored array but mean opposite things. If the
    roots are background, lumen, fat or segmentation failures, then pseudotime 0
    is anchored on an artifact and the early->late orientation of the axis is not
    interpretable.

    root_sensitivity already showed the roots do NOT determine the ordering
    (uniformly random 20-root sets reproduce production pseudotime at |rho|
    0.78-0.89) — they determine only which END is called "early". That is exactly
    the thing this script tests.

WHAT IT PRODUCES
    For each section:
      root_patches_<section>.png     20 root patches, annotated
      root_context_<section>.png     the same patches with surrounding tissue
      control_patches_<section>.png  20 median-density patches, same layout
      root_patches_<section>/        individual full-resolution crops
    and a shared root_patch_report.{md,json} with per-patch quantitative measures.

    The control panel is not decoration. "Is this background?" is not answerable
    from the root patches alone — you need to see what an ordinary patch from the
    same slides looks like at the same magnification.

READS (read-only): <run_dir>/results.csv and the source slide PNGs.
WRITES: a new output directory only. No pipeline module is modified or re-run.

NOTE ON PIXELS
    run_all.py:324 stain-normalises each slide BEFORE patching, so the pipeline
    saw normalised pixels. This script shows RAW pixels by default, because the
    question is what tissue is present and normalisation is itself a candidate
    failure. Pass --stain-normalised to reproduce what the pipeline actually fed
    to the segmenter; the quantitative columns are computed on whatever is shown.

Usage:
    python -m cancer_trajectory_atlas.diagnostics.inspect_root_patches \
        --sections 2M-1 2M-2 \
        --run-dirs $SCRATCH/results/per_section/atlas_2M-1 \
                   $SCRATCH/results/per_section/atlas_2M-2 \
        --png-dir  $SCRATCH/data/png \
        --output-dir $SCRATCH/results/root_patches
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # whole-slide PNGs exceed the default bomb guard

# Matches features/patching.py defaults and the white_thresh used by its HSV filter.
DEFAULT_PATCH_SIZE = 112
WHITE_THRESH = 220
GRID = (4, 5)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(run_dir: Path) -> pd.DataFrame:
    """Load results.csv. Row order matches adata row order — both are built from
    the same concatenated per-slide arrays in run_all.py."""
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. This diagnostic reuses an existing per-section run."
        )
    df = pd.read_csv(csv)
    missing = [c for c in ("x", "y", "slide_name", "nuclear_density", "pseudotime")
               if c not in df.columns]
    if missing:
        raise KeyError(f"{csv} is missing columns {missing}.")
    return df


def pick_roots(df: pd.DataFrame, n_roots: int, explicit: list[int] | None) -> list[int]:
    """Reproduce diffusion.py:165 — argsort(nuclear_density)[:n_roots].

    NOTE: production selects on compute_nuclear_density_quick(all_patches)
    (run_all.py:564), a separate float32 pass, while results.csv stores the value
    from the full feature pass. The rule is identical; array identity is not
    verifiable because compute_dpt_multi_root never persists root_candidates. Pass
    --root-indices to use the exact indices from an eccentricity_check run instead.
    """
    if explicit:
        return [int(i) for i in explicit]
    return [int(i) for i in np.argsort(df["nuclear_density"].values)[:n_roots]]


# ── Cropping ──────────────────────────────────────────────────────────────────

def crop(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """Crop the patch the pipeline extracted: img[y:y+size, x:x+size].

    coords are appended as (x, y) at features/patching.py:277, where x indexes
    columns and y indexes rows.
    """
    return img[y:y + size, x:x + size]


def crop_context(img: np.ndarray, x: int, y: int, size: int, factor: int):
    """Wider crop centred on the patch, plus the patch's box within it.

    Returns (context_image, (box_x, box_y)) so the patch can be outlined. Clamped
    at the slide edges, which is itself informative — a root patch sitting hard
    against an edge is a strong hint the origin is a boundary artifact.
    """
    pad = (factor - 1) * size // 2
    h, w = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + size + pad), min(h, y + size + pad)
    return img[y0:y1, x0:x1], (x - x0, y - y0)


# ── Quantitative measures ─────────────────────────────────────────────────────

def patch_stats(patch: np.ndarray) -> dict:
    """Measures that separate 'background' from 'tissue the segmenter missed'.

    A patch that is mostly white is background and should never have been a root.
    A patch with low white fraction and normal saturation IS tissue — which means
    nuclear_density = 0 came from a segmentation failure, not from absent nuclei.
    That distinction is the whole point of this diagnostic.
    """
    if patch.size == 0:
        return {"empty": True}
    rgb = patch[..., :3].astype(float)
    gray = rgb.mean(axis=2)

    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)

    return {
        "empty": False,
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "frac_pixels_white": float((gray > WHITE_THRESH).mean()),
        "mean_saturation": float(sat.mean()),
        "frac_low_saturation": float((sat < 0.05).mean()),
        "shape": list(patch.shape[:2]),
    }


def classify(st: dict) -> str:
    """Plain-language call. Deliberately conservative: anything that is clearly
    tissue is reported as a segmentation failure, because that is the finding
    that would invalidate the root rule while looking innocuous in the array."""
    if st.get("empty"):
        return "EMPTY CROP — coordinates fall outside the image"
    if st["frac_pixels_white"] >= 0.70:
        return "BACKGROUND — >=70% white pixels; not tissue"
    if st["frac_pixels_white"] >= 0.40:
        return "MOSTLY BACKGROUND — 40-70% white; edge or lumen"
    if st["std_intensity"] < 8.0:
        return "FEATURELESS — near-uniform; fat, lumen or blur"
    return ("TISSUE PRESENT — nuclear_density 0 here means the segmenter found "
            "nothing in real tissue, i.e. a silent failure")


# ── Figures ───────────────────────────────────────────────────────────────────

def contact_sheet(items: list[dict], title: str, path: Path, key: str = "patch",
                  boxes: bool = False) -> None:
    rows, cols = GRID
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax, it in zip(axes.ravel(), items):
        img = it.get(key)
        if img is None or getattr(img, "size", 0) == 0:
            ax.text(0.5, 0.5, "no crop", ha="center", va="center", fontsize=7)
        else:
            ax.imshow(img)
            if boxes and it.get("box") is not None:
                bx, by = it["box"]
                s = it["patch_size"]
                ax.add_patch(plt.Rectangle((bx, by), s, s, fill=False,
                                           edgecolor="#D62728", lw=1.6))
        st = it["stats"]
        lab = (f"#{it['index']}  {it['slide_name'][:16]}\n"
               f"nd={it['nuclear_density']:.2e}  pt={it['pseudotime']:.3f}\n"
               f"white={st.get('frac_pixels_white', float('nan')):.0%}  "
               f"sd={st.get('std_intensity', float('nan')):.0f}")
        ax.set_title(lab, fontsize=6)
        ax.axis("off")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ── Per-section driver ────────────────────────────────────────────────────────

def inspect_section(section: str, run_dir: Path, png_dir: Path, out_dir: Path,
                    n_roots: int, patch_size: int, context: int,
                    explicit_roots: list[int] | None,
                    stain_normalised: bool) -> dict:
    df = load_results(run_dir)
    roots = pick_roots(df, n_roots, explicit_roots)

    nd = df["nuclear_density"].values
    # Controls: patches at the MEDIAN of the density distribution, so the eye has
    # a same-magnification reference for "ordinary tissue from these slides".
    order = np.argsort(nd)
    mid = len(order) // 2
    controls = [int(i) for i in order[mid - n_roots // 2: mid - n_roots // 2 + n_roots]]

    indiv = out_dir / f"root_patches_{section}"
    indiv.mkdir(parents=True, exist_ok=True)

    slide_cache: dict[str, np.ndarray] = {}

    def load_slide(name: str) -> np.ndarray | None:
        if name in slide_cache:
            return slide_cache[name]
        p = png_dir / f"{name}.png"
        if not p.exists():
            hits = list(png_dir.glob(f"{name}.*"))
            if not hits:
                print(f"    WARNING: no slide image for '{name}' in {png_dir}")
                slide_cache[name] = None
                return None
            p = hits[0]
        arr = np.array(Image.open(p).convert("RGB"))
        if stain_normalised:
            try:
                from ..features.stain_norm import normalize_slide  # optional
                arr = normalize_slide(arr, None, name)
            except Exception as exc:
                print(f"    WARNING: stain normalisation unavailable ({exc}); raw pixels")
        slide_cache[name] = arr
        return arr

    def gather(indices: list[int], save: bool) -> list[dict]:
        items = []
        for rank, i in enumerate(indices):
            row = df.iloc[i]
            name = str(row["slide_name"])
            x, y = int(row["x"]), int(row["y"])
            img = load_slide(name)
            if img is None:
                p = np.zeros((0, 0, 3), dtype=np.uint8)
                ctx, box = p, None
            else:
                p = crop(img, x, y, patch_size)
                ctx, box = crop_context(img, x, y, patch_size, context)
            st = patch_stats(p)
            if save and p.size:
                Image.fromarray(p).save(indiv / f"root{rank:02d}_idx{i}_{name}.png")
            items.append({
                "index": i, "rank": rank, "slide_name": name, "x": x, "y": y,
                "nuclear_density": float(row["nuclear_density"]),
                "pseudotime": float(row["pseudotime"]),
                "stats": st, "classification": classify(st),
                "patch": p, "context": ctx, "box": box, "patch_size": patch_size,
            })
        return items

    print(f"  [{section}] cropping {len(roots)} root patches ...")
    root_items = gather(roots, save=True)
    print(f"  [{section}] cropping {len(controls)} control patches ...")
    ctrl_items = gather(controls, save=False)

    contact_sheet(root_items,
                  f"{section}: the {n_roots} DPT ROOT patches (lowest nuclear_density)",
                  out_dir / f"root_patches_{section}")
    contact_sheet(root_items,
                  f"{section}: root patches in context ({context}x window, red box = patch)",
                  out_dir / f"root_context_{section}", key="context", boxes=True)
    contact_sheet(ctrl_items,
                  f"{section}: CONTROL — {n_roots} patches at MEDIAN nuclear_density",
                  out_dir / f"control_patches_{section}")

    counts: dict[str, int] = {}
    for it in root_items:
        head = it["classification"].split(" —")[0]
        counts[head] = counts.get(head, 0) + 1

    n_tissue = counts.get("TISSUE PRESENT", 0)
    n_bad = sum(v for k, v in counts.items() if k != "TISSUE PRESENT")

    verdict = (
        f"ROOTS ARE NOT TISSUE — {n_bad} of {len(root_items)} root patches are "
        "background, mostly-background or featureless. The pseudotime origin is an "
        "imaging artifact, so the early->late ORIENTATION of the axis is not "
        "interpretable. The ordering itself is unaffected (random roots reproduce "
        "it at |rho| 0.78-0.89) — it is the direction that fails."
        if n_bad > len(root_items) / 2 else
        f"ROOTS ARE REAL TISSUE — {n_tissue} of {len(root_items)} root patches "
        "contain tissue with normal saturation and contrast, so nuclear_density = 0 "
        "came from SEGMENTATION FAILURE rather than from absent nuclei "
        "(morphological_features.py:177-178 swallows the exception). The origin is "
        "still not a biological 'earliest' state, but for a different reason: it is "
        "wherever the segmenter breaks."
        if n_tissue > len(root_items) / 2 else
        f"MIXED — {n_tissue} tissue, {n_bad} background/featureless of "
        f"{len(root_items)}. Inspect the contact sheet directly; no single "
        "explanation covers the root set."
    )

    def strip(items):
        return [{k: v for k, v in it.items()
                 if k not in ("patch", "context", "box")} for it in items]

    return {
        "section": section,
        "n_roots": len(root_items),
        "patch_size": patch_size,
        "root_rule": "np.argsort(nuclear_density)[:n_roots]  (diffusion.py:165)",
        "roots_from": "explicit --root-indices" if explicit_roots else "reconstructed",
        "pixels_shown": "stain-normalised" if stain_normalised else "raw",
        "classification_counts": counts,
        "n_zero_density_in_section": int((nd <= 0).sum()),
        "root_density_range": [float(nd[roots].min()), float(nd[roots].max())],
        "roots": strip(root_items),
        "controls": strip(ctrl_items),
        "verdict": verdict,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(out_dir: Path, results: dict) -> None:
    L = ["# DPT root patch inspection", "",
         "Are the 20 lowest-`nuclear_density` patches — the DPT origin — actually",
         "tissue? `compute_nuclear_density_quick` returns 0.0 both for acellular",
         "tissue and for any patch whose segmentation raised",
         "(`validation/morphological_features.py:177-178`), so the stored array",
         "cannot tell those apart. The images can.", "",
         "Root choice does not affect the pseudotime ORDERING — random 20-root sets",
         "reproduce it at |rho| 0.78-0.89 — only which end is called *early*.", ""]

    for section, r in results.items():
        L += [f"## {section}", "",
              f"- Roots: {r['n_roots']} ({r['roots_from']}), rule `{r['root_rule']}`",
              f"- Patches with zero nuclear_density in this section: "
              f"**{r['n_zero_density_in_section']}**",
              f"- Root density range: [{r['root_density_range'][0]:.3e}, "
              f"{r['root_density_range'][1]:.3e}]",
              f"- Pixels shown: {r['pixels_shown']}, patch size {r['patch_size']}px", "",
              "**Classification of the root patches**", ""]
        for k, v in sorted(r["classification_counts"].items(), key=lambda kv: -kv[1]):
            L.append(f"- {k}: **{v}**")
        L += ["", f"**Verdict.** {r['verdict']}", "",
              "| # | index | slide | x | y | nuclear_density | pseudotime | white% | sd | call |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for it in r["roots"]:
            st = it["stats"]
            L.append(
                f"| {it['rank']} | {it['index']} | {it['slide_name']} | {it['x']} | "
                f"{it['y']} | {it['nuclear_density']:.3e} | {it['pseudotime']:.4f} | "
                f"{st.get('frac_pixels_white', float('nan')):.0%} | "
                f"{st.get('std_intensity', float('nan')):.1f} | "
                f"{it['classification'].split(' —')[0]} |")
        L += ["", f"Figures: `root_patches_{section}.png`, "
                  f"`root_context_{section}.png`, `control_patches_{section}.png`", ""]

    L += ["## How to read this", "",
          "- **Background / mostly background** — the root rule is selecting empty",
          "  space. The axis orientation is an artifact; re-root excluding these.",
          "- **Tissue present** — worse in a sense: the segmenter silently failed on",
          "  real tissue, so `nuclear_density = 0` is a bug, not a measurement, and",
          "  it also contaminates the feature wherever else it occurs.",
          "- Compare against `control_patches_*.png` before judging. Median-density",
          "  patches from the same slides are the reference for what normal looks like.", ""]
    (out_dir / "root_patch_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--png-dir", type=Path, required=True,
                    help="Directory of source slide PNGs, named <slide_name>.png")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-roots", type=int, default=20)
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                    help=f"Must match the run's --patch-size (default {DEFAULT_PATCH_SIZE})")
    ap.add_argument("--context", type=int, default=7,
                    help="Context window as a multiple of patch size (default 7)")
    ap.add_argument("--root-indices", nargs="+", type=int, default=None,
                    help="Exact root indices (e.g. from eccentricity_check Task 0). "
                         "Applies to the FIRST section only; omit to reconstruct.")
    ap.add_argument("--stain-normalised", action="store_true",
                    help="Show the normalised pixels the pipeline fed the segmenter "
                         "instead of raw slide pixels.")
    args = ap.parse_args()

    if len(args.sections) != len(args.run_dirs):
        ap.error("--sections and --run-dirs must match in length and order")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  DPT root patch inspection")
    print("=" * 64)

    results = {}
    for k, (section, run_dir) in enumerate(zip(args.sections, args.run_dirs)):
        print(f"\n  {section}  <-  {run_dir}")
        results[section] = inspect_section(
            section, Path(run_dir), args.png_dir, args.output_dir,
            args.n_roots, args.patch_size, args.context,
            args.root_indices if k == 0 else None,
            args.stain_normalised,
        )
        print(f"    {results[section]['verdict']}")

    with open(args.output_dir / "root_patch_report.json", "w") as f:
        json.dump(results, f, indent=2)
    write_report(args.output_dir, results)

    print(f"\n  JSON:     {args.output_dir / 'root_patch_report.json'}")
    print(f"  Markdown: {args.output_dir / 'root_patch_report.md'}")
    print("\n" + "=" * 64)
    for s, r in results.items():
        print(f"\n  [{s}]\n  {r['verdict']}")


if __name__ == "__main__":
    main()
