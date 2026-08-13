"""Root-patch contact sheets for the v3 root/filter experiment.

WHY A SIBLING RATHER THAN AN EXTENSION OF inspect_root_patches.py
----------------------------------------------------------------
``diagnostics/inspect_root_patches.py`` RE-DERIVES its roots as
``argsort(nuclear_density)[:n]`` from results.csv. That is wrong for the v3
configs in two ways: Configs A and C do not use the density rule at all, and even
for the density configs the re-derivation is not guaranteed to reproduce the
indices the run actually used (production ranks on a separate
``compute_nuclear_density_quick`` pass, and ties are resolved by argsort order).

This module instead READS ``adata.uns['dpt_root_candidates']`` — the roots the run
genuinely used, persisted by ``compute_dpt_multi_root``. That works uniformly for
v2 and for all three v3 configs, and it removes the re-derivation risk entirely.
``inspect_root_patches.py`` is NOT modified; it remains as provenance for the
earlier 2M-1/2M-2 inspection.

WHAT IT PRODUCES, per run
    roots_<label>_patches.{png,pdf}   20 native 112x112 crops, labelled 4x5 sheet
    roots_<label>_context.{png,pdf}   1500x1500 px windows, patch outlined
    roots_<label>/                    the individual crops at native resolution
    roots_<label>.json                per-root measures + the label table

Labels carry root index, slide, (x, y), nuclear_density and nucleus count; for
holeyness-rooted configs they additionally carry the duct's hole % and duct ID,
read from ``<run_dir>/holeyness_roots.json``.

PRESENTED NEUTRALLY. This module deliberately prints no judgement about whether
a root looks early or late, and no classification of tissue vs artifact. It shows
the images and the numbers. ``inspect_root_patches.py:classify`` exists if you
want the background-vs-tissue call; it is not invoked here.

READS (read-only): <run_dir>/adata_full.h5ad, <run_dir>/results.csv,
                   <run_dir>/holeyness_roots.json (optional), and the slide PNGs.
WRITES: a new output directory only. No run tree is modified.

Usage:
    python -m cancer_trajectory_atlas.diagnostics.inspect_roots_v3 \\
        --labels    v2 v3a v3b v3c \\
        --run-dirs  $SCRATCH/results/per_section_v2/atlas_2M-1 \\
                    $SCRATCH/results/per_section_v3a_holeyroot/atlas_2M-1 \\
                    $SCRATCH/results/per_section_v3b_relaxed/atlas_2M-1 \\
                    $SCRATCH/results/per_section_v3c_both/atlas_2M-1 \\
        --png-dir   $SCRATCH/data/MCF7_x5_cropped \\
        --output-dir $SCRATCH/results/v3_root_experiment/root_sheets
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

Image.MAX_IMAGE_PIXELS = None

GRID = (4, 5)
DPI = 300
CONTEXT_PX_DEFAULT = 1500
WHITE_THRESH = 220


# ── Loading ───────────────────────────────────────────────────────────────────

def load_root_indices(run_dir: Path) -> tuple[list[int], str]:
    """Read the roots the run ACTUALLY used, plus the source tag.

    Never re-derives. If dpt_root_candidates is absent the run predates the
    persistence added in v2 and cannot be inspected reliably — that is an error,
    not something to paper over with a guess.
    """
    import anndata as ad

    h5 = run_dir / "adata_full.h5ad"
    if not h5.exists():
        raise FileNotFoundError(f"{h5} not found — cannot read the root set.")
    adata = ad.read_h5ad(h5, backed="r")
    if "dpt_root_candidates" not in adata.uns:
        raise KeyError(
            f"{h5} has no uns['dpt_root_candidates']. That run predates root "
            "persistence; its root set is not recoverable and must not be guessed."
        )
    roots = [int(i) for i in np.asarray(adata.uns["dpt_root_candidates"]).ravel()]
    source = str(adata.uns.get("dpt_root_source", "nuclear_density (pre-v3 run)"))
    return roots, source


def load_results(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found.")
    return pd.read_csv(csv)


def load_holeyness(run_dir: Path) -> dict[int, dict]:
    """Map patch_index -> duct info, when this run was holeyness-rooted."""
    p = run_dir / "holeyness_roots.json"
    if not p.exists():
        return {}
    with open(p) as f:
        rep = json.load(f)
    return {int(r["patch_index"]): r for r in rep.get("selected_roots", [])}


# ── Cropping ──────────────────────────────────────────────────────────────────

def crop_patch(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """The exact window the pipeline extracted: img[y:y+size, x:x+size]."""
    return img[y:y + size, x:x + size]


def crop_context(img: np.ndarray, x: int, y: int, size: int, window: int):
    """`window` x `window` px crop centred on the patch.

    Returns (image, (box_x, box_y), clamped). Clamped at the slide edges, and the
    clamp is REPORTED rather than silently padded — a root sitting hard against
    an edge is itself informative.
    """
    pad = (window - size) // 2
    h, w = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + size + pad), min(h, y + size + pad)
    clamped = (x0 == 0 or y0 == 0 or x1 == w or y1 == h)
    return img[y0:y1, x0:x1], (x - x0, y - y0), bool(clamped)


def patch_stats(patch: np.ndarray) -> dict:
    if patch.size == 0:
        return {"empty": True}
    rgb = patch[..., :3].astype(float)
    gray = rgb.mean(axis=2)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    return {
        "empty": False,
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "frac_pixels_white": float((gray > WHITE_THRESH).mean()),
        "mean_saturation": float(sat.mean()),
        "shape": list(patch.shape[:2]),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def _label(it: dict) -> str:
    base = (f"#{it['rank']}  idx {it['patch_index']}\n"
            f"{it['slide_name'][:18]}\n"
            f"({it['x']}, {it['y']})\n"
            f"nd={it['nuclear_density']:.3e}  n={it['nucleus_count']}")
    if it.get("hole_pct") is not None:
        base += f"\nhole%={it['hole_pct']:.3f}  duct {str(it['duct_id'])[:8]}"
    if it.get("clamped"):
        base += "\n[edge-clamped]"
    return base


def contact_sheet(items: list[dict], title: str, path: Path,
                  key: str, boxes: bool) -> None:
    rows, cols = GRID
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.4))
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
                                           edgecolor="#00FF00", lw=2.0))
                ax.add_patch(plt.Rectangle((bx, by), s, s, fill=False,
                                           edgecolor="#000000", lw=0.7))
        ax.set_title(_label(it), fontsize=5.5)
        ax.axis("off")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Driver ────────────────────────────────────────────────────────────────────

def inspect_run(label: str, run_dir: Path, png_dir: Path, out_dir: Path,
                patch_size: int, window: int) -> dict:
    print(f"\n=== {label}  ({run_dir}) ===")
    roots, source = load_root_indices(run_dir)
    df = load_results(run_dir)
    holey = load_holeyness(run_dir)
    print(f"  root source: {source}   n_roots: {len(roots)}")
    if holey:
        print(f"  holeyness_roots.json present — duct labels available")

    bad = [i for i in roots if i < 0 or i >= len(df)]
    if bad:
        raise IndexError(
            f"{label}: root indices {bad[:5]} are out of range for a results.csv "
            f"with {len(df)} rows. The h5ad and results.csv are from different runs."
        )

    indiv = out_dir / f"roots_{label}"
    indiv.mkdir(parents=True, exist_ok=True)

    cache: dict[str, np.ndarray] = {}

    def slide(name: str):
        if name not in cache:
            p = png_dir / f"{name}.png"
            if not p.exists():
                hits = list(png_dir.glob(f"{name}.*"))
                p = hits[0] if hits else None
            cache[name] = np.array(Image.open(p).convert("RGB")) if p else None
        return cache[name]

    items, n_clamped = [], 0
    for rank, idx in enumerate(roots):
        row = df.iloc[idx]
        name = str(row["slide_name"])
        x, y = int(row["x"]), int(row["y"])
        img = slide(name)

        patch = crop_patch(img, x, y, patch_size) if img is not None else np.array([])
        if img is not None:
            ctx, box, clamped = crop_context(img, x, y, patch_size, window)
        else:
            ctx, box, clamped = np.array([]), None, False
        n_clamped += int(clamped)

        hk = holey.get(int(idx), {})
        it = {
            "rank": rank,
            "patch_index": int(idx),
            "slide_name": name,
            "x": x, "y": y,
            "patch_size": patch_size,
            "nuclear_density": float(row.get("nuclear_density", float("nan"))),
            "nucleus_count": int(row["nucleus_count"]) if "nucleus_count" in df.columns
                             and np.isfinite(row["nucleus_count"]) else -1,
            "pseudotime": float(row.get("pseudotime", float("nan"))),
            "hole_pct": hk.get("hole_pct"),
            "duct_id": hk.get("duct_id"),
            "duct_area_um2": hk.get("duct_area_um2"),
            "clamped": clamped,
            "stats": patch_stats(patch),
            "patch": patch,
            "context": ctx,
            "box": box,
        }
        items.append(it)
        if patch.size:
            Image.fromarray(patch).save(indiv / f"root{rank:02d}_idx{idx}_{name}.png")

    contact_sheet(items, f"{label} — DPT roots, native {patch_size}px  "
                         f"(source: {source})",
                  out_dir / f"roots_{label}_patches", key="patch", boxes=False)
    contact_sheet(items, f"{label} — DPT roots in {window}x{window}px context  "
                         f"({n_clamped} edge-clamped)",
                  out_dir / f"roots_{label}_context", key="context", boxes=True)

    rec = {
        "label": label,
        "run_dir": str(run_dir),
        "root_source": source,
        "n_roots": len(roots),
        "context_window_px": window,
        "n_edge_clamped": n_clamped,
        "roots": [{k: v for k, v in it.items()
                   if k not in ("patch", "context", "box")} for it in items],
    }
    with open(out_dir / f"roots_{label}.json", "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"  wrote roots_{label}_patches / _context (.png+.pdf, {DPI} dpi), "
          f"{n_clamped} edge-clamped")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--png-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=112,
                    help="MUST match the run's --patch-size or the wrong window "
                         "is cropped, silently. (default: 112)")
    ap.add_argument("--context-window", type=int, default=CONTEXT_PX_DEFAULT,
                    help=f"Side length in px of the context crop. "
                         f"(default: {CONTEXT_PX_DEFAULT})")
    args = ap.parse_args()

    if len(args.labels) != len(args.run_dirs):
        ap.error("--labels and --run-dirs must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for label, rd in zip(args.labels, args.run_dirs):
        out.append(inspect_run(label, rd, args.png_dir, args.output_dir,
                               args.patch_size, args.context_window))

    with open(args.output_dir / "root_sheets_index.json", "w") as f:
        json.dump({"runs": out}, f, indent=2, default=str)
    print(f"\nAll sheets written to {args.output_dir}")
    print("Presented without judgement — no early/late or tissue/artifact call is "
          "made here.")


if __name__ == "__main__":
    main()
