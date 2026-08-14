"""Task 3: browsing index for the per-slide timepoint runs.

================================================================================
NON-COMPARABILITY CONSTRAINT
================================================================================
Every slide below was run through ``run_individual.py``, which fits a SEPARATE
PCA basis per slide, with no patch cap, no feature cache, no batch correction,
and single-root cluster-anchored DPT. PSEUDOTIME VALUES FROM ONE SLIDE ARE NOT
COMPARABLE TO ANY OTHER SLIDE, NOR TO ANY PER-SECTION OR PROJECTED RESULT
ELSEWHERE IN THIS PROJECT.

This index groups slides by timepoint FOR BROWSING ONLY. It deliberately
computes NO statistic that crosses the grouping — no per-timepoint mean
pseudotime, no trend, no test. Such a number would be meaningless here, because
each slide's pseudotime is defined on its own axis with its own arbitrary scale
and its own separately-chosen origin.

A cross-slide or cross-timepoint comparison requires a DIFFERENT pipeline:
either running every slide through the main pipeline's shared-PCA logic, or the
existing projection pathway. Both are explicitly out of scope here.

This run is also independent of, and does not resolve, the earlier finding that
projecting these slides onto the trained manifold produced 100% extrapolation,
or that they differ substantially in staining from the 2M cohort.
================================================================================

Reads (read-only): the Pass 3 output tree, root_choices.json, and each slide's
results.csv. Writes only --output.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BANNER = """> ## ⚠ These pseudotimes are NOT comparable across slides
>
> Each slide was fitted with its **own PCA basis**, its own Leiden clustering, and
> its own separately-chosen DPT root, by `run_individual.py` — no patch cap, no
> feature cache, no batch correction, single-root cluster-anchored DPT. A
> pseudotime of 0.8 on one slide and 0.8 on another mean **nothing in common**.
>
> The grouping by timepoint below is **for browsing only**. No statistic crossing
> that grouping is computed or reported, because none would be meaningful.
>
> A cross-slide or cross-timepoint comparison needs a different pipeline — the
> main pipeline's shared-PCA logic, or the existing projection pathway. Out of
> scope here.
>
> This is independent of, and does not resolve, the 100%-extrapolation projection
> finding or the staining differences versus the 2M cohort."""

TIMEPOINT_RE = re.compile(r"-(\d+)W\b", re.IGNORECASE)
MOUSE_RE = re.compile(r"^(\d+)")
TIMEPOINT_ORDER = ["4W", "7W", "8W", "12W"]


def parse_slide(stem: str) -> tuple[str, str]:
    """('6054-4L-8W_x5') -> ('6054', '8W'). Unparseable parts return '?'."""
    base = stem[:-3] if stem.endswith("_x5") else stem
    m_mouse = MOUSE_RE.match(base)
    m_tp = TIMEPOINT_RE.search(base)
    return (m_mouse.group(1) if m_mouse else "?",
            (m_tp.group(1) + "W").upper() if m_tp else "?")


def collect(final_dir: Path, root_choices: dict, min_patches: int) -> list[dict]:
    rows = []
    for d in sorted(p for p in final_dir.iterdir() if p.is_dir()):
        csv = d / "results.csv"
        if not csv.exists():
            rows.append({"slide": d.name, "status": "FAILED — no results.csv"})
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:                                     # noqa: BLE001
            rows.append({"slide": d.name, "status": f"FAILED — unreadable ({e})"})
            continue

        mouse, tp = parse_slide(d.name)
        rc = root_choices.get(d.name, {})
        n = len(df)
        status = "ok"
        if n < min_patches:
            status = (f"UNSTABLE — {n} patches (< {min_patches}); "
                      "per-slide PCA/Leiden/DPT is not trustworthy at this size")

        rows.append({
            "slide": d.name,
            "mouse": mouse,
            "timepoint": tp,
            "n_patches": n,
            "n_clusters": int(df["cluster"].nunique()) if "cluster" in df.columns else None,
            "root_cluster": rc.get("root_cluster"),
            "root_median_nuclear_density": rc.get("root_median_nuclear_density"),
            "root_tie": bool(rc.get("tie_broken_by_lowest_cluster_id", False)),
            "png_width": rc.get("png_width"),
            "png_height": rc.get("png_height"),
            "has_overlay": (d / "overlays").is_dir() and any((d / "overlays").glob("*.html")),
            "has_patches": (d / "patch_export").is_dir(),
            "status": status,
            "dir": d,
        })
    return rows


def write_report(rows: list[dict], out: Path, final_dir: Path,
                 min_patches: int, wide_ratio: float) -> None:
    L = ["# Per-slide pseudotime — timepoint cohort", "", BANNER, "",
         f"Source tree: `{final_dir}`", ""]

    ok = [r for r in rows if r.get("status") == "ok"]
    unstable = [r for r in rows if str(r.get("status", "")).startswith("UNSTABLE")]
    failed = [r for r in rows if str(r.get("status", "")).startswith("FAILED")]
    L += [f"**{len(ok)} slide(s) processed**, {len(unstable)} flagged unstable, "
          f"{len(failed)} failed.", ""]

    # Possible duplicate-tissue flag: the MCF7 NDPIs hold two side-by-side copies
    # of the same slide, and these PNGs were converted FULL WIDTH with no crop. A
    # very wide aspect ratio is the signature of that layout surviving into the
    # patch set, which would duplicate tissue within a slide's own manifold.
    wide = [r for r in ok
            if r.get("png_width") and r.get("png_height")
            and r["png_width"] / r["png_height"] >= wide_ratio]
    if wide:
        L += ["## ⚠ Possible duplicate tissue", "",
              f"{len(wide)} slide(s) have a width:height ratio >= {wide_ratio}. These "
              "PNGs were converted **full width, no crop**. The MCF7 NDPIs contain two "
              "side-by-side copies of the same physical slide; if these do too, "
              "whole-slide patching ingested each region **twice**, duplicating tissue "
              "inside that slide's own manifold. That inflates patch counts and cluster "
              "sizes and distorts the diffusion map.", "",
              "This is flagged, not corrected — verify against the source NDPIs before "
              "reading structure into the affected slides.", "",
              "| slide | width | height | ratio |", "|---|---|---|---|"]
        for r in wide:
            L.append(f"| `{r['slide']}` | {r['png_width']} | {r['png_height']} | "
                     f"{r['png_width']/r['png_height']:.2f} |")
        L.append("")

    if unstable:
        L += ["## Flagged unstable — do not read structure into these", ""]
        for r in unstable:
            L.append(f"- `{r['slide']}` — {r['status']}")
        L.append("")
    if failed:
        L += ["## Failed", ""]
        for r in failed:
            L.append(f"- `{r['slide']}` — {r['status']}")
        L.append("")

    L += ["## Slides, grouped by timepoint", "",
          "*Grouping is for browsing only. No statistic is computed across groups — "
          "see the constraint above.*", ""]

    seen = {r.get("timepoint") for r in rows if r.get("timepoint")}
    order = [t for t in TIMEPOINT_ORDER if t in seen] + sorted(seen - set(TIMEPOINT_ORDER))
    for tp in order:
        grp = [r for r in rows if r.get("timepoint") == tp and r.get("status") == "ok"]
        if not grp:
            continue
        L += [f"### {tp}  ({len(grp)} slide(s))", "",
              "*Each row's pseudotime lives on its own axis. Rows are neighbours in "
              "this table and nowhere else.*", "",
              "| slide | mouse | patches | clusters | root cluster | root median "
              "nuclear density | overlay | patches |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sorted(grp, key=lambda x: x["slide"]):
            rd = r["root_median_nuclear_density"]
            rd_s = "n/a" if rd is None else f"{rd:.5g}"
            root_s = "n/a" if r["root_cluster"] is None else str(r["root_cluster"])
            if r["root_tie"]:
                root_s += " *(tie)*"
            ov = (f"[overlay]({r['slide']}/overlays/)" if r["has_overlay"] else "—")
            px = (f"[patches]({r['slide']}/patch_export/)" if r["has_patches"] else "—")
            L.append(f"| `{r['slide']}` | {r['mouse']} | {r['n_patches']} | "
                     f"{r['n_clusters']} | {root_s} | {rd_s} | {ov} | {px} |")
        L.append("")

    L += ["## How the root cluster was chosen", "",
          "`run_individual.py` defaults to the **lowest-numbered** Leiden cluster, and "
          "Leiden IDs are arbitrary labels — so that default is an arbitrary origin. "
          "Instead each slide's root is the cluster with the **lowest median "
          "`nuclear_density`**, computed by "
          "`validation/morphological_features.compute_nuclear_density_quick` — the same "
          "function the atlas pipeline uses to rank its own DPT roots. Ties break to "
          "the lowest cluster ID.", "",
          "Every cluster's median density is recorded in `root_choices.json`, not just "
          "the winner, so the choice is auditable.", "",
          "Because clusters are per-slide objects, the chosen root cluster ID is also a "
          "per-slide label. **Root cluster 2 on one slide has no relationship to root "
          "cluster 2 on another.**", "",
          "## Out of scope", "",
          "This run says nothing about differences **between** slides or timepoints. If "
          "a cross-slide or cross-timepoint comparison is wanted, it needs the main "
          "pipeline's shared-PCA logic or the existing projection pathway — a different "
          "pipeline, not a summary of this one.", ""]

    out.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", type=Path, required=True,
                    help="Pass 3 output root, one subdirectory per slide.")
    ap.add_argument("--root-choices", type=Path, required=True,
                    help="root_choices.json written by Pass 2.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Markdown index to write (NEW file).")
    ap.add_argument("--min-patches", type=int, default=500,
                    help="Slides below this are flagged UNSTABLE rather than silently "
                         "included. run_individual's own floor is 50 patches, which is "
                         "far too low for a stable per-slide PCA/Leiden/DPT. "
                         "(default: 500)")
    ap.add_argument("--wide-ratio", type=float, default=1.8,
                    help="Flag slides whose PNG width:height meets this, as possible "
                         "two-copy full-width images. (default: 1.8)")
    args = ap.parse_args()

    print(BANNER.replace("> ", "").replace(">", ""))

    if not args.final_dir.is_dir():
        raise SystemExit(f"ERROR: --final-dir not found: {args.final_dir}")
    if args.output.exists():
        raise SystemExit(f"ERROR: {args.output} exists; refusing to overwrite.")

    rc = json.loads(args.root_choices.read_text()) if args.root_choices.exists() else {}
    rows = collect(args.final_dir, rc.get("choices", {}), args.min_patches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(rows, args.output, args.final_dir, args.min_patches, args.wide_ratio)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nIndexed {len(rows)} slide dir(s); {n_ok} ok.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
