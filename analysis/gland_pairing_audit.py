"""Task 1 — establish the gland pairing empirically, from the per-duct tables.

WHY THIS EXISTS
---------------
Every analysis to date has treated the 16 slides as 16 independent samples. They are
not: they are **8 matched pairs**. Each mouse-flank (gland) contributes one slide to
2M-1 (Carnoy's/carmine alum) and one to 2M-2 (PFA). The collaborating pathologist's
own methods text treats them as matched pairs, n = 8.

Consequence: any between-section test that permutes the 16 slides freely — including
the C(16,8) = 12,870 exact test currently on record — admits between-gland and
between-mouse variation that the paired design already controls. Its null is wider
than it should be, so its minimum detectable difference is inflated.

WHAT THIS MODULE DOES
---------------------
It establishes the pairing **from the data**, not from the brief and not from the
slide-list files. `slide_name` is parsed out of the per-duct tables themselves, and
the audit refuses to proceed if the design is not the balanced 8x2 it is claimed to
be. That refusal is the point: an unbalanced design would make the sign-flip null of
Task 2 invalid, and it is better to stop here than to produce a paired p-value on an
unpaired design.

It also reports per-gland marginals (n ducts, median duct area, median hole %, median
duct-level pseudotime) so a reader can see whether the two halves of each gland are
comparable in the quantities being tested.

WHAT IT DOES NOT DO
-------------------
It does not fix anything, recompute anything, or rerun any pipeline stage. It reads
two CSVs. The mis-specification audit in the report accompanying this module is
report-only by instruction, so that the scope of each correction stays visible.

READ-ONLY. Writes only to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 6027-4L-2M-1_x5  ->  mouse 6027, flank 4L, section 2M-1
SLIDE_RE = re.compile(r"^(?P<mouse>\d+)-(?P<flank>4[LR])-(?P<section>2M-[12])(?:_x\d+)?$")

REQUIRED_COLUMNS = ["slide_name", "hole_pct", "area_um2", "pseudotime"]
EXPECTED_N_GLANDS = 8


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def parse_slide(name: str) -> dict:
    """Split a slide_name into mouse / flank / section.

    Raises rather than returning a partial parse: a slide whose name does not match
    the expected pattern cannot be assigned to a gland, and silently dropping it
    would change the design without saying so.
    """
    m = SLIDE_RE.match(str(name).strip())
    if not m:
        raise ValueError(
            f"slide_name {name!r} does not match the expected "
            "'<mouse>-<4L|4R>-<2M-1|2M-2>[_x5]' pattern, so its gland cannot be "
            "determined. Refusing to guess."
        )
    d = m.groupdict()
    d["gland"] = f"{d['mouse']}-{d['flank']}"
    return d


def load_per_duct(path: Path, expect_section: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This audit reads the per-duct tables the holeyness "
            "validation wrote and must not recompute them."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns {missing}.")

    parsed = pd.DataFrame([parse_slide(s) for s in df["slide_name"]])
    df = df.copy()
    for col in ("mouse", "flank", "section", "gland"):
        df[col] = parsed[col].values

    found = sorted(df["section"].unique())
    if found != [expect_section]:
        raise ValueError(
            f"{path} was expected to contain only section {expect_section!r} but "
            f"contains {found}. The per-duct tables are per-section by construction; "
            "this means the wrong file was supplied."
        )
    return df


# ── (a) + (b) the pairing table and the balance gate ─────────────────────────

def build_pairing(frames: dict) -> dict:
    sections = list(frames)
    by_section = {s: sorted(frames[s]["gland"].unique()) for s in sections}
    all_glands = sorted(set().union(*by_section.values()))

    table = []
    for g in all_glands:
        row = {"gland": g,
               "mouse": g.split("-")[0],
               "flank": g.split("-")[1]}
        for s in sections:
            sub = frames[s][frames[s]["gland"] == g]
            row[f"{s}_n_slides"] = int(sub["slide_name"].nunique())
            row[f"{s}_slide"] = (sorted(sub["slide_name"].unique())[0]
                                 if len(sub) else None)
        table.append(row)

    problems = []
    for g in all_glands:
        for s in sections:
            n = next(r[f"{s}_n_slides"] for r in table if r["gland"] == g)
            if n == 0:
                problems.append(f"gland {g} has NO slide in {s}")
            elif n > 1:
                problems.append(f"gland {g} has {n} slides in {s} (expected 1)")
    if len(all_glands) != EXPECTED_N_GLANDS:
        problems.append(
            f"found {len(all_glands)} glands, expected {EXPECTED_N_GLANDS}")

    balanced = not problems
    return {
        "sections": sections,
        "n_glands": len(all_glands),
        "glands": all_glands,
        "glands_by_section": by_section,
        "pairing_table": table,
        "identical_gland_sets": bool(
            len(set(map(tuple, by_section.values()))) == 1),
        "balanced": balanced,
        "problems": problems,
        "verdict": (
            f"BALANCED — {len(all_glands)} glands, each contributing exactly one "
            "slide to each section. The paired design is valid and Task 2's "
            "2^8 = 256 sign-flip null is well defined."
            if balanced else
            "NOT BALANCED — the paired design does NOT hold as claimed. Task 2 must "
            "not run: a sign-flip null assumes one value per gland per section. "
            "Problems: " + "; ".join(problems)),
    }


# ── (c) per-gland marginals ──────────────────────────────────────────────────

def per_gland_marginals(frames: dict) -> dict:
    sections = list(frames)
    rows = []
    for s in sections:
        df = frames[s]
        for g, sub in df.groupby("gland"):
            rows.append({
                "gland": g, "section": s,
                "slide_name": sorted(sub["slide_name"].unique())[0],
                "n_ducts": int(len(sub)),
                "median_area_um2": float(np.nanmedian(sub["area_um2"])),
                "median_hole_pct": float(np.nanmedian(sub["hole_pct"])),
                "median_pseudotime": float(np.nanmedian(sub["pseudotime"])),
            })
    long = pd.DataFrame(rows)

    # within-gland ratios, so a reader can see comparability at a glance
    a, b = sections
    wide = long.pivot(index="gland", columns="section")
    comp = []
    for g in sorted(long["gland"].unique()):
        e = {"gland": g}
        for q in ("n_ducts", "median_area_um2", "median_hole_pct",
                  "median_pseudotime"):
            va = float(wide[(q, a)][g]); vb = float(wide[(q, b)][g])
            e[f"{q}_{a}"] = va
            e[f"{q}_{b}"] = vb
            e[f"{q}_ratio_{a}_over_{b}"] = (va / vb) if vb not in (0, np.nan) else None
        comp.append(e)

    return {
        "per_gland_per_section": rows,
        "within_gland_comparison": comp,
        "note": (
            "Marginals only. A large within-gland imbalance in n_ducts or duct area "
            "does not invalidate the paired design — the paired test differences "
            "correlations, not duct counts — but it does bear on whether the two "
            "halves of a gland sample comparable tissue, which is a question for "
            "the pathologist rather than for this audit."),
    }


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs=2, default=["2M-1", "2M-2"])
    ap.add_argument("--per-duct-csvs", nargs=2, type=Path, required=True,
                    help="holeyness_per_duct.csv per section, SAME ORDER as --sections.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    secs = list(args.sections)
    print("=" * 78)
    print("  TASK 1 — gland pairing audit")
    print("=" * 78)

    frames, paths = {}, {}
    for i, s in enumerate(secs):
        p = args.per_duct_csvs[i]
        frames[s] = load_per_duct(p, s)
        paths[s] = str(p)
        print(f"  {s}: {p}")
        print(f"      {len(frames[s])} ducts, "
              f"{frames[s]['slide_name'].nunique()} slides, "
              f"{frames[s]['gland'].nunique()} glands")

    pairing = build_pairing(frames)
    print("\n  === (a) pairing table ===")
    a, b = secs
    print(f"  {'gland':<12} {a:<22} {b:<22}")
    for r in pairing["pairing_table"]:
        print(f"  {r['gland']:<12} {str(r[f'{a}_slide']):<22} "
              f"{str(r[f'{b}_slide']):<22}")

    print(f"\n  === (b) balance gate ===")
    print(f"  identical gland sets between sections: "
          f"{pairing['identical_gland_sets']}")
    print(f"  {pairing['verdict']}")

    marg = per_gland_marginals(frames)
    print("\n  === (c) per-gland marginals ===")
    print(f"  {'gland':<12} {'sec':<6} {'n_ducts':>8} {'med_area':>11} "
          f"{'med_hole':>9} {'med_pt':>8}")
    for r in marg["per_gland_per_section"]:
        print(f"  {r['gland']:<12} {r['section']:<6} {r['n_ducts']:>8} "
              f"{r['median_area_um2']:>11.0f} {r['median_hole_pct']:>9.3f} "
              f"{r['median_pseudotime']:>8.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "analysis": "gland_pairing_audit",
        "inputs": {"per_duct_tables": paths,
                   "n_ducts": {s: int(len(frames[s])) for s in secs},
                   "recomputed_anything": False},
        "pairing": pairing,
        "marginals": marg,
    }
    out = args.output_dir / "gland_pairing_audit.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default),
                   encoding="utf-8")
    write_report(res, args.output_dir / "gland_pairing_audit.md")
    print(f"\n  JSON:     {out}")
    print(f"  Markdown: {args.output_dir / 'gland_pairing_audit.md'}")

    if not pairing["balanced"]:
        raise SystemExit(
            "\nSTOPPING: the design is not the balanced 8x2 the paired analysis "
            "requires. Task 2 must not be run until this is resolved.")


def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    p = res["pairing"]
    a, b = p["sections"]

    add("# Task 1 — gland pairing audit\n")
    add("The 16 slides are **8 matched pairs**, not 16 independent samples: each "
        "mouse-flank (gland) contributes one slide to each section. This audit "
        "establishes that from the per-duct tables themselves rather than from "
        "filenames or from the claim.\n")

    add("## Inputs\n")
    for s, q in res["inputs"]["per_duct_tables"].items():
        add(f"- **{s}**: `{q}` — {res['inputs']['n_ducts'][s]} ducts")
    add("\nNothing was recomputed.\n")

    add("## (a) Pairing table\n")
    add(f"| gland | mouse | flank | {a} | {b} |")
    add("|---|---|---|---|---|")
    for r in p["pairing_table"]:
        add(f"| {r['gland']} | {r['mouse']} | {r['flank']} | "
            f"{r[f'{a}_slide']} | {r[f'{b}_slide']} |")

    add("\n## (b) Balance gate\n")
    add(f"- glands found: **{p['n_glands']}** (expected {EXPECTED_N_GLANDS})")
    add(f"- identical gland sets between sections: **{p['identical_gland_sets']}**")
    add(f"- balanced: **{p['balanced']}**")
    if p["problems"]:
        for q in p["problems"]:
            add(f"  - PROBLEM: {q}")
    add(f"\n**Verdict:** {p['verdict']}\n")

    add("## (c) Per-gland marginals\n")
    add(f"| gland | section | n ducts | median area (um^2) | median hole % | median pseudotime |")
    add("|---|---|---|---|---|---|")
    for r in res["marginals"]["per_gland_per_section"]:
        add(f"| {r['gland']} | {r['section']} | {r['n_ducts']} | "
            f"{r['median_area_um2']:.0f} | {r['median_hole_pct']:.3f} | "
            f"{r['median_pseudotime']:.4f} |")

    add("\n### Within-gland ratios (Carnoy's / PFA)\n")
    add("| gland | n ducts | median area | median hole % | median pseudotime |")
    add("|---|---|---|---|---|")
    for e in res["marginals"]["within_gland_comparison"]:
        cells = []
        for q in ("n_ducts", "median_area_um2", "median_hole_pct",
                  "median_pseudotime"):
            v = e.get(f"{q}_ratio_{a}_over_{b}")
            cells.append(f"{v:.3f}" if v is not None else "—")
        add(f"| {e['gland']} | " + " | ".join(cells) + " |")
    add(f"\n> {res['marginals']['note']}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
