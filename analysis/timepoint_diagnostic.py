"""
Timepoint cohort: Stage E -- diagnostic analysis (NOT a validation claim).

WHAT THIS IS, AND WHAT IT IS NOT
---------------------------------
This is a diagnostic run executed DESPITE a failed stain gate. Stage B v2 found
that timepoint groups separate on hematoxylin intensity specifically -- the RGB
channels are mostly negligible-to-small -- at or above the project's own
reference confound threshold. That hematoxylin effect is consistent with EITHER
a reagent-side confound OR genuine cellularity change with tumor age; this
cohort cannot distinguish those two. No correction was applied and the gate
still stands.

The question this module asks is narrow: does projected pseudotime track
timepoint INDEPENDENTLY of that one already-confounded channel, or does the
raw association collapse once hematoxylin is controlled for? That is a
diagnostic, not a result. Nothing here validates or supports the timepoint
hypothesis, and no output may be interpreted or shown to anyone as a timepoint
result until the PI's biology question and the stain-versus-cellularity
disambiguation are resolved.

Reuses (does not reimplement) from analysis/holeyness.py:
  _safe_spearman, _partial_spearman (= rho(x, y | z)), _format_perm_p.
Hematoxylin values are READ from Stage B v2's existing output and never
recomputed, so the numbers here and there refer to the same quantity.

Joins Stage D and Stage B v2 on the composite key (mouse_id, timepoint_weeks),
never mouse_id alone -- mouse 6072 contributes both a 7W and a 12W row from a
confirmed staggered harvest of the same animal, and collapsing on mouse_id
would silently merge two different timepoints into one.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_diagnostic \\
      --stageD-json $SCRATCH/results/timepoint_cohort/stageD_projection/stageD_projection.json \\
      --stageB-json $SCRATCH/results/timepoint_cohort/stageB_v2_fullres/stageB_v2_fullres.json \\
      --output-dir  $SCRATCH/results/timepoint_cohort/stageE_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .holeyness import _format_perm_p, _partial_spearman, _safe_spearman
from .timepoint_stage2_stain_check import _fmt

H_MEASURES = ["median_h_intensity_mean_masked", "median_h_intensity_median_masked"]
DUAL_TIMEPOINT_MOUSE = "6072"

# _partial_spearman returns nan below this many valid observations (guard lives
# in holeyness.py). Recorded here so the report can say plainly WHY a partial is
# nan rather than leaving a blank cell.
PARTIAL_MIN_N = 10


# ── Joining Stage D and Stage B v2 ────────────────────────────────────────────

def join_mouse_tables(stageD_rows: list[dict], stageB_rows: list[dict],
                      variant: str) -> list[dict]:
    """Composite-key join. Hard-fails on any mismatch rather than silently
    dropping rows -- a missing mouse would quietly change n and every
    correlation computed from it."""
    d_by_key = {(r["mouse_id"], r["timepoint_weeks"]): r for r in stageD_rows}
    b_by_key = {(r["mouse_id"], r["timepoint_weeks"]): r for r in stageB_rows}

    only_d = sorted(set(d_by_key) - set(b_by_key))
    only_b = sorted(set(b_by_key) - set(d_by_key))
    if only_d or only_b:
        sys.exit(
            f"ERROR: Stage D and Stage B v2 mouse tables disagree for variant "
            f"'{variant}'.\n"
            f"  Only in Stage D (projected but no hematoxylin): {only_d}\n"
            f"  Only in Stage B (hematoxylin but not projected): {only_b}\n"
            "Refusing to compute correlations on a silently-reduced mouse set. "
            "A slide that failed projection in Stage D is the most likely cause -- "
            "check Stage D's failed_slides."
        )

    joined = []
    for key in sorted(d_by_key, key=lambda k: (k[1], k[0])):
        d, b = d_by_key[key], b_by_key[key]
        row = {
            "mouse_id": key[0],
            "timepoint_weeks": key[1],
            "pseudotime": d["median_projected_pseudotime"],
            "has_suffix_slide": d["has_suffix_slide"],
            "dual_timepoint_mouse": d["dual_timepoint_mouse"],
            "frac_beyond_training_p99": d.get("median_frac_beyond_training_p99"),
        }
        for m in H_MEASURES:
            if m not in b:
                sys.exit(f"ERROR: Stage B v2 row {key} is missing '{m}'.")
            row[m] = b[m]
        joined.append(row)
    return joined


# ── The three quantities, computed side by side on one mouse set ─────────────

def compute_block(rows: list[dict], n_permutations: int,
                  rng: np.random.Generator) -> dict:
    """RAW, PARTIAL (one per hematoxylin measure), and hematoxylin-vs-weeks --
    all on the IDENTICAL mouse set, so n and membership are directly
    comparable across the three. Permutation shuffles the timepoint label
    across mouse rows; every statistic in a given iteration uses the same
    shuffled labels, so the nulls are comparable too."""
    weeks = np.array([r["timepoint_weeks"] for r in rows], dtype=float)
    pt = np.array([r["pseudotime"] for r in rows], dtype=float)
    h = {m: np.array([r[m] for r in rows], dtype=float) for m in H_MEASURES}
    n = len(rows)

    raw_rho, raw_p = _safe_spearman(pt, weeks)
    partial = {m: _partial_spearman(pt, weeks, h[m]) for m in H_MEASURES}
    h_vs_weeks = {m: _safe_spearman(h[m], weeks) for m in H_MEASURES}

    null_raw = np.empty(n_permutations)
    null_partial = {m: np.empty(n_permutations) for m in H_MEASURES}
    null_h = {m: np.empty(n_permutations) for m in H_MEASURES}
    for i in range(n_permutations):
        w_perm = rng.permutation(weeks)
        null_raw[i] = abs(_safe_spearman(pt, w_perm)[0])
        for m in H_MEASURES:
            null_partial[m][i] = abs(_partial_spearman(pt, w_perm, h[m]))
            null_h[m][i] = abs(_safe_spearman(h[m], w_perm)[0])

    def _perm_p(null: np.ndarray, obs: float) -> float:
        if not np.isfinite(obs):
            return float("nan")
        null = null[np.isfinite(null)]
        if null.size == 0:
            return float("nan")
        return float(np.mean(null >= abs(obs)))

    return {
        "n_mice_rows": n,
        "mouse_rows": [f"{r['mouse_id']}@{r['timepoint_weeks']}W" for r in rows],
        "raw": {
            "rho": raw_rho, "scipy_p": raw_p,
            "perm_p": _perm_p(null_raw, raw_rho),
            "null95": float(np.percentile(null_raw[np.isfinite(null_raw)], 95))
                      if np.any(np.isfinite(null_raw)) else float("nan"),
        },
        "partial": {
            m: {
                "rho": partial[m],
                "perm_p": _perm_p(null_partial[m], partial[m]),
                "control": m,
                "insufficient_n": bool(n < PARTIAL_MIN_N),
            } for m in H_MEASURES
        },
        "hematoxylin_vs_weeks": {
            m: {
                "rho": h_vs_weeks[m][0], "scipy_p": h_vs_weeks[m][1],
                "perm_p": _perm_p(null_h[m], h_vs_weeks[m][0]),
            } for m in H_MEASURES
        },
    }


def dual_timepoint_aside(rows: list[dict]) -> dict:
    """Mouse 6072's two pseudotime values, reported individually."""
    mine = [r for r in rows if r["mouse_id"] == DUAL_TIMEPOINT_MOUSE]
    return {
        "mouse_id": DUAL_TIMEPOINT_MOUSE,
        "present": bool(mine),
        "values": [
            {"timepoint_weeks": r["timepoint_weeks"], "pseudotime": r["pseudotime"]}
            for r in sorted(mine, key=lambda r: r["timepoint_weeks"])
        ],
        "contributes_n_rows": len(mine),
    }


# ── Output writers ────────────────────────────────────────────────────────────

def _p_cell(perm_p, n_perm) -> str:
    return _format_perm_p(perm_p, n_perm) if perm_p is not None else "n/a"


def _fmt_p(p) -> str:
    """p-values only. `_fmt` is for correlations -- it forces a leading '+' and
    fixed 4dp, which on a p-value both reads oddly and renders anything below
    1e-4 as '+0.0000', i.e. indistinguishable from exactly zero."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"


def _short_control(measure: str) -> str:
    """'median_h_intensity_mean_masked' -> 'h_intensity_mean'. Trimming to a
    bare 'mean'/'median' would be ambiguous about WHICH quantity is being
    controlled for."""
    return measure.replace("median_h_intensity_", "h_intensity_").replace("_masked", "")


def write_report(result: dict, output_dir: Path) -> None:
    n_perm = result["n_permutations"]
    lines = ["# Timepoint cohort — Stage E: diagnostic analysis (NOT a validation claim)", ""]

    lines.append(
        "**What this is.** This is a diagnostic run executed *despite* a failed stain "
        "gate. Stage B v2 found that timepoint groups separate specifically on "
        "hematoxylin intensity — the RGB channels are mostly negligible-to-small — at "
        "or above this project's own reference confound threshold, and that "
        "hematoxylin effect is consistent with **either** a reagent-side confound "
        "**or** genuine cellularity change with tumor age; this cohort cannot tell "
        "those apart. No correction has been applied and the gate still stands. The "
        "only question asked below is whether projected pseudotime tracks timepoint "
        "*independently of that one already-confounded channel*, or whether the raw "
        "association collapses once hematoxylin is controlled for. **This is not a "
        "validated timepoint result. It must not be interpreted or shown to anyone "
        "as one** until the PI's biology question and the stain-versus-cellularity "
        "disambiguation are resolved."
    )
    lines.append("")

    # Projection validity FIRST — before any correlation number.
    pv = result["projection_validity"]
    lines.append("## Projection validity — the correlations below rest on this")
    lines.append("")
    if not pv.get("baseline_available"):
        lines.append(
            f"> **Extrapolation could not be assessed** ({pv.get('reason')}). Every "
            "correlation below is therefore of unknown validity."
        )
    elif pv["n_slides_substantially_extrapolated"] > 0:
        lines.append(
            f"> **{pv['n_slides_substantially_extrapolated']} of {pv['n_slides_total']} "
            f"slides are substantially extrapolated** (more than "
            f"{pv['extrapolation_fraction_warn']:.0%} of their patches lie beyond the "
            f"training manifold's p99 mean-k-NN distance). For those slides the KNN "
            f"still returns a pseudotime for every patch, but it describes regions of "
            f"feature space the manifold does not cover. **Any correlation below that "
            f"includes them is limited accordingly — this is a limitation on "
            f"interpretation, not a footnote.**"
        )
        lines.append("")
        lines.append("Per timepoint group:")
        lines.append("")
        lines.append("| weeks | n slides | median % patches beyond training p99 | slides flagged |")
        lines.append("|---|---|---|---|")
        for w, v in pv["by_timepoint"].items():
            frac = v["median_frac_beyond_training_p99"]
            lines.append(
                f"| {w} | {v['n_slides']} | "
                f"{'n/a' if frac is None else f'{frac:.1%}'} | "
                f"{v['n_slides_substantially_extrapolated']} |")
    else:
        lines.append(
            f"> No slide of {pv['n_slides_total']} exceeds the extrapolation threshold "
            f"({pv['extrapolation_fraction_warn']:.0%} of patches beyond training p99). "
            "Projected patches sit within the training manifold's support on this "
            "measure, so the correlations below are not obviously resting on "
            "extrapolated values."
        )
    lines.append("")

    for label, key in [("excluding the 3 ambiguous-provenance suffix slides (PRIMARY)",
                        "excluding_suffix"),
                       ("including suffix slides", "including_suffix")]:
        block = result["blocks"][key]
        lines.append(f"## Correlations — {label}")
        lines.append("")
        lines.append(f"n = {block['n_mice_rows']} (mouse, timepoint) rows. "
                     f"Permutation: {n_perm} shuffles of the timepoint label, "
                     f"two-tailed on |rho|.")
        lines.append("")
        lines.append("| # | quantity | rho | perm p | scipy p |")
        lines.append("|---|---|---|---|---|")
        r = block["raw"]
        lines.append(f"| 1 | **RAW** — pseudotime vs weeks | {_fmt(r['rho'])} | "
                     f"{_p_cell(r['perm_p'], n_perm)} | {_fmt_p(r['scipy_p'])} |")
        for m in H_MEASURES:
            p = block["partial"][m]
            lines.append(
                f"| 2 | **PARTIAL** — pseudotime vs weeks, controlling "
                f"`{_short_control(m)}` | {_fmt(p['rho'])} | "
                f"{_p_cell(p['perm_p'], n_perm)} | — |")
        for m in H_MEASURES:
            h = block["hematoxylin_vs_weeks"][m]
            lines.append(
                f"| 3 | comparison — `{_short_control(m)}` itself vs weeks | "
                f"{_fmt(h['rho'])} | {_p_cell(h['perm_p'], n_perm)} | "
                f"{_fmt_p(h['scipy_p'])} |")
        lines.append("")
        lines.append(
            "Row 3 recomputes Stage B v2's trend on exactly this mouse set, so rows "
            "1–3 are directly comparable on identical n and membership. Any difference "
            "from Stage B's own reported trend reflects the mouse set, not the measure."
        )
        lines.append("")

    aside = result["dual_timepoint_aside"]
    lines.append("## Aside — mouse 6072 (same animal, two timepoints)")
    lines.append("")
    if aside["present"]:
        vals = ", ".join(f"{v['timepoint_weeks']}W → pseudotime {_fmt(v['pseudotime'])}"
                         for v in aside["values"])
        lines.append(f"Mouse 6072 contributes **{aside['contributes_n_rows']} rows**: {vals}.")
        lines.append("")
        lines.append(
            "**These are not two independent animals.** 6072 is a confirmed staggered "
            "harvest of one mouse, so it enters every correlation above as two "
            "correlated points. The supplementary block below drops 6072 entirely."
        )
    else:
        lines.append("Mouse 6072 is not present in this mouse set.")
    lines.append("")

    if result.get("blocks_drop_6072"):
        lines.append("### Supplementary — same three quantities with mouse 6072 dropped")
        lines.append("")
        lines.append("| variant | n | RAW rho | perm p | partial (mean) | partial (median) |")
        lines.append("|---|---|---|---|---|---|")
        for key, lbl in [("excluding_suffix", "excluding suffix"),
                         ("including_suffix", "including suffix")]:
            b = result["blocks_drop_6072"][key]
            pm = b["partial"]["median_h_intensity_mean_masked"]["rho"]
            pmed = b["partial"]["median_h_intensity_median_masked"]["rho"]
            lines.append(
                f"| {lbl} | {b['n_mice_rows']} | {_fmt(b['raw']['rho'])} | "
                f"{_p_cell(b['raw']['perm_p'], n_perm)} | {_fmt(pm)} | {_fmt(pmed)} |")
        lines.append("")

    lines.append("## Power — read before drawing any conclusion")
    lines.append("")
    n_primary = result["blocks"]["excluding_suffix"]["n_mice_rows"]
    lines.append(
        f"The primary comparison has n={n_primary} (mouse, timepoint) rows. A partial "
        f"correlation with one covariate leaves df={n_primary - 3}. **At this n, an "
        f"attenuated partial correlation cannot distinguish \"pseudotime adds nothing "
        f"beyond hematoxylin\" from \"there are too few mice to tell.\"** Both readings "
        f"remain open on this evidence. Every number above is descriptive; do not lean "
        f"on the p-values, and do not read a sign flip at this n as meaningful."
    )
    lines.append("")
    lines.append(
        "**Reminder:** the confound this controls for is hematoxylin-specific, and "
        "hematoxylin intensity is itself a plausible proxy for cellularity. Controlling "
        "for it may therefore remove genuine biological signal as well as any reagent "
        "artifact — the partial correlation is not a clean 'artifact-removed' number, "
        "and is not interpretable as one until the stain-versus-cellularity question "
        "is resolved."
    )
    lines.append("")

    (output_dir / "stageE_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stageE_diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(result, output_dir)
    print(f"  Markdown report: {output_dir / 'stageE_diagnostic.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint cohort Stage E: diagnostic analysis (not a validation claim)"
    )
    parser.add_argument("--stageD-json", required=True, type=Path)
    parser.add_argument("--stageB-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint cohort — Stage E: diagnostic analysis")
    print("=" * 60)
    print("\n  DIAGNOSTIC despite a FAILED (hematoxylin-specific) stain gate.")
    print("  NOT a validated timepoint result.\n")

    for p in (args.stageD_json, args.stageB_json):
        if not p.exists():
            sys.exit(f"ERROR: required input not found:\n  {p}")

    with open(args.stageD_json) as f:
        stageD = json.load(f)
    with open(args.stageB_json) as f:
        stageB = json.load(f)

    if "mouse_level_by_suffix" not in stageD:
        sys.exit("ERROR: Stage D JSON has no 'mouse_level_by_suffix' — it predates "
                 "the both-variants output. Re-run Stage D.")

    rng_seed = args.seed
    blocks, blocks_drop, joined_by_variant = {}, {}, {}
    for variant in ("excluding_suffix", "including_suffix"):
        d_rows = stageD["mouse_level_by_suffix"][variant]
        b_rows = stageB["mouse_level_features"][variant]
        joined = join_mouse_tables(d_rows, b_rows, variant)
        joined_by_variant[variant] = joined
        print(f"\n=== {variant}: n={len(joined)} (mouse, timepoint) rows ===")

        blocks[variant] = compute_block(
            joined, args.n_permutations, np.random.default_rng(rng_seed))
        b = blocks[variant]
        print(f"  RAW rho={_fmt(b['raw']['rho'])} perm_p="
              f"{_format_perm_p(b['raw']['perm_p'], args.n_permutations)}")
        for m in H_MEASURES:
            print(f"  PARTIAL | {m}: rho={_fmt(b['partial'][m]['rho'])}")
            print(f"  {m} vs weeks: rho={_fmt(b['hematoxylin_vs_weeks'][m]['rho'])}")

        dropped = [r for r in joined if r["mouse_id"] != DUAL_TIMEPOINT_MOUSE]
        blocks_drop[variant] = compute_block(
            dropped, args.n_permutations, np.random.default_rng(rng_seed))

    per_slide = stageD.get("per_slide", [])
    baseline_ok = stageD.get("training_support_baseline", {}).get("available", False)
    projection_validity = {
        "baseline_available": baseline_ok,
        "reason": stageD.get("training_support_baseline", {}).get("reason"),
        "extrapolation_fraction_warn": stageD.get("extrapolation_fraction_warn"),
        "n_slides_total": len(per_slide),
        "n_slides_substantially_extrapolated": sum(
            1 for e in per_slide if e.get("substantially_extrapolated")),
        "extrapolated_slides": [
            e["raw_stem"] for e in per_slide if e.get("substantially_extrapolated")],
        "by_timepoint": stageD.get("by_timepoint", {}),
    }

    result = {
        "what_this_is": (
            "Diagnostic run despite a FAILED, hematoxylin-specific stain gate. Reports "
            "whether projected pseudotime carries information beyond the one known-"
            "confounded channel. NOT a validated timepoint result; not to be "
            "interpreted or shown as one until the PI's biology question and the "
            "stain-versus-cellularity disambiguation are resolved."
        ),
        "n_permutations": args.n_permutations,
        "projection_validity": projection_validity,
        "blocks": blocks,
        "blocks_drop_6072": blocks_drop,
        "dual_timepoint_aside": dual_timepoint_aside(
            joined_by_variant["including_suffix"]),
        "joined_mouse_rows": joined_by_variant,
    }
    write_outputs(result, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE E COMPLETE")
    print("=" * 60)
    print("\n  Report the RAW and PARTIAL correlations together with the projection-"
          "validity\n  section. Do not present any of it as a validated timepoint "
          "result. STOP HERE.")


if __name__ == "__main__":
    main()
