"""Repaired-axis SENSITIVITY check, reported beside the primary result.

PRE-DECLARED BEFORE ANY NUMBER WAS SEEN
---------------------------------------
This is a SENSITIVITY CHECK. The PRIMARY external validation remains the v2
density-rooted result — rho(pt, hole_pct) = +0.2763 (2M-1) and +0.1906 (2M-2),
with the exact slide-level permutation test showing no evidence the sections
differ (p = 0.313 on the within-slide normal-scores variant, minimum detectable
difference 0.133). The repaired axis does NOT replace those numbers, and nothing
in this report should be read as the headline.

The hypothesis is MECHANISTIC, not a search for a better correlation: that the
decisive between-section divergence in rho(duct area, pseudotime) — +0.4325 vs
-0.0844, exact p = 1.55e-4 — is an artifact of 2M-2's degenerate root set rather
than a property of the tissue.

THREE PREDICTIONS, RECORDED IN ADVANCE:

  (a) 2M-2's rho(area, pseudotime) moves from -0.0844 toward zero-or-positive.
  (b) 2M-2's pseudotime_std drops from 27.70% of axis range toward ~3.40%.
  (c) The between-section difference in rho(area, pseudotime) falls below the
      0.353 minimum detectable difference of the primary run.

Each is reported as HELD or DID NOT HOLD, computed rather than asserted. A failed
prediction is reported plainly; the hypothesis is not reframed around whatever
the data turns out to show.

THE BUILT-IN NEGATIVE CONTROL, AND THE GATE IT NEEDS
----------------------------------------------------
2M-1 had ZERO discordant roots, so its repaired axis should be identical to its
source and every 2M-1 quantity should be unchanged. Any movement there is
unexpected and is flagged prominently.

That control has a confound which this module checks rather than assumes. The two
sections' PRIMARY validations were run against different run trees:

    2M-1 primary  ->  $SCRATCH/results/per_section/atlas_2M-1/results.csv
    2M-2 primary  ->  $SCRATCH/results/per_section_v2/atlas_2M-2/results.csv

but the repaired axis derives from ``per_section_v2`` for BOTH. So for 2M-1 the
sensitivity run changes two things at once: the repair (a no-op there) and the
baseline tree. If those two trees carry different pseudotime, a 2M-1 "change"
would be a tree difference rather than anything about the repair. ``--baseline-
csvs`` supplies the two 2M-1 results.csv files and this module compares their
pseudotime columns directly. If they disagree, the 2M-1 arm is declared
CONFOUNDED and the negative control is not interpreted.

READ-ONLY. Reads finished JSON and per-duct tables. Recomputes no pseudotime and
no duct table, reruns no pipeline stage, modifies no module, and writes only to
--output-dir. It never touches the primary outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Pre-declared reference values. Carried so the report can state them before any
# recomputed number appears, and so a mismatch against the finished JSON is
# visible rather than silent.
PRIMARY_REFERENCE = {
    "raw_rho_pt_hole": {"2M-1": 0.2763, "2M-2": 0.1906},
    "rho_area_pseudotime": {"2M-1": 0.4325, "2M-2": -0.0844},
    "pseudotime_std_pct": {"2M-1": 5.03, "2M-2": 27.70},
    "within_slide_variant_p": 0.313,
    "within_slide_variant_mdd": 0.133,
    "area_pt_exact_p": 1.55e-4,
    "area_pt_mdd": 0.3528,
}
PREDICTION_B_TARGET_PCT = 3.40
NEG_CONTROL_TOL = 0.005          # |delta| in rho above this is "changed"
REFERENCE_TOL = 0.01             # tolerance when checking JSON against the brief

ESTIMAND_KEYS = ("raw_rho_pt_hole", "partial_given_area", "partial_given_area_nd")
DIFF_KEYS = ("raw_rho_pt_hole", "partial_given_area", "rho_area_pseudotime",
             "raw_within_slide_normalised")


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


def _tree_label(p) -> str:
    """`<run tree>/<section dir>` for a results.csv path.

    The baseline gate compares two files that share a parent NAME —
    per_section/atlas_2M-1/results.csv and per_section_v2/atlas_2M-1/results.csv
    — so labelling by parent alone printed "atlas_2M-1 vs atlas_2M-1" and left
    the reader unable to tell what was compared. Two components disambiguate the
    run trees, which is the whole point of the gate.
    """
    q = Path(p)
    return f"{q.parent.parent.name}/{q.parent.name}"


def _rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return None
    return float(spearmanr(x[ok], y[ok]).statistic)


def _load_json(p: Path, what: str) -> dict:
    if not Path(p).exists():
        raise FileNotFoundError(
            f"{p} not found — {what}. This report reads finished outputs and does "
            "not regenerate them.")
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ── the baseline gate ────────────────────────────────────────────────────────

def baseline_gate(csv_a: Path, csv_b: Path, section: str) -> dict:
    """Do the two run trees carry the same pseudotime for this section?

    Without this, a 2M-1 "change" could be a baseline-tree difference rather than
    anything about the repair, and the negative control would be meaningless.
    """
    a = pd.read_csv(csv_a)
    b = pd.read_csv(csv_b)
    out = {"section": section, "csv_a": str(csv_a), "csv_b": str(csv_b),
           "n_rows_a": int(len(a)), "n_rows_b": int(len(b))}
    if len(a) != len(b):
        out.update({"identical": False,
                    "reason": f"row counts differ ({len(a)} vs {len(b)})"})
        return out
    pa = a["pseudotime"].values.astype(float)
    pb = b["pseudotime"].values.astype(float)
    max_abs = float(np.nanmax(np.abs(pa - pb)))
    rho = _rho(pa, pb)
    out.update({
        "max_abs_difference": max_abs,
        "spearman": rho,
        "identical": bool(max_abs < 1e-9),
        "interpretation": (
            "If identical, the 2M-1 sensitivity arm changes only the repair (a "
            "no-op there) and the negative control is valid. If not, the arm also "
            "changes the baseline tree and the control is CONFOUNDED."),
    })
    return out


# ── per-section quantities recomputed from the per-duct tables ───────────────

def per_section_quantities(csv: Path) -> dict:
    """rho(area, pseudotime) and rho(pt, hole_pct), straight from a per-duct table.

    The section-comparison JSON stores rho(area, pseudotime) only as a
    between-section DIFFERENCE, so the per-section values are recomputed here
    rather than quoted from the analysis brief.
    """
    d = pd.read_csv(csv)
    for c in ("pseudotime", "hole_pct", "area_um2"):
        if c not in d.columns:
            raise KeyError(f"{csv} is missing '{c}'.")
    return {
        "path": str(csv),
        "n_ducts": int(len(d)),
        "n_slides": int(d["slide_name"].nunique()) if "slide_name" in d else None,
        "rho_area_pseudotime": _rho(d["area_um2"].values, d["pseudotime"].values),
        "rho_pt_hole_pct": _rho(d["pseudotime"].values, d["hole_pct"].values),
    }


# ── predictions ──────────────────────────────────────────────────────────────

def evaluate_predictions(primary_q: dict, repaired_q: dict, anchor_rules: dict,
                         sens_json: dict, prim_json: dict) -> dict:
    preds = {}

    # (a) 2M-2 rho(area, pseudotime) moves toward zero-or-positive
    was = primary_q["2M-2"]["rho_area_pseudotime"]
    now = repaired_q["2M-2"]["rho_area_pseudotime"]
    moved_up = bool(now is not None and was is not None and now > was)
    reached = bool(now is not None and now >= 0.0)
    preds["a"] = {
        "statement": ("2M-2's rho(area, pseudotime) moves from -0.0844 toward "
                      "zero-or-positive"),
        "primary": was, "repaired": now,
        "delta": (float(now - was) if (now is not None and was is not None) else None),
        "moved_toward_zero_or_positive": moved_up,
        "reached_zero_or_positive": reached,
        "held": bool(moved_up),
        "note": ("'Held' requires movement in the predicted direction. Whether it "
                 "actually reached zero-or-positive is reported separately, since "
                 "the prediction as written asks only for movement toward it."),
    }

    # (b) 2M-2 pseudotime_std drops from 27.70% toward ~3.40%
    ar = anchor_rules.get("2M-2", {})
    before = ar.get("pseudotime_std_pct_of_range_all_roots")
    after = ar.get("pseudotime_std_pct_of_range_repaired")
    preds["b"] = {
        "statement": ("2M-2's pseudotime_std drops from 27.70% of axis range "
                      f"toward ~{PREDICTION_B_TARGET_PCT}%"),
        "all_roots_pct": before, "repaired_pct": after,
        "target_pct": PREDICTION_B_TARGET_PCT,
        "primary_reference_pct": PRIMARY_REFERENCE["pseudotime_std_pct"]["2M-2"],
        "held": bool(after is not None and before is not None
                     and after < before
                     and abs(after - PREDICTION_B_TARGET_PCT) <= 1.0),
        "note": ("Read from anchor_axis.json, not recomputed. 'Held' requires both "
                 "a drop and landing within 1 percentage point of the value Task E "
                 "recorded."),
    }

    # (c) between-section difference in rho(area, pt) falls below the primary MDD
    mdd = (prim_json.get("task_2", {}).get("rho_area_pseudotime", {})
           .get("minimum_detectable_difference_alpha05"))
    obs_prim = (prim_json.get("task_2", {}).get("rho_area_pseudotime", {})
                .get("observed_difference"))
    obs_sens = (sens_json.get("task_2", {}).get("rho_area_pseudotime", {})
                .get("observed_difference"))
    p_sens = (sens_json.get("task_2", {}).get("rho_area_pseudotime", {})
              .get("exact_p_two_sided"))
    preds["c"] = {
        "statement": ("the between-section difference in rho(area, pseudotime) "
                      "falls below the primary run's minimum detectable "
                      "difference"),
        "primary_observed_difference": obs_prim,
        "primary_mdd_threshold": mdd,
        "repaired_observed_difference": obs_sens,
        "repaired_exact_p": p_sens,
        "repaired_own_mdd": (sens_json.get("task_2", {})
                             .get("rho_area_pseudotime", {})
                             .get("minimum_detectable_difference_alpha05")),
        "held": bool(obs_sens is not None and mdd is not None
                     and abs(obs_sens) < mdd),
        "note": ("Judged against the PRE-DECLARED threshold — the primary run's "
                 "MDD — not against the repaired run's own. The repaired run's MDD "
                 "and exact p are reported alongside for context."),
    }
    preds["n_held"] = int(sum(1 for k in ("a", "b", "c") if preds[k]["held"]))
    return preds


# ── negative control ─────────────────────────────────────────────────────────

def negative_control(primary_q: dict, repaired_q: dict, prim_json: dict,
                     sens_json: dict, anchor_rules: dict, gate: dict | None,
                     section: str = "2M-1") -> dict:
    ar = anchor_rules.get(section, {})
    deltas = {}
    for key in ("rho_area_pseudotime", "rho_pt_hole_pct"):
        was, now = primary_q[section][key], repaired_q[section][key]
        deltas[key] = {"primary": was, "repaired": now,
                       "delta": (float(now - was)
                                 if (was is not None and now is not None) else None)}
    for key in ESTIMAND_KEYS:
        was = (prim_json["task_1"]["per_section"][section][key]["point_estimate"])
        now = (sens_json["task_1"]["per_section"][section][key]["point_estimate"])
        deltas[key] = {"primary": was, "repaired": now, "delta": float(now - was)}

    moved = {k: v for k, v in deltas.items()
             if v["delta"] is not None and abs(v["delta"]) > NEG_CONTROL_TOL}
    confounded = bool(gate is not None and not gate.get("identical", False))

    return {
        "section": section,
        "identical_to_source_recorded": ar.get("identical_to_source"),
        "n_dropped_roots": ar.get("n_dropped"),
        "deltas": deltas,
        "tolerance": NEG_CONTROL_TOL,
        "n_quantities_moved": len(moved),
        "quantities_moved": sorted(moved),
        "baseline_gate": gate,
        "confounded_by_baseline_tree": confounded,
        "verdict": (
            (f"CONFOUNDED. The two run trees do not carry the same {section} "
             "pseudotime, so this arm changes the baseline as well as the repair "
             "and the negative control cannot be interpreted. Any movement below "
             "may be the tree, not the repair.")
            if confounded else
            (f"CLEAN. {section} had {ar.get('n_dropped')} discordant root(s) and "
             f"nothing moved by more than {NEG_CONTROL_TOL} — the negative control "
             "behaves as predicted, so any change seen in the other section comes "
             "from that section alone.")
            if not moved else
            (f"UNEXPECTED. {section} should be unchanged "
             f"(identical_to_source = {ar.get('identical_to_source')}, "
             f"{ar.get('n_dropped')} roots dropped) but "
             f"{sorted(moved)} moved by more than {NEG_CONTROL_TOL}. Investigate "
             "before reading anything into the other section's changes.")),
    }


# ── report ───────────────────────────────────────────────────────────────────

def write_report(res: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    secs = res["sections"]
    prim, sens = res["primary_json"], res["sensitivity_json"]
    preds, nc = res["predictions"], res["negative_control"]

    add("# Repaired-axis sensitivity check — reported beside the primary result\n")

    add("## Pre-declaration (recorded before any number here was seen)\n")
    add("- **This is a SENSITIVITY CHECK.** The PRIMARY external validation "
        "remains the v2 density-rooted result. The repaired axis does not replace "
        "it and nothing below is the headline.")
    add("- The hypothesis is **mechanistic**, not a search for a better "
        "correlation: that the between-section divergence in "
        "`rho(duct area, pseudotime)` is an artifact of 2M-2's degenerate root "
        "set rather than a property of the tissue.")
    add("- Three predictions were recorded in advance and each is reported below "
        "as held or not held, computed rather than asserted.")
    add("- 2M-1 had zero discordant roots and acts as a **negative control**.\n")

    add("## The PRIMARY result (unchanged, and still the headline)\n")
    add("| quantity | 2M-1 | 2M-2 |")
    add("|---|---|---|")
    for key, label in (("raw_rho_pt_hole", "`rho(pt, hole_pct)` raw"),
                       ("partial_given_area", "`rho(pt, hole_pct \\| area)`"),
                       ("partial_given_area_nd", "`rho(pt, hole_pct \\| area, nd)`")):
        cells = []
        for s in secs:
            e = prim["task_1"]["per_section"][s][key]
            ci = e["bootstrap"]["ci95"]
            cells.append(f"{e['point_estimate']:+.4f} "
                         + (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))
        add(f"| {label} | {cells[0]} | {cells[1]} |")
    wsv = prim["task_2"]["raw_within_slide_normalised"]
    add(f"\nExact slide-level permutation on the between-section difference, "
        f"within-slide normal-scores variant: observed "
        f"{wsv['observed_difference']:+.4f}, p = {wsv['exact_p_two_sided']:.4g}, "
        f"MDD {wsv['minimum_detectable_difference_alpha05']:.4f} — **no evidence "
        "the sections differ.**\n")

    add("## Predictions\n")
    add("| # | prediction | result | held? |")
    add("|---|---|---|---|")
    a, b, c = preds["a"], preds["b"], preds["c"]
    add(f"| (a) | {a['statement']} | {a['primary']:+.4f} → {a['repaired']:+.4f} "
        f"(delta {a['delta']:+.4f}) | **{'HELD' if a['held'] else 'DID NOT HOLD'}** |")
    add(f"| (b) | {b['statement']} | "
        + (f"{b['all_roots_pct']:.2f}% → {b['repaired_pct']:.2f}%"
           if b["repaired_pct"] is not None else "unavailable")
        + f" | **{'HELD' if b['held'] else 'DID NOT HOLD'}** |")
    add(f"| (c) | {c['statement']} | |diff| "
        + (f"{abs(c['repaired_observed_difference']):.4f} vs threshold "
           f"{c['primary_mdd_threshold']:.4f}"
           if c["repaired_observed_difference"] is not None else "unavailable")
        + f" | **{'HELD' if c['held'] else 'DID NOT HOLD'}** |")
    add(f"\n{preds['n_held']} of 3 predictions held.\n")
    for k in ("a", "b", "c"):
        add(f"- **({k})** {preds[k]['note']}")
    add("")

    add("## Side by side — primary vs repaired\n")
    add("### Four-cell correlation table\n")
    add("| section | estimand | primary (v2) | repaired | delta |")
    add("|---|---|---|---|---|")
    for s in secs:
        for key in ESTIMAND_KEYS:
            p = prim["task_1"]["per_section"][s][key]["point_estimate"]
            q = sens["task_1"]["per_section"][s][key]["point_estimate"]
            add(f"| {s} | `{key}` | {p:+.4f} | {q:+.4f} | {q - p:+.4f} |")

    add("\n### Per-section `rho(area, pseudotime)`\n")
    add("| section | primary (v2) | repaired | delta |")
    add("|---|---|---|---|")
    for s in secs:
        p = res["per_section"]["primary"][s]["rho_area_pseudotime"]
        q = res["per_section"]["repaired"][s]["rho_area_pseudotime"]
        add(f"| {s} | {p:+.4f} | {q:+.4f} | {q - p:+.4f} |")

    add("\n### Exact between-section permutation tests\n")
    add("| quantity | primary diff | primary p | repaired diff | repaired p | "
        "primary MDD |")
    add("|---|---|---|---|---|---|")
    for key in DIFF_KEYS:
        p = prim["task_2"][key]
        q = sens["task_2"][key]
        add(f"| `{key}` | {p['observed_difference']:+.4f} | "
            f"{p['exact_p_two_sided']:.4g} | {q['observed_difference']:+.4f} | "
            f"{q['exact_p_two_sided']:.4g} | "
            f"{p['minimum_detectable_difference_alpha05']:.4f} |")

    add("\n### pseudotime_std, as a percentage of each run's own raw range\n")
    add("| section | v2 (all roots) | repaired | roots dropped |")
    add("|---|---|---|---|")
    for s in secs:
        ar = res["anchor_rules"].get(s, {})
        add(f"| {s} | "
            + (f"{ar.get('pseudotime_std_pct_of_range_all_roots'):.2f}%"
               if ar.get("pseudotime_std_pct_of_range_all_roots") is not None else "—")
            + " | "
            + (f"{ar.get('pseudotime_std_pct_of_range_repaired'):.2f}%"
               if ar.get("pseudotime_std_pct_of_range_repaired") is not None else "—")
            + f" | {ar.get('n_dropped')} |")
    add("\nRead from `anchor_axis.json`, not recomputed.\n")

    add("## Negative control — 2M-1\n")
    add(f"**Verdict:** {nc['verdict']}\n")
    g = nc.get("baseline_gate")
    if g:
        add(f"- baseline gate: `{_tree_label(g['csv_a'])}` vs "
            f"`{_tree_label(g['csv_b'])}` pseudotime — identical: "
            f"**{g.get('identical')}**"
            + (f", max abs difference {g['max_abs_difference']:.3g}"
               if g.get("max_abs_difference") is not None else "")
            + (f", Spearman {g['spearman']:.6f}" if g.get("spearman") else ""))
        add(f"  - {g['interpretation']}")
    else:
        add("- baseline gate: **NOT RUN** (`--baseline-csvs` not supplied). The "
            "2M-1 arm changes the baseline tree as well as the repair, and that "
            "has not been checked.")
    add(f"- `identical_to_source` recorded in anchor_axis.json: "
        f"**{nc['identical_to_source_recorded']}**; roots dropped: "
        f"{nc['n_dropped_roots']}")
    add(f"- quantities moving by more than {nc['tolerance']}: "
        f"**{nc['n_quantities_moved']}**"
        + (f" ({', '.join(nc['quantities_moved'])})" if nc["quantities_moved"] else ""))
    add("")

    add("## Provenance — `anchor_rule`, copied verbatim\n")
    for s in secs:
        add(f"### {s}\n")
        add("```json")
        add(json.dumps(res["anchor_rules"].get(s, {}), indent=2,
                       default=_json_default))
        add("```\n")

    add("## Standing caveats\n")
    add("- **The drop rule keeps whichever orientation the majority of roots "
        "share.** It removes minority disagreement and cannot detect a majority "
        "of bad roots. It works here only because 3 of 20 disagree — a fact about "
        "this data, not a property of the rule.")
    add("- **The rule was applied after the discordance was observed.** It was "
        "fixed in advance, but this is still a second look, which is why the "
        "repaired axis is a sensitivity check and not the primary.")
    add("- **Fixation is perfectly collinear with section.** Every Carnoy's slide "
        "is 2M-1 and every PFA slide is 2M-2, so nothing here attributes anything "
        "to fixation chemistry as opposed to anatomical region. Bridge samples "
        "would be required.")
    add("- **Eight slides per section.** Failure to reject is not evidence of "
        "equivalence; the minimum detectable differences above are large and "
        "differences up to them cannot be excluded.")
    add("- Nothing was recomputed from images or embeddings, and no primary "
        "output directory was written to.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs=2, default=["2M-1", "2M-2"])
    ap.add_argument("--primary-json", type=Path, required=True)
    ap.add_argument("--sensitivity-json", type=Path, required=True)
    ap.add_argument("--primary-per-duct-csvs", nargs=2, type=Path, required=True)
    ap.add_argument("--repaired-per-duct-csvs", nargs=2, type=Path, required=True)
    ap.add_argument("--anchor-axis-jsons", nargs=2, type=Path, required=True,
                    help="anchor_axis.json per section, SAME ORDER as --sections.")
    ap.add_argument("--baseline-csvs", nargs=2, type=Path, default=None,
                    help="The two 2M-1 results.csv files (per_section and "
                         "per_section_v2) for the negative-control gate. Omitted "
                         "-> the gate is reported as NOT RUN, never assumed.")
    ap.add_argument("--baseline-section", default="2M-1")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    secs = list(args.sections)
    print("=" * 78)
    print("  Repaired-axis sensitivity check")
    print("=" * 78)

    prim = _load_json(args.primary_json, "the PRIMARY section comparison")
    sens = _load_json(args.sensitivity_json, "the repaired-axis section comparison")
    anchor_rules = {}
    for i, s in enumerate(secs):
        aj = _load_json(args.anchor_axis_jsons[i], f"{s}'s anchor_axis.json")
        anchor_rules[s] = aj.get("anchor_rule", {})
        if aj.get("anchor") != "v2_repaired":
            print(f"  WARNING: {args.anchor_axis_jsons[i]} records anchor "
                  f"{aj.get('anchor')!r}, not 'v2_repaired'.")

    per_section = {"primary": {}, "repaired": {}}
    for i, s in enumerate(secs):
        per_section["primary"][s] = per_section_quantities(args.primary_per_duct_csvs[i])
        per_section["repaired"][s] = per_section_quantities(args.repaired_per_duct_csvs[i])
        pq, rq = per_section["primary"][s], per_section["repaired"][s]
        print(f"  {s}: ducts {pq['n_ducts']} -> {rq['n_ducts']}   "
              f"rho(area,pt) {pq['rho_area_pseudotime']:+.4f} -> "
              f"{rq['rho_area_pseudotime']:+.4f}")
        if pq["n_ducts"] != rq["n_ducts"]:
            raise ValueError(
                f"{s}: duct retention changed ({pq['n_ducts']} -> "
                f"{rq['n_ducts']}). Duct assignment does not depend on pseudotime, "
                "so this means the two runs used different inputs and the "
                "comparison is not like-for-like.")

    gate = None
    if args.baseline_csvs:
        gate = baseline_gate(args.baseline_csvs[0], args.baseline_csvs[1],
                             args.baseline_section)
        print(f"\n  baseline gate ({args.baseline_section}): identical = "
              f"{gate.get('identical')}, max abs diff "
              f"{gate.get('max_abs_difference')}")
    else:
        print("\n  baseline gate: NOT RUN (--baseline-csvs not supplied)")

    preds = evaluate_predictions(per_section["primary"], per_section["repaired"],
                                 anchor_rules, sens, prim)
    for k in ("a", "b", "c"):
        print(f"  prediction ({k}): "
              f"{'HELD' if preds[k]['held'] else 'DID NOT HOLD'}")

    nc = negative_control(per_section["primary"], per_section["repaired"],
                          prim, sens, anchor_rules, gate, args.baseline_section)
    print(f"\n  negative control: {nc['verdict'][:100]}...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "analysis": "holeyness_repaired_sensitivity",
        "status": "SENSITIVITY CHECK — the primary v2 result is unchanged and "
                  "remains the headline",
        "sections": secs,
        "pre_declared_reference": PRIMARY_REFERENCE,
        "inputs": {
            "primary_json": str(args.primary_json),
            "sensitivity_json": str(args.sensitivity_json),
            "primary_per_duct_csvs": [str(p) for p in args.primary_per_duct_csvs],
            "repaired_per_duct_csvs": [str(p) for p in args.repaired_per_duct_csvs],
            "anchor_axis_jsons": [str(p) for p in args.anchor_axis_jsons],
            "recomputed_anything": False,
        },
        "per_section": per_section,
        "anchor_rules": anchor_rules,
        "predictions": preds,
        "negative_control": nc,
        "primary_json": prim,
        "sensitivity_json": sens,
    }
    out = args.output_dir / "holeyness_repaired_sensitivity.json"
    out.write_text(json.dumps(res, indent=2, default=_json_default),
                   encoding="utf-8")
    write_report(res, args.output_dir / "holeyness_repaired_sensitivity.md")
    print(f"\n  JSON:     {out}")
    print(f"  Markdown: {args.output_dir / 'holeyness_repaired_sensitivity.md'}")


if __name__ == "__main__":
    main()
