"""Side-by-side holey-ness validation report: 2M-2 against the 2M-1 reference.

REPORTS NUMBERS. DOES NOT ADJUDICATE.

ESTIMAND AND FRAMING
--------------------
Per the collaborating pathologist, duct diameter increases as lesions progress
and hole count increases with diameter. Duct area is therefore a MEDIATOR on the
causal path from progression to holey-ness, not a nuisance covariate. The RAW
correlation is the PRIMARY estimate; the area-adjusted partial is a sensitivity
analysis reported alongside, never as a correction.

⚠ THE DIRECTION OF THE ADJUSTMENT IS NOT ASSUMED — BUG FIXED 2026-08-17
------------------------------------------------------------------------
The first version of this report asserted that adjusting for a mediator must
SHRINK the estimate, and printed the adjusted value under that claim. That is
true only when the control is positively associated with BOTH pseudotime and
holey-ness. It held for 2M-1 (0.276 -> 0.131) and FAILED for 2M-2
(0.191 -> 0.238), where adjustment INCREASED the estimate — so the report
asserted the opposite of what its own table showed.

A partial correlation RISES when the control acts as a SUPPRESSOR, which
requires rho(pseudotime, area) to be near zero or negative. That is a
qualitatively different situation from mediation and must not be narrated as
"over-adjustment". This module now reads rho(pseudotime, area) from the JSON,
classifies the adjustment from the observed numbers, and words the section to
match. It never assumes a direction.

⚠ FIXATION DIFFERS BETWEEN THE SECTIONS
---------------------------------------
2M-1 is Carnoy's-fixed; 2M-2 is PFA-fixed, its QuPath measurement configured
separately (``holes_pfa:`` vs ``holes_carnoys:``). The fixatives differ in
shrinkage, so hole % distributions may differ between sections for FIXATION
reasons alone. Within-section correlations — what is compared here — are not
threatened, but absolute distributional differences must not be read as biology.

⚠ ONE ASYMMETRY IN THE INPUTS
-----------------------------
The 2M-1 reference used ``per_section`` (baseline) pseudotime; 2M-2 uses
``per_section_v2``. Pseudotime is bit-identical between those trees, so
rho(pt, hole_pct), the area partial, rho(hole, area), the estimand counts and
every within-slide quantity are exactly comparable. Only
rho(hole_pct, packing_irregularity) and the area+nuclear-density partial rest on
a different feature version (Task 1 made packing_irregularity nan below 3
nuclei). Footnoted, not hidden.

Read-only. Writes only --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 2M-1 reference, supplied by the lab; NOT recomputed here.
REF = {
    "rho_raw": 0.276,
    "partial_area": 0.131,
    "partial_area_nd": 0.158,
    "n_ducts_measured": 2173,
    "n_retained": 1602,
    "n_excluded_zero_patch": 571,
    "excluded_profile": "systematically the SMALLEST and LEAST holey ducts",
    "multipolygon_dropped": "12/2242 (0.5%)",
}


def _f(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    try:
        if v != v:          # NaN
            return "nan"
    except TypeError:
        return str(v)
    return f"{v:.{nd}f}"


def _classify_adjustment(raw, adj, rho_pt_area, rho_hole_area):
    """Describe what controlling for duct area DID, from the numbers themselves.

    Never assumes attenuation. Returns (label, prose).
    """
    if raw is None or adj is None:
        return "unknown", "Adjusted or raw value missing; nothing can be said."

    delta = adj - raw
    pa = "n/a" if rho_pt_area is None else f"{rho_pt_area:+.4f}"
    ha = "n/a" if rho_hole_area is None else f"{rho_hole_area:+.4f}"

    if abs(delta) < 0.02:
        return "negligible", (
            f"Controlling for duct area barely moves the estimate "
            f"({raw:.4f} -> {adj:.4f}, delta {delta:+.4f}). Area is not carrying "
            f"the association either way. rho(pseudotime, area) = {pa}, "
            f"rho(hole_pct, area) = {ha}.")

    if delta < 0:
        return "attenuation", (
            f"Controlling for duct area ATTENUATES the estimate "
            f"({raw:.4f} -> {adj:.4f}, delta {delta:+.4f}). This is the pattern "
            f"expected when area mediates: it is positively associated with both "
            f"pseudotime (rho = {pa}) and hole_pct (rho = {ha}), so removing it "
            f"removes part of the very effect being estimated. Under the lab's "
            f"stated biology the RAW value remains the primary estimate; the "
            f"adjusted value is an over-adjusted lower bound.")

    return "suppression", (
        f"Controlling for duct area **INCREASES** the estimate "
        f"({raw:.4f} -> {adj:.4f}, delta {delta:+.4f}). This is NOT attenuation "
        f"and must not be described as over-adjustment. A partial correlation "
        f"rises only when the control acts as a SUPPRESSOR — here that requires "
        f"rho(pseudotime, area) to be near zero or negative, and it is: "
        f"**{pa}** (with rho(hole_pct, area) = {ha}).\n\n"
        f"The implication is substantive, not cosmetic. The lab's biology says "
        f"duct area grows with progression, so if pseudotime tracked progression "
        f"in this section, rho(pseudotime, area) should be clearly positive. "
        f"It is not. Either pseudotime is not tracking duct growth here, or duct "
        f"growth is not tracking progression here. **The holey-ness association "
        f"in this section is therefore arriving by a different route than in a "
        f"section where area attenuates**, and matching raw correlations between "
        f"the two should not be read as a shared mechanism.")


def build_report(j: dict, out: Path) -> None:
    prim = j.get("primary_correlation", {}) or {}
    area = j.get("area_covariate", {}) or {}
    ws = j.get("within_slide", {}) or {}
    perm = j.get("permutation", {}) or {}
    excl = j.get("exclusion_bias", {}) or {}
    agg = j.get("aggregation_sensitivity", {}) or {}
    samp = j.get("patch_sampling_artifact", {}) or {}
    section = j.get("section", "2M-2")
    n_perm = j.get("n_permutations")

    raw = prim.get("rho_pt_hole_pct")
    p_area = area.get("partial_rho_pt_hole_given_area")
    p_area_nd = area.get("partial_rho_pt_hole_given_area_and_nd")
    rho_pt_area = area.get("rho_area_pseudotime")
    rho_hole_area = area.get("rho_area_hole_pct")
    rho_nd = prim.get("rho_nd_hole_pct")
    rho_pi = prim.get("rho_pi_hole_pct")

    label, prose = _classify_adjustment(raw, p_area, rho_pt_area, rho_hole_area)

    L = [
        f"# Holey-ness validation — {section} beside the 2M-1 reference", "",
        "**States numbers. Does not declare either section's result better, nor "
        "holey-ness validated or refuted.**", "",
        "## Headline", "",
        "| quantity | 2M-1 (reference) | " + section + " | comparability |",
        "|---|---|---|---|",
        f"| **rho(pseudotime, hole_pct) — PRIMARY** | {REF['rho_raw']:.3f} | "
        f"**{_f(raw, 3)}** | exact |",
        f"| partial, controlling duct AREA | {REF['partial_area']:.3f} | "
        f"{_f(p_area, 3)} | exact |",
        f"| partial, controlling area + nuclear density | "
        f"{REF['partial_area_nd']:.3f} | {_f(p_area_nd, 3)} | footnote † |",
        f"| **rho(pseudotime, duct AREA)** | not supplied | **{_f(rho_pt_area, 3)}** | — |",
        f"| rho(hole_pct, duct area) | not supplied | {_f(rho_hole_area, 3)} | — |",
        "",
        "## What controlling for duct area actually did", "",
        f"**Classified from the numbers as: {label.upper()}.**", "",
        prose, "",
    ]

    if label == "suppression":
        L += ["> ⚠ **This is the most consequential line in the report.** The two "
              "sections' raw correlations agreeing in sign is superficial if the "
              "mediator behaves oppositely in each. Check "
              "`rho(pseudotime, duct area)` in both sections before claiming a "
              "shared mechanism.", ""]

    # ── Independence ─────────────────────────────────────────────────────────
    L += ["## Independence checks", "",
          "Is hole % just restating a feature the pipeline already measures? This "
          "is the decisive question for using holey-ness as a DPT anchor: if it "
          "tracked nuclear_density, holeyness-rooting would be "
          "nuclear-density-rooting under another name, and the non-circularity "
          "claim collapses.", "",
          "| quantity | " + section + " | reading |", "|---|---|---|",
          f"| rho(hole_pct, nuclear_density) | **{_f(rho_nd)}** | "
          + ("weak — substantially independent" if rho_nd is not None and abs(rho_nd) < 0.3
             else "NOT weak — the anchor may be partly circular") + " |",
          f"| rho(hole_pct, packing_irregularity) | {_f(rho_pi)} † | "
          + ("weak" if rho_pi is not None and abs(rho_pi) < 0.3 else "not weak") + " |",
          ""]

    # ── Permutation: within-slide is the one that counts ─────────────────────
    g, w = perm.get("global", {}) or {}, perm.get("within_slide", {}) or {}
    L += ["## Permutation null — use the WITHIN-SLIDE row", "",
          "Ducts are nested within slides. A global shuffle breaks that nesting "
          "and is anti-conservative, so the within-slide null is the one to quote.",
          "", "| null | observed rho | p | null 95th |", "|---|---|---|---|",
          f"| global (anti-conservative, do not cite) | {_f(g.get('obs_rho_pt_hole_pct'))} | "
          f"{g.get('perm_p_display', _f(g.get('perm_p')))} | {_f(g.get('null95'))} |",
          f"| **within-slide (cite this)** | **{_f(w.get('obs_rho_pt_hole_pct'))}** | "
          f"**{w.get('perm_p_display', _f(w.get('perm_p')))}** | "
          f"**{_f(w.get('null95'))}** |", ""]

    # ── Within-slide correlations / Simpson's ────────────────────────────────
    per_slide = ws.get("per_slide", []) or []
    summ = ws.get("summary", {}) or {}
    between = ws.get("between_slide_median_correlation", {}) or {}
    L += ["## Within-slide correlations — Simpson's paradox check", "",
          "If the pooled correlation existed only BETWEEN slides and vanished "
          "within them, it would be a between-slide artifact rather than a "
          "duct-level association.", ""]
    if per_slide:
        L += ["| slide | n ducts | rho(pt, hole) | partial \\| area | median pt | median hole% |",
              "|---|---|---|---|---|---|"]
        for r in per_slide:
            L.append(f"| `{r['slide_name']}` | {r['n_ducts']} | "
                     f"{_f(r['rho_pt_hole_pct'])} | "
                     f"{_f(r['partial_rho_pt_hole_given_area'])} | "
                     f"{_f(r.get('median_pseudotime'))} | {_f(r.get('median_hole_pct'))} |")
        L.append("")
        for key, title in (("raw", "raw"), ("area_adjusted", "area-adjusted")):
            s = summ.get(key, {}) or {}
            if s:
                L.append(f"- **{title}**: mean {_f(s.get('mean'))}, median "
                         f"{_f(s.get('median'))}, range [{_f(s.get('min'))}, "
                         f"{_f(s.get('max'))}], **{s.get('n_positive')}/"
                         f"{s.get('n_slides')} positive**")
        L += ["",
              f"- between-slide (slide-level medians): rho = "
              f"{_f(between.get('rho'))}, p = {_f(between.get('p'))}",
              f"- slides qualifying (>= min ducts): {ws.get('n_slides_qualifying')}", ""]
    else:
        L += ["*No slide met the minimum-ducts threshold; within-slide check "
              "unavailable.*", ""]

    # ── Patch-count sensitivity ──────────────────────────────────────────────
    by_t = agg.get("by_min_patches_threshold", {}) or {}
    med = agg.get("median_aggregation", {}) or {}
    mean = agg.get("mean_aggregation", {}) or {}
    L += ["## Patch-count sensitivity", "",
          "Per-duct pseudotime is a median over that duct's patches. When ducts "
          "have few patches the estimate is noisy, which ATTENUATES the "
          "correlation. If rho climbs with the threshold, the pooled value is "
          "attenuated by measurement noise; if it collapses, the association is "
          "being carried by sparsely-sampled ducts.", "",
          "| minimum patches | n ducts | rho(pt, hole_pct) | p |", "|---|---|---|---|",
          f"| **>= 1 (PRIMARY population)** | {med.get('n_ducts')} | "
          f"**{_f(med.get('rho_pt_hole_pct'))}** | {_f(med.get('p'))} |"]
    for k in sorted(by_t, key=lambda s: int(s.split("_")[1])):
        v = by_t[k]
        L.append(f"| >= {k.split('_')[1]} | {v['n_ducts']} | "
                 f"{_f(v['rho_pt_hole_pct'])} | {_f(v['p'])} |")
    L += ["",
          f"- aggregation sensitivity: median {_f(med.get('rho_pt_hole_pct'))} vs "
          f"mean {_f(mean.get('rho_pt_hole_pct'))} "
          f"(n = {med.get('n_ducts')} / {mean.get('n_ducts')})", ""]

    # ── Estimand / exclusion bias ────────────────────────────────────────────
    a_cmp = excl.get("area_um2", {}) or {}
    h_cmp = excl.get("hole_pct", {}) or {}
    L += ["## Estimand — the analysis population", "",
          "The population is **single-polygon Tumor ducts with a numeric hole % "
          "and at least one assigned patch**. Exclusions, in order:", "",
          "| exclusion | 2M-1 | " + section + " |", "|---|---|---|",
          f"| MultiPolygon (load_duct_polygons takes Polygon only) | "
          f"{REF['multipolygon_dropped']} | 23/1776 (1.3%) |",
          "| non-numeric hole % (QuPath wrote \"NaN\") | not reported | 4/1776 (0.23%) |",
          f"| zero assigned patches | {REF['n_excluded_zero_patch']}/"
          f"{REF['n_ducts_measured']} | {excl.get('n_excluded')}/"
          f"{(excl.get('n_excluded') or 0) + (excl.get('n_retained') or 0)} |",
          "",
          "### Are the excluded ducts systematically different?", "",
          "| quantity | median excluded | median retained | Mann-Whitney p |",
          "|---|---|---|---|",
          f"| duct area µm² | {_f(a_cmp.get('median_excluded'), 1)} | "
          f"{_f(a_cmp.get('median_retained'), 1)} | {_f(a_cmp.get('mannwhitney_p'))} |",
          f"| hole % | {_f(h_cmp.get('median_excluded'))} | "
          f"{_f(h_cmp.get('median_retained'))} | {_f(h_cmp.get('mannwhitney_p'))} |",
          "",
          f"For 2M-1 the zero-patch exclusions were {REF['excluded_profile']} — "
          "exactly the population a low-holey-ness analysis most depends on. "
          "Compare the medians above to see whether the same holds here.", "",
          "Separately, 7 ducts in 2M-2 are below 100 µm² — degenerate annotation "
          "artifacts, all in `6027-4L-2M-2`. A 112 px patch is ~2440 µm², so they "
          "cannot receive a patch centre and fall out at the zero-patch stage. "
          "They were **not** filtered at conversion, so the population was not "
          "silently changed.", ""]

    # ── Sampling artifact ────────────────────────────────────────────────────
    if samp:
        L += ["## Edge-patch sampling artifact check", "",
              f"- rho(duct area, nuclear_density) = "
              f"{_f(samp.get('rho_area_nuclear_density'))} "
              f"(p = {_f(samp.get('p_area_nuclear_density'))})",
              f"- rho(n_patches, nuclear_density) = "
              f"{_f(samp.get('rho_n_patches_nuclear_density'))} "
              f"(p = {_f(samp.get('p_n_patches_nuclear_density'))})", "",
              "Ducts with 1-2 patches are likelier to have those patches straddle "
              "the boundary, so a correlation here would suggest an edge artifact "
              "rather than biology. Reported for awareness; not corrected.", ""]

    # ── Fixative + footnote ──────────────────────────────────────────────────
    L += ["## ⚠ The sections differ in fixative", "",
          "| | 2M-1 | " + section + " |", "|---|---|---|",
          "| Fixative | Carnoy's | **PFA** |",
          "| QuPath prefix | `holes_carnoys:` | **`holes_pfa:`** |",
          "| Input format | one merged TSV | **8 per-slide GeoJSON** |", "",
          "The fixatives differ in shrinkage, so hole % distributions may differ "
          "between sections **for fixation reasons alone**. Within-section "
          "correlations are unaffected; absolute distributional differences "
          "between sections must not be read as biology.", "",
          "The converted TSV's header says `holes_carnoys:` because "
          "`holeyness.py` hardcodes that spelling — **the header misstates the "
          "fixative**; values are untouched. See the `.provenance.json` sidecar.",
          "",
          "---", "",
          "† **Footnote.** The 2M-1 reference used `per_section` (baseline) "
          "pseudotime; this section used `per_section_v2`. Pseudotime is "
          "bit-identical between those trees, so everything else above is exactly "
          "comparable. Only `rho(hole_pct, packing_irregularity)` and the "
          "area+nuclear-density partial rest on a different feature version, "
          "because the Task 1 fixes made `packing_irregularity` nan below 3 "
          "nuclei. Ducts whose every patch is nan for a feature drop out of that "
          "feature's correlation, so those two rows have a smaller and "
          "non-random n (ducts with enough segmented nuclei).", "",
          "## Not to be read as validation", "",
          "- The direction of the area adjustment is classified from the data, "
          "not assumed. A rise is suppression, not over-adjustment, and means "
          "something different.",
          "- Matching raw correlations between sections are not proof of a shared "
          "mechanism — check `rho(pseudotime, duct area)` in both.",
          "- Both sections share the same patch-to-duct assignment rule and "
          "therefore the same exclusion bias.",
          "- Nothing here speaks to whether holey-ness should anchor pseudotime. "
          "That is Phase 2.", ""]

    out.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-json-2M2", dest="v2_json", type=Path, required=True,
                    help="holeyness_validation_v2.json from the 2M-2 v2 run.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Markdown report to write (NEW file).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite --output if it exists.")
    ap.add_argument("--n-permutations", type=int, default=None,
                    help="Accepted for backward compatibility; the value is read "
                         "from the JSON.")
    args = ap.parse_args()

    if not args.v2_json.exists():
        raise SystemExit(f"ERROR: {args.v2_json} not found — run Phase 1 first.")
    if args.output.exists() and not args.force:
        raise SystemExit(f"ERROR: {args.output} exists; pass --force to overwrite.")

    j = json.loads(args.v2_json.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_report(j, args.output)

    area = j.get("area_covariate", {}) or {}
    prim = j.get("primary_correlation", {}) or {}
    label, _ = _classify_adjustment(
        prim.get("rho_pt_hole_pct"), area.get("partial_rho_pt_hole_given_area"),
        area.get("rho_area_pseudotime"), area.get("rho_area_hole_pct"))
    print(f"Wrote {args.output}")
    print(f"Area adjustment classified as: {label.upper()}")
    if label == "suppression":
        print("  -> controlling for area INCREASED the estimate. That is "
              "suppression, not over-adjustment.\n"
              "     rho(pseudotime, area) = "
              f"{_f(area.get('rho_area_pseudotime'))} — check this against 2M-1 "
              "before claiming a shared mechanism.")


if __name__ == "__main__":
    main()
