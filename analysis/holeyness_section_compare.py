"""Side-by-side holey-ness validation report: 2M-2 against the 2M-1 reference.

REPORTS NUMBERS. DOES NOT ADJUDICATE.

ESTIMAND AND FRAMING — fixed before the numbers are seen
--------------------------------------------------------
Per the collaborating pathologist, **duct diameter increases as lesions progress
and hole count increases with diameter**. Duct area is therefore a MEDIATOR on
the causal path from progression to holey-ness, not a nuisance covariate.

Consequently:
  * the RAW correlation is the PRIMARY estimate;
  * the area-adjusted partial is a deliberately OVER-ADJUSTED sensitivity
    analysis, reported alongside and never as a correction.
Adjusting for a mediator removes part of the very effect being estimated, so a
smaller adjusted value is expected under the lab's own biology and is not
evidence against the association.

⚠ FIXATION DIFFERS BETWEEN THE SECTIONS
---------------------------------------
2M-1 is Carnoy's-fixed; 2M-2 is PFA-fixed, and its QuPath measurement was
configured separately (source prefix ``holes_pfa:`` vs ``holes_carnoys:``). The
two fixatives differ in shrinkage behaviour, so hole % distributions may differ
between sections for FIXATION reasons alone, independent of biology.

That does not threaten either section's WITHIN-section correlation, which is what
this report compares. It does mean the two sections' hole % values are not on a
guaranteed-common scale, so absolute distributional differences must not be read
as biology.

⚠ ONE ASYMMETRY IN THE INPUTS, STATED NOT HIDDEN
------------------------------------------------
The 2M-1 reference values were computed against ``per_section`` (baseline)
pseudotime; 2M-2 is run against ``per_section_v2``. Pseudotime is bit-identical
between those trees (verified rho = 1.0 in the v2 re-run), so:

  EXACTLY comparable : rho(pt, hole_pct), the area-adjusted partial,
                       rho(hole_pct, area), the estimand counts, and every
                       within-slide quantity
  FOOTNOTED          : rho(hole_pct, packing_irregularity) and the
                       area+nuclear_density partial, because the Task 1 feature
                       fixes changed packing_irregularity (nan below 3 nuclei)

Read-only. Writes only --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 2M-1 reference, supplied by the lab; NOT recomputed here.
REF_2M1 = {
    "section": "2M-1",
    "fixative": "Carnoy's",
    "rho_raw": 0.276,
    "partial_area": 0.131,
    "partial_area_nd": 0.158,
    "n_ducts_measured": 2173,
    "n_retained": 1602,
    "n_excluded_zero_patch": 571,
    "excluded_profile": "systematically the SMALLEST and LEAST holey ducts",
    "pseudotime_source": "per_section (baseline)",
    "multipolygon_dropped": "12/2242 (0.5%)",
}

FOOTNOTED = {"rho_pi_hole_pct", "partial_rho_pt_hole_given_area_and_nd"}


def _g(d: dict, *keys, default=None):
    """Fetch the first present key from a possibly-nested result dict."""
    for k in keys:
        if k in d:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            got = _g(v, *keys, default=None)
            if got is not None:
                return got
    return default


def _f(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def build_report(j2: dict, out: Path, n_perm: int | None) -> None:
    raw = _g(j2, "rho_pt_hole_pct")
    p_area = _g(j2, "partial_rho_pt_hole_given_area")
    p_area_nd = _g(j2, "partial_rho_pt_hole_given_area_and_nd")
    rho_nd = _g(j2, "rho_nd_hole_pct")
    rho_pi = _g(j2, "rho_pi_hole_pct")
    rho_area_hole = _g(j2, "rho_area_hole_pct")

    L = [
        "# Holey-ness validation — 2M-2 beside the 2M-1 reference", "",
        "**This report states numbers. It does not declare either section's result "
        "better, nor holey-ness validated or refuted.**", "",
        "## Framing, fixed before the numbers", "",
        "Per the collaborating pathologist, duct diameter increases as lesions "
        "progress and hole count increases with diameter. **Duct area is a mediator "
        "of progression, not a nuisance covariate.** The raw correlation is "
        "therefore the **primary estimate**; the area-adjusted partial is a "
        "deliberately **over-adjusted sensitivity analysis**, never a correction. "
        "Adjusting for a mediator removes part of the effect being estimated, so a "
        "smaller adjusted value is expected under the lab's own biology.", "",
        "## ⚠ The two sections differ in fixative", "",
        "| | 2M-1 | 2M-2 |", "|---|---|---|",
        "| Fixative | Carnoy's | **PFA** |",
        "| QuPath measurement prefix | `holes_carnoys:` | **`holes_pfa:`** |",
        "| Input format | one merged TSV | **8 per-slide GeoJSON** |",
        "",
        "Carnoy's and PFA differ in shrinkage behaviour, so hole % distributions may "
        "differ between sections **for fixation reasons alone**, independent of "
        "biology. This does not threaten either section's within-section "
        "correlation — which is what is compared below — but absolute "
        "distributional differences between sections must not be read as biology.",
        "",
        "The converted TSV's header says `holes_carnoys:` because `holeyness.py` "
        "hardcodes that spelling. **The header misstates the fixative for 2M-2**; "
        "values are untouched. See the `.provenance.json` sidecar.", "",
        "## Primary estimate", "",
        "| quantity | 2M-1 (reference) | 2M-2 | note |",
        "|---|---|---|---|",
        f"| **rho(pseudotime, hole_pct) — PRIMARY** | {REF_2M1['rho_raw']:.3f} | "
        f"**{_f(raw, 3)}** | exactly comparable |",
        f"| partial, controlling duct AREA *(over-adjusted)* | "
        f"{REF_2M1['partial_area']:.3f} | {_f(p_area, 3)} | exactly comparable |",
        f"| partial, controlling area + nuclear density *(over-adjusted)* | "
        f"{REF_2M1['partial_area_nd']:.3f} | {_f(p_area_nd, 3)} | see footnote † |",
        "",
        "## Independence checks", "",
        "Is hole % just restating a feature the pipeline already measures?", "",
        "| quantity | 2M-2 |", "|---|---|",
        f"| rho(hole_pct, region-level nuclear_density) | {_f(rho_nd)} |",
        f"| rho(hole_pct, region-level packing_irregularity) | {_f(rho_pi)} † |",
        f"| rho(hole_pct, duct area) | {_f(rho_area_hole)} |",
        "",
        "## Estimand — the analysis population", "",
        "The population is **single-polygon Tumor ducts with a numeric hole % and "
        "at least one assigned patch**. Three exclusions apply, in order:", "",
        "| exclusion | 2M-1 | 2M-2 |", "|---|---|---|",
        f"| MultiPolygon (load_duct_polygons accepts Polygon only) | "
        f"{REF_2M1['multipolygon_dropped']} | 23/1776 (1.3%) |",
        "| non-numeric hole % (QuPath wrote \"NaN\") | not reported | 4/1776 (0.23%) |",
        f"| zero assigned patches | {REF_2M1['n_excluded_zero_patch']}/"
        f"{REF_2M1['n_ducts_measured']} | see v2 report |",
        "",
        f"For 2M-1 the zero-patch exclusions were {REF_2M1['excluded_profile']} — "
        "i.e. exactly the population a low-holey-ness analysis most depends on. "
        "The 2M-2 exclusion-bias block in `holeyness_validation_v2.md` reports "
        "whether the same holds; read it before generalising either section's "
        "correlation beyond this population.", "",
        "Separately, 7 ducts in 2M-2 are below 100 µm² — degenerate annotation "
        "artifacts, all in `6027-4L-2M-2`. A 112 px patch is ~2440 µm², so these "
        "are orders of magnitude smaller than a single patch and cannot receive a "
        "patch centre; they fall out at the zero-patch stage. They were **not** "
        "filtered at conversion, so the population was not silently changed.", "",
        "## Permutation null", "",
        "Ducts are nested within slides, so the null shuffles **within slide**, not "
        "globally — a global shuffle would break the nesting and inflate "
        "significance. p-values are reported as `< 1/n_permutations` rather than "
        "0.0 when no shuffle exceeds the observed value"
        + (f" (n = {n_perm})." if n_perm else ".") +
        " See `within_slide_permutation` in the v2 JSON.", "",
        "## Within-slide correlations", "",
        "All 8 slides, with mean / median / range / number positive, are in the v2 "
        "report's within-slide block. If the pooled correlation held only between "
        "slides and vanished within them, that would be a Simpson's-paradox "
        "artifact; `holeyness.py` computes the slide-level median correlation "
        "explicitly to rule that in or out.", "",
        "## Patch-count sensitivity", "",
        "Thresholds ≥3, ≥5, ≥10, ≥20 assigned patches are in the v2 report. "
        "**≥1 is the primary population** and is the headline value above, not a "
        "separate row.", "",
        "---", "",
        "† **Footnote.** The 2M-1 reference was computed against `per_section` "
        "(baseline) pseudotime; 2M-2 was run against `per_section_v2`. Pseudotime "
        "is bit-identical between those trees, so every other quantity above is "
        "exactly comparable. Only `rho(hole_pct, packing_irregularity)` and the "
        "area+nuclear-density partial rest on a different feature version, because "
        "the Task 1 fixes changed `packing_irregularity` to nan below 3 nuclei.", "",
        "## Not to be read as validation", "",
        "- A smaller area-adjusted value is **expected** under the lab's stated "
        "biology (area is a mediator) and is not evidence against the association.",
        "- Agreement between the two sections' raw correlations would be "
        "encouraging but is not proof: the sections differ in fixative, and both "
        "analyses share the same patch-to-duct assignment rule and therefore the "
        "same exclusion bias.",
        "- Nothing here speaks to whether holey-ness should anchor pseudotime. "
        "That is Phase 2.", "",
    ]
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-json-2M2", type=Path, required=True,
                    help="holeyness_validation_v2.json from the 2M-2 v2 run.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Markdown report to write (NEW file).")
    ap.add_argument("--n-permutations", type=int, default=None)
    args = ap.parse_args()

    if not args.v2_json_2M2.exists():
        raise SystemExit(f"ERROR: {args.v2_json_2M2} not found — run Phase 1 first.")
    if args.output.exists():
        raise SystemExit(f"ERROR: {args.output} exists; refusing to overwrite.")

    j2 = json.loads(args.v2_json_2M2.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_report(j2, args.output, args.n_permutations)
    print(f"Wrote {args.output}")
    print("Raw rho is the PRIMARY estimate; area-adjusted is an over-adjusted "
          "sensitivity analysis, not a correction.")


if __name__ == "__main__":
    main()
