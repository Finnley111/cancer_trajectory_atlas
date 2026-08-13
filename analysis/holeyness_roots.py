"""Select DPT roots from expert-annotated per-duct holey-ness (v3 Config A / C).

WHY THIS EXISTS
---------------
The production root rule anchors pseudotime at the 20 patches with the lowest
MEASURED ``nuclear_density``. But ``nuclear_density`` is simultaneously

  (a) the root selector,
  (b) one of the six morphological features pseudotime is VALIDATED against
      (``validation/correlations.py``), and
  (c) the covariate partialled out in ``analysis/cellularity_confound.py``.

So the axis is partly DEFINED by a quantity it is later validated against, and
partly by the quantity used to adjust that validation. Neither the reported
nuclear_density correlation nor the "collapses under cellularity adjustment"
finding is independent of the anchor.

Miranda's per-duct ``holes_carnoys: hole %`` comes from HAND ANNOTATION in
QuPath, not from the pipeline's pixels. Rooting on it therefore REMOVES the
circularity rather than reducing it, and makes all six morphological features
independent validators for the first time. It also supplies a defensible
DIRECTION: expert judgment is that duct diameter increases as lesions progress
and holes increase with diameter, so LOW holey-ness = EARLY.

WHAT TO EXPECT, STATED BEFORE RUNNING
-------------------------------------
Uniformly random 20-root sets already reproduce the production pseudotime at
|rho| 0.78-0.89. The manifold fixes the ORDERING; roots fix only which end is
zero. So changing the root rule is EXPECTED to change the axis ORIENTATION and
the root set, and NOT to change the ordering. Assess the result against that
expectation. |rho| < 0.7 against v2 would CONTRADICT the random-root finding and
needs explaining before anything downstream is trusted.

THE SELECTION RULE, FIXED IN ADVANCE AND NOT TUNED
--------------------------------------------------
1. Ducts are the ``"Tumor"`` polygons in ``data/annotations_ratio``, joined to
   the measurement export by QuPath object UUID. Loaded by
   ``holeyness.load_duct_polygons`` / ``holeyness.build_duct_table``, reused
   unmodified.
2. Patches inherit a duct by CENTRE-IN-POLYGON, via
   ``holeyness.assign_patches_to_ducts``, reused unmodified so this experiment is
   comparable to the existing holey-ness analysis, inherited bias and all.
3. Patches in NO duct are EXCLUDED from root candidacy. They are NOT assigned
   hole % = 0: that would make them the lowest-holeyness patches in the cohort
   and therefore the PREFERRED roots, which is the exact failure this anchor is
   meant to avoid.
4. The candidate pool is every duct at or below the ``percentile``-th percentile
   of the PER-DUCT hole % distribution. Over ducts, not over patches, so a few
   large ducts cannot drag the threshold. A percentile rather than "the 20
   lowest" so the anchor is not a handful of extreme ducts.
5. Ducts need ``min_patches_per_duct`` assigned patches (default 1). Raising it
   would exclude more SMALL ducts, and small ducts are already the population the
   centre-in-polygon rule under-samples — see the inherited limitation below.
6. Roots are taken ONE PER DUCT, ducts ascending by hole %, patch = the one whose
   centre is nearest its duct centroid. This stops all 20 roots collapsing into
   one large duct.
7. TIE-BREAK. A bottom decile of hole % is very likely to be entirely 0.0, which
   makes the percentile degenerate. Ties are broken by ``object_id`` UUID string
   order. That is arbitrary but reproducible, and — the point — INDEPENDENT of
   every downstream quantity, so it cannot leak result-dependence into the anchor.

INHERITED LIMITATION, REPORTED NOT FIXED
----------------------------------------
The centre-in-polygon rule excluded 571 of 2,173 ducts (26%) in the earlier
holey-ness analysis, and those were systematically the SMALLEST and LEAST holey
ducts — i.e. exactly the population a low-holeyness root rule draws from. This
module does not attempt to fix that (``holeyness_final.py``'s Task F explores an
area-overlap rule instead); it REPORTS how many ducts are available as candidates
here so the comparison is visible.

READ-ONLY. Reads the measurement export, the ratio annotations and
slide_dimensions.json. Writes nothing itself — ``run_all.py`` saves the returned
report to ``<out>/holeyness_roots.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .holeyness import (
    PATCH_SIZE_DEFAULT,
    load_slide_dimensions,
    parse_measurement_export,
    load_duct_polygons,
    build_duct_table,
    assign_patches_to_ducts,
)


def select_holeyness_roots(
    coords: np.ndarray,
    slide_ids: np.ndarray,
    slide_names: Sequence[str],
    annotation_dir: Path,
    export_path: Path,
    slide_dimensions_path: Path,
    n_roots: int = 20,
    percentile: float = 10.0,
    min_patches_per_duct: int = 1,
    patch_size: int = PATCH_SIZE_DEFAULT,
) -> tuple[list[int], dict]:
    """Return (root_indices, report).

    ``coords`` are the pipeline's top-left (x, y) in cropped-PNG pixel space and
    ``slide_ids`` index into ``slide_names`` — i.e. exactly what ``run_all``
    holds at PHASE 4, in the same row order as ``adata``. The returned indices
    are positions in that array, which is what ``compute_dpt_multi_root`` wants.

    Raises rather than degrading if the candidate pool cannot supply ``n_roots``
    distinct ducts. A silently short root set would change the median-across-
    roots aggregation without saying so.
    """
    print("\n  === Holeyness root selection ===")
    coords = np.asarray(coords)
    slide_ids = np.asarray(slide_ids)
    if coords.shape[0] != slide_ids.shape[0]:
        raise ValueError(
            f"coords has {coords.shape[0]} rows but slide_ids has "
            f"{slide_ids.shape[0]} — these must be the same patch array."
        )

    # results.csv-shaped frame; assign_patches_to_ducts needs only these three
    # columns, and building it here avoids depending on a completed run.
    results_df = pd.DataFrame({
        "x": coords[:, 0].astype(float),
        "y": coords[:, 1].astype(float),
        "slide_name": [slide_names[int(s)] for s in slide_ids],
    })
    pipeline_slides = sorted(set(results_df["slide_name"]))

    slide_dims = load_slide_dimensions(Path(slide_dimensions_path))
    measurements = parse_measurement_export(Path(export_path), pipeline_slides)
    polygons = load_duct_polygons(Path(annotation_dir), pipeline_slides, slide_dims)
    duct_table = build_duct_table(measurements, polygons)

    if len(duct_table) == 0:
        raise ValueError(
            "The duct table is empty — no measurement row joined to a Tumor "
            "polygon by UUID. Check that --holeyness-export matches the slides "
            "being run and that --annotation-dir is the RATIO directory."
        )

    assigned = assign_patches_to_ducts(results_df, duct_table, patch_size=patch_size)
    duct_id = assigned["duct_id"].values

    n_no_duct = int(sum(1 for d in duct_id if d is None))
    n_total_patches = len(results_df)
    print(f"  Patches in no duct (EXCLUDED from candidacy): "
          f"{n_no_duct}/{n_total_patches} ({100*n_no_duct/n_total_patches:.1f}%)")

    # ── Per-duct patch counts, and the hole % of each duct with >=1 patch ─────
    hole_by_duct = dict(zip(duct_table["object_id"], duct_table["hole_pct"]))
    area_by_duct = dict(zip(duct_table["object_id"], duct_table["area_um2"]))
    slide_by_duct = dict(zip(duct_table["object_id"], duct_table["slide_name"]))

    patches_by_duct: dict[str, list[int]] = {}
    for i, d in enumerate(duct_id):
        if d is not None:
            patches_by_duct.setdefault(d, []).append(i)

    n_ducts_total = len(duct_table)
    n_ducts_with_patches = len(patches_by_duct)
    eligible = {
        d: idxs for d, idxs in patches_by_duct.items()
        if len(idxs) >= min_patches_per_duct and np.isfinite(hole_by_duct.get(d, np.nan))
    }
    print(f"  Ducts in table            : {n_ducts_total}")
    print(f"  Ducts with >=1 patch      : {n_ducts_with_patches} "
          f"({n_ducts_total - n_ducts_with_patches} with 0 patches, excluded — this is "
          "the inherited centre-in-polygon bias)")
    print(f"  Ducts with >={min_patches_per_duct} patch(es) : {len(eligible)}")

    if not eligible:
        raise ValueError(
            "No duct has enough assigned patches to be a root candidate. Either "
            "the annotations do not match these slides or "
            "--holeyness-min-patches is too high."
        )

    # ── Candidate pool: bottom `percentile` of the PER-DUCT hole % ────────────
    eligible_holes = np.array([hole_by_duct[d] for d in eligible], dtype=float)
    threshold = float(np.percentile(eligible_holes, percentile))
    pool = [d for d in eligible if hole_by_duct[d] <= threshold]
    degenerate = bool(np.all(np.asarray([hole_by_duct[d] for d in pool]) == threshold))
    print(f"  Hole % P{percentile:g} threshold   : {threshold:.4f}")
    print(f"  Ducts in candidate pool   : {len(pool)}")
    if degenerate:
        print("  NOTE: every duct in the pool has the SAME hole % as the threshold "
              "— the percentile is degenerate and the UUID tie-break is doing the "
              "selecting. This is expected when many ducts have 0 holes.")

    if len(pool) < n_roots:
        raise ValueError(
            f"Candidate pool has {len(pool)} ducts but {n_roots} roots were "
            f"requested, and roots are taken one per duct. Raise "
            f"--holeyness-percentile or lower --n-roots. Refusing to return a "
            "short root set, which would silently reweight the median across roots."
        )

    # ── One root per duct, ducts ascending by hole % then UUID ────────────────
    pool_sorted = sorted(pool, key=lambda d: (float(hole_by_duct[d]), str(d)))

    root_indices: list[int] = []
    root_rows: list[dict] = []
    half = patch_size / 2.0
    for d in pool_sorted[:n_roots]:
        idxs = eligible[d]
        # The most CENTRAL of this duct's patches: nearest the mean of its own
        # assigned patch centres, all in cropped-PNG pixels. Deliberately NOT the
        # QuPath centroid, which is in um and would need a scale conversion this
        # module has no reason to own. Picking the central patch keeps the root
        # off the duct boundary, where a 112 px window straddles two tissues.
        # Deterministic, and independent of every downstream quantity.
        px = coords[idxs, 0].astype(float) + half
        py = coords[idxs, 1].astype(float) + half
        d2 = (px - px.mean()) ** 2 + (py - py.mean()) ** 2
        chosen = int(idxs[int(np.argmin(d2))])
        root_indices.append(chosen)
        root_rows.append({
            "patch_index":  chosen,
            "duct_id":      str(d),
            "slide_name":   str(slide_by_duct.get(d)),
            "hole_pct":     float(hole_by_duct[d]),
            "duct_area_um2": float(area_by_duct.get(d, np.nan)),
            "n_patches_in_duct": int(len(idxs)),
            "x": int(coords[chosen, 0]),
            "y": int(coords[chosen, 1]),
        })

    if len(set(root_indices)) != len(root_indices):
        raise ValueError(
            "Selected root indices contain duplicates — one patch was reachable "
            "from two ducts. Overlapping Tumor polygons break the one-root-per-"
            "duct guarantee; inspect the annotations before proceeding."
        )

    print(f"  Selected {len(root_indices)} roots from {len(root_indices)} distinct ducts")
    print(f"    hole % range   : [{root_rows[0]['hole_pct']:.4f}, "
          f"{root_rows[-1]['hole_pct']:.4f}]")
    areas = np.array([r["duct_area_um2"] for r in root_rows], dtype=float)
    finite_areas = areas[np.isfinite(areas)]
    if finite_areas.size:
        print(f"    duct area um^2 : median {np.median(finite_areas):.0f}, "
              f"range [{finite_areas.min():.0f}, {finite_areas.max():.0f}]")
    slides_hit = sorted({r["slide_name"] for r in root_rows})
    print(f"    slides covered : {len(slides_hit)} of {len(pipeline_slides)}")

    all_eligible_areas = np.array(
        [area_by_duct.get(d, np.nan) for d in eligible], dtype=float)
    all_eligible_areas = all_eligible_areas[np.isfinite(all_eligible_areas)]

    report = {
        "rule": {
            "assignment": "patch centre inside Tumor polygon (holeyness.assign_patches_to_ducts, reused)",
            "no_duct_patches": "EXCLUDED from candidacy, never assigned hole %=0",
            "pool": f"ducts at or below the P{percentile:g} of the PER-DUCT hole % distribution",
            "min_patches_per_duct": min_patches_per_duct,
            "selection": "one patch per duct, ducts ascending by hole %, patch nearest the mean of that duct's own assigned patch centres",
            "tie_break": "object_id UUID string order — arbitrary, reproducible, and independent of every downstream quantity",
            "direction": "LOW hole % = EARLY (expert judgment: duct diameter increases with progression, holes increase with diameter)",
        },
        "expectation": (
            "Random 20-root sets reproduce v2 pseudotime at |rho| 0.78-0.89, so a "
            "root-rule change is EXPECTED to alter orientation and root membership, "
            "not ordering. |rho| < 0.7 vs v2 would contradict that and needs "
            "explanation before any downstream number is trusted."
        ),
        "counts": {
            "n_patches_total": int(n_total_patches),
            "n_patches_no_duct": int(n_no_duct),
            "frac_patches_no_duct": float(n_no_duct / n_total_patches),
            "n_ducts_in_table": int(n_ducts_total),
            "n_ducts_with_zero_patches": int(n_ducts_total - n_ducts_with_patches),
            "n_ducts_with_min_patches": int(len(eligible)),
            "n_ducts_in_candidate_pool": int(len(pool)),
            "percentile": float(percentile),
            "hole_pct_threshold": threshold,
            "pool_is_degenerate_all_at_threshold": degenerate,
        },
        "inherited_limitation": (
            "The centre-in-polygon rule excluded 571/2173 ducts (26%) in the earlier "
            "holeyness analysis, and those were systematically the SMALLEST and LEAST "
            "holey — the same population this low-holeyness rule draws from. Here "
            f"{n_ducts_total - n_ducts_with_patches}/{n_ducts_total} "
            f"({100*(n_ducts_total-n_ducts_with_patches)/max(n_ducts_total,1):.1f}%) "
            "ducts have zero patches and are unavailable as candidates. Not fixed "
            "here; reported so the estimand is visible."
        ),
        "selected_roots": root_rows,
        "root_duct_area_um2": {
            "median": float(np.median(finite_areas)) if finite_areas.size else None,
            "min": float(finite_areas.min()) if finite_areas.size else None,
            "max": float(finite_areas.max()) if finite_areas.size else None,
        },
        "all_eligible_duct_area_um2_median": (
            float(np.median(all_eligible_areas)) if all_eligible_areas.size else None
        ),
        "slides_covered": slides_hit,
        "n_slides_in_run": len(pipeline_slides),
    }
    return root_indices, report
