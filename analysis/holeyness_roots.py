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

OVERLAP_MIN_FRACTION_DEFAULT = 0.25


# ── Fix 1: area-overlap patch-to-duct assignment ──────────────────────────────

def assign_patches_to_ducts_overlap(
    results_df: pd.DataFrame,
    duct_table: pd.DataFrame,
    patch_size: int = PATCH_SIZE_DEFAULT,
    overlap_min_fraction: float = OVERLAP_MIN_FRACTION_DEFAULT,
) -> pd.DataFrame:
    """Assign each patch to the duct covering the largest share of its AREA.

    The centre-in-polygon rule (``holeyness.assign_patches_to_ducts``) requires
    the patch's single centre pixel to fall inside a duct. A duct smaller than a
    112 px patch, or narrower than one, can therefore contain tissue in several
    patches while capturing the centre of none — which is how 571 of 2173 ducts
    (26%) ended up with zero patches, systematically the SMALLEST and LEAST holey
    ones. That is precisely the population a low-holeyness root rule must draw
    from, so under the centre rule the anchor is blind to it.

    This rule instead treats the patch as a 112x112 box and assigns it to the
    duct claiming the largest absolute intersection area, provided that area is
    at least ``overlap_min_fraction`` of the patch. Ties therefore resolve to
    "most covered", which is what the task specifies.

    Lifted from ``holeyness_final.py:run_overlap_sensitivity`` (Task F), which
    established the approach and the 0.25 default. That module is NOT modified —
    its v1-v3b consolidation output stays exactly as it was.

    NOTE ON HOLES. ``load_duct_polygons`` builds each MplPath from
    ``geometry["coordinates"][0]``, the OUTER ring only, so a duct's own lumen is
    not subtracted here. Overlap is with the duct's outer boundary. That matches
    the centre rule's behaviour, so the two assignments stay comparable.

    Raises ImportError with an actionable message if shapely is missing rather
    than silently falling back to the centre rule — a silent fallback would make
    the two configurations indistinguishable in the output.
    """
    try:
        from shapely.geometry import Polygon, box
        from shapely.strtree import STRtree
    except ImportError as e:                                     # pragma: no cover
        raise ImportError(
            f"Area-overlap assignment requires shapely ({e!r}). It is listed in "
            "requirements.txt; install it into ~/envs/atlas. Refusing to fall "
            "back to the centre-in-polygon rule, which would silently produce a "
            "different experiment under the same output label."
        ) from e

    print(f"\n=== Patch-to-duct assignment (AREA OVERLAP >= "
          f"{overlap_min_fraction:.0%} of patch) ===")
    results_df = results_df.copy()
    results_df["duct_id"] = None
    patch_area = float(patch_size) ** 2

    for slide_name, slide_df in results_df.groupby("slide_name"):
        slide_ducts = duct_table[duct_table["slide_name"] == slide_name].reset_index(drop=True)
        if len(slide_ducts) == 0:
            print(f"  {slide_name}: 0 ducts — {len(slide_df)} patches left unassigned")
            continue

        duct_polys, duct_ids = [], []
        n_repaired = 0
        for _, row in slide_ducts.iterrows():
            poly = Polygon(row["polygon"].vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)          # self-intersection repair
                n_repaired += 1
            duct_polys.append(poly)
            duct_ids.append(row["object_id"])
        tree = STRtree(duct_polys)

        xs = slide_df["x"].values.astype(float)
        ys = slide_df["y"].values.astype(float)
        assigned = np.full(len(slide_df), None, dtype=object)
        for i in range(len(slide_df)):
            pbox = box(xs[i], ys[i], xs[i] + patch_size, ys[i] + patch_size)
            best_id, best_area = None, 0.0
            for idx in tree.query(pbox):
                a = pbox.intersection(duct_polys[int(idx)]).area
                if a > best_area:                # largest ABSOLUTE overlap wins
                    best_area, best_id = a, duct_ids[int(idx)]
            if best_id is not None and (best_area / patch_area) >= overlap_min_fraction:
                assigned[i] = best_id
        results_df.loc[slide_df.index, "duct_id"] = assigned

        n_ass = int(sum(1 for a in assigned if a is not None))
        print(f"  {slide_name}: {n_ass}/{len(slide_df)} patches assigned "
              f"({100*n_ass/len(slide_df):.1f}%)  |  {len(slide_ducts)} ducts"
              + (f"  [{n_repaired} polygon(s) repaired]" if n_repaired else ""))

    total = int(results_df["duct_id"].notna().sum())
    print(f"  Overall: {total}/{len(results_df)} patches assigned "
          f"({100*total/len(results_df):.1f}%)")
    return results_df


# ── Fix 2: topological contiguity, as an ASSERTION not a selector ─────────────

def assert_roots_connected(adata, root_indices: Sequence[int]) -> dict:
    """Verify the roots occupy ONE connected component of the DPT k-NN graph.

    WHY AN ASSERTION RATHER THAN A SELECTION CRITERION
    --------------------------------------------------
    Selecting the seed's 19 nearest PCA neighbours would guarantee tightness, but
    20 mutually-adjacent patches are very likely 20 patches of the SAME duct on
    the SAME slide — which collapses the anchor to a single location and lets any
    local artifact there define the origin. Diversity across ducts is the thing
    the one-root-per-duct rule buys; this check keeps that and tests the property
    that actually matters instead.

    WHAT ACTUALLY GOES WRONG, mechanically. ``compute_dpt_multi_root`` runs each
    root through its OWN ``sc.tl.dpt`` call and medians the results — the walks
    never "collide". The real failure is the clamp: a root sitting in a small
    component returns inf for every patch outside it, those inf values are
    clamped to that root's own maximum, and a near-constant vector enters the
    median. Config B is exactly this: pseudotime_std at 30.5% of its range.

    Fails loudly if the neighbour graph is absent — it must be, since
    ``compute_diffusion_map`` runs before root selection in ``run_all``.

    Returns a diagnostic dict; also reports the max pairwise latent distance, in
    the SAME space and metric ``sc.pp.neighbors`` used (``adata.X``, euclidean —
    scanpy's default, since no ``metric=`` is ever passed). Reported only, never
    used to select.
    """
    from scipy.sparse.csgraph import connected_components

    if "connectivities" not in adata.obsp:
        raise RuntimeError(
            "adata.obsp['connectivities'] is missing, so root connectivity cannot "
            "be verified. compute_diffusion_map() must run before root selection. "
            "Refusing to proceed: an unverified root set is exactly what produced "
            "Config B's 30%-of-range pseudotime_std."
        )

    roots = [int(i) for i in root_indices]
    n_comp, labels = connected_components(adata.obsp["connectivities"], directed=False)
    root_comps = labels[roots]
    uniq, counts = np.unique(root_comps, return_counts=True)

    sizes = {int(c): int((labels == c).sum()) for c in uniq}
    print(f"\n  === Root topology check ===")
    print(f"  Graph components: {n_comp}   roots span: {len(uniq)}")
    for c, n in zip(uniq, counts):
        print(f"    component {int(c)}: {int(n)} root(s), {sizes[int(c)]} patches total")

    X = np.asarray(adata.X)
    sub = X[roots]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    max_pair = float(d.max())
    # Scale reference: how far apart are two RANDOM patches in the same space?
    rng = np.random.default_rng(0)
    samp = X[rng.choice(len(X), size=min(500, len(X)), replace=False)]
    ds = np.sqrt(((samp[:, None, :] - samp[None, :, :]) ** 2).sum(-1))
    median_random = float(np.median(ds[np.triu_indices_from(ds, k=1)]))
    print(f"  Max pairwise latent distance among roots : {max_pair:.4f}")
    print(f"  Median pairwise distance, random patches : {median_random:.4f}")
    print(f"  Ratio (roots / random)                   : {max_pair/median_random:.3f}"
          "   (<1 means the roots are tighter than chance)")

    if len(uniq) > 1:
        raise RuntimeError(
            f"The {len(roots)} DPT roots span {len(uniq)} DISCONNECTED components "
            f"of the k-NN graph (sizes {sizes}). Multi-root DPT would clamp inf to "
            "each root's own maximum and feed near-constant vectors into the "
            "median, which is what broke Config B. Refusing to proceed.\n"
            "  Fixes, in order of preference: (1) run with production tissue "
            "filters, which is where the manifold is connected; (2) raise "
            "--holeyness-percentile so the pool is not confined to one island; "
            "(3) investigate the graph with qc/graph_connectivity.py."
        )

    return {
        "n_graph_components": int(n_comp),
        "n_components_spanned_by_roots": int(len(uniq)),
        "root_component_sizes": sizes,
        "max_pairwise_latent_distance": max_pair,
        "median_pairwise_distance_random_patches": median_random,
        "tightness_ratio_vs_random": float(max_pair / median_random),
        "space": "adata.X (PCA / X_embed), euclidean — the same space and metric "
                 "sc.pp.neighbors used to build the DPT graph. NOT UMAP.",
        "note": "Reported as a diagnostic. Never used to select roots.",
    }


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
    assignment: str = "centre",
    overlap_min_fraction: float = OVERLAP_MIN_FRACTION_DEFAULT,
    max_roots_per_duct: int = 1,
    allow_degenerate_pool: bool = False,
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

    if assignment == "overlap":
        assigned = assign_patches_to_ducts_overlap(
            results_df, duct_table, patch_size=patch_size,
            overlap_min_fraction=overlap_min_fraction)
        # Also run the centre rule, purely to quantify what the overlap rule
        # SALVAGES. This is the 571/2173 (26%) number made concrete for this run.
        centre_assigned = assign_patches_to_ducts(
            results_df, duct_table, patch_size=patch_size)
        centre_ducts = {d for d in centre_assigned["duct_id"].values if d is not None}
    elif assignment == "centre":
        assigned = assign_patches_to_ducts(results_df, duct_table, patch_size=patch_size)
        centre_ducts = None
    else:
        raise ValueError(f"assignment must be 'centre' or 'overlap', got {assignment!r}")
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
          f"({n_ducts_total - n_ducts_with_patches} with 0 patches, excluded)")
    print(f"  Ducts with >={min_patches_per_duct} patch(es) : {len(eligible)}")

    # ── What the overlap rule salvaged ───────────────────────────────────────
    salvage = None
    if centre_ducts is not None:
        overlap_ducts = set(patches_by_duct)
        rescued = overlap_ducts - centre_ducts
        lost = centre_ducts - overlap_ducts
        r_area = np.array([area_by_duct.get(d, np.nan) for d in rescued], float)
        r_area = r_area[np.isfinite(r_area)]
        k_area = np.array([area_by_duct.get(d, np.nan) for d in centre_ducts], float)
        k_area = k_area[np.isfinite(k_area)]
        r_hole = np.array([hole_by_duct.get(d, np.nan) for d in rescued], float)
        r_hole = r_hole[np.isfinite(r_hole)]
        k_hole = np.array([hole_by_duct.get(d, np.nan) for d in centre_ducts], float)
        k_hole = k_hole[np.isfinite(k_hole)]
        print(f"\n  === Overlap rule vs centre rule ===")
        print(f"  Ducts with patches, centre rule : {len(centre_ducts)}")
        print(f"  Ducts with patches, overlap rule: {len(overlap_ducts)}")
        print(f"  SALVAGED (0 patches -> >=1)     : {len(rescued)}")
        print(f"  Lost (had patches -> 0)         : {len(lost)}")
        if r_area.size and k_area.size:
            print(f"  Salvaged duct area um^2 : median {np.median(r_area):.0f}  "
                  f"vs {np.median(k_area):.0f} for centre-rule ducts")
        if r_hole.size and k_hole.size:
            print(f"  Salvaged duct hole %    : median {np.median(r_hole):.3f}  "
                  f"vs {np.median(k_hole):.3f} for centre-rule ducts")
        salvage = {
            "n_ducts_centre_rule": len(centre_ducts),
            "n_ducts_overlap_rule": len(overlap_ducts),
            "n_salvaged": len(rescued),
            "n_lost": len(lost),
            "salvaged_area_um2_median": float(np.median(r_area)) if r_area.size else None,
            "centre_rule_area_um2_median": float(np.median(k_area)) if k_area.size else None,
            "salvaged_hole_pct_median": float(np.median(r_hole)) if r_hole.size else None,
            "centre_rule_hole_pct_median": float(np.median(k_hole)) if k_hole.size else None,
            "reference": ("The earlier holeyness analysis excluded 571/2173 ducts "
                          "(26%) under the centre rule, systematically the smallest "
                          "and least holey. Compare n_salvaged against that."),
        }

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
    n_strictly_below = int(sum(1 for d in pool if hole_by_duct[d] < threshold))
    print(f"  Hole % P{percentile:g} threshold   : {threshold:.4f}")
    print(f"  Ducts in candidate pool   : {len(pool)} "
          f"({n_strictly_below} strictly below the threshold, "
          f"{len(pool)-n_strictly_below} exactly at it)")

    # ── Degenerate pool: FAIL LOUDLY ─────────────────────────────────────────
    # If every duct in the pool sits exactly at the threshold, "lowest holeyness"
    # is not ordering anything — the tie-break is. The tie-break is UUID order,
    # which is arbitrary but at least independent of every downstream quantity;
    # it is emphatically NOT nuclear_density, because using density here would
    # quietly reinstate the circularity this whole anchor exists to remove.
    # Either way the anchor would then be "an arbitrary 20 of the N zero-hole
    # ducts", which is a materially weaker claim than "the least holey ducts" and
    # must never pass silently.
    if degenerate:
        msg = (
            f"DEGENERATE CANDIDATE POOL: all {len(pool)} ducts at or below the "
            f"P{percentile:g} threshold have hole % == {threshold:.4f} exactly. "
            "'Lowest holeyness' therefore selects nothing; the arbitrary UUID "
            "tie-break does. The resulting anchor is 'an arbitrary "
            f"{n_roots} of {len(pool)} zero-hole ducts', NOT 'the least holey "
            "ducts', and every downstream statement must say so.\n"
            "  Options: (1) raise --holeyness-percentile until the pool contains "
            "ducts with distinct hole %; (2) pass --allow-degenerate-pool to "
            "proceed deliberately with the arbitrary tie-break."
        )
        if not allow_degenerate_pool:
            raise ValueError(msg)
        print("  WARNING (proceeding, --allow-degenerate-pool was set):\n  " + msg)

    max_available = len(pool) * max_roots_per_duct
    if max_available < n_roots:
        raise ValueError(
            f"Candidate pool has {len(pool)} ducts x {max_roots_per_duct} roots/duct "
            f"= {max_available} available, but {n_roots} roots were requested. Raise "
            "--holeyness-percentile, raise --holeyness-max-roots-per-duct, or lower "
            "--n-roots. Refusing to return a short root set, which would silently "
            "reweight the median across roots."
        )

    # ── Round-robin over ducts, ascending by hole % then UUID ────────────────
    # Round-robin rather than "fill each duct before moving on" so duct diversity
    # is maximised for any n_roots: every duct contributes its 1st root before any
    # duct contributes a 2nd. With max_roots_per_duct=1 this is exactly the
    # previous one-per-duct behaviour.
    pool_sorted = sorted(pool, key=lambda d: (float(hole_by_duct[d]), str(d)))
    half = patch_size / 2.0

    # Within a duct, order patches by distance from that duct's own patch centroid.
    # Taking the most central first keeps a root off the duct boundary, where a
    # 112 px window straddles two tissues. Deterministic, and independent of every
    # downstream quantity. Deliberately NOT the QuPath centroid, which is in um.
    ranked: dict[str, list[int]] = {}
    for d in pool_sorted:
        idxs = np.asarray(eligible[d])
        px = coords[idxs, 0].astype(float) + half
        py = coords[idxs, 1].astype(float) + half
        d2 = (px - px.mean()) ** 2 + (py - py.mean()) ** 2
        ranked[d] = [int(i) for i in idxs[np.argsort(d2, kind="stable")]]

    root_indices: list[int] = []
    root_rows: list[dict] = []
    for rank_in_duct in range(max_roots_per_duct):
        for d in pool_sorted:
            if len(root_indices) >= n_roots:
                break
            if rank_in_duct >= len(ranked[d]):
                continue
            chosen = ranked[d][rank_in_duct]
            if chosen in root_indices:      # overlapping polygons can share a patch
                continue
            root_indices.append(chosen)
            root_rows.append({
                "patch_index":  chosen,
                "duct_id":      str(d),
                "slide_name":   str(slide_by_duct.get(d)),
                "hole_pct":     float(hole_by_duct[d]),
                "duct_area_um2": float(area_by_duct.get(d, np.nan)),
                "n_patches_in_duct": int(len(eligible[d])),
                "rank_within_duct": rank_in_duct,
                "x": int(coords[chosen, 0]),
                "y": int(coords[chosen, 1]),
            })
        if len(root_indices) >= n_roots:
            break

    if len(root_indices) < n_roots:
        raise ValueError(
            f"Only {len(root_indices)} distinct roots could be selected but "
            f"{n_roots} were requested — ducts in the pool have fewer patches than "
            "expected, or patches are shared between overlapping polygons."
        )
    if len(set(root_indices)) != len(root_indices):
        raise ValueError("Selected root indices contain duplicates.")

    n_distinct_ducts = len({r["duct_id"] for r in root_rows})
    print(f"  Selected {len(root_indices)} roots from {n_distinct_ducts} distinct ducts "
          f"(max {max_roots_per_duct}/duct)")
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
            "assignment": (
                f"AREA OVERLAP >= {overlap_min_fraction:.0%} of the 112px patch box, "
                "assigned to the duct claiming the largest absolute overlap area"
                if assignment == "overlap" else
                "patch centre inside Tumor polygon (holeyness.assign_patches_to_ducts, reused)"),
            "assignment_mode": assignment,
            "overlap_min_fraction": (float(overlap_min_fraction)
                                     if assignment == "overlap" else None),
            "no_duct_patches": "EXCLUDED from candidacy, never assigned hole %=0",
            "pool": f"ducts at or below the P{percentile:g} of the PER-DUCT hole % distribution",
            "min_patches_per_duct": min_patches_per_duct,
            "max_roots_per_duct": max_roots_per_duct,
            "selection": (
                f"round-robin over ducts ascending by hole %, up to {max_roots_per_duct} "
                "root(s) per duct; within a duct, patches ordered by distance from that "
                "duct's own patch centroid (most central first)"),
            "tie_break": ("object_id UUID string order — arbitrary, reproducible, and "
                          "independent of every downstream quantity. Deliberately NOT "
                          "nuclear_density: using density to break ties would reinstate "
                          "the circularity this anchor exists to remove."),
            "direction": "LOW hole % = EARLY (expert judgment: duct diameter increases with progression, holes increase with diameter)",
            "topology": ("Roots are verified to occupy ONE connected component of the "
                         "DPT k-NN graph (assert_roots_connected). Connectivity is "
                         "ASSERTED, never used to select — selecting the seed's nearest "
                         "neighbours would collapse the anchor onto a single duct."),
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
            "n_pool_ducts_strictly_below_threshold": n_strictly_below,
            "n_distinct_ducts_among_roots": n_distinct_ducts,
        },
        "overlap_vs_centre": salvage,
        "inherited_limitation": (
            "The centre-in-polygon rule excluded 571/2173 ducts (26%) in the earlier "
            "holeyness analysis, systematically the SMALLEST and LEAST holey — the "
            "same population this low-holeyness rule draws from. "
            + (f"This run uses the AREA-OVERLAP rule, which addresses that directly; "
               f"see overlap_vs_centre for how many ducts it salvaged. "
               if assignment == "overlap" else
               "This run uses the CENTRE rule, so the bias is present and merely "
               "documented, not fixed. ")
            + f"Here {n_ducts_total - n_ducts_with_patches}/{n_ducts_total} "
            f"({100*(n_ducts_total-n_ducts_with_patches)/max(n_ducts_total,1):.1f}%) "
            "ducts still have zero patches and are unavailable as candidates."
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
