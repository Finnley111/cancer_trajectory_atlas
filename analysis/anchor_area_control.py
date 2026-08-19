"""Phase 3: is the holey-ness anchor's rho(pt, duct AREA) a DUCT-SIZE artifact?

WHY THIS EXISTS
---------------
Phase 2 nominated ``rho(pseudotime, duct area)`` as the sharpest NON-CIRCULAR
discriminator between the v2 (nuclear-density) and holeyroot (hole %) anchors,
because duct area is not used in either root rule. In 2M-2 it moved from -0.084
to +0.249 — from contradicting the lab's stated biology to agreeing with it.

``holeyness_roots.json`` shows that test is not non-circular after all:

    root duct area, median         :  7,763 um^2
    ALL eligible ducts, median     : 33,807 um^2
    largest of the 20 root ducts   : 30,822 um^2

Every one of the 20 root ducts is BELOW the median eligible duct. Under any
size-blind rule that has probability 0.5^20 ~ 1e-6. The anchor is not merely
correlated with size through hole % (rho(hole, area) = +0.361) — its REALISED
selection sits in the small tail. Pseudotime is graph distance from those 20
patches, so any duct-level quantity tracking size tracks pseudotime by
construction, holey-ness or no holey-ness.

Conditioning on hole % does NOT remove this. "The roots happen to sit in small
ducts" is a different path from "hole % mediates area", and a partial correlation
given hole % leaves it intact.

WHAT THIS MODULE DOES ABOUT IT
------------------------------
It re-runs DPT from alternative root sets on the FROZEN graph and diffusion map
of an existing run — only ``sc.tl.dpt`` re-runs, exactly as
``root_sensitivity.py`` Check A does, reusing that module's
``build_dpt_adata`` / ``run_multi_root_dpt`` unmodified so the aggregation
(per-root clamp, median across roots, min-max normalisation) is identical to
production's ``diffusion.compute_dpt_multi_root``.

  TASK A  ANCHOR AREA EXTREMITY. Where the 20 root ducts sit in the eligible
          duct area distribution: per-root percentile, count below median, and
          the exact binomial p-value for that count. Descriptive; it quantifies
          the problem rather than testing anything.

  TASK B  AREA-MATCHED SURROGATE ANCHOR — the decisive control. Draw 20 ducts
          matched one-to-one to the holey-ness roots' AREAS but selected WITHOUT
          reference to hole %, and re-run DPT. If rho(pt, area) under these
          surrogate anchors covers the observed +0.249, then duct-size geometry
          reproduces the discriminator and holey-ness is not needed to explain
          it. Repeated over --n-draws seeds to give a null band, not a point.
          The surrogate roots' OWN hole % distribution is reported: with
          rho(hole, area) = +0.36 some holey-ness inevitably leaks in, and the
          reader must be able to see how much.

  TASK C  AREA-STRATIFIED HOLEY-NESS ANCHOR — the complement. Keep "lowest hole
          %" but force the root set to span the eligible AREA distribution
          (lowest-hole ducts within each area stratum). This preserves the
          anchor's meaning while removing its size extremity. If rho(pt, hole)
          survives while rho(pt, area) collapses toward zero, holey-ness is
          doing work that size is not. If BOTH collapse, the Phase 2 result was
          size.

  TASK D  UNIFORM RANDOM DUCT ANCHOR — the reference null. 20 ducts drawn
          uniformly, one patch each. Re-derives the |rho| 0.78-0.89 band this
          project already relies on, under the SAME duct-restricted candidate
          pool as Tasks B and C, so all three nulls are comparable.

  TASK E  V2 ROOT REPAIR. In 2M-2, all 20 v2 roots have duct_id null (none lies
          inside a Tumor annotation) and two of them — adjacent patches 9906 and
          9907 on 6027-4L — sit at pseudotime 0.717 / 0.673 while the other 18
          span 0.009-0.144. Two roots anchoring from the far end of the axis the
          other eighteen agree on explains the 27.7%-of-range pseudotime_std
          arithmetically: 2/20 discordant at ~0.7 gives std ~ sqrt(.1*.9)*.7 ~
          0.21 against 0.259 observed, with n_roots_clamped = 0 ruling out the
          connectivity story. This task re-runs DPT from v2's OWN stored roots,
          drops discordant ones by a rule fixed in advance — leave-one-out
          Spearman of a root's vector against the median of the other 19,
          dropped if NEGATIVE, i.e. it orders the manifold backwards relative to
          its peers — and asks whether the repaired v2 axis re-enters the
          0.78-0.89 band against holeyroot. If it does, Phase 2's sub-floor
          rho of 0.7105 is v2's defect, not the new anchor's.

  TASK F  ECCENTRICITY, EVERY ANCHOR. root_sensitivity run 4 found
          rho(PT, diffusion-map centroid distance) = 0.808 / 0.802 versus
          rho(PT, DC1) = 0.543 / 0.467 — the axis is a RADIAL coordinate, not a
          directed trajectory. Phase 2 never touched the embedding (identical
          patch sets, PCA width 261/241 unchanged), so if the holeyroot axis is
          still ~0.80 on centroid distance then every Phase 2 delta is
          re-orientation of the same radial coordinate. Computed for the
          production, holeyroot and surrogate axes alike, reusing
          ``eccentricity_check``'s own centroid and diffusion-component
          functions so the numbers are comparable to run 4's.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not declare an anchor correct. Task B answering "the surrogate covers
+0.249" would show the discriminator is uninformative, NOT that holey-ness is
wrong. Task B answering "it does not" would show size alone is insufficient, NOT
that the anchor is right — the roots would still be drawn from a duct population
that under-represents the smallest and least holey ducts (389/1749 = 22.2% of
2M-2's ducts have zero assigned patches and cannot be candidates at all).

READ-ONLY on every run tree. Writes only --output-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from .holeyness import (
    PATCH_SIZE_DEFAULT, FEATURES_TO_AGGREGATE,
    load_slide_list, load_slide_dimensions, parse_measurement_export,
    load_duct_polygons, build_duct_table, assign_patches_to_ducts,
    aggregate_per_duct, _partial_spearman,
)

from .root_sensitivity import (
    build_dpt_adata, run_multi_root_dpt, load_run, pick_diffusion_component,
)
from .eccentricity_check import _centroid_distance, _median_centroid_distance

# The obs columns aggregate_per_duct reads, minus pseudotime, which every caller
# here supplies itself. Taken from holeyness.py rather than restated, so a change
# there cannot silently leave this module aggregating a stale column set.
AGG_FEATURES = [f for f in FEATURES_TO_AGGREGATE if f != "pseudotime"]

N_ROOTS_DEFAULT = 20
N_DRAWS_DEFAULT = 25
N_STRATA_DEFAULT = 5
AREA_MATCH_TOL = 0.25      # acceptable MEDIAN relative area error of a surrogate
MATCH_POOL_DEFAULT = 10    # nearest-in-log-area donors each target draws from
RANDOM_ROOT_BAND = (0.78, 0.89)   # the project's established uniform-null range
KEY = ["slide_name", "x", "y"]


# ── small helpers ─────────────────────────────────────────────────────────────

def _safe_rho(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _pct(values: np.ndarray, q) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


# ── row-order verification between adata and results.csv ─────────────────────

def _verify_row_alignment(adata, results: pd.DataFrame, path: Path) -> dict:
    """Prove results.csv is in adata ROW ORDER before trusting it for coordinates.

    run_all.py writes both from the same arrays in the same block, so they agree
    by construction — but "by construction" is exactly the kind of assumption
    that survives a refactor and then silently attaches every patch to the wrong
    duct. Root indices from ``adata.uns['dpt_root_candidates']`` index INTO this
    frame, so a shifted row order would produce a completely different anchor and
    entirely plausible-looking numbers.

    The check uses every column the two files share. ``pseudotime`` and the
    morphological features are continuous and effectively unique per patch, so
    an order mismatch cannot survive them; ``slide_id`` is categorical and
    partitions the rows into 8 blocks, which catches a block-level permutation
    that continuous columns might tolerate at low precision.

    Raises rather than warning. There is no safe degraded mode here.
    """
    if len(results) != adata.n_obs:
        raise ValueError(
            f"{path} has {len(results)} rows but the h5ad has {adata.n_obs}. These "
            "are from different runs; root indices from the h5ad do not address "
            "this frame."
        )

    obs = adata.obs
    shared = [c for c in results.columns if c in obs.columns]
    checked: dict[str, str] = {}
    mismatches: list[str] = []

    for c in shared:
        a = obs[c].values
        b = results[c].values
        try:
            af = np.asarray(a, dtype=float)
            bf = np.asarray(b, dtype=float)
        except (TypeError, ValueError):
            # categorical / string columns (slide_id is stored as str in obs and
            # as int in the csv, so compare their string forms)
            same = (pd.Series(a).astype(str).values
                    == pd.Series(b).astype(str).values)
            n_bad = int((~same).sum())
            checked[c] = f"categorical, {len(same) - n_bad}/{len(same)} equal"
            if n_bad:
                mismatches.append(f"{c}: {n_bad} rows differ")
            continue

        nan_a, nan_b = np.isnan(af), np.isnan(bf)
        if not np.array_equal(nan_a, nan_b):
            mismatches.append(f"{c}: nan masks differ")
            checked[c] = "nan mask mismatch"
            continue
        ok = ~nan_a
        if ok.sum() == 0:
            checked[c] = "all nan, uninformative"
            continue
        if np.unique(af[ok]).size < 2:
            # A constant column agrees under every permutation, so it proves
            # nothing about row order. Counted as checked but NOT as evidence.
            checked[c] = "constant, uninformative"
            continue
        close = np.allclose(af[ok], bf[ok], rtol=1e-5, atol=1e-8)
        checked[c] = ("numeric, identical" if close else "numeric, DIFFERS")
        if not close:
            n_bad = int((~np.isclose(af[ok], bf[ok], rtol=1e-5, atol=1e-8)).sum())
            mismatches.append(f"{c}: {n_bad}/{int(ok.sum())} values differ")

    informative = [c for c, v in checked.items() if "uninformative" not in v]
    if not informative:
        raise ValueError(
            f"{path} and the h5ad share no column that could confirm row order "
            f"(shared columns: {shared}). Refusing to assume alignment — the "
            "coordinates would be attached to arbitrary patches."
        )
    if mismatches:
        raise ValueError(
            f"{path} is NOT in adata row order. Disagreements: "
            + "; ".join(mismatches)
            + ". run_all.py writes results.csv and adata_full.h5ad from the same "
              "arrays, so this means the two files are from different runs. Every "
              "root index would address the wrong patch."
        )

    print(f"  row alignment verified against {path.name}: "
          f"{len(informative)} shared column(s) agree "
          f"({', '.join(informative[:6])}{'...' if len(informative) > 6 else ''})")
    return {"n_rows": int(len(results)), "columns_checked": checked,
            "n_informative_columns": len(informative)}


# ── duct context: everything the root rules and the duct-level rhos need ──────

class DuctContext:
    """Patch->duct assignment and per-duct metadata, in adata ROW ORDER.

    Coordinates come from the run's results.csv, not from adata.obs, which never
    carries them; see the note in __init__ for why, and _verify_row_alignment for
    the check that makes indexing one with the other's root indices safe.

    Built with the SAME loaders and the SAME centre-in-polygon assignment that
    holeyness.py, holeyness_roots.py and holeyroot_compare.py use, so every
    number this module produces is directly comparable to Phase 1 and Phase 2.
    The inherited exclusion bias (ducts whose patches capture no patch CENTRE
    get no patches, and those are systematically the smallest and least holey)
    is therefore present here too — deliberately, because changing it would make
    the comparison to Phase 2 meaningless. It is reported, not fixed.
    """

    def __init__(self, adata, run_dir: Path, export: Path, ann_dir: Path,
                 dims: Path, slide_list: Path, patch_size: int):
        # WHERE THE COORDINATES LIVE, AND WHY NOT IN adata.obs
        # ----------------------------------------------------
        # run_all.py builds the AnnData from X_pca and adds only cluster,
        # slide_id, mouse_id, section_number, pseudotime, pseudotime_std and the
        # morphological features (run_all.py:664). The patch COORDINATES are
        # never written to obs — they go to results.csv, built in the same block
        # from the same all_coords / slide_ids arrays (run_all.py:718), so the
        # two files are in identical row order by construction.
        #
        # That invariant is what diagnostics/inspect_roots_v3.py already relies
        # on when it indexes results.csv positionally with root indices read from
        # the h5ad. This module relies on it too, and VERIFIES it below rather
        # than assuming it: a silent row-order mismatch would attach every patch
        # to the wrong duct while still producing entirely plausible numbers.
        results_csv = Path(run_dir) / "results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(
                f"{results_csv} not found. The patch coordinates are not in "
                "adata.obs — run_all.py writes them only to results.csv — so the "
                "patch->duct assignment cannot be built from the h5ad alone."
            )
        results = pd.read_csv(results_csv)
        self.row_alignment = _verify_row_alignment(adata, results, results_csv)

        slides = load_slide_list(slide_list)
        meas = parse_measurement_export(export, slides)
        polys = load_duct_polygons(ann_dir, slides, load_slide_dimensions(dims))
        self.duct_table = build_duct_table(meas, polys)
        if len(self.duct_table) == 0:
            raise ValueError(
                "Empty duct table — no measurement row joined to a Tumor polygon "
                "by UUID. Check that the export matches the slides being run and "
                "that the annotation dir is the RATIO directory."
            )

        missing = [c for c in KEY if c not in results.columns]
        if missing:
            raise KeyError(
                f"{results_csv} is missing {missing} — assign_patches_to_ducts "
                "needs all three and there is no other source for them.")
        self.results_df = pd.DataFrame({
            "x": results["x"].values.astype(float),
            "y": results["y"].values.astype(float),
            "slide_name": results["slide_name"].astype(str).values,
        })
        # aggregate_per_duct's other inputs come from the SAME csv, not from
        # adata.obs, so these duct-level numbers are produced by exactly the
        # inputs holeyroot_compare.duct_level used in Phase 2.
        self._feature_cache = {}
        for feat in AGG_FEATURES:
            if feat not in results.columns:
                raise KeyError(
                    f"{results_csv} has no '{feat}' column — "
                    "holeyness.aggregate_per_duct requires it and this module "
                    "reuses that function rather than reimplementing it.")
            self._feature_cache[feat] = results[feat].values.astype(float)
        self.coords = self.results_df[["x", "y"]].values
        self.patch_size = int(patch_size)

        assigned = assign_patches_to_ducts(
            self.results_df, self.duct_table, patch_size=self.patch_size)
        self.duct_id = assigned["duct_id"].values

        self.hole_by_duct = dict(zip(self.duct_table["object_id"],
                                     self.duct_table["hole_pct"]))
        self.area_by_duct = dict(zip(self.duct_table["object_id"],
                                     self.duct_table["area_um2"]))
        self.slide_by_duct = dict(zip(self.duct_table["object_id"],
                                      self.duct_table["slide_name"]))

        patches_by_duct: dict[str, list[int]] = {}
        for i, d in enumerate(self.duct_id):
            if d is not None:
                patches_by_duct.setdefault(d, []).append(int(i))
        # Eligible == the candidate population every root rule here draws from:
        # >=1 assigned patch AND a finite hole %. Identical to holeyness_roots.
        self.eligible = {
            d: idxs for d, idxs in patches_by_duct.items()
            if np.isfinite(self.hole_by_duct.get(d, np.nan))
            and np.isfinite(self.area_by_duct.get(d, np.nan))
        }
        if len(self.eligible) < N_ROOTS_DEFAULT:
            raise ValueError(
                f"Only {len(self.eligible)} eligible ducts — fewer than the "
                f"{N_ROOTS_DEFAULT} roots every task needs. The annotations do not "
                "match these slides."
            )

        self.n_ducts_total = int(len(self.duct_table))
        self.n_ducts_zero_patches = self.n_ducts_total - len(patches_by_duct)
        self.eligible_ids = list(self.eligible)
        self.eligible_areas = np.array(
            [self.area_by_duct[d] for d in self.eligible_ids], dtype=float)
        self.eligible_holes = np.array(
            [self.hole_by_duct[d] for d in self.eligible_ids], dtype=float)

        # Per-duct patch ranking by distance from that duct's OWN patch centroid,
        # most central first — byte-identical to holeyness_roots.select_holeyness_roots
        # so a root duct maps to the same root PATCH under every rule below.
        half = self.patch_size / 2.0
        self.ranked: dict[str, list[int]] = {}
        for d, idxs in self.eligible.items():
            a = np.asarray(idxs)
            px = self.coords[a, 0].astype(float) + half
            py = self.coords[a, 1].astype(float) + half
            d2 = (px - px.mean()) ** 2 + (py - py.mean()) ** 2
            self.ranked[d] = [int(i) for i in a[np.argsort(d2, kind="stable")]]

    # ── root construction ────────────────────────────────────────────────────
    def roots_from_ducts(self, ducts) -> list[int]:
        """One root patch per duct: the most central, same rule for every anchor."""
        out: list[int] = []
        for d in ducts:
            for cand in self.ranked[d]:
                if cand not in out:
                    out.append(cand)
                    break
        if len(out) != len(ducts):
            raise ValueError(
                f"{len(ducts)} ducts yielded only {len(out)} distinct root patches — "
                "overlapping polygons share patches. Refusing a short root set, which "
                "would silently reweight the median across roots."
            )
        return out

    # ── duct-level correlations ──────────────────────────────────────────────
    def per_duct(self, pseudotime: np.ndarray) -> pd.DataFrame:
        """Median-aggregate a pseudotime vector to ducts via holeyness.py's own
        aggregator, so these rhos and Phase 1/2's come from identical code."""
        df = self.results_df.copy()
        df["duct_id"] = self.duct_id
        df["pseudotime"] = np.asarray(pseudotime, dtype=float)
        for feat in AGG_FEATURES:
            df[feat] = self._feature_cache[feat]
        return aggregate_per_duct(df, self.duct_table, np.nanmedian, "median")

def duct_rhos(ctx: DuctContext, pseudotime: np.ndarray) -> dict:
    """The Phase 2 duct-level block, recomputed against an arbitrary axis."""
    pd_ = ctx.per_duct(pseudotime)
    pt = pd_["pseudotime"].values
    hole = pd_["hole_pct"].values
    area = pd_["area_um2"].values
    nd = pd_["nuclear_density"].values
    return {
        "n_ducts": int(len(pd_)),
        "rho_pt_hole_pct": _safe_rho(pt, hole),
        "rho_pt_area": _safe_rho(pt, area),
        "rho_pt_nuclear_density": _safe_rho(pt, nd),
        "rho_hole_area": _safe_rho(hole, area),
        "partial_pt_hole_given_area": _partial_spearman(pt, hole, area),
        "partial_pt_area_given_hole": _partial_spearman(pt, area, hole),
    }


# ── TASK A: where the anchor sits in the area distribution ───────────────────

def task_a_area_extremity(ctx: DuctContext, root_ducts: list[str]) -> dict:
    """Descriptive. Quantifies the selection bias; tests nothing."""
    areas = ctx.eligible_areas
    med = float(np.median(areas))
    root_areas = np.array([ctx.area_by_duct[d] for d in root_ducts], dtype=float)
    pctiles = [float((areas < a).mean() * 100.0) for a in root_areas]
    n_below = int((root_areas < med).sum())
    bt = binomtest(n_below, len(root_areas), 0.5, alternative="two-sided")

    print("\n=== TASK A — anchor area extremity ===")
    print(f"  eligible ducts               : {len(areas)}")
    print(f"  eligible duct area, median   : {med:.0f} um^2")
    print(f"  root duct area, median       : {np.median(root_areas):.0f} um^2")
    print(f"  root ducts below that median : {n_below}/{len(root_areas)}  "
          f"(exact binomial p = {bt.pvalue:.3g})")
    print(f"  root duct area percentiles   : median {np.median(pctiles):.1f}, "
          f"max {max(pctiles):.1f}")

    return {
        "n_eligible_ducts": int(len(areas)),
        "eligible_area_um2": {
            "median": med, "p25": _pct(areas, 25), "p75": _pct(areas, 75),
            "min": float(areas.min()), "max": float(areas.max()),
        },
        "root_area_um2": {
            "median": float(np.median(root_areas)),
            "min": float(root_areas.min()), "max": float(root_areas.max()),
        },
        "root_area_percentile_within_eligible": pctiles,
        "n_root_ducts_below_eligible_median": n_below,
        "n_root_ducts": int(len(root_areas)),
        "exact_binomial_p_vs_half": float(bt.pvalue),
        "size_ratio_eligible_median_over_root_median":
            float(med / np.median(root_areas)) if np.median(root_areas) > 0 else None,
        "interpretation": (
            "Descriptive only. A count far from n/2 means the realised anchor is "
            "size-extreme, so rho(pt, duct area) is NOT the non-circular test "
            "Phase 2 treated it as — Task B measures how much of it size explains."
        ),
    }


# ── TASK B/C/D: alternative root rules ───────────────────────────────────────

def select_area_matched_surrogate(ctx: DuctContext, target_ducts: list[str],
                                  rng: np.random.Generator,
                                  tol: float = AREA_MATCH_TOL,
                                  match_pool: int = MATCH_POOL_DEFAULT
                                  ) -> tuple[list[str], dict]:
    """20 ducts matched one-to-one to the target AREAS, chosen without hole %.

    MATCHING BY NEAREST RANK, NOT BY A FIXED WINDOW
    -----------------------------------------------
    An earlier version drew each surrogate from a +/-tol window around its
    target and refused if any window was empty. That refuses on a technicality
    exactly where it matters most: the anchor's smallest root duct can BE close
    to the smallest eligible duct, leaving no donor below it, and the whole
    control then fails to build even though 19 of 20 slots matched perfectly.

    Instead each target draws from its ``match_pool`` nearest unused donors in
    log-area, and the REALISED match quality is measured and returned. Quality
    is judged on the finished set — median and max relative area error — not on
    whether one boundary slot happened to have a neighbour. A surrogate whose
    median error exceeds ``tol`` is refused, because it is then not area-matched
    in any useful sense; that check looks at the outcome rather than the
    procedure.

    Targets are matched in random order and sampled without replacement, so the
    draw is a genuine random variable rather than a deterministic nearest-
    neighbour map. The real root ducts are excluded from the donor pool: leaving
    them in would let the surrogate literally re-select the anchor being tested.
    """
    exclude = set(target_ducts)
    donors = [d for d in ctx.eligible_ids if d not in exclude]
    if len(donors) < len(target_ducts):
        raise ValueError(
            f"Only {len(donors)} donor ducts outside the anchor, fewer than the "
            f"{len(target_ducts)} the surrogate needs.")
    donor_log = np.log(np.array([ctx.area_by_duct[d] for d in donors], dtype=float))

    used = np.zeros(len(donors), dtype=bool)
    chosen: list[str] = []
    errors: list[float] = []
    for t in rng.permutation(len(target_ducts)):
        target = float(ctx.area_by_duct[target_ducts[int(t)]])
        d = np.abs(donor_log - np.log(target))
        d[used] = np.inf
        k = min(match_pool, int((~used).sum()))
        near = np.argpartition(d, k - 1)[:k] if k > 1 else np.array([int(d.argmin())])
        pick = int(rng.choice(near))
        used[pick] = True
        chosen.append(donors[pick])
        errors.append(abs(float(np.exp(donor_log[pick])) - target) / target)

    err = np.array(errors, dtype=float)
    quality = {
        "median_relative_area_error": float(np.median(err)),
        "max_relative_area_error": float(err.max()),
        "n_slots_outside_tol": int((err > tol).sum()),
        "tol": float(tol),
        "match_pool": int(match_pool),
    }
    if quality["median_relative_area_error"] > tol:
        raise ValueError(
            f"Surrogate is not area-matched: median relative area error "
            f"{quality['median_relative_area_error']:.1%} exceeds the "
            f"{tol:.0%} tolerance. The eligible pool cannot supply size-matched "
            "donors for this anchor, which is itself a finding — report it rather "
            "than accepting an unmatched control."
        )
    return chosen, quality


def select_area_stratified_holeyness(ctx: DuctContext, n_roots: int,
                                     n_strata: int) -> list[str]:
    """Lowest hole % WITHIN each area stratum — holey-ness without size extremity.

    Strata are equal-count quantile bins of the eligible duct area distribution,
    so the resulting root set spans the full size range by construction while
    every member is still the least holey duct available at its size.
    """
    if n_roots % n_strata != 0:
        raise ValueError(
            f"n_roots ({n_roots}) must divide evenly into n_strata ({n_strata}) so "
            "every stratum contributes equally; an uneven split would reintroduce a "
            "size weighting through the back door."
        )
    per = n_roots // n_strata
    edges = np.quantile(ctx.eligible_areas, np.linspace(0, 1, n_strata + 1))
    edges[-1] = np.inf

    chosen: list[str] = []
    for s in range(n_strata):
        lo, hi = edges[s], edges[s + 1]
        members = [d for d in ctx.eligible_ids
                   if lo <= ctx.area_by_duct[d] < hi]
        if len(members) < per:
            raise ValueError(
                f"Area stratum {s} has {len(members)} ducts, fewer than the {per} "
                "roots it must supply. Lower --n-strata."
            )
        # ascending hole %, UUID tie-break — the same arbitrary-but-downstream-
        # independent tie-break holeyness_roots.py uses.
        members.sort(key=lambda d: (float(ctx.hole_by_duct[d]), str(d)))
        chosen.extend(members[:per])
    return chosen


def select_random_ducts(ctx: DuctContext, n_roots: int,
                        rng: np.random.Generator) -> list[str]:
    idx = rng.choice(len(ctx.eligible_ids), size=n_roots, replace=False)
    return [ctx.eligible_ids[int(i)] for i in idx]


def describe_anchor(ctx: DuctContext, ducts: list[str]) -> dict:
    areas = np.array([ctx.area_by_duct[d] for d in ducts], dtype=float)
    holes = np.array([ctx.hole_by_duct[d] for d in ducts], dtype=float)
    med_area = float(np.median(ctx.eligible_areas))
    # Where each root duct sits in the eligible area distribution. Task C's
    # stratification equalises this ACROSS strata but cannot remove the skew
    # WITHIN one: hole % rises with area, so the lowest-hole ducts of a stratum
    # sit near that stratum's small edge. Reported so the residual is visible
    # rather than assumed away.
    pctiles = [float((ctx.eligible_areas < a).mean() * 100.0) for a in areas]
    return {
        "n_ducts": len(ducts),
        "n_slides": len({ctx.slide_by_duct.get(d) for d in ducts}),
        "duct_area_um2": {"median": float(np.median(areas)),
                          "min": float(areas.min()), "max": float(areas.max())},
        "hole_pct": {"median": float(np.median(holes)),
                     "min": float(holes.min()), "max": float(holes.max())},
        "n_below_eligible_area_median": int((areas < med_area).sum()),
        "area_percentile_within_eligible": {
            "median": float(np.median(pctiles)),
            "min": float(min(pctiles)), "max": float(max(pctiles)),
        },
    }


def run_anchor(label: str, ctx: DuctContext, ducts: list[str], base,
               dm_sub: np.ndarray, reference: dict,
               match_quality: dict | None = None) -> tuple[dict, np.ndarray]:
    """Re-run DPT from one root set on the frozen graph, then measure it."""
    roots = ctx.roots_from_ducts(ducts)
    pt, pt_std, notes = run_multi_root_dpt(None, roots, base=base)

    rng_raw = notes["median_pt_range_before_normalisation"]
    raw_range = float(rng_raw[1] - rng_raw[0])
    out = {
        "label": label,
        "anchor": describe_anchor(ctx, ducts),
        "root_patch_indices": roots,
        "dpt_notes": notes,
        "duct_level": duct_rhos(ctx, pt),
        "pseudotime_std": {
            "median_raw": float(np.median(pt_std)),
            "median_pct_of_raw_range":
                float(100 * np.median(pt_std) / raw_range) if raw_range > 0 else None,
        },
        "eccentricity": {
            "rho_pt_diffmap_centroid_distance": _safe_rho(pt, _centroid_distance(dm_sub)),
            "rho_pt_diffmap_median_centroid_distance":
                _safe_rho(pt, _median_centroid_distance(dm_sub)),
            "note": ("DPT pseudotime IS a diffusion distance, so this is PARTLY "
                     "definitional. Reported for continuity with root_sensitivity "
                     "run 4 (0.808 / 0.802), where it was the basis for calling the "
                     "axis radial rather than directed."),
        },
        "rho_vs_reference": {
            k: _safe_rho(pt, v) for k, v in reference.items()
        },
        "area_match_quality": match_quality,
    }
    return out, pt


# ── TASK E: v2 root repair ───────────────────────────────────────────────────

def _loo_concordance(pt_matrix: np.ndarray) -> np.ndarray:
    """Spearman of each root's own DPT vector against the median of the others.

    Separated out so the drop rule can be tested against a constructed matrix
    (n concordant roots plus k deliberately reversed ones) without needing
    scanpy, a graph, or a run tree.
    """
    n = pt_matrix.shape[0]
    return np.array([
        _safe_rho(pt_matrix[i], np.median(np.delete(pt_matrix, i, axis=0), axis=0))
        for i in range(n)
    ], dtype=float)


def task_e_v2_root_repair(adata_v2, holeyroot_pt_aligned: np.ndarray,
                          dm_sub_v2: np.ndarray) -> dict:
    """Re-run DPT from v2's own stored roots, drop discordant ones, re-measure.

    THE DROP RULE, FIXED BEFORE LOOKING AT THE RESULT
    -------------------------------------------------
    For each root i, take the Spearman correlation of that root's own DPT vector
    against the median of the OTHER 19. Drop root i if that correlation is
    NEGATIVE — i.e. it orders the manifold backwards relative to every peer.
    This is a statement about internal consistency of the root set alone. It
    never consults hole %, duct area, any morphological feature, or the
    holeyroot axis, so it cannot be tuned toward a desired answer.
    """
    if "dpt_root_candidates" not in adata_v2.uns:
        raise KeyError(
            "adata.uns['dpt_root_candidates'] missing from the v2 run — the roots "
            "it actually used cannot be recovered, and re-deriving them from "
            "nuclear_density would apply a rule that may not be the one the run "
            "used (see diffusion.compute_dpt_multi_root's docstring)."
        )
    import scanpy as sc

    roots = [int(i) for i in np.asarray(adata_v2.uns["dpt_root_candidates"]).ravel()]
    base = build_dpt_adata(adata_v2)

    pt_matrix = np.zeros((len(roots), base.n_obs), dtype=np.float64)
    for i, r in enumerate(roots):
        tmp = base.copy()
        tmp.uns["iroot"] = int(r)
        sc.tl.dpt(tmp)
        v = tmp.obs["dpt_pseudotime"].values.copy()
        finite = np.isfinite(v)
        if not finite.all():
            v[~finite] = v[finite].max() if finite.any() else 0.0
        pt_matrix[i] = v

    loo = _loo_concordance(pt_matrix)
    keep = np.flatnonzero(loo >= 0)
    dropped = np.flatnonzero(loo < 0)

    def _agg(mat):
        med = np.median(mat, axis=0)
        lo, hi = float(med.min()), float(med.max())
        return (med - lo) / (hi - lo) if hi - lo > 1e-10 else np.zeros_like(med), (lo, hi)

    pt_all, rng_all = _agg(pt_matrix)
    result = {
        "n_roots": len(roots),
        "root_patch_indices": roots,
        "leave_one_out_rho_vs_median_of_others": loo.tolist(),
        "n_dropped": int(dropped.size),
        "dropped_root_positions": dropped.tolist(),
        "dropped_root_patch_indices": [roots[int(i)] for i in dropped],
        "drop_rule": ("leave-one-out Spearman of a root's own DPT vector against the "
                      "median of the other 19; dropped if NEGATIVE. Fixed in advance; "
                      "consults nothing outside the root set."),
        "reproduced_stored_axis_rho": _safe_rho(
            pt_all, adata_v2.obs["pseudotime"].values.astype(float)),
        "all_roots": {
            "pseudotime_std_median_pct_of_range":
                float(100 * np.median(np.std(pt_matrix, axis=0)) / (rng_all[1] - rng_all[0]))
                if rng_all[1] > rng_all[0] else None,
            "rho_vs_holeyroot": _safe_rho(pt_all, holeyroot_pt_aligned),
            "rho_pt_diffmap_centroid_distance": _safe_rho(pt_all, _centroid_distance(dm_sub_v2)),
        },
    }

    if dropped.size == 0:
        result["repaired"] = None
        result["verdict"] = (
            "No root ordered the manifold backwards relative to its peers, so the "
            "rule dropped nothing and there is no repaired axis to compare. The "
            "27.7%-of-range pseudotime_std is then NOT explained by sign-discordant "
            "roots and needs a different explanation."
        )
        return result

    pt_rep, rng_rep = _agg(pt_matrix[keep])
    rho_rep = _safe_rho(pt_rep, holeyroot_pt_aligned)
    lo_band, hi_band = RANDOM_ROOT_BAND
    result["repaired"] = {
        "n_roots_kept": int(keep.size),
        "pseudotime_std_median_pct_of_range":
            float(100 * np.median(np.std(pt_matrix[keep], axis=0)) / (rng_rep[1] - rng_rep[0]))
            if rng_rep[1] > rng_rep[0] else None,
        "rho_vs_holeyroot": rho_rep,
        "rho_vs_stored_v2_axis": _safe_rho(pt_rep, adata_v2.obs["pseudotime"].values.astype(float)),
        "rho_pt_diffmap_centroid_distance": _safe_rho(pt_rep, _centroid_distance(dm_sub_v2)),
        "reenters_random_root_band": bool(lo_band <= abs(rho_rep) <= hi_band),
        "band": list(RANDOM_ROOT_BAND),
    }
    result["verdict"] = (
        f"Dropping {dropped.size} sign-discordant root(s) moves rho(v2, holeyroot) "
        f"from {result['all_roots']['rho_vs_holeyroot']:.4f} to {rho_rep:.4f}. If that "
        f"lands inside {RANDOM_ROOT_BAND}, Phase 2's sub-floor 0.7105 was v2's defect "
        "and not evidence about the holey-ness anchor. It does NOT make either axis "
        "correct — a repaired v2 is still rooted on 20 patches with zero measured "
        "nuclei, none of which lies inside a Tumor annotation."
    )
    return result


# ── alignment between two run trees ──────────────────────────────────────────

def load_aligned_results(adata, run_dir: Path) -> pd.DataFrame:
    """results.csv for a run, verified to be in that run's adata row order."""
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Patch coordinates live only in results.csv "
            "(run_all.py:718), never in adata.obs, so cross-run alignment cannot "
            "be established from the h5ad alone.")
    results = pd.read_csv(path)
    _verify_row_alignment(adata, results, path)
    return results


def align_to(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Row indices mapping source order -> target order, keyed on (slide, x, y).

    Both frames are results.csv tables that ``_verify_row_alignment`` has already
    tied to their own run's adata row order, so a permutation computed here is
    valid for the h5ad arrays too.

    Phase 2 verified the two runs hold identical patch SETS, but nothing
    guarantees identical row ORDER, and a silent misalignment would corrupt every
    cross-run correlation while still producing plausible numbers.
    """
    def key(o):
        return list(zip(o["slide_name"].astype(str).values,
                        o["x"].values.astype(np.int64),
                        o["y"].values.astype(np.int64)))
    src, tgt = key(source), key(target)
    if set(src) != set(tgt):
        raise ValueError(
            f"Patch sets differ between runs: {len(set(src) - set(tgt))} only in "
            f"source, {len(set(tgt) - set(src))} only in target. These axes are not "
            "comparable patch-by-patch."
        )
    pos = {k: i for i, k in enumerate(tgt)}
    return np.array([pos[k] for k in src], dtype=int)


# ── driver ───────────────────────────────────────────────────────────────────

def run_section(section: str, hr_dir: Path, v2_dir: Path, export: Path,
                ann_dir: Path, dims: Path, slide_list: Path, patch_size: int,
                n_roots: int, n_draws: int, n_strata: int, seed: int,
                area_tol: float, match_pool: int) -> dict:
    print("\n" + "=" * 78)
    print(f"  SECTION {section}")
    print("=" * 78)

    adata_hr, _, _ = load_run(hr_dir)
    ctx = DuctContext(adata_hr, hr_dir, export, ann_dir, dims, slide_list,
                      patch_size)

    dc_info = pick_diffusion_component(adata_hr)
    dm = np.asarray(adata_hr.obsm["X_diffmap"], dtype=float)
    trivial = set(dc_info["trivial_columns_detected"])
    dm_sub = dm[:, [i for i in range(dm.shape[1]) if i not in trivial]]

    hr_pt = adata_hr.obs["pseudotime"].values.astype(float)
    reference = {"holeyroot_production_pt": hr_pt}

    # The holeyroot run's OWN roots, recovered from what it actually used.
    if "dpt_root_candidates" not in adata_hr.uns:
        raise KeyError(f"{hr_dir}: adata.uns['dpt_root_candidates'] missing.")
    hr_roots = [int(i) for i in np.asarray(adata_hr.uns["dpt_root_candidates"]).ravel()]
    hr_root_ducts = []
    for r in hr_roots:
        d = ctx.duct_id[r]
        if d is None:
            raise ValueError(
                f"holeyroot root patch {r} is assigned to NO duct under this "
                "rebuild of the assignment. The duct context does not reproduce "
                "the run's own root selection; refusing to build controls against "
                "an assignment that disagrees with the anchor."
            )
        hr_root_ducts.append(d)

    # Consistency: does re-running DPT from the stored roots reproduce the stored axis?
    base = build_dpt_adata(adata_hr)
    check_pt, _, _ = run_multi_root_dpt(None, hr_roots, base=base)
    repro = _safe_rho(check_pt, hr_pt)
    print(f"\n  consistency: re-run of the stored holeyroot roots vs the stored "
          f"axis, rho = {repro:.6f}")
    if repro < 0.999:
        print("  WARNING: the frozen-graph re-run does NOT reproduce the stored axis. "
              "Every control below is then measured against a different DPT than "
              "production used, and the comparison is unsafe.")

    result = {
        "section": section,
        "consistency_rerun_vs_stored_rho": repro,
        "consistency_ok": bool(repro >= 0.999),
        "duct_context": {
            "n_ducts_in_table": ctx.n_ducts_total,
            "n_ducts_with_zero_patches": ctx.n_ducts_zero_patches,
            "n_eligible_ducts": len(ctx.eligible_ids),
        },
        "task_a_area_extremity": task_a_area_extremity(ctx, hr_root_ducts),
        "observed_holeyroot": duct_rhos(ctx, hr_pt),
    }

    rng = np.random.default_rng(seed)

    print("\n=== TASK B — area-matched surrogate anchors ===")
    task_b = []
    for k in range(n_draws):
        ducts, quality = select_area_matched_surrogate(
            ctx, hr_root_ducts, rng, area_tol, match_pool)
        rec, _ = run_anchor(f"surrogate_{k}", ctx, ducts, base, dm_sub, reference,
                            match_quality=quality)
        task_b.append(rec)
        print(f"  draw {k:2d}: rho(pt,area)={rec['duct_level']['rho_pt_area']:+.4f}  "
              f"rho(pt,hole)={rec['duct_level']['rho_pt_hole_pct']:+.4f}  "
              f"root hole% median={rec['anchor']['hole_pct']['median']:.3f}  "
              f"area match err med={quality['median_relative_area_error']:.1%}")

    print("\n=== TASK C — area-stratified holey-ness anchor ===")
    ducts_c = select_area_stratified_holeyness(ctx, n_roots, n_strata)
    task_c, _ = run_anchor("area_stratified_holeyness", ctx, ducts_c, base,
                           dm_sub, reference)
    print(f"  rho(pt,area)={task_c['duct_level']['rho_pt_area']:+.4f}  "
          f"rho(pt,hole)={task_c['duct_level']['rho_pt_hole_pct']:+.4f}")

    print("\n=== TASK D — uniform random duct anchors ===")
    task_d = []
    for k in range(n_draws):
        ducts = select_random_ducts(ctx, n_roots, rng)
        rec, _ = run_anchor(f"random_{k}", ctx, ducts, base, dm_sub, reference)
        task_d.append(rec)
        print(f"  draw {k:2d}: rho(pt,area)={rec['duct_level']['rho_pt_area']:+.4f}  "
              f"rho vs holeyroot={rec['rho_vs_reference']['holeyroot_production_pt']:+.4f}")

    result["task_b_area_matched_surrogate"] = task_b
    result["task_c_area_stratified_holeyness"] = task_c
    result["task_d_uniform_random"] = task_d
    result["task_b_summary"] = _null_summary(
        task_b, result["observed_holeyroot"])
    result["task_d_summary"] = _null_summary(
        task_d, result["observed_holeyroot"])

    # TASK E
    print("\n=== TASK E — v2 root repair ===")
    adata_v2, _, _ = load_run(v2_dir)
    results_v2 = load_aligned_results(adata_v2, v2_dir)
    idx = align_to(results_v2, ctx.results_df)
    hr_pt_in_v2_order = hr_pt[idx]
    dc_v2 = pick_diffusion_component(adata_v2)
    dm_v2 = np.asarray(adata_v2.obsm["X_diffmap"], dtype=float)
    triv_v2 = set(dc_v2["trivial_columns_detected"])
    dm_sub_v2 = dm_v2[:, [i for i in range(dm_v2.shape[1]) if i not in triv_v2]]
    result["task_e_v2_root_repair"] = task_e_v2_root_repair(
        adata_v2, hr_pt_in_v2_order, dm_sub_v2)

    return result


def _null_summary(draws: list[dict], observed: dict) -> dict:
    """Where the observed holeyroot value sits in a null of alternative anchors."""
    out = {}
    for q in ("rho_pt_area", "rho_pt_hole_pct", "rho_pt_nuclear_density"):
        vals = np.array([d["duct_level"][q] for d in draws], dtype=float)
        vals = vals[np.isfinite(vals)]
        obs = float(observed[q])
        if vals.size == 0:
            out[q] = None
            continue
        # Two-sided: how often does an anchor that knows nothing about hole %
        # reach a value at least as extreme as the one the anchor produced?
        p = float((np.abs(vals) >= abs(obs)).mean())
        out[q] = {
            "observed_holeyroot": obs,
            "null_median": float(np.median(vals)),
            "null_p05": _pct(vals, 5), "null_p95": _pct(vals, 95),
            "null_min": float(vals.min()), "null_max": float(vals.max()),
            "observed_inside_null_range": bool(vals.min() <= obs <= vals.max()),
            "frac_null_at_least_as_extreme": p,
            "n_draws": int(vals.size),
        }
    rho_ref = np.array(
        [d["rho_vs_reference"]["holeyroot_production_pt"] for d in draws], dtype=float)
    out["rho_vs_holeyroot_production"] = {
        "median": float(np.median(rho_ref)),
        "min": float(rho_ref.min()), "max": float(rho_ref.max()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=["2M-1", "2M-2"])
    ap.add_argument("--holeyroot-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--v2-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--exports", nargs="+", type=Path, required=True,
                    help="One per section, SAME ORDER as --sections. 2M-1 and 2M-2 "
                         "read different export files.")
    ap.add_argument("--slide-lists", nargs="+", type=Path, required=True)
    ap.add_argument("--annotation-dir", type=Path, required=True)
    ap.add_argument("--slide-dimensions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    ap.add_argument("--n-roots", type=int, default=N_ROOTS_DEFAULT)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--n-strata", type=int, default=N_STRATA_DEFAULT)
    ap.add_argument("--area-match-tol", type=float, default=AREA_MATCH_TOL,
                    help="Acceptable MEDIAN relative area error of a surrogate set.")
    ap.add_argument("--match-pool", type=int, default=MATCH_POOL_DEFAULT,
                    help="Nearest-in-log-area donors each target samples from.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = len(args.sections)
    for name, seq in (("--holeyroot-dirs", args.holeyroot_dirs),
                      ("--v2-dirs", args.v2_dirs),
                      ("--exports", args.exports),
                      ("--slide-lists", args.slide_lists)):
        if len(seq) != n:
            raise SystemExit(
                f"{name} has {len(seq)} entries but --sections has {n}. These are "
                "positional: 2M-1 and 2M-2 read different export files and a "
                "mismatch would silently analyse one section with the other's "
                "annotations.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"config": {k: str(v) for k, v in vars(args).items()}, "sections": []}
    for i, sec in enumerate(args.sections):
        results["sections"].append(run_section(
            sec, args.holeyroot_dirs[i], args.v2_dirs[i], args.exports[i],
            args.annotation_dir, args.slide_dimensions, args.slide_lists[i],
            args.patch_size, args.n_roots, args.n_draws, args.n_strata, args.seed,
            float(args.area_match_tol), int(args.match_pool)))

    out = args.output_dir / "anchor_area_control.json"
    out.write_text(json.dumps(results, indent=2, default=_json_default),
                   encoding="utf-8")
    write_report(results, args.output_dir / "anchor_area_control.md")
    print(f"\nWrote {out}")


def write_report(results: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    add("# Phase 3 — is rho(pseudotime, duct area) a duct-size artifact?\n")
    add("Phase 2 treated `rho(pt, duct area)` as the non-circular discriminator "
        "between anchors. This asks whether an anchor that knows nothing about "
        "holey-ness, but sits in ducts of the same SIZE, reproduces it.\n")

    for s in results["sections"]:
        add(f"\n## {s['section']}\n")
        a = s["task_a_area_extremity"]
        add(f"- eligible ducts: **{a['n_eligible_ducts']}**, median area "
            f"{a['eligible_area_um2']['median']:.0f} um^2")
        add(f"- root ducts: median area {a['root_area_um2']['median']:.0f} um^2 "
            f"({a['size_ratio_eligible_median_over_root_median']:.2f}x smaller)")
        add(f"- **{a['n_root_ducts_below_eligible_median']}/{a['n_root_ducts']} root "
            f"ducts below the eligible median** (exact binomial p = "
            f"{a['exact_binomial_p_vs_half']:.3g})")
        add(f"- frozen-graph re-run reproduces the stored axis at rho = "
            f"{s['consistency_rerun_vs_stored_rho']:.6f} "
            f"({'OK' if s['consistency_ok'] else '**FAILED — numbers below are unsafe**'})\n")

        obs = s["observed_holeyroot"]
        add("### Observed holeyroot axis (duct level)\n")
        add("| quantity | value |")
        add("|---|---|")
        for k in ("rho_pt_hole_pct", "rho_pt_area", "rho_pt_nuclear_density",
                  "partial_pt_area_given_hole"):
            add(f"| `{k}` | {obs[k]:+.4f} |")

        for key, title in (("task_b_summary", "Task B — area-matched surrogate "
                            "(knows nothing about hole %)"),
                           ("task_d_summary", "Task D — uniform random ducts")):
            summ = s[key]
            add(f"\n### {title}\n")
            add("| quantity | observed | null median | null range | "
                "frac null as extreme |")
            add("|---|---|---|---|---|")
            for q in ("rho_pt_area", "rho_pt_hole_pct", "rho_pt_nuclear_density"):
                r = summ.get(q)
                if not r:
                    continue
                add(f"| `{q}` | {r['observed_holeyroot']:+.4f} | "
                    f"{r['null_median']:+.4f} | "
                    f"[{r['null_min']:+.4f}, {r['null_max']:+.4f}] | "
                    f"{r['frac_null_at_least_as_extreme']:.2f} |")
            rr = summ["rho_vs_holeyroot_production"]
            add(f"\nrho(null axis, holeyroot production) median {rr['median']:+.4f}, "
                f"range [{rr['min']:+.4f}, {rr['max']:+.4f}].")

        c = s["task_c_area_stratified_holeyness"]
        add("\n### Task C — lowest hole % WITHIN area strata\n")
        add(f"- root duct area median {c['anchor']['duct_area_um2']['median']:.0f} um^2, "
            f"{c['anchor']['n_below_eligible_area_median']}/{c['anchor']['n_ducts']} "
            "below the eligible median")
        add(f"- `rho_pt_area` = {c['duct_level']['rho_pt_area']:+.4f}, "
            f"`rho_pt_hole_pct` = {c['duct_level']['rho_pt_hole_pct']:+.4f}, "
            f"`rho_pt_nuclear_density` = "
            f"{c['duct_level']['rho_pt_nuclear_density']:+.4f}")
        add("\n> If `rho_pt_hole_pct` survives here while `rho_pt_area` collapses, "
            "holey-ness is doing work that size is not. If both collapse, Phase 2's "
            "discriminator was size.")

        add("\n### Eccentricity — is this still a radial coordinate?\n")
        add("| axis | rho(pt, diffmap centroid distance) |")
        add("|---|---|")
        add(f"| area-stratified holeyness | "
            f"{c['eccentricity']['rho_pt_diffmap_centroid_distance']:+.4f} |")
        for lbl, draws in (("surrogate", s["task_b_area_matched_surrogate"]),
                           ("random", s["task_d_uniform_random"])):
            v = np.array([d["eccentricity"]["rho_pt_diffmap_centroid_distance"]
                          for d in draws], dtype=float)
            add(f"| {lbl} (median of {len(v)} draws) | {np.median(v):+.4f} |")
        add("\n> root_sensitivity run 4 measured 0.808 (2M-1) / 0.802 (2M-2) on the "
            "production axis. Values near those mean the anchor change did not alter "
            "what the axis IS.")

        e = s["task_e_v2_root_repair"]
        add("\n### Task E — v2 root repair\n")
        add(f"- {e['n_dropped']}/{e['n_roots']} v2 roots order the manifold backwards "
            "relative to their peers")
        add(f"- pseudotime_std, all roots: "
            f"{e['all_roots']['pseudotime_std_median_pct_of_range']:.2f}% of range")
        add(f"- rho(v2 all roots, holeyroot) = {e['all_roots']['rho_vs_holeyroot']:+.4f}")
        if e["repaired"]:
            r = e["repaired"]
            add(f"- **repaired ({r['n_roots_kept']} roots): std "
                f"{r['pseudotime_std_median_pct_of_range']:.2f}% of range, "
                f"rho vs holeyroot = {r['rho_vs_holeyroot']:+.4f}, "
                f"re-enters {r['band']}: {r['reenters_random_root_band']}**")
        add(f"\n> {e['verdict']}")

    add("\n## Not to be read as validation\n")
    add("- Task B covering the observed value shows the discriminator is "
        "uninformative, NOT that holey-ness is wrong.")
    add("- Task B failing to cover it shows size alone is insufficient, NOT that "
        "the anchor is right: the candidate pool still excludes every duct with no "
        "assigned patch, systematically the smallest and least holey.")
    add("- Task E repairing v2 does not make either axis correct. v2's roots are "
        "20 patches with zero measured nuclei, none inside a Tumor annotation.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
