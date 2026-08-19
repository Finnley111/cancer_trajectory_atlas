"""Persist an alternative-anchor pseudotime as a run dir the other tools can read.

WHY THIS EXISTS
---------------
``anchor_area_control`` showed that the holeyroot anchor is duct-size-extreme
(20/20 root ducts below the eligible median in both sections) and that its
``rho(pt, duct area)`` is fully reproduced by size-matched anchors that know
nothing about hole %. Its Task C built the fix — the AREA-STRATIFIED anchor,
lowest hole % WITHIN each area stratum — which removes the size extremity while
keeping the anchor's meaning, and which drove ``rho(pt, area)`` back to the
random-anchor baseline (+0.2664 in 2M-1, +0.0205 in 2M-2) while leaving
``rho(pt, hole_pct)`` intact.

But ``anchor_area_control`` computes that pseudotime and DISCARDS the vector — it
keeps only summary correlations. So every downstream tool that reads a run
directory (``eccentricity_check``, ``eccentricity_within_slide``,
``holeyroot_duct_checks``) can only ever see the production axis.

That is the gap this closes, and it matters for a specific reason.
``eccentricity_check`` returned DIRECTIONAL IN MORPHOLOGY for the holeyroot axis,
counting ``nuclear_density`` among 2M-2's directional, within-slide-surviving
features. ``anchor_area_control`` found that same relationship to be a duct-size
artifact of the anchor: size-matched surrogates over-explain it, and under the
area-stratified anchor it goes to **-0.244** at duct level. So the trajectory
verdict is conditional on an anchor we now know to be size-extreme.
``eccentricity_check`` cannot test that — it reads whatever pseudotime it is
given. Re-running it against the area-stratified axis is the direct test, and it
needs that axis to exist on disk.

WHAT IT WRITES
--------------
A DERIVED run directory containing:

    adata_full.h5ad   the source run's AnnData with obs['pseudotime'] and
                      obs['pseudotime_std'] REPLACED, uns['dpt_root_candidates']
                      set to the new roots, and uns['anchor_provenance'] recording
                      what was done. X, obsm, obsp and uns['neighbors'] /
                      uns['diffmap_evals'] are carried over untouched, so the
                      diffusion map and graph are the SAME objects the production
                      axis was built on and nothing is re-embedded.
    results.csv       the source run's csv with the same two columns replaced, so
                      the derived dir also satisfies the row-alignment check in
                      anchor_area_control.DuctContext and can be read by
                      holeyroot_duct_checks.
    anchor_axis.json  provenance, the root set, and the consistency gate below.

Everything is written under --output-dir. No run tree is modified.

THE CONSISTENCY GATE
--------------------
The area-stratified rule is fully deterministic — no RNG — so rebuilding it here
must reproduce ``anchor_area_control``'s Task C exactly. Pass
``--expect-json`` pointing at ``anchor_area_control.json`` and this module
recomputes the duct-level correlations and compares them to the recorded Task C
values, refusing if they differ beyond --tolerance. Without that check a silent
divergence in the rebuild would hand every downstream tool a different axis under
the same name.

Cost is ~20 ``sc.tl.dpt`` calls per section on a frozen graph — minutes, not the
hours ``anchor_area_control`` takes, because none of the 50 null draws are
rebuilt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .holeyness import PATCH_SIZE_DEFAULT
from .root_sensitivity import build_dpt_adata, run_multi_root_dpt, load_run
from .anchor_area_control import (
    DuctContext, duct_rhos, describe_anchor, _json_default, _safe_rho,
    select_area_stratified_holeyness, select_area_matched_surrogate,
    N_ROOTS_DEFAULT, N_STRATA_DEFAULT, AREA_MATCH_TOL, MATCH_POOL_DEFAULT,
)

TOLERANCE_DEFAULT = 1e-6
COMPARED_KEYS = ("rho_pt_hole_pct", "rho_pt_area", "rho_pt_nuclear_density")


def build_anchor(ctx: DuctContext, anchor: str, n_roots: int, n_strata: int,
                 seed: int, area_tol: float, match_pool: int,
                 source_roots: list[int]) -> tuple[list[str], dict]:
    """The duct set for a named anchor. Returns (ducts, provenance)."""
    if anchor == "area_stratified":
        ducts = select_area_stratified_holeyness(ctx, n_roots, n_strata)
        prov = {"rule": ("lowest hole % WITHIN each of "
                         f"{n_strata} equal-count duct-area strata"),
                "deterministic": True, "n_strata": n_strata}
    elif anchor == "area_matched_surrogate":
        # The size-matched control, exported so its axis can also be examined.
        # This one IS random, so the seed is part of its identity and is recorded.
        target_ducts = []
        for r in source_roots:
            d = ctx.duct_id[r]
            if d is None:
                raise ValueError(
                    f"source root patch {r} lies in no duct under this rebuild of the "
                    "assignment, so it cannot supply an area target.")
            target_ducts.append(d)
        ducts, quality = select_area_matched_surrogate(
            ctx, target_ducts, np.random.default_rng(seed), area_tol, match_pool)
        prov = {"rule": ("20 ducts size-matched to the source anchor's root ducts, "
                         "selected WITHOUT reference to hole %"),
                "deterministic": False, "seed": seed, "match_quality": quality}
    else:
        raise ValueError(
            f"unknown anchor {anchor!r}. Supported: 'area_stratified' (the "
            "anchor_area_control Task C fix) and 'area_matched_surrogate' (its "
            "Task B control).")
    return ducts, prov


def check_against_expected(section: str, anchor: str, observed: dict,
                           expect_json: Path, tolerance: float) -> dict:
    """Refuse if the rebuild does not reproduce anchor_area_control's own numbers."""
    payload = json.loads(Path(expect_json).read_text(encoding="utf-8"))
    sec = next((s for s in payload.get("sections", [])
                if s.get("section") == section), None)
    if sec is None:
        raise ValueError(
            f"{expect_json} has no section {section!r}; sections present: "
            f"{[s.get('section') for s in payload.get('sections', [])]}")
    if anchor != "area_stratified":
        return {"checked": False,
                "reason": (f"anchor {anchor!r} is seed-dependent; "
                           "anchor_area_control records no single value to match.")}

    expected = sec["task_c_area_stratified_holeyness"]["duct_level"]
    diffs, worst = {}, 0.0
    for k in COMPARED_KEYS:
        e, o = float(expected[k]), float(observed[k])
        diffs[k] = {"expected": e, "observed": o, "abs_diff": abs(e - o)}
        worst = max(worst, abs(e - o))
    if worst > tolerance:
        raise ValueError(
            f"{section}: the rebuilt {anchor} axis does NOT reproduce "
            f"anchor_area_control's Task C (worst |diff| = {worst:.3g} > "
            f"{tolerance:g}). Details: {json.dumps(diffs, indent=2)}\n"
            "The area-stratified rule is deterministic, so a difference means the "
            "duct context or the DPT inputs differ between the two runs. Refusing "
            "to export an axis that is not the one Task C measured."
        )
    print(f"  consistency gate PASSED — worst |diff| vs Task C = {worst:.3g}")
    return {"checked": True, "worst_abs_diff": float(worst),
            "tolerance": float(tolerance), "per_quantity": diffs}


def write_derived_run(out_dir: Path, adata, results: pd.DataFrame,
                      pseudotime: np.ndarray, pseudotime_std: np.ndarray,
                      roots: list[int], provenance: dict) -> dict:
    """Write adata_full.h5ad + results.csv with the axis replaced, nothing else."""
    out_dir.mkdir(parents=True, exist_ok=True)

    derived = adata.copy()
    derived.obs["pseudotime"] = np.asarray(pseudotime, dtype=float)
    derived.obs["pseudotime_std"] = np.asarray(pseudotime_std, dtype=float)
    derived.uns["dpt_root_candidates"] = np.asarray(roots, dtype=np.int64)
    # Recorded so no later reader mistakes this for a production run. The name is
    # deliberately not 'dpt_*' so it cannot be confused with pipeline output.
    derived.uns["anchor_provenance"] = json.dumps(provenance, default=_json_default)
    derived.write(out_dir / "adata_full.h5ad")

    csv = results.copy()
    csv["pseudotime"] = np.asarray(pseudotime, dtype=float)
    csv["pseudotime_std"] = np.asarray(pseudotime_std, dtype=float)
    csv.to_csv(out_dir / "results.csv", index=False)

    return {"adata_full.h5ad": str(out_dir / "adata_full.h5ad"),
            "results.csv": str(out_dir / "results.csv")}


def run_section(section: str, run_dir: Path, out_dir: Path, anchor: str,
                export: Path, ann_dir: Path, dims: Path, slide_list: Path,
                patch_size: int, n_roots: int, n_strata: int, seed: int,
                area_tol: float, match_pool: int, expect_json: Path | None,
                tolerance: float) -> dict:
    print("\n" + "=" * 78)
    print(f"  SECTION {section}   anchor={anchor}")
    print("=" * 78)

    adata, _, _ = load_run(run_dir)
    ctx = DuctContext(adata, run_dir, export, ann_dir, dims, slide_list, patch_size)
    results = pd.read_csv(Path(run_dir) / "results.csv")

    if "dpt_root_candidates" not in adata.uns:
        raise KeyError(f"{run_dir}: adata.uns['dpt_root_candidates'] missing.")
    source_roots = [int(i) for i in
                    np.asarray(adata.uns["dpt_root_candidates"]).ravel()]

    # Gate first: the frozen-graph re-run must reproduce the SOURCE axis, or the
    # exported axis is measured against a different DPT than production used.
    base = build_dpt_adata(adata)
    src_pt = adata.obs["pseudotime"].values.astype(float)
    check_pt, _, _ = run_multi_root_dpt(None, source_roots, base=base)
    repro = _safe_rho(check_pt, src_pt)
    print(f"  frozen-graph re-run of the SOURCE roots vs stored axis: rho = {repro:.6f}")
    if repro < 0.999:
        raise ValueError(
            f"{section}: re-running DPT from the source run's own roots reproduces "
            f"its stored axis at only rho = {repro:.6f}. The graph or diffusion map "
            "on disk is not the one that produced the stored pseudotime, so any "
            "axis exported from it would not be comparable. Refusing to export.")

    ducts, anchor_prov = build_anchor(ctx, anchor, n_roots, n_strata, seed,
                                      area_tol, match_pool, source_roots)
    roots = ctx.roots_from_ducts(ducts)
    pt, pt_std, notes = run_multi_root_dpt(None, roots, base=base)

    rhos = duct_rhos(ctx, pt)
    print(f"  rebuilt axis: rho(pt,area)={rhos['rho_pt_area']:+.4f}  "
          f"rho(pt,hole)={rhos['rho_pt_hole_pct']:+.4f}  "
          f"rho(pt,nd)={rhos['rho_pt_nuclear_density']:+.4f}")
    print(f"  rho(rebuilt axis, source production axis) = "
          f"{_safe_rho(pt, src_pt):+.4f}")

    gate = ({"checked": False, "reason": "--expect-json not supplied"}
            if expect_json is None else
            check_against_expected(section, anchor, rhos, expect_json, tolerance))

    provenance = {
        "derived_from": str(run_dir),
        "anchor": anchor,
        "anchor_rule": anchor_prov,
        "n_roots": len(roots),
        "root_patch_indices": roots,
        "source_root_patch_indices": source_roots,
        "source_axis_reproduced_rho": repro,
        "rho_vs_source_axis": _safe_rho(pt, src_pt),
        "duct_level": rhos,
        "anchor_description": describe_anchor(ctx, ducts),
        "dpt_notes": notes,
        "consistency_gate": gate,
        "warning": (
            "DERIVED, NOT PRODUCTION. obs['pseudotime'] here comes from an "
            "alternative anchor. Everything else — X, X_diffmap, the neighbour "
            "graph, every morphological feature — is carried over unchanged from "
            "the source run. Note that eccentricity_check's Task 0 reconstructs "
            "roots as argsort(nuclear_density)[:20] and will therefore profile the "
            "WRONG root set for this directory, exactly as it does for a holeyroot "
            "run; read uns['anchor_provenance'] instead and ignore its Task 0."
        ),
    }

    files = write_derived_run(out_dir, adata, results, pt, pt_std, roots, provenance)
    (out_dir / "anchor_axis.json").write_text(
        json.dumps(provenance, indent=2, default=_json_default), encoding="utf-8")
    print(f"  wrote {out_dir}")
    return {"section": section, "output_dir": str(out_dir), "files": files,
            **{k: v for k, v in provenance.items() if k != "root_patch_indices"}}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True,
                    help="Source runs, one per section, SAME ORDER.")
    ap.add_argument("--exports", nargs="+", type=Path, required=True,
                    help="One per section, SAME ORDER. 2M-1 and 2M-2 differ.")
    ap.add_argument("--slide-lists", nargs="+", type=Path, required=True)
    ap.add_argument("--annotation-dir", type=Path, required=True)
    ap.add_argument("--slide-dimensions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Parent; one atlas_<section> subdir is written per section.")
    ap.add_argument("--anchor", default="area_stratified",
                    choices=["area_stratified", "area_matched_surrogate"])
    ap.add_argument("--expect-json", type=Path, default=None,
                    help="anchor_area_control.json — enables the consistency gate.")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_DEFAULT)
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    ap.add_argument("--n-roots", type=int, default=N_ROOTS_DEFAULT)
    ap.add_argument("--n-strata", type=int, default=N_STRATA_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--area-match-tol", type=float, default=AREA_MATCH_TOL)
    ap.add_argument("--match-pool", type=int, default=MATCH_POOL_DEFAULT)
    args = ap.parse_args()

    n = len(args.sections)
    for name, seq in (("--run-dirs", args.run_dirs), ("--exports", args.exports),
                      ("--slide-lists", args.slide_lists)):
        if len(seq) != n:
            raise SystemExit(
                f"{name} has {len(seq)} entries but --sections has {n}. These are "
                "positional; a mismatch would export one section's axis built from "
                "another section's annotations.")

    res = {"analysis": "export_anchor_axis",
           "config": {k: str(v) for k, v in vars(args).items()}, "sections": []}
    for i, sec in enumerate(args.sections):
        res["sections"].append(run_section(
            sec, args.run_dirs[i], args.output_dir / f"atlas_{sec}", args.anchor,
            args.exports[i], args.annotation_dir, args.slide_dimensions,
            args.slide_lists[i], args.patch_size, args.n_roots, args.n_strata,
            args.seed, float(args.area_match_tol), int(args.match_pool),
            args.expect_json, float(args.tolerance)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_anchor_axis.json").write_text(
        json.dumps(res, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {args.output_dir / 'export_anchor_axis.json'}")
    print("\nThese directories can now be passed to --run-dirs of "
          "eccentricity_check and eccentricity_within_slide.")


if __name__ == "__main__":
    main()
