"""v3 root/filter experiment — comparison report. REPORTS NUMBERS, ADJUDICATES NOTHING.

Compares three configs against the v2 per-section reference for section 2M-1:

    v3a  holeyness roots,   production filters   (root rule alone)
    v3b  production roots,  relaxed filters      (filter change alone)
    v3c  holeyness roots,   relaxed filters      (both, to expose interaction)

and v3a vs v3c, which isolates the filter effect at a FIXED root rule.

THE EXPECTATION, FIXED BEFORE THE RUN
-------------------------------------
Uniformly random 20-root sets already reproduce the v2 pseudotime at |rho|
0.78-0.89. The manifold fixes the ORDERING; roots fix only which end is zero. So
a root-rule change is EXPECTED to alter orientation and root membership and to
leave the ordering largely intact. v3a landing near 0.78-0.89 is the PREDICTION,
not a discovery. |rho| < 0.7 CONTRADICTS the random-root result and is flagged
prominently as needing explanation before anything downstream is trusted.

WHAT THIS MODULE REFUSES TO DO
------------------------------
It does not declare any configuration better. In particular a reduced
``pseudotime_std``, or a ``nuclear_density`` correlation that flips to the
"expected" sign, are NOT treated as validating the new anchor — either can occur
for reasons unrelated to the anchor being more biologically correct, and both are
printed with that caveat attached.

TWO HONEST GAPS, REPORTED RATHER THAN PAPERED OVER
--------------------------------------------------
1. RAW PSEUDOTIME RANGE. ``pseudotime`` in results.csv is min-max normalised;
   ``pseudotime_std`` is NOT. Expressing std as a fraction of the run's own raw
   range needs ``pt_max - pt_min``, which ``compute_dpt_multi_root`` PRINTS
   (analysis/diffusion.py) but never persists. This module parses it from a
   SLURM log when one is supplied via --log, and reports it as UNRECOVERABLE
   otherwise. It never invents a normalisation.
   (One-line production fix, deliberately NOT applied here: store pt_min/pt_max
   into adata.uns alongside dpt_root_candidates.)
2. CELLULARITY CONFOUND FOR v2. ``cellularity_confound.analyze_run_nuclear_density``
   WRITES into the run directory it analyses — see the warning in
   analysis/v3_regression_check.py. Re-running it against per_section_v2 would
   modify the reference tree, which this experiment is forbidden from doing. So
   v2's confound verdicts are READ from its existing
   cellularity_confound/cellularity_confound.json, and reported as unavailable if
   that file is absent.

READ-ONLY with respect to every run tree. Writes only into --output-dir.

Usage:
    python -m cancer_trajectory_atlas.analysis.v3_root_experiment_compare \\
        --v2-dir   $SCRATCH/results/per_section_v2/atlas_2M-1 \\
        --config-labels v3a v3b v3c \\
        --config-dirs $SCRATCH/results/per_section_v3a_holeyroot/atlas_2M-1 \\
                      $SCRATCH/results/per_section_v3b_relaxed/atlas_2M-1 \\
                      $SCRATCH/results/per_section_v3c_both/atlas_2M-1 \\
        --output-dir $SCRATCH/results/v3_root_experiment/compare \\
        --logs logs/per_section_v2-123.out logs/v3a-124.out ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = ["nuclear_density", "nc_ratio", "mean_nuclear_area",
            "texture_entropy", "packing_irregularity", "h_intensity",
            "h_intensity_wholepatch"]

DELTA_FLAG = 0.1
RHO_CONTRADICTION_FLOOR = 0.7
KEY = ["slide_name", "x", "y"]

# "  Pseudotime median range: [0.0000, 0.4211] → normalized [0, 1]"
_PT_RANGE_RE = re.compile(
    r"Pseudotime median range:\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> dict:
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found.")
    df = pd.read_csv(csv)
    for c in KEY:
        if c not in df.columns:
            raise KeyError(f"{csv} lacks required key column '{c}'.")

    rec = {"dir": run_dir, "df": df, "n_patches": len(df)}

    h5 = run_dir / "adata_full.h5ad"
    rec["roots"], rec["root_source"], rec["pca_width"] = None, None, None
    if h5.exists():
        try:
            import anndata as ad
            a = ad.read_h5ad(h5, backed="r")
            if "dpt_root_candidates" in a.uns:
                rec["roots"] = [int(i) for i in
                                np.asarray(a.uns["dpt_root_candidates"]).ravel()]
            rec["root_source"] = str(a.uns.get("dpt_root_source", "nuclear_density"))
            rec["pca_width"] = int(a.X.shape[1]) if a.X is not None else None
        except Exception as e:                      # noqa: BLE001
            rec["h5ad_error"] = f"{type(e).__name__}: {e}"

    hr = run_dir / "holeyness_roots.json"
    rec["holeyness"] = json.loads(hr.read_text()) if hr.exists() else None

    cc = run_dir / "cellularity_confound" / "cellularity_confound.json"
    rec["confound"] = json.loads(cc.read_text()) if cc.exists() else None
    return rec


def parse_pt_range(log_paths: list[Path], run_dir: Path) -> tuple[float | None, str]:
    """Recover pt_max - pt_min from a SLURM log. Never fabricated, never guessed.

    Matching is on the EXACT output-directory string, which every v3 job script
    echoes. A looser match on the parent directory name is NOT safe: the relaxed
    cache prepop job writes to <V3B_BASE>/_prepop_discard and also prints a
    pseudotime range, so a parent-name match would silently attribute the
    prepop run's range to Config B.

    If several logs claim the same run directory with DIFFERENT ranges — a
    re-run, say — this reports the ambiguity rather than picking one.
    """
    # Path-BOUNDARY match, not a plain substring: "<...>/atlas_2M-1" is a prefix
    # of "<...>/atlas_2M-1_retry", and "<...>/per_section_v3b_relaxed" is a prefix
    # of the prepop job's "<...>/per_section_v3b_relaxed/_prepop_discard". A
    # substring test would silently attribute the wrong run's range.
    # The path must END here: followed by whitespace or end-of-line. Anything
    # else means it is a PREFIX of a different path, which must not match —
    # both "<dir>_retry" (sibling run) and "<dir>/_prepop_discard" (the cache
    # prepop job, which also prints a pseudotime range) are near-misses that a
    # looser test attributes to the wrong run.
    needle = re.compile(re.escape(str(run_dir)) + r"(?=\s|$)")
    found: list[tuple[float, str]] = []
    for p in log_paths:
        if not p.exists():
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if not needle.search(txt):
            continue
        hits = _PT_RANGE_RE.findall(txt)
        if hits:
            lo, hi = float(hits[-1][0]), float(hits[-1][1])
            found.append((hi - lo, p.name))

    if not found:
        return None, ("UNRECOVERABLE — compute_dpt_multi_root prints pt_min/pt_max "
                      "but never persists them, and no log naming this exact run "
                      "directory was supplied. NOT fabricated.")
    ranges = {round(r, 12) for r, _ in found}
    if len(ranges) > 1:
        return None, ("AMBIGUOUS — logs " + ", ".join(n for _, n in found) +
                      f" name this run directory but report {len(ranges)} different "
                      "pseudotime ranges. Refusing to pick one.")
    return found[0][0], f"parsed from {found[0][1]}"


# ── Comparisons ───────────────────────────────────────────────────────────────

def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inner-join on (slide_name, x, y). Reports whether the sets are identical."""
    for d, nm in ((a, "left"), (b, "right")):
        if d.duplicated(subset=KEY).any():
            raise ValueError(
                f"{nm} results.csv has duplicate (slide_name, x, y) keys — the "
                "patch identity join is not well defined.")
    m = a[KEY + [c for c in FEATURES + ["pseudotime", "pseudotime_std"]
                 if c in a.columns]].merge(
        b[KEY + [c for c in FEATURES + ["pseudotime", "pseudotime_std"]
                 if c in b.columns]],
        on=KEY, how="inner", suffixes=("_ref", "_cfg"))
    info = {
        "n_ref": len(a), "n_cfg": len(b), "n_shared": len(m),
        "identical_patch_sets": bool(len(a) == len(b) == len(m)),
        "frac_of_ref_shared": float(len(m) / len(a)) if len(a) else 0.0,
    }
    return m, info


def feature_rhos(df: pd.DataFrame, suffix: str) -> dict:
    out = {}
    pt = df[f"pseudotime{suffix}"].values
    for f in FEATURES:
        col = f"{f}{suffix}"
        if col not in df.columns:
            out[f] = None
            continue
        v = df[col].values
        mask = np.isfinite(pt) & np.isfinite(v)
        out[f] = float(spearmanr(pt[mask], v[mask]).statistic) if mask.sum() >= 4 else None
    return out


def compare(ref: dict, cfg: dict, label: str, logs: list[Path]) -> dict:
    merged, align_info = align(ref["df"], cfg["df"])
    identical = align_info["identical_patch_sets"]

    pt_r = merged["pseudotime_ref"].values
    pt_c = merged["pseudotime_cfg"].values
    mask = np.isfinite(pt_r) & np.isfinite(pt_c)
    rho = float(spearmanr(pt_r[mask], pt_c[mask]).statistic) if mask.sum() >= 4 else None

    # Root-set overlap
    roots_r = set(ref["roots"] or [])
    roots_c = set(cfg["roots"] or [])
    if identical and ref["roots"] is not None and cfg["roots"] is not None:
        n_diff, root_note = len(roots_c - roots_r), None
    else:
        n_diff, root_note = None, (
            "Root indices are positions in each run's own patch array. The patch "
            "sets differ, so index overlap is MEANINGLESS and is not reported. "
            "Compare the roots' properties instead.")

    def root_props(rec: dict) -> dict:
        if rec["roots"] is None:
            return {"unavailable": True}
        d = rec["df"]
        sub = d.iloc[[i for i in rec["roots"] if 0 <= i < len(d)]]
        p = {}
        for c in ("nuclear_density", "nucleus_count"):
            if c in sub.columns:
                v = sub[c].values.astype(float)
                v = v[np.isfinite(v)]
                p[c] = ({"median": float(np.median(v)), "min": float(v.min()),
                         "max": float(v.max())} if v.size else None)
        if rec["holeyness"]:
            hp = np.array([r["hole_pct"] for r in
                           rec["holeyness"]["selected_roots"]], float)
            p["hole_pct"] = {"median": float(np.median(hp)),
                             "min": float(hp.min()), "max": float(hp.max())}
            p["n_distinct_ducts"] = len({r["duct_id"] for r in
                                         rec["holeyness"]["selected_roots"]})
        return p

    rr, rc = feature_rhos(merged, "_ref"), feature_rhos(merged, "_cfg")
    feats = {}
    for f in FEATURES:
        a, b = rr.get(f), rc.get(f)
        delta = None if (a is None or b is None) else abs(b - a)
        feats[f] = {"v2": a, label: b, "abs_delta": delta,
                    "flagged": bool(delta is not None and delta > DELTA_FLAG)}

    rng, rng_src = parse_pt_range(logs, cfg["dir"])
    std = cfg["df"]["pseudotime_std"].values.astype(float) if \
        "pseudotime_std" in cfg["df"].columns else np.array([])
    std = std[np.isfinite(std)]
    std_block = {
        "median_raw": float(np.median(std)) if std.size else None,
        "p25_raw": float(np.percentile(std, 25)) if std.size else None,
        "p75_raw": float(np.percentile(std, 75)) if std.size else None,
        "raw_pt_range": rng,
        "raw_pt_range_source": rng_src,
        "median_as_pct_of_raw_range": (
            float(100 * np.median(std) / rng) if (std.size and rng) else None),
        "caveat": ("A SMALLER pseudotime_std does NOT validate the anchor. It can "
                   "fall for reasons unrelated to the anchor being more "
                   "biologically correct."),
    }

    nd_note = None
    if cfg["holeyness"] is not None:
        nd_note = ("nuclear_density is NOT the root variable in this config, so its "
                   "correlation with pseudotime is non-circular here for the first "
                   "time. A sign flip toward the 'expected' direction is NOT "
                   "evidence the anchor is better.")

    return {
        "label": label,
        "alignment": align_info,
        "elementwise_defined": identical,
        "spearman_vs_v2": rho,
        "spearman_scope": ("element-wise, identical patch sets" if identical else
                           f"shared subset only, n={align_info['n_shared']} of "
                           f"{align_info['n_ref']} v2 patches — element-wise "
                           "comparison is NOT defined for differing patch sets"),
        "contradicts_random_root_finding": bool(
            rho is not None and abs(rho) < RHO_CONTRADICTION_FLOOR),
        "root_source": cfg.get("root_source"),
        "n_roots_differing_from_v2": n_diff,
        "root_index_note": root_note,
        "root_properties_v2": root_props(ref),
        "root_properties_config": root_props(cfg),
        "feature_correlations": feats,
        "n_flagged_features": sum(1 for v in feats.values() if v["flagged"]),
        "pca_width": {"v2": ref.get("pca_width"), label: cfg.get("pca_width")},
        "patch_counts": {"v2": ref["n_patches"], label: cfg["n_patches"],
                         "pct_change": (float(100 * (cfg["n_patches"] - ref["n_patches"])
                                              / ref["n_patches"])
                                        if ref["n_patches"] else None)},
        "pseudotime_std": std_block,
        "frac_patches_no_duct": (
            cfg["holeyness"]["counts"]["frac_patches_no_duct"]
            if cfg["holeyness"] else None),
        "nuclear_density_circularity_note": nd_note,
        "confound": {
            "v2": ref["confound"] if ref["confound"] else
                  "UNAVAILABLE — per_section_v2 has no cellularity_confound.json and "
                  "this module will not generate one, because analyze_run_nuclear_density "
                  "writes into the tree it analyses and v2 must not be modified.",
            label: cfg["confound"] if cfg["confound"] else "not run for this config",
        },
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_report(res: dict, out: Path) -> None:
    L = ["# v3 root/filter experiment — comparison vs per_section_v2 (2M-1)", "",
         "**This report adjudicates nothing.** No configuration is declared better.",
         "", "## Expectation, fixed before running", "",
         "Random 20-root sets reproduce v2 pseudotime at |rho| 0.78-0.89. A root-rule",
         "change is therefore EXPECTED to change orientation and root membership, and",
         "NOT the ordering. Results are assessed against that, not post hoc.", ""]

    for c in res["configs"]:
        lb = c["label"]
        L += [f"## {lb}", "",
              f"- root source: `{c['root_source']}`",
              f"- patches: v2 {c['patch_counts']['v2']} -> {lb} "
              f"{c['patch_counts'][lb]}  ({_f(c['patch_counts']['pct_change'], 1)}%)",
              f"- PCA width: v2 {c['pca_width']['v2']} -> {lb} {c['pca_width'][lb]}",
              f"- shared patches: {c['alignment']['n_shared']} "
              f"({'identical sets' if c['elementwise_defined'] else 'PARTIAL — element-wise undefined'})",
              f"- **Spearman(v2, {lb}) = {_f(c['spearman_vs_v2'])}**  _{c['spearman_scope']}_", ""]
        if c["contradicts_random_root_finding"]:
            L += [f"> ⚠ **|rho| < {RHO_CONTRADICTION_FLOOR} — CONTRADICTS the random-root",
                  "> finding that the ordering is root-insensitive. Explain this before",
                  "> trusting any number below.**", ""]
        if c["n_roots_differing_from_v2"] is not None:
            L += [f"- roots differing from v2: **{c['n_roots_differing_from_v2']}/20**", ""]
        else:
            L += [f"- roots: {c['root_index_note']}", ""]
        if c["frac_patches_no_duct"] is not None:
            L += [f"- patches in no duct: {100*c['frac_patches_no_duct']:.1f}%", ""]

        L += ["### Feature correlations vs pseudotime", "",
              "| feature | v2 | " + lb + " | \\|delta\\| | flag |", "|---|---|---|---|---|"]
        for f, v in c["feature_correlations"].items():
            L.append(f"| `{f}` | {_f(v['v2'])} | {_f(v[lb])} | "
                     f"{_f(v['abs_delta'])} | {'**>0.1**' if v['flagged'] else ''} |")
        L += ["", f"{c['n_flagged_features']} feature(s) moved by more than {DELTA_FLAG}.", ""]

        s = c["pseudotime_std"]
        L += ["### pseudotime_std", "",
              f"- median (raw): {_f(s['median_raw'])}  IQR [{_f(s['p25_raw'])}, {_f(s['p75_raw'])}]",
              f"- raw pseudotime range: {_f(s['raw_pt_range'])}  ({s['raw_pt_range_source']})",
              f"- median as % of that range: "
              f"{'n/a — range unrecoverable, NOT fabricated' if s['median_as_pct_of_raw_range'] is None else _f(s['median_as_pct_of_raw_range'], 2) + '%'}",
              f"- {s['caveat']}", ""]
        if c["nuclear_density_circularity_note"]:
            L += [f"> {c['nuclear_density_circularity_note']}", ""]

    if res.get("a_vs_c"):
        av = res["a_vs_c"]
        L += ["## v3a vs v3c — filter effect at a FIXED root rule", "",
              f"- shared patches: {av['alignment']['n_shared']}",
              f"- Spearman = {_f(av['spearman_vs_v2'])}  _{av['spearman_scope']}_", ""]

    L += ["## Not to be read as validation", "",
          "- A reduced `pseudotime_std` does not validate the anchor.",
          "- A `nuclear_density` correlation flipping to the 'expected' sign does not",
          "  validate the anchor. In holeyness configs it merely becomes non-circular.",
          "- Configs B and C changed the patch set, hence the PCA basis and every",
          "  absolute number. They are comparable to v2 on STRUCTURE only.", ""]
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--config-labels", nargs="+", required=True)
    ap.add_argument("--config-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--logs", nargs="*", type=Path, default=[],
                    help="SLURM .out files. The ONLY way to recover each run's raw "
                         "pre-normalisation pseudotime range; without them that "
                         "figure is reported unrecoverable, never invented.")
    args = ap.parse_args()

    if len(args.config_labels) != len(args.config_dirs):
        ap.error("--config-labels and --config-dirs must have the same length")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref = load_run(args.v2_dir)
    print(f"v2 reference: {ref['n_patches']} patches, PCA width {ref['pca_width']}")

    runs, configs = {}, []
    for lb, d in zip(args.config_labels, args.config_dirs):
        runs[lb] = load_run(d)
        configs.append(compare(ref, runs[lb], lb, args.logs))
        print(f"  {lb}: rho={configs[-1]['spearman_vs_v2']}, "
              f"{configs[-1]['n_flagged_features']} flagged feature(s)")

    res = {"reference": str(args.v2_dir), "configs": configs}
    if "v3a" in runs and "v3c" in runs:
        res["a_vs_c"] = compare(runs["v3a"], runs["v3c"], "v3c", args.logs)
        res["a_vs_c"]["note"] = ("Reference here is v3a, NOT v2 — this isolates the "
                                 "filter effect at a fixed (holeyness) root rule.")

    (args.output_dir / "v3_comparison.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_report(res, args.output_dir / "v3_comparison.md")
    print(f"\nWrote {args.output_dir}/v3_comparison.{{json,md}}")
    print("No configuration is declared better. Read the numbers.")


if __name__ == "__main__":
    main()
