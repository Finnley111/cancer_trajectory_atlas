"""
Read-only diagnostics for the morphological feature audit.

Implements the four diagnostics (D1–D4) and failure-rate cross-check (FC)
recommended in reports/morphological_features_audit.md, operating exclusively
on existing results.csv files already produced by the per-section pipeline runs.

Does NOT modify any pipeline output, rerun any pipeline stage, or overwrite
any existing results.csv, validation.json, or cellularity_confound.json.

Usage:
    python -m cancer_trajectory_atlas.diagnostics.audit_feature_diagnostics \\
        --section1-results $SCRATCH/results/per_section/atlas_2M-1/results.csv \\
        --section2-results $SCRATCH/results/per_section/atlas_2M-2/results.csv \\
        --output-report ~/cancer_trajectory_atlas/reports/morphological_features_diagnostics_results.md
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


FEATURES = [
    "nuclear_density",
    "mean_nuclear_area",
    "nc_ratio",
    "texture_entropy",
    "h_intensity",
    "packing_irregularity",
]
REQUIRED_COLS = FEATURES + ["pseudotime", "slide_name"]
STAIN_FEATURES = ["h_intensity", "nuclear_density", "mean_nuclear_area"]

# Thresholds from the audit
D1_ZERO_THRESHOLD_PCT = 0.5
D2_INF_THRESHOLD_PCT = 0.1
FC_THRESHOLD_PCT = 0.5
D3_LARGE_EFFECT = 0.3
D3_SMALL_EFFECT = 0.1


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"ERROR: {label} results.csv not found at:\n  {path}\n"
            "  Confirm the per-section run completed on Narval before running this diagnostic."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {label} results.csv missing required columns: {missing}\n"
            f"  File: {path}"
        )
    print(f"  [{label}] Loaded {len(df):,} rows from:\n    {path}")
    return df


# ── Diagnostics ───────────────────────────────────────────────────────────────

def d1_failure_zeros(df: pd.DataFrame, label: str) -> dict:
    """D1: Count patches with texture_entropy == 0.0 as an upper-bound failure proxy."""
    n = len(df)
    te = df["texture_entropy"].astype(float)
    zero_count = int((te == 0.0).sum())
    nan_count = int(te.isna().sum())
    zero_pct = zero_count / n * 100

    print(f"\n[D1 | {label}] Failure-zeroed patches (texture_entropy == 0.0)")
    print(f"  Total patches          : {n:,}")
    print(f"  texture_entropy == 0.0 : {zero_count:,}  ({zero_pct:.3f}%)")
    print(f"  texture_entropy is NaN : {nan_count:,}  ({nan_count / n * 100:.3f}%)")
    flag = "ABOVE" if zero_pct > D1_ZERO_THRESHOLD_PCT else "below"
    print(f"  [{flag} {D1_ZERO_THRESHOLD_PCT}% threshold]")

    return {
        "n": n,
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "nan_count": nan_count,
        "nan_pct": nan_count / n * 100,
        "above_threshold": zero_pct > D1_ZERO_THRESHOLD_PCT,
    }


def d2_nc_ratio_inf(df: pd.DataFrame, label: str) -> dict:
    """D2: nc_ratio infinite-value count and pseudotime/density of affected patches."""
    n = len(df)
    ncr = df["nc_ratio"].astype(float).values
    pt = df["pseudotime"].astype(float).values
    nd = df["nuclear_density"].astype(float).values

    inf_mask = np.isinf(ncr)
    inf_count = int(inf_mask.sum())
    nan_count = int(np.isnan(ncr).sum())
    inf_pct = inf_count / n * 100

    print(f"\n[D2 | {label}] nc_ratio infinite values")
    print(f"  Total patches   : {n:,}")
    print(f"  nc_ratio is inf : {inf_count:,}  ({inf_pct:.3f}%)")
    print(f"  nc_ratio is NaN : {nan_count:,}  ({nan_count / n * 100:.3f}%)")

    result: dict = {
        "n": n,
        "inf_count": inf_count,
        "inf_pct": inf_pct,
        "nan_count": nan_count,
        "above_threshold": inf_pct > D2_INF_THRESHOLD_PCT,
        "inf_pt_min": None, "inf_pt_median": None, "inf_pt_max": None,
        "inf_nd_min": None, "inf_nd_median": None, "inf_nd_max": None,
        "overall_pt_median": float(np.nanmedian(pt)),
        "overall_nd_median": float(np.nanmedian(nd)),
    }

    if inf_count > 0:
        pt_inf = pt[inf_mask]
        nd_inf = nd[inf_mask]
        result.update({
            "inf_pt_min": float(np.nanmin(pt_inf)),
            "inf_pt_median": float(np.nanmedian(pt_inf)),
            "inf_pt_max": float(np.nanmax(pt_inf)),
            "inf_nd_min": float(np.nanmin(nd_inf)),
            "inf_nd_median": float(np.nanmedian(nd_inf)),
            "inf_nd_max": float(np.nanmax(nd_inf)),
        })
        print(f"  inf-nc_ratio patch distribution:")
        print(f"    pseudotime      min={result['inf_pt_min']:.4f}  "
              f"median={result['inf_pt_median']:.4f}  max={result['inf_pt_max']:.4f}")
        print(f"    nuclear_density min={result['inf_nd_min']:.5f}  "
              f"median={result['inf_nd_median']:.5f}  max={result['inf_nd_max']:.5f}")
        print(f"  Overall medians:  pseudotime={result['overall_pt_median']:.4f}  "
              f"nuclear_density={result['overall_nd_median']:.5f}")
    else:
        print(f"  No inf values — nc_ratio produced no degenerate results in this section.")

    flag = "ABOVE" if result["above_threshold"] else "below"
    print(f"  [{flag} {D2_INF_THRESHOLD_PCT}% threshold]")

    return result


def d3_cross_section(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str = "2M-1",
    label2: str = "2M-2",
) -> dict:
    """D3: Cross-section Mann-Whitney U + rank-biserial r for stain-derived features."""
    print(f"\n[D3] Cross-section distribution comparison ({label1} vs {label2})")
    results: dict = {}

    for feat in STAIN_FEATURES:
        v1 = df1[feat].astype(float).values
        v2 = df2[feat].astype(float).values
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        n1, n2 = len(v1), len(v2)

        stat, pvalue = mannwhitneyu(v1, v2, alternative="two-sided")
        # r_rb in [-1, 1]: positive = v1 > v2 more often; negative = v2 > v1
        r_rb = 2.0 * stat / (n1 * n2) - 1.0

        if abs(r_rb) >= D3_LARGE_EFFECT:
            effect_label = "large" if abs(r_rb) >= 0.5 else "medium"
        elif abs(r_rb) >= D3_SMALL_EFFECT:
            effect_label = "small"
        else:
            effect_label = "negligible"

        def _s(arr):
            return (
                float(np.mean(arr)),
                float(np.median(arr)),
                float(np.percentile(arr, 5)),
                float(np.percentile(arr, 95)),
                float(np.std(arr)),
            )

        m1, med1, p5_1, p95_1, sd1 = _s(v1)
        m2, med2, p5_2, p95_2, sd2 = _s(v2)

        print(f"\n  {feat}")
        print(f"    {label1}: n={n1:,}  mean={m1:.4f}  median={med1:.4f}  "
              f"p5={p5_1:.4f}  p95={p95_1:.4f}  std={sd1:.4f}")
        print(f"    {label2}: n={n2:,}  mean={m2:.4f}  median={med2:.4f}  "
              f"p5={p5_2:.4f}  p95={p95_2:.4f}  std={sd2:.4f}")
        print(f"    U={stat:.0f}  p={pvalue:.3e}  r_rb={r_rb:+.3f}  [{effect_label}]")

        results[feat] = {
            "n1": n1, "n2": n2,
            "mean_1": m1, "median_1": med1, "p5_1": p5_1, "p95_1": p95_1, "std_1": sd1,
            "mean_2": m2, "median_2": med2, "p5_2": p5_2, "p95_2": p95_2, "std_2": sd2,
            "mwu_U": float(stat), "mwu_p": float(pvalue),
            "r_rb": float(r_rb), "effect_label": effect_label,
        }

    return results


def d4_packing_zeros(df: pd.DataFrame, label: str) -> dict:
    """D4: packing_irregularity == 0.0 cross-checked with nuclear_density."""
    n = len(df)
    pi = df["packing_irregularity"].astype(float).values
    nd = df["nuclear_density"].astype(float).values

    pi_zero = pi == 0.0
    pi_zero_count = int(pi_zero.sum())
    pi_zero_pct = pi_zero_count / n * 100

    nd_zero_pi_zero = int(((nd == 0.0) & pi_zero).sum())
    nd_pos_pi_zero = int(((nd > 0.0) & pi_zero).sum())
    nd_floor_subset = nd[(nd > 0.0) & pi_zero]

    print(f"\n[D4 | {label}] packing_irregularity zero-count")
    print(f"  Total patches             : {n:,}")
    print(f"  packing_irregularity == 0 : {pi_zero_count:,}  ({pi_zero_pct:.2f}%)")
    if pi_zero_count > 0:
        print(f"    nuclear_density == 0.0  : {nd_zero_pi_zero:,}  "
              f"({nd_zero_pi_zero / pi_zero_count * 100:.1f}% of PI=0) — sparse, expected")
        print(f"    nuclear_density  > 0.0  : {nd_pos_pi_zero:,}  "
              f"({nd_pos_pi_zero / pi_zero_count * 100:.1f}% of PI=0) — <3-nucleus floor")
        if len(nd_floor_subset) > 0:
            print(f"    nuclear_density in <3-nucleus subset: "
                  f"min={nd_floor_subset.min():.5f}  "
                  f"median={float(np.median(nd_floor_subset)):.5f}  "
                  f"max={nd_floor_subset.max():.5f}")

    return {
        "n": n,
        "pi_zero_count": pi_zero_count,
        "pi_zero_pct": pi_zero_pct,
        "nd_zero_pi_zero": nd_zero_pi_zero,
        "nd_pos_pi_zero": nd_pos_pi_zero,
        "nd_pos_pct_of_total": nd_pos_pi_zero / n * 100,
        "suspicious_nd_min": float(nd_floor_subset.min()) if len(nd_floor_subset) > 0 else None,
        "suspicious_nd_median": float(np.median(nd_floor_subset)) if len(nd_floor_subset) > 0 else None,
        "suspicious_nd_max": float(nd_floor_subset.max()) if len(nd_floor_subset) > 0 else None,
    }


def fc_all_zero(df: pd.DataFrame, label: str) -> dict:
    """FC: Patches where all six features are simultaneously 0.0 (best failure estimate)."""
    n = len(df)
    mask = (
        (df["nuclear_density"].astype(float) == 0.0)
        & (df["mean_nuclear_area"].astype(float) == 0.0)
        & (df["nc_ratio"].astype(float) == 0.0)
        & (df["texture_entropy"].astype(float) == 0.0)
        & (df["h_intensity"].astype(float) == 0.0)
        & (df["packing_irregularity"].astype(float) == 0.0)
    )
    count = int(mask.sum())
    pct = count / n * 100

    print(f"\n[FC | {label}] All-six-features-zero (silent-failure cross-check)")
    print(f"  Patches with ALL 6 features == 0.0 : {count:,}  ({pct:.3f}%)")
    flag = "ABOVE" if pct > FC_THRESHOLD_PCT else "below"
    print(f"  [{flag} {FC_THRESHOLD_PCT}% threshold]")

    return {"n": n, "count": count, "pct": pct, "above_threshold": pct > FC_THRESHOLD_PCT}


# ── Interpretation text generators ───────────────────────────────────────────

def _interpret_d1(d1_1: dict, d1_2: dict) -> str:
    above_1, above_2 = d1_1["above_threshold"], d1_2["above_threshold"]
    if above_1 or above_2:
        which = []
        if above_1:
            which.append(f"2M-1 ({d1_1['zero_pct']:.3f}%)")
        if above_2:
            which.append(f"2M-2 ({d1_2['zero_pct']:.3f}%)")
        return (
            f"{' and '.join(which)} exceed{'s' if len(which) == 1 else ''} the "
            f"{D1_ZERO_THRESHOLD_PCT}% threshold. "
            "The failure rate is non-trivial. "
            "The current `correlations.py` code filters by `np.isfinite()` only, which does NOT "
            "exclude exact-zero texture_entropy values — these are silently included in the "
            "Spearman rank computation at the distribution floor. "
            "If these represent truly failed patches, they compress the dynamic range of "
            "texture_entropy and bias the reported rho toward zero. "
            "**Actionable:** add an explicit failure-flag column to results.csv (e.g., based on "
            "all-six-zeros, see FC below) and exclude flagged patches from all feature "
            "correlations before rerunning validation.py. "
            "The texture_entropy rho reported in the manuscript may change."
        )
    return (
        f"Both sections are below the {D1_ZERO_THRESHOLD_PCT}% audit threshold "
        f"(2M-1: {d1_1['zero_pct']:.3f}%, 2M-2: {d1_2['zero_pct']:.3f}%). "
        "The failure rate is negligible and is unlikely to materially affect the reported "
        "Spearman correlations. "
        "A single supplementary sentence suffices: 'Fewer than "
        f"{D1_ZERO_THRESHOLD_PCT}% of patches failed feature extraction and were retained "
        "with zero-initialized feature values; this does not affect the correlation analysis.' "
        "No reanalysis is required."
    )


def _interpret_d2_section(d2: dict, label: str) -> str:
    if d2["inf_count"] == 0:
        return (
            f"No inf values in {label}. "
            "All patches had at least one cytoplasm pixel and nc_ratio is finite throughout. "
            "This check can be closed for this section."
        )

    pt_shift = d2["inf_pt_median"] - d2["overall_pt_median"]
    nd_shift = d2["inf_nd_median"] - d2["overall_nd_median"]
    pt_dir = "lower" if pt_shift < 0 else "higher"
    nd_dir = "higher" if nd_shift > 0 else "lower"

    context = (
        f"The {d2['inf_count']:,} inf-nc_ratio patches ({d2['inf_pct']:.3f}%) have "
        f"median pseudotime {d2['inf_pt_median']:.3f} vs. overall {d2['overall_pt_median']:.3f} "
        f"({pt_dir} by {abs(pt_shift):.3f}), and median nuclear_density "
        f"{d2['inf_nd_median']:.5f} vs. overall {d2['overall_nd_median']:.5f} "
        f"({nd_dir} density). "
    )

    if pt_shift < -0.05 and nd_shift > 0:
        consequence = (
            "This confirms the audit's concern: inf-nc_ratio patches are concentrated in "
            "high-density, early-pseudotime regions — exactly where nc_ratio should be highest. "
            "Their correct exclusion by `np.isfinite()` removes these high-value points from "
            "the Spearman ranking, compressing the dynamic range and biasing the reported "
            "nc_ratio rho toward zero."
        )
    elif pt_shift > 0.05:
        consequence = (
            "The inf-nc_ratio patches are concentrated at higher pseudotime than average, "
            "not at the high-density pole. "
            "Their exclusion is unlikely to systematically bias the reported rho."
        )
    else:
        consequence = (
            "The inf-nc_ratio patches are not strongly concentrated in high-density or "
            "early-pseudotime regions, moderating concern about directional bias in the "
            "nc_ratio correlation."
        )

    severity = (
        f"**Above the {D2_INF_THRESHOLD_PCT}% threshold** — worth a limitations note. "
        if d2["above_threshold"]
        else f"Below the {D2_INF_THRESHOLD_PCT}% threshold — a supplementary note suffices. "
    )
    return severity + context + consequence


def _interpret_d3(d3: dict) -> str:
    hi = d3["h_intensity"]
    nd = d3["nuclear_density"]
    mna = d3["mean_nuclear_area"]
    hi_r = hi["r_rb"]
    nd_r = nd["r_rb"]

    if abs(hi_r) >= D3_LARGE_EFFECT:
        if abs(nd_r) < 0.2:
            verdict = (
                f"**Supports the stain-matrix concern.** "
                f"h_intensity shows a {hi['effect_label']} inter-section shift "
                f"(rank-biserial r = {hi_r:+.3f}) while nuclear_density shifts only "
                f"{nd['effect_label']}ly (r = {nd_r:+.3f}). "
                "A large h_intensity difference without a correspondingly large density "
                "difference is not easily explained by biology alone, and is consistent "
                "with the fixed Ruifrok–Johnston stain matrix responding differently to the "
                "two sections' different staining reagents. "
                "**Manuscript note required:** cross-section h_intensity comparisons are "
                "partially confounded by stain chemistry; within-section pseudotime "
                "correlations are less affected because the stain matrix is consistent "
                "within a single section."
            )
        elif abs(nd_r) >= D3_LARGE_EFFECT:
            verdict = (
                f"**Ambiguous — stain matrix or biology.** "
                f"Both h_intensity (r = {hi_r:+.3f}, {hi['effect_label']}) and "
                f"nuclear_density (r = {nd_r:+.3f}, {nd['effect_label']}) shift "
                "substantially between sections. "
                "Since h_intensity is computed as a whole-patch mean (not nuclear-masked), "
                "it mechanically increases with density; the h_intensity shift may "
                "largely reflect the density difference rather than stain chemistry. "
                "Disentangling the two requires restricting h_intensity to nuclear pixels "
                "only (audit follow-up experiment E2). "
                "Acknowledge both potential confounds in the manuscript."
            )
        else:
            verdict = (
                f"**Moderately supports the stain-matrix concern.** "
                f"h_intensity shifts {hi['effect_label']}ly (r = {hi_r:+.3f}); "
                f"nuclear_density shifts {nd['effect_label']}ly (r = {nd_r:+.3f}). "
                "Some of the h_intensity difference is attributable to density differences "
                "via the whole-patch mean confound, but the stain-matrix concern cannot "
                "be dismissed. "
                "Interpret cross-section h_intensity comparisons with caution."
            )
    elif abs(hi_r) >= D3_SMALL_EFFECT:
        verdict = (
            f"**Weakly supports the stain-matrix concern** "
            f"(h_intensity rank-biserial r = {hi_r:+.3f}, {hi['effect_label']}). "
            "The inter-section shift is statistically detectable at this sample size "
            "but the effect size is small, suggesting modest bias from the fixed stain matrix. "
            "A supplementary methods note is sufficient; no manuscript correction is warranted."
        )
    else:
        verdict = (
            f"**Does not confirm the stain-matrix concern** for h_intensity "
            f"(rank-biserial r = {hi_r:+.3f}, {hi['effect_label']} effect). "
            "The h_intensity distributions are similar between sections, suggesting the fixed "
            "Ruifrok–Johnston matrix does not introduce a large systematic bias at 5× "
            "magnification. "
            "The audit concern is not confirmed by this data, though it cannot be fully "
            "ruled out without per-slide stain calibration."
        )

    mna_r = mna["r_rb"]
    mna_note = (
        f" Mean nuclear area shows a {mna['effect_label']} inter-section shift "
        f"(r = {mna_r:+.3f}); this is independent of stain chemistry (area is computed from "
        "the labeled mask, not pixel intensities) and reflects genuine morphological "
        "differences between sections if non-negligible."
    )
    return verdict + mna_note


def _interpret_d4_section(d4: dict, label: str) -> str:
    if d4["pi_zero_count"] == 0:
        return f"No zero packing_irregularity patches in {label}. No floor effect."

    nd_pos_pct = d4["nd_pos_pct_of_total"]
    nd_pos = d4["nd_pos_pi_zero"]
    total_pi = d4["pi_zero_count"]
    nd_pos_of_pi_pct = nd_pos / total_pi * 100 if total_pi > 0 else 0.0

    base = (
        f"Section {label}: {total_pi:,} patches ({d4['pi_zero_pct']:.2f}%) have "
        f"packing_irregularity = 0.0. "
        f"Of these, {nd_pos:,} ({nd_pos_of_pi_pct:.1f}%) have nuclear_density > 0, "
        f"meaning 1–2 nuclei were segmented but the <3-nucleus threshold fired. "
        f"These represent {nd_pos_pct:.2f}% of all patches in this section. "
    )

    if nd_pos_pct > 2.0:
        action = (
            "**Non-trivial fraction.** packing_irregularity = 0 is not biologically "
            "meaningful for these patches (it reflects the threshold, not genuine uniform "
            "packing). This structural floor attenuates the packing_irregularity–pseudotime "
            "correlation and should be disclosed in methods. "
            "The cellularity confound analysis (partial Spearman controlling for "
            "nuclear_density) partially absorbs this covariance, but the mechanism "
            "should be made explicit."
        )
    elif nd_pos_pct > 0.5:
        action = (
            "Modest fraction — the structural floor is worth a methods note but is "
            "unlikely to substantially alter the reported rho or confound conclusion."
        )
    else:
        action = (
            "Negligible fraction — the <3-nucleus floor does not materially affect "
            "the packing_irregularity correlation or confound analysis."
        )

    return base + action


def _interpret_fc(fc_1: dict, fc_2: dict) -> str:
    if fc_1["above_threshold"] or fc_2["above_threshold"]:
        which = []
        if fc_1["above_threshold"]:
            which.append(f"2M-1 ({fc_1['pct']:.3f}%)")
        if fc_2["above_threshold"]:
            which.append(f"2M-2 ({fc_2['pct']:.3f}%)")
        return (
            f"At least one section exceeds the {FC_THRESHOLD_PCT}% threshold "
            f"({', '.join(which)}). "
            "The all-six-zeros count is a conservative indicator of true silent exceptions — "
            "it requires every feature to be exactly 0.0 simultaneously, which is extremely "
            "unlikely for real tissue. "
            "**Actionable:** add an explicit failure-flag column to results.csv and exclude "
            "flagged patches from all six feature correlations before rerunning validation.py. "
            "The reported correlation coefficients for multiple features are likely affected."
        )
    return (
        f"Both sections are below the {FC_THRESHOLD_PCT}% threshold "
        f"(2M-1: {fc_1['pct']:.3f}%, 2M-2: {fc_2['pct']:.3f}%). "
        "The true failure rate lies between this count (lower bound) and the D1 "
        "texture_entropy zero count (upper bound). "
        "Neither bound exceeds the threshold, confirming the failure rate is negligible. "
        "No manuscript correction is required for failure handling."
    )


def _overall_verdict(r: dict) -> str:
    d1_any = r["d1_2m1"]["above_threshold"] or r["d1_2m2"]["above_threshold"]
    fc_any = r["fc_2m1"]["above_threshold"] or r["fc_2m2"]["above_threshold"]
    d2_any = r["d2_2m1"]["above_threshold"] or r["d2_2m2"]["above_threshold"]
    hi_r = r["d3"]["h_intensity"]["r_rb"]
    nd_r = r["d3"]["nuclear_density"]["r_rb"]
    hi_effect_large = abs(hi_r) >= D3_LARGE_EFFECT
    d4_1_nontrivial = r["d4_2m1"]["nd_pos_pct_of_total"] > 2.0
    d4_2_nontrivial = r["d4_2m2"]["nd_pos_pct_of_total"] > 2.0

    flags = []
    if fc_any or d1_any:
        flags.append(
            "**Silent failure rate is non-trivial** — add failure flag to results.csv "
            "and rerun validation.py before manuscript submission (see D1 and FC)"
        )
    if hi_effect_large and abs(nd_r) < D3_LARGE_EFFECT:
        flags.append(
            "**h_intensity cross-section shift is large without a correspondingly large "
            "density shift** — stain-matrix limitation must be disclosed in methods (see D3)"
        )
    elif hi_effect_large:
        flags.append(
            "**h_intensity and nuclear_density both shift between sections** — "
            "stain matrix vs. biology is ambiguous; acknowledge both confounds (see D3)"
        )
    if d2_any:
        flags.append(
            "**nc_ratio inf count is above threshold** — check for systematic "
            "exclusion of high-density patches; add to limitations (see D2)"
        )
    if d4_1_nontrivial or d4_2_nontrivial:
        flags.append(
            "**packing_irregularity <3-nucleus floor affects >2% of patches** in at "
            "least one section — structural confound mechanism should be disclosed (see D4)"
        )

    if not flags:
        return (
            "No audit flag exceeded its threshold. The reported morphological feature "
            "correlations appear methodologically sound given the pipeline design choices. "
            "The audit findings (fixed stain matrix, whole-patch h_intensity, single-angle "
            "GLCM) remain as **limitations to disclose** in the methods section, but the "
            "quantitative results do not require reanalysis. "
            "Proceed to manuscript submission with the limitation notes identified in "
            "`reports/morphological_features_audit.md`."
        )

    bullet_list = "\n".join(f"- {f}" for f in flags)
    return (
        "**One or more audit flags exceeded their threshold. "
        "Address the following before manuscript submission:**\n\n"
        + bullet_list
        + "\n\nFor findings not listed above, the pipeline choices fall within acceptable "
        "methodological limits and require only supplementary disclosure."
    )


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(
    r: dict,
    path: Path,
    sec1_path: Path,
    sec2_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    d1_1, d1_2 = r["d1_2m1"], r["d1_2m2"]
    d2_1, d2_2 = r["d2_2m1"], r["d2_2m2"]
    d3 = r["d3"]
    d4_1, d4_2 = r["d4_2m1"], r["d4_2m2"]
    fc_1, fc_2 = r["fc_2m1"], r["fc_2m2"]

    def _flag(above: bool) -> str:
        return "⚠ above threshold" if above else "✓ within threshold"

    def _d2_pt(d2) -> str:
        return "N/A" if d2["inf_count"] == 0 else f"{d2['inf_pt_median']:.3f}"

    def _d2_nd(d2) -> str:
        return "N/A" if d2["inf_count"] == 0 else f"{d2['inf_nd_median']:.5f}"

    def _nd_floor_median(d4) -> str:
        v = d4.get("suspicious_nd_median")
        return "N/A" if v is None else f"{v:.5f}"

    lines = [
        "# Morphological Feature Diagnostics Results",
        "",
        f"**Generated:** {today}  ",
        "**Script:** `diagnostics/audit_feature_diagnostics.py`  ",
        f"**Section 2M-1 input:** `{sec1_path}`  ",
        f"**Section 2M-2 input:** `{sec2_path}`  ",
        "",
        "> This file is generated automatically by the script above.",
        "> Do not edit by hand — rerun `jobs/run_feature_diagnostics.sh` to update.",
        "",
        "---",
        "",
        "## D1: Failure-Zeroed Patches (texture_entropy == 0.0)",
        "",
        "A genuine tissue patch cannot have zero GLCM entropy. Exact-zero values are a reliable "
        "proxy for patches where `compute_morphological_features` caught an exception (after "
        "index 4) and left all features at the initialised 0.0. The zero count is an upper bound "
        "on true failures; the FC all-six-zeros count is the lower bound.",
        "",
        "| Section | N patches | zero count | zero % | NaN count | Assessment |",
        "|---------|-----------|------------|--------|-----------|------------|",
        (
            f"| 2M-1 | {d1_1['n']:,} | {d1_1['zero_count']:,} | {d1_1['zero_pct']:.3f}% | "
            f"{d1_1['nan_count']:,} | {_flag(d1_1['above_threshold'])} |"
        ),
        (
            f"| 2M-2 | {d1_2['n']:,} | {d1_2['zero_count']:,} | {d1_2['zero_pct']:.3f}% | "
            f"{d1_2['nan_count']:,} | {_flag(d1_2['above_threshold'])} |"
        ),
        "",
        "### Interpretation",
        "",
        _interpret_d1(d1_1, d1_2),
        "",
        "---",
        "",
        "## D2: nc_ratio Infinite Values",
        "",
        "`compute_nc_ratio` returns `float('inf')` when cytoplasm_pixels == 0. These patches "
        "are correctly excluded by `np.isfinite()` in `correlations.py` and "
        "`cellularity_confound.py`. The question is whether their exclusion is systematically "
        "concentrated in high-density, early-pseudotime regions, which would bias the reported "
        "nc_ratio rho toward zero.",
        "",
        "| Section | N | inf count | inf % | NaN count | pt median (inf subset) | nd median (inf subset) | Assessment |",
        "|---------|---|-----------|-------|-----------|------------------------|------------------------|------------|",
        (
            f"| 2M-1 | {d2_1['n']:,} | {d2_1['inf_count']:,} | {d2_1['inf_pct']:.3f}% | "
            f"{d2_1['nan_count']:,} | {_d2_pt(d2_1)} | {_d2_nd(d2_1)} | "
            f"{_flag(d2_1['above_threshold'])} |"
        ),
        (
            f"| 2M-2 | {d2_2['n']:,} | {d2_2['inf_count']:,} | {d2_2['inf_pct']:.3f}% | "
            f"{d2_2['nan_count']:,} | {_d2_pt(d2_2)} | {_d2_nd(d2_2)} | "
            f"{_flag(d2_2['above_threshold'])} |"
        ),
        "",
        (
            f"Overall medians for reference — "
            f"2M-1: pt={d2_1['overall_pt_median']:.3f}, nd={d2_1['overall_nd_median']:.5f};  "
            f"2M-2: pt={d2_2['overall_pt_median']:.3f}, nd={d2_2['overall_nd_median']:.5f}"
        ),
        "",
        "### Section 2M-1",
        "",
        _interpret_d2_section(d2_1, "2M-1"),
        "",
        "### Section 2M-2",
        "",
        _interpret_d2_section(d2_2, "2M-2"),
        "",
        "---",
        "",
        "## D3: Cross-Section Distribution Comparison",
        "",
        "If the fixed Ruifrok–Johnston stain matrix responds differently to the two sections' "
        "different staining reagents, h_intensity should shift substantially between sections "
        "independent of biology. nuclear_density and mean_nuclear_area serve as controls "
        "(stain-matrix independent). "
        "At ~10 k patches Mann-Whitney U always reaches p < 0.001; the interpretable metric is "
        "the rank-biserial correlation r (effect size, range [−1, 1]); |r| ≥ 0.3 is the "
        "threshold for a medium/large effect.",
        "",
        "### Section 2M-1 summary statistics",
        "",
        "| Feature | n | mean | median | p5 | p95 | std |",
        "|---------|---|------|--------|----|-----|-----|",
    ]

    for feat in STAIN_FEATURES:
        v = d3[feat]
        lines.append(
            f"| {feat} | {v['n1']:,} | {v['mean_1']:.4f} | {v['median_1']:.4f} | "
            f"{v['p5_1']:.4f} | {v['p95_1']:.4f} | {v['std_1']:.4f} |"
        )

    lines += [
        "",
        "### Section 2M-2 summary statistics",
        "",
        "| Feature | n | mean | median | p5 | p95 | std |",
        "|---------|---|------|--------|----|-----|-----|",
    ]

    for feat in STAIN_FEATURES:
        v = d3[feat]
        lines.append(
            f"| {feat} | {v['n2']:,} | {v['mean_2']:.4f} | {v['median_2']:.4f} | "
            f"{v['p5_2']:.4f} | {v['p95_2']:.4f} | {v['std_2']:.4f} |"
        )

    lines += [
        "",
        "### Cross-section comparison",
        "",
        "| Feature | U | p-value | rank-biserial r | effect size |",
        "|---------|---|---------|-----------------|-------------|",
    ]

    for feat in STAIN_FEATURES:
        v = d3[feat]
        lines.append(
            f"| {feat} | {v['mwu_U']:.0f} | {v['mwu_p']:.3e} | "
            f"{v['r_rb']:+.3f} | {v['effect_label']} |"
        )

    lines += [
        "",
        "### Interpretation",
        "",
        _interpret_d3(d3),
        "",
        "---",
        "",
        "## D4: packing_irregularity Zero-Count",
        "",
        "`compute_packing_irregularity` returns 0.0 for patches with fewer than 3 segmented "
        "nuclei. Patches with nuclear_density == 0 and packing_irregularity == 0 are expected "
        "(truly sparse). Patches with nuclear_density > 0 and packing_irregularity == 0 "
        "represent exactly 1–2 nuclei: their packing_irregularity is not biologically "
        "meaningful (the threshold fired, not genuine uniform packing), creating a structural "
        "confound with nuclear_density.",
        "",
        "| Section | N | PI=0 count | PI=0 % | nd=0 & PI=0 | nd>0 & PI=0 (<3 floor) | nd median in floor subset |",
        "|---------|---|------------|--------|-------------|------------------------|---------------------------|",
        (
            f"| 2M-1 | {d4_1['n']:,} | {d4_1['pi_zero_count']:,} | {d4_1['pi_zero_pct']:.2f}% | "
            f"{d4_1['nd_zero_pi_zero']:,} | {d4_1['nd_pos_pi_zero']:,} | {_nd_floor_median(d4_1)} |"
        ),
        (
            f"| 2M-2 | {d4_2['n']:,} | {d4_2['pi_zero_count']:,} | {d4_2['pi_zero_pct']:.2f}% | "
            f"{d4_2['nd_zero_pi_zero']:,} | {d4_2['nd_pos_pi_zero']:,} | {_nd_floor_median(d4_2)} |"
        ),
        "",
        "### Section 2M-1",
        "",
        _interpret_d4_section(d4_1, "2M-1"),
        "",
        "### Section 2M-2",
        "",
        _interpret_d4_section(d4_2, "2M-2"),
        "",
        "---",
        "",
        "## FC: All-Six-Features-Zero (Silent Failure Cross-Check)",
        "",
        "A patch where all six features are simultaneously 0.0 almost certainly represents a "
        "silent exception — real tissue cannot have zero hematoxylin intensity, zero entropy, "
        "and zero density simultaneously. This count is the lower bound on true failures; "
        "the D1 texture_entropy zero count is the upper bound.",
        "",
        "| Section | N | all-zero count | all-zero % | Assessment |",
        "|---------|---|----------------|------------|------------|",
        (
            f"| 2M-1 | {fc_1['n']:,} | {fc_1['count']:,} | {fc_1['pct']:.3f}% | "
            f"{_flag(fc_1['above_threshold'])} |"
        ),
        (
            f"| 2M-2 | {fc_2['n']:,} | {fc_2['count']:,} | {fc_2['pct']:.3f}% | "
            f"{_flag(fc_2['above_threshold'])} |"
        ),
        "",
        "### Interpretation",
        "",
        _interpret_fc(fc_1, fc_2),
        "",
        "---",
        "",
        "## Overall Verdict",
        "",
        _overall_verdict(r),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to:\n    {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only morphological feature diagnostics for the cancer trajectory atlas. "
            "Reads existing per-section results.csv files; does not modify any pipeline output."
        )
    )
    parser.add_argument(
        "--section1-results",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to atlas_2M-1/results.csv on Narval",
    )
    parser.add_argument(
        "--section2-results",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to atlas_2M-2/results.csv on Narval",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/morphological_features_diagnostics_results.md"),
        metavar="PATH",
        help="Destination markdown report (default: reports/morphological_features_diagnostics_results.md)",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  Cancer Trajectory Atlas — Morphological Feature Diagnostics")
    print("=" * 68)
    print(f"\nSection 2M-1 : {args.section1_results}")
    print(f"Section 2M-2 : {args.section2_results}")
    print(f"Report path  : {args.output_report}")

    df1 = load_csv(args.section1_results, "2M-1")
    df2 = load_csv(args.section2_results, "2M-2")

    r: dict = {}

    print("\n" + "=" * 68)
    print("  D1: Failure-zeroed patches")
    print("=" * 68)
    r["d1_2m1"] = d1_failure_zeros(df1, "2M-1")
    r["d1_2m2"] = d1_failure_zeros(df2, "2M-2")

    print("\n" + "=" * 68)
    print("  D2: nc_ratio infinite values")
    print("=" * 68)
    r["d2_2m1"] = d2_nc_ratio_inf(df1, "2M-1")
    r["d2_2m2"] = d2_nc_ratio_inf(df2, "2M-2")

    print("\n" + "=" * 68)
    print("  D3: Cross-section distribution comparison")
    print("=" * 68)
    r["d3"] = d3_cross_section(df1, df2)

    print("\n" + "=" * 68)
    print("  D4: packing_irregularity zero-count")
    print("=" * 68)
    r["d4_2m1"] = d4_packing_zeros(df1, "2M-1")
    r["d4_2m2"] = d4_packing_zeros(df2, "2M-2")

    print("\n" + "=" * 68)
    print("  FC: All-six-features-zero cross-check")
    print("=" * 68)
    r["fc_2m1"] = fc_all_zero(df1, "2M-1")
    r["fc_2m2"] = fc_all_zero(df2, "2M-2")

    print("\n" + "=" * 68)
    print("  Writing report")
    print("=" * 68)
    write_report(r, args.output_report, args.section1_results, args.section2_results)

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
