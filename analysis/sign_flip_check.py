"""Task 3 — are the two sections' axes opposed, or merely oriented oppositely?

Every feature directional in both sections carries the opposite sign
(nuclear_density +0.445 vs -0.150; nc_ratio +0.328 vs -0.401;
packing_irregularity -0.222 vs +0.168). Root choice determines which end of the
axis is labelled zero, and root_sensitivity showed the roots do NOT determine the
ordering (random 20-root sets reproduce it at |rho| 0.78-0.89) — only the
orientation. So the two axes may be the same axis read in opposite directions.

INTERPRETIVE CONSTRAINT — enforced in the output, not just documented
    Negating 2M-2's pseudotime flips every one of its correlations' signs. That
    is ARITHMETIC, NOT EVIDENCE, and sign agreement after a flip is therefore
    vacuous on its own. Two things are informative:

      (i) whether flipped 2M-2 agrees with 2M-1 in sign AND approximate
          MAGNITUDE, reported per feature as an absolute difference;
      (ii) whether features non-directional in one section (h_intensity in 2M-2;
           texture_entropy and mean_nuclear_area in 2M-1) STAY non-directional —
           a flip cannot manufacture or destroy a relationship, so a feature that
           is strong in one section and absent in the other is evidence AGAINST
           a shared axis no matter which way either is pointed.

Read-only: consumes adata_full.h5ad. Writes a new directory only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

FEATURES = ["nuclear_density", "mean_nuclear_area", "nc_ratio",
            "texture_entropy", "h_intensity", "packing_irregularity"]

DIRECTIONAL_MIN = 0.15   # |rho| to call a feature directional (matches eccentricity_check)
AGREE_ABS_DIFF = 0.10    # |rho_A - rho_B_flipped| under which magnitudes "agree"


def _rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def section_rhos(run_dir: Path, flip: bool) -> tuple[dict, int]:
    import anndata as ad
    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(f"{h5ad} not found — this check reuses an existing run.")
    adata = ad.read_h5ad(h5ad)
    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)
    if flip:
        # 1 - pt preserves [0,1] and reverses order. Spearman is invariant to any
        # strictly decreasing transform, so this negates every rho exactly.
        pt = 1.0 - pt
    out = {}
    for f in FEATURES:
        out[f] = _rho(pt, obs[f].values.astype(float)) if f in obs.columns else float("nan")
    return out, int(adata.n_obs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir-a", type=Path, required=True, help="Reference section (2M-1)")
    ap.add_argument("--run-dir-b", type=Path, required=True, help="Section to flip (2M-2)")
    ap.add_argument("--label-a", default="2M-1")
    ap.add_argument("--label-b", default="2M-2")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Sign-flip check — opposed biology, or opposite orientation?")
    print("=" * 64)

    a, n_a = section_rhos(args.run_dir_a, flip=False)
    b, n_b = section_rhos(args.run_dir_b, flip=False)
    bf, _ = section_rhos(args.run_dir_b, flip=True)

    # Sanity: the flip must negate exactly. If it does not, something other than
    # a monotone reversal happened and the whole comparison is invalid.
    max_dev = max((abs(bf[f] + b[f]) for f in FEATURES
                   if np.isfinite(b[f]) and np.isfinite(bf[f])), default=0.0)

    rows, agree, disagree, mismatched_directionality = [], [], [], []
    for f in FEATURES:
        ra, rb, rbf = a[f], b[f], bf[f]
        dif = abs(ra - rbf) if np.isfinite(ra) and np.isfinite(rbf) else float("nan")
        dir_a = np.isfinite(ra) and abs(ra) >= DIRECTIONAL_MIN
        dir_b = np.isfinite(rb) and abs(rb) >= DIRECTIONAL_MIN
        same_sign = np.isfinite(ra) and np.isfinite(rbf) and np.sign(ra) == np.sign(rbf)
        mag_agrees = np.isfinite(dif) and dif <= AGREE_ABS_DIFF

        if dir_a != dir_b:
            status = ("DIRECTIONALITY MISMATCH — directional in "
                      f"{args.label_a if dir_a else args.label_b} only; a flip cannot "
                      "create or remove a relationship, so this is evidence against a "
                      "shared axis regardless of orientation")
            mismatched_directionality.append(f)
        elif not (dir_a or dir_b):
            status = "not directional in either section — uninformative"
        elif same_sign and mag_agrees:
            status = f"AGREES after flip (|diff| {dif:.3f} <= {AGREE_ABS_DIFF})"
            agree.append(f)
        elif same_sign:
            status = (f"sign agrees but MAGNITUDE DIFFERS (|diff| {dif:.3f} > "
                      f"{AGREE_ABS_DIFF}) — sign alone is arithmetic, not evidence")
            disagree.append(f)
        else:
            status = "still disagrees after flip"
            disagree.append(f)

        rows.append({
            "feature": f, f"rho_{args.label_a}": ra, f"rho_{args.label_b}_asis": rb,
            f"rho_{args.label_b}_flipped": rbf, "abs_diff_a_vs_b_flipped": dif,
            f"directional_{args.label_a}": bool(dir_a),
            f"directional_{args.label_b}": bool(dir_b),
            "sign_agrees_after_flip": bool(same_sign),
            "magnitude_agrees": bool(mag_agrees), "status": status,
        })

    n_testable = len(agree) + len(disagree)
    if mismatched_directionality:
        verdict = (
            f"NOT A SHARED AXIS — {len(mismatched_directionality)} feature(s) "
            f"({', '.join(mismatched_directionality)}) are directional in one section "
            "and not the other. Reorientation cannot create or destroy a relationship, "
            "so this difference survives any flip. Of the features testable on both, "
            f"{len(agree)}/{n_testable} agree in sign AND magnitude after flipping."
        )
    elif n_testable and len(agree) == n_testable:
        verdict = (
            f"CONSISTENT WITH ONE AXIS READ IN OPPOSITE DIRECTIONS — all {n_testable} "
            f"testable features agree in sign and to within {AGREE_ABS_DIFF} in "
            "magnitude after flipping, and no feature changes directionality status. "
            "Note this does NOT establish a shared trajectory: it establishes that the "
            "opposite signs are explained by orientation rather than by opposing "
            "morphology. The orientation itself is set by the root rule."
        )
    else:
        verdict = (
            f"PARTIAL — {len(agree)}/{n_testable} testable features agree in sign and "
            f"magnitude after flipping; {len(disagree)} do not "
            f"({', '.join(disagree) or 'none'}). Orientation explains some but not all "
            "of the cross-section disagreement."
        )

    payload = {
        "analysis": "sign_flip_check",
        "labels": {"a": args.label_a, "b": args.label_b},
        "n_patches": {args.label_a: n_a, args.label_b: n_b},
        "thresholds": {"directional_min_abs_rho": DIRECTIONAL_MIN,
                       "magnitude_agreement_max_abs_diff": AGREE_ABS_DIFF},
        "flip_definition": "pseudotime -> 1 - pseudotime (preserves [0,1], reverses order)",
        "flip_negation_max_deviation": max_dev,
        "interpretive_constraint": (
            "Flipping NECESSARILY negates every correlation. Sign agreement after a "
            "flip is arithmetic, not evidence, and is never reported here as a "
            "finding on its own. The informative quantities are (i) agreement in "
            "MAGNITUDE, given as abs_diff_a_vs_b_flipped, and (ii) whether a "
            "feature's directional/non-directional status matches across sections, "
            "which no reorientation can change."
        ),
        "per_feature": rows,
        "summary": {
            "agree_sign_and_magnitude": agree,
            "disagree": disagree,
            "directionality_mismatch": mismatched_directionality,
        },
        "verdict": verdict,
    }

    with open(args.output_dir / "sign_flip_check.json", "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: None if isinstance(o, float) else str(o))

    L = ["# Sign-flip check", "",
         "**Flipping necessarily negates every correlation. That is arithmetic, not",
         "evidence.** Sign agreement after a flip is not reported as a finding. What",
         "matters is magnitude agreement, and whether a feature is directional in both",
         "sections — no reorientation can change that.", "",
         f"Flip: `pseudotime -> 1 - pseudotime` on {args.label_b}. Exactness of the "
         f"negation: max |rho_flipped + rho_asis| = {max_dev:.2e}.", "",
         f"| Feature | {args.label_a} | {args.label_b} as-is | {args.label_b} flipped "
         f"| \\|diff\\| | Status |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| `{r['feature']}` | {r[f'rho_{args.label_a}']:+.3f} | "
            f"{r[f'rho_{args.label_b}_asis']:+.3f} | "
            f"{r[f'rho_{args.label_b}_flipped']:+.3f} | "
            f"{r['abs_diff_a_vs_b_flipped']:.3f} | {r['status']} |")
    L += ["", f"**Verdict.** {verdict}", ""]
    (args.output_dir / "sign_flip_report.md").write_text("\n".join(L), encoding="utf-8")

    # Figure: A vs B-flipped, with the identity line magnitude agreement implies.
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    xs = [r[f"rho_{args.label_a}"] for r in rows]
    ys = [r[f"rho_{args.label_b}_flipped"] for r in rows]
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
    lim = 0.6
    ax.plot([-lim, lim], [-lim, lim], color="0.6", ls="--", lw=1,
            label="identity (magnitudes agree)")
    ax.scatter(xs, ys, s=45, color="#4878CF", zorder=3)
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(r["feature"].replace("_", "\n"), (x, y), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"rho, {args.label_a}")
    ax.set_ylabel(f"rho, {args.label_b} FLIPPED")
    ax.set_title("Points off the dashed line disagree in magnitude\n"
                 "even though every sign was forced to match", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"sign_flip_scatter.{ext}", dpi=170)
    plt.close(fig)

    print(f"\n  JSON:     {args.output_dir / 'sign_flip_check.json'}")
    print(f"  Markdown: {args.output_dir / 'sign_flip_report.md'}")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
