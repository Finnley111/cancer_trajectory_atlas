# KNOWN_ISSUES.md — defects and limitations, with scope

**Current as of 2026-08-23.** Each entry states what the issue is, why it matters, how it was
found, **whether it affects a reported number**, and what fixing it would require.

**Nothing here is fixed.** These are recorded so the scope of each correction stays visible and
so the next person does not rediscover them. Where an issue has been *measured* rather than
merely suspected, the measurement is given.

**Companion documents.** `docs/ANCHOR_VALIDATION_RECORD.md` §4 is the error log — mistakes made
and how each was caught. §5 there lists statistics that should not be trusted. This file is
about defects that are still live in the code and the design.

---

## 1. Statistical mis-specification

### 1.1 The 16 slides are 8 matched pairs — CORRECTED for one analysis, still live for others

**What.** Every mouse-flank gland contributes one slide to 2M-1 (Carnoy's) and one to 2M-2
(PFA). The 16 slides are therefore 8 matched pairs, not 16 independent samples. This was
overlooked in every analysis before 2026-08-23.

**Why it matters.** Any between-section test that permutes the 16 slides freely admits
between-gland and between-mouse variation that the paired design already controls. The null is
wider than it should be and the minimum detectable difference is inflated.

**Found by.** The collaborating pathologist's methods text, then verified empirically from the
per-duct tables (`analysis/gland_pairing_audit.py`) rather than from filenames.

**Affects reported numbers.** Yes, for the between-section comparison — now corrected in
`analysis/holeyness_paired_comparison.py` and reported side by side with the unpaired version.
See `ANCHOR_VALIDATION_RECORD.md` §3.11.

**Still live for.** Nothing else on the between-section path, but see 1.2 and 1.3 below, which
are separate defects surfaced by the same audit.

**Not affected, and worth stating.** The within-section slide-clustered bootstraps in
`holeyness_section_comparison.py:224`, `holeyroot_duct_checks.py:192` and
`duct_white_fraction.py:259` are **unaffected**: within one section no two slides share a gland,
so the slide *is* the independent unit there.

### 1.2 Patch-level global shuffles ignore slide nesting

**What.** Two permutation tests shuffle at the level of individual patches, with no slide
structure preserved:

- `validation/correlations.py:50` `permutation_test` — `rng.permutation(pseudotime)` over all
  patches
- `analysis/cellularity_confound.py:267` — `rng.permutation(pt)`, same pattern

**Why it matters.** Patches are nested within slides. Treating ~8,000–10,000 nested patches as
exchangeable makes these tests **anti-conservative** — the null is too narrow and p-values are
too small. This is the opposite direction from 1.1.

**Found by.** The Task 1(d) repo audit, 2026-08-23.

**Affects reported numbers.** Yes — every permutation p-value in `validation.json` and in the
cellularity-confound output. Note the practical impact may be limited because these p-values are
already reported as `< 1/n_permutations` in most runs; the concern is the *nominal* significance
level, not the ranking of features.

**These are pre-existing and unrelated to the pairing.** They are within-section tests, so gland
pairing does not enter. They were wrong before the pairing was discovered.

**Fixing would require** a within-slide shuffle, as already implemented in
`analysis/holeyness.py` and `analysis/holeyness_asymmetry.py`. Both of those get it right; these
two do not.

### 1.3 Leave-one-out leakage across the matched pair

**What.** `data/loo_slides.txt` holds all **16** slides and `jobs/submit_loo_array.sh` holds out
one slide at a time. Under the paired design, holding out `6027-4L-2M-1` leaves
**`6027-4L-2M-2` — the same gland — in the training set**.

**Why it matters.** The held-out slide's matched partner trains the model it is then projected
onto. That is not a clean held-out test, and the resulting projection accuracy is
**optimistically biased**.

**Found by.** The Task 1(d) repo audit, 2026-08-23.

**Affects reported numbers.** Yes, for any full-atlas 16-slide LOO result.

**The within-section LOO is fine.** `jobs/run_per_section.sh` runs LOO within a section, where
the 8 slides are 8 distinct glands, so no partner is present. Only the cross-section array leaks.

**Fixing would require** holding out the *gland* — both slides — rather than the slide, which
reduces the fold count from 16 to 8.

---

## 2. Measurement and design

### 2.1 `hole_pct` is not fixation-invariant — MEASURED

> **The underlying measurement is a RESULT, not a limitation.** Carnoy's deforms ductal
> architecture anisotropically — the lumen collapses to roughly a quarter of what the duct does
> — and that belongs in the manuscript as a finding. See `ANCHOR_VALIDATION_RECORD.md` §3.12.
> What is recorded *here* is only the consequence it has for cross-section comparability.

**What.** On matched tissue, the lumen collapses far harder than the duct: hole area shrinks
**5.48×** under Carnoy's while duct area shrinks only **1.64×**. Anisotropy (hole-area ratio
over duct-area ratio) is **0.261 median, 8/8 glands, exact sign-test p = 0.0078**.

**Why it matters.** `hole_pct` measured under Carnoy's and `hole_pct` measured under PFA **are
not the same quantity**. Any cross-section comparison of it compares two different measurements.

**Scope, precisely.** The validation **within** each section is unaffected — `hole_pct` still
ranks ducts correctly there, and `duct_white_fraction` confirmed it tracks measured white space
at rho 0.92 / 0.79. **Only cross-section replication claims are affected**, and they are not
invalidated: Spearman is invariant to monotone rescaling, so the replication holds. It should be
stated as *the correlation replicates despite `hole_pct` being systematically rescaled between
conditions*, which arguably strengthens it.

**Found by.** `analysis/holeyness_paired_comparison.py`, 2026-08-23. See
`ANCHOR_VALIDATION_RECORD.md` §3.12.

**Fixing would require** either a fixation-invariant holeyness measure, or restricting
replication claims to within-section statements.

### 2.2 Fixation is perfectly collinear with section

**What.** Every Carnoy's slide is 2M-1 and every PFA slide is 2M-2.

**Why it matters.** No analysis of this cohort can attribute any difference — or any absence of
one — to fixation chemistry as opposed to anatomical region. The shrinkage measurement in 2.1
is a *fixation-or-region* effect; the chemistry is the plausible mechanism, not the demonstrated
one.

**Fixing would require** bridge samples: serial sections from one block, split across both
fixations, stained in one run. **Time-sensitive** — tissue blocks degrade and are consumed.

### 2.3 The duct-validation estimand excludes ~26% of ducts

**What.** Patch-centre assignment (`analysis/holeyness.py:286`) leaves 571/2173 ducts (2M-1) and
389/1749 (2M-2) with no assigned patch, systematically the smallest and least holey.

**Why it matters.** Every duct-level correlation describes the *retained* population, not all
annotated ducts.

**Partly addressed.** `duct_white_fraction` measured the zero-patch ducts directly and found
they correlate just as well (+0.9078 vs +0.9157 in 2M-1), so the exclusion does not bias *that*
relationship. It remains an estimand statement that should be made explicitly.

### 2.4 Eight pairs is the ceiling on everything

**What.** 16 slides, 8 glands, 4 mice, one timepoint.

**Why it matters.** The paired design's p-value floor is **2/256 = 0.0078**; minimum detectable
differences run **0.13–0.42** depending on estimand. Failure to reject is never evidence of
equivalence. At n = 8 the paired test is effectively a **sign test** — power comes from
consistency, not effect size, and any 8/8 sign agreement gives exactly p = 0.0078 regardless of
magnitude.

---

## 3. Pipeline defects

### 3.1 Root-selection circularity

`nuclear_density` is simultaneously the DPT root selector (`analysis/diffusion.py:301-316`), one
of the six validation features (`run_all.py:694`), and the covariate partialled out in the
cellularity-confound analysis. The axis is partly *defined* by what it is validated against.

**Mitigating evidence:** 25 random 20-root sets reproduce the axis at |rho| 0.78–0.89, and the
holeyness-anchored axis agrees with the density-anchored one at rho 0.9476 in 2M-1 despite 19 of
20 roots differing. The manifold, not the root rule, does most of the work.

### 3.2 `compute_nuclear_density_quick` returns 0.0 for two different things

Genuinely acellular tissue **and** a segmentation failure both yield `0.0`
(`validation/morphological_features.py`). Indistinguishable in the stored value. This produced
the degenerate 2M-2 anchor, where all 20 roots sit at exactly 0.0 and **none lies inside a Tumor
annotation**. Fixing would require a learned segmenter that reports failure as failure.

### 3.3 `n_roots` is silently clamped

`n_roots = min(n_roots, finite_idx.size)` (`analysis/diffusion.py:314`). Ask for 20, get fewer,
no warning. Read `len(adata.uns['dpt_root_candidates'])` for the true count.

### 3.4 `pseudotime_std` is a poor anchor-health check

It read 27.70% of range in 2M-2 and fell to 3.40% after repair, yet the axis moved by only
rho 0.9621 — a median across 20 roots is robust to 3 outliers; `std` is not. **Use mean
leave-one-out concordance across roots instead** (2M-1 scores 0.726, 2M-2 only 0.478). Keep
`std` as a per-patch uncertainty map. Full reasoning in `ANCHOR_VALIDATION_RECORD.md` §7.

### 3.5 The feature cache key is the slide name alone

The guard compares row count only, not dimensionality (`run_all.py:363-403`). `--model resnet50`
against a Phikon-populated cache **passes** and silently uses 768-dim Phikon features. **One
cache directory must serve exactly one `(model, patch_size, stride, min_roi_coverage,
stain_method)` combination.** Unenforced convention, not a checked constraint.

### 3.6 Defaults that were never chosen

Eleven operative values are library or function defaults rather than deliberate choices — five
tissue-filter thresholds, the PCA variance target, Leiden k and metric, the diffusion map's
euclidean metric (scanpy's default, no CLI flag), UMAP's parameters, and the PAGA gate's
`threshold=0.05`. Full table in `reports/codebase_inventory.md` §3. Notably **k=30 for the
diffusion graph sits below the k≥50 plateau** reported by Vig et al.

### 3.7 Fixed generic stain matrix across two fixations

`rgb2hed` applies one stain matrix to both sections. Hematoxylin separates the sections at
rank-biserial 0.71. Otsu-on-hematoxylin segmentation ties four of six morphological features to
a channel that differs by fixation. Per-section stain estimation (Macenko/Vahadane) would
address this.

### 3.8 `packing_irregularity` returns NaN below 3 nuclei

Structurally couples it to nuclear density: the patches where it is missing are exactly the
low-density ones.

---

## 4. Open scientific questions

### 4.1 `rho(duct area, pseudotime)` diverges between sections and remains unexplained

**+0.4325 (2M-1) vs −0.0844 (2M-2)**, 8/8 glands, paired p at the design floor. Ruled out so far:

- **not** root-repair-dependent — repairing 2M-2's three discordant roots moved it by 0.017
- **not** duct-size/density geometry — `rho(area, nuclear_density)` is +0.389 vs +0.342
- **not** fixation shrinkage — per-gland correlations are invariant to monotone rescaling by
  construction (verified at 0.00e+00), and the pooled difference survives within-gland
  rank-normalisation at 95%
- **anchor-dependent** — the gap is +0.517 density-rooted, +0.169 holeyroot, +0.246
  area-stratified

**Leading untested hypothesis.** If PFA preserves open lumens while Carnoy's collapses them, PFA
yields more genuinely-empty patches, and the anchor selects the 20 *lowest-density* patches —
placing PFA's anchor on lumen or background and Carnoy's elsewhere. Consistent with 2M-2's roots
all sitting at density 0.0 with none inside a duct, and with zero-density patches being 0.208%
of 2M-2 against 0.133% of 2M-1 (**1.56×**, against a 1.64× duct-area ratio). **Suggestive only:**
n = 21 and n = 11 are far too small, and the ratio coincidence could easily be chance.

### 4.2 Are the two halves of a gland equivalent tissue?

The paired correction fixes the *unit of analysis*. It does not establish that the two pieces of
each gland sample comparable tissue. If they sample different regions, within-gland regional
variation remains mixed with the fixation effect and cannot be separated by any analysis of this
data. **Outstanding clarification with the pathologist.**

### 4.3 Is the compression monotone?

A pure multiplicative shrink leaves log-area spread unchanged, and it is unchanged (ratio 1.018,
3/8 glands below 1, p = 0.73) — evidence for monotonicity. But a **non-monotone** distortion, one
that reorders ducts by size, cannot be tested at all: the two halves of a gland are different
physical slides with different ducts, so there is no duct-to-duct correspondence.

### 4.4 Late-pseudotime tail is concentrated in one slide

3.3–3.6× one slide against 12.5% uniform, on every axis tested. Characterised
(`eccentricity_within_slide.py`) but not fixed. Cramér's V shows the cohort late subclusters
largely **are** slides.
