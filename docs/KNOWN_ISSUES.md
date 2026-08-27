# KNOWN_ISSUES.md — defects and limitations, with scope

**Current as of 2026-08-24.** Each entry states what the issue is, why it matters, how it was
found, **whether it affects a reported number**, and what fixing it would require.

**Sections 5 to 9 were added by the 2026-08-24 correctness audit** and record defects found by
reading the code rather than by analysing results. Start at **section 5**, which is an ordered
shortlist; the rest is detail behind it. Every entry there states plainly whether it is **LIVE**
(a recorded number is affected) or **LATENT** (a real defect that no recorded run triggered).
That distinction is load-bearing. Most of what follows is latent.

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

- `validation/correlations.py:84`, inside `permutation_test` — `rng.permutation(pseudotime)`
  over all patches
- `analysis/cellularity_confound.py:272` — `pt_shuf = rng.permutation(pt)`, same pattern

**No comment marks either site.** See §9. `correlations.py` carries a second, unrelated defect
in the same loop: see §6.1.

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

**What.** Patch-centre assignment (`analysis/holeyness.py:305`) leaves 571/2173 ducts (2M-1) and
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

`nuclear_density` is simultaneously the DPT root selector (`analysis/diffusion.py:303-317`), one
of the six validation features (`run_all.py:702`), and the covariate partialled out in the
cellularity-confound analysis. The axis is partly *defined* by what it is validated against.

**Mitigating evidence:** 25 random 20-root sets reproduce the axis at |rho| 0.78–0.89, and the
holeyness-anchored axis agrees with the density-anchored one at rho 0.9476 in 2M-1 despite 19 of
20 roots differing. The manifold, not the root rule, does most of the work.

### 3.2 A genuinely acellular patch reads 0.0 and is therefore preferentially selected as a root

> **Corrected 2026-08-24.** This entry previously read *"failures and acellular tissue both yield
> 0.0"*. That is **no longer accurate**: FIX 1a made segmentation failures return `np.nan`, and
> `compute_nuclear_density_quick` documents this at
> `validation/morphological_features.py:256-259`. The sentinel collision is gone. The mechanism
> below is what is still live, and it is the one that matters.

**What.** `compute_nuclear_density_quick` is a *correct* measurement that returns `0.0` for a
patch containing no nuclei. Nothing is wrong with the value. The problem is what consumes it:
`compute_dpt_multi_root` selects the **n lowest-density patches** as DPT roots
(`analysis/diffusion.py:303-317`). A patch of empty lumen, background, or acellular stroma
therefore sorts to the very front of the root queue **by construction**.

**Why it matters.** The pseudotime origin is anchored on whatever tissue is emptiest, which is
not the same thing as whatever tissue is earliest. Low cellularity is being used as a proxy for
trajectory origin with no evidence that the two coincide.

**This is the mechanism behind the degenerate 2M-2 anchor.** All 20 of 2M-2's roots sit at
`nuclear_density` **exactly 0.0**, and **none lies inside a Tumor annotation**. The root rule did
exactly what it was written to do; the rule is the defect, not a bug in its implementation.

**Affects reported numbers.** Yes, for 2M-2. Its anchor is placed on non-tumor tissue. See
`ANCHOR_VALIDATION_RECORD.md` §4 error #2, and §4.1 above for the leading hypothesis that this
also explains the `rho(duct area, pseudotime)` divergence between sections.

**Mitigating evidence, same as §3.1.** 25 random 20-root sets reproduce the axis at |rho|
0.78-0.89, so the manifold fixes the ordering and the roots fix mainly which end is zero.

**Fixing would require** either a root rule that does not use density as its sole criterion (the
holeyness anchor in `analysis/holeyness_roots.py` is exactly this experiment), or excluding
zero-density patches from the candidate pool, or a segmenter that distinguishes "no nuclei
present" from "no nuclei detected".

### 3.3 `n_roots` is silently clamped

`n_roots = min(n_roots, finite_idx.size)` (`analysis/diffusion.py:315`). Ask for 20, get fewer,
no warning. Read `len(adata.uns['dpt_root_candidates'])` for the true count.

**Not triggered in either reference section** (zero feature failures, so `finite_idx.size` was
the full patch count). Siblings and the contrasting good example are in §6.5.

### 3.4 `pseudotime_std` is a poor anchor-health check

It read 27.70% of range in 2M-2 and fell to 3.40% after repair, yet the axis moved by only
rho 0.9621 — a median across 20 roots is robust to 3 outliers; `std` is not. **Use mean
leave-one-out concordance across roots instead** (2M-1 scores 0.726, 2M-2 only 0.478). Keep
`std` as a per-patch uncertainty map. Full reasoning in `ANCHOR_VALIDATION_RECORD.md` §7.

### 3.5 The feature cache key is the slide name alone

The guard compares row count only, not dimensionality (`run_all.py:371-411`). `--model resnet50`
against a Phikon-populated cache **passes** and silently uses 768-dim Phikon features. **One
cache directory must serve exactly one `(model, patch_size, stride, min_roi_coverage,
stain_method)` combination.** Unenforced convention, not a checked constraint. Restated with the
other unenforced conventions in §7.2; the site itself is well commented and needs nothing.

### 3.6 Defaults that were never chosen

Eleven operative values are library or function defaults rather than deliberate choices — five
tissue-filter thresholds, the PCA variance target, Leiden k and metric, the diffusion map's
euclidean metric (scanpy's default, no CLI flag), UMAP's parameters, and the PAGA gate's
`threshold=0.05`. Full table in `reports/codebase_inventory.md` §3. Notably **k=30 for the
diffusion graph sits below the k≥50 plateau** reported by Vig et al. **That k point is not
noted at the code site**, though the metric-is-a-default point is; see §9.

### 3.7 Fixed generic stain matrix across two fixations

`rgb2hed` applies one stain matrix to both sections. Hematoxylin separates the sections at
rank-biserial 0.71. Otsu-on-hematoxylin segmentation ties four of six morphological features to
a channel that differs by fixation. Per-section stain estimation (Macenko/Vahadane) would
address this. **Nothing at `validation/morphological_features.py:70` says any of this**; see §9.

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

---

## 5. If you fix one thing, fix these — in this order

Added 2026-08-24. **The decision has been made not to fix any of sections 6 to 8.** This ordering
exists so the next person can pick up any of it without re-deriving the analysis.

**0. Read `mpp_verification.json` from the Stage D conversion (§6.4b).** Two minutes, no
compute. It settles whether the timepoint cohort was converted at twice the atlas's
resolution, which would be an independent and *fixable* explanation for the 100%
extrapolation result. Cheapest item on this list and the one with the most riding on it.

**1. Quantify the projection clamp (§6.4).** Read-only, needs no GPU, changes nothing, and it
bears on a conclusion already on record. Count how many projected pseudotime values sit at
*exactly* 0.0 or *exactly* 1.0 in the Stage D timepoint output. That single number converts "the
projection was 100% extrapolation" from an inference into a measurement. Do this first because it
is cheap and because it firms up something you would otherwise take on trust.

**2. Add the row-alignment assertion (§7.1).** Widest blast radius in the codebase, and a length
assertion at write time is behaviour-preserving while the invariant holds. Sixteen modules depend
on an unchecked positional join, and one of them already crashed because of it.

**3. Fix the 0.0 padding in the permutation null (§6.1).** Anti-conservative, and it is the exact
sibling of a bug that already bit this project once. The correct pattern is not a design question:
`analysis/cellularity_confound.py` already implements it, at lines 277 and 305. Copy it.

**4. Add the site comment for the global shuffle (§6.2).** Already known and already recorded as
§1.2. It simply is not marked at the code site, so a reader of `permutation_test` has no way to
learn that the null is mis-specified. Comment only, no behaviour change.

**Everything else can wait.** §6.3 and §6.5 through §6.9 are **latent**: real defects that no
recorded run triggered. None affects a number in `validation.json`, in either reference section,
or in any figure in the manuscript. Fixing them is hygiene, not correction.

---

## 6. Correctness audit — 2026-08-24

Found by reading the code, not by analysing results. Method: repo-wide sweeps for bare `except`,
sentinel substitution, silent clamping, and boundary handling, then targeted reading of every hit
on a path that feeds a recorded number.

**There are no bare `except:` and no `except: pass` anywhere in live code.** That sweep came back
clean.

### 6.1 The permutation null is padded with 0.0 on a failed computation — LATENT

**What.** `validation/correlations.py:89`, inside the permutation loop:

```python
valid = np.isfinite(values) & np.isfinite(shuffled_pt)
if valid.sum() < 10:
    null_distributions[name].append(0.0)      # <-- sentinel
    continue
```

`0.0` is indistinguishable from a genuine permutation that produced |rho| = 0, and it is the
**minimum possible value** of |rho|. At `correlations.py:107` the null is consumed without
filtering, so every padded entry drags the 95th percentile down and the empirical p down with it.

**Direction of error: ANTI-CONSERVATIVE.** It manufactures significance rather than hiding it.
The affected quantity feeds `n_significant_permutations`, one of the two inputs to the headline
verdict.

**Affects reported numbers: NO.** The guard cannot fire in any recorded run. `pseudotime` is
all-finite after min-max normalisation, so `valid.sum()` reduces to `np.isfinite(values).sum()`,
which is constant across permutations and far above 10 for every feature. This is a live landmine,
not a live error. It would fire on a small cohort, or on a feature that failed extraction almost
everywhere.

**The same module already gets this right elsewhere.** At `correlations.py:102` the *real*
statistic's under-10 case correctly writes `np.nan`. Only the permutation loop uses `0.0`.

**Fixing would require** two changes, both already written and working in
`analysis/cellularity_confound.py`: append `np.nan` instead of `0.0`, as that module does at line
277, and filter non-finite values before computing the percentile and the p-value, as it does at
line 305. This is not a design decision, it is copying an established pattern between two files.

**This is the exact sibling of the already-fixed 0.0-for-failed-patches bug** (FIX 1a, where a
failed patch's features were stored as 0.0 and were then preferentially selected as DPT roots).
Same error, same module family, found because that one taught us to look for it.

### 6.2 Global patch shuffle ignores slide nesting — LIVE, already recorded as §1.2

**What.** `validation/correlations.py:84` shuffles all patch labels globally. Patches within a
slide overlap by 16 px (stride 96 against patch size 112) and share a slide-level mean, so they
are not exchangeable. The null is narrower than the true null.

**Direction of error: ANTI-CONSERVATIVE**, by an unquantified amount.

**Affects reported numbers: YES** — see §1.2, which records this and its second site at
`analysis/cellularity_confound.py:272`.

**What is new in this audit:** nothing in `permutation_test` says so. The function's docstring is
otherwise careful, explaining the p = 0 convention at length, which makes the omission more
misleading than it would be in an undocumented function. A reader has no way to learn from the
code that the test is mis-specified.

**Fixing the mis-specification** requires a within-slide shuffle, already implemented in
`analysis/holeyness.py` and `analysis/holeyness_asymmetry.py`. **Marking it** requires only a
comment.

### 6.3 Projection silently substitutes a different estimator — LATENT

**What.** `analysis/projector.py:130-132`:

```python
except Exception as exc:
    print(f"  WARNING: Ingest failed ({exc}), falling back to KNN.")
    return self._project_knn(adata_test, adata_test.X)
```

Any exception from the scanpy ingest path switches to KNN projection. These are different
estimators producing different pseudotime. The only record is one stdout line; **the output
artifact does not record which method produced it.**

**Can it produce a wrong number.** Not "wrong" so much as *not the method you believe you ran*,
and not comparable across slides if it fired for some and not others.

**Affects reported numbers: NO.** Every recorded projection passes `method="knn"` explicitly
(`analysis/timepoint_projection.py`, `analysis/loo_project.py`), which does not enter the ingest
path at all, so this fallback was unreachable for those runs.

**Fixing would require** recording the method actually used in the output artifact, and narrowing
the `except` to the exceptions ingest is expected to raise.

### 6.4 Projected pseudotime is clamped to [0,1] — BEARS ON A RECORDED CONCLUSION

**What.** `analysis/projector.py:127` and `:175` both end with `np.clip(pt, 0.0, 1.0)`. A test
patch outside the training manifold is mapped onto the boundary, so **a clamped 1.0 and a genuine
late-trajectory 1.0 are indistinguishable in the stored output**. Nothing counts how often it
fires.

**Why this one matters.** The timepoint-cohort Stage D result found that projection was **100%
extrapolation on all 29 slides**. Under that condition every projected patch is being asked about
a region the model never saw, so the pseudotime returned is drawn from whichever training patches
happen to be least far away, and the boundary is where those values pile up. **The projected
pseudotime is boundary saturation, not measurement.**

**This STRENGTHENS the existing decision to stop the timepoint work; it does not contradict it.**
Stage D was halted because the projection was extrapolating. This identifies the specific
mechanism by which that extrapolation still produced confident-looking numbers, and explains why
the projected values looked plausible rather than obviously broken.

**The cheap check that was NOT run.** Count how many projected values sit at exactly 0.0 or
exactly 1.0, per slide and pooled. That directly quantifies the saturation. It is read-only, needs
no GPU, and can reuse the feature cache Stage D already populated. It is item 1 in §5.

**A caveat so that check is read correctly when someone runs it.** A KNN regressor returns a
weighted mean of training targets, so its raw output is bounded by the training range by
construction, and the clip is expected to move few or no values. **A near-zero clamp rate would
therefore NOT clear the projection.** The quantity that matters is the fraction sitting *at* a
boundary, not the fraction the clip had to move.

**Affects reported numbers.** It affects how the Stage D projected pseudotime should be described.
It does not affect the per-section atlas results, which do not project.

### 6.4b Conversion-scale mismatch between the atlas and the timepoint cohort — HYPOTHESIS

**Found 2026-08-26, while auditing the fallout of the scale-0.5 discovery.**

**What.** The atlas cohort was converted at `--ndpi-scale 0.5` (established
bit-identically, job 1648162). `jobs/run_timepoint_convert_nocrop.sh` converts the
timepoint cohort at `--ndpi-scale 1.0`, and its header states that 1.0 is *"the original
pipeline's actual conversion scale — confirmed against `jobs/convert_ndpi.sh` and
`pipeline_config.py`'s default, NOT the previously-assumed 0.5"*.

**That confirmation used the broken script as its evidence.** `convert_ndpi.sh` passed
1.0 and could not regenerate the cohort; `pipeline_config.py`'s default of 1.0 is
likewise not what production used. A correct assumption of 0.5 was overturned by two
sources that were both wrong. This is the clearest instance in the repo of the stale
conversion recipe propagating into a downstream decision.

**Why it might matter scientifically.** If the two cohorts share level-0 MPP, which that
script's pre-flight check requires before it proceeds, then:

| | 1 PNG pixel | a 112x112 patch covers |
|---|---|---|
| atlas (scale 0.5) | 2 level-0 px | 224 x 224 level-0 px |
| timepoint (scale 1.0) | 1 level-0 px | 112 x 112 level-0 px |

Timepoint patches would show tissue at **twice the magnification**, covering **a quarter
of the area**, that the atlas was trained on. Phikon embeddings of systematically
finer-grained tissue would sit off the training manifold.

**That is a candidate mechanism for the Stage D result** that projection was **100%
extrapolation on all 29 slides**. A whole-cohort scale mismatch would produce exactly
that signature, and it would do so regardless of biology.

**NOT ESTABLISHED. Recorded as a hypothesis, not a finding.** What would settle it:

1. Read `mpp_verification.json` from the Stage D conversion. If the two cohorts' level-0
   MPP matched, the 2x mismatch is real. If the timepoint slides were scanned at half the
   atlas magnification, scale 1.0 was correct and there is no mismatch.
2. Compare a timepoint PNG's pixel-per-micron against an atlas PNG's.

**Consequence either way.** The timepoint work is already halted as
staining-confounded (`project_timepoint_cohort`). If this mismatch is real it is a
*second, independent* reason the projection could not work, and one that is fixable by
re-converting rather than by more tissue. That distinction matters if the cohort is ever
revisited.

**Nothing was changed.** `run_timepoint_convert_nocrop.sh` is untouched: it belongs to
halted work, its scale is chosen dynamically at runtime, and altering it without reading
`mpp_verification.json` would be guessing.

### 6.5 `n_roots` is silently clamped, and it has siblings — LATENT

**What.** `analysis/diffusion.py:315`, `n_roots = min(n_roots, finite_idx.size)`. Ask for 20, get
fewer, no warning line. Recorded as §3.3.

**Auditable after the fact** via `len(adata.uns['dpt_root_candidates'])`, which is persisted.

**Affects reported numbers: NO.** Both reference sections have zero feature failures, so
`finite_idx.size` was the full patch count and the clamp did not engage.

**Siblings found by the same sweep:**

| Site | Behaviour | Status |
|---|---|---|
| `analysis/projector.py:74` | `n_neighbors=min(15, len(train_pca) - 1)`, silent | latent |
| `analysis/clustering.py:106` | `max_components` **refits PCA from scratch** rather than truncating, giving a different basis, not a subset of the first. It does print. | dead in practice: `run_all.py` never passes it |
| `sc.tl.diffmap(n_comps=...)` | scanpy reduces `n_comps` internally if the graph supplies fewer eigenvectors | library behaviour, **not verified either way** |

**The contrast worth copying.** The inf-to-root-max clamp in `compute_dpt_multi_root` is handled
properly: it prints a loud `CLAMPING FIRED` block (`analysis/diffusion.py:354`) naming how many
roots and how many unreached patches, and it **persists `dpt_n_nonfinite_per_root`** so the event
is recoverable from the artifact rather than only from the log. That is the model the others
should follow, and it exists because this exact failure mode produced Config B's inflated
`pseudotime_std` while nothing in the output said so.

### 6.6 Half-pixel offset in every patch-centre computation — LATENT, negligible

**What.** A patch spans pixel indices `x` through `x + 111`, so its geometric centre is
`x + 55.5`. Three sites compute `half = patch_size / 2.0` and use `x + 56.0`:

- `features/patching.py:338`
- `analysis/holeyness.py:318`
- `analysis/holeyness_roots.py:486`

**Can it produce a wrong number.** In principle, for a patch whose centre lies within half a pixel
of a polygon boundary the centre-in-polygon test could flip.

**Affects reported numbers: NO, not meaningfully.** The bias is 0.5 px against a 112 px patch at
5x downsampling, and it is **systematic and identical across all three sites**, so patch-to-duct
assignment stays internally consistent. Recorded for completeness, not as a concern.

### 6.7 Boundary handling in extraction and cropping — LATENT

Three separate items, none affecting a recorded number.

**Trailing strip never covered.** `features/patching.py:334`, `range(0, h - patch_size + 1,
stride)` means up to `patch_size - 1` pixels along the right and bottom edges fall outside every
patch. Conventional and correct, but worth knowing when comparing patch counts against slide area.

**Right-half polygon discard uses `<=` on a vertex mean.** `features/patching.py:225` keeps a
polygon when `cx <= cropped_w`, where `cx` is the mean of the path's vertices. A duct straddling
the midline is therefore kept **whole**, including the portion outside the cropped PNG. No patches
exist out there, so patch selection is unaffected, **but a polygon-area computation over such a
duct would include out-of-image area.** Relevant to any future duct-area measurement; the current
duct areas come from the annotation export, not from this path.

**`coords + 56` hardcode.** `validation/correlations.py:200`, in `spatial_depth_correlation`'s
`roi_polygon` branch, hardcodes half of patch size 112 and would be wrong for any other patch
size. **That branch is unreachable** — `run_all.py` never passes `roi_polygon` — and the reachable
branch is already flagged as `spatial_depth_secondary` and excluded from the verdict.

### 6.8 A zero-patch slide vanishes without a persisted record — LATENT

**What.** `run_all.py:366`:

```python
if len(patches) == 0:
    print(f"    WARNING: No patches found - skipping")
    continue
```

The slide was already appended to `slide_names` before this point, so **index bookkeeping stays
correct**: `slide_idx = i` is stored in Pass 1 and `slide_names[sid]` lookups remain valid. This
was verified, not assumed.

**What is lost.** The slide never reaches `slide_data`, so it gets **no row in
`sampling_manifest.csv`**. A 16-slide run would produce a 15-row manifest with nothing stating
which slide dropped out. The printed "from N slides" total also overstates the contributing count.

**Affects reported numbers: NO.** Both reference sections have full patch counts on every slide.

### 6.9 Other silent-substitution paths — ALL LATENT

None of these was triggered in any recorded run.

| Where | Behaviour | Why it did not fire |
|---|---|---|
| `validation/morphological_features.py:114` | StarDist unavailable falls back to Otsu, printing one line. Nothing in the output records which segmenter ran. **StarDist is absent from `requirements.txt`**, so on a clean environment the fallback is the *default* outcome of asking for StarDist. | `--use-stardist` has never been passed |
| `data/stain_normalization.py:66` | Fewer than 1000 tissue pixels causes statistics to be computed over **all** pixels including glass, silently. A mostly-blank reference would normalise the whole cohort toward background. | `--stain-method none` |
| `data/stain_normalization.py:205` | A failed `transform` returns the **un-normalised** array with only a stdout warning, so a run can be silently half-normalised. Already documented at the site. | `--stain-method none` |
| `features/patching.py:479-480` | `get_patches` swallows every load exception and returns empty arrays, indistinguishable from "loaded fine, nothing survived the filters". | `run_all.py` uses `Image.open` directly and would raise |
| `features/extractors.py:154` | The ResNet branch scales to [0,1] and stops, skipping the ImageNet channel mean and std normalisation its pretrained weights expect. The features are deterministic and still cluster, which is why it went unnoticed. | every run used Phikon |

---

## 7. Unenforced conventions

Not defects in themselves. Each is a rule the code depends on but does not check.

### 7.1 `results.csv` and `adata.obs` are aligned by ROW POSITION ONLY

**The widest-blast-radius fragility in the codebase.**

**What.** Both are built from the same arrays in the same order (`run_all.py:728` and
`build_adata`), so row *i* of one **is** row *i* of the other. But there is **no shared key column
and no assertion**. `adata.obs` carries `cluster` and `slide_id` and nothing else: **no `x`, no
`y`, no `slide_name`.** Anything needing coordinates must read `results.csv` and join
**positionally**.

**Sixteen modules read both files.** Only `analysis/anchor_area_control.py` verifies the alignment
(`_verify_row_alignment`, line 167), and it does so only because this assumption **already crashed
job 1200392**. The other fifteen trust it silently.

**Why it matters.** A future change to either writer misaligns every dependent module **with no
error** — every downstream number would simply be computed against the wrong rows.

**The invariant currently holds.** This was verified during the audit; nothing is presently
misaligned.

**Cheapest mitigation:** a length assertion at write time, comparing `len(df)` against
`adata.n_obs` before writing. **Behaviour-preserving while the invariant holds**, and it converts
a silent future misalignment into an immediate, obvious failure. This is item 2 in §5.

### 7.2 Feature cache key is the slide name alone

Recorded as §3.5. The guard compares row count, not dimensionality. **The site itself is
extensively commented and needs no addition** — it already enumerates what the guard catches and
what it does not.

### 7.3 `slide_id` integers are run-local

**What.** They are positions in `slide_names`, which comes from `sorted(png_dir.glob("*.png"))`
filtered by `--slides`. **Two runs over different subsets assign different integers to the same
slide.**

**Consequence.** `results.csv` carries both `slide_name` and `slide_id`; `adata.obs` carries only
the id, as a string. **Any cross-run comparison must join on `slide_name`**, never on `slide_id`.
Joining on the integer across two runs with different slide subsets silently compares different
slides.

---

## 8. Weak gates

### 8.1 `check_slide_independence` passes more easily than it appears to

**What.** `analysis/clustering.py:317` uses a fixed `dominance_threshold=0.80` with no adjustment
for cluster size or for the number of slides.

**Why it is weak.** With 16 slides, chance dominance is about 6%. An 80% bar therefore catches
only egregious cases, and **a cluster that is 79% one slide passes silently**. Cluster size is not
considered either: a 5-patch cluster drawn entirely from one slide is unremarkable but trips the
threshold, while a large, genuinely slide-skewed cluster at 79% does not.

**Why it matters.** Its verdict is persisted to `slide_independence.json`. **A passing verdict
there is weaker evidence of slide independence than it appears to be**, and it should not be cited
as though it were a test with a calibrated false-positive rate. The quantitative batch-mixing work
in `analysis/batch_mixing.py` is the stronger evidence.

**Not a wrong number.** It reports the per-cluster proportions alongside the verdict, and those
are correct. It is the *verdict* that is over-interpretable.

---

## 9. Documentation gaps: issues with no comment at the code site

Recorded as a to-do rather than acted on, because the 2026-08-24 pass was documentation-only and
did not touch source.

| Issue | Site that should carry it | Present? |
|---|---|---|
| §1.2 global shuffle ignores slide nesting | `validation/correlations.py:50` `permutation_test`, and `analysis/cellularity_confound.py:272` | **No** |
| §1.3 LOO leakage across the matched pair | `analysis/loo_summary.py`, `analysis/loo_project.py` | **No** |
| §3.7 fixed generic stain matrix across two fixations | `validation/morphological_features.py:70` `_deconvolve_hematoxylin` | **No.** That docstring is four lines and mentions none of it, despite the issue tying four of six morphological features to a channel that separates the sections at rank-biserial 0.71 |
| §3.6 diffusion k=30 below the k>=50 plateau | `analysis/diffusion.py:44` `compute_diffusion_map` | **Partial.** The metric-is-a-default point is made well; the k point is not |
| §6.1, §6.2, §6.4, §6.5 | as cited in each entry | **No** |

§2.3, §3.2, §3.3, §3.5 and §3.8 are all reflected at their sites already.
