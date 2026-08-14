#!/bin/bash
# CONFIG D — HOLEYNESS ROOTS via AREA-OVERLAP assignment, PRODUCTION FILTERS.
# Section 2M-1 only.
#
# This is Config A plus the two methodological fixes. It is the successor to A,
# not a new axis: same production filters, same production feature cache, same
# everything downstream. A vs D isolates the assignment rule alone.
#
# ── FIX 1: AREA-OVERLAP PATCH-TO-DUCT ASSIGNMENT ─────────────────────────────
#   Config A used the centre-in-polygon rule: a patch joins a duct only if its
#   single centre pixel lands inside the polygon. A duct narrower or smaller than
#   a 112 px patch can hold tissue in several patches while capturing the centre
#   of none — which is how 571/2173 ducts (26%) had zero patches, systematically
#   the SMALLEST and LEAST holey. Those are exactly the "Time 0" branch ducts a
#   low-holeyness anchor should be drawing from, so Config A's anchor was blind
#   to its own target population.
#
#   --holeyness-assignment overlap treats each patch as a 112x112 box and assigns
#   it to the duct claiming the largest ABSOLUTE intersection area, provided that
#   area is >= 25% of the patch. The run also computes the centre-rule assignment
#   purely to report how many ducts the overlap rule SALVAGES — see
#   holeyness_roots.json -> overlap_vs_centre.
#
#   Requires shapely (in requirements.txt). run_all pre-flights the import and
#   refuses to fall back to the centre rule, which would silently produce a
#   different experiment under this label.
#
# ── FIX 2: TOPOLOGICAL CONTIGUITY, AS AN ASSERTION ───────────────────────────
#   The roots are verified to occupy ONE connected component of the k=30
#   euclidean DPT graph, and the job ABORTS if they do not.
#
#   This is deliberately an assertion rather than a selection rule. Selecting the
#   seed's 19 nearest PCA neighbours would guarantee tightness but would very
#   likely return 20 patches of the SAME duct on the SAME slide, collapsing the
#   anchor to one location — the failure the round-robin one-per-duct rule exists
#   to prevent. Max pairwise latent distance IS reported (in adata.X, euclidean,
#   the same space and metric sc.pp.neighbors uses — never UMAP), as a diagnostic
#   only.
#
#   compute_dpt_multi_root now also records how many patches each root failed to
#   reach BEFORE clamping, into adata.uns['dpt_n_nonfinite_per_root'], and warns
#   loudly if any root clamped. That is the mechanism behind Config B's
#   pseudotime_std at 30% of range, and it was previously invisible.
#
# ── FIX 3: DEGENERATE POOL IS NOW A HARD ERROR ───────────────────────────────
#   If every duct at or below the percentile threshold has the SAME hole %, then
#   "lowest holeyness" orders nothing and the arbitrary UUID tie-break picks the
#   roots. The anchor would then be "an arbitrary 20 of the N zero-hole ducts",
#   which is a materially weaker claim. The run now FAILS rather than proceeding
#   silently. Pass --holeyness-allow-degenerate-pool to override deliberately.
#   The tie-break is UUID order and NOT nuclear_density: breaking ties on density
#   would quietly reinstate the exact circularity this anchor removes.
#
# ── WHY PRODUCTION FILTERS, NOT RELAXED ──────────────────────────────────────
#   The relaxed-filter branch (Configs B/C) is where the manifold breaks: B vs C
#   on identical patch sets gives rho 0.39, because background patches form
#   islands and the clamp fires. The root-insensitivity result (random roots
#   reproduce v2 at 0.78-0.89) holds on a CONNECTED manifold. So these fixes are
#   applied to the strict-filter lineage, which is also the one whose morphology
#   trends look cleaner.
#
# EXPECTATION. Same as Config A: rho vs v2 in or above 0.78-0.89, orientation
#   possibly changed, ordering not. Config A gave 0.9476. If D departs sharply
#   from A, the salvaged small ducts moved the anchor — which is the point of the
#   fix and should be read off overlap_vs_centre, not treated as noise.
#
# READS (READ-ONLY): $SCRATCH/data/features_cache  <- never written; aborts if
#                    incomplete rather than regenerating
#                    $PNG_DIR, $ANN_DIR, $SLIDE_DIMS, $HOLEYNESS_EXPORT
# WRITES (NEW ONLY): $SCRATCH/results/per_section_v3d_overlap/atlas_2M-1
#
# Usage:  sbatch ~/cancer_trajectory_atlas/jobs/run_v3d_overlap.sh

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=v3d_overlap
#SBATCH --output=logs/v3d_overlap-%j.out

set -euo pipefail
mkdir -p logs
# NOT $(dirname "$0"): sbatch copies the script to a spool dir, so $0 is not
# the repo. SLURM_SUBMIT_DIR is unset for an interactive run, hence the fallback.
V3_JOBS_DIR="${SLURM_SUBMIT_DIR:-$HOME/cancer_trajectory_atlas}/jobs"
[ -f "$V3_JOBS_DIR/_v3_common.sh" ] || V3_JOBS_DIR="$HOME/cancer_trajectory_atlas/jobs"
# shellcheck disable=SC1091
source "$V3_JOBS_DIR/_v3_common.sh"

OUT_DIR="$V3D_BASE/atlas_${SECTION}"
OVERLAP_MIN_FRACTION=0.25
PERCENTILE=5
MAX_ROOTS_PER_DUCT=3

echo "============================================================"
echo "  CONFIG D — holeyness roots, AREA-OVERLAP assignment"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Output   : $OUT_DIR   (NEW)"
echo "  Cache    : $PROD_CACHE   (READ-ONLY, production filters)"
echo "  Assign   : overlap >= ${OVERLAP_MIN_FRACTION} of patch area"
echo "  Pool     : P${PERCENTILE} of per-duct hole %, max ${MAX_ROOTS_PER_DUCT} roots/duct"
echo "  Degenerate pool: HARD ERROR (no --holeyness-allow-degenerate-pool)"
echo "============================================================"

v3_assert_output_safe "$OUT_DIR"

echo ""
echo "=== Pre-run checks ==="
v3_assert_inputs_exist "$PNG_DIR" "$ANN_DIR" "$SLIDE_DIMS" "$HOLEYNESS_EXPORT" "$PROD_CACHE"
v3_assert_prod_cache_complete

v3_load_env

python -c "import shapely; print('  shapely', shapely.__version__)" || {
    echo "ERROR: shapely is not importable in ~/envs/atlas, and Config D requires"
    echo "       it for area-overlap assignment. It is in requirements.txt:"
    echo "         pip install 'shapely>=2.0.0'"
    echo "       Refusing to fall back to the centre rule."
    exit 1
}

CACHE_BEFORE=$(find "$PROD_CACHE" -name '*_features.npy' -printf '%f %s\n' \
               | sort | md5sum | cut -d' ' -f1)
echo "  Production cache fingerprint (pre-run): $CACHE_BEFORE"
echo "============================================"

mkdir -p "$OUT_DIR"
cd ~

python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir                        "$PNG_DIR" \
    --annotation-dir                 "$ANN_DIR" \
    --output-dir                     "$OUT_DIR" \
    --stain-method                   none \
    --batch-method                   none \
    --model                          phikon \
    --patch-size                     "$PATCH_SIZE" \
    --stride                         "$STRIDE" \
    --clustering-method              leiden \
    --leiden-resolution              "$LEIDEN_RES" \
    --n-roots                        "$N_ROOTS" \
    --n-permutations                 "$N_PERMUTATIONS" \
    --features-cache-dir             "$PROD_CACHE" \
    --cap-strategy                   median \
    --slides                         "$SLIDES_CSV" \
    --root-source                    holeyness \
    --holeyness-export               "$HOLEYNESS_EXPORT" \
    --holeyness-slide-dims           "$SLIDE_DIMS" \
    --holeyness-assignment           overlap \
    --holeyness-overlap-min-fraction "$OVERLAP_MIN_FRACTION" \
    --holeyness-percentile           "$PERCENTILE" \
    --holeyness-min-patches          1 \
    --holeyness-max-roots-per-duct   "$MAX_ROOTS_PER_DUCT"

CACHE_AFTER=$(find "$PROD_CACHE" -name '*_features.npy' -printf '%f %s\n' \
              | sort | md5sum | cut -d' ' -f1)
echo ""
if [ "$CACHE_BEFORE" != "$CACHE_AFTER" ]; then
    echo "  WARNING: the production cache CHANGED during this run. It should not"
    echo "           have. Investigate before trusting this or any sharing run."
else
    echo "  Production cache unchanged, as expected."
fi

echo ""
echo "  --- Fix 1: what the overlap rule salvaged ---"
python - "$OUT_DIR" <<'PY'
import json, sys
p = f"{sys.argv[1]}/holeyness_roots.json"
try:
    d = json.load(open(p))
except Exception as e:
    sys.exit(f"  (holeyness_roots.json unreadable: {e})")
s, c = d.get("overlap_vs_centre"), d["counts"]
if s:
    print(f"  ducts with patches, centre  : {s['n_ducts_centre_rule']}")
    print(f"  ducts with patches, overlap : {s['n_ducts_overlap_rule']}")
    print(f"  SALVAGED                    : {s['n_salvaged']}   (lost: {s['n_lost']})")
    if s['salvaged_area_um2_median'] is not None:
        print(f"  salvaged duct area um^2 med : {s['salvaged_area_um2_median']:.0f} "
              f"vs {s['centre_rule_area_um2_median']:.0f} centre-rule")
    if s['salvaged_hole_pct_median'] is not None:
        print(f"  salvaged duct hole % med    : {s['salvaged_hole_pct_median']:.3f} "
              f"vs {s['centre_rule_hole_pct_median']:.3f} centre-rule")
print(f"  patches in no duct          : {100*c['frac_patches_no_duct']:.1f}%")
print(f"  candidate pool              : {c['n_ducts_in_candidate_pool']} ducts "
      f"({c['n_pool_ducts_strictly_below_threshold']} strictly below threshold)")
print(f"  distinct ducts among roots  : {c['n_distinct_ducts_among_roots']}/20")
t = d.get("topology", {})
if t:
    print(f"  --- Fix 2: topology ---")
    print(f"  graph components            : {t['n_graph_components']}")
    print(f"  components spanned by roots : {t['n_components_spanned_by_roots']}")
    print(f"  max pairwise latent dist    : {t['max_pairwise_latent_distance']:.4f}")
    print(f"  tightness vs random         : {t['tightness_ratio_vs_random']:.3f}")
PY

echo ""
echo "  --- clamping (from the run log above) ---"
echo "  Look for 'CLAMPING FIRED' or 'All 20 roots reached every patch'."

echo ""
echo "============================================================"
echo "  CONFIG D COMPLETE — $OUT_DIR"
echo "  Compare against Config A to isolate the assignment rule."
echo "============================================================"
