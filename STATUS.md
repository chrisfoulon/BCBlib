# STATUS — BCBlib / lesion_features

_Last touched: 2026-09-16_

## Goal

Ship a robust, server-deployable `bcb-lesion-features` pipeline that extracts
per-subject lesion and disconnectome damage profiles across a standard atlas set
(EBRAINS) with optional private enrichment (TDI, streamline ratio).

## State of play

**PyPI is at `0.7.2`** (2026-09-16), a bugfix release — see below. `main` HEAD = `fb8b31e`
(tag `v0.7.2`). `dev` HEAD = `f4975b8` (merge of `main` into `dev`, carrying the fix forward
past the disconnectome2 Phase 1+2 work described below, which is still unreleased/private).

**`devel` branch** is the live branch (shared with external team; cannot be retired yet).
It is now behind `main`/`dev` — it still contains the pre-fix `pwll_normalised`/
`continuous_dice` bug (see below) and does not have the disconnectome2 work. Not
retired yet per the existing decision below (Paolo's Docker still points at it).

**2026-09-16 — fixed `pwll_normalised`/`continuous_dice` bound violation, released as
`v0.7.2`.** Both metrics in `compute_region_stats()` (`bcblib/tools/damage_profile/_stats.py`)
used the raw unweighted overlap sum (`sum_ov`) as their numerator instead of the
probability-weighted sum `Σ(subject_data × atlas_weight)`. This let them exceed their
documented `[0, 1]` bound whenever the atlas has non-uniform, low per-voxel probability
spread over many voxels — the normal case for real tractography atlases (e.g. Yeh
HCP1065). Existing tests only exercised uniform atlas weights, where the buggy and
correct formulas coincide, which is why it shipped in `0.7.0`/`0.7.1`. Root cause found
and fix applied to `main` directly (files were byte-identical between `main` and `dev`,
so the fix didn't need to wait on disconnectome2's release); 2 new regression tests with
non-uniform weights added, 1 pre-existing test's hardcoded expected values corrected
(they encoded the buggy behavior). 90/90 `test_damage_profile.py` tests pass on both
`main` and `dev` post-merge. **Any `pwll_normalised`/`continuous_dice` values already
computed with `bcblib<=0.7.1` should be treated as unreliable and recomputed** —
`sum_overlap`, `sum_atlas_in_tract`, and `max_atlas_prob_in_overlap` were unaffected.

Checked the same session whether the Yeh HCP1065 atlas encodes hemisphere via signed
values (which would break `mask = weights > 0` in the same function) — **ruled out**:
each `_L`/`_R` tract is already a separate file in the cached atlas
(`~/.bcblib/atlases/yeh_hcp1065/prob/`), and none of the 64 files contain negative
values. No action needed.

All planned features for the 0.7.0 release are implemented and pushed to `devel`:
- TDI private hook (`_tdi.py`) — hooks `/opt/tdi` or `$TDI_DIR`, skips silently if absent
- Yeh HCP1065 atlas warp fix (`order=1` trilinear interpolation)
- Interpolation overhaul: trilinear for probability atlases, ANTs genericLabel for binary masks
- `pip install bcblib[ants]` / `bcblib[dipy]` / `bcblib[ebrains]` optional extras
- Streamline ratio: per-subject `*_atlas-yeh_hcp1065_streamline.csv` (87 tracts, `tract`/`streamline_ratio` columns); TRK files auto-downloaded on first `--ebrains` run (588 MB)
- Output flattened (`LF_SUBDIR=""`) so prep + feature outputs land directly under `sub-XXX/`

Update email sent to Sebastiano and Paolo on 2026-07-14 with install instructions.
Awaiting confirmation from PHI that the update went through.

**2026-09-09 — disconnectome2 Phase 1 landed on `dev`.** disconnectome2
(`~/neuro_apps/disconnectome2`, private, not on PyPI) is a faster/more-accurate
CLI-based drop-in for BCBToolKit's `run_disco.sh`; output naming already matches
`predict_disco_output()` byte-for-byte, no glue code needed. Added to
`_disco.py`: `disco2_ready()` (non-raising auto-detect: package importable +
`DISCO2_INDEX_PATH`/`--disco2-index` resolves), `require_disco2()` (raising
variant for forced selection), `run_disco2_batch()` (subprocess wrapper,
mirrors `run_disco_batch`). `run_lf_preprocess.py` gained `--engine
{auto,bcbtoolkit,disco2}` (default `auto`, prefers disco2 when ready) and
`--disco2-index`; `_select_disco_engine()` holds the selection logic, unit
tested directly (new `test_run_lf_preprocess.py`). BCBToolKit path untouched.
29 new tests, 396 total passing (one pre-existing, unrelated
`test_parcitron.py` collection error). Spec:
`~/neuro_apps/disconnectome2/docs/BCBLIB_INTEGRATION.md`.

**2026-09-09 — disconnectome2 Phase 2 landed on `dev`.** `run_disco2_batch`
gained `fiber_class`/`out_voxel_size`/`len_min`/`len_max` (all optional,
default `None` → no behavior change unless a caller opts in; disco2's own
CLI validates `fiber_class`, not duplicated here). `run_lf_preprocess.py`
gained matching `--fiber-class`/`--out-voxel-size`/`--len-min`/`--len-max`
flags, disco2-only — `_select_disco_engine` now warns (non-fatal) to stderr
if any are set but the resolved engine is BCBToolKit (forced or auto
fallback), since BCBToolKit has no equivalent. 8 new tests, 404 total
passing (same pre-existing, unrelated `test_parcitron.py` collection
error). `--backend` stays unexposed (perf knob, no caller-facing reason to
pick it) per spec.

Out of scope for both phases: `--aggregate`/weighted-graded severity input
(needs a graded lesion input entering the pipeline upstream — separate
design decision, not unlocked by this wiring).

## Decided strategy

- **Space strategy for cross-template warps**: warp images (lesion masks, atlases) to
  the target space, never move tractography coordinates.
- **Interpolation**: trilinear (`order=1`) for continuous probability volumes; ANTs
  `genericLabel` (or NN fallback) for binary masks.
- **TDI**: private EBRAINS script/atlas installed at `/opt/tdi` by sysadmin; not
  distributed with BCBlib due to patent restriction.
- **Streamline ratio**: separate CSV from overlap CSV (TRK names are abbreviated, e.g.
  `AF_L`, vs full prob-atlas names, e.g. `Arcuate_Fasciculus_L` — can't merge).
- **ANTs + dipy**: optional deps; `bcblib[ebrains]` installs both for PHI.
- **Branch/release model** (decided 2026-07-31): `main` = stable, always equals the
  latest PyPI release; `dev` = active development + what EBRAINS pulls while conventions
  are unsettled; `devel` = frozen (EBRAINS already updated from it — do NOT commit to it
  again; delete once Paolo's Docker moves off it). PyPI follows `main` **via tags**, never
  a branch. Release = bump `setup.py` version, commit, `git tag vX.Y.Z && git push origin vX.Y.Z`;
  a GitHub Action (`.github/workflows/publish.yml`) builds + publishes on the tag push via
  PyPI Trusted Publishing. EBRAINS path: `@devel` now → `@dev` next update → `pip install bcblib==X`
  once they go live.
- **disconnectome2 engine** (decided 2026-09-09): optional dependency, not a hard
  requirement — private/pre-1.0, no PyPI release yet. `--engine auto` prefers it
  when ready, falls back to BCBToolKit silently; `--engine disco2` fails loudly
  instead of falling back, since a silent swap to the slower engine would be a
  worse surprise than an explicit error. BCBToolKit route stays until disco2 is
  public and two gaps close: arbitrary-target-space output, and offline/air-gapped
  sites with local tracks but no path to fetch a hosted index.

## Open questions / next

- [ ] Await Paolo's confirmation that PHI update succeeded (TRK download + new CSV output)
- [ ] Retire `devel` once Paolo's Docker points at `@dev` — note `devel` still carries the
      `pwll_normalised`/`continuous_dice` bug fixed in `0.7.2`, so flag that to Paolo too
- [ ] Decide, separately, where a graded/severity lesion input would enter
      `iter_bids_lesions`/`preprocess_batch` before `--aggregate` is usable
- [ ] disconnectome2 Phase 1+2 (on `dev`) still pending a decision on when/whether to
      merge into `main` and release — private dependency, no PyPI release needed yet
