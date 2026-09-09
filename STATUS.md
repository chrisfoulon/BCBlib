# STATUS — BCBlib / lesion_features

_Last touched: 2026-09-09_

## Goal

Ship a robust, server-deployable `bcb-lesion-features` pipeline that extracts
per-subject lesion and disconnectome damage profiles across a standard atlas set
(EBRAINS) with optional private enrichment (TDI, streamline ratio).

## State of play

**`devel` branch** is the live branch (shared with external team; cannot be retired yet).
`dev` and `main` were fast-forwarded to match `devel` on 2026-07-31 (both at `758d82b`).

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

**Phase 2 planned, not yet implemented**: pass-through kwargs on
`run_disco2_batch` for disco2's `--fiber-class`/`--out-voxel-size`/
`--len-min`/`--len-max` (all disco2-only, optional, default off). Plan at
`~/.claude/plans/this-repo-s-bcblib-tools-lesion-features-compiled-metcalfe.md`.
Out of scope for both phases: `--backend`, `--aggregate`/weighted-graded
severity input (needs a graded lesion input entering the pipeline upstream —
separate design decision, not unlocked by this wiring).

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

- [ ] **Publish 0.7.0**: (1) add PyPI Trusted Publisher for repo `chrisfoulon/BCBlib`,
      workflow `publish.yml`, environment `pypi`; (2) create GitHub environment `pypi`;
      (3) `git push origin v0.7.0` to trigger the publish Action. (tag exists locally, unpushed)
- [ ] Await Paolo's confirmation that PHI update succeeded (TRK download + new CSV output)
- [ ] Retire `devel` once Paolo's Docker points at `@dev`
- [ ] Implement disconnectome2 Phase 2 (fiber_class/out_voxel_size/len_min/len_max
      pass-through) per the plan file referenced above
- [ ] Decide, separately, where a graded/severity lesion input would enter
      `iter_bids_lesions`/`preprocess_batch` before `--aggregate` is usable
