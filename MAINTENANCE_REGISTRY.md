# Maintenance Registry

Technical debt tracking for BCBlib. Items are ordered by impact.

---

## High Priority

### Layered design absent in `bcblib.imaging.stats` and `bcblib.imaging.math`

- **Files**: `bcblib/imaging/stats.py`, `bcblib/imaging/math.py`, `bcblib/imaging/manipulate.py`
- **Issue**: All public functions load a `NiftiLike` and operate on it in a single layer.
  No underlying numpy-level functions exist, so the core logic cannot be reused with
  pre-loaded arrays. Newly added functions (`fraction_covered`, `weighted_region_mean`)
  use the correct layered pattern and serve as reference implementations.
- **Impact**: Code duplication when damage_profile (and any future tool) needs to operate
  on already-loaded arrays. Performance cost from redundant I/O.
- **Proposed fix**: For each public function `fn(img, ...)`, extract a
  `_fn_array(data, ...)` core. Full plan in `docs/imaging_stats_layered/plan.md`.
- **Estimated effort**: ~2 days for stats.py + math.py + manipulate.py
- **Discovered**: 2026-05-13 during damage_profile implementation

---

## Deferred (not blocking current work)

### Pre-existing flake8 violations in legacy code

- **Files**: most of `bcblib/tools/`, `bcblib/imaging/` (pre-dating this branch)
- **Issue**: ~1100 E501 (line length), ~30 F401 (unused imports), various style issues.
  Added `setup.cfg` with `max-line-length = 120` to eliminate spurious E501 false
  positives on lines that are stylistically acceptable. True style violations remain.
- **Impact**: cosmetic; no functional issues
- **Action**: clean up incrementally following the Boy Scout Rule

### `_check_image_space` import from `best_overlap` internal namespace

- **File**: `bcblib/tools/damage_profile/_space.py` line 8
- **Issue**: Imports a private function from a sibling tool's internal namespace.
  If `best_overlap` is refactored, this import will break.
- **Impact**: low (both modules are in the same package; coupling is visible)
- **Proposed fix**: move `_check_image_space` to a shared `bcblib.imaging.space` module.

---

## Recently Completed

- Added `setup.cfg` with `max-line-length = 120` to codify actual project line-length
  standard (pre-existing code already exceeded 79 chars; 2026-05-13).
- Implemented layered pattern for `fraction_covered` and `weighted_region_mean`
  in `bcblib/imaging/stats.py` as reference for future refactor (2026-05-13).
