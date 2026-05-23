# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dev install (core + web + dev + docs)
pip install -e ".[all]"

# Tests
pytest tests/ -v                              # full suite
pytest tests/test_homogenization.py -v        # one module
pytest tests/test_fe_solver.py::TestName::test_x -v   # one test

# Lint / type-check (matches .github/workflows/tests.yml)
ruff check .
mypy porosity_fe porosity_fe_analysis.py
# The mypy job in CI pins `numpy<2` because the type stubs were validated
# against numpy 1.x; numpy 2.x stubs surface unrelated errors. Runtime
# stays unpinned. Install `pip install "numpy<2"` before running mypy
# locally if you want to reproduce the CI signal.

# Streamlit web app (requires the `web` extra)
pip install -e ".[web]"
streamlit run app.py                          # http://localhost:8501

# Batch analysis CLI (entry points from pyproject.toml [project.scripts])
porosity-analyze                              # sweep 5 Vp × 5 configs, PNG + JSON
validate_porosity --help                      # run vs. bundled validation datasets

# Standalone CLI executable (PyInstaller)
python -m PyInstaller ValidatePorosity.spec --noconfirm --clean

# Docs (Sphinx)
cd docs && make html                          # output under docs/_build/html
```

## Architecture

### Package layout (post #119 split)

The implementation lives in the `porosity_fe/` package. The top-level
`porosity_fe_analysis.py` is a 62-line **compatibility shim** that re-exports
everything from `porosity_fe` so legacy `from porosity_fe_analysis import X`
callers keep working. **New code should import from `porosity_fe` directly**;
do not add new symbols to `porosity_fe_analysis.py`.

The package re-exports its full public API from `porosity_fe/__init__.py` — the
`__all__` list there is the authoritative public surface and what the Sphinx
API page documents. Adding a public symbol means adding it both to its
module and to `__init__.py` (re-export + `__all__` entry).

### Two solver paths, one mesh

Both solvers consume the same `PorosityField` + `CompositeMesh` + `MaterialProperties`
triple but apply degradation differently. **They are designed to give
different numerical answers for the same inputs**, not the same answer:

- `EmpiricalSolver` (`porosity_fe/empirical.py`) — closed-form
  knockdown applied **once at the laminate level** using the
  specimen-average `Vp`. The distribution shape (`uniform` vs `clustered`
  vs `interface`) has **no effect** here because all are renormalized to
  the same mean. Layup enters via a matrix-dominated fraction `f_md`
  computed from ply angles (`alpha_eff = alpha_QI · (f_md / 0.5)`, with
  floors `_F_MD_FLOOR` / `_F_MD_FLOOR_ILSS`).
- `FESolver` (`porosity_fe/fe/solver.py`) — builds a hex8 mesh and
  applies stiffness degradation **per element** via Eshelby/Mori-Tanaka
  micromechanics on the local `Vp(x, y, z)`. Strength degradation uses a
  heuristic `strength ~ sqrt(stiffness_retention)` scaling (see
  `_degraded_strengths`). The FE path **does** pick up distribution-shape
  differences.

When changing one solver, check whether the matching behavior in the
other is intentional or needs to move in lockstep — they're independent
implementations of the same physical claim.

### `Vp` is always a fraction, never a percent

`PorosityField.__init__` enforces `Vp ∈ [0, 1]` and emits a percent-vs-fraction
hint when the value looks like a percent (`Vp ≥ 1.001`). Any new API that
accepts porosity input must keep that convention; plotting may multiply by
100 for display only. See "Inputs and Conventions" in README.md for the
full table.

### Ply-angle resolution

`ply_angles` is a single string sentinel (`'QI'` / `'UD'`) or an explicit
list, resolved through `porosity_fe._ply_angles._resolve_ply_angles`.
`EmpiricalSolver`, `CompositeMesh`, and `FESolver` all funnel through this
helper to avoid divergent defaults (the bug #44 item 2 fixed). Don't
re-implement sentinel handling at a call site.

### Sweep orchestration

`porosity_fe/pipeline.py` owns the multi-config sweep. `_analyze_one` is
the parallel worker invoked by `compare_configurations`; `_DEFAULT_MESH_RES
= (30, 10, 12)` is the single source of truth for production mesh size.
`build_empirical_pipeline` is the canonical factory used by examples,
tests, and the UQ helper — when changing mesh defaults or ply-angle
handling, change it here, not at the call sites.

### Validation database

`validation/datasets/*.json` holds 13 peer-reviewed experimental
datasets, schema-validated against
`validation/schemas/validation_dataset_schema.json`. The whole tree is
gitignored (digitized from published figures, kept out of the repo); the
PyInstaller spec bundles it into the CLI executable so end users get an
offline-runnable validator. `validate_porosity_cli.py` calls
`validation/validate_all.py` to run every dataset through the empirical
pipeline and emit `validation_master_report.png` + `validation_detail_report.md`.

### Streamlit app bootstrap

`app.py` calls `matplotlib.use("Agg")` and adjusts `sys.path` **before**
importing `pyplot` and sibling modules — this is why
`pyproject.toml` adds `E402` / `I001` to `[tool.ruff.lint.per-file-ignores]`
for `app.py`. Don't reorder those imports.

### Test conftest layout

The repo-root `conftest.py` adjusts `sys.path` so tests can find
`porosity_fe`, `app`, `validate_porosity_cli`, and `validation` without an
editable install (CI installs deps but historically not the package
itself). `tests/conftest.py` adds the `_restore_porosity_logger`
autouse fixture that undoes `_configure_cli_logging`'s
`propagate = False` between tests — required because alphabetical
collection puts `test_cli.py` first.

## Units and conventions

- **Stiffnesses and strengths: MPa. Thicknesses: mm.** Documented at the
  API surface; look for the `Notes` block on
  `MaterialProperties.get_stiffness_matrix`, `Hex8Element.B_matrix`,
  `FieldResults`, and `EmpiricalSolver.apply_loading` for Voigt order,
  engineering vs. tensor strain, and compression-sign conventions.
- **Tsai-Wu `F_12`**: default is Tsai's `F_12 = -0.5·√(F_11·F_22)`; users
  can override via `MaterialProperties(tsai_wu_F12=...)` (must be in
  `[-1, 0]`).
- **Calibration scope**: empirical coefficients are validated to
  `Vp ≲ 0.05`; flag extrapolations explicitly.

## CI matrix

`.github/workflows/tests.yml` runs three jobs: `lint` (ruff + mypy on
Python 3.12 with `numpy<2`), `test` (pytest on `{ubuntu, macos, windows}` ×
`{3.10, 3.11, 3.12, 3.13}`), and `streamlit_smoke` (decoupled
single-OS/Python import-only check — Streamlit wheel availability on 3.13
can't be allowed to drop the whole library matrix red, see issue #157).
`security.yml` runs `pip-audit` weekly. Match the matrix when adding
version-sensitive code.

## Adding materials and empirical coefficients

- **New material preset**: add a `MaterialProperties(...)` entry to the
  `MATERIALS` dict in `porosity_fe/materials.py`. All orthotropic
  constants are required — there are no defaults. Both the FE
  micromechanics path and the empirical path read from the same dataclass,
  so populate constituent fields (`matrix_modulus`, `fiber_volume_fraction`,
  etc.) even if you only plan to exercise the empirical solver.
- **New empirical correlation**: add to `EmpiricalSolver`. Take `Vp` as a
  fraction in `[0, 1]`, return `KD ∈ (0, 1]`. Document the calibration
  set, the regression form (`ln(KD)` vs `Vp` for Judd-Wright,
  `ln(KD)` vs `ln(1-Vp)` for power law), and the validity bound. Custom
  per-mode overrides go through the keyword-only `judd_wright_alpha=` /
  `power_law_n=` / `linear_beta=` constructor args, which layer on top of
  the QI defaults with the same `f_md` scaling.
