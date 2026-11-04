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
# In Claude Code on the web containers, `pytest` is uv-isolated and
# cannot see the project deps installed by the SessionStart hook. Use
# `python -m pytest ...` there.

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

# Bit-identity fingerprint of the FE results (132 production cases; run
# both sides with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1, see the script)
python scripts/fingerprint.py <repo_root> out.json [--quick]
python scripts/fingerprint.py --compare base.json out.json

# The numeric FE regression gate CI runs (job `fe_fingerprint`, 30 cases,
# rtol 1e-9). A PR that changes FE results on purpose regenerates the
# baseline single-threaded and says why in the CHANGELOG:
python scripts/fingerprint.py --check scripts/fingerprint_baseline.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/fingerprint.py --write-baseline scripts/fingerprint_baseline.json
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
  the same mean. Layup enters through `EmpiricalSolver._layup_scales()`
  (`porosity_fe/_layup.py`, IMPROVEMENT_PLAN 2.7): `shear` / `ilss` /
  `transverse_tension` are layup-independent, and `tension` /
  `compression` scale by `max(1, CLT-energy blend / alpha_QI)`, so QI
  layups are unchanged and a scale above 1 is flagged as unvalidated. The
  old `f_md` / `F_MD_*` constants are deprecated and drive nothing.
- `FESolver` (`porosity_fe/fe/solver.py`) — builds a hex mesh of
  incompatible-mode bricks (`formulation='hex8i'`, the default; plain
  `'hex8'` stays selectable), eliminates the Dirichlet BCs exactly, and
  applies stiffness degradation **per element** via Eshelby/Mori-Tanaka
  micromechanics on the local `Vp(x, y, z)`. Assembly and stress recovery
  run on batched per-element arrays (`porosity_fe/fe/batch.py`), not
  per-element `Hex8Element` objects (kept as the public reference
  implementation and tested for agreement); the assembled `K` and its LU
  factorization are cached on the assembler / solver and reused across
  solves until the mesh, material or porosity change. Every factorization
  is estimated first and refused over `FESolver(max_memory_gb=)` (default
  80 % of RAM; `fe/_memory.py`, the model `refinement_study` plans with),
  and the knockdown's pristine reference follows `solver`, so `'cg'`
  factorizes nothing. Strength degradation uses a
  heuristic `strength ~ sqrt(stiffness_retention)` scaling (see
  `degraded_strengths` in `fe/failure.py`). The FE path **does** pick up
  distribution-shape differences. `solver.py` holds only the
  orchestration; the steps live in private `fe/` modules: `_load_cases`
  (loading modes, checks, BC dispatch), `_constrained_solve` (elimination,
  LU cache, CG), `_knockdown` (pristine reference, solved before the
  porous model so one factorization is alive at a time), `_thermal`
  (unit-`delta_T` solve, factorization freed once cached), `_postprocess`
  (recovery, failure metrics); `FieldResults` is in `fe/results.py`.

When changing one solver, check whether the matching behavior in the
other is intentional or needs to move in lockstep — they're independent
implementations of the same physical claim.

### Delamination and fracture mechanics (FE only)

Porosity is smeared into the stiffness; a delamination is an explicit
crack. `Delamination` (`porosity_fe/delamination.py`) is a strip,
rectangle or ellipse at interface `k` (between plies `k` and `k + 1`).
`CompositeMesh(..., delaminations=...)` duplicates that interface's node
plane, appending the duplicates after the `n_structured_nodes` structured
nodes, and ties every pair outside the delamination (`bonded_ties`,
exact master-slave elimination in `_ConstraintReduction`; a tied
duplicate is never prescribed). Structured helpers must slice
`[:n_structured_nodes]`. Every pair bonded gives the plain mesh's solution
to round-off, and the pristine reference always has every pair bonded.
On top of the solve:
- `fe/_contact.py`: frictionless crack-face contact, a primal-dual active
  set that changes only the ties (one extra LU cache entry).
- `fe/vcct.py` / `fe/_fracture.py`: VCCT energy release rates at the
  front (only the mode I product is clipped at zero; the shear products
  keep their sign), the Benzeggagh-Kenane toughness with Reeder's mode III
  term, and the critical load factor `lambda_c` (quadratic with a thermal
  prestress). The toughness at each front node is knocked down by the local
  porosity (`porosity_fe/toughness.py`, `ToughnessKnockdown`); the default
  exponential law has **uncalibrated placeholder** coefficients and every
  result it changes carries `toughness_uncalibrated = True`.
- `fe/interlaminar.py`: interface traction recovery and the stress-based
  Brewer-Lagace / Ye onset criteria (post-processing only);
  `porosity_fe/edge_delamination.py`: O'Brien's free-edge G (lamination
  theory).
- `porosity_fe/fracture_specimens.py` and `CompositeMesh.for_dcb` /
  `for_enf` / `for_mmb` with the `'dcb'` / `'enf'` / `'mmb'` loadings:
  the fracture specimens, verified against Krueger's NASA benchmarks in
  `tests/test_fracture_specimens.py` (the slow ones behind
  `POROSITY_FE_FRACTURE_BENCH=1`).

The empirical path has no delamination. The linear VCCT load factor is
growth *onset* only: no propagation, cohesive zone or buckling.
`DELAMINATION_PLAN.md` holds the plan, its status and the open decisions.

### `Vp` is always a fraction, never a percent

`PorosityField.__init__` enforces `Vp ∈ [0, 1]` and emits a percent-vs-fraction
hint when the value looks like a percent (`Vp ≥ 1.001`). Any new API that
accepts porosity input must keep that convention; plotting may multiply by
100 for display only. The only percent inputs, each labelled as such, are the
Streamlit "Void content Vp (%)" field, the CLI's `--vp-pct`, and
`reporting.recommend_disposition(Vp_percent, ...)` (NCR convention). See "Inputs and Conventions" in README.md for the
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
= (30, 10, None)` is the single source of truth for production mesh size:
`nz = None` means one element layer per ply, capped at `_DEFAULT_NZ_MAX`
(48, the `nz_max` of `build_empirical_pipeline`), resolved by
`_default_mesh_resolution(n_plies, ply_angles)`, which the pipeline, the
app and `refinement_study` all call (never read the tuple directly). Above
the cap it takes the largest `nz` whose element layers each cover plies of
one angle (the rule of `CompositeMesh.layup_discrepancies`), else the cap
with a warning.
`build_empirical_pipeline` is the canonical factory used by examples,
tests, and the UQ helper — when changing mesh defaults or ply-angle
handling, change it here, not at the call sites.

### Validation database

`validation/datasets/*.json` holds 18 experimental datasets (15
strength and modulus, plus 3 interlaminar-toughness ones,
`hakim_2017`, `agius_fox_2015` and `olivier_2007`, whose MAE is reported apart from the
strength headline), schema-validated against
`validation/schemas/validation_dataset_schema.json`. The datasets are
tracked in the repository (it is where they are backed up), and CI runs
the validation tests against them. The PyInstaller spec bundles them into
the CLI executable so end users get an offline-runnable validator. `validate_porosity_cli.py` calls
`validation/validate_all.py` to run every dataset through the empirical
pipeline and emit `validation_master_report.png` + `validation_detail_report.md`.

### Streamlit app bootstrap

The app lives in the `porosity_app/` package (#224): `config` (sidebar
keys, defaults, cache key, labels), `runners` (`run_analysis`, the FE solve,
the cached wrappers), `summaries` (result tables), `convergence`, `plots`,
`sidebar`, `tabs/` (one module per tab plus `common`) and `render`
(`_render`). A module imports only modules of a lower layer, in that
order (`tests/test_app_package.py`). `app.py` is the entry point
(`streamlit run app.py`): it calls `matplotlib.use("Agg")` and adjusts
`sys.path` **before** importing `porosity_fe` and `porosity_app` — this is
why `pyproject.toml` adds `E402` / `I001` to
`[tool.ruff.lint.per-file-ignores]` for `app.py`; don't reorder those
imports. It re-exports every `porosity_app` name, so `app.X` keeps working;
new code imports from `porosity_app`, and a monkeypatch targets the module
that looks the name up (`porosity_app.runners.FESolver`, not `app.FESolver`,
which patches nothing). The app tests skip in the core CI matrix (no
Streamlit there) and run in the `streamlit_smoke` job.

### Test conftest layout

Tests import `porosity_fe`, `app`, `validate_porosity_cli`, and
`validation` from the installed package, so run them after
`pip install -e ".[dev]"` (CI does the same; there is no repo-root
`sys.path` shim). `tests/conftest.py` adds the `_restore_porosity_logger`
autouse fixture that undoes `_configure_cli_logging`'s
`propagate = False` between tests — required because alphabetical
collection puts `test_cli.py` first.

## Units and conventions

- **Stiffnesses and strengths: MPa. Thicknesses: mm. Energy release
  rates and toughness: N/mm (= kJ/m²).** Documented at the
  API surface; look for the `Notes` block on
  `MaterialProperties.get_stiffness_matrix`, `Hex8Element.B_matrix`,
  `FieldResults`, and `EmpiricalSolver.apply_loading` for Voigt order,
  engineering vs. tensor strain, and compression-sign conventions.
- **Tsai-Wu `F_12`**: default is Tsai's `F_12 = -0.5·√(F_11·F_22)`; users
  can override via `MaterialProperties(tsai_wu_F12=...)` (must be in
  `[-1, 0]`).
- **Interlaminar strengths**: optional `sigma_3t` (`Z_t`) and `tau_13`
  (`S_13`); when `None` every criterion keeps its stand-in (`sigma_2t`;
  `tau_12` in Tsai-Wu and maximum stress, `tau_ilss` in Hashin), so the
  default results do not change.
- **Calibration scope**: empirical coefficients are validated to
  `Vp ≲ 0.05`; flag extrapolations explicitly.

## CI matrix

`.github/workflows/tests.yml` runs four jobs: `lint` (ruff + mypy on
Python 3.12 with `numpy<2`), `test` (pytest on `{ubuntu, macos, windows}` ×
`{3.11, 3.12, 3.13, 3.14}` against an editable install; the ubuntu/3.12
cell also measures statement and branch coverage with a 95 % floor
(`--cov-fail-under`), uploads `coverage.xml` as an artifact, and
runs the production-mesh FE timing guard via `POROSITY_FE_BENCHMARK=1`),
`fe_fingerprint` (the FE regression gate of `scripts/fingerprint.py` on
ubuntu/3.12, single threaded, on the numpy / scipy recorded in the
baseline, run on the runner's BLAS kernel and again on Sandybridge's; see
Commands; a case whose fields are round-off by physics names them in
`_CASE_OPTIONS`), and `streamlit_smoke` (decoupled
single-OS/Python job: the `app` import and the app tests, which the core
matrix skips without Streamlit — Streamlit wheel availability on a new
Python can't be allowed to drop the whole library matrix red, see issue
#157). The floor is Python 3.11 (`requires-python`, ruff `target-version`,
mypy `python_version`; 3.10 reached end of life in October 2026, #216).
`security.yml` runs `pip-audit` weekly, and Dependabot proposes action and
pip updates (`.github/dependabot.yml`). Match the matrix when adding
version-sensitive code.

## Adding materials and empirical coefficients

- **New material preset**: add a `MaterialProperties(...)` entry to the
  `MATERIALS` dict in `porosity_fe/materials.py`. All orthotropic
  constants are required — there are no defaults. Both the FE
  micromechanics path and the empirical path read from the same dataclass,
  so populate constituent fields (`matrix_modulus`, `fiber_volume_fraction`,
  etc.) even if you only plan to exercise the empirical solver. The CTEs,
  the interlaminar strengths and the toughness (`G_Ic`, `G_IIc`,
  `G_IIIc`, `bk_eta`) are optional and default to `None`; set them only
  from a cited source.
- **New empirical correlation**: register the law function in
  `_KNOCKDOWN_LAWS` (`porosity_fe/empirical.py`), the one place every
  solver path looks laws up; add its QI table to `Calibration`, its name to
  `KnockdownModel`, and a `local_sensitivities` branch. Take `Vp` as a
  fraction in `[0, 1]`, return `KD ∈ (0, 1]`. Document the calibration
  set, the regression form (`ln(KD)` vs `Vp` for Judd-Wright,
  `ln(KD)` vs `ln(1-Vp)` for power law), and the validity bound. Custom
  per-mode overrides go through the keyword-only `judd_wright_alpha=` /
  `power_law_n=` / `linear_beta=` constructor args, which layer on top of
  the QI defaults with the same layup scaling.
