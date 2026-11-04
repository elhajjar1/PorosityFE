# Deprecations removed in 2.0

Everything below still works in 1.x and is removed in 2.0. The ones that
warn emit a `DeprecationWarning` naming 2.0 when they are used; the others
are private names or documented aliases that warn nothing. Importing the
package emits none of them. To find what your code still uses, run it
(or its tests) with `python -W error::DeprecationWarning`.

`tests/test_deprecations.py` holds the same list. Before 2.0 it checks that
every entry still works (so 1.x removes nothing early) and that every
`DeprecationWarning` in the code belongs to an entry. From version 2.0 on
it fails while an entry is left. Each removal gets a CHANGELOG "Removed"
entry and loses its row here.

| Id | Deprecated | Use instead | Warns | Reference |
|---|---|---|---|---|
| `f-md-constants` | `Calibration.F_MD_REF`, `F_MD_FLOOR`, `F_MD_FLOOR_ILSS` and the `EmpiricalSolver._F_MD_*` aliases: the retired `f_md / 0.5` layup rule, without effect | `EmpiricalSolver.layup_scale` | yes | IMPROVEMENT_PLAN 2.7 |
| `calibration-aliases` | `EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI`, `_POWER_LAW_N_QI`, `_LINEAR_BETA_QI`, `_MATRIX_DOMINATED_MODES`; `porosity_fe._UQ_DEFAULT_PERCENTILES` (private) | the `Calibration` attributes of the same names without the underscore | no | #121 |
| `solver-minres` | `FESolver.solve(solver='minres')`, `refinement_study(solver='minres')` | `solver='cg'`: `K_ff` is SPD and CG is more accurate at the same `rtol` | yes | CHANGELOG, Unreleased |
| `penalty-diag-scale` | `FESolver.solve(penalty_factor=..., diag_scale=...)`, without effect | nothing: the BCs are imposed exactly by elimination | yes | CHANGELOG, Unreleased |
| `apply-penalty` | `BoundaryHandler.apply_penalty` | `BoundaryHandler.apply_elimination` | yes | CHANGELOG, Unreleased |
| `ply-angles-none` | `ply_angles=None` in `EmpiricalSolver`, `CompositeMesh` and `FESolver` | `'QI'`, `'UD'` or an explicit list | yes | #44 |
| `provenance-aliases` | the JSON provenance keys `package_version`, `python`, `numpy`, `scipy`, `generated_utc`, `git_sha` (marked `deprecated` in the results schema) | `porosity_fe_version`, `python_version`, `numpy_version`, `scipy_version`, `timestamp_utc`, `git_commit`; schema 2.0 | no (written beside the canonical keys) | IMPROVEMENT_PLAN 4.7 |
| `result-dict-access` | dict-style access to `FailureResult` and `ConfigResult` (`result['knockdown']`, `.get()`, `.keys()`) | attribute access (`result.knockdown`) | no (documented) | #148 |
| `resolve-material-strict` | `validation.validate_all.resolve_material(strict=...)`, ignored | nothing: an unknown `(fiber, matrix)` pair always raises `KeyError` | yes | #34 |
| `mesh-dimension-assignment` | assigning `CompositeMesh.L_x`, `L_y` or `x_grid` after the mesh is generated (marks it stale; `FESolver` refuses a stale mesh) | `CompositeMesh(..., length=, width=, x_grid=)` | yes | #225 |

## Open decisions for 2.0

These are not deprecated. Whether they go is decided at 2.0 (#218):

- **`porosity_fe_analysis.py`**, the compatibility shim that re-exports
  `porosity_fe` for `from porosity_fe_analysis import ...` callers. It is
  not marked for removal.
- **`EmpiricalSolver.f_md`**, the legacy binned matrix-dominated
  fraction. It is kept for display and no longer drives the layup scaling.
- **The silent entries above** (`calibration-aliases`,
  `result-dict-access`). They could start to warn in a last 1.x release,
  so that callers see them before they go.
