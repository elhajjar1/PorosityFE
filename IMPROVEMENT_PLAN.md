# PorosityFE Improvement Plan

Prepared 2026-07-16 from a full scan of the codebase: every module in
`porosity_fe/`, the Streamlit app, both CLIs, the validation runner, the test
suite, CI workflows, packaging metadata, and docs. Findings below were
cross-checked against the current tree; measured numbers come from running the
gates and profiling the FE solver in this environment.

## Current state (measured)

The project is in very good health, which shapes what "improvement" means here:

| Gate | Result |
|---|---|
| `python -m pytest tests/` | 832 passed, 1 skipped |
| `ruff check .` | clean |
| `mypy porosity_fe porosity_fe_analysis.py` | clean (even on numpy 2.4) |
| Line coverage (`pytest --cov`) | **97%** overall; no module below 90% |
| FE solve, production mesh 30×10×12 (3,600 elements) | **~18–24 s per load case** |

The FE timing is the standout: profiling one `FESolver.solve(loading='tension')`
shows the sparse linear solve itself takes **2.0 s**, while Python-loop element
assembly takes **11.5 s** and stress recovery **9.2 s** (which re-instantiates
every element and recomputes everything assembly already computed). There are
~200,000 calls to `np.linalg.inv` on small matrices in a single solve, and a
second `solve()` on the same solver rebuilds and refactorizes everything from
scratch (measured: no speedup on repeat solves).

So the plan prioritizes: (1) FE performance, (2) a small set of real
correctness/physics-fidelity gaps, (3) high-value missing features, (4) app/CLI
polish, (5) architecture cleanups, and (6) CI/packaging/docs/hygiene. Items are
tagged **[S]**mall (≤ half day), **[M]**edium (1–3 days), **[L]**arge (multi-day
/ design needed).

---

## Workstream 1 — FE solver performance (biggest measurable win)

Target: bring a production-mesh solve from ~20 s down to ~2–3 s (solver-bound),
and make multi-load-case runs nearly free after the first. Ordered by payoff:

### 1.1 Stop recomputing everything during stress recovery — [M]
`FESolver._recover_stresses` (`porosity_fe/fe/solver.py:816-877`) calls
`assembler.create_element(e)` per element, re-running porosity validation and
`_degraded_stiffness` at all 8 Gauss points — all computed during assembly and
discarded. Profile attributes ~9 s of the 24 s here. Cache per-element `B`
matrices and degraded `C` during assembly (keyed like the existing `_ke_cache`)
and reuse them in recovery. Also vectorize the per-Gauss-point local-frame
rotation (`solver.py:873-875`): `sig_local = sig_g @ T_sigma.T` over the whole
`(8, 6)` block, with `T_sigma`/`T_eps` built once per unique ply angle rather
than per element.

### 1.2 Reuse the assembled K and its factorization across load cases — [M]
`solve()` re-assembles and re-factorizes on every call (`fe/solver.py:505`,
`fe/assembler.py:149`), yet K is identical for compression/tension/shear on the
same mesh/material/porosity — only BCs and RHS differ. Cache K on the assembler
(invalidate on mesh/material/porosity change) and expose
`scipy.sparse.linalg.splu` / `factorized` so repeated solves reuse the
factorization. The Streamlit app and `compare_configurations` sweeps benefit
directly.

### 1.3 Vectorize `_compute_knockdown` — [S]
`fe/solver.py:922-932` is an O(n_elem × n_gp) Python double loop that calls
`rotate_stiffness_3d` per element. Ply angles take a handful of unique values:
precompute one rotated C per angle, then a single
`np.einsum('j,egj->eg', C_row, strain_global)` over the full array.

### 1.4 Batch the element stiffness math — [L]
`Hex8Element.stiffness_matrix` / `B_matrix` / `shape_derivatives`
(`fe/element.py:132-285`) are pure-Python per-Gauss-point loops; ~200k small
`np.linalg.inv` calls per solve. Two options, in increasing effort:
(a) vectorize across Gauss points within an element (stack the 8 Jacobians and
use batched `np.linalg.inv` on a `(8,3,3)` array); (b) vectorize across
elements sharing geometry+angle groups (the mesh is structured, so per-layer
element geometry is uniform — the existing `_ke_cache` already exploits this
for Ke, but B/J recomputation doesn't). Option (a) alone should cut assembly
time several-fold with no architectural change.

### 1.5 Small assembly wins — [S]
- `_element_cache_key` is computed twice per element (`fe/assembler.py:113-136`
  pre-pass, then again at `:180`). Fold into one lazy pass.
- `element_dof_indices` (`fe/assembler.py:138-147`): replace the loop with
  `(3*node_ids[:,None] + np.arange(3)).ravel()`.
- `apply_penalty` (`fe/assembler.py:495-505`): replace the LIL round-trip +
  Python loop with a vectorized sparse diagonal update.

### 1.6 Replace penalty BCs with direct elimination — [L]
The penalty method (`fe/assembler.py:458-505`) is the root cause of the
conditioning machinery the code has accrued (diag-ratio logging, Jacobi
pre-scaling, penalty tuning at `fe/solver.py:679-743`) and of the ~1e-4
CG-vs-direct tolerance in tests. Partitioning into free/constrained DOFs and
solving `K_ff u_f = F_f − K_fc u_c` enforces Dirichlet BCs exactly, keeps
cond(K) physical, makes iterative solvers viable, and lets the workaround code
retire. Do this after 1.1–1.4 so before/after benchmarks are clean.

### 1.7 Guard total mesh size, not per-axis — [S]
`_MAX_ELEMENTS_PER_AXIS = 10_000` (`mesh.py:129`) permits `10_000³` elements;
the COO assembler pre-allocates `n_elem × 576` entries eagerly
(`fe/assembler.py:163-166`) → instant OOM. Add a total-element / estimated-RAM
cap in `CompositeMesh.__init__`.

---

## Workstream 2 — Correctness and physics fidelity

### 2.1 Extrapolation warning: mislabeled and blind to local peaks — [S] (verified)
`EmpiricalSolver._warn_if_extrapolated` (`porosity_fe/empirical.py:362-370`)
assigns the specimen **mean** Vp to a variable named `vp_max` and prints
"max Vp = …". Worse, `apply_loading` evaluates knockdowns at each node's
*local* Vp, and for `clustered`/`interface` distributions the local peak can be
several × the mean — so the nodal field can be deep in the extrapolated regime
with no warning while the mean sits below 0.05. Fix the label, and also check
`Vp_arr.max()` against `_VP_CALIBRATION_MAX` for the nodal path (warn once per
call).

### 2.2 Hashin criterion is blind to interlaminar shear under ILSS — [M]
`_evaluate_hashin` (`fe/solver.py:1242-1319`) uses only σ11, σ22, τ12. The ILSS
load case exists specifically to produce τ13/τ23, so
`loading='ilss', failure_criterion='hashin'` returns an index blind to the
governing stress (`max_stress` handles out-of-plane shear; the inconsistency is
criterion-specific). Minimum fix: warn or refuse the hashin+ilss pairing.
Better: add the 3D delamination mode (quadratic interlaminar term in τ13/τ23
vs S13/S23).

### 2.3 Unify the three inconsistent "void element" notions — [M]
The assembler flags voids geometrically (centroid inside a `VoidGeometry`,
`fe/assembler.py:62`) and assigns ~1 MPa stiffness; `_evaluate_failure` skips
only elements with nodal-mean Vp > 0.95 (`fe/solver.py:1122`);
`_degraded_stiffness` clamps Vp at 0.99 (`fe/element.py:221`). A geometric void
with low nodal porosity is still run through the failure polynomial with
near-pristine strengths; a high-Vp non-geometric element is silently dropped
from the max-FI search. Define one shared void/degradation threshold constant
and have `_evaluate_failure` consult `mesh.void_element_set`.

### 2.4 FE knockdown metric: signed-mean instability and silent clamp — [M]
`_compute_knockdown` averages *signed* stresses over the domain then takes
`abs(mean)/abs(pristine mean)` and silently clamps to ≤ 1.0
(`fe/solver.py:915-940`). For sign-changing fields (ILSS bending) the signed
mean can approach zero. Preferred fix (pairs with 3.2 below): compute the
knockdown from reaction-force–derived effective modulus, which is robust and
physical. If the clamp stays, log when the raw ratio exceeds 1 instead of
silently truncating.

### 2.5 `_degraded_composite_stiffness` assumes an isotropic MT tensor — [M]
`homogenization.py:245-251` extracts `mu_eff = C_eff[3,3]`,
`lam_eff = C_eff[0,1]` — valid only for spherical voids; for
cylindrical/penny `void_shape_radii` the Mori-Tanaka result is anisotropic and
this silently mischaracterizes the degraded matrix. Currently masked because
the public CLT entry points always pass spherical, but the parameter is live.
Either assert spherical-only or derive effective isotropic moduli from tensor
invariants (`mu_eff = (C[3,3]+C[4,4]+C[5,5])/3`, etc.).

### 2.6 Replace ad-hoc SCF heuristics with elasticity-based values — [M]
`stress_concentration_factor` (`void_geometry.py:144-162`) uses uncited
piecewise-linear-in-aspect-ratio SCFs that grow unbounded (penny ar=10 →
SCF 17 in transverse tension). Classical closed forms exist
(Kirsch/Inglis; Eshelby-based cavity SCFs). At minimum cap and cite the
heuristic. Also: unify the void-shape classification thresholds —
`homogenization.py:101` calls radii within 1% a sphere while
`void_geometry.py:148` uses aspect ratio < 1.2, so the same void can be
"spherical" to stiffness but not to strength.

### 2.7 Layup scaling rule (`f_md/0.5`) has known ~33% error — [L]
`_layup_scale` (`empirical.py:322-348`, TODO #140) and the uncalibrated floors
`F_MD_FLOOR`/`F_MD_FLOOR_ILSS` (#139) are the largest physics-fidelity gap in
the empirical path (documented error up to 33.5% vs a CLT proxy for UD-heavy
layups). Replace the linear rule with a CLT-derived `f_md → scale` map
calibrated against the same reference set used to justify the floors; close
#139/#140 together.

### 2.8 Smaller physics items — [S] each
- **Fatigue `R` is accepted but inert** (`fatigue.py:98-149`): add a
  Goodman/Walker mean-stress correction, or at least warn on non-default `R`.
- **Hardcoded isotropic fiber** (`homogenization.py:258-259`, `nu_f = 0.2`):
  add optional `fiber_poisson`/`fiber_shear_modulus` to `MaterialProperties`.
- **G23 reuses the G12 Halpin-Tsai form byte-for-byte**
  (`homogenization.py:278-286`): use a G23-appropriate ξ or document the
  intentional equality and drop the redundant recompute.
- **Silent triaxial→prolate fallback** (`homogenization.py:123-128`): emit a
  `UserWarning` when taken.
- **`perturb` can crash a UQ sweep** when a lognormal draw pushes
  `fiber_volume_fraction` past 1.0 (`materials.py:471-486`): clip bounded
  fields in `_perturbed_value`.
- **Mesh quality check tests detJ only at the element center**
  (`mesh.py:417-419`) while assembly enforces positivity at all 8 Gauss points
  (`fe/element.py:270`): check all GPs so diagnostics match enforcement.
- **`effective_porosity_profile` hardcodes (25, 10) mm sample coordinates**
  (`porosity_field.py:261-263`): accept `(x, y)` parameters or move the method
  to `CompositeMesh` where domain extents are known.

---

## Workstream 3 — Missing capabilities users will actually want

### 3.1 Expose `sigma_2c`: add a `transverse_compression` loading mode — [S]
`MaterialProperties.sigma_2c` is stored and validated (`materials.py:133,187`)
but unreachable — `EmpiricalSolver.PRISTINE_STRENGTH_KEY`
(`empirical.py:170-174`) has no `transverse_compression` entry. Add the key and
coefficient-table entries (or document the exclusion).

### 3.2 Reaction forces and effective modulus from FE — [S]
Displacement-controlled solves never recover reactions (`R = K @ u` at
constrained DOFs is essentially free post-solve). Reaction sum ÷ face area ÷
applied strain gives the effective modulus directly — the robust knockdown
basis 2.4 needs, and a headline number users ask for.

### 3.3 First-ply-failure load factor / margin of safety — [S]
Failure indices are reported (`fe/solver.py:541-568`) but never inverted to a
load. For linear analysis the factor is the positive root of the Tsai-Wu
quadratic (or `1/max_FI` for max-stress/Hashin). Surface it on `FieldResults`.

### 3.4 Surface UQ in the CLI and app — [M]
`propagate_uncertainty` (`uq.py:122`) is a complete, tested capability
reachable only by import. Add `--uq/--uq-samples` to `porosity-analyze` and a
UQ expander in the app (knockdown histogram + p5/p50/p95 band). While there:
add `n_jobs` to the sampling loop (currently serial, rebuilds the mesh per
draw), reuse the mesh geometry across draws, optionally perturb the empirical
coefficients themselves (they carry the dominant calibration uncertainty), and
fix `material_name` reporting `"MaterialProperties"` for passed instances
(`uq.py:194`).

### 3.5 Thermal/residual-stress load case — [L]
Cure-induced thermal stress interacts strongly with porosity-degraded matrix
strength. A `loading='thermal'` mode building `∫ Bᵀ C α ΔT dV` reuses the
existing assembly machinery; needs CTE fields on `MaterialProperties`.
Prerequisite: the production mesh (`nz = 12`) resolves no preset's plies,
so its element layup is unsymmetric and a free laminate warps under
`ΔT`. Decide first between a ply-resolving default (`nz = n_plies`, about
5x the first FE solve) and thickness-averaged multi-ply elements;
`CompositeMesh.layup_discrepancies()` reports the mismatch.

### 3.6 Stretch: richer FE toolbox — [L each]
- **Locking mitigation**: fully integrated hex8 locks in bending — the likely
  reason the ILSS beam test needs 16×4×8 and still shows 15% error
  (`tests/test_fe_solver.py:1030-1083`). Options: B-bar / selective reduced
  integration / incompatible modes; or Hex20. Also make integration order
  configurable (plumbed but pinned at `fe/element.py:97`).
- **Mesh-convergence helper**: `refinement_study()` solving at 2–3 resolutions
  and reporting knockdown/peak-FI convergence.
- **4-point bend (ASTM D7264) and flexure (D790) BC builders** — the
  `BoundaryHandler` docstring already flags the missing 4-pt variant.
- **Periodic BCs** for RVE homogenization, closing the loop with the
  analytical MT path.
- **Binary VTU/XDMF export** (current writer is legacy ASCII with per-float
  `repr`, `fe/solver.py:221-317`) and nodal stress recovery (GP→node
  extrapolation) so ParaView output resolves peaks instead of element means.

---

## Workstream 4 — App, CLI, and pipeline polish

### 4.1 App: FE legend entry vanishes for non-compression modes — [S] (verified)
`plot_results` (`app.py:303-318`): the legend label attaches only when
`bx == bar_x[0]` (float equality on the first mode's bar), but the FE series is
NaN everywhere except the selected loading mode — so for tension/shear/ILSS the
FE bar renders with no legend entry. Track "first drawn bar" with a boolean.

### 4.2 App: make FE failure degrade gracefully — [S] (verified)
`run_analysis` (`app.py:145-175`) hard-codes `"fe_skipped_reason": None` and
lets any FE exception destroy the whole result, so the existing fallback UI
(`app.py:656-665`, `693-698`) is unreachable dead code. Wrap only the FE solve
in try/except, set the reason, and return the empirical results regardless.

### 4.3 App: route through the canonical pipeline factory — [S]
`run_analysis` hand-builds field→mesh→solver (`app.py:130-145`), bypassing
`build_empirical_pipeline` — the factory CLAUDE.md designates as the single
point of change. Any future change to mesh defaults or ply-angle handling will
silently miss the GUI. Call the factory and layer the FE solve on top.

### 4.4 App: caching hygiene — [S]
- `run_analysis_cached` (`app.py:105-111`) pickles mesh + fields into an
  unbounded `st.cache_data`; add `max_entries`/`ttl` (and consider caching only
  the JSON-friendly numbers).
- NCR serializers run on every rerun (`app.py:799-823`) — `serialise_ncr_pdf`
  spins up PdfPages per rerun, and the export JSON is computed twice
  (`app.py:715,730`). Memoize.
- Add per-figure PNG download buttons (figures are already built for
  `st.pyplot`; `fig.savefig(BytesIO())` is all that's missing).

### 4.5 Close the `FEVisualizer` figure leak — [S]
Every `viz.py` plot method saves and returns a figure without `plt.close()`
(`viz.py:44-293`); the CLI `--plots` sweep creates 100+ figures
(`cli.py:333-359`). The validation runner already does this correctly
(`validate_all.py:566`). Close after save, matching it.

### 4.6 CLI: honor the exit-code contract end-to-end; parallelize across Vp — [S/M]
- The documented 0/2/3 contract (`cli.py:237-243`) is violated by unguarded
  `save_fn`/plot calls (`cli.py:333-359`) — an `OSError` on write exits 1.
  Wrap and map (OSError→2, other→3).
- The Vp loop is serial; `--jobs` only parallelizes configs within one Vp,
  rebuilding the pool per Vp (`cli.py:311-354`). Dispatch the full
  (Vp × config) task list to a single pool.

### 4.7 Pipeline/I-O cleanups — [S] each
- `datetime.utcnow()` is deprecated on 3.12+ (`io.py:132`); use
  `datetime.now(timezone.utc)` (the validate CLI already does).
- Cache the `git rev-parse` provenance SHA per process
  (`io.py:117-129`) instead of a subprocess per JSON write.
- Read the `validate_porosity_cli.py:39` version fallback from
  `porosity_fe.__version__` instead of a hand-synced literal.
- Deduplicate the provenance block (`io.py:134-157` emits every field twice
  under parallel names, and the schema requires both spellings): pick one
  convention, alias the other for a deprecation cycle.
- Either remove the inert `applied_stress` parameter (threaded through
  `cli.py` → `pipeline.py`, documented "currently unused") or warn at parse
  time.

---

## Workstream 5 — Architecture and code quality

### 5.1 Split `fe/solver.py` (1,497 lines, five responsibilities) — [M]
Extract: (a) the three failure criteria + `_degraded_strengths`
(`fe/solver.py:953-1367`, ~415 lines) into `porosity_fe/fe/failure.py` —
this also removes the reach into the private
`homogenization._mt_effective_stiffness` (`fe/solver.py:1011`);
(b) `to_vtk` + `export_results` (~290 lines) into `porosity_fe/fe/export.py`.
Leaves `FESolver` as a ~500-line orchestrator (assemble → BC → solve →
recover). Pairs naturally with Workstream 1 refactors.

### 5.2 Single source of truth for the knockdown laws — [S]
`exp(-αVp)`, `(1-Vp)^n`, `max(1-βVp, 0)` are each implemented in ~4 places
(`empirical.py:440-467`, `:665-669`, `:887-916`, `:985-996`), plus the model
dispatch is duplicated between `_resolve_knockdown_model` and `apply_loading`
(`empirical.py:567-576` vs `:660-675`). Centralize each law as one
`(Vp, coef)` function; all call sites (scalar, vectorized, FD, analytic)
reference it.

### 5.3 Avoid the redundant nodal pass in `get_failure_load` — [S]
`empirical.py:746-761` runs the full O(n_nodes) `apply_loading` (whose nodal
field it discards), then recomputes `env_kd`/`fat_kd` a second time, and
user-supplied callables get grid-validated twice (`:677` and `:578`). Make
nodal population opt-in and validate callables once. Pure speedup for the
sweep/validation paths — no behavior change.

### 5.4 Misc dedup — [S] each
- `compute_clt_effective_modulus` reimplements the A-matrix assembly that
  `_build_clt_abd` provides (`homogenization.py:347-373` vs `:377-415`).
- The mid-y cross-section index loop appears 3× (`app.py:205-213`, `:381-386`,
  `viz.py:101-109`) — add a `CompositeMesh.mid_y_section_indices()` helper.
- The JSON envelope `{schema_version, format, provenance, units, …}` is built
  in 4 places (`reporting.py:152,208,448`, `io.py:215`) — one
  `_wrap_envelope()` in `io.py`.
- Cache `PorosityField._compute_normalization` in `__init__` instead of
  rebuilding a 1,000-point profile per `local_porosity` call
  (`porosity_field.py:195-234`).
- `validate_all.py` scatters imports across 7 body locations; dead
  `strict=` param and empty `_UNSUPPORTED_STRENGTH_PROPS` guards
  (`validate_all.py:100-122`, `:174`, `:228-235`).
- Fix `save_path: str = None` annotations in `viz.py` (7 sites) →
  `str | None`; add return-type annotations to the app's plot functions;
  consider a `TypedDict` for the analysis-result dict.
- `recommend_disposition` silently coerces an invalid `structural_class` to
  `"primary"` (`reporting.py:283`) — in an MRB/NCR context that's a hazard;
  validate and raise.

---

## Workstream 6 — Tests, CI, packaging, docs, hygiene

### 6.1 Repo identity: one canonical owner — [S] (verified; needs maintainer decision)
`pyproject.toml:67-70` declares `ranipdx-glitch/PorosityFE` the active home,
while README badges/clone URL (`README.md:5-7,46,469,513`), `CITATION.cff:7-8`,
and `CONTRIBUTING.md:7,23` all point at `elhajjar1/PorosityFE`
(`docs/*.rst` use `ranipdx-glitch`). For a scientific package the citation and
issue-tracker URLs must be right. Decide the canonical owner and sweep every
badge/link/citation to it in one commit.

### 6.2 Fix the `validation/` gitignore trap — [S] (verified)
`.gitignore` ignores the entire `validation/` tree, yet 13 dataset JSONs,
2 schemas, and `validate_all.py` are force-committed — so the
`add-validation-dataset` workflow writes a file that `git add` silently
ignores. Replace the blanket ignore with targeted ignores for generated
outputs only. Relatedly, `validation_detail_report.md` / `_master_report.png`
are simultaneously tracked *and* gitignored — untrack the generated artifacts.

### 6.3 CI: measure coverage, install the package, cache pip — [S]
- Add `pytest-cov` + `--cov-report=xml` + Codecov upload to `tests.yml` (the
  `.gitignore` already anticipates coverage files; CI never produces them).
- Switch CI to `pip install -e ".[dev]"` so packaging (entry points,
  `py.typed`, package-data) is exercised, then delete the repo-root
  `conftest.py` sys.path shim (its own docstring says exactly this).
- Add `cache: pip` to `setup-python` in `tests.yml`, `security.yml`,
  `build-executables.yml` (only `docs.yml` has it) — the 12-cell matrix
  reinstalls numpy/scipy/matplotlib from scratch every run.
- `build-executables.yml:46` hand-lists deps (`pip install numpy scipy …`)
  instead of installing from `requirements.txt`/the package — drift risk.

### 6.4 Release automation: publish to PyPI — [M]
There is no wheel/sdist publishing workflow; releases ship only the frozen
CLI zip. Add `publish.yml` on `v*` tags with `python -m build` +
`pypa/gh-action-pypi-publish` (trusted publishing, `id-token: write`), and
have the release workflow assert the CHANGELOG `[Unreleased]` section is
empty at the tag. Add a `.pre-commit-config.yaml` (ruff, ruff-format, mypy)
so contributors catch CI failures locally.

### 6.5 Docs: sync the metadata, fill the theory gap — [M]
- **Stale claims** (verified by the docs sweep): README badge and
  `docs/index.rst:18` say Python 3.9+, `pyproject.toml` requires ≥3.10;
  CHANGELOG references a `porosity-fe` CLI, a `[gui]` extra, PyQt6, and
  `PorosityFE.spec` — none exist (scripts are `porosity-analyze` /
  `validate_porosity`; the spec is `ValidatePorosity.spec`);
  `CONTRIBUTING.md:46` still directs material presets to the shim file.
  One reconciliation pass fixes all of these.
- **`docs/api.rst` documents ~10 of ~40 public symbols** — missing
  `MaterialProperties`, `MATERIALS`, `propagate_uncertainty`, `FatigueModel`,
  the CLT functions, transforms, io, pipeline entry points. Drive it from
  `__all__` (or `automodule`) so it can't silently drift.
- **No theory pages**: for a scientific package the math is the product —
  add narrative pages for Mori-Tanaka/Eshelby, the three knockdown laws and
  their calibration sets, CLT/ABD, Tsai-Wu/Hashin, and the S-N fatigue model
  (`myst-parser` is already a dependency). Add a CLI reference
  (`sphinx-argparse`) and include the CHANGELOG in the toctree.

### 6.6 Test additions (coverage is 97%, so target *kinds* not lines) — [M]
- Dedicated unit-test files for the modules tested only incidentally:
  `test_reporting.py` (687-line module, largest without one), `test_viz.py`,
  `test_fatigue.py`, `test_pipeline.py`.
- **Property-based tests** (no hypothesis anywhere today): rotation
  round-trip `R(θ)R(−θ) = I`, Tsai-Wu frame invariance, knockdown
  monotonicity in Vp on [0, 0.05] for all three laws, strain-transform
  shear factor-of-2 (a bug class already fixed once per the CHANGELOG).
- **Golden regression on the headline validation metric**: pin the
  property-weighted MAE (~7.69%) with a tolerance so silent model/dataset
  drift reddens CI.
- Explicit `rtol`/`atol` on FE numerical comparisons instead of `np.allclose`
  defaults (atol=1e-8 is meaningless against MPa-scale stiffness entries).
- Once Workstream 1 lands: a benchmark test (or `pytest-benchmark` job)
  pinning production-mesh solve time to catch performance regressions.

### 6.7 Packaging details — [S]
- Deduplicate `requirements*.txt` against `pyproject.toml` extras (3-way
  copy today); make `all = ["porosity-fe[web,dev,docs]"]` self-referential.
- `py-modules` ships top-level `app` — an `import app` collision risk in
  user environments; consider namespacing or documenting.
- `ValidatePorosity.spec`: assert the dataset glob found files (a silent
  empty glob ships an executable with zero datasets), and add
  `collect_submodules('porosity_fe')` to `hiddenimports` (currently lists
  only the shim).
- Ship `CITATION.cff` in the sdist via `MANIFEST.in`.

---

## Suggested sequencing

**Phase 1 — quick wins (a few days total).** All [S] items with verified bugs
or zero design risk: extrapolation-warning fix (2.1), app legend + graceful FE
degradation + factory routing (4.1–4.3), figure leak (4.5), `utcnow` and
version-fallback fixes (4.7), knockdown-law dedup (5.2), `get_failure_load`
fast path (5.3), normalization caching (5.4), gitignore trap (6.2), CI
coverage/caching/editable-install (6.3), docs metadata reconciliation and the
repo-identity sweep once the owner decision is made (6.1, 6.5 first bullet).

**Phase 2 — FE performance sprint.** 1.1 → 1.3 → 1.5 → 1.2 → 1.4, each with
before/after timings on the production mesh; target ≥5× end-to-end. Add the
benchmark pin (6.6 last bullet) at the end.

**Phase 3 — correctness & features.** 2.2–2.6, 3.1–3.4, the `fe/solver.py`
split (5.1), remaining Workstream 4/5 items, dedicated test files and
property tests (6.6), PyPI publishing (6.4).

**Phase 4 — deeper physics & FE (design-first).** Layup-scale recalibration
(2.7), BC elimination (1.6), locking mitigation / thermal loading / RVE /
VTU export (3.5, 3.6), theory documentation (6.5).

Each item above is self-contained enough to become one issue/PR; the file:line
references point at the exact code in question as of this commit.
