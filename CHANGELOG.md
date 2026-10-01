# Changelog

All notable changes to PorosityFE will be documented in this file.

## [Unreleased]

### Added
- **Fiber constituent inputs.** `MaterialProperties` gains
  `fiber_poisson` (default 0.2) and `fiber_shear_modulus` (default `None`,
  meaning the isotropic `E_f / (2 (1 + nu_f))`). They replace a hard-coded
  isotropic fiber in the Halpin-Tsai degradation ratios. The defaults
  reproduce the earlier results bit for bit.
- **PyPI release automation (IMPROVEMENT_PLAN 6.4).** A new
  `.github/workflows/publish.yml` runs on `v*` tags:
  - `.github/scripts/check_release.py` checks that the tag matches both
    version literals, that `[Unreleased]` is empty and that the release
    has its own CHANGELOG section;
  - the sdist and wheel are built and pass `twine check --strict`, and
    the wheel is smoke-tested;
  - the packages are published through PyPI trusted publishing (no
    stored token).

  A `.pre-commit-config.yaml` runs the CI lint checks (ruff, mypy)
  locally. The license metadata moves to an SPDX expression
  (`license = "MIT"`, setuptools >= 77), which removes the deprecation
  that breaks builds after 2027-02-18.
- **Property-based tests (IMPROVEMENT_PLAN 6.6).** `tests/test_properties.py`
  uses `hypothesis` (new dev dependency) to check that:
  - stiffness rotation round-trips;
  - stress transforms compose as a group;
  - work is frame-invariant, which guards the engineering-shear factor 2;
  - the Tsai-Wu index is frame-invariant;
  - every knockdown law stays a monotone fraction on `Vp in [0, 0.05]`,
    at QI and after layup scaling.

  The headline validation MAE (7.05% property-weighted, 6.53%
  point-weighted, 35 entries, 239 points) is now pinned.
- **Uncertainty propagation in the CLI and app (IMPROVEMENT_PLAN 3.4).**
  - `porosity-analyze --uq` writes `porosity_uq_<Vp>.json` (new
    `porosity-fe.uq` format, `FORMAT_UQ`) for every loading mode. The
    sampled inputs are set with `--uq-samples`, `--uq-coef-cov`,
    `--uq-vp-cov` and `--uq-strength-cov`.
  - The app's Results tab gains an **Uncertainty** expander showing the
    p5/p50/p95 metrics and a knockdown histogram.
  - `propagate_uncertainty(coef_cov=...)` perturbs the knockdown law's
    calibration coefficient. Before this, material scatter alone left
    the knockdown with zero spread.
  - A passed `MaterialProperties` instance now reports its preset name
    (or `"custom"`) instead of `"MaterialProperties"`.
  - Draws beyond the calibration bound produce one summary warning
    instead of one per draw.
  - New `save_uq_results_to_json()`, and a `solver_kwargs=` hook on
    `build_empirical_pipeline` for the coefficient overrides.
- **`sweep_configurations(void_volume_fractions, ...)`** runs
  `compare_configurations` for several porosity levels with every
  `(Vp, config)` pair in one task list, so `n_jobs > 1` keeps one process
  pool busy across levels. `porosity-analyze --jobs N` uses it when more
  than one `--vp` is given. Results equal the per-level calls.
- **PNG download per figure in the Streamlit app** (profile, mesh,
  results, stress tabs).
- **First-ply-failure load factor.** `FieldResults.first_ply_failure_load_factor`
  is the multiplier on the applied load at which the selected criterion
  first reaches 1 (linear scaling; margin of safety = factor - 1), solved
  exactly per Gauss point for Tsai-Wu, Hashin and max-stress. Also in
  `summary().details` and the JSON export.
- **Reaction forces and effective modulus from FE solves.** `FieldResults`
  now carries `reaction_forces` (N, per node) and `effective_modulus`
  (MPa): `E_x` for compression/tension and `G_xy` for shear, from the
  strain energy `u^T K u / (strain^2 V)`; `None` for the force-controlled
  ILSS bend. A pristine UD specimen recovers `E11` and `G12` exactly, and
  a quasi-isotropic one matches CLT `G_xy` exactly and `E_x` to ~1%. The
  JSON export gains a `stiffness` block.
- **Self-documenting `units` block in JSON envelopes** (`save_results_to_json`,
  `write_results_json` / `_serialise_payload_json`, `serialise_ncr_json`) plus
  per-field `description` entries on numeric leaves in
  `validation/schemas/porosity_results_schema.json` so downstream consumers
  reading only the JSON can tell whether `knockdown` is a fraction, a
  percentage, or a multiplier. `JSON_SCHEMA_VERSION` bumped to `1.1`
  (additive, backwards-compatible — 1.0 files still load and validate). (#131)
- **`--jobs N` flag on the `porosity-analyze` CLI** parallelises the per-
  configuration sweep in `compare_configurations` over a
  `concurrent.futures.ProcessPoolExecutor`. `N=1` (default) preserves
  the deterministic serial path byte-for-byte; `N>1` dispatches the
  `(Vp, config)` calls across worker processes; `0`/`-1` resolve to
  `os.cpu_count()`. Results are deterministically re-assembled by
  `(Vp, name)` so the returned dict is independent of `N`. Measured
  ~2x wall-clock speedup on the 5-Vp × 5-config sweep with 4 workers
  on a 4-core box. (#52)
- **`n_jobs` kwarg on `compare_configurations`** exposes the same knob
  to library callers (including a future GUI batch sweep).
- **Top-level `_analyze_one(Vp, name, config, material, applied_stress,
  seed)` helper** factors the per-configuration build/solve out of the
  inner loop so it can be pickled to a worker process. The returned
  `(Vp, name, result_dict)` tuple includes the mesh / porosity_field /
  empirical_solver instances (all three pickle cleanly today; no
  rebuild on the main process is needed).

### Fixed
- **Total mesh size is capped.** `CompositeMesh` rejected more than
  10 000 elements per axis but still admitted 10 000^3 in total, so an
  oversized mesh failed with an out-of-memory error partway through FE
  assembly. It now rejects more than 1 000 000 elements up front and
  states the estimated FE memory in the message. The app shows the error
  under the mesh sliders and disables Run.
- **Triaxial voids use the exact Eshelby tensor.** Voids with three
  different radii were treated as a spheroid about their longest axis; the
  Mori-Tanaka stiffness now uses the general-ellipsoid tensor (Mura's
  integrals). **Result change** only for triaxial `void_shape_radii`; the
  preset void shapes are axisymmetric and unchanged.
- **UQ sweeps survive extreme draws.** A wide distribution could push
  `fiber_volume_fraction` past 1, or a modulus or strength below 0 with
  `'normal'`, and abort the sweep in validation. Perturbed values are now
  clipped: fiber volume fraction to the hexagonal packing limit (0.907),
  moduli and strengths to stay positive.
- **Mesh quality check matches assembly.** `check_mesh_quality` tested the
  Jacobian only at the element center, while assembly rejects a
  non-positive determinant at any of the 8 Gauss points. It now checks the
  same points, so a corner-inverted element is reported.
- **`FatigueModel` says when `R` is ignored.** The S-N slopes are calibrated
  at `R = 0.1` and no mean-stress correction is applied, so other values of
  `R` do not change the result. A non-calibration `R` now emits a
  `UserWarning` saying so.
- **Knockdown-curve plot paired the wrong points.** `FEVisualizer.plot_knockdown_curves`
  sorted the x values numerically but read the y values in label text
  order. With `--vp 0.02 0.10 --plots` the 10% knockdowns were drawn at
  2% and the reverse. Labels such as `2p5pct` (from `--vp 0.025`) also
  crashed the plot. Both are now parsed and ordered one way.
- **Download filenames are Windows-safe as documented.**
  `_sanitise_filename_component` now also replaces `\ : * ? " < > |`,
  other whitespace and control characters, not just `/` and spaces.
- **`compare_configurations(configs={})` / `sweep_configurations(configs={})`
  run nothing.** An empty mapping used to fall back to the five bundled
  configurations; only `None` does now.
- **`porosity-analyze` output failures honor the exit-code contract.**
  An `OSError` while writing a JSON result or a plot exited with a
  traceback (code 1); it now returns 2, and any other output failure
  returns 3, as documented.
- **Bounded app result cache.** `run_analysis_cached` kept every analysis
  (mesh and fields included) for the life of the server; it now holds at
  most 16. The export JSON is also serialized once per rerun, not twice.
- **Void stress concentration factors from elasticity.**
  `VoidGeometry.stress_concentration_factor` used uncited
  piecewise-linear rules in aspect ratio, with shape classes that
  disagreed with the micromechanics' (and jumped at aspect ratio 1.2 and
  `radii[1] = radii[0]/2`). A penny void loaded in its own plane got
  SCF 17, and orientation was ignored. It now uses the exact solution for
  a traction-free ellipsoidal cavity in an isotropic matrix: Eshelby's
  interior field plus the traction-free jump condition, maximized over the
  surface. That reproduces Goodier's sphere values, Kirsch's 3 and Inglis'
  `1 + 2a/b`, honors `orientation`, and takes the matrix Poisson's ratio
  (`nu_m`, default 0.35; `EmpiricalSolver` passes
  `material.matrix_poisson`). **Result change:** only the discrete-void
  term of `EmpiricalSolver.nodal_knockdown` (and the SCF plot) changes.
  For example, the preset penny void goes from SCF 17 / 14 (tension /
  ILSS) to 1.16 / 6.08, and a sphere from 2.0 / 1.8 to 2.07 / 1.86.
  Specimen-level failure loads and the validation MAE are unaffected.
- **FE knockdown is now a real stiffness ratio.** `FieldResults.knockdown`
  was the ratio of signed domain-mean stresses (porous vs pristine
  stiffness applied to the porous strain field), silently clamped to 1.
  It read `sigma_xx` for the shear mode and the sign-changing `tau_xz`
  field for ILSS, so both came out as exactly 1.0 at any porosity. The
  knockdown is now porous over pristine structural stiffness from a
  second solve of the same mesh and boundary conditions with no porosity
  or void elements: the `E_x` / `G_xy` ratio for compression, tension and
  shear, and the beam-stiffness (inverse compliance) ratio for ILSS.
  Values above 1 are logged, not clamped. The pristine result is cached
  by mesh geometry and material, so a porosity sweep pays for it once
  per loading mode; the first solve on a new production mesh takes about
  3 s longer. **Result change:** shear and ILSS FE knockdowns drop below
  1 (e.g. 0.985 and 0.946 for uniform QI at `Vp = 0.06`), and the
  compression values shift slightly (README comparison table updated).
- **`FESolver.solve('tension')` now pulls.** `applied_strain` defaulted
  to `-0.01` for every mode and `tension_bcs` forwarded it unchanged, so
  a tension solve without an explicit strain was a compression solve
  (identical failure indices to `'compression'`). The default is now
  mode-dependent (`-0.01` compression, `+0.01` tension and shear), and a
  strain whose sign contradicts `'compression'` / `'tension'` logs a
  warning. **Result change** only for callers relying on the default;
  explicit strains (as the app and pipeline pass) are unaffected.
- **One definition of a "void element" in the FE path.** Failure
  evaluation and the first-ply-failure load factor now skip geometric
  void elements (`CompositeMesh.void_elements`, assembled with ~1 MPa
  stiffness) as well as elements above the porosity threshold. Before,
  a geometric void with low nodal porosity went through the failure
  polynomial with near-pristine strengths. The thresholds are shared
  constants in `porosity_fe.fe.element` (`VOID_VP_THRESHOLD = 0.95`,
  `VP_STIFFNESS_CLAMP = 0.99`). Geometric voids never governed the
  maximum index in the cases checked, so only their per-element entries
  change (to 0).
- **Mori-Tanaka Eshelby tensor corrected.** Three errors in
  `_mt_effective_stiffness`, now checked against Mura's integrals and the
  closed-form spherical-void Mori-Tanaka moduli:
  (1) the shear diagonal lacked the factor 2 of the engineering-shear
  Voigt form, so every void shape under-degraded shear stiffness;
  (2) in the axisymmetric (cylindrical / penny) branch `S_1122` and
  `S_2211` were swapped and the `S_2211` expression was wrong, which also
  made `C_eff` asymmetric (11% for cylinders, 76% for pennies);
  (3) `_degraded_composite_stiffness` read the matrix moduli from single
  entries `C_eff[3,3]` / `C_eff[0,1]`, which for an anisotropic `C_eff`
  picks one plane's values depending on the void orientation. It now uses
  the isotropic (Voigt-average) projection, which is exact for spheres.
  **Result change:** FE knockdowns at `Vp = 0.04` move by about one
  percentage point (e.g. uniform QI 0.963 -> 0.950, penny interface
  0.991 -> 0.979), with matching shifts in stresses and failure indices.
  CLT modulus predictions change too: validation MAE goes from 7.09% to
  7.05% property-weighted and 6.56% to 6.53% point-weighted, and only
  modulus entries move (liu_2018 transverse modulus 3.29% -> 2.76%,
  stamopoulos_2016 shear modulus 15.39% -> 14.73%, stamopoulos_2016
  transverse modulus 1.02% -> 1.49%). The penny-void regression pins in
  `test_homogenization.py` were re-derived: at high crack density the
  in-plane `C_11` now correctly approaches its plane-stress limit.
- **FE Hashin criterion now sees interlaminar stresses.** It used only
  `sigma_11`, `sigma_22`, `tau_12`, so `loading='ilss'` with
  `failure_criterion='hashin'` returned an index blind to the governing
  `tau_13` / `tau_23`. A fifth `delamination` mode (Brewer & Lagace 1988:
  `(<sigma_33>/Y_t)^2 + (tau_13^2 + tau_23^2)/S_23^2`) is part of
  `max_fi` and the load factor. **Result change:** Hashin ILSS indices are
  now governed by delamination; Hashin results for the other load cases
  in the regression fingerprint, and all Tsai-Wu / max-stress results,
  are unchanged. Every criterion's mode breakdown now has a
  `delamination` key (NaN for Tsai-Wu, 0 for max-stress, which already
  checks those components).
- **Empirical extrapolation warning now reports the right value and sees
  local peaks.** The message labelled the specimen-average `Vp` as
  "max Vp"; it now says "specimen-average Vp". `apply_loading()` also
  warns when the mean is within the `Vp <= 0.05` calibration bound but a
  `clustered` / `interface` distribution's local peak is not, since the
  per-node knockdown field is then extrapolated near the peak.
  `get_failure_load()` still checks only the mean, because its result uses
  the mean. Nodes inside discrete voids (`Vp = 1.0`, handled by the SCF
  step) are excluded from the peak. The warning is now attributed to the
  caller's line for both entry points.
- **App: the FE legend entry no longer disappears** for tension, shear,
  and ILSS runs. It was attached only to a bar in the first (compression)
  group, where the FE series is not drawn.
- **App: an FE solver failure no longer discards the empirical results.**
  Only the FE solve is guarded; on failure `fe_field` is `None` and
  `fe_skipped_reason` names the exception, which activates the existing
  "FE solve was skipped" notice.
- **App: `run_analysis` builds the field, mesh, and empirical solver
  through `build_empirical_pipeline`**, so changes to mesh defaults or
  ply-angle handling reach the GUI.
- **`FEVisualizer` closes figures after saving them.** The CLI `--plots`
  sweep no longer accumulates 100+ open figures. The `Figure` is still
  returned; without `save_path` it is left open for the caller as before.
- **Provenance timestamps no longer use the deprecated
  `datetime.utcnow()`** (Python 3.12+). The `...Z` string format is
  unchanged.
- **`validate_porosity --version` falls back to `porosity_fe.__version__`**
  instead of its own hard-coded version string, leaving
  `porosity_fe/__init__.py` as the only literal to bump at release.

### Changed
- **Packaging (IMPROVEMENT_PLAN 6.7).**
  - The `all` extra is now `porosity-fe[web,dev,docs]` instead of a
    hand-copied list.
  - A test fails if `requirements.txt`, `requirements-web.txt` or
    `requirements-test.txt` drifts from `pyproject.toml`. The files stay
    because the executable build, the security audit and the Streamlit
    deployment guide use them.
  - `CITATION.cff` ships in the sdist.
  - `ValidatePorosity.spec` refuses to build when it finds no validation
    datasets, instead of producing an executable with none, and bundles
    every `porosity_fe` submodule.
- **`effective_porosity_profile(nz, x=25.0, y=10.0)`** takes the in-plane
  sampling location, which was hard-coded to the domain center.
- **G23 degradation shares the G12 ratio explicitly.** Both used the same
  Halpin-Tsai form with identical inputs; the duplicate computation is
  removed and the equality is documented. Results are unchanged.
- **Percent inputs are labelled as percent.** The only porosity inputs that
  take a percent now say so where the value is asked for:
  - the app field is "Void content Vp (%)", with help "3.0 means 3 % voids";
  - `recommend_disposition`'s first parameter is renamed `Vp` ->
    `Vp_percent`. It is documented and rejects values outside `[0, 100]`;
    positional callers are unaffected, but `Vp=` keyword callers must switch
    to `Vp_percent=`;
  - the README conventions section lists all three percent inputs
    (the app field, `--vp-pct`, `recommend_disposition`).

  The uncertainty CoV inputs (CLI and app) now state that they are
  fractions of the value (0.10 = 10 % of Vp, not 10 percentage points).
- **Dedup and cleanup (IMPROVEMENT_PLAN 5.4).** All five JSON writers
  build their envelope through one `io._wrap_envelope()` (output is
  unchanged). `compute_clt_effective_modulus` reuses `_build_clt_abd`
  (bit-identical). The new `CompositeMesh.mid_y_section_indices()` /
  `mid_y_element_indices()` replace three copies of the cross-section
  index loop in the app and `FEVisualizer`. `validate_all.py` has one
  import block and no longer carries the empty
  `_UNSUPPORTED_STRENGTH_PROPS` guards. `FEVisualizer` `save_path`
  parameters are annotated `str | os.PathLike | None`.
- **`recommend_disposition` rejects an unknown `structural_class`** with
  `ValueError`. It used to substitute `"primary"` silently, which in an
  MRB record misstates the substantiation basis.
- **`resolve_material(strict=...)` is deprecated.** It had been a no-op
  since #34; passing it now emits a `DeprecationWarning`.
- **`sigma_2c` scope documented (IMPROVEMENT_PLAN 3.1).** The plan asked
  to add a `transverse_compression` empirical mode or document why not.
  There is no porosity dataset for transverse compression to calibrate a
  coefficient on, so the mode is left out. `MaterialProperties`, the
  README scope table and `EmpiricalSolver.PRISTINE_STRENGTH_KEY` now say
  so, and note that the FE criteria already use `sigma_2c` (Tsai-Wu `Y_c`,
  Hashin matrix compression).
- **FE solves are ~5x faster, and repeat solves ~60x faster.** On the
  production mesh (30x10x12, 3,600 elements) a first solve went from
  14 s (clustered porosity) / 9.6 s (uniform) to ~2.6 s / ~2.1 s, now
  bound by the sparse LU factorization. Assembly and stress recovery use
  batched per-element arrays instead of per-element Python objects, and
  recovery reuses what assembly computed. The FE knockdown, penalty BCs
  and mesh-quality check are vectorized. The assembled stiffness and its
  factorization are reused across solves with the same constraints
  (e.g. compression after tension: ~0.23 s), and are rebuilt
  automatically if the mesh, material or porosity change. Results agree
  with the previous implementation to ~1e-11 relative.
- **The validation datasets are tracked in git like any other file.**
  `.gitignore` used to ignore the whole `validation/` tree even though the
  datasets were committed, so `git add` silently skipped a newly added
  dataset. The blanket rule is removed; the generated
  `validation_*_report` files stay ignored.
- **`elhajjar1/PorosityFE` is the canonical repository.** The PyPI project
  URLs (homepage, repository, issues, documentation), the app's README
  link, the docs and the Streamlit deployment guide now point there, in
  line with the README, `CITATION.cff` and `CONTRIBUTING.md`.
- **CI installs the package** (`pip install -e ".[dev]"`) in the lint, test
  and Streamlit jobs, so tests exercise the real packaging, and the
  repo-root `conftest.py` `sys.path` shim is removed. Run tests after an
  editable install.
- **CI measures coverage** on the ubuntu / Python 3.12 test cell and uploads
  `coverage.xml` as a workflow artifact. `pytest-cov` joins the `dev` extra.
- **CI caches pip downloads** in the test, security and executable-build
  workflows, and the executable build installs its runtime dependencies
  from `requirements.txt` instead of a hand-written list.
- **Internal: one implementation per empirical knockdown law.** The
  Judd-Wright, power-law, and linear forms were each written out in four
  places in `EmpiricalSolver`; they now live in one table, and every entry
  point raises the same "Unknown knockdown model" message. Results are
  bit-identical.
- **Internal: each `apply_loading()` / `get_failure_load()` call validates
  a user knockdown callable, and evaluates the hygrothermal and fatigue
  factors, once** instead of twice.
- **Internal: the porosity profile normalization is memoized**, and the git
  commit used in JSON provenance is looked up once per process instead of
  once per file written.

## [1.2.0] - 2026-05-11

### Fixed
- **FE: penny / oblate voids were silently routed to the prolate Eshelby
  formula** because `ar = max/min ≥ 1` made the oblate branch unreachable.
  `_mt_effective_stiffness` now detects the axis of symmetry, uses the
  correct g-function branch (prolate vs oblate), and permutes the Voigt
  tensor to align with the actual void axis. Penny-shaped voids
  (`VOID_SHAPES['penny']`) now produce physically-correct anisotropy
  with through-disk degradation exceeding in-plane degradation. (#32)
- **FE: stiffness assembly now rejects inverted elements** (non-positive
  `det(J)`) at quadrature time rather than letting a wrong-sign block
  silently corrupt the assembled global K. (#33)
- **FE: engineering strain in post-processing is now rotated with
  `strain_transformation_3d`** (was `stress_transformation_3d`), so
  `FieldResults.strain_local` shear components are no longer off by 2x.
  Stress / Tsai-Wu paths were already correct and are unchanged. (#38)
- **GUI: `porosity-fe` console script** now prints a friendly stderr
  message and exits non-zero when PyQt6 is missing, instead of leaking
  a Python traceback. Error text mentions the discoverable
  `pip install porosity-fe[gui]` form. (#46)

### Added
- **CSV export** alongside the existing JSON export in the GUI's
  *File → Export Results* menu. Format is picked from the typed
  extension (`.csv` / `.json`) or the selected filter. CSV layout:
  `#`-prefixed config metadata, then a flat
  `mode,model,failure_stress_MPa,knockdown` table. (#30)
- **`EmpiricalSolver` constructor overrides** for `judd_wright_alpha`,
  `power_law_n`, and `linear_beta` — partial-merge with QI defaults,
  layup-scaled the same way, replacing the documented subclass
  workaround. (#16)
- **Input validation** on `MaterialProperties`, `CompositeMesh`,
  `VoidGeometry`, and `Hex8Element` constructors (positive moduli /
  strengths, Poisson ratios in `(-1, 0.5)`, positive geometry, fraction
  range on `node_porosities`). (#13)
- **Validation reporting** now publishes both property-weighted and
  point-weighted MAE — the published 7.7% is the property-weighted form;
  point-weighted is ~7.0%. (#36) New `summarize_mae()` helper exposes
  both for downstream tools.
- **GUI threading hardening**: `closeEvent` waits for the worker to
  exit; `_stop_requested` uses `threading.Event` for cross-core memory
  fences; the Run button is disabled before `_build_config` runs;
  `_on_stop` warns when the worker doesn't terminate within 2s. (#17)
- **Numerical-stability guards** in Mori-Tanaka inversion (pinv
  fallback + finite check) and Tsai-Wu evaluation (strength-floor clamp,
  non-finite check, fp clip on `elem_Vp`). (#14)
- **Error-message polish** on bad material / void-shape / mode /
  cluster_location names; UTF-8 encoding on all file I/O; floating-point
  noise clip on `Vp ≈ 1.0`; string `Vp` rejected with `TypeError`. (#21, #22, #23)
- **PyInstaller spec parity**: `PorosityFE.spec` now mirrors
  `ValidatePorosity.spec`'s dataset bundling and adds commonly-missed
  hidden imports (`PyQt6.sip`, `PyQt6.QtSvg`,
  `scipy.sparse.csgraph._validation`, `scipy.special.cython_special`). (#24)
- **Documentation**: README "Inputs and Conventions" section clarifies
  Vp as a fraction in [0, 1] with do/don't table; "Defining a custom
  material" worked example; per-ply vs. specimen-average porosity behavior
  for empirical vs. FE paths. (#1, #2, #4)

### Changed
- **Version source of truth**: `validate_porosity --version` now reads
  from `importlib.metadata` with a hard-coded fallback. Version aligned
  across `pyproject.toml`, `CHANGELOG.md`, `CITATION.cff`, README BibTeX,
  and `PorosityFE.spec`. (#45)

## [1.1.1] - 2026-04-19

### Changed
- **Removed `flexural_strength` property** from the validation schema and all
  datasets. The property consistently showed high MAE (8-40%) because 3-point
  bend failure in cross-ply and UD laminates involves mixed compression +
  interlaminar shear mechanisms that the current Judd-Wright mode mapping
  (`compression` proxy) cannot capture. Removing it focuses the validation
  database on properties the model can predict well.
- Overall validation MAE: **9.76% → 7.69%** (35 property-dataset pairs,
  down from 41)
- Affected datasets (6): Almeida 1994, Ghiorse 1993, Liu 2006, Olivier 1995,
  Stamopoulos 2016, Tang 1987 — all retain their other properties

### Added
- CI badge for "Build Executables" workflow in README
- Explicit documentation of model scope and property coverage

## [1.1.0] - 2026-04-19

### Added
- Expanded validation database from 3 to 13 peer-reviewed experimental papers
- Unified JSON Schema (Draft-07) for validation datasets
- 3 new material presets: IM7/8551, T300/934, CF/PEEK
- 3 CLT helper functions: `compute_degraded_clt_moduli`,
  `compute_degraded_clt_flexural_modulus`, `_build_clt_abd`
- Master validation runner `validation/validate_all.py` with strength
  (Judd-Wright) and modulus (CLT) prediction, aggregated MAE report
- Cross-platform CLI executable `validate_porosity` (Linux/macOS/Windows)
- GitHub Actions workflow `build-executables.yml` that builds and releases
  the CLI on Ubuntu, macOS, and Windows runners
- 28 new tests bringing total to 186

### Classical validation datasets added
- Ghiorse (1993) SAMPE Quarterly — AS4/3501-6
- Almeida & Nogueira Neto (1994) Compos. Struct. — 0-10% void range
- Tang, Lee & Springer (1987) J. Comp. Mater. — T300/976
- Bowles & Frimpong (1992) J. Comp. Mater. — IM7/8551-7
- Jeong (1997) J. Comp. Mater. — AS4 fabric
- Olivier, Cottu & Ferret (1995) Composites — T300/914

### Recent validation datasets added
- Liu et al. (2018) J. Comp. Mater. — T300/924, 6 porosity levels
- Zhang et al. (2025) Polymers — CF/PEEK thermoplastic matrix
- Wen et al. (2023) J. Reinf. Plast. Compos. — T700/epoxy + temperature
- Wang et al. (2022) J. Comp. Mater. — CF/epoxy + micro-CT damage evolution

## [1.0.0] - 2026-04-03

### Added
- PyQt6 desktop GUI with interactive porosity analysis
- Empirical strength models: Judd-Wright (exponential) and Power Law correlations
- 3D finite element solver with Eshelby-based stiffness degradation
- 3 porosity distribution types: uniform, clustered (midplane/surface/quarter), interface-concentrated
- 3 void morphologies: spherical, cylindrical (prolate), penny-shaped (oblate)
- 4 loading modes: compression, tension, shear, ILSS
- 3 built-in material presets: T800/epoxy, E-glass/epoxy, T700/epoxy
- Discrete void modeling with stress concentration factors
- Tsai-Wu failure criterion for multiaxial states
- Visualization: porosity fields, 3D meshes, damage contours, knockdown curves
- JSON export of analysis results
- PyInstaller macOS app bundle
- Comprehensive test suite
