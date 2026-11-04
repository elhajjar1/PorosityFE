# What's new in 1.3

PorosityFE 1.3 is the first release since 1.2.0 (2026-05-11). This page is
an overview; the {doc}`changelog` has every change, with its numbers. Read
[Results that change on upgrade](#results-that-change-on-upgrade) before
you compare 1.3 results with 1.2 ones.

## Delamination and fracture mechanics (FE)

A delamination is now an explicit crack in the FE model, alongside the
smeared porosity ({doc}`theory/delamination`):

- **`Delamination`**: a strip, rectangle or ellipse at a ply interface,
  passed as `CompositeMesh(..., delaminations=...)`. The interface's node
  plane is duplicated and tied outside the crack. Results without a
  delamination are unchanged bit for bit.
- **Crack-face contact** keeps the faces from interpenetrating.
- **VCCT energy release rates** along the front, with Benzeggagh-Kenane
  toughness. The critical load factor `fracture_load_factor` is joined by
  `interior_fracture_load_factor`, the minimum away from the mesh-dependent
  side faces.
- **Porosity-dependent toughness.** The default law's coefficients are
  uncalibrated placeholders, and every result they change carries
  `toughness_uncalibrated = True`.
- **An uncertainty band** on the load factor (opt-in).
- **Interlaminar stresses** at the ply interfaces, with stress-based onset
  criteria, and O'Brien's free-edge energy release rate.
- **The ASTM fracture specimens** DCB, ENF and MMB (`CompositeMesh.for_dcb`
  / `for_enf` / `for_mmb`, loadings `'dcb'` / `'enf'` / `'mmb'`), verified
  against the NASA benchmark reports.
- **Delamination in the app** (a Defects panel), the CLI and the NCR
  record, all opt-in.

The load factor predicts growth onset only: there is no propagation,
cohesive zone or buckling.

## Specimens and load cases

- **`'ilss'` is the ASTM D2344 short-beam test** on its own specimen mesh
  (`CompositeMesh.for_ilss`). It was a three-point bend of whatever coupon
  the mesh was.
- **Three- and four-point flexure** on roller supports (ASTM D7264
  procedures A and B, `CompositeMesh.for_flexure`), in the library and the
  app.
- **Thermal and cure residual stress**: optional lamina CTEs on
  `MaterialProperties`, `FESolver.solve(delta_T=...)`, and a thermal
  section in the app.

## The FE solver

- **The incompatible-mode brick `'hex8i'` is the default formulation.**
  Plain `'hex8'` stays selectable.
- **Boundary conditions are imposed exactly** by eliminating the
  prescribed degrees of freedom, not by a penalty.
- **The production mesh resolves every ply**: 30 x 10 x `n_plies`, capped
  at 48 element layers. Above the cap, thick laminates keep their stacking
  sequence where an `nz` allows it.
- **Faster solves**: assembly and stress recovery are batched, and the
  sparse LU factorizations of the last two constraint sets are cached. On
  the production mesh a first solve became about 5 times faster and a
  repeat solve about 60 times.
- **A memory budget** (`FESolver(max_memory_gb=...)`, default 80 % of
  physical memory). Each factorization is estimated before it is built,
  and refused with the estimate and the alternatives when it does not
  fit. With `solver='cg'` nothing is factorized, the knockdown's pristine
  reference included.
- **New results**:
  - reaction forces and the effective modulus
  - the first-ply-failure load factor
  - interior failure metrics, away from the stress singularities
  - nodal stresses
  - the failure-mode maxima
- **Binary VTU export** for ParaView.
- **`refinement_study`**: a mesh-convergence study of any load case, with
  observed orders, Richardson extrapolates and a convergence flag per
  quantity. The app runs it from its Results tab.
- **An opt-in direction-aware porosity degradation**
  (`porosity_degradation='directional'`) for flat voids.

## The app, the CLI and reporting

- **The Streamlit app** gained:
  - the flexure, thermal and fracture modes
  - the Defects panel
  - the mesh-convergence check
  - uncertainty analysis
  - a PNG download per figure and a VTU download

  The app is now the `porosity_app` package; `streamlit run app.py` is
  unchanged.
- **`porosity-analyze`** gained `--jobs N` (parallel sweeps), `--uq`
  (uncertainty propagation) and delamination screening.
- **The NCR disposition bands are a `DispositionPolicy`.** The defaults
  are labelled illustrative: the released drawing or process
  specification governs.
- **JSON exports** carry a `units` block. The results schema is at 1.3,
  with additive blocks only.
- **An `IM7_8552_epoxy` material preset** carries interlaminar toughness
  (NASA/CR-2011-217305), so the fracture features work on it without
  manual entry. Its strengths are the NCAMP qualification means. Its resin
  Poisson ratio is an assumption, flagged in the entry.

## Results that change on upgrade

These change numbers you may have recorded with 1.2. Each is explained in
the {doc}`changelog`.

- **FE knockdown**: it is now a real stiffness ratio (porous over pristine
  structural stiffness). It was a ratio of signed domain-mean stresses,
  clamped to 1.
- **Micromechanics**:
  - The Mori-Tanaka Eshelby tensor had three errors, now corrected.
  - Voids with three different radii use the general-ellipsoid tensor.
- **FE formulation and mesh**:
  - `'hex8i'` is the default element.
  - The boundary conditions are exact.
  - The production mesh has one element layer per ply.
  - Elements take the ply at their centroid exactly.
- **`'ilss'`** solves the D2344 short-beam specimen.
- **Failure**:
  - Hashin sees the interlaminar stresses, with a delamination mode.
  - `tsai_wu_F12` is the normalized coefficient it was documented as.
  - Geometric void elements are skipped consistently.
- **`solve('tension')`** pulls by default; it used to be a compression
  solve.
- **Iterative solves**: `solver='cg'` / `'minres'` returned wrong answers
  at the default `rtol` under the old penalty constraints.
- **Empirical layup scaling**: the binned `f_md / 0.5` rule is replaced by
  a lamination-theory energy blend. Quasi-isotropic layups are unchanged.
- **The `interface` porosity profile** is scaled to its requested mean in
  the FE.
- **Round-off only**: the sparse LU now pivots on the diagonal, which
  moves FE results within the regression gate's 1e-9 tolerance.

## Deprecations and platforms

- **Deprecations.** Ten names and paths are deprecated in 1.x and removed
  in 2.0. The warning ones emit a `DeprecationWarning` that names 2.0. See
  {doc}`deprecations`.
- **Python 3.11 to 3.14.** Python 3.10 reached end of life in October 2026.
- **Releases** publish to PyPI from a version tag through trusted
  publishing.
- **The PyQt6 desktop GUI** is no longer shipped. The supported interfaces
  are the Streamlit app, the `porosity-analyze` and `validate_porosity`
  commands, and the library.
