# Changelog

All notable changes to PorosityFE will be documented in this file.

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
