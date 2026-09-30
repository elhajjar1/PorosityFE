# Contributing to PorosityFE

Thank you for your interest in contributing to PorosityFE!

## Reporting Bugs

Open an [issue](https://github.com/elhajjar1/PorosityFE/issues) with:
- Python version and OS
- Steps to reproduce the problem
- Expected vs actual behavior
- Error messages or screenshots

## Suggesting Features

Open an issue describing:
- The use case (what composite analysis problem you're solving)
- Expected behavior
- Any relevant references or equations

## Development Setup

```bash
git clone https://github.com/elhajjar1/PorosityFE.git
cd PorosityFE
pip install -e ".[all]"
pytest tests/ -v
```

To catch lint and type errors before CI does, install the pre-commit hooks
(they run `ruff check` and the same `mypy` command as the CI lint job):

```bash
pip install pre-commit
pre-commit install
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request with a clear description

## Releasing

1. Bump `version` in `pyproject.toml` and the fallback `__version__` in
   `porosity_fe/__init__.py`.
2. Move the `## [Unreleased]` entries in `CHANGELOG.md` under
   `## [X.Y.Z] - <date>`, leaving `[Unreleased]` empty.
3. Tag `vX.Y.Z` and push the tag. `.github/workflows/publish.yml` checks
   steps 1-2 (`.github/scripts/check_release.py`), builds the sdist and
   wheel, smoke-tests the wheel and publishes to PyPI via trusted
   publishing. `build-executables.yml` attaches the frozen
   `validate_porosity` CLIs to the GitHub release.

## Code Style

- Follow existing patterns in the codebase
- Add docstrings to public functions
- Include units in variable names and docstrings (MPa, mm, rad)
- Use SI units throughout

## Adding New Materials

To add a material system, add a new entry to the `MATERIALS` dictionary in `porosity_fe/materials.py` using the `MaterialProperties` dataclass. Include all orthotropic constants, the constituent fields the FE micromechanics path reads (`matrix_modulus`, `fiber_volume_fraction`, etc.), and source references. `porosity_fe_analysis.py` is a compatibility shim; don't add new symbols there.

## Adding New Porosity Models

Each built-in knockdown law is a single function registered in `_KNOCKDOWN_LAWS` in `porosity_fe/empirical.py`; every solver path (scalar, per-node, sensitivities) looks it up there. Each law should:
- Accept porosity volume fraction as input (`Vp`, dimensionless fraction in `[0, 1]`, scalar or array)
- Return a knockdown factor (0 to 1)
- Include a literature reference in the docstring

A new law also needs its QI coefficient table on `Calibration` (scaled per layup in `EmpiricalSolver.__init__`), its name in `KnockdownModel` (`porosity_fe/_types.py`), a derivative branch in `EmpiricalSolver.local_sensitivities` (otherwise only `sensitivity_fd` supports it), and an entry in the model list of `get_all_failure_loads` if sweeps should report it. For a one-off model, passing a callable `model(Vp, mode) -> KD` to `get_failure_load` needs none of this.

When introducing or recalibrating coefficients (e.g. a custom `alpha` for Judd-Wright or `n` for the power law):
- Document the calibration coupon set (Vp range, layup, fiber/matrix system) and the test standard used (ASTM D2344 for ILSS, D7264 for flexure, D3039 for tension, D6641 for compression).
- Fit on log-transformed data: regress `ln(KD)` vs `Vp` for the slope `−alpha`, or `ln(KD)` vs `ln(1 − Vp)` for `n`.
- Note the validity bounds — both forms are commonly bounded to `Vp ≲ 0.05`.
- Cross-reference the README "Empirical Strength Knockdown" section so user-facing docs stay in sync.
