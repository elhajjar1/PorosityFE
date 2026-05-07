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

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request with a clear description

## Code Style

- Follow existing patterns in the codebase
- Add docstrings to public functions
- Include units in variable names and docstrings (MPa, mm, rad)
- Use SI units throughout

## Adding New Materials

To add a material system, add a new entry to the `MATERIALS` dictionary in `porosity_fe_analysis.py` using the `MaterialProperties` dataclass. Include all orthotropic constants and source references.

## Adding New Porosity Models

New empirical correlation models can be added to the `EmpiricalSolver` class. Each model should:
- Accept porosity volume fraction as input (`Vp`, dimensionless fraction in `[0, 1]`)
- Return a knockdown factor (0 to 1)
- Include a literature reference in the docstring

When introducing or recalibrating coefficients (e.g. a custom `alpha` for Judd-Wright or `n` for the power law):
- Document the calibration coupon set (Vp range, layup, fiber/matrix system) and the test standard used (ASTM D2344 for ILSS, D7264 for flexure, D3039 for tension, D6641 for compression).
- Fit on log-transformed data: regress `ln(KD)` vs `Vp` for the slope `−alpha`, or `ln(KD)` vs `ln(1 − Vp)` for `n`.
- Note the validity bounds — both forms are commonly bounded to `Vp ≲ 0.05`.
- Cross-reference the README "Empirical Strength Knockdown" section so user-facing docs stay in sync.
