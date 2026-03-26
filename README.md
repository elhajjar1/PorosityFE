# Porosity FE Analysis Tool

A Python tool for assessing the effects of porosity defects on composite laminate strength. Supports distributed microporosity, discrete macrovoids, and multiple loading modes.

## Features

- **Two porosity types**: Distributed microporosity (continuous field) + discrete macrovoids (explicit geometry)
- **Five configurations**: uniform/spherical, uniform/cylindrical, clustered/midplane, clustered/surface, interface/penny
- **Four loading modes**: Compression, tension, shear, ILSS
- **Two solver tiers**: Empirical (Judd-Wright, power law, linear) + Mori-Tanaka micromechanics
- **Three material presets**: T800/epoxy, T700/epoxy, glass/epoxy
- **Publication-quality plots**: 7 visualization types with consistent styling

## Quick Start

```bash
pip install -r requirements.txt
python3 porosity_fe_analysis.py
```

This runs the full analysis across 5 porosity levels (1%-8%) and 5 configurations, generating PNG plots and JSON results.

## Installation

```bash
pip install numpy scipy matplotlib
```

Optional for testing:
```bash
pip install pytest
python3 -m pytest tests/ -v
```

## Usage

### Run full analysis
```bash
python3 porosity_fe_analysis.py
```

### Use as a library
```python
from porosity_fe_analysis import *

# Single configuration analysis
results = compare_configurations(0.03, material_name='T800_epoxy')

# Custom analysis
material = MATERIALS['T700_epoxy']
pf = PorosityField(material, 0.05, distribution='clustered', cluster_location='midplane')
mesh = CompositeMesh(pf, material, nx=50, ny=20, nz=24)

# Fast empirical solver
solver = EmpiricalSolver(mesh, material)
result = solver.get_failure_load(mode='compression', model='judd_wright')
print(f"Knockdown: {result['knockdown']:.3f}")

# Detailed Mori-Tanaka solver
mt = MoriTanakaSolver(mesh, material)
result = mt.get_failure_load(mode='ilss')
print(f"ILSS knockdown: {result['knockdown']:.3f}")
```

## Output Files

- `porosity_profile_{config}_{Vp}.png` - Through-thickness porosity profiles
- `porosity_mesh_3d_{config}_{Vp}.png` - 3D mesh visualizations
- `porosity_mesh_detail_{config}_{Vp}.png` - Cross-section details
- `porosity_damage_{config}_{Vp}.png` - Stiffness reduction maps
- `porosity_comparison_{Vp}.png` - Model comparison bar charts
- `porosity_knockdown_curves.png` - Cross-severity knockdown curves
- `porosity_analysis_results_{Vp}.json` - Numerical results

## Physics Models

### Empirical (Judd-Wright)
```
sigma/sigma_0 = exp(-alpha * Vp)
```
ILSS is most sensitive (alpha ~ 5.5), compression moderate (alpha ~ 3.0).

### Mori-Tanaka Micromechanics
```
C_eff = C_m * {I - Vp * [I - (1-Vp)*S]^-1}
```
Uses Eshelby tensor for void shape effects (sphere, prolate, oblate).

### Failure: 3D Tsai-Wu criterion
Full 6-component stress state evaluation with degraded strengths.

## References

- Judd & Wright - Empirical porosity-strength relationships
- Mori & Tanaka (1973) - Mean-field micromechanics
- Eshelby (1957) - Ellipsoidal inclusion theory
- Mura (1987) - Micromechanics of Defects in Solids
- Tsai & Wu (1971) - General theory of strength for anisotropic materials
