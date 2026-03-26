# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Repository Overview

This repository contains a **Python-based analytical and micromechanics model** for predicting the effects of porosity defects on composite laminate strength. The tool mirrors the architecture of the Double_Wrinkle `jin_fe_analysis` tool but focuses on porosity rather than fiber waviness.

**Key features:**
- Distributed microporosity + discrete macrovoids
- Empirical models (Judd-Wright, power law, linear) for fast predictions
- Mori-Tanaka micromechanics with Eshelby tensor for rigorous analysis
- Full 3D Tsai-Wu failure criterion
- Five porosity configurations, four loading modes, three material presets

## File Structure

```
Porosity_FE/
├── porosity_fe_analysis.py          # Main implementation (~1100 lines)
├── requirements.txt                 # Dependencies (numpy, scipy, matplotlib)
├── tests/test_porosity_fe.py        # 79 tests
├── README.md                        # Usage guide
├── CLAUDE.md                        # This file
├── docs/superpowers/
│   ├── specs/                       # Design specification
│   └── plans/                       # Implementation plan
└── (output files generated at runtime: PNG plots + JSON results)
```

## Running the Tool

```bash
# Full analysis (5 porosity levels x 5 configurations)
python3 porosity_fe_analysis.py

# Run tests
python3 -m pytest tests/ -v
```

## Code Architecture

The single file `porosity_fe_analysis.py` is organized into 8 sections:

### Section 1: MaterialProperties (dataclass)
- Full orthotropic properties + constituent properties for micromechanics
- `get_stiffness_matrix()`, `get_compliance_matrix()`, `get_isotropic_matrix_stiffness()`
- Three presets: `MATERIALS['T800_epoxy']`, `MATERIALS['T700_epoxy']`, `MATERIALS['glass_epoxy']`

### Section 2: VoidGeometry
- Single void: center, radii (semi-axes), orientation
- `contains()`, `distance_field()`, `stress_concentration_factor()`, `volume()`, `aspect_ratio`
- Shape presets: `VOID_SHAPES` = spherical, cylindrical, penny

### Section 3: PorosityField
- Distributed + discrete porosity field
- Three distributions: uniform, clustered (midplane/surface), interface (ply boundaries)
- `local_porosity()`, `local_stiffness_reduction()`, `effective_porosity_profile()`
- Five config presets: `POROSITY_CONFIGS`

### Section 4: CompositeMesh
- 3D structured hexahedral mesh (flat geometry, porosity as stiffness reduction)
- `generate_mesh()` creates nodes, elements, porosity field samples, ply IDs
- Identifies void elements (porosity > 0.95)

### Section 5: EmpiricalSolver
- Fast analytical: Judd-Wright (`exp(-alpha*Vp)`), power law (`(1-Vp)^n`), linear (`1-beta*Vp`)
- Loading-dependent decay constants (ILSS most sensitive)
- Discrete void SCF amplification via `_apply_discrete_void_scf()`
- `get_failure_load()`, `get_all_failure_loads()`

### Section 6: MoriTanakaSolver
- Eshelby tensor: closed-form for sphere, prolate, oblate (Mura 1987)
- Mori-Tanaka: `C_eff = C_m @ {I - Vp * inv[I - (1-Vp)*S]}`
- Strength degradation: `sigma_degraded = sigma_pristine * sqrt(C_eff_ii / C_pristine_ii)`
- Full 3D Tsai-Wu failure criterion (F1-F66 + all interaction terms)
- Cached nodal knockdown computation by unique porosity values

### Section 7: FEVisualizer
- 7 static plot methods: porosity_field, mesh_3d, mesh_detail, damage_contour, void_scf, knockdown_curves, model_comparison
- Publication-quality: 300 DPI, consistent fonts, proper colorbars

### Section 8: Analysis Pipeline
- `compare_configurations()` - loops through configs, runs both solvers, prints rankings
- `save_results_to_json()` - exports to JSON
- `main()` - full parametric study across porosity levels

## Key Mathematical Relationships

**Empirical knockdown (Judd-Wright):**
```
sigma/sigma_0 = exp(-alpha * Vp)
alpha: compression=3.0, tension=2.0, shear=4.0, ilss=5.5
```

**Mori-Tanaka effective stiffness (void inclusions):**
```
T = (I - S)^-1                              (void strain concentration tensor)
C_eff = C_m * {I - Vp * [I - (1-Vp)*S]^-1}  (effective stiffness)
```

**Eshelby tensor S:** Closed-form from Mura (1987) for sphere, prolate, oblate spheroids.

**3D Tsai-Wu failure:**
```
F_i * s_i + F_ij * s_i * s_j = 1    (i, j = 1..6, Voigt notation)
```

## Typical Parameter Ranges

| Parameter | Range | Notes |
|-----------|-------|-------|
| Void volume fraction Vp | 0.01 - 0.08 | 1% to 8% |
| Void aspect ratio | 1.0 - 10.0 | Sphere to elongated |
| Distribution | uniform, clustered, interface | Through-thickness profile |

## Modifying the Models

**To change material properties:**
```python
material = MaterialProperties(E11=..., E22=..., ...)
```

**To add a new configuration:**
```python
POROSITY_CONFIGS['custom'] = {'distribution': 'clustered', 'void_shape': 'penny', 'cluster_location': 'quarter'}
```

**To change mesh resolution:**
```python
mesh = CompositeMesh(porosity_field, material, nx=80, ny=30, nz=24)
```

**To add a new empirical model:**
Add a new method to `EmpiricalSolver` and register it in `get_failure_load()`.

## Expected Results

For 3% porosity with T800/epoxy:
- **Compression knockdown (J-W):** ~0.914 (exp(-3.0 * 0.03))
- **ILSS knockdown (J-W):** ~0.847 (exp(-5.5 * 0.03)) — most sensitive
- **Tension knockdown (J-W):** ~0.942 (exp(-2.0 * 0.03))

ILSS is always the most porosity-sensitive property. Interface/penny configurations show highest local damage due to stress concentration at ply boundaries.

## Scientific References

1. Judd, N.C.W. & Wright, W.W. - Voids and their effects on mechanical properties of composites
2. Mori, T. & Tanaka, K. (1973) - Acta Metallurgica (mean-field homogenization)
3. Eshelby, J.D. (1957) - Proc. Royal Society (inclusion theory)
4. Mura, T. (1987) - Micromechanics of Defects in Solids (Eshelby tensor derivations)
5. Tsai, S.W. & Wu, E.M. (1971) - J. Composite Materials (failure criterion)
6. Nemat-Nasser, S. & Hori, M. (1993) - Micromechanics (homogenization methods)
