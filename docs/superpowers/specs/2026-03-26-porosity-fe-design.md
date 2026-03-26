# Porosity FE Analysis Tool — Design Specification

**Date:** 2026-03-26
**Repository:** `/Users/elhajjar/Library/CloudStorage/OneDrive-UWM/AI/Porosity_FE/`
**Architecture:** Single-file mirror of `jin_fe_analysis/dual_wrinkle_fe_analysis.py`

## Overview

A standalone Python tool for assessing the effects of porosity defects on composite laminate strength. Supports both distributed microporosity and discrete macrovoids. Mirrors the structure, feel, and features of the Double_Wrinkle `jin_fe_analysis` tool.

## Scope

- **Porosity types:** Distributed microporosity (continuous field) + discrete macrovoids (explicit geometry)
- **Parametric space:** Location (through-thickness) x void shape/orientation x spatial distribution
- **Loading modes:** Compression, tension, shear, ILSS
- **Solvers:** Empirical (Judd-Wright, power law, linear) + Mori-Tanaka micromechanics
- **Statistics:** Basic statistics (mean, std, percentiles) from parametric sweeps
- **Materials:** Multiple presets (T800/epoxy, T700/epoxy, glass/epoxy)

## File Structure

```
Porosity_FE/
├── porosity_fe_analysis.py          # Main implementation (~2000 lines)
├── requirements.txt                 # Dependencies
├── README.md                        # Usage guide
├── CLAUDE.md                        # Claude Code instructions
├── docs/superpowers/specs/          # This spec
└── (output files generated at runtime)
```

## Class Architecture

### Section 1: MaterialProperties

```python
@dataclass
class MaterialProperties:
    # Lamina-level orthotropic properties
    E11: float          # Longitudinal modulus (MPa)
    E22: float          # Transverse modulus (MPa)
    E33: float          # Through-thickness modulus (MPa)
    G12: float          # In-plane shear modulus (MPa)
    G13: float          # Interlaminar shear modulus (MPa)
    G23: float          # Transverse shear modulus (MPa)
    nu12: float         # Major Poisson's ratio
    nu13: float         # Through-thickness Poisson's ratio
    nu23: float         # Transverse Poisson's ratio
    # Longitudinal strengths
    sigma_1c: float     # Longitudinal compression strength (MPa)
    sigma_1t: float     # Longitudinal tension strength (MPa)
    # Transverse strengths
    sigma_2t: float     # Transverse tension strength (MPa)
    sigma_2c: float     # Transverse compression strength (MPa)
    # Shear strengths
    tau_12: float       # In-plane shear strength (MPa)
    tau_ilss: float     # Interlaminar shear strength (MPa)
    t_ply: float        # Ply thickness (mm)
    n_plies: int        # Number of plies

    # Constituent properties (for Mori-Tanaka)
    matrix_modulus: float         # E_m (MPa)
    matrix_poisson: float         # nu_m
    fiber_modulus: float          # E_f (MPa)
    fiber_volume_fraction: float  # V_f (pristine, typically 0.55-0.65)

    def get_stiffness_matrix(self) -> np.ndarray:
        """6x6 stiffness matrix [C] from engineering constants"""

    def get_compliance_matrix(self) -> np.ndarray:
        """6x6 compliance matrix [S] = [C]^-1"""

    def get_isotropic_matrix_stiffness(self) -> np.ndarray:
        """6x6 isotropic stiffness tensor C_m from matrix_modulus and matrix_poisson.
        Used by MoriTanakaSolver for homogenization."""
```

**Presets dictionary:**

```python
MATERIALS = {
    'T800_epoxy': MaterialProperties(
        E11=161000, E22=11380, E33=11380,
        G12=5170, G13=5170, G23=3980,
        nu12=0.32, nu13=0.32, nu23=0.40,
        sigma_1c=1500, sigma_1t=2800, sigma_2t=80, sigma_2c=250,
        tau_12=100, tau_ilss=90,
        t_ply=0.183, n_plies=24,
        matrix_modulus=3500, matrix_poisson=0.35,
        fiber_modulus=294000, fiber_volume_fraction=0.60
    ),
    'T700_epoxy': MaterialProperties(
        E11=132000, E22=10300, E33=10300,
        G12=4700, G13=4700, G23=3500,
        nu12=0.30, nu13=0.30, nu23=0.40,
        sigma_1c=1200, sigma_1t=2400, sigma_2t=65, sigma_2c=200,
        tau_12=85, tau_ilss=80,
        t_ply=0.125, n_plies=24,
        matrix_modulus=3200, matrix_poisson=0.35,
        fiber_modulus=230000, fiber_volume_fraction=0.58
    ),
    'glass_epoxy': MaterialProperties(
        E11=45000, E22=12000, E33=12000,
        G12=5500, G13=5500, G23=4000,
        nu12=0.28, nu13=0.28, nu23=0.40,
        sigma_1c=600, sigma_1t=1100, sigma_2t=40, sigma_2c=140,
        tau_12=70, tau_ilss=55,
        t_ply=0.200, n_plies=24,
        matrix_modulus=3500, matrix_poisson=0.35,
        fiber_modulus=73000, fiber_volume_fraction=0.55
    ),
}
```

### Section 2: VoidGeometry

```python
class VoidGeometry:
    """Single void parameterization — equivalent of WrinkleGeometry"""

    def __init__(self, center, radii, orientation=0.0):
        """
        Args:
            center: (x, y, z) void center in mm
            radii: (a, b, c) semi-axes in mm — sphere when a=b=c
            orientation: rotation of major axis about z-axis (rad)
        """
        self.center = np.array(center)
        self.radii = np.array(radii)
        self.orientation = orientation

    def contains(self, x, y, z) -> np.ndarray:
        """Boolean array: True if point is inside the void"""
        # Transform to void-local coordinates (rotation + translation)
        # Evaluate ellipsoid equation: (x'/a)^2 + (y'/b)^2 + (z'/c)^2 <= 1

    def distance_field(self, x, y, z) -> np.ndarray:
        """Signed distance from void surface (negative inside)"""

    def stress_concentration_factor(self) -> dict:
        """SCF for each loading mode from Eshelby inclusion theory"""
        # Sphere: SCF_compression ~ 2.0, SCF_tension ~ 2.0, SCF_shear ~ 1.5
        # Cylinder: depends on orientation relative to loading
        # Penny: high SCF (crack-like behavior)

    def volume(self) -> float:
        """Void volume: (4/3)*pi*a*b*c"""

    @property
    def aspect_ratio(self) -> float:
        """Major/minor axis ratio"""
```

**Shape presets:**

```python
VOID_SHAPES = {
    'spherical':   (1.0, 1.0, 1.0),   # Isotropic void
    'cylindrical': (3.0, 1.0, 1.0),   # Elongated along fiber direction
    'penny':       (3.0, 3.0, 0.3),   # Flat disc at ply interface
}
```

### Section 3: PorosityField

```python
class PorosityField:
    """Distributed + discrete porosity — equivalent of DualWrinkleMorphology"""

    DISTRIBUTIONS = {
        'uniform':   'Constant Vp through thickness',
        'clustered': 'Gaussian concentration at specified z-location',
        'interface': 'Concentrated at ply interfaces',
    }

    def __init__(self, material, void_volume_fraction,
                 distribution='uniform', void_shape='spherical',
                 cluster_location='midplane', discrete_voids=None):
        """
        Args:
            material: MaterialProperties instance
            void_volume_fraction: overall Vp (0.0 to ~0.10)
            distribution: 'uniform', 'clustered', 'interface'
            void_shape: key from VOID_SHAPES or custom (a, b, c) tuple
            cluster_location: 'midplane', 'surface', 'quarter' (for clustered)
            discrete_voids: optional list of VoidGeometry for macrovoids
        """

    def local_porosity(self, x, y, z) -> np.ndarray:
        """Local porosity fraction at arbitrary points (0 to 1)"""
        # Combines distributed field + discrete void contributions:
        #   Vp_local = min(Vp_distributed(z) + Vp_discrete(x,y,z), 1.0)
        # Discrete voids contribute 1.0 inside their boundary, 0.0 outside.
        # Result is clamped to [0, 1].

    def local_stiffness_reduction(self, x, y, z) -> np.ndarray:
        """Stiffness multiplier at each point (1.0 = pristine, 0.0 = void)"""
        # For distributed: smooth reduction based on local Vp
        # For discrete voids: zero stiffness inside void boundary

    def get_void_locations(self) -> list:
        """List of discrete void centers and radii for visualization"""

    def effective_porosity_profile(self, nz=100) -> tuple:
        """Returns (z_coords, porosity_values) for through-thickness plot"""
```

**Configuration presets (equivalent of MORPHOLOGY_PHASES):**

```python
POROSITY_CONFIGS = {
    'uniform_spherical': {
        'distribution': 'uniform',
        'void_shape': 'spherical',
    },
    'uniform_cylindrical': {
        'distribution': 'uniform',
        'void_shape': 'cylindrical',
    },
    'clustered_midplane': {
        'distribution': 'clustered',
        'void_shape': 'spherical',
        'cluster_location': 'midplane',
    },
    'clustered_surface': {
        'distribution': 'clustered',
        'void_shape': 'spherical',
        'cluster_location': 'surface',
    },
    'interface_penny': {
        'distribution': 'interface',
        'void_shape': 'penny',
    },
}
```

### Section 4: CompositeMesh

```python
class CompositeMesh:
    """3D structured hex mesh with porosity — mirrors wrinkle tool's CompositeMesh"""

    def __init__(self, porosity_field, material, nx=50, ny=20, nz=24):
        self.porosity_field = porosity_field
        self.material = material
        self.nx, self.ny, self.nz = nx, ny, nz
        self.generate_mesh()

    def generate_mesh(self):
        """Create nodes, elements, and porosity-derived fields"""
        # Domain: Lx x Ly x Lz (default 50 x 20 x n_plies*t_ply mm)
        # Node loop: for k in nz+1: for j in ny+1: for i in nx+1
        # Flat geometry (no displacement unlike wrinkle tool)
        # At each node compute local porosity and stiffness reduction

        self.nodes = ...              # (N, 3) coordinates
        self.elements = ...           # (M, 8) hex connectivity
        self.porosity = ...           # (N,) local porosity at each node
        self.stiffness_reduction = ...# (N,) multiplier 0-1 per node
        self.ply_ids = ...            # (N,) ply number
        self.void_elements = ...      # (V,) element indices inside discrete voids
```

### Section 5: EmpiricalSolver

```python
class EmpiricalSolver:
    """Fast analytical solver — equivalent of SimplifiedFESolver"""

    # Empirical decay constants by loading mode
    JUDD_WRIGHT_ALPHA = {
        'compression': 3.0,
        'tension': 2.0,
        'shear': 4.0,
        'ilss': 5.5,
    }

    POWER_LAW_N = {
        'compression': 1.5,
        'tension': 1.2,
        'shear': 2.0,
        'ilss': 2.5,
    }

    LINEAR_BETA = {
        'compression': 10.0,
        'tension': 7.0,
        'shear': 12.0,
        'ilss': 15.0,
    }

    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material

    def apply_loading(self, stress, mode='compression'):
        """Compute nodal stresses with porosity knockdown"""
        # At each node: apply local knockdown based on porosity
        # Store stress field in 6-component format

    def get_failure_load(self, mode='compression', model='judd_wright') -> dict:
        """Predict failure stress for given loading mode"""
        # Returns: {'failure_stress': float, 'knockdown': float,
        #           'critical_location': (x,y,z), 'model': str}

    def get_all_failure_loads(self) -> dict:
        """Run all loading modes, return comprehensive results"""
        # Returns nested dict: {mode: {model: failure_data}}

    def _judd_wright(self, Vp, mode):
        """Exponential decay: sigma/sigma_0 = exp(-alpha * Vp)"""

    def _power_law(self, Vp, mode):
        """Power law: sigma/sigma_0 = (1 - Vp)^n"""

    def _linear(self, Vp, mode):
        """Linear: sigma/sigma_0 = 1 - beta * Vp"""

    def _apply_discrete_void_scf(self, base_knockdown, mode):
        """Apply stress concentration factors from discrete macrovoids.
        For nodes near discrete voids, multiply base knockdown by 1/SCF.
        SCF values come from VoidGeometry.stress_concentration_factor().
        Only affects nodes within influence radius of discrete voids;
        nodes in distributed-only regions use base knockdown unchanged."""
```

### Section 6: MoriTanakaSolver

```python
class MoriTanakaSolver:
    """Micromechanics solver — equivalent of RealFESolver"""

    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material

    def apply_loading(self, stress, mode='compression'):
        """Compute effective properties via Mori-Tanaka, then stress field"""

    def get_failure_load(self, mode='compression') -> dict:
        """Failure from degraded effective properties + Tsai-Wu criterion"""

    def get_all_failure_loads(self) -> dict:
        """All loading modes"""

    def _eshelby_tensor(self, aspect_ratio, nu_m) -> np.ndarray:
        """Eshelby tensor S for ellipsoidal void in isotropic matrix"""
        # Closed-form expressions:
        # Sphere (a=b=c): S components from nu_m only
        # Prolate spheroid (a>b=c): S from aspect ratio + nu_m
        # Oblate spheroid (a=b>c): S from aspect ratio + nu_m

    def _effective_stiffness(self, Vp, void_shape) -> np.ndarray:
        """Mori-Tanaka effective stiffness for void inclusions (C_i = 0)"""
        # For voids, the strain concentration tensor is T = (I - S)^-1
        # Mori-Tanaka effective stiffness:
        #   C_eff = C_m * {I - Vp * [(1-Vp)*T^-1 + Vp*I]^-1}
        #         = C_m * {I - Vp * [(1-Vp)*(I-S) + Vp*I]^-1}
        #         = C_m * {I - Vp * [I - (1-Vp)*S]^-1}
        # C_m is the 6x6 isotropic matrix stiffness from material.get_isotropic_matrix_stiffness()
        # S is the Eshelby tensor from _eshelby_tensor(aspect_ratio, nu_m)

    def _degraded_strengths(self, C_eff) -> dict:
        """Map degraded stiffness to degraded strength via stiffness ratio scaling.
        For each direction i: sigma_i_degraded = sigma_i_pristine * (C_eff_ii / C_m_ii)^0.5
        Returns dict with all 6 strength components needed for 3D Tsai-Wu."""

    def _tsai_wu_failure(self, stress_state, strengths) -> float:
        """Full 3D Tsai-Wu failure index.
        stress_state: 6-component (s1, s2, s3, s4, s5, s6) Voigt notation
        strengths: dict with sigma_1t, sigma_1c, sigma_2t, sigma_2c, tau_12, tau_ilss

        F_i*s_i + F_ij*s_i*s_j = 1  (full 3D, i,j = 1..6)

        Coefficients computed from strengths:
          F1 = 1/sigma_1t - 1/sigma_1c
          F11 = 1/(sigma_1t * sigma_1c)
          F2 = F3 = 1/sigma_2t - 1/sigma_2c
          F22 = F33 = 1/(sigma_2t * sigma_2c)
          F44 = 1/tau_23^2, F55 = F66 = 1/tau_12^2
          F12 = F13 = -0.5 * sqrt(F11 * F22)  (stability approximation)
        Returns failure index (>=1.0 means failure)."""
```

### Section 7: FEVisualizer

```python
class FEVisualizer:
    """Publication-quality plotting — all static methods"""

    @staticmethod
    def plot_porosity_field(porosity_field, save_path=None):
        """2-panel: through-thickness porosity profile (left) +
        plan-view distribution at midplane (right)"""

    @staticmethod
    def plot_mesh_3d(mesh, save_path=None):
        """3D hex mesh wireframe with void elements highlighted in red,
        top/bottom surfaces with grid, midplane interface"""

    @staticmethod
    def plot_mesh_detail(mesh, save_path=None):
        """Left: cross-section with porosity contour overlay
        Right: single 8-node hex element with node numbering"""

    @staticmethod
    def plot_damage_contour(mesh, solver, save_path=None):
        """2D stiffness reduction / damage map at midplane"""

    @staticmethod
    def plot_void_scf(void_geometry, save_path=None):
        """Stress concentration field around a single void
        (polar plot or 2D contour)"""

    @staticmethod
    def plot_knockdown_curves(results_by_porosity, save_path=None):
        """Strength vs porosity % for all loading modes
        Lines for each model (Judd-Wright, power law, Mori-Tanaka)
        Key summary/comparison plot"""

    @staticmethod
    def plot_model_comparison(results, save_path=None):
        """Empirical vs Mori-Tanaka predictions side by side"""
```

### Section 8: Analysis Pipeline

```python
def compare_configurations(void_volume_fraction, material_name='T800_epoxy',
                           applied_stress=-1500.0, configs=None):
    """Main analysis function — equivalent of compare_morphologies()"""
    material = MATERIALS[material_name]
    configs = configs or POROSITY_CONFIGS
    results = {}

    for name, config in configs.items():
        # Create porosity field
        porosity_field = PorosityField(material, void_volume_fraction, **config)

        # Generate mesh
        mesh = CompositeMesh(porosity_field, material)

        # Run both solvers
        empirical = EmpiricalSolver(mesh, material)
        mori_tanaka = MoriTanakaSolver(mesh, material)

        results[name] = {
            'config': config,
            'mesh': mesh,
            'empirical_solver': empirical,       # Solver object (for visualization)
            'mori_tanaka_solver': mori_tanaka,   # Solver object (for visualization)
            'empirical': empirical.get_all_failure_loads(),
            'mori_tanaka': mori_tanaka.get_all_failure_loads(),
            'porosity_field': porosity_field,
        }

    # Print comparison table with rankings
    # Rank by: compression strength, ILSS, tension strength
    return results


def main():
    """Entry point — loops over porosity severity levels"""
    porosity_levels = [0.01, 0.02, 0.03, 0.05, 0.08]  # 1% to 8%

    all_results = {}
    for Vp in porosity_levels:
        Vp_label = f"{int(Vp*100)}pct"
        results = compare_configurations(Vp)
        all_results[Vp_label] = results

        # Generate per-configuration plots
        for name in results:
            FEVisualizer.plot_porosity_field(
                results[name]['porosity_field'],
                save_path=f"porosity_profile_{name}_{Vp_label}.png")
            FEVisualizer.plot_mesh_3d(
                results[name]['mesh'],
                save_path=f"porosity_mesh_3d_{name}_{Vp_label}.png")
            FEVisualizer.plot_mesh_detail(
                results[name]['mesh'],
                save_path=f"porosity_mesh_detail_{name}_{Vp_label}.png")
            FEVisualizer.plot_damage_contour(
                results[name]['mesh'],
                results[name]['empirical_solver'],
                save_path=f"porosity_damage_{name}_{Vp_label}.png")

        # Comparison plot for this porosity level
        FEVisualizer.plot_model_comparison(
            results,
            save_path=f"porosity_comparison_{Vp_label}.png")

        # Save JSON
        save_results_to_json(results, f"porosity_analysis_results_{Vp_label}.json")

    # Cross-severity knockdown curves
    FEVisualizer.plot_knockdown_curves(
        all_results,
        save_path="porosity_knockdown_curves.png")


def save_results_to_json(results, filename):
    """Export numerical results — same pattern as wrinkle tool"""
    # Strips non-serializable objects (mesh, solver)
    # Keeps: floats, ints, lists, config strings


if __name__ == "__main__":
    main()
```

## Output Files

**Per configuration per porosity level (5 configs x 5 levels = 25 sets):**
```
porosity_profile_{config}_{Vp}.png
porosity_mesh_3d_{config}_{Vp}.png
porosity_mesh_detail_{config}_{Vp}.png
porosity_damage_{config}_{Vp}.png
```

**Per porosity level (5 files):**
```
porosity_comparison_{Vp}.png
porosity_analysis_results_{Vp}.json
```

**Cross-severity (1 file):**
```
porosity_knockdown_curves.png
```

## JSON Output Structure

```json
{
  "uniform_spherical": {
    "config": {"distribution": "uniform", "void_shape": "spherical"},
    "void_volume_fraction": 0.03,
    "empirical": {
      "compression": {
        "judd_wright": {"failure_stress_MPa": 1365, "knockdown": 0.910},
        "power_law":   {"failure_stress_MPa": 1434, "knockdown": 0.956},
        "linear":      {"failure_stress_MPa": 1350, "knockdown": 0.900}
      },
      "tension": {...},
      "shear": {...},
      "ilss": {...}
    },
    "mori_tanaka": {
      "compression": {"failure_stress_MPa": 1380, "knockdown": 0.920},
      "tension": {...},
      "shear": {...},
      "ilss": {...}
    }
  },
  "clustered_midplane": {...},
  ...
}
```

## Key Physics

### Empirical Models

**Judd-Wright exponential decay:**
```
sigma/sigma_0 = exp(-alpha * Vp)
```
where alpha depends on loading mode (ILSS most sensitive, alpha ~ 5.5).

**Power law:**
```
sigma/sigma_0 = (1 - Vp)^n
```

**Linear:**
```
sigma/sigma_0 = 1 - beta * Vp
```

### Mori-Tanaka Micromechanics

For void inclusions (C_i = 0), the Mori-Tanaka effective stiffness is:
```
T = (I - S)^-1                              (void strain concentration tensor)
C_eff = C_m * {I - Vp * [I - (1-Vp)*S]^-1}  (effective stiffness)
```
where S is the Eshelby tensor for the void shape (sphere, prolate, or oblate spheroid),
and C_m is the isotropic matrix stiffness tensor assembled from E_m and nu_m.

Eshelby tensor shape mapping:
- `spherical` (a=b=c) → isotropic Eshelby tensor, closed-form from nu_m
- `cylindrical` (a>b=c) → prolate spheroid Eshelby tensor
- `penny` (a=b>c) → oblate spheroid Eshelby tensor

### Stress Concentration Factors

From Eshelby inclusion theory (used for discrete macrovoids):
- Spherical void: SCF ~ 2.0 (all modes)
- Cylindrical void: SCF depends on loading direction vs void axis
- Penny-shaped void: SCF >> 2.0 (crack-like, mode-dependent)

SCFs are applied as local amplification factors at discrete void boundaries,
combined multiplicatively with the distributed porosity knockdown.

### Failure Criterion

Full 3D Tsai-Wu for multi-axial assessment:
```
F_i * s_i + F_ij * s_i * s_j = 1    (i, j = 1..6, Voigt notation)
```

Coefficients from ply-level strengths (sigma_1t, sigma_1c, sigma_2t, sigma_2c, tau_12, tau_ilss):
```
F1  = 1/sigma_1t - 1/sigma_1c
F11 = 1/(sigma_1t * sigma_1c)
F2  = F3 = 1/sigma_2t - 1/sigma_2c
F22 = F33 = 1/(sigma_2t * sigma_2c)
F44 = 1/tau_23^2,  F55 = F66 = 1/tau_12^2
F12 = F13 = -0.5 * sqrt(F11 * F22)   (stability approximation)
```

Strength degradation from stiffness: sigma_i_degraded = sigma_i_pristine * sqrt(C_eff_ii / C_pristine_ii)

## Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
```

No optional dependencies — fully self-contained unlike the wrinkle tool.

## Comparison with Wrinkle Tool

| Aspect | Wrinkle Tool | Porosity Tool |
|--------|-------------|---------------|
| Defect geometry | Gaussian-sinusoidal profile | Ellipsoidal voids + field |
| Configurations | 3 (stack/convex/concave) | 5 (shape x distribution) |
| Loading modes | Compression, tension | Compression, tension, shear, ILSS |
| Fast solver | Budiansky-Fleck kink-band | Judd-Wright / power law |
| Detailed solver | Real FE (stiffness assembly) | Mori-Tanaka micromechanics |
| Mesh deformation | Nodes displaced by wrinkle | Flat mesh, stiffness reduced |
| Materials | 1 preset (T800) | 3 presets (T800, T700, glass) |
| Statistics | Full Monte Carlo + Jensen gap | Basic (mean, std, percentiles) |
| Severity levels | 2A, 3A, 4A, 5A | 1%, 2%, 3%, 5%, 8% |
