# Porosity FE Analysis Tool Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python tool for assessing porosity defect effects on composite laminate strength, mirroring the architecture of the Double_Wrinkle `jin_fe_analysis` tool.

**Architecture:** Single-file implementation (`porosity_fe_analysis.py`) with 8 sections: MaterialProperties, VoidGeometry, PorosityField, CompositeMesh, EmpiricalSolver, MoriTanakaSolver, FEVisualizer, and Analysis Pipeline. Two solver tiers — fast empirical (Judd-Wright/power-law/linear) and detailed Mori-Tanaka micromechanics.

**Tech Stack:** Python 3.8+, numpy, scipy, matplotlib

**Spec:** `docs/superpowers/specs/2026-03-26-porosity-fe-design.md`
**Reference implementation:** `/Users/elhajjar/Library/CloudStorage/OneDrive-UWM/AI/Double_Wrinkle/jin_fe_analysis/dual_wrinkle_fe_analysis.py`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `porosity_fe_analysis.py` (header + imports only)
- Create: `tests/test_porosity_fe.py` (test scaffold)

- [ ] **Step 1: Create requirements.txt**

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
pytest>=7.0.0
```

- [ ] **Step 2: Create porosity_fe_analysis.py with header and imports**

```python
#!/usr/bin/env python3
"""
POROSITY DEFECT ANALYSIS FOR COMPOSITE LAMINATES
==================================================

Evaluates the effects of porosity (distributed microporosity and discrete
macrovoids) on composite laminate strength under multiple loading modes
(compression, tension, shear, ILSS).

Supports five porosity configurations across three material presets.
Two solver tiers: empirical (Judd-Wright, power law, linear) and
Mori-Tanaka micromechanics.

Based on:
- Judd & Wright - Empirical porosity-strength relationships
- Mori-Tanaka (1973) - Mean-field micromechanics homogenization
- Eshelby (1957) - Inclusion theory for void stress concentration
- Tsai-Wu - 3D failure criterion

Dependencies:
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union
import json
```

- [ ] **Step 3: Create test scaffold**

```python
#!/usr/bin/env python3
"""Tests for porosity_fe_analysis.py"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 4: Install dependencies and verify**

Run: `cd /Users/elhajjar/Library/CloudStorage/OneDrive-UWM/AI/Porosity_FE && pip install -r requirements.txt`
Expected: Successfully installed all packages

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: scaffold porosity FE analysis project"
```

---

### Task 2: MaterialProperties Dataclass + Presets

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 1)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for MaterialProperties**

Add to `tests/test_porosity_fe.py`:

```python
from porosity_fe_analysis import MaterialProperties, MATERIALS


class TestMaterialProperties:
    def test_dataclass_creation(self):
        mat = MATERIALS['T800_epoxy']
        assert mat.E11 == 161000.0
        assert mat.sigma_1c == 1500.0
        assert mat.sigma_2t == 80.0
        assert mat.tau_12 == 100.0
        assert mat.tau_ilss == 90.0
        assert mat.matrix_modulus == 3500.0
        assert mat.fiber_volume_fraction == 0.60

    def test_all_presets_exist(self):
        assert 'T800_epoxy' in MATERIALS
        assert 'T700_epoxy' in MATERIALS
        assert 'glass_epoxy' in MATERIALS

    def test_total_thickness(self):
        mat = MATERIALS['T800_epoxy']
        expected = 0.183 * 24
        assert abs(mat.total_thickness - expected) < 1e-10

    def test_stiffness_matrix_shape(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        assert C.shape == (6, 6)

    def test_stiffness_matrix_symmetric(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        np.testing.assert_allclose(C, C.T, atol=1e-6)

    def test_stiffness_matrix_positive_definite(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > 0)

    def test_compliance_is_inverse_of_stiffness(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        S = mat.get_compliance_matrix()
        np.testing.assert_allclose(C @ S, np.eye(6), atol=1e-6)

    def test_isotropic_matrix_stiffness_shape(self):
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        assert C_m.shape == (6, 6)

    def test_isotropic_matrix_stiffness_symmetric(self):
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        np.testing.assert_allclose(C_m, C_m.T, atol=1e-6)

    def test_isotropic_matrix_stiffness_values(self):
        """C_m should reflect E_m=3500, nu_m=0.35"""
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        E_m, nu_m = 3500.0, 0.35
        lam = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
        mu = E_m / (2 * (1 + nu_m))
        assert abs(C_m[0, 0] - (lam + 2 * mu)) < 1.0
        assert abs(C_m[0, 1] - lam) < 1.0
        assert abs(C_m[3, 3] - mu) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/elhajjar/Library/CloudStorage/OneDrive-UWM/AI/Porosity_FE && python -m pytest tests/test_porosity_fe.py::TestMaterialProperties -v`
Expected: FAIL — ImportError, MaterialProperties not defined

- [ ] **Step 3: Implement MaterialProperties and MATERIALS**

Add to `porosity_fe_analysis.py` after imports:

```python
# ============================================================
# SECTION 1: MATERIAL PROPERTIES AND CONSTANTS
# ============================================================

@dataclass
class MaterialProperties:
    """Composite material properties with constituent data for micromechanics."""
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

    # Geometric
    t_ply: float        # Ply thickness (mm)
    n_plies: int        # Number of plies

    # Constituent properties (for Mori-Tanaka)
    matrix_modulus: float         # E_m (MPa)
    matrix_poisson: float         # nu_m
    fiber_modulus: float          # E_f (MPa)
    fiber_volume_fraction: float  # V_f (pristine)

    @property
    def total_thickness(self) -> float:
        return self.t_ply * self.n_plies

    def get_compliance_matrix(self) -> np.ndarray:
        """6x6 compliance matrix [S] for orthotropic material."""
        S = np.zeros((6, 6))
        S[0, 0] = 1.0 / self.E11
        S[1, 1] = 1.0 / self.E22
        S[2, 2] = 1.0 / self.E33
        S[0, 1] = S[1, 0] = -self.nu12 / self.E11
        S[0, 2] = S[2, 0] = -self.nu13 / self.E11
        S[1, 2] = S[2, 1] = -self.nu23 / self.E22
        S[3, 3] = 1.0 / self.G23
        S[4, 4] = 1.0 / self.G13
        S[5, 5] = 1.0 / self.G12
        return S

    def get_stiffness_matrix(self) -> np.ndarray:
        """6x6 stiffness matrix [C] = [S]^-1."""
        return np.linalg.inv(self.get_compliance_matrix())

    def get_isotropic_matrix_stiffness(self) -> np.ndarray:
        """6x6 isotropic stiffness tensor C_m from matrix_modulus and matrix_poisson."""
        E_m = self.matrix_modulus
        nu_m = self.matrix_poisson
        lam = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
        mu = E_m / (2 * (1 + nu_m))
        C_m = np.zeros((6, 6))
        C_m[0, 0] = C_m[1, 1] = C_m[2, 2] = lam + 2 * mu
        C_m[0, 1] = C_m[0, 2] = C_m[1, 0] = C_m[1, 2] = C_m[2, 0] = C_m[2, 1] = lam
        C_m[3, 3] = C_m[4, 4] = C_m[5, 5] = mu
        return C_m


MATERIALS = {
    'T800_epoxy': MaterialProperties(
        E11=161000.0, E22=11380.0, E33=11380.0,
        G12=5170.0, G13=5170.0, G23=3980.0,
        nu12=0.32, nu13=0.32, nu23=0.40,
        sigma_1c=1500.0, sigma_1t=2800.0, sigma_2t=80.0, sigma_2c=250.0,
        tau_12=100.0, tau_ilss=90.0,
        t_ply=0.183, n_plies=24,
        matrix_modulus=3500.0, matrix_poisson=0.35,
        fiber_modulus=294000.0, fiber_volume_fraction=0.60,
    ),
    'T700_epoxy': MaterialProperties(
        E11=132000.0, E22=10300.0, E33=10300.0,
        G12=4700.0, G13=4700.0, G23=3500.0,
        nu12=0.30, nu13=0.30, nu23=0.40,
        sigma_1c=1200.0, sigma_1t=2400.0, sigma_2t=65.0, sigma_2c=200.0,
        tau_12=85.0, tau_ilss=80.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=3200.0, matrix_poisson=0.35,
        fiber_modulus=230000.0, fiber_volume_fraction=0.58,
    ),
    'glass_epoxy': MaterialProperties(
        E11=45000.0, E22=12000.0, E33=12000.0,
        G12=5500.0, G13=5500.0, G23=4000.0,
        nu12=0.28, nu13=0.28, nu23=0.40,
        sigma_1c=600.0, sigma_1t=1100.0, sigma_2t=40.0, sigma_2c=140.0,
        tau_12=70.0, tau_ilss=55.0,
        t_ply=0.200, n_plies=24,
        matrix_modulus=3500.0, matrix_poisson=0.35,
        fiber_modulus=73000.0, fiber_volume_fraction=0.55,
    ),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/elhajjar/Library/CloudStorage/OneDrive-UWM/AI/Porosity_FE && python -m pytest tests/test_porosity_fe.py::TestMaterialProperties -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add MaterialProperties dataclass with 3 material presets"
```

---

### Task 3: VoidGeometry Class + Shape Presets

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 2)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for VoidGeometry**

```python
from porosity_fe_analysis import VoidGeometry, VOID_SHAPES


class TestVoidGeometry:
    def test_sphere_creation(self):
        void = VoidGeometry(center=(10, 5, 2), radii=(1.0, 1.0, 1.0))
        np.testing.assert_array_equal(void.center, [10, 5, 2])
        np.testing.assert_array_equal(void.radii, [1.0, 1.0, 1.0])
        assert void.orientation == 0.0

    def test_contains_center(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])
        assert void.contains(x, y, z)[0] == True

    def test_contains_outside(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        x = np.array([2.0])
        y = np.array([0.0])
        z = np.array([0.0])
        assert void.contains(x, y, z)[0] == False

    def test_contains_boundary(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        x = np.array([1.0])
        y = np.array([0.0])
        z = np.array([0.0])
        assert void.contains(x, y, z)[0] == True  # <= 1

    def test_ellipsoidal_contains(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1))
        # Inside along major axis
        assert void.contains(np.array([2.5]), np.array([0.0]), np.array([0.0]))[0] == True
        # Outside along minor axis
        assert void.contains(np.array([0.0]), np.array([1.5]), np.array([0.0]))[0] == False

    def test_volume_sphere(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(2, 2, 2))
        expected = (4 / 3) * np.pi * 8
        assert abs(void.volume() - expected) < 1e-10

    def test_volume_ellipsoid(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 2, 1))
        expected = (4 / 3) * np.pi * 6
        assert abs(void.volume() - expected) < 1e-10

    def test_aspect_ratio_sphere(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        assert void.aspect_ratio == 1.0

    def test_aspect_ratio_elongated(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1))
        assert void.aspect_ratio == 3.0

    def test_scf_sphere(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        scf = void.stress_concentration_factor()
        assert isinstance(scf, dict)
        assert 'compression' in scf
        assert 'tension' in scf
        assert 'shear' in scf
        assert 'ilss' in scf
        assert scf['compression'] > 1.0

    def test_distance_field_inside_negative(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        d = void.distance_field(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert d[0] < 0  # Inside → negative

    def test_distance_field_outside_positive(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        d = void.distance_field(np.array([2.0]), np.array([0.0]), np.array([0.0]))
        assert d[0] > 0  # Outside → positive

    def test_void_shapes_presets(self):
        assert 'spherical' in VOID_SHAPES
        assert 'cylindrical' in VOID_SHAPES
        assert 'penny' in VOID_SHAPES
        assert VOID_SHAPES['spherical'] == (1.0, 1.0, 1.0)

    def test_orientation_rotation(self):
        """Rotated cylindrical void should contain points along rotated axis"""
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1), orientation=np.pi / 2)
        # After 90-degree rotation, major axis is along y
        assert void.contains(np.array([0.0]), np.array([2.5]), np.array([0.0]))[0] == True
        assert void.contains(np.array([2.5]), np.array([0.0]), np.array([0.0]))[0] == False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestVoidGeometry -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement VoidGeometry and VOID_SHAPES**

Add to `porosity_fe_analysis.py`:

```python
# ============================================================
# SECTION 2: VOID GEOMETRY MODEL
# ============================================================

VOID_SHAPES = {
    'spherical':   (1.0, 1.0, 1.0),
    'cylindrical': (3.0, 1.0, 1.0),
    'penny':       (3.0, 3.0, 0.3),
}


class VoidGeometry:
    """Single void parameterization — equivalent of WrinkleGeometry."""

    def __init__(self, center: Tuple, radii: Tuple, orientation: float = 0.0):
        self.center = np.array(center, dtype=float)
        self.radii = np.array(radii, dtype=float)
        self.orientation = orientation

    def _to_local(self, x, y, z):
        """Transform world coordinates to void-local (translated + rotated)."""
        dx = np.asarray(x, dtype=float) - self.center[0]
        dy = np.asarray(y, dtype=float) - self.center[1]
        dz = np.asarray(z, dtype=float) - self.center[2]
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        z_loc = dz
        return x_loc, y_loc, z_loc

    def contains(self, x, y, z) -> np.ndarray:
        x_l, y_l, z_l = self._to_local(x, y, z)
        val = (x_l / self.radii[0])**2 + (y_l / self.radii[1])**2 + (z_l / self.radii[2])**2
        return val <= 1.0

    def distance_field(self, x, y, z) -> np.ndarray:
        x_l, y_l, z_l = self._to_local(x, y, z)
        val = np.sqrt((x_l / self.radii[0])**2 + (y_l / self.radii[1])**2 + (z_l / self.radii[2])**2)
        r_eff = np.sqrt(x_l**2 + y_l**2 + z_l**2)
        r_eff = np.maximum(r_eff, 1e-12)
        return r_eff * (val - 1.0) / val

    def stress_concentration_factor(self) -> dict:
        ar = self.aspect_ratio
        if ar < 1.2:  # Spherical
            return {'compression': 2.0, 'tension': 2.0, 'shear': 1.5, 'ilss': 1.8}
        elif self.radii[0] > self.radii[2]:
            if self.radii[1] < self.radii[0] * 0.5:  # Cylindrical (prolate)
                return {'compression': 1.5 + 0.5 * ar, 'tension': 1.5 + 0.5 * ar,
                        'shear': 1.3 + 0.3 * ar, 'ilss': 1.5 + 0.4 * ar}
            else:  # Penny (oblate)
                return {'compression': 2.0 + 1.0 * ar, 'tension': 2.0 + 1.5 * ar,
                        'shear': 1.5 + 0.8 * ar, 'ilss': 2.0 + 1.2 * ar}
        else:
            return {'compression': 2.0, 'tension': 2.0, 'shear': 1.5, 'ilss': 1.8}

    def volume(self) -> float:
        return (4.0 / 3.0) * np.pi * self.radii[0] * self.radii[1] * self.radii[2]

    @property
    def aspect_ratio(self) -> float:
        return float(np.max(self.radii) / np.min(self.radii))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestVoidGeometry -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add VoidGeometry class with shape presets and SCF"
```

---

### Task 4: PorosityField Class + Configuration Presets

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 3)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for PorosityField**

```python
from porosity_fe_analysis import PorosityField, POROSITY_CONFIGS, MATERIALS


class TestPorosityField:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']

    def test_uniform_constant_porosity(self):
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        Lz = self.material.total_thickness
        z_mid = Lz / 2
        Vp = pf.local_porosity(np.array([10.0]), np.array([5.0]), np.array([z_mid]))
        assert abs(Vp[0] - 0.03) < 1e-10

    def test_uniform_same_everywhere(self):
        pf = PorosityField(self.material, 0.05, distribution='uniform')
        Lz = self.material.total_thickness
        z_vals = np.linspace(0, Lz, 10)
        x = np.full_like(z_vals, 10.0)
        y = np.full_like(z_vals, 5.0)
        Vp = pf.local_porosity(x, y, z_vals)
        np.testing.assert_allclose(Vp, 0.05, atol=1e-10)

    def test_clustered_midplane_higher_at_center(self):
        pf = PorosityField(self.material, 0.05, distribution='clustered',
                           cluster_location='midplane')
        Lz = self.material.total_thickness
        Vp_mid = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                    np.array([Lz / 2]))[0]
        Vp_edge = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                     np.array([0.0]))[0]
        assert Vp_mid > Vp_edge

    def test_clustered_surface_higher_at_surface(self):
        pf = PorosityField(self.material, 0.05, distribution='clustered',
                           cluster_location='surface')
        Lz = self.material.total_thickness
        Vp_surface = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                        np.array([0.0]))[0]
        Vp_mid = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                    np.array([Lz / 2]))[0]
        assert Vp_surface > Vp_mid

    def test_interface_peaks_at_ply_boundaries(self):
        pf = PorosityField(self.material, 0.05, distribution='interface')
        Lz = self.material.total_thickness
        t = self.material.t_ply
        Vp_interface = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                          np.array([t]))[0]
        Vp_midply = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                       np.array([t / 2]))[0]
        assert Vp_interface > Vp_midply

    def test_stiffness_reduction_pristine_is_one(self):
        pf = PorosityField(self.material, 0.0, distribution='uniform')
        sr = pf.local_stiffness_reduction(np.array([10.0]), np.array([5.0]),
                                           np.array([1.0]))
        assert abs(sr[0] - 1.0) < 1e-10

    def test_stiffness_reduction_decreases_with_porosity(self):
        pf = PorosityField(self.material, 0.05, distribution='uniform')
        sr = pf.local_stiffness_reduction(np.array([10.0]), np.array([5.0]),
                                           np.array([1.0]))
        assert sr[0] < 1.0
        assert sr[0] > 0.0

    def test_porosity_clamped_to_one(self):
        """With very high Vp and a discrete void, should not exceed 1.0"""
        void = VoidGeometry(center=(10, 5, 1), radii=(1, 1, 0.5))
        pf = PorosityField(self.material, 0.90, distribution='uniform',
                           discrete_voids=[void])
        Vp = pf.local_porosity(np.array([10.0]), np.array([5.0]), np.array([1.0]))
        assert Vp[0] <= 1.0

    def test_effective_porosity_profile_shape(self):
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        z, Vp = pf.effective_porosity_profile(nz=50)
        assert len(z) == 50
        assert len(Vp) == 50

    def test_configs_all_exist(self):
        assert len(POROSITY_CONFIGS) == 5
        for name in ['uniform_spherical', 'uniform_cylindrical',
                      'clustered_midplane', 'clustered_surface', 'interface_penny']:
            assert name in POROSITY_CONFIGS

    def test_void_shape_string_resolved(self):
        pf = PorosityField(self.material, 0.03, void_shape='cylindrical')
        assert pf.void_shape_radii == (3.0, 1.0, 1.0)

    def test_void_shape_tuple_accepted(self):
        pf = PorosityField(self.material, 0.03, void_shape=(2.0, 1.5, 0.5))
        assert pf.void_shape_radii == (2.0, 1.5, 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestPorosityField -v`
Expected: FAIL

- [ ] **Step 3: Implement PorosityField and POROSITY_CONFIGS**

Add to `porosity_fe_analysis.py`:

```python
# ============================================================
# SECTION 3: POROSITY FIELD MODEL
# ============================================================

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


class PorosityField:
    """Distributed + discrete porosity field."""

    def __init__(self, material: MaterialProperties, void_volume_fraction: float,
                 distribution: str = 'uniform', void_shape: Union[str, Tuple] = 'spherical',
                 cluster_location: str = 'midplane',
                 discrete_voids: Optional[List[VoidGeometry]] = None):
        self.material = material
        self.Vp = void_volume_fraction
        self.distribution = distribution
        self.cluster_location = cluster_location
        self.discrete_voids = discrete_voids or []
        self.Lz = material.total_thickness

        # Resolve void shape
        if isinstance(void_shape, str):
            self.void_shape_radii = VOID_SHAPES[void_shape]
        else:
            self.void_shape_radii = tuple(void_shape)

    def _distributed_porosity(self, z: np.ndarray) -> np.ndarray:
        """Through-thickness distributed porosity profile."""
        z = np.asarray(z, dtype=float)
        if self.distribution == 'uniform':
            return np.full_like(z, self.Vp)
        elif self.distribution == 'clustered':
            if self.cluster_location == 'midplane':
                z0 = self.Lz / 2
            elif self.cluster_location == 'surface':
                z0 = 0.0
            else:  # quarter
                z0 = self.Lz / 4
            sigma = self.Lz / 6
            profile = np.exp(-0.5 * ((z - z0) / sigma)**2)
            # Normalize so average equals Vp
            norm = np.mean(profile) if np.mean(profile) > 0 else 1.0
            return self.Vp * profile / norm
        elif self.distribution == 'interface':
            t = self.material.t_ply
            n = self.material.n_plies
            profile = np.zeros_like(z)
            for k in range(1, n):
                z_int = k * t
                profile += np.exp(-0.5 * ((z - z_int) / (t * 0.15))**2)
            norm = np.mean(profile) if np.mean(profile) > 0 else 1.0
            return self.Vp * profile / norm
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def local_porosity(self, x, y, z) -> np.ndarray:
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        Vp_dist = self._distributed_porosity(z)
        Vp_discrete = np.zeros_like(Vp_dist)
        for void in self.discrete_voids:
            Vp_discrete = np.maximum(Vp_discrete,
                                      void.contains(x, y, z).astype(float))
        return np.minimum(Vp_dist + Vp_discrete, 1.0)

    def local_stiffness_reduction(self, x, y, z) -> np.ndarray:
        Vp_local = self.local_porosity(x, y, z)
        return 1.0 - Vp_local

    def get_void_locations(self) -> list:
        return [(v.center.tolist(), v.radii.tolist()) for v in self.discrete_voids]

    def effective_porosity_profile(self, nz: int = 100) -> tuple:
        """Through-thickness profile including discrete void contributions."""
        z_coords = np.linspace(0, self.Lz, nz)
        x_mid = np.full(nz, 25.0)  # Sample at domain center
        y_mid = np.full(nz, 10.0)
        Vp_vals = self.local_porosity(x_mid, y_mid, z_coords)
        return z_coords, Vp_vals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestPorosityField -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add PorosityField with 3 distribution types and 5 config presets"
```

---

### Task 5: CompositeMesh Class

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 4)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for CompositeMesh**

```python
from porosity_fe_analysis import CompositeMesh


class TestCompositeMesh:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_mesh_creation(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.nodes is not None
        assert mesh.elements is not None

    def test_node_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        expected_nodes = 11 * 6 * 7  # (nx+1)*(ny+1)*(nz+1)
        assert len(mesh.nodes) == expected_nodes

    def test_element_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        expected_elements = 10 * 5 * 6
        assert len(mesh.elements) == expected_elements

    def test_nodes_3d(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.nodes.shape[1] == 3

    def test_hex_elements_8_nodes(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.elements.shape[1] == 8

    def test_porosity_field_sampled(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.porosity) == len(mesh.nodes)
        # Uniform 3% → all nodes should be ~0.03
        np.testing.assert_allclose(mesh.porosity, 0.03, atol=1e-10)

    def test_stiffness_reduction_sampled(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.stiffness_reduction) == len(mesh.nodes)
        np.testing.assert_allclose(mesh.stiffness_reduction, 0.97, atol=1e-10)

    def test_ply_ids_range(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert np.min(mesh.ply_ids) >= 0
        assert np.max(mesh.ply_ids) <= self.material.n_plies

    def test_domain_bounds(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert np.min(mesh.nodes[:, 0]) >= 0
        assert np.min(mesh.nodes[:, 2]) >= 0
        assert abs(np.max(mesh.nodes[:, 2]) - self.material.total_thickness) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestCompositeMesh -v`
Expected: FAIL

- [ ] **Step 3: Implement CompositeMesh**

Add to `porosity_fe_analysis.py`:

```python
# ============================================================
# SECTION 4: MESH GENERATION
# ============================================================

class CompositeMesh:
    """3D structured hex mesh with porosity."""

    def __init__(self, porosity_field: PorosityField, material: MaterialProperties,
                 nx: int = 50, ny: int = 20, nz: int = 24):
        self.porosity_field = porosity_field
        self.material = material
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.L_x = 50.0
        self.L_y = 20.0
        self.L_z = material.total_thickness

        self.nodes = None
        self.elements = None
        self.porosity = None
        self.stiffness_reduction = None
        self.ply_ids = None
        self.void_elements = None

        self.generate_mesh()

    def generate_mesh(self):
        x = np.linspace(0, self.L_x, self.nx + 1)
        y = np.linspace(0, self.L_y, self.ny + 1)
        z = np.linspace(0, self.L_z, self.nz + 1)

        nodes = []
        porosity_vals = []
        ply_ids = []

        for k, zk in enumerate(z):
            for j, yj in enumerate(y):
                for i, xi in enumerate(x):
                    nodes.append([xi, yj, zk])

        self.nodes = np.array(nodes)

        # Sample porosity at all nodes
        self.porosity = self.porosity_field.local_porosity(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])
        self.stiffness_reduction = self.porosity_field.local_stiffness_reduction(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])

        # Ply IDs
        z_normalized = self.nodes[:, 2] / self.L_z
        self.ply_ids = np.clip((z_normalized * self.material.n_plies).astype(int),
                               0, self.material.n_plies - 1)

        # Hex element connectivity
        elements = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n0 = k * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + i
                    n1 = n0 + 1
                    n2 = n0 + (self.nx + 1) + 1
                    n3 = n0 + (self.nx + 1)
                    n4 = n0 + (self.ny + 1) * (self.nx + 1)
                    n5 = n4 + 1
                    n6 = n4 + (self.nx + 1) + 1
                    n7 = n4 + (self.nx + 1)
                    elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

        self.elements = np.array(elements)

        # Identify void elements (average porosity > 0.95)
        elem_porosity = np.mean(self.porosity[self.elements], axis=1)
        self.void_elements = np.where(elem_porosity > 0.95)[0]

        print(f"Mesh generated: {len(self.nodes)} nodes, {len(self.elements)} elements")
        print(f"  Domain: {self.L_x:.1f} x {self.L_y:.1f} x {self.L_z:.2f} mm")
        print(f"  Void elements: {len(self.void_elements)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestCompositeMesh -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add CompositeMesh with hex element connectivity and porosity sampling"
```

---

### Task 6: EmpiricalSolver

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 5)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for EmpiricalSolver**

```python
from porosity_fe_analysis import EmpiricalSolver


class TestEmpiricalSolver:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = EmpiricalSolver(self.mesh, self.material)

    def test_judd_wright_zero_porosity(self):
        """At Vp=0, knockdown should be 1.0"""
        kd = self.solver._judd_wright(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_judd_wright_decreasing(self):
        """Higher porosity → lower knockdown"""
        kd1 = self.solver._judd_wright(0.01, 'compression')
        kd2 = self.solver._judd_wright(0.05, 'compression')
        assert kd1 > kd2

    def test_power_law_zero_porosity(self):
        kd = self.solver._power_law(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_linear_zero_porosity(self):
        kd = self.solver._linear(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_ilss_most_sensitive(self):
        """ILSS should have largest knockdown for same porosity"""
        Vp = 0.05
        kd_comp = self.solver._judd_wright(Vp, 'compression')
        kd_ilss = self.solver._judd_wright(Vp, 'ilss')
        assert kd_ilss < kd_comp

    def test_get_failure_load_returns_dict(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert 'failure_stress' in result
        assert 'knockdown' in result
        assert 'model' in result

    def test_failure_load_positive(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert result['failure_stress'] > 0
        assert 0 < result['knockdown'] <= 1.0

    def test_get_all_failure_loads(self):
        results = self.solver.get_all_failure_loads()
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            assert mode in results
            for model in ['judd_wright', 'power_law', 'linear']:
                assert model in results[mode]

    def test_failure_stress_below_pristine(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert result['failure_stress'] < self.material.sigma_1c

    def test_discrete_void_scf_amplifies_knockdown(self):
        """Discrete macrovoid should cause worse knockdown near the void."""
        material = MATERIALS['T800_epoxy']
        void = VoidGeometry(center=(25, 10, material.total_thickness / 2),
                            radii=(2, 2, 0.5))
        pf = PorosityField(material, 0.02, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=10, ny=5, nz=6)
        solver = EmpiricalSolver(mesh, material)
        result_with_void = solver.get_failure_load('compression', 'judd_wright')

        pf_no_void = PorosityField(material, 0.02, distribution='uniform')
        mesh_no_void = CompositeMesh(pf_no_void, material, nx=10, ny=5, nz=6)
        solver_no_void = EmpiricalSolver(mesh_no_void, material)
        result_no_void = solver_no_void.get_failure_load('compression', 'judd_wright')

        # Discrete void should reduce strength further
        assert result_with_void['knockdown'] < result_no_void['knockdown']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestEmpiricalSolver -v`
Expected: FAIL

- [ ] **Step 3: Implement EmpiricalSolver**

Add to `porosity_fe_analysis.py`:

```python
# ============================================================
# SECTION 5: EMPIRICAL SOLVER
# ============================================================

class EmpiricalSolver:
    """Fast analytical solver using empirical porosity-strength models."""

    JUDD_WRIGHT_ALPHA = {
        'compression': 3.0, 'tension': 2.0, 'shear': 4.0, 'ilss': 5.5,
    }
    POWER_LAW_N = {
        'compression': 1.5, 'tension': 1.2, 'shear': 2.0, 'ilss': 2.5,
    }
    LINEAR_BETA = {
        'compression': 10.0, 'tension': 7.0, 'shear': 12.0, 'ilss': 15.0,
    }
    PRISTINE_STRENGTH_KEY = {
        'compression': 'sigma_1c', 'tension': 'sigma_1t',
        'shear': 'tau_12', 'ilss': 'tau_ilss',
    }

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties):
        self.mesh = mesh
        self.material = material
        self.nodal_knockdown = None

    def _judd_wright(self, Vp: float, mode: str) -> float:
        alpha = self.JUDD_WRIGHT_ALPHA[mode]
        return float(np.exp(-alpha * Vp))

    def _power_law(self, Vp: float, mode: str) -> float:
        n = self.POWER_LAW_N[mode]
        return float((1.0 - Vp)**n)

    def _linear(self, Vp: float, mode: str) -> float:
        beta = self.LINEAR_BETA[mode]
        return float(max(1.0 - beta * Vp, 0.0))

    def _get_pristine_strength(self, mode: str) -> float:
        return getattr(self.material, self.PRISTINE_STRENGTH_KEY[mode])

    def _apply_discrete_void_scf(self, base_knockdown: np.ndarray, mode: str) -> np.ndarray:
        kd = base_knockdown.copy()
        for void in self.mesh.porosity_field.discrete_voids:
            scf_dict = void.stress_concentration_factor()
            scf = scf_dict.get(mode, 1.0)
            dist = void.distance_field(self.mesh.nodes[:, 0],
                                        self.mesh.nodes[:, 1],
                                        self.mesh.nodes[:, 2])
            influence = np.exp(-np.maximum(dist, 0) / max(void.radii))
            kd *= (1.0 - influence * (1.0 - 1.0 / scf))
        return kd

    def apply_loading(self, mode: str = 'compression', model: str = 'judd_wright'):
        model_func = {'judd_wright': self._judd_wright,
                      'power_law': self._power_law,
                      'linear': self._linear}[model]
        kd = np.array([model_func(Vp, mode) for Vp in self.mesh.porosity])
        kd = self._apply_discrete_void_scf(kd, mode)
        self.nodal_knockdown = kd

    def get_failure_load(self, mode: str = 'compression', model: str = 'judd_wright') -> dict:
        self.apply_loading(mode, model)
        sigma_0 = self._get_pristine_strength(mode)
        min_kd = float(np.min(self.nodal_knockdown))
        critical_idx = int(np.argmin(self.nodal_knockdown))
        return {
            'failure_stress': sigma_0 * min_kd,
            'knockdown': min_kd,
            'critical_location': self.mesh.nodes[critical_idx].tolist(),
            'model': model,
        }

    def get_all_failure_loads(self) -> dict:
        results = {}
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            results[mode] = {}
            for model in ['judd_wright', 'power_law', 'linear']:
                results[mode][model] = self.get_failure_load(mode, model)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestEmpiricalSolver -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add EmpiricalSolver with Judd-Wright, power law, and linear models"
```

---

### Task 7: MoriTanakaSolver

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 6)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for MoriTanakaSolver**

```python
from porosity_fe_analysis import MoriTanakaSolver


class TestMoriTanakaSolver:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = MoriTanakaSolver(self.mesh, self.material)

    def test_eshelby_tensor_sphere_shape(self):
        S = self.solver._eshelby_tensor(1.0, 0.35)
        assert S.shape == (6, 6)

    def test_eshelby_tensor_sphere_symmetric(self):
        S = self.solver._eshelby_tensor(1.0, 0.35)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_eshelby_tensor_sphere_known_value(self):
        """For sphere, S_1111 = (7-5*nu) / (15*(1-nu))"""
        nu = 0.35
        S = self.solver._eshelby_tensor(1.0, nu)
        expected_S11 = (7 - 5 * nu) / (15 * (1 - nu))
        assert abs(S[0, 0] - expected_S11) < 1e-10

    def test_effective_stiffness_zero_porosity(self):
        C_eff = self.solver._effective_stiffness(0.0, (1, 1, 1))
        C_m = self.material.get_isotropic_matrix_stiffness()
        np.testing.assert_allclose(C_eff, C_m, atol=1e-6)

    def test_effective_stiffness_decreases_with_porosity(self):
        C_0 = self.solver._effective_stiffness(0.01, (1, 1, 1))
        C_5 = self.solver._effective_stiffness(0.05, (1, 1, 1))
        assert C_0[0, 0] > C_5[0, 0]

    def test_effective_stiffness_positive_definite(self):
        C_eff = self.solver._effective_stiffness(0.05, (1, 1, 1))
        eigenvalues = np.linalg.eigvalsh(C_eff)
        assert np.all(eigenvalues > 0)

    def test_tsai_wu_no_load_below_one(self):
        """Zero stress should give failure index 0"""
        strengths = {
            'sigma_1t': 2800, 'sigma_1c': 1500,
            'sigma_2t': 80, 'sigma_2c': 250,
            'tau_12': 100, 'tau_ilss': 90,
        }
        fi = self.solver._tsai_wu_failure(np.zeros(6), strengths)
        assert abs(fi) < 1e-10

    def test_tsai_wu_compression_failure(self):
        """At pristine compression strength, should be near 1.0"""
        strengths = {
            'sigma_1t': 2800, 'sigma_1c': 1500,
            'sigma_2t': 80, 'sigma_2c': 250,
            'tau_12': 100, 'tau_ilss': 90,
        }
        stress = np.array([-1500, 0, 0, 0, 0, 0], dtype=float)
        fi = self.solver._tsai_wu_failure(stress, strengths)
        assert fi > 0.9

    def test_get_failure_load_returns_dict(self):
        result = self.solver.get_failure_load(mode='compression')
        assert 'failure_stress' in result
        assert 'knockdown' in result

    def test_get_all_failure_loads(self):
        results = self.solver.get_all_failure_loads()
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            assert mode in results

    def test_degraded_strengths_keys(self):
        C_eff = self.solver._effective_stiffness(0.03, (1, 1, 1))
        strengths = self.solver._degraded_strengths(C_eff)
        for key in ['sigma_1t', 'sigma_1c', 'sigma_2t', 'sigma_2c', 'tau_12', 'tau_ilss']:
            assert key in strengths
            assert strengths[key] > 0

    def test_degraded_strengths_less_than_pristine(self):
        C_eff = self.solver._effective_stiffness(0.05, (1, 1, 1))
        strengths = self.solver._degraded_strengths(C_eff)
        assert strengths['sigma_1c'] < self.material.sigma_1c
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestMoriTanakaSolver -v`
Expected: FAIL

- [ ] **Step 3: Implement MoriTanakaSolver**

Add to `porosity_fe_analysis.py`:

```python
# ============================================================
# SECTION 6: MORI-TANAKA MICROMECHANICS SOLVER
# ============================================================

class MoriTanakaSolver:
    """Micromechanics solver using Mori-Tanaka homogenization for void inclusions."""

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties):
        self.mesh = mesh
        self.material = material
        self.nodal_knockdown = None

    def _eshelby_tensor(self, aspect_ratio: float, nu_m: float) -> np.ndarray:
        """Eshelby tensor S for ellipsoidal void in isotropic matrix.
        aspect_ratio: a1/a3 where a1 is the symmetry axis (x1 direction).
        - ar > 1: prolate (cylindrical), a1 > a2 = a3
        - ar < 1: oblate (penny), a1 < a2 = a3
        - ar = 1: sphere
        Ref: Mura (1987) Ch. 11; Nemat-Nasser & Hori (1993) Sec. 11.3."""
        S = np.zeros((6, 6))
        ar = aspect_ratio
        nu = nu_m

        if abs(ar - 1.0) < 0.01:  # Sphere
            S[0, 0] = S[1, 1] = S[2, 2] = (7 - 5 * nu) / (15 * (1 - nu))
            S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = \
                (5 * nu - 1) / (15 * (1 - nu))
            S[3, 3] = S[4, 4] = S[5, 5] = (4 - 5 * nu) / (15 * (1 - nu))

        elif ar > 1.0:  # Prolate spheroid (cylindrical void, a1 > a2 = a3)
            # g function for prolate: Mura Eq. 11.18
            g = ar / (ar**2 - 1)**1.5 * (ar * np.sqrt(ar**2 - 1) - np.arccosh(ar))
            a2 = ar**2

            S[0, 0] = (1.0 / (2 * (1 - nu))) * (
                1 - 2 * nu + (3 * a2 - 1) / (a2 - 1) - (1 - 2 * nu + 3 * a2 / (a2 - 1)) * g)

            S[1, 1] = S[2, 2] = (3.0 / (8 * (1 - nu))) * a2 / (a2 - 1) + \
                (1.0 / (4 * (1 - nu))) * (1 - 2 * nu - 9.0 / (4 * (a2 - 1))) * g

            S[0, 1] = S[0, 2] = -(1.0 / (2 * (1 - nu))) * a2 / (a2 - 1) + \
                (1.0 / (4 * (1 - nu))) * (3 * a2 / (a2 - 1) - (1 - 2 * nu)) * g

            S[1, 0] = S[2, 0] = -(1.0 / (2 * (1 - nu))) * 1.0 / (a2 - 1) + \
                (1.0 / (4 * (1 - nu))) * (3.0 / (a2 - 1) - (1 - 2 * nu)) * g

            S[1, 2] = S[2, 1] = (1.0 / (4 * (1 - nu))) * (
                a2 / (2 * (a2 - 1)) - (1 - 2 * nu + 3.0 / (4 * (a2 - 1))) * g)

            S[3, 3] = (1.0 / (4 * (1 - nu))) * (
                a2 / (2 * (a2 - 1)) + (1 - 2 * nu - 3.0 / (4 * (a2 - 1))) * g)

            S[4, 4] = S[5, 5] = (1.0 / (4 * (1 - nu))) * (
                1 - 2 * nu - (a2 + 1) / (a2 - 1) -
                0.5 * (1 - 2 * nu - 3 * (a2 + 1) / (a2 - 1)) * g)

        else:  # Oblate spheroid (penny void, a1 < a2 = a3)
            # g function for oblate: Mura Eq. 11.19
            # For oblate, ar < 1, use p = 1/ar > 1 as the "other" aspect ratio
            p = 1.0 / ar  # p > 1
            g_ob = p / (p**2 - 1)**1.5 * (np.arccos(1.0 / p) - (1.0 / p) * np.sqrt(1 - 1.0 / p**2))
            # g_ob is defined such that as p -> 1, g_ob -> 2/3 (sphere)
            a2 = ar**2  # < 1 for oblate

            S[0, 0] = (1.0 / (2 * (1 - nu))) * (
                1 - 2 * nu + (3 * a2 - 1) / (a2 - 1) - (1 - 2 * nu + 3 * a2 / (a2 - 1)) * g_ob)

            S[1, 1] = S[2, 2] = (3.0 / (8 * (1 - nu))) * a2 / (a2 - 1) + \
                (1.0 / (4 * (1 - nu))) * (1 - 2 * nu - 9.0 / (4 * (a2 - 1))) * g_ob

            S[0, 1] = S[0, 2] = -(1.0 / (2 * (1 - nu))) * a2 / (a2 - 1) + \
                (1.0 / (4 * (1 - nu))) * (3 * a2 / (a2 - 1) - (1 - 2 * nu)) * g_ob

            S[1, 0] = S[2, 0] = -(1.0 / (2 * (1 - nu))) * 1.0 / (a2 - 1) + \
                (1.0 / (4 * (1 - nu))) * (3.0 / (a2 - 1) - (1 - 2 * nu)) * g_ob

            S[1, 2] = S[2, 1] = (1.0 / (4 * (1 - nu))) * (
                a2 / (2 * (a2 - 1)) - (1 - 2 * nu + 3.0 / (4 * (a2 - 1))) * g_ob)

            S[3, 3] = (1.0 / (4 * (1 - nu))) * (
                a2 / (2 * (a2 - 1)) + (1 - 2 * nu - 3.0 / (4 * (a2 - 1))) * g_ob)

            S[4, 4] = S[5, 5] = (1.0 / (4 * (1 - nu))) * (
                1 - 2 * nu - (a2 + 1) / (a2 - 1) -
                0.5 * (1 - 2 * nu - 3 * (a2 + 1) / (a2 - 1)) * g_ob)

        return S

    def _effective_stiffness(self, Vp: float, void_shape: Tuple) -> np.ndarray:
        """Mori-Tanaka effective stiffness for void inclusions (C_i = 0)."""
        C_m = self.material.get_isotropic_matrix_stiffness()
        if Vp < 1e-12:
            return C_m.copy()

        ar = max(void_shape) / min(void_shape)
        S = self._eshelby_tensor(ar, self.material.matrix_poisson)
        I = np.eye(6)

        # C_eff = C_m @ {I - Vp @ inv[I - (1-Vp)*S]}
        inner = I - (1 - Vp) * S
        inner_inv = np.linalg.inv(inner)
        C_eff = C_m @ (I - Vp * inner_inv)

        return C_eff

    def _degraded_strengths(self, C_eff: np.ndarray) -> dict:
        """Map degraded stiffness to degraded strengths via sqrt stiffness ratio.
        Reference is C_pristine at Vp=0 (which equals C_m for the Mori-Tanaka
        void-in-matrix model). Pristine strengths are the composite-level values,
        so the ratio C_eff/C_pristine gives the fractional degradation from
        adding voids to the matrix phase."""
        C_pristine = self._effective_stiffness(0.0, self.mesh.porosity_field.void_shape_radii)
        ratio_11 = np.sqrt(max(C_eff[0, 0] / C_pristine[0, 0], 0))
        ratio_22 = np.sqrt(max(C_eff[1, 1] / C_pristine[1, 1], 0))
        ratio_shear = np.sqrt(max(C_eff[5, 5] / C_pristine[5, 5], 0))
        ratio_ilss = np.sqrt(max(C_eff[3, 3] / C_pristine[3, 3], 0))
        return {
            'sigma_1t': self.material.sigma_1t * ratio_11,
            'sigma_1c': self.material.sigma_1c * ratio_11,
            'sigma_2t': self.material.sigma_2t * ratio_22,
            'sigma_2c': self.material.sigma_2c * ratio_22,
            'tau_12': self.material.tau_12 * ratio_shear,
            'tau_ilss': self.material.tau_ilss * ratio_ilss,
        }

    def _tsai_wu_failure(self, stress_state: np.ndarray, strengths: dict) -> float:
        """Full 3D Tsai-Wu failure index.
        Voigt notation: s = [s1, s2, s3, s4(=tau_23), s5(=tau_13), s6(=tau_12)]
        Assumes transverse isotropy: direction 2 = direction 3.
        tau_23 is approximated as tau_ilss (standard for interlaminar shear)."""
        s = stress_state
        st = strengths
        # Linear terms
        F1 = 1.0 / st['sigma_1t'] - 1.0 / st['sigma_1c']
        F2 = 1.0 / st['sigma_2t'] - 1.0 / st['sigma_2c']
        F3 = F2  # Transverse isotropy
        # Quadratic diagonal terms
        F11 = 1.0 / (st['sigma_1t'] * st['sigma_1c'])
        F22 = 1.0 / (st['sigma_2t'] * st['sigma_2c'])
        F33 = F22  # Transverse isotropy
        # Shear terms: F44 for tau_23, F55 for tau_13, F66 for tau_12
        F44 = 1.0 / st['tau_ilss']**2   # tau_23 ~ tau_ilss
        F55 = 1.0 / st['tau_12']**2     # tau_13 ~ tau_12
        F66 = 1.0 / st['tau_12']**2
        # Interaction terms (stability approximation)
        F12 = -0.5 * np.sqrt(F11 * F22)
        F13 = F12  # Transverse isotropy
        F23 = -0.5 * np.sqrt(F22 * F33)  # = -0.5 * F22

        fi = (F1 * s[0] + F2 * s[1] + F3 * s[2] +
              F11 * s[0]**2 + F22 * s[1]**2 + F33 * s[2]**2 +
              F44 * s[3]**2 + F55 * s[4]**2 + F66 * s[5]**2 +
              2 * F12 * s[0] * s[1] + 2 * F13 * s[0] * s[2] +
              2 * F23 * s[1] * s[2])
        return float(fi)

    def get_failure_load(self, mode: str = 'compression') -> dict:
        """Predict failure from degraded effective properties."""
        Vp_avg = float(np.mean(self.mesh.porosity))
        C_eff = self._effective_stiffness(Vp_avg, self.mesh.porosity_field.void_shape_radii)
        strengths = self._degraded_strengths(C_eff)

        pristine_key = {'compression': 'sigma_1c', 'tension': 'sigma_1t',
                        'shear': 'tau_12', 'ilss': 'tau_ilss'}[mode]
        pristine = getattr(self.material, pristine_key)
        degraded = strengths[pristine_key]

        stress_dir = {
            'compression': np.array([-degraded, 0, 0, 0, 0, 0]),
            'tension': np.array([degraded, 0, 0, 0, 0, 0]),
            'shear': np.array([0, 0, 0, 0, 0, degraded]),
            'ilss': np.array([0, 0, 0, degraded, 0, 0]),
        }[mode]

        fi = self._tsai_wu_failure(stress_dir, strengths)
        knockdown = degraded / pristine

        # Store knockdown per node for visualization (cached by unique Vp)
        C_pristine = self._effective_stiffness(0.0, self.mesh.porosity_field.void_shape_radii)
        unique_Vp = np.unique(self.mesh.porosity)
        vp_to_ratio = {}
        for vp in unique_Vp:
            C_vp = self._effective_stiffness(float(vp), self.mesh.porosity_field.void_shape_radii)
            vp_to_ratio[float(vp)] = C_vp[0, 0] / C_pristine[0, 0]
        self.nodal_knockdown = np.array([vp_to_ratio[float(vp)] for vp in self.mesh.porosity])

        return {
            'failure_stress': degraded,
            'knockdown': knockdown,
            'tsai_wu_index': fi,
            'effective_stiffness_ratio': float(C_eff[0, 0] / self.material.get_isotropic_matrix_stiffness()[0, 0]),
        }

    def get_all_failure_loads(self) -> dict:
        results = {}
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            results[mode] = self.get_failure_load(mode)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestMoriTanakaSolver -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add MoriTanakaSolver with Eshelby tensor and 3D Tsai-Wu criterion"
```

---

### Task 8: FEVisualizer

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 7)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for FEVisualizer**

```python
class TestFEVisualizer:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = EmpiricalSolver(self.mesh, self.material)
        self.solver.apply_loading('compression', 'judd_wright')
        self.pf = pf

    def test_plot_porosity_field_returns_fig(self):
        fig = FEVisualizer.plot_porosity_field(self.pf)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_3d_returns_fig(self):
        fig = FEVisualizer.plot_mesh_3d(self.mesh)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_detail_returns_fig(self):
        fig = FEVisualizer.plot_mesh_detail(self.mesh)
        assert fig is not None
        plt.close(fig)

    def test_plot_damage_contour_returns_fig(self):
        fig = FEVisualizer.plot_damage_contour(self.mesh, self.solver)
        assert fig is not None
        plt.close(fig)

    def test_plot_porosity_field_saves(self, tmp_path):
        path = str(tmp_path / "test_profile.png")
        FEVisualizer.plot_porosity_field(self.pf, save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_void_scf_returns_fig(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        fig = FEVisualizer.plot_void_scf(void)
        assert fig is not None
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestFEVisualizer -v`
Expected: FAIL

- [ ] **Step 3: Implement FEVisualizer**

Add to `porosity_fe_analysis.py`. This is a large section — implement all 7 static methods following the wrinkle tool's visualization patterns:

```python
# ============================================================
# SECTION 7: VISUALIZATION
# ============================================================

class FEVisualizer:
    """Publication-quality plotting for porosity analysis."""

    @staticmethod
    def plot_porosity_field(porosity_field: PorosityField, save_path: str = None):
        """2-panel: through-thickness porosity profile + plan view."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: through-thickness profile
        z, Vp = porosity_field.effective_porosity_profile(nz=200)
        axes[0].plot(Vp * 100, z, 'b-', linewidth=2)
        axes[0].set_xlabel('Porosity (%)', fontsize=12)
        axes[0].set_ylabel('z (mm)', fontsize=12)
        axes[0].set_title('Through-Thickness Porosity Profile', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(left=0)

        # Right: plan view at midplane
        Lz = porosity_field.Lz
        x = np.linspace(0, 50, 100)
        y = np.linspace(0, 20, 50)
        X, Y = np.meshgrid(x, y)
        Z_mid = np.full_like(X, Lz / 2)
        Vp_map = porosity_field.local_porosity(X.ravel(), Y.ravel(), Z_mid.ravel())
        Vp_map = Vp_map.reshape(X.shape)
        im = axes[1].contourf(X, Y, Vp_map * 100, levels=20, cmap='YlOrRd')
        plt.colorbar(im, ax=axes[1], label='Porosity (%)')
        axes[1].set_xlabel('x (mm)', fontsize=12)
        axes[1].set_ylabel('y (mm)', fontsize=12)
        axes[1].set_title('Porosity at Midplane', fontsize=14, fontweight='bold')

        # Mark discrete voids
        for center, radii in porosity_field.get_void_locations():
            circle = plt.Circle((center[0], center[1]), radii[0],
                               fill=False, color='red', linewidth=2)
            axes[1].add_patch(circle)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    @staticmethod
    def plot_mesh_3d(mesh: CompositeMesh, save_path: str = None):
        """3D hex mesh wireframe with void elements highlighted."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot top and bottom surface grids
        nx, ny = mesh.nx, mesh.ny
        n_per_layer = (nx + 1) * (ny + 1)

        for layer_idx in [0, mesh.nz]:
            start = layer_idx * n_per_layer
            end = start + n_per_layer
            layer_nodes = mesh.nodes[start:end]
            X = layer_nodes[:, 0].reshape(ny + 1, nx + 1)
            Y = layer_nodes[:, 1].reshape(ny + 1, nx + 1)
            Z = layer_nodes[:, 2].reshape(ny + 1, nx + 1)
            ax.plot_wireframe(X, Y, Z, alpha=0.3, color='gray', linewidth=0.5)

        # Highlight void elements
        if len(mesh.void_elements) > 0:
            for eidx in mesh.void_elements[:20]:  # Limit for performance
                elem_nodes = mesh.nodes[mesh.elements[eidx]]
                center = elem_nodes.mean(axis=0)
                ax.scatter(*center, color='red', s=20, alpha=0.8)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.set_title('3D Mesh with Porosity', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    @staticmethod
    def plot_mesh_detail(mesh: CompositeMesh, save_path: str = None):
        """Cross-section with porosity contour + single hex element."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: cross-section at mid-y
        ny_mid = mesh.ny // 2
        nx1 = mesh.nx + 1
        ny1 = mesh.ny + 1
        indices = []
        for k in range(mesh.nz + 1):
            for i in range(mesh.nx + 1):
                idx = k * ny1 * nx1 + ny_mid * nx1 + i
                indices.append(idx)
        indices = np.array(indices)
        X = mesh.nodes[indices, 0].reshape(mesh.nz + 1, mesh.nx + 1)
        Z = mesh.nodes[indices, 2].reshape(mesh.nz + 1, mesh.nx + 1)
        P = mesh.porosity[indices].reshape(mesh.nz + 1, mesh.nx + 1)

        im = axes[0].contourf(X, Z, P * 100, levels=20, cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0], label='Porosity (%)')
        axes[0].set_xlabel('x (mm)', fontsize=12)
        axes[0].set_ylabel('z (mm)', fontsize=12)
        axes[0].set_title('Cross-Section Porosity', fontsize=14, fontweight='bold')
        axes[0].set_aspect('equal')

        # Right: single hex element diagram
        ax = axes[1]
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            pts = corners[list(e)]
            ax.plot(pts[:, 0] + pts[:, 1]*0.3, pts[:, 2] + pts[:, 1]*0.3,
                   'b-', linewidth=1.5)
        for idx, c in enumerate(corners):
            ax.plot(c[0] + c[1]*0.3, c[2] + c[1]*0.3, 'ko', markersize=6)
            ax.annotate(str(idx), (c[0] + c[1]*0.3 + 0.05, c[2] + c[1]*0.3 + 0.05),
                       fontsize=10, fontweight='bold')
        ax.set_title('8-Node Hexahedral Element', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    @staticmethod
    def plot_damage_contour(mesh: CompositeMesh, solver, save_path: str = None):
        """2D stiffness reduction map at midplane."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Get midplane slice
        nz_mid = mesh.nz // 2
        nx1 = mesh.nx + 1
        ny1 = mesh.ny + 1
        start = nz_mid * ny1 * nx1
        end = start + ny1 * nx1
        X = mesh.nodes[start:end, 0].reshape(ny1, nx1)
        Y = mesh.nodes[start:end, 1].reshape(ny1, nx1)

        if solver.nodal_knockdown is not None:
            kd = solver.nodal_knockdown[start:end].reshape(ny1, nx1)
        else:
            kd = mesh.stiffness_reduction[start:end].reshape(ny1, nx1)

        im = ax.contourf(X, Y, kd, levels=20, cmap='RdYlGn')
        plt.colorbar(im, ax=ax, label='Stiffness Retention')
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title('Stiffness Reduction at Midplane', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    @staticmethod
    def plot_void_scf(void_geometry: VoidGeometry, save_path: str = None):
        """Stress concentration field around a single void."""
        fig, ax = plt.subplots(figsize=(8, 8))

        r_max = 3 * max(void_geometry.radii)
        x = np.linspace(-r_max, r_max, 200)
        y = np.linspace(-r_max, r_max, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        dist = void_geometry.distance_field(X.ravel(), Y.ravel(), Z.ravel())
        dist = dist.reshape(X.shape)

        scf = void_geometry.stress_concentration_factor()
        scf_max = scf['compression']
        field = np.where(dist < 0, 0, 1.0 + (scf_max - 1) * np.exp(-dist / max(void_geometry.radii)))

        im = ax.contourf(X, Y, field, levels=30, cmap='hot_r')
        plt.colorbar(im, ax=ax, label='Stress Concentration Factor')
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title(f'SCF Field (aspect ratio={void_geometry.aspect_ratio:.1f})',
                     fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    @staticmethod
    def plot_knockdown_curves(results_by_porosity: dict, save_path: str = None):
        """Strength vs porosity % for all loading modes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        modes = ['compression', 'tension', 'shear', 'ilss']
        colors = {'judd_wright': 'blue', 'power_law': 'red', 'linear': 'green',
                  'mori_tanaka': 'black'}

        for idx, mode in enumerate(modes):
            ax = axes[idx]
            Vp_vals = sorted([float(k.replace('pct', '')) for k in results_by_porosity.keys()])

            for config_name in list(list(results_by_porosity.values())[0].keys()):
                # Empirical models
                for model in ['judd_wright', 'power_law', 'linear']:
                    kd_vals = []
                    for Vp_label in sorted(results_by_porosity.keys()):
                        r = results_by_porosity[Vp_label][config_name]['empirical']
                        kd_vals.append(r[mode][model]['knockdown'])
                    ax.plot(Vp_vals, kd_vals, color=colors[model],
                           linestyle='-' if 'uniform' in config_name else '--',
                           alpha=0.7, linewidth=1.5)

            ax.set_xlabel('Porosity (%)', fontsize=11)
            ax.set_ylabel('Knockdown Factor', fontsize=11)
            ax.set_title(mode.upper(), fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

        plt.suptitle('Porosity Knockdown Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    @staticmethod
    def plot_model_comparison(results: dict, save_path: str = None):
        """Empirical vs Mori-Tanaka comparison bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        configs = list(results.keys())
        x = np.arange(len(configs))
        width = 0.15

        # Left: compression knockdown
        for i, model in enumerate(['judd_wright', 'power_law', 'linear']):
            vals = [results[c]['empirical']['compression'][model]['knockdown'] for c in configs]
            axes[0].bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        mt_vals = [results[c]['mori_tanaka']['compression']['knockdown'] for c in configs]
        axes[0].bar(x + 3 * width, mt_vals, width, label='Mori-Tanaka', color='black')
        axes[0].set_xticks(x + 1.5 * width)
        axes[0].set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=8)
        axes[0].set_ylabel('Knockdown Factor')
        axes[0].set_title('Compression', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right: ILSS knockdown
        for i, model in enumerate(['judd_wright', 'power_law', 'linear']):
            vals = [results[c]['empirical']['ilss'][model]['knockdown'] for c in configs]
            axes[1].bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        mt_vals = [results[c]['mori_tanaka']['ilss']['knockdown'] for c in configs]
        axes[1].bar(x + 3 * width, mt_vals, width, label='Mori-Tanaka', color='black')
        axes[1].set_xticks(x + 1.5 * width)
        axes[1].set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=8)
        axes[1].set_ylabel('Knockdown Factor')
        axes[1].set_title('ILSS', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestFEVisualizer -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add FEVisualizer with 7 publication-quality plot types"
```

---

### Task 9: Analysis Pipeline (compare_configurations, main, save_results_to_json)

**Files:**
- Modify: `porosity_fe_analysis.py` (add Section 8)
- Modify: `tests/test_porosity_fe.py`

- [ ] **Step 1: Write failing tests for analysis pipeline**

```python
from porosity_fe_analysis import compare_configurations, save_results_to_json


class TestAnalysisPipeline:
    def test_compare_configurations_returns_all_configs(self):
        results = compare_configurations(0.03, material_name='T800_epoxy',
                                          configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        assert 'uniform_spherical' in results

    def test_compare_configurations_has_both_solvers(self):
        results = compare_configurations(0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        r = results['uniform_spherical']
        assert 'empirical' in r
        assert 'mori_tanaka' in r
        assert 'mesh' in r
        assert 'empirical_solver' in r
        assert 'mori_tanaka_solver' in r

    def test_compare_configurations_empirical_has_all_modes(self):
        results = compare_configurations(0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        emp = results['uniform_spherical']['empirical']
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            assert mode in emp

    def test_save_results_to_json(self, tmp_path):
        results = compare_configurations(0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "test_results.json")
        save_results_to_json(results, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert 'uniform_spherical' in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_porosity_fe.py::TestAnalysisPipeline -v`
Expected: FAIL

- [ ] **Step 3: Implement analysis pipeline**

Add to `porosity_fe_analysis.py`:

```python
# ============================================================
# SECTION 8: ANALYSIS PIPELINE
# ============================================================

def compare_configurations(void_volume_fraction: float,
                           material_name: str = 'T800_epoxy',
                           applied_stress: float = -1500.0,
                           configs: Optional[Dict] = None) -> Dict:
    """Main analysis function — loops through porosity configurations."""
    material = MATERIALS[material_name]
    configs = configs or POROSITY_CONFIGS
    results = {}

    print(f"\n{'='*70}")
    print(f"POROSITY ANALYSIS: Vp = {void_volume_fraction*100:.1f}%")
    print(f"Material: {material_name}")
    print(f"{'='*70}")

    for name, config in configs.items():
        print(f"\n  Configuration: {name}")
        porosity_field = PorosityField(material, void_volume_fraction, **config)
        mesh = CompositeMesh(porosity_field, material, nx=30, ny=10, nz=12)

        empirical = EmpiricalSolver(mesh, material)
        mori_tanaka = MoriTanakaSolver(mesh, material)

        emp_results = empirical.get_all_failure_loads()
        mt_results = mori_tanaka.get_all_failure_loads()

        results[name] = {
            'config': config,
            'mesh': mesh,
            'porosity_field': porosity_field,
            'empirical_solver': empirical,
            'mori_tanaka_solver': mori_tanaka,
            'empirical': emp_results,
            'mori_tanaka': mt_results,
        }

        # Print summary
        comp_kd = emp_results['compression']['judd_wright']['knockdown']
        ilss_kd = emp_results['ilss']['judd_wright']['knockdown']
        print(f"    Compression KD (J-W): {comp_kd:.3f}")
        print(f"    ILSS KD (J-W):        {ilss_kd:.3f}")

    # Rankings
    print(f"\n{'='*70}")
    print("RANKINGS (by compression strength, Judd-Wright)")
    print(f"{'='*70}")
    ranked = sorted(results.keys(),
                   key=lambda c: results[c]['empirical']['compression']['judd_wright']['failure_stress'],
                   reverse=True)
    for i, name in enumerate(ranked, 1):
        fs = results[name]['empirical']['compression']['judd_wright']['failure_stress']
        print(f"  {i}. {name}: {fs:.1f} MPa")

    return results


def save_results_to_json(results: Dict, filename: str):
    """Export numerical results to JSON."""
    output = {}
    for name, data in results.items():
        entry = {
            'config': data['config'],
            'void_volume_fraction': float(data['porosity_field'].Vp),
            'empirical': {},
            'mori_tanaka': {},
        }
        for mode in data['empirical']:
            entry['empirical'][mode] = {}
            for model in data['empirical'][mode]:
                r = data['empirical'][mode][model]
                entry['empirical'][mode][model] = {
                    'failure_stress_MPa': r['failure_stress'],
                    'knockdown': r['knockdown'],
                }
        for mode in data['mori_tanaka']:
            r = data['mori_tanaka'][mode]
            entry['mori_tanaka'][mode] = {
                'failure_stress_MPa': r['failure_stress'],
                'knockdown': r['knockdown'],
            }
        output[name] = entry

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {filename}")


def main():
    """Entry point — loops over porosity severity levels."""
    porosity_levels = [0.01, 0.02, 0.03, 0.05, 0.08]

    all_results = {}
    for Vp in porosity_levels:
        Vp_label = f"{int(Vp*100)}pct"
        results = compare_configurations(Vp)
        all_results[Vp_label] = results

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

        FEVisualizer.plot_model_comparison(
            results,
            save_path=f"porosity_comparison_{Vp_label}.png")

        save_results_to_json(results, f"porosity_analysis_results_{Vp_label}.json")

    FEVisualizer.plot_knockdown_curves(
        all_results,
        save_path="porosity_knockdown_curves.png")

    print(f"\n{'='*70}")
    print("COMPLETE ANALYSIS FINISHED")
    print(f"{'='*70}")
    print(f"Porosity levels analyzed: {[f'{v*100:.0f}%' for v in porosity_levels]}")
    print(f"Configurations: {list(POROSITY_CONFIGS.keys())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_porosity_fe.py::TestAnalysisPipeline -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add porosity_fe_analysis.py tests/test_porosity_fe.py
git commit -m "feat: add analysis pipeline with compare_configurations and JSON export"
```

---

### Task 10: Full Integration Test + README

**Files:**
- Modify: `tests/test_porosity_fe.py`
- Create: `README.md`

- [ ] **Step 1: Write integration test**

```python
class TestIntegration:
    """End-to-end test with reduced parameters for speed."""

    def test_full_pipeline_single_config(self, tmp_path):
        """Run complete pipeline with one config and one porosity level."""
        os.chdir(str(tmp_path))
        results = compare_configurations(
            0.03, material_name='T800_epoxy',
            configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})

        # Check results structure
        assert 'uniform_spherical' in results
        r = results['uniform_spherical']

        # Empirical results exist and are reasonable
        emp_comp = r['empirical']['compression']['judd_wright']
        assert 0 < emp_comp['knockdown'] < 1.0
        assert emp_comp['failure_stress'] < MATERIALS['T800_epoxy'].sigma_1c

        # Mori-Tanaka results exist
        mt_comp = r['mori_tanaka']['compression']
        assert 0 < mt_comp['knockdown'] < 1.0

        # ILSS is more sensitive than compression
        emp_ilss = r['empirical']['ilss']['judd_wright']['knockdown']
        emp_comp_kd = r['empirical']['compression']['judd_wright']['knockdown']
        assert emp_ilss < emp_comp_kd

        # JSON export works
        save_results_to_json(results, "test_output.json")
        assert os.path.exists("test_output.json")

        # Plots generate without error
        FEVisualizer.plot_porosity_field(r['porosity_field'],
                                         save_path="test_profile.png")
        assert os.path.exists("test_profile.png")
        plt.close('all')

    def test_all_materials(self):
        """Verify all 3 material presets work."""
        for mat_name in ['T800_epoxy', 'T700_epoxy', 'glass_epoxy']:
            results = compare_configurations(
                0.02, material_name=mat_name,
                configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
            assert 'uniform_spherical' in results
            kd = results['uniform_spherical']['empirical']['compression']['judd_wright']['knockdown']
            assert 0 < kd < 1.0
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_porosity_fe.py::TestIntegration -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Write README.md**

Create `README.md` with: overview, installation, quick start, output files, configuration guide, and references. Follow the pattern of `jin_fe_analysis/README.md`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/test_porosity_fe.py -v`
Expected: All tests PASS (approximately 60+ tests total)

- [ ] **Step 5: Run the tool end-to-end with default parameters**

Run: `cd /Users/elhajjar/Library/CloudStorage/OneDrive-UWM/AI/Porosity_FE && python porosity_fe_analysis.py`
Expected: Console output showing analysis progress, PNG files and JSON files generated

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_porosity_fe.py
git commit -m "feat: add integration tests and README documentation"
```

---

### Task 11: CLAUDE.md Project Documentation

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

Create `CLAUDE.md` following the structure and level of detail of the Double_Wrinkle CLAUDE.md. Include:
- Repository overview
- File structure
- Running the model (prerequisites, execution, output)
- Code architecture (all 8 sections with key functions documented)
- Key mathematical relationships (empirical models, Mori-Tanaka, Eshelby, Tsai-Wu)
- Typical parameter ranges for porosity
- Modifying the models (how to change material, distribution, mesh, etc.)
- Expected results per configuration
- Scientific references

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add comprehensive CLAUDE.md project documentation"
```
