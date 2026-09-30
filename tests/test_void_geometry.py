#!/usr/bin/env python3
"""Tests for porosity_fe.void_geometry.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (VoidGeometry, VOID_SHAPES)


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

    @pytest.mark.parametrize("nu", [0.2, 0.35, 0.45])
    def test_scf_sphere_matches_goodier(self, nu):
        """Spherical cavity: Goodier (1933) closed forms."""
        scf = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1)
                           ).stress_concentration_factor(nu)
        tension = (27 - 15 * nu) / (2 * (7 - 5 * nu))
        shear = 15 * (1 - nu) / (7 - 5 * nu)
        for mode in ('tension', 'compression', 'transverse_tension'):
            assert scf[mode] == pytest.approx(tension, rel=1e-8)
        for mode in ('shear', 'ilss'):
            assert scf[mode] == pytest.approx(shear, rel=1e-8)

    def test_scf_long_circular_cylinder_matches_kirsch(self):
        """Cavity elongated along z, loaded along x: Kirsch's 3."""
        scf = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 200)
                           ).stress_concentration_factor()
        assert scf['tension'] == pytest.approx(3.0, rel=1e-3)

    def test_scf_long_elliptic_cylinder_matches_inglis(self):
        """Elliptic cross-section a=1 (x), b=0.25 (y): 1 + 2 a / b."""
        scf = VoidGeometry(center=(0, 0, 0), radii=(1, 0.25, 200)
                           ).stress_concentration_factor()
        assert scf['transverse_tension'] == pytest.approx(1 + 2 * 1 / 0.25, rel=1e-3)
        assert scf['tension'] == pytest.approx(1 + 2 * 0.25 / 1, rel=1e-3)

    def test_scf_penny_depends_on_load_direction(self):
        """A penny void barely concentrates in-plane loads but strongly
        concentrates interlaminar shear (the old heuristic gave 17 for
        in-plane tension)."""
        scf = VoidGeometry(center=(0, 0, 0), radii=VOID_SHAPES['penny']
                           ).stress_concentration_factor()
        assert 1.0 < scf['tension'] < 1.3
        assert scf['ilss'] > 5.0
        thinner = VoidGeometry(center=(0, 0, 0), radii=(3, 3, 0.03)
                               ).stress_concentration_factor()
        assert thinner['ilss'] > 5 * scf['ilss']

    def test_scf_tension_equals_compression(self):
        scf = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 0.5)
                           ).stress_concentration_factor()
        assert scf['tension'] == pytest.approx(scf['compression'], rel=1e-12)

    def test_scf_honors_orientation(self):
        """Rotating a void 90 deg about z swaps the x and y loadings."""
        import math
        base = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1)
                            ).stress_concentration_factor()
        rotated = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1),
                               orientation=math.pi / 2).stress_concentration_factor()
        swapped = VoidGeometry(center=(0, 0, 0), radii=(1, 3, 1)
                               ).stress_concentration_factor()
        assert rotated['tension'] == pytest.approx(base['transverse_tension'], rel=1e-6)
        assert rotated['transverse_tension'] == pytest.approx(base['tension'], rel=1e-6)
        for mode in rotated:
            assert rotated[mode] == pytest.approx(swapped[mode], rel=1e-6)

    def test_scf_continuous_across_former_shape_thresholds(self):
        """The old rules jumped at aspect ratio 1.2 and radii[1] = radii[0]/2."""
        near = [VoidGeometry(center=(0, 0, 0), radii=r).stress_concentration_factor()
                for r in [(1.19, 1, 1), (1.21, 1, 1), (3, 1.49, 1), (3, 1.51, 1)]]
        for lo, hi in ((near[0], near[1]), (near[2], near[3])):
            for mode in lo:
                assert lo[mode] == pytest.approx(hi[mode], rel=0.02)

    def test_scf_rejects_bad_poisson_ratio(self):
        with pytest.raises(ValueError, match="nu_m"):
            VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1)
                         ).stress_concentration_factor(0.5)

    def test_distance_field_inside_negative(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        d = void.distance_field(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert d[0] < 0  # Inside -> negative

    def test_distance_field_outside_positive(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        d = void.distance_field(np.array([2.0]), np.array([0.0]), np.array([0.0]))
        assert d[0] > 0  # Outside -> positive

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

    def test_zero_radius_rejected(self):
        with pytest.raises(ValueError, match=r"radii.*positive"):
            VoidGeometry(center=(0, 0, 0), radii=(0.0, 1.0, 1.0))

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError, match=r"radii.*positive"):
            VoidGeometry(center=(0, 0, 0), radii=(1.0, -1.0, 1.0))

    def test_wrong_radii_shape_rejected(self):
        with pytest.raises(ValueError, match=r"radii must have 3 components"):
            VoidGeometry(center=(0, 0, 0), radii=(1.0, 1.0))

    def test_non_finite_orientation_rejected(self):
        with pytest.raises(ValueError, match=r"orientation"):
            VoidGeometry(center=(0, 0, 0), radii=(1.0, 1.0, 1.0),
                         orientation=float('nan'))

    def test_wrong_center_shape_rejected(self):
        with pytest.raises(ValueError, match=r"center must have 3 components"):
            VoidGeometry(center=(0, 0), radii=(1.0, 1.0, 1.0))

    def test_non_finite_center_rejected(self):
        with pytest.raises(ValueError, match=r"center must be finite"):
            VoidGeometry(center=(0.0, float('inf'), 0.0), radii=(1.0, 1.0, 1.0))
