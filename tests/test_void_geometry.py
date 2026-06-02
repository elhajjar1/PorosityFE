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

    def test_scf_sphere_exact_values(self):
        """The near-spherical branch (ar < 1.2) returns fixed SCFs."""
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        assert void.stress_concentration_factor() == {
            'compression': 2.0, 'tension': 2.0, 'shear': 1.5,
            'ilss': 1.8, 'transverse_tension': 2.0,
        }

    def test_scf_cylindrical_prolate_values(self):
        """Prolate ('cylindrical') void: radii[0] > radii[2] and
        radii[1] < radii[0]/2. Pins the aspect-ratio scaling and the
        ``VOID_SHAPES['cylindrical']`` preset -> regime mapping."""
        void = VoidGeometry(center=(0, 0, 0), radii=VOID_SHAPES['cylindrical'])
        ar = void.aspect_ratio  # (3, 1, 1) -> 3.0
        assert ar == 3.0
        scf = void.stress_concentration_factor()
        assert scf == {
            'compression': 1.5 + 0.5 * ar,        # 3.0
            'tension': 1.5 + 0.5 * ar,            # 3.0
            'shear': 1.3 + 0.3 * ar,              # 2.2
            'ilss': 1.5 + 0.4 * ar,               # 2.7
            'transverse_tension': 1.5 + 0.5 * ar,  # 3.0
        }

    def test_scf_penny_oblate_values(self):
        """Oblate ('penny') void: radii[0] > radii[2] and
        radii[1] >= radii[0]/2. Pennies carry the highest SCFs, so a
        regression in this branch would understate their severity."""
        void = VoidGeometry(center=(0, 0, 0), radii=VOID_SHAPES['penny'])
        ar = void.aspect_ratio  # (3, 3, 0.3) -> 10.0
        assert ar == 10.0
        scf = void.stress_concentration_factor()
        assert scf == {
            'compression': 2.0 + 1.0 * ar,        # 12.0
            'tension': 2.0 + 1.5 * ar,            # 17.0
            'shear': 1.5 + 0.8 * ar,              # 9.5
            'ilss': 2.0 + 1.2 * ar,               # 14.0
            'transverse_tension': 2.0 + 1.5 * ar,  # 17.0
        }
        # Penny tension SCF must exceed an equally-elongated prolate void's.
        prolate = VoidGeometry(center=(0, 0, 0), radii=(10, 1, 1))
        assert scf['tension'] > prolate.stress_concentration_factor()['tension']

    def test_scf_through_thickness_falls_back_to_sphere(self):
        """A void elongated through-thickness (radii[0] <= radii[2]) is not
        a recognised in-plane regime, so the SCF defaults to the spherical
        values rather than the prolate/oblate scaling."""
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 3))
        assert void.aspect_ratio == 3.0  # >= 1.2, so not the spherical branch
        assert void.stress_concentration_factor() == {
            'compression': 2.0, 'tension': 2.0, 'shear': 1.5,
            'ilss': 1.8, 'transverse_tension': 2.0,
        }

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
