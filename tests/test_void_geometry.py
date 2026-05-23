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
