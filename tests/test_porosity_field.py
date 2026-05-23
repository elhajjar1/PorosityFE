#!/usr/bin/env python3
"""Tests for porosity_fe.porosity_field.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (MATERIALS, VoidGeometry, PorosityField, POROSITY_CONFIGS)


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

    def test_negative_Vp_raises(self):
        with pytest.raises(ValueError, match=r"finite fraction in \[0, 1\]"):
            PorosityField(self.material, -0.01, distribution='uniform')

    def test_Vp_above_one_raises_with_percent_hint(self):
        with pytest.raises(ValueError, match=r"Did you pass a percent\?"):
            PorosityField(self.material, 3.0, distribution='uniform')

    def test_nan_Vp_raises(self):
        with pytest.raises(ValueError, match=r"finite fraction"):
            PorosityField(self.material, float('nan'), distribution='uniform')

    def test_inf_Vp_raises(self):
        with pytest.raises(ValueError, match=r"finite fraction"):
            PorosityField(self.material, float('inf'), distribution='uniform')

    def test_Vp_boundary_zero_and_one_accepted(self):
        # Both boundaries should be accepted (no exception)
        PorosityField(self.material, 0.0, distribution='uniform')
        PorosityField(self.material, 1.0, distribution='uniform')

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

    def test_unknown_void_shape_string_raises(self):
        with pytest.raises(ValueError, match=r"Unknown void_shape"):
            PorosityField(self.material, 0.03, void_shape='spheroidal')

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match=r"Unknown distribution"):
            PorosityField(self.material, 0.03, distribution='gradient')

    def test_unknown_cluster_location_raises(self):
        with pytest.raises(ValueError, match=r"Unknown cluster_location"):
            PorosityField(self.material, 0.03,
                          distribution='clustered', cluster_location='midplne')

    def test_quarter_cluster_location_supported(self):
        # 'quarter' is one of the documented cluster locations and should round-trip.
        pf = PorosityField(self.material, 0.03,
                           distribution='clustered', cluster_location='quarter')
        assert pf.cluster_location == 'quarter'

    def test_Vp_snap_to_one_from_fp_noise(self):
        # numerical noise just above 1.0 should snap to 1.0 instead of raising
        pf = PorosityField(self.material, 1.0 + 5e-10, distribution='uniform')
        assert pf.Vp == 1.0

    def test_Vp_just_above_one_no_percent_hint(self):
        # Values barely above the boundary are likely numerical noise, not
        # percent confusion — the percent hint should be suppressed.
        with pytest.raises(ValueError) as exc:
            PorosityField(self.material, 1.0001, distribution='uniform')
        assert "Did you pass a percent?" not in str(exc.value)

    def test_Vp_string_rejected_with_typeerror(self):
        with pytest.raises(TypeError, match=r"numeric type"):
            PorosityField(self.material, "0.5", distribution='uniform')

    def test_Vp_none_rejected(self):
        with pytest.raises(ValueError, match=r"None"):
            PorosityField(self.material, None, distribution='uniform')
