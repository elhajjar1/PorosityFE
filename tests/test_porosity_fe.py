#!/usr/bin/env python3
"""Tests for porosity_fe_analysis.py"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from porosity_fe_analysis import (MaterialProperties, MATERIALS, VoidGeometry, VOID_SHAPES,
                                   PorosityField, POROSITY_CONFIGS, CompositeMesh,
                                   EmpiricalSolver, MoriTanakaSolver, FEVisualizer,
                                   compare_configurations, save_results_to_json)


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
        # Uniform 3% -> all nodes should be ~0.03
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
        """Higher porosity -> lower knockdown"""
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


class TestIntegration:
    """End-to-end test with reduced parameters for speed."""

    def test_full_pipeline_single_config(self, tmp_path):
        os.chdir(str(tmp_path))
        results = compare_configurations(
            0.03, material_name='T800_epoxy',
            configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})

        r = results['uniform_spherical']
        emp_comp = r['empirical']['compression']['judd_wright']
        assert 0 < emp_comp['knockdown'] < 1.0
        assert emp_comp['failure_stress'] < MATERIALS['T800_epoxy'].sigma_1c

        mt_comp = r['mori_tanaka']['compression']
        assert 0 < mt_comp['knockdown'] < 1.0

        emp_ilss = r['empirical']['ilss']['judd_wright']['knockdown']
        emp_comp_kd = r['empirical']['compression']['judd_wright']['knockdown']
        assert emp_ilss < emp_comp_kd

        save_results_to_json(results, "test_output.json")
        assert os.path.exists("test_output.json")

        FEVisualizer.plot_porosity_field(r['porosity_field'],
                                         save_path="test_profile.png")
        assert os.path.exists("test_profile.png")
        plt.close('all')

    def test_all_materials(self):
        for mat_name in ['T800_epoxy', 'T700_epoxy', 'glass_epoxy']:
            results = compare_configurations(
                0.02, material_name=mat_name,
                configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
            assert 'uniform_spherical' in results
            kd = results['uniform_spherical']['empirical']['compression']['judd_wright']['knockdown']
            assert 0 < kd < 1.0
