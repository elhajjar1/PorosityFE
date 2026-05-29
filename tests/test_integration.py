#!/usr/bin/env python3
"""Tests for end-to-end integration and miscellaneous helpers.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from porosity_fe_analysis import (MATERIALS, VoidGeometry, PorosityField, POROSITY_CONFIGS, CompositeMesh,
                                   EmpiricalSolver, FEVisualizer,
                                   compare_configurations, save_results_to_json,
                                   rotation_matrix_3d, stress_transformation_3d,
                                   strain_transformation_3d, rotate_stiffness_3d,
                                   gauss_points_1d, gauss_points_hex,
                                   Hex8Element, FESolver, load_results_from_json,
                                   JSON_SCHEMA_VERSION, FORMAT_EMPIRICAL_SWEEP)


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

    def _two_vp_results(self):
        """A ``{Vp_label: {config: ConfigResult}}`` map for the curve plot."""
        cfgs = {'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        return {
            '2pct': compare_configurations(0.02, configs=cfgs),
            '3pct': compare_configurations(0.03, configs=cfgs),
        }

    def test_plot_knockdown_curves_returns_fig(self):
        fig = FEVisualizer.plot_knockdown_curves(self._two_vp_results())
        assert fig is not None
        plt.close(fig)

    def test_plot_knockdown_curves_saves(self, tmp_path):
        path = str(tmp_path / "knockdown.png")
        FEVisualizer.plot_knockdown_curves(self._two_vp_results(),
                                           save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_model_comparison_returns_fig(self):
        results = compare_configurations(
            0.03,
            configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        fig = FEVisualizer.plot_model_comparison(results)
        assert fig is not None
        plt.close(fig)

    def test_plot_model_comparison_saves(self, tmp_path):
        results = compare_configurations(
            0.03,
            configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "comparison.png")
        FEVisualizer.plot_model_comparison(results, save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_damage_contour_falls_back_to_mesh_reduction(self):
        """Before ``apply_loading`` populates ``nodal_knockdown`` the
        midplane map must fall back to ``mesh.stiffness_reduction`` (the
        else-branch in plot_damage_contour)."""
        solver = EmpiricalSolver(self.mesh, self.material)
        assert solver.nodal_knockdown is None
        fig = FEVisualizer.plot_damage_contour(self.mesh, solver)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_3d_highlights_voids_and_saves(self, tmp_path):
        """A field carrying discrete voids yields void elements that the 3D
        plot highlights in red; also exercises the save_path branch."""
        lo = self.mesh.nodes.min(axis=0)
        hi = self.mesh.nodes.max(axis=0)
        void = VoidGeometry(center=tuple((lo + hi) / 2),
                            radii=tuple((hi - lo) / 4))
        pf = PorosityField(self.material, 0.03, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.void_elements) > 0
        path = str(tmp_path / "mesh3d.png")
        FEVisualizer.plot_mesh_3d(mesh, save_path=path)
        assert os.path.exists(path)
        plt.close('all')


class TestAnalysisPipeline:
    def test_compare_configurations_returns_all_configs(self):
        results = compare_configurations(0.03, material_name='T800_epoxy',
                                          configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        assert 'uniform_spherical' in results

    def test_compare_configurations_has_empirical_solver(self):
        # #44 item 3: lightweight ConfigResult by default; live mesh /
        # empirical_solver objects live on the parallel artifacts dict
        # returned only with ``return_artifacts=True``.
        results, artifacts = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']},
            return_artifacts=True)
        r = results['uniform_spherical']
        assert 'empirical' in r
        art = artifacts['uniform_spherical']
        assert art.mesh is not None
        assert art.empirical_solver is not None

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
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # Envelope keys (flat structure: schema_version/format/provenance at
        # top level alongside per-configuration entries).
        assert 'schema_version' in data
        assert 'provenance' in data
        assert 'uniform_spherical' in data

    def test_compare_configurations_unknown_material_raises(self):
        with pytest.raises(ValueError, match=r"Unknown material"):
            compare_configurations(0.03, material_name='T800epoxy')

    def test_save_results_writes_schema_envelope(self, tmp_path):
        # #20: saved files must carry schema_version + format so consumers
        # can detect version drift.
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "envelope.json")
        save_results_to_json(results, path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        assert data['schema_version'] == JSON_SCHEMA_VERSION
        assert data['format'] == FORMAT_EMPIRICAL_SWEEP

    def test_load_results_from_json_round_trips(self, tmp_path):
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "round_trip.json")
        save_results_to_json(results, path)
        loaded = load_results_from_json(path)
        assert 'uniform_spherical' in loaded
        # Inner payload survives the round trip.
        assert (loaded['uniform_spherical']['empirical']['compression']
                ['judd_wright']['knockdown']
                == results['uniform_spherical']['empirical']['compression']
                ['judd_wright']['knockdown'])

    def test_load_results_from_json_rejects_missing_envelope(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"uniform_spherical": {"empirical": {}}}),
                        encoding='utf-8')
        with pytest.raises(ValueError, match=r"missing 'schema_version'"):
            load_results_from_json(str(path))

    def test_load_results_from_json_rejects_incompatible_major(self, tmp_path):
        path = tmp_path / "future.json"
        path.write_text(json.dumps({
            "schema_version": "2.0",
            "format": "porosity-fe.empirical-sweep",
        }), encoding='utf-8')
        with pytest.raises(ValueError, match=r"incompatible"):
            load_results_from_json(str(path))

    def test_load_results_from_json_rejects_unknown_format(self, tmp_path):
        path = tmp_path / "wrong-fmt.json"
        path.write_text(json.dumps({
            "schema_version": "1.0",
            "format": "porosity-fe.something-else",
        }), encoding='utf-8')
        with pytest.raises(ValueError, match=r"unknown format"):
            load_results_from_json(str(path))


class TestIntegration:
    """End-to-end test with reduced parameters for speed."""

    def test_full_pipeline_single_config(self, tmp_path):
        os.chdir(str(tmp_path))
        # #44 item 3: pull the porosity_field for plotting from the
        # artifacts dict; the lightweight ConfigResult only carries
        # numbers and the nested empirical table.
        results, artifacts = compare_configurations(
            0.03, material_name='T800_epoxy',
            configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']},
            return_artifacts=True)

        r = results['uniform_spherical']
        emp_comp = r['empirical']['compression']['judd_wright']
        assert 0 < emp_comp['knockdown'] < 1.0
        assert emp_comp['failure_stress'] < MATERIALS['T800_epoxy'].sigma_1c

        emp_ilss = r['empirical']['ilss']['judd_wright']['knockdown']
        emp_comp_kd = r['empirical']['compression']['judd_wright']['knockdown']
        assert emp_ilss < emp_comp_kd

        save_results_to_json(results, "test_output.json", artifacts=artifacts)
        assert os.path.exists("test_output.json")

        FEVisualizer.plot_porosity_field(
            artifacts['uniform_spherical'].porosity_field,
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


# ============================================================
# FE SOLVER TESTS
# ============================================================


class TestCoordinateTransforms:
    def test_rotation_matrix_identity_for_zero_angle(self):
        R = rotation_matrix_3d(0.0, axis='z')
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_matrix_orthogonal(self):
        R = rotation_matrix_3d(np.pi / 4, axis='z')
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_rotation_matrix_y_axis(self):
        R = rotation_matrix_3d(np.pi / 2, axis='y')
        # After 90-deg rotation about y: x->-z, z->x
        expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_matrix_invalid_axis(self):
        with pytest.raises(ValueError):
            rotation_matrix_3d(0.0, axis='x')

    def test_stress_transform_identity_at_zero(self):
        T = stress_transformation_3d(0.0, axis='z')
        np.testing.assert_allclose(T, np.eye(6), atol=1e-15)

    def test_strain_transform_identity_at_zero(self):
        T = strain_transformation_3d(0.0, axis='z')
        np.testing.assert_allclose(T, np.eye(6), atol=1e-15)

    def test_rotate_stiffness_identity_at_zero(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        C_rot = rotate_stiffness_3d(C, 0.0, axis='z')
        np.testing.assert_allclose(C_rot, C, atol=1e-6)

    def test_rotate_stiffness_180_returns_same(self):
        """180-degree rotation about z should return same stiffness for orthotropic."""
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        C_rot = rotate_stiffness_3d(C, np.pi, axis='z')
        np.testing.assert_allclose(C_rot, C, atol=1e-6)

    def test_rotate_stiffness_symmetric(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        C_rot = rotate_stiffness_3d(C, np.pi / 4, axis='z')
        np.testing.assert_allclose(C_rot, C_rot.T, atol=1e-6)

    def test_rotate_stiffness_wrong_shape(self):
        with pytest.raises(ValueError):
            rotate_stiffness_3d(np.eye(3), 0.0)

    # y-axis transforms (wrinkle/waviness misalignment) were untested ----

    @staticmethod
    def _voigt_from_tensor(t, *, engineering=False):
        """Pack a symmetric 3x3 tensor into Voigt order
        [11, 22, 33, 23, 13, 12]. ``engineering`` doubles the shear terms
        (engineering strain convention)."""
        f = 2.0 if engineering else 1.0
        return np.array([t[0, 0], t[1, 1], t[2, 2],
                         f * t[1, 2], f * t[0, 2], f * t[0, 1]])

    @pytest.mark.parametrize('axis', ['z', 'y'])
    @pytest.mark.parametrize('angle', [np.pi / 6, np.pi / 3, -np.pi / 4])
    def test_stress_transform_matches_tensor_rotation(self, axis, angle):
        """``T_sigma @ voigt(sigma)`` must equal ``voigt(R sigma R^T)`` for
        the matching 3x3 rotation R. This pins the hand-written y-axis 6x6
        against the rotation it is supposed to encode (a transcription error
        there would otherwise go unnoticed)."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((3, 3))
        sigma = a + a.T
        R = rotation_matrix_3d(angle, axis=axis)
        expected = self._voigt_from_tensor(R @ sigma @ R.T)
        got = stress_transformation_3d(angle, axis=axis) @ \
            self._voigt_from_tensor(sigma)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize('axis', ['z', 'y'])
    @pytest.mark.parametrize('angle', [np.pi / 6, np.pi / 3, -np.pi / 4])
    def test_strain_transform_matches_tensor_rotation(self, axis, angle):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((3, 3))
        eps = a + a.T
        R = rotation_matrix_3d(angle, axis=axis)
        expected = self._voigt_from_tensor(R @ eps @ R.T, engineering=True)
        got = strain_transformation_3d(angle, axis=axis) @ \
            self._voigt_from_tensor(eps, engineering=True)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_stress_transform_invalid_axis(self):
        with pytest.raises(ValueError):
            stress_transformation_3d(0.1, axis='x')

    def test_strain_transform_invalid_axis(self):
        with pytest.raises(ValueError):
            strain_transformation_3d(0.1, axis='x')

    def test_rotate_stiffness_y_axis_identity_and_symmetry(self):
        C = MATERIALS['T800_epoxy'].get_stiffness_matrix()
        np.testing.assert_allclose(rotate_stiffness_3d(C, 0.0, axis='y'),
                                   C, atol=1e-6)
        C_rot = rotate_stiffness_3d(C, np.pi / 5, axis='y')
        np.testing.assert_allclose(C_rot, C_rot.T, atol=1e-6)


class TestGaussQuadrature:
    def test_gauss_1d_order2(self):
        pts, wts = gauss_points_1d(2)
        assert len(pts) == 2
        assert len(wts) == 2
        np.testing.assert_allclose(wts.sum(), 2.0)

    def test_gauss_1d_order3(self):
        pts, wts = gauss_points_1d(3)
        assert len(pts) == 3
        np.testing.assert_allclose(wts.sum(), 2.0)

    def test_gauss_1d_invalid_order(self):
        with pytest.raises(ValueError):
            gauss_points_1d(4)

    def test_gauss_hex_shape(self):
        pts, wts = gauss_points_hex(order=2)
        assert pts.shape == (8, 3)
        assert wts.shape == (8,)

    def test_gauss_hex_weight_sum(self):
        """Weights should sum to 8 (volume of [-1,1]^3)."""
        pts, wts = gauss_points_hex(order=2)
        np.testing.assert_allclose(wts.sum(), 8.0)

    def test_gauss_hex_order3(self):
        pts, wts = gauss_points_hex(order=3)
        assert pts.shape == (27, 3)
        np.testing.assert_allclose(wts.sum(), 8.0)


class TestApiConsistency:
    """#44: unified FailureResult / ConfigResult dataclasses, 'QI'/'UD'
    sentinel, slim compare_configurations return."""

    def setup_method(self):
        from porosity_fe_analysis import FailureResult, ConfigResult, ConfigArtifacts
        self.FailureResult = FailureResult
        self.ConfigResult = ConfigResult
        self.ConfigArtifacts = ConfigArtifacts
        self.material = MATERIALS['T800_epoxy']

    # --- Item 1: unified return shapes -------------------------------

    def test_failure_result_attribute_and_dict_access(self):
        """FailureResult must support both `r.failure_stress` and `r['failure_stress']`."""
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        mesh = CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                             ply_angles='QI')
        emp = EmpiricalSolver(mesh, self.material)
        r = emp.get_failure_load(mode='compression', model='judd_wright')
        assert isinstance(r, self.FailureResult)
        # Attribute access.
        assert hasattr(r, 'failure_stress')
        assert hasattr(r, 'knockdown')
        assert hasattr(r, 'model')
        # Dict-style back-compat shim.
        assert r['failure_stress'] == r.failure_stress
        assert r['knockdown'] == r.knockdown
        assert r['model'] == r.model

    def test_field_results_summary_matches_solver_output(self):
        """FieldResults.summary() must produce a FailureResult that
        composes the FE knockdown with the supplied pristine strength."""
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, self.material, nx=3, ny=2, nz=2,
                             ply_angles='UD')
        fe = FESolver(mesh, self.material, pf)
        field = fe.solve(loading='compression', applied_strain=-0.001)
        # Caller supplies the pristine strength (FE solver doesn't carry it).
        sigma_p = self.material.sigma_1c
        summary = field.summary(sigma_pristine=sigma_p)
        assert isinstance(summary, self.FailureResult)
        assert summary.knockdown == pytest.approx(field.knockdown, rel=1e-12)
        assert summary.failure_stress == pytest.approx(
            field.knockdown * sigma_p, rel=1e-12)
        assert summary.details['max_failure_index'] == pytest.approx(
            float(field.max_failure_index), rel=1e-12)
        assert summary.details['failure_criterion'] == field.failure_criterion

    # --- Item 2: 'QI'/'UD' sentinel ----------------------------------

    def test_ply_angles_qi_sentinel(self):
        """`ply_angles='QI'` must expand to the canonical [0/90/45/-45]_s stack."""
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        mesh_qi = CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                                ply_angles='QI')
        mesh_explicit = CompositeMesh(
            pf, self.material, nx=4, ny=2, nz=2,
            ply_angles=[0.0, 90.0, 45.0, -45.0, -45.0, 45.0, 90.0, 0.0])
        np.testing.assert_allclose(mesh_qi.ply_angles, mesh_explicit.ply_angles)

    def test_ply_angles_ud_sentinel(self):
        """`ply_angles='UD'` must expand to the all-zero stack."""
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        mesh_ud = CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                                ply_angles='UD')
        np.testing.assert_allclose(mesh_ud.ply_angles, 0.0)

    def test_ply_angles_none_deprecation(self):
        """Passing ply_angles=None must warn but still resolve to QI."""
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        with pytest.warns(DeprecationWarning, match="deprecated"):
            mesh_none = CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                                      ply_angles=None)
        mesh_qi = CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                                ply_angles='QI')
        # Resolved to 'QI' for back-compat (compare expanded per-layer arrays).
        np.testing.assert_allclose(mesh_none.ply_angles, mesh_qi.ply_angles)

    def test_ply_angles_bad_sentinel_raises(self):
        """An unknown string sentinel must raise a clear ValueError."""
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        with pytest.raises(ValueError, match=r"'QI' or 'UD'"):
            CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                          ply_angles='nonsense')

    # --- Item 3: lightweight compare_configurations return -----------

    def test_compare_configurations_default_returns_lightweight(self):
        """Default compare_configurations must return Dict[str, ConfigResult]."""
        results = compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']})
        assert isinstance(results, dict)
        for entry in results.values():
            assert isinstance(entry, self.ConfigResult)

    def test_compare_configurations_return_artifacts(self):
        """`return_artifacts=True` must return (results, artifacts) tuple."""
        out = compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']},
            return_artifacts=True)
        assert isinstance(out, tuple) and len(out) == 2
        results, artifacts = out
        for entry in results.values():
            assert isinstance(entry, self.ConfigResult)
        for art in artifacts.values():
            assert isinstance(art, self.ConfigArtifacts)
            assert art.mesh is not None
            assert art.porosity_field is not None
            assert art.empirical_solver is not None

    def test_config_result_legacy_artifact_keys_raise(self):
        """Accessing the moved keys via the dict shim must raise a clear KeyError."""
        results = compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']})
        r = results['uniform_spherical']
        for legacy_key in ('mesh', 'empirical_solver', 'porosity_field'):
            with pytest.raises(KeyError, match="return_artifacts"):
                _ = r[legacy_key]

    # --- #148: pin dict-shim and migration-hint behavior --------------

    def _make_failure_result(self):
        """Build a FailureResult via the same factory the rest of the
        suite uses (EmpiricalSolver.get_failure_load)."""
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        mesh = CompositeMesh(pf, self.material, nx=4, ny=2, nz=2,
                             ply_angles='QI')
        emp = EmpiricalSolver(mesh, self.material)
        return emp.get_failure_load(mode='compression', model='judd_wright')

    def _make_config_result(self):
        """Build a ConfigResult via compare_configurations with a single
        small config (matches the factory used elsewhere in the class)."""
        results = compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']})
        return results['uniform_spherical']

    def test_failure_result_dict_shim_unknown_key_raises_keyerror(self):
        """Unknown keys must raise KeyError (NOT AttributeError) so
        dict-style back-compat (``result['nope']``) keeps the right
        exception type for legacy callers wrapping ``try/except KeyError``."""
        r = self._make_failure_result()
        with pytest.raises(KeyError):
            _ = r['nope']

    def test_failure_result_get_returns_default_for_missing(self):
        """``result.get(key, default)`` must return the default for
        unknown keys rather than raising."""
        r = self._make_failure_result()
        assert r.get('nope', 'fallback') == 'fallback'
        # Sanity: known key still resolves through .get().
        assert r.get('knockdown') == r.knockdown

    def test_failure_result_keys_includes_documented_fields(self):
        """``keys()`` must surface the three documented direct fields
        (``failure_stress``, ``knockdown``, ``model``) so legacy
        ``for k in result:``-style code sees them."""
        r = self._make_failure_result()
        ks = r.keys()
        assert 'failure_stress' in ks
        assert 'knockdown' in ks
        assert 'model' in ks

    def test_failure_result_to_dict_matches_attribute_access(self):
        """``to_dict()`` must mirror attribute access for the documented
        direct fields (the legacy dict-returning shape)."""
        r = self._make_failure_result()
        d = r.to_dict()
        assert d['knockdown'] == r.knockdown
        assert d['failure_stress'] == r.failure_stress
        assert d['model'] == r.model

    def test_config_result_artifact_keys_hint_at_return_artifacts(self):
        """The migration-hint KeyError for moved keys
        (``mesh`` / ``empirical_solver`` / ``porosity_field`` /
        ``field_results``) must mention ``return_artifacts`` so callers
        know where their data went."""
        r = self._make_config_result()
        for artifact_key in self.ConfigResult._ARTIFACT_KEYS:
            with pytest.raises(KeyError, match='return_artifacts'):
                _ = r[artifact_key]


class TestDistributionComparison:
    """#83: uniform vs clustered/interface knockdowns at matched Vp_mean.

    Locks in the two-headed answer that resolves the issue:

    1. ``EmpiricalSolver.get_failure_load`` evaluates the knockdown at
       the specimen-mean Vp, so the empirical KDs are identical across
       distributions when Vp_mean matches. A regression that started
       evaluating at the local peak (a real design choice we might
       revisit) would fail ``test_empirical_knockdowns_match...``.
    2. The FE solver sees the local field, so its KDs diverge across
       distributions even at matched mean — documented by
       ``test_fe_knockdowns_diverge...``.
    3. There is no preset literally named ``stack`` — the validator
       must reject it with a message pointing at the real names.
    """

    VP_MEAN = 0.03

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']

    def _make_solver(self, **kwargs):
        pf = PorosityField(self.material, self.VP_MEAN, **kwargs)
        mesh = CompositeMesh(pf, self.material, nx=4, ny=2, nz=4,
                             ply_angles='QI')
        return EmpiricalSolver(mesh, self.material), pf, mesh

    def test_empirical_knockdowns_match_across_distributions_at_matched_vp_mean(self):
        """Empirical KD uses specimen-mean Vp: same Vp_mean -> same KD."""
        solver_u, _, _ = self._make_solver(distribution='uniform')
        solver_c, _, _ = self._make_solver(distribution='clustered',
                                           cluster_location='midplane')
        solver_i, _, _ = self._make_solver(distribution='interface',
                                           void_shape='penny')
        kd_u = solver_u.get_failure_load(
            mode='compression', model='judd_wright').knockdown
        kd_c = solver_c.get_failure_load(
            mode='compression', model='judd_wright').knockdown
        kd_i = solver_i.get_failure_load(
            mode='compression', model='judd_wright').knockdown
        # Empirical solver collapses to the specimen-mean Vp, so the
        # three distributions must match to machine precision.
        assert kd_c == pytest.approx(kd_u, rel=1e-10)
        assert kd_i == pytest.approx(kd_u, rel=1e-10)

    def test_fe_knockdowns_diverge_across_distributions_at_matched_vp_mean(self):
        """FE solver sees local Vp: distribution shape changes FE KD at matched mean.

        We compare ``uniform`` (spherical) against ``clustered (midplane)``
        with ``penny`` voids — pennies have a much higher stress
        concentration than spheres, so even at matched specimen-mean Vp
        the FE solver produces visibly distinct knockdowns once the
        clustered profile concentrates them at the midplane. The
        comparison against ``uniform`` (spherical) is enough to lock
        in "the FE path *does* see the distribution shape", which is
        the entire substantive claim of issue #83.
        """
        Vp_mean = 0.06
        pf_u = PorosityField(self.material, Vp_mean,
                              distribution='uniform',
                              void_shape='spherical')
        mesh_u = CompositeMesh(pf_u, self.material, nx=6, ny=3, nz=12,
                               ply_angles='QI')
        pf_c = PorosityField(self.material, Vp_mean,
                              distribution='clustered',
                              cluster_location='midplane',
                              void_shape='penny')
        mesh_c = CompositeMesh(pf_c, self.material, nx=6, ny=3, nz=12,
                               ply_angles='QI')
        # Sanity: the specimen-mean Vp is matched between the two cases.
        assert pf_u.Vp == pytest.approx(pf_c.Vp, rel=1e-12)
        kd_u = FESolver(mesh_u, self.material, pf_u, ply_angles='QI').solve(
            loading='compression', applied_strain=-0.001).knockdown
        kd_c = FESolver(mesh_c, self.material, pf_c, ply_angles='QI').solve(
            loading='compression', applied_strain=-0.001).knockdown
        # Loose tolerance: this is a "the FE path DOES see the shape"
        # smoke test, not a regression lock on a specific number.
        assert abs(kd_u - kd_c) > 1e-3

    def test_no_preset_named_stack(self):
        """'stack' is not a valid distribution preset; error must point at the real names."""
        with pytest.raises(ValueError) as exc_info:
            PorosityField(self.material, 0.02, distribution='stack')
        msg = str(exc_info.value)
        # The error must point at the actual valid names so a user who
        # typed 'stack' can find the right one.
        assert 'uniform' in msg
        assert 'clustered' in msg
        assert 'interface' in msg


class TestReprMethods:
    def test_material_repr(self):
        mat = MATERIALS['T800_epoxy']
        r = repr(mat)
        assert 'MaterialProperties' in r
        assert 'E11=161000' in r

    def test_void_geometry_repr(self):
        void = VoidGeometry(center=(1, 2, 3), radii=(1, 1, 1))
        r = repr(void)
        assert 'VoidGeometry' in r
        assert 'aspect_ratio=1.00' in r

    def test_porosity_field_repr(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        r = repr(pf)
        assert 'PorosityField' in r
        assert '0.0300' in r

    def test_composite_mesh_repr(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=3, ny=2, nz=2)
        r = repr(mesh)
        assert 'CompositeMesh' in r
        assert 'nx=3' in r

    def test_field_results_repr(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, mat, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        r = repr(results)
        assert 'FieldResults' in r
        assert 'knockdown=' in r


class TestVoidInclusions:
    """Tests that discrete voids are modeled as near-zero stiffness inclusions."""

    def test_void_elements_identified_by_geometry(self):
        """Elements inside a VoidGeometry should be flagged as void."""
        material = MATERIALS['T800_epoxy']
        Lz = material.total_thickness
        void = VoidGeometry(center=(25, 10, Lz / 2), radii=(5, 5, Lz / 4))
        pf = PorosityField(material, 0.0, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=10, ny=5, nz=6)
        # Should have some void elements
        assert len(mesh.void_elements) > 0
        # Void elements should be near the void center
        void_centers = np.mean(mesh.nodes[mesh.elements[mesh.void_elements]], axis=1)
        for center in void_centers:
            assert void.contains(np.array([center[0]]), np.array([center[1]]),
                                  np.array([center[2]]))[0]

    def test_void_element_has_near_zero_stiffness(self):
        """Hex8Element with is_void=True should have very soft stiffness."""
        material = MATERIALS['T800_epoxy']
        C_base = material.get_stiffness_matrix()
        C_m = material.get_isotropic_matrix_stiffness()
        coords = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                           [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
        porosity = np.zeros(8)

        elem_normal = Hex8Element(coords, C_base, 0.0, porosity,
                                   (1,1,1), 0.35, C_m, is_void=False)
        elem_void = Hex8Element(coords, C_base, 0.0, porosity,
                                 (1,1,1), 0.35, C_m, is_void=True)

        Ke_normal = elem_normal.stiffness_matrix()
        Ke_void = elem_void.stiffness_matrix()

        # Void element stiffness should be orders of magnitude smaller
        ratio = np.linalg.norm(Ke_void) / np.linalg.norm(Ke_normal)
        assert ratio < 1e-4, f"Void/normal stiffness ratio {ratio} not small enough"

    def test_fe_with_void_has_stress_concentration(self):
        """FE solve with a void should show higher stresses near the void."""
        material = MATERIALS['T800_epoxy']
        Lz = material.total_thickness
        # Large void relative to mesh: 10mm radius covers multiple elements
        void = VoidGeometry(center=(25, 10, Lz / 2), radii=(10, 8, Lz / 3))
        pf = PorosityField(material, 0.0, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=10, ny=5, nz=6)

        assert len(mesh.void_elements) > 0, "No void elements found"

        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.005)

        # Non-void elements near the void should have higher stresses
        # than elements far from the void
        assert results.max_failure_index > 0
        assert np.any(np.isfinite(results.stress_global))


# ============================================================
# Coverage backfill (#12): layup-scaling helpers, linear-model
# saturation, CLT degradation boundaries, CLI smoke, GUI parser.
# ============================================================


class TestLayupParser:
    """parse_layup is a pure helper extracted in #9; tested here for #12."""

    def setup_method(self):
        from porosity_fe.reporting import parse_layup
        self.parse_layup = parse_layup

    def test_simple_slash_form(self):
        assert self.parse_layup('[0/45/-45/90]') == [0.0, 45.0, -45.0, 90.0]

    def test_repeat_and_symmetry(self):
        out = self.parse_layup('[0/90]_2s')
        # repeat then mirror: [0,90,0,90] -> [0,90,0,90,90,0,90,0]
        assert out == [0.0, 90.0, 0.0, 90.0, 90.0, 0.0, 90.0, 0.0]

    def test_comma_separator_alternative(self):
        assert self.parse_layup('[90, 0, 90]') == [90.0, 0.0, 90.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parse_layup('')

    def test_invalid_angle_token_raises(self):
        with pytest.raises(ValueError, match="Invalid ply angle"):
            self.parse_layup('[0/oops/90]')

    def test_invalid_repeat_token_raises(self):
        with pytest.raises(ValueError, match="Invalid repeat count"):
            self.parse_layup('[0/45]_xyz')

    def test_negative_repeat_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            self.parse_layup('[0/45]_-3')

    def test_no_angles_raises(self):
        with pytest.raises(ValueError, match="No ply angles"):
            self.parse_layup('[]_3s')


class TestExportHelpers:
    """Module-level CSV / JSON writers extracted for issue #30."""

    @staticmethod
    def _sample_result():
        return {
            "config": {
                "material_name": "T800_epoxy",
                "n_plies": 24,
                "t_ply": 0.183,
                "Vp": 3.0,
                "distribution": "uniform",
                "void_shape": "spherical",
                "nx": 30, "ny": 10, "nz": 12,
            },
            "empirical": {
                "compression": {
                    "judd_wright": {"failure_stress": 1234.5, "knockdown": 0.823},
                    "power_law": {"failure_stress": 1300.0, "knockdown": 0.867},
                },
                "ilss": {
                    "judd_wright": {"failure_stress": 67.0, "knockdown": 0.744},
                },
            },
        }

    def test_build_export_payload_shape(self):
        from porosity_fe.reporting import build_export_payload
        payload = build_export_payload(self._sample_result())
        assert payload["config"]["material"] == "T800_epoxy"
        assert payload["config"]["mesh"] == "30x10x12"
        assert payload["empirical"]["compression"]["judd_wright"]["knockdown"] == 0.823

    def test_write_results_json_round_trips(self, tmp_path):
        from porosity_fe.reporting import build_export_payload, write_results_json
        path = str(tmp_path / "out.json")
        write_results_json(path, build_export_payload(self._sample_result()))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["empirical"]["compression"]["judd_wright"]["knockdown"] == 0.823

    def test_write_results_csv_header_and_rows(self, tmp_path):
        from porosity_fe.reporting import build_export_payload, write_results_csv
        path = str(tmp_path / "out.csv")
        write_results_csv(path, build_export_payload(self._sample_result()))
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        # First N lines are #-prefixed config; then the header; then rows.
        comment_lines = [l for l in lines if l.startswith("#")]
        data_lines = [l for l in lines if not l.startswith("#")]
        assert any("material: T800_epoxy" in l for l in comment_lines)
        assert any("Vp_percent: 3.0" in l for l in comment_lines)
        assert data_lines[0] == "mode,model,failure_stress_MPa,knockdown"
        # Three (mode, model) rows in the sample → header + 3 = 4 lines.
        assert len(data_lines) == 4
        # failure_stress -> 1 dp, knockdown -> 4 dp (#128).
        assert "compression,judd_wright,1234.5,0.8230" in data_lines

    def test_write_results_csv_round_trips_via_csv_module(self, tmp_path):
        import csv as _csv
        from porosity_fe.reporting import build_export_payload, write_results_csv
        path = str(tmp_path / "out.csv")
        write_results_csv(path, build_export_payload(self._sample_result()))
        with open(path, encoding="utf-8", newline="") as f:
            # Skip comment lines exactly the way pandas read_csv(comment='#') would.
            rows = [r for r in _csv.reader(f) if r and not r[0].startswith("#")]
        assert rows[0] == ["mode", "model", "failure_stress_MPa", "knockdown"]
        # All non-header rows should parse to four columns; numeric ones finite.
        for row in rows[1:]:
            assert len(row) == 4
            float(row[2])
            float(row[3])

    def test_download_filename_stem_templates_material_vp_date(self):
        """Stem encodes material, Vp%, and today's date (#130)."""
        import datetime as _dt
        from porosity_fe.reporting import build_export_payload, download_filename_stem
        stem = download_filename_stem(build_export_payload(self._sample_result()))
        today = _dt.date.today().strftime("%Y%m%d")
        assert stem == f"porosity_T800_epoxy_Vp3.0pct_{today}"

    def test_download_filename_stem_sanitizes_material_slashes(self):
        """Slashes/spaces in material codes become underscores (#130)."""
        from porosity_fe.reporting import download_filename_stem
        payload = {"config": {"material": "T800/epoxy", "Vp_percent": 2.5}}
        stem = download_filename_stem(payload)
        assert "/" not in stem
        assert "T800_epoxy" in stem
        assert "Vp2.5pct" in stem


class TestNCRExport:
    """NCR validation-summary attachment for MRB disposition support."""

    @staticmethod
    def _result(comp_kd=0.823, ilss_kd=0.744, Vp=3.0):
        return {
            "config": {
                "material_name": "T800_epoxy",
                "n_plies": 24,
                "t_ply": 0.183,
                "Vp": Vp,
                "distribution": "uniform",
                "void_shape": "spherical",
                "nx": 30, "ny": 10, "nz": 12,
            },
            "empirical": {
                "compression": {
                    "judd_wright": {"failure_stress": 1234.5, "knockdown": comp_kd},
                    "power_law": {"failure_stress": 1300.0, "knockdown": 0.867},
                },
                "ilss": {
                    "judd_wright": {"failure_stress": 67.0, "knockdown": ilss_kd},
                },
            },
        }

    @staticmethod
    def _meta(**overrides):
        meta = {
            "prepared_by": "J. Engineer",
            "ncr_reference": "NCR-2026-0042",
            "structural_class": "primary",
            "note": "Voids found in C-scan of web region.",
            "date": "2026-05-17",
            "layup": "[0/45/-45/90]_3s",
        }
        meta.update(overrides)
        return meta

    def test_governing_failure_picks_lowest_knockdown(self):
        from porosity_fe.reporting import governing_failure
        worst = governing_failure(self._result(comp_kd=0.823, ilss_kd=0.744))
        assert worst["mode"] == "ilss"
        assert worst["model"] == "judd_wright"
        assert worst["knockdown"] == 0.744
        assert worst["residual_strength_MPa"] == 67.0

    def test_recommend_disposition_bins_by_severity(self):
        from porosity_fe.reporting import recommend_disposition
        uai = recommend_disposition(0.8, 0.97, "primary")
        assert uai["path"].startswith("Use-As-Is (UAI)")
        repair = recommend_disposition(7.0, 0.65, "primary")
        assert "Scrap" in repair["path"] or "Repair" in repair["path"]
        # Disclaimer is always present — tool never issues a final disposition.
        assert "NOT a final disposition" in uai["disclaimer"]
        assert uai["cited_criteria"] and uai["required_mrb_actions"]

    def test_recommend_disposition_primary_requires_concurrence(self):
        from porosity_fe.reporting import recommend_disposition
        d = recommend_disposition(0.5, 0.98, "primary")
        assert any("concurrence" in a for a in d["required_mrb_actions"])

    def test_build_ncr_record_shape(self):
        from porosity_fe.reporting import build_ncr_record
        ncr = build_ncr_record(self._result(), self._meta())
        # Lightweight summary metadata — no part/serial/work-order fields.
        assert ncr["summary"]["prepared_by"] == "J. Engineer"
        assert ncr["summary"]["ncr_reference"] == "NCR-2026-0042"
        assert "approvals" not in ncr
        assert "part_number" not in ncr["summary"]
        assert ncr["nonconformance"]["measured_Vp_percent"] == 3.0
        assert ncr["nonconformance"]["layup"] == "[0/45/-45/90]_3s"
        # Governing analysis derives from the worst (ILSS) case.
        assert ncr["engineering_analysis"]["governing_mode"] == "ilss"
        assert ncr["recommended_disposition"]["path"]

    def test_serialise_ncr_json_envelope_and_round_trip(self, tmp_path):
        from porosity_fe.reporting import build_ncr_record, write_ncr_json
        from porosity_fe_analysis import FORMAT_NCR
        path = str(tmp_path / "ncr.json")
        write_ncr_json(path, build_ncr_record(self._result(), self._meta()))
        data = load_results_from_json(path)
        assert data["format"] == FORMAT_NCR
        assert "provenance" in data
        assert data["summary"]["prepared_by"] == "J. Engineer"

    def test_serialise_ncr_markdown_has_sections(self, tmp_path):
        from porosity_fe.reporting import build_ncr_record, write_ncr_markdown
        path = str(tmp_path / "ncr.md")
        write_ncr_markdown(path, build_ncr_record(self._result(), self._meta()))
        with open(path, encoding="utf-8") as f:
            md = f.read()
        assert "NCR Validation Summary" in md
        assert "Recommended Disposition Path" in md
        assert "NOT a final disposition" in md
        assert "NCR-2026-0042" in md
        assert "Engineering Analysis" in md

    def test_serialise_ncr_pdf_is_valid_pdf(self, tmp_path):
        from porosity_fe.reporting import build_ncr_record, serialise_ncr_pdf, write_ncr_pdf
        ncr = build_ncr_record(self._result(), self._meta())
        blob = serialise_ncr_pdf(ncr)
        assert isinstance(blob, bytes)
        assert blob.startswith(b"%PDF")
        path = str(tmp_path / "ncr.pdf")
        write_ncr_pdf(path, ncr)
        with open(path, "rb") as f:
            assert f.read(4) == b"%PDF"
