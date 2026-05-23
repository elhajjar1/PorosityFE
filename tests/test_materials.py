#!/usr/bin/env python3
"""Tests for porosity_fe.materials.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""

import dataclasses

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (MaterialProperties, MATERIALS, PorosityField, CompositeMesh,
                                   EmpiricalSolver)


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

    def test_im7_preset_exists(self):
        assert 'IM7_8551_epoxy' in MATERIALS
        mat = MATERIALS['IM7_8551_epoxy']
        assert 170000 <= mat.E11 <= 180000

    def test_t300_934_preset_exists(self):
        assert 'T300_934_epoxy' in MATERIALS
        mat = MATERIALS['T300_934_epoxy']
        assert 125000 <= mat.E11 <= 140000

    def test_cf_peek_preset_exists(self):
        assert 'CF_PEEK' in MATERIALS
        mat = MATERIALS['CF_PEEK']
        assert 130000 <= mat.E11 <= 150000

    @staticmethod
    def _kwargs(**overrides):
        base = dict(
            E11=140000.0, E22=10500.0, E33=10500.0,
            G12=4900.0, G13=4900.0, G23=3700.0,
            nu12=0.30, nu13=0.30, nu23=0.42,
            sigma_1c=1300.0, sigma_1t=2500.0,
            sigma_2t=70.0, sigma_2c=210.0,
            tau_12=90.0, tau_ilss=85.0,
            t_ply=0.180, n_plies=24,
            matrix_modulus=3400.0, matrix_poisson=0.36,
            fiber_modulus=240000.0, fiber_volume_fraction=0.60,
        )
        base.update(overrides)
        return base

    def test_zero_modulus_rejected(self):
        with pytest.raises(ValueError, match=r"E11.*positive finite"):
            MaterialProperties(**self._kwargs(E11=0.0))

    def test_negative_strength_rejected(self):
        with pytest.raises(ValueError, match=r"sigma_1c.*positive finite"):
            MaterialProperties(**self._kwargs(sigma_1c=-100.0))

    def test_poisson_at_isotropic_limit_rejected(self):
        # nu = 0.5 makes (1 - 2*nu) = 0 in the isotropic matrix stiffness
        with pytest.raises(ValueError, match=r"matrix_poisson.*\(-1, 0\.5\)"):
            MaterialProperties(**self._kwargs(matrix_poisson=0.5))

    def test_negative_t_ply_rejected(self):
        with pytest.raises(ValueError, match=r"t_ply.*positive"):
            MaterialProperties(**self._kwargs(t_ply=-0.1))

    def test_zero_n_plies_rejected(self):
        with pytest.raises(ValueError, match=r"n_plies.*positive integer"):
            MaterialProperties(**self._kwargs(n_plies=0))

    def test_fiber_fraction_above_one_rejected(self):
        with pytest.raises(ValueError, match=r"fiber_volume_fraction"):
            MaterialProperties(**self._kwargs(fiber_volume_fraction=60.0))


class TestMaterialPropertiesPerturb:
    """Unit tests for the MaterialProperties.perturb sampling primitive."""

    def test_zero_draw_lognormal_is_identity(self):
        m = MATERIALS['T800_epoxy']
        out = m.perturb({'sigma_1c': 0.0}, {'sigma_1c': ('lognormal', 0.08)})
        # exp(sigma_ln * 0) == 1 -> nominal preserved.
        assert out.sigma_1c == pytest.approx(m.sigma_1c)
        # Untouched fields are copied through.
        assert out.E22 == m.E22

    def test_returns_new_instance_validated(self):
        m = MATERIALS['T800_epoxy']
        out = m.perturb({'E22': 1.5}, {'E22': ('lognormal', 0.05)})
        assert out is not m
        assert out.E22 > m.E22  # positive draw -> larger modulus
        assert isinstance(out, MaterialProperties)

    def test_unknown_distribution_rejected(self):
        m = MATERIALS['T800_epoxy']
        with pytest.raises(ValueError, match="Unknown distribution"):
            m.perturb({'sigma_1c': 0.1}, {'sigma_1c': ('weibull', 0.1)})

    def test_perturbed_value_zero_cov_lognormal_returns_nominal(self):
        """Lognormal at CoV=0 short-circuits to the exact nominal value."""
        m = MATERIALS['T800_epoxy']
        # Use a non-zero unit draw to prove the short-circuit returns nominal
        # without consuming the variate (otherwise exp(0 * 2.5) would still
        # equal 1.0 and we wouldn't be exercising the guard branch).
        out = m.perturb({'sigma_1c': 2.5}, {'sigma_1c': ('lognormal', 0.0)})
        assert out.sigma_1c == m.sigma_1c

    def test_perturbed_value_zero_cov_normal_returns_nominal(self):
        """Normal at CoV=0 short-circuits to the exact nominal value."""
        m = MATERIALS['T800_epoxy']
        out = m.perturb({'E22': -1.7}, {'E22': ('normal', 0.0)})
        assert out.E22 == m.E22

    def test_perturbed_value_zero_width_uniform_returns_nominal(self):
        """Uniform at half-width=0 short-circuits to the exact nominal value."""
        m = MATERIALS['T800_epoxy']
        # ``unit_draw`` is a U(0,1) variate for uniform; pick 0.75 so the
        # short-circuit (rather than the affine map evaluating to nominal by
        # coincidence at 0.5) is what enforces the identity.
        out = m.perturb({'tau_12': 0.75}, {'tau_12': ('uniform', 0.0)})
        assert out.tau_12 == m.tau_12


class TestEnvironmentalKnockdown:
    """#59: hygrothermal (T/M) and S-N fatigue knockdown surfaces.

    Threads ``environment=`` / ``cycles=`` / ``R=`` into
    :meth:`EmpiricalSolver.get_failure_load` and asserts they compose
    multiplicatively with the existing porosity knockdown.
    """

    def setup_method(self):
        from porosity_fe_analysis import FatigueModel, _FATIGUE_B_QI
        self.FatigueModel = FatigueModel
        self._FATIGUE_B_QI = _FATIGUE_B_QI
        # T800/epoxy with hygrothermal calibration: typical aerospace epoxy
        # T_g_dry ~ 200 deg C. Defaults T_ref = 23 C, M_ref = 0 wt%.
        self.material = dataclasses.replace(
            MATERIALS['T800_epoxy'], T_g_dry=200.0,
        )
        # Reference material with no hygrothermal calibration (T_g_dry=None)
        # so the env knockdown is a clean no-op.
        self.material_no_tg = MATERIALS['T800_epoxy']

    def _make_solver(self, material=None):
        material = material if material is not None else self.material
        pf = PorosityField(material, 0.02, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=4, ny=2, nz=2,
                             ply_angles='QI')
        return EmpiricalSolver(mesh, material), pf, mesh

    # -- Item 1: hygrothermal knockdown --------------------------------

    def test_environment_knockdown_noop_when_unspecified(self):
        """No ``environment`` kwarg -> factor 1.0, FailureResult unchanged."""
        solver, _, _ = self._make_solver()
        base = solver.get_failure_load(mode='ilss', model='judd_wright')
        env_off = solver.get_failure_load(mode='ilss', model='judd_wright',
                                          environment=None)
        assert base.knockdown == pytest.approx(env_off.knockdown, rel=1e-12)
        assert base.failure_stress == pytest.approx(env_off.failure_stress,
                                                     rel=1e-12)
        # ``environment_knockdown`` is only surfaced when active.
        assert 'environment_knockdown' not in base.details
        assert 'environment_knockdown' not in env_off.details

    def test_environment_knockdown_matrix_dominated_reduces(self):
        """Hot/wet conditioning must reduce matrix-dominated allowables."""
        solver, _, _ = self._make_solver()
        env = {'T': 80.0, 'M': 1.2}
        for mode in ('ilss', 'transverse_tension'):
            base = solver.get_failure_load(mode=mode, model='judd_wright')
            env_on = solver.get_failure_load(mode=mode, model='judd_wright',
                                             environment=env)
            assert env_on.details['environment_knockdown'] < 1.0, mode
            assert env_on.knockdown < base.knockdown, mode
            assert env_on.failure_stress < base.failure_stress, mode

    def test_environment_knockdown_fiber_dominated_unaffected(self):
        """Fiber-dominated 'tension' must see factor 1.0 even hot/wet."""
        solver, _, _ = self._make_solver()
        env = {'T': 80.0, 'M': 1.2}
        env_on = solver.get_failure_load(mode='tension', model='judd_wright',
                                         environment=env)
        # Even when ``environment`` is passed, the fiber-dominated mode
        # gets factor 1.0 from ``environment_knockdown``.
        assert env_on.details['environment_knockdown'] == pytest.approx(
            1.0, rel=1e-12)

    def test_environment_knockdown_below_glass_transition_safe(self):
        """``T_service`` well below dry ``T_g`` -> factor close to 1.0."""
        solver, _, _ = self._make_solver()
        # Cool & dry: T = 23 C, M = 0 -> ratio = 1.0 exactly.
        env_on = solver.get_failure_load(mode='ilss', model='judd_wright',
                                         environment={'T': 23.0, 'M': 0.0})
        assert env_on.details['environment_knockdown'] == pytest.approx(
            1.0, rel=1e-9)
        # Mildly warm & nearly dry: still close to 1.0.
        env_mild = solver.get_failure_load(mode='ilss', model='judd_wright',
                                            environment={'T': 30.0, 'M': 0.1})
        assert env_mild.details['environment_knockdown'] > 0.95

    # -- Item 1b: defensive branches of MaterialProperties.environment_knockdown

    def test_environment_knockdown_no_op_when_tg_dry_none(self):
        """No ``T_g_dry`` calibration -> direct call returns 1.0 (no-op)."""
        # ``self.material_no_tg`` is the preset MaterialProperties with
        # ``T_g_dry=None`` (the default), so the environment knockdown
        # short-circuits to the no-op identity even at hot/wet service.
        mat = self.material_no_tg
        assert mat.T_g_dry is None
        factor = mat.environment_knockdown('ilss', T=80.0, M=1.2)
        assert factor == 1.0

    def test_environment_knockdown_pathological_tref_above_tg(self):
        """``T_ref >= T_g_dry`` is pathological -> refuse to scale (return 1.0)."""
        # Force the (T_g_dry - T_ref) denominator to be non-positive by setting
        # T_g_dry below the default T_ref (=23 C). The code refuses to divide
        # by zero / negative and returns 1.0 instead.
        mat = dataclasses.replace(
            MATERIALS['T800_epoxy'], T_g_dry=20.0,
        )
        assert mat.T_ref >= mat.T_g_dry
        factor = mat.environment_knockdown('ilss', T=10.0, M=0.0)
        assert factor == 1.0

    def test_environment_knockdown_above_wet_tg_clamps_floor(self):
        """Service T above the wet T_g -> clamp to the 0.01 floor."""
        # T_g_wet = T_g_dry - 25 * M = 200 - 25 * 2 = 150 C; service at 200 C
        # is well above T_g_wet, so the numerator (T_g_wet - T_eff) is
        # negative and the code clamps to the documented 0.01 floor.
        mat = self.material  # T_g_dry = 200.0
        factor = mat.environment_knockdown('ilss', T=200.0, M=2.0)
        assert factor == pytest.approx(0.01, rel=1e-12)

    # -- Item 2: S-N fatigue knockdown ---------------------------------

    def test_fatigue_knockdown_noop_when_cycles_none(self):
        """``cycles=None`` -> factor 1.0, no ``fatigue_knockdown`` in details."""
        solver, _, _ = self._make_solver()
        base = solver.get_failure_load(mode='compression', model='judd_wright')
        fat_off = solver.get_failure_load(mode='compression', model='judd_wright',
                                          cycles=None)
        assert base.knockdown == pytest.approx(fat_off.knockdown, rel=1e-12)
        assert 'fatigue_knockdown' not in fat_off.details

    def test_fatigue_knockdown_log_linear_compression(self):
        """At N=1e6, compression knockdown ~ 1 - b * 6 with the canonical b."""
        solver, _, _ = self._make_solver()
        fat = solver.get_failure_load(mode='compression', model='judd_wright',
                                       cycles=1e6)
        b = self._FATIGUE_B_QI['compression']
        expected = 1.0 - b * 6.0
        assert fat.details['fatigue_knockdown'] == pytest.approx(expected,
                                                                  rel=1e-9)

    def test_fatigue_knockdown_floor_clamp_emits_warning(self):
        """At N=1e20 the log-linear extrapolation goes negative -> clamp+warn."""
        fm = self.FatigueModel()
        with pytest.warns(UserWarning, match="floor"):
            kd = fm.knockdown_factor('compression', 1e20)
        assert kd == pytest.approx(0.01, rel=1e-12)

    # -- Item 3: multiplicative composition ----------------------------

    def test_porosity_environment_fatigue_compose_multiplicatively(self):
        """All three knockdowns must compose as kd_porosity * kd_env * kd_fat."""
        solver, _, _ = self._make_solver()
        # Individual factors from solo runs.
        kd_porosity_only = solver.get_failure_load(
            mode='ilss', model='judd_wright').knockdown
        env = {'T': 80.0, 'M': 1.2}
        kd_env = solver.get_failure_load(
            mode='ilss', model='judd_wright',
            environment=env,
        ).details['environment_knockdown']
        kd_fat = solver.get_failure_load(
            mode='ilss', model='judd_wright',
            cycles=1e6,
        ).details['fatigue_knockdown']
        # Combined: porosity x env x fatigue.
        combined = solver.get_failure_load(
            mode='ilss', model='judd_wright',
            environment=env, cycles=1e6,
        )
        expected = kd_porosity_only * kd_env * kd_fat
        assert combined.knockdown == pytest.approx(expected, rel=1e-9)
        # Both extras must appear in details so the caller can audit.
        assert combined.details['environment_knockdown'] == pytest.approx(
            kd_env, rel=1e-12)
        assert combined.details['fatigue_knockdown'] == pytest.approx(
            kd_fat, rel=1e-12)
