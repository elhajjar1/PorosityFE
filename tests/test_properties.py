#!/usr/bin/env python3
"""Property-based tests (IMPROVEMENT_PLAN 6.6).

Invariants that must hold for every input, checked with ``hypothesis``
rather than a handful of hand-picked cases: frame transformations form a
group and preserve work, the Tsai-Wu index is frame invariant, every
built-in knockdown law is a monotone fraction on the calibrated range, and
the empirical layup scale stays inside its convex bounds.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from porosity_fe import MATERIALS, build_empirical_pipeline
from porosity_fe._layup import _membrane_energy_partition
from porosity_fe.empirical import _KNOCKDOWN_LAWS, Calibration, _layup_scales
from porosity_fe.fe.failure import (
    SUPPORTED_FAILURE_CRITERIA,
    _point_load_factors_prestressed,
    degraded_strengths,
    evaluate_hashin,
    evaluate_max_stress,
    evaluate_tsai_wu,
)
from porosity_fe.transforms import (
    rotate_stiffness_3d,
    strain_transformation_3d,
    stress_transformation_3d,
)

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi,
                   allow_nan=False, allow_infinity=False)
voigt = st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
                 min_size=6, max_size=6).map(np.array)

MAT = MATERIALS['T800_epoxy']
C_PLY = MAT.get_stiffness_matrix()


class TestFrameTransforms:
    @given(angles)
    def test_rotation_round_trip(self, theta):
        C = rotate_stiffness_3d(rotate_stiffness_3d(C_PLY, theta), -theta)
        np.testing.assert_allclose(C, C_PLY, rtol=0, atol=1e-9 * np.abs(C_PLY).max())

    @given(angles, angles)
    def test_stress_transform_is_a_group(self, a, b):
        np.testing.assert_allclose(
            stress_transformation_3d(a) @ stress_transformation_3d(b),
            stress_transformation_3d(a + b), rtol=0, atol=1e-12)

    @given(angles, voigt, voigt)
    def test_work_is_frame_invariant(self, theta, sigma, eps):
        """sigma . eps (engineering shear) is a scalar; the strain transform
        must carry the engineering-shear factor of 2 for this to hold."""
        work = sigma @ eps
        rotated = (stress_transformation_3d(theta) @ sigma) @ \
            (strain_transformation_3d(theta) @ eps)
        assert rotated == pytest.approx(work, rel=1e-9, abs=1e-6)

    @given(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False).filter(
        lambda g: abs(g) > 1e-9))
    def test_pure_shear_at_45_degrees(self, gamma):
        """gamma_xy rotated by 45 deg is eps_11 = gamma/2, eps_22 = -gamma/2."""
        eps = np.zeros(6)
        eps[5] = gamma
        out = strain_transformation_3d(np.pi / 4) @ eps
        np.testing.assert_allclose(out[:2], [gamma / 2, -gamma / 2], rtol=1e-12)
        assert out[5] == pytest.approx(0.0, abs=1e-12 * abs(gamma))


class TestTsaiWuFrameInvariance:
    @settings(max_examples=50)
    @given(angles, angles, voigt)
    def test_index_depends_only_on_the_ply_local_stress(self, theta, phi, sigma):
        """Rotating the global load and the ply by the same angle leaves the
        local stress, and so the Tsai-Wu index, unchanged."""
        strengths = degraded_strengths(MAT, (1.0, 1.0, 1.0), 0.02)
        local = stress_transformation_3d(theta) @ sigma
        # Global stress rotated by -phi, seen by a ply at theta + phi.
        local_rotated = stress_transformation_3d(theta + phi) @ \
            (stress_transformation_3d(-phi) @ sigma)
        fi = evaluate_tsai_wu(local[None], strengths, 0, 0.02)
        fi_rotated = evaluate_tsai_wu(local_rotated[None], strengths, 0, 0.02)
        np.testing.assert_allclose(fi_rotated, fi, rtol=1e-9, atol=1e-9)


class TestPrestressedFirstPlyFailure:
    """The load factor with a fixed pre-stress (IMPROVEMENT_PLAN 3.5) is the
    first crossing: below it every criterion stays under 1, just above it
    the criterion has reached 1, including Hashin modes that switch on at a
    stress sign change."""

    strengths = degraded_strengths(MAT, (1.0, 1.0, 1.0), 0.02)

    @staticmethod
    def _fi(criterion, s, strengths):
        if criterion == 'tsai_wu':
            return evaluate_tsai_wu(s, strengths, 0, 0.02)
        if criterion == 'hashin':
            return evaluate_hashin(s, strengths)['max_fi']
        return evaluate_max_stress(s, strengths)['max_fi']

    @settings(max_examples=60, deadline=None)
    @pytest.mark.parametrize("criterion", SUPPORTED_FAILURE_CRITERIA)
    @given(s_th=voigt, s_m=voigt)
    def test_closed_form_is_the_first_crossing(self, criterion, s_th, s_m):
        s_th = 0.2 * s_th
        lam = float(_point_load_factors_prestressed(
            s_th[None], s_m[None], self.strengths, criterion, None)[0])
        fi_th = self._fi(criterion, s_th[None], self.strengths)[0]
        if lam == 0.0:
            assert fi_th >= 1.0
            return
        assert fi_th < 1.0
        top = lam if np.isfinite(lam) else 1e3
        below = np.linspace(0.0, top * (1.0 - 1e-7), 400)
        fi = self._fi(criterion, s_th[None] + below[:, None] * s_m[None],
                      self.strengths)
        assert fi.max() < 1.0 + 1e-9
        if np.isfinite(lam):
            above = lam * (1.0 + 1e-9) + 1e-12
            assert self._fi(criterion, (s_th + above * s_m)[None],
                            self.strengths)[0] >= 1.0 - 1e-6


class TestKnockdownLaws:
    vps = st.lists(st.floats(min_value=0.0, max_value=0.05, allow_nan=False),
                   min_size=2, max_size=20)

    @pytest.mark.parametrize("model", sorted(_KNOCKDOWN_LAWS))
    @pytest.mark.parametrize("mode", sorted(Calibration.JUDD_WRIGHT_ALPHA_QI))
    @given(vps=vps)
    def test_qi_law_is_monotone_fraction(self, model, mode, vps):
        law, table = _KNOCKDOWN_LAWS[model]
        coef = getattr(Calibration, table + '_QI')[mode]
        vp = np.sort(np.asarray(vps))
        kd = law(vp, coef)
        assert np.all((kd > 0.0) & (kd <= 1.0))
        assert np.all(np.diff(kd) <= 1e-15)
        assert float(law(0.0, coef)) == 1.0

    @settings(max_examples=15, deadline=None)
    @given(st.lists(st.sampled_from([0.0, 45.0, -45.0, 90.0, 30.0, -60.0]),
                    min_size=4, max_size=12),
           st.sampled_from(sorted(_KNOCKDOWN_LAWS)))
    def test_layup_scaled_law_is_monotone_fraction(self, layup, model):
        """After the layup scaling (amplification up to 2.56x for tension),
        every mode's knockdown is still a non-increasing fraction on
        [0, 0.05]."""
        _pf, _mesh, emp = build_empirical_pipeline(
            MAT, 0.02, ply_angles=layup, mesh_res=(4, 3, 3))
        vp = np.linspace(0.0, 0.05, 11)
        for mode in Calibration.JUDD_WRIGHT_ALPHA_QI:
            law, coef = emp._builtin_law(model, mode)
            kd = law(vp, coef)
            assert np.all((kd > 0.0) & (kd <= 1.0)), (layup, model, mode)
            assert np.all(np.diff(kd) <= 1e-15), (layup, model, mode)


class TestLayupScale:
    layups = st.lists(st.floats(min_value=-90.0, max_value=90.0,
                                allow_nan=False, allow_infinity=False),
                      min_size=1, max_size=12)

    @settings(max_examples=60, deadline=None)
    @given(layups, st.sampled_from(sorted(MATERIALS)))
    def test_scale_within_convex_bounds(self, layup, material_name):
        """Fiber-direction modes: 1 <= scale <= max(matrix anchors) / alpha;
        matrix modes: scale == 1, for any layup and material preset."""
        scales = _layup_scales(MATERIALS[material_name], layup)
        alpha = Calibration.JUDD_WRIGHT_ALPHA_QI
        for mode, scale in scales.items():
            if mode in Calibration.FIBER_DIRECTION_MODES:
                anchors = Calibration.LAYUP_MATRIX_ANCHORS[mode]
                upper = max(alpha[a] for a in anchors) / alpha[mode]
                assert 1.0 <= scale <= upper * (1 + 1e-12), (layup, mode, scale)
            else:
                assert scale == 1.0, (layup, mode)

    @settings(max_examples=40, deadline=None)
    @given(layups, st.sampled_from(['x', 'y', 'xy']))
    def test_energy_partition_sums_to_one_and_ignores_order(self, layup, direction):
        e = _membrane_energy_partition(MAT, layup, direction)
        assert sum(e) == pytest.approx(1.0, abs=1e-12)
        assert min(e) > -0.01  # Poisson-coupling terms stay small
        e_rev = _membrane_energy_partition(MAT, layup[::-1], direction)
        np.testing.assert_allclose(e_rev, e, rtol=0, atol=1e-12)
