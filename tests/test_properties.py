#!/usr/bin/env python3
"""Property-based tests (IMPROVEMENT_PLAN 6.6).

Invariants that must hold for every input, checked with ``hypothesis``
rather than a handful of hand-picked cases: frame transformations form a
group and preserve work, the Tsai-Wu index is frame invariant, and every
built-in knockdown law is a monotone fraction on the calibrated range.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from porosity_fe import MATERIALS, build_empirical_pipeline
from porosity_fe.empirical import _KNOCKDOWN_LAWS, Calibration
from porosity_fe.fe.failure import degraded_strengths, evaluate_tsai_wu
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
        """After the f_md layup scaling, every mode's knockdown is still a
        non-increasing fraction on [0, 0.05]."""
        _pf, _mesh, emp = build_empirical_pipeline(
            MAT, 0.02, ply_angles=layup, mesh_res=(4, 3, 3))
        vp = np.linspace(0.0, 0.05, 11)
        for mode in Calibration.JUDD_WRIGHT_ALPHA_QI:
            law, coef = emp._builtin_law(model, mode)
            kd = law(vp, coef)
            assert np.all((kd > 0.0) & (kd <= 1.0)), (layup, model, mode)
            assert np.all(np.diff(kd) <= 1e-15), (layup, model, mode)
