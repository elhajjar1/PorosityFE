"""Tests for porosity_fe._layup (CLT membrane strain-energy partition).

The partition drives the empirical layup scaling (IMPROVEMENT_PLAN 2.7);
see tests/test_empirical.py for the scaling itself.
"""

import numpy as np
import pytest

from porosity_fe import MATERIALS
from porosity_fe._layup import _membrane_energy_partition
from porosity_fe.empirical import Calibration

MAT = MATERIALS['T800_epoxy']
QI = list(Calibration.PLY_ANGLES_QI)


class TestMembraneEnergyPartition:
    def test_ud_under_x_is_fiber_energy(self):
        e1, _e2, _e6 = _membrane_energy_partition(MAT, [0] * 8, 'x')
        assert e1 > 0.99

    def test_ninety_under_x_is_transverse_energy(self):
        _e1, e2, _e6 = _membrane_energy_partition(MAT, [90] * 8, 'x')
        assert e2 > 0.99

    def test_angle_ply_under_shear_is_fiber_energy(self):
        e1, _e2, _e6 = _membrane_energy_partition(MAT, [45, -45, -45, 45], 'xy')
        assert e1 > 0.9

    def test_ud_under_shear_is_shear_energy(self):
        e = _membrane_energy_partition(MAT, [0] * 4, 'xy')
        np.testing.assert_allclose(e, (0.0, 0.0, 1.0), atol=1e-12)

    @pytest.mark.parametrize("direction", ['x', 'y', 'xy'])
    def test_sums_to_one(self, direction):
        for layup in (QI, [0, 0, 90], [30, -30, 60], [15]):
            assert sum(_membrane_energy_partition(MAT, layup, direction)) == \
                pytest.approx(1.0, abs=1e-12)

    def test_invariant_to_stacking_permutation(self):
        a = _membrane_energy_partition(MAT, [0, 45, -45, 90, 90, -45, 45, 0])
        b = _membrane_energy_partition(MAT, [45, 90, 0, -45, 0, -45, 90, 45])
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_in_plane_isotropic_layups_agree(self):
        a = _membrane_energy_partition(MAT, QI)
        b = _membrane_energy_partition(MAT, [0, 60, -60, -60, 60, 0])
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_continuous_in_angle(self):
        thetas = np.linspace(0.0, 90.0, 361)
        parts = np.array([_membrane_energy_partition(MAT, [t, -t, -t, t])
                          for t in thetas])
        assert np.max(np.abs(np.diff(parts, axis=0))) < 0.02

    def test_matches_independent_textbook_clt(self):
        """Cross-check against a plain textbook CLT (explicit Q-bar formulas,
        Reuter-matrix strain rotation) built from the engineering constants,
        independent of the package's 3D rotation path."""
        layup = [0, 30, -60, 90, 15]
        E1, E2, G12, nu12 = MAT.E11, MAT.E22, MAT.G12, MAT.nu12
        nu21 = nu12 * E2 / E1
        d = 1.0 - nu12 * nu21
        Q = np.array([[E1 / d, nu12 * E2 / d, 0.0],
                      [nu12 * E2 / d, E2 / d, 0.0],
                      [0.0, 0.0, G12]])

        def T_sigma(theta):
            c, s_ = np.cos(theta), np.sin(theta)
            return np.array([[c * c, s_ * s_, 2 * c * s_],
                             [s_ * s_, c * c, -2 * c * s_],
                             [-c * s_, c * s_, c * c - s_ * s_]])

        R = np.diag([1.0, 1.0, 2.0])
        q_bars, T_eps = [], []
        for a in layup:
            Ts = T_sigma(np.radians(a))
            Te = R @ Ts @ np.linalg.inv(R)
            q_bars.append(np.linalg.inv(Ts) @ Q @ Te)
            T_eps.append(Te)
        eps0 = np.linalg.solve(sum(q_bars), np.array([1.0, 0.0, 0.0]))
        energy = sum(Q @ (Te @ eps0) * (Te @ eps0) for Te in T_eps)
        expected = energy / energy.sum()
        got = _membrane_energy_partition(MAT, layup, 'x')
        # The package reduces the full 3D stiffness to plane stress, which
        # equals the 2D Q built from E11/E22/G12/nu12 to round-off.
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError, match="load direction"):
            _membrane_energy_partition(MAT, QI, 'z')
        with pytest.raises(ValueError, match="at least one ply"):
            _membrane_energy_partition(MAT, [], 'x')
