"""Regression tests for the core-physics review of November 2026.

Each test reproduces a defect the review found and checks the fix:
the Tsai-Wu interaction range, a layup longer than the laminate, the
void-shape radii and the Mori-Tanaka cache, a porosity field of another
laminate, interface voids on one ply, a percent passed as ``Vp`` and the
reference moisture of the hygrothermal factor.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings

import numpy as np
import pytest

from porosity_fe import (MATERIALS, CompositeMesh, EmpiricalSolver, PorosityField,
                         compute_degraded_clt_flexural_modulus,
                         compute_degraded_clt_moduli)
from porosity_fe.fe import failure
from porosity_fe.homogenization import _degraded_composite_stiffness

T800 = MATERIALS['T800_epoxy']


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


class TestTsaiWuInteractionRange:
    """With F*_13 = F*_12 and F*_23 = -0.5 the normalized normal block has
    determinant 0.75 - 3 F*_12^2: below -0.5 the 3-D envelope is open."""

    @pytest.mark.parametrize('value', [-0.51, -0.8, -1.0])
    def test_values_below_minus_half_are_refused(self, value):
        with pytest.raises(ValueError, match=r'\[-0\.5, 0\].*opens the 3-D'):
            dataclasses.replace(T800, tsai_wu_F12=value)

    @pytest.mark.parametrize('value', [None, -0.5, -0.25, 0.0])
    def test_allowed_values_close_the_envelope(self, value):
        mat = dataclasses.replace(T800, tsai_wu_F12=value)
        strengths = failure.degraded_strengths(mat, (1, 1, 1), 0.0)
        (_, _, _, F11, F22, F33, _, _, _, F12, F13, F23) = \
            failure._tsai_wu_coefficients(strengths, mat.tsai_wu_F12)
        block = np.array([[F11, F12, F13], [F12, F22, F23], [F13, F23, F33]])
        d = np.sqrt(np.diag(block))
        assert np.linalg.eigvalsh(block / np.outer(d, d)).min() >= -1e-12
        # Every normal allowable exceeded 1.5 times: the index is past 1 and
        # the first-ply-failure load factor finite and below 1.
        s = np.array([[1.5 * mat.sigma_1t, 1.5 * mat.sigma_2t, 1.5 * mat.sigma_2t,
                       0.0, 0.0, 0.0]])
        assert failure.evaluate_tsai_wu(s, strengths, 0, 0.0, mat.tsai_wu_F12)[0] > 1.0
        lam = failure._point_load_factors(s, strengths, 'tsai_wu', mat.tsai_wu_F12)[0]
        assert 0.0 < lam < 1.0


class TestLayupLongerThanLaminate:
    def test_explicit_layup_longer_than_n_plies_raises(self):
        mat = dataclasses.replace(T800, n_plies=8)
        pf = PorosityField(mat, 0.02)
        with pytest.raises(ValueError, match='layup has 16 plies but the material has n_plies=8'):
            CompositeMesh(pf, mat, nx=4, ny=2, nz=8, ply_angles=[90] * 8 + [0] * 8)

    def test_patterns_still_fit_the_laminate(self):
        mat = dataclasses.replace(T800, n_plies=4)
        pf = PorosityField(mat, 0.02)
        # The 8-ply 'QI' pattern is cut, a shorter explicit list tiled.
        assert len(CompositeMesh(pf, mat, nx=4, ny=2, nz=4).element_layup) == 4
        mesh = CompositeMesh(pf, mat, nx=4, ny=2, nz=4, ply_angles=[0, 90])
        assert list(mesh.element_layup) == [0.0, 90.0, 0.0, 90.0]


class TestVoidShapeRadii:
    @pytest.mark.parametrize('radii', [(3, 3, -0.3), (3, 3, 0), (1, 1), (1, 1, float('nan')),
                                       (1, True, 1)])
    def test_invalid_radii_are_refused(self, radii):
        with pytest.raises(ValueError, match='three finite radii > 0'):
            PorosityField(T800, 0.03, void_shape=radii)

    @pytest.mark.parametrize('degradation', ['isotropic', 'directional'])
    def test_only_the_ratios_matter(self, degradation):
        """Radii in metres gave a crack (a3 rounded to 0) and the isotropic
        stiffness changed with the units."""
        ref = _degraded_composite_stiffness(0.05, (3, 3, 0.3), T800,
                                            porosity_degradation=degradation)
        for scale in (1e-6, 1e-3, 1e3):
            C = _degraded_composite_stiffness(0.05, (3 * scale, 3 * scale, 0.3 * scale), T800,
                                              porosity_degradation=degradation)
            np.testing.assert_allclose(C, ref, rtol=1e-9, atol=1e-9 * np.abs(ref).max())

    def test_presets_unchanged(self):
        """The ratio key keeps the presets' radii to 12 decimals."""
        pf = PorosityField(T800, 0.03, void_shape='penny')
        assert pf.void_shape_radii == (3.0, 3.0, 0.3)


class TestFieldOfAnotherLaminate:
    def test_mismatched_plies_raise(self):
        pf = PorosityField(T800, 0.03, distribution='clustered')   # 24 plies
        thin = dataclasses.replace(T800, n_plies=8)
        with pytest.raises(ValueError, match='porosity field was built for a laminate'):
            CompositeMesh(pf, thin, nx=4, ny=2, nz=8)
        thinner_plies = dataclasses.replace(T800, t_ply=T800.t_ply / 2)
        with pytest.raises(ValueError, match='porosity field was built for a laminate'):
            CompositeMesh(pf, thinner_plies, nx=4, ny=2, nz=8)

    def test_same_plies_other_stiffness_is_fine(self):
        stiffer = dataclasses.replace(T800, n_plies=4, E11=2 * T800.E11)
        pf = PorosityField(dataclasses.replace(T800, n_plies=4), 0.03)
        assert CompositeMesh(pf, stiffer, nx=4, ny=2, nz=4).material is stiffer


def test_interface_voids_need_two_plies():
    one = dataclasses.replace(T800, n_plies=1)
    with pytest.raises(ValueError, match="needs at least two plies"):
        PorosityField(one, 0.03, distribution='interface')
    PorosityField(one, 0.03, distribution='uniform')
    PorosityField(dataclasses.replace(T800, n_plies=2), 0.03, distribution='interface')


class TestPercentAsVp:
    def test_local_sensitivities_refuses_a_percent(self):
        mat = dataclasses.replace(T800, n_plies=8)
        pf = PorosityField(mat, 0.03)
        solver = EmpiricalSolver(CompositeMesh(pf, mat, nx=4, ny=2, nz=8), mat)
        with pytest.raises(ValueError, match='Did you pass a percent'):
            solver.local_sensitivities('compression', 'judd_wright', Vp=3.0)
        with pytest.raises(ValueError, match=r'void fraction in \[0, 1\]'):
            solver.local_sensitivities('compression', 'judd_wright', Vp=-0.5)
        assert solver.local_sensitivities('compression', 'judd_wright', Vp=0.03)['KD'] \
            == pytest.approx(np.exp(-6.9 * 0.03))

    @pytest.mark.parametrize('fn', [compute_degraded_clt_moduli,
                                    compute_degraded_clt_flexural_modulus])
    def test_clt_moduli_refuse_a_percent(self, fn):
        angles = [0, 90, 90, 0]
        with pytest.raises(ValueError, match='Did you pass a percent'):
            fn(T800, angles, 3.0)
        with pytest.raises(ValueError, match=r'void fraction in \[0, 1\]'):
            fn(T800, angles, -0.03)
        assert fn(T800, angles, 0.03)


@pytest.mark.parametrize('M_ref', [0.0, 1.0, 2.0])
def test_hygrothermal_factor_is_one_at_the_reference(M_ref):
    """M_ref was validated but never read: the factor at (T_ref, M_ref)
    was 0.92 for M_ref = 1."""
    mat = dataclasses.replace(T800, T_g_dry=180.0, T_ref=23.0, M_ref=M_ref)
    assert mat.environment_knockdown('shear', T=23.0, M=M_ref) == pytest.approx(1.0)
    # Wetter or hotter than the reference still knocks down.
    assert mat.environment_knockdown('shear', T=23.0, M=M_ref + 1.0) < 1.0
    assert mat.environment_knockdown('shear', T=120.0, M=M_ref) < 1.0
