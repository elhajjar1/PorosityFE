"""Regression tests for the FE orchestration review of November 2026.

A plane-strain fracture specimen under a thermal load, a load that releases
no energy at the crack front, and the lamination-theory flexural modulus of
an unsymmetric layup. The zero-load knockdown (NaN, not 1.0) is checked in
``test_fe_solver.py`` and ``test_fe_flexure.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings

import numpy as np
import pytest

from porosity_fe import (MATERIALS, CompositeMesh, Delamination, FESolver, PorosityField,
                         compute_degraded_clt_flexural_modulus)
from porosity_fe.homogenization import (VOID_SHAPES, _build_clt_abd, _clt_abd_full,
                                        _degraded_composite_stiffness)

# The CTE preset, with the IM7/8552 toughness for the fracture load factor.
MAT = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4,
                          G_Ic=0.21, G_IIc=0.77, bk_eta=2.1)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


def test_thermal_load_on_a_plane_strain_specimen_is_refused():
    """The thermal state was solved free-standing in 3-D, without the
    specimen's u_y = 0: the onset factor of a cross-ply DCB moved from 1.01
    to 0.77 with ny."""
    mat = dataclasses.replace(MAT, n_plies=8)
    pf = PorosityField(mat, 0.0)
    layup = [0, 90, 0, 90, 90, 0, 90, 0]
    mesh = CompositeMesh.for_dcb(pf, mat, crack_length=10.0, length=30.0, width=4.0,
                                 ply_angles=layup, plane_strain=True, ny=2)
    solver = FESolver(mesh, mat, pf, ply_angles=layup)
    with pytest.raises(ValueError, match='plane-strain fracture specimen'):
        solver.solve('dcb', applied_load=5.0, delta_T=-150.0)
    assert np.isfinite(solver.solve('dcb', applied_load=5.0).fracture_load_factor)


class TestNoReleaseIsInfinite:
    """A load that releases no energy gave a finite load factor from
    round-off (1e7 to 1e14) and a meaningless governing node."""

    @staticmethod
    def _mesh(layup, interface):
        pf = PorosityField(MAT, 0.0)
        crack = Delamination.ellipse(interface, (5.0, 3.0), (2.0, 1.5))
        return pf, CompositeMesh(pf, MAT, nx=10, ny=6, nz=4, length=10.0, width=6.0,
                                 ply_angles=layup, delaminations=[crack])

    def test_ud_laminate_under_a_uniform_temperature(self):
        pf, mesh = self._mesh('UD', 1)
        r = FESolver(mesh, MAT, pf, ply_angles='UD').solve('thermal', delta_T=-100.0)
        assert r.fracture_load_factor == np.inf
        assert r.fracture_governing_node is None

    @pytest.mark.parametrize('solver_name', ['direct', 'cg'])
    def test_mid_plane_of_a_symmetric_laminate_in_shear(self, solver_name):
        layup = [0, 90, 90, 0]
        pf, mesh = self._mesh(layup, 1)
        r = FESolver(mesh, MAT, pf, ply_angles=layup).solve('shear', solver=solver_name)
        assert r.fracture_load_factor == np.inf

    def test_a_front_that_opens_keeps_its_factor(self):
        layup = [0, 90, 90, 0]
        pf, mesh = self._mesh(layup, 0)
        r = FESolver(mesh, MAT, pf, ply_angles=layup).solve('tension')
        assert 1.0 < r.fracture_load_factor < 1e3
        assert r.fracture_governing_node is not None


class TestCltFlexuralModulus:
    def test_unsymmetric_layup_uses_the_full_abd_inverse(self):
        """inv(D) alone ignored the bending-stretching coupling: [0_4/90_4]
        read 76.5 GPa where the full inverse (and the FE beam) give 33.5."""
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        layup = [0] * 4 + [90] * 4
        ef = compute_degraded_clt_flexural_modulus(mat, layup, 0.0)['Ef_x']
        A, B, D = _clt_abd_full(mat.get_stiffness_matrix(), layup, mat.t_ply)
        h = 8 * mat.t_ply
        # The bending block of the ABD inverse is the inverse of the Schur
        # complement D - B A^-1 B.
        d_star = np.linalg.inv(D - B @ np.linalg.inv(A) @ B)
        assert ef == pytest.approx(12.0 / (h ** 3 * d_star[0, 0]), rel=1e-10)
        uncoupled = 12.0 / (h ** 3 * np.linalg.inv(D)[0, 0])
        assert ef < 0.6 * uncoupled

    def test_symmetric_layup_is_unchanged(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        layup = [0, 90, 45, -45, -45, 45, 90, 0]
        C = _degraded_composite_stiffness(0.03, VOID_SHAPES['spherical'], mat)
        _, D = _build_clt_abd(mat, layup, C)
        h = 8 * mat.t_ply
        expected = 12.0 / (h ** 3 * np.linalg.inv(D)[0, 0])
        ef = compute_degraded_clt_flexural_modulus(mat, layup, 0.03)['Ef_x']
        assert ef == pytest.approx(expected, rel=1e-12)
