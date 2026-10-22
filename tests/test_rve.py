#!/usr/bin/env python3
"""Tests for the periodic RVE homogenization (IMPROVEMENT_PLAN 3.6, D1).

``porosity_fe.fe._rve`` is an internal verification tool for the
Mori-Tanaka void model. Its tests use cells of at most 12^3 elements (or
12 x 12 x 24), so the module runs in seconds. The 24^3 comparison behind the
table in ``docs/theory/micromechanics.md`` runs only with
``POROSITY_FE_RVE_SWEEP=1`` (about five minutes). Floats are compared with
tolerances throughout.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from porosity_fe import MATERIALS, CompositeMesh, PorosityField
from porosity_fe.fe import _rve
from porosity_fe.fe.assembler import GlobalAssembler
from porosity_fe.homogenization import _degraded_composite_stiffness, _mt_effective_stiffness

MAT = MATERIALS['T800_epoxy']
C_M = MAT.get_isotropic_matrix_stiffness()
FORMULATIONS = ('hex8i', 'hex8')
SPHERE = (1.0, 1.0, 1.0)
#: Oblate spheroid of aspect 5, the penny of the plan's measurements.
PENNY = (1.0, 1.0, 0.2)


def _mt(result, radii):
    """Mori-Tanaka stiffness at the RVE's voxel void fraction."""
    return _mt_effective_stiffness(C_M, result.void_fraction, tuple(radii),
                                   MAT.matrix_poisson)


def _isotropic_moduli(C):
    """Voigt-average bulk and shear moduli (the projection the FE path uses)."""
    K = (np.trace(C[:3, :3]) + 2.0 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    mu = (np.trace(C[:3, :3]) - (C[0, 1] + C[0, 2] + C[1, 2])
          + 3.0 * np.trace(C[3:, 3:])) / 15.0
    return K, mu


# ---------------------------------------------------------------------------
# Periodic cell: consistency
# ---------------------------------------------------------------------------

class TestPeriodicCellConsistency:

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    @pytest.mark.parametrize("n, size", [(4, 1.0), ((3, 5, 2), (1.0, 2.5, 0.4))])
    def test_homogeneous_cell_reproduces_matrix_stiffness(self, formulation, n, size):
        res = _rve.void_cell_stiffness(MAT, None, n=n, size=size, formulation=formulation)
        np.testing.assert_allclose(res.stiffness, C_M, rtol=0, atol=1e-12 * C_M.max())
        np.testing.assert_allclose(res.mean_strain, np.eye(6), rtol=0, atol=1e-12)
        assert res.void_fraction == 0.0
        assert res.volume == pytest.approx(np.prod(np.broadcast_to(size, 3)), rel=1e-12)
        # Every node but the tied images and the pinned corner is unknown.
        assert res.n_reduced == 3 * (np.prod(np.broadcast_to(n, 3)) - 1)

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_homogeneous_periodic_solution_is_the_uniform_strain(self, formulation):
        # Patch test: the periodic fluctuation vanishes, so u = E x exactly
        # and every Gauss point carries the macro strain.
        mesh, matrix, field = _rve.void_unit_cell(MAT, None, n=(3, 4, 5), size=(1.0, 1.3, 0.7))
        res = _rve.homogenize(mesh, matrix, field, formulation=formulation)
        batch = GlobalAssembler(mesh, matrix, field, formulation=formulation).element_batch()
        for j in range(6):
            E = _rve.macro_strain_tensor(np.eye(6)[j])
            u = res.displacements[j]
            np.testing.assert_allclose(u, (mesh.nodes @ E.T).ravel(), rtol=0, atol=1e-13)
            strain = batch.strains(u)
            np.testing.assert_allclose(strain, np.broadcast_to(np.eye(6)[j], strain.shape),
                                       rtol=0, atol=1e-12)

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_anisotropic_porous_ply_is_reproduced(self, formulation):
        # The real composite with smeared porosity, rotated 30 deg so the
        # stiffness couples normal and shear terms: a homogeneous periodic
        # cell must return the element stiffness itself.
        mat = dataclasses.replace(MAT, n_plies=1)
        field = PorosityField(mat, 0.04, distribution='uniform', void_shape='penny')
        mesh = CompositeMesh(field, mat, nx=3, ny=2, nz=2, ply_angles=[30.0])
        mesh.L_x, mesh.L_y = 0.6, 0.4
        mesh.generate_mesh()
        res = _rve.homogenize(mesh, mat, field, formulation=formulation)
        C = GlobalAssembler(mesh, mat, field, formulation=formulation).element_batch().C[0, 0]
        assert abs(C[0, 5]) > 1e3                  # normal-shear coupling present
        np.testing.assert_allclose(res.stiffness, C, rtol=0, atol=1e-11 * C.max())
        ply = _degraded_composite_stiffness(0.04, field.void_shape_radii, mat)
        assert ply[0, 0] > C[0, 0]                 # it really is rotated

    def test_mismatched_face_grids_are_rejected(self):
        mesh, matrix, field = _rve.void_unit_cell(MAT, None, n=3)
        x_max = mesh.nodes_on_face('x_max')
        interior_of_face = [i for i in x_max if 0 < mesh.nodes[i, 1] < 1 and 0 < mesh.nodes[i, 2] < 1]
        mesh.nodes[interior_of_face[0], 1] += 0.01
        with pytest.raises(ValueError, match="do not match"):
            _rve.periodic_node_pairs(mesh)

    def test_void_overlapping_its_images_is_rejected(self):
        with pytest.raises(ValueError, match="half the cell"):
            _rve.void_unit_cell(MAT, (0.5, 0.2, 0.2))
        with pytest.raises(ValueError, match="half the cell"):
            _rve.void_unit_cell(MAT, (0.4, 0.4, 0.3), size=(1.0, 1.0, 0.5))

    def test_void_radii_fill_the_requested_fraction(self):
        r = _rve.void_radii_for(PENNY, 0.05, size=(1.0, 2.0, 0.5))
        assert 4.0 / 3.0 * np.pi * np.prod(r) == pytest.approx(0.05, rel=1e-12)
        assert r[0] == pytest.approx(r[1]) and r[2] == pytest.approx(0.2 * r[0])

    def test_nested_dissection_is_a_permutation(self):
        ijk = np.stack(np.meshgrid(np.arange(7), np.arange(5), np.arange(3),
                                   indexing='ij'), axis=-1).reshape(-1, 3)
        perm = _rve._nested_dissection(np.repeat(ijk, 3, axis=0), leaf=6)
        np.testing.assert_array_equal(np.sort(perm), np.arange(3 * len(ijk)))


@pytest.fixture(scope='module')
def sphere_cells():
    """Spherical void, Vp ~ 0.05, 12^3 cell, per formulation."""
    radii = _rve.void_radii_for(SPHERE, 0.05)
    out = {}
    for formulation in FORMULATIONS:
        mesh, matrix, field = _rve.void_unit_cell(MAT, radii, n=12)
        res = _rve.homogenize(mesh, matrix, field, formulation=formulation)
        out[formulation] = (mesh, matrix, field, res, radii)
    return out


class TestPeriodicVoidCell:

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_effective_stiffness_is_symmetric_and_energy_consistent(self, sphere_cells, formulation):
        mesh, matrix, field, res, _ = sphere_cells[formulation]
        C = res.stiffness
        np.testing.assert_allclose(C, C.T, rtol=0, atol=1e-12 * C.max())
        np.testing.assert_allclose(res.mean_strain, np.eye(6), rtol=0, atol=1e-12)
        # Hill-Mandel: e_i . C e_j = u_i^T K u_j / V.
        K = GlobalAssembler(mesh, matrix, field, formulation=formulation).stiffness()
        U = res.displacements
        energy = U @ (K @ U.T) / res.volume
        np.testing.assert_allclose(energy, C, rtol=0, atol=1e-10 * C.max())

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_pinned_corner_carries_no_force(self, sphere_cells, formulation):
        # The pin removes only the rigid translations: the net force on the
        # corner and its seven periodic images vanishes.
        mesh, matrix, field, res, _ = sphere_cells[formulation]
        K = GlobalAssembler(mesh, matrix, field, formulation=formulation).stiffness()
        red = _rve.periodic_reduction(mesh, np.eye(6)[0])
        for j in range(6):
            r = K @ res.displacements[j]
            for c in range(3):
                dof = red.fixed[c]
                group = np.concatenate([[dof], red.tied[red.masters == dof]])
                assert group.size == 8
                assert abs(r[group].sum()) <= 1e-9 * np.abs(r).max()

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_cubic_symmetry_of_a_centred_sphere(self, sphere_cells, formulation):
        C = sphere_cells[formulation][3].stiffness
        tol = 1e-10 * C.max()
        assert C[0, 0] == pytest.approx(C[1, 1], abs=tol) == pytest.approx(C[2, 2], abs=tol)
        assert C[3, 3] == pytest.approx(C[4, 4], abs=tol) == pytest.approx(C[5, 5], abs=tol)
        assert C[0, 1] == pytest.approx(C[0, 2], abs=tol) == pytest.approx(C[1, 2], abs=tol)
        assert np.abs(C[:3, 3:]).max() <= tol      # no normal-shear coupling

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_sphere_matches_mori_tanaka(self, sphere_cells, formulation):
        # 12^3 cell, voxel void fraction 0.0509. Measured FE / MT:
        # hex8i 0.9976 (C11), 0.9908 (C12), 0.9955 (C44); hex8 1.0001,
        # 0.9941, 0.9969. At 24^3 the normal terms are within 0.2 %, see the
        # micromechanics docs page. The knockdown 1 - C11/Cm is 13 %, so
        # 0.5 % of C11 is about 3 % of the knockdown.
        res, radii = sphere_cells[formulation][3], sphere_cells[formulation][4]
        assert 0.045 < res.void_fraction < 0.055
        C, ref = res.stiffness, _mt(res, radii)
        assert C[0, 0] / ref[0, 0] == pytest.approx(1.0, abs=0.005)
        assert C[0, 1] / ref[0, 1] == pytest.approx(1.0, abs=0.015)
        assert C[3, 3] / ref[3, 3] == pytest.approx(1.0, abs=0.01)
        # Both clearly below the pristine matrix: the void is resolved.
        assert C[0, 0] / C_M[0, 0] < 0.9

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_sphere_at_ten_percent_matches_mori_tanaka(self, formulation):
        # 12^3 cell, voxel void fraction 0.0926. A simple cubic array of
        # voids is not isotropic: its two shear moduli, C44 and
        # (C11 - C12) / 2, split about 2 % to either side of the isotropic
        # Mori-Tanaka value (measured C44 / MT = 0.978 and 0.980,
        # (C11 - C12) / 2 / MT = 1.019 and 1.021 for hex8i and hex8), while
        # the isotropic averages agree within 1.1 %.
        radii = _rve.void_radii_for(SPHERE, 0.10)
        res = _rve.void_cell_stiffness(MAT, radii, n=12, formulation=formulation)
        C, ref = res.stiffness, _mt(res, radii)
        assert 0.09 < res.void_fraction < 0.11
        assert C[0, 0] / ref[0, 0] == pytest.approx(1.0, abs=0.005)
        assert C[0, 1] / ref[0, 1] == pytest.approx(1.0, abs=0.03)
        assert C[3, 3] / ref[3, 3] == pytest.approx(1.0, abs=0.03)
        mu_mt = ref[3, 3]
        assert C[3, 3] < mu_mt < 0.5 * (C[0, 0] - C[0, 1])
        K, mu = _isotropic_moduli(C)
        K_mt, mu_mt = _isotropic_moduli(ref)
        assert K / K_mt == pytest.approx(1.0, abs=0.015)
        assert mu / mu_mt == pytest.approx(1.0, abs=0.01)

    def test_formulations_agree_closely(self, sphere_cells):
        # hex8i relaxes the stiff hex8 bricks around the void: slightly
        # softer, by a fraction of a percent.
        a = sphere_cells['hex8i'][3].stiffness
        b = sphere_cells['hex8'][3].stiffness
        assert np.all(np.diag(a) <= np.diag(b))
        np.testing.assert_allclose(np.diag(a), np.diag(b), rtol=0.005)


class TestPennyVoidKnownLimitation:
    """Crowded oblate voids: where Mori-Tanaka and the periodic cell part ways.

    At Vp ~ 0.10 a penny of aspect 5 spans 98.5 % of the cell width. The
    ligament to its periodic neighbour (1.5 % of the cell) is thinner than an
    element at any affordable resolution, so whether the voxel voids touch
    across it depends on the mesh. Either way the cell is markedly softer out
    of plane (C33, C44) than Mori-Tanaka, which has no notion of void
    spacing: FE / MT for C33 is 0.72 on this 12 x 12 x 24 cell (the voxel
    voids touch), 0.87 at 16^3 (they do not) and 0.81 at 24^3 (they touch);
    see the docs table. Only the sign and rough size of the gap are pinned
    here, not its value.
    """

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_crowded_penny_is_softer_out_of_plane_than_mori_tanaka(self, formulation):
        radii = _rve.void_radii_for(PENNY, 0.10)
        res = _rve.void_cell_stiffness(MAT, radii, n=(12, 12, 24), formulation=formulation)
        C, ref = res.stiffness, _mt(res, radii)
        assert 0.08 < res.void_fraction < 0.12
        # In plane the two still agree within a few percent ...
        assert C[0, 0] / ref[0, 0] == pytest.approx(1.0, abs=0.08)
        assert C[5, 5] / ref[5, 5] == pytest.approx(1.0, abs=0.03)
        # ... out of plane the cell is clearly softer.
        assert 0.6 < C[2, 2] / ref[2, 2] < 0.95
        assert 0.6 < C[3, 3] / ref[3, 3] < 0.95


@pytest.mark.skipif(
    os.environ.get("POROSITY_FE_RVE_SWEEP") != "1",
    reason="set POROSITY_FE_RVE_SWEEP=1 to run the 24^3 RVE vs Mori-Tanaka sweep (~5 min)",
)
class TestRVESweep:
    """The 24^3 cells behind the docs table (plan tolerances for spheres)."""

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    @pytest.mark.parametrize("vp", [0.05, 0.10])
    def test_sphere_24(self, formulation, vp):
        radii = _rve.void_radii_for(SPHERE, vp)
        res = _rve.void_cell_stiffness(MAT, radii, n=24, formulation=formulation)
        C, ref = res.stiffness, _mt(res, radii)
        for i in range(3):
            assert C[i, i] / ref[i, i] == pytest.approx(1.0, abs=0.002)
        for i in range(3, 6):    # cubic-array anisotropy, see the 12^3 test
            assert C[i, i] / ref[i, i] == pytest.approx(1.0, abs=0.025)
        assert C[0, 1] / ref[0, 1] == pytest.approx(1.0, abs=0.02)
        K, mu = _isotropic_moduli(C)
        K_mt, mu_mt = _isotropic_moduli(ref)
        assert K / K_mt == pytest.approx(1.0, abs=0.01)
        assert mu / mu_mt == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("formulation", FORMULATIONS)
    def test_penny_24(self, formulation):
        radii = _rve.void_radii_for(PENNY, 0.05)
        res = _rve.void_cell_stiffness(MAT, radii, n=24, formulation=formulation)
        C, ref = res.stiffness, _mt(res, radii)
        for i in (0, 1, 2, 5):
            assert C[i, i] / ref[i, i] == pytest.approx(1.0, abs=0.015)
        assert C[3, 3] / ref[3, 3] == pytest.approx(1.0, abs=0.04)
        radii = _rve.void_radii_for(PENNY, 0.10)
        res = _rve.void_cell_stiffness(MAT, radii, n=24, formulation=formulation)
        C, ref = res.stiffness, _mt(res, radii)
        assert 0.7 < C[2, 2] / ref[2, 2] < 0.9
        assert 0.7 < C[3, 3] / ref[3, 3] < 0.9
