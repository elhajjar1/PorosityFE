#!/usr/bin/env python3
"""Tests for porosity_fe.fe.recovery: Gauss-point to node stress recovery."""

import dataclasses

import numpy as np
import pytest
import scipy.sparse.linalg

from porosity_fe import (MATERIALS, CompositeMesh, FESolver, GlobalAssembler,
                         PorosityField, VoidGeometry, extrapolate_to_nodes,
                         gauss_points_hex)
from porosity_fe.fe.element import Hex8Element
from porosity_fe.fe.recovery import _GP_TO_NODE

MAT = MATERIALS['T800_epoxy']


def _mesh(nx=4, ny=2, nz=4, ply_angles='QI', material=MAT, **pf_kw):
    pf = PorosityField(material, pf_kw.pop('vp', 0.0),
                       distribution=pf_kw.pop('distribution', 'uniform'),
                       **pf_kw)
    return pf, CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz,
                             ply_angles=ply_angles)


def _gauss_point_coords(mesh):
    """Physical coordinates ``(E, 8, 3)`` of every element's Gauss points."""
    points, _ = gauss_points_hex(2)
    N = np.stack([Hex8Element.shape_functions(*p) for p in points])
    return np.einsum('gn,enk->egk', N, mesh.nodes[mesh.elements])


def _linear(xyz, coef):
    """Field ``a + b x + c y + d z`` per component; ``coef`` is ``(4, k)``."""
    return coef[0] + xyz @ coef[1:]


class TestExtrapolationOperator:
    def test_inverse_of_shape_functions_at_gauss_points(self):
        points, _ = gauss_points_hex(2)
        N_gp = np.stack([Hex8Element.shape_functions(*p) for p in points])
        np.testing.assert_allclose(N_gp @ _GP_TO_NODE, np.eye(8), atol=1e-14)

    def test_constant_field_maps_to_itself(self):
        # Rows sum to 1: a constant is recovered exactly.
        np.testing.assert_allclose(_GP_TO_NODE.sum(axis=1), 1.0, atol=1e-14)

    def test_linear_field_is_recovered_exactly(self):
        _, mesh = _mesh(nx=3, ny=2, nz=3)
        coef = np.array([[1.0, -2.0], [0.3, 0.0], [-0.7, 1.1], [5.0, -4.0]])
        field = _linear(_gauss_point_coords(mesh), coef)
        nodal, corner = extrapolate_to_nodes(field, mesh, average='all')
        np.testing.assert_allclose(nodal, _linear(mesh.nodes, coef),
                                   rtol=0, atol=1e-11)
        np.testing.assert_allclose(
            corner, _linear(mesh.nodes[mesh.elements], coef), rtol=0, atol=1e-11)

    def test_scalar_input_returns_scalar_shapes(self):
        _, mesh = _mesh(nx=2, ny=2, nz=2)
        field = _gauss_point_coords(mesh)[:, :, 2]
        nodal, corner = extrapolate_to_nodes(field, mesh)
        assert nodal.shape == (mesh.n_nodes,)
        assert corner.shape == (mesh.n_elements, 8)
        np.testing.assert_allclose(nodal, mesh.nodes[:, 2], atol=1e-12)

    @pytest.mark.parametrize('bad', [
        lambda m: np.zeros((m.n_elements, 4, 6)),
        lambda m: np.zeros((m.n_elements + 1, 8, 6)),
        lambda m: np.zeros(m.n_elements),
    ])
    def test_shape_mismatch_raises(self, bad):
        _, mesh = _mesh(nx=2, ny=2, nz=2)
        with pytest.raises(ValueError, match='field_gp must have shape'):
            extrapolate_to_nodes(bad(mesh), mesh)

    def test_unknown_average_raises(self):
        _, mesh = _mesh(nx=2, ny=2, nz=2)
        with pytest.raises(ValueError, match='average'):
            extrapolate_to_nodes(np.zeros((mesh.n_elements, 8)), mesh,
                                 average='mean')


class TestPlyAwareAveraging:
    """A field that is linear inside each ply but jumps between plies."""

    def _jump_field(self, mesh):
        xyz = _gauss_point_coords(mesh)
        ply = np.asarray(mesh.elem_ply_ids)[:, None, None]
        # Different slope and offset per ply: discontinuous at interfaces.
        return 10.0 * ply + (1.0 + ply) * xyz[:, :, :1] - 0.5 * xyz[:, :, 2:3]

    def _exact(self, mesh, ply, xyz):
        return 10.0 * ply + (1.0 + ply) * xyz[..., 0] - 0.5 * xyz[..., 2]

    def test_ply_average_keeps_each_ply_exact(self):
        _, mesh = _mesh(nx=3, ny=2, nz=4, ply_angles='QI')
        assert len(np.unique(mesh.elem_ply_ids)) == 4
        field = self._jump_field(mesh)
        nodal, corner = extrapolate_to_nodes(field, mesh, average='ply')
        ply = np.asarray(mesh.elem_ply_ids)[:, None]
        np.testing.assert_allclose(
            corner[:, :, 0], self._exact(mesh, ply, mesh.nodes[mesh.elements]),
            atol=1e-10)
        # A node on a ply interface reports the ply above it.
        z_nodes = np.unique(mesh.nodes[:, 2])
        above = np.empty(mesh.n_nodes, dtype=int)
        layer = np.searchsorted(z_nodes, mesh.nodes[:, 2])
        layer_ply = np.asarray(mesh.elem_ply_ids).reshape(mesh.nz, -1)[:, 0]
        above[:] = layer_ply[np.minimum(layer, mesh.nz - 1)]
        np.testing.assert_allclose(
            nodal[:, 0], self._exact(mesh, above, mesh.nodes), atol=1e-10)

    def test_all_average_smears_interfaces(self):
        _, mesh = _mesh(nx=3, ny=2, nz=4, ply_angles='QI')
        field = self._jump_field(mesh)
        _, corner_ply = extrapolate_to_nodes(field, mesh, average='ply')
        _, corner_all = extrapolate_to_nodes(field, mesh, average='all')
        # Interior element layers touch interfaces on both faces.
        jump = np.abs(corner_all - corner_ply).max()
        assert jump > 4.0  # half the 10-unit offset between neighbouring plies

    def test_none_returns_raw_corners_and_ply_nodal(self):
        _, mesh = _mesh(nx=3, ny=2, nz=4, ply_angles='QI')
        rng = np.random.default_rng(3)
        field = rng.normal(size=(mesh.n_elements, 8, 2))
        nodal_n, corner_n = extrapolate_to_nodes(field, mesh, average='none')
        nodal_p, corner_p = extrapolate_to_nodes(field, mesh, average='ply')
        np.testing.assert_allclose(
            corner_n, np.einsum('ng,egk->enk', _GP_TO_NODE, field), atol=1e-13)
        assert not np.allclose(corner_n, corner_p)
        np.testing.assert_array_equal(nodal_n, nodal_p)
        # 'ply' corners are the group means of the raw corners.
        node = mesh.elements[0, 6]
        same = (mesh.elements == node) & \
            (np.asarray(mesh.elem_ply_ids)[:, None] == mesh.elem_ply_ids[0])
        np.testing.assert_allclose(corner_p[0, 6], corner_n[same].mean(axis=0),
                                   atol=1e-13)

    def test_void_elements_do_not_pollute_solid_nodes(self):
        _, mesh = _mesh(nx=10, ny=4, nz=6, ply_angles='UD', discrete_voids=[
            VoidGeometry(center=(25.0, 10.0, 2.2), radii=(6.0, 4.0, 1.0))])
        void = np.asarray(mesh.void_elements)
        assert void.size > 0
        field = np.ones((mesh.n_elements, 8))
        field[void] = 0.0
        nodal_p, corner_p = extrapolate_to_nodes(field, mesh, average='ply')
        nodal_a, _ = extrapolate_to_nodes(field, mesh, average='all')
        boundary = np.intersect1d(
            mesh.elements[void].ravel(),
            np.delete(mesh.elements, void, axis=0).ravel())
        assert boundary.size > 0
        np.testing.assert_allclose(nodal_p[boundary], 1.0)
        assert np.all(nodal_a[boundary] < 1.0)
        np.testing.assert_allclose(corner_p[void], 0.0, atol=1e-13)


def _pure_bending(material, nx, ny, nz, theta=1e-3):
    """UD beam under a prescribed end rotation (uniform moment).

    ``u_x = 0`` on ``x = 0``, ``u_x = -theta (z - h/2)`` on ``x = L``,
    ``u_z = 0`` along the ``x = 0`` neutral line, one ``u_y`` fixed;
    exact Dirichlet elimination. Returns the mesh, Gauss-point stress and
    curvature.
    """
    pf, mesh = _mesh(nx, ny, nz, ply_angles='UD', material=material)
    asm = GlobalAssembler(mesh, material, pf)
    K = asm.stiffness().tocsr()
    x, z = mesh.nodes[:, 0], mesh.nodes[:, 2]
    tol = 1e-9
    con = {}
    for n in np.flatnonzero(np.abs(x) < tol):
        con[3 * n] = 0.0
    for n in np.flatnonzero(np.abs(x - mesh.L_x) < tol):
        con[3 * n] = -theta * (z[n] - mesh.L_z / 2)
    axis = np.flatnonzero((np.abs(x) < tol) & (np.abs(z - mesh.L_z / 2) < tol))
    for n in axis:
        con[3 * n + 2] = 0.0
    con[3 * axis[0] + 1] = 0.0
    fixed = np.fromiter(con.keys(), np.intp)
    values = np.fromiter(con.values(), float)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    u = np.zeros(mesh.n_dof)
    u[fixed] = values
    u[free] = scipy.sparse.linalg.spsolve(K[free][:, free].tocsc(),
                                          -K[free][:, fixed] @ values)
    batch = asm.element_batch()
    return mesh, batch.stresses(batch.strains(u)), theta / mesh.L_x


class TestPureBendingSurfaceStress:
    """Closed form: surface ``sigma_xx = E11 * kappa * h / 2`` in pure bending."""

    @pytest.mark.parametrize('nz, mean_ratio', [(2, 0.504), (4, 0.756)])
    def test_nodal_surface_stress_matches_beam_theory(self, nz, mean_ratio):
        ud = dataclasses.replace(MAT, n_plies=4, t_ply=0.5)
        mesh, stress, kappa = _pure_bending(ud, 16, 4, nz)
        exact = ud.E11 * kappa * mesh.L_z / 2
        centroid = mesh.nodes[mesh.elements].mean(axis=1)
        mid = (centroid[:, 0] > 0.3 * mesh.L_x) & (centroid[:, 0] < 0.7 * mesh.L_x)
        element_mean = np.abs(stress[mid, :, 0].mean(axis=1)).max() / exact
        gp_max = np.abs(stress[mid, :, 0]).max() / exact

        nodal, _ = extrapolate_to_nodes(stress, mesh)
        top = (np.abs(mesh.nodes[:, 2] - mesh.L_z) < 1e-9) \
            & (mesh.nodes[:, 0] > 0.3 * mesh.L_x) & (mesh.nodes[:, 0] < 0.7 * mesh.L_x)
        bottom = (np.abs(mesh.nodes[:, 2]) < 1e-9) \
            & (mesh.nodes[:, 0] > 0.3 * mesh.L_x) & (mesh.nodes[:, 0] < 0.7 * mesh.L_x)
        top_ratio = nodal[top, 0] / exact
        bottom_ratio = nodal[bottom, 0] / exact

        # Element means (what to_vtk writes) and Gauss points under-read
        # the surface stress; the recovered nodal value is within 3 %.
        assert element_mean == pytest.approx(mean_ratio, abs=0.01)
        assert gp_max < 0.95
        assert np.all(np.abs(np.abs(top_ratio) - 1.0) < 0.03)
        assert np.all(np.abs(np.abs(bottom_ratio) - 1.0) < 0.03)
        # u_x(L) = -theta (z - h/2) shortens the top fibre: compression on
        # top, tension at the bottom.
        assert np.all(top_ratio < 0) and np.all(bottom_ratio > 0)


class TestFieldResultsNodal:
    def _solve(self, ply_angles=(0.0, 90.0, 90.0, 0.0)):
        mat = dataclasses.replace(MAT, n_plies=4, t_ply=0.5)
        pf, mesh = _mesh(nx=8, ny=4, nz=4, ply_angles=list(ply_angles),
                         material=mat)
        return mesh, FESolver(mesh, mat, pf).solve('compression')

    def test_matches_extrapolate_to_nodes(self):
        mesh, r = self._solve()
        for method, glob, loc in (
                (r.nodal_stress, r.stress_global, r.stress_local),
                (r.nodal_strain, r.strain_global, r.strain_local)):
            nodal, corner = method(mesh)
            ref_n, ref_c = extrapolate_to_nodes(glob, mesh)
            np.testing.assert_array_equal(nodal, ref_n)
            np.testing.assert_array_equal(corner, ref_c)
            nodal_l, _ = method(mesh, frame='local')
            np.testing.assert_array_equal(
                nodal_l, extrapolate_to_nodes(loc, mesh)[0])
            assert nodal.shape == (mesh.n_nodes, 6)
            assert corner.shape == (mesh.n_elements, 8, 6)

    def test_cross_ply_interface_jump_is_preserved(self):
        # Uniform compression of a [0/90]s laminate: sigma_xx is ~E11 eps in
        # the 0 deg plies and ~E22 eps in the 90 deg plies. Ply-aware
        # corners stay at their own ply's level; 'all' mixes them.
        mesh, r = self._solve()
        _, corner = r.nodal_stress(mesh)
        _, corner_all = r.nodal_stress(mesh, average='all')
        centroid = mesh.nodes[mesh.elements].mean(axis=1)
        inner = (centroid[:, 0] > 0.25 * mesh.L_x) & (centroid[:, 0] < 0.75 * mesh.L_x) \
            & (centroid[:, 1] > 0.2 * mesh.L_y) & (centroid[:, 1] < 0.8 * mesh.L_y)
        zero = inner & (np.asarray(mesh.ply_angles) == 0.0)
        ninety = inner & (np.asarray(mesh.ply_angles) == 90.0)
        s0 = r.stress_global[zero, :, 0].mean()
        s90 = r.stress_global[ninety, :, 0].mean()
        assert abs(s0) > 5 * abs(s90)
        np.testing.assert_allclose(corner[zero, :, 0], s0, rtol=0.02)
        np.testing.assert_allclose(corner[ninety, :, 0], s90, rtol=0.05)
        assert np.abs(corner_all[ninety, :, 0] - s90).max() > abs(s0 - s90) / 4

    def test_local_frame_all_average_rejected_on_multi_angle_laminate(self):
        mesh, r = self._solve()
        with pytest.raises(ValueError, match="average='all'"):
            r.nodal_stress(mesh, frame='local', average='all')
        mesh_ud, r_ud = self._solve(ply_angles=(0.0,) * 4)
        r_ud.nodal_stress(mesh_ud, frame='local', average='all')  # allowed

    def test_unknown_frame_raises(self):
        mesh, r = self._solve()
        with pytest.raises(ValueError, match='frame'):
            r.nodal_strain(mesh, frame='material')
