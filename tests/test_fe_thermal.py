#!/usr/bin/env python3
"""Thermal / cure residual-stress load case (IMPROVEMENT_PLAN 3.5).

Checks the thermal load vector against exact solutions (free expansion,
uniform CTE, an uncondensed incompatible-mode reference), the residual
stresses of free-standing symmetric laminates against closed-form CLT,
the 3-2-1 supports, superposition with the mechanical modes, the
first-ply-failure factor with a residual pre-stress against a bisection,
the guards, and the exports. Numbers that depend on the element
formulation pass ``formulation=`` explicitly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
from porosity_fe.fe import failure
from porosity_fe.fe.batch import (
    assemble_load_vector,
    build_element_batch,
    global_cte,
    thermal_stress_moduli,
)
from porosity_fe.homogenization import _degraded_composite_stiffness
from porosity_fe.transforms import (
    rotate_stiffness_3d,
    strain_transformation_3d,
    stress_transformation_3d,
)

FORMULATIONS = ('hex8', 'hex8i')

#: Representative T800/epoxy lamina CTEs (1/K) used by the IMPROVEMENT_PLAN
#: 3.5 prototype; a method check, not a preset value.
ALPHA_1, ALPHA_2 = -0.1e-6, 31.0e-6
DT = -150.0
CROSS_PLY = [0.0, 90.0, 0.0, 90.0, 90.0, 0.0, 90.0, 0.0]     # [0/90]_2s
ANGLE_PLY = [45.0, -45.0, -45.0, 45.0]                      # [+-45]_s
QI_8 = [0.0, 90.0, 45.0, -45.0, -45.0, 45.0, 90.0, 0.0]     # [0/90/45/-45]_s
IDX = [0, 1, 5]


def _material(layup, t_ply=0.125, **overrides):
    fields = dict(n_plies=len(layup), t_ply=t_ply, alpha_1=ALPHA_1, alpha_2=ALPHA_2)
    fields.update(overrides)
    return dataclasses.replace(MATERIALS['T800_epoxy'], **fields)


def _solver(layup, nx, ny, *, formulation, vp=0.0, nz_per_ply=1,
            material=None, **pf_kwargs):
    mat = material if material is not None else _material(layup)
    pf = PorosityField(mat, vp, **pf_kwargs)
    mesh = CompositeMesh(pf, mat, nx=nx, ny=ny, nz=nz_per_ply * len(layup),
                         ply_angles=list(layup))
    return FESolver(mesh, mat, pf, ply_angles=list(layup),
                    formulation=formulation)


def _clt(material, layup, dT, vp=0.0, radii=(1.0, 1.0, 1.0)):
    """Plane-stress CLT of a symmetric laminate: mid-plane strain and ply stresses.

    Returns ``eps0`` ``[e_x, e_y, g_xy]`` and a list of ply-local
    ``[s11, s22, t12]`` per ply.
    """
    C = _degraded_composite_stiffness(vp, radii, material)
    alpha = material.cte_vector()
    A = np.zeros((3, 3))
    N = np.zeros(3)
    plies = []
    for ang in layup:
        Cg = rotate_stiffness_3d(C, np.radians(ang), axis='z') if ang else C
        Q = np.array([[Cg[i, j] - Cg[i, 2] * Cg[j, 2] / Cg[2, 2] for j in IDX]
                      for i in IDX])
        a = global_cte(alpha, ang)[IDX]
        A += Q
        N += Q @ a * dT
        plies.append((ang, Q, a))
    eps0 = np.linalg.solve(A, N)
    out = []
    for ang, Q, a in plies:
        s = Q @ (eps0 - a * dT)
        s6 = np.array([s[0], s[1], 0.0, 0.0, 0.0, s[2]])
        out.append((stress_transformation_3d(np.radians(ang), axis='z') @ s6)[IDX])
    return eps0, out


def _interior(mesh, frac=0.3):
    cent = mesh.nodes[mesh.elements].mean(axis=1)
    return ((cent[:, 0] > frac * mesh.L_x) & (cent[:, 0] < (1 - frac) * mesh.L_x)
            & (cent[:, 1] > frac * mesh.L_y) & (cent[:, 1] < (1 - frac) * mesh.L_y))


def _ply_mean_error(solver, result, layup, vp=0.0):
    """Worst ply-mean interior error against CLT, relative to max |CLT|."""
    mesh = solver.mesh
    _, ref = _clt(solver.material, layup, result.delta_T, vp,
                  solver.porosity_field.void_shape_radii)
    m = _interior(mesh)
    worst = 0.0
    for k, r in enumerate(ref):
        fe = result.stress_local[m & (mesh.elem_ply_ids == k)].reshape(-1, 6)
        worst = max(worst, np.abs(fe.mean(axis=0)[IDX] - r).max() / np.abs(r).max())
    return worst


@pytest.fixture(scope='module', params=FORMULATIONS)
def cross_ply(request):
    """[0/90]_2s, 50 x 20 mm, 40 x 16 x 8 (one element per ply), dT = -150 K."""
    solver = _solver(CROSS_PLY, 40, 16, formulation=request.param)
    return solver, solver.solve('thermal', delta_T=DT)


# --------------------------------------------------------------------------
# Thermal load vector: exact solutions
# --------------------------------------------------------------------------

class TestThermalLoadVector:
    def test_global_cte_matches_textbook_rotation(self):
        alpha = np.array([ALPHA_1, ALPHA_2, 29e-6, 0.0, 0.0, 0.0])
        for theta in (30.0, 45.0, -60.0, 90.0):
            c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
            expected = [ALPHA_1 * c * c + ALPHA_2 * s * s,
                        ALPHA_1 * s * s + ALPHA_2 * c * c, 29e-6, 0.0, 0.0,
                        2.0 * (ALPHA_1 - ALPHA_2) * s * c]
            np.testing.assert_allclose(global_cte(alpha, theta), expected,
                                       rtol=0, atol=1e-18)
            # Rotating the global vector back gives the ply vector.
            T_eps = strain_transformation_3d(np.radians(theta), axis='z')
            np.testing.assert_allclose(T_eps @ global_cte(alpha, theta), alpha,
                                       rtol=0, atol=1e-18)

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_free_expansion_patch_test_on_distorted_mesh(self, formulation):
        """Homogeneous isotropic block, uniform dT, free: zero stress, u = a dT x."""
        E, nu, a = 70000.0, 0.3, 20e-6
        G = E / (2 * (1 + nu))
        layup = [0.0, 90.0, 45.0, -30.0]
        mat = _material(layup, t_ply=0.5, E11=E, E22=E, E33=E, G12=G, G13=G,
                        G23=G, nu12=nu, nu13=nu, nu23=nu, alpha_1=a, alpha_2=a)
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=4, ny=3, nz=4, ply_angles=layup)
        mesh.L_x, mesh.L_y = 6.0, 4.0
        mesh.generate_mesh()
        nodes = mesh.nodes
        lo, hi = nodes.min(axis=0), nodes.max(axis=0)
        inner = np.all((nodes > lo + 1e-9) & (nodes < hi - 1e-9), axis=1)
        h = np.array([6.0 / 4, 4.0 / 3, 2.0 / 4])
        rng = np.random.default_rng(0)
        nodes[inner] += rng.uniform(-0.25, 0.25, size=(int(inner.sum()), 3)) * h
        assert inner.sum() == 18
        solver = FESolver(mesh, mat, pf, ply_angles=layup, formulation=formulation)
        r = solver.solve('thermal', delta_T=-100.0)
        np.testing.assert_allclose(r.displacement, -100.0 * a * mesh.nodes,
                                   rtol=0, atol=1e-13)
        assert np.abs(r.stress_global).max() < 1e-8
        free_strain = -100.0 * a * np.array([1, 1, 1, 0, 0, 0])
        np.testing.assert_allclose(r.strain_global,
                                   np.broadcast_to(free_strain, r.strain_global.shape),
                                   rtol=0, atol=1e-14)
        assert np.abs(r.reaction_forces).max() < 1e-9

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_uniform_cte_on_any_layup_is_stress_free(self, formulation):
        mat = _material(QI_8, alpha_1=20e-6, alpha_2=20e-6)
        solver = _solver(QI_8, 6, 3, formulation=formulation, material=mat)
        r = solver.solve('thermal', delta_T=DT)
        assert np.abs(r.stress_local).max() < 1e-6
        assert r.max_failure_index < 1e-12

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_unidirectional_off_axis_laminate_is_stress_free(self, formulation):
        layup = [30.0] * 4
        solver = _solver(layup, 6, 3, formulation=formulation)
        r = solver.solve('thermal', delta_T=DT)
        assert np.abs(r.stress_local).max() < 1e-6
        # ... and its total strain is the free thermal strain, rotated.
        expected = global_cte(solver.material.cte_vector(), 30.0) * DT
        np.testing.assert_allclose(
            r.strain_global, np.broadcast_to(expected, r.strain_global.shape),
            rtol=0, atol=1e-12)

    def test_hex8i_matches_uncondensed_reference_with_graded_porosity(self):
        """Condensed load f_u + H^T f_a and strain B_eff u + G Kaa^-1 f_a are exact.

        Solves the same problem with the nine internal modes of every
        element kept as unknowns and compares; graded porosity makes the
        stiffness vary inside elements, so the internal modes carry load.
        """
        layup = [0.0, 45.0, 90.0, -45.0]
        mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4, t_ply=0.5)
        pf = PorosityField(mat, 0.05, distribution='clustered',
                           cluster_location='surface')
        mesh = CompositeMesh(pf, mat, nx=4, ny=3, nz=4, ply_angles=layup)
        mesh.L_x, mesh.L_y = 8.0, 6.0
        mesh.generate_mesh()
        solver = FESolver(mesh, mat, pf, ply_angles=layup, formulation='hex8i')
        dT = -120.0
        r = solver.solve('thermal', delta_T=dT)

        b_i = solver.assembler.element_batch()
        b_0 = build_element_batch(mesh, mat, pf.void_shape_radii, formulation='hex8')
        beta = thermal_stress_moduli(b_0, mesh, mat.cte_vector()) * dT
        n_el, nd = len(mesh.elements), mesh.n_dof
        BG = np.concatenate([b_0.B, b_i.G_inc], axis=3)              # (E, 8, 6, 33)
        Ke = np.einsum('egki,egkl,eglj,eg->eij', BG, b_0.C, BG, b_0.detJ_w)
        fe = np.einsum('egki,egk,eg->ei', BG, beta, b_0.detJ_w)
        dofs = np.concatenate(
            [b_0.dofs, nd + 9 * np.arange(n_el)[:, None] + np.arange(9)], axis=1)
        n = nd + 9 * n_el
        K = scipy.sparse.coo_matrix(
            (Ke.ravel(), (np.broadcast_to(dofs[:, :, None], Ke.shape).ravel(),
                          np.broadcast_to(dofs[:, None, :], Ke.shape).ravel())),
            shape=(n, n)).tocsc()
        F = np.bincount(dofs.ravel(), weights=fe.ravel(), minlength=n)
        fixed = np.array(sorted(solver.bc_handler.free_bcs()[0]))
        free = np.setdiff1d(np.arange(n), fixed)
        x = np.zeros(n)
        x[free] = scipy.sparse.linalg.spsolve(K[free][:, free], F[free])
        u, amp = x[:nd], x[nd:].reshape(n_el, 9)
        eps = (np.einsum('egij,ej->egi', b_0.B, u[b_0.dofs])
               + np.einsum('egij,ej->egi', b_i.G_inc, amp))
        sig = np.einsum('egij,egj->egi', b_0.C, eps) - beta

        np.testing.assert_allclose(r.displacement.ravel(), u, rtol=0,
                                   atol=1e-11 * np.abs(u).max())
        np.testing.assert_allclose(r.stress_global, sig, rtol=0,
                                   atol=1e-10 * np.abs(sig).max())
        np.testing.assert_allclose(r.strain_global, eps, rtol=0,
                                   atol=1e-10 * np.abs(eps).max())
        # The internal-mode strain driven directly by the load matters here.
        corr = b_i.incompatible_mode_strains(beta)
        assert np.abs(np.einsum('egij,egj->egi', b_0.C, corr)).max() \
            > 1e-2 * np.abs(sig).max()

    def test_hex8_has_no_internal_mode_strain(self):
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8')
        solver.solve('thermal', delta_T=DT)
        batch = solver.assembler.element_batch()
        assert batch.G_inc is None and batch.Kaa is None
        beta = np.ones(batch.C.shape[:3])
        assert not batch.incompatible_mode_strains(beta).any()

    def test_void_elements_carry_no_thermal_stress_modulus(self):
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8')
        mesh = solver.mesh
        mesh.void_elements = np.array([3, 10])
        batch = build_element_batch(mesh, solver.material, (1.0, 1.0, 1.0))
        beta = thermal_stress_moduli(batch, mesh, solver.material.cte_vector())
        assert not beta[[3, 10]].any()
        assert np.abs(beta[0]).max() > 0.0


# --------------------------------------------------------------------------
# Residual stresses against classical lamination theory
# --------------------------------------------------------------------------

class TestAgainstCLT:
    def test_cross_ply_analytic_value(self):
        """|dT| (a2 - a1)(Q11 Q22 - Q12^2) / (Q11 + Q22 + 2 Q12) = 47.573 MPa."""
        mat = _material(CROSS_PLY)
        C = mat.get_stiffness_matrix()
        Q = np.array([[C[i, j] - C[i, 2] * C[j, 2] / C[2, 2] for j in (0, 1)]
                      for i in (0, 1)])
        analytic = abs(DT) * (ALPHA_2 - ALPHA_1) * (Q[0, 0] * Q[1, 1] - Q[0, 1] ** 2) \
            / (Q[0, 0] + Q[1, 1] + 2 * Q[0, 1])
        assert analytic == pytest.approx(47.573, abs=5e-4)
        # CLT agrees in every ply: fiber direction in compression, transverse
        # in tension, after a cool-down.
        _, ref = _clt(mat, CROSS_PLY, DT)
        for r in ref:
            np.testing.assert_allclose(r, [-analytic, analytic, 0.0], atol=1e-9)

    def test_cross_ply_interior_matches_clt(self, cross_ply):
        solver, r = cross_ply
        mesh = solver.mesh
        assert _ply_mean_error(solver, r, CROSS_PLY) < 5e-3
        m = _interior(mesh)
        s = r.stress_local[m]
        assert (s[..., 1] > 0).all() and (s[..., 0] < 0).all()
        _, ref = _clt(solver.material, CROSS_PLY, DT)
        if solver.formulation == 'hex8':
            # One element per ply already reproduces CLT to 0.1 % pointwise.
            np.testing.assert_allclose(s[..., 1], ref[0][1], rtol=1e-3)
            assert np.abs(s[..., 2:5]).max() < 1e-3 * ref[0][1]

    def test_mid_plane_strain_matches_clt(self, cross_ply):
        solver, r = cross_ply
        eps0, _ = _clt(solver.material, CROSS_PLY, DT)
        e = r.strain_global[_interior(solver.mesh)].reshape(-1, 6).mean(axis=0)
        # hex8i carries a slowly decaying, element-to-element oscillation in
        # from the free edges on this coarse in-plane mesh (dx / t_ply = 10);
        # it shrinks with dx (0.3 % ply-mean stress error here, 0.14 % at
        # half the element size), hex8 does not show it.
        tol = 2e-3 if solver.formulation == 'hex8' else 1e-2
        np.testing.assert_allclose(e[[0, 1, 5]], eps0, rtol=0,
                                   atol=tol * np.abs(eps0).max())

    def test_reactions_vanish_at_statically_determinate_supports(self, cross_ply):
        solver, r = cross_ply
        batch = solver.assembler.element_batch()
        beta = thermal_stress_moduli(batch, solver.mesh, solver.material.cte_vector())
        F = assemble_load_vector(batch, batch.thermal_loads(beta) * DT,
                                 solver.mesh.n_dof)
        assert np.abs(F).max() > 1.0                       # N; a real load
        assert np.abs(r.reaction_forces).max() < 1e-6 * np.abs(F).max()
        assert np.count_nonzero(np.abs(r.reaction_forces) > 0) <= 6

    def test_free_edge_peak_is_reported_next_to_interior(self, cross_ply):
        solver, r = cross_ply
        assert r.interior_max_failure_index is not None
        assert r.interior_max_failure_index < r.max_failure_index
        assert r.interior_first_ply_failure_load_factor > r.first_ply_failure_load_factor
        # Interlaminar stresses concentrate at the free edges.
        m = _interior(solver.mesh)
        assert np.abs(r.stress_local[..., 2]).max() \
            > 10 * np.abs(r.stress_local[m][..., 2]).max()

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_angle_ply_matches_clt(self, formulation):
        """[+-45]_s exercises the engineering-shear CTE term a_xy = 2 (a1 - a2) s c."""
        solver = _solver(ANGLE_PLY, 20, 8, formulation=formulation)
        r = solver.solve('thermal', delta_T=DT)
        assert _ply_mean_error(solver, r, ANGLE_PLY) < 5e-3
        # In-plane shear stress in the ply axes is zero in CLT; a wrong sign
        # or factor on a_xy would put shear here.
        m = _interior(solver.mesh)
        assert np.abs(r.stress_local[m][..., 5]).max() < 0.01 * 47.573

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_quasi_isotropic_matches_clt(self, formulation):
        solver = _solver(QI_8, 20, 8, formulation=formulation)
        r = solver.solve('thermal', delta_T=DT)
        assert _ply_mean_error(solver, r, QI_8) < 1.5e-2

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_unsymmetric_laminate_curvature(self, formulation):
        """[0_4/90_4] warps on cool-down; hex8 locks in bending, hex8i does not."""
        layup = [0.0] * 4 + [90.0] * 4
        solver = _solver(layup, 20, 8, formulation=formulation)
        r = solver.solve('thermal', delta_T=DT)
        mat, t, h = solver.material, 0.125, 1.0
        C = mat.get_stiffness_matrix()
        ABD = np.zeros((6, 6))
        NM = np.zeros(6)
        for k, ang in enumerate(layup):
            z0, z1 = -h / 2 + k * t, -h / 2 + (k + 1) * t
            Cg = rotate_stiffness_3d(C, np.radians(ang), axis='z') if ang else C
            Q = np.array([[Cg[i, j] - Cg[i, 2] * Cg[j, 2] / Cg[2, 2] for j in IDX]
                          for i in IDX])
            a = global_cte(mat.cte_vector(), ang)[IDX] * DT
            for blk, w in ((slice(0, 3), z1 - z0), (slice(3, 6), (z1**3 - z0**3) / 3)):
                ABD[blk, blk] += Q * w
            ABD[:3, 3:] += Q * (z1**2 - z0**2) / 2
            ABD[3:, :3] += Q * (z1**2 - z0**2) / 2
            NM[:3] += Q @ a * (z1 - z0)
            NM[3:] += Q @ a * (z1**2 - z0**2) / 2
        kappa = np.linalg.solve(ABD, NM)[3:]
        n, w = solver.mesh.nodes, r.displacement[:, 2]
        sel = ((np.abs(n[:, 2] - h / 2) < 1e-9) & (n[:, 0] > 10) & (n[:, 0] < 40)
               & (n[:, 1] > 4) & (n[:, 1] < 16))
        x, y = n[sel, 0], n[sel, 1]
        fit = np.linalg.lstsq(np.column_stack(
            [np.ones_like(x), x, y, -0.5 * x**2, -0.5 * y**2, -x * y]), w[sel],
            rcond=None)[0]
        ratio = fit[3:5] / kappa[:2]
        if formulation == 'hex8i':
            np.testing.assert_allclose(ratio, 1.0, atol=0.01)
        else:
            assert (ratio < 0.8).all()          # dx / h = 2.5: locked

    @pytest.mark.parametrize('formulation', FORMULATIONS)
    def test_porous_cross_ply_matches_porous_clt(self, formulation):
        """Porosity enters through the degraded stiffness only (CTE held pristine)."""
        s22 = []
        for vp in (0.0, 0.02, 0.05):
            solver = _solver(CROSS_PLY, 20, 8, formulation=formulation, vp=vp)
            r = solver.solve('thermal', delta_T=DT)
            assert _ply_mean_error(solver, r, CROSS_PLY, vp) < 1e-2
            s22.append(r.stress_local[_interior(solver.mesh)][..., 1].mean())
        assert s22[0] > s22[1] > s22[2] > 0.9 * s22[0]


# --------------------------------------------------------------------------
# Thermal-only results
# --------------------------------------------------------------------------

class TestThermalOnlyMode:
    def test_result_fields(self, cross_ply):
        solver, r = cross_ply
        assert np.isnan(r.knockdown)
        assert r.effective_modulus is None
        assert r.delta_T == DT
        assert r.load_factor_basis == 'delta_T'
        assert r.cte_local == (ALPHA_1, ALPHA_2, ALPHA_2)
        np.testing.assert_array_equal(r.residual_stress_local, r.stress_local)
        assert r.residual_max_failure_index == r.max_failure_index
        assert 'nan' in repr(r)
        summary = r.summary()
        assert np.isnan(summary.knockdown)
        assert summary.details['delta_T'] == DT
        assert summary.details['load_factor_basis'] == 'delta_T'

    def test_linear_in_delta_T(self, cross_ply):
        solver, r = cross_ply
        r2 = solver.solve('thermal', delta_T=2 * DT)
        np.testing.assert_allclose(r2.stress_global, 2 * r.stress_global,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r2.displacement, 2 * r.displacement,
                                   rtol=1e-12, atol=1e-16)
        # The load factor multiplies delta_T: the critical change is fixed.
        assert r2.first_ply_failure_load_factor * 2 * DT == pytest.approx(
            r.first_ply_failure_load_factor * DT, rel=1e-9)
        # A heat-up reverses every stress.
        r_up = solver.solve('thermal', delta_T=-DT)
        np.testing.assert_allclose(r_up.stress_global, -r.stress_global,
                                   rtol=1e-12, atol=1e-12)

    def test_first_ply_failure_factor_is_critical_temperature_change(self):
        solver = _solver(CROSS_PLY, 12, 4, formulation='hex8')
        r = solver.solve('thermal', delta_T=DT)
        lam = r.first_ply_failure_load_factor
        assert 1.0 < lam < np.inf
        at_crit = solver.solve('thermal', delta_T=lam * DT)
        assert at_crit.max_failure_index == pytest.approx(1.0, rel=1e-9)

    def test_failure_on_cool_down_is_logged(self, caplog):
        solver = _solver(CROSS_PLY, 12, 4, formulation='hex8')
        with caplog.at_level(logging.WARNING, logger='porosity_fe_analysis'):
            r = solver.solve('thermal', delta_T=-400.0)
        assert r.residual_max_failure_index >= 1.0
        assert r.first_ply_failure_load_factor < 1.0
        assert 'fails on cool-down' in caplog.text

    def test_unit_solution_is_cached(self):
        solver = _solver(CROSS_PLY, 6, 2, formulation='hex8')
        solver.solve('thermal', delta_T=DT)
        cached = solver._thermal_cache
        solver.solve('thermal', delta_T=-50.0)
        assert solver._thermal_cache is cached
        solver.solve('thermal', delta_T=-50.0, solver='cg')
        assert solver._thermal_cache is not cached


# --------------------------------------------------------------------------
# Combined thermal + mechanical
# --------------------------------------------------------------------------

class TestCombinedLoading:
    @pytest.mark.parametrize('formulation', FORMULATIONS)
    @pytest.mark.parametrize('loading', ['compression', 'tension', 'shear', 'ilss'])
    def test_superposition(self, formulation, loading):
        solver = _solver(CROSS_PLY, 12, 4, formulation=formulation, vp=0.02,
                         distribution='clustered')
        th = solver.solve('thermal', delta_T=DT)
        mech = solver.solve(loading)
        comb = solver.solve(loading, delta_T=DT)
        for name in ('stress_global', 'stress_local', 'strain_global',
                     'strain_local', 'displacement'):
            total = getattr(th, name) + getattr(mech, name)
            np.testing.assert_allclose(getattr(comb, name), total, rtol=0,
                                       atol=1e-10 * np.abs(total).max())
        np.testing.assert_array_equal(comb.residual_stress_local, th.stress_local)
        # Stiffness measures and grip reactions are the mechanical ones.
        assert comb.knockdown == mech.knockdown
        assert comb.effective_modulus == mech.effective_modulus
        np.testing.assert_array_equal(comb.reaction_forces, mech.reaction_forces)
        assert comb.load_factor_basis == 'mechanical_with_residual'
        assert comb.residual_max_failure_index == pytest.approx(th.max_failure_index)
        assert mech.delta_T is None and mech.residual_stress_local is None
        assert mech.load_factor_basis == 'mechanical'

    @pytest.mark.parametrize('criterion', failure.SUPPORTED_FAILURE_CRITERIA)
    def test_load_factor_scales_only_the_mechanical_part(self, criterion):
        solver = _solver(CROSS_PLY, 12, 4, formulation='hex8', vp=0.02)
        mesh, mat = solver.mesh, solver.material
        radii = solver.porosity_field.void_shape_radii
        for loading, strain in (('tension', 0.001), ('compression', -0.001)):
            th = solver.solve('thermal', delta_T=DT, failure_criterion=criterion)
            mech = solver.solve(loading, applied_strain=strain,
                                failure_criterion=criterion)
            comb = solver.solve(loading, applied_strain=strain, delta_T=DT,
                                failure_criterion=criterion)
            lam = comb.first_ply_failure_load_factor
            assert 0.0 < lam < mech.first_ply_failure_load_factor

            def fi(L, s_th=th.stress_local, s_m=mech.stress_local):
                return failure.evaluate_failure(
                    s_th + L * s_m, mesh.porosity, mesh.elements, mat, radii,
                    criterion)[0]
            assert fi(lam) == pytest.approx(1.0, rel=1e-7)
            assert fi(lam * (1 - 1e-6)) < 1.0
            # Scaling the total stress instead (the naive factor) is wrong.
            naive = failure.first_ply_failure_load_factor(
                comb.stress_local, mesh.porosity, mesh.elements, mat, radii,
                criterion)
            assert abs(naive - lam) > 0.1 * lam

    @pytest.mark.parametrize('cross_ply', ['hex8'], indirect=True)
    def test_cross_ply_residual_uses_transverse_capacity(self, cross_ply):
        """Plan 2.7 numbers (prototype, hex8): T800 [0/90]_2s, dT = -150 K, interior."""
        solver, r = cross_ply
        assert r.interior_max_failure_index == pytest.approx(0.541, abs=0.01)
        mech = solver.solve('tension', applied_strain=0.001)
        comb = solver.solve('tension', applied_strain=0.001, delta_T=DT)
        assert mech.interior_first_ply_failure_load_factor == pytest.approx(7.00, abs=0.05)
        assert comb.interior_first_ply_failure_load_factor == pytest.approx(2.69, abs=0.05)

    def test_delta_T_zero_adds_nothing(self):
        solver = _solver(CROSS_PLY, 6, 2, formulation='hex8')
        mech = solver.solve('tension')
        comb = solver.solve('tension', delta_T=0.0)
        np.testing.assert_allclose(comb.stress_local, mech.stress_local,
                                   rtol=0, atol=1e-12)
        assert comb.first_ply_failure_load_factor == pytest.approx(
            mech.first_ply_failure_load_factor, rel=1e-12)


# --------------------------------------------------------------------------
# First-ply failure with a pre-stress: closed form against bisection
# --------------------------------------------------------------------------

STRENGTHS = failure.degraded_strengths(MATERIALS['T800_epoxy'], (1, 1, 1), 0.02)


def _fi(criterion, s, strengths=STRENGTHS, F12=None):
    s = np.atleast_2d(s)
    if criterion == 'tsai_wu':
        return failure.evaluate_tsai_wu(s, strengths, 0, 0.0, F12)
    if criterion == 'hashin':
        return failure.evaluate_hashin(s, strengths)['max_fi']
    return failure.evaluate_max_stress(s, strengths)['max_fi']


def _bisect_first_failure(criterion, s_th, s_m, lam_hi, n_scan=4000):
    """Smallest lam in [0, lam_hi] with FI(s_th + lam s_m) >= 1, by scan + bisection."""
    def f(L):
        return _fi(criterion, s_th + L * s_m)[0]
    if f(0.0) >= 1.0:
        return 0.0
    grid = np.linspace(0.0, lam_hi, n_scan + 1)
    vals = _fi(criterion, s_th[None, :] + grid[:, None] * s_m[None, :])
    hit = np.flatnonzero(vals >= 1.0)
    if hit.size == 0:
        return np.inf
    lo, hi = grid[hit[0] - 1], grid[hit[0]]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 1.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * max(hi, 1.0):
            break
    return hi


def _random_pair(rng):
    scale = np.array([800.0, 40.0, 40.0, 30.0, 40.0, 40.0])
    return rng.normal(size=6) * scale * 0.4, rng.normal(size=6) * scale


class TestPrestressedLoadFactor:
    @pytest.mark.parametrize('criterion', failure.SUPPORTED_FAILURE_CRITERIA)
    def test_closed_form_matches_bisection(self, criterion):
        rng = np.random.default_rng(35)
        checked = 0
        for _ in range(300):
            s_th, s_m = _random_pair(rng)
            lam = float(failure._point_load_factors_prestressed(
                s_th[None], s_m[None], STRENGTHS, criterion, None)[0])
            hi = 3.0 * lam if np.isfinite(lam) and lam > 0 else 50.0
            ref = _bisect_first_failure(criterion, s_th, s_m, hi)
            if lam == 0.0:
                assert ref == 0.0
            else:
                assert lam == pytest.approx(ref, rel=1e-6)
                checked += 1
        assert checked > 200

    @pytest.mark.parametrize('criterion', failure.SUPPORTED_FAILURE_CRITERIA)
    def test_zero_prestress_reduces_to_existing_factor(self, criterion):
        rng = np.random.default_rng(7)
        s = rng.normal(size=(50, 6)) * np.array([800, 40, 40, 30, 40, 40])
        old = failure._point_load_factors(s, STRENGTHS, criterion, None)
        new = failure._point_load_factors_prestressed(
            np.zeros_like(s), s, STRENGTHS, criterion, None)
        np.testing.assert_allclose(new, old, rtol=1e-12)

    @pytest.mark.parametrize('criterion', failure.SUPPORTED_FAILURE_CRITERIA)
    def test_failed_prestress_gives_zero(self, criterion):
        s_th = np.array([0.0, 2.0 * STRENGTHS[2], 0, 0, 0, 0])
        s_m = np.array([100.0, 1.0, 0, 0, 0, 1.0])
        lam = failure._point_load_factors_prestressed(
            s_th[None], s_m[None], STRENGTHS, criterion, None)
        assert lam[0] == 0.0

    def test_tsai_wu_closed_form_matches_quadratic(self):
        rng = np.random.default_rng(3)
        co = failure._tsai_wu_coefficients(STRENGTHS, None)
        for _ in range(50):
            s_th, s_m = _random_pair(rng)
            lam = failure._point_load_factors_prestressed(
                s_th[None], s_m[None], STRENGTHS, 'tsai_wu', None)[0]
            lin_th, q_th = failure._tsai_wu_forms(s_th, s_th, co)
            if lin_th + q_th >= 1.0:
                assert lam == 0.0
                continue
            assert _fi('tsai_wu', s_th + lam * s_m)[0] == pytest.approx(1.0, abs=1e-9)

    def test_max_stress_is_linear_with_offset(self):
        Xt, Xc, Yt, Yc, S12, S23 = STRENGTHS
        s_th = np.array([100.0, 30.0, 0, 0, 0, 10.0])
        s_m = np.array([10.0, 1.0, 0, 0, 0, -2.0])
        lam = failure._point_load_factors_prestressed(
            s_th[None], s_m[None], STRENGTHS, 'max_stress', None)[0]
        expected = min((Xt - 100.0) / 10.0, (Yt - 30.0) / 1.0, (-S12 - 10.0) / -2.0)
        assert lam == pytest.approx(expected, rel=1e-14)

    def test_hashin_fiber_tension_switches_on_with_a_jump(self):
        """sigma_11 goes from compression to tension at lam = 50 with a shear
        term above 1: fiber tension fails right at the sign change."""
        Xt, Xc, Yt, Yc, S12, S23 = STRENGTHS
        # sigma_22 at the minimum of the matrix-compression mode keeps the
        # matrix below 1 despite a shear term of 1.05.
        k = ((Yc / (2.0 * S23)) ** 2 - 1.0) / Yc
        s_th = np.array([-50.0, -2.0 * k * S23 ** 2, 0, 0, 0, np.sqrt(1.05) * S12])
        s_m = np.array([1.0, 0.0, 0, 0, 0, 0.0])
        assert _fi('hashin', s_th)[0] < 1.0
        lam = failure._point_load_factors_prestressed(
            s_th[None], s_m[None], STRENGTHS, 'hashin', None)[0]
        assert lam == pytest.approx(50.0, rel=1e-12)
        assert _bisect_first_failure('hashin', s_th, s_m, 200.0) == pytest.approx(
            50.0, rel=1e-9)
        # The existing sign-preserving logic would miss the switch.
        assert failure._point_load_factors(s_m[None], STRENGTHS, 'hashin', None)[0] \
            == pytest.approx(Xt, rel=1e-12)

    def test_hashin_matrix_mode_after_sign_change(self):
        """sigma_22 tensile in the residual state, driven into compression."""
        s_th = np.array([0.0, 20.0, 0, 0, 0, 0.0])
        s_m = np.array([0.0, -1.0, 0, 0, 0, 0.0])
        lam = failure._point_load_factors_prestressed(
            s_th[None], s_m[None], STRENGTHS, 'hashin', None)[0]
        ref = _bisect_first_failure('hashin', s_th, s_m, 1000.0)
        assert lam == pytest.approx(ref, rel=1e-9)
        assert lam > 20.0                      # fails in matrix compression
        assert _fi('hashin', s_th + lam * s_m)[0] == pytest.approx(1.0, rel=1e-9)

    def test_hashin_delamination_branches(self):
        """sigma_33 compressive in the residual state, then tensile."""
        Xt, Xc, Yt, Yc, S12, S23 = STRENGTHS
        s_th = np.array([0.0, 0.0, -10.0, 0.5 * S23, 0.0, 0.0])
        s_m = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        lam = failure._point_load_factors_prestressed(
            s_th[None], s_m[None], STRENGTHS, 'hashin', None)[0]
        assert lam == pytest.approx(10.0 + np.sqrt(0.75) * Yt, rel=1e-12)

    def test_never_failing_direction_is_inf(self):
        s_th = np.zeros(6)
        s_m = np.zeros(6)
        for criterion in failure.SUPPORTED_FAILURE_CRITERIA:
            lam = failure._point_load_factors_prestressed(
                s_th[None], s_m[None], STRENGTHS, criterion, None)
            assert np.isinf(lam[0])

    def test_public_function_validates_prestress_shape(self):
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8')
        r = solver.solve('tension')
        mesh = solver.mesh
        with pytest.raises(ValueError, match='prestress_local has shape'):
            failure.first_ply_failure_load_factor(
                r.stress_local, mesh.porosity, mesh.elements, solver.material,
                (1, 1, 1), prestress_local=r.stress_local[:2])


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

class TestGuards:
    def test_thermal_needs_explicit_delta_T(self):
        mat = _material(CROSS_PLY, T_stress_free=177.0)
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8', material=mat)
        with pytest.raises(ValueError, match='explicit delta_T'):
            solver.solve('thermal')

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '150'])
    def test_delta_T_must_be_a_finite_number(self, bad):
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8')
        with pytest.raises(ValueError, match='delta_T must be a finite number'):
            solver.solve('thermal', delta_T=bad)

    @pytest.mark.parametrize('loading', ['thermal', 'tension'])
    def test_material_without_cte_is_rejected(self, loading):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8, t_ply=0.125)
        assert not mat.has_cte
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8', material=mat)
        with pytest.raises(ValueError, match='thermal expansion coefficients'):
            solver.solve(loading, delta_T=DT)
        solver.solve('tension')              # mechanical-only is unaffected

    @pytest.mark.parametrize('loading', ['thermal', 'compression'])
    def test_unresolved_layup_raises_unless_allowed(self, loading, caplog):
        mat = MATERIALS['AS4_3501_6_epoxy']       # 24 plies, QI
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=6, ny=2, nz=12, ply_angles='QI')
        assert mesh.layup_discrepancies()
        solver = FESolver(mesh, mat, pf, formulation='hex8')
        with pytest.raises(ValueError, match=r'nz = k \* n_plies'):
            solver.solve(loading, delta_T=DT)
        with caplog.at_level(logging.WARNING, logger='porosity_fe_analysis'):
            r = solver.solve(loading, delta_T=DT, allow_unresolved_layup=True)
        assert 'allow_unresolved_layup=True' in caplog.text
        assert r.delta_T == DT
        solver.solve('compression')          # mechanical-only still runs

    def test_resolved_preset_layup_runs(self):
        mat = MATERIALS['AS4_3501_6_epoxy']
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=4, ny=2, nz=24, ply_angles='QI')
        assert not mesh.layup_discrepancies()
        r = FESolver(mesh, mat, pf, formulation='hex8').solve('thermal', delta_T=DT)
        assert r.max_failure_index > 0.0

    def test_unsymmetric_layup_warns(self, caplog):
        layup = [0.0, 0.0, 90.0, 90.0]
        solver = _solver(layup, 6, 2, formulation='hex8')
        with caplog.at_level(logging.WARNING, logger='porosity_fe_analysis'):
            r = solver.solve('thermal', delta_T=DT)
        assert 'unsymmetric' in caplog.text and "formulation='hex8i'" in caplog.text
        assert np.abs(r.displacement[:, 2]).max() > 0.01   # it warps

    @pytest.mark.parametrize('bad', [-1.0, float('nan')])
    def test_interior_margin_validated(self, bad):
        solver = _solver(CROSS_PLY, 4, 2, formulation='hex8')
        with pytest.raises(ValueError, match='interior_margin'):
            solver.solve('tension', interior_margin=bad)

    def test_interior_margin_controls_interior_set(self):
        solver = _solver(CROSS_PLY, 10, 4, formulation='hex8')
        wide = solver.solve('thermal', delta_T=DT, interior_margin=0.0)
        assert wide.interior_max_failure_index == wide.max_failure_index
        none = solver.solve('thermal', delta_T=DT, interior_margin=30.0)
        assert none.interior_max_failure_index is None
        assert none.interior_first_ply_failure_load_factor is None


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

_SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'validation', 'schemas', 'porosity_results_schema.json')


class TestExport:
    @pytest.fixture(scope='class')
    @classmethod
    def solved(cls):
        solver = _solver(CROSS_PLY, 6, 2, formulation='hex8')
        return (solver, solver.solve('thermal', delta_T=DT),
                solver.solve('tension', delta_T=DT), solver.solve('tension'))

    def test_json_thermal_block_and_null_knockdown(self, solved, tmp_path):
        import jsonschema
        solver, th, comb, mech = solved
        schema = json.loads(open(_SCHEMA, encoding='utf-8').read())
        for name, r in (('th', th), ('comb', comb), ('mech', mech)):
            path = tmp_path / f'{name}.json'
            FESolver.export_results(r, path, include_raw=True)
            doc = json.loads(path.read_text(encoding='utf-8'))
            jsonschema.validate(instance=doc, schema=schema)
            fail = doc['failure']
            assert fail['load_factor_basis'] == r.load_factor_basis
            if r.delta_T is None:
                assert 'thermal' not in doc and 'delta_T_K' not in doc['provenance']
                assert fail['knockdown_factor'] == r.knockdown
                continue
            assert doc['provenance']['delta_T_K'] == DT
            block = doc['thermal']
            assert block['delta_T_K'] == DT
            assert block['alpha_local_per_K'] == [ALPHA_1, ALPHA_2, ALPHA_2]
            assert block['residual_max_failure_index'] == pytest.approx(
                r.residual_max_failure_index)
            assert block['residual_stress_local']['sigma_22']['max'] == pytest.approx(
                float(r.residual_stress_local[..., 1].max()))
            raw = np.load(str(path) + '.npz')
            np.testing.assert_array_equal(raw['residual_stress_local'],
                                          r.residual_stress_local)
        text = (tmp_path / 'th.json').read_text(encoding='utf-8')
        assert '"knockdown_factor": null' in text

    def test_vtk_has_residual_fields_and_no_nan(self, solved, tmp_path):
        solver, th, comb, mech = solved
        path = tmp_path / 'th.vtk'
        th.to_vtk(solver.mesh, path)
        text = path.read_text(encoding='utf-8')
        assert 'nan' not in text.lower()
        assert 'SCALARS residual_sigma_22_local' in text
        assert 'SCALARS knockdown' not in text
        path = tmp_path / 'mech.vtk'
        mech.to_vtk(solver.mesh, path)
        text = path.read_text(encoding='utf-8')
        assert 'SCALARS knockdown' in text and 'residual_' not in text

    def test_vtu_has_residual_fields(self, solved, tmp_path):
        solver, th, comb, mech = solved
        path = tmp_path / 'comb.vtu'
        comb.to_vtu(solver.mesh, path)
        head = path.read_bytes().split(b'<AppendedData')[0].decode('ascii')
        for name in ('residual_sigma_11_local', 'residual_sigma_22_local',
                     'residual_tau_12_local', 'knockdown'):
            assert f'Name="{name}"' in head
