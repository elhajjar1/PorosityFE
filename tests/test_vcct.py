"""Crack-face contact, virtual crack closure and the mixed-mode fracture criterion.

Delamination plan, phase 2.4-2.8. On explicit delaminations
(:class:`Delamination`) this checks:

- double-cantilever beam (the feasibility probe's 16-ply UD beam): VCCT
  ``G_I`` against the compliance derivative and root-rotation-corrected beam
  theory, ``G_II`` and ``G_III`` vanishing, the energy balance, the
  distribution across the width and the convergence under refinement;
- end-notched flexure through :class:`FESolver` (``'flexure_3pt'`` on a
  :meth:`CompositeMesh.for_flexure` beam): ``G_II`` against the compliance
  derivative, ``G_I = 0`` with crack-face contact, and the penetration
  without it;
- a single-leg (mixed-mode) beam: the mode mix against corrected beam
  theory and ``G_T`` against the compliance derivative;
- the critical load factor (by hand, under load scaling, with a prestress,
  through the thermal solve) and the Benzeggagh-Kenane limits;
- a penny-shaped crack in an isotropic block under remote tension against
  ``G = (1 - nu^2) K_I^2 / E``, ``K_I = 2 sigma sqrt(a / pi)``;
- the toughness inputs, the exports, and that a solve without a
  delamination is unchanged.

Energy release rates and toughness are in N/mm (1 N/mm = 1 kJ/m^2).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    CrackFront,
    Delamination,
    FESolver,
    GlobalAssembler,
    MaterialProperties,
    PorosityField,
    critical_energy_release_rate,
    fracture_load_factors,
    vcct_energy_release_rates,
)
from porosity_fe.fe._constrained_solve import free_dof_system, solve_constrained
from porosity_fe.fe.assembler import BoundaryHandler, _mesh_ties
from porosity_fe.fe.vcct import _pair_forces

#: IM7/8552 toughness and B-K exponent (N/mm), Krueger NASA/CR-2012-217562
#: (plan Appendix A, [P]); explicit test inputs on other elastic constants.
TOUGHNESS = dict(G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)

T800_16 = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=16)
MID16 = 7          # mid-plane interface of 16 plies

_SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'validation', 'schemas', 'porosity_results_schema.json')


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _solve(mesh, K, constrained, F=None, *, solver='direct'):
    """Solve with the mesh's bonded ties: ``u`` and ``r = K u - F`` (unfolded)."""
    F = np.zeros(K.shape[0]) if F is None else F
    system = partial(free_dof_system, OrderedDict(), max_entries=1)
    u, _ = solve_constrained(K, F, constrained, system=system, ties=_mesh_ties(mesh),
                             solver=solver, rtol=1e-10)
    return u, K @ u - F


def _width_average(front: CrackFront, values: np.ndarray, width: float) -> float:
    return float(np.sum(values * front.width) / width)


def _chi_h(material: MaterialProperties) -> float:
    """Root-rotation length ``chi h`` of a DCB arm (Hashemi, Kinloch and
    Williams; the same correction as tests/test_delamination.py)."""
    E11, E22, G13 = material.E11, material.E22, material.G13
    gamma = 1.18 * math.sqrt(E11 * E22) / G13
    chi = math.sqrt(E11 / (11.0 * G13) * (3.0 - 2.0 * (gamma / (1.0 + gamma)) ** 2))
    return chi * 0.5 * material.total_thickness


# --------------------------------------------------------------------------
# Double-cantilever beam (the probe's beam)
# --------------------------------------------------------------------------

DCB_W, DCB_L, DCB_A = 4.0, 60.0, 21.0


def _beam_mesh(material, a, *, nx=60, ny=2, width=DCB_W):
    """16-ply UD beam, ``DCB_L`` long, mid-plane strip from ``x = 0`` to ``a``."""
    pf = PorosityField(material, 0.0)
    mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=16, ply_angles='UD',
                         delaminations=[Delamination.strip(MID16, 0.0, a)])
    mesh.L_x, mesh.L_y = DCB_L, width
    mesh.generate_mesh()
    return mesh


def _end_lines(mesh):
    x, z = mesh.nodes[:, 0], mesh.nodes[:, 2]
    top = np.flatnonzero((np.abs(x) < 1e-9) & (np.abs(z - mesh.L_z) < 1e-9))
    bottom = np.flatnonzero((np.abs(x) < 1e-9) & (np.abs(z) < 1e-9))
    clamp = {3 * int(n) + c: 0.0
             for n in BoundaryHandler(mesh)._face_nodes('x_max') for c in range(3)}
    return top, bottom, clamp


def _dcb(mesh, K, delta=0.5):
    """Hinge lines of the two arms opened by ``+-delta``, far end clamped."""
    top, bottom, constrained = _end_lines(mesh)
    constrained.update({3 * int(n) + 2: delta for n in top})
    constrained.update({3 * int(n) + 2: -delta for n in bottom})
    u, r = _solve(mesh, K, constrained)
    return dict(u=u, r=r, P=float(r[3 * top + 2].sum()), U=0.5 * float(u @ (K @ u)),
                ties=_mesh_ties(mesh), front=vcct_energy_release_rates(mesh, u, r))


def _single_leg(mesh, K, load=1.0):
    """Upper arm loaded by ``load`` (N, +z) on its end line, lower arm free,
    far end clamped: the load point compliance ``C = u.F / load^2``."""
    top, _, constrained = _end_lines(mesh)
    y = mesh.nodes[top, 1]
    w = np.zeros(top.size)                     # trapezoidal weights across the width
    order = np.argsort(y)
    seg = np.diff(y[order])
    w[order[:-1]] += 0.5 * seg
    w[order[1:]] += 0.5 * seg
    F = np.zeros(K.shape[0])
    F[3 * top + 2] = load * w / w.sum()
    u, r = _solve(mesh, K, constrained, F)
    return dict(u=u, r=r, C=float(u @ F) / load ** 2,
                front=vcct_energy_release_rates(mesh, u, r))


@pytest.fixture(scope='module')
def beam():
    """The beam mesh with the crack at 21 mm and its stiffness (reused for
    every crack length: only the ties change)."""
    mesh = _beam_mesh(T800_16, DCB_A)
    K = GlobalAssembler(mesh, T800_16, mesh.porosity_field).stiffness()
    return mesh, K


@pytest.fixture(scope='module')
def dcb(beam):
    mesh, K = beam
    out = {}
    for a in (DCB_A - 1.0, DCB_A + 1.0, DCB_A):        # ends at DCB_A
        mesh.delaminations = (Delamination.strip(MID16, 0.0, a),)
        out[a] = _dcb(mesh, K)
    return out


class TestDoubleCantileverBeam:
    def test_front_and_frame(self, dcb):
        front = dcb[DCB_A]['front']
        assert len(front) == 3
        np.testing.assert_allclose(front.x, DCB_A)
        np.testing.assert_allclose(front.y, [0.0, 2.0, 4.0])
        # x' is exactly +x on a straight front; Delta_a one element;
        # half the tributary width at the two plate edges.
        np.testing.assert_array_equal(front.direction, [[1.0, 0.0]] * 3)
        np.testing.assert_allclose(front.delta_a, 1.0)
        np.testing.assert_allclose(front.width, [1.0, 2.0, 1.0])
        np.testing.assert_allclose(front.area.sum(), DCB_W * 1.0)
        assert front.G_c is None and front.lambda_c is None

    def test_mode_I_against_the_compliance_derivative(self, dcb):
        """VCCT within 5 % of ``P^2 / (2 b) dC/da`` over one element (it
        reads 0.959: the one-element difference is centred half an element
        ahead, where ``G`` is 4 % higher) and within 0.5 % of the centred
        difference over +-1 element (0.9991)."""
        a, P = DCB_A, dcb[DCB_A]['P']
        G = _width_average(dcb[a]['front'], dcb[a]['front'].G_I, DCB_W)
        C = {k: 1.0 / v['P'] for k, v in dcb.items()}       # 2 delta / P, delta = 0.5
        G_forward = P ** 2 / (2 * DCB_W) * (C[a + 1] - C[a])
        G_central = P ** 2 / (2 * DCB_W) * (C[a + 1] - C[a - 1]) / 2.0
        assert G == pytest.approx(G_forward, rel=0.05)
        assert G == pytest.approx(G_central, rel=5e-3)

    def test_against_corrected_beam_theory(self, dcb):
        """``G = 12 P^2 (a + chi h)^2 / (E11 b^2 h^3)``: 0.988 at 1 mm elements."""
        a, P = DCB_A, dcb[DCB_A]['P']
        h = 0.5 * T800_16.total_thickness
        G_beam = 12 * P ** 2 * (a + _chi_h(T800_16)) ** 2 / (T800_16.E11 * DCB_W ** 2 * h ** 3)
        G = _width_average(dcb[a]['front'], dcb[a]['front'].G_I, DCB_W)
        assert G == pytest.approx(G_beam, rel=0.03)

    def test_symmetric_beam_is_pure_mode_I(self, dcb):
        front = dcb[DCB_A]['front']
        assert np.all(front.G_I > 0.0)
        assert np.max(front.G_II / front.G_I) < 1e-3
        assert np.max(front.G_III / front.G_I) < 1e-3
        np.testing.assert_allclose(front.psi, 0.0, atol=1e-3)

    def test_energy_balance(self, dcb, beam):
        """At fixed grips, the strain energy released by a one-element
        extension equals the crack closure work of the released pairs,
        ``-1/2 sum F(a) . Delta(a + da)``, exactly (1e-9); and the one-step
        VCCT sum ``sum G_T b Delta_a`` (from the opening one element behind)
        matches the energy released per one-element extension, centred on
        ``a`` (``[U(a - da) - U(a + da)] / 2``), within 2 % (0.9938)."""
        mesh, _ = beam
        a = DCB_A
        s1, s2 = dcb[a], dcb[a + 1.0]
        front = s1['front']
        forces = _pair_forces(mesh, s1['r'], s1['ties'])[front.pair]
        u2 = s2['u'].reshape(-1, 3)
        closure = -0.5 * float(np.sum(forces * (u2[front.duplicate] - u2[front.node])))
        released = s1['U'] - s2['U']
        assert closure == pytest.approx(released, rel=1e-9)
        vcct = float(np.sum(front.G_T * front.area))
        centred = 0.5 * (dcb[a - 1.0]['U'] - s2['U'])
        assert vcct == pytest.approx(centred, rel=0.02)
        assert vcct == pytest.approx(-0.5 * float(np.sum(front.force_local * front.opening_local)),
                                     rel=1e-12)

    def test_full_residual_and_pair_forces_agree(self, dcb, beam):
        mesh, _ = beam
        mesh.delaminations = (Delamination.strip(MID16, 0.0, DCB_A),)
        s = dcb[DCB_A]
        by_pairs = vcct_energy_release_rates(
            mesh, s['u'], _pair_forces(mesh, s['r'], s['ties']))
        np.testing.assert_array_equal(by_pairs.G_I, s['front'].G_I)
        with pytest.raises(ValueError, match='interface_forces'):
            vcct_energy_release_rates(mesh, s['u'], np.zeros(7))

    def test_uniform_across_the_width(self):
        """Without Poisson coupling the beam is in a 2-D state, so ``G_I`` is
        the same at every front node, edge nodes (half width) included. With
        it (T800, a 20 mm wide beam) the interior nodes, at least 4 mm from
        the free edges, are within 2 % of their mean (1.0305 to 1.0432 of the
        width average); the free-edge nodes read lower (0.74 of the
        average), the corner singularity where the front meets a free
        surface."""
        nu0 = dataclasses.replace(T800_16, nu12=0.0, nu13=0.0, nu23=0.0)
        mesh = _beam_mesh(nu0, DCB_A, ny=4)
        front = _dcb(mesh, GlobalAssembler(mesh, nu0, mesh.porosity_field).stiffness())['front']
        np.testing.assert_allclose(front.G_I, front.G_I.mean(), rtol=1e-6)

        mesh = _beam_mesh(T800_16, DCB_A, nx=30, ny=10, width=20.0)
        front = _dcb(mesh, GlobalAssembler(mesh, T800_16, mesh.porosity_field).stiffness())['front']
        ratio = front.G_I / _width_average(front, front.G_I, 20.0)
        interior = (front.y >= 4.0) & (front.y <= 16.0)
        assert np.ptp(ratio[interior]) / ratio[interior].mean() < 0.02
        assert ratio[0] == pytest.approx(ratio[-1], rel=1e-9)
        assert 0.6 < ratio[0] < 0.9

    def test_mesh_convergence(self):
        """``G`` over corrected beam theory at a = 20 mm: 0.9792, 0.9878,
        0.9931 at Delta_a = 2, 1, 0.5 mm (0.9961 at 0.25 mm), each with its
        own load: monotone, the steps shrink, within 1 % at 0.5 mm."""
        a = 20.0
        h = 0.5 * T800_16.total_thickness
        ratios = []
        for nx in (30, 60, 120):
            mesh = _beam_mesh(T800_16, a, nx=nx)
            s = _dcb(mesh, GlobalAssembler(mesh, T800_16, mesh.porosity_field).stiffness())
            G = _width_average(s['front'], s['front'].G_T, DCB_W)
            G_beam = 12 * s['P'] ** 2 * (a + _chi_h(T800_16)) ** 2 / (
                T800_16.E11 * DCB_W ** 2 * h ** 3)
            ratios.append(G / G_beam)
        steps = np.diff(ratios)
        assert np.all(steps > 0.0)
        assert abs(steps[1]) < abs(steps[0])
        assert ratios[-1] == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------
# Single-leg (mixed-mode) beam
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def single_leg(beam):
    mesh, K = beam
    out = {}
    for a in (DCB_A - 1.0, DCB_A + 1.0, DCB_A):
        mesh.delaminations = (Delamination.strip(MID16, 0.0, a),)
        s = _single_leg(mesh, K)
        originals, duplicates = mesh.interface_node_pairs()
        u = s['u'].reshape(-1, 3)
        s['min_opening'] = float(np.min(
            (u[duplicates, 2] - u[originals, 2])[~mesh.bonded_pairs()]))
        out[a] = s
    return out


class TestSingleLegBending:
    """Only the upper arm is loaded, so at the front it carries the moment
    ``M = P a`` and the lower arm none. Williams' beam partition for equal
    arms, ``G_I = 12 M_I^2 / (E b^2 h^3)``, ``G_II = 9 M_II^2 / (E b^2 h^3)``
    with ``M_I = M_II = M / 2``, gives ``G_II / G_T = 3/7`` for a long crack;
    root rotation lengthens each mode's arm, by ``chi h`` in mode I and
    ``0.42 chi h`` in mode II (attributed to Wang and Williams 1992, not
    checked against the paper), which gives 0.3936 at ``a / h = 14``."""

    def test_faces_open(self, single_leg):
        assert all(s['min_opening'] > 0.0 for s in single_leg.values())

    def test_mode_mix_against_corrected_beam_theory(self, single_leg):
        a = DCB_A
        front = single_leg[a]['front']
        psi = _width_average(front, front.G_II + front.G_III, DCB_W) / \
            _width_average(front, front.G_T, DCB_W)
        chi_h = _chi_h(T800_16)
        g_I = 12.0 * (1.0 + chi_h / a) ** 2
        g_II = 9.0 * (1.0 + 0.42 * chi_h / a) ** 2
        psi_beam = g_II / (g_I + g_II)
        assert psi == pytest.approx(psi_beam, rel=0.05)     # 0.3897 / 0.3936
        assert psi < 3.0 / 7.0
        assert np.max(front.G_III / front.G_T) < 0.02

    def test_total_against_the_compliance_derivative(self, single_leg):
        a = DCB_A
        front = single_leg[a]['front']
        G = _width_average(front, front.G_T, DCB_W)
        G_central = (single_leg[a + 1]['C'] - single_leg[a - 1]['C']) / (2 * DCB_W * 2.0)
        assert G == pytest.approx(G_central, rel=0.05)      # 0.9991


# --------------------------------------------------------------------------
# End-notched flexure through FESolver (crack-face contact)
# --------------------------------------------------------------------------

ENF_MAT = dataclasses.replace(T800_16, **TOUGHNESS)
ENF_LOAD = -100.0


def _enf_mesh():
    pf = PorosityField(ENF_MAT, 0.0)
    return CompositeMesh.for_flexure(
        pf, ENF_MAT, span=50.0, width=DCB_W, overhang=5.0, load_width=0.0,
        nz=16, ny=2, element_length=1.0, ply_angles='UD',
        delaminations=[Delamination.strip(MID16, -1.0, 10.0)])


def _openings(mesh, result):
    originals, duplicates = mesh.interface_node_pairs()
    u = result.displacement
    return (u[duplicates, 2] - u[originals, 2])[~mesh.bonded_pairs()]


@pytest.fixture(scope='module')
def enf():
    """ENF beam, crack tip at the node line nearest 0.6 of the half span
    from the left support, and one element either side; solved with
    contact, once without, and once at twice the load."""
    mesh = _enf_mesh()
    xs = np.unique(mesh.nodes[:mesh.n_structured_nodes, 0])
    s0 = mesh.flexure_geometry.support_x[0]
    i_tip = int(np.argmin(np.abs(xs - (s0 + 0.6 * 25.0))))
    solver = FESolver(mesh, ENF_MAT, mesh.porosity_field, ply_angles='UD')
    K = solver.assembler.stiffness()
    out = {'mesh': mesh, 'solver': solver, 'dx': float(xs[1] - xs[0])}
    for di in (-1, 1, 0):                                  # ends at the tip
        mesh.delaminations = (Delamination.strip(MID16, -1.0, float(xs[i_tip + di])),)
        r = solver.solve('flexure_3pt', applied_load=ENF_LOAD)
        u = r.displacement.ravel()
        out[di] = r
        out[('C', di)] = float(u @ (K @ u)) / ENF_LOAD ** 2
    out['free'] = solver.solve('flexure_3pt', applied_load=ENF_LOAD, crack_contact=False)
    out['double'] = solver.solve('flexure_3pt', applied_load=2 * ENF_LOAD)
    out['repeat'] = solver.solve('flexure_3pt', applied_load=ENF_LOAD)
    return out


class TestEndNotchedFlexure:
    def test_mode_II_against_the_compliance_derivative(self, enf):
        """Within 2 % of the centred compliance derivative (0.9983); the
        one-element forward difference is centred half an element ahead
        (0.943 at 0.96 mm elements)."""
        front = enf[0].crack_front
        G = _width_average(front, front.G_T, DCB_W)
        G_central = ENF_LOAD ** 2 / (2 * DCB_W) * (enf[('C', 1)] - enf[('C', -1)]) / (2 * enf['dx'])
        assert G == pytest.approx(G_central, rel=0.02)
        assert _width_average(front, front.G_II, DCB_W) / G > 0.98

    def test_contact_closes_the_crack(self, enf):
        r, mesh = enf[0], enf['mesh']
        assert r.crack_contact is True and r.contact_converged is True
        assert r.contact_pairs > 0 and r.contact_iterations >= 2
        # The pair behind the front is open by 1e-5 mm: G_I ~ 1e-6 G_T.
        assert np.max(r.crack_front.G_I / r.crack_front.G_T) < 1e-4
        scale = float(np.abs(r.displacement).max())
        assert _openings(mesh, r).min() >= -1e-12 * scale

    def test_without_contact_the_faces_penetrate(self, enf):
        free, r, mesh = enf['free'], enf[0], enf['mesh']
        assert free.crack_contact is False
        assert free.contact_iterations is None and free.contact_pairs is None
        scale = float(np.abs(free.displacement[:, 2]).max())
        assert _openings(mesh, free).min() < -0.1 * scale    # a visible overlap
        assert free.crack_front.G_I.max() > r.crack_front.G_II.max()

    def test_contact_forces(self, enf):
        """Pairs in contact push the upper face up (compressive), open pairs
        carry nothing, and the front pairs carry their tie force."""
        r, mesh = enf[0], enf['mesh']
        forces = r.interface_forces
        bonded = mesh.bonded_pairs()
        in_contact = ~bonded & (np.abs(forces[:, 2]) > 0.0)
        assert in_contact.sum() == r.contact_pairs
        assert np.all(forces[in_contact, 2] > 0.0)
        np.testing.assert_array_equal(forces[~bonded][:, :2], 0.0)
        recomputed = r.energy_release_rates(mesh)
        np.testing.assert_array_equal(recomputed.G_T, r.crack_front.G_T)
        assert recomputed.lambda_c is None

    def test_repeat_solve_reuses_the_active_set(self, enf):
        r, again = enf[0], enf['repeat']
        assert again.contact_iterations == 1
        assert again.contact_pairs == r.contact_pairs
        np.testing.assert_array_equal(again.displacement, r.displacement)


class TestMultidirectionalShearFront:
    """A mid-plane crack in the shear span of the QI short-beam specimen,
    where the mode III product changes sign across the width. The shear
    products keep their sign, so the front still sums to the energy
    release: the width average is within 2 % of the centred compliance
    derivative (0.989 at 0.55 mm elements). Zeroing the negative products
    gave 1.041 here, and 1.070 at 0.37 mm."""

    def test_signed_shear_products_sum_to_the_energy(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], **TOUGHNESS)
        pf = PorosityField(mat, 0.0)
        kw = dict(nx=48, ny=4, ply_angles='QI')
        base = CompositeMesh.for_ilss(pf, mat, **kw)
        width, dx = base.L_y, base.L_x / base.nx
        geometry = base.ilss_geometry
        x0, xm = geometry.support_x[0], geometry.midspan
        x_left, x_right = x0 + 0.25 * (xm - x0), x0 + 0.6 * (xm - x0)
        load = -10.0
        out = {}
        for k in (-1, 0, 1):
            mesh = CompositeMesh.for_ilss(
                pf, mat, delaminations=[Delamination.strip(11, x_left, x_right + k * dx)],
                **kw)
            r = FESolver(mesh, mat, pf).solve('ilss', applied_load=load,
                                              crack_contact=False)
            _, F = BoundaryHandler(mesh).ilss_bcs(load)
            out[k] = (r, float(F @ r.displacement.ravel()) / load ** 2)
        front = out[0][0].crack_front
        right = front.x > 0.5 * (x_left + x_right)
        G = float(np.sum(front.G_T[right] * front.width[right]) / width)
        G_central = load ** 2 / 2 * (out[1][1] - out[-1][1]) / (2 * dx * width)
        assert G == pytest.approx(G_central, rel=0.02)
        assert np.min(front.G_III[right]) < 0.0 < np.max(front.G_III[right])
        assert np.all(front.G_I >= 0.0)


class TestLoadFactor:
    def test_by_hand(self, enf):
        front, r = enf[0].crack_front, enf[0]
        mat = ENF_MAT
        psi = (front.G_II + front.G_III) / front.G_T
        G_c = mat.G_Ic + (mat.G_IIc - mat.G_Ic) * psi ** mat.bk_eta
        np.testing.assert_allclose(front.G_c, G_c, rtol=1e-12)
        np.testing.assert_allclose(front.lambda_c, np.sqrt(G_c / front.G_T), rtol=1e-12)
        assert r.fracture_load_factor == pytest.approx(float(np.min(front.lambda_c)), rel=1e-15)
        row = int(np.argmin(front.lambda_c))
        assert r.fracture_governing_node == int(front.node[row])
        details = r.summary().details
        assert details['fracture_load_factor'] == r.fracture_load_factor
        assert details['fracture_governing_psi'] == pytest.approx(psi[row], rel=1e-12)
        assert details['contact_pairs'] == r.contact_pairs

    def test_scales_inversely_with_the_load(self, enf):
        """Frictionless contact from a zero gap is positively homogeneous:
        the same pairs touch at twice the load and ``lambda_c`` halves."""
        r, double = enf[0], enf['double']
        assert double.contact_pairs == r.contact_pairs
        np.testing.assert_allclose(double.crack_front.lambda_c,
                                   0.5 * r.crack_front.lambda_c, rtol=1e-9)
        np.testing.assert_allclose(double.crack_front.G_T, 4.0 * r.crack_front.G_T,
                                   rtol=1e-9)

    def test_benzeggagh_kenane_limits(self):
        mat = ENF_MAT
        assert critical_energy_release_rate(mat, 1.0, 0.0) == pytest.approx(mat.G_Ic)
        assert critical_energy_release_rate(mat, 0.0, 1.0) == pytest.approx(mat.G_IIc)
        assert critical_energy_release_rate(mat, 0.0, 0.0, 1.0) == pytest.approx(mat.G_IIc)
        assert critical_energy_release_rate(mat, 1.0, 1.0) == pytest.approx(
            mat.G_Ic + (mat.G_IIc - mat.G_Ic) * 0.5 ** mat.bk_eta)
        assert np.isnan(critical_energy_release_rate(mat, 0.0, 0.0))
        tear = dataclasses.replace(mat, G_IIIc=1.2)
        assert critical_energy_release_rate(tear, 0.0, 0.0, 1.0) == pytest.approx(1.2)
        assert critical_energy_release_rate(tear, 0.0, 1.0, 0.0) == pytest.approx(mat.G_IIc)
        # The porosity hook does not change the toughness yet.
        assert critical_energy_release_rate(mat, 1.0, 1.0, Vp=0.05) == \
            critical_energy_release_rate(mat, 1.0, 1.0)
        with pytest.raises(ValueError, match='G_Ic'):
            critical_energy_release_rate(T800_16, 1.0, 0.0)
        with pytest.raises(ValueError, match='G_Ic'):
            fracture_load_factors(enf_front_stub(), T800_16)

    def test_prestress_shifts_the_factor_exactly(self, enf):
        """A prestress ``c`` times the load state gives ``u = (c + lam) u_M``,
        so ``lambda_c = sqrt(G_c / G_T) - c`` (0 once ``c`` reaches it)."""
        front = enf[0].energy_release_rates(enf['mesh'])
        lam = fracture_load_factors(front, ENF_MAT).lambda_c
        for c in (0.3, -0.4):
            pre = dataclasses.replace(front, force_local=c * front.force_local,
                                      opening_local=c * front.opening_local)
            shifted = fracture_load_factors(front, ENF_MAT, prestress=pre)
            np.testing.assert_allclose(shifted.lambda_c, lam - c, rtol=1e-10)
            np.testing.assert_allclose(shifted.G_T, (1.0 + c) ** 2 * front.G_T, rtol=1e-12)
        pre = dataclasses.replace(front, force_local=2 * lam.max() * front.force_local,
                                  opening_local=2 * lam.max() * front.opening_local)
        np.testing.assert_array_equal(
            fracture_load_factors(front, ENF_MAT, prestress=pre).lambda_c, 0.0)

    def test_missing_toughness(self, caplog):
        mesh = _enf_mesh()
        mesh.delaminations = (Delamination.strip(MID16, -1.0, 15.0),)
        logging.disable(logging.NOTSET)
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            r = FESolver(mesh, T800_16, mesh.porosity_field, ply_angles='UD').solve(
                'flexure_3pt', applied_load=ENF_LOAD)
        assert r.crack_front is not None and r.crack_front.G_c is None
        assert r.fracture_load_factor is None and r.fracture_governing_node is None
        assert "No interlaminar toughness" in caplog.text


def enf_front_stub() -> CrackFront:
    z = np.zeros(1)
    return CrackFront(node=np.zeros(1, int), duplicate=np.zeros(1, int),
                      pair=np.zeros(1, int), interface=np.zeros(1, int),
                      behind=np.zeros(1, int), x=z, y=z, z=z,
                      direction=np.array([[1.0, 0.0]]), delta_a=np.ones(1),
                      width=np.ones(1), force_local=np.zeros((1, 3)),
                      opening_local=np.zeros((1, 3)), G_I=z, G_II=z, G_III=z,
                      G_T=z, psi=np.full(1, np.nan), Vp=z)


# --------------------------------------------------------------------------
# Thermal prestress
# --------------------------------------------------------------------------

class TestThermalLoadFactor:
    """Cross-ply [0/90/0/90]s AS4/3501-6 (it has CTEs) in three-point
    bending with a delamination at a 0/90 interface, cooled by 100 K."""

    MAT = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=8, G_Ic=0.08,
                              G_IIc=0.6, bk_eta=2.0)
    LAYUP = [0, 90, 0, 90, 90, 0, 90, 0]

    @pytest.fixture(scope='class')
    @classmethod
    def solved(cls):
        pf = PorosityField(cls.MAT, 0.0)
        mesh = CompositeMesh.for_flexure(
            pf, cls.MAT, span=20.0, width=4.0, overhang=2.0, load_width=0.0, nz=8,
            ny=2, element_length=0.5, ply_angles=cls.LAYUP,
            delaminations=[Delamination.strip(1, -1.0, 8.0)])
        solver = FESolver(mesh, cls.MAT, pf, ply_angles=cls.LAYUP)
        out = {'mesh': mesh}
        for contact in (False, True):
            out[contact] = (
                solver.solve('flexure_3pt', applied_load=-20.0, crack_contact=contact),
                solver.solve('thermal', delta_T=-100.0, crack_contact=contact),
                solver.solve('flexure_3pt', applied_load=-20.0, delta_T=-100.0,
                             crack_contact=contact))
        return out

    def test_quadratic_load_factor(self, solved):
        """Without contact the thermal and mechanical states superpose
        exactly, so the combined solve's ``lambda_c`` is the root of
        ``G_T(lam) = G_c(psi(lam))`` on ``u_T + lam u_M`` of the two separate
        solves."""
        mesh = solved['mesh']
        mech, thermal, combined = solved[False]
        assert combined.load_factor_basis == 'mechanical_with_residual'
        assert thermal.load_factor_basis == 'delta_T'
        f_M = mech.energy_release_rates(mesh)
        f_T = thermal.energy_release_rates(mesh)
        lam = fracture_load_factors(f_M, self.MAT, prestress=f_T).lambda_c
        np.testing.assert_allclose(combined.crack_front.lambda_c, lam, rtol=1e-10)
        assert combined.fracture_load_factor != pytest.approx(mech.fracture_load_factor,
                                                              rel=0.01)
        for i, lam_i in enumerate(lam):
            scaled = dataclasses.replace(f_M, force_local=lam_i * f_M.force_local,
                                         opening_local=lam_i * f_M.opening_local)
            at = fracture_load_factors(scaled, self.MAT, prestress=f_T)
            assert at.G_T[i] == pytest.approx(at.G_c[i], rel=1e-8)
            assert at.lambda_c[i] == pytest.approx(1.0, rel=1e-8)
        # The thermal state alone scales with delta_T.
        np.testing.assert_allclose(
            thermal.crack_front.lambda_c,
            np.sqrt(thermal.crack_front.G_c / thermal.crack_front.G_T), rtol=1e-12)

    def test_contact_of_the_total_state(self, solved):
        mesh = solved['mesh']
        for r in solved[True]:
            assert r.crack_contact and r.contact_converged
            scale = float(np.abs(r.displacement).max())
            assert _openings(mesh, r).min() >= -1e-9 * scale
        assert solved[True][2].contact_pairs != solved[True][0].contact_pairs


# --------------------------------------------------------------------------
# Penny-shaped crack in an isotropic block
# --------------------------------------------------------------------------

def _coupon(material, pf, delaminations=None):
    """A 4-ply 8 x 4 mm coupon, 1 mm elements in plane."""
    mesh = CompositeMesh(pf, material, nx=8, ny=4, nz=4, delaminations=delaminations)
    mesh.L_x, mesh.L_y = 8.0, 4.0
    mesh.generate_mesh()
    return mesh


def _isotropic(t_ply: float, E: float = 10000.0, nu: float = 0.3) -> MaterialProperties:
    G = E / (2.0 * (1.0 + nu))
    return MaterialProperties(
        E11=E, E22=E, E33=E, G12=G, G13=G, G23=G, nu12=nu, nu13=nu, nu23=nu,
        sigma_1c=100.0, sigma_1t=100.0, sigma_2t=100.0, sigma_2c=100.0,
        tau_12=100.0, tau_ilss=100.0, t_ply=t_ply, n_plies=2,
        matrix_modulus=E, matrix_poisson=nu, fiber_modulus=E,
        fiber_volume_fraction=0.5)


def _graded(n_fine: int, h: float, n_coarse: int, length: float) -> np.ndarray:
    """``n_fine`` steps of ``h``, then ``n_coarse`` growing steps to ``length``."""
    x0 = n_fine * h
    lo, hi = 1.0 + 1e-9, 3.0
    for _ in range(100):
        q = 0.5 * (lo + hi)
        if h * q * (q ** n_coarse - 1.0) / (q - 1.0) > length - x0:
            hi = q
        else:
            lo = q
    steps = h * q ** np.arange(1, n_coarse + 1)
    steps *= (length - x0) / steps.sum()
    return np.concatenate([np.arange(n_fine + 1) * h, x0 + np.cumsum(steps)])


@pytest.fixture(scope='module')
def penny():
    """Quarter model (symmetry planes x = 0 and y = 0) of a 4a x 4a x 8a
    block, crack radius a = 4 mm at mid-height, a/8 elements over 1.5 a
    in plane and next to the crack plane, graded beyond; remote tension
    on the top and bottom faces; conjugate gradients."""
    a, h = 4.0, 0.5
    xg = _graded(12, h, 6, 4 * a)
    zh = _graded(3, h, 6, 4 * a)
    zg = np.concatenate([4 * a - zh[::-1], 4 * a + zh[1:]])
    mat = _isotropic(4 * a)
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh(pf, mat, nx=xg.size - 1, ny=xg.size - 1, nz=zg.size - 1,
                         ply_angles='UD',
                         delaminations=[Delamination.ellipse(0, (0.0, 0.0), (a, a))])
    mesh.L_x = mesh.L_y = 4 * a
    mesh.generate_mesh()

    def remap(v, grid):
        return grid[np.rint(v / v.max() * (grid.size - 1)).astype(int)]

    # The same structured grid, graded: originals and duplicates alike.
    mesh.nodes = np.column_stack([remap(mesh.nodes[:, 0], xg), remap(mesh.nodes[:, 1], xg),
                                  remap(mesh.nodes[:, 2], zg)])
    K = GlobalAssembler(mesh, mat, pf).stiffness()
    handler = BoundaryHandler(mesh)
    constrained = {3 * int(n): 0.0 for n in handler._face_nodes('x_min')}
    constrained.update({3 * int(n) + 1: 0.0 for n in handler._face_nodes('y_min')})
    n0 = mesh.n_structured_nodes
    far = np.flatnonzero(np.all(np.abs(mesh.nodes[:n0] - [4 * a, 4 * a, 4 * a]) < 1e-9, axis=1))
    constrained[3 * int(far[0]) + 2] = 0.0          # the z translation
    sigma = 1.0
    w = np.zeros(xg.size)
    w[:-1] += 0.5 * np.diff(xg)
    w[1:] += 0.5 * np.diff(xg)
    F = np.zeros(K.shape[0])
    for face, sign in (('z_max', 1.0), ('z_min', -1.0)):
        nodes = handler._face_nodes(face)
        i = np.searchsorted(xg, mesh.nodes[nodes, 0] - 1e-9)
        j = np.searchsorted(xg, mesh.nodes[nodes, 1] - 1e-9)
        F[3 * nodes + 2] += sign * sigma * w[i] * w[j]
    u, r = _solve(mesh, K, constrained, F, solver='cg')
    front = vcct_energy_release_rates(mesh, u, r)
    G_ref = 4.0 * sigma ** 2 * a * (1.0 - 0.3 ** 2) / (math.pi * mat.E11)
    return dict(front=front, a=a, G_ref=G_ref, mesh=mesh)


class TestPennyCrack:
    """``G = (1 - nu^2) K_I^2 / E`` with ``K_I = 2 sigma sqrt(a / pi)``. The
    crack tip runs through the bonded front nodes of the staircase, between
    ``a`` and ``a + sqrt(2) h`` from the centre (mean 1.058 a here), and the
    closed form scales with ``a``. Measured: the front mean of ``G_T`` is
    1.061 of the closed form at the nominal radius and 1.003 at the mean
    front radius; around the staircase it ranges 0.785 to 1.434 of it."""

    def test_front_average(self, penny):
        front, G_ref = penny['front'], penny['G_ref']
        mean = float(front.G_T.mean())
        r_mean = float(np.hypot(front.x, front.y).mean())
        assert mean == pytest.approx(G_ref, rel=0.10)
        assert mean == pytest.approx(G_ref * r_mean / penny['a'], rel=0.05)
        assert 0.6 < front.G_T.min() / G_ref and front.G_T.max() / G_ref < 1.6

    def test_mode_I_and_the_staircase_frame(self, penny):
        front = penny['front']
        # The front is the mesh's crack_front_pairs(), in the same order.
        originals, duplicates = penny['mesh'].crack_front_pairs()
        np.testing.assert_array_equal(front.node, originals)
        np.testing.assert_array_equal(front.duplicate, duplicates)
        assert np.min(front.G_I / front.G_T) > 0.999
        # Eight grid directions, the radial one on the axes.
        on_x = np.abs(front.y) < 1e-9
        np.testing.assert_array_equal(front.direction[on_x], [[1.0, 0.0]])
        np.testing.assert_allclose(np.linalg.norm(front.direction, axis=1), 1.0)
        np.testing.assert_allclose(front.width[on_x], 0.25)    # half the cell at the plane
        np.testing.assert_allclose(front.area[~on_x & (np.abs(front.x) > 1e-9)], 0.25)


# --------------------------------------------------------------------------
# Inputs, exports and no-delamination solves
# --------------------------------------------------------------------------

class TestToughnessInputs:
    """Validation of the inputs is in tests/test_materials.py."""

    def test_toughness_changes_no_stiffness_result(self):
        """The fields enter the cache keys (every material field does) but
        no stiffness, stress or load factor of the existing outputs."""
        base = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)
        tough = dataclasses.replace(base, **TOUGHNESS)
        pf = PorosityField(base, 0.02)
        mesh = _coupon(base, pf)
        r0 = FESolver(mesh, base, pf).solve('compression')
        r1 = FESolver(mesh, tough, PorosityField(tough, 0.02)).solve('compression')
        np.testing.assert_array_equal(r0.displacement, r1.displacement)
        assert r0.knockdown == r1.knockdown
        assert r0.first_ply_failure_load_factor == r1.first_ply_failure_load_factor
        assert r1.crack_front is None


class TestExports:
    def test_json_fracture_block(self, enf, tmp_path):
        import jsonschema
        r = enf[0]
        path = tmp_path / 'enf.json'
        FESolver.export_results(r, path)
        text = path.read_text(encoding='utf-8')
        assert 'NaN' not in text and 'Infinity' not in text
        doc = json.loads(text)
        jsonschema.validate(instance=doc, schema=json.loads(
            open(_SCHEMA, encoding='utf-8').read()))
        block = doc['fracture']
        assert block['crack_contact'] is True
        assert block['contact_pairs'] == r.contact_pairs
        assert block['fracture_load_factor'] == pytest.approx(r.fracture_load_factor)
        assert block['governing_node'] == r.fracture_governing_node
        front = block['front']
        assert block['n_front_nodes'] == len(r.crack_front) == len(front['G_T_N_per_mm'])
        assert front['lambda_c'] == pytest.approx(list(r.crack_front.lambda_c))
        # Infinite and undefined values are null.
        free = dataclasses.replace(r, fracture_load_factor=math.inf)
        FESolver.export_results(free, tmp_path / 'inf.json')
        assert json.loads((tmp_path / 'inf.json').read_text())['fracture'][
            'fracture_load_factor'] is None

    def test_vtu_front_point_data(self, enf, tmp_path):
        import base64
        import xml.etree.ElementTree as ET
        r, mesh = enf[0], enf['mesh']
        path = tmp_path / 'enf.vtu'
        r.to_vtu(mesh, path, encoding='base64', nodal=False)
        arrays = ET.parse(path).getroot().find('UnstructuredGrid/Piece/PointData')
        (node,) = [da for da in arrays.findall('DataArray') if da.get('Name') == 'G_T']
        assert node.get('type') == 'Float64'
        G_T = np.frombuffer(base64.b64decode(node.text.strip())[8:], '<f8')
        front = r.crack_front
        np.testing.assert_array_equal(G_T[front.node], front.G_T)
        np.testing.assert_array_equal(G_T[front.duplicate], front.G_T)
        others = np.ones(G_T.size, dtype=bool)
        others[front.node] = others[front.duplicate] = False
        assert np.isnan(G_T[others]).all()
        r.to_vtk(mesh, tmp_path / 'enf.vtk')        # legacy ASCII: no NaN fields
        assert 'G_T' not in (tmp_path / 'enf.vtk').read_text()

    def test_no_delamination_no_fracture_output(self, tmp_path):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4, **TOUGHNESS)
        pf = PorosityField(mat, 0.02)
        r = FESolver(_coupon(mat, pf), mat, pf).solve('compression')
        for name in ('crack_front', 'fracture_load_factor', 'fracture_governing_node',
                     'interface_forces', 'crack_contact', 'contact_iterations',
                     'contact_pairs', 'contact_converged'):
            assert getattr(r, name) is None
        details = r.summary().details
        assert details['crack_contact'] is None and details['fracture_load_factor'] is None
        FESolver.export_results(r, tmp_path / 'r.json')
        assert 'fracture' not in json.loads((tmp_path / 'r.json').read_text())

    def test_bonded_duplicated_mesh_has_forces_but_no_front(self):
        """Duplicated interface, every pair bonded: no contact, no front,
        but the tie forces are reported."""
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)
        pf = PorosityField(mat, 0.0)
        mesh = _coupon(mat, pf, [Delamination.strip(1, 0.3, 0.6)])
        r = FESolver(mesh, mat, pf).solve('compression')
        assert r.crack_contact is None and r.crack_front is None
        assert r.interface_forces.shape == (mesh.interface_node_pairs()[0].size, 3)
        assert np.abs(r.interface_forces).max() > 0.0
