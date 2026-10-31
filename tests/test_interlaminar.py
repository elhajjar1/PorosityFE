"""Interlaminar tractions, delamination onset and free-edge delamination.

Delamination plan, phase 1. This checks:

- the optional interlaminar strengths ``sigma_3t`` and ``tau_13``: their
  validation, that every criterion keeps its historical stand-in when they
  are ``None`` (bit for bit for Tsai-Wu and maximum stress), and that each
  criterion and load factor uses them when set;
- the interface traction recovery: exact for fields linear in ``x``, ``y``
  and ``z``, the two sides and their jump, delaminated interfaces;
- free-edge stresses of laminates in tension: the jump shrinking under
  refinement, the sign of the edge stresses of a cross-ply and an angle-ply
  laminate with peaks growing under refinement, and the equilibrium of the
  interface shear with the in-plane ply force of lamination theory;
- the Brewer-Lagace and Ye onset criteria by hand, the averaging window,
  the porosity degradation, the load factor (scaling, thermal prestress)
  and the interior variant;
- O'Brien's free-edge delamination energy release rate against an
  independent lamination-theory calculation and his published
  ``[+-30/+-30/90/90bar]s`` numbers.

Stresses and strengths in MPa, lengths in mm.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    Delamination,
    DelaminationOnset,
    EdgeDelamination,
    FESolver,
    FieldResults,
    InterfaceTractions,
    MaterialProperties,
    PorosityField,
    delamination_onset,
    interface_tractions,
    obrien_edge_delamination,
    rotate_stiffness_3d,
    stress_transformation_3d,
)
from porosity_fe.fe import failure
from porosity_fe.fe.batch import _gauss_point_tables
from porosity_fe.uq import _normalize_uq_spec

T800 = MATERIALS['T800_epoxy']
#: A thin 4-ply T800 laminate for the free-edge solves (h = 1 mm).
MAT4 = dataclasses.replace(T800, n_plies=4, t_ply=0.25)


def _trapezoid(f: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))


def _results(mesh: CompositeMesh, stress_global: np.ndarray, *,
             residual_local: np.ndarray | None = None,
             basis: str = 'mechanical') -> FieldResults:
    """A FieldResults carrying a given Gauss-point stress field."""
    zero = np.zeros_like(stress_global)
    return FieldResults(
        displacement=np.zeros((mesh.n_nodes, 3)), stress_global=stress_global,
        stress_local=zero, strain_global=zero, strain_local=zero,
        max_failure_index=0.0, knockdown=1.0,
        residual_stress_local=residual_local, load_factor_basis=basis)


def _gauss_xyz(mesh: CompositeMesh) -> np.ndarray:
    """``(n_elem, 8, 3)`` Gauss-point coordinates."""
    N, _, _, _ = _gauss_point_tables()
    return np.einsum('gn,enk->egk', N, np.asarray(mesh.nodes)[mesh.elements])


def _small_mesh(n_plies=2, layup=(0, 90), nx=1, ny=1, nz=2, Vp=0.0, material=T800):
    mat = dataclasses.replace(material, n_plies=n_plies)
    pf = PorosityField(mat, Vp)
    return mat, CompositeMesh(pf, mat, nx=nx, ny=ny, nz=nz, ply_angles=list(layup))


def _uniform(mesh: CompositeMesh, *, szz=0.0, tyz=0.0, txz=0.0) -> np.ndarray:
    S = np.zeros((mesh.n_elements, 8, 6))
    S[..., 2], S[..., 3], S[..., 4] = szz, tyz, txz
    return S


# ---------------------------------------------------------------------------
# Optional interlaminar strengths sigma_3t and tau_13
# ---------------------------------------------------------------------------

class TestInterlaminarStrengths:

    @pytest.mark.parametrize('name', ['sigma_3t', 'tau_13'])
    @pytest.mark.parametrize('bad', [0.0, -10.0, float('nan'), float('inf'), True, 'x'])
    def test_validation(self, name, bad):
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(T800, **{name: bad})

    def test_defaults_none_everywhere(self):
        assert all(m.sigma_3t is None and m.tau_13 is None for m in MATERIALS.values())
        assert failure.degraded_interlaminar_strengths(T800, (1, 1, 1), 0.02) == (None, None)

    def test_perturbable_only_when_set(self):
        for name in ('sigma_3t', 'tau_13'):
            assert name in MaterialProperties.PERTURBABLE_FIELDS
            with pytest.raises(ValueError, match='is None'):
                _normalize_uq_spec(T800, {name: 0.1}, None)
        mat = dataclasses.replace(T800, sigma_3t=60.0, tau_13=95.0)
        spec = _normalize_uq_spec(mat, {'sigma_3t': 0.1, 'tau_13': 0.05}, None)
        assert set(spec) == {'sigma_3t', 'tau_13'}

    def test_degraded_by_the_matrix_ratio(self):
        mat = dataclasses.replace(T800, sigma_3t=60.0, tau_13=95.0)
        Vp = 0.03
        st = failure.degraded_strengths(mat, (3, 3, 0.3), Vp)
        Zt, S13 = failure.degraded_interlaminar_strengths(mat, (3, 3, 0.3), Vp)
        r_m = st[2] / mat.sigma_2t
        assert r_m < 1.0
        assert Zt == pytest.approx(60.0 * r_m, rel=1e-12)
        assert S13 == pytest.approx(95.0 * r_m, rel=1e-12)
        assert st[5] / mat.tau_ilss == pytest.approx(r_m, rel=1e-12)

    @staticmethod
    def _random_state(seed=3):
        """Random ply-local stresses on a porous 4-ply mesh, every component
        large enough to govern somewhere, and a random prestress."""
        mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4)
        pf = PorosityField(mat, 0.03, distribution='clustered', seed=1)
        mesh = CompositeMesh(pf, mat, nx=3, ny=2, nz=4, ply_angles=[0, 90, 90, 0])
        rng = np.random.default_rng(seed)
        scale = np.array([600.0, 30.0, 30.0, 40.0, 40.0, 40.0])
        s = rng.normal(size=(mesh.n_elements, 8, 6)) * scale
        pre = rng.normal(size=s.shape) * scale * 0.3
        return mat, mesh, s, pre

    @staticmethod
    def _evaluate(mat, mesh, s, pre, criterion):
        fields: dict = {}
        out = failure.evaluate_failure(s, mesh.porosity, mesh.elements, mat, (1, 1, 1),
                                       criterion, mode_fields=fields)
        lam = failure._element_load_factors(s, mesh.porosity, mesh.elements, mat,
                                            (1, 1, 1), criterion)
        lam_p = failure._element_load_factors(s, mesh.porosity, mesh.elements, mat,
                                              (1, 1, 1), criterion, prestress_local=pre)
        return out, fields, lam, lam_p

    @pytest.mark.parametrize('criterion', ['tsai_wu', 'max_stress'])
    def test_explicit_fallback_values_are_bit_identical(self, criterion):
        """sigma_3t = sigma_2t and tau_13 = tau_12 reproduce the default path
        bit for bit for Tsai-Wu and maximum stress: indices, modes and load
        factors with and without a prestress. Other values change them."""
        mat, mesh, s, pre = self._random_state()
        same = dataclasses.replace(mat, sigma_3t=mat.sigma_2t, tau_13=mat.tau_12)
        a, b = self._evaluate(mat, mesh, s, pre, criterion), \
            self._evaluate(same, mesh, s, pre, criterion)
        assert a[0][0] == b[0][0]
        assert np.array_equal(a[0][1], b[0][1])
        assert a[0][2] == b[0][2] or criterion == 'tsai_wu'
        for k in a[1]:
            assert np.array_equal(a[1][k], b[1][k])
        assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
        for kw in ({'sigma_3t': 0.8 * mat.sigma_2t}, {'tau_13': 0.8 * mat.tau_12}):
            c = self._evaluate(dataclasses.replace(mat, **kw), mesh, s, pre, criterion)
            assert not np.array_equal(a[0][1], c[0][1])
            assert not np.array_equal(a[3], c[3])

    def test_explicit_fallback_values_hashin(self):
        mat, mesh, s, pre = self._random_state()
        same = dataclasses.replace(mat, sigma_3t=mat.sigma_2t, tau_13=mat.tau_ilss)
        a, b = self._evaluate(mat, mesh, s, pre, 'hashin'), \
            self._evaluate(same, mesh, s, pre, 'hashin')
        np.testing.assert_allclose(b[0][1], a[0][1], rtol=1e-12)
        np.testing.assert_allclose(b[1]['delamination'], a[1]['delamination'], rtol=1e-12)
        np.testing.assert_allclose(b[2], a[2], rtol=1e-12)
        np.testing.assert_allclose(b[3], a[3], rtol=1e-12)
        c = self._evaluate(dataclasses.replace(mat, tau_13=0.8 * mat.tau_ilss),
                           mesh, s, pre, 'hashin')
        assert np.all(c[1]['delamination'] >= a[1]['delamination'])
        assert not np.allclose(c[1]['delamination'], a[1]['delamination'])

    def test_opt_in_reaches_every_criterion(self):
        mat = dataclasses.replace(T800, sigma_3t=40.0, tau_13=60.0)
        st = failure.degraded_strengths(mat, (1, 1, 1), 0.0)
        inter = failure.degraded_interlaminar_strengths(mat, (1, 1, 1), 0.0)
        assert inter == (40.0, 60.0)
        Xt, Xc, Yt, Yc, S12, S23 = st
        s = np.zeros((2, 6))
        s[0, 2] = 20.0       # sigma_33 tension
        s[1, 4] = 30.0       # tau_13
        h = failure.evaluate_hashin(s, st, interlaminar=inter)
        np.testing.assert_allclose(h['delamination'], [(20 / 40) ** 2, (30 / 60) ** 2])
        h0 = failure.evaluate_hashin(s, st)
        np.testing.assert_allclose(h0['delamination'], [(20 / Yt) ** 2, (30 / S23) ** 2])
        m = failure.evaluate_max_stress(s, st, interlaminar=inter)
        assert m['matrix_t'][0] == pytest.approx(20 / 40)
        assert m['shear'][1] == pytest.approx(30 / 60)
        m0 = failure.evaluate_max_stress(s, st)
        assert m0['matrix_t'][0] == pytest.approx(20 / Yt)
        assert m0['shear'][1] == pytest.approx(30 / S12)
        tw = failure.evaluate_tsai_wu(s, st, 0, 0.0, None, interlaminar=inter)
        F3 = 1 / 40 - 1 / Yc
        F33 = 1 / (40 * Yc)
        assert tw[0] == pytest.approx(F3 * 20 + F33 * 20 ** 2, rel=1e-12)
        assert tw[1] == pytest.approx((30 / 60) ** 2, rel=1e-12)
        tw0 = failure.evaluate_tsai_wu(s, st, 0, 0.0, None)
        assert tw0[1] == pytest.approx((30 / S12) ** 2, rel=1e-12)

    def test_tsai_wu_interactions_follow_Zt(self):
        st = failure.degraded_strengths(T800, (1, 1, 1), 0.0)
        co = failure._tsai_wu_coefficients(st, None, (40.0, None))
        F11, F22, F33, F13, F23 = co[3], co[4], co[5], co[10], co[11]
        assert F13 == pytest.approx(-0.5 * math.sqrt(F11 * F33), rel=1e-12)
        assert F23 == pytest.approx(-0.5 * math.sqrt(F22 * F33), rel=1e-12)
        assert failure._tsai_wu_coefficients(st, None, (None, None)) == \
            failure._tsai_wu_coefficients(st, None)

    @pytest.mark.parametrize('criterion', ['tsai_wu', 'hashin', 'max_stress'])
    def test_load_factors_use_them(self, criterion):
        """The plain and the prestressed load factor reach index 1 with the
        opt-in strengths."""
        mat = dataclasses.replace(T800, sigma_3t=40.0, tau_13=60.0)
        st = failure.degraded_strengths(mat, (1, 1, 1), 0.0)
        inter = failure.degraded_interlaminar_strengths(mat, (1, 1, 1), 0.0)
        s_m = np.array([[5.0, 3.0, 12.0, 4.0, 15.0, 2.0]])
        s_th = np.array([[-2.0, 6.0, -3.0, 1.0, 5.0, -1.0]])

        def index(s):
            if criterion == 'tsai_wu':
                return failure.evaluate_tsai_wu(s, st, 0, 0.0, None, interlaminar=inter)[0]
            fn = failure.evaluate_hashin if criterion == 'hashin' else failure.evaluate_max_stress
            return fn(s, st, interlaminar=inter)['max_fi'][0]

        lam = failure._point_load_factors(s_m, st, criterion, None, inter)[0]
        assert index(lam * s_m) == pytest.approx(1.0, rel=1e-9)
        lam_p = failure._point_load_factors_prestressed(
            s_th, s_m, st, criterion, None, interlaminar=inter)[0]
        assert index(s_th + lam_p * s_m) == pytest.approx(1.0, rel=1e-9)
        assert index(s_th + 0.99 * lam_p * s_m) < 1.0


# ---------------------------------------------------------------------------
# Traction recovery
# ---------------------------------------------------------------------------

class TestTractionRecovery:

    def test_uniform_field(self):
        mat, mesh = _small_mesh(n_plies=4, layup=(0, 90, 90, 0), nx=2, ny=3, nz=4)
        res = _results(mesh, _uniform(mesh, szz=30.0, tyz=10.0, txz=40.0))
        tr = interface_tractions(res, mesh)
        assert isinstance(tr, InterfaceTractions)
        assert tr.interfaces == (0, 1, 2)
        assert tr.shape == (4, 3) and len(tr) == 3 * 12
        assert tr.valid.all()
        np.testing.assert_allclose(tr.traction, np.tile([40.0, 10.0, 30.0], (36, 1)))
        np.testing.assert_allclose(tr.jump, 0.0, atol=1e-12)
        np.testing.assert_allclose(tr.shear, math.hypot(40.0, 10.0))
        np.testing.assert_allclose(tr.z, np.repeat([0.183, 2 * 0.183, 3 * 0.183], 12))
        assert np.array_equal(tr.node[:12], 12 + np.arange(12))

    def test_linear_field_is_exact_and_maps_to_the_grid(self):
        """A traction linear in x, y and z is recovered exactly at every node,
        continuous across the interface (zero jump)."""
        mat, mesh = _small_mesh(n_plies=4, layup=(0, 45, -45, 90), nx=3, ny=2, nz=8)
        xyz = _gauss_xyz(mesh)
        S = np.zeros((mesh.n_elements, 8, 6))
        coef = {2: (5.0, 0.3, -0.7, 2.0), 3: (-1.0, 0.1, 0.2, 0.5), 4: (2.0, -0.2, 0.4, -3.0)}
        for slot, (a, bx, by, bz) in coef.items():
            S[..., slot] = a + bx * xyz[..., 0] + by * xyz[..., 1] + bz * xyz[..., 2]
        tr = interface_tractions(_results(mesh, S), mesh)
        for col, slot in enumerate((4, 3, 2)):
            a, bx, by, bz = coef[slot]
            expect = a + bx * tr.x + by * tr.y + bz * tr.z
            np.testing.assert_allclose(tr.below[:, col], expect, atol=1e-9)
            np.testing.assert_allclose(tr.above[:, col], expect, atol=1e-9)
        np.testing.assert_allclose(tr.jump, 0.0, atol=1e-9)
        grid = tr.grid(tr.traction[:, 2], 1)
        assert grid.shape == (3, 4)
        X = tr.grid(tr.x, 1)
        Y = tr.grid(tr.y, 1)
        np.testing.assert_allclose(X[0], np.linspace(0, mesh.L_x, 4))
        np.testing.assert_allclose(Y[:, 0], np.linspace(0, mesh.L_y, 3))
        with pytest.raises(ValueError, match='not covered'):
            tr.grid(tr.x, 7)

    def test_two_sides_and_jump(self):
        mat, mesh = _small_mesh(n_plies=2, nx=2, ny=2, nz=4)
        S = np.zeros((mesh.n_elements, 8, 6))
        layer = np.repeat(np.arange(4), 4)
        S[..., 2] = np.where(layer < 2, 1.0, 3.0)[:, None]
        tr = interface_tractions(_results(mesh, S), mesh)
        assert tr.interfaces == (0,)
        np.testing.assert_allclose(tr.below[:, 2], 1.0)
        np.testing.assert_allclose(tr.above[:, 2], 3.0)
        np.testing.assert_allclose(tr.traction[:, 2], 2.0)
        np.testing.assert_allclose(tr.jump[:, 2], 2.0)
        np.testing.assert_allclose(tr.jump_norm, 2.0)

    def test_only_node_layer_interfaces(self):
        mat, mesh = _small_mesh(n_plies=4, layup=(0, 90, 90, 0), nz=2)
        res = _results(mesh, _uniform(mesh, szz=1.0))
        assert interface_tractions(res, mesh).interfaces == (1,)
        with pytest.raises(ValueError, match='not node layers'):
            interface_tractions(res, mesh, interfaces=[0])
        with pytest.raises(ValueError, match='do not exist'):
            interface_tractions(res, mesh, interfaces=[3])
        with pytest.raises(ValueError, match='integers'):
            interface_tractions(res, mesh, interfaces=[1.0])
        mat, mesh3 = _small_mesh(n_plies=4, layup=(0, 90, 90, 0), nz=3)
        with pytest.raises(ValueError, match='no ply interface'):
            interface_tractions(_results(mesh3, _uniform(mesh3)), mesh3)
        with pytest.raises(ValueError, match='Pass the mesh'):
            interface_tractions(res, mesh3)

    def test_delaminated_interface_uses_bonded_faces(self):
        mat = dataclasses.replace(T800, n_plies=2)
        pf = PorosityField(mat, 0.0)
        crack = Delamination.rectangle(0, center=(25.0, 10.0), half_size=(10.0, 4.0))
        mesh = CompositeMesh(pf, mat, nx=10, ny=5, nz=2, ply_angles=[0, 90],
                             delaminations=[crack])
        res = _results(mesh, _uniform(mesh, szz=5.0, txz=8.0))
        tr = interface_tractions(res, mesh)
        bonded = mesh.bonded_pairs(0)
        assert not bonded.all()
        # Inside the crack no face is bonded: no value. Bonded nodes keep one.
        assert not tr.valid[~bonded].any()
        assert np.isnan(tr.traction[~bonded]).all()
        assert tr.valid[bonded].all()
        np.testing.assert_allclose(tr.traction[bonded], np.tile([8.0, 0.0, 5.0], (bonded.sum(), 1)))
        onset = delamination_onset(res, mesh, mat)
        assert not onset.evaluated[~bonded].any()
        assert onset.evaluated[bonded].all()

    def test_does_not_change_the_results(self):
        mat, mesh = _small_mesh(n_plies=2, nx=2, ny=2, nz=2)
        S = _uniform(mesh, szz=3.0, txz=1.0)
        res = _results(mesh, S.copy())
        res.interface_tractions(mesh)
        res.delamination_onset(mesh, mat)
        assert np.array_equal(res.stress_global, S)


# ---------------------------------------------------------------------------
# Free-edge stresses of laminates in tension
# ---------------------------------------------------------------------------

#: (nx, ny, element layers per ply): in-plane and through-thickness
#: refinement together.
_LEVELS = ((10, 10, 1), (10, 20, 2), (10, 40, 4))


def _tension(layup, nx, ny, per_ply, *, material=MAT4, strain=0.001, delta_T=None):
    pf = PorosityField(material, 0.0)
    mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=material.n_plies * per_ply,
                         ply_angles=list(layup))
    mesh.L_x, mesh.L_y = 10.0, 5.0
    mesh.generate_mesh()
    res = FESolver(mesh, material, pf, ply_angles=list(layup)).solve(
        'tension', applied_strain=strain, delta_T=delta_T)
    return mesh, res


@pytest.fixture(scope='module')
def cross_ply():
    return [_tension([0, 90, 90, 0], *lv) for lv in _LEVELS]


@pytest.fixture(scope='module')
def angle_ply():
    return [_tension([45, -45, -45, 45], *lv) for lv in _LEVELS]


def _profile(mesh, tr, k):
    """Rows of interface k along y at mid-length, sorted by y."""
    sel = np.flatnonzero((tr.interface == k) & np.isclose(tr.x, mesh.L_x / 2))
    return sel[np.argsort(tr.y[sel])]


def _clt_ply_force(layup, ply, strain=0.001, material=MAT4):
    """In-plane force per width (N/mm) of one ply of a laminate under
    uniaxial strain, N_x free: [N_xx, N_yy, N_xy] from lamination theory."""
    C = material.get_stiffness_matrix()
    idx = [0, 1, 5]

    def qbar(a):
        Cr = rotate_stiffness_3d(C, math.radians(a), axis='z')
        return Cr[np.ix_(idx, idx)] - np.outer(Cr[idx, 2], Cr[idx, 2]) / Cr[2, 2]

    A = sum(qbar(a) * material.t_ply for a in layup)
    eps = np.r_[strain, np.linalg.solve(A[1:, 1:], -A[1:, 0] * strain)]
    return qbar(layup[ply]) @ eps * material.t_ply


class TestFreeEdge:

    def test_jump_shrinks_under_refinement(self, cross_ply):
        """Relative RMS jump near the free edge (0.5 to 2 mm in from it)
        falls as the mesh is refined in-plane and through the thickness."""
        rel = {0: [], 1: []}
        for mesh, res in cross_ply:
            tr = res.interface_tractions(mesh)
            for k in rel:
                sel = (tr.interface == k) & (tr.x > 2.0) & (tr.x < 8.0) \
                    & (tr.y > mesh.L_y - 2.0) & (tr.y < mesh.L_y - 0.5)
                j = np.sqrt(np.mean(np.sum(tr.jump[sel] ** 2, axis=1)))
                t = np.sqrt(np.mean(np.sum(tr.traction[sel] ** 2, axis=1)))
                rel[k].append(j / t)
        for k, r in rel.items():
            assert r[0] > r[1] > r[2], (k, r)
            assert r[2] < 0.5 * r[0], (k, r)

    def test_cross_ply_sigma_zz_tensile_at_the_free_edge(self, cross_ply):
        """[0/90]s: the 0 deg plies are stretched across the width, so the
        moment of the in-plane force they shed at the free edge pulls the
        interfaces open there (sigma_zz > 0), and the edge value grows with
        refinement (a singular field)."""
        edge = {0: [], 1: []}
        for mesh, res in cross_ply:
            tr = res.interface_tractions(mesh)
            for k in edge:
                rows = _profile(mesh, tr, k)
                szz = tr.traction[rows, 2]
                assert szz[-1] > 0.0
                assert np.argmax(szz) == len(szz) - 1          # peak at the edge
                edge[k].append(szz[-1])
            # tau_yz carries the 0 deg ply's transverse force at interface 0.
            rows = _profile(mesh, tr, 0)
            assert tr.traction[rows[-1], 1] > 0.0
        for k, v in edge.items():
            assert v[0] < v[1] < v[2], (k, v)

    def test_angle_ply_tau_xz_sign_and_peak(self, angle_ply):
        """[45/-45]s: the +45 ply's shear force tau_xy t (positive) is shed
        at the faces free of tau_xy through interface shear tau_xz =
        -dN_xy/dy: positive at y = L_y, negative at y = 0, peaking there and
        growing with refinement."""
        peaks = []
        for mesh, res in angle_ply:
            tr = res.interface_tractions(mesh)
            rows = _profile(mesh, tr, 0)
            txz = tr.traction[rows, 0]
            assert txz[-1] > 0.0 > txz[0]
            assert np.argmax(np.abs(txz)) in (0, len(txz) - 1)
            assert txz[-1] == pytest.approx(np.max(txz))
            peaks.append(txz[-1])
            # The interlaminar normal stress is secondary here.
            assert np.max(np.abs(tr.traction[rows, 2])) < 0.5 * np.max(np.abs(txz))
        assert peaks[0] < peaks[1] < peaks[2]

    def test_angle_ply_sign_follows_the_layup(self):
        mesh, res = _tension([-45, 45, 45, -45], *_LEVELS[1])
        tr = res.interface_tractions(mesh)
        rows = _profile(mesh, tr, 0)
        assert tr.traction[rows[-1], 0] < 0.0 < tr.traction[rows[0], 0]

    @pytest.mark.parametrize('case', ['angle', 'cross'])
    def test_equilibrium_with_the_ply_force(self, case, cross_ply, angle_ply):
        """The interface shear integrated from mid-width to the free edge
        equals the in-plane force the ply below sheds there (lamination
        theory in the interior): N_xy of the +45 ply for tau_xz, N_yy of
        the 0 deg ply for tau_yz. First-order convergence: the two finer
        meshes extrapolate (Richardson) to within 2 %."""
        layup, runs, col, comp = (
            ([45, -45, -45, 45], angle_ply, 0, 2) if case == 'angle'
            else ([0, 90, 90, 0], cross_ply, 1, 1))
        target = _clt_ply_force(layup, 0)[comp]
        integrals = []
        for mesh, res in runs:
            tr = res.interface_tractions(mesh)
            rows = _profile(mesh, tr, 0)
            half = tr.y[rows] >= mesh.L_y / 2 - 1e-9
            integrals.append(_trapezoid(tr.traction[rows[half], col], tr.y[rows[half]]))
        assert integrals[0] < integrals[1] < integrals[2] < target
        assert integrals[2] == pytest.approx(target, rel=0.15)
        assert 2 * integrals[2] - integrals[1] == pytest.approx(target, rel=0.02)


# ---------------------------------------------------------------------------
# Onset criteria
# ---------------------------------------------------------------------------

class TestOnset:

    def test_brewer_lagace_by_hand(self):
        mat, mesh = _small_mesh()
        res = _results(mesh, _uniform(mesh, szz=30.0, tyz=10.0, txz=40.0))
        onset = delamination_onset(res, mesh, mat)
        assert isinstance(onset, DelaminationOnset)
        hand = (30 / mat.sigma_2t) ** 2 + (40 / mat.tau_ilss) ** 2 + (10 / mat.tau_ilss) ** 2
        np.testing.assert_allclose(onset.index, hand, rtol=1e-12)
        np.testing.assert_allclose(onset.load_factor, 1 / math.sqrt(hand), rtol=1e-12)
        np.testing.assert_allclose(onset.Zt, mat.sigma_2t)
        np.testing.assert_allclose(onset.S13, mat.tau_ilss)
        np.testing.assert_allclose(onset.S23, mat.tau_ilss)
        assert onset.x_avg == pytest.approx(mat.t_ply)
        assert onset.max_index == pytest.approx(hand, rel=1e-12)
        assert onset.onset_load_factor == pytest.approx(1 / math.sqrt(hand), rel=1e-12)
        assert onset.critical_interface == 0
        assert onset.load_factor_basis == 'mechanical'
        summary = onset.summary()
        json.dumps(summary, allow_nan=False)                   # strict JSON
        assert summary['interface_max_index'] == {'0': pytest.approx(hand)}
        # The one-element plate has no interior.
        assert summary['interior_max_index'] is None and not onset.interior.any()

    def test_opt_in_strengths(self):
        mat, mesh = _small_mesh(material=dataclasses.replace(T800, sigma_3t=50.0, tau_13=70.0))
        res = _results(mesh, _uniform(mesh, szz=30.0, tyz=10.0, txz=40.0))
        onset = delamination_onset(res, mesh, mat)
        hand = (30 / 50) ** 2 + (40 / 70) ** 2 + (10 / mat.tau_ilss) ** 2
        np.testing.assert_allclose(onset.index, hand, rtol=1e-12)

    def test_load_factor_scales_with_the_load(self):
        mat, mesh = _small_mesh(nx=3, ny=2, nz=4)
        S = _uniform(mesh, szz=12.0, tyz=-4.0, txz=9.0)
        a = delamination_onset(_results(mesh, S), mesh, mat)
        b = delamination_onset(_results(mesh, 2.5 * S), mesh, mat)
        np.testing.assert_allclose(b.index, 2.5 ** 2 * a.index, rtol=1e-12)
        np.testing.assert_allclose(b.load_factor, a.load_factor / 2.5, rtol=1e-12)

    def test_compression_term(self):
        mat, mesh = _small_mesh()
        res = _results(mesh, _uniform(mesh, szz=-100.0))
        onset = delamination_onset(res, mesh, mat)
        np.testing.assert_allclose(onset.index, 0.0)
        assert np.isinf(onset.load_factor).all()
        with_zc = delamination_onset(res, mesh, mat, Zc=250.0)
        np.testing.assert_allclose(with_zc.index, (100 / 250) ** 2, rtol=1e-12)
        np.testing.assert_allclose(with_zc.load_factor, 2.5, rtol=1e-12)
        with pytest.raises(ValueError, match='Zc'):
            delamination_onset(res, mesh, mat, criterion='ye', Zc=250.0)
        with pytest.raises(ValueError, match='Zc'):
            delamination_onset(res, mesh, mat, Zc=0.0)

    def test_ye_works_in_the_ply_axes(self):
        """At a 0/90 interface a pure tau_xz is tau_13 of the 0 deg ply and
        -tau_23 of the 90 deg ply; Ye takes the worse ply, Brewer-Lagace
        pairs tau_xz with S_13."""
        mat, mesh = _small_mesh(material=dataclasses.replace(T800, tau_13=120.0))
        res = _results(mesh, _uniform(mesh, txz=40.0))
        bl = delamination_onset(res, mesh, mat)
        ye = delamination_onset(res, mesh, mat, criterion='ye')
        np.testing.assert_allclose(bl.index, (40 / 120) ** 2, rtol=1e-12)
        np.testing.assert_allclose(ye.index, (40 / mat.tau_ilss) ** 2, rtol=1e-12)
        np.testing.assert_allclose(ye.load_factor, mat.tau_ilss / 40, rtol=1e-12)
        # With the default strengths (S_13 = S_23) the two coincide.
        mat0, mesh0 = _small_mesh(layup=(30, -60))
        res0 = _results(mesh0, _uniform(mesh0, szz=20.0, tyz=15.0, txz=-25.0))
        np.testing.assert_allclose(
            delamination_onset(res0, mesh0, mat0, criterion='ye').index,
            delamination_onset(res0, mesh0, mat0).index, rtol=1e-12)

    def test_averaging_window(self):
        """sigma_zz linear in y: the window average is the field at the
        window centre, which shifts inward at the plate edges."""
        mat, mesh = _small_mesh(nx=4, ny=10, nz=2)
        xyz = _gauss_xyz(mesh)
        S = np.zeros((mesh.n_elements, 8, 6))
        S[..., 2] = 10.0 + 3.0 * xyz[..., 1]
        res = _results(mesh, S)
        w = 4.0
        onset = delamination_onset(res, mesh, mat, x_avg=w)
        y = onset.y
        centre = np.clip(y, w / 2, mesh.L_y - w / 2)
        np.testing.assert_allclose(onset.traction[:, 2], 10.0 + 3.0 * centre, rtol=1e-12)
        point = delamination_onset(res, mesh, mat, x_avg=0.0)
        np.testing.assert_allclose(point.traction[:, 2], 10.0 + 3.0 * y, rtol=1e-12)
        whole = delamination_onset(res, mesh, mat, x_avg=1e3)
        np.testing.assert_allclose(whole.traction[:, 2], 10.0 + 1.5 * mesh.L_y, rtol=1e-12)
        # A window narrower than an element integrates the bilinear field exactly.
        narrow = delamination_onset(res, mesh, mat, x_avg=0.5)
        np.testing.assert_allclose(narrow.traction[:, 2],
                                   10.0 + 3.0 * np.clip(y, 0.25, mesh.L_y - 0.25), rtol=1e-12)
        for bad in (-1.0, float('nan'), True):
            with pytest.raises(ValueError, match='x_avg'):
                delamination_onset(res, mesh, mat, x_avg=bad)

    def test_porosity_degrades_the_strengths(self):
        mat, mesh = _small_mesh(nx=2, ny=2, nz=2, Vp=0.03)
        res = _results(mesh, _uniform(mesh, szz=20.0, txz=30.0))
        onset = delamination_onset(res, mesh, mat)
        pristine_mat, pristine_mesh = _small_mesh(nx=2, ny=2, nz=2)
        ref = delamination_onset(_results(pristine_mesh, _uniform(pristine_mesh, szz=20.0, txz=30.0)),
                                 pristine_mesh, pristine_mat)
        for row in range(len(onset)):
            st = failure.degraded_strengths(mat, mesh.porosity_field.void_shape_radii,
                                            float(onset.Vp[row]))
            assert onset.Zt[row] == pytest.approx(st[2])
            assert onset.S13[row] == pytest.approx(st[5])
        assert (onset.Vp > 0).all()
        assert (onset.index > ref.index).all()

    def test_thermal_prestress_by_hand(self):
        """With a residual sigma_zz in compression the mechanical tension
        first has to cancel it: index((s_th + lam s_m)) = 1 solved branch by
        branch."""
        mat, mesh = _small_mesh(layup=(0, 90))
        res_g = _uniform(mesh, szz=-10.0, txz=5.0)          # residual, global
        mech = _uniform(mesh, szz=25.0, txz=10.0)
        angles = np.asarray(mesh.ply_angles, dtype=float)
        T = np.stack([stress_transformation_3d(math.radians(a), axis='z') for a in angles])
        res_local = np.einsum('eij,egj->egi', T, res_g)
        res = _results(mesh, res_g + mech, residual_local=res_local,
                       basis='mechanical_with_residual')
        onset = delamination_onset(res, mesh, mat)
        Zt, S = mat.sigma_2t, mat.tau_ilss

        def index(lam):
            szz, txz = -10.0 + 25.0 * lam, 5.0 + 10.0 * lam
            return (max(szz, 0.0) / Zt) ** 2 + (txz / S) ** 2

        lam = onset.load_factor[0]
        assert index(lam) == pytest.approx(1.0, rel=1e-10)
        assert index(0.999 * lam) < 1.0
        # Tension branch: ((25 lam - 10)/Zt)^2 + ((10 lam + 5)/S)^2 = 1.
        a = (25 / Zt) ** 2 + (10 / S) ** 2
        b = 2 * (-250 / Zt ** 2 + 50 / S ** 2)
        c = (10 / Zt) ** 2 + (5 / S) ** 2 - 1
        assert lam == pytest.approx((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a), rel=1e-10)
        np.testing.assert_allclose(onset.index, index(1.0), rtol=1e-12)
        # The prestress alone at onset: factor 0.
        big = _results(mesh, 20 * res_g + mech, residual_local=20 * res_local,
                       basis='mechanical_with_residual')
        assert delamination_onset(big, mesh, mat).onset_load_factor == 0.0

    def test_thermal_prestress_fe(self):
        """Combined tension + cool-down: the onset factor is the root of the
        superposed thermal and mechanical tractions; the thermal solve alone
        scales the temperature change."""
        mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4, t_ply=0.25)
        layup = [0, 90, 90, 0]
        lv = (6, 12, 1)
        mesh, comb = _tension(layup, *lv, material=mat, strain=0.004, delta_T=-150.0)
        _, mech = _tension(layup, *lv, material=mat, strain=0.004)
        solver = FESolver(mesh, mat, mesh.porosity_field, ply_angles=layup)
        th = solver.solve('thermal', delta_T=-150.0)
        on_c = delamination_onset(comb, mesh, mat)
        on_m = delamination_onset(mech, mesh, mat)
        on_t = delamination_onset(th, mesh, mat)
        assert on_c.load_factor_basis == 'mechanical_with_residual'
        assert on_t.load_factor_basis == 'delta_T'
        np.testing.assert_allclose(on_c.traction, on_t.traction + on_m.traction, atol=1e-8)
        ok = on_t.evaluated & np.isfinite(on_t.load_factor)
        np.testing.assert_allclose(on_t.load_factor[ok], 1 / np.sqrt(on_t.index[ok]), rtol=1e-12)
        st = failure.degraded_strengths(mat, (1, 1, 1), 0.0)
        Zt, S = st[2], st[5]
        # Rows the load drives (in the interior of the cross-ply the
        # mechanical traction is round-off and its factor meaningless).
        rows = np.flatnonzero(on_c.evaluated & (on_c.load_factor > 0)
                              & (on_c.load_factor < 50))
        assert rows.size > 20
        for r in rows[::3]:
            t = on_t.traction[r] + on_c.load_factor[r] * on_m.traction[r]
            idx = (max(t[2], 0.0) / Zt) ** 2 + (t[0] / S) ** 2 + (t[1] / S) ** 2
            assert idx == pytest.approx(1.0, rel=1e-8)
        # The cool-down lowers the onset load of this cross-ply.
        assert on_c.onset_load_factor < on_m.onset_load_factor

    def test_interior_variant(self, cross_ply):
        mesh, res = cross_ply[1]
        onset = res.delamination_onset(mesh, MAT4)
        h = mesh.L_z
        assert onset.interior.any() and not onset.interior.all()
        x, y = onset.x[onset.interior], onset.y[onset.interior]
        assert x.min() >= h - 1e-12 and y.min() >= h - 1e-12
        assert y.max() <= mesh.L_y - h + 1e-12
        # The free edge governs the domain, not the interior.
        assert onset.critical_location[1] == pytest.approx(mesh.L_y)
        assert onset.interior_max_index < 0.5 * onset.max_index
        assert onset.interior_onset_load_factor > onset.onset_load_factor
        wide = res.delamination_onset(mesh, MAT4, interior_margin=2.0)
        assert wide.interior.sum() < onset.interior.sum()
        assert onset.interface_max_index()[onset.critical_interface] == onset.max_index
        with pytest.raises(ValueError, match='interior_margin'):
            res.delamination_onset(mesh, MAT4, interior_margin=-1.0)

    def test_averaging_lowers_the_free_edge_index(self, cross_ply):
        mesh, res = cross_ply[2]
        point = res.delamination_onset(mesh, MAT4, x_avg=0.0)
        avg = res.delamination_onset(mesh, MAT4)
        wide = res.delamination_onset(mesh, MAT4, x_avg=4 * MAT4.t_ply)
        assert point.max_index > avg.max_index > wide.max_index

    def test_errors(self):
        mat, mesh = _small_mesh()
        res = _results(mesh, _uniform(mesh, szz=1.0))
        with pytest.raises(ValueError, match='onset criterion'):
            delamination_onset(res, mesh, mat, criterion='hashin')
        assert interface_tractions(res, mesh).interfaces == \
            res.interface_tractions(mesh).interfaces
        np.testing.assert_array_equal(
            res.delamination_onset(mesh, mat, criterion='ye').index,
            delamination_onset(res, mesh, mat, criterion='ye').index)


# ---------------------------------------------------------------------------
# O'Brien free-edge delamination
# ---------------------------------------------------------------------------

def _hand_Q(E1, E2, nu12, G12, theta):
    """Plane-stress reduced stiffness, rotated (classical formulas)."""
    nu21 = nu12 * E2 / E1
    d = 1 - nu12 * nu21
    Q11, Q22, Q12, Q66 = E1 / d, E2 / d, nu12 * E2 / d, G12
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return np.array([
        [Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s * s * c * c + Q22 * s**4,
         (Q11 + Q22 - 4 * Q66) * s * s * c * c + Q12 * (s**4 + c**4),
         (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c],
        [(Q11 + Q22 - 4 * Q66) * s * s * c * c + Q12 * (s**4 + c**4),
         Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s * s * c * c + Q22 * c**4,
         (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3],
        [(Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c,
         (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3,
         (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s * s * c * c + Q66 * (s**4 + c**4)]])


def _hand_modulus(mat, angles, h, coupling):
    n = len(angles)
    z = [-n * h / 2 + k * h for k in range(n + 1)]
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, a in enumerate(angles):
        Q = _hand_Q(mat.E11, mat.E22, mat.nu12, mat.G12, a)
        A += Q * (z[k + 1] - z[k])
        B += Q * (z[k + 1] ** 2 - z[k] ** 2) / 2
        D += Q * (z[k + 1] ** 3 - z[k] ** 3) / 3
    inv = np.linalg.inv(np.block([[A, B], [B, D]])) if coupling else np.linalg.inv(A)
    return 1.0 / (n * h * inv[0, 0])


#: T300/5208 of O'Brien (NASA TM-81940): E11 = 138 GPa, E22 = 15 GPa,
#: G12 = 5.9 GPa, nu12 = 0.21; the 11-ply laminate is 1.51 mm thick. The
#: other constants only complete the dataclass (lamination theory does not
#: read them).
T300_5208 = dataclasses.replace(
    T800, E11=138e3, E22=15e3, E33=15e3, G12=5.9e3, G13=5.9e3, G23=5.9e3,
    nu12=0.21, nu13=0.21, nu23=0.21, t_ply=1.51 / 11, n_plies=11)
EDT_LAYUP = [30, -30, 30, -30, 90, 90, 90, -30, 30, -30, 30]


class TestOBrien:

    @pytest.mark.parametrize('coupling', [False, True])
    def test_against_hand_lamination_theory(self, coupling):
        layup = [25, -25, 90, 0, 0, 90, -25, 25]
        r = obrien_edge_delamination(T800, layup, [1, 5], 0.005, bending_coupling=coupling)
        assert isinstance(r, EdgeDelamination)
        h = T800.t_ply
        assert r.sublaminates == ((0, 1), (2, 3, 4, 5), (6, 7))
        E_lam = _hand_modulus(T800, layup, h, coupling)
        subs = [layup[:2], layup[2:6], layup[6:]]
        E_i = [_hand_modulus(T800, s, h, coupling) for s in subs]
        E_star = (2 * E_i[0] + 4 * E_i[1] + 2 * E_i[2]) / 8
        assert r.E_LAM == pytest.approx(E_lam, rel=1e-9)
        assert r.sublaminate_moduli == pytest.approx(tuple(E_i), rel=1e-9)
        assert r.E_star == pytest.approx(E_star, rel=1e-9)
        assert r.thickness == pytest.approx(8 * h)
        assert r.G == pytest.approx(0.005 ** 2 * 8 * h / 2 * (E_lam - E_star), rel=1e-9)
        assert r.G_simultaneous == pytest.approx(r.G / 2)

    def test_obrien_edge_delamination_specimen(self):
        """[+-30/+-30/90/90bar]s delaminated at both -30/90 interfaces:
        O'Brien reports E* = 0.743 E_LAM without coupling, 0.69 E_LAM with
        it, and G_c = 137 J/m^2 at the onset strain 3470e-6."""
        r = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], 3470e-6)
        assert r.sublaminates == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9, 10))
        assert r.thickness == pytest.approx(1.51)
        assert r.sublaminate_moduli[1] == pytest.approx(15e3, rel=1e-9)   # (90)_3 = E22
        assert r.stiffness_ratio == pytest.approx(0.743, rel=0.005)
        assert r.G == pytest.approx(0.137, rel=0.02)                      # N/mm
        assert r.critical_strain(0.137) == pytest.approx(3470e-6, rel=0.015)
        coupled = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], 3470e-6,
                                           bending_coupling=True)
        assert coupled.stiffness_ratio == pytest.approx(0.69, rel=0.005)
        assert coupled.E_LAM == pytest.approx(r.E_LAM, rel=1e-12)          # symmetric
        assert coupled.G > r.G

    def test_fe_cross_check(self):
        """FE edge delaminations in both -30/90 interfaces of the specimen,
        1.5 mm in from the free edge along the whole length: the energy
        released per crack front (fixed grip, from the effective modulus at
        two delamination widths) matches G / 2, O'Brien's value for two
        interfaces growing together, and the VCCT G_T at mid-length is
        within 7 % (Delta_a = 0.125 mm at a bimaterial interface)."""
        Lx, Ly, nx, ny = 3.0, 6.0, 6, 48
        eps = 1e-3
        dy = Ly / ny

        def run(a):
            pf = PorosityField(T300_5208, 0.0)
            delams = [Delamination.rectangle(k, center=(Lx / 2, Ly), half_size=(Lx, a))
                      for k in (3, 6)]
            mesh = CompositeMesh(pf, T300_5208, nx=nx, ny=ny, nz=11,
                                 ply_angles=EDT_LAYUP, delaminations=delams)
            mesh.L_x, mesh.L_y = Lx, Ly
            mesh.generate_mesh()
            res = FESolver(mesh, T300_5208, pf, ply_angles=EDT_LAYUP).solve(
                'tension', applied_strain=eps)
            return mesh, res

        a1, a2 = 1.5, 1.5 + 2 * dy
        mesh, r1 = run(a1 + 0.5 * dy)          # crack boundary between node lines
        _, r2 = run(a2 + 0.5 * dy)
        volume = Lx * Ly * mesh.L_z
        dE_da = (r2.effective_modulus - r1.effective_modulus) / (a2 - a1)
        G_front = -0.5 * eps ** 2 * volume * dE_da / (2 * Lx)
        ref = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], eps)
        assert G_front == pytest.approx(ref.G_simultaneous, rel=0.02)
        front = r1.crack_front
        mid = np.abs(front.x - Lx / 2) < Lx / 4
        assert mid.sum() >= 4
        assert front.G_T[mid].mean() == pytest.approx(ref.G_simultaneous, rel=0.07)
        assert np.all(front.G_III[mid] < 0.05 * front.G_T[mid])

    def test_scaling_and_limits(self):
        r1 = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], 1e-3)
        r2 = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], 2e-3)
        assert r2.G == pytest.approx(4 * r1.G, rel=1e-12)
        assert r1.load_factor(r2.G) == pytest.approx(2.0, rel=1e-12)
        assert r1.critical_strain(r1.G) == pytest.approx(1e-3, rel=1e-12)
        ud = obrien_edge_delamination(T800, 'UD', [1], 0.01)
        assert ud.G == pytest.approx(0.0, abs=1e-12 * ud.E_LAM)
        assert math.isinf(ud.critical_strain(0.2))
        thin = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], 1e-3, t_ply=0.5 * T300_5208.t_ply)
        assert thin.G == pytest.approx(0.5 * r1.G, rel=1e-12)

    @pytest.mark.parametrize('kwargs, match', [
        ({'interfaces': []}, 'at least one'),
        ({'interfaces': [10]}, 'do not exist'),
        ({'interfaces': [-1]}, 'do not exist'),
        ({'strain': float('nan')}, 'strain'),
        ({'strain': True}, 'strain'),
        ({'t_ply': 0.0}, 't_ply'),
        ({'ply_angles': []}, 'ply_angles'),
    ])
    def test_validation(self, kwargs, match):
        args = {'ply_angles': EDT_LAYUP, 'interfaces': [3, 6], 'strain': 1e-3}
        args.update(kwargs)
        t_ply = args.pop('t_ply', None)
        with pytest.raises(ValueError, match=match):
            obrien_edge_delamination(T300_5208, args['ply_angles'], args['interfaces'],
                                     args['strain'], t_ply=t_ply)
        r = obrien_edge_delamination(T300_5208, EDT_LAYUP, [3, 6], 1e-3)
        with pytest.raises(ValueError, match='G_c'):
            r.load_factor(0.0)
