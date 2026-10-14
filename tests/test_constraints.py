#!/usr/bin/env python3
"""Tests for the constraint reduction u = T u_r + g (IMPROVEMENT_PLAN 3.6, D1).

``_ConstraintReduction`` (``porosity_fe.fe.assembler``) eliminates prescribed
values and master-slave ties. With ties it is checked against a dense
Lagrange-multiplier solve of the same constrained problem and against the
equivalent Dirichlet problem on an FE mesh; without ties it is the Dirichlet
partition the solver has always used. Floats are compared with tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg

from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
from porosity_fe.fe.assembler import BoundaryHandler, _ConstraintReduction, _DirichletPartition


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _kkt_solve(K, F, constrained, ties):
    """Minimize u.K.u/2 - F.u subject to the constraints, by Lagrange multipliers."""
    n = K.shape[0]
    rows, rhs = [], []
    for dof, value in constrained.items():
        r = np.zeros(n)
        r[dof] = 1.0
        rows.append(r)
        rhs.append(value)
    for s, m, c in zip(*ties, strict=True):
        r = np.zeros(n)
        r[s], r[m] = 1.0, -1.0
        rows.append(r)
        rhs.append(c)
    A = np.array(rows)
    kkt = np.block([[K, A.T], [A, np.zeros((len(rows), len(rows)))]])
    return np.linalg.solve(kkt, np.concatenate([F, rhs]))[:n]


class TestConstraintReduction:
    """Prescribed values plus master-slave ties, against a dense KKT solve."""

    N = 10
    CONSTRAINED = {0: 0.25, 7: -0.5}
    # 9 -> 8 -> 3 is a chain; 5 -> 7 has a prescribed master; 2 -> 4 is plain.
    TIES = (np.array([9, 8, 5, 2]), np.array([8, 3, 7, 4]),
            np.array([0.1, -0.3, 0.05, 1.5]))

    def _reduction(self, ties=TIES, constrained=None):
        constrained = self.CONSTRAINED if constrained is None else constrained
        return _ConstraintReduction.from_constraints(self.N, constrained, ties=ties)

    def test_historical_name_is_the_same_class(self):
        assert _DirichletPartition is _ConstraintReduction

    def test_without_ties_it_is_the_dirichlet_partition(self):
        for ties in (None, (np.array([], int), np.array([], int), np.array([]))):
            red = self._reduction(ties=ties)
            assert red.transform is None
            assert red.key == np.array([0, 7], dtype=np.intp).tobytes()
            np.testing.assert_array_equal(red.free, [1, 2, 3, 4, 5, 6, 8, 9])
            assert red.tied.size == 0
            np.testing.assert_array_equal(red.lift(), [0.25, 0, 0, 0, 0, 0, 0, -0.5, 0, 0])

    def test_chains_and_prescribed_masters_are_resolved(self):
        red = self._reduction()
        np.testing.assert_array_equal(red.tied, [2, 5, 8, 9])
        np.testing.assert_array_equal(red.masters, [4, 7, 3, 3])
        np.testing.assert_allclose(red.offsets, [1.5, 0.05, -0.3, -0.2], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(red.free, [1, 3, 4, 6])
        T = red.transform.toarray()
        assert T.shape == (10, 4)
        assert np.all(T[5] == 0.0)            # master 7 is prescribed: all in g
        np.testing.assert_array_equal(T[9], T[3])
        g = red.lift()
        np.testing.assert_allclose(g[[0, 7, 5, 9, 2]], [0.25, -0.5, -0.45, -0.2, 1.5],
                                   rtol=0, atol=1e-15)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_reduced_solve_matches_lagrange_multipliers(self, seed):
        K = _spd(self.N, seed)
        F = np.random.default_rng(seed + 10).standard_normal(self.N)
        red = self._reduction()
        K_r = red.reduce_matrix(K).toarray()
        np.testing.assert_allclose(K_r, K_r.T, rtol=0, atol=1e-12 * np.abs(K_r).max())
        u = red.expand(np.linalg.solve(K_r, red.reduce_rhs(K, F)))
        u_ref = _kkt_solve(K, F, self.CONSTRAINED, self.TIES)
        np.testing.assert_allclose(u, u_ref, rtol=0, atol=1e-12 * np.abs(u_ref).max())
        # Prescribed values hold bit for bit, ties to round-off.
        assert u[0] == 0.25 and u[7] == -0.5
        np.testing.assert_allclose(u[self.TIES[0]] - u[self.TIES[1]], self.TIES[2],
                                   rtol=0, atol=1e-14)

    def test_key_depends_on_topology_not_values(self):
        a = self._reduction()
        b = self._reduction(ties=(self.TIES[0], self.TIES[1], 2.0 * self.TIES[2]),
                            constrained={0: 1.0, 7: 3.0})
        assert a.key == b.key
        c = self._reduction(ties=(self.TIES[0][:3], self.TIES[1][:3], self.TIES[2][:3]))
        assert c.key != a.key
        # A tie key can never equal a Dirichlet key (fixed.tobytes()).
        assert len(a.key) % np.dtype(np.intp).itemsize != 0

    @pytest.mark.parametrize("ties, constrained, match", [
        (([1, 1], [2, 3], [0.0, 0.0]), {}, "more than one master"),
        (([0], [2], [0.0]), {0: 0.0}, "both tied and prescribed"),
        (([1, 2], [2, 1], [0.0, 0.0]), {}, "cycle"),
        (([1], [1], [0.0]), {}, "cycle"),
        (([1], [10], [0.0]), {}, "out of range"),
        (([1], [2], [np.nan]), {}, "finite"),
        (([1, 2], [3], [0.0, 0.0]), {}, "equal-length"),
    ])
    def test_invalid_ties_are_rejected(self, ties, constrained, match):
        with pytest.raises(ValueError, match=match):
            _ConstraintReduction.from_constraints(
                self.N, constrained, ties=tuple(np.asarray(t) for t in ties))


@pytest.fixture(scope='module')
def model():
    mat = MATERIALS['T800_epoxy']
    pf = PorosityField(mat, 0.03, distribution='clustered')
    mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=4)
    solver = FESolver(mesh, mat, pf)
    return mesh, solver, solver.assembler.stiffness()


class TestTiesOnAnFEMesh:
    """Ties on a real stiffness give the same field as the Dirichlet BCs."""

    @staticmethod
    def _solve(K, red, F):
        K_r = red.reduce_matrix(K)
        return red.expand(scipy.sparse.linalg.spsolve(K_r.tocsc(), red.reduce_rhs(K, F)))

    def test_face_tied_to_a_prescribed_master_matches_dirichlet(self, model):
        # Compression BCs, with every x_max u_x but one expressed as a tie
        # (zero offset) to the one that stays prescribed.
        mesh, solver, K = model
        constrained, F = BoundaryHandler(mesh).compression_bcs(-0.001)
        x_max = mesh.nodes_on_face('x_max')
        keep, others = 3 * int(x_max[0]), 3 * x_max[1:]
        reduced = {d: v for d, v in constrained.items() if d not in set(others.tolist())}
        ties = (others, np.full(others.size, keep), np.zeros(others.size))
        u_tied = self._solve(K, _ConstraintReduction.from_constraints(K.shape[0], reduced, ties=ties), F)
        u_ref = solver.solve('compression', applied_strain=-0.001).displacement.ravel()
        np.testing.assert_allclose(u_tied, u_ref, rtol=0, atol=1e-10 * np.abs(u_ref).max())

    def test_rigid_platen_carries_the_load_on_a_free_master(self, model):
        # Tie every x_max u_x to one free master and load the master: a
        # rigid platen. The face then moves uniformly and the equilibrium of
        # the reduced system is the platen force balance.
        mesh, solver, K = model
        constrained, F = BoundaryHandler(mesh).compression_bcs(-0.001)
        x_max = mesh.nodes_on_face('x_max')
        master, others = 3 * int(x_max[0]), 3 * x_max[1:]
        for d in [master, *others.tolist()]:
            constrained.pop(int(d))
        F = F.copy()
        F[master] = -100.0
        red = _ConstraintReduction.from_constraints(
            K.shape[0], constrained, ties=(others, np.full(others.size, master), np.zeros(others.size)))
        u = self._solve(K, red, F)
        np.testing.assert_array_equal(u[others], np.full(others.size, u[master]))
        assert u[master] < 0.0
        # The reaction of the platen face balances the load.
        r = K @ u - F
        assert r[3 * x_max].sum() == pytest.approx(0.0, abs=1e-8 * 100.0)
        assert r[3 * mesh.nodes_on_face('x_min')].sum() == pytest.approx(100.0, rel=1e-8)
