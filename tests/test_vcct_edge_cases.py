"""Edge paths of the VCCT front and its load-factor roots (issue #219).

The branches the fracture tests do not reach: a front with no finite
factor or none at all, a one-node line, a mesh without a delamination, a
one-node ligament (its growth direction falls back on the first delaminated
neighbour), ties-free pair forces, and the prestress roots: the bisection on
the sign where the prestress closes the crack (``G_T = 0`` below the
crossing), a root beyond the bracketing grid (``inf``), and the batch root
with its bisection budget spent. The roots are checked against closed forms
on one synthetic node. Forces N, lengths mm, rates N/mm.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings

import numpy as np
import pytest

from porosity_fe import MATERIALS, CompositeMesh, Delamination, FESolver, PorosityField
from porosity_fe.fe import vcct

MAT4 = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4,
                           G_Ic=0.2, G_IIc=0.6, bk_eta=2.0)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


def _mesh(*delaminations, nx=10, ny=2):
    pf = PorosityField(MAT4, 0.0)
    return CompositeMesh(pf, MAT4, nx=nx, ny=ny, nz=4, delaminations=list(delaminations)), pf


@pytest.fixture(scope='module')
def front():
    logging.disable(logging.WARNING)
    try:
        mesh, pf = _mesh(Delamination.strip(1, 20.0, 30.0))
        res = FESolver(mesh, MAT4, pf).solve('tension')
    finally:
        logging.disable(logging.NOTSET)
    assert res.crack_front is not None and len(res.crack_front) > 0
    return res.crack_front


class _Front:
    """The three things the batch root reads from a front."""

    def __init__(self, force_local, opening_local, area):
        self.force_local = np.asarray(force_local, dtype=float)
        self.opening_local = np.asarray(opening_local, dtype=float)
        self.area = np.asarray(area, dtype=float)

    def __len__(self):
        return self.area.size


#: Local components are (x', z', y'): index 1 is the opening (mode I).
def _mode_I(F, D):
    return np.array([0.0, F, 0.0]), np.array([0.0, D, 0.0])


#: Mode I toughness only matters on a pure mode I node.
TOUGH_I = (np.array(1e-9), np.array(1.0), np.array(1.0))


class TestCrackFrontProperties:
    def test_no_finite_factor(self, front):
        none_driven = dataclasses.replace(front, lambda_c=np.full(len(front), np.inf))
        assert none_driven.governing is None
        assert none_driven.load_factor == math.inf

    def test_no_toughness_or_empty(self, front):
        for lam in (None, np.array([])):
            f = dataclasses.replace(front, lambda_c=lam)
            assert f.governing is None and f.load_factor is None


class TestFrontGeometry:
    def test_one_node_line(self):
        np.testing.assert_array_equal(vcct._tributary(np.array([3.0])), [0.0])
        np.testing.assert_allclose(vcct._tributary(np.array([0.0, 1.0, 3.0])), [0.5, 1.5, 1.0])

    def test_no_delamination(self):
        mesh, _pf = _mesh()
        assert vcct._front_geometry(mesh) is None
        zeros = np.zeros(3 * mesh.n_nodes)
        assert vcct.vcct_energy_release_rates(mesh, zeros, zeros) is None

    def test_one_node_ligament(self):
        """Two strips leave one bonded node column at x = 25 mm (nodes at
        y = 0, 10, 20). At every node there the edge-sharing delaminated
        neighbours (x = 20 and 30) cancel, so the direction falls back on
        the mean of all its delaminated neighbours: at y = 0 and 20 that
        points inward and the node grows from a diagonal neighbour; at
        y = 10 the six cancel too and the node grows from the first entry of
        the neighbour table, (+1, 0), that is along -x."""
        mesh, _pf = _mesh(Delamination.strip(1, -1.0, 24.9), Delamination.strip(1, 25.1, 51.0))
        geometry = vcct._front_geometry(mesh)
        assert geometry is not None
        np.testing.assert_allclose(geometry.xyz[:, 0], 25.0)
        np.testing.assert_allclose(geometry.xyz[:, 1], [0.0, 10.0, 20.0])
        diagonal = np.array([-5.0, 10.0]) / math.hypot(5.0, 10.0)
        np.testing.assert_allclose(geometry.direction,
                                   [diagonal * [1, -1], [-1.0, 0.0], diagonal], atol=1e-15)
        np.testing.assert_allclose(geometry.delta_a, [math.hypot(5.0, 10.0), 5.0,
                                                      math.hypot(5.0, 10.0)])

    def test_nodal_array_size(self):
        with pytest.raises(ValueError, match='3 entries per node'):
            vcct._as_nodal(np.zeros(5), 4, 'displacement')

    def test_pair_forces_without_ties(self):
        mesh, _pf = _mesh(Delamination.strip(1, 20.0, 30.0))
        n_pairs = mesh.interface_node_pairs()[0].size
        for ties in (None, (np.array([], dtype=int),) * 3):
            out = vcct._pair_forces(mesh, np.ones(3 * mesh.n_nodes), ties)
            assert out.shape == (n_pairs, 3) and not out.any()


class TestPrestressRoots:
    def test_prestress_closing_the_crack(self):
        """Thermal opening -1, mechanical opening +1 and force -1 on a unit
        area: G_I(lam) = lam (lam - 1) / 2, zero (clipped) below lam = 1.
        With G_Ic = 1e-9 the root is lam = (1 + sqrt(1 + 8e-9)) / 2, found by
        bisecting on the sign from the last grid point where G_T = 0."""
        F_T, D_T = _mode_I(0.0, -1.0)
        F_M, D_M = _mode_I(-1.0, 1.0)
        exact = 0.5 * (1.0 + math.sqrt(1.0 + 8e-9))
        lam = vcct._thermal_factor(F_T, D_T, F_M, D_M, 1.0, TOUGH_I, 2.0)
        assert lam == pytest.approx(exact, rel=1e-14)
        batch = vcct._thermal_factors_batch(
            _Front([F_T], [D_T], [1.0]), _Front([F_M], [D_M], [1.0]),
            tuple(t.reshape(1, 1) for t in TOUGH_I), 2.0)
        assert batch.shape == (1, 1) and batch[0, 0] == pytest.approx(exact, rel=1e-13)

    def test_root_beyond_the_grid_is_inf(self):
        """A thermal closure of 1e9 holds the crack shut until lam = 1e9, far
        past the grid (1e6 times the mechanical estimate sqrt(2) here): no
        factor, as for a node no load drives."""
        F_T, D_T = _mode_I(0.0, -1e9)
        F_M, D_M = _mode_I(-1.0, 1.0)
        tough = (np.array(1.0), np.array(1.0), np.array(1.0))
        assert vcct._thermal_factor(F_T, D_T, F_M, D_M, 1.0, tough, 2.0) == math.inf

    def test_batch_with_an_unbracketed_node_and_few_bisections(self):
        """Node 0 has a root, node 1 none on the grid (so the bracketing
        loop runs to its end); with two bisections the result is the upper
        end of node 0's bracket, above the root by less than one grid step
        (12 %)."""
        F_T = np.array([_mode_I(0.0, 0.0)[0], _mode_I(0.0, -1e9)[0]])
        D_T = np.array([_mode_I(0.0, 0.0)[1], _mode_I(0.0, -1e9)[1]])
        F_M = np.tile(_mode_I(-1.0, 1.0)[0], (2, 1))
        D_M = np.tile(_mode_I(-1.0, 1.0)[1], (2, 1))
        area = np.ones(2)
        tough = (np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
        exact = math.sqrt(2.0)                   # G_I = lam^2 / 2 = G_Ic = 1
        full = vcct._thermal_factors_batch(_Front(F_T, D_T, area), _Front(F_M, D_M, area),
                                           tough, 2.0)
        assert full[0, 0] == pytest.approx(exact, rel=1e-13) and full[0, 1] == math.inf
        coarse = vcct._thermal_factors_batch(_Front(F_T, D_T, area), _Front(F_M, D_M, area),
                                             tough, 2.0, max_bisections=2)
        assert exact <= coarse[0, 0] < exact * 1.13 and coarse[0, 1] == math.inf
