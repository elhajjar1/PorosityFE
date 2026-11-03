"""``CompositeMesh`` takes the coupon size in its constructor (issue #225).

``length=``, ``width=`` and ``x_grid=`` replace assigning ``L_x`` / ``L_y`` /
``x_grid`` and calling ``generate_mesh()`` again. The old assignment still
works for one release: it warns (``DeprecationWarning`` naming the new
argument) and marks the mesh stale, and ``FESolver`` refuses a stale mesh
until ``generate_mesh()`` runs, so its nodes, porosity and elements can no
longer silently disagree with the dimensions. Lengths in mm.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pytest

from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField

MAT = MATERIALS['T800_epoxy']


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def pf():
    return PorosityField(MAT, 0.02)


def test_constructor_sizes_the_coupon(pf):
    default = CompositeMesh(pf, MAT, nx=4, ny=2, nz=4)
    assert (default.L_x, default.L_y) == (50.0, 20.0) and not default.is_stale
    mesh = CompositeMesh(pf, MAT, nx=8, ny=4, nz=4, length=80.0, width=25.0)
    assert (mesh.L_x, mesh.L_y, mesh.L_z) == (80.0, 25.0, MAT.total_thickness)
    np.testing.assert_allclose(mesh.nodes.max(axis=0), [80.0, 25.0, MAT.total_thickness])
    np.testing.assert_allclose(mesh.nodes.min(axis=0), 0.0)
    assert mesh.x_grid is None
    np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), np.linspace(0.0, 80.0, 9))


def test_constructor_takes_a_graded_grid(pf):
    grid = [0.0, 1.0, 3.0, 7.0, 15.0]
    mesh = CompositeMesh(pf, MAT, nx=4, ny=2, nz=4, length=15.0, x_grid=grid)
    np.testing.assert_array_equal(np.unique(mesh.nodes[:, 0]), grid)
    np.testing.assert_array_equal(mesh.x_grid, grid)
    with pytest.raises(ValueError, match='x_grid must hold nx \\+ 1 = 5'):
        CompositeMesh(pf, MAT, nx=4, ny=2, nz=4, length=16.0, x_grid=grid)
    with pytest.raises(ValueError, match='pass x_grid=None'):
        CompositeMesh(pf, MAT, nx=3, ny=2, nz=4, length=15.0, x_grid=grid)


@pytest.mark.parametrize('kwargs, name', [
    ({'length': 0.0}, 'length'), ({'length': -5.0}, 'length'),
    ({'width': math.nan}, 'width'), ({'width': math.inf}, 'width'),
    ({'length': True}, 'length'), ({'width': '20'}, 'width'),
])
def test_invalid_dimensions(pf, kwargs, name):
    with pytest.raises(ValueError, match=f"{name} must be a finite length > 0"):
        CompositeMesh(pf, MAT, nx=4, ny=2, nz=4, **kwargs)


def test_same_mesh_either_way(pf):
    """The deprecated two-step route and the constructor give the same mesh."""
    direct = CompositeMesh(pf, MAT, nx=8, ny=4, nz=4, length=80.0, width=25.0)
    old = CompositeMesh(pf, MAT, nx=8, ny=4, nz=4)
    with pytest.warns(DeprecationWarning, match='pass length= to CompositeMesh'):
        old.L_x = 80.0
    with pytest.warns(DeprecationWarning, match='pass width= to CompositeMesh'):
        old.L_y = 25.0
    old.generate_mesh()
    assert not old.is_stale
    for name in ('nodes', 'elements', 'porosity', 'stiffness_reduction', 'ply_angles'):
        np.testing.assert_array_equal(getattr(old, name), getattr(direct, name), err_msg=name)


def test_a_stale_mesh_is_refused(pf):
    mesh = CompositeMesh(pf, MAT, nx=6, ny=3, nz=4)
    fe = FESolver(mesh, MAT, pf)
    with pytest.warns(DeprecationWarning, match='FESolver refuses the mesh'):
        mesh.L_x = 60.0
    assert mesh.is_stale
    # The arrays still describe the 50 mm coupon.
    assert mesh.nodes[:, 0].max() == pytest.approx(50.0)
    with pytest.raises(ValueError, match='stale'):
        fe.solve('compression')
    with pytest.raises(ValueError, match='length=, width=, x_grid='):
        FESolver(mesh, MAT, pf)
    mesh.generate_mesh()
    assert not mesh.is_stale and mesh.nodes[:, 0].max() == pytest.approx(60.0)
    res = FESolver(mesh, MAT, pf).solve('compression')
    assert 0.0 < res.knockdown <= 1.0


def test_x_grid_assignment_is_deprecated_too(pf):
    mesh = CompositeMesh(pf, MAT, nx=4, ny=2, nz=4, length=15.0)
    with pytest.warns(DeprecationWarning, match='pass x_grid= to CompositeMesh'):
        mesh.x_grid = [0.0, 1.0, 3.0, 7.0, 15.0]
    assert mesh.is_stale
    mesh.generate_mesh()
    np.testing.assert_array_equal(np.unique(mesh.nodes[:, 0]), [0.0, 1.0, 3.0, 7.0, 15.0])
    with pytest.raises(ValueError, match='L_x must be a finite length > 0'):
        mesh.L_x = -1.0


def test_builders_do_not_warn(pf):
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        flex = CompositeMesh.for_flexure(pf, MAT)
        ilss = CompositeMesh.for_ilss(pf, MAT)
        dcb = CompositeMesh.for_dcb(PorosityField(MAT, 0.0), MAT)
    for mesh in (flex, ilss, dcb):
        assert not mesh.is_stale
    assert dcb.x_grid is not None
