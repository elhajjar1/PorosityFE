#!/usr/bin/env python3
"""Tests for porosity_fe.mesh / porosity_fe.fe.mesh.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (MATERIALS, PorosityField, CompositeMesh,
                                   check_mesh_quality)


class TestCompositeMesh:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_mesh_creation(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.nodes is not None
        assert mesh.elements is not None

    def test_node_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        expected_nodes = 11 * 6 * 7  # (nx+1)*(ny+1)*(nz+1)
        assert len(mesh.nodes) == expected_nodes

    def test_element_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        expected_elements = 10 * 5 * 6
        assert len(mesh.elements) == expected_elements

    def test_nodes_3d(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.nodes.shape[1] == 3

    def test_hex_elements_8_nodes(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.elements.shape[1] == 8

    def test_porosity_field_sampled(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.porosity) == len(mesh.nodes)
        # Uniform 3% -> all nodes should be ~0.03
        np.testing.assert_allclose(mesh.porosity, 0.03, atol=1e-10)

    def test_stiffness_reduction_sampled(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.stiffness_reduction) == len(mesh.nodes)
        np.testing.assert_allclose(mesh.stiffness_reduction, 0.97, atol=1e-10)

    def test_ply_ids_range(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert np.min(mesh.ply_ids) >= 0
        assert np.max(mesh.ply_ids) <= self.material.n_plies

    def test_domain_bounds(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert np.min(mesh.nodes[:, 0]) >= 0
        assert np.min(mesh.nodes[:, 2]) >= 0
        assert abs(np.max(mesh.nodes[:, 2]) - self.material.total_thickness) < 1e-6

    def test_zero_axis_count_rejected(self):
        with pytest.raises(ValueError, match=r"nx.*positive integer"):
            CompositeMesh(self.pf, self.material, nx=0, ny=5, nz=6)

    def test_negative_axis_count_rejected(self):
        with pytest.raises(ValueError, match=r"ny.*positive integer"):
            CompositeMesh(self.pf, self.material, nx=10, ny=-2, nz=6)

    def test_huge_axis_count_rejected(self):
        with pytest.raises(ValueError, match=r"exhaust memory|exceeds"):
            CompositeMesh(self.pf, self.material, nx=20_000, ny=5, nz=6)


class TestCompositeMeshFE:
    """Tests for the FE-related additions to CompositeMesh."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_nodes_on_face_x_min(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        nodes = mesh.nodes_on_face('x_min')
        assert len(nodes) > 0
        np.testing.assert_allclose(mesh.nodes[nodes, 0], 0.0, atol=1e-10)

    def test_nodes_on_face_x_max(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        nodes = mesh.nodes_on_face('x_max')
        assert len(nodes) > 0
        np.testing.assert_allclose(mesh.nodes[nodes, 0], mesh.L_x, atol=1e-10)

    def test_nodes_on_face_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        # x_min face should have (ny+1)*(nz+1) nodes
        nodes = mesh.nodes_on_face('x_min')
        assert len(nodes) == (3 + 1) * (4 + 1)

    def test_nodes_on_face_invalid(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        with pytest.raises(ValueError):
            mesh.nodes_on_face('invalid')

    def test_n_dof(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        assert mesh.n_dof == mesh.n_nodes * 3

    def test_domain_size(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        Lx, Ly, Lz = mesh.domain_size
        assert abs(Lx - 50.0) < 1e-10
        assert abs(Ly - 20.0) < 1e-10

    def test_ply_angles_ud_sentinel_all_zero(self):
        # #44 item 2: the all-zero behaviour moved from the implicit
        # default to the explicit 'UD' sentinel.
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4,
                             ply_angles='UD')
        np.testing.assert_allclose(mesh.ply_angles, 0.0)

    def test_ply_angles_custom(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4,
                            ply_angles=[0, 45, 90, -45])
        # Should have angles from the layup
        unique_angles = np.unique(mesh.ply_angles)
        assert len(unique_angles) > 1

    def test_elem_ply_ids(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        assert hasattr(mesh, 'elem_ply_ids')
        assert len(mesh.elem_ply_ids) == mesh.n_elements


class TestMeshQuality:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_returns_dict(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        result = check_mesh_quality(mesh)
        assert isinstance(result, dict)
        assert 'min_aspect_ratio' in result
        assert 'max_aspect_ratio' in result
        assert 'min_jacobian_det' in result
        assert 'n_inverted' in result
        assert 'n_distorted' in result

    def test_structured_mesh_no_inverted(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        result = check_mesh_quality(mesh)
        assert result['n_inverted'] == 0

    def test_positive_jacobian(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        result = check_mesh_quality(mesh)
        assert result['min_jacobian_det'] > 0

    def test_verbose_mode(self):
        mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)
        result = check_mesh_quality(mesh, verbose=True)
        assert result['n_elements'] == mesh.n_elements


class TestCompositeMeshFindNodesNear:
    """Unit tests for the CompositeMesh.find_nodes_near helper."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.02, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=4, ny=2, nz=2)

    def test_finds_corner_node(self):
        ids = self.mesh.find_nodes_near(x=0.0, y=0.0, z=0.0)
        assert ids.size >= 1
        # First-found node should have coords very close to origin.
        coord = self.mesh.nodes[ids[0]]
        assert np.linalg.norm(coord) < 1e-6

    def test_axis_subset_match(self):
        # Search only on x: should hit a whole column (constant x) of nodes.
        Lx = self.mesh.L_x
        ids = self.mesh.find_nodes_near(x=Lx / 2.0)
        assert ids.size > 0
        for nid in ids:
            assert abs(self.mesh.nodes[nid, 0] - Lx / 2.0) <= 1e-6 + 1e-9

    def test_requires_at_least_one_axis(self):
        with pytest.raises(ValueError):
            self.mesh.find_nodes_near()

    def test_default_tol_finds_exact_corner(self):
        # An exact mesh-node coordinate should always be found regardless
        # of element aspect ratio: distance from query to node is zero.
        ids = self.mesh.find_nodes_near(x=self.mesh.L_x, y=self.mesh.L_y,
                                        z=self.mesh.L_z)
        assert ids.size >= 1
        # The opposite-corner node coordinates should match.
        coord = self.mesh.nodes[ids[0]]
        assert abs(coord[0] - self.mesh.L_x) < 1e-9
        assert abs(coord[1] - self.mesh.L_y) < 1e-9
        assert abs(coord[2] - self.mesh.L_z) < 1e-9

    def test_explicit_tol(self):
        # With a generous tol we should pick up multiple neighbours.
        ids_loose = self.mesh.find_nodes_near(x=self.mesh.L_x / 2.0,
                                              z=self.mesh.L_z,
                                              tol=self.mesh.L_x)
        # With a tiny tol on a non-coincident point we get nothing.
        ids_tight = self.mesh.find_nodes_near(
            x=self.mesh.L_x / 2.0 + 1e-3,
            z=self.mesh.L_z + 1e-3,
            tol=1e-9,
        )
        assert ids_loose.size > ids_tight.size
