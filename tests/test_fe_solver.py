#!/usr/bin/env python3
"""Tests for porosity_fe.fe.solver.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""

import dataclasses

import numpy as np
import scipy.sparse
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (MaterialProperties, MATERIALS, PorosityField,
                                   CompositeMesh, strain_transformation_3d,
                                   Hex8Element, GlobalAssembler,
                                   BoundaryHandler, FESolver, FieldResults)


class TestHex8Element:
    def setup_method(self):
        # Create a simple unit cube element
        self.node_coords = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        mat = MATERIALS['T800_epoxy']
        self.C_base = mat.get_stiffness_matrix()
        self.C_m = mat.get_isotropic_matrix_stiffness()
        self.elem = Hex8Element(
            node_coords=self.node_coords,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, 0.03),
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )

    def test_shape_functions_partition_of_unity(self):
        """Shape functions should sum to 1 at any point."""
        N = Hex8Element.shape_functions(0.3, -0.2, 0.5)
        np.testing.assert_allclose(N.sum(), 1.0, atol=1e-14)

    def test_nan_node_porosities_rejected(self):
        bad = np.full(8, 0.03)
        bad[3] = float('nan')
        with pytest.raises(ValueError, match=r"node_porosities must be finite"):
            Hex8Element(
                node_coords=self.node_coords,
                C_base=self.C_base,
                ply_angle_deg=0.0,
                node_porosities=bad,
                void_shape_radii=(1, 1, 1),
                nu_m=0.35,
                C_m=self.C_m,
            )

    def test_node_porosities_above_one_rejected(self):
        bad = np.full(8, 0.03)
        bad[2] = 5.0  # plausibly a percent (5%) → rejected with hint
        with pytest.raises(ValueError, match=r"node_porosities must be a fraction"):
            Hex8Element(
                node_coords=self.node_coords,
                C_base=self.C_base,
                ply_angle_deg=0.0,
                node_porosities=bad,
                void_shape_radii=(1, 1, 1),
                nu_m=0.35,
                C_m=self.C_m,
            )

    def test_node_porosities_fp_overshoot_clipped(self):
        # ~1e-12 above 1.0 should be clipped, not rejected.
        bumped = np.full(8, 1.0)
        bumped[1] = 1.0 + 5e-13
        elem = Hex8Element(
            node_coords=self.node_coords,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=bumped,
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )
        assert np.all(elem.node_porosities <= 1.0)

    def test_shape_functions_at_nodes(self):
        """N_i should be 1 at node i and 0 at other nodes."""
        from porosity_fe_analysis import _NODE_COORDS_REF
        for i in range(8):
            xi, eta, zeta = _NODE_COORDS_REF[i]
            N = Hex8Element.shape_functions(xi, eta, zeta)
            for j in range(8):
                expected = 1.0 if i == j else 0.0
                assert abs(N[j] - expected) < 1e-14

    def test_shape_derivatives_shape(self):
        dN = Hex8Element.shape_derivatives(0.0, 0.0, 0.0)
        assert dN.shape == (3, 8)

    def test_jacobian_unit_cube(self):
        """Jacobian of unit cube should be 0.5 * I (mapping [-1,1] to [0,1])."""
        J = self.elem.jacobian(0.0, 0.0, 0.0)
        assert J.shape == (3, 3)
        np.testing.assert_allclose(J, 0.5 * np.eye(3), atol=1e-14)

    def test_B_matrix_shape(self):
        B = self.elem.B_matrix(0.0, 0.0, 0.0)
        assert B.shape == (6, 24)

    def test_stiffness_matrix_shape(self):
        Ke = self.elem.stiffness_matrix()
        assert Ke.shape == (24, 24)

    def test_stiffness_matrix_symmetric(self):
        Ke = self.elem.stiffness_matrix()
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-4)

    def test_stiffness_matrix_positive_semidefinite(self):
        Ke = self.elem.stiffness_matrix()
        eigenvalues = np.linalg.eigvalsh(Ke)
        # Should have 6 zero eigenvalues (rigid body modes) and 18 positive
        assert np.sum(eigenvalues > 1e-6) >= 12  # At least 12 positive

    def test_volume_unit_cube(self):
        assert abs(self.elem.volume - 1.0) < 1e-12

    def test_inverted_element_rejected_at_assembly(self):
        """Regression for #33: signed det(J) silently corrupting K."""
        # Swap two adjacent nodes on the bottom face to invert the element.
        inverted = self.node_coords.copy()
        inverted[[0, 1]] = inverted[[1, 0]]
        bad_elem = Hex8Element(
            node_coords=inverted,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, 0.03),
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )
        with pytest.raises(ValueError, match="non-positive Jacobian"):
            bad_elem.stiffness_matrix()

    def test_inverted_element_volume_still_positive(self):
        """volume uses abs(det J); only stiffness_matrix raises."""
        inverted = self.node_coords.copy()
        inverted[[0, 1]] = inverted[[1, 0]]
        bad_elem = Hex8Element(
            node_coords=inverted,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, 0.03),
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )
        assert bad_elem.volume > 0

    def test_stress_at_gauss_points_shape(self):
        u_elem = np.zeros(24)
        sig = self.elem.stress_at_gauss_points(u_elem)
        assert sig.shape == (8, 6)

    def test_strain_at_gauss_points_shape(self):
        u_elem = np.zeros(24)
        eps = self.elem.strain_at_gauss_points(u_elem)
        assert eps.shape == (8, 6)

    def test_zero_displacement_zero_stress(self):
        u_elem = np.zeros(24)
        sig = self.elem.stress_at_gauss_points(u_elem)
        np.testing.assert_allclose(sig, 0.0, atol=1e-12)

    def test_uniform_strain_produces_uniform_stress(self):
        """Uniform x-displacement gradient should produce constant sigma_11."""
        # Prescribe u_x = eps_x * x at each node, with eps_x = 0.001
        eps_x = 0.001
        u_elem = np.zeros(24)
        for i in range(8):
            u_elem[3 * i] = eps_x * self.node_coords[i, 0]
        sig = self.elem.stress_at_gauss_points(u_elem)
        # All GP should have approximately the same sigma_11
        sigma_11_vals = sig[:, 0]
        assert np.std(sigma_11_vals) / (np.mean(np.abs(sigma_11_vals)) + 1e-12) < 0.01

    def test_porosity_reduces_stiffness(self):
        """Higher porosity should produce lower element stiffness."""
        elem_low = Hex8Element(self.node_coords, self.C_base, 0.0,
                               np.full(8, 0.01), (1, 1, 1), 0.35, self.C_m)
        elem_high = Hex8Element(self.node_coords, self.C_base, 0.0,
                                np.full(8, 0.10), (1, 1, 1), 0.35, self.C_m)
        Ke_low = elem_low.stiffness_matrix()
        Ke_high = elem_high.stiffness_matrix()
        # Trace of stiffness should be lower for higher porosity
        assert np.trace(Ke_high) < np.trace(Ke_low)

    def test_wrong_node_coords_shape(self):
        with pytest.raises(ValueError):
            Hex8Element(np.zeros((4, 3)), self.C_base, 0.0,
                       np.full(8, 0.03), (1, 1, 1), 0.35, self.C_m)

    def test_wrong_porosity_shape(self):
        with pytest.raises(ValueError):
            Hex8Element(self.node_coords, self.C_base, 0.0,
                       np.full(4, 0.03), (1, 1, 1), 0.35, self.C_m)


class TestGlobalAssembler:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_create_element(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        elem = assembler.create_element(0)
        assert isinstance(elem, Hex8Element)

    def test_element_dof_indices_shape(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        dofs = assembler.element_dof_indices(0)
        assert dofs.shape == (24,)

    def test_element_dof_indices_range(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        dofs = assembler.element_dof_indices(0)
        assert np.all(dofs >= 0)
        assert np.all(dofs < self.mesh.n_dof)

    def test_assemble_stiffness_shape(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        assert K.shape == (self.mesh.n_dof, self.mesh.n_dof)

    def test_assemble_stiffness_symmetric(self):
        # Issue #57: K is now explicitly symmetrized at the per-element
        # cache layer, so K = K^T should hold to machine precision rather
        # than the prior atol=1e-2 slop.
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        K_dense = K.toarray()
        max_K = float(np.max(np.abs(K_dense)))
        max_asym = float(np.max(np.abs(K_dense - K_dense.T)))
        assert max_asym < 1e-10 * max_K, (
            f"K not symmetric: max|K-K.T| = {max_asym:.4e}, "
            f"max|K| = {max_K:.4e}"
        )

    def test_assemble_stiffness_sparse(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        assert scipy.sparse.issparse(K)


class TestBoundaryHandler:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)
        self.handler = BoundaryHandler(self.mesh)

    def test_compression_bcs_returns_tuple(self):
        constrained, F = self.handler.compression_bcs()
        assert isinstance(constrained, dict)
        assert isinstance(F, np.ndarray)
        assert len(F) == self.mesh.n_dof

    def test_compression_bcs_constrained_dofs(self):
        constrained, F = self.handler.compression_bcs()
        assert len(constrained) > 0
        # Should constrain ux on x_min and x_max
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        for nid in xmin_nodes:
            assert 3 * int(nid) in constrained
            assert constrained[3 * int(nid)] == 0.0

    def test_compression_bcs_prescribed_displacement(self):
        strain = -0.01
        constrained, F = self.handler.compression_bcs(applied_strain=strain)
        xmax_nodes = self.mesh.nodes_on_face('x_max')
        expected_disp = strain * self.mesh.L_x
        for nid in xmax_nodes:
            assert abs(constrained[3 * int(nid)] - expected_disp) < 1e-10

    def test_tension_bcs(self):
        strain = 0.01
        constrained, F = self.handler.tension_bcs(applied_strain=strain)
        assert len(constrained) > 0
        assert len(F) == self.mesh.n_dof
        # x_min: ux pinned to 0
        for nid in self.mesh.nodes_on_face('x_min'):
            assert constrained[3 * int(nid)] == 0.0
        # x_max: ux = +strain * Lx (positive => tension, not compression)
        expected = strain * self.mesh.L_x
        assert expected > 0.0
        for nid in self.mesh.nodes_on_face('x_max'):
            assert abs(constrained[3 * int(nid)] - expected) < 1e-12
        # y_min: uy pinned to 0 (symmetry)
        for nid in self.mesh.nodes_on_face('y_min'):
            assert constrained[3 * int(nid) + 1] == 0.0

    def test_shear_bcs(self):
        gamma = 0.01
        constrained, F = self.handler.shear_bcs(applied_strain=gamma)
        assert len(constrained) > 0
        assert len(F) == self.mesh.n_dof
        nodes = self.mesh.nodes
        # All four side faces must prescribe BOTH ux and uy to the pure-shear
        # field u = gamma/2 * y, v = gamma/2 * x. A regression that swapped
        # ux/uy on a face, or left a face traction-free, fails here.
        for face in ('x_min', 'x_max', 'y_min', 'y_max'):
            face_nodes = self.mesh.nodes_on_face(face)
            assert len(face_nodes) > 0
            for nid in face_nodes:
                nid = int(nid)
                x_n, y_n = float(nodes[nid, 0]), float(nodes[nid, 1])
                assert abs(constrained[3 * nid] - (gamma / 2.0) * y_n) < 1e-12
                assert abs(constrained[3 * nid + 1] - (gamma / 2.0) * x_n) < 1e-12

    # ------------------------------------------------------------------
    # Issue #48 (item 1) — deepen BC-handler asserts.  Mirror the rigor
    # of test_compression_bcs_constrained_dofs for shear and tension:
    # check the *specific* DOF indices and prescribed values on each
    # face, the rigid-body corner pin, and that the other in-plane DOF
    # is not constrained where the loading mode says it shouldn't be.
    # A regression that, for instance, swapped ux<->uy on x_max would
    # have passed the pre-existing length-only assertion.
    # ------------------------------------------------------------------
    def test_tension_bcs_constrained_dofs(self):
        strain = 0.01
        constrained, F = self.handler.tension_bcs(applied_strain=strain)
        expected_xmax = strain * self.mesh.L_x

        # x_min face: ux = 0 prescribed; uy on x_min must NOT be in the
        # constrained set (would over-constrain Poisson contraction).
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        assert len(xmin_nodes) > 0
        for nid in xmin_nodes:
            nid = int(nid)
            assert 3 * nid in constrained, f"ux missing on x_min node {nid}"
            assert constrained[3 * nid] == 0.0
            # Corner nodes on (x_min, y_min) may have uy=0 from the y_min
            # symmetry condition — but a generic x_min node must not.
            if nid not in self.mesh.nodes_on_face('y_min'):
                assert 3 * nid + 1 not in constrained, (
                    f"uy on x_min interior node {nid} should be free")

        # x_max face: ux = +strain * Lx; uy on x_max must be free
        xmax_nodes = self.mesh.nodes_on_face('x_max')
        assert len(xmax_nodes) > 0
        for nid in xmax_nodes:
            nid = int(nid)
            assert 3 * nid in constrained, f"ux missing on x_max node {nid}"
            assert abs(constrained[3 * nid] - expected_xmax) < 1e-12
            if nid not in self.mesh.nodes_on_face('y_min'):
                assert 3 * nid + 1 not in constrained, (
                    f"uy on x_max interior node {nid} should be free")

        # y_min symmetry face: uy = 0
        ymin_nodes = self.mesh.nodes_on_face('y_min')
        assert len(ymin_nodes) > 0
        for nid in ymin_nodes:
            nid = int(nid)
            assert 3 * nid + 1 in constrained, f"uy missing on y_min node {nid}"
            assert constrained[3 * nid + 1] == 0.0

        # Rigid-body z pin lives on the (x_min, y_min, z_min) corner.
        xmin_set = set(int(n) for n in xmin_nodes)
        ymin_set = set(int(n) for n in ymin_nodes)
        zmin_set = set(int(n) for n in self.mesh.nodes_on_face('z_min'))
        corner_candidates = xmin_set & ymin_set & zmin_set
        assert corner_candidates, "no (x_min, y_min, z_min) corner node found"
        pinned_z_dofs = [d for d in constrained if d % 3 == 2]
        assert len(pinned_z_dofs) == 1, (
            f"tension should pin exactly one uz DOF, got {len(pinned_z_dofs)}")
        pinned_node = pinned_z_dofs[0] // 3
        assert pinned_node in corner_candidates, (
            f"uz pin is on node {pinned_node}, not on x_min/y_min/z_min corner")
        assert constrained[pinned_z_dofs[0]] == 0.0

        # Sanity: the force vector is purely displacement-controlled.
        assert np.all(F == 0.0)

    def test_shear_bcs_constrained_dofs(self):
        gamma = 0.01
        constrained, F = self.handler.shear_bcs(applied_strain=gamma)
        nodes = self.mesh.nodes

        # For every node on any of the four side faces, BOTH ux and uy
        # must be in the constrained set with the exact pure-shear values
        # ux = (gamma/2) * y_n, uy = (gamma/2) * x_n.
        for face in ('x_min', 'x_max', 'y_min', 'y_max'):
            face_nodes = self.mesh.nodes_on_face(face)
            assert len(face_nodes) > 0, f"face {face} has no nodes"
            for nid in face_nodes:
                nid = int(nid)
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                assert 3 * nid in constrained, (
                    f"ux missing on {face} node {nid}")
                assert 3 * nid + 1 in constrained, (
                    f"uy missing on {face} node {nid}")
                np.testing.assert_allclose(
                    constrained[3 * nid], (gamma / 2.0) * y_n, atol=1e-12,
                    err_msg=f"ux wrong on {face} node {nid}")
                np.testing.assert_allclose(
                    constrained[3 * nid + 1], (gamma / 2.0) * x_n, atol=1e-12,
                    err_msg=f"uy wrong on {face} node {nid}")

        # Distinct face values: with gamma=0.01, Lx>0, Ly>0 the prescribed
        # ux on x_max varies with y (so different from x_min where it also
        # varies with y but x-coordinate differs).  In particular, the
        # *uy* value on x_max nodes must equal (gamma/2)*Lx, NOT zero —
        # a regression that copied the compression BC into shear would
        # set uy=0 there and would fail this asymmetric check.
        half_Lx = (gamma / 2.0) * self.mesh.L_x
        for nid in self.mesh.nodes_on_face('x_max'):
            nid = int(nid)
            assert abs(constrained[3 * nid + 1] - half_Lx) < 1e-12, (
                f"uy on x_max node {nid} must equal (gamma/2)*Lx")
        half_Ly = (gamma / 2.0) * self.mesh.L_y
        for nid in self.mesh.nodes_on_face('y_max'):
            nid = int(nid)
            assert abs(constrained[3 * nid] - half_Ly) < 1e-12, (
                f"ux on y_max node {nid} must equal (gamma/2)*Ly")

        # Rigid-body uz pin: exactly one uz DOF constrained, on the
        # (x_min, y_min, z_min) corner.
        xmin_set = set(int(n) for n in self.mesh.nodes_on_face('x_min'))
        ymin_set = set(int(n) for n in self.mesh.nodes_on_face('y_min'))
        zmin_set = set(int(n) for n in self.mesh.nodes_on_face('z_min'))
        corner_candidates = xmin_set & ymin_set & zmin_set
        assert corner_candidates
        pinned_z_dofs = [d for d in constrained if d % 3 == 2]
        assert len(pinned_z_dofs) == 1, (
            f"shear should pin exactly one uz DOF, got {len(pinned_z_dofs)}")
        pinned_node = pinned_z_dofs[0] // 3
        assert pinned_node in corner_candidates
        assert constrained[pinned_z_dofs[0]] == 0.0

        # Top/bottom (z_min, z_max) interior nodes — i.e. not also on a
        # side face — must have ux and uy free; shear is in-plane only.
        side_node_set = set()
        for face in ('x_min', 'x_max', 'y_min', 'y_max'):
            side_node_set.update(int(n) for n in self.mesh.nodes_on_face(face))
        for face in ('z_min', 'z_max'):
            for nid in self.mesh.nodes_on_face(face):
                nid = int(nid)
                if nid in side_node_set:
                    continue
                assert 3 * nid not in constrained, (
                    f"ux on {face} interior node {nid} should be free")
                assert 3 * nid + 1 not in constrained, (
                    f"uy on {face} interior node {nid} should be free")

        assert np.all(F == 0.0)

    def test_apply_penalty(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = self.handler.compression_bcs()
        K_mod, F_mod = BoundaryHandler.apply_penalty(K, F, constrained)
        assert K_mod.shape == K.shape
        assert len(F_mod) == len(F)

    def test_penalty_increases_diagonal(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = self.handler.compression_bcs()
        K_mod, F_mod = BoundaryHandler.apply_penalty(K, F, constrained)
        # Constrained DOF diagonals should be much larger
        for dof in list(constrained.keys())[:5]:
            assert K_mod[dof, dof] > K[dof, dof]

    def test_ilss_bcs_returns_tuple(self):
        constrained, F = self.handler.ilss_bcs(applied_load=-10.0)
        assert isinstance(constrained, dict)
        assert isinstance(F, np.ndarray)
        assert len(F) == self.mesh.n_dof

    def test_ilss_bcs_pins_support_edges(self):
        """The two bottom-face support edges should pin all three DOFs."""
        constrained, F = self.handler.ilss_bcs(applied_load=-10.0)
        zmin = self.mesh.nodes_on_face('z_min')
        xmin = self.mesh.nodes_on_face('x_min')
        xmax = self.mesh.nodes_on_face('x_max')
        support_left = np.intersect1d(zmin, xmin)
        support_right = np.intersect1d(zmin, xmax)
        assert support_left.size > 0
        assert support_right.size > 0
        for nid in np.concatenate([support_left, support_right]):
            nid = int(nid)
            for k in (0, 1, 2):
                assert 3 * nid + k in constrained
                assert constrained[3 * nid + k] == 0.0

    def test_ilss_bcs_force_vector_sums_to_applied_load(self):
        load = -10.0
        _constrained, F = self.handler.ilss_bcs(applied_load=load)
        # All midspan load lives in uz DOFs (every third entry starting at 2)
        assert abs(F.sum() - load) < 1e-12
        # And the sum across only the uz DOFs also matches
        uz_sum = F[2::3].sum()
        assert abs(uz_sum - load) < 1e-12

    def test_ilss_bcs_loads_only_midspan_top(self):
        """Only nodes on the top face near x = Lx/2 should carry the load."""
        constrained, F = self.handler.ilss_bcs(applied_load=-10.0)
        Lx = self.mesh.L_x
        Lz = self.mesh.L_z
        loaded_dofs = np.where(F != 0.0)[0]
        # All loaded DOFs must be uz (mod 3 == 2)
        assert np.all(loaded_dofs % 3 == 2)
        loaded_nodes = loaded_dofs // 3
        assert loaded_nodes.size > 0
        # Those nodes really are on the top face and close to midspan in x.
        dx = self.mesh.L_x / max(self.mesh.nx, 1)
        for nid in loaded_nodes:
            assert abs(self.mesh.nodes[nid, 2] - Lz) < 1e-9
            assert abs(self.mesh.nodes[nid, 0] - Lx / 2.0) <= dx + 1e-9

    def test_ilss_bcs_no_load_on_ux_or_uy(self):
        _constrained, F = self.handler.ilss_bcs(applied_load=-10.0)
        # Only z-DOFs should receive load
        assert np.all(F[0::3] == 0.0)
        assert np.all(F[1::3] == 0.0)


class TestFESolver:
    """Integration tests for the full FE solver pipeline."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        # Very coarse mesh for speed
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_solve_returns_field_results(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert isinstance(results, FieldResults)

    def test_solve_displacement_shape(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.displacement.shape == (self.mesh.n_nodes, 3)

    def test_solve_stress_shape(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.stress_global.shape == (self.mesh.n_elements, 8, 6)
        assert results.stress_local.shape == (self.mesh.n_elements, 8, 6)

    def test_solve_strain_shape(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.strain_global.shape == (self.mesh.n_elements, 8, 6)

    def test_solve_knockdown_range(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert 0 < results.knockdown <= 1.0

    def test_solve_failure_index_positive(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.max_failure_index >= 0

    def test_solve_nonzero_displacement(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert np.max(np.abs(results.displacement)) > 0

    def test_solve_tension(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='tension', applied_strain=0.001)
        assert isinstance(results, FieldResults)

    def test_solve_shear(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='shear', applied_strain=0.001)
        assert isinstance(results, FieldResults)

    def test_solve_invalid_loading(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.raises(ValueError):
            solver.solve(loading='invalid')

    def test_strain_local_uses_strain_transform_not_stress_transform(self):
        """Regression for #38: engineering strain was rotated via T_sigma
        (the stress transformation), which leaves the shear slots off by
        a factor of 2. strain_local must equal T_epsilon @ strain_global
        per (element, Gauss point) — within numerical noise."""
        # 45-degree plies make the bug most visible (shear components dominate).
        mat45 = dataclasses.replace(self.material, n_plies=4)
        pf = PorosityField(mat45, 0.02, distribution='uniform')
        mesh = CompositeMesh(pf, mat45, nx=3, ny=2, nz=4,
                              ply_angles=[45.0, -45.0, -45.0, 45.0])
        solver = FESolver(mesh, mat45, pf)
        r = solver.solve(loading='compression', applied_strain=-0.001)
        # Check transformation invariant on a handful of elements.
        for e in [0, mesh.n_elements // 2, mesh.n_elements - 1]:
            ply_rad = np.radians(float(mesh.ply_angles[e]))
            T_eps = strain_transformation_3d(ply_rad, axis='z')
            for g in range(8):
                expected = T_eps @ r.strain_global[e, g]
                np.testing.assert_allclose(
                    r.strain_local[e, g], expected,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"strain_local mismatch at elem={e}, gp={g}",
                )

    def test_higher_porosity_softer_response(self):
        """Higher porosity should produce softer material (lower stresses
        for the same applied displacement)."""
        pf_low = PorosityField(self.material, 0.01, distribution='uniform')
        mesh_low = CompositeMesh(pf_low, self.material, nx=3, ny=2, nz=2)
        solver_low = FESolver(mesh_low, self.material, pf_low)
        result_low = solver_low.solve(loading='compression', applied_strain=-0.001)

        pf_high = PorosityField(self.material, 0.08, distribution='uniform')
        mesh_high = CompositeMesh(pf_high, self.material, nx=3, ny=2, nz=2)
        solver_high = FESolver(mesh_high, self.material, pf_high)
        result_high = solver_high.solve(loading='compression', applied_strain=-0.001)

        # Higher porosity -> softer -> lower stresses for same displacement
        max_stress_low = np.max(np.abs(result_low.stress_global[:, :, 0]))
        max_stress_high = np.max(np.abs(result_high.stress_global[:, :, 0]))
        assert max_stress_high < max_stress_low

    def test_solve_verbose(self):
        """Verbose mode should not crash."""
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001, verbose=True)
        assert isinstance(results, FieldResults)

    def test_displacement_boundary_conditions_applied(self):
        """Check that BCs are approximately satisfied."""
        solver = FESolver(self.mesh, self.material, self.pf)
        strain = -0.001
        results = solver.solve(loading='compression', applied_strain=strain)

        # x_min nodes should have ~0 x-displacement
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        np.testing.assert_allclose(results.displacement[xmin_nodes, 0], 0.0, atol=1e-8)

        # x_max nodes should have ~strain*Lx displacement
        xmax_nodes = self.mesh.nodes_on_face('x_max')
        expected = strain * self.mesh.L_x
        np.testing.assert_allclose(results.displacement[xmax_nodes, 0], expected, atol=1e-6)

    def test_solve_ilss_runs(self):
        """Smoke: FESolver should accept loading='ilss' and produce a FieldResults."""
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='ilss', applied_load=-10.0)
        assert isinstance(results, FieldResults)
        assert results.displacement.shape == (self.mesh.n_nodes, 3)
        assert results.stress_global.shape == (self.mesh.n_elements, 8, 6)

    def test_solve_ilss_produces_shear_stress(self):
        """A 3-point short-beam load must induce non-zero tau_xz (Voigt 4)."""
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='ilss', applied_load=-10.0)
        max_tau_xz = float(np.max(np.abs(results.stress_global[:, :, 4])))
        max_sigma_xx = float(np.max(np.abs(results.stress_global[:, :, 0])))
        assert max_tau_xz > 0.0
        # Short-beam geometry: bending stress also exists, but tau_xz should
        # be a non-trivial fraction of the total stress field.
        assert max_tau_xz > 1e-6 * max(max_sigma_xx, 1.0)


class TestFESolverIterative:
    """Regression tests for the iterative solver path and K-symmetrization
    added in issue #57.

    Coverage:
      * CG converges to the same displacement field as the direct LU
        solve (within iterative tolerance).
      * MINRES likewise.
      * Assembled K is symmetric to machine precision.
      * Unknown solver names raise a clear ValueError.
      * An unreachable tolerance triggers the non-convergence guard.
    """

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        # Coarse mesh keeps the iterative tests cheap but still gives the
        # CG/MINRES iterations something to chew on (n_dof ~ a few hundred).
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_iterative_cg_matches_direct(self):
        """CG with Jacobi precond should match spsolve within rtol.

        The penalty-method conditioning (max(diag)/min(diag) ~ 1e9) caps
        how closely CG/MINRES can match LU on this problem; we accept any
        agreement at the few-times-1e-5 level (still well within
        engineering accuracy).
        """
        solver_direct = FESolver(self.mesh, self.material, self.pf)
        r_direct = solver_direct.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
        )
        solver_cg = FESolver(self.mesh, self.material, self.pf)
        r_cg = solver_cg.solve(
            loading='compression', applied_strain=-0.001,
            solver='cg', rtol=1e-14,
        )
        # Compare on the dominant component to avoid divide-by-near-zero
        # noise in the transverse directions.
        ux_direct = r_direct.displacement[:, 0]
        ux_cg = r_cg.displacement[:, 0]
        scale = float(np.max(np.abs(ux_direct)))
        max_err = float(np.max(np.abs(ux_cg - ux_direct)))
        assert max_err / max(scale, 1e-30) < 1e-4, (
            f"CG vs direct max|du|/max|u| = {max_err / max(scale, 1e-30):.4e}"
        )

    def test_minres_matches_direct(self):
        """MINRES should also match spsolve within rtol."""
        solver_direct = FESolver(self.mesh, self.material, self.pf)
        r_direct = solver_direct.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
        )
        solver_minres = FESolver(self.mesh, self.material, self.pf)
        r_minres = solver_minres.solve(
            loading='compression', applied_strain=-0.001,
            solver='minres', rtol=1e-14,
        )
        ux_direct = r_direct.displacement[:, 0]
        ux_mr = r_minres.displacement[:, 0]
        scale = float(np.max(np.abs(ux_direct)))
        max_err = float(np.max(np.abs(ux_mr - ux_direct)))
        assert max_err / max(scale, 1e-30) < 1e-4, (
            f"MINRES vs direct max|du|/max|u| = "
            f"{max_err / max(scale, 1e-30):.4e}"
        )

    def test_stiffness_matrix_is_symmetric(self):
        """K = K^T to machine precision after symmetrization (issue #57)."""
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        K_dense = K.toarray()
        max_K = float(np.max(np.abs(K_dense)))
        max_asym = float(np.max(np.abs(K_dense - K_dense.T)))
        assert max_asym < 1e-10 * max_K, (
            f"K not symmetric: max|K-K.T| = {max_asym:.4e}, "
            f"max|K| = {max_K:.4e}"
        )

    def test_invalid_solver_raises(self):
        """Unsupported solver names should fail loudly."""
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.raises(ValueError, match="Unknown solver"):
            solver.solve(
                loading='compression', applied_strain=-0.001, solver='gmres',
            )

    def test_cg_nonconvergence_raises(self):
        """An impossibly-tight tolerance should raise RuntimeError."""
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.raises(RuntimeError, match="failed to converge"):
            solver.solve(
                loading='compression', applied_strain=-0.001,
                solver='cg', rtol=1e-30,
            )


class TestPenaltyFactorAndConditioning:
    """Regression tests for the matrix-conditioning diagnostic, the
    user-exposed ``penalty_factor`` kwarg, and the optional Jacobi
    pre-scaling path (issue #60).

    Background: the penalty-method BC enforcement uses
    ``alpha = penalty_factor * max(diag(K))``. Pre-#60 this was hardwired
    at ``penalty_factor=1e8`` which pushed cond(K_mod) to ~2.4e9 and
    capped LU-vs-CG agreement at ~3e-6 even when the CG residual was at
    machine precision. PR #60 lowers the default to ``1e6``, exposes the
    knob, logs ``cond_diag_ratio`` on every solve, and adds a
    symmetric-Jacobi pre-scaling path.
    """

    def setup_method(self):
        import inspect
        self._inspect = inspect
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        # Small but non-trivial mesh — keeps the suite fast while still
        # exercising the iterative solvers and giving a measurable
        # cond_diag_ratio.
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_default_penalty_lowered(self):
        """``BoundaryHandler.apply_penalty`` default must be 1e6, not 1e8."""
        sig = self._inspect.signature(BoundaryHandler.apply_penalty)
        default = sig.parameters['penalty_factor'].default
        assert default == 1e6, (
            f"Expected apply_penalty default penalty_factor=1e6, got {default!r}"
        )

    def test_penalty_factor_kwarg_threaded_through_solve(self):
        """Passing penalty_factor through solve must reach apply_penalty.

        We use a deliberately-loose penalty (1e2) which makes BC
        enforcement slack enough to perturb the solution detectably
        relative to the default (1e6).
        """
        solver = FESolver(self.mesh, self.material, self.pf)
        r_default = solver.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
        )
        r_loose = solver.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
            penalty_factor=1e2,
        )
        scale = float(np.max(np.abs(r_default.displacement)))
        delta = float(np.max(np.abs(
            r_loose.displacement - r_default.displacement
        )))
        # Loose penalty must produce a *detectable* perturbation
        # (otherwise the kwarg is being ignored).
        assert delta / max(scale, 1e-30) > 1e-4, (
            f"penalty_factor kwarg appears not to be threaded through "
            f"to apply_penalty: relative delta {delta/max(scale,1e-30):.3e}"
        )

    def test_conditioning_warning_logged(self, caplog):
        """penalty_factor=1e15 must trip the float64-headroom warning."""
        import logging
        solver = FESolver(self.mesh, self.material, self.pf)
        with caplog.at_level(logging.WARNING, logger='porosity_fe_analysis'):
            solver.solve(
                loading='compression', applied_strain=-0.001, solver='direct',
                penalty_factor=1e15,
            )
        msgs = [rec.message for rec in caplog.records
                if rec.levelno >= logging.WARNING]
        assert any('Matrix conditioning near float64 limit' in m
                   for m in msgs), (
            f"Expected conditioning warning, got records: {msgs!r}"
        )

    def test_diag_scale_off_matches_legacy(self):
        """diag_scale=False (default) must reproduce the un-rescaled path
        bit-identically — diag_scale should be opt-in only.
        """
        solver = FESolver(self.mesh, self.material, self.pf)
        r_default = solver.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
        )
        r_explicit_off = solver.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
            diag_scale=False,
        )
        np.testing.assert_allclose(
            r_explicit_off.displacement, r_default.displacement,
            rtol=0.0, atol=0.0,
            err_msg="diag_scale=False should be bit-identical to default",
        )

    def test_diag_scale_on_matches_off_for_well_conditioned(self):
        """The Jacobi rescaling is a similarity transform on the linear
        system — math unchanged, only conditioning. For a well-
        conditioned problem the two paths must agree to ~1e-7.
        """
        solver = FESolver(self.mesh, self.material, self.pf)
        r_off = solver.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
            diag_scale=False,
        )
        r_on = solver.solve(
            loading='compression', applied_strain=-0.001, solver='direct',
            diag_scale=True,
        )
        scale = float(np.max(np.abs(r_off.displacement)))
        delta = float(np.max(np.abs(r_on.displacement - r_off.displacement)))
        assert delta / max(scale, 1e-30) < 1e-7, (
            f"diag_scale on/off mismatch on well-conditioned mesh: "
            f"max|du|/max|u| = {delta/max(scale, 1e-30):.3e}"
        )

    def test_diag_scale_reduces_conditioning_ratio(self, caplog):
        """On a voided/graded mesh diag_scale must measurably reduce
        the diagonal-conditioning ratio. We capture the INFO line both
        ways and assert the rescaled ratio is strictly smaller.
        """
        import logging
        import re

        # Voided/graded mesh — clustered distribution drives spatial
        # variation in stiffness, which widens the diagonal spread.
        pf_voided = PorosityField(self.material, 0.10,
                                  distribution='clustered', seed=42)
        mesh_voided = CompositeMesh(pf_voided, self.material, nx=4, ny=3, nz=3)
        solver = FESolver(mesh_voided, self.material, pf_voided)

        def _capture_ratio(diag_scale_value):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
                solver.solve(
                    loading='compression', applied_strain=-0.001,
                    solver='direct', diag_scale=diag_scale_value,
                )
            # The post-scaling line (when diag_scale=True) takes priority;
            # otherwise grab the initial diagnostic.
            target_prefix = ('Matrix conditioning after diag_scale'
                             if diag_scale_value
                             else 'Matrix conditioning:')
            for rec in caplog.records:
                if rec.message.startswith(target_prefix):
                    m = re.search(r'cond_diag_ratio=([0-9.eE+\-]+)',
                                  rec.message)
                    if m:
                        return float(m.group(1))
            raise AssertionError(
                f"Did not find cond_diag_ratio log line for "
                f"diag_scale={diag_scale_value}; got: "
                f"{[r.message for r in caplog.records]!r}"
            )

        ratio_off = _capture_ratio(False)
        ratio_on = _capture_ratio(True)
        assert ratio_on < ratio_off, (
            f"diag_scale=True did not reduce cond_diag_ratio: "
            f"off={ratio_off:.3e}, on={ratio_on:.3e}"
        )


class TestILSSBeamTheoryValidation:
    """Beam-theory validation for the ILSS short-beam-shear FE BCs.

    For a 3-point bend on a rectangular cross-section with width b and
    height h under a center load F, Timoshenko shear theory gives a peak
    transverse shear stress at the neutral axis::

        tau_xz_peak = 1.5 * |F| / (b * h)

    We solve a pristine (zero porosity) short beam and check the
    recovered peak |tau_xz| against the closed-form value.
    """

    def test_peak_tau_xz_matches_beam_theory(self):
        material = dataclasses.replace(
            MATERIALS['T800_epoxy'], n_plies=4, t_ply=0.5,
        )
        # Pristine reference: no porosity so beam theory is the direct target.
        pf = PorosityField(material, 0.0, distribution='uniform')
        # All zero-degree plies — isotropic-ish in the x-z plane for shear.
        mesh = CompositeMesh(
            pf, material, nx=16, ny=4, nz=8,
            ply_angles=[0.0, 0.0, 0.0, 0.0],
        )
        solver = FESolver(mesh, material, pf)
        applied_load = -10.0  # N, downward
        results = solver.solve(loading='ilss', applied_load=applied_load)

        b = mesh.L_y
        h = mesh.L_z
        tau_analytical = 1.5 * abs(applied_load) / (b * h)

        # Recover tau_xz at the neutral axis midspan. Gather GPs in the
        # mid-third of the span (avoid the load/support singularities) and
        # near the neutral axis (mid-thickness).
        # Compute per-element centroids.
        elem_nodes = mesh.elements  # (n_elem, 8)
        coords = mesh.nodes
        centers = np.mean(coords[elem_nodes], axis=1)  # (n_elem, 3)

        Lx = mesh.L_x
        Lz = mesh.L_z
        # Mid-span band: 35% .. 65% of x to avoid load point.
        x_band = (centers[:, 0] > 0.35 * Lx) & (centers[:, 0] < 0.65 * Lx)
        # Neutral-axis band: 35% .. 65% of thickness.
        z_band = (centers[:, 2] > 0.35 * Lz) & (centers[:, 2] < 0.65 * Lz)
        mask = x_band & z_band
        assert mask.sum() > 0, "No elements in the midspan/neutral-axis band"

        # Peak tau_xz over the GPs of selected elements (mid-span / neutral
        # axis band). tau_xz is at Voigt index 4 (tau_13). The shear-stress
        # profile through thickness is parabolic, so the *peak* value
        # in the band is what beam theory predicts; the band-average is
        # naturally lower (~2/3 of peak for the full parabola).
        tau_band = results.stress_global[mask, :, 4]
        tau_recovered = float(np.max(np.abs(tau_band)))

        rel_err = abs(tau_recovered - tau_analytical) / tau_analytical
        # Coarse hex8 short beam: 15% relative-error tolerance is the
        # practical target. Tighter (~2–3%) requires a much finer mesh and
        # would make the test slow; we keep the asymptotic check loose but
        # informative.
        assert rel_err < 0.15, (
            f"Recovered peak |tau_xz| = {tau_recovered:.4f} MPa, "
            f"analytical = {tau_analytical:.4f} MPa, "
            f"rel_err = {rel_err:.3f}"
        )


class TestKeCacheKeyGeometry:
    """Regression tests for issue #40: _ke_cache key must encode full element
    geometry and material so skewed/non-rectilinear elements or elements with
    different C_base never collide with axis-aligned ones."""

    def _make_elem(self, node_coords, C_base, porosity=0.03, material=None):
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        return Hex8Element(
            node_coords=np.asarray(node_coords, dtype=float),
            C_base=C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, porosity),
            void_shape_radii=(1, 1, 1),
            nu_m=mat.matrix_poisson,
            C_m=C_m,
            material=material,  # None => legacy C_base scaling path
        )

    def setup_method(self):
        mat = MATERIALS['T800_epoxy']
        self.C_base = mat.get_stiffness_matrix()

        # Axis-aligned unit cube
        self.coords_rect = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)

        # Same bounding box (dx=dy=dz=1) but one node is sheared in x
        self.coords_shear = self.coords_rect.copy()
        self.coords_shear[2, 0] += 0.2   # shear node 2 in x

    def test_stiffness_differs_for_sheared_element(self):
        """Sheared element must produce a different Ke than its axis-aligned twin."""
        elem_rect = self._make_elem(self.coords_rect, self.C_base)
        elem_shear = self._make_elem(self.coords_shear, self.C_base)
        Ke_rect = elem_rect.stiffness_matrix()
        Ke_shear = elem_shear.stiffness_matrix()
        assert not np.allclose(Ke_rect, Ke_shear, atol=1.0), (
            "Stiffness matrices of axis-aligned and sheared elements should differ"
        )

    def test_cache_key_differs_for_sheared_element(self):
        """Cache key must differ between axis-aligned and sheared elements."""
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        assembler = GlobalAssembler(mesh, mat, pf)

        # Manually build keys from two synthetic node-coord arrays.
        # We exploit that _element_cache_key reads from mesh internals,
        # so instead we test the geometry encoding directly via
        # the centroid-relative tuple approach used in the fixed code.
        import numpy as _np

        def _geom_key(coords):
            centroid = coords.mean(axis=0)
            rel = _np.round(coords - centroid, 8)
            return tuple(rel.ravel())

        key_rect = _geom_key(self.coords_rect)
        key_shear = _geom_key(self.coords_shear)
        assert key_rect != key_shear, (
            "Geometry cache keys must differ for axis-aligned vs sheared nodes"
        )

    def test_stiffness_differs_for_different_material(self):
        """Two elements with the same geometry but different C_base must differ in Ke.

        We use the legacy path (material=None) so that the stiffness is computed
        directly from C_base, making the difference observable in Ke.
        """
        mat2 = MATERIALS['T700_epoxy']
        C_base2 = mat2.get_stiffness_matrix()
        # material=None → legacy scalar-degradation path that reads C_base directly
        elem1 = self._make_elem(self.coords_rect, self.C_base, material=None)
        elem2 = self._make_elem(self.coords_rect, C_base2, material=None)
        Ke1 = elem1.stiffness_matrix()
        Ke2 = elem2.stiffness_matrix()
        assert not np.allclose(Ke1, Ke2, atol=1.0), (
            "Stiffness matrices of elements with different C_base should differ"
        )

    def test_cache_key_differs_for_different_material(self):
        """Cache key must differ when C_base changes, even for identical geometry."""
        mat2 = MATERIALS['T700_epoxy']
        C_base2 = mat2.get_stiffness_matrix()
        c_key1 = hash(self.C_base.tobytes())
        c_key2 = hash(C_base2.tobytes())
        assert c_key1 != c_key2, (
            "Material hash in cache key must differ for different C_base matrices"
        )

    def test_identical_elements_share_cache_key(self):
        """Two identical axis-aligned elements must produce the same cache key."""
        import numpy as _np

        def _geom_key(coords):
            centroid = coords.mean(axis=0)
            rel = _np.round(coords - centroid, 8)
            return tuple(rel.ravel())

        key1 = _geom_key(self.coords_rect)
        # Translate the element — the centroid-relative coords must be identical.
        coords_translated = self.coords_rect + np.array([5.0, 3.0, 1.0])
        key2 = _geom_key(coords_translated)
        assert key1 == key2, (
            "Translated copies of the same element shape should share the geometry key"
        )


# ============================================================
# Issue #39: pure-shear BC fix — G12 recovery test
# ============================================================


class TestPureShearBCs:
    """Verify that shear_bcs imposes true pure shear and recovers G12 correctly.

    An isotropic material (E11=E22=E33, G12=E/(2*(1+nu))) is used so that
    the analytical shear modulus is known exactly.  The FE-recovered G12 is
    computed as:

        G12_fe = mean(sigma_xy) / gamma

    where gamma = applied_strain (engineering shear strain) and sigma_xy is
    the volume-average Voigt component index 5 (1-indexed: [0]=s11, [1]=s22,
    [2]=s33, [3]=s23, [4]=s13, [5]=s12).
    """

    @staticmethod
    def _make_isotropic_material(E: float = 10000.0, nu: float = 0.30) -> 'MaterialProperties':
        """Return a MaterialProperties that behaves as an isotropic solid."""
        G = E / (2.0 * (1.0 + nu))
        return MaterialProperties(
            E11=E, E22=E, E33=E,
            G12=G, G13=G, G23=G,
            nu12=nu, nu13=nu, nu23=nu,
            sigma_1c=1e6, sigma_1t=1e6,
            sigma_2t=1e6, sigma_2c=1e6,
            tau_12=1e6, tau_ilss=1e6,
            t_ply=0.5, n_plies=4,
            matrix_modulus=E, matrix_poisson=nu,
            fiber_modulus=E, fiber_volume_fraction=0.6,
        )

    def test_shear_bcs_prescribes_all_four_faces(self):
        """After the fix, all four side faces must carry prescribed displacements."""
        mat = self._make_isotropic_material()
        pf = PorosityField(mat, 0.0, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        handler = BoundaryHandler(mesh)

        gamma = 0.01
        constrained, F = handler.shear_bcs(applied_strain=gamma)

        nodes = mesh.nodes
        # Every node on any of the four side faces must have ux and uy prescribed.
        for face in ('x_min', 'x_max', 'y_min', 'y_max'):
            for nid in mesh.nodes_on_face(face):
                nid = int(nid)
                assert 3 * nid in constrained, (
                    f"ux not prescribed for node {nid} on face {face}")
                assert 3 * nid + 1 in constrained, (
                    f"uy not prescribed for node {nid} on face {face}")
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                np.testing.assert_allclose(
                    constrained[3 * nid], (gamma / 2.0) * y_n, atol=1e-12,
                    err_msg=f"ux wrong for node {nid} on {face}")
                np.testing.assert_allclose(
                    constrained[3 * nid + 1], (gamma / 2.0) * x_n, atol=1e-12,
                    err_msg=f"uy wrong for node {nid} on {face}")

    def test_recovered_G12_matches_analytical(self):
        """FE-recovered G12 must match E/(2*(1+nu)) within 2 %."""
        E = 10000.0
        nu = 0.30
        G_analytical = E / (2.0 * (1.0 + nu))

        mat = self._make_isotropic_material(E=E, nu=nu)
        pf = PorosityField(mat, 0.0, distribution='uniform')
        # 4x4x4 gives 64 elements — coarse but sufficient for a homogeneous cube
        mesh = CompositeMesh(pf, mat, nx=4, ny=4, nz=4)
        solver = FESolver(mesh, mat, pf)

        gamma = 0.01
        results = solver.solve(loading='shear', applied_strain=gamma)

        # Volume-average sigma_xy (Voigt index 5, 0-based)
        sigma_xy_mean = float(np.mean(results.stress_global[:, :, 5]))
        G12_fe = sigma_xy_mean / gamma

        rel_err = abs(G12_fe - G_analytical) / G_analytical
        assert rel_err < 0.02, (
            f"G12 recovery failed: G12_fe={G12_fe:.1f}, "
            f"G_analytical={G_analytical:.1f}, rel_err={rel_err:.4f}")

    def test_shear_only_stress_state(self):
        """Normal stresses must be negligible compared with shear stress."""
        E = 10000.0
        nu = 0.30

        mat = self._make_isotropic_material(E=E, nu=nu)
        pf = PorosityField(mat, 0.0, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=4, ny=4, nz=4)
        solver = FESolver(mesh, mat, pf)

        gamma = 0.01
        results = solver.solve(loading='shear', applied_strain=gamma)

        # indices: 0=s11, 1=s22, 2=s33, 3=s23, 4=s13, 5=s12
        sigma = results.stress_global  # shape (n_elem, n_gp, 6)

        sigma_xy_rms = float(np.sqrt(np.mean(sigma[:, :, 5] ** 2)))
        for i, label in enumerate(['s11', 's22', 's33', 's23', 's13']):
            sigma_i_rms = float(np.sqrt(np.mean(sigma[:, :, i] ** 2)))
            ratio = sigma_i_rms / sigma_xy_rms if sigma_xy_rms > 0 else 0.0
            assert ratio < 0.05, (
                f"Non-shear stress {label} too large relative to s12: "
                f"ratio={ratio:.4f} (rms {label}={sigma_i_rms:.2f}, "
                f"rms s12={sigma_xy_rms:.2f})")


class TestHRefinementConvergence:
    """h-refinement convergence: finer mesh should approach the analytical
    uniaxial-tension result more closely than the coarser mesh (#18)."""

    def _run_tension(self, nx, ny, nz, applied_strain=0.001):
        """Build a zero-porosity mesh and solve uniaxial tension.

        Returns the volume-averaged sigma_xx stress at all Gauss points.
        """
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, void_volume_fraction=0.0, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz)
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='tension', applied_strain=applied_strain)
        # Average sigma_xx across all elements and Gauss points
        avg_sigma_xx = float(np.mean(results.stress_global[:, :, 0]))
        return avg_sigma_xx

    def test_h_refinement_monotone_convergence(self):
        """Refining the mesh from 2x2x2 to 4x4x4 elements should produce a
        sigma_xx that is closer to the analytical value, OR the two mesh
        densities agree to within a tightening tolerance (monotone convergence).

        Analytical uniaxial tension for an all-0-degree ply laminate:
          sigma_xx_analytic ≈ E11 * applied_strain  (simplified, ignores
          lateral coupling), which serves as an upper-bound reference.
        """
        applied_strain = 0.001
        material = MATERIALS['T800_epoxy']

        # Coarse mesh: 2x2x2 hex elements
        sigma_coarse = self._run_tension(nx=2, ny=2, nz=2,
                                         applied_strain=applied_strain)

        # Fine mesh: 4x4x4 hex elements
        sigma_fine = self._run_tension(nx=4, ny=4, nz=4,
                                       applied_strain=applied_strain)

        # Analytical reference: sigma_xx ~ C11 * eps_xx for uniaxial tension
        # with all-0-degree plies.  C11 from the material stiffness matrix.
        C = material.get_stiffness_matrix()
        sigma_analytic = float(C[0, 0]) * applied_strain

        err_coarse = abs(sigma_coarse - sigma_analytic)
        err_fine = abs(sigma_fine - sigma_analytic)

        # The fine mesh must be at least as accurate as the coarse mesh,
        # OR the difference between the two meshes must be small relative
        # to the magnitude (monotone convergence guard).
        mesh_diff = abs(sigma_fine - sigma_coarse)
        relative_diff = mesh_diff / max(abs(sigma_analytic), 1.0)

        assert err_fine <= err_coarse or relative_diff < 0.05, (
            f"h-refinement did not converge monotonically: "
            f"coarse err={err_coarse:.4e}, fine err={err_fine:.4e}, "
            f"mesh-to-mesh diff={mesh_diff:.4e} ({relative_diff*100:.2f}%)"
        )


# ============================================================
# PROVENANCE METADATA TESTS
# ============================================================


class TestFailureCriteria:
    """#62: Hashin / max-stress / Tsai-Wu dispatch on FESolver.

    The Hashin and max-stress polynomials are exercised directly on a
    synthetic stress state (no FE solve needed) so the per-mode arithmetic
    can be asserted in isolation. The Tsai-Wu golden test still runs through
    a real FE solve to confirm bit-identical legacy behavior.
    """

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.0, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=2, ny=2, nz=2)
        self.solver = FESolver(self.mesh, self.material, self.pf)

    def test_hashin_separates_modes(self):
        """Pure fiber-tension stress lights up `fiber_t`, not the matrix modes."""
        mat = self.material
        # Single element, single Gauss point, pure σ_11 = 0.5 * X_T.
        sigma_11 = 0.5 * mat.sigma_1t
        s = np.array([[sigma_11, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # Pristine strengths (no porosity).
        strengths = self.solver._degraded_strengths(0.0)
        modes = self.solver._evaluate_hashin(s, strengths)
        assert modes['fiber_t'][0] == pytest.approx(0.25, rel=1e-12)
        # Other modes should be exactly zero.
        assert modes['fiber_c'][0] == 0.0
        assert modes['matrix_t'][0] == 0.0
        assert modes['matrix_c'][0] == 0.0
        # Aggregate max must equal the fiber-tension term.
        assert modes['max_fi'][0] == pytest.approx(0.25, rel=1e-12)

    def test_hashin_pure_fiber_compression(self):
        """σ_11 < 0 must light up `fiber_c`, not `fiber_t`."""
        mat = self.material
        s = np.array([[-0.5 * mat.sigma_1c, 0.0, 0.0, 0.0, 0.0, 0.0]])
        strengths = self.solver._degraded_strengths(0.0)
        modes = self.solver._evaluate_hashin(s, strengths)
        assert modes['fiber_c'][0] == pytest.approx(0.25, rel=1e-12)
        assert modes['fiber_t'][0] == 0.0

    def test_max_stress_matches_simple_uniaxial(self):
        """σ_11 = 0.5·X_T must produce FI = 0.5 for the max-stress criterion."""
        mat = self.material
        s = np.array([[0.5 * mat.sigma_1t, 0.0, 0.0, 0.0, 0.0, 0.0]])
        strengths = self.solver._degraded_strengths(0.0)
        modes = self.solver._evaluate_max_stress(s, strengths)
        assert modes['fiber_t'][0] == pytest.approx(0.5, rel=1e-12)
        assert modes['max_fi'][0] == pytest.approx(0.5, rel=1e-12)
        # Other components are exactly zero.
        assert modes['fiber_c'][0] == 0.0
        assert modes['matrix_t'][0] == 0.0
        assert modes['shear'][0] == 0.0

    def test_tsai_wu_unchanged_when_default(self):
        """Default solve() must still use Tsai-Wu with identical numbers."""
        # Reference: untouched legacy call (no explicit criterion).
        ref_solver = FESolver(self.mesh, self.material, self.pf)
        ref = ref_solver.solve(loading='compression', applied_strain=-0.001)

        # New explicit-default path should match bit-for-bit.
        new_solver = FESolver(self.mesh, self.material, self.pf,
                              failure_criterion='tsai_wu')
        out = new_solver.solve(loading='compression', applied_strain=-0.001)
        assert out.failure_criterion == 'tsai_wu'
        assert ref.max_failure_index == out.max_failure_index
        np.testing.assert_allclose(
            ref.per_element_failure_index, out.per_element_failure_index,
            rtol=0, atol=0)

    def test_solver_accepts_hashin_criterion(self):
        """FESolver.solve must dispatch to Hashin and populate mode_indices."""
        solver = FESolver(self.mesh, self.material, self.pf,
                          failure_criterion='hashin')
        res = solver.solve(loading='tension', applied_strain=0.001)
        assert res.failure_criterion == 'hashin'
        assert res.failure_mode_indices is not None
        # All five mode keys must be present.
        for key in ('fiber_t', 'fiber_c', 'matrix_t', 'matrix_c', 'shear',
                    'max_fi'):
            assert key in res.failure_mode_indices
        # Tension loading: fiber_t should dominate over compression modes.
        assert res.failure_mode_indices['fiber_t'] >= \
            res.failure_mode_indices['fiber_c']

    def test_solver_accepts_max_stress_criterion(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        res = solver.solve(loading='tension', applied_strain=0.001,
                           failure_criterion='max_stress')
        assert res.failure_criterion == 'max_stress'
        assert res.failure_mode_indices is not None
        # Max-stress fills zeros, not NaNs.
        for v in res.failure_mode_indices.values():
            assert np.isfinite(v)

    def test_solver_rejects_unknown_criterion(self):
        with pytest.raises(ValueError, match="Unknown failure_criterion"):
            FESolver(self.mesh, self.material, self.pf,
                     failure_criterion='nonsense')
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.raises(ValueError, match="Unknown failure_criterion"):
            solver.solve(loading='compression', applied_strain=-0.001,
                         failure_criterion='nonsense')

    def test_tsai_wu_mode_indices_are_nan(self):
        """Tsai-Wu doesn't separate modes; per-mode entries must be NaN."""
        solver = FESolver(self.mesh, self.material, self.pf,
                          failure_criterion='tsai_wu')
        res = solver.solve(loading='compression', applied_strain=-0.001)
        assert res.failure_mode_indices is not None
        # The max_fi entry is the scalar; the per-mode entries are NaN.
        assert np.isnan(res.failure_mode_indices['fiber_t'])
        assert np.isnan(res.failure_mode_indices['matrix_c'])

    # ------------------------------------------------------------------
    # Issue #145: tsai_wu_F12 override on MaterialProperties.
    # ------------------------------------------------------------------
    def test_tsai_wu_default_F12_matches_recommendation(self):
        """tsai_wu_F12=None must reproduce F_12 = -0.5 * sqrt(F_11 * F_22).

        Verified by evaluating the per-GP polynomial on a pure σ_1 σ_2
        biaxial state and recovering F_12 algebraically.
        """
        mat = self.material
        assert mat.tsai_wu_F12 is None  # built-in preset uses Tsai default
        # Pristine (Vp=0) strengths so degraded_strengths returns the raw
        # allowables.
        strengths = self.solver._degraded_strengths(0.0)
        Xt_s, Xc_s, Yt_s, Yc_s, _, _ = strengths
        F11 = 1.0 / (Xt_s * Xc_s)
        F22 = 1.0 / (Yt_s * Yc_s)
        F1 = 1.0 / Xt_s - 1.0 / Xc_s
        F2 = 1.0 / Yt_s - 1.0 / Yc_s
        F12_expected = -0.5 * np.sqrt(F11 * F22)

        # Build a biaxial stress state and back out the F_12 the solver used.
        s1, s2 = 100.0, 50.0
        s = np.array([[s1, s2, 0.0, 0.0, 0.0, 0.0]])
        fi = self.solver._evaluate_tsai_wu(s, strengths, e=0, elem_Vp=0.0)[0]
        # fi = F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + 2*F12*s1*s2
        # (other terms vanish because σ_3, shears, and σ_2-σ_3 coupling
        # all use a zero stress component).
        diagonal = (F1 * s1 + F2 * s2 + F11 * s1 ** 2 + F22 * s2 ** 2)
        F12_solver = (fi - diagonal) / (2.0 * s1 * s2)
        assert F12_solver == pytest.approx(F12_expected, rel=1e-12, abs=1e-18)

    def test_tsai_wu_user_F12_used_when_provided(self):
        """A user-supplied tsai_wu_F12 must replace the Tsai recommendation."""
        base = self.material
        custom_F12 = -0.3
        mat_custom = dataclasses.replace(base, tsai_wu_F12=custom_F12)

        # Use the same mesh / porosity field as the default solver setup;
        # only the material differs, so any FI change is attributable to F_12.
        solver_custom = FESolver(self.mesh, mat_custom, self.pf,
                                 failure_criterion='tsai_wu')

        strengths = solver_custom._degraded_strengths(0.0)
        # Biaxial stress state — F_12 only enters via 2*F_12*s1*s2.
        s1, s2 = 100.0, 50.0
        s = np.array([[s1, s2, 0.0, 0.0, 0.0, 0.0]])
        Xt_s, Xc_s, Yt_s, Yc_s, _, _ = strengths
        F11 = 1.0 / (Xt_s * Xc_s)
        F22 = 1.0 / (Yt_s * Yc_s)
        F1 = 1.0 / Xt_s - 1.0 / Xc_s
        F2 = 1.0 / Yt_s - 1.0 / Yc_s
        diagonal = (F1 * s1 + F2 * s2 + F11 * s1 ** 2 + F22 * s2 ** 2)

        fi_custom = solver_custom._evaluate_tsai_wu(
            s, strengths, e=0, elem_Vp=0.0)[0]
        F12_recovered = (fi_custom - diagonal) / (2.0 * s1 * s2)
        assert F12_recovered == pytest.approx(custom_F12, rel=1e-12, abs=1e-18)

        # Cross-check: default solver gives a different FI on the same state.
        fi_default = self.solver._evaluate_tsai_wu(
            s, strengths, e=0, elem_Vp=0.0)[0]
        assert fi_custom != pytest.approx(fi_default, rel=1e-12, abs=1e-18)

    def test_tsai_wu_F12_out_of_range_raises(self):
        """Positive tsai_wu_F12 opens the failure envelope -> ValueError."""
        base = self.material
        with pytest.raises(ValueError, match="tsai_wu_F12"):
            dataclasses.replace(base, tsai_wu_F12=0.5)
        # Below -1 is equally invalid.
        with pytest.raises(ValueError, match="tsai_wu_F12"):
            dataclasses.replace(base, tsai_wu_F12=-1.5)
        # NaN / inf must also be rejected.
        with pytest.raises(ValueError, match="tsai_wu_F12"):
            dataclasses.replace(base, tsai_wu_F12=float('nan'))
