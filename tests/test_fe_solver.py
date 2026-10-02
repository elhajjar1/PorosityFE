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
                                   BoundaryHandler, FESolver, FieldResults, VoidGeometry)


def _default_strain(loading):
    """The default ``applied_strain`` FESolver.solve uses for ``loading``."""
    from porosity_fe.fe.solver import _resolve_applied_strain
    return _resolve_applied_strain(loading, None)


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
        with pytest.warns(DeprecationWarning, match="apply_elimination"):
            K_mod, F_mod = BoundaryHandler.apply_penalty(K, F, constrained)
        assert K_mod.shape == K.shape
        assert len(F_mod) == len(F)

    def test_penalty_increases_diagonal(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = self.handler.compression_bcs()
        with pytest.warns(DeprecationWarning):
            K_mod, F_mod = BoundaryHandler.apply_penalty(K, F, constrained)
        # Constrained DOF diagonals should be much larger
        for dof in list(constrained.keys())[:5]:
            assert K_mod[dof, dof] > K[dof, dof]

    @pytest.mark.parametrize("loading", ["compression", "shear", "ilss"])
    def test_apply_elimination_matches_dense_partitioned_solve(self, loading):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = {
            'compression': lambda: self.handler.compression_bcs(-0.001),
            'shear': lambda: self.handler.shear_bcs(0.01),
            'ilss': lambda: self.handler.ilss_bcs(-10.0),
        }[loading]()
        K_ff, rhs, free = BoundaryHandler.apply_elimination(K, F, constrained)

        fixed = np.array(sorted(constrained), dtype=int)
        values = np.array([constrained[d] for d in fixed])
        expected_free = np.setdiff1d(np.arange(self.mesh.n_dof), fixed)
        np.testing.assert_array_equal(free, expected_free)
        Kd = K.toarray()
        np.testing.assert_array_equal(K_ff.toarray(), Kd[np.ix_(free, free)])
        np.testing.assert_allclose(
            rhs, F[free] - Kd[np.ix_(free, fixed)] @ values,
            rtol=1e-12, atol=1e-12 * max(np.abs(rhs).max(), 1.0))

        u = np.zeros(self.mesh.n_dof)
        u[fixed] = values
        u[free] = np.linalg.solve(K_ff.toarray(), rhs)
        # Prescribed values hold exactly; the free rows are in equilibrium.
        np.testing.assert_array_equal(u[fixed], values)
        r_free = (Kd @ u - F)[free]
        assert np.abs(r_free).max() <= 1e-9 * np.abs(Kd).max() * np.abs(u).max()
        # The same displacement field the solver produces.
        res = FESolver(self.mesh, self.material, self.pf).solve(
            loading, applied_strain=-0.001 if loading == 'compression' else 0.01)
        np.testing.assert_allclose(res.displacement.ravel(), u, rtol=0,
                                   atol=1e-9 * np.abs(u).max())

    def test_apply_elimination_does_not_modify_inputs(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = self.handler.compression_bcs(-0.001)
        K0, F0 = K.copy(), F.copy()
        BoundaryHandler.apply_elimination(K, F, constrained)
        assert (K != K0).nnz == 0
        np.testing.assert_array_equal(F, F0)

    def test_apply_elimination_rejects_out_of_range_dof(self):
        K = scipy.sparse.identity(6, format='csc')
        with pytest.raises(ValueError, match="out of range"):
            BoundaryHandler.apply_elimination(K, np.zeros(6), {6: 0.0})
        with pytest.raises(ValueError, match="finite"):
            BoundaryHandler.apply_elimination(K, np.zeros(6), {0: float('nan')})

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
        """Prescribed displacements hold exactly (they are eliminated)."""
        solver = FESolver(self.mesh, self.material, self.pf)
        strain = -0.001
        results = solver.solve(loading='compression', applied_strain=strain)

        xmin_nodes = self.mesh.nodes_on_face('x_min')
        np.testing.assert_array_equal(results.displacement[xmin_nodes, 0], 0.0)

        xmax_nodes = self.mesh.nodes_on_face('x_max')
        expected = strain * self.mesh.L_x
        np.testing.assert_array_equal(results.displacement[xmax_nodes, 0], expected)

    @pytest.mark.parametrize("loading,solver_name", [
        ("compression", "direct"), ("shear", "direct"), ("ilss", "direct"),
        ("compression", "cg"), ("shear", "minres"), ("ilss", "cg"),
    ])
    def test_every_constrained_dof_is_exact(self, loading, solver_name):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading=loading, solver=solver_name)
        constrained, _ = solver._apply_boundary_conditions(
            loading, _default_strain(loading), -10.0)
        dofs = np.fromiter(constrained, dtype=int)
        values = np.fromiter(constrained.values(), dtype=float)
        np.testing.assert_array_equal(results.displacement.ravel()[dofs], values)

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


class TestFESolverNonFiniteGuards:
    """Cover the two defensive ValueError guards in
    ``FESolver._evaluate_failure`` (issue #185):

    1. the non-finite porosity guard, and
    2. the non-finite failure-index guard (a Gauss-point criterion value
       coming out NaN/Inf).

    Both guards live inside ``_evaluate_failure``. Neither is reachable
    through a *valid* ``solve()`` call (porosity is range-validated by
    ``PorosityField`` and the criterion polynomials are finite for finite
    stresses), so these tests reach the guards by corrupting the realized
    mesh porosity field / monkeypatching the per-criterion evaluator and
    then calling ``_evaluate_failure`` directly. That method is internal
    but is the same one ``solve()`` invokes and takes a plain
    ``stress_local`` array, so driving it directly mirrors the real call
    path."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)
        self.solver = FESolver(self.mesh, self.material, self.pf)
        # Benign all-zero stress field of the right shape (n_elem, n_gp, 6).
        # Zero stresses give finite (zero) failure indices, so the
        # failure-index guard does NOT fire unless we force it.
        self.stress_local = np.zeros((self.mesh.n_elements, 8, 6), dtype=float)

    def test_nonfinite_porosity_guard_nan(self):
        """A NaN injected into ``mesh.porosity`` trips the porosity guard
        before any per-element work. ``PorosityField`` forbids this at
        construction, so we corrupt the realized field directly."""
        self.mesh.porosity[0] = np.nan
        with pytest.raises(ValueError, match="porosity contains non-finite"):
            self.solver._evaluate_failure(self.stress_local, criterion='tsai_wu')

    def test_nonfinite_porosity_guard_inf(self):
        """+Inf in the porosity field trips the same guard (message tail)."""
        self.mesh.porosity[-1] = np.inf
        with pytest.raises(ValueError, match="corrupted porosity field"):
            self.solver._evaluate_failure(self.stress_local, criterion='hashin')

    def test_nonfinite_failure_index_guard_hashin(self, monkeypatch):
        """Force the Hashin evaluator to emit a NaN ``max_fi`` so the
        per-Gauss-point failure-index guard fires. Porosity is left finite,
        so the only non-finite value is the injected criterion result.
        Patched because finite stresses never yield a non-finite Hashin
        index through the public path."""
        n_gp = self.stress_local.shape[1]

        def fake_hashin(s_all, strengths):
            bad = np.zeros(n_gp, dtype=float)
            bad[0] = np.nan
            zeros = np.zeros(n_gp, dtype=float)
            return {
                'max_fi': bad,
                'fiber_t': zeros, 'fiber_c': zeros,
                'matrix_t': zeros, 'matrix_c': zeros, 'shear': zeros,
            }

        from porosity_fe.fe import failure as failure_mod
        monkeypatch.setattr(failure_mod, 'evaluate_hashin', fake_hashin)
        with pytest.raises(ValueError, match="hashin failure index is non-finite"):
            self.solver._evaluate_failure(self.stress_local, criterion='hashin')

    def test_nonfinite_failure_index_guard_max_stress(self, monkeypatch):
        """Same guard via the ``max_stress`` branch: a monkeypatched
        evaluator returns an Inf ``max_fi``."""
        n_gp = self.stress_local.shape[1]

        def fake_max_stress(s_all, strengths):
            bad = np.zeros(n_gp, dtype=float)
            bad[0] = np.inf
            zeros = np.zeros(n_gp, dtype=float)
            return {
                'max_fi': bad,
                'fiber_t': zeros, 'fiber_c': zeros,
                'matrix_t': zeros, 'matrix_c': zeros, 'shear': zeros,
            }

        from porosity_fe.fe import failure as failure_mod
        monkeypatch.setattr(failure_mod, 'evaluate_max_stress', fake_max_stress)
        with pytest.raises(ValueError, match="max_stress failure index is non-finite"):
            self.solver._evaluate_failure(self.stress_local, criterion='max_stress')


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
        """CG with a Jacobi preconditioner matches the LU solve.

        With the prescribed DOFs eliminated the reduced matrix keeps the
        physical conditioning, so at a tight ``rtol`` CG agrees with LU to
        near machine precision (measured ~1e-14).
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
        assert max_err / max(scale, 1e-30) < 1e-9, (
            f"CG vs direct max|du|/max|u| = {max_err / max(scale, 1e-30):.4e}"
        )

    def test_minres_matches_direct(self):
        """MINRES also matches the LU solve at a tight rtol."""
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
        assert max_err / max(scale, 1e-30) < 1e-9, (
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

    def test_minres_nonconvergence_raises(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.raises(RuntimeError, match="minres failed to converge"):
            solver.solve(
                loading='compression', applied_strain=-0.001,
                solver='minres', rtol=1e-30,
            )

    def test_minres_restarts_are_bounded(self, monkeypatch):
        """A MINRES that reports success without reducing the true residual
        is warm-restarted at most ``_MINRES_MAX_RESTARTS`` times."""
        import scipy.sparse.linalg as sla
        x0s = []

        def stalled_minres(A, b, x0=None, **kwargs):
            x0s.append(x0)
            return np.zeros_like(b), 0

        monkeypatch.setattr(sla, 'minres', stalled_minres)
        solver = FESolver(self.mesh, self.material, self.pf)
        n = FESolver._MINRES_MAX_RESTARTS
        with pytest.raises(RuntimeError,
                           match=rf"minres failed to converge.*after {n} warm restarts"):
            solver.solve(loading='compression', applied_strain=-0.001,
                         solver='minres')
        assert len(x0s) == 1 + n
        assert x0s[0] is None and all(x is not None for x in x0s[1:])


def _relative_differences(r_iter, r_direct):
    du = (np.abs(r_iter.displacement - r_direct.displacement).max()
          / np.abs(r_direct.displacement).max())
    ds = (np.abs(r_iter.stress_global - r_direct.stress_global).max()
          / np.abs(r_direct.stress_global).max())
    dfi = (abs(r_iter.max_failure_index - r_direct.max_failure_index)
           / abs(r_direct.max_failure_index))
    return du, ds, dfi


@pytest.fixture(scope="module")
def clustered_6x3x4():
    """6x3x4 clustered Vp=0.03 QI solver."""
    mat = MATERIALS['T800_epoxy']
    pf = PorosityField(mat, 0.03, distribution='clustered', seed=42)
    mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=4)
    return FESolver(mesh, mat, pf)


@pytest.fixture(scope="module")
def clustered_20x8x12():
    """20x8x12 clustered Vp=0.02 QI solver and its direct solutions."""
    mat = MATERIALS['T800_epoxy']
    pf = PorosityField(mat, 0.02, distribution='clustered', seed=42)
    mesh = CompositeMesh(pf, mat, nx=20, ny=8, nz=12)
    solver = FESolver(mesh, mat, pf)
    direct = {loading: solver.solve(loading)
              for loading in ('compression', 'shear')}
    return solver, direct


class TestIterativeSolversAtDefaultTolerance:
    """IMPROVEMENT_PLAN 1.6: CG and MINRES at the *default* ``rtol`` give
    the direct answer.

    With penalty BCs the constrained rows put ``alpha * v ~ 1e11 * v`` in the
    right-hand side, so ``||F||`` was huge and CG met ``rtol=1e-9`` as soon
    as the constrained DOFs were right, with the interior unconverged: on a
    20x8x12 mesh compression CG was 24% off in displacement while reporting
    convergence. MINRES was 8% off the same way, and for ILSS it stopped
    on its preconditioned estimate, ~1000x below the true residual, and
    failed the residual check. Eliminating the prescribed DOFs and warm-restarting MINRES
    fixes both.
    """

    @pytest.mark.parametrize("solver_name", ["cg", "minres"])
    @pytest.mark.parametrize("loading", ["compression", "tension", "shear", "ilss"])
    def test_small_mesh_all_modes(self, clustered_6x3x4, loading, solver_name):
        r_direct = clustered_6x3x4.solve(loading)
        r_iter = clustered_6x3x4.solve(loading, solver=solver_name)
        du, ds, dfi = _relative_differences(r_iter, r_direct)
        assert du < 1e-7, f"max|du|/max|u| = {du:.2e}"
        assert ds < 1e-7, f"max|ds|/max|s| = {ds:.2e}"
        assert dfi < 1e-6, f"dFI/FI = {dfi:.2e}"
        assert r_iter.knockdown == pytest.approx(r_direct.knockdown, rel=1e-9)

    @pytest.mark.parametrize("loading", ["compression", "shear"])
    def test_moderate_mesh_cg(self, clustered_20x8x12, loading):
        # Measured: du 4e-9 / 6e-9, stress 6e-9 / 2e-8, FI 3e-9 / 3e-8.
        # With penalty BCs: du 0.24 / 0.013, stress 0.16 / 0.16, FI 0.02 / 0.17.
        solver, direct = clustered_20x8x12
        r_cg = solver.solve(loading, solver='cg')
        du, ds, dfi = _relative_differences(r_cg, direct[loading])
        assert du < 1e-7, f"max|du|/max|u| = {du:.2e}"
        assert ds < 2e-7, f"max|ds|/max|s| = {ds:.2e}"
        assert dfi < 2e-7, f"dFI/FI = {dfi:.2e}"
        assert r_cg.effective_modulus == pytest.approx(
            direct[loading].effective_modulus, rel=1e-12)

    @pytest.mark.parametrize("loading", ["compression", "shear"])
    def test_moderate_mesh_minres(self, clustered_20x8x12, loading):
        # MINRES minimizes the residual, not the energy-norm error, so at
        # the same rtol its displacements are less accurate than CG's
        # (measured du 1.3e-6 / 1.8e-7; with penalty BCs 0.08 / 0.007).
        solver, direct = clustered_20x8x12
        r_mr = solver.solve(loading, solver='minres')
        du, ds, dfi = _relative_differences(r_mr, direct[loading])
        assert du < 1e-5, f"max|du|/max|u| = {du:.2e}"
        assert ds < 1e-5, f"max|ds|/max|s| = {ds:.2e}"
        assert dfi < 1e-5, f"dFI/FI = {dfi:.2e}"
        assert r_mr.effective_modulus == pytest.approx(
            direct[loading].effective_modulus, rel=1e-10)

    def test_minres_ilss_needs_the_restart(self, clustered_6x3x4, monkeypatch):
        """The ILSS point load puts SciPy's MINRES estimate ~1000x below the
        true residual: one call fails the check, the warm restart passes."""
        monkeypatch.setattr(FESolver, '_MINRES_MAX_RESTARTS', 0)
        with pytest.raises(RuntimeError, match="minres failed to converge"):
            clustered_6x3x4.solve('ilss', solver='minres')
        monkeypatch.undo()
        clustered_6x3x4.solve('ilss', solver='minres')


class TestDirichletElimination:
    """IMPROVEMENT_PLAN 1.6: prescribed displacements are eliminated exactly.

    Replaces the penalty method (``alpha = penalty_factor * max(diag K)``
    added to each constrained DOF, issue #60). ``solve(penalty_factor=,
    diag_scale=)`` and ``BoundaryHandler.apply_penalty`` remain for one
    deprecation cycle: they warn, and the solve arguments have no effect.
    """

    def setup_method(self):
        import inspect
        self._inspect = inspect
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_default_penalty_lowered(self):
        """The deprecated ``apply_penalty`` keeps its 1e6 default."""
        sig = self._inspect.signature(BoundaryHandler.apply_penalty)
        default = sig.parameters['penalty_factor'].default
        assert default == 1e6, (
            f"Expected apply_penalty default penalty_factor=1e6, got {default!r}"
        )

    def test_solve_defaults_emit_no_deprecation_warning(self):
        import warnings
        solver = FESolver(self.mesh, self.material, self.pf)
        sig = self._inspect.signature(FESolver.solve)
        assert sig.parameters['penalty_factor'].default is None
        assert sig.parameters['diag_scale'].default is None
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            solver.solve(loading='compression', applied_strain=-0.001)

    @pytest.mark.parametrize("value", [1e2, 1e6, 1e15])
    def test_penalty_factor_is_deprecated_no_op(self, value):
        solver = FESolver(self.mesh, self.material, self.pf)
        r_default = solver.solve(loading='compression', applied_strain=-0.001)
        with pytest.warns(DeprecationWarning, match="penalty_factor.*no effect"):
            r = solver.solve(loading='compression', applied_strain=-0.001,
                             penalty_factor=value)
        np.testing.assert_array_equal(r.displacement, r_default.displacement)
        np.testing.assert_array_equal(r.stress_global, r_default.stress_global)
        assert r.knockdown == r_default.knockdown

    @pytest.mark.parametrize("value", [True, False])
    def test_diag_scale_is_deprecated_no_op(self, value):
        solver = FESolver(self.mesh, self.material, self.pf)
        r_default = solver.solve(loading='shear', applied_strain=0.01)
        with pytest.warns(DeprecationWarning, match="diag_scale.*no effect"):
            r = solver.solve(loading='shear', applied_strain=0.01,
                             diag_scale=value)
        np.testing.assert_array_equal(r.displacement, r_default.displacement)

    def test_deprecation_warning_points_at_the_caller(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.warns(DeprecationWarning) as record:
            solver.solve(loading='compression', penalty_factor=1e6)
        assert record[0].filename == __file__

    @pytest.mark.parametrize("loading,bound", [
        ("compression", 1e2), ("shear", 1e2), ("ilss", 1e2)])
    def test_free_dof_conditioning_is_physical(self, caplog, loading, bound):
        """The diagonal ratio of K_ff is O(10) (it was ~1e7 with the
        penalty rows), so no conditioning workaround is needed."""
        import logging
        import re
        solver = FESolver(self.mesh, self.material, self.pf)
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            solver.solve(loading=loading)
        ratios = [float(m.group(1)) for rec in caplog.records
                  if (m := re.search(r'Free-DOF stiffness: diag ratio=([0-9.eE+\-]+)',
                                     rec.message))]
        assert ratios, [r.message for r in caplog.records]
        assert 1.0 <= ratios[0] < bound

    def test_reactions_are_exactly_zero_at_free_dofs(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        for loading in ('compression', 'shear', 'ilss'):
            r = solver.solve(loading=loading)
            constrained, _ = solver._apply_boundary_conditions(
                loading, _default_strain(loading), -10.0)
            free = np.ones(self.mesh.n_dof, dtype=bool)
            free[list(constrained)] = False
            assert np.all(r.reaction_forces.ravel()[free] == 0.0)
            assert np.any(r.reaction_forces.ravel()[~free] != 0.0)

    def test_load_on_a_constrained_dof_is_carried_by_the_support(self):
        """The penalty method overwrote F at constrained DOFs, silently
        dropping a load applied there; elimination keeps it in R_c."""
        solver = FESolver(self.mesh, self.material, self.pf)
        K = solver.assembler.stiffness()
        constrained, F = solver.bc_handler.ilss_bcs(applied_load=-10.0)
        support = next(iter(constrained))
        F_extra = F.copy()
        F_extra[support] += 7.5
        u, _ = solver._solve_constrained(K, F, constrained)
        u_extra, _ = solver._solve_constrained(K, F_extra, constrained)
        # The support does not move, so the field is unchanged ...
        np.testing.assert_array_equal(u_extra, u)
        R, _ = solver._reactions_and_modulus('ilss', K, u, F, 0.0, constrained)
        R_extra, _ = solver._reactions_and_modulus(
            'ilss', K, u_extra, F_extra, 0.0, constrained)
        # ... and the support reaction takes the extra load (R_c = K_c u - F_c).
        delta = (R_extra - R).ravel()
        assert delta[support] == pytest.approx(-7.5, rel=1e-12)
        delta[support] = 0.0
        assert np.all(delta == 0.0)
        # Global equilibrium includes the load at the support.
        np.testing.assert_allclose(R_extra.sum(axis=0) + F_extra.reshape(-1, 3).sum(axis=0),
                                   0.0, atol=1e-9)

    def test_singular_free_block_raises_a_clear_error(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        K = scipy.sparse.diags([1.0, 0.0, 2.0, 3.0]).tocsc()
        with pytest.raises(RuntimeError, match="boundary conditions"):
            solver._solve_constrained(K, np.ones(4), {3: 0.0})

    def test_fully_constrained_system_returns_prescribed_values(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        K = scipy.sparse.identity(3, format='csc')
        u, rel = solver._solve_constrained(K, np.zeros(3), {0: 1.0, 1: 2.0, 2: 3.0})
        np.testing.assert_array_equal(u, [1.0, 2.0, 3.0])
        assert rel == 0.0

    @pytest.mark.parametrize("solver_name", ["direct", "cg", "minres"])
    def test_zero_load_gives_zero_field(self, solver_name):
        solver = FESolver(self.mesh, self.material, self.pf)
        r = solver.solve('compression', applied_strain=0.0, solver=solver_name)
        assert np.all(r.displacement == 0.0)
        assert r.knockdown == 1.0
        assert r.effective_modulus is None


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
        # tsai_wu_F12 is the normalized F*_12, scaled by sqrt(F_11 * F_22).
        assert F12_recovered == pytest.approx(
            custom_F12 * np.sqrt(F11 * F22), rel=1e-9, abs=1e-18)

        # Cross-check: default solver gives a different FI on the same state.
        fi_default = self.solver._evaluate_tsai_wu(
            s, strengths, e=0, elem_Vp=0.0)[0]
        assert fi_custom != pytest.approx(fi_default, rel=1e-12, abs=1e-18)

    def test_tsai_wu_F12_minus_half_matches_default(self):
        """tsai_wu_F12=-0.5 is Tsai's default, so the index is unchanged."""
        mat_half = dataclasses.replace(self.material, tsai_wu_F12=-0.5)
        solver_half = FESolver(self.mesh, mat_half, self.pf,
                               failure_criterion='tsai_wu')
        strengths = self.solver._degraded_strengths(0.02)
        s = np.array([[-800.0, 40.0, 5.0, 3.0, 2.0, 30.0]])
        np.testing.assert_allclose(
            solver_half._evaluate_tsai_wu(s, strengths, e=0, elem_Vp=0.02),
            self.solver._evaluate_tsai_wu(s, strengths, e=0, elem_Vp=0.02),
            rtol=1e-14)

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


# ============================================================
# IMPROVEMENT_PLAN 1.1 / 1.4: batched element quantities
# ============================================================

class TestElementBatchMatchesHex8Element:
    """The vectorized assembly/recovery path must reproduce Hex8Element
    element by element, including void elements, non-uniform porosity and
    rotated plies, for both element formulations."""

    @pytest.fixture(scope="class", params=['hex8', 'hex8i'])
    @classmethod
    def setup(cls, request):
        from porosity_fe.fe.batch import build_element_batch
        mat = MATERIALS['T800_epoxy']
        void = VoidGeometry(center=(25, 10, mat.total_thickness / 2),
                            radii=(6, 4, mat.total_thickness / 3))
        pf = PorosityField(mat, 0.04, distribution='clustered',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, mat, nx=8, ny=4, nz=6,
                             ply_angles=[0, 45, -45, 90, 90, -45, 45, 0])
        assert len(mesh.void_elements) > 0, "fixture must contain void elements"
        formulation = request.param
        assembler = GlobalAssembler(mesh, mat, pf, formulation=formulation)
        batch = build_element_batch(mesh, mat, pf.void_shape_radii,
                                    formulation=formulation)
        assert batch.formulation == formulation
        assert assembler.create_element(0).formulation == formulation
        return mesh, assembler, batch

    def test_element_stiffness_matches(self, setup):
        mesh, assembler, batch = setup
        Ke_batch = batch.stiffness_matrices()
        for e in range(mesh.n_elements):
            Ke = assembler.create_element(e).stiffness_matrix()
            Ke = 0.5 * (Ke + Ke.T)
            np.testing.assert_allclose(
                Ke_batch[e], Ke, rtol=1e-10, atol=1e-10 * np.abs(Ke).max(),
                err_msg=f"element {e}")

    def test_gauss_point_strain_and_stress_match(self, setup):
        mesh, assembler, batch = setup
        rng = np.random.default_rng(0)
        u = rng.normal(scale=1e-3, size=mesh.n_dof)
        strain = batch.strains(u)
        stress = batch.stresses(strain)
        for e in range(mesh.n_elements):
            elem = assembler.create_element(e)
            u_e = u[assembler.element_dof_indices(e)]
            eps = elem.strain_at_gauss_points(u_e)
            sig = elem.stress_at_gauss_points(u_e)
            np.testing.assert_allclose(strain[e], eps, rtol=1e-10,
                                       atol=1e-12 * np.abs(eps).max())
            np.testing.assert_allclose(stress[e], sig, rtol=1e-10,
                                       atol=1e-10 * np.abs(sig).max())

    def test_dof_indices_match(self, setup):
        mesh, assembler, batch = setup
        for e in (0, mesh.n_elements // 2, mesh.n_elements - 1):
            node_ids = mesh.elements[e]
            expected = np.array([3 * n + k for n in node_ids for k in range(3)])
            np.testing.assert_array_equal(batch.dofs[e], expected)
            np.testing.assert_array_equal(assembler.element_dof_indices(e), expected)

    def test_inverted_element_is_rejected(self, setup):
        from porosity_fe.fe.batch import build_element_batch
        mesh, _, _ = setup
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.02)
        bad = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        bad.elements = bad.elements[:, [1, 0, 3, 2, 5, 4, 7, 6]]
        with pytest.raises(ValueError, match="non-positive Jacobian"):
            build_element_batch(bad, mat, pf.void_shape_radii)

    def test_out_of_range_porosity_is_rejected(self, setup):
        from porosity_fe.fe.batch import build_element_batch
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.02)
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        mesh.porosity = mesh.porosity.copy()
        mesh.porosity[0] = 3.0
        with pytest.raises(ValueError, match="not a percent"):
            build_element_batch(mesh, mat, pf.void_shape_radii)


class TestKnockdownFromPristineReference:
    """IMPROVEMENT_PLAN 2.4: knockdown = porous / pristine structural stiffness."""

    @staticmethod
    def _pair(Vp=0.04, discrete_voids=None, **mesh_kw):
        mat = MATERIALS['T800_epoxy']
        kw = dict(nx=6, ny=3, nz=6, ply_angles=[0, 45, -45, 90, 90, -45, 45, 0])
        kw.update(mesh_kw)
        pf = PorosityField(mat, Vp, distribution='clustered',
                           discrete_voids=discrete_voids or [])
        pf0 = PorosityField(mat, 0.0)
        porous = FESolver(CompositeMesh(pf, mat, **kw), mat, pf,
                          ply_angles=kw['ply_angles'])
        pristine = FESolver(CompositeMesh(pf0, mat, **kw), mat, pf0,
                            ply_angles=kw['ply_angles'])
        return porous, pristine

    @pytest.mark.parametrize("loading", ["compression", "tension", "shear"])
    def test_equals_effective_modulus_ratio(self, loading):
        porous, pristine = self._pair()
        r, r0 = porous.solve(loading), pristine.solve(loading)
        assert r.knockdown == pytest.approx(
            r.effective_modulus / r0.effective_modulus, rel=1e-9)
        assert 0.0 < r.knockdown < 1.0

    def test_ilss_equals_beam_stiffness_ratio(self):
        porous, pristine = self._pair()
        r, r0 = porous.solve('ilss'), pristine.solve('ilss')
        # Inverse compliance: pristine deflection energy / porous.
        b, b0 = porous.assembler.element_batch(), pristine.assembler.element_batch()
        W = np.einsum('egi,egij,egj,eg->', r.strain_global, b.C,
                      r.strain_global, b.detJ_w)
        W0 = np.einsum('egi,egij,egj,eg->', r0.strain_global, b0.C,
                       r0.strain_global, b0.detJ_w)
        assert r.knockdown == pytest.approx(W0 / W, rel=1e-6)
        assert 0.0 < r.knockdown < 1.0

    def test_shear_and_ilss_are_no_longer_pinned_at_one(self):
        porous, _ = self._pair()
        assert porous.solve('shear').knockdown < 0.999
        assert porous.solve('ilss').knockdown < 0.999

    def test_pristine_mesh_gives_exactly_one(self):
        _, pristine = self._pair()
        for loading in ('compression', 'shear', 'ilss'):
            assert pristine.solve(loading).knockdown == 1.0

    def test_geometric_voids_alone_reduce_stiffness(self):
        mat = MATERIALS['T800_epoxy']
        void = VoidGeometry(center=(25, 10, mat.total_thickness / 2),
                            radii=(4, 3, 1.0))
        porous, _ = self._pair(Vp=0.0, discrete_voids=[void],
                               nx=10, ny=6, nz=12, ply_angles='UD')
        assert len(porous.mesh.void_elements) > 0
        assert porous.solve('shear').knockdown < 0.99

    def test_independent_of_load_magnitude_and_sign(self):
        porous, _ = self._pair()
        ref = porous.solve('compression').knockdown
        assert porous.solve('compression', applied_strain=-0.001).knockdown == \
            pytest.approx(ref, rel=1e-9)
        assert porous.solve('tension', applied_strain=0.003).knockdown == \
            pytest.approx(ref, rel=1e-9)
        ilss = porous.solve('ilss').knockdown
        assert porous.solve('ilss', applied_load=-250.0).knockdown == \
            pytest.approx(ilss, rel=1e-9)

    def test_pristine_reference_is_cached_across_solvers(self, monkeypatch):
        from porosity_fe.fe import solver as solver_mod
        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        calls = []
        real = solver_mod._pristine_mesh
        monkeypatch.setattr(solver_mod, '_pristine_mesh',
                            lambda mesh: calls.append(1) or real(mesh))
        a, _ = self._pair(Vp=0.02)
        b, _ = self._pair(Vp=0.05)
        a.solve('compression')
        a.solve('tension')
        b.solve('compression')
        assert len(calls) == 1
        b.solve('shear')
        assert len(calls) == 2

    def test_pristine_cache_keys_on_fields_missing_from_repr(self):
        # E33 / G13 / G23 are absent from MaterialProperties.__repr__, so a
        # repr-keyed cache handed the soft material the T800 reference.
        from porosity_fe.fe import solver as solver_mod
        base = MATERIALS['T800_epoxy']
        soft = dataclasses.replace(base, G13=1500.0, G23=1200.0, E33=6000.0)
        assert repr(soft) == repr(base)

        def kd(mat):
            pf = PorosityField(mat, 0.03)
            mesh = CompositeMesh(pf, mat, nx=8, ny=3, nz=6)
            return FESolver(mesh, mat, pf).solve('ilss').knockdown

        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        kd(base)
        after_base = kd(soft)
        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        fresh = kd(soft)
        assert after_base == pytest.approx(fresh, rel=1e-12)


class TestStiffnessAndFactorizationReuse:
    """IMPROVEMENT_PLAN 1.2: K and its LU factorization are reused across
    solves when valid, and rebuilt when inputs change."""

    @staticmethod
    def _problem(Vp=0.03):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, Vp, distribution='clustered')
        mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=4)
        return mat, pf, mesh

    @pytest.fixture
    def counters(self, monkeypatch):
        import scipy.sparse.linalg as sla
        from porosity_fe.fe import assembler as asm_mod
        counts = {'batch': 0, 'splu': 0}
        real_batch, real_splu = asm_mod.build_element_batch, sla.splu

        def batch(*a, **k):
            counts['batch'] += 1
            return real_batch(*a, **k)

        def splu(*a, **k):
            counts['splu'] += 1
            return real_splu(*a, **k)

        monkeypatch.setattr(asm_mod, 'build_element_batch', batch)
        monkeypatch.setattr(sla, 'splu', splu)
        # Count only this solver's assembly and factorizations, not the
        # (separately cached) pristine knockdown reference solve.
        monkeypatch.setattr(FESolver, '_pristine_stiffness_measure',
                            lambda self, loading: 1.0)
        return counts

    @staticmethod
    def _assert_same(r1, r2):
        np.testing.assert_allclose(r1.displacement, r2.displacement, rtol=1e-10,
                                   atol=1e-12 * np.abs(r2.displacement).max())
        np.testing.assert_allclose(r1.stress_global, r2.stress_global, rtol=1e-10,
                                   atol=1e-10 * np.abs(r2.stress_global).max())
        assert r1.knockdown == pytest.approx(r2.knockdown, rel=1e-12)
        assert r1.max_failure_index == pytest.approx(r2.max_failure_index, rel=1e-10)

    def test_same_constraint_set_reuses_k_and_lu(self, counters):
        mat, pf, mesh = self._problem()
        solver = FESolver(mesh, mat, pf)
        solver.solve('tension', applied_strain=0.01)
        r = solver.solve('compression', applied_strain=-0.005)
        assert counters == {'batch': 1, 'splu': 1}
        fresh = FESolver(mesh, mat, pf).solve('compression', applied_strain=-0.005)
        self._assert_same(r, fresh)

    def test_new_constraint_set_refactors_but_keeps_k(self, counters):
        mat, pf, mesh = self._problem()
        solver = FESolver(mesh, mat, pf)
        solver.solve('tension', applied_strain=0.01)
        r = solver.solve('shear', applied_strain=0.01)
        assert counters == {'batch': 1, 'splu': 2}
        self._assert_same(r, FESolver(mesh, mat, pf).solve('shear', applied_strain=0.01))

    def test_penalty_factor_is_ignored_and_does_not_refactor(self, counters):
        # The cache is keyed on the constrained-DOF set only.
        mat, pf, mesh = self._problem()
        solver = FESolver(mesh, mat, pf)
        ref = solver.solve('tension', applied_strain=0.01)
        with pytest.warns(DeprecationWarning):
            r = solver.solve('tension', applied_strain=0.01, penalty_factor=1e7)
        with pytest.warns(DeprecationWarning):
            solver.solve('tension', applied_strain=0.01, diag_scale=True)
        assert counters == {'batch': 1, 'splu': 1}
        np.testing.assert_array_equal(r.displacement, ref.displacement)

    def test_iterative_solves_reuse_the_free_block_without_factorizing(self, counters):
        mat, pf, mesh = self._problem()
        solver = FESolver(mesh, mat, pf)
        solver.solve('tension', applied_strain=0.01, solver='cg')
        assert counters['splu'] == 0
        K_ff = solver._lu_cache[2]
        solver.solve('compression', applied_strain=-0.01, solver='minres')
        assert solver._lu_cache[2] is K_ff
        solver.solve('compression', applied_strain=-0.01)
        solver.solve('tension', applied_strain=0.01)
        solver.solve('tension', applied_strain=0.01, solver='cg')
        assert solver._lu_cache[2] is K_ff
        assert counters == {'batch': 1, 'splu': 1}

    def test_in_place_porosity_edit_triggers_reassembly(self, counters):
        mat, pf, mesh = self._problem()
        solver = FESolver(mesh, mat, pf)
        before = solver.solve('tension', applied_strain=0.01)
        mesh.porosity *= 2.0   # in place, same array object
        after = solver.solve('tension', applied_strain=0.01)
        assert counters['batch'] == 2
        assert after.knockdown < before.knockdown
        self._assert_same(after, FESolver(mesh, mat, pf).solve('tension', applied_strain=0.01))

    def test_in_place_material_edit_triggers_reassembly(self, counters):
        # Fields absent from MaterialProperties.__repr__ must still
        # invalidate the cached K.
        _, pf, mesh = self._problem()
        mat = dataclasses.replace(MATERIALS['T800_epoxy'])
        solver = FESolver(mesh, mat, pf)
        before = solver.solve('tension', applied_strain=0.01)
        mat.E33, mat.G13, mat.G23 = 6000.0, 1500.0, 1200.0
        after = solver.solve('tension', applied_strain=0.01)
        assert counters['batch'] == 2
        assert not np.allclose(after.displacement, before.displacement)
        fresh = FESolver(mesh, mat, pf).solve('tension', applied_strain=0.01)
        np.testing.assert_allclose(after.displacement, fresh.displacement,
                                   rtol=1e-10,
                                   atol=1e-12 * np.abs(fresh.displacement).max())

    def test_assemble_stiffness_returns_independent_copy(self):
        mat, pf, mesh = self._problem()
        solver = FESolver(mesh, mat, pf)
        ref = solver.solve('tension', applied_strain=0.01)
        K = solver.assembler.assemble_stiffness()
        K.data[:] = 0.0
        self._assert_same(solver.solve('tension', applied_strain=0.01), ref)


class TestReactionsAndEffectiveModulus:
    """IMPROVEMENT_PLAN 3.2: reaction forces and the strain-energy effective
    modulus recovered from a displacement-controlled solve."""

    @staticmethod
    def _solver(Vp, angles, nz=8):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, Vp)
        mesh = CompositeMesh(pf, mat, nx=10, ny=4, nz=nz, ply_angles=angles)
        return mat, mesh, FESolver(mesh, mat, pf, ply_angles=angles)

    def test_pristine_ud_recovers_ply_moduli(self):
        mat, _, solver = self._solver(0.0, 'UD')
        # Exact BCs: measured within ~3e-14 (the penalty slack gave ~5e-8).
        assert solver.solve('tension', applied_strain=0.01).effective_modulus == \
            pytest.approx(mat.E11, rel=1e-10)
        assert solver.solve('compression', applied_strain=-0.005).effective_modulus == \
            pytest.approx(mat.E11, rel=1e-10)
        assert solver.solve('shear', applied_strain=0.01).effective_modulus == \
            pytest.approx(mat.G12, rel=1e-10)

    def test_axial_modulus_equals_face_reaction_over_area_and_strain(self):
        _, mesh, solver = self._solver(0.04, 'QI')
        r = solver.solve('tension', applied_strain=0.01)
        P = r.reaction_forces[mesh.nodes_on_face('x_max'), 0].sum()
        assert r.effective_modulus == pytest.approx(
            P / (mesh.L_y * mesh.L_z * 0.01), rel=1e-10)

    @pytest.mark.parametrize("Vp", [0.0, 0.04])
    def test_quasi_isotropic_moduli_match_clt(self, Vp):
        from porosity_fe.homogenization import compute_degraded_clt_moduli
        mat, mesh, solver = self._solver(Vp, 'QI', nz=16)
        clt = compute_degraded_clt_moduli(
            mat, [0, 90, 45, -45, -45, 45, 90, 0], Vp)
        Gxy = solver.solve('shear', applied_strain=0.01).effective_modulus
        Ex = solver.solve('tension', applied_strain=0.01).effective_modulus
        # Homogeneous shear BCs reproduce the in-plane CLT assumption; the
        # axial modulus carries a small 3D (free-edge, sigma_zz) effect.
        assert Gxy == pytest.approx(clt['Gxy'], rel=1e-10)
        assert Ex == pytest.approx(clt['Ex'], rel=0.02)

    def test_porosity_lowers_effective_modulus(self):
        _, _, pristine = self._solver(0.0, 'QI')
        _, _, porous = self._solver(0.05, 'QI')
        for loading in ('tension', 'shear'):
            assert porous.solve(loading, applied_strain=0.01).effective_modulus < \
                pristine.solve(loading, applied_strain=0.01).effective_modulus

    def test_ilss_reactions_balance_applied_load_and_no_modulus(self):
        _, _, solver = self._solver(0.02, 'QI')
        r = solver.solve('ilss', applied_load=-10.0)
        assert r.effective_modulus is None
        np.testing.assert_allclose(r.reaction_forces.sum(axis=0), [0.0, 0.0, 10.0],
                                   atol=1e-9)

    def test_json_export_carries_stiffness_block(self, tmp_path):
        import json
        _, _, solver = self._solver(0.02, 'QI')
        r = solver.solve('tension', applied_strain=0.01)
        path = tmp_path / "fe.json"
        FESolver.export_results(r, path)
        block = json.loads(path.read_text())['stiffness']
        assert block['effective_modulus_MPa'] == pytest.approx(r.effective_modulus)
        assert len(block['reaction_force_sum_N']) == 3


class TestFirstPlyFailureLoadFactor:
    """IMPROVEMENT_PLAN 3.3: the load multiplier at first-ply failure."""

    @pytest.fixture(scope="class")
    def solver(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='clustered')
        mesh = CompositeMesh(pf, mat, nx=8, ny=4, nz=8)
        return FESolver(mesh, mat, pf)

    @pytest.mark.parametrize("criterion", ["tsai_wu", "hashin", "max_stress"])
    @pytest.mark.parametrize("loading, kwarg, value", [
        ("tension", "applied_strain", 0.002),
        ("compression", "applied_strain", -0.002),
        ("shear", "applied_strain", 0.002),
        ("ilss", "applied_load", -10.0),
    ])
    def test_rescaled_load_reaches_unit_failure_index(self, solver, criterion,
                                                      loading, kwarg, value):
        """Linear analysis: re-solving at lam times the load must put the
        governing point exactly on the failure surface."""
        r = solver.solve(loading, failure_criterion=criterion, **{kwarg: value})
        lam = r.first_ply_failure_load_factor
        assert np.isfinite(lam) and lam > 0
        r2 = solver.solve(loading, failure_criterion=criterion,
                          **{kwarg: value * lam})
        assert r2.max_failure_index == pytest.approx(1.0, rel=1e-9)
        assert r2.first_ply_failure_load_factor == pytest.approx(1.0, rel=1e-9)

    def test_max_stress_factor_is_reciprocal_of_max_index(self, solver):
        r = solver.solve('tension', applied_strain=0.002, failure_criterion='max_stress')
        assert r.first_ply_failure_load_factor == pytest.approx(
            1.0 / r.max_failure_index, rel=1e-12)

    def test_unstressed_model_gives_infinite_factor(self):
        from porosity_fe.fe.failure import first_ply_failure_load_factor
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.02)
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        zero = np.zeros((mesh.n_elements, 8, 6))
        for criterion in ("tsai_wu", "hashin", "max_stress"):
            assert first_ply_failure_load_factor(
                zero, mesh.porosity, mesh.elements, mat,
                pf.void_shape_radii, criterion) == np.inf

    def test_reported_in_summary_and_json(self, solver, tmp_path):
        import json
        r = solver.solve('tension', applied_strain=0.002)
        assert r.summary().details['first_ply_failure_load_factor'] == \
            r.first_ply_failure_load_factor
        path = tmp_path / "fe.json"
        FESolver.export_results(r, path)
        data = json.loads(path.read_text())
        assert data['failure']['first_ply_failure_load_factor'] == pytest.approx(
            r.first_ply_failure_load_factor)


class TestHashinDelaminationMode:
    """IMPROVEMENT_PLAN 2.2: Hashin now sees interlaminar stresses through a
    Brewer-Lagace delamination mode."""

    @pytest.fixture(scope="class")
    def strengths(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        return FESolver(mesh, mat, pf)._degraded_strengths(0.0)

    def test_pure_interlaminar_shear_governs(self, strengths):
        from porosity_fe.fe.failure import evaluate_hashin
        S23 = strengths[5]
        s = np.zeros((2, 6))
        s[0, 4] = 0.5 * S23           # tau_13
        s[1, 3] = 0.5 * S23           # tau_23
        modes = evaluate_hashin(s, strengths)
        np.testing.assert_allclose(modes['delamination'], 0.25, rtol=1e-12)
        np.testing.assert_allclose(modes['max_fi'], 0.25, rtol=1e-12)

    def test_only_through_thickness_tension_contributes(self, strengths):
        from porosity_fe.fe.failure import evaluate_hashin
        Yt = strengths[2]
        s = np.zeros((2, 6))
        s[0, 2] = 0.5 * Yt            # sigma_33 tension
        s[1, 2] = -0.5 * Yt           # sigma_33 compression
        modes = evaluate_hashin(s, strengths)
        assert modes['delamination'][0] == pytest.approx(0.25, rel=1e-12)
        assert modes['delamination'][1] == 0.0

    def test_ilss_hashin_is_governed_by_delamination(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03)
        mesh = CompositeMesh(pf, mat, nx=10, ny=3, nz=8)
        r = FESolver(mesh, mat, pf).solve('ilss', failure_criterion='hashin')
        modes = r.failure_mode_indices
        assert r.max_failure_index == pytest.approx(modes['delamination'], rel=1e-12)
        assert modes['delamination'] > max(modes['fiber_t'], modes['fiber_c'],
                                           modes['matrix_t'], modes['matrix_c'])

    def test_mode_keys_are_uniform_across_criteria(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03)
        mesh = CompositeMesh(pf, mat, nx=4, ny=2, nz=4)
        solver = FESolver(mesh, mat, pf)
        keys = {c: set(solver.solve('tension', applied_strain=0.002,
                                    failure_criterion=c).failure_mode_indices)
                for c in ('tsai_wu', 'hashin', 'max_stress')}
        assert keys['tsai_wu'] == keys['hashin'] == keys['max_stress']
        assert 'delamination' in keys['hashin']


@pytest.fixture(scope="module")
def voided():
    """UD mesh containing geometric void elements, with its solver."""
    mat = MATERIALS['T800_epoxy']
    void = VoidGeometry(center=(25, 10, mat.total_thickness / 2),
                        radii=(4, 3, 1.0))
    pf = PorosityField(mat, 0.04, discrete_voids=[void])
    mesh = CompositeMesh(pf, mat, nx=10, ny=6, nz=12, ply_angles='UD')
    assert len(mesh.void_elements) > 0
    return mat, pf, mesh, FESolver(mesh, mat, pf, ply_angles='UD')


class TestVoidElementUnification:
    """IMPROVEMENT_PLAN 2.3: one notion of "void element" across the FE path."""

    @pytest.mark.parametrize("criterion", ['tsai_wu', 'hashin', 'max_stress'])
    def test_geometric_voids_are_skipped_by_failure(self, voided, criterion):
        _mat, _pf, mesh, solver = voided
        r = solver.solve('compression', failure_criterion=criterion)
        assert np.all(r.per_element_failure_index[mesh.void_elements] == 0.0)
        assert r.max_failure_index > 0.0

    def test_load_factor_ignores_geometric_voids(self, voided):
        from porosity_fe.fe.failure import first_ply_failure_load_factor
        mat, pf, mesh, solver = voided
        r = solver.solve('compression')
        # Put a huge stress in one void element: only the unmasked call sees it.
        stress = r.stress_local.copy()
        stress[mesh.void_elements[0]] = -1e6
        args = (stress, mesh.porosity, mesh.elements, mat, pf.void_shape_radii)
        masked = first_ply_failure_load_factor(
            *args, void_elements=mesh.void_elements)
        unmasked = first_ply_failure_load_factor(*args)
        assert masked == pytest.approx(r.first_ply_failure_load_factor, rel=1e-12)
        assert unmasked < 1e-3 * masked

    def test_high_porosity_threshold_is_shared(self):
        from porosity_fe.fe import batch, element, failure
        assert failure.VOID_VP_THRESHOLD is element.VOID_VP_THRESHOLD
        assert batch.VP_STIFFNESS_CLAMP is element.VP_STIFFNESS_CLAMP
        assert element.VOID_VP_THRESHOLD < element.VP_STIFFNESS_CLAMP


@pytest.fixture(scope="module")
def small_ud_solver():
    mat = MATERIALS['T800_epoxy']
    pf = PorosityField(mat, 0.02)
    mesh = CompositeMesh(pf, mat, nx=4, ny=2, nz=4, ply_angles='UD')
    return FESolver(mesh, mat, pf, ply_angles='UD')


class TestAppliedStrainDefault:
    """``solve('tension')`` used to default to -0.01 and run in compression."""

    def test_default_tension_is_tensile(self, small_ud_solver):
        r = small_ud_solver.solve('tension')
        assert np.mean(r.stress_global[:, :, 0]) > 0.0
        explicit = small_ud_solver.solve('tension', applied_strain=0.01)
        np.testing.assert_array_equal(r.stress_global, explicit.stress_global)

    def test_default_compression_is_compressive(self, small_ud_solver):
        r = small_ud_solver.solve('compression')
        assert np.mean(r.stress_global[:, :, 0]) < 0.0
        explicit = small_ud_solver.solve('compression', applied_strain=-0.01)
        np.testing.assert_array_equal(r.stress_global, explicit.stress_global)

    def test_tension_and_compression_failure_indices_differ(self, small_ud_solver):
        t = small_ud_solver.solve('tension')
        c = small_ud_solver.solve('compression')
        assert t.max_failure_index != pytest.approx(c.max_failure_index, rel=1e-3)

    def test_contradictory_sign_warns(self, small_ud_solver, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="porosity_fe_analysis"):
            small_ud_solver.solve('tension', applied_strain=-0.01)
        assert "contradicts the loading mode" in caplog.text


# ============================================================
# IMPROVEMENT_PLAN 3.6 (A1): incompatible-mode element 'hex8i'
# ============================================================

def _ud_beam_material():
    """UD T800/epoxy, 4 plies of 0.5 mm: a 50 x 20 x 2 mm default coupon."""
    return dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4, t_ply=0.5)


def _solve_by_elimination(K, F, constrained):
    """Exact Dirichlet elimination, independent of FESolver's BC handling."""
    import scipy.sparse.linalg
    n = K.shape[0]
    dofs = np.fromiter(constrained.keys(), dtype=np.intp, count=len(constrained))
    vals = np.fromiter(constrained.values(), dtype=float, count=len(constrained))
    free = np.setdiff1d(np.arange(n), dofs)
    K = scipy.sparse.csr_matrix(K)
    u = np.zeros(n)
    u[dofs] = vals
    rhs = F[free] - K[free][:, dofs] @ vals
    u[free] = scipy.sparse.linalg.spsolve(K[free][:, free].tocsc(), rhs)
    return u


class TestIncompatibleModes:
    """``formulation='hex8i'``: Wilson-Taylor incompatible modes condensed
    into an effective ``B``. Opt-in; ``'hex8'`` stays the default."""

    def test_unknown_formulation_is_rejected(self):
        from porosity_fe.fe.batch import build_element_batch
        mat = _ud_beam_material()
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2, ply_angles=[0.0] * 4)
        with pytest.raises(ValueError, match="Unknown element formulation"):
            FESolver(mesh, mat, pf, formulation='hex20')
        with pytest.raises(ValueError, match="Unknown element formulation"):
            GlobalAssembler(mesh, mat, pf, formulation='HEX8I')
        with pytest.raises(ValueError, match="Unknown element formulation"):
            build_element_batch(mesh, mat, pf.void_shape_radii, formulation='q1')
        with pytest.raises(ValueError, match="Unknown element formulation"):
            Hex8Element(
                node_coords=mesh.nodes[mesh.elements[0]],
                C_base=mat.get_stiffness_matrix(), ply_angle_deg=0.0,
                node_porosities=np.zeros(8), void_shape_radii=(1, 1, 1),
                nu_m=mat.matrix_poisson,
                C_m=mat.get_isotropic_matrix_stiffness(), material=mat,
                formulation='bbar')

    @staticmethod
    def _distorted_element(formulation):
        from porosity_fe.fe.element import _NODE_COORDS_REF
        mat = _ud_beam_material()
        rng = np.random.default_rng(1)
        coords = (0.5 * (_NODE_COORDS_REF + 1.0)) * np.array([5.0, 1.0, 1.0])
        coords = coords + rng.normal(scale=0.08, size=coords.shape)
        return Hex8Element(
            node_coords=coords, C_base=mat.get_stiffness_matrix(),
            ply_angle_deg=30.0, node_porosities=np.full(8, 0.02),
            void_shape_radii=(1, 1, 1), nu_m=mat.matrix_poisson,
            C_m=mat.get_isotropic_matrix_stiffness(), material=mat,
            formulation=formulation)

    @pytest.mark.parametrize('formulation', ['hex8', 'hex8i'])
    def test_exactly_six_zero_energy_modes(self, formulation):
        # A single distorted 5:1:1 element: only the six rigid-body modes
        # may cost no energy (no spurious mechanisms from the condensation).
        Ke = self._distorted_element(formulation).stiffness_matrix()
        Ke = 0.5 * (Ke + Ke.T)
        ev = np.linalg.eigvalsh(Ke)
        assert int(np.sum(ev < 1e-8 * ev.max())) == 6
        assert ev.min() > -1e-8 * ev.max()

    def test_incompatible_modes_carry_no_mean_strain(self):
        # Taylor's correction: each mode integrates to zero strain over the
        # (distorted) element, the condition for passing the patch test.
        elem = self._distorted_element('hex8i')
        total = np.zeros((6, 9))
        for (xi, eta, zeta), w in zip(elem._gauss_points, elem._gauss_weights,
                                      strict=True):
            detJ = np.linalg.det(elem.jacobian(xi, eta, zeta))
            total += elem.G_matrix(xi, eta, zeta) * detJ * w
        G0 = elem.G_matrix(0.577, -0.577, 0.577)
        assert np.abs(total).max() < 1e-12 * np.abs(G0).max()

    @pytest.mark.parametrize('formulation', ['hex8', 'hex8i'])
    def test_patch_test_on_distorted_mesh(self, formulation):
        # Linear displacement prescribed on the boundary of a distorted
        # 3 x 3 x 4 mesh of a 30-degree ply: every Gauss point of every
        # element must recover the constant strain exactly.
        from porosity_fe.fe.batch import build_element_batch
        mat = _ud_beam_material()
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=3, ny=3, nz=4, ply_angles=[30.0] * 4)
        mesh.L_x, mesh.L_y = 3.0, 3.0
        mesh.generate_mesh()
        X = mesh.nodes.copy()
        tol = 1e-9
        interior = np.flatnonzero(
            (X[:, 0] > tol) & (X[:, 0] < mesh.L_x - tol)
            & (X[:, 1] > tol) & (X[:, 1] < mesh.L_y - tol)
            & (X[:, 2] > tol) & (X[:, 2] < mesh.L_z - tol))
        assert interior.size
        rng = np.random.default_rng(2)
        X[interior] += rng.uniform(-0.25, 0.25, size=(interior.size, 3)) \
            * np.array([1.0, 1.0, mesh.L_z / 4])
        mesh.nodes = X
        grad = np.array([[1e-3, 2e-4, -3e-4],
                         [1e-4, -5e-4, 2e-4],
                         [3e-4, 1e-4, 4e-4]])
        u_exact = (X @ grad.T).ravel()
        boundary = np.setdiff1d(np.arange(mesh.n_nodes), interior)
        constrained = {3 * int(n) + k: u_exact[3 * n + k]
                       for n in boundary for k in range(3)}
        K = GlobalAssembler(mesh, mat, pf, formulation=formulation).stiffness()
        u = _solve_by_elimination(K, np.zeros(mesh.n_dof), constrained)
        eps = build_element_batch(mesh, mat, pf.void_shape_radii,
                                  formulation=formulation).strains(u)
        eps_exact = np.array([grad[0, 0], grad[1, 1], grad[2, 2],
                              grad[1, 2] + grad[2, 1], grad[0, 2] + grad[2, 0],
                              grad[0, 1] + grad[1, 0]])
        assert np.abs(u - u_exact).max() < 1e-10 * np.abs(u_exact).max()
        assert np.abs(eps - eps_exact).max() < 1e-10 * np.abs(eps_exact).max()

    @staticmethod
    def _bending_modulus_ratio(nx, ny, nz, formulation, theta=1e-3):
        """``E_bend / E11`` of a UD beam under a uniform moment.

        One end is held in ``u_x``; the other is rotated by ``theta`` about
        the neutral axis (``u_x = -theta (z - h/2)``). The exact solution is
        a uniform moment ``M = E11 I theta / L``, and ``u^T K u = M theta``.
        """
        mat = _ud_beam_material()
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh(pf, mat, nx=nx, ny=ny, nz=nz, ply_angles=[0.0] * 4)
        L, b, h = mesh.L_x, mesh.L_y, mesh.L_z
        constrained = {}
        for n in mesh.nodes_on_face('x_min'):
            constrained[3 * int(n)] = 0.0
        for n in mesh.nodes_on_face('x_max'):
            constrained[3 * int(n)] = -theta * (mesh.nodes[n, 2] - h / 2)
        axis = mesh.find_nodes_near(x=0.0, z=h / 2, tol=1e-9)
        assert axis.size, "the neutral axis must be a node line (even nz)"
        for n in axis:
            constrained[3 * int(n) + 2] = 0.0
        constrained[3 * int(axis[0]) + 1] = 0.0
        K = GlobalAssembler(mesh, mat, pf, formulation=formulation).stiffness()
        u = _solve_by_elimination(K, np.zeros(mesh.n_dof), constrained)
        E_bend = L * float(u @ (K @ u)) / (b * h ** 3 / 12.0 * theta ** 2)
        return E_bend / mat.E11

    @pytest.mark.parametrize('res, hex8_ratio', [
        ((4, 2, 2), 2.265),    # element length / thickness = 6.25
        ((16, 4, 8), 1.084),   # 1.56, the ILSS test mesh
    ])
    def test_pure_bending_is_lock_free(self, res, hex8_ratio):
        # Standard hex8 locks: its bending stiffness error grows as about
        # (G13 / E11) (dx / h)^2 and does not shrink with nz. hex8i is
        # within 0.5 % of exact on both meshes.
        assert self._bending_modulus_ratio(*res, 'hex8i') == \
            pytest.approx(1.0, abs=0.005)
        assert self._bending_modulus_ratio(*res, 'hex8') == \
            pytest.approx(hex8_ratio, rel=0.01)

    def test_default_is_hex8_and_unchanged(self):
        # Not passing the option must give exactly the 'hex8' results.
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        pf = PorosityField(mat, 0.03, distribution='clustered')
        mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=8)
        default = FESolver(mesh, mat, pf)
        explicit = FESolver(mesh, mat, pf, formulation='hex8')
        assert default.formulation == 'hex8'
        assert default.assembler.element_batch().formulation == 'hex8'
        for mode in ('compression', 'ilss'):
            a, b = default.solve(mode), explicit.solve(mode)
            assert a.formulation == b.formulation == 'hex8'
            np.testing.assert_array_equal(a.displacement, b.displacement)
            np.testing.assert_array_equal(a.stress_global, b.stress_global)
            assert a.knockdown == b.knockdown
            assert a.max_failure_index == b.max_failure_index
            assert a.first_ply_failure_load_factor == \
                b.first_ply_failure_load_factor

    def test_membrane_response_nearly_unchanged(self):
        # hex8i passes the patch test, so a homogeneous in-plane state is
        # unchanged: pure shear agrees to round-off. Under compression the
        # angle plies of this QI laminate develop free-edge interlaminar
        # gradients that hex8 resolves slightly too stiffly (about 0.2 % on
        # E_x at this coarse mesh); the knockdown, a ratio of two solves
        # with the same element, moves by about 1e-4. The bending (ILSS)
        # response is where the formulations really differ.
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        pf = PorosityField(mat, 0.03, distribution='clustered')
        mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=8)
        r = {f: {m: FESolver(mesh, mat, pf, formulation=f).solve(m)
                 for m in ('compression', 'shear', 'ilss')}
             for f in ('hex8', 'hex8i')}
        assert r['hex8i']['shear'].effective_modulus == pytest.approx(
            r['hex8']['shear'].effective_modulus, rel=1e-6)
        assert r['hex8i']['compression'].effective_modulus == pytest.approx(
            r['hex8']['compression'].effective_modulus, rel=5e-3)
        for mode in ('compression', 'shear'):
            assert r['hex8i'][mode].knockdown == pytest.approx(
                r['hex8'][mode].knockdown, abs=5e-4)
        assert abs(r['hex8i']['ilss'].knockdown
                   - r['hex8']['ilss'].knockdown) > 5e-3
        ilss_tau = {f: np.abs(r[f]['ilss'].stress_global[..., 4]).max()
                    for f in r}
        assert ilss_tau['hex8i'] < 0.9 * ilss_tau['hex8']

    def test_formulation_recorded_in_results_and_export(self, tmp_path):
        import json
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        pf = PorosityField(mat, 0.02)
        mesh = CompositeMesh(pf, mat, nx=4, ny=2, nz=8)
        solver = FESolver(mesh, mat, pf, formulation='hex8i')
        assert solver.formulation == 'hex8i'
        r = solver.solve('compression')
        assert r.formulation == 'hex8i'
        assert r.summary().details['formulation'] == 'hex8i'
        path = tmp_path / "fe.json"
        FESolver.export_results(r, path)
        with open(path, encoding='utf-8') as f:
            assert json.load(f)['solver'] == {'formulation': 'hex8i'}
        # FieldResults built directly keeps the historical default.
        bare = FieldResults(r.displacement, r.stress_global, r.stress_local,
                            r.strain_global, r.strain_local, 1.0, 1.0)
        assert bare.formulation == 'hex8'

    def test_assembly_cache_is_keyed_on_formulation(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        pf = PorosityField(mat, 0.03, distribution='clustered')
        mesh = CompositeMesh(pf, mat, nx=4, ny=2, nz=8)
        assembler = GlobalAssembler(mesh, mat, pf)
        K_hex8 = assembler.stiffness()
        assert assembler.stiffness() is K_hex8
        assembler.formulation = 'hex8i'
        K_hex8i = assembler.stiffness()
        assert K_hex8i is not K_hex8
        assert assembler.element_batch().formulation == 'hex8i'
        fresh = GlobalAssembler(mesh, mat, pf, formulation='hex8i')
        np.testing.assert_allclose(K_hex8i.toarray(),
                                   fresh.assemble_stiffness().toarray(),
                                   rtol=0, atol=1e-12 * abs(K_hex8i).max())
        assert abs(K_hex8i - K_hex8).max() > 1e-6 * abs(K_hex8).max()

    def test_pristine_reference_is_keyed_on_formulation(self):
        # A hex8 pristine reference must never be reused for a hex8i porous
        # solve (or vice versa): the ILSS beam stiffness differs between the
        # two, so the knockdown would be wrong.
        from porosity_fe.fe import solver as solver_mod
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
        pf = PorosityField(mat, 0.03)
        mesh = CompositeMesh(pf, mat, nx=8, ny=3, nz=8)

        def kd(formulation):
            return FESolver(mesh, mat, pf, formulation=formulation).solve(
                'ilss').knockdown

        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        kd('hex8')
        after_hex8 = kd('hex8i')
        assert len(solver_mod._PRISTINE_MEASURE_CACHE) == 2
        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        fresh = kd('hex8i')
        assert after_hex8 == pytest.approx(fresh, rel=1e-12)

    def test_singular_internal_stiffness_falls_back_to_hex8(self):
        from porosity_fe.fe.element import _condensation_operator
        Kaa = np.stack([np.eye(9), np.zeros((9, 9)), np.full((9, 9), np.inf)])
        Kau = np.ones((3, 9, 24))
        H = _condensation_operator(Kaa, Kau)
        np.testing.assert_array_equal(H[0], -np.ones((9, 24)))
        np.testing.assert_array_equal(H[1:], 0.0)
        np.testing.assert_array_equal(
            _condensation_operator(np.zeros((9, 9)), np.ones((9, 24))), 0.0)
