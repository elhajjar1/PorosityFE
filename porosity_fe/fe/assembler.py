"""Global FE assembler + boundary handler."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import astuple

import numpy as np
import scipy.sparse

from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..porosity_field import PorosityField
from .batch import ElementBatch, build_element_batch, element_dofs
from .element import Hex8Element

logger = logging.getLogger("porosity_fe_analysis")

# ============================================================
# SECTION 7e: GLOBAL ASSEMBLER
# ============================================================

class GlobalAssembler:
    """Assembles global stiffness matrix from Hex8Element contributions.

    Uses COO format for assembly, converts to CSC for solving.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    material : MaterialProperties
        Material properties.
    porosity_field : PorosityField
        Porosity field for degradation.
    """

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField) -> None:
        self.mesh = mesh
        self.material = material
        self.porosity_field = porosity_field
        self._C_base = material.get_stiffness_matrix()
        self._C_m = material.get_isotropic_matrix_stiffness()
        self._nu_m = material.matrix_poisson
        self._void_shape = porosity_field.void_shape_radii
        # Assembly results, valid while _state_key() still returns _key.
        self._key: tuple | None = None
        self._batch: ElementBatch | None = None
        self._K: scipy.sparse.csc_matrix | None = None

    def create_element(self, elem_idx: int) -> Hex8Element:
        """Create a Hex8Element for the given element index.

        Elements whose centroid falls inside a discrete void get is_void=True,
        which sets their stiffness to near-zero (~1 MPa), creating an explicit
        void inclusion in the mesh.
        """
        node_ids = self.mesh.elements[elem_idx]
        node_coords = self.mesh.nodes[node_ids]
        ply_angle = float(self.mesh.ply_angles[elem_idx])
        node_porosities = self.mesh.porosity[node_ids]
        is_void = elem_idx in self.mesh.void_element_set
        return Hex8Element(
            node_coords=node_coords,
            C_base=self._C_base,
            ply_angle_deg=ply_angle,
            node_porosities=node_porosities,
            void_shape_radii=self._void_shape,
            nu_m=self._nu_m,
            C_m=self._C_m,
            is_void=is_void,
            material=self.material,
        )

    def element_dof_indices(self, elem_idx: int) -> np.ndarray:
        """Global DOF indices (24,) for an element's 8 nodes."""
        return element_dofs(self.mesh.elements[elem_idx:elem_idx + 1])[0]

    def _state_key(self) -> tuple:
        """Fingerprint of every input the stiffness depends on.

        Hashes the mesh arrays by content (so in-place edits are seen) and
        the material (every field, not its abbreviated ``repr``) and void
        shape by value.
        """
        h = hashlib.blake2b(digest_size=16)
        mesh = self.mesh
        for arr in (mesh.nodes, mesh.elements, mesh.porosity,
                    mesh.ply_angles, mesh.void_elements):
            a = np.ascontiguousarray(arr)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
        return (h.hexdigest(), astuple(self.material),
                tuple(self.porosity_field.void_shape_radii))

    def element_batch(self) -> ElementBatch:
        """Per-element, per-Gauss-point ``B``, ``C`` and ``det(J) w`` arrays.

        Reuses the batch from the last assembly while the mesh, material and
        porosity inputs are unchanged; rebuilds it otherwise.
        """
        key = self._state_key()
        if self._batch is None or key != self._key:
            self._batch = build_element_batch(
                self.mesh, self.material, self.porosity_field.void_shape_radii)
            self._key = key
            self._K = None
        return self._batch

    def stiffness(self, verbose: bool = False) -> scipy.sparse.csc_matrix:
        """Global K, re-assembled only when its inputs have changed.

        The returned matrix is shared with later calls; treat it as
        read-only (``BoundaryHandler.apply_penalty`` returns a new matrix).
        """
        if self._K is None or self._state_key() != self._key:
            self.assemble_stiffness(verbose=verbose)
        assert self._K is not None
        return self._K

    def assemble_stiffness(self, verbose: bool = False) -> scipy.sparse.csc_matrix:
        """Assemble global stiffness matrix K in CSC format.

        All element stiffness matrices are computed together from an
        :class:`ElementBatch` (see :mod:`porosity_fe.fe.batch`) and scattered
        into a COO matrix of ``n_elem * 576`` entries. Always assembles;
        :meth:`stiffness` returns the cached result when inputs are unchanged.
        """
        self._batch = build_element_batch(
            self.mesh, self.material, self.porosity_field.void_shape_radii)
        self._key = self._state_key()
        batch = self._batch
        Ke = batch.stiffness_matrices()

        # An element whose contribution overflows is recomputed through
        # Hex8Element, which zeroes non-finite terms per Gauss point.
        overflowed = np.flatnonzero(~np.isfinite(Ke).all(axis=(1, 2)))
        for e in overflowed:
            Ke_e = self.create_element(int(e)).stiffness_matrix()
            Ke[e] = 0.5 * (Ke_e + Ke_e.T)

        n_elem = self.mesh.n_elements
        n_dof = self.mesh.n_dof
        dofs = batch.dofs
        coo_rows = np.broadcast_to(dofs[:, :, None], Ke.shape).ravel()
        coo_cols = np.broadcast_to(dofs[:, None, :], Ke.shape).ravel()
        coo_vals = Ke.ravel()

        if verbose:
            n_void = len(self.mesh.void_elements)
            logger.info("  Assembled %d elements", n_elem)
            logger.info(
                "  Void inclusion elements: %d (E ~ %s MPa)",
                n_void, Hex8Element.VOID_MODULUS,
            )
            logger.info(
                "  Building sparse matrix: %d DOFs, %d COO entries",
                n_dof, coo_vals.size,
            )

        K_coo = scipy.sparse.coo_matrix(
            (coo_vals, (coo_rows, coo_cols)),
            shape=(n_dof, n_dof),
        )
        K_csc = K_coo.tocsc()

        if verbose:
            logger.info(
                "  CSC matrix: %d stored entries (%.1f per DOF)",
                K_csc.nnz, K_csc.nnz / n_dof,
            )

        self._K = K_csc
        return K_csc.copy()


# ============================================================
# SECTION 7f: BOUNDARY HANDLER
# ============================================================

class BoundaryHandler:
    """Handles boundary conditions for the porosity FE model.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    """

    def __init__(self, mesh: CompositeMesh) -> None:
        self.mesh = mesh

    def nodes_on_face(self, face: str) -> np.ndarray:
        """Return node indices on the specified face."""
        return self.mesh.nodes_on_face(face)

    def compression_bcs(self, applied_strain: float = -0.01
                        ) -> tuple[dict[int, float], np.ndarray]:
        """Standard uniaxial compression boundary conditions.

        - x_min: ux = 0
        - x_max: ux = applied_strain * Lx
        - y_min: uy = 0
        - one corner fixed in z

        Parameters
        ----------
        applied_strain : float
            Applied nominal strain (negative for compression).

        Returns
        -------
        constrained_dofs : dict
            {global_dof: prescribed_value}
        F : np.ndarray
            Shape (n_dof,) force vector (zeros for displacement-controlled).
        """
        n_dof = self.mesh.n_dof
        Lx = self.mesh.L_x
        prescribed_disp = applied_strain * Lx

        constrained: dict[int, float] = {}

        # Fix ux on x_min
        for nid in self.mesh.nodes_on_face('x_min'):
            constrained[3 * int(nid)] = 0.0

        # Prescribe ux on x_max
        for nid in self.mesh.nodes_on_face('x_max'):
            constrained[3 * int(nid)] = prescribed_disp

        # Fix uy on y_min (symmetry)
        for nid in self.mesh.nodes_on_face('y_min'):
            constrained[3 * int(nid) + 1] = 0.0

        # Fix uz on one corner node (rigid body)
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        ymin_nodes = self.mesh.nodes_on_face('y_min')
        zmin_nodes = self.mesh.nodes_on_face('z_min')
        corner = np.intersect1d(np.intersect1d(xmin_nodes, ymin_nodes), zmin_nodes)
        if corner.size > 0:
            constrained[3 * int(corner[0]) + 2] = 0.0
        else:
            constrained[3 * int(xmin_nodes[0]) + 2] = 0.0

        F = np.zeros(n_dof, dtype=np.float64)
        return constrained, F

    def tension_bcs(self, applied_strain: float = 0.01
                    ) -> tuple[dict[int, float], np.ndarray]:
        """Uniaxial tension boundary conditions (same structure, positive strain)."""
        return self.compression_bcs(applied_strain=applied_strain)

    def shear_bcs(self, applied_strain: float = 0.01
                  ) -> tuple[dict[int, float], np.ndarray]:
        """Pure shear boundary conditions for engineering shear strain gamma_12.

        Prescribes the deformation field consistent with pure shear on ALL four
        side faces (±x and ±y) so that no spurious bending or traction-free
        condition biases the computed shear modulus G12.

        For engineering shear strain gamma = applied_strain, the displacement
        field is::

            u(x, y) = (gamma / 2) * y
            v(x, y) = (gamma / 2) * x
            w       = 0

        This gives ε_xy = gamma/2 (Voigt ε_6 = gamma), with all normal strains
        and out-of-plane shear strains exactly zero.

        BCs applied
        -----------
        - x_min (x = 0): ux = 0,              uy = (gamma/2) * y_node
        - x_max (x = Lx): ux = (gamma/2)*y_node, uy = (gamma/2)*Lx
        - y_min (y = 0): ux = 0,              uy = (gamma/2)*x_node
        - y_max (y = Ly): ux = (gamma/2)*Ly,  uy = (gamma/2)*x_node
        - uz = 0 pinned at the (x_min, y_min, z_min) corner to remove rigid-body
          motion in z.
        """
        n_dof = self.mesh.n_dof
        gamma = applied_strain
        nodes = self.mesh.nodes  # shape (n_nodes, 3)

        constrained: dict[int, float] = {}

        # ±x faces: u = gamma/2 * y_node,  v = gamma/2 * x_node
        for face in ('x_min', 'x_max'):
            for nid in self.mesh.nodes_on_face(face):
                nid = int(nid)
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                constrained[3 * nid]     = (gamma / 2.0) * y_n   # ux
                constrained[3 * nid + 1] = (gamma / 2.0) * x_n   # uy

        # ±y faces: u = gamma/2 * y_node,  v = gamma/2 * x_node
        for face in ('y_min', 'y_max'):
            for nid in self.mesh.nodes_on_face(face):
                nid = int(nid)
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                constrained[3 * nid]     = (gamma / 2.0) * y_n   # ux
                constrained[3 * nid + 1] = (gamma / 2.0) * x_n   # uy

        # Fix uz at one corner to prevent rigid-body translation in z
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        ymin_nodes = self.mesh.nodes_on_face('y_min')
        zmin_nodes = self.mesh.nodes_on_face('z_min')
        corner = np.intersect1d(np.intersect1d(xmin_nodes, ymin_nodes), zmin_nodes)
        if corner.size > 0:
            constrained[3 * int(corner[0]) + 2] = 0.0
        else:
            constrained[3 * int(xmin_nodes[0]) + 2] = 0.0

        F = np.zeros(n_dof, dtype=np.float64)
        return constrained, F

    def ilss_bcs(self, applied_load: float = -10.0
                 ) -> tuple[dict[int, float], np.ndarray]:
        """ILSS (interlaminar short-beam shear) boundary conditions, ASTM D2344.

        Three-point short-beam-shear setup:

        - Two simple supports at the bottom face (``z_min``), one along the
          ``x_min`` edge and one along the ``x_max`` edge. All three
          translational DOFs are pinned at the support nodes so the beam
          is fully simply-supported in the FE sense.
        - A downward (``-z``) midspan load applied as a nodal force on
          the top face (``z_max``) at ``x = L_x / 2``. The total load is
          ``applied_load`` (typically negative for "downward"); it is
          distributed equally across the midspan-top nodes.

        Unlike the compression/tension/shear BCs which are *displacement*
        controlled, ILSS is **force controlled** — the returned ``F``
        vector carries the load directly rather than being routed through
        ``apply_penalty``. The penalty path is still used for the support
        DOF constraints.

        Parameters
        ----------
        applied_load : float
            Total midspan load in the ``z`` direction (negative = downward).

        Returns
        -------
        constrained_dofs : dict
            ``{global_dof: prescribed_value}`` — all-zero values at the
            two support edges (left/right of the bottom face).
        F : np.ndarray
            Shape ``(n_dof,)`` force vector with the midspan load applied
            to the top-face nodes near ``x = Lx/2``.

        Notes
        -----
        This implementation assumes the **three-point bend** geometry of
        ASTM D2344. The standard four-point bend variant (ASTM D7264)
        requires a different BC method — either a new ``ilss_4pt_bcs``
        with two upper load rollers, or the empirical-only path. The
        Tsai-Wu / strength-recovery code already handles the multi-axial
        stress state recovered from this solve.
        """
        n_dof = self.mesh.n_dof
        Lx = self.mesh.L_x
        Lz = self.mesh.L_z

        zmin = self.mesh.nodes_on_face('z_min')
        xmin = self.mesh.nodes_on_face('x_min')
        xmax = self.mesh.nodes_on_face('x_max')

        support_left = np.intersect1d(zmin, xmin)
        support_right = np.intersect1d(zmin, xmax)

        if support_left.size == 0 or support_right.size == 0:
            raise RuntimeError(
                "ilss_bcs: failed to locate bottom-face support edges "
                "(intersection of z_min with x_min / x_max is empty). "
                "Check mesh generation."
            )

        constrained: dict[int, float] = {}
        for nid in np.concatenate([support_left, support_right]):
            nid = int(nid)
            constrained[3 * nid]     = 0.0  # ux
            constrained[3 * nid + 1] = 0.0  # uy
            constrained[3 * nid + 2] = 0.0  # uz

        F = np.zeros(n_dof, dtype=np.float64)
        # Tolerance must be wide enough to bracket *some* x-column when the
        # exact midspan does not fall on a mesh node. Use half the x-edge
        # plus a small fp epsilon so odd nx values still resolve a column.
        dx = Lx / max(self.mesh.nx, 1)
        dz = Lz / max(self.mesh.nz, 1)
        tol = 0.5 * np.hypot(dx, dz) + 1e-9
        midspan_top = self.mesh.find_nodes_near(x=Lx / 2.0, z=Lz, tol=tol)
        if midspan_top.size == 0:
            raise RuntimeError(
                "ilss_bcs: no top-face nodes found near midspan "
                f"(x = {Lx / 2.0}, z = {Lz}). Refine the mesh."
            )
        F[3 * midspan_top + 2] = applied_load / float(midspan_top.size)
        return constrained, F

    @staticmethod
    def apply_penalty(K: scipy.sparse.csc_matrix, F: np.ndarray,
                      constrained_dofs: dict[int, float],
                      penalty_factor: float = 1e6
                      ) -> tuple[scipy.sparse.csc_matrix, np.ndarray]:
        """Apply penalty method for prescribed displacements.

        For each constrained DOF ``i`` with value ``v``,
        ``K[i, i] += alpha`` and ``F[i] = alpha * v``, where
        ``alpha = penalty_factor * max(diag(K))``.

        Parameters
        ----------
        K : scipy.sparse.csc_matrix
            Global stiffness matrix.
        F : np.ndarray
            Global force vector.
        constrained_dofs : dict
            {dof_index: prescribed_value}.
        penalty_factor : float
            Multiplier for max diagonal entry. Defaults to ``1e6`` (six
            decades of BC enforcement), lowered from the historical
            ``1e8`` because the latter pushed ``cond(K_mod)`` to ~2.4e9
            and capped LU-vs-CG agreement at ~3e-6 even when the
            iterative residual was at machine precision (issue #60).

        Returns
        -------
        K_mod : scipy.sparse.csc_matrix
        F_mod : np.ndarray
        """
        if not constrained_dofs:
            return K, F

        if not scipy.sparse.issparse(K):
            K = scipy.sparse.csc_matrix(K)
        F_mod = F.copy()

        diag_max = np.abs(K.diagonal()).max()
        alpha = penalty_factor * max(diag_max, 1.0)

        n = len(constrained_dofs)
        dofs = np.fromiter(constrained_dofs.keys(), dtype=np.intp, count=n)
        vals = np.fromiter(constrained_dofs.values(), dtype=float, count=n)
        penalty = scipy.sparse.csc_matrix(
            (np.full(n, alpha), (dofs, dofs)), shape=K.shape)
        F_mod[dofs] = alpha * vals

        return (K + penalty).tocsc(), F_mod
