"""Global FE assembler + boundary handler."""

from __future__ import annotations

import hashlib
import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import astuple, dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse

from .._types import FEFormulation
from ..materials import MaterialProperties
from ..mesh import CompositeMesh, FlexureGeometry
from ..porosity_field import PorosityField
from .batch import ElementBatch, build_element_batch, element_dofs
from .element import _DEFAULT_FORMULATION, Hex8Element, _check_formulation
from .short_beam import require_ilss_geometry

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
    formulation : {'hex8i', 'hex8'}, optional
        Element formulation (see :class:`Hex8Element`), default
        ``'hex8i'``. Part of the assembly cache key, so changing
        ``self.formulation`` re-assembles.
    """

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField, *,
                 formulation: FEFormulation = _DEFAULT_FORMULATION) -> None:
        self.mesh = mesh
        self.material = material
        self.porosity_field = porosity_field
        _check_formulation(formulation)
        self.formulation: FEFormulation = formulation
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
            formulation=self.formulation,
        )

    def element_dof_indices(self, elem_idx: int) -> np.ndarray:
        """Global DOF indices (24,) for an element's 8 nodes."""
        return element_dofs(self.mesh.elements[elem_idx:elem_idx + 1])[0]

    def _state_key(self) -> tuple:
        """Fingerprint of every input the stiffness depends on.

        Hashes the mesh arrays by content (so in-place edits are seen) and
        the material (every field, not its abbreviated ``repr``), void
        shape and element formulation by value.
        """
        h = hashlib.blake2b(digest_size=16)
        mesh = self.mesh
        for arr in (mesh.nodes, mesh.elements, mesh.porosity,
                    mesh.ply_angles, mesh.void_elements):
            a = np.ascontiguousarray(arr)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
        return (h.hexdigest(), astuple(self.material),
                tuple(self.porosity_field.void_shape_radii),
                self.formulation)

    def element_batch(self) -> ElementBatch:
        """Per-element, per-Gauss-point ``B``, ``C`` and ``det(J) w`` arrays.

        Reuses the batch from the last assembly while the mesh, material and
        porosity inputs are unchanged; rebuilds it otherwise.
        """
        key = self._state_key()
        if self._batch is None or key != self._key:
            self._batch = build_element_batch(
                self.mesh, self.material, self.porosity_field.void_shape_radii,
                formulation=self.formulation)
            self._key = key
            self._K = None
        return self._batch

    def stiffness(self, verbose: bool = False) -> scipy.sparse.csc_matrix:
        """Global K, re-assembled only when its inputs have changed.

        The returned matrix is shared with later calls; treat it as
        read-only (``BoundaryHandler.apply_elimination`` returns a new
        matrix).
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
            self.mesh, self.material, self.porosity_field.void_shape_radii,
            formulation=self.formulation)
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

def _hat_weights(grid: np.ndarray, x0: float, width: float,
                 tol: float) -> np.ndarray:
    """Consistent nodal shares of a unit load on a 1D grid of linear elements.

    ``width = 0``: the linear shape functions at ``x0`` (a point within
    ``tol`` of a node goes wholly to that node). ``width > 0``: the shape
    functions averaged over ``[x0 - width/2, x0 + width/2]``, a uniform
    pressure. The weights sum to 1.
    """
    n = grid.size
    if n == 0:
        raise ValueError("The loaded face has no nodes.")
    lo, hi = float(grid[0]), float(grid[-1])
    a, b = x0 - 0.5 * width, x0 + 0.5 * width
    if a < lo - tol or b > hi + tol:
        raise ValueError(
            f"Load at x = {x0:g} mm (width {width:g} mm) lies outside the "
            f"loaded face, x in [{lo:g}, {hi:g}] mm.")
    w = np.zeros(n)
    if width == 0.0:
        near = int(np.argmin(np.abs(grid - x0)))
        if abs(float(grid[near]) - x0) <= tol or n == 1:
            w[near] = 1.0
            return w
        i = int(np.searchsorted(grid, x0))
        t = (x0 - grid[i - 1]) / (grid[i] - grid[i - 1])
        w[i - 1], w[i] = 1.0 - t, t
        return w
    left, right = grid[:-1], grid[1:]
    p, q = np.clip(a, left, right), np.clip(b, left, right)
    h = right - left
    w[:-1] += ((right - p) ** 2 - (right - q) ** 2) / (2.0 * h)
    w[1:] += ((q - left) ** 2 - (p - left) ** 2) / (2.0 * h)
    return w / w.sum()


@dataclass(frozen=True)
class _ConstraintReduction:
    """Affine reduction ``u = T u_r + g`` of the global DOFs.

    The one place where constraints are removed from the system. It takes
    two kinds of constraint:

    - prescribed (Dirichlet) values, ``u_c = values``;
    - ties (master-slave multi-point constraints), ``u_s = u_m + offset``,
      for example the periodic conditions ``u(x + L) = u(x) + E L`` of a
      unit cell under the macroscopic strain ``E``.

    The independent unknowns ``u_r`` are the DOFs that are neither
    prescribed nor tied (:attr:`free`). ``T`` (``n_dof x n_free``) has a
    unit entry in each free row and in each tied row whose master is free;
    ``g`` (:meth:`lift`) holds the prescribed values and the tie offsets.
    The reduced system is ``T^T K T u_r = T^T (F - K g)``: the stationary
    point of the potential energy on the constrained subspace, symmetric
    positive definite when the constraints remove every rigid-body mode.

    Without ties ``T`` only selects the free columns, so ``T^T K T`` is the
    free-DOF block ``K_ff`` and the right-hand side is ``F_f - K_fc u_c``.
    That case is computed by slicing, as the solver always has (``T`` is
    not built), so Dirichlet-only results are unchanged bit for bit. Every
    reduction goes through :meth:`reduce_matrix`, :meth:`reduce_rhs` and
    :meth:`expand`.

    Attributes
    ----------
    n_dof : int
        Total number of DOFs.
    free : np.ndarray
        Sorted independent DOF indices (the columns of ``T``), shape
        ``(n_f,)``.
    fixed : np.ndarray
        Sorted prescribed DOF indices, shape ``(n_c,)``.
    values : np.ndarray
        Prescribed values aligned with ``fixed``.
    key : bytes
        Identifies the reduced matrix, which depends on the constrained
        *topology* only (the prescribed set and the resolved tie map), not
        on the values or offsets. ``fixed.tobytes()`` without ties, as
        before; with ties a longer string whose length is not a multiple of
        the index size, so the two kinds never collide.
    tied : np.ndarray
        Sorted tied (slave) DOF indices, shape ``(n_t,)``; empty without
        ties.
    masters : np.ndarray
        Ultimate master of each tied DOF, aligned with ``tied``: a chain
        ``a -> b -> c`` is resolved to ``a -> c``. A master is free or
        prescribed, never tied.
    offsets : np.ndarray
        Accumulated offsets along each chain, aligned with ``tied``:
        ``u[tied] = u[masters] + offsets``.
    transform : scipy.sparse.csr_matrix or None
        ``T``; ``None`` without ties.
    """

    n_dof: int
    free: np.ndarray
    fixed: np.ndarray
    values: np.ndarray
    key: bytes
    tied: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    masters: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    transform: scipy.sparse.csr_matrix | None = None

    @classmethod
    def from_constraints(
        cls, n_dof: int, constrained: dict[int, float], *,
        ties: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> _ConstraintReduction:
        """Build the reduction from prescribed values and optional ties.

        Parameters
        ----------
        n_dof : int
            Total number of DOFs.
        constrained : dict[int, float]
            ``{dof: prescribed_value}``.
        ties : (tied, master, offset), optional
            Three equal-length 1-D arrays: ``u[tied[k]] = u[master[k]] +
            offset[k]``. Each DOF is tied at most once and may not also be
            prescribed; a master may itself be tied (chains are resolved)
            or prescribed (the tied DOF then takes the master's value plus
            the offset). ``None`` or empty arrays give the Dirichlet-only
            reduction.

        Raises
        ------
        ValueError
            On a DOF index out of range, a non-finite value or offset, a DOF
            tied twice or both tied and prescribed, or a cycle of ties.
        """
        n = len(constrained)
        fixed = np.fromiter(constrained.keys(), dtype=np.intp, count=n)
        values = np.fromiter(constrained.values(), dtype=float, count=n)
        if n and (fixed.min() < 0 or fixed.max() >= n_dof):
            raise ValueError(
                f"Constrained DOF index out of range [0, {n_dof}): "
                f"min {int(fixed.min())}, max {int(fixed.max())}.")
        if not np.isfinite(values).all():
            raise ValueError("Prescribed displacements must be finite.")
        order = np.argsort(fixed, kind='stable')
        fixed, values = fixed[order], values[order]
        is_free = np.ones(n_dof, dtype=bool)
        is_free[fixed] = False
        if ties is None or len(ties[0]) == 0:
            return cls(n_dof=n_dof, free=np.flatnonzero(is_free), fixed=fixed,
                       values=values, key=fixed.tobytes())

        tied, masters, offsets = cls._resolve_ties(n_dof, is_free, ties)
        is_free[tied] = False
        free = np.flatnonzero(is_free)
        column = np.full(n_dof, -1, dtype=np.intp)
        column[free] = np.arange(free.size)
        # One unit entry per free row and per tied row with a free master;
        # a tied DOF whose master is prescribed has a zero row (all in g).
        master_col = column[masters]
        on_free = master_col >= 0
        rows = np.concatenate([free, tied[on_free]])
        cols = np.concatenate([np.arange(free.size), master_col[on_free]])
        transform = scipy.sparse.csr_matrix(
            (np.ones(rows.size), (rows, cols)), shape=(n_dof, free.size))
        # bytes of intp arrays have a length divisible by the index size;
        # the 6-byte separator keeps tie keys off every Dirichlet key.
        key = fixed.tobytes() + b'|ties|' + tied.tobytes() + masters.tobytes()
        return cls(n_dof=n_dof, free=free, fixed=fixed, values=values, key=key,
                   tied=tied, masters=masters, offsets=offsets,
                   transform=transform)

    @staticmethod
    def _resolve_ties(n_dof: int, is_free: np.ndarray,
                      ties: tuple[np.ndarray, np.ndarray, np.ndarray]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate ties and resolve chains to their ultimate masters.

        Returns ``(tied, masters, offsets)`` sorted by the tied DOF.
        """
        tied = np.asarray(ties[0], dtype=np.intp).ravel()
        masters = np.asarray(ties[1], dtype=np.intp).ravel()
        offsets = np.asarray(ties[2], dtype=float).ravel()
        if not tied.size == masters.size == offsets.size:
            raise ValueError(
                f"ties must be three equal-length arrays (tied, master, "
                f"offset); got lengths {tied.size}, {masters.size}, "
                f"{offsets.size}.")
        for name, idx in (('Tied', tied), ('Master', masters)):
            if idx.min() < 0 or idx.max() >= n_dof:
                raise ValueError(
                    f"{name} DOF index out of range [0, {n_dof}): "
                    f"min {int(idx.min())}, max {int(idx.max())}.")
        if not np.isfinite(offsets).all():
            raise ValueError("Tie offsets must be finite.")
        order = np.argsort(tied, kind='stable')
        tied, masters, offsets = tied[order], masters[order], offsets[order]
        if np.any(tied[1:] == tied[:-1]):
            dup = int(tied[1:][tied[1:] == tied[:-1]][0])
            raise ValueError(f"DOF {dup} is tied to more than one master.")
        prescribed = ~is_free[tied]
        if np.any(prescribed):
            raise ValueError(
                f"DOF {int(tied[prescribed][0])} is both tied and prescribed.")

        # Follow the chains: replace each master that is itself tied by its
        # own master, accumulating offsets. A chain cannot be longer than
        # the number of ties, so more passes than that means a cycle.
        link = np.full(n_dof, -1, dtype=np.intp)
        link[tied] = masters
        link_offset = np.zeros(n_dof, dtype=float)
        link_offset[tied] = offsets
        masters, offsets = masters.copy(), offsets.copy()
        for _ in range(tied.size + 1):
            step = link[masters] >= 0
            if not step.any():
                break
            offsets[step] += link_offset[masters[step]]
            masters[step] = link[masters[step]]
        else:
            raise ValueError("The ties contain a cycle (a DOF is its own master).")
        return tied, masters, offsets

    def lift(self) -> np.ndarray:
        """``g``: the prescribed values, the tie offsets (plus the value of a
        prescribed master) and zero at the independent DOFs."""
        u = np.zeros(self.n_dof, dtype=float)
        u[self.fixed] = self.values
        if self.tied.size:
            u[self.tied] = u[self.masters] + self.offsets
        return u

    def reduce_matrix(self, K: scipy.sparse.spmatrix) -> scipy.sparse.csc_matrix:
        """``T^T K T``; without ties the free-DOF block ``K_ff`` (column
        slice, then rows)."""
        K_csc = scipy.sparse.csc_matrix(K)
        if self.transform is None:
            return K_csc[:, self.free][self.free, :].tocsc()
        T = self.transform
        return (T.T @ (K_csc @ T)).tocsc()

    def reduce_rhs(self, K: scipy.sparse.spmatrix, F: np.ndarray) -> np.ndarray:
        """``T^T (F - K g)``; without ties ``F_f - K_fc u_c``, without
        building ``K_fc``."""
        F = np.asarray(F, dtype=float)
        if self.transform is None:
            return F[self.free] - (K @ self.lift())[self.free]
        return self.transform.T @ (F - K @ self.lift())

    def expand(self, u_f: np.ndarray) -> np.ndarray:
        """Full displacement vector ``T u_f + g`` from the reduced solution.

        Prescribed DOFs hold their values exactly.
        """
        u = self.lift()
        u[self.free] = u_f
        if self.transform is not None:
            # Tied rows of T u_f: the master's value, or 0 for a prescribed
            # master (whose value is already in g).
            u[self.tied] += (self.transform @ u_f)[self.tied]
        return u



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

    def ilss_geometry(self) -> FlexureGeometry:
        """Supports and loading-nose band of the short-beam (ILSS) test.

        :attr:`CompositeMesh.ilss_geometry`, set by
        :meth:`CompositeMesh.for_ilss`.

        Raises
        ------
        ValueError
            If the mesh has no ILSS geometry; the message says how to
            build the specimen (see
            :func:`porosity_fe.fe.short_beam.require_ilss_geometry`).
        """
        return require_ilss_geometry(self.mesh)

    def ilss_bcs(self, applied_load: float = -10.0, *,
                 geometry: FlexureGeometry | None = None,
                 ) -> tuple[dict[int, float], np.ndarray]:
        """ASTM D2344 short-beam-shear (ILSS) boundary conditions.

        Three-point bending of the short-beam specimen: :meth:`flexure_bcs`
        with ``procedure='3pt'`` at the specimen's supports and loading
        band, i.e.

        - roller supports (:meth:`roller_supports`) on the bottom-face node
          lines ``geometry.support_x``, which exert no horizontal reaction;
        - ``applied_load`` (N, along ``z``, negative = downward) at
          mid-span as a uniform pressure over the band
          ``geometry.load_width`` across the top face, with consistent
          nodal forces (:meth:`line_load`).

        Force controlled: the supports are prescribed to zero and the load
        is in ``F``.

        Parameters
        ----------
        applied_load : float
            Total load along ``z`` (N), default ``-10.0``.
        geometry : FlexureGeometry, optional
            Keyword-only supports and band; default :meth:`ilss_geometry`
            (the specimen from :meth:`CompositeMesh.for_ilss`).

        Returns
        -------
        constrained_dofs : dict
            ``{global_dof: 0.0}`` from :meth:`roller_supports`.
        F : np.ndarray
            Shape ``(n_dof,)`` force vector on the top face.

        Raises
        ------
        ValueError
            If the mesh has no ILSS geometry and none is given, or a
            support is not on a node line.

        Notes
        -----
        Changed in the ASTM D2344 rework (IMPROVEMENT_PLAN 3.6, C2). Before,
        this pinned ``u_x``, ``u_y`` and ``u_z`` along both bottom end edges
        of whatever coupon the mesh was (span ``L_x``, 11.4 thicknesses on
        the 50 mm coupon) and split the load equally over the mid-span top
        nodes. The pinned edges made a tied arch (horizontal reaction about
        4 P, deflection about half the simply supported one), and the long
        span made the test mostly flexure. The closest thing to the old
        model is :meth:`flexure_bcs` (``loading='flexure_3pt'``) on a mesh
        from ``CompositeMesh.for_flexure(field, material, span=50.0,
        width=20.0, overhang=0.0)``, the old coupon on rollers instead of
        pinned edges.
        """
        geom = self.ilss_geometry() if geometry is None else geometry
        return self.flexure_bcs(applied_load, procedure='3pt', geometry=geom)

    def _line_tolerance(self) -> float:
        """Distance (mm) within which a node lies on a requested ``x`` line."""
        x = self.mesh.nodes[:, 0]
        return 1e-9 * max(float(x.max() - x.min()), 1.0)

    def face_line_nodes(self, face: Literal['z_min', 'z_max'], x: float
                        ) -> np.ndarray:
        """Nodes of the bottom or top face on the line ``x = x0``.

        Parameters
        ----------
        face : {'z_min', 'z_max'}
            Bottom or top face.
        x : float
            ``x`` of the line (mm). Matched to within ``1e-9`` of the
            specimen length.

        Returns
        -------
        np.ndarray
            Node indices sorted by ``y``; empty when no node line lies at
            ``x``.
        """
        if face not in ('z_min', 'z_max'):
            raise ValueError(f"face must be 'z_min' or 'z_max', got {face!r}.")
        nodes = self.mesh.nodes_on_face(face)
        on_line = nodes[np.abs(self.mesh.nodes[nodes, 0] - float(x))
                        <= self._line_tolerance()]
        return on_line[np.argsort(self.mesh.nodes[on_line, 1], kind='stable')]

    def roller_supports(self, support_x: Sequence[float]) -> dict[int, float]:
        """Simply supported line supports (rollers) on the bottom face.

        Each support is the node line ``x = x_s`` of the bottom face
        (``z_min``) across the full width, and holds ``u_z = 0`` at every
        node of the line. The supports must lie on node lines. The rollers
        leave the in-plane rigid-body motions free (translation along ``x``
        and ``y``, rotation about ``z``); three more constraints at the
        mid-width node of the outer lines remove them: ``u_x = 0`` on the
        first line, ``u_y = 0`` on the first and the last. These three are
        statically determinate in the ``x``-``y`` plane, so under loads
        normal to the faces they carry no force: the supports exert no
        horizontal reaction, and the beam is free to shorten, stretch and
        contract across its width. Pinning ``u_x`` along both supports (what
        :meth:`ilss_bcs` did before the ASTM D2344 rework) instead ties the
        support lines together, and the beam carries part of the load as an
        arch.

        Parameters
        ----------
        support_x : sequence of float
            ``x`` of the support lines (mm), at least two distinct values.

        Returns
        -------
        dict
            ``{global_dof: 0.0}`` for every constrained DOF.

        Raises
        ------
        ValueError
            If fewer than two distinct supports are given or a support is
            not on a bottom-face node line.
        """
        xs = sorted({float(x) for x in support_x})
        if len(xs) < 2:
            raise ValueError(
                "roller_supports needs at least two distinct support lines; "
                f"got {list(support_x)!r}.")
        lines = []
        for x in xs:
            line = self.face_line_nodes('z_min', x)
            if line.size == 0:
                raise ValueError(
                    f"No bottom-face node line at x = {x:g} mm for a roller "
                    "support. Supports must lie on node lines: use "
                    "CompositeMesh.for_flexure, or choose L_x and nx so that "
                    "the support falls on a node.")
            lines.append(line)

        nodes = self.mesh.nodes
        constrained: dict[int, float] = {}
        for line in lines:
            for nid in line:
                constrained[3 * int(nid) + 2] = 0.0

        def mid_width(line: np.ndarray) -> int:
            y = nodes[line, 1]
            return int(line[np.argmin(np.abs(y - 0.5 * (y.min() + y.max())))])

        first, last = mid_width(lines[0]), mid_width(lines[-1])
        constrained[3 * first] = 0.0
        constrained[3 * first + 1] = 0.0
        constrained[3 * last + 1] = 0.0
        return constrained

    def line_load(self, x: float, total_load: float, *,
                  width: float = 0.0) -> np.ndarray:
        """Consistent nodal forces of a load across the width of the top face.

        Spreads ``total_load`` (N, along ``z``; negative is downward)
        uniformly along ``y`` over the full width of the top face
        (``z_max``), on the line ``x = x0`` (``width = 0``) or as a uniform
        pressure over the band ``|x - x0| <= width / 2``. The nodal forces
        are the work-equivalent (consistent) loads of the bilinear face shape
        functions:

        - along ``y``, trapezoidal weights: interior nodes take
          ``dy / b`` of the load, the two edge nodes half that;
        - along ``x``, each node line takes its linear shape function
          evaluated at ``x0`` (``width = 0``) or averaged over the band, so a
          line load on a node line loads only that line, and one between
          two node lines is shared in proportion to the distance.

        Equal shares (what :meth:`ilss_bcs` used before the ASTM D2344
        rework) put the same force on the two edge nodes as on the interior
        ones (with five nodes across, 1/5 each instead of 1/8), which
        concentrates load at the free edges.

        Parameters
        ----------
        x : float
            Centre of the line or band (mm), within the top face.
        total_load : float
            Total force along ``z`` (N).
        width : float
            Band width along ``x`` (mm), default ``0.0`` (a line load).

        Returns
        -------
        np.ndarray
            Shape ``(n_dof,)`` force vector, with ``z`` components only,
            summing to ``total_load``.

        Raises
        ------
        ValueError
            If ``total_load`` or ``width`` is not finite, ``width`` is
            negative, or the line or band leaves the top face.
        """
        total = float(total_load)
        w = float(width)
        if not math.isfinite(total):
            raise ValueError(f"total_load must be finite, got {total_load!r}.")
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"width must be a finite length >= 0, got {width!r}.")
        top = self.mesh.nodes_on_face('z_max')
        grid = np.unique(self.mesh.nodes[top, 0])
        weights = _hat_weights(grid, float(x), w, self._line_tolerance())

        F = np.zeros(self.mesh.n_dof, dtype=np.float64)
        for xi, wx in zip(grid, weights, strict=True):
            if wx == 0.0:
                continue
            line = self.face_line_nodes('z_max', float(xi))
            y = self.mesh.nodes[line, 1]
            wy = np.zeros(line.size)
            if line.size == 1:
                wy[0] = 1.0
            else:
                seg = np.diff(y)
                wy[:-1] += 0.5 * seg
                wy[1:] += 0.5 * seg
                wy /= wy.sum()
            F[3 * line + 2] += total * wx * wy
        return F

    def flexure_geometry(self) -> FlexureGeometry:
        """Flexure test geometry of the mesh.

        :attr:`CompositeMesh.flexure_geometry` when set (by
        :meth:`CompositeMesh.for_flexure`); otherwise supports at the two
        bottom end edges (span ``L_x``, no overhang) and a four-point load
        span of half the span.
        """
        geometry = getattr(self.mesh, 'flexure_geometry', None)
        if geometry is not None:
            return geometry
        x = self.mesh.nodes[:, 0]
        x0, x1 = float(x.min()), float(x.max())
        return FlexureGeometry(support_x=(x0, x1), load_span=0.5 * (x1 - x0))

    def flexure_bcs(self, applied_load: float = -10.0, *,
                    procedure: Literal['3pt', '4pt'] = '3pt',
                    geometry: FlexureGeometry | None = None,
                    ) -> tuple[dict[int, float], np.ndarray]:
        """Three- or four-point bending on roller supports.

        - Supports: :meth:`roller_supports` at ``geometry.support_x``
          (span ``L``), so the supports exert no horizontal reaction.
        - Loads: ``'3pt'`` (ASTM D790, D7264 procedure A) puts
          ``applied_load`` at mid-span; ``'4pt'`` (D7264 procedure B) puts
          half of it on each of two lines ``geometry.load_span`` apart,
          centred on mid-span (the quarter points for the D7264 load span
          ``L / 2``). Each load is a consistent line load (or band, with
          ``geometry.load_width``) across the top face, see
          :meth:`line_load`.

        Force controlled, like :meth:`ilss_bcs`: the supports are
        prescribed to zero and the load is in ``F``.

        Parameters
        ----------
        applied_load : float
            Total load along ``z`` (N, negative = downward), default
            ``-10.0``.
        procedure : {'3pt', '4pt'}
            Three-point (default) or four-point bending.
        geometry : FlexureGeometry, optional
            Supports and load lines; defaults to :meth:`flexure_geometry`.

        Returns
        -------
        constrained_dofs : dict
            ``{global_dof: 0.0}`` from :meth:`roller_supports`.
        F : np.ndarray
            Shape ``(n_dof,)`` force vector on the top face.

        Raises
        ------
        ValueError
            If ``procedure`` is unknown, a support is not on a node line,
            or a load line or band leaves the top face.
        """
        geom = self.flexure_geometry() if geometry is None else geometry
        load_x = geom.load_x(procedure)
        constrained = self.roller_supports(geom.support_x)
        F = np.zeros(self.mesh.n_dof, dtype=np.float64)
        for x in load_x:
            F += self.line_load(x, applied_load / len(load_x),
                                width=geom.load_width)
        return constrained, F

    def free_bcs(self) -> tuple[dict[int, float], np.ndarray]:
        """Statically determinate 3-2-1 supports for a free-standing laminate.

        Removes only the six rigid-body modes, on the ``z_min`` face:

        - corner A ``(x_min, y_min)``: ``ux = uy = uz = 0``;
        - corner B ``(x_max, y_min)``: ``uy = uz = 0``;
        - corner C ``(x_min, y_max)``: ``uz = 0``.

        Because the set is statically determinate, a self-equilibrated
        load (such as a thermal load) leaves zero reactions, and the
        laminate expands, contracts and warps freely. Used by the
        ``'thermal'`` loading mode.

        Returns
        -------
        constrained_dofs : dict
            Six ``{global_dof: 0.0}`` entries.
        F : np.ndarray
            Shape ``(n_dof,)`` zero force vector (the solver adds the
            thermal load).

        Raises
        ------
        RuntimeError
            If a support corner cannot be located on the mesh.
        """
        faces = {f: self.mesh.nodes_on_face(f)
                 for f in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min')}

        def corner(fx: str, fy: str) -> int:
            nodes = np.intersect1d(np.intersect1d(faces[fx], faces[fy]),
                                   faces['z_min'])
            if nodes.size == 0:
                raise RuntimeError(
                    f"free_bcs: no node on the ({fx}, {fy}, z_min) corner. "
                    "Check mesh generation.")
            return int(nodes[0])

        a = corner('x_min', 'y_min')
        b = corner('x_max', 'y_min')
        c = corner('x_min', 'y_max')
        constrained = {3 * a: 0.0, 3 * a + 1: 0.0, 3 * a + 2: 0.0,
                       3 * b + 1: 0.0, 3 * b + 2: 0.0,
                       3 * c + 2: 0.0}
        return constrained, np.zeros(self.mesh.n_dof, dtype=np.float64)

    @staticmethod
    def apply_elimination(K: scipy.sparse.spmatrix, F: np.ndarray,
                          constrained_dofs: dict[int, float]
                          ) -> tuple[scipy.sparse.csc_matrix, np.ndarray, np.ndarray]:
        """Eliminate prescribed DOFs exactly (partitioned system).

        Splits the DOFs into free (``f``) and constrained (``c``) sets and
        returns the reduced system ``K_ff u_f = F_f - K_fc u_c``. Solve it
        with any linear solver, then assemble the full vector::

            K_ff, rhs, free = BoundaryHandler.apply_elimination(K, F, constrained)
            u = np.zeros(K.shape[0])
            u[list(constrained)] = list(constrained.values())
            u[free] = scipy.sparse.linalg.spsolve(K_ff, rhs)

        The constrained DOFs then hold their prescribed values exactly, and
        ``K_ff`` keeps the conditioning of the physical stiffness (no
        penalty term). A load in ``F`` at a constrained DOF is carried by
        the support: it appears in the reaction ``(K u - F)[c]``.

        Parameters
        ----------
        K : scipy.sparse matrix
            Global stiffness matrix (not modified).
        F : np.ndarray
            Global force vector (not modified).
        constrained_dofs : dict
            ``{dof_index: prescribed_value}``.

        Returns
        -------
        K_ff : scipy.sparse.csc_matrix
            Free-DOF block of ``K``, shape ``(n_f, n_f)``.
        rhs : np.ndarray
            ``F_f - K_fc u_c``, shape ``(n_f,)``.
        free_dofs : np.ndarray
            Sorted free DOF indices, shape ``(n_f,)``.

        Raises
        ------
        ValueError
            If a constrained DOF index is out of range or a prescribed
            value is not finite.
        """
        part = _ConstraintReduction.from_constraints(K.shape[0], constrained_dofs)
        return part.reduce_matrix(K), part.reduce_rhs(K, F), part.free

    @staticmethod
    def apply_penalty(K: scipy.sparse.csc_matrix, F: np.ndarray,
                      constrained_dofs: dict[int, float],
                      penalty_factor: float = 1e6
                      ) -> tuple[scipy.sparse.csc_matrix, np.ndarray]:
        """Apply penalty method for prescribed displacements (deprecated).

        Deprecated: :class:`FESolver` now eliminates prescribed DOFs
        exactly; use :meth:`apply_elimination`. This method still returns
        the same penalty-modified system, emits a
        :class:`DeprecationWarning`, and will be removed in 2.0.

        For each constrained DOF ``i`` with value ``v``,
        ``K[i, i] += alpha`` and ``F[i] = alpha * v``, where
        ``alpha = penalty_factor * max(diag(K))``. Any load already in
        ``F`` at a constrained DOF is overwritten.

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
        warnings.warn(
            "BoundaryHandler.apply_penalty is deprecated: FESolver imposes "
            "prescribed displacements exactly by elimination. Use "
            "BoundaryHandler.apply_elimination instead; apply_penalty will be "
            "removed in 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
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
