"""Structured hex mesh and quality checks."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ._ply_angles import _resolve_ply_angles
from ._types import MeshFace
from .gauss import gauss_points_hex
from .materials import MaterialProperties
from .porosity_field import PorosityField

logger = logging.getLogger("porosity_fe_analysis")

# Two normalized ply angles closer than this (degrees) are the same angle.
_ANGLE_TOL = 1e-6


def _ply_at(num: np.ndarray, den: int, n_plies: int) -> np.ndarray:
    """Ply index at the through-thickness positions ``num / den`` (ply units).

    Evaluated in integer arithmetic, so a position exactly on a ply
    interface always takes the ply above it instead of whichever side
    floating-point rounding happens to land on. The top surface maps to
    the last ply.
    """
    return np.clip(np.asarray(num) // den, 0, n_plies - 1)


def _normalize_angles(angles: np.ndarray) -> np.ndarray:
    """Map ply angles (degrees) into ``(-90, 90]``, so 90 and -90 compare equal."""
    a = 90.0 - np.mod(90.0 - np.asarray(angles, dtype=float), 180.0)
    return np.round(a, 6) + 0.0  # + 0.0 turns -0.0 into 0.0


def _angle_fractions(angles: np.ndarray) -> dict[float, float]:
    """Fraction of the entries at each normalized angle (equal thicknesses)."""
    values, counts = np.unique(_normalize_angles(angles), return_counts=True)
    return {float(v): int(c) / angles.size for v, c in zip(values, counts, strict=True)}


def _is_balanced(fractions: dict[float, float]) -> bool:
    """Every off-axis angle ``+theta`` has as much thickness as ``-theta``."""
    return all(abs(f - fractions.get(-a, 0.0)) <= 1e-9
               for a, f in fractions.items() if _ANGLE_TOL < abs(a) < 90.0 - _ANGLE_TOL)


# ============================================================
# SECTION 4: MESH GENERATION
# ============================================================

class CompositeMesh:
    """3D structured hexahedral mesh of a composite coupon with porosity.

    Builds a regular grid of 8-node hexahedral elements over a
    rectangular coupon of dimensions ``L_x x L_y x L_z``, samples the
    porosity field at every node, assigns each element the ply id and
    ply angle (degrees) of the ply containing its centroid, and flags
    elements whose centroid falls inside any explicit
    :class:`VoidGeometry` for the explicit-inclusion solver path.

    The in-plane coupon size is **hard-coded** to ``L_x = 50.0`` mm and
    ``L_y = 20.0`` mm (a standard ASTM-style coupon). Through-thickness
    ``L_z`` is taken from ``material.total_thickness``
    (``t_ply * n_plies``). To analyze a different coupon size, set
    ``self.L_x`` / ``self.L_y`` on the instance and call
    :meth:`generate_mesh` again.

    Parameters
    ----------
    porosity_field : PorosityField
        Source of nodal porosity values. Sampled by
        :meth:`generate_mesh` at every node coordinate.
    material : MaterialProperties
        Composite material; supplies ``total_thickness`` (``L_z``) and
        ``n_plies`` for the per-element ply id assignment.
    nx, ny, nz : int, optional
        Number of elements along each axis (defaults
        ``nx=50``, ``ny=20``, ``nz=24``). Each must be a positive
        integer not greater than ``_MAX_ELEMENTS_PER_AXIS`` (10 000), and
        ``nx * ny * nz`` may not exceed ``_MAX_TOTAL_ELEMENTS`` (1 000 000;
        an FE solve needs about 30 kB per element before factorization).
    ply_angles : list of float or {'QI', 'UD'}, optional
        Per-ply orientation in degrees, OR a string sentinel — ``'QI'``
        (default, expands to the 8-ply quasi-isotropic baseline
        ``[0, 90, 45, -45]_s``) or ``'UD'`` (all-zero unidirectional).
        Explicit lists shorter than ``n_plies`` are tiled. Passing
        ``None`` is deprecated and is currently resolved to all-zero
        plies (the historical default) with a
        :class:`DeprecationWarning` (#44 item 2); this back-compat path
        will be removed in a future major version.

    Attributes
    ----------
    porosity_field : PorosityField
        Stored input.
    material : MaterialProperties
        Stored input.
    nx, ny, nz : int
        Element counts along each axis.
    L_x, L_y, L_z : float
        Coupon dimensions in mm. ``L_x = 50.0`` and ``L_y = 20.0`` are
        defaults; ``L_z = material.total_thickness``.
    nodes : np.ndarray
        Shape ``(n_nodes, 3)`` float array of node coordinates (mm),
        populated by :meth:`generate_mesh`.
    elements : np.ndarray
        Shape ``(n_elem, 8)`` int array of node indices per element
        (VTK hexahedron ordering).
    porosity : np.ndarray
        Shape ``(n_nodes,)`` nodal porosity values in ``[0, 1]``.
    stiffness_reduction : np.ndarray
        Shape ``(n_nodes,)`` complementary ``1 - Vp`` field.
    ply_ids : np.ndarray
        Shape ``(n_nodes,)`` per-node ply id (``0`` to ``n_plies - 1``).
        A node on a ply interface takes the ply above it; the top-surface
        nodes take the last ply.
    elem_ply_ids : np.ndarray
        Shape ``(n_elem,)`` id of the ply containing each element
        centroid. A centroid exactly on a ply interface takes the ply
        above it (exact integer arithmetic, no floating-point rounding).
    ply_angles : np.ndarray
        Shape ``(n_elem,)`` per-element ply orientation in degrees: the
        angle of ply ``elem_ply_ids``. :attr:`element_layup` gives the
        through-thickness sequence.
    void_elements : np.ndarray
        Int indices of elements whose centroid lies inside any discrete
        void.
    void_element_set : set of int
        Same content as ``void_elements`` for O(1) membership tests.
    n_nodes, n_elements, n_dof : int
        Read-only sizes.

    Examples
    --------
    A default-size coupon with the T800 ply and 2 % uniform porosity:

    >>> mat = MATERIALS['T800_epoxy']
    >>> field = PorosityField(mat, void_volume_fraction=0.02,
    ...                       distribution='uniform')
    >>> mesh = CompositeMesh(field, mat, nx=10, ny=4, nz=4)
    >>> mesh.L_x, mesh.L_y
    (50.0, 20.0)
    >>> mesh.n_elements
    160

    Override the in-plane coupon size to 80 mm x 25 mm:

    >>> mesh = CompositeMesh(field, mat, nx=8, ny=4, nz=4)
    >>> mesh.L_x = 80.0
    >>> mesh.L_y = 25.0
    >>> mesh.generate_mesh()

    Notes
    -----
    ``ply_angles`` defaults — ``'QI'`` is the standardised default across
    :class:`EmpiricalSolver`, :class:`CompositeMesh`, and :class:`FESolver`
    (#44 item 2). The string sentinels expand to canonical baselines
    (``'QI'`` -> ``[0, 90, 45, -45]_s``; ``'UD'`` -> all-zero plies);
    explicit lists pass through unchanged. Pass ``ply_angles='UD'`` to
    reproduce the pre-#44 behaviour of leaving every element at
    ``ply_angle = 0``.

    The element layers resolve the layup only when none of them spans
    plies of different angles, which ``nz`` a multiple of ``n_plies``
    guarantees. Otherwise each element layer keeps just the ply at its
    centroid and the FE solver analyzes a different laminate: on the
    production mesh (``nz = 12``) a 24-ply ``'QI'`` layup becomes
    ``[90, -45, 45, 0]`` repeated three times, which is no longer
    symmetric. :meth:`layup_discrepancies` lists what is lost, and
    :class:`FESolver` logs it as a warning.
    """

    # Cap mesh dimensions to prevent accidental memory blowup. A million-element
    # mesh is already ~100x what the GUI spinboxes allow; an order of magnitude
    # above that is almost certainly a typo or unit confusion.
    _MAX_ELEMENTS_PER_AXIS = 10_000
    # The per-axis cap alone still admits 10_000**3 elements. FE assembly
    # holds roughly 30 kB per element before the sparse factorization (B, C,
    # element stiffness and COO triplets), so a million elements is already
    # ~30 GB; refuse anything larger up front instead of failing with an
    # out-of-memory error mid-assembly (IMPROVEMENT_PLAN 1.7).
    _MAX_TOTAL_ELEMENTS = 1_000_000
    _FE_BYTES_PER_ELEMENT = 30_000

    def __init__(self, porosity_field: PorosityField, material: MaterialProperties,
                 nx: int = 50, ny: int = 20, nz: int = 24,
                 ply_angles: list[float] | str | None = 'QI'):
        for axis_name, value in (('nx', nx), ('ny', ny), ('nz', nz)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(
                    f"CompositeMesh.{axis_name} must be a positive integer "
                    f"(elements per axis), got {value!r}."
                )
            if value > self._MAX_ELEMENTS_PER_AXIS:
                raise ValueError(
                    f"CompositeMesh.{axis_name}={value} exceeds the "
                    f"{self._MAX_ELEMENTS_PER_AXIS} per-axis cap. "
                    f"Such a fine mesh would exhaust memory; "
                    f"reduce or split the analysis."
                )
        n_total = int(nx) * int(ny) * int(nz)
        if n_total > self._MAX_TOTAL_ELEMENTS:
            gb = n_total * self._FE_BYTES_PER_ELEMENT / 1e9
            raise ValueError(
                f"CompositeMesh {nx}x{ny}x{nz} has {n_total:,} elements, above "
                f"the {self._MAX_TOTAL_ELEMENTS:,}-element cap. An FE solve "
                f"would need roughly {gb:,.0f} GB before factorization; "
                f"reduce the resolution or split the analysis."
            )

        self.porosity_field = porosity_field
        self.material = material
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.L_x = 50.0
        self.L_y = 20.0
        self.L_z = material.total_thickness

        self.nodes = None
        self.elements = None
        self.porosity: np.ndarray | None = None
        self.stiffness_reduction = None
        self.ply_ids = None
        self.ply_angles = None  # Per-element ply orientation angles (degrees)
        self.void_elements: np.ndarray | None = None

        # Resolve the ply_angles sentinel (#44 item 2). ``None`` is the
        # deprecated path and emits a DeprecationWarning inside
        # ``_resolve_ply_angles``.
        self._input_ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='CompositeMesh.ply_angles')
        self.generate_mesh()

    def generate_mesh(self):
        x = np.linspace(0, self.L_x, self.nx + 1)
        y = np.linspace(0, self.L_y, self.ny + 1)
        z = np.linspace(0, self.L_z, self.nz + 1)

        # Vectorized node construction. Node ordering is z (outer) ->
        # y (middle) -> x (inner), i.e. node_id = k*(ny+1)*(nx+1)
        # + j*(nx+1) + i. ``indexing='ij'`` with axis order (z, y, x)
        # then C-order ravel reproduces that exact ordering, matching the
        # original triple-nested ``for zk: for yj: for xi`` loop.
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        self.nodes = np.column_stack(
            (xx.ravel(), yy.ravel(), zz.ravel()))

        # Sample porosity at all nodes
        self.porosity = self.porosity_field.local_porosity(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])
        self.stiffness_reduction = self.porosity_field.local_stiffness_reduction(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])

        # Vectorized hex element connectivity. Element ordering is
        # k (outer) -> j (middle) -> i (inner), matching the original
        # triple-nested loop. ``indexing='ij'`` with axis order
        # (k, j, i) then a C-order ravel of each corner offset array
        # reproduces that exact element ordering and per-element corner
        # ordering.
        nx1 = self.nx + 1
        layer = (self.ny + 1) * nx1
        k, j, i = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx),
            indexing='ij')
        n0 = (k * layer + j * nx1 + i).ravel()
        n1 = n0 + 1
        n2 = n0 + nx1 + 1
        n3 = n0 + nx1
        n4 = n0 + layer
        n5 = n4 + 1
        n6 = n4 + nx1 + 1
        n7 = n4 + nx1
        self.elements = np.column_stack((n0, n1, n2, n3, n4, n5, n6, n7))

        # Identify void elements: check if element centroid falls inside
        # any discrete void geometry (explicit inclusion modeling)
        elem_centers = np.mean(self.nodes[self.elements], axis=1)  # (n_elem, 3)
        void_mask = np.zeros(len(self.elements), dtype=bool)
        for void in self.porosity_field.discrete_voids:
            inside = void.contains(elem_centers[:, 0], elem_centers[:, 1], elem_centers[:, 2])
            void_mask |= inside
        self.void_elements = np.where(void_mask)[0]
        # Also create a set for O(1) lookup
        self.void_element_set = set(self.void_elements.tolist())

        # Ply ids in exact integer arithmetic, so a node or element centroid
        # on a ply interface is not rounded either way by floating-point
        # noise. Node layer k sits at k * n_plies / nz in ply units; each
        # element takes the ply containing its centroid, which for element
        # layer k is at (2k + 1) * n_plies / (2 nz).
        n_plies = self.material.n_plies
        self.ply_ids = np.repeat(
            _ply_at(np.arange(self.nz + 1) * n_plies, self.nz, n_plies),
            (self.ny + 1) * (self.nx + 1))
        self.elem_ply_ids = np.repeat(
            _ply_at((2 * np.arange(self.nz) + 1) * n_plies, 2 * self.nz, n_plies),
            self.ny * self.nx)

        if self._input_ply_angles is not None:
            angle_list = list(self._input_ply_angles)
            if len(angle_list) < n_plies:
                # Repeat to fill all plies
                angle_list = (angle_list * (n_plies // len(angle_list) + 1))[:n_plies]
            self._ply_layup = np.array(angle_list[:n_plies], dtype=float)
        else:
            # Default: all 0-degree plies
            self._ply_layup = np.zeros(n_plies, dtype=float)
        self.ply_angles = self._ply_layup[self.elem_ply_ids]
        self._layup_warning_logged = False

        logger.info("Mesh generated: %d nodes, %d elements",
                    len(self.nodes), len(self.elements))
        logger.info("  Domain: %.1f x %.1f x %.2f mm",
                    self.L_x, self.L_y, self.L_z)
        logger.info("  Void elements: %d", len(self.void_elements))

    @property
    def element_layup(self) -> np.ndarray:
        """Ply angle (degrees) of each element layer, bottom to top, shape ``(nz,)``.

        This is the laminate the FE solver analyzes. It equals the
        requested layup only when no element layer spans plies of
        different angles, e.g. when ``nz`` is a multiple of ``n_plies``;
        :meth:`layup_discrepancies` reports how it differs otherwise.
        """
        return np.asarray(self.ply_angles, dtype=float).reshape(self.nz, -1)[:, 0].copy()

    def layup_discrepancies(self) -> list[str]:
        """Ways the element layup misrepresents the requested laminate.

        Each element layer takes the angle of the ply at its centroid, so
        an element layer that spans several plies of different angles
        drops all but one of them. That happens whenever ``nz`` is not a
        multiple of ``n_plies`` (unless the merged plies share an angle).

        Returns
        -------
        list of str
            Empty when every element layer lies within plies of a single
            angle. Otherwise one sentence (no final period) per finding:
            element layers merge plies of different angles (with the
            change in each angle's thickness fraction); a symmetric
            requested layup is unsymmetric in the elements; a balanced
            requested layup is unbalanced in the elements.
        """
        n_plies = self.material.n_plies
        requested = _normalize_angles(self._ply_layup)
        issues: list[str] = []

        # Element layer k spans [k, k + 1] * n_plies / nz in ply units.
        k = np.arange(self.nz)
        first = k * n_plies // self.nz
        last = -(-(k + 1) * n_plies // self.nz) - 1
        if any(np.ptp(requested[a:b + 1]) > _ANGLE_TOL for a, b in zip(first, last, strict=True)):
            req_frac = _angle_fractions(requested)
            elem_frac = _angle_fractions(np.asarray(self.ply_angles, dtype=float))
            msg = (f"{self.nz} element layers for {n_plies} plies merge plies "
                   f"of different angles, so each layer keeps only the ply at "
                   f"its centroid")
            if req_frac != elem_frac:
                changes = ", ".join(
                    f"{a:g} deg {100 * req_frac.get(a, 0.0):.0f}% -> "
                    f"{100 * elem_frac.get(a, 0.0):.0f}%"
                    for a in sorted(set(req_frac) | set(elem_frac)))
                msg += f" and the angle thickness fractions change ({changes})"
            issues.append(msg)

        elements = _normalize_angles(np.asarray(self.ply_angles, dtype=float))
        elements = elements.reshape(self.nz, -1)
        if (np.allclose(requested, requested[::-1], atol=_ANGLE_TOL)
                and not np.allclose(elements, elements[::-1], atol=_ANGLE_TOL)):
            issues.append("The requested layup is symmetric but the element "
                          "layup is not")
        if (_is_balanced(_angle_fractions(requested))
                and not _is_balanced(_angle_fractions(elements))):
            issues.append("The requested layup is balanced but the element "
                          "layup is not")
        return issues

    def _log_layup_warning(self) -> None:
        """Log :meth:`layup_discrepancies` as a warning, once per generated mesh.

        Called by :class:`FESolver`: only the FE path uses the element
        layup, so empirical-only runs on the same mesh stay quiet.
        """
        if self._layup_warning_logged:
            return
        self._layup_warning_logged = True
        issues = self.layup_discrepancies()
        if issues:
            layup = ", ".join(f"{a:g}" for a in self.element_layup)
            logger.warning(
                "The FE mesh does not represent the requested %d-ply layup. "
                "%s. FE stiffness, stresses and knockdown are computed for "
                "the element layup [%s] (bottom to top). Use nz = %d, or a "
                "multiple, to give every ply its own element layer.",
                self.material.n_plies, ". ".join(issues), layup,
                self.material.n_plies)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def domain_size(self) -> tuple[float, float, float]:
        return (self.L_x, self.L_y, self.L_z)

    def mid_y_section_indices(self) -> np.ndarray:
        """Node indices of the mid-y (``j = ny // 2``) cross-section.

        Shape ``(nz + 1, nx + 1)``: row ``k`` is one through-thickness node
        layer, column ``i`` one x position.
        """
        nx1, ny1 = self.nx + 1, self.ny + 1
        k = np.arange(self.nz + 1)[:, None]
        i = np.arange(nx1)[None, :]
        return k * ny1 * nx1 + (self.ny // 2) * nx1 + i

    def mid_y_element_indices(self) -> np.ndarray:
        """Element indices of the mid-y element row, shape ``(nz, nx)``."""
        k = np.arange(self.nz)[:, None]
        i = np.arange(self.nx)[None, :]
        return k * self.ny * self.nx + (self.ny // 2) * self.nx + i

    def nodes_on_face(self, face: MeshFace) -> np.ndarray:
        """Return node indices on the specified face.

        Parameters
        ----------
        face : str
            One of 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.

        Returns
        -------
        np.ndarray
            1-D array of node indices on that face.
        """
        tol = 1e-8
        coords = self.nodes
        match face:
            case 'x_min':
                return np.where(np.abs(coords[:, 0] - coords[:, 0].min()) < tol)[0]
            case 'x_max':
                return np.where(np.abs(coords[:, 0] - coords[:, 0].max()) < tol)[0]
            case 'y_min':
                return np.where(np.abs(coords[:, 1] - coords[:, 1].min()) < tol)[0]
            case 'y_max':
                return np.where(np.abs(coords[:, 1] - coords[:, 1].max()) < tol)[0]
            case 'z_min':
                return np.where(np.abs(coords[:, 2] - coords[:, 2].min()) < tol)[0]
            case 'z_max':
                return np.where(np.abs(coords[:, 2] - coords[:, 2].max()) < tol)[0]
            case _:
                raise ValueError(f"Unknown face '{face}'. Use x_min/x_max/y_min/y_max/z_min/z_max.")

    def find_nodes_near(self, x: float | None = None,
                        y: float | None = None,
                        z: float | None = None,
                        tol: float | None = None) -> np.ndarray:
        """Return node indices within ``tol`` of the specified target coords.

        Any of ``x``, ``y``, ``z`` may be ``None``, in which case that axis
        is not used in the distance computation (i.e. the search becomes a
        line/plane match rather than a point match). Distances are computed
        with ``np.linalg.norm`` on the subset of axes that were specified.

        Parameters
        ----------
        x, y, z : float or None
            Target coordinate per axis. Pass ``None`` to ignore an axis.
        tol : float or None
            Distance tolerance. If ``None``, defaults to half of a typical
            element edge length (``0.5 * min(L_x/nx, L_y/ny, L_z/nz)``).

        Returns
        -------
        np.ndarray
            Sorted 1-D array of node indices whose distance to the target
            (restricted to the specified axes) is ``<= tol``.

        Notes
        -----
        Used by ILSS short-beam BCs to locate midspan-top loading nodes
        even when ``Lx / 2`` does not coincide with a mesh node.
        """
        if x is None and y is None and z is None:
            raise ValueError(
                "find_nodes_near: at least one of x/y/z must be specified."
            )
        if tol is None:
            dxs = []
            if self.nx > 0:
                dxs.append(self.L_x / self.nx)
            if self.ny > 0:
                dxs.append(self.L_y / self.ny)
            if self.nz > 0:
                dxs.append(self.L_z / self.nz)
            tol = 0.5 * min(dxs)

        coords = self.nodes
        targets = []
        cols = []
        if x is not None:
            targets.append(float(x))
            cols.append(0)
        if y is not None:
            targets.append(float(y))
            cols.append(1)
        if z is not None:
            targets.append(float(z))
            cols.append(2)

        diffs = coords[:, cols] - np.asarray(targets, dtype=float)
        dist = np.linalg.norm(diffs, axis=1)
        return np.where(dist <= tol)[0]

    def __repr__(self) -> str:
        return (f"CompositeMesh(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
                f"domain={self.L_x:.1f}x{self.L_y:.1f}x{self.L_z:.2f}mm, "
                f"void_elements={len(self.void_elements)})")


def check_mesh_quality(mesh: CompositeMesh, verbose: bool = False) -> dict:
    """Check mesh quality: element aspect ratios and Jacobian determinants.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh to check.
    verbose : bool
        Print detailed quality report.

    Returns
    -------
    dict
        Quality metrics: min/max aspect ratio, min Jacobian determinant over
        the 8 Gauss points, number of inverted elements (non-positive
        determinant at any Gauss point, matching what assembly rejects),
        number of highly distorted elements.

    Raises
    ------
    Warning messages are printed for inverted or highly distorted elements.
    """
    # Lazy import — :class:`Hex8Element` lives in :mod:`porosity_fe.fe.element`
    # which is one layer above us in the dependency graph (the FE subpackage
    # imports from :mod:`porosity_fe.mesh`, not the other way around). Importing
    # here keeps :mod:`porosity_fe.mesh` cycle-free at module load time.
    from .fe.element import Hex8Element

    n_elem = mesh.n_elements
    coords = mesh.nodes[mesh.elements]  # (n_elem, 8, 3)

    # Aspect ratio: ratio of max edge length to min edge length over the
    # 12 edges of each hexahedron.
    edge_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3])  # bottom, top,
    edge_b = np.array([1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7])  # vertical
    edge_lengths = np.linalg.norm(coords[:, edge_a] - coords[:, edge_b], axis=2)
    min_len = edge_lengths.min(axis=1)
    max_len = edge_lengths.max(axis=1)
    with np.errstate(divide='ignore'):
        aspect_ratios = np.where(min_len > 1e-15, max_len / min_len, np.inf)

    # Jacobian at the 8 Gauss points, the same points where assembly
    # rejects a non-positive determinant; checking only the element center
    # missed elements that are inverted near a corner (IMPROVEMENT_PLAN 2.8).
    points, _ = gauss_points_hex(order=2)
    dN = np.stack([Hex8Element.shape_derivatives(*p) for p in points])  # (G, 3, 8)
    detJ = np.linalg.det(np.einsum('gij,ejk->egik', dN, coords))         # (E, G)
    min_detJ_per_elem = detJ.min(axis=1)

    n_inverted = int(np.sum(min_detJ_per_elem <= 0))
    n_distorted = int(np.sum(aspect_ratios > 20.0))

    result = {
        'min_aspect_ratio': float(np.min(aspect_ratios)),
        'max_aspect_ratio': float(np.max(aspect_ratios)),
        'mean_aspect_ratio': float(np.mean(aspect_ratios)),
        'min_jacobian_det': float(np.min(min_detJ_per_elem)),
        'n_inverted': n_inverted,
        'n_distorted': n_distorted,
        'n_elements': n_elem,
    }

    if verbose:
        logger.info("  Mesh quality: %d elements", n_elem)
        logger.info(
            "    Aspect ratio: min=%.2f, max=%.2f, mean=%.2f",
            result['min_aspect_ratio'],
            result['max_aspect_ratio'],
            result['mean_aspect_ratio'],
        )
        logger.info("    Min Jacobian det: %.6e", result['min_jacobian_det'])
        if n_inverted > 0:
            logger.warning(
                "    WARNING: %d inverted elements (negative Jacobian)!",
                n_inverted,
            )
        if n_distorted > 0:
            logger.warning(
                "    WARNING: %d highly distorted elements (aspect ratio > 20)!",
                n_distorted,
            )

    if n_inverted > 0:
        warnings.warn(
            f"Mesh has {n_inverted} inverted elements (negative Jacobian determinant).",
            stacklevel=2,
        )
    if n_distorted > 0:
        warnings.warn(
            f"Mesh has {n_distorted} highly distorted elements (aspect ratio > 20).",
            stacklevel=2,
        )

    return result

