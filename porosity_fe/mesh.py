"""Structured hex mesh and quality checks."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._ply_angles import _resolve_ply_angles
from ._types import MeshFace
from .delamination import Delamination
from .fracture_specimens import (
    _D5528_CRACK_LENGTH,
    _D5528_LENGTH,
    _D5528_WIDTH,
    _D6671_CRACK_LENGTH,
    _D6671_MODE_RATIO,
    _D6671_SPAN,
    _D6671_WIDTH,
    _D7905_CRACK_LENGTH,
    _D7905_SPAN,
    _D7905_WIDTH,
    FractureSpecimenGeometry,
    _build_specimen,
    _finite_positive,
    _SpecimenLayout,
    mmb_lever_length,
)
from .gauss import gauss_points_hex
from .materials import MaterialProperties
from .porosity_field import PorosityField

logger = logging.getLogger("porosity_fe_analysis")

# Two normalized ply angles closer than this (degrees) are the same angle.
_ANGLE_TOL = 1e-6

# Largest element aspect ratio check_mesh_quality accepts without a warning.
_MAX_ASPECT_RATIO = 20.0

# ASTM D2344/D2344M short-beam-strength specimen, as multiples of the
# laminate thickness h: support span 4 h, length 6 h, width 2 h. The loading
# nose is a 6.0 mm diameter cylinder (radius 3.0 mm), the supports 3.0 mm.
_D2344_SPAN_TO_THICKNESS = 4.0
_D2344_LENGTH_TO_THICKNESS = 6.0
_D2344_WIDTH_TO_THICKNESS = 2.0
_D2344_NOSE_RADIUS = 3.0
# Default in-plane element length of CompositeMesh.for_ilss, h / 4: the
# quarter-span shear and the knockdown are within 0.2 % and 1.2 % (of the
# stiffness loss) of an h / 8 mesh there (theory page, "Short-beam shear").
_D2344_ELEMENTS_PER_THICKNESS = 4


def _hertz_contact_width(material: MaterialProperties, *,
                         nose_radius: float = _D2344_NOSE_RADIUS) -> float:
    """Contact width (mm) of a short-beam loading nose at the pristine strength.

    Hertz line contact of a rigid cylinder of radius ``R`` on an elastic
    half-space: half-width ``a = sqrt(4 q R / (pi E*))`` for a load ``q``
    per unit length, with ``E* = E33 / (1 - nu23^2)`` the through-thickness
    plane-strain modulus of the ply. ``q`` is the load per unit width at
    which the pristine laminate reaches its short-beam strength,
    ``q = P / b = (4/3) tau_ilss h`` (ASTM D2344: ``F_sbs = 0.75 P / (b h)``).
    Returns ``2 a``: about 0.77 mm (0.18 h) for the 24-ply T800/epoxy
    preset. An estimate: a laminate is stiffer in plane than ``E33``, which
    narrows the contact, and damage under the nose near failure widens it.
    """
    h = float(material.total_thickness)
    q = 4.0 / 3.0 * float(material.tau_ilss) * h
    e_star = float(material.E33) / (1.0 - float(material.nu23) ** 2)
    return 4.0 * math.sqrt(q * float(nose_radius) / (math.pi * e_star))


@dataclass(frozen=True)
class FlexureGeometry:
    """Support and load positions of a flexure test on a mesh (mm).

    The supports are line supports (rollers) along ``y`` on the bottom face
    (``z_min``); the loads act along ``z`` on the top face (``z_max``),
    spread evenly across the width. The three-point test (ASTM D790, D7264
    procedure A) loads mid-span; the four-point test (D7264 procedure B)
    loads two lines ``load_span`` apart, centred on mid-span.
    :meth:`CompositeMesh.for_flexure` builds a mesh with these lines on node
    lines and stores the geometry as :attr:`CompositeMesh.flexure_geometry`;
    :meth:`~porosity_fe.BoundaryHandler.flexure_bcs` and the
    ``'flexure_3pt'`` / ``'flexure_4pt'`` loading modes of
    :class:`~porosity_fe.FESolver` read it from there.

    Attributes
    ----------
    support_x : tuple of float
        ``x`` of the left and right support lines (mm). The supports must
        lie on node lines of the mesh.
    load_span : float
        Distance between the two load lines of the four-point test (mm),
        ``0 < load_span < span``. D7264 procedure B uses half the support
        span, so the loads sit at the quarter points. Not used by the
        three-point test.
    load_width : float
        Width along ``x`` of the band each load is spread over as a uniform
        pressure (mm). ``0.0`` (default) is a line load; a band about as
        wide as the contact under a loading nose removes the stress
        singularity under the load line. Each band must stay between the
        supports and, in the four-point test, clear of the other band.

    Raises
    ------
    ValueError
        If a value is not finite, the supports are not in increasing order,
        or the load span or width does not fit between the supports.
    """

    support_x: tuple[float, float]
    load_span: float
    load_width: float = 0.0

    def __post_init__(self) -> None:
        try:
            x0, x1 = (float(v) for v in self.support_x)
            load_span, load_width = float(self.load_span), float(self.load_width)
        except (TypeError, ValueError):
            raise ValueError(
                "FlexureGeometry needs two support positions and numeric "
                f"load_span and load_width, got support_x={self.support_x!r}, "
                f"load_span={self.load_span!r}, load_width={self.load_width!r}."
            ) from None
        if not all(math.isfinite(v) for v in (x0, x1, load_span, load_width)):
            raise ValueError("FlexureGeometry values must be finite (mm).")
        if not x1 > x0:
            raise ValueError(
                f"support_x must be increasing (left, right), got {(x0, x1)}.")
        span = x1 - x0
        if not 0.0 < load_span < span:
            raise ValueError(
                f"load_span must lie in (0, span = {span:g}) mm, got {load_span:g}.")
        if not 0.0 <= load_width < min(load_span, span - load_span):
            raise ValueError(
                f"load_width must be >= 0 and leave the load bands clear of "
                f"each other and of the supports (below "
                f"{min(load_span, span - load_span):g} mm here), got "
                f"{load_width:g}.")
        object.__setattr__(self, 'support_x', (x0, x1))
        object.__setattr__(self, 'load_span', load_span)
        object.__setattr__(self, 'load_width', load_width)

    @property
    def span(self) -> float:
        """Support span ``L`` (mm)."""
        return self.support_x[1] - self.support_x[0]

    @property
    def midspan(self) -> float:
        """``x`` of mid-span (mm)."""
        return 0.5 * (self.support_x[0] + self.support_x[1])

    def load_x(self, procedure: Literal['3pt', '4pt']) -> tuple[float, ...]:
        """``x`` of the load line(s): mid-span (``'3pt'``) or the two lines
        ``load_span`` apart (``'4pt'``)."""
        if procedure == '3pt':
            return (self.midspan,)
        if procedure == '4pt':
            half = 0.5 * self.load_span
            return (self.midspan - half, self.midspan + half)
        raise ValueError(f"procedure must be '3pt' or '4pt', got {procedure!r}.")

    def shear_span(self, procedure: Literal['3pt', '4pt']) -> float:
        """Distance ``a`` from each support to the nearer load line (mm):
        ``L / 2`` (``'3pt'``) or ``(L - load_span) / 2`` (``'4pt'``)."""
        return self.load_x(procedure)[0] - self.support_x[0]


def _ply_at(num: np.ndarray, den: int, n_plies: int) -> np.ndarray:
    """Ply index at the through-thickness positions ``num / den`` (ply units).

    Evaluated in integer arithmetic, so a position exactly on a ply
    interface always takes the ply above it instead of whichever side
    floating-point rounding happens to land on. The top surface maps to
    the last ply.
    """
    return np.clip(np.asarray(num) // den, 0, n_plies - 1)


def _as_delaminations(value: object) -> tuple[Delamination, ...]:
    """``delaminations=`` as a tuple of :class:`Delamination` (``None``: empty)."""
    if value is None:
        return ()
    if isinstance(value, Delamination):
        return (value,)
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise ValueError(
            f"delaminations must be a sequence of Delamination, got {value!r}.") from None
    bad = [d for d in items if not isinstance(d, Delamination)]
    if bad:
        raise ValueError(
            f"delaminations must be Delamination instances, got {bad[0]!r}.")
    return items


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
    delaminations : sequence of Delamination, optional
        Keyword-only explicit delaminations (interlaminar cracks), default
        none. For every ply interface that has at least one, the node plane
        of the interface is duplicated once: the copies are appended after
        the structured nodes (so the structured node ids do not change) and
        the element layer above the interface is rewired to them. The
        solver ties each copy to its original outside the delaminations
        (:meth:`bonded_ties`); inside, the two crack faces are free. Needs
        ``nz`` a multiple of ``n_plies``, so that every interface is a node
        layer. Checked against the plate in :meth:`generate_mesh`, so a
        builder that resizes the plate (:meth:`for_flexure`) checks the
        final size.

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
        populated by :meth:`generate_mesh`. The first
        :attr:`n_structured_nodes` form the structured grid (node id
        ``k (ny+1)(nx+1) + j (nx+1) + i``); the duplicated interface nodes
        of the delaminations follow.
    elements : np.ndarray
        Shape ``(n_elem, 8)`` int array of node indices per element
        (VTK hexahedron ordering).
    porosity : np.ndarray
        Shape ``(n_nodes,)`` nodal porosity values in ``[0, 1]``; a
        duplicated node carries its original's value.
    stiffness_reduction : np.ndarray
        Shape ``(n_nodes,)`` complementary ``1 - Vp`` field.
    ply_ids : np.ndarray
        Shape ``(n_nodes,)`` per-node ply id (``0`` to ``n_plies - 1``).
        A node on a ply interface takes the ply above it; the top-surface
        nodes take the last ply.
    delaminations : tuple of Delamination
        The delaminations. Replacing them with others at the same
        interfaces changes only the bonded ties, not the mesh, so a crack
        can grow without reassembly; an interface without a duplicated
        node plane needs :meth:`generate_mesh` again.
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
    flexure_geometry : FlexureGeometry or None
        Support and load lines of a flexure test, set by
        :meth:`for_flexure` (and :meth:`for_ilss`); ``None`` otherwise, and
        the flexure loading modes then put the supports at the two bottom
        end edges.
    ilss_geometry : FlexureGeometry or None
        Supports and loading-nose band of the ASTM D2344 short-beam test,
        set by :meth:`for_ilss`; the ``'ilss'`` loading mode of
        :class:`~porosity_fe.FESolver` needs it and raises without it. Set
        it by hand (supports on node lines) to run the short-beam test on
        another mesh.
    fracture_geometry : FractureSpecimenGeometry or None
        Loads, supports and insert of a delamination fracture specimen, set
        by :meth:`for_dcb`, :meth:`for_enf` and :meth:`for_mmb`; the
        ``'dcb'``, ``'enf'`` and ``'mmb'`` loading modes of
        :class:`~porosity_fe.FESolver` need it and raise without it.
    x_grid : np.ndarray or None
        ``x`` (mm) of the ``nx + 1`` node lines, increasing from 0 to
        ``L_x``; ``None`` (default) spaces them evenly. The fracture-specimen
        builders set it for a mesh graded towards the crack front (plan
        decision D5). :meth:`generate_mesh` raises when it does not match
        ``nx`` and ``L_x``: set it back to ``None`` before changing either.
    n_nodes, n_elements, n_dof, n_structured_nodes : int
        Read-only sizes, duplicated interface nodes included in the first
        three; :attr:`n_structured_nodes` is ``(nx+1)(ny+1)(nz+1)``.

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
    centroid and the FE solver analyzes a different laminate: with
    ``nz = 12`` a 24-ply ``'QI'`` layup becomes ``[90, -45, 45, 0]``
    repeated three times, which is no longer symmetric. The production
    resolution (:func:`~porosity_fe.build_empirical_pipeline`, the app)
    uses ``nz = n_plies``. :meth:`layup_discrepancies` lists what is lost,
    and :class:`FESolver` logs it as a warning.
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
                 ply_angles: list[float] | str | None = 'QI', *,
                 delaminations: Iterable[Delamination] | Delamination | None = None):
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
        self.flexure_geometry: FlexureGeometry | None = None
        self.ilss_geometry: FlexureGeometry | None = None
        self.fracture_geometry: FractureSpecimenGeometry | None = None
        self.x_grid: np.ndarray | None = None
        self.delaminations: tuple[Delamination, ...] = _as_delaminations(delaminations)
        # Duplicated interface node pairs (see _duplicate_interfaces): the
        # original node of pair p is _pair_original[p], its duplicate is
        # node n_structured_nodes + p, at interface _pair_interface[p].
        self._n_structured_nodes = 0
        self._pair_original = np.zeros(0, dtype=np.intp)
        self._pair_interface = np.zeros(0, dtype=np.intp)

        # Resolve the ply_angles sentinel (#44 item 2). ``None`` is the
        # deprecated path and emits a DeprecationWarning inside
        # ``_resolve_ply_angles``.
        self._input_ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='CompositeMesh.ply_angles')
        self.generate_mesh()

    @classmethod
    def for_flexure(cls, porosity_field: PorosityField,
                    material: MaterialProperties, *,
                    span_to_thickness: float = 32.0,
                    span: float | None = None,
                    width: float = 13.0,
                    overhang: float | None = None,
                    load_span: float | None = None,
                    load_width: float = 0.0,
                    nz: int | None = None,
                    ny: int | None = None,
                    element_length: float | None = None,
                    nx: int | None = None,
                    ply_angles: list[float] | str | None = 'QI',
                    delaminations: Iterable[Delamination] | Delamination | None = None,
                    ) -> CompositeMesh:
        """Mesh of a flexure specimen (ASTM D7264 / D790 proportions).

        Sets the coupon length and width for a support span ``L`` of
        ``span_to_thickness`` laminate thicknesses, chooses ``nx`` so that
        the support lines, mid-span and (when possible) the four-point load
        lines fall on node lines, and records them as
        :attr:`flexure_geometry` for the ``'flexure_3pt'`` and
        ``'flexure_4pt'`` loading modes of :class:`~porosity_fe.FESolver`.
        One mesh serves both procedures.

        ASTM D7264 (polymer-matrix composites) uses a standard
        span-to-thickness ratio of 32:1 (16, 20, 40 and 60 are alternatives),
        a 13 mm wide specimen about 20 % longer than the span, three-point
        loading (procedure A) or four-point loading with the load span half
        the support span (procedure B). ASTM D790 (three-point, plastics and
        composites) uses 16:1 by default and 32:1 to 60:1 for high-modulus
        composites, with at least 10 % of the span overhanging each support.
        At 32:1 transverse shear adds only 1 to 4 % to the deflection of a
        carbon-fibre laminate; at 16:1 it is four times as much.

        Parameters
        ----------
        porosity_field : PorosityField
            Porosity field, sampled at the nodes (its discrete voids keep
            their absolute coordinates).
        material : MaterialProperties
            Ply material; the laminate thickness is
            ``material.total_thickness``.
        span_to_thickness : float
            Support span over laminate thickness, default 32 (D7264
            standard). Ignored when ``span`` is given.
        span : float, optional
            Support span in mm; overrides ``span_to_thickness``.
        width : float
            Specimen width ``L_y`` in mm, default 13 (D7264 standard; D790
            uses 12.7).
        overhang : float, optional
            Length beyond each support in mm, default 10 % of the span. It is
            rounded up to whole elements, so the specimen is at least as long
            as requested.
        load_span : float, optional
            Distance between the four-point load lines in mm, default half
            the span (D7264 procedure B: loads at the quarter points).
        load_width : float
            Width of the band each load is spread over in mm, default 0 (a
            line load); see :class:`FlexureGeometry`.
        nz : int, optional
            Elements through the thickness, default ``material.n_plies`` so
            every ply has its own element layer.
        ny : int, optional
            Elements across the width, default the smallest even count with
            elements no wider than ``element_length``.
        element_length : float, optional
            Largest element length along ``x`` and ``y`` in mm. The default,
            ``min(h, 20 h / nz)`` with ``h`` the laminate thickness, keeps
            the element aspect ratio at or below 20 (no
            :func:`check_mesh_quality` warning) and the element length at or
            below ``h``, where the default ``'hex8i'`` element is free of
            shear locking. The plain ``'hex8'`` element needs a much finer
            mesh in bending. Ignored when ``nx`` is given.
        nx : int, optional
            Elements along the length instead of ``element_length``; even,
            so mid-span is a node line. The span gets the most elements that
            leave each overhang at least ``overhang`` long (two fewer when
            that puts the four-point load lines on node lines), and the
            overhangs take the rest, so they can come out longer than
            requested. The span itself is exact at every ``nx``, which is
            what :func:`~porosity_fe.refinement_study` refines.
        ply_angles : list of float or {'QI', 'UD'}, optional
            Layup, as for :class:`CompositeMesh`.
        delaminations : sequence of Delamination, optional
            Delaminations, as for :class:`CompositeMesh`, in the specimen's
            coordinates (``x = 0`` at its left end); checked against the
            final specimen size.

        Returns
        -------
        CompositeMesh
            Mesh with ``L_x = span + 2 * overhang``, ``L_y = width`` and
            :attr:`flexure_geometry` set.

        Raises
        ------
        ValueError
            If a length or ratio is not positive and finite, the load
            span or width does not fit (see :class:`FlexureGeometry`), or
            ``nx`` is not an even integer that leaves at least two elements
            between the supports.

        Examples
        --------
        A D7264 specimen of the 24-ply T800 QI laminate (span 140.5 mm,
        48 x 4 x 24 elements):

        >>> mat = MATERIALS['T800_epoxy']
        >>> field = PorosityField(mat, 0.02)
        >>> mesh = CompositeMesh.for_flexure(field, mat)
        >>> mesh.nx, mesh.ny, mesh.nz
        (48, 4, 24)
        >>> round(mesh.flexure_geometry.span / mat.total_thickness, 6)
        32.0
        """
        h = float(material.total_thickness)

        def positive(name: str, value: float | None) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
                    or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"CompositeMesh.for_flexure: {name} must be a positive "
                    f"finite number, got {value!r}.")
            return float(value)

        ratio = positive('span_to_thickness', span_to_thickness)
        L = positive('span', span)
        if L is None:
            assert ratio is not None
            L = ratio * h
        width_f = positive('width', width)
        assert width_f is not None
        if overhang is None:
            overhang_f = 0.1 * L
        elif isinstance(overhang, bool) or not isinstance(overhang, (int, float, np.number)) \
                or not math.isfinite(float(overhang)) or float(overhang) < 0.0:
            raise ValueError(
                f"CompositeMesh.for_flexure: overhang must be a finite length "
                f">= 0 in mm, got {overhang!r}.")
        else:
            overhang_f = float(overhang)
        Ls = 0.5 * L if load_span is None else positive('load_span', load_span)
        assert Ls is not None
        # Validates load_span / load_width against the span up front.
        FlexureGeometry(support_x=(0.0, L), load_span=Ls, load_width=load_width)
        nz_f = int(material.n_plies) if nz is None else nz
        if not isinstance(nz_f, (int, np.integer)) or nz_f <= 0:
            raise ValueError(
                f"CompositeMesh.for_flexure: nz must be a positive integer, got {nz!r}.")
        shear_span = 0.5 * (L - Ls)

        def aligned(m: int) -> bool:
            k = shear_span * m / L
            return abs(k - round(k)) < 1e-6

        # Elements between the supports: even, so mid-span is a node line,
        # and if possible such that the four-point load lines are too.
        n_over: int | None = None
        if nx is not None:
            if isinstance(nx, bool) or not isinstance(nx, (int, np.integer)) \
                    or nx < 2 or nx % 2:
                raise ValueError(
                    f"CompositeMesh.for_flexure: nx must be an even integer "
                    f">= 2 (mid-span on a node line), got {nx!r}.")
            # The most span elements that leave each overhang long enough.
            m = 2 * math.floor(nx * L / (2.0 * (L + 2.0 * overhang_f)) + 1e-9)
            if m < 2:
                raise ValueError(
                    f"CompositeMesh.for_flexure: nx={nx} leaves fewer than two "
                    f"elements between the supports with {overhang_f:g} mm "
                    f"overhangs on a {L:g} mm span; use a larger nx.")
            if not aligned(m) and m >= 4 and aligned(m - 2):
                m -= 2
            n_over = (int(nx) - m) // 2
            dx_max = L / m
        else:
            dx_max_arg = positive('element_length', element_length)
            dx_max = min(h, _MAX_ASPECT_RATIO * h / nz_f) if dx_max_arg is None \
                else dx_max_arg
            m0 = 2 * math.ceil(L / (2.0 * dx_max) - 1e-9)
            m = next((c for c in range(m0, 2 * m0 + 1, 2) if aligned(c)), m0)
        if not aligned(m):
            logger.info(
                "CompositeMesh.for_flexure: the four-point load lines fall "
                "between node lines; their load is shared by the neighbouring "
                "nodes with consistent (linear shape-function) weights.")
        dx = L / m
        if n_over is None:
            n_over = math.ceil(overhang_f / dx - 1e-9) if overhang_f > 0.0 else 0
        nx_total = m + 2 * n_over
        if ny is None:
            ny = max(2, 2 * math.ceil(width_f / (2.0 * dx_max) - 1e-9))

        mesh = cls(porosity_field, material, nx=nx_total, ny=ny, nz=nz_f,
                   ply_angles=ply_angles)
        mesh.L_x = nx_total * dx
        mesh.L_y = width_f
        # Set after the default-size build, so they are checked against the
        # specimen; generate_mesh() duplicates their interfaces last.
        mesh.delaminations = _as_delaminations(delaminations)
        mesh.generate_mesh()
        x = np.linspace(0.0, mesh.L_x, nx_total + 1)
        mesh.flexure_geometry = FlexureGeometry(
            support_x=(float(x[n_over]), float(x[n_over + m])),
            load_span=Ls, load_width=load_width)
        return mesh

    @classmethod
    def for_ilss(cls, porosity_field: PorosityField,
                 material: MaterialProperties, *,
                 span_to_thickness: float = _D2344_SPAN_TO_THICKNESS,
                 length_to_thickness: float = _D2344_LENGTH_TO_THICKNESS,
                 width_to_thickness: float = _D2344_WIDTH_TO_THICKNESS,
                 load_width: float | None = None,
                 nose_radius: float = _D2344_NOSE_RADIUS,
                 nx: int | None = None,
                 ny: int | None = None,
                 nz: int | None = None,
                 element_length: float | None = None,
                 ply_angles: list[float] | str | None = 'QI',
                 delaminations: Iterable[Delamination] | Delamination | None = None,
                 ) -> CompositeMesh:
        """Mesh of an ASTM D2344 short-beam-shear (ILSS) specimen.

        The specimen the ``'ilss'`` loading mode of
        :class:`~porosity_fe.FESolver` needs: ASTM D2344/D2344M proportions
        in laminate thicknesses ``h`` (support span ``4 h``, length ``6 h``,
        width ``2 h``), roller supports on node lines, and the loading nose
        modelled as a uniform pressure over a contact band at mid-span.
        Built on :meth:`for_flexure`; the supports and band are recorded as
        :attr:`ilss_geometry` (and, as the same object,
        :attr:`flexure_geometry`, so ``'flexure_3pt'`` on this mesh runs the
        same test but reports a flexural modulus, which at this span is
        mostly shear correction).

        The nose (a 6.0 mm diameter cylinder in D2344) touches the specimen
        over a strip whose Hertz width grows with the load. The default band
        is that width at the pristine short-beam strength,
        ``P / b = (4/3) tau_ilss h``: ``2 a`` with
        ``a = sqrt(4 (P / b) R / (pi E*))``, ``R`` the nose radius and
        ``E* = E33 / (1 - nu23^2)`` (rigid nose; 0.77 mm, or 0.18 h, for
        the 24-ply T800/epoxy preset). It removes the line-load singularity
        from the problem, so refinement converges; on the default mesh it
        is narrower than an element and acts on the nearest node lines with
        consistent weights. The supports (3.0 mm cylinders) stay line
        supports: their stress concentration is left out of the interior
        quantities of the solve instead (see :meth:`FESolver.solve`).

        Parameters
        ----------
        porosity_field : PorosityField
            Porosity field, sampled at the nodes (discrete voids keep their
            absolute coordinates, with the specimen from ``x = 0``,
            ``y = 0``).
        material : MaterialProperties
            Ply material; the laminate thickness ``h`` is
            ``material.total_thickness``.
        span_to_thickness, length_to_thickness, width_to_thickness : float
            Support span, specimen length and width in thicknesses; default
            4, 6 and 2 (D2344). ISO 14130 uses 5, 10 and 5. The overhang
            ``(length - span) / 2`` is rounded up to whole elements.
        load_width : float, optional
            Width of the loading-nose band in mm; ``0.0`` is a line load.
            Default: the Hertz contact width above.
        nose_radius : float
            Loading-nose radius in mm for the default ``load_width``,
            default 3.0 (D2344; ISO 14130 uses 5.0).
        nx : int, optional
            Elements along the length. It must put both supports and
            mid-span on node lines: for the D2344 proportions a multiple of
            6. Default: from ``element_length``.
        ny : int, optional
            Elements across the width; default the smallest even count with
            elements no wider than the element length.
        nz : int, optional
            Elements through the thickness, default ``material.n_plies`` (one
            element layer per ply).
        element_length : float, optional
            Largest in-plane element length in mm, default ``h / 4`` (24 x 8
            elements in plane for the D2344 proportions). Ignored when
            ``nx`` is given.
        ply_angles : list of float or {'QI', 'UD'}, optional
            Layup, as for :class:`CompositeMesh`.
        delaminations : sequence of Delamination, optional
            Delaminations, as for :meth:`for_flexure`.

        Returns
        -------
        CompositeMesh
            Mesh of the specimen with :attr:`ilss_geometry` set.

        Raises
        ------
        ValueError
            If a ratio or length is not positive and finite, the specimen
            is shorter than the span, the band does not fit between the
            supports, or ``nx`` puts a support or mid-span between node
            lines.

        Examples
        --------
        The 24-ply T800 QI laminate (h = 4.392 mm): a 26.4 x 8.8 mm
        specimen on a 17.6 mm span, 24 x 8 x 24 elements:

        >>> mat = MATERIALS['T800_epoxy']
        >>> field = PorosityField(mat, 0.03)
        >>> mesh = CompositeMesh.for_ilss(field, mat)
        >>> mesh.nx, mesh.ny, mesh.nz
        (24, 8, 24)
        >>> round(mesh.ilss_geometry.span / mat.total_thickness, 6)
        4.0
        >>> round(mesh.ilss_geometry.load_width, 2)
        0.77
        """
        h = float(material.total_thickness)

        def positive(name: str, value: float) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
                    or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"CompositeMesh.for_ilss: {name} must be a positive finite "
                    f"number, got {value!r}.")
            return float(value)

        span = positive('span_to_thickness', span_to_thickness) * h
        length = positive('length_to_thickness', length_to_thickness) * h
        width = positive('width_to_thickness', width_to_thickness) * h
        if length < span * (1.0 - 1e-12):
            raise ValueError(
                f"CompositeMesh.for_ilss: the specimen ({length_to_thickness:g} h) "
                f"must be at least as long as the span ({span_to_thickness:g} h).")
        overhang = max(0.0, 0.5 * (length - span))
        if load_width is None:
            band = _hertz_contact_width(
                material, nose_radius=positive('nose_radius', nose_radius))
        elif isinstance(load_width, bool) \
                or not isinstance(load_width, (int, float, np.number)) \
                or not math.isfinite(float(load_width)) or float(load_width) < 0.0:
            raise ValueError(
                f"CompositeMesh.for_ilss: load_width must be a finite width "
                f">= 0 in mm, got {load_width!r}.")
        else:
            band = float(load_width)
        if nx is not None:
            if isinstance(nx, bool) or not isinstance(nx, (int, np.integer)) or nx <= 0:
                raise ValueError(
                    f"CompositeMesh.for_ilss: nx must be a positive integer, got {nx!r}.")

            def aligned(n: int) -> bool:
                # Supports at overhang and overhang + span, mid-span between:
                # each a whole number of elements of length / n.
                n_over, n_span = overhang * n / length, span * n / length
                return (abs(n_over - round(n_over)) <= 1e-6
                        and abs(n_span - round(n_span)) <= 1e-6
                        and round(n_span) % 2 == 0)

            if not aligned(int(nx)):
                unit = next((n for n in range(1, cls._MAX_ELEMENTS_PER_AXIS + 1)
                             if aligned(n)), None)
                hint = (f"use a multiple of {unit}" if unit is not None
                        else "pass element_length instead")
                raise ValueError(
                    f"CompositeMesh.for_ilss: nx={nx} puts a support or mid-span "
                    f"between node lines (a {length_to_thickness:g} h specimen on "
                    f"a {span_to_thickness:g} h span); {hint}.")
            dx_max: float | None = length / nx
        else:
            dx_max = h / _D2344_ELEMENTS_PER_THICKNESS if element_length is None \
                else positive('element_length', element_length)
        mesh = cls.for_flexure(
            porosity_field, material, span=span, width=width, overhang=overhang,
            load_width=band, nz=nz, ny=ny, element_length=dx_max,
            ply_angles=ply_angles, delaminations=delaminations)
        if nx is not None and mesh.nx != nx:   # pragma: no cover - guarded above
            raise RuntimeError(
                f"CompositeMesh.for_ilss: built nx={mesh.nx}, requested {nx}.")
        mesh.ilss_geometry = mesh.flexure_geometry
        return mesh

    # ------------------------------------------------------------------
    # Delamination fracture specimens (plan phase 3)
    # ------------------------------------------------------------------

    @classmethod
    def for_dcb(cls, porosity_field: PorosityField, material: MaterialProperties, *,
                crack_length: float = _D5528_CRACK_LENGTH,
                length: float = _D5528_LENGTH,
                width: float = _D5528_WIDTH,
                interface: int | None = None,
                element_length: float | None = None,
                max_element_length: float | None = None,
                nx: int | None = None,
                ny: int | None = None,
                nz: int | None = None,
                plane_strain: bool = False,
                ply_angles: list[float] | str | None = 'UD',
                ) -> CompositeMesh:
        """Mesh of a double-cantilever-beam (DCB, mode I) specimen.

        The ``'dcb'`` loading mode of :class:`~porosity_fe.FESolver` runs
        it: the two arms are pulled apart by equal and opposite hinge loads
        on the line ``x = 0`` (top and bottom faces), and a statically
        determinate set of six constraints at the far end holds the
        specimen without carrying load. The insert is a through-width
        :class:`~porosity_fe.Delamination` strip at ``interface`` from
        ``x = 0`` (the loaded end, open there) to ``crack_length``; the
        crack tip and the hinge line lie on node lines. The geometry is
        recorded as :attr:`fracture_geometry`.

        The defaults are the delamination plan's reading of ASTM D5528
        (unidirectional laminate, specimen at least 125 mm long, 20 to
        25 mm wide, insert about 50 mm from the load line). The ASTM text
        was not available: check them against the purchased standard. The
        hinge line is the specimen end here, so ``crack_length`` is measured
        from the load line as in the standard, and the end block or hinge
        beyond it is not modelled.

        Parameters
        ----------
        porosity_field : PorosityField
            Porosity field, sampled at the nodes.
        material : MaterialProperties
            Ply material; the specimen thickness is
            ``material.total_thickness``.
        crack_length : float
            Insert length ``a0`` from the load line (mm), default 50.
        length : float
            Specimen length from the load line (mm), default 125.
        width : float
            Specimen width ``b`` (mm), default 25.
        interface : int, optional
            Ply interface of the insert, default the mid-plane
            (``n_plies // 2 - 1``; an odd ply count needs it).
        element_length : float, optional
            Element length along ``x`` at the crack front (``Delta a`` of
            the virtual crack closure), default a quarter of the specimen
            thickness. Six elements of exactly this length lie on each side
            of the front; beyond them the length grows by 25 % per element
            up to ``max_element_length`` (a graded mesh, plan decision D5;
            :attr:`x_grid`). Ignored when ``nx`` is given.
        max_element_length : float, optional
            Largest element length along ``x`` and ``y`` (mm), default
            ``min(h, 20 h / nz)`` with ``h`` the specimen thickness (element
            aspect ratio at most 20, length at most ``h``, as
            :meth:`for_flexure`).
        nx : int, optional
            Elements along ``x`` on a uniform grid instead of the graded
            mesh; it must put the crack tip on a node line (for the
            defaults a multiple of 5).
        ny : int, optional
            Elements across the width, default the smallest even count with
            elements no wider than ``max_element_length``.
        nz : int, optional
            Elements through the thickness, a multiple of ``n_plies``,
            default ``n_plies``.
        plane_strain : bool
            Solve the specimen in plane strain, ``u_y = 0`` at every node,
            the counterpart of a two-dimensional plane-strain model (the
            fields of a unidirectional laminate are then the same across
            the width, so ``ny = 1`` is enough, at the cost of an
            element-quality warning for the wide elements). Default ``False``: the full three-dimensional
            specimen, whose energy release rate varies across the width.
        ply_angles : list of float or {'QI', 'UD'}, optional
            Layup, default ``'UD'`` (the standard tests unidirectional
            laminates).

        Returns
        -------
        CompositeMesh
            The specimen, ``L_x = length``, ``L_y = width``, with
            :attr:`fracture_geometry` and :attr:`delaminations` set.

        Raises
        ------
        ValueError
            If a length is not positive and finite, the insert is not
            shorter than the specimen, ``nz`` is not a multiple of
            ``n_plies``, or ``nx`` puts the crack tip between node lines.

        Examples
        --------
        The D5528 defaults on the T800 ply, with the IM7/8552 toughness of
        the theory page's table for illustration (N/mm):

        >>> mat = dataclasses.replace(MATERIALS['T800_epoxy'], G_Ic=0.212,
        ...                           G_IIc=0.774, bk_eta=2.1)
        >>> field = PorosityField(mat, 0.0)
        >>> mesh = CompositeMesh.for_dcb(field, mat)
        >>> mesh.fracture_geometry.crack_length, mesh.L_x, mesh.L_y
        (50.0, 125.0, 25.0)
        >>> result = FESolver(mesh, mat, field).solve(
        ...     'dcb', applied_load=50.0)                     # doctest: +SKIP
        >>> 50.0 * result.fracture_load_factor   # onset load, N  # doctest: +SKIP
        """
        a0 = _finite_positive('crack_length', crack_length)
        total = _finite_positive('length', length)
        if not a0 < total:
            raise ValueError(
                f"CompositeMesh.for_dcb: the insert ({a0:g} mm) must be shorter "
                f"than the specimen ({total:g} mm).")
        layout = _SpecimenLayout(
            test='dcb', points=(0.0, a0, total), front=1,
            width=_finite_positive('width', width),
            geometry_kwargs=dict(crack_length=a0, load_end=0.0))
        return _build_specimen(
            cls, porosity_field, material, layout, interface=interface,
            element_length=element_length, max_element_length=max_element_length,
            nx=nx, ny=ny, nz=nz, plane_strain=plane_strain, ply_angles=ply_angles)

    @classmethod
    def for_enf(cls, porosity_field: PorosityField, material: MaterialProperties, *,
                crack_length: float = _D7905_CRACK_LENGTH,
                span: float = _D7905_SPAN,
                width: float = _D7905_WIDTH,
                overhang: float = 0.0,
                interface: int | None = None,
                element_length: float | None = None,
                max_element_length: float | None = None,
                nx: int | None = None,
                ny: int | None = None,
                nz: int | None = None,
                plane_strain: bool = False,
                ply_angles: list[float] | str | None = 'UD',
                ) -> CompositeMesh:
        """Mesh of an end-notched-flexure (ENF, mode II) specimen.

        The ``'enf'`` loading mode of :class:`~porosity_fe.FESolver` runs
        it: three-point bending on roller supports ``span`` apart (bottom
        face), the load at mid-span on the top face, with crack-face
        contact. The insert is a through-width
        :class:`~porosity_fe.Delamination` strip at ``interface`` from the
        cracked end (``x = 0``) to ``crack_length`` beyond the support
        under it; the supports, the load line and the crack tip lie on node
        lines. The geometry is recorded as :attr:`fracture_geometry`.

        The defaults are the delamination plan's reading of ASTM D7905
        (support span 2L = 100 mm, ``a0`` = 30 mm from the support; the
        width is not in the plan). The ASTM text was not available: check
        them against the purchased standard. The standard specimen is longer
        than the span; its unloaded overhangs are left out by default
        (``overhang = 0``).

        Parameters
        ----------
        porosity_field, material
            As :meth:`for_dcb`.
        crack_length : float
            ``a0`` (mm) from the support under the cracked end to the crack
            tip, default 30; less than the span, and not at mid-span.
        span : float
            Support span ``2 L`` (mm), default 100.
        width : float
            Specimen width (mm), default 25.
        overhang : float
            Length beyond each support (mm), default 0. The insert runs
            through the overhang at the cracked end.
        interface, element_length, max_element_length, ny, nz, plane_strain, ply_angles
            As :meth:`for_dcb`.
        nx : int, optional
            Elements along ``x`` on a uniform grid instead of the graded
            mesh; it must put the supports, mid-span and the crack tip on
            node lines (for the defaults a multiple of 10).

        Returns
        -------
        CompositeMesh
            The specimen, ``L_x = span + 2 overhang``, with
            :attr:`fracture_geometry` and :attr:`delaminations` set.

        Raises
        ------
        ValueError
            As :meth:`for_dcb`, or if the crack tip is not inside the span
            or lies at mid-span.
        """
        a0 = _finite_positive('crack_length', crack_length)
        L2 = _finite_positive('span', span)
        over = _finite_positive('overhang', overhang, allow_zero=True)
        if not a0 < L2 or math.isclose(a0, 0.5 * L2, rel_tol=1e-9):
            raise ValueError(
                f"CompositeMesh.for_enf: the crack tip must lie inside the "
                f"{L2:g} mm span and not under the mid-span load, got "
                f"crack_length {a0:g} mm.")
        points = sorted({0.0, over, over + a0, over + 0.5 * L2, over + L2, 2.0 * over + L2})
        layout = _SpecimenLayout(
            test='enf', points=tuple(points), front=points.index(over + a0),
            width=_finite_positive('width', width),
            geometry_kwargs=dict(crack_length=a0, load_end=over, span=L2))
        return _build_specimen(
            cls, porosity_field, material, layout, interface=interface,
            element_length=element_length, max_element_length=max_element_length,
            nx=nx, ny=ny, nz=nz, plane_strain=plane_strain, ply_angles=ply_angles)

    @classmethod
    def for_mmb(cls, porosity_field: PorosityField, material: MaterialProperties, *,
                crack_length: float = _D6671_CRACK_LENGTH,
                span: float = _D6671_SPAN,
                lever_length: float | None = None,
                mode_ratio: float | None = None,
                width: float = _D6671_WIDTH,
                overhang: float = 0.0,
                interface: int | None = None,
                element_length: float | None = None,
                max_element_length: float | None = None,
                nx: int | None = None,
                ny: int | None = None,
                nz: int | None = None,
                plane_strain: bool = False,
                ply_angles: list[float] | str | None = 'UD',
                ) -> CompositeMesh:
        """Mesh of a mixed-mode-bending (MMB, mode I and II) specimen.

        The ``'mmb'`` loading mode of :class:`~porosity_fe.FESolver` runs
        it with the loads of the MMB lever
        (:meth:`FractureSpecimenGeometry.mmb_loads`): a hinge load on the
        upper arm at the cracked end (``x = 0``, top face), the saddle load
        at mid-span (top face), and roller supports under the lower arm's
        hinge (``x = 0``) and at the far end of the span (bottom face). The
        lever is statically determinate, so applying its forces is exact in
        a linear analysis. The insert is a through-width
        :class:`~porosity_fe.Delamination` strip at ``interface`` from
        ``x = 0`` to ``crack_length``; the hinge line, mid-span, the far
        support and the crack tip lie on node lines. The geometry is
        recorded as :attr:`fracture_geometry`.

        The defaults are the delamination plan's reading of ASTM D6671
        (half-span L = 50 mm, ``a0`` about 25 mm; the width is not in the
        plan) and ``G_II / G_T = 0.5``. The ASTM text was not available:
        check them against the purchased standard.

        Parameters
        ----------
        porosity_field, material
            As :meth:`for_dcb`.
        crack_length : float
            ``a0`` (mm) from the hinge line, default 25; less than the span,
            and not at mid-span.
        span : float
            Support span ``2 L`` (mm), default 100.
        lever_length : float, optional
            Lever length ``c`` (mm) from the saddle to the lever load.
            Default: :func:`~porosity_fe.mmb_lever_length` for
            ``mode_ratio``.
        mode_ratio : float, optional
            Target ``G_II / G_T`` of the beam theory of
            :func:`~porosity_fe.mmb_lever_length` when ``lever_length`` is not
            given, default 0.5. The FE mode mix differs from it by a few
            per cent; read :attr:`CrackFront.psi <porosity_fe.CrackFront.psi>`.
        width : float
            Specimen width (mm), default 25.
        overhang : float
            Length beyond the far support (mm), default 0.
        interface, element_length, max_element_length, ny, nz, plane_strain, ply_angles
            As :meth:`for_dcb`.
        nx : int, optional
            Elements along ``x`` on a uniform grid instead of the graded
            mesh; it must put the hinge line, mid-span, the far support and
            the crack tip on node lines (for the defaults a multiple of 4).

        Returns
        -------
        CompositeMesh
            The specimen, ``L_x = span + overhang``, with
            :attr:`fracture_geometry` and :attr:`delaminations` set.

        Raises
        ------
        ValueError
            As :meth:`for_enf`, or if both ``lever_length`` and
            ``mode_ratio`` are given.
        """
        a0 = _finite_positive('crack_length', crack_length)
        L2 = _finite_positive('span', span)
        over = _finite_positive('overhang', overhang, allow_zero=True)
        if not a0 < L2 or math.isclose(a0, 0.5 * L2, rel_tol=1e-9):
            raise ValueError(
                f"CompositeMesh.for_mmb: the crack tip must lie inside the "
                f"{L2:g} mm span and not under the saddle, got crack_length "
                f"{a0:g} mm.")
        if lever_length is not None and mode_ratio is not None:
            raise ValueError(
                "CompositeMesh.for_mmb: pass lever_length or mode_ratio, not both.")
        if lever_length is None:
            c = mmb_lever_length(
                material, _D6671_MODE_RATIO if mode_ratio is None else mode_ratio,
                span=L2, crack_length=a0)
        else:
            c = _finite_positive('lever_length', lever_length, allow_zero=True)
        points = sorted({0.0, a0, 0.5 * L2, L2, L2 + over})
        layout = _SpecimenLayout(
            test='mmb', points=tuple(points), front=points.index(a0),
            width=_finite_positive('width', width),
            geometry_kwargs=dict(crack_length=a0, load_end=0.0, span=L2,
                                 lever_length=c))
        return _build_specimen(
            cls, porosity_field, material, layout, interface=interface,
            element_length=element_length, max_element_length=max_element_length,
            nx=nx, ny=ny, nz=nz, plane_strain=plane_strain, ply_angles=ply_angles)

    def _x_node_lines(self) -> np.ndarray:
        """``x`` of the node lines: :attr:`x_grid`, or evenly spaced."""
        if self.x_grid is None:
            return np.linspace(0, self.L_x, self.nx + 1)
        x = np.asarray(self.x_grid, dtype=float)
        tol = 1e-9 * max(float(self.L_x), 1.0)
        if x.shape != (self.nx + 1,) or not np.all(np.isfinite(x)) \
                or abs(float(x[0])) > tol or abs(float(x[-1]) - float(self.L_x)) > tol \
                or not np.all(np.diff(x) > 0.0):
            raise ValueError(
                f"CompositeMesh.x_grid must hold nx + 1 = {self.nx + 1} increasing "
                f"node-line positions from 0 to L_x = {self.L_x:g} mm, got shape "
                f"{x.shape}; set mesh.x_grid = None for evenly spaced node lines.")
        return x

    def generate_mesh(self):
        x = self._x_node_lines()
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

        # Explicit delaminations: duplicate their interfaces' node planes
        # last, after every per-node array exists (and after a builder such
        # as for_flexure has set the final plate size).
        self._duplicate_interfaces()

        logger.info("Mesh generated: %d nodes, %d elements",
                    len(self.nodes), len(self.elements))
        logger.info("  Domain: %.1f x %.1f x %.2f mm",
                    self.L_x, self.L_y, self.L_z)
        logger.info("  Void elements: %d", len(self.void_elements))

    # ------------------------------------------------------------------
    # Explicit delaminations: duplicated interface nodes and bonded ties
    # ------------------------------------------------------------------

    def _plate(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the structured grid (mm)."""
        xy = np.asarray(self.nodes, dtype=float)[:self.n_structured_nodes, :2]
        return (float(xy[:, 0].min()), float(xy[:, 0].max()),
                float(xy[:, 1].min()), float(xy[:, 1].max()))

    def _tolerance(self) -> float:
        """Distance (mm) within which a node lies on a delamination boundary."""
        return 1e-9 * max(float(self.L_x), float(self.L_y), 1.0)

    def _checked_delaminations(self) -> tuple[Delamination, ...]:
        """:attr:`delaminations`, checked against the laminate and the plate."""
        delams = _as_delaminations(self.delaminations)
        if not delams:
            return ()
        n_plies = int(self.material.n_plies)
        if self.nz % n_plies:
            raise ValueError(
                f"A delamination needs every ply interface on a node layer: "
                f"nz must be a multiple of n_plies = {n_plies} (interface k "
                f"at node layer (k + 1) * nz / n_plies), got nz = {self.nz}.")
        plate, tol = self._plate(), self._tolerance()
        for d in delams:
            if d.interface > n_plies - 2:
                raise ValueError(
                    f"Delamination interface {d.interface} does not exist: a "
                    f"{n_plies}-ply laminate has interfaces 0 to {n_plies - 2} "
                    f"(interface k lies between ply k and ply k + 1, from the "
                    f"bottom).")
            if not d.overlaps_plate(plate, tol=tol):
                raise ValueError(
                    f"{d!r} does not overlap the {plate[1] - plate[0]:g} x "
                    f"{plate[3] - plate[2]:g} mm plate (x in [{plate[0]:g}, "
                    f"{plate[1]:g}], y in [{plate[2]:g}, {plate[3]:g}] mm).")
        return delams

    def _duplicate_interfaces(self):
        """Duplicate the node plane of every delaminated interface.

        The copies of interface ``k`` (node layer ``(k + 1) nz / n_plies``)
        are appended after the structured nodes, interfaces in increasing
        order, each plane in structured order; the bottom face of the
        element layer above the interface is rewired to them. Every
        per-node array (``porosity``, ``stiffness_reduction``,
        ``ply_ids``) is extended with the original's values. Element arrays
        are unchanged: rewiring moves no element centroid.
        """
        # Unannotated, like generate_mesh(), its caller: mypy reads the
        # lazily initialised node arrays as None.
        self._n_structured_nodes = len(self.nodes)
        self._pair_original = np.zeros(0, dtype=np.intp)
        self._pair_interface = np.zeros(0, dtype=np.intp)
        delams = self._checked_delaminations()
        if not delams:
            return
        n0 = self._n_structured_nodes
        layer = (self.nx + 1) * (self.ny + 1)
        elements_per_layer = self.nx * self.ny
        n_plies = int(self.material.n_plies)
        elements = np.array(self.elements, copy=True)
        originals, interfaces = [], []
        for q, k in enumerate(sorted({d.interface for d in delams})):
            node_layer = (k + 1) * self.nz // n_plies
            first = node_layer * layer
            above = slice(node_layer * elements_per_layer,
                          (node_layer + 1) * elements_per_layer)
            # Corners 0-3 are the bottom face of a hexahedron, here node
            # layer ``node_layer``: shift them onto plane q of the copies.
            elements[above, :4] += n0 + q * layer - first
            originals.append(first + np.arange(layer, dtype=np.intp))
            interfaces.append(np.full(layer, k, dtype=np.intp))
        orig = np.concatenate(originals)
        self.nodes = np.vstack([self.nodes, self.nodes[orig]])
        self.porosity = np.concatenate([self.porosity, self.porosity[orig]])
        self.stiffness_reduction = np.concatenate(
            [self.stiffness_reduction, self.stiffness_reduction[orig]])
        self.ply_ids = np.concatenate([self.ply_ids, self.ply_ids[orig]])
        self.elements = elements
        self._pair_original = orig
        self._pair_interface = np.concatenate(interfaces)
        self.bonded_pairs()   # fails early if an interface is cut through

    @property
    def n_structured_nodes(self) -> int:
        """Nodes of the structured grid, ``(nx+1)(ny+1)(nz+1)``.

        They are the first nodes of :attr:`nodes`, in structured order (node
        id ``k (ny+1)(nx+1) + j (nx+1) + i``); the duplicated interface
        nodes of the delaminations follow. Equal to :attr:`n_nodes` without
        delaminations.
        """
        return self._n_structured_nodes

    @property
    def delaminated_interfaces(self) -> tuple[int, ...]:
        """Interfaces whose node plane is duplicated, in increasing order."""
        return tuple(int(k) for k in np.unique(self._pair_interface))

    def interface_node_pairs(self, interface: int | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Original and duplicate node of every duplicated interface node.

        Parameters
        ----------
        interface : int, optional
            Only the pairs of this interface; default all, interfaces in
            increasing order, each in structured order (``x`` fastest).

        Returns
        -------
        originals, duplicates : np.ndarray
            Aligned node indices. The original belongs to the elements
            below the interface, the duplicate to the elements above.
            Empty without delaminations.
        """
        n = self._pair_original.size
        duplicates = self._n_structured_nodes + np.arange(n, dtype=np.intp)
        if interface is None:
            return self._pair_original.copy(), duplicates
        sel = self._pair_interface == int(interface)
        return self._pair_original[sel], duplicates[sel]

    def bonded_pairs(self, interface: int | None = None) -> np.ndarray:
        """Which duplicated node pairs are bonded (tied), from :attr:`delaminations`.

        A pair is delaminated when its node lies strictly inside one of the
        delaminations of its interface, and bonded otherwise; a node on the
        boundary of a delamination stays bonded (the crack front). An edge
        of a ``'strip'`` or ``'rectangle'`` on or beyond the plate edge is
        open (see :class:`Delamination`).

        Parameters
        ----------
        interface : int, optional
            Only the pairs of this interface; default all.

        Returns
        -------
        np.ndarray
            Boolean mask aligned with :meth:`interface_node_pairs`.

        Raises
        ------
        ValueError
            If a delamination lies at an interface without duplicated nodes
            (call :meth:`generate_mesh` after adding it), or a duplicated
            interface has no bonded pair left: the laminate would separate
            into two unconnected parts.
        """
        delams = self._checked_delaminations()
        orig = self._pair_original
        iface = self._pair_interface
        bonded = np.ones(orig.size, dtype=bool)
        if orig.size:
            xy = np.asarray(self.nodes, dtype=float)[orig, :2]
            plate, tol = self._plate(), self._tolerance()
            for d in delams:
                sel = iface == d.interface
                if not sel.any():
                    raise ValueError(
                        f"{d!r} lies at interface {d.interface}, which has no "
                        f"duplicated nodes; call generate_mesh() after adding "
                        f"a delamination at a new interface.")
                bonded[sel] &= ~d.contains(xy[sel, 0], xy[sel, 1],
                                           plate=plate, tol=tol)
            for k in self.delaminated_interfaces:
                if not bonded[iface == k].any():
                    raise ValueError(
                        f"The delaminations at interface {k} cover every node "
                        f"of it: the laminate would separate into two "
                        f"unconnected parts.")
        elif delams:
            raise ValueError(
                "The mesh has delaminations but no duplicated interface "
                "nodes; call generate_mesh().")
        if interface is None:
            return bonded
        return bonded[iface == int(interface)]

    def bonded_ties(self) -> tuple[np.ndarray, np.ndarray]:
        """Degrees of freedom of the bonded interface: ``u[slave] = u[master]``.

        Returns
        -------
        slave_dofs, master_dofs : np.ndarray
            Aligned DOF indices (three per node, ``x`` / ``y`` / ``z``) of
            the duplicate (slave) and original (master) node of every
            bonded pair (:meth:`bonded_pairs`). Empty without
            delaminations. :class:`~porosity_fe.FESolver` imposes them as
            master-slave ties.
        """
        orig, dup = self.interface_node_pairs()
        bonded = self.bonded_pairs()
        comp = np.arange(3, dtype=np.intp)
        slaves = (3 * dup[bonded][:, None] + comp).ravel()
        masters = (3 * orig[bonded][:, None] + comp).ravel()
        return slaves, masters

    def crack_front_pairs(self, interface: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Bonded node pairs on the crack front.

        A bonded pair is on the front when one of its eight in-plane
        neighbours in the interface (the nodes it shares an element face
        with) is delaminated; it is where the virtual crack closure
        technique reads the tie forces.

        Parameters
        ----------
        interface : int, optional
            Only the front of this interface; default all.

        Returns
        -------
        originals, duplicates : np.ndarray
            Aligned node indices, a subset of :meth:`interface_node_pairs`
            in the same order. Empty without delaminations.
        """
        orig, dup = self.interface_node_pairs()
        bonded = self.bonded_pairs()
        front = np.zeros(orig.size, dtype=bool)
        nx1, ny1 = self.nx + 1, self.ny + 1
        for k in self.delaminated_interfaces:
            sel = np.flatnonzero(self._pair_interface == k)
            b = bonded[sel].reshape(ny1, nx1)
            open_ = np.pad(~b, 1)
            near = np.zeros_like(b)
            for dj in range(3):
                for di in range(3):
                    near |= open_[dj:dj + ny1, di:di + nx1]
            front[sel] = (b & near).ravel()
        keep = front if interface is None else \
            front & (self._pair_interface == int(interface))
        return orig[keep], dup[keep]

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
        layer, column ``i`` one x position. Structured nodes only: on a
        delaminated interface they are the original nodes, which belong to
        the elements below it.
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
            element edge length (``0.5 * min(L_x/nx, L_y/ny, L_z/nz)``, with
            the shortest element along ``x`` instead of ``L_x/nx`` on a
            graded mesh, :attr:`x_grid`).

        Returns
        -------
        np.ndarray
            Sorted 1-D array of node indices whose distance to the target
            (restricted to the specified axes) is ``<= tol``.

        Notes
        -----
        A general lookup; the flexure and short-beam boundary conditions
        locate their node lines with
        :meth:`~porosity_fe.BoundaryHandler.face_line_nodes` instead.
        """
        if x is None and y is None and z is None:
            raise ValueError(
                "find_nodes_near: at least one of x/y/z must be specified."
            )
        if tol is None:
            dxs = []
            if self.x_grid is not None:
                # A graded mesh: half its shortest element along x.
                dxs.append(float(np.min(np.diff(np.asarray(self.x_grid, dtype=float)))))
            elif self.nx > 0:
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
        delams = (f", delaminations={len(self.delaminations)}"
                  if self.delaminations else "")
        return (f"CompositeMesh(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
                f"domain={self.L_x:.1f}x{self.L_y:.1f}x{self.L_z:.2f}mm, "
                f"void_elements={len(self.void_elements)}{delams})")


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
    n_distorted = int(np.sum(aspect_ratios > _MAX_ASPECT_RATIO))

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

