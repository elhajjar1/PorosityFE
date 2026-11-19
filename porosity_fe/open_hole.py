"""Open-hole coupons: a mapped hex mesh around a central hole, and its notch analysis.

:class:`OpenHoleMesh` (built by :meth:`CompositeMesh.for_open_hole
<porosity_fe.CompositeMesh.for_open_hole>`) meshes the coupon of the
open-hole tension and compression tests (ASTM D5766, D6484: 36 mm wide, a
6 mm hole) with a ring of elements around the hole, mapped from the circle
to a square of the coupon's width, and rectangular blocks out to the ends,
extruded through the plies like the coupon. The ``'compression'`` and
``'tension'`` loads of :class:`~porosity_fe.FESolver` run on it unchanged
(the end faces are planar). :func:`open_hole_analysis` reads a solve: the
stress concentration at the hole against Lekhnitskii's orthotropic plate and
Tan's finite-width correction, the stress along the net section, and the
notched strength by the point- and average-stress criteria of Whitney and
Nuismer. Porosity enters through the degraded stiffness of every element
(and the unnotched strength the criteria scale); with a random porosity
field (:meth:`PorosityField.random <porosity_fe.PorosityField.random>`) the
voids at the hole edge count most.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .homogenization import _clt_abd_full
from .materials import MaterialProperties
from .mesh import CompositeMesh, _expand_layup, _ply_at
from .porosity_field import PorosityField

if TYPE_CHECKING:
    from .fe.results import FieldResults

#: Load cases of an open-hole coupon: the end-loaded ones.
OPEN_HOLE_LOADINGS: tuple[str, ...] = ('compression', 'tension')

#: ASTM D5766 / D6484 coupon: 36 mm wide with a 6 mm hole (W / D = 6).
_ASTM_WIDTH = 36.0
_ASTM_HOLE = 6.0

#: Largest D / W the mapping meshes well (the ring elements at the square's
#: corners skew as the hole grows).
_MAX_D_OVER_W = 0.6

#: Width of the first ring at the hole over the elements' length around it:
#: the peak at the edge is steep, steeper the more orthotropic the laminate.
_FIRST_RING = 0.25

#: Growth of the end blocks' element lengths away from the hole, and their
#: cap in element-layer thicknesses (the mesh-quality check warns above 20).
_END_GROWTH = 1.25
_END_ASPECT = 18.0


@dataclass(frozen=True)
class OpenHoleGeometry:
    """The coupon of an :class:`OpenHoleMesh` (mm).

    Attributes
    ----------
    hole_diameter, width, length : float
        ``D``, ``W`` and ``L``; the coupon spans ``[0, L] x [0, W]``.
    center : (float, float)
        The hole's centre, ``(L / 2, W / 2)``.
    n_circumferential : int
        Elements per quarter of the hole (an even number, so node lines run
        through the hole's top and bottom on the net section).
    n_radial : int
        Element rings from the hole to the square of side ``W`` around it.
    radial_growth : float
        Ratio of consecutive ring widths, outward.
    n_end : int
        Element columns of each end block, from the square to the coupon's
        end (0 when ``L = W``); they lengthen by ``1.25`` per column away
        from the hole, from the length of the square's side elements, up to
        ``end_cap``.
    end_cap : float
        Longest end-block element (mm): 18 element-layer thicknesses, or the
        square's side elements if longer.
    """

    hole_diameter: float
    width: float
    length: float
    center: tuple[float, float]
    n_circumferential: int
    n_radial: int
    radial_growth: float
    n_end: int
    end_cap: float = math.inf

    @property
    def radius(self) -> float:
        """Hole radius ``D / 2`` (mm)."""
        return 0.5 * self.hole_diameter

    @property
    def diameter_to_width(self) -> float:
        """``D / W``."""
        return self.hole_diameter / self.width


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
            or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive finite number (mm), got {value!r}.")
    return float(value)


def _count(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _matching_growth(first: float, span: float, n: int) -> float:
    """Growth ratio ``q`` with ``first (q^n - 1) / (q - 1) = span``: rings that
    start ``first`` wide at the hole and fill ``span`` in ``n``."""
    if first * n >= span:
        return 1.0
    lo, hi = 1.0 + 1e-12, 10.0
    for _ in range(200):
        q = 0.5 * (lo + hi)
        if first * (q ** n - 1.0) / (q - 1.0) < span:
            lo = q
        else:
            hi = q
    return 0.5 * (lo + hi)


def _end_widths(first: float, cap: float, n: int) -> np.ndarray:
    return np.minimum(first * _END_GROWTH ** np.arange(n), max(cap, first))


def _end_lines(span: float, first: float, n: int, cap: float = math.inf) -> np.ndarray:
    """Distances of ``n`` element boundaries from the square, ``first``
    wide at the square and growing by :data:`_END_GROWTH` up to ``cap``,
    scaled to end at ``span``."""
    widths = _end_widths(first, cap, n)
    return span * np.cumsum(widths) / widths.sum()


def _end_columns(span: float, first: float, cap: float = math.inf) -> int:
    """Columns of an end block of length ``span`` (see :func:`_end_lines`)."""
    n = 1
    while _end_widths(first, cap, n).sum() < span * (1.0 - 1e-9):
        n += 1
    return n


def _open_hole_quads(g: OpenHoleGeometry) -> tuple[np.ndarray, np.ndarray]:
    """In-plane nodes ``(n, 2)`` and counter-clockwise quads ``(m, 4)``.

    The ring: node ``(r, c)`` is ``r * 4 n_c + c``, ``c`` around the hole
    from -45 degrees (counter-clockwise, periodic) and ``r`` outward, on the
    straight line from the hole point at its angle to the point of the
    square's side at the same fraction of the side. Then the right and left
    end blocks, which share the ring's outer nodes on the square's sides.
    """
    nc, nr, ne = g.n_circumferential, g.n_radial, g.n_end
    m = 4 * nc
    xc, yc = g.center
    a, R = 0.5 * g.width, g.radius
    c = np.arange(m)
    theta = -0.25 * np.pi + 2.0 * np.pi * c / m
    inner = np.column_stack((xc + R * np.cos(theta), yc + R * np.sin(theta)))
    side, s = c // nc, (c % nc) / nc
    outer = np.empty((m, 2))
    outer[side == 0] = np.column_stack((np.full((side == 0).sum(), xc + a),
                                        yc - a + 2.0 * a * s[side == 0]))
    outer[side == 1] = np.column_stack((xc + a - 2.0 * a * s[side == 1],
                                        np.full((side == 1).sum(), yc + a)))
    outer[side == 2] = np.column_stack((np.full((side == 2).sum(), xc - a),
                                        yc + a - 2.0 * a * s[side == 2]))
    outer[side == 3] = np.column_stack((xc - a + 2.0 * a * s[side == 3],
                                        np.full((side == 3).sum(), yc - a)))
    q = g.radial_growth
    r = np.arange(nr + 1, dtype=float)
    t = r / nr if q == 1.0 else (q ** r - 1.0) / (q ** nr - 1.0)
    ring = inner[None, :, :] + t[:, None, None] * (outer - inner)[None, :, :]
    xy = [ring.reshape(-1, 2)]
    rr, cc = np.meshgrid(np.arange(nr), np.arange(m), indexing='ij')
    quads = [np.column_stack(((rr * m + cc).ravel(), ((rr + 1) * m + cc).ravel(),
                              ((rr + 1) * m + (cc + 1) % m).ravel(),
                              (rr * m + (cc + 1) % m).ravel()))]
    n = (nr + 1) * m
    outer_id = nr * m
    if ne:
        # Right block: column 0 is the square's right side (ring c = 0..nc).
        span, first = g.length - (xc + a), g.width / nc
        ys = ring[nr, :nc + 1, 1]
        gx, gy = np.meshgrid(xc + a + _end_lines(span, first, ne, g.end_cap), ys,
                             indexing='ij')
        xy.append(np.column_stack((gx.ravel(), gy.ravel())))
        right = np.empty((ne + 1, nc + 1), dtype=np.intp)
        right[0] = outer_id + np.arange(nc + 1)
        right[1:] = n + np.arange(ne * (nc + 1)).reshape(ne, nc + 1)
        n += ne * (nc + 1)
        # Left block: its last column is the square's left side (ring
        # c = 3 nc down to 2 nc, y ascending).
        ys = ring[nr, 3 * nc - np.arange(nc + 1), 1]
        lines = (xc - a) - _end_lines(span, first, ne, g.end_cap)[::-1]
        gx, gy = np.meshgrid(np.concatenate(([0.0], lines[1:])), ys, indexing='ij')
        xy.append(np.column_stack((gx.ravel(), gy.ravel())))
        left = np.empty((ne + 1, nc + 1), dtype=np.intp)
        left[:ne] = n + np.arange(ne * (nc + 1)).reshape(ne, nc + 1)
        left[ne] = outer_id + 3 * nc - np.arange(nc + 1)
        for block in (right, left):
            i, j = np.meshgrid(np.arange(ne), np.arange(nc), indexing='ij')
            quads.append(np.column_stack((block[i, j].ravel(), block[i + 1, j].ravel(),
                                          block[i + 1, j + 1].ravel(),
                                          block[i, j + 1].ravel())))
    return np.vstack(xy), np.vstack(quads).astype(np.intp)


class OpenHoleMesh(CompositeMesh):
    """Hex mesh of an open-hole coupon; see :meth:`CompositeMesh.for_open_hole`.

    A :class:`~porosity_fe.CompositeMesh` whose in-plane mesh is mapped
    around the hole instead of a rectangular grid: the nodes are extruded
    through the thickness layer by layer, so every per-node and per-element
    array (porosity, ply ids, ply angles, void elements) is the coupon's.
    ``nx`` and ``ny`` hold the element counts along the coupon's long and
    short edges, for the lateral margins of the interior metrics.
    Delaminations, the interface traction recovery and the grid plots need
    the rectangular grid and are refused.

    Attributes
    ----------
    open_hole_geometry : OpenHoleGeometry
        The coupon and its mesh parameters.
    in_plane_nodes : numpy.ndarray
        ``(n, 2)`` coordinates of one node layer (mm); node ``k n + p`` of
        :attr:`nodes` is in-plane node ``p`` at node layer ``k``.
    in_plane_quads : numpy.ndarray
        ``(m, 4)`` in-plane node indices of the quadrilaterals;
        element ``k m + e`` of :attr:`elements` is quad ``e`` in element
        layer ``k``.
    """

    def __init__(self, porosity_field: PorosityField, material: MaterialProperties, *,
                 geometry: OpenHoleGeometry, nz: int,
                 ply_angles: list[float] | str | None = 'QI') -> None:
        self.open_hole_geometry = geometry
        super().__init__(porosity_field, material,
                         nx=2 * geometry.n_end + geometry.n_circumferential,
                         ny=geometry.n_circumferential, nz=nz, ply_angles=ply_angles,
                         length=geometry.length, width=geometry.width)

    def generate_mesh(self):
        if self.delaminations:
            raise ValueError(
                "An open-hole mesh takes no delamination: the duplicated "
                "interface nodes need the rectangular grid of CompositeMesh.")
        xy, quads = _open_hole_quads(self.open_hole_geometry)
        self.in_plane_nodes, self.in_plane_quads = xy, quads
        n2, nq = len(xy), len(quads)
        z = np.linspace(0.0, self.L_z, self.nz + 1)
        self.nodes = np.column_stack((np.tile(xy[:, 0], self.nz + 1),
                                      np.tile(xy[:, 1], self.nz + 1),
                                      np.repeat(z, n2)))
        self.porosity = self._sample_porosity(z)
        self.stiffness_reduction = 1.0 - self.porosity
        offset = (np.arange(self.nz) * n2)[:, None, None]
        bottom = (quads[None, :, :] + offset).reshape(-1, 4)
        self.elements = np.hstack((bottom, bottom + n2))
        centers = self.nodes[self.elements].mean(axis=1)
        void_mask = np.zeros(len(self.elements), dtype=bool)
        for void in self.porosity_field.discrete_voids:
            void_mask |= void.contains(centers[:, 0], centers[:, 1], centers[:, 2])
        self.void_elements = np.where(void_mask)[0]
        self.void_element_set = set(self.void_elements.tolist())
        n_plies = self.material.n_plies
        self.ply_ids = np.repeat(_ply_at(np.arange(self.nz + 1) * n_plies, self.nz, n_plies),
                                 n2)
        self.elem_ply_ids = np.repeat(
            _ply_at((2 * np.arange(self.nz) + 1) * n_plies, 2 * self.nz, n_plies), nq)
        self._ply_layup = _expand_layup(list(self._input_ply_angles), n_plies) \
            if self._input_ply_angles is not None else np.zeros(n_plies)
        self.ply_angles = self._ply_layup[self.elem_ply_ids]
        self._layup_warning_logged = False
        self._duplicate_interfaces()
        self._stale = False

    def mid_y_section_indices(self) -> np.ndarray:
        raise ValueError("An open-hole mesh has no mid-y grid section: it is mapped "
                         "around its hole (use net_section_nodes or a VTU export).")

    def mid_y_element_indices(self) -> np.ndarray:
        raise ValueError("An open-hole mesh has no mid-y grid section: it is mapped "
                         "around its hole (use net_section_nodes or a VTU export).")

    def __repr__(self) -> str:
        g = self.open_hole_geometry
        return (f"OpenHoleMesh(D={g.hole_diameter:g} mm, W={g.width:g} mm, "
                f"L={g.length:g} mm, {len(self.in_plane_quads)} quads x nz={self.nz}, "
                f"n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
                f"void_elements={len(self.void_elements)})")

    def net_section_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """In-plane nodes on the net section ``x = L / 2``, above and below
        the hole, each ordered outward from the hole edge."""
        g = self.open_hole_geometry
        m = 4 * g.n_circumferential
        r = np.arange(g.n_radial + 1)
        top = r * m + (3 * g.n_circumferential) // 2
        bottom = r * m + (7 * g.n_circumferential) // 2
        return top, bottom


def _laminate_constants(material: MaterialProperties, layup: np.ndarray
                        ) -> tuple[float, float, float, float]:
    """In-plane engineering constants ``(E_x, E_y, G_xy, nu_xy)`` of the
    pristine laminate from lamination theory (MPa)."""
    A, _, _ = _clt_abd_full(material.get_stiffness_matrix(), list(layup), material.t_ply)
    h = len(layup) * material.t_ply
    a = np.linalg.inv(A)
    return (float(1.0 / (h * a[0, 0])), float(1.0 / (h * a[1, 1])),
            float(1.0 / (h * a[2, 2])), float(-a[0, 1] / a[0, 0]))


def orthotropic_stress_concentration(E_x: float, E_y: float, G_xy: float, nu_xy: float,
                                     diameter_to_width: float = 0.0) -> float:
    """Stress concentration of a circular hole in an orthotropic plate.

    Lekhnitskii's infinite-plate value at the hole edge under a uniaxial load
    along ``x``,

    .. math::

        K_T^\\infty = 1 + \\sqrt{2\\left(\\sqrt{E_x / E_y} - \\nu_{xy}\\right)
                                 + E_x / G_{xy}},

    corrected for a finite width by Tan's factor (Tan, *J. Compos. Mater.*
    22, 1988), with ``lambda = D / W``:

    .. math::

        \\frac{K_T^\\infty}{K_T} = \\frac{3(1-\\lambda)}{2 + (1-\\lambda)^3}
            + \\tfrac12 (\\lambda M)^6 (K_T^\\infty - 3)\\left[1 - (\\lambda M)^2\\right],

        M^2 = \\frac{\\sqrt{1 - 8\\left[\\frac{3(1-\\lambda)}{2 + (1-\\lambda)^3}
              - 1\\right]} - 1}{2\\lambda^2}.

    ``K_T`` is the peak stress over the gross (far-field) stress; an
    in-plane isotropic laminate gives 3 for an infinite plate.

    Examples
    --------
    >>> round(orthotropic_stress_concentration(70e3, 70e3, 70e3 / 2.6, 0.3), 6)
    3.0
    >>> round(orthotropic_stress_concentration(70e3, 70e3, 70e3 / 2.6, 0.3, 1 / 6), 3)
    3.094
    """
    k_inf = 1.0 + math.sqrt(2.0 * (math.sqrt(E_x / E_y) - nu_xy) + E_x / G_xy)
    lam = float(diameter_to_width)
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"diameter_to_width must lie in [0, 1), got {lam!r}.")
    if lam == 0.0:
        return k_inf
    iso = 3.0 * (1.0 - lam) / (2.0 + (1.0 - lam) ** 3)
    M = math.sqrt((math.sqrt(1.0 - 8.0 * (iso - 1.0)) - 1.0) / (2.0 * lam * lam))
    ratio = iso + 0.5 * (lam * M) ** 6 * (k_inf - 3.0) * (1.0 - (lam * M) ** 2)
    return k_inf / ratio


@dataclass(frozen=True, eq=False)
class OpenHoleAnalysis:
    """The notch of an open-hole solve; see :func:`open_hole_analysis`.

    Attributes
    ----------
    geometry : OpenHoleGeometry
        The coupon.
    loading : str
        The load case of the solve.
    far_field_stress : float
        Gross stress, the end load over ``W h`` (MPa, negative in
        compression).
    stress_concentration : float
        ``K_T``: the laminate-average axial stress at the hole edge (on the
        net section, the mean of the top and bottom edge) over the far-field
        stress.
    distance : numpy.ndarray
        Distance from the hole edge along the net section (mm), from 0 to
        ``W / 2 - R``.
    stress_ratio : numpy.ndarray
        Laminate-average axial stress over the far-field stress at each
        :attr:`distance` (mean of the two ligaments).
    reference_stress_concentration : float
        Lekhnitskii's ``K_T`` of the pristine laminate with Tan's finite-width
        correction (:func:`orthotropic_stress_concentration`).
    laminate_constants : dict
        ``E_x``, ``E_y``, ``G_xy`` (MPa) and ``nu_xy`` of the pristine
        laminate (lamination theory) the reference uses.
    """

    geometry: OpenHoleGeometry
    loading: str
    far_field_stress: float
    stress_concentration: float
    distance: np.ndarray
    stress_ratio: np.ndarray
    reference_stress_concentration: float
    laminate_constants: dict[str, float]

    @property
    def net_stress_concentration(self) -> float:
        """``K_T`` over the net-section stress: ``K_T (1 - D / W)``."""
        return self.stress_concentration * (1.0 - self.geometry.diameter_to_width)

    def _ratio_at(self, d: float) -> float:
        return float(np.interp(d, self.distance, self.stress_ratio))

    def point_stress_factor(self, d0: float) -> float:
        """Notched-to-unnotched strength ratio by the point-stress criterion
        (Whitney and Nuismer 1974): the far-field stress at which the stress
        a characteristic distance ``d0`` (mm) from the hole edge reaches the
        unnotched strength, ``1 / ratio(d0)``."""
        d0 = self._distance('d0', d0)
        return 1.0 / self._ratio_at(d0)

    def average_stress_factor(self, a0: float) -> float:
        """Notched-to-unnotched strength ratio by the average-stress
        criterion (Whitney and Nuismer 1974): the stress averaged over
        ``a0`` (mm) from the hole edge reaches the unnotched strength,
        ``a0 / integral_0^a0 ratio``."""
        a0 = self._distance('a0', a0)
        d = np.linspace(0.0, a0, 201)
        ratio = np.interp(d, self.distance, self.stress_ratio)
        mean = float(np.sum(0.5 * (ratio[1:] + ratio[:-1]) * np.diff(d)) / a0)
        return 1.0 / mean

    def notched_strength(self, unnotched_strength: float, *, d0: float | None = None,
                         a0: float | None = None) -> float:
        """Notched strength (MPa, gross) from the unnotched laminate
        strength: by the point-stress criterion with ``d0`` or the
        average-stress criterion with ``a0`` (give one). The characteristic
        distances are the material's, fitted to notched tests (about 1 mm
        for ``d0`` and 4 mm for ``a0`` in Whitney and Nuismer's data): none
        ships."""
        strength = _positive('unnotched_strength', unnotched_strength)
        if (d0 is None) == (a0 is None):
            raise ValueError("Give one of d0 (point stress) or a0 (average stress), in mm.")
        factor = self.point_stress_factor(d0) if d0 is not None \
            else self.average_stress_factor(a0)  # type: ignore[arg-type]
        return strength * factor

    def _distance(self, name: str, value: float) -> float:
        d = _positive(name, value)
        if d > float(self.distance[-1]):
            raise ValueError(
                f"{name} = {d:g} mm reaches past the ligament ({self.distance[-1]:g} mm "
                f"from the hole edge to the coupon's edge).")
        return d

    def summary(self, *, d0: float | None = None, a0: float | None = None
                ) -> dict[str, Any]:
        """The analysis as a JSON-ready dict; with ``d0`` or ``a0`` (mm) it
        adds the ``notch_criteria`` block, the point- and average-stress
        factors at them."""
        g = self.geometry
        criteria: dict[str, float] = {}
        if d0 is not None:
            criteria.update(point_stress_distance_mm=float(d0),
                            point_stress_factor=self.point_stress_factor(d0))
        if a0 is not None:
            criteria.update(average_stress_distance_mm=float(a0),
                            average_stress_factor=self.average_stress_factor(a0))
        extra = {'notch_criteria': criteria} if criteria else {}
        return {
            'hole_diameter_mm': g.hole_diameter, 'width_mm': g.width, 'length_mm': g.length,
            'loading': self.loading,
            'far_field_stress_MPa': self.far_field_stress,
            'stress_concentration': self.stress_concentration,
            'net_stress_concentration': self.net_stress_concentration,
            'reference_stress_concentration': self.reference_stress_concentration,
            'laminate_constants': dict(self.laminate_constants),
            'distance_mm': [float(v) for v in self.distance],
            'stress_ratio': [float(v) for v in self.stress_ratio],
            **extra,
        }


def _laminate_axial_stress(results: FieldResults, mesh: OpenHoleMesh,
                           points: np.ndarray) -> np.ndarray:
    """Laminate-average ``sigma_xx`` at in-plane nodes ``points``: per element
    layer, the mean of the recovered corner values of the elements around
    the node at the layer's bottom and top, averaged over the layers."""
    from .fe.recovery import extrapolate_to_nodes

    _, corner = extrapolate_to_nodes(np.asarray(results.stress_global)[..., 0], mesh)
    quads = mesh.in_plane_quads
    nq = len(quads)
    out = np.empty(len(points))
    for i, p in enumerate(points):
        around, slot = np.nonzero(quads == p)
        layers = []
        for k in range(mesh.nz):
            elems = k * nq + around
            layers.append(0.5 * (corner[elems, slot].mean() + corner[elems, slot + 4].mean()))
        out[i] = float(np.mean(layers))
    return out


def open_hole_analysis(results: FieldResults, mesh: OpenHoleMesh) -> OpenHoleAnalysis:
    """The stress concentration and net-section stress of an open-hole solve.

    Parameters
    ----------
    results : FieldResults
        A ``'compression'`` or ``'tension'`` solve of ``mesh``
        (:meth:`FESolver.solve <porosity_fe.FESolver.solve>`).
    mesh : OpenHoleMesh
        The coupon (:meth:`CompositeMesh.for_open_hole
        <porosity_fe.CompositeMesh.for_open_hole>`).

    Returns
    -------
    OpenHoleAnalysis

    Raises
    ------
    ValueError
        For a mesh that is not an open-hole mesh, or results without
        reactions (a solve of another load case).

    Notes
    -----
    The stresses are the laminate averages through the thickness of the
    recovered (extrapolated, ply-averaged) axial stress, so ``K_T`` is the
    laminate's, comparable with Lekhnitskii's solution for the laminate's
    in-plane constants; the ply stresses at the edge differ ply by ply.
    The far-field stress is the end reaction over the gross section.
    """
    if not isinstance(mesh, OpenHoleMesh):
        raise ValueError("open_hole_analysis needs an OpenHoleMesh "
                         "(CompositeMesh.for_open_hole).")
    reactions = getattr(results, 'reaction_forces', None)
    if reactions is None:
        raise ValueError("open_hole_analysis needs the reactions of a 'compression' or "
                         "'tension' solve.")
    g = mesh.open_hole_geometry
    x = mesh.nodes[:, 0]
    end = np.isclose(x, mesh.L_x, atol=1e-9 * mesh.L_x)
    force = float(np.asarray(reactions)[end, 0].sum())
    far = force / (mesh.L_y * mesh.L_z)
    if far == 0.0:
        raise ValueError("The solve carries no end load: no far-field stress.")
    top, bottom = mesh.net_section_nodes()
    sigma = 0.5 * (_laminate_axial_stress(results, mesh, top)
                   + _laminate_axial_stress(results, mesh, bottom))
    ratio = sigma / far
    distance = mesh.in_plane_nodes[top, 1] - (g.center[1] + g.radius)
    E_x, E_y, G_xy, nu = _laminate_constants(mesh.material, mesh._ply_layup)
    loading = 'tension' if far > 0.0 else 'compression'
    return OpenHoleAnalysis(
        geometry=g, loading=loading, far_field_stress=far,
        stress_concentration=float(ratio[0]), distance=distance, stress_ratio=ratio,
        reference_stress_concentration=orthotropic_stress_concentration(
            E_x, E_y, G_xy, nu, g.diameter_to_width),
        laminate_constants={'E_x': E_x, 'E_y': E_y, 'G_xy': G_xy, 'nu_xy': nu})


def build_open_hole_mesh(porosity_field: PorosityField, material: MaterialProperties, *,
                         hole_diameter: float = _ASTM_HOLE, width: float = _ASTM_WIDTH,
                         length: float | None = None, n_circumferential: int = 12,
                         n_radial: int = 16, radial_growth: float | None = None,
                         nz: int | None = None,
                         ply_angles: list[float] | str | None = 'QI') -> OpenHoleMesh:
    """The mesh of :meth:`CompositeMesh.for_open_hole`; see there."""
    D, W = _positive('hole_diameter', hole_diameter), _positive('width', width)
    if D / W > _MAX_D_OVER_W:
        raise ValueError(
            f"The hole is too large for the mapped mesh: D / W = {D / W:.3g} > "
            f"{_MAX_D_OVER_W:g}.")
    L = 3.0 * W if length is None else _positive('length', length)
    if L < W:
        raise ValueError(f"length ({L:g} mm) must be at least the width ({W:g} mm).")
    nc = _count('n_circumferential', n_circumferential, 4)
    if nc % 2:
        raise ValueError(f"n_circumferential must be even (a node line on the net "
                         f"section), got {nc}.")
    nr = _count('n_radial', n_radial, 2)
    a, R = 0.5 * W, 0.5 * D
    if radial_growth is None:
        growth = _matching_growth(_FIRST_RING * 0.5 * math.pi * R / nc, a - R, nr)
    else:
        growth = float(radial_growth)
        if not (math.isfinite(growth) and growth >= 1.0):
            raise ValueError(f"radial_growth must be a finite number >= 1, got "
                             f"{radial_growth!r}.")
    if nz is None:
        from .pipeline import _default_mesh_resolution
        nz = _default_mesh_resolution(material.n_plies, ply_angles)[2]
    cap = max(W / nc, _END_ASPECT * material.total_thickness / nz)
    n_end = 0 if L == W else _end_columns(0.5 * (L - W), W / nc, cap)
    geometry = OpenHoleGeometry(
        hole_diameter=D, width=W, length=L, center=(0.5 * L, 0.5 * W),
        n_circumferential=nc, n_radial=nr, radial_growth=growth, n_end=n_end,
        end_cap=cap)
    return OpenHoleMesh(porosity_field, material, geometry=geometry, nz=nz,
                        ply_angles=ply_angles)
