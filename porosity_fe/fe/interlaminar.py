"""Interlaminar tractions at the ply interfaces and stress-based delamination onset.

Post-processing of a solved :class:`~porosity_fe.FieldResults`; nothing here
changes what :meth:`FESolver.solve <porosity_fe.FESolver.solve>` computes.

**Traction recovery** (:func:`interface_tractions`). At every ply interface
that is a node layer of the mesh (all of them when ``nz`` is a multiple of
``n_plies``) the interlaminar traction ``t = sigma . e_z = (tau_xz, tau_yz,
sigma_zz)`` is recovered in the laminate (global) frame from both adjacent
element layers. Each element's eight Gauss-point stresses are extrapolated
to the shared face with the trilinear extrapolation of
:func:`~porosity_fe.extrapolate_to_nodes` (``fe/recovery.py``), which gives a
bilinear traction field on every face of each side. The face values are
averaged at each interface node over the faces of that side, and the two
sides are reported with their average and their jump. Equilibrium makes the
exact traction continuous across a bonded interface, so the jump measures
how well the mesh resolves the interlaminar stress there; it does not
vanish at a free-edge singularity.

**Onset** (:func:`delamination_onset`). The traction, averaged over an
in-plane window of side ``x_avg`` (one ply thickness by default, Kim & Soni
1984), is checked against porosity-degraded interlaminar strengths with the
Brewer & Lagace (1988) quadratic criterion, or optionally the quadratic
criterion attributed to Ye (1988) in the ply axes. The analysis is linear,
so the onset index is quadratic in the load factor: the factor is closed
form for a mechanical load and the positive root of a quadratic with a
thermal residual prestress.

See the theory page "Failure criteria", section "Interlaminar stresses and
delamination onset". Units: MPa and mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._types import PorosityDegradation
from ..empirical import Calibration
from ..homogenization import _DEFAULT_POROSITY_DEGRADATION, _check_porosity_degradation
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..transforms import stress_transformation_3d
from . import failure
from .element import VOID_VP_THRESHOLD
from .recovery import _GP_TO_NODE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .results import FieldResults

#: Onset criteria accepted by :func:`delamination_onset`.
OnsetCriterion = Literal['brewer_lagace', 'ye']
ONSET_CRITERIA: tuple[str, ...] = ('brewer_lagace', 'ye')

#: Voigt slots of the traction ``sigma . e_z``: ``(tau_xz, tau_yz, sigma_zz)``.
_TRACTION_SLOTS = (4, 3, 2)
#: Face corner ``c`` (0-3: ``(i, j)``, ``(i+1, j)``, ``(i+1, j+1)``,
#: ``(i, j+1)``, the in-plane order of both the bottom and the top face of a
#: hexahedron) laid out as ``[b][a]`` with ``a`` the ``x`` and ``b`` the
#: ``y`` offset.
_CORNER_BA = ((0, 1), (3, 2))
#: Extrapolation from the Gauss points to the top face (corners 4-7) of the
#: element below an interface, and to the bottom face (0-3) of the one above.
_GP_TO_TOP = _GP_TO_NODE[4:8]
_GP_TO_BOTTOM = _GP_TO_NODE[0:4]


def _covered(interface: np.ndarray) -> tuple[int, ...]:
    """The interfaces of a row layout, increasing."""
    return tuple(int(k) for k in np.unique(interface))


def _grid_of(interface: np.ndarray, shape: tuple[int, int], values: np.ndarray,
             k: int) -> np.ndarray:
    """The rows of interface ``k`` of a per-row array as an ``(ny+1, nx+1)`` map."""
    sel = interface == int(k)
    if not sel.any():
        raise ValueError(
            f"interface {k} is not covered; the rows hold interfaces "
            f"{list(_covered(interface))}.")
    v = np.asarray(values)[sel]
    return v.reshape(shape + v.shape[1:])


@dataclass(frozen=True, eq=False)
class InterfaceTractions:
    """Interlaminar traction at the nodes of the ply interfaces.

    Rows are interface nodes (see :meth:`grid` for maps). The traction is
    ``t = sigma . e_z`` in the laminate (global) frame, components
    ``(tau_xz, tau_yz, sigma_zz)`` in MPa: the stress vector on the
    interface plane, the same physical vector whichever ply it is computed
    from.

    Attributes
    ----------
    interface : np.ndarray
        Interface index ``k`` of each row (between ply ``k`` and ply
        ``k + 1`` from the bottom).
    node : np.ndarray
        Structured node id of each row (the original node of a duplicated
        pair on a delaminated interface).
    x, y, z : np.ndarray
        Node coordinates (mm).
    below, above : np.ndarray
        ``(n, 3)`` traction extrapolated from the element layer below and
        above the interface, averaged over that side's faces around the
        node (MPa).
    traction : np.ndarray
        ``(n, 3)`` average of the two sides (MPa).
    jump : np.ndarray
        ``(n, 3)`` ``above - below`` (MPa), the resolution indicator: the
        exact traction is continuous across a bonded interface.
    Vp : np.ndarray
        Porosity (fraction) at the node.
    valid : np.ndarray
        Whether the node has a bonded face between solid elements on both
        sides. On a delaminated interface only bonded faces (all four
        corner pairs tied) count, so the nodes inside a delamination, whose
        faces are traction free or in contact, are not valid; next to a
        geometric void element neither are they. The traction arrays are
        ``NaN`` where not valid.
    shape : tuple of int
        ``(ny + 1, nx + 1)``, the node layout of one interface.

    See Also
    --------
    interface_tractions, delamination_onset
    """

    interface: np.ndarray
    node: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    below: np.ndarray
    above: np.ndarray
    traction: np.ndarray
    jump: np.ndarray
    Vp: np.ndarray
    valid: np.ndarray
    shape: tuple[int, int]

    def __len__(self) -> int:
        return int(self.node.size)

    @property
    def interfaces(self) -> tuple[int, ...]:
        """The interfaces covered, in increasing order."""
        return _covered(self.interface)

    def grid(self, values: np.ndarray, interface: int) -> np.ndarray:
        """One interface of a per-row array as an ``(ny + 1, nx + 1)`` map.

        Parameters
        ----------
        values : np.ndarray
            ``(n,)`` or ``(n, k)``, one entry per row, e.g. :attr:`traction`.
        interface : int
            Interface index.

        Returns
        -------
        np.ndarray
            ``(ny + 1, nx + 1)`` or ``(ny + 1, nx + 1, k)``, indexed ``[j, i]``
            (``y`` row, ``x`` column).
        """
        return _grid_of(self.interface, self.shape, values, interface)

    @property
    def shear(self) -> np.ndarray:
        """Resultant interlaminar shear ``sqrt(tau_xz^2 + tau_yz^2)`` (MPa)."""
        return np.hypot(self.traction[:, 0], self.traction[:, 1])

    @property
    def jump_norm(self) -> np.ndarray:
        """Euclidean norm of :attr:`jump` (MPa)."""
        return np.linalg.norm(self.jump, axis=1)


@dataclass(frozen=True, eq=False)
class DelaminationOnset:
    """Stress-based delamination onset at the ply interfaces.

    Rows are interface nodes, as in :class:`InterfaceTractions` (see
    :meth:`grid` for maps of :attr:`index`). Built by
    :func:`delamination_onset`.

    Attributes
    ----------
    criterion : str
        ``'brewer_lagace'`` or ``'ye'``.
    x_avg : float
        Side of the square in-plane averaging window (mm); 0 means the
        nodal traction, without averaging.
    Zc : float or None
        Pristine interlaminar compressive strength of the optional
        compression term (MPa), ``None`` without it.
    load_factor_basis : str
        What :attr:`load_factor` multiplies, as
        :attr:`FieldResults.load_factor_basis
        <porosity_fe.FieldResults.load_factor_basis>`: ``'mechanical'``,
        ``'delta_T'`` or ``'mechanical_with_residual'`` (the mechanical load,
        with the thermal residual stress held fixed).
    interface, node, x, y, z : np.ndarray
        Interface index, structured node id and coordinates (mm) per row.
    traction : np.ndarray
        ``(n, 3)`` averaged traction ``(tau_xz, tau_yz, sigma_zz)`` at the
        applied state (MPa, laminate frame).
    index : np.ndarray
        Onset index per row (1 = onset at the applied state); ``NaN`` where
        not evaluated.
    load_factor : np.ndarray
        Onset load factor per row: the multiplier on the load named by
        :attr:`load_factor_basis` at which the index reaches 1; ``inf``
        where the load never drives it there, ``NaN`` where not evaluated.
    Zt, S13, S23 : np.ndarray
        Porosity-degraded strengths used at each row (MPa).
    Vp : np.ndarray
        Porosity (fraction) at the node, which sets the degradation.
    evaluated : np.ndarray
        Whether the row was evaluated: a bonded node between solid
        elements (see :attr:`InterfaceTractions.valid`) with
        ``Vp <= 0.95``.
    interior : np.ndarray
        Rows at least the interior margin from every lateral face.
    shape : tuple of int
        ``(ny + 1, nx + 1)``.

    See Also
    --------
    delamination_onset
    """

    criterion: str
    x_avg: float
    Zc: float | None
    load_factor_basis: str
    interface: np.ndarray
    node: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    traction: np.ndarray
    index: np.ndarray
    load_factor: np.ndarray
    Zt: np.ndarray
    S13: np.ndarray
    S23: np.ndarray
    Vp: np.ndarray
    evaluated: np.ndarray
    interior: np.ndarray
    shape: tuple[int, int]

    def __len__(self) -> int:
        return int(self.node.size)

    @property
    def interfaces(self) -> tuple[int, ...]:
        """The interfaces covered, in increasing order."""
        return _covered(self.interface)

    def grid(self, values: np.ndarray, interface: int) -> np.ndarray:
        """One interface of a per-row array as an ``(ny + 1, nx + 1)`` map.

        Parameters
        ----------
        values : np.ndarray
            ``(n,)`` or ``(n, k)``, one entry per row, e.g. :attr:`index`.
        interface : int
            Interface index.

        Returns
        -------
        np.ndarray
            ``(ny + 1, nx + 1)`` or ``(ny + 1, nx + 1, k)``, indexed ``[j, i]``
            (``y`` row, ``x`` column).
        """
        return _grid_of(self.interface, self.shape, values, interface)

    def _argmax(self, mask: np.ndarray) -> int | None:
        sel = mask & self.evaluated
        if not sel.any():
            return None
        return int(np.argmax(np.where(sel, self.index, -np.inf)))

    @property
    def critical(self) -> int | None:
        """Row of the largest :attr:`index`, ``None`` when nothing was evaluated."""
        return self._argmax(np.ones(self.index.shape, dtype=bool))

    @property
    def max_index(self) -> float | None:
        """Largest onset index over the evaluated rows."""
        row = self.critical
        return None if row is None else float(self.index[row])

    @property
    def critical_interface(self) -> int | None:
        """Interface of the :attr:`critical` row."""
        row = self.critical
        return None if row is None else int(self.interface[row])

    @property
    def critical_location(self) -> tuple[float, float, float] | None:
        """``(x, y, z)`` of the :attr:`critical` row (mm)."""
        row = self.critical
        return None if row is None else (
            float(self.x[row]), float(self.y[row]), float(self.z[row]))

    @property
    def onset_load_factor(self) -> float | None:
        """Smallest :attr:`load_factor` over the evaluated rows (``inf``
        when the load drives no row to onset)."""
        if not self.evaluated.any():
            return None
        return float(np.min(self.load_factor[self.evaluated]))

    @property
    def interior_critical(self) -> int | None:
        """Row of the largest :attr:`index` among the :attr:`interior` rows."""
        return self._argmax(self.interior)

    @property
    def interior_max_index(self) -> float | None:
        """Largest onset index over the evaluated interior rows."""
        row = self.interior_critical
        return None if row is None else float(self.index[row])

    @property
    def interior_onset_load_factor(self) -> float | None:
        """Smallest :attr:`load_factor` over the evaluated interior rows."""
        sel = self.evaluated & self.interior
        if not sel.any():
            return None
        return float(np.min(self.load_factor[sel]))

    def interface_max_index(self) -> dict[int, float | None]:
        """Largest onset index of each interface (``None`` where none evaluated)."""
        out: dict[int, float | None] = {}
        for k in self.interfaces:
            sel = (self.interface == k) & self.evaluated
            out[k] = float(np.max(self.index[sel])) if sel.any() else None
        return out

    def summary(self) -> dict:
        """Scalar outputs as a JSON-ready dict (non-finite values become ``None``).

        Keys: ``criterion``, ``x_avg_mm``, ``Zc_MPa``, ``load_factor_basis``,
        ``max_index``, ``critical_interface``, ``critical_x_mm``,
        ``critical_y_mm``, ``critical_z_mm``, ``onset_load_factor``,
        ``interior_max_index``, ``interior_onset_load_factor`` and
        ``interface_max_index`` (a dict keyed by the interface as a string).
        """
        def num(v: float | None) -> float | None:
            return None if v is None or not math.isfinite(v) else float(v)

        loc = self.critical_location
        return {
            'criterion': self.criterion,
            'x_avg_mm': float(self.x_avg),
            'Zc_MPa': self.Zc,
            'load_factor_basis': self.load_factor_basis,
            'max_index': num(self.max_index),
            'critical_interface': self.critical_interface,
            'critical_x_mm': None if loc is None else loc[0],
            'critical_y_mm': None if loc is None else loc[1],
            'critical_z_mm': None if loc is None else loc[2],
            'onset_load_factor': num(self.onset_load_factor),
            'interior_max_index': num(self.interior_max_index),
            'interior_onset_load_factor': num(self.interior_onset_load_factor),
            'interface_max_index': {
                str(k): num(v) for k, v in self.interface_max_index().items()},
        }


# ----------------------------------------------------------------------
# Mesh layout
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class _Grid:
    """Rectilinear structured layout of a :class:`CompositeMesh`."""

    xs: np.ndarray        # (nx + 1,) node x coordinates
    ys: np.ndarray        # (ny + 1,) node y coordinates
    zs: np.ndarray        # (nz + 1,) node-layer z coordinates
    nx: int
    ny: int
    nz: int
    n_plies: int


def _grid(mesh: CompositeMesh) -> _Grid:
    """The structured grid of ``mesh``, checked to be rectilinear."""
    nx, ny, nz = int(mesh.nx), int(mesh.ny), int(mesh.nz)
    n0 = (nx + 1) * (ny + 1) * (nz + 1)
    nodes = np.asarray(mesh.nodes, dtype=float)
    elements = np.asarray(mesh.elements)
    if nodes.shape[0] < n0 or elements.shape != (nx * ny * nz, 8):
        raise ValueError(
            f"interface recovery needs the structured {nx} x {ny} x {nz} "
            f"hexahedral mesh of CompositeMesh; got {nodes.shape[0]} nodes "
            f"and elements of shape {elements.shape}.")
    g = nodes[:n0].reshape(nz + 1, ny + 1, nx + 1, 3)
    xs, ys, zs = g[0, 0, :, 0], g[0, :, 0, 1], g[:, 0, 0, 2]
    tol = 1e-9 * max(float(np.ptp(xs)), float(np.ptp(ys)), float(np.ptp(zs)), 1.0)
    if (np.abs(g[..., 0] - xs).max() > tol or np.abs(g[..., 1] - ys[:, None]).max() > tol
            or np.abs(g[..., 2] - zs[:, None, None]).max() > tol):
        raise ValueError(
            "interface recovery needs a rectilinear structured grid (node "
            "x, y and z coordinates on lines); this mesh's nodes are not.")
    return _Grid(xs=xs.copy(), ys=ys.copy(), zs=zs.copy(), nx=nx, ny=ny,
                 nz=nz, n_plies=int(mesh.material.n_plies))


def _interface_layers(grid: _Grid, interfaces: Iterable[int] | None
                      ) -> list[tuple[int, int]]:
    """``(interface k, node layer)`` of the interfaces that are node layers."""
    n = grid.n_plies
    on_layer = [k for k in range(n - 1) if ((k + 1) * grid.nz) % n == 0]
    if interfaces is None:
        chosen = on_layer
    else:
        given = list(interfaces)
        if any(isinstance(k, bool) or not isinstance(k, (int, np.integer))
               for k in given):
            raise ValueError(f"interfaces must be integers, got {given!r}.")
        chosen = sorted({int(k) for k in given})
        bad = [k for k in chosen if not 0 <= k <= n - 2]
        if bad:
            raise ValueError(
                f"interfaces {bad} do not exist: a {n}-ply laminate has "
                f"interfaces 0 to {n - 2} (interface k lies between ply k and "
                f"ply k + 1, from the bottom).")
        off = [k for k in chosen if k not in on_layer]
        if off:
            raise ValueError(
                f"interfaces {off} are not node layers of this mesh (nz = "
                f"{grid.nz}, {n} plies): interface k is a node layer when "
                f"(k + 1) * nz is a multiple of n_plies; use nz a multiple "
                f"of n_plies.")
    if not chosen:
        raise ValueError(
            f"no ply interface of this mesh is a node layer (nz = {grid.nz}, "
            f"{n} plies); use nz a multiple of n_plies.")
    return [(k, (k + 1) * grid.nz // n) for k in chosen]


def _solid_columns(mesh: CompositeMesh, grid: _Grid, layer: int) -> np.ndarray:
    """``(ny, nx)``: elements of element layer ``layer`` the failure criteria check.

    The skip rule of :func:`porosity_fe.fe.failure.evaluate_failure`: mean
    nodal porosity above 0.95, or a geometric void element.
    """
    per = grid.nx * grid.ny
    elems = np.arange(layer * per, (layer + 1) * per)
    _, skip = failure._element_porosity_and_skip(
        np.asarray(mesh.porosity), np.asarray(mesh.elements)[elems])
    void_idx = np.asarray(getattr(mesh, 'void_elements', ()), dtype=np.intp).ravel()
    if void_idx.size:
        skip |= np.isin(elems, void_idx)
    return (~skip).reshape(grid.ny, grid.nx)


def _bonded_faces(mesh: CompositeMesh, grid: _Grid, k: int) -> np.ndarray:
    """``(ny, nx)``: faces of interface ``k`` with all four corner pairs bonded."""
    if k not in tuple(getattr(mesh, 'delaminated_interfaces', ())):
        return np.ones((grid.ny, grid.nx), dtype=bool)
    b = np.asarray(mesh.bonded_pairs(k), dtype=bool).reshape(grid.ny + 1, grid.nx + 1)
    return b[:-1, :-1] & b[1:, :-1] & b[1:, 1:] & b[:-1, 1:]


def _face_corners(stress_global: np.ndarray, grid: _Grid, layer: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Traction at the face corners of node layer ``layer`` from both sides.

    Returns ``(below, above)``, each ``(ny, nx, 2, 2, 3)`` indexed
    ``[j, i, b, a]`` (corner ``(i + a, j + b)``), components
    ``(tau_xz, tau_yz, sigma_zz)``.
    """
    per = grid.nx * grid.ny
    slots = list(_TRACTION_SLOTS)
    lower = stress_global[(layer - 1) * per:layer * per][:, :, slots]   # (per, 8, 3)
    upper = stress_global[layer * per:(layer + 1) * per][:, :, slots]
    top = np.einsum('cg,egk->eck', _GP_TO_TOP, lower)                   # (per, 4, 3)
    bottom = np.einsum('cg,egk->eck', _GP_TO_BOTTOM, upper)
    order = np.array(_CORNER_BA)
    shape = (grid.ny, grid.nx, 2, 2, 3)
    return top[:, order].reshape(shape), bottom[:, order].reshape(shape)


def _face_count(valid: np.ndarray, grid: _Grid) -> np.ndarray:
    """Number of valid faces around each node, ``(ny+1, nx+1)``."""
    count = np.zeros((grid.ny + 1, grid.nx + 1))
    w = valid.astype(float)
    for b in (0, 1):
        for a in (0, 1):
            count[b:grid.ny + b, a:grid.nx + a] += w
    return count


def _nodal(corners: np.ndarray, valid: np.ndarray, grid: _Grid) -> np.ndarray:
    """Average of the valid faces' corner values at each node, ``(ny+1, nx+1, 3)``."""
    total = np.zeros((grid.ny + 1, grid.nx + 1, 3))
    w = valid.astype(float)
    for b in (0, 1):
        for a in (0, 1):
            total[b:grid.ny + b, a:grid.nx + a] += corners[:, :, b, a] * w[..., None]
    count = _face_count(valid, grid)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = total / count[..., None]
    out[count == 0] = np.nan
    return out


def _window_weights(coords: np.ndarray, width: float) -> np.ndarray:
    """Integrals of the 1-D linear shape functions over each node's window.

    The window of node ``p`` has length ``width``, is centred on it and
    shifted inward where it would cross an end of ``coords`` (all of
    ``coords`` when ``width`` exceeds their span). Returns ``(n + 1, n, 2)``:
    ``[p, cell, a]`` is the integral over the window's part in ``cell`` of
    the shape function of the cell's ``a``-th end (0 left, 1 right).
    """
    c0, c1 = coords[:-1], coords[1:]
    h = c1 - c0
    span = float(coords[-1] - coords[0])
    if width >= span:
        lo = np.full(coords.shape, coords[0])
        hi = np.full(coords.shape, coords[-1])
    else:
        lo = np.clip(coords - 0.5 * width, coords[0], coords[-1] - width)
        hi = lo + width
    lo_c = np.maximum(lo[:, None], c0[None, :])
    hi_c = np.minimum(hi[:, None], c1[None, :])
    inside = hi_c > lo_c
    lo_c = np.where(inside, lo_c, c0[None, :])
    hi_c = np.where(inside, hi_c, c0[None, :])
    w0 = ((c1 - lo_c) ** 2 - (c1 - hi_c) ** 2) / (2.0 * h)
    w1 = ((hi_c - c0) ** 2 - (lo_c - c0) ** 2) / (2.0 * h)
    return np.stack([w0, w1], axis=-1) * inside[..., None]


def _window_average(corners: np.ndarray, valid: np.ndarray,
                    wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """Exact window average of the bilinear face fields over the valid faces.

    ``corners`` ``(ny, nx, 2, 2, 3)``, ``valid`` ``(ny, nx)``; ``wx``, ``wy``
    from :func:`_window_weights`. Returns ``(ny+1, nx+1, 3)``, ``NaN`` where
    a window holds no valid face.
    """
    v = np.where(valid[:, :, None, None, None], corners, 0.0)
    num = np.einsum('qjb,pia,jibak->qpk', wy, wx, v, optimize=True)
    den = np.einsum('qjb,pia,ji->qp', wy, wx, valid.astype(float), optimize=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den[..., None]
    out[den <= 0.0] = np.nan
    return out


def _residual_global(results: FieldResults, mesh: CompositeMesh) -> np.ndarray | None:
    """The thermal residual stress of a combined solve in the global frame."""
    if results.load_factor_basis != 'mechanical_with_residual' \
            or results.residual_stress_local is None:
        return None
    local = np.asarray(results.residual_stress_local, dtype=float)
    angles, idx = np.unique(np.asarray(mesh.ply_angles, dtype=float),
                            return_inverse=True)
    # The inverse of the global-to-ply rotation by theta is the rotation by -theta.
    T = np.stack([stress_transformation_3d(np.radians(-a), axis='z')
                  for a in angles])[idx.reshape(-1)]
    return np.einsum('eij,egj->egi', T, local)


def _checked_stress(results: FieldResults, mesh: CompositeMesh) -> np.ndarray:
    stress = np.asarray(results.stress_global, dtype=float)
    n_elem = int(np.asarray(mesh.elements).shape[0])
    if stress.shape != (n_elem, _GP_TO_NODE.shape[1], 6):
        raise ValueError(
            f"results.stress_global has shape {stress.shape}; this mesh needs "
            f"({n_elem}, {_GP_TO_NODE.shape[1]}, 6). Pass the mesh the results "
            f"were computed on.")
    return stress


# ----------------------------------------------------------------------
# Traction recovery
# ----------------------------------------------------------------------

def interface_tractions(results: FieldResults, mesh: CompositeMesh, *,
                        interfaces: Iterable[int] | None = None
                        ) -> InterfaceTractions:
    """Interlaminar traction at the ply interfaces, from both sides.

    At every ply interface that is a node layer of the mesh (every one when
    ``nz`` is a multiple of ``n_plies``) the traction ``t = sigma . e_z =
    (tau_xz, tau_yz, sigma_zz)`` is recovered in the laminate frame from the
    element layer below and the one above. Each element's Gauss-point
    stresses are extrapolated to the shared face with the trilinear
    extrapolation of :func:`~porosity_fe.extrapolate_to_nodes`, and each
    side's face values are averaged at the interface nodes. The result
    reports both sides, their average and their jump. Post-processing only:
    the solve is not changed.

    Parameters
    ----------
    results : FieldResults
        A solve on ``mesh`` (its :attr:`~porosity_fe.FieldResults.stress_global`).
    mesh : CompositeMesh
        The mesh of the solve, with its delaminations as solved.
    interfaces : iterable of int, optional
        Only these interfaces (each must be a node layer); default all
        that are.

    Returns
    -------
    InterfaceTractions
        One row per interface node.

    Raises
    ------
    ValueError
        If the results do not match the mesh, the mesh is not the
        rectilinear structured grid of :class:`~porosity_fe.CompositeMesh`,
        or no requested interface is a node layer.

    Notes
    -----
    On a delaminated interface only faces whose four corner pairs are
    bonded are used: the crack faces are traction free, or carry contact
    pressure, and the extrapolated stress next to the front is not an
    interface traction. A face next to a geometric void element, or an
    element the failure criteria skip (mean porosity above 0.95), is left
    out the same way.

    Examples
    --------
    >>> tr = interface_tractions(result, mesh)
    >>> edge = (tr.interface == 3) & np.isclose(tr.x, mesh.L_x / 2)
    >>> tr.traction[edge, 2]        # sigma_zz across the width at mid-length
    """
    stress = _checked_stress(results, mesh)
    grid = _grid(mesh)
    rows = _assemble_rows(mesh, grid, _interface_layers(grid, interfaces))
    below, above, valid = [], [], []
    for k, layer in rows['layers']:
        lo, up = _face_corners(stress, grid, layer)
        ok = _valid_faces(mesh, grid, k, layer)
        nb, na = _nodal(lo, ok, grid), _nodal(up, ok, grid)
        below.append(nb.reshape(-1, 3))
        above.append(na.reshape(-1, 3))
        valid.append(np.isfinite(nb[..., 0]).ravel())
    b = np.concatenate(below)
    a = np.concatenate(above)
    return InterfaceTractions(
        interface=rows['interface'], node=rows['node'], x=rows['x'], y=rows['y'],
        z=rows['z'], below=b, above=a, traction=0.5 * (b + a), jump=a - b,
        Vp=rows['Vp'], valid=np.concatenate(valid), shape=(grid.ny + 1, grid.nx + 1))


def _valid_faces(mesh: CompositeMesh, grid: _Grid, k: int, layer: int) -> np.ndarray:
    """Bonded faces of interface ``k`` between two solid elements."""
    return (_bonded_faces(mesh, grid, k) & _solid_columns(mesh, grid, layer - 1)
            & _solid_columns(mesh, grid, layer))


def _assemble_rows(mesh: CompositeMesh, grid: _Grid,
                   layers: list[tuple[int, int]]) -> dict:
    """Row identifiers of the interface nodes of ``layers``."""
    per_layer = (grid.nx + 1) * (grid.ny + 1)
    X, Y = np.meshgrid(grid.xs, grid.ys)
    iface, node, x, y, z, vp = [], [], [], [], [], []
    for k, layer in layers:
        ids = layer * per_layer + np.arange(per_layer)
        iface.append(np.full(per_layer, k, dtype=np.intp))
        node.append(ids)
        x.append(X.ravel())
        y.append(Y.ravel())
        z.append(np.full(per_layer, grid.zs[layer]))
        # The field on the interface plane, not the renormalized stiffness
        # field (CompositeMesh.field_porosity).
        vp.append(mesh.field_porosity(ids))
    return {'layers': layers, 'interface': np.concatenate(iface),
            'node': np.concatenate(node), 'x': np.concatenate(x),
            'y': np.concatenate(y), 'z': np.concatenate(z),
            'Vp': np.concatenate(vp)}


# ----------------------------------------------------------------------
# Onset criteria
# ----------------------------------------------------------------------

def _check_length(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
            or not math.isfinite(float(value)) or float(value) < 0.0 \
            or (positive and float(value) == 0.0):
        bound = '> 0' if positive else '>= 0'
        raise ValueError(f"{name} must be a finite number {bound}, got {value!r}.")
    return float(value)


def _interior_rows(mesh: CompositeMesh, grid: _Grid, x: np.ndarray, y: np.ndarray,
                   margin: float | None) -> np.ndarray:
    """Rows at least ``margin`` from every lateral face.

    The default margin is that of the interior failure metrics of the
    in-plane loading modes: ``max(h, 2 element widths)`` in each in-plane
    direction, ``h`` the laminate thickness.
    """
    x0, x1, y0, y1 = grid.xs[0], grid.xs[-1], grid.ys[0], grid.ys[-1]
    if margin is None:
        h = float(grid.zs[-1] - grid.zs[0])
        mx = max(h, 2.0 * (x1 - x0) / max(grid.nx, 1))
        my = max(h, 2.0 * (y1 - y0) / max(grid.ny, 1))
    else:
        mx = my = _check_length('interior_margin', margin)
    return (x >= x0 + mx) & (x <= x1 - mx) & (y >= y0 + my) & (y <= y1 - my)


def _quadratic_index(t: np.ndarray, w0: np.ndarray, w1: np.ndarray,
                     wt: np.ndarray, wc: np.ndarray | float) -> np.ndarray:
    """``w0 t0^2 + w1 t1^2 + (wt if t2 >= 0 else wc) t2^2`` per row."""
    normal = np.where(t[:, 2] >= 0.0, wt, wc) * t[:, 2] ** 2
    return w0 * t[:, 0] ** 2 + w1 * t[:, 1] ** 2 + normal


def _quadratic_load_factor(t_m: np.ndarray, t_th: np.ndarray | None,
                           w0: np.ndarray, w1: np.ndarray, wt: np.ndarray,
                           wc: np.ndarray | float) -> np.ndarray:
    """Smallest ``lam >= 0`` with index ``(t_th + lam t_m) = 1``, per row.

    Without a prestress the index is ``lam^2`` times that of ``t_m`` (the
    normal term keeps its branch for ``lam > 0``), so ``lam = 1 /
    sqrt(index)``. With one, each branch of the normal term is solved on the
    interval of ``lam`` where its sign holds
    (:func:`porosity_fe.fe.failure._piece_load_factor`).
    """
    if t_th is None:
        A = _quadratic_index(t_m, w0, w1, wt, wc)
        return failure._positive_root_offset(A, np.zeros_like(A), 1.0)
    tension = failure._piece_load_factor(t_th, t_m, (2, True),
                                         {0: w0, 1: w1, 2: wt}, {})
    comp_quad = {0: w0, 1: w1}
    if np.any(np.asarray(wc) != 0.0):
        comp_quad[2] = wc  # type: ignore[assignment]
    compression = failure._piece_load_factor(t_th, t_m, (2, False), comp_quad, {})
    return np.minimum(tension, compression)


def _to_ply_axes(t: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Rows ``(tau_xz, tau_yz, sigma_zz)`` in the axes of plies at
    ``angle_deg`` (one per row): ``(tau_13, tau_23, sigma_33)``."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * t[:, 0] + s * t[:, 1], -s * t[:, 0] + c * t[:, 1],
                     t[:, 2]], axis=1)


def _degraded_rows(material: MaterialProperties, radii: tuple, vp: np.ndarray,
                   Zc: float | None, *, criterion: str,
                   porosity_degradation: PorosityDegradation =
                   _DEFAULT_POROSITY_DEGRADATION) -> tuple[np.ndarray, ...]:
    """``(Z_t, S_13, S_23, Z_c)`` per row, degraded at each row's porosity.

    One evaluation per distinct porosity level; ``Z_c`` is ``NaN`` without
    a compression strength. With ``porosity_degradation='directional'``
    the strengths take the through-thickness ratios of
    :func:`porosity_fe.fe.failure._interlaminar_strength_ratios` (``Z_c``
    that of ``Z_t``), with the ``S_13`` stand-in of ``criterion``.
    """
    Zt, S13, S23, Zc_s = (np.full(vp.size, np.nan) for _ in range(4))
    levels, inverse = np.unique(np.clip(vp, 0.0, 1.0), return_inverse=True)
    inverse = inverse.reshape(-1)
    for i, level in enumerate(levels):
        st = failure.degraded_strengths(material, radii, float(level),
                                        porosity_degradation=porosity_degradation)
        zt_o, s13_o = failure.degraded_interlaminar_strengths(
            material, radii, float(level),
            porosity_degradation=porosity_degradation, criterion=criterion)
        sel = inverse == i
        Zt[sel] = st[2] if zt_o is None else zt_o
        S13[sel] = st[5] if s13_o is None else s13_o
        S23[sel] = st[5]
        if Zc is not None:
            r_Zt, _, _ = failure._interlaminar_strength_ratios(
                material, radii, float(level),
                porosity_degradation=porosity_degradation)
            Zc_s[sel] = max(Zc * r_Zt, Calibration.STRENGTH_FLOOR_MPA)
    return Zt, S13, S23, Zc_s


def delamination_onset(results: FieldResults, mesh: CompositeMesh,
                       material: MaterialProperties, *,
                       criterion: OnsetCriterion = 'brewer_lagace',
                       x_avg: float | None = None,
                       Zc: float | None = None,
                       interfaces: Iterable[int] | None = None,
                       interior_margin: float | None = None,
                       void_shape_radii: tuple | None = None,
                       porosity_degradation: PorosityDegradation =
                       _DEFAULT_POROSITY_DEGRADATION,
                       ) -> DelaminationOnset:
    """Stress-based delamination onset at the ply interfaces.

    Recovers the interlaminar traction at the interface nodes
    (:func:`interface_tractions`), averages it over an in-plane window,
    and checks it against porosity-degraded interlaminar strengths.

    ``'brewer_lagace'`` (default), in the laminate frame (Brewer & Lagace
    1988)::

        (<s_zz>/Z_t)^2 [+ (<-s_zz>/Z_c)^2] + (t_xz/S_13)^2 + (t_yz/S_23)^2 = 1

    on the averaged traction, ``<.>`` the positive part. The compression
    term is added only when ``Zc`` is given. ``'ye'``: the quadratic
    criterion attributed to Ye (1988), ``(<s_33>/Z_t)^2 + (t_13/S_13)^2 +
    (t_23/S_23)^2 = 1`` with the shears in the material axes of each of the
    two adjacent plies (the larger index of the two), never with a
    compression term. With the default strengths (``S_13 = S_23 =
    tau_ilss``) the two coincide.

    Parameters
    ----------
    results : FieldResults
        A solve on ``mesh``.
    mesh : CompositeMesh
        The mesh of the solve.
    material : MaterialProperties
        The ply material of the solve (strengths and constituents).
    criterion : {'brewer_lagace', 'ye'}
        Onset criterion, keyword-only.
    x_avg : float, optional
        Side of the square in-plane averaging window, in mm (default one
        ply thickness, ``L_z / n_plies``, after Kim & Soni 1984). It is a
        calibration parameter: the stress at a free edge is singular, and
        the averaged value, so the predicted onset, depends on it. The
        window is centred on each node and shifted inward at the plate
        edges, so at a straight free edge it spans ``x_avg`` from the edge,
        the averaging distance of Brewer & Lagace. ``0`` uses the nodal
        traction without averaging.
    Zc : float, optional
        Pristine interlaminar compressive strength in MPa for the
        ``'brewer_lagace'`` compression term, degraded like the others.
        ``None`` (default): no compression term, so through-thickness
        compression does not drive onset.
    interfaces : iterable of int, optional
        Only these interfaces; default every one that is a node layer.
    interior_margin : float, optional
        Rows at least this far (mm) from every lateral face form the
        interior variant. Default ``max(h, 2 element widths)`` per in-plane
        direction, the margin of the interior failure metrics of the
        in-plane loading modes.
    void_shape_radii : tuple, optional
        Void shape of the strength degradation; default that of
        ``mesh.porosity_field``.
    porosity_degradation : {'isotropic', 'directional'}, optional
        Porosity degradation of the strengths (see
        :data:`porosity_fe.PorosityDegradation`), default ``'isotropic'``:
        pass the value the solve used (``FESolver(...,
        porosity_degradation=...)``); the results do not record it. With
        ``'directional'`` ``Z_t`` (and ``Zc``) take the through-thickness
        ratio ``r_Zt``, ``S_13`` the ``r_S13`` and ``S_23`` the ``r_S23`` of
        :func:`porosity_fe.fe.failure._interlaminar_strength_ratios`
        instead of ``r_m``, so flat voids weaken the interface more; for
        spherical voids nothing changes.

    Returns
    -------
    DelaminationOnset
        Per-node index and load factor, and the critical interface,
        location and load factor (also over the interior).

    Raises
    ------
    ValueError
        For an unknown criterion or ``porosity_degradation``, a negative or
        non-finite ``x_avg`` or ``interior_margin``, a non-positive ``Zc``,
        ``Zc`` with ``'ye'``, or results and mesh that do not match.

    Notes
    -----
    **Strengths.** ``Z_t`` is ``material.sigma_3t``, or ``sigma_2t`` when
    it is ``None``; ``S_13`` is ``material.tau_13``, or ``tau_ilss`` when
    it is ``None``; ``S_23`` is ``tau_ilss``. All are degraded by the
    matrix ratio ``r_m`` of :func:`porosity_fe.fe.failure.degraded_strengths`
    at the porosity of the interface node, like the matrix strengths of the
    ply criteria (by their own ratios with
    ``porosity_degradation='directional'``).

    **Load factor.** Linear analysis: the traction is ``t_th + lam t_m``,
    with ``t_th`` the averaged thermal residual traction of a combined
    solve (``load_factor_basis='mechanical_with_residual'``) and ``0``
    otherwise. The index is then quadratic in ``lam``: without a prestress
    ``lam = 1 / sqrt(index)``; with one, each branch of the normal term is
    solved on the interval where its sign holds, with the rationalized root
    of :func:`porosity_fe.fe.failure._positive_root_offset`. ``0`` where the
    prestress alone has reached onset.

    Examples
    --------
    >>> onset = delamination_onset(result, mesh, material)
    >>> onset.max_index, onset.critical_interface, onset.onset_load_factor
    >>> onset.grid(onset.index, onset.critical_interface)   # (ny+1, nx+1) map
    """
    if criterion not in ONSET_CRITERIA:
        raise ValueError(
            f"Unknown onset criterion {criterion!r}. Use one of {list(ONSET_CRITERIA)}.")
    _check_porosity_degradation(porosity_degradation)
    if Zc is not None:
        if criterion == 'ye':
            raise ValueError(
                "Zc applies to the 'brewer_lagace' compression term; the 'ye' "
                "criterion has none (through-thickness compression does not "
                "drive onset).")
        Zc = _check_length('Zc', Zc, positive=True)
    stress = _checked_stress(results, mesh)
    grid = _grid(mesh)
    t_ply = float(grid.zs[-1] - grid.zs[0]) / grid.n_plies
    width = t_ply if x_avg is None else _check_length('x_avg', x_avg)
    radii = tuple(mesh.porosity_field.void_shape_radii) if void_shape_radii is None \
        else tuple(void_shape_radii)
    layers = _interface_layers(grid, interfaces)
    rows = _assemble_rows(mesh, grid, layers)
    residual = _residual_global(results, mesh)
    layup = np.asarray(mesh.ply_angles, dtype=float).reshape(grid.nz, -1)[:, 0]

    wx = _window_weights(grid.xs, width) if width > 0.0 else None
    wy = _window_weights(grid.ys, width) if width > 0.0 else None

    def averaged(field: np.ndarray, layer: int, ok: np.ndarray) -> np.ndarray:
        lo, up = _face_corners(field, grid, layer)
        mean = 0.5 * (lo + up)
        if wx is None or wy is None:
            return _nodal(mean, ok, grid).reshape(-1, 3)
        return _window_average(mean, ok, wx, wy).reshape(-1, 3)

    t_tot, t_th, bonded, plies = [], [], [], []
    for k, layer in layers:
        ok = _valid_faces(mesh, grid, k, layer)
        t_tot.append(averaged(stress, layer, ok))
        if residual is not None:
            t_th.append(averaged(residual, layer, ok))
        # A node is evaluated only where it is bonded and next to a valid face.
        node_ok = _face_count(ok, grid) > 0
        if k in tuple(getattr(mesh, 'delaminated_interfaces', ())):
            node_ok &= np.asarray(mesh.bonded_pairs(k), dtype=bool).reshape(node_ok.shape)
        bonded.append(node_ok.ravel())
        n = node_ok.size
        plies.append(np.repeat([[layup[layer - 1], layup[layer]]], n, axis=0))
    traction = np.concatenate(t_tot)
    prestress = np.concatenate(t_th) if residual is not None else None
    ply_pair = np.concatenate(plies)
    vp = rows['Vp']
    evaluated = np.concatenate(bonded) & np.isfinite(traction).all(axis=1) \
        & (vp <= VOID_VP_THRESHOLD)
    if prestress is not None:
        evaluated &= np.isfinite(prestress).all(axis=1)

    Zt, S13, S23, Zc_s = _degraded_rows(material, radii, vp, Zc, criterion=criterion,
                                        porosity_degradation=porosity_degradation)
    index = np.full(vp.size, np.nan)
    lam = np.full(vp.size, np.nan)
    e = evaluated
    wt = 1.0 / Zt[e] ** 2
    w13 = 1.0 / S13[e] ** 2
    w23 = 1.0 / S23[e] ** 2
    wc: np.ndarray | float = 0.0 if Zc is None else 1.0 / Zc_s[e] ** 2
    t_e = traction[e]
    t_th_e = None if prestress is None else prestress[e]
    t_m_e = t_e if t_th_e is None else t_e - t_th_e
    if criterion == 'brewer_lagace':
        index[e] = _quadratic_index(t_e, w13, w23, wt, wc)
        lam[e] = _quadratic_load_factor(t_m_e, t_th_e, w13, w23, wt, wc)
    else:
        # The worse of the two plies at the interface, in their own axes.
        idx_sides, lam_sides = [], []
        for side in (0, 1):
            ang = ply_pair[e, side]
            rot_th = None if t_th_e is None else _to_ply_axes(t_th_e, ang)
            idx_sides.append(_quadratic_index(_to_ply_axes(t_e, ang), w13, w23, wt, 0.0))
            lam_sides.append(_quadratic_load_factor(
                _to_ply_axes(t_m_e, ang), rot_th, w13, w23, wt, 0.0))
        index[e] = np.maximum(*idx_sides)
        lam[e] = np.minimum(*lam_sides)

    return DelaminationOnset(
        criterion=criterion, x_avg=width, Zc=Zc,
        load_factor_basis=str(results.load_factor_basis),
        interface=rows['interface'], node=rows['node'], x=rows['x'], y=rows['y'],
        z=rows['z'], traction=traction, index=index, load_factor=lam,
        Zt=Zt, S13=S13, S23=S23, Vp=vp, evaluated=evaluated,
        interior=_interior_rows(mesh, grid, rows['x'], rows['y'], interior_margin),
        shape=(grid.ny + 1, grid.nx + 1))
