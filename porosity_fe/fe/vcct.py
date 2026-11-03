"""Virtual crack closure (VCCT) on an explicit delamination, and its growth criterion.

Energy release rates along the front of a :class:`~porosity_fe.Delamination`
from the tie forces of the bonded front pairs and the openings of the pairs
one element behind (Krueger 2004, the 8-node brick formulas), the mixed-mode
Benzeggagh-Kenane toughness, knocked down by the porosity at each front node
(:class:`~porosity_fe.ToughnessKnockdown`), and the critical load factor. The
duplicated interface nodes, the bonded ties and the front come from
:class:`~porosity_fe.CompositeMesh`; :meth:`FESolver.solve
<porosity_fe.FESolver.solve>` fills :attr:`FieldResults.crack_front
<porosity_fe.FieldResults.crack_front>` with the table built here. See the
theory page "Delamination" for the frame, the tributary widths and the load
factor.

Units: forces in N, lengths in mm, energy release rates and toughness in
N/mm (1 N/mm = 1 kJ/m^2).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize

from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..toughness import (
    ToughnessKnockdown,
    ToughnessKnockdownSpec,
    _log_uncalibrated_once,
    _placeholder_applied,
    _resolve_toughness_knockdown,
)

logger = logging.getLogger("porosity_fe_analysis")

#: In-plane offsets ``(di, dj)`` of the eight grid neighbours of a node, the
#: four edge-sharing ones first.
_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
_N_EDGE = 4


@dataclass(frozen=True, eq=False)
class CrackFront:
    """Virtual-crack-closure table of a delamination front, one row per front node.

    The front nodes are the bonded duplicated node pairs with a delaminated
    pair among their eight in-plane neighbours
    (:meth:`CompositeMesh.crack_front_pairs
    <porosity_fe.CompositeMesh.crack_front_pairs>`), interfaces in increasing
    order, each in structured order. Every array has one entry per front
    node (the vectors one row of three components).

    Each node has a local frame: ``x'`` is the in-plane growth direction, the
    unit vector from the delaminated pair behind it to the node (exactly
    ``+-x`` along a straight front across ``y``, one of eight grid
    directions on a staircase front); ``y' = +z`` is the interface normal;
    ``z' = x' x y'`` runs along the front. With ``F'`` the force the tie
    exerts on the upper crack face (the duplicate) and ``Delta'`` the
    opening (upper minus lower face) of the pair behind, both in that frame,
    the energy release rates are (Krueger 2004, 8-node bricks)::

        G_I = -F'_y Delta'_y / (2 Delta_a b)
        G_II = -F'_x Delta'_x / (2 Delta_a b)
        G_III = -F'_z Delta'_z / (2 Delta_a b)

    with ``Delta_a`` the distance to the pair behind and ``b`` the tributary
    width, so ``Delta_a b`` is the node's tributary area of the interface
    (half at a plate edge, a quarter at a plate corner). A negative mode I
    product is set to zero: the faces behind are closed (contact) and
    release no opening energy. The shear products ``G_II`` and ``G_III``
    keep their sign. One node's can be negative where the local field is
    not self-similar (the mode III product changes sign across the width
    of a multidirectional laminate, for instance). The signed values are
    what makes the sum over the front, ``sum(G_T * Delta_a * b)``, equal
    the energy the crack extension releases: zeroing the negative ones
    over-predicted the total by up to 7 % on a QI short-beam specimen.

    Attributes
    ----------
    node, duplicate : np.ndarray
        Original (lower face) and duplicate (upper face) node of each front
        pair, ``int``.
    pair : np.ndarray
        Index of the pair in :meth:`CompositeMesh.interface_node_pairs
        <porosity_fe.CompositeMesh.interface_node_pairs>`.
    interface : np.ndarray
        Ply interface of the pair.
    behind : np.ndarray
        Original node of the delaminated pair behind, whose opening enters
        the formulas.
    x, y, z : np.ndarray
        Coordinates of the front node (mm).
    direction : np.ndarray
        ``(n, 2)`` in-plane unit vector ``x'`` (growth direction).
    delta_a, width : np.ndarray
        ``Delta_a`` and ``b`` (mm).
    force_local : np.ndarray
        ``(n, 3)`` tie force on the upper face, ``(F'_x, F'_y, F'_z)`` (N).
    opening_local : np.ndarray
        ``(n, 3)`` opening of the pair behind, ``(Delta'_x, Delta'_y,
        Delta'_z)`` (mm): sliding, normal opening, tearing.
    G_I, G_II, G_III, G_T : np.ndarray
        Energy release rates (N/mm); ``G_T`` is their sum.
    psi : np.ndarray
        Mode mixity ``(G_II + G_III) / G_T``; ``NaN`` where ``G_T = 0``.
    Vp : np.ndarray
        Porosity (fraction) at the crack plane under the front node: the
        mean of :attr:`CompositeMesh.porosity
        <porosity_fe.CompositeMesh.porosity>` at the pair's two nodes, the
        porosity field sampled on the interface. The toughness is knocked
        down at it (:class:`~porosity_fe.ToughnessKnockdown`).
    G_c : np.ndarray or None
        Mixed-mode toughness at the node's ``psi`` and ``Vp``
        (:func:`critical_energy_release_rate`, N/mm); ``None`` without
        toughness data.
    lambda_c : np.ndarray or None
        Critical load factor: the multiplier on the load at which ``G_T``
        reaches ``G_c`` (``inf`` where the load releases no energy);
        ``None`` without toughness data. See
        :func:`fracture_load_factors`.
    toughness_knockdown : ToughnessKnockdown or None
        The porosity law that gave ``G_c``; ``None`` without toughness
        data.
    toughness_uncalibrated : bool or None
        ``True`` when an uncalibrated placeholder coefficient of that law
        lowered the toughness (the default exponential law and some front
        node with ``Vp > 0``); ``False`` otherwise, ``None`` without
        toughness data.
    parts : (CrackFront, CrackFront) or None
        The two parts of a front under a prestress, ``(moving, fixed)``:
        the part the load factor scales (the mechanical load) and the
        fixed part (the thermal residual state), each a table on the same
        front nodes without toughness; this table is their sum (``lam =
        1``). Set by :func:`fracture_load_factors` with ``prestress`` and
        by a combined solve (``delta_T`` with a mechanical load), ``None``
        for a front with a single part. Kept so the load factor can be
        re-evaluated for other toughness values without a solve
        (:func:`~porosity_fe.fracture_load_factor_uncertainty`).

    See Also
    --------
    vcct_energy_release_rates, fracture_load_factors
    """

    node: np.ndarray
    duplicate: np.ndarray
    pair: np.ndarray
    interface: np.ndarray
    behind: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    direction: np.ndarray
    delta_a: np.ndarray
    width: np.ndarray
    force_local: np.ndarray
    opening_local: np.ndarray
    G_I: np.ndarray
    G_II: np.ndarray
    G_III: np.ndarray
    G_T: np.ndarray
    psi: np.ndarray
    Vp: np.ndarray
    G_c: np.ndarray | None = None
    lambda_c: np.ndarray | None = None
    toughness_knockdown: ToughnessKnockdown | None = None
    toughness_uncalibrated: bool | None = None
    parts: tuple[CrackFront, CrackFront] | None = None

    def __len__(self) -> int:
        return int(self.node.size)

    @property
    def area(self) -> np.ndarray:
        """Tributary area ``Delta_a b`` of each front node (mm^2)."""
        return self.delta_a * self.width

    @property
    def governing(self) -> int | None:
        """Row of the smallest :attr:`lambda_c`, or ``None`` (no toughness,
        an empty front, or no finite factor)."""
        if self.lambda_c is None or self.lambda_c.size == 0:
            return None
        finite = np.isfinite(self.lambda_c)
        if not finite.any():
            return None
        return int(np.argmin(np.where(finite, self.lambda_c, np.inf)))

    @property
    def load_factor(self) -> float | None:
        """Smallest :attr:`lambda_c` over the front: ``None`` without
        toughness data or front, ``inf`` when no node is driven."""
        if self.lambda_c is None or self.lambda_c.size == 0:
            return None
        return float(np.min(self.lambda_c))

    def to_dict(self) -> dict[str, list | None]:
        """The table as a dict of equal-length lists (JSON-ready).

        Non-finite values (``psi`` where ``G_T = 0``, an infinite
        ``lambda_c``) become ``None``; ``G_c`` and ``lambda_c`` are
        ``None`` columns without toughness data. Keys carry their unit:
        ``x_mm``, ``G_T_N_per_mm`` and so on.
        """
        def col(values: np.ndarray | None) -> list | None:
            if values is None:
                return None
            return [float(v) if math.isfinite(float(v)) else None
                    for v in np.asarray(values, dtype=float)]

        return {
            'node': [int(v) for v in self.node],
            'interface': [int(v) for v in self.interface],
            'x_mm': col(self.x),
            'y_mm': col(self.y),
            'z_mm': col(self.z),
            'direction_x': col(self.direction[:, 0]),
            'direction_y': col(self.direction[:, 1]),
            'delta_a_mm': col(self.delta_a),
            'width_mm': col(self.width),
            'G_I_N_per_mm': col(self.G_I),
            'G_II_N_per_mm': col(self.G_II),
            'G_III_N_per_mm': col(self.G_III),
            'G_T_N_per_mm': col(self.G_T),
            'psi': col(self.psi),
            'Vp': col(self.Vp),
            'G_c_N_per_mm': col(self.G_c),
            'lambda_c': col(self.lambda_c),
        }


@dataclass(frozen=True)
class _FrontGeometry:
    """Front nodes, the pairs behind and their frames; see :class:`CrackFront`."""

    pair: np.ndarray
    behind_pair: np.ndarray
    node: np.ndarray
    duplicate: np.ndarray
    behind_node: np.ndarray
    behind_duplicate: np.ndarray
    interface: np.ndarray
    xyz: np.ndarray
    direction: np.ndarray
    delta_a: np.ndarray
    width: np.ndarray


def _tributary(coords: np.ndarray) -> np.ndarray:
    """Half the spacing to each neighbour on a 1-D node line (mm): the dual
    cell length, half a spacing at either end."""
    w = np.zeros(coords.size, dtype=float)
    if coords.size > 1:
        seg = np.diff(coords)
        w[:-1] += 0.5 * seg
        w[1:] += 0.5 * seg
    return w


def _front_geometry(mesh: CompositeMesh) -> _FrontGeometry | None:
    """Front pairs of every delaminated interface, the pair behind each, the
    growth direction, ``Delta_a`` and the tributary width; ``None`` when
    there is no front.

    The pair behind is the delaminated grid neighbour most aligned with the
    mean direction from the node's delaminated edge-sharing neighbours (the
    diagonal ones when it has none): straight back along a straight front,
    the diagonal at a staircase corner. ``x'`` points from it to the node and
    ``Delta_a`` is their distance, so ``Delta_a b`` is the node's tributary
    area of the interface plane (its dual cell on the structured grid, half
    at a plate edge).
    """
    originals, duplicates = mesh.interface_node_pairs()
    if originals.size == 0:
        return None
    bonded = mesh.bonded_pairs()
    nodes = np.asarray(mesh.nodes, dtype=float)
    nx1, ny1 = mesh.nx + 1, mesh.ny + 1
    layer = nx1 * ny1
    out: dict[str, list[np.ndarray]] = {k: [] for k in (
        'pair', 'behind_pair', 'direction', 'delta_a', 'width', 'interface')}
    # The pairs of the duplicated interfaces follow each other, interfaces
    # in increasing order, each a full node plane in structured order.
    for q, k in enumerate(mesh.delaminated_interfaces):
        sel = q * layer + np.arange(layer)
        b = bonded[sel].reshape(ny1, nx1)
        xyz = nodes[originals[sel]].reshape(ny1, nx1, 3)
        X, Y = xyz[..., 0], xyz[..., 1]
        area = np.outer(_tributary(Y[:, 0]), _tributary(X[0, :]))
        jj, ii = np.nonzero(b)
        n_cand = len(_NEIGHBOURS)
        cand_i = ii[:, None] + np.array([d[0] for d in _NEIGHBOURS])
        cand_j = jj[:, None] + np.array([d[1] for d in _NEIGHBOURS])
        inside = (cand_i >= 0) & (cand_i < nx1) & (cand_j >= 0) & (cand_j < ny1)
        ci, cj = np.clip(cand_i, 0, nx1 - 1), np.clip(cand_j, 0, ny1 - 1)
        open_ = inside & ~b[cj, ci]                         # delaminated neighbour
        front = open_.any(axis=1)
        if not front.any():
            continue
        ii, jj, ci, cj, open_ = ii[front], jj[front], ci[front], cj[front], open_[front]
        vec = np.stack([X[jj, ii][:, None] - X[cj, ci],
                        Y[jj, ii][:, None] - Y[cj, ci]], axis=-1)   # q -> p
        dist = np.linalg.norm(vec, axis=-1)
        unit = vec / np.where(dist > 0.0, dist, 1.0)[..., None]
        edge = np.zeros(n_cand, dtype=bool)
        edge[:_N_EDGE] = True
        use = np.where(open_[:, :_N_EDGE].any(axis=1)[:, None], open_ & edge, open_)
        mean = (unit * use[..., None]).sum(axis=1)
        weak = np.linalg.norm(mean, axis=1) <= 1e-12
        if weak.any():
            # Opposite delaminated neighbours (a one-node ligament): fall back
            # on every delaminated neighbour, then on the first one.
            mean[weak] = (unit[weak] * open_[weak][..., None]).sum(axis=1)
            still = weak & (np.linalg.norm(mean, axis=1) <= 1e-12)
            first = np.argmax(open_, axis=1)
            mean[still] = unit[still, first[still]]
        score = np.einsum('nck,nk->nc', unit, mean)
        score += 1e-9 * edge            # ties: prefer an edge-sharing neighbour
        score[~open_] = -np.inf
        best = np.argmax(score, axis=1)
        rows = np.arange(best.size)
        bi, bj = ci[rows, best], cj[rows, best]
        delta_a = dist[rows, best]
        out['pair'].append(sel[jj * nx1 + ii])
        out['behind_pair'].append(sel[bj * nx1 + bi])
        out['direction'].append(unit[rows, best])
        out['delta_a'].append(delta_a)
        out['width'].append(area[jj, ii] / delta_a)
        out['interface'].append(np.full(best.size, k, dtype=np.intp))
    if not out['pair']:
        return None
    pair = np.concatenate(out['pair'])
    behind = np.concatenate(out['behind_pair'])
    return _FrontGeometry(
        pair=pair, behind_pair=behind,
        node=originals[pair], duplicate=duplicates[pair],
        behind_node=originals[behind], behind_duplicate=duplicates[behind],
        interface=np.concatenate(out['interface']),
        xyz=nodes[originals[pair]],
        direction=np.concatenate(out['direction']),
        delta_a=np.concatenate(out['delta_a']),
        width=np.concatenate(out['width']))


def _local(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Global ``(n, 3)`` vectors in the front frames ``(x', y', z')``."""
    dx, dy = direction[:, 0], direction[:, 1]
    vx, vy, vz = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return np.column_stack([vx * dx + vy * dy, vz, vx * dy - vy * dx])


def _clip_mode_I(work: np.ndarray) -> np.ndarray:
    """``work`` (columns ``x', y', z'``) with a negative mode I (``y'``)
    entry set to zero; the shear entries keep their sign (see
    :class:`CrackFront`)."""
    out = np.array(work, dtype=float, copy=True)
    out[..., 1] = np.maximum(out[..., 1], 0.0)
    return out


def _rates(force_local: np.ndarray, opening_local: np.ndarray,
           area: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(G_I, G_II, G_III)`` from the local force and opening (mode I
    clipped at 0, the shear modes signed)."""
    work = _clip_mode_I(-force_local * opening_local / (2.0 * area[:, None]))
    return work[:, 1], work[:, 0], work[:, 2]


def _mixity(G_I: np.ndarray, G_II: np.ndarray, G_III: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """``G_T`` and ``psi = (G_II + G_III) / G_T`` (``NaN`` where ``G_T = 0``)."""
    G_T = G_I + G_II + G_III
    with np.errstate(invalid='ignore', divide='ignore'):
        psi = np.where(G_T > 0.0, (G_II + G_III) / np.where(G_T > 0.0, G_T, 1.0),
                       np.nan)
    return G_T, psi


def _as_nodal(values: np.ndarray, n_nodes: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size != 3 * n_nodes:
        raise ValueError(
            f"{name} must have 3 entries per node ({3 * n_nodes}), got "
            f"shape {arr.shape}.")
    return arr.reshape(n_nodes, 3)


def _evaluate(geometry: _FrontGeometry, mesh: CompositeMesh,
              displacement: np.ndarray, pair_forces: np.ndarray) -> CrackFront:
    """:class:`CrackFront` of a displacement field and interface forces."""
    u = _as_nodal(displacement, int(np.asarray(mesh.nodes).shape[0]), 'displacement')
    forces = np.asarray(pair_forces, dtype=float)[geometry.pair]
    opening = u[geometry.behind_duplicate] - u[geometry.behind_node]
    force_local = _local(forces, geometry.direction)
    opening_local = _local(opening, geometry.direction)
    area = geometry.delta_a * geometry.width
    G_I, G_II, G_III = _rates(force_local, opening_local, area)
    G_T, psi = _mixity(G_I, G_II, G_III)
    return CrackFront(
        node=geometry.node, duplicate=geometry.duplicate, pair=geometry.pair,
        interface=geometry.interface, behind=geometry.behind_node,
        x=geometry.xyz[:, 0], y=geometry.xyz[:, 1], z=geometry.xyz[:, 2],
        direction=geometry.direction, delta_a=geometry.delta_a,
        width=geometry.width, force_local=force_local,
        opening_local=opening_local, G_I=G_I, G_II=G_II, G_III=G_III,
        G_T=G_T, psi=psi, Vp=_front_porosity(mesh, geometry))


def _front_porosity(mesh: CompositeMesh, geometry: _FrontGeometry) -> np.ndarray:
    """Porosity at the crack plane of every front pair: the mean of the
    porosity field at its original and duplicate node
    (:meth:`CompositeMesh.field_porosity`, without the ``'interface'``
    renormalization of the stiffness field). The duplicate holds a copy of
    the original's value, so this is the field sampled at the interface
    node."""
    return 0.5 * (mesh.field_porosity(geometry.node)
                  + mesh.field_porosity(geometry.duplicate))


def _pair_forces(mesh: CompositeMesh, nodal_forces: np.ndarray,
                 ties: tuple[np.ndarray, np.ndarray, np.ndarray] | None
                 ) -> np.ndarray:
    """Force on the upper crack face of every duplicated pair, from its tie.

    Parameters
    ----------
    mesh : CompositeMesh
        Mesh with duplicated interface nodes.
    nodal_forces : np.ndarray
        ``K u - F`` at every DOF (``(n_dof,)`` or ``(n_nodes, 3)``), before
        the tie forces are folded onto the masters: at a tied DOF it is the
        force its tie exerts on it.
    ties : (tied, master, offset) or None
        The ties of the solve (bonded pairs on all three components, a pair
        in contact on ``z`` only).

    Returns
    -------
    np.ndarray
        ``(n_pairs, 3)`` force (N) the lower face exerts on the upper face
        (the duplicate) through each tied component, in global axes and
        :meth:`CompositeMesh.interface_node_pairs` order; zero for an
        untied component (an open crack face).
    """
    originals, duplicates = mesh.interface_node_pairs()
    out = np.zeros((originals.size, 3), dtype=float)
    if originals.size == 0 or ties is None or len(ties[0]) == 0:
        return out
    r = np.asarray(nodal_forces, dtype=float).ravel()
    master_of = np.full(r.size, -1, dtype=np.intp)
    master_of[np.asarray(ties[0], dtype=np.intp)] = np.asarray(ties[1], dtype=np.intp)
    comp = np.arange(3)
    dup_dofs = 3 * duplicates[:, None] + comp
    orig_dofs = 3 * originals[:, None] + comp
    dup_slave = master_of[dup_dofs] == orig_dofs
    orig_slave = master_of[orig_dofs] == dup_dofs
    out[dup_slave] = r[dup_dofs[dup_slave]]
    out[orig_slave] = -r[orig_dofs[orig_slave]]
    return out


def vcct_energy_release_rates(mesh: CompositeMesh, displacement: np.ndarray,
                              interface_forces: np.ndarray) -> CrackFront | None:
    """Energy release rates along the delamination front by virtual crack closure.

    The front, the frames and the tributary widths come from the mesh
    (:class:`CrackFront`); the tie force of each front pair from
    ``interface_forces`` and the opening of the pair behind from
    ``displacement``. :meth:`FESolver.solve <porosity_fe.FESolver.solve>`
    calls it and stores the table as :attr:`FieldResults.crack_front
    <porosity_fe.FieldResults.crack_front>`; call it to recompute the
    table from a result (``results.displacement``,
    ``results.interface_forces``), or after a solve of your own boundary
    conditions with
    :func:`~porosity_fe.fe._constrained_solve.solve_constrained`.

    Parameters
    ----------
    mesh : CompositeMesh
        The mesh of the solve, with :attr:`CompositeMesh.delaminations
        <porosity_fe.CompositeMesh.delaminations>` as solved.
    displacement : np.ndarray
        Nodal displacements, ``(n_nodes, 3)`` or ``(n_dof,)`` (mm).
    interface_forces : np.ndarray
        ``(n_pairs, 3)`` force on the upper face of each duplicated pair
        (:attr:`FieldResults.interface_forces
        <porosity_fe.FieldResults.interface_forces>`), or the full ``K u -
        F`` vector of a solve whose ties are the mesh's bonded ties
        (``(n_dof,)`` or ``(n_nodes, 3)``), from which the bonded front
        pairs' tie forces are read.

    Returns
    -------
    CrackFront or None
        The table, without toughness (``G_c`` and ``lambda_c`` are
        ``None``; see :func:`fracture_load_factors`). ``None`` when the mesh
        has no delaminated pair, so no front.

    Raises
    ------
    ValueError
        If an array does not match the mesh.

    Examples
    --------
    >>> front = vcct_energy_release_rates(mesh, result.displacement,
    ...                                   result.interface_forces)
    >>> front.G_T.max()                       # N/mm               # doctest: +SKIP
    """
    geometry = _front_geometry(mesh)
    if geometry is None:
        return None
    n_nodes = int(np.asarray(mesh.nodes).shape[0])
    n_pairs = mesh.interface_node_pairs()[0].size
    forces = np.asarray(interface_forces, dtype=float)
    if forces.size == 3 * n_pairs:
        pair_forces = forces.reshape(n_pairs, 3)
    elif forces.size == 3 * n_nodes:
        slaves, masters = mesh.bonded_ties()
        pair_forces = _pair_forces(
            mesh, forces, (slaves, masters, np.zeros(slaves.size)))
    else:
        raise ValueError(
            f"interface_forces must have shape (n_pairs, 3) = ({n_pairs}, 3) "
            f"or 3 entries per node ({3 * n_nodes}), got shape {forces.shape}.")
    return _evaluate(geometry, mesh, displacement, pair_forces)


def _interface_toughness(material: MaterialProperties, Vp: np.ndarray,
                         knockdown: ToughnessKnockdown
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mode I, II and III toughness (N/mm) at interface porosity ``Vp``.

    The pristine values of ``material`` times the porosity factor of
    ``knockdown`` for each mode (``G_IIIc`` takes the mode III factor, which
    follows mode II by default); exactly the pristine values where
    ``Vp = 0``. Logs the uncalibrated-placeholder INFO line once per process
    when a placeholder lowers a value. ``material.has_toughness`` must be
    true.
    """
    Vp = np.asarray(Vp, dtype=float)
    assert material.G_Ic is not None and material.G_IIc is not None
    G_IIIc = material.G_IIIc_eff
    assert G_IIIc is not None
    if _placeholder_applied(knockdown, Vp):
        _log_uncalibrated_once(knockdown)
    return (np.full(Vp.shape, float(material.G_Ic)) * knockdown.factor(Vp, 'I'),
            np.full(Vp.shape, float(material.G_IIc)) * knockdown.factor(Vp, 'II'),
            np.full(Vp.shape, G_IIIc) * knockdown.factor(Vp, 'III'))


def _bk_toughness(toughness: tuple[np.ndarray, np.ndarray, np.ndarray], eta: float,
                  G_I: np.ndarray, G_II: np.ndarray, G_III: np.ndarray) -> np.ndarray:
    """Benzeggagh-Kenane ``G_c`` with Reeder's mode III term, from the
    pure-mode toughness ``(G_Ic, G_IIc, G_IIIc)`` and the mode mix of
    ``(G_I, G_II, G_III)`` (all broadcast); ``NaN`` where ``G_T = 0``."""
    G_Ic, G_IIc, G_IIIc = toughness
    shear = G_II + G_III
    G_T = G_I + shear
    with np.errstate(invalid='ignore', divide='ignore'):
        psi = np.clip(shear / G_T, 0.0, 1.0)
        tear = np.clip(np.where(shear > 0.0,
                                G_III / np.where(shear > 0.0, shear, 1.0), 0.0),
                       0.0, 1.0)
        mix = psi ** eta
        G_c = G_Ic + (G_IIc - G_Ic) * mix + (G_IIIc - G_IIc) * tear * mix
    return np.where(G_T > 0.0, G_c, np.nan)


def _check_toughness(material: MaterialProperties, what: str) -> float:
    """``material.bk_eta``, after checking that the toughness inputs exist."""
    if not material.has_toughness:
        raise ValueError(
            f"{what} needs MaterialProperties.G_Ic, G_IIc "
            "and bk_eta (N/mm, N/mm, dimensionless); "
            f"got G_Ic={material.G_Ic!r}, G_IIc={material.G_IIc!r}, "
            f"bk_eta={material.bk_eta!r}.")
    assert material.bk_eta is not None
    return float(material.bk_eta)


def critical_energy_release_rate(material: MaterialProperties,
                                 G_I: np.ndarray | float,
                                 G_II: np.ndarray | float,
                                 G_III: np.ndarray | float = 0.0, *,
                                 Vp: np.ndarray | float = 0.0,
                                 toughness_knockdown: ToughnessKnockdownSpec = 'exponential',
                                 ) -> np.ndarray:
    """Mixed-mode toughness ``G_c`` (N/mm) at the mode mix of ``(G_I, G_II, G_III)``.

    Benzeggagh-Kenane (1996), with Reeder's (2006) term for a separate mode
    III toughness::

        psi = (G_II + G_III) / G_T
        G_c = G_Ic + (G_IIc - G_Ic) psi^eta
              + (G_IIIc - G_IIc) (G_III / (G_II + G_III)) psi^eta

    which is plain B-K when ``G_IIIc = G_IIc`` (the default,
    ``material.G_IIIc = None``): ``G_Ic`` in pure mode I (``psi = 0``),
    ``G_IIc`` in pure mode II and ``G_IIIc`` in pure mode III. ``psi`` is
    clipped to ``[0, 1]``. Each pure-mode toughness is the material's
    pristine value knocked down by the porosity ``Vp`` (``G_Ic f_I(Vp)``,
    ``G_IIc f_II(Vp)``, ``G_IIIc f_III(Vp)``; see
    :class:`~porosity_fe.ToughnessKnockdown`); ``eta`` does not change.

    Parameters
    ----------
    material : MaterialProperties
        Needs ``G_Ic``, ``G_IIc`` and ``bk_eta``
        (:attr:`MaterialProperties.has_toughness`).
    G_I, G_II, G_III : array_like
        Energy release rates (N/mm), broadcast together; only their ratios
        matter.
    Vp : array_like
        Interface porosity (fraction in ``[0, 1]``) at each point; ``0``
        (the default) gives the pristine toughness.
    toughness_knockdown : ToughnessKnockdown, str, callable or None
        The porosity law (:class:`~porosity_fe.ToughnessKnockdown`):
        default ``'exponential'`` with the uncalibrated placeholders
        ``b_I = 6.2`` and ``b_II = b_III = 14`` per unit ``Vp``; ``None``
        for the pristine toughness at every ``Vp``. The placeholders log
        one INFO line per process the first time they lower a toughness.

    Returns
    -------
    np.ndarray
        ``G_c`` (N/mm); ``NaN`` where ``G_T = 0`` (no mode mix).

    Raises
    ------
    ValueError
        If the material has no toughness data, ``Vp`` is not a fraction in
        ``[0, 1]`` or the law is invalid.
    """
    eta = _check_toughness(material, "The mixed-mode toughness")
    law = _resolve_toughness_knockdown(toughness_knockdown)
    G_I, G_II, G_III, Vp = np.broadcast_arrays(
        np.asarray(G_I, dtype=float), np.asarray(G_II, dtype=float),
        np.asarray(G_III, dtype=float), np.asarray(Vp, dtype=float))
    return _bk_toughness(_interface_toughness(material, Vp, law), eta,
                         G_I, G_II, G_III)


#: Geometric grid that brackets the prestress root, ``scale`` times these
#: (12 % steps over twelve decades), in :func:`_thermal_factor` and
#: :func:`_thermal_factors_batch`.
_ROOT_GRID = np.geomspace(1e-6, 1e6, 241)


def _thermal_factor(F_T: np.ndarray, D_T: np.ndarray, F_M: np.ndarray,
                    D_M: np.ndarray, area: float,
                    toughness: tuple[np.ndarray, np.ndarray, np.ndarray],
                    eta: float) -> float:
    """Critical factor ``lam`` of one node under ``u = u_T + lam u_M``.

    ``toughness`` is the node's pure-mode toughness ``(G_Ic, G_IIc,
    G_IIIc)``, already knocked down by its porosity. Each mode's closure
    work is quadratic in ``lam``; the smallest positive root of ``G_T(lam)
    - G_c(psi(lam))`` is bracketed on a geometric grid and refined with
    :func:`scipy.optimize.brentq`. 0 when the prestress alone reaches the
    toughness; ``inf`` when no load drives the node.
    """
    def rates(lam: float) -> np.ndarray:
        return _clip_mode_I(-(F_T + lam * F_M) * (D_T + lam * D_M) / (2.0 * area))

    def G_c_at(g: np.ndarray) -> float:
        return float(_bk_toughness(toughness, eta, np.asarray(g[1]),
                                   np.asarray(g[0]), np.asarray(g[2])))

    def excess(lam: float) -> float:
        g = rates(lam)
        G_T = float(g.sum())
        if G_T <= 0.0:
            return -math.inf
        return G_T - G_c_at(g)

    if excess(0.0) >= 0.0:
        return 0.0
    g_M = _clip_mode_I(-F_M * D_M / (2.0 * area))
    if g_M.sum() <= 0.0:
        return math.inf
    # Mechanical estimate as the scale of the grid; it brackets the root
    # well inside 1e-6 .. 1e6 of it.
    G_c_M = G_c_at(g_M)
    scale = math.sqrt(G_c_M / float(g_M.sum()))
    grid = scale * _ROOT_GRID
    prev, f_prev = 0.0, excess(0.0)
    for lam in grid:
        f = excess(float(lam))
        if f >= 0.0:
            if f_prev == -math.inf:
                # G_T was zero at prev: bisect on the sign alone.
                lo, hi = prev, float(lam)
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if excess(mid) >= 0.0:
                        hi = mid
                    else:
                        lo = mid
                return hi
            return float(scipy.optimize.brentq(excess, prev, float(lam),
                                               xtol=1e-14 * float(lam), rtol=1e-12))
        prev, f_prev = float(lam), f
    return math.inf


def _rates_at(F_P: np.ndarray, D_P: np.ndarray, F_M: np.ndarray, D_M: np.ndarray,
              area: np.ndarray, lam: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(G_I, G_II, G_III)`` of every front node at ``u = u_P + lam u_M``.

    ``F_P, D_P, F_M, D_M`` are the ``(n, 3)`` local forces and openings of
    the fixed and moving parts, ``area`` the ``(n,)`` tributary areas and
    ``lam`` an array of factors whose last axis is the ``n`` nodes (one row
    per draw, say). The mode I product is clipped at zero, the shear ones
    signed, as in :func:`_rates`; each output has the shape of ``lam``.
    """
    lam = np.asarray(lam, dtype=float)[..., None]
    work = _clip_mode_I(-(F_P + lam * F_M) * (D_P + lam * D_M) / (2.0 * area[:, None]))
    return work[..., 1], work[..., 0], work[..., 2]


def _thermal_factors_batch(prestress: CrackFront, front: CrackFront,
                           toughness: tuple[np.ndarray, np.ndarray, np.ndarray],
                           eta: float | np.ndarray, *,
                           max_bisections: int = 128) -> np.ndarray:
    """Critical factors of every front node for a batch of toughness values.

    The vectorised :func:`_thermal_factor`: ``toughness`` holds the
    pure-mode toughness ``(G_Ic, G_IIc, G_IIIc)`` of ``m`` toughness sets
    at the ``n`` front nodes, each broadcastable to ``(m, n)``, and ``eta``
    is a float or an ``(m, 1)`` array; the front is ``prestress + lam *
    front`` for every set. Returns ``(m, n)``: the smallest positive root
    of ``G_T(lam) = G_c(psi(lam))`` per node and set, 0 where the prestress
    alone reaches the toughness and ``inf`` where the load drives no
    energy.

    The root is bracketed on the same geometric grid as
    :func:`_thermal_factor` and then refined by bisection on the sign of
    ``G_T - G_c`` until the bracket is a few ULP wide (``max_bisections``
    halvings at most), instead of :func:`scipy.optimize.brentq`; the two
    agree to brentq's tolerance (about 1e-12 relative). One toughness set
    at the nominal values reproduces :func:`fracture_load_factors` to that
    tolerance, not bit for bit.
    """
    F_P, D_P = prestress.force_local, prestress.opening_local
    F_M, D_M = front.force_local, front.opening_local
    area = front.area
    n = len(front)
    G_Ic, G_IIc, G_IIIc = (np.asarray(t, dtype=float) for t in toughness)
    # A scalar eta stays a float: ``psi ** eta`` then takes the same path
    # as the scalar root.
    eta_b: float | np.ndarray = (float(eta) if np.ndim(eta) == 0
                                 else np.asarray(eta, dtype=float))
    shape = np.broadcast_shapes(G_Ic.shape, G_IIc.shape, G_IIIc.shape,
                                np.shape(eta_b), (1, n))
    if len(shape) != 2 or shape[1] != n:
        raise ValueError(
            f"toughness arrays must broadcast to (m, {n}) (one row per set, "
            f"one column per front node); got {shape}.")
    tough = (G_Ic, G_IIc, G_IIIc)

    def excess(lam: np.ndarray) -> np.ndarray:
        """``G_T(lam) - G_c(psi(lam))``; ``-inf`` where ``G_T = 0``."""
        G_I, G_II, G_III = _rates_at(F_P, D_P, F_M, D_M, area, lam)
        G_T = G_I + G_II + G_III
        with np.errstate(invalid='ignore'):
            G_c = _bk_toughness(tough, eta_b, G_I, G_II, G_III)
            return np.where(G_T > 0.0, G_T - G_c, -np.inf)

    out = np.full(shape, np.inf)
    # The prestress alone reaches the toughness: the root is 0.
    at_zero = excess(np.zeros(shape)) >= 0.0
    out[at_zero] = 0.0
    # The scale of the grid is the mechanical estimate sqrt(G_c / G_T) of
    # the moving part alone; nodes it does not drive stay at inf.
    G_I_M, G_II_M, G_III_M = _rates(F_M, D_M, area)
    G_T_M = G_I_M + G_II_M + G_III_M
    with np.errstate(invalid='ignore', divide='ignore'):
        G_c_M = _bk_toughness(tough, eta_b, G_I_M, G_II_M, G_III_M)
        scale = np.sqrt(G_c_M / np.where(G_T_M > 0.0, G_T_M, 1.0))
    active = ~at_zero & np.broadcast_to(G_T_M > 0.0, shape)
    scale = np.where(active, scale, 1.0)
    lo = np.zeros(shape)
    hi = np.full(shape, np.inf)
    found = np.zeros(shape, dtype=bool)
    for g in _ROOT_GRID:
        lam = scale * g
        f = excess(lam)
        hit = active & ~found & (f >= 0.0)
        hi[hit] = lam[hit]
        found |= hit
        pending = active & ~found
        lo[pending] = lam[pending]
        if not pending.any():
            break
    # Bisect on the sign in every bracket at once. Brentq would reach the
    # same root; the sign alone also covers a bracket whose lower end has
    # G_T = 0 (excess -inf), as the scalar version's fallback does. An
    # entry without a bracket stays finite (and unused) so the products
    # never meet inf * 0.
    hi[~found] = lo[~found]
    for _ in range(max_bisections):
        mid = 0.5 * (lo + hi)
        up = excess(mid) >= 0.0
        hi = np.where(found & up, mid, hi)
        lo = np.where(found & ~up, mid, lo)
        if not np.any(found & (hi - lo > 4.0 * np.finfo(float).eps * hi)):
            break
    out[found] = hi[found]
    return out


def fracture_load_factors(front: CrackFront, material: MaterialProperties, *,
                          prestress: CrackFront | None = None,
                          toughness_knockdown: ToughnessKnockdownSpec = 'exponential',
                          ) -> CrackFront:
    """Mixed-mode toughness and critical load factor at every front node.

    Without ``prestress`` every quantity scales with the load: the energy
    release rates grow as ``lam^2`` and the mode mix does not change, so the
    factor at which node ``i`` starts to grow is
    ``lambda_c = sqrt(G_c(psi_i) / G_T,i)`` (``inf`` where ``G_T = 0``).
    With crack-face contact this needs the contact zone not to change with
    the load, which holds for proportional loading from a zero gap
    (frictionless contact is positively homogeneous).

    With ``prestress`` (a fixed state such as the thermal residual one) the
    front is ``u = u_P + lam u``: each mode's closure work is quadratic in
    ``lam`` and the mix changes with it, so ``lambda_c`` is the smallest
    positive root of ``G_T(lam) = G_c(psi(lam))``, bracketed on a grid and
    refined with :func:`scipy.optimize.brentq` per node; 0 where the
    prestress alone reaches the toughness. The contact zone of the applied
    state is held for every ``lam``, an approximation (see the theory page).

    The toughness at each node is the material's, knocked down by the
    porosity at the node (``front.Vp``) with ``toughness_knockdown``.

    Parameters
    ----------
    front : CrackFront
        The part that the factor scales (the mechanical load).
    material : MaterialProperties
        Needs ``G_Ic``, ``G_IIc`` and ``bk_eta``.
    prestress : CrackFront, optional
        The fixed part, on the same front nodes.
    toughness_knockdown : ToughnessKnockdown, str, callable or None
        The porosity law (:class:`~porosity_fe.ToughnessKnockdown`):
        default ``'exponential'`` with the uncalibrated placeholders;
        ``None`` for the pristine toughness.

    Returns
    -------
    CrackFront
        With ``G_c``, ``lambda_c``, ``toughness_knockdown`` and
        ``toughness_uncalibrated`` filled; with ``prestress`` the energy
        release rates are those of the sum (``lam = 1``) and ``parts`` is
        ``(front, prestress)``, without it ``parts`` is ``None`` (the
        factor scales the whole of ``front``).

    Raises
    ------
    ValueError
        If the material has no toughness data, ``prestress`` is on other
        nodes, or the law is invalid.
    """
    eta = _check_toughness(material, "The critical load factor")
    law = _resolve_toughness_knockdown(toughness_knockdown)
    record = _toughness_record(law, front.Vp)
    if prestress is None:
        G_c = critical_energy_release_rate(
            material, front.G_I, front.G_II, front.G_III, Vp=front.Vp,
            toughness_knockdown=law)
        with np.errstate(invalid='ignore', divide='ignore'):
            lam = np.where(front.G_T > 0.0,
                           np.sqrt(G_c / np.where(front.G_T > 0.0, front.G_T, 1.0)),
                           np.inf)
        return dataclasses.replace(front, G_c=G_c, lambda_c=lam, parts=None, **record)
    _check_same_front(prestress, front)
    area = front.area
    force = prestress.force_local + front.force_local
    opening = prestress.opening_local + front.opening_local
    G_I, G_II, G_III = _rates(force, opening, area)
    G_T, psi = _mixity(G_I, G_II, G_III)
    toughness = _interface_toughness(material, front.Vp, law)
    lam = np.array([
        _thermal_factor(prestress.force_local[i], prestress.opening_local[i],
                        front.force_local[i], front.opening_local[i],
                        float(area[i]),
                        (toughness[0][i], toughness[1][i], toughness[2][i]), eta)
        for i in range(len(front))], dtype=float)
    G_c = _bk_toughness(toughness, eta, G_I, G_II, G_III)
    return dataclasses.replace(
        front, force_local=force, opening_local=opening, G_I=G_I, G_II=G_II,
        G_III=G_III, G_T=G_T, psi=psi, G_c=G_c, lambda_c=lam,
        parts=(front, prestress), **record)


def _check_same_front(prestress: CrackFront, front: CrackFront) -> None:
    """Raise unless ``prestress`` is a table on the nodes of ``front``."""
    if prestress.node.shape != front.node.shape or \
            not np.array_equal(prestress.node, front.node):
        raise ValueError("prestress must be a CrackFront on the same front nodes.")


def _toughness_record(law: ToughnessKnockdown, Vp: np.ndarray) -> dict:
    """The ``toughness_knockdown`` and ``toughness_uncalibrated`` fields of
    a :class:`CrackFront` whose ``G_c`` came from ``law`` at ``Vp``."""
    return {'toughness_knockdown': law,
            'toughness_uncalibrated': _placeholder_applied(law, Vp)}
