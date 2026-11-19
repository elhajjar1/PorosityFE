"""Fracture outputs of a solve on a delaminated mesh.

The interface forces of every duplicated pair, the virtual-crack-closure
table of the front (:mod:`porosity_fe.fe.vcct`) and, when the material has
toughness data, the mixed-mode toughness (knocked down by the porosity at
each front node) and critical load factor of every front node. A step of :meth:`FESolver.solve
<porosity_fe.fe.solver.FESolver.solve>`, which calls it only on a mesh with
a delaminated node pair.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np

from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..toughness import ToughnessKnockdown
from . import vcct
from ._load_cases import interior_point_mask
from .assembler import Ties

logger = logging.getLogger("porosity_fe_analysis")


@dataclass(frozen=True)
class FractureOutputs:
    """The delamination fields of :class:`~porosity_fe.FieldResults`."""

    crack_front: vcct.CrackFront | None
    load_factor: float | None
    governing_node: int | None
    interface_forces: np.ndarray
    interior_load_factor: float | None = None
    interior_governing_node: int | None = None


def fracture_outputs(
    mesh: CompositeMesh, material: MaterialProperties, ties: Ties | None, *,
    load: tuple[np.ndarray, np.ndarray],
    prestress: tuple[np.ndarray, np.ndarray] | None = None,
    toughness_knockdown: ToughnessKnockdown | None = None,
    interior_margin: float | None = None,
) -> FractureOutputs:
    """Interface forces, front table and critical load factor of a solve.

    Parameters
    ----------
    mesh : CompositeMesh
        The delaminated mesh.
    material : MaterialProperties
        Its toughness (``G_Ic``, ``G_IIc``, ``bk_eta``) gives ``G_c`` and the
        load factor; without it they are ``None`` (logged once at INFO).
    ties : Ties or None
        The ties of the solve (bonded pairs, plus the normal ties of the
        pairs in contact).
    load : (u, r)
        Displacement and residual ``K u - F`` (tie forces not folded) of
        the part the load factor scales: the mechanical solve, or the
        thermal one for ``loading='thermal'``.
    prestress : (u, r), optional
        The fixed part (the thermal residual state of a combined solve).
    toughness_knockdown : ToughnessKnockdown, optional
        The porosity law of the toughness at the front nodes; ``None`` (the
        default here) takes the default exponential law with its
        uncalibrated placeholders, as :meth:`FESolver.solve
        <porosity_fe.fe.solver.FESolver.solve>` does.

    interior_margin : float, optional
        The lateral margin (mm) of the interior front nodes, as for the
        interior failure metrics (default ``max(h, 2 element widths)`` per
        direction; :func:`~porosity_fe.fe._load_cases.lateral_margins`).

    Returns
    -------
    FractureOutputs
        The front table at the applied state (``load`` plus
        ``prestress``), its smallest critical load factor and that node,
        and the same over the interior front nodes (issue #227).
        With ``prestress`` the table's ``parts`` are the two parts'
        tables (``(moving, fixed)``).
    """
    u, r = load
    if prestress is not None:
        u, r = u + prestress[0], r + prestress[1]
    forces = vcct._pair_forces(mesh, r, ties)
    geometry = vcct._front_geometry(mesh)
    if geometry is None:
        return FractureOutputs(None, None, None, forces)
    front = vcct._evaluate(geometry, mesh, u, forces)
    if not material.has_toughness:
        logger.info(
            "No interlaminar toughness (MaterialProperties.G_Ic, G_IIc, "
            "bk_eta): energy release rates only, no fracture load factor.")
        return FractureOutputs(front, None, None, forces)
    law = ToughnessKnockdown() if toughness_knockdown is None else toughness_knockdown
    if prestress is None:
        front = vcct.fracture_load_factors(front, material, toughness_knockdown=law)
        front = _without_roundoff_release(front, u)
    else:
        moving = vcct._evaluate(geometry, mesh, load[0],
                                vcct._pair_forces(mesh, load[1], ties))
        fixed = vcct._evaluate(geometry, mesh, prestress[0],
                               vcct._pair_forces(mesh, prestress[1], ties))
        lam = vcct.fracture_load_factors(moving, material, prestress=fixed,
                                         toughness_knockdown=law).lambda_c
        G_c = vcct.critical_energy_release_rate(
            material, front.G_I, front.G_II, front.G_III, Vp=front.Vp,
            toughness_knockdown=law)
        # The two parts stay on the table, so the factor can be
        # re-evaluated for other toughness values without a solve.
        front = dataclasses.replace(front, G_c=G_c, lambda_c=lam,
                                    parts=(moving, fixed),
                                    **vcct._toughness_record(law, front.Vp))
    row = front.governing
    interior_factor, interior_node = _interior_factor(mesh, front, interior_margin)
    return FractureOutputs(
        crack_front=front, load_factor=front.load_factor,
        governing_node=None if row is None else int(front.node[row]),
        interface_forces=forces, interior_load_factor=interior_factor,
        interior_governing_node=interior_node)


#: A front node whose G_T is below this fraction of its own tie force times
#: the solve's largest displacement (over its area) releases no energy: its
#: crack-face opening is round-off of the solve, as under a load that does
#: not open the crack by symmetry. Measured: about 1e-16 for a direct solve
#: and 1e-8 for CG at its default tolerance, against 5e-5 (median) to 3e-3
#: (largest) at the front nodes of a delamination that does open.
_ROUNDOFF_RELEASE = 1e-6


def _without_roundoff_release(front: vcct.CrackFront, u: np.ndarray) -> vcct.CrackFront:
    """``front`` with ``lambda_c = inf`` where ``G_T`` is round-off.

    A load that releases no energy gave a finite factor of 1e7 to 1e14 at
    some node (``G_T > 0`` on round-off), and with it a governing node and
    mode mix that mean nothing; ``FieldResults.fracture_load_factor`` is
    documented as ``inf`` then.
    """
    if front.lambda_c is None or len(front) == 0:
        return front
    u_max = float(np.max(np.abs(u))) if np.size(u) else 0.0
    scale = (np.linalg.norm(front.force_local, axis=1) * u_max
             / np.where(front.area > 0.0, front.area, np.inf))
    negligible = np.asarray(front.G_T) <= _ROUNDOFF_RELEASE * scale
    if not negligible.any():
        return front
    lam = np.where(negligible, np.inf, front.lambda_c)
    return dataclasses.replace(front, lambda_c=lam)


def _interior_factor(mesh: CompositeMesh, front: vcct.CrackFront,
                     margin: float | None) -> tuple[float | None, int | None]:
    """Smallest ``lambda_c`` over the front nodes at least the lateral
    margin from every lateral face, and that node.

    At a lateral face (the side faces of a fracture specimen, the edges of
    a coupon a delamination reaches) the energy release rate is mesh
    dependent, and the plain minimum usually sits there. ``(None, None)``
    without toughness data or interior front node; ``(inf, None)`` when no
    interior node is driven, as for the plain minimum.
    """
    lam = front.lambda_c
    if lam is None or len(front) == 0:
        return None, None
    interior = interior_point_mask(mesh, front.x, front.y, margin)
    if not interior.any():
        return None, None
    rows = np.flatnonzero(interior)
    values = lam[rows]
    finite = np.isfinite(values)
    if not finite.any():
        return float(np.min(values)), None
    best = rows[np.argmin(np.where(finite, values, np.inf))]
    return float(lam[best]), int(front.node[best])
