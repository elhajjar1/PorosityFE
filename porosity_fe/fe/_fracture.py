"""Fracture outputs of a solve on a delaminated mesh.

The interface forces of every duplicated pair, the virtual-crack-closure
table of the front (:mod:`porosity_fe.fe.vcct`) and, when the material has
toughness data, the mixed-mode toughness and critical load factor of every
front node. A step of :meth:`FESolver.solve
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
from . import vcct
from .assembler import Ties

logger = logging.getLogger("porosity_fe_analysis")


@dataclass(frozen=True)
class FractureOutputs:
    """The delamination fields of :class:`~porosity_fe.FieldResults`."""

    crack_front: vcct.CrackFront | None
    load_factor: float | None
    governing_node: int | None
    interface_forces: np.ndarray


def fracture_outputs(
    mesh: CompositeMesh, material: MaterialProperties, ties: Ties | None, *,
    load: tuple[np.ndarray, np.ndarray],
    prestress: tuple[np.ndarray, np.ndarray] | None = None,
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

    Returns
    -------
    FractureOutputs
        The front table at the applied state (``load`` plus
        ``prestress``), its smallest critical load factor and that node.
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
    if prestress is None:
        front = vcct.fracture_load_factors(front, material)
    else:
        moving = vcct._evaluate(geometry, mesh, load[0],
                                vcct._pair_forces(mesh, load[1], ties))
        fixed = vcct._evaluate(geometry, mesh, prestress[0],
                               vcct._pair_forces(mesh, prestress[1], ties))
        lam = vcct.fracture_load_factors(moving, material, prestress=fixed).lambda_c
        G_c = vcct.critical_energy_release_rate(
            material, front.G_I, front.G_II, front.G_III, Vp=front.Vp)
        front = dataclasses.replace(front, G_c=G_c, lambda_c=lam)
    row = front.governing
    return FractureOutputs(
        crack_front=front, load_factor=front.load_factor,
        governing_node=None if row is None else int(front.node[row]),
        interface_forces=forces)
