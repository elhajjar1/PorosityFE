"""Free-standing thermal solve: the cure residual stress of the FE model.

The layup check a thermal solve needs, and the solve for ``delta_T = 1`` K
with statically determinate 3-2-1 supports, which :meth:`FESolver.solve
<porosity_fe.fe.solver.FESolver.solve>` caches and scales by ``delta_T``
(alone for ``loading='thermal'``, superposed on the mechanical solve
otherwise). Split out of ``fe/solver.py`` (IMPROVEMENT_PLAN 5.1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse

from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from .assembler import _MESH_TIES, Ties, _fold_tied, _mesh_ties
from .batch import ElementBatch, assemble_load_vector, thermal_stress_moduli

logger = logging.getLogger("porosity_fe_analysis")


@dataclass(frozen=True)
class _ThermalUnitSolution:
    """Free-standing thermal solve for ``delta_T = 1`` K (scale by ``delta_T``).

    ``strain_global`` is the total strain ``B u`` (plus the hex8i
    internal-mode part) and ``stress_global = C strain - beta``.
    ``residual`` is ``K u - F`` before the tie forces are folded (the force
    each tie carries, for the virtual crack closure of a delaminated mesh);
    ``None`` on a mesh without ties.
    """

    u: np.ndarray
    strain_global: np.ndarray
    stress_global: np.ndarray
    reactions: np.ndarray
    alpha_local: np.ndarray
    residual: np.ndarray | None = None


def check_layup_for_thermal(mesh: CompositeMesh, material: MaterialProperties,
                            formulation: str,
                            allow_unresolved_layup: bool) -> None:
    """Refuse (or warn about) a thermal solve on an unresolved layup.

    Residual stresses come from the ply-to-ply mismatch, so an element
    layup that merges plies (``nz`` not a multiple of ``n_plies``)
    gives wrong stresses and spurious warping, not just a less
    accurate answer. Also warns when the resolved element layup is
    unsymmetric: the free laminate then warps, which linear theory
    captures only while the deflection is small next to the thickness.
    """
    issues = mesh.layup_discrepancies()
    layup = ", ".join(f"{a:g}" for a in mesh.element_layup)
    n_plies = material.n_plies
    if issues:
        msg = (f"Thermal residual stresses depend on the stacking "
               f"sequence, and the FE mesh does not resolve the requested "
               f"{n_plies}-ply layup: {'. '.join(issues)}. The element "
               f"layup is [{layup}] (bottom to top).")
        if not allow_unresolved_layup:
            raise ValueError(
                f"{msg} Use nz = k * n_plies (e.g. nz={n_plies}) so every "
                f"ply has its own element layer, or pass "
                f"allow_unresolved_layup=True to analyze the element "
                f"layup anyway.")
        logger.warning("%s Analyzing it anyway "
                       "(allow_unresolved_layup=True).", msg)
    angles = np.asarray(mesh.element_layup, dtype=float)
    angles = (angles + 90.0) % 180.0 - 90.0   # fold to [-90, 90)
    if not np.allclose(angles, angles[::-1], atol=1e-6):
        logger.warning(
            "The element layup [%s] is unsymmetric, so the free laminate "
            "warps under delta_T. The solve is linear: it is valid only "
            "while the warping deflection stays small next to the "
            "laminate thickness (larger shapes, e.g. bistable ones, need "
            "geometric nonlinearity)%s.", layup,
            "" if formulation == 'hex8i' else
            ", and the 'hex8' element locks in bending and underpredicts "
            "the curvature (formulation='hex8i' does not)")


def unit_solution(
    K: scipy.sparse.spmatrix, batch: ElementBatch, mesh: CompositeMesh,
    supports: tuple[dict[int, float], np.ndarray], alpha_local: np.ndarray, *,
    solve: Callable[[scipy.sparse.spmatrix, np.ndarray, dict[int, float]],
                    tuple[np.ndarray, float]],
    verbose: bool = False, ties: Ties | None | object = _MESH_TIES,
) -> _ThermalUnitSolution:
    """Free-standing thermal solve for ``delta_T = 1`` K.

    Builds the thermal load ``F = sum B^T C alpha det(J) w`` from the
    element batch (:func:`~porosity_fe.fe.batch.thermal_stress_moduli`),
    solves with ``supports``, the 3-2-1 supports and zero force vector of
    :meth:`BoundaryHandler.free_bcs` (``solve(K, F, constrained)``, the
    solver's constrained solve) and
    recovers ``sigma = C (B u - alpha)``, with the hex8i internal-mode
    strain included. The analysis is linear, so every field at ``delta_T``
    is ``delta_T`` times this one.

    The supports are statically determinate, so the reactions must
    vanish; a reaction above ``1e-6`` of the largest nodal thermal
    force is logged as a warning (it would point to a support bug).
    On a delaminated mesh ``solve`` ties the bonded interface (and the
    pairs in crack-face contact), and the residual of each tied DOF is
    folded onto its master before the reactions are read; ``ties`` are
    those of ``solve``, by default the mesh's bonded ties.
    """
    beta = thermal_stress_moduli(batch, mesh, alpha_local)
    F_th = assemble_load_vector(batch, batch.thermal_loads(beta), mesh.n_dof)
    constrained, F0 = supports
    F = F0 + F_th
    if verbose:
        logger.info("Solving the free-standing thermal problem "
                    "(3-2-1 supports, unit delta_T)...")
    u, _ = solve(K, F, constrained)
    strain = batch.strains(u) + batch.incompatible_mode_strains(beta)
    stress = batch.stresses(strain) - beta

    tie_set = _mesh_ties(mesh) if ties is _MESH_TIES else ties
    assert tie_set is None or isinstance(tie_set, tuple)
    unfolded = K @ u - F
    residual = _fold_tied(unfolded, tie_set)
    fixed = np.fromiter(constrained.keys(), dtype=np.intp,
                        count=len(constrained))
    reactions = np.zeros_like(residual)
    reactions[fixed] = residual[fixed]
    f_max = float(np.abs(F_th).max()) if F_th.size else 0.0
    r_max = float(np.abs(reactions).max())
    if r_max > 1e-6 * f_max + 1e-12:
        logger.warning(
            "Thermal solve: support reactions %.3e N against nodal thermal "
            "forces up to %.3e N (per K). The 3-2-1 supports should carry "
            "no load; check the boundary conditions.", r_max, f_max)
    return _ThermalUnitSolution(
        u=u, strain_global=strain, stress_global=stress,
        reactions=reactions.reshape(-1, 3), alpha_local=alpha_local,
        residual=None if tie_set is None else unfolded)
