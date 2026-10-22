"""Post-processing of a solved FE displacement field.

Gauss-point stress and strain recovery in the global and ply frames, nodal
reactions and the effective modulus, and the failure metrics
:class:`~porosity_fe.fe.results.FieldResults` reports (failure indices,
first-ply-failure load factors, residual and interior values). Split out of
``fe/solver.py`` (IMPROVEMENT_PLAN 5.1); :class:`FESolver
<porosity_fe.fe.solver.FESolver>` keeps ``_rotate_to_local`` and
``_reactions_and_modulus`` as thin wrappers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse

from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..transforms import strain_transformation_3d, stress_transformation_3d
from . import failure
from ._load_cases import _FLEXURE_PROCEDURE
from .assembler import BoundaryHandler, GlobalAssembler
from .batch import ElementBatch
from .flexure import flexural_modulus

logger = logging.getLogger("porosity_fe_analysis")


def rotate_to_local(ply_angles: np.ndarray, field_global: np.ndarray, *,
                    strain: bool = False) -> np.ndarray:
    """Rotate a ``(n_elem, n_gp, 6)`` global field into the ply axes.

    ``ply_angles`` holds each element's ply angle in degrees. One matrix
    per distinct ply angle. Stress uses ``T_sigma``; engineering strain
    (``strain=True``, with ``gamma_ij = 2 eps_ij`` in slots 3-5) uses
    ``T_epsilon``: ``T_sigma`` applied to engineering strain leaves the
    shear components off by 2x.
    """
    transform = strain_transformation_3d if strain else stress_transformation_3d
    angles, angle_idx = np.unique(
        np.asarray(ply_angles, dtype=float), return_inverse=True)
    T = np.stack([transform(np.radians(a), axis='z')
                  for a in angles])[angle_idx]
    return np.einsum('eij,egj->egi', T, field_global)


def recover_stresses(
    batch: ElementBatch, mesh: CompositeMesh, u: np.ndarray, *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Element stresses and strains in the global and local frames.

    Evaluates stress/strain at the 8 Gauss points of every element from
    the displacement field ``u``, reusing the ``B`` and ``C`` arrays the
    assembler already computed (``batch``), and rotates each into the
    ply-local frame (stress via ``T_sigma``, engineering strain via
    ``T_epsilon``; :func:`rotate_to_local`).

    Returns
    -------
    (stress_global, stress_local, strain_global, strain_local) :
        tuple of np.ndarray
        Each shape ``(n_elem, n_gp, 6)`` in Voigt order
        ``[11, 22, 33, 23, 13, 12]``.
    """
    strain_global = batch.strains(u)
    stress_global = batch.stresses(strain_global)
    if verbose:
        logger.info("  Post-processed %d elements", mesh.n_elements)

    stress_local = rotate_to_local(mesh.ply_angles, stress_global)
    strain_local = rotate_to_local(mesh.ply_angles, strain_global, strain=True)
    return stress_global, stress_local, strain_global, strain_local


def reactions_and_modulus(
    loading: str, K: scipy.sparse.spmatrix, u: np.ndarray,
    F: np.ndarray, applied_strain: float, constrained: dict[int, float],
    *, applied_load: float, mesh: CompositeMesh, assembler: GlobalAssembler,
    bc_handler: BoundaryHandler,
) -> tuple[np.ndarray, float | None]:
    """Nodal reactions and the effective modulus.

    Reactions are ``K u - F`` at the constrained DOFs (including any
    load applied there, which the support carries) and exactly zero at
    the free DOFs, where ``K u - F`` is only the solver residual.

    For the displacement-controlled modes only the prescribed boundary
    moves work through the reactions, so ``u^T K u = sum(R_i u_i)`` is
    twice the strain energy ``0.5 M strain^2 V``, giving ``M``: axial
    ``E_x = P / (A strain)`` for compression/tension, ``G_xy`` for the
    homogeneous pure-shear BCs. For the flexure modes it is the
    shear-corrected flexural modulus from the mid-span deflection under
    ``applied_load`` (:func:`~porosity_fe.fe.flexure.flexural_modulus`);
    ``None`` for ILSS, whose short span makes the deflection about half
    shear (see :attr:`FieldResults.effective_modulus`).
    """
    Ku = K @ u
    residual = Ku - F
    is_fixed = np.zeros(residual.size, dtype=bool)
    is_fixed[np.fromiter(constrained.keys(), dtype=np.intp,
                         count=len(constrained))] = True
    if logger.isEnabledFor(logging.DEBUG) and not is_fixed.all():
        logger.debug(
            "Equilibrium residual at free DOFs: max|K u - F| = %.3e",
            float(np.abs(residual[~is_fixed]).max()))
    residual[~is_fixed] = 0.0
    reactions = residual.reshape(-1, 3)
    if loading in _FLEXURE_PROCEDURE:
        return reactions, flexural_modulus(
            mesh, assembler.element_batch(), u, applied_load,
            bc_handler.flexure_geometry(), _FLEXURE_PROCEDURE[loading])
    if loading == 'ilss' or applied_strain == 0.0:
        return reactions, None
    volume = mesh.L_x * mesh.L_y * mesh.L_z
    modulus = float(u @ Ku) / (applied_strain ** 2 * volume)
    return reactions, modulus


@dataclass(frozen=True)
class FailureMetrics:
    """Failure outputs of one solve (see :class:`FieldResults`)."""

    max_fi: float
    per_element_fi: np.ndarray
    mode_indices: dict[str, float]
    load_factor: float
    residual_max_fi: float | None
    interior_max_fi: float | None
    interior_load_factor: float | None
    mode_maxima: dict[str, float] | None = None
    per_element_mode_fi: dict[str, np.ndarray] | None = None


def failure_metrics(
    mesh: CompositeMesh, material: MaterialProperties,
    void_shape_radii: tuple, criterion: str, *,
    stress_local: np.ndarray, load_stress_local: np.ndarray,
    prestress_local: np.ndarray | None, residual_local: np.ndarray | None,
    interior: np.ndarray, thermal_only: bool, delta_T: float | None,
) -> FailureMetrics:
    """Failure indices and first-ply-failure load factors of a solve.

    ``stress_local`` is the total ply-local stress, ``load_stress_local``
    the part the load factor scales (the mechanical stress, or the whole
    thermal stress for ``loading='thermal'``) on top of the fixed
    ``prestress_local`` (the residual stress of a combined solve, else
    ``None``). ``residual_local`` is the residual stress of a combined
    solve, whose own maximum failure index is reported (for
    ``thermal_only`` it is the total one). ``interior`` masks the elements
    of the interior metrics.

    The per-element failure index is the max over Gauss points (0.0 for
    skipped void elements), ``max_fi`` its overall maximum, and the mode
    breakdown has ``NaN`` entries for Tsai-Wu, which does not separate
    modes (:func:`porosity_fe.fe.failure.evaluate_failure`). For the
    criteria that do, ``per_element_mode_fi`` holds each mode's
    per-element maximum and ``mode_maxima`` its maximum over the model
    (``None`` for Tsai-Wu).
    """
    def evaluate(stress: np.ndarray, mode_fields: dict | None = None
                 ) -> tuple[float, np.ndarray, dict[str, float]]:
        return failure.evaluate_failure(
            stress, mesh.porosity, mesh.elements, material, void_shape_radii,
            criterion, void_elements=mesh.void_elements, mode_fields=mode_fields)

    mode_fields: dict[str, np.ndarray] = {}
    max_fi, per_elem_fi, mode_indices = evaluate(stress_local, mode_fields)
    mode_maxima = ({k: float(v.max()) if v.size else 0.0
                    for k, v in mode_fields.items()} if mode_fields else None)
    lam_elem = failure._element_load_factors(
        load_stress_local, mesh.porosity, mesh.elements, material,
        void_shape_radii, criterion, void_elements=mesh.void_elements,
        prestress_local=prestress_local)
    load_factor = float(lam_elem.min()) if lam_elem.size else float('inf')

    residual_max_fi: float | None = None
    if thermal_only:
        residual_max_fi = max_fi
    elif residual_local is not None:
        residual_max_fi = evaluate(residual_local)[0]
    if residual_max_fi is not None and residual_max_fi >= 1.0:
        logger.warning(
            "The thermal residual stress alone reaches the %s criterion "
            "(max failure index %.3f at delta_T=%g K): the laminate "
            "fails on cool-down before any load, so the first-ply-"
            "failure load factor is %s.", criterion, residual_max_fi, delta_T,
            '<= 1' if thermal_only else '0')

    has_interior = bool(interior.any())
    return FailureMetrics(
        max_fi=max_fi, per_element_fi=per_elem_fi, mode_indices=mode_indices,
        load_factor=load_factor, residual_max_fi=residual_max_fi,
        interior_max_fi=float(per_elem_fi[interior].max()) if has_interior else None,
        interior_load_factor=float(lam_elem[interior].min()) if has_interior else None,
        mode_maxima=mode_maxima,
        per_element_mode_fi=mode_fields or None,
    )
