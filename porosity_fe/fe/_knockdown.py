"""FE stiffness knockdown: porous over pristine structural stiffness.

The structural stiffness measure of a linear solve, the pristine reference
solve of the same mesh with no porosity or voids, and the module-level cache
of those references, shared across solvers so a porosity sweep on one mesh
solves each reference once. Split out of ``fe/solver.py``
(IMPROVEMENT_PLAN 5.1); :mod:`porosity_fe.fe.solver` re-exports the
underscored names, and :class:`FESolver <porosity_fe.fe.solver.FESolver>`
keeps ``_pristine_stiffness_measure`` and ``_compute_knockdown`` as thin
wrappers.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import logging
from collections import OrderedDict
from dataclasses import astuple

import numpy as np
import scipy.sparse

from .._types import FEFormulation
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..porosity_field import PorosityField
from ._constrained_solve import free_dof_system, solve_constrained
from ._load_cases import (
    _DEFAULT_APPLIED_STRAIN,
    _FLEXURE_PROCEDURE,
    _FORCE_CONTROLLED,
    apply_boundary_conditions,
)
from .assembler import BoundaryHandler, GlobalAssembler

logger = logging.getLogger("porosity_fe_analysis")

#: Pristine stiffness measures (see :func:`_stiffness_measure`) keyed by
#: loading, mesh geometry and material. Shared across solvers so a porosity
#: sweep on one mesh solves each pristine reference once.
_PRISTINE_MEASURE_CACHE: OrderedDict[tuple, float] = OrderedDict()
_PRISTINE_MEASURE_CACHE_MAXSIZE = 64


def _stiffness_measure(loading: str, K: scipy.sparse.spmatrix, u: np.ndarray,
                       applied_strain: float, applied_load: float) -> float:
    """Structural stiffness of a linear solve, independent of load magnitude.

    ``u^T K u / strain^2`` for the displacement-controlled modes (the
    effective modulus times the volume) and ``load^2 / u^T K u`` (inverse
    compliance) for the force-controlled ILSS and flexure bends.
    """
    energy = float(u @ (K @ u))
    if loading in _FORCE_CONTROLLED:
        return applied_load ** 2 / energy
    return energy / applied_strain ** 2


def _pristine_mesh(mesh: CompositeMesh) -> CompositeMesh:
    """Shallow copy of ``mesh`` with zero porosity and no void elements."""
    pristine = copy.copy(mesh)
    pristine.porosity = np.zeros_like(mesh.porosity)
    pristine.void_elements = np.zeros(0, dtype=np.intp)
    pristine.void_element_set = set()
    return pristine


def needs_pristine_reference(mesh: CompositeMesh, loading: str,
                             applied_strain: float | None,
                             applied_load: float) -> bool:
    """Whether :func:`knockdown` needs the pristine reference of ``loading``.

    Not for ``'thermal'`` (no stiffness measure), a pristine mesh (the
    knockdown is exactly 1) or a zero load. ``applied_strain`` is the
    resolved strain of the displacement-controlled modes.
    """
    if loading == 'thermal':
        return False
    porosity = np.asarray(mesh.porosity, dtype=float)
    if not np.any(porosity > 0.0) and np.size(mesh.void_elements) == 0:
        return False
    return (applied_load if loading in _FORCE_CONTROLLED
            else applied_strain) != 0.0


def knockdown(loading: str, porous: float, pristine: float) -> float:
    """Stiffness knockdown ``porous / pristine`` (:func:`_stiffness_measure`).

    For compression, tension and shear the ratio is ``E_x`` or ``G_xy``
    porous / pristine; for the ILSS and flexure bends it is the ratio of the
    beam stiffnesses (inverse compliances). IMPROVEMENT_PLAN 2.4 replaced the
    earlier ratio of signed domain-mean stresses, which read ``sigma_xx``
    for shear, went unstable for the sign-changing ILSS ``tau_xz`` field,
    and was silently clamped to 1. A ratio above 1 is not clamped; it is
    logged as a warning.
    """
    ratio = porous / pristine
    if ratio > 1.0 + 1e-6:
        logger.warning(
            "FE knockdown %.6f exceeds 1 for loading=%r: the porous model "
            "came out stiffer than the pristine one.", ratio, loading)
    return float(ratio)


def pristine_stiffness_measure(
    loading: str, mesh: CompositeMesh, material: MaterialProperties,
    porosity_field: PorosityField, formulation: FEFormulation,
    bc_handler: BoundaryHandler,
) -> float:
    """:func:`_stiffness_measure` of ``mesh`` with no porosity or voids, cached.

    The reference is a direct solve of the same loading and boundary
    conditions on :func:`_pristine_mesh`, at the default strain or load
    (the measure does not depend on its magnitude). Its assembly and
    factorization are local to this call and freed on return. Cached in
    :data:`_PRISTINE_MEASURE_CACHE` by loading (tension and compression
    share one), mesh nodes, connectivity and ply angles, every material
    field, the element formulation and, for the bends, the support and load
    lines.
    """
    # Tension and compression share one linear pristine problem.
    key_loading = 'compression' if loading == 'tension' else loading
    h = hashlib.blake2b(digest_size=16)
    for arr in (mesh.nodes, mesh.elements, mesh.ply_angles):
        a = np.ascontiguousarray(arr)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    # Key on every material field: repr() omits E33, G13, G23, ... .
    key: tuple = (key_loading, h.hexdigest(), astuple(material), formulation)
    if key_loading in _FLEXURE_PROCEDURE:
        # Supports and load lines are not part of the node arrays.
        key += (bc_handler.flexure_geometry(),)
    elif key_loading == 'ilss':
        # Nor are the short-beam supports and loading band. The shallow
        # pristine mesh copy below carries the same ilss_geometry.
        key += (bc_handler.ilss_geometry(),)
    cached = _PRISTINE_MEASURE_CACHE.get(key)
    if cached is not None:
        _PRISTINE_MEASURE_CACHE.move_to_end(key)
        return cached

    pristine = _pristine_mesh(mesh)
    assembler = GlobalAssembler(pristine, material, porosity_field,
                                formulation=formulation)
    strain = _DEFAULT_APPLIED_STRAIN.get(key_loading, -0.01)
    load = -10.0
    constrained, F = apply_boundary_conditions(
        BoundaryHandler(pristine), key_loading, strain, load)
    K0 = assembler.stiffness()
    system = functools.partial(free_dof_system, OrderedDict(), max_entries=1)
    u0, _ = solve_constrained(K0, F, constrained, system=system)
    measure = _stiffness_measure(key_loading, K0, u0, strain, load)

    _PRISTINE_MEASURE_CACHE[key] = measure
    if len(_PRISTINE_MEASURE_CACHE) > _PRISTINE_MEASURE_CACHE_MAXSIZE:
        _PRISTINE_MEASURE_CACHE.popitem(last=False)
    return measure
