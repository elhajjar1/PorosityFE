"""Load cases of :meth:`FESolver.solve <porosity_fe.fe.solver.FESolver.solve>`.

The loading modes and their defaults, the input checks that run before
anything is assembled, the boundary-condition dispatch to
:class:`~porosity_fe.fe.assembler.BoundaryHandler` and the interior region
the interior failure metrics are taken over. Split out of ``fe/solver.py``
(IMPROVEMENT_PLAN 5.1); :mod:`porosity_fe.fe.solver` re-exports the
underscored names.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np

from ..fracture_specimens import _FRACTURE_TEST_NAMES
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from . import flexure, short_beam
from .assembler import BoundaryHandler

logger = logging.getLogger("porosity_fe_analysis")

_DEFAULT_APPLIED_STRAIN = {'compression': -0.01, 'tension': 0.01, 'shear': 0.01}

#: Loading modes :meth:`FESolver.solve` accepts (see :data:`FELoadingMode`).
_FE_LOADINGS: tuple[str, ...] = (
    'compression', 'tension', 'shear', 'ilss', 'flexure_3pt', 'flexure_4pt',
    'thermal', 'dcb', 'enf', 'mmb')

#: Flexure loading modes and their :class:`FlexureGeometry` procedure.
_FLEXURE_PROCEDURE: dict[str, Literal['3pt', '4pt']] = {
    'flexure_3pt': '3pt', 'flexure_4pt': '4pt'}

#: Delamination fracture tests (:mod:`porosity_fe.fracture_specimens`):
#: loaded by ``applied_load`` on the specimen of
#: :attr:`CompositeMesh.fracture_geometry`.
_FRACTURE_TESTS = frozenset(_FRACTURE_TEST_NAMES)

#: Force-controlled modes: loaded by ``applied_load``, stiffness measured as
#: inverse compliance.
_FORCE_CONTROLLED = frozenset({'ilss', *_FLEXURE_PROCEDURE, *_FRACTURE_TESTS})


def _minres_deprecation(caller: str) -> str:
    """Message of the ``solver='minres'`` :class:`DeprecationWarning`."""
    return (f"{caller}(solver='minres') is deprecated and will be removed in "
            "2.0; use solver='cg'. The reduced stiffness K_ff is symmetric "
            "positive definite, so conjugate gradients cover every case and "
            "give more accurate displacements at the same rtol.")


def check_solve_arguments(loading: str, criterion: str, solver: str, *,
                          supported_criteria: tuple[str, ...],
                          penalty_factor: float | None,
                          diag_scale: bool | None) -> None:
    """Deprecation warnings and the loading / criterion checks of ``solve``.

    Called by :meth:`FESolver.solve` before anything else; the warnings are
    attributed to its caller (``stacklevel=3``).
    """
    for name, value in (('penalty_factor', penalty_factor),
                        ('diag_scale', diag_scale)):
        if value is not None:
            warnings.warn(
                f"FESolver.solve({name}=...) is deprecated and has no "
                "effect: prescribed displacements are now imposed "
                "exactly by eliminating the constrained DOFs. It will "
                "be removed in 2.0.",
                DeprecationWarning,
                stacklevel=3,
            )
    if solver == 'minres':
        warnings.warn(_minres_deprecation('FESolver.solve'),
                      DeprecationWarning, stacklevel=3)
    if criterion not in supported_criteria:
        raise ValueError(
            f"Unknown failure_criterion {criterion!r}. "
            f"Use one of {list(supported_criteria)}."
        )
    if loading not in _FE_LOADINGS:
        raise ValueError(
            f"Unknown loading '{loading}'. "
            f"Use one of {'/'.join(_FE_LOADINGS)}."
        )


def _resolve_applied_strain(loading: str, applied_strain: float | None) -> float:
    """Mode-dependent default strain; warn when the sign contradicts the mode."""
    if applied_strain is None:
        return _DEFAULT_APPLIED_STRAIN.get(loading, -0.01)
    if (loading == 'compression' and applied_strain > 0) or \
            (loading == 'tension' and applied_strain < 0):
        logger.warning(
            "loading=%r with applied_strain=%g: the strain sign contradicts "
            "the loading mode, so the solve is really in %s.",
            loading, applied_strain,
            'tension' if applied_strain > 0 else 'compression')
    return float(applied_strain)


def _check_delta_T(loading: str, delta_T: float | None,
                   material: MaterialProperties) -> float | None:
    """Validate ``delta_T`` for ``loading``; ``None`` for a mechanical-only solve.

    The library never infers the temperature change (not even from
    ``material.T_stress_free``): ``loading='thermal'`` without an explicit
    ``delta_T`` raises, and any ``delta_T`` needs the lamina CTEs.
    """
    if delta_T is None:
        if loading == 'thermal':
            raise ValueError(
                "loading='thermal' needs an explicit delta_T: the temperature "
                "change from the stress-free state in K (or deg C, the same "
                "for a difference), negative for a cool-down, e.g. "
                "delta_T = T_service - T_stress_free = 20 - 177 = -157. It "
                "is not inferred from MaterialProperties.T_stress_free.")
        return None
    if isinstance(delta_T, bool) or not isinstance(delta_T, (int, float, np.number)) \
            or not np.isfinite(float(delta_T)):
        raise ValueError(
            f"delta_T must be a finite number of kelvin (a temperature "
            f"difference), got {delta_T!r}.")
    if not material.has_cte:
        from ..materials import MATERIALS
        with_cte = sorted(k for k, m in MATERIALS.items() if m.has_cte)
        raise ValueError(
            "delta_T needs the lamina thermal expansion coefficients, but "
            "this MaterialProperties has none: set alpha_1 and alpha_2 in "
            "1/K (e.g. dataclasses.replace(material, alpha_1=-1.0e-6, "
            f"alpha_2=26e-6)). Presets with CTEs: {with_cte}.")
    return float(delta_T)


def _interior_element_mask(mesh: CompositeMesh,
                           margin: float | None = None) -> np.ndarray:
    """Elements whose centroid lies at least ``margin`` from every lateral face.

    The lateral faces are ``x_min``, ``x_max``, ``y_min`` and ``y_max``.
    The default margin is ``max(h, 2 element widths)`` in each in-plane
    direction, with ``h`` the laminate thickness: wide enough to leave out
    the free-edge interlaminar singularity of a thermal solve.
    """
    if margin is not None and (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float, np.number))
            or not np.isfinite(float(margin)) or float(margin) < 0.0):
        raise ValueError(
            f"interior_margin must be a finite length >= 0 in mm, got {margin!r}.")
    nodes = np.asarray(mesh.nodes, dtype=float)
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    if margin is None:
        h = hi[2] - lo[2]
        mx = max(h, 2.0 * (hi[0] - lo[0]) / max(mesh.nx, 1))
        my = max(h, 2.0 * (hi[1] - lo[1]) / max(mesh.ny, 1))
    else:
        mx = my = float(margin)
    cent = nodes[mesh.elements].mean(axis=1)
    return ((cent[:, 0] >= lo[0] + mx) & (cent[:, 0] <= hi[0] - mx)
            & (cent[:, 1] >= lo[1] + my) & (cent[:, 1] <= hi[1] - my))


def interior_mask(loading: str, mesh: CompositeMesh,
                  bc_handler: BoundaryHandler,
                  margin: float | None) -> np.ndarray:
    """Interior elements of ``loading``'s failure metrics.

    For ``'ilss'``, the ASTM D2344 shear spans, clear of the contact zones
    (supports, loading band) and the side faces
    (:func:`~porosity_fe.fe.short_beam.interior_element_mask`; raises
    :class:`ValueError` without a specimen mesh). For the flexure modes,
    the span clear of a half-disc around each support and load line and of
    the side faces (:func:`~porosity_fe.fe.flexure.interior_element_mask`).
    Otherwise the elements at least ``margin`` from the lateral faces
    (:func:`_interior_element_mask`); for the fracture tests ('dcb',
    'enf', 'mmb') after checking that the mesh is that test's specimen
    (raises :class:`ValueError` otherwise).
    """
    if loading in _FRACTURE_TESTS:
        bc_handler.fracture_geometry(loading)
    if loading == 'ilss':
        return short_beam.interior_element_mask(
            mesh, bc_handler.ilss_geometry(), margin)
    if loading in _FLEXURE_PROCEDURE:
        return flexure.interior_element_mask(
            mesh, bc_handler.flexure_geometry(), _FLEXURE_PROCEDURE[loading],
            margin)
    return _interior_element_mask(mesh, margin)


def apply_boundary_conditions(
    bc_handler: BoundaryHandler, loading: str, applied_strain: float,
    applied_load: float,
) -> tuple[dict[int, float], np.ndarray]:
    """Dirichlet BCs and force vector of ``loading``.

    Dispatches to the matching :class:`BoundaryHandler` builder. The
    displacement-controlled modes ('compression', 'tension', 'shear') use
    ``applied_strain``; the force-controlled ILSS short-beam shear
    (:meth:`BoundaryHandler.ilss_bcs`, on the mesh's
    :attr:`CompositeMesh.ilss_geometry`) and flexure modes use
    ``applied_load``; ``'thermal'`` returns the 3-2-1 supports of
    :meth:`BoundaryHandler.free_bcs` with a zero force vector (the thermal
    load is built by :func:`porosity_fe.fe._thermal.unit_solution`). The
    fracture tests ('dcb', 'enf', 'mmb') use
    :meth:`BoundaryHandler.fracture_bcs` on the mesh's
    :attr:`CompositeMesh.fracture_geometry` with ``applied_load``.

    Returns
    -------
    constrained : dict[int, float]
        Map of constrained DOF index -> prescribed value.
    F : np.ndarray
        Global force vector.

    Raises
    ------
    ValueError
        If ``loading`` is not one of the supported modes.
    """
    if loading == 'thermal':
        return bc_handler.free_bcs()
    if loading == 'compression':
        return bc_handler.compression_bcs(applied_strain)
    if loading == 'tension':
        return bc_handler.tension_bcs(applied_strain)
    if loading == 'shear':
        return bc_handler.shear_bcs(applied_strain)
    if loading == 'ilss':
        return bc_handler.ilss_bcs(applied_load)
    if loading in _FLEXURE_PROCEDURE:
        return bc_handler.flexure_bcs(
            applied_load, procedure=_FLEXURE_PROCEDURE[loading])
    if loading in _FRACTURE_TESTS:
        return bc_handler.fracture_bcs(
            applied_load, geometry=bc_handler.fracture_geometry(loading))
    raise ValueError(
        f"Unknown loading '{loading}'. "
        f"Use one of {'/'.join(_FE_LOADINGS)}."
    )
