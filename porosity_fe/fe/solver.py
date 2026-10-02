"""FE solver and FieldResults dataclass."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import time
import warnings
from collections import OrderedDict
from dataclasses import astuple, dataclass
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .._ply_angles import _resolve_ply_angles
from .._types import FELoadingMode
from ..materials import MaterialProperties
from ..mesh import CompositeMesh, check_mesh_quality
from ..porosity_field import PorosityField
from ..results import FailureResult
from ..transforms import strain_transformation_3d, stress_transformation_3d
from . import failure
from .assembler import BoundaryHandler, GlobalAssembler, _DirichletPartition
from .export import export_results as _export_results
from .export import write_vtk

logger = logging.getLogger("porosity_fe_analysis")

# ============================================================
# SECTION 7g: FE SOLVER AND FIELD RESULTS
# ============================================================

@dataclass(frozen=True, slots=True)
class FieldResults:
    """Results from a finite element solve.

    Attributes
    ----------
    displacement : np.ndarray
        Shape (n_nodes, 3) nodal displacements.
    stress_global : np.ndarray
        Shape (n_elem, n_gp, 6) stress in global coordinates.
    stress_local : np.ndarray
        Shape (n_elem, n_gp, 6) stress in local (material) coordinates.
    strain_global : np.ndarray
        Shape (n_elem, n_gp, 6) strain in global coordinates.
    strain_local : np.ndarray
        Shape (n_elem, n_gp, 6) strain in local coordinates.
    max_failure_index : float
        Maximum failure index across all Gauss points (criterion-dependent).
    knockdown : float
        Stiffness knockdown factor: porous over pristine structural
        stiffness from a pristine reference solve of the same mesh (``E_x``
        or ``G_xy`` ratio; beam-stiffness ratio for ILSS).
    per_element_failure_index : np.ndarray or None
        Shape (n_elem,) max-over-Gauss-point failure index per element.
        Optional (defaults to ``None`` for back-compatibility with callers
        that construct ``FieldResults`` directly); populated by
        ``FESolver.solve`` and consumed by the VTK export so failure
        hot-spots can be sliced in ParaView.
    failure_criterion : str
        Which failure criterion was used to produce ``max_failure_index`` and
        ``per_element_failure_index``. One of ``'tsai_wu'`` (default for
        back-compat), ``'hashin'``, or ``'max_stress'``.
    failure_mode_indices : dict or None
        Per-mode breakdown of the maximum failure index across the model with
        keys ``'fiber_t'``, ``'fiber_c'``, ``'matrix_t'``, ``'matrix_c'``,
        ``'shear'``, ``'delamination'`` (plus ``'max_fi'``). For Tsai-Wu the per-mode entries are
        ``NaN`` (the polynomial does not separate modes); for ``max_stress``
        the unused entries are zero. Lets the GUI and JSON exporter report
        the dominant failure mode, not just severity.
    reaction_forces : np.ndarray or None
        Shape (n_nodes, 3) nodal reaction forces (N): ``K u - F`` at the
        constrained DOFs and exactly zero at the free DOFs. A load applied
        at a constrained DOF is carried by the support and enters its
        reaction with the opposite sign. Summed over a loaded face they
        give the resultant the boundary conditions apply.
    effective_modulus : float or None
        Homogenized modulus of the specimen (MPa) for the displacement-
        controlled modes: ``E_x`` for ``'compression'`` / ``'tension'``
        and ``G_xy`` for ``'shear'``, from the strain energy
        ``u^T K u / (strain^2 * V)``. ``None`` for the force-controlled
        ``'ilss'`` three-point bend, which has no single modulus.
    first_ply_failure_load_factor : float or None
        Multiplier on the applied load at which ``failure_criterion`` first
        reaches 1 at any Gauss point of a non-void element (linear
        scaling); the margin of safety is this value minus 1. ``inf`` if no
        point is stressed.

    Notes
    -----
    Voigt order for both stress and strain: ``[11, 22, 33, 23, 13, 12]``
    (normals first, then 23 / 13 / 12 shears). The last three strain
    components are **engineering** strain (``gamma_ij = 2 * eps_ij``); the
    last three stress components are the matching shear stresses
    ``[tau_23, tau_13, tau_12]``. Sign convention: tensile normals are
    positive, compressive normals are negative — these arrays are signed
    (unlike the empirical ``failure_stress_MPa`` returned by
    :meth:`EmpiricalSolver.apply_loading`, which is a positive magnitude).
    Use :func:`strain_transformation_3d` / :func:`stress_transformation_3d`
    to rotate these arrays between frames.
    """
    displacement: np.ndarray
    stress_global: np.ndarray
    stress_local: np.ndarray
    strain_global: np.ndarray
    strain_local: np.ndarray
    max_failure_index: float
    knockdown: float
    per_element_failure_index: np.ndarray | None = None
    failure_criterion: str = 'tsai_wu'
    failure_mode_indices: dict[str, float] | None = None
    reaction_forces: np.ndarray | None = None
    effective_modulus: float | None = None
    first_ply_failure_load_factor: float | None = None

    def __repr__(self) -> str:
        n_nodes = self.displacement.shape[0] if self.displacement is not None else 0
        n_elem = self.stress_global.shape[0] if self.stress_global is not None else 0
        return (f"FieldResults(n_nodes={n_nodes}, n_elements={n_elem}, "
                f"max_FI={self.max_failure_index:.4f}, "
                f"knockdown={self.knockdown:.4f})")

    def summary(self, sigma_pristine: float | None = None,
                model_label: str | None = None) -> FailureResult:
        """Distill the field result into a :class:`FailureResult`.

        Unifies the FE return shape with the empirical solver (#44 item 1)
        so callers can treat the two solver outputs polymorphically.

        Parameters
        ----------
        sigma_pristine : float, optional
            Pristine reference stress (MPa) used to compute
            ``failure_stress = knockdown * sigma_pristine``. If ``None``
            (default), the ``failure_stress`` field is set to
            ``knockdown`` itself (unit knockdown — the bare ratio); pass
            the loading-mode-specific pristine strength
            (``material.sigma_1c`` for compression, ``material.tau_ilss``
            for ILSS, etc.) to get a meaningful magnitude.
        model_label : str, optional
            Label used for the ``model`` field. Defaults to
            ``f"fe_{self.failure_criterion}"`` so the FE summary is
            self-describing and distinguishable from the empirical labels.

        Returns
        -------
        FailureResult
            Unified summary with the FE ``knockdown``, derived
            ``failure_stress``, FE-criterion-tagged ``model`` and a
            ``details`` dict carrying ``max_failure_index``,
            ``failure_criterion`` and ``failure_mode_indices`` for
            downstream consumers that need the richer field-result data.
        """
        kd = float(self.knockdown)
        sigma_ref = float(sigma_pristine) if sigma_pristine is not None else 1.0
        return FailureResult(
            failure_stress=kd * sigma_ref,
            knockdown=kd,
            model=str(model_label) if model_label is not None
            else f"fe_{self.failure_criterion}",
            details={
                'max_failure_index': float(self.max_failure_index),
                'failure_criterion': self.failure_criterion,
                'failure_mode_indices': (
                    dict(self.failure_mode_indices)
                    if self.failure_mode_indices is not None else None
                ),
                'first_ply_failure_load_factor': self.first_ply_failure_load_factor,
                'effective_modulus': self.effective_modulus,
            },
        )

    def to_vtk(self, mesh: CompositeMesh,
               filename: str | os.PathLike) -> None:
        """Write the hex mesh and per-element FE fields to a legacy ASCII VTK
        file (``UNSTRUCTURED_GRID``) for inspection in ParaView / VisIt / PyVista.

        The writer is dependency-free: it emits the legacy VTK 3.0 ASCII
        format by hand. The 8-node hex connectivity already stored in
        ``mesh.elements`` follows the standard VTK hexahedron ordering
        (bottom face CCW then top face CCW, see ``_NODE_COORDS_REF``), so the
        cells are written verbatim with cell type 12 (``VTK_HEXAHEDRON``).

        Point data
        -----------
        - ``displacement`` (3-vector), and the scalars ``porosity``,
          ``stiffness_reduction``, ``ply_id`` if present on the mesh.

        Cell data
        ---------
        - element-averaged ``von_mises`` and the six global stress
          (``sigma_xx`` .. ``tau_xy``) and strain (``eps_xx`` .. ``gamma_xy``)
          components (Gauss points reduced by mean),
        - ``tsai_wu_index`` (max-over-GP per element, if available),
        - ``Vp_elem`` (mean nodal porosity over the 8 corners),
        - ``ply_id``, ``ply_angle_deg``, ``is_void`` and ``knockdown`` where
          available.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results (supplies geometry,
            connectivity, porosity and ply metadata).
        filename : str or os.PathLike
            Output ``.vtk`` file path. ``pathlib.Path`` objects are accepted.
        """
        write_vtk(self, mesh, filename)


_DEFAULT_APPLIED_STRAIN = {'compression': -0.01, 'tension': 0.01, 'shear': 0.01}


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
    compliance) for the force-controlled ILSS bend.
    """
    energy = float(u @ (K @ u))
    if loading == 'ilss':
        return applied_load ** 2 / energy
    return energy / applied_strain ** 2


def _pristine_mesh(mesh: CompositeMesh) -> CompositeMesh:
    """Shallow copy of ``mesh`` with zero porosity and no void elements."""
    pristine = copy.copy(mesh)
    pristine.porosity = np.zeros_like(mesh.porosity)
    pristine.void_elements = np.zeros(0, dtype=np.intp)
    pristine.void_element_set = set()
    return pristine


def _relative_residual(A: scipy.sparse.spmatrix, x: np.ndarray,
                       b: np.ndarray) -> float:
    """``||A x - b|| / ||b||``; the absolute residual when ``b = 0``."""
    r = float(np.linalg.norm(A @ x - b))
    norm_b = float(np.linalg.norm(b))
    return r / norm_b if norm_b > 0.0 else r


def _iterative_solve(K_ff: scipy.sparse.spmatrix, rhs: np.ndarray,
                     diag: np.ndarray, *, solver: str, rtol: float,
                     max_restarts: int = 3) -> tuple[np.ndarray, float]:
    """Jacobi-preconditioned CG or MINRES on the reduced SPD system.

    A result is accepted when the true relative residual
    ``||K_ff u - rhs|| / ||rhs||`` is at most ``10 * rtol`` (the slack
    absorbs the floating-point floor). SciPy's CG stops on its recurrence
    residual, which tracks the true one, so one call lands at about
    ``rtol``. SciPy's MINRES stops on ``||r||_M / (||A|| ||x||)``, an
    estimate in the preconditioner norm that sits a roughly constant factor
    (10x to 1000x on these meshes) below the true relative residual. So
    while the true residual is above ``rtol``, MINRES is warm-started from
    its last iterate with the internal tolerance scaled by
    ``rtol / achieved``, at most ``max_restarts`` times (one restart
    suffices on the shipped load cases). MINRES minimizes the residual,
    not the energy-norm error, so at the same residual its displacement
    error is larger than CG's.

    Raises
    ------
    RuntimeError
        On a non-positive diagonal (no Jacobi preconditioner), or when the
        solver stops without meeting the tolerance.
    """
    if not np.all(diag > 0.0):
        raise RuntimeError(
            "Cannot build Jacobi preconditioner: the free-DOF stiffness has "
            "a non-positive diagonal entry. Check the assembly.")
    M = scipy.sparse.diags(1.0 / diag)
    maxiter = max(10 * rhs.size, 100)
    accept = 10.0 * rtol
    restarts = 0
    if solver == 'cg':
        u_f, info = scipy.sparse.linalg.cg(
            K_ff, rhs, M=M, rtol=rtol, maxiter=maxiter)
        rel_res = _relative_residual(K_ff, u_f, rhs)
    else:  # solver == 'minres'
        eps = float(np.finfo(float).eps)
        inner = rtol
        x0: np.ndarray | None = None
        while True:
            u_f, info = scipy.sparse.linalg.minres(
                K_ff, rhs, x0=x0, M=M, rtol=inner, maxiter=maxiter)
            rel_res = _relative_residual(K_ff, u_f, rhs)
            if info != 0 or rel_res <= rtol or restarts >= max_restarts:
                break
            # Below eps the estimate cannot tighten further, but a warm
            # restart from the better iterate can still make progress.
            inner = max(inner * rtol / rel_res, eps)
            x0 = u_f
            restarts += 1
    if not (info == 0 and rel_res <= accept):
        raise RuntimeError(
            f"{solver} failed to converge: info={info}, achieved relative "
            f"residual {rel_res:.4e} (requested rtol={rtol:.4e}"
            + (f", after {restarts} warm restarts" if restarts else "")
            + ").")
    logger.info(
        "%s converged: relative residual %.4e (rtol=%.4e, restarts=%d)",
        solver, rel_res, rtol, restarts)
    return u_f, rel_res


class FESolver:
    """Linear static FE solver for porosity-degraded composite laminates.

    Workflow:
    1. Assemble K via GlobalAssembler
    2. Build BCs via BoundaryHandler
    3. Eliminate the prescribed DOFs: ``K_ff u_f = F_f - K_fc u_c``
    4. Solve the reduced system (sparse LU, or CG / MINRES)
    5. Recover stresses at Gauss points
    6. Evaluate failure criterion at each GP (Tsai-Wu, Hashin, or max-stress)
    7. Compute knockdown factor

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    material : MaterialProperties
        Material properties.
    porosity_field : PorosityField
        Porosity field for stiffness degradation.
    ply_angles : list of float or {'QI', 'UD'}, optional
        Optional list of ply angles (degrees), OR a string sentinel —
        ``'QI'`` (default, ``[0, 90, 45, -45]_s``) or ``'UD'`` (all-zero
        unidirectional). When provided, the resolved angle list is
        forwarded into the underlying :class:`CompositeMesh` per-element
        ``ply_angles`` array via :meth:`CompositeMesh.generate_mesh` only
        if the mesh has not yet been laid up with it — the mesh's
        ``ply_angles`` field remains the authoritative source for the
        per-element transformations used during solve. Passing ``None``
        is deprecated and resolved to ``'QI'`` with a
        :class:`DeprecationWarning` (#44 item 2).
    failure_criterion : {'tsai_wu', 'hashin', 'max_stress'}, optional
        Default failure criterion used by :meth:`solve` when no per-call
        override is supplied. ``'tsai_wu'`` (default) applies the
        quadratic Tsai-Wu interaction polynomial and preserves the
        historical bit-identical behavior. The Tsai-Wu in-plane
        interaction coefficient is ``F_12 = F*_12 * sqrt(F_11 * F_22)``
        with the normalized ``F*_12`` from
        :attr:`MaterialProperties.tsai_wu_F12` when set, otherwise the
        Tsai & Wu (1971) recommendation ``F*_12 = -0.5`` (see
        :meth:`_evaluate_tsai_wu`). ``'hashin'`` uses the Hashin
        2D criterion with separate fiber/matrix tension/compression
        modes, plus a Brewer-Lagace delamination mode for the
        interlaminar stresses. ``'max_stress'`` uses an uncoupled maximum-stress check
        against each lamina strength. Validated against
        :attr:`SUPPORTED_FAILURE_CRITERIA`; an unknown value raises
        :class:`ValueError`.

    Notes
    -----
    ``ply_angles`` defaults — ``'QI'`` is the standardised default across
    :class:`EmpiricalSolver`, :class:`CompositeMesh`, and
    :class:`FESolver` (#44 item 2). The string sentinels expand to
    canonical baselines; explicit lists pass through unchanged. The
    constructor stores the resolved value on ``self.ply_angles`` so
    callers can introspect what layup the solver was built for; the
    actual per-element angles used during ``solve()`` come from
    ``self.mesh.ply_angles``, which is set when the mesh was constructed
    (passing ``ply_angles`` here does *not* relayup the mesh).

    Each element takes the angle of the ply at its centroid, so a mesh
    whose element layers span several plies of different angles (``nz``
    not a multiple of ``n_plies``) analyzes a different laminate. The
    constructor logs a warning, once per mesh, naming the element layup
    and what it loses (see :meth:`CompositeMesh.layup_discrepancies`).
    """

    #: Supported failure criteria for :meth:`solve`. Used both at runtime
    #: (for validation) and as the documented enumeration.
    SUPPORTED_FAILURE_CRITERIA: tuple[str, ...] = failure.SUPPORTED_FAILURE_CRITERIA

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField,
                 ply_angles: list[float] | str | None = 'QI',
                 failure_criterion: Literal[
                     'tsai_wu', 'hashin', 'max_stress'] = 'tsai_wu') -> None:
        self.mesh = mesh
        self.material = material
        self.porosity_field = porosity_field
        # Resolve the ply_angles sentinel (#44 item 2). The resolved value
        # is stored for introspection; the per-element angles consumed by
        # solve() come from ``mesh.ply_angles`` (set by the mesh's own
        # constructor). Previously ``ply_angles`` was stored-but-unused;
        # documenting the intentional decoupling here so the field has an
        # explicit contract.
        self.ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='FESolver.ply_angles')
        # Warn (once per mesh) when the element layers merge plies of
        # different angles, so the solve sees a different laminate.
        mesh._log_layup_warning()
        self.assembler = GlobalAssembler(mesh, material, porosity_field)
        self.bc_handler = BoundaryHandler(mesh)
        if failure_criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {failure_criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )
        self.failure_criterion = failure_criterion
        # Most recent free-DOF stiffness block and its sparse LU
        # factorization (built lazily by direct solves), reused while K and
        # the constrained-DOF set are unchanged: (K, key, K_ff, SuperLU|None).
        self._lu_cache: tuple | None = None

    def solve(self, loading: FELoadingMode = 'compression',
              applied_strain: float | None = None,
              applied_load: float = -10.0,
              verbose: bool = False,
              failure_criterion: Literal['tsai_wu', 'hashin', 'max_stress'] | None = None,
              solver: Literal['direct', 'cg', 'minres'] = 'direct',
              rtol: float = 1e-9,
              diag_scale: bool | None = None,
              penalty_factor: float | None = None) -> FieldResults:
        """Solve the static FE problem.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', or 'ilss'.
        applied_strain : float, optional
            Applied nominal strain (negative for compression). Used by the
            displacement-controlled modes ('compression', 'tension',
            'shear'). Defaults to ``-0.01`` for ``'compression'`` and
            ``+0.01`` for ``'tension'`` and ``'shear'``. (Before 2.3 the
            default was ``-0.01`` for every mode, so ``solve('tension')``
            silently ran a compression solve.) A strain whose sign
            contradicts ``'compression'`` / ``'tension'`` logs a warning.
        applied_load : float
            Total midspan load (force) used by the force-controlled ILSS
            short-beam-shear mode (ASTM D2344). Ignored for the other
            modes.
        verbose : bool
            Print progress information.
        failure_criterion : {'tsai_wu', 'hashin', 'max_stress'}, optional
            Per-call override for the failure criterion. Defaults to the
            value passed to :meth:`__init__` (``'tsai_wu'`` if unset). When
            ``'tsai_wu'`` the result is bit-identical to the historical
            behavior; ``'hashin'`` and ``'max_stress'`` populate the
            per-mode breakdown on :class:`FieldResults`.
        solver : {'direct', 'cg', 'minres'}
            Linear solver for the reduced system ``K_ff u_f = F_f - K_fc u_c``
            left after the prescribed DOFs are eliminated. ``'direct'``
            (default) uses a sparse LU factorization
            (:func:`scipy.sparse.linalg.splu`), cached and reused while
            the mesh, material, porosity and constrained-DOF set are
            unchanged. ``K_ff`` is symmetric positive definite, so
            ``'cg'`` (conjugate gradient with a Jacobi preconditioner) is
            a memory-light alternative; it needs no LU fill-in and is
            the better choice above roughly 40k DOF (about 3x the default
            production mesh), where LU time and memory grow fastest.
            ``'minres'`` is offered for completeness; SciPy's MINRES
            stops on a preconditioned residual estimate, so it is
            warm-restarted with a tighter internal tolerance until the
            true relative residual meets ``rtol``. MINRES minimizes the
            residual rather than the energy-norm error, so at the same
            ``rtol`` its displacements are less accurate than CG's (about
            1e-6 vs 1e-8 relative to direct on the production mesh at the
            default ``rtol``); prefer ``'cg'``. Auto-switching is
            intentionally *not* performed: callers select the path
            explicitly (issue #57).
        rtol : float
            Relative-residual tolerance ``||K_ff u_f - rhs|| / ||rhs||``
            for the iterative solvers; a result is accepted when the true
            residual is at most ``10 * rtol``. Ignored when
            ``solver='direct'``.
        diag_scale : None
            Deprecated, no effect. It Jacobi-scaled the penalty-modified
            system; with exact elimination the reduced matrix is already
            as well conditioned. Passing any value emits a
            :class:`DeprecationWarning`; the argument will be removed in
            2.0.
        penalty_factor : None
            Deprecated, no effect. Prescribed displacements are now
            imposed exactly by eliminating the constrained DOFs instead of
            a penalty stiffness. Passing any value emits a
            :class:`DeprecationWarning`; the argument will be removed in
            2.0.

        Returns
        -------
        FieldResults
            Complete solution data.

        Raises
        ------
        ValueError
            If ``failure_criterion`` is not one of ``'tsai_wu'``,
            ``'hashin'``, ``'max_stress'`` (validated against
            :attr:`SUPPORTED_FAILURE_CRITERIA`), or if ``solver`` is not
            one of ``'direct'``, ``'cg'``, ``'minres'``.
        RuntimeError
            If the iterative solver fails to converge to ``rtol``, if the
            direct solve produces non-finite values / a residual above
            ``1e-6``, or if the free-DOF stiffness is singular (boundary
            conditions that leave a rigid-body mode free).
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
                    stacklevel=2,
                )
        t0 = time.perf_counter()
        criterion = failure_criterion if failure_criterion is not None \
            else self.failure_criterion
        if criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )

        applied_strain = _resolve_applied_strain(loading, applied_strain)

        # 0. Mesh quality check
        check_mesh_quality(self.mesh, verbose=verbose)

        # 1. Global stiffness (re-assembled only if the mesh, material or
        #    porosity changed since the last solve on this solver).
        if verbose:
            logger.info("Assembling global stiffness matrix...")
        K = self.assembler.stiffness(verbose=verbose)

        if verbose:
            t1 = time.perf_counter()
            logger.info("  Assembly time: %.2f s", t1 - t0)

        # 2. Build BCs and the force vector for the requested loading mode.
        constrained, F = self._apply_boundary_conditions(
            loading, applied_strain, applied_load, verbose=verbose)

        # 3-4. Eliminate the prescribed DOFs and solve the reduced system.
        u, _rel_res = self._solve_constrained(
            K, F, constrained, solver=solver, rtol=rtol, verbose=verbose)

        if verbose:
            t2 = time.perf_counter()
            logger.info(
                "  Solve time: %.2f s, residual: %.4e", t2 - t1, _rel_res)
            t1 = t2

        # 5. Recover stresses and strains (global + local frames).
        if verbose:
            logger.info("Recovering element stresses and strains...")
        (stress_global, stress_local,
         strain_global, strain_local) = self._recover_stresses(u, verbose=verbose)

        # 6. Evaluate the selected failure criterion at each GP.
        #    per_elem_fi[e] is the max-over-GP failure index for element e
        #    (0.0 for skipped void elements); the scalar max_fi is its
        #    overall maximum. mode_indices captures the per-mode breakdown
        #    (NaN entries for Tsai-Wu, which does not separate modes).
        max_fi, per_elem_fi, mode_indices = self._evaluate_failure(
            stress_local, criterion=criterion)
        fpf_load_factor = failure.first_ply_failure_load_factor(
            stress_local, self.mesh.porosity, self.mesh.elements,
            self.material, self.porosity_field.void_shape_radii, criterion,
            void_elements=self.mesh.void_elements)

        # 7. Knockdown: porous / pristine structural stiffness.
        knockdown = self._compute_knockdown(
            loading, K, u, applied_strain, applied_load)

        displacement = u.reshape(-1, 3)
        reactions, effective_modulus = self._reactions_and_modulus(
            loading, K, u, F, applied_strain, constrained)

        if verbose:
            t3 = time.perf_counter()
            logger.info("  Post-processing time: %.2f s", t3 - t1)
            logger.info("Total solve time: %.2f s", t3 - t0)
            logger.info("  Max %s FI: %.4f", criterion, max_fi)
            logger.info("  Knockdown factor: %.4f", knockdown)

        return FieldResults(
            displacement=displacement,
            stress_global=stress_global,
            stress_local=stress_local,
            strain_global=strain_global,
            strain_local=strain_local,
            max_failure_index=max_fi,
            knockdown=knockdown,
            per_element_failure_index=per_elem_fi,
            failure_criterion=criterion,
            failure_mode_indices=mode_indices,
            reaction_forces=reactions,
            effective_modulus=effective_modulus,
            first_ply_failure_load_factor=fpf_load_factor,
        )

    def _apply_boundary_conditions(
        self, loading: str, applied_strain: float, applied_load: float,
        *, verbose: bool = False,
    ) -> tuple[dict[int, float], np.ndarray]:
        """Build the Dirichlet BCs and force vector for a loading mode.

        Dispatches to the matching :class:`BoundaryHandler` builder for the
        requested ``loading`` ('compression', 'tension', 'shear', 'ilss').
        The displacement-controlled modes use ``applied_strain``; the
        force-controlled ILSS short-beam-shear mode uses ``applied_load``.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', or 'ilss'.
        applied_strain : float
            Applied nominal strain (used by the displacement-controlled
            modes).
        applied_load : float
            Total midspan load (used by the ILSS mode; ignored otherwise).
        verbose : bool
            Log the number of applied BCs.

        Returns
        -------
        constrained : dict[int, float]
            Map of constrained DOF index -> prescribed value, as returned
            by the :class:`BoundaryHandler` builders.
        F : np.ndarray
            Global force vector.

        Raises
        ------
        ValueError
            If ``loading`` is not one of the four supported modes.
        """
        if loading == 'compression':
            constrained, F = self.bc_handler.compression_bcs(applied_strain)
        elif loading == 'tension':
            constrained, F = self.bc_handler.tension_bcs(applied_strain)
        elif loading == 'shear':
            constrained, F = self.bc_handler.shear_bcs(applied_strain)
        elif loading == 'ilss':
            constrained, F = self.bc_handler.ilss_bcs(applied_load)
        else:
            raise ValueError(
                f"Unknown loading '{loading}'. "
                "Use compression/tension/shear/ilss."
            )

        if verbose:
            logger.info("  Applied %d displacement BCs", len(constrained))

        return constrained, F

    #: Warm restarts allowed after the first MINRES call (see
    #: :func:`_iterative_solve`).
    _MINRES_MAX_RESTARTS = 3

    def _solve_constrained(
        self, K: scipy.sparse.spmatrix, F: np.ndarray, constrained: dict[int, float],
        *, solver: Literal['direct', 'cg', 'minres'] = 'direct',
        rtol: float = 1e-9, verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Impose the Dirichlet BCs exactly and solve ``K u = F``.

        Partitions the DOFs into free (``f``) and constrained (``c``) sets
        (``_DirichletPartition`` in :mod:`porosity_fe.fe.assembler`), solves
        ``K_ff u_f = F_f - K_fc u_c`` with the requested linear solver and
        returns the full vector with ``u_c`` equal to the prescribed values
        bit for bit.

        Parameters
        ----------
        K : scipy.sparse matrix
            Assembled global stiffness matrix (not modified).
        F : np.ndarray
            Global force vector.
        constrained : dict[int, float]
            Constrained-DOF map from :meth:`_apply_boundary_conditions`.
        solver : {'direct', 'cg', 'minres'}
            Linear solver to use. See :meth:`solve` for details.
        rtol : float
            Relative-residual tolerance for the iterative solvers.
        verbose : bool
            Log progress information.

        Returns
        -------
        u : np.ndarray
            Full displacement vector.
        rel_res : float
            Achieved relative residual ``||K_ff u_f - rhs|| / ||rhs||``
            of the reduced system (absolute residual when ``rhs = 0``).

        Raises
        ------
        ValueError
            If ``solver`` is not one of 'direct', 'cg', 'minres'.
        RuntimeError
            On a singular free-DOF stiffness, a non-positive diagonal for
            the preconditioner, a non-finite or non-converged solution, or
            a residual above tolerance.
        """
        if solver not in ('direct', 'cg', 'minres'):
            raise ValueError(
                f"Unknown solver '{solver}'. "
                "Use 'direct', 'cg', or 'minres'."
            )
        part = _DirichletPartition.from_constraints(K.shape[0], constrained)
        if part.free.size == 0:
            return part.lift(), 0.0

        K_ff, lu = self._free_dof_system(K, part, factorize=solver == 'direct')
        rhs = part.reduce_rhs(K, F)

        # Cheap conditioning diagnostic: with exact elimination the ratio
        # reflects the physical stiffness contrast (ply anisotropy, void
        # elements), not a boundary-condition artefact.
        diag = K_ff.diagonal()
        positive = diag[diag > 0.0]
        diag_ratio = float(positive.max() / positive.min()) \
            if positive.size == diag.size else float('inf')
        logger.info(
            "Free-DOF stiffness: diag ratio=%.3e (n_free=%d, n_fixed=%d)",
            diag_ratio, part.free.size, part.fixed.size)
        if verbose:
            logger.info(
                "Solving system (%d free of %d DOFs) with solver='%s'...",
                part.free.size, part.n_dof, solver)

        if solver == 'direct':
            assert lu is not None
            u_f = lu.solve(rhs)
            if not np.isfinite(u_f).all():
                raise RuntimeError(
                    "The sparse LU solve produced non-finite values (NaN or "
                    "Inf). Check the material stiffness and the boundary "
                    "conditions.")
            rel_res = _relative_residual(K_ff, u_f, rhs)
            if rel_res >= 1e-6:
                raise RuntimeError(
                    f"Sparse LU residual {rel_res:.4e} exceeds tolerance "
                    "1e-6. Check the material stiffness and the boundary "
                    "conditions.")
        else:
            u_f, rel_res = _iterative_solve(
                K_ff, rhs, diag, solver=solver, rtol=rtol,
                max_restarts=self._MINRES_MAX_RESTARTS)

        return part.expand(u_f), rel_res

    def _free_dof_system(
        self, K: scipy.sparse.spmatrix, part: _DirichletPartition,
        *, factorize: bool,
    ) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.linalg.SuperLU | None]:
        """``K_ff`` and (when ``factorize``) its sparse LU, cached.

        ``K_ff`` depends only on the assembled ``K`` and the *set* of
        constrained DOFs, not their prescribed values, so e.g. compression
        and tension on the same mesh, or repeat solves of one load case,
        share it and its factorization. Only the most recent one is kept,
        bounding memory; the LU is built only for direct solves.
        """
        cached = self._lu_cache
        if cached is not None and cached[0] is K and cached[1] == part.key:
            K_ff, lu = cached[2], cached[3]
        else:
            K_ff, lu = part.reduce_matrix(K), None
        if factorize and lu is None:
            # K_ff is symmetric: a symmetric fill-reducing ordering with
            # diagonal pivoting factors ~20% faster than SuperLU's default
            # COLAMD here.
            try:
                lu = scipy.sparse.linalg.splu(
                    K_ff, permc_spec='MMD_AT_PLUS_A',
                    options={'SymmetricMode': True})
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Sparse LU factorization of the free-DOF stiffness "
                    f"failed ({exc}). A singular matrix usually means the "
                    "boundary conditions leave a rigid-body mode "
                    "unrestrained.") from exc
        self._lu_cache = (K, part.key, K_ff, lu)
        return K_ff, lu

    def _reactions_and_modulus(
        self, loading: str, K: scipy.sparse.spmatrix, u: np.ndarray,
        F: np.ndarray, applied_strain: float, constrained: dict[int, float],
    ) -> tuple[np.ndarray, float | None]:
        """Nodal reactions and the strain-energy effective modulus.

        Reactions are ``K u - F`` at the constrained DOFs (including any
        load applied there, which the support carries) and exactly zero at
        the free DOFs, where ``K u - F`` is only the solver residual.

        For the displacement-controlled modes only the prescribed boundary
        moves work through the reactions, so ``u^T K u = sum(R_i u_i)`` is
        twice the strain energy ``0.5 M strain^2 V``, giving ``M``: axial
        ``E_x = P / (A strain)`` for compression/tension, ``G_xy`` for the
        homogeneous pure-shear BCs.
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
        if loading == 'ilss' or applied_strain == 0.0:
            return reactions, None
        volume = self.mesh.L_x * self.mesh.L_y * self.mesh.L_z
        modulus = float(u @ Ku) / (applied_strain ** 2 * volume)
        return reactions, modulus

    def _recover_stresses(
        self, u: np.ndarray, *, verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Recover element stresses and strains in global and local frames.

        Evaluates stress/strain at the 8 Gauss points of every element from
        the recovered displacement field, reusing the ``B`` and ``C`` arrays
        the assembler already computed, and rotates each into the ply-local
        frame (stress via ``T_sigma``, engineering strain via
        ``T_epsilon``).

        Parameters
        ----------
        u : np.ndarray
            Flat global displacement vector.
        verbose : bool
            Log per-element post-processing progress.

        Returns
        -------
        (stress_global, stress_local, strain_global, strain_local) :
            tuple of np.ndarray
            Each shape ``(n_elem, n_gp, 6)`` in Voigt order
            ``[11, 22, 33, 23, 13, 12]``.
        """
        batch = self.assembler.element_batch()
        strain_global = batch.strains(u)
        stress_global = batch.stresses(strain_global)
        if verbose:
            logger.info("  Post-processed %d elements", self.mesh.n_elements)

        # Transform to local coordinates, one pair of matrices per distinct
        # ply angle. Stress uses T_sigma; engineering strain (with
        # gamma_ij = 2*eps_ij in slots 3-5) uses T_epsilon — T_sigma applied
        # to engineering strain leaves the shear components off by 2x.
        angles, angle_idx = np.unique(
            np.asarray(self.mesh.ply_angles, dtype=float), return_inverse=True)
        T_sigma = np.stack([stress_transformation_3d(np.radians(a), axis='z')
                            for a in angles])[angle_idx]
        T_eps = np.stack([strain_transformation_3d(np.radians(a), axis='z')
                          for a in angles])[angle_idx]
        stress_local = np.einsum('eij,egj->egi', T_sigma, stress_global)
        strain_local = np.einsum('eij,egj->egi', T_eps, strain_global)

        return stress_global, stress_local, strain_global, strain_local

    def _compute_knockdown(
        self, loading: str, K: scipy.sparse.spmatrix, u: np.ndarray,
        applied_strain: float, applied_load: float,
    ) -> float:
        """Stiffness knockdown: porous over pristine structural stiffness.

        The pristine reference is a second solve of the same mesh, loading
        and boundary conditions with zero porosity and no void elements
        (cached across solvers by mesh geometry, see
        :meth:`_pristine_stiffness_measure`). For compression, tension and
        shear the ratio is ``E_x`` or ``G_xy`` porous / pristine; for the
        ILSS bend it is the ratio of the beam stiffnesses (inverse
        compliances). IMPROVEMENT_PLAN 2.4 replaced the earlier ratio of
        signed domain-mean stresses, which read ``sigma_xx`` for shear, went
        unstable for the sign-changing ILSS ``tau_xz`` field, and was
        silently clamped to 1.

        Returns 1.0 for a pristine mesh or a zero load. A ratio above 1 is
        not clamped; it is logged as a warning.
        """
        porosity = np.asarray(self.mesh.porosity, dtype=float)
        if not np.any(porosity > 0.0) and np.size(self.mesh.void_elements) == 0:
            return 1.0
        if (applied_load if loading == 'ilss' else applied_strain) == 0.0:
            return 1.0
        porous = _stiffness_measure(loading, K, u, applied_strain, applied_load)
        pristine = self._pristine_stiffness_measure(loading)
        knockdown = porous / pristine
        if knockdown > 1.0 + 1e-6:
            logger.warning(
                "FE knockdown %.6f exceeds 1 for loading=%r: the porous model "
                "came out stiffer than the pristine one.", knockdown, loading)
        return float(knockdown)

    def _pristine_stiffness_measure(self, loading: str) -> float:
        """:func:`_stiffness_measure` of this mesh with no porosity or voids."""
        # Tension and compression share one linear pristine problem.
        key_loading = 'compression' if loading == 'tension' else loading
        h = hashlib.blake2b(digest_size=16)
        for arr in (self.mesh.nodes, self.mesh.elements, self.mesh.ply_angles):
            a = np.ascontiguousarray(arr)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
        # Key on every material field: repr() omits E33, G13, G23, ... .
        key = (key_loading, h.hexdigest(), astuple(self.material))
        cached = _PRISTINE_MEASURE_CACHE.get(key)
        if cached is not None:
            _PRISTINE_MEASURE_CACHE.move_to_end(key)
            return cached

        pristine = FESolver(_pristine_mesh(self.mesh), self.material,
                            self.porosity_field, ply_angles=self.ply_angles)
        strain = _DEFAULT_APPLIED_STRAIN.get(key_loading, -0.01)
        load = -10.0
        constrained, F = pristine._apply_boundary_conditions(
            key_loading, strain, load)
        K0 = pristine.assembler.stiffness()
        u0, _ = pristine._solve_constrained(K0, F, constrained)
        measure = _stiffness_measure(key_loading, K0, u0, strain, load)

        _PRISTINE_MEASURE_CACHE[key] = measure
        if len(_PRISTINE_MEASURE_CACHE) > _PRISTINE_MEASURE_CACHE_MAXSIZE:
            _PRISTINE_MEASURE_CACHE.popitem(last=False)
        return measure

    #: Empty per-mode failure-index dict (see :mod:`porosity_fe.fe.failure`).
    _EMPTY_MODE_FI: dict[str, float] = failure.EMPTY_MODE_FI

    def _degraded_strengths(self, elem_Vp: float
                            ) -> tuple[float, float, float, float, float, float]:
        """Porosity-degraded ply strengths; see :func:`failure.degraded_strengths`."""
        return failure.degraded_strengths(
            self.material, self.porosity_field.void_shape_radii, elem_Vp)

    def _evaluate_failure(self, stress_local: np.ndarray,
                          criterion: str = 'tsai_wu'
                          ) -> tuple[float, np.ndarray, dict[str, float]]:
        """Failure indices for every Gauss point; see :func:`failure.evaluate_failure`."""
        return failure.evaluate_failure(
            stress_local, self.mesh.porosity, self.mesh.elements,
            self.material, self.porosity_field.void_shape_radii, criterion,
            void_elements=self.mesh.void_elements)

    def _evaluate_tsai_wu(self, s_all: np.ndarray,
                          strengths: tuple[float, float, float, float, float, float],
                          e: int, elem_Vp: float) -> np.ndarray:
        """Tsai-Wu index for one element; see :func:`failure.evaluate_tsai_wu`."""
        return failure.evaluate_tsai_wu(
            s_all, strengths, e, elem_Vp, self.material.tsai_wu_F12)

    def _evaluate_hashin(self, s_all: np.ndarray,
                         strengths: tuple[float, float, float, float, float, float]
                         ) -> dict[str, np.ndarray]:
        """Hashin mode indices for one element; see :func:`failure.evaluate_hashin`."""
        return failure.evaluate_hashin(s_all, strengths)

    def _evaluate_max_stress(self, s_all: np.ndarray,
                             strengths: tuple[float, float, float, float, float, float]
                             ) -> dict[str, np.ndarray]:
        """Max-stress mode indices; see :func:`failure.evaluate_max_stress`."""
        return failure.evaluate_max_stress(s_all, strengths)

    @staticmethod
    def export_results(field_results: FieldResults,
                       filename: str | os.PathLike,
                       fmt: str = 'json',
                       mesh: CompositeMesh | None = None,
                       include_raw: bool = False) -> None:
        """Export FE results to a JSON summary or a VTK field file.

        With ``fmt='json'`` (the default, unchanged legacy behavior) this
        saves displacement statistics, stress/strain summaries, failure data,
        and knockdown factor; large arrays are summarized (min/max/mean/std)
        rather than stored in full.

        With ``fmt='vtk'`` it delegates to :meth:`FieldResults.to_vtk` and
        writes the full hex mesh plus per-element fields as a legacy ASCII
        ``UNSTRUCTURED_GRID`` for ParaView / VisIt / PyVista. The richer
        per-element/per-node API lives on ``FieldResults.to_vtk`` directly;
        this ``fmt='vtk'`` path is a convenience shim for callers that
        already hold an ``FESolver``.

        Parameters
        ----------
        field_results : FieldResults
            Results from FESolver.solve().
        filename : str or os.PathLike
            Output file path (``.json`` or ``.vtk``). ``pathlib.Path``
            objects are accepted.
        fmt : str
            ``'json'`` (default) or ``'vtk'``.
        mesh : CompositeMesh, optional
            Required when ``fmt='vtk'`` (supplies geometry/connectivity).
        include_raw : bool
            When ``True`` (and ``fmt='json'``), also write a sidecar
            ``<filename>.npz`` containing the raw displacement/stress/strain
            arrays so a full audit can re-derive the per-key summary
            statistics. Default ``False`` so existing outputs are not
            bloated (#55).
        """
        _export_results(field_results, filename, fmt=fmt, mesh=mesh,
                        include_raw=include_raw)
