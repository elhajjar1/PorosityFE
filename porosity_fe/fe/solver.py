"""FE solver and FieldResults dataclass."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
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
from ..transforms import rotate_stiffness_3d, strain_transformation_3d, stress_transformation_3d
from . import failure
from .assembler import BoundaryHandler, GlobalAssembler
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
        Stiffness knockdown factor (modulus ratio: E_porous/E_pristine).
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
        ``'shear'`` (plus ``'max_fi'``). For Tsai-Wu the per-mode entries are
        ``NaN`` (the polynomial does not separate modes); for ``max_stress``
        the unused entries are zero. Lets the GUI and JSON exporter report
        the dominant failure mode, not just severity.
    reaction_forces : np.ndarray or None
        Shape (n_nodes, 3) nodal reaction forces ``K u - F`` (N). Non-zero
        only at constrained DOFs, up to solver residual; summed over a
        loaded face they give the resultant the boundary conditions apply.
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


class FESolver:
    """Linear static FE solver for porosity-degraded composite laminates.

    Workflow:
    1. Assemble K via GlobalAssembler
    2. Build BCs via BoundaryHandler
    3. Apply penalty method
    4. Solve K*u = F via spsolve
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
        interaction coefficient ``F_12`` is taken from
        :attr:`MaterialProperties.tsai_wu_F12` when set; otherwise the
        Tsai & Wu (1971) recommendation
        ``F_12 = -0.5 * sqrt(F_11 * F_22)`` is used (see
        :meth:`_evaluate_tsai_wu`). ``'hashin'`` uses the Hashin
        2D criterion with separate fiber/matrix tension/compression
        modes. ``'max_stress'`` uses an uncoupled maximum-stress check
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
        self.assembler = GlobalAssembler(mesh, material, porosity_field)
        self.bc_handler = BoundaryHandler(mesh)
        if failure_criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {failure_criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )
        self.failure_criterion = failure_criterion
        # Most recent sparse LU factorization, reused by direct solves whose
        # penalty-modified matrix is unchanged: (K, key, SuperLU).
        self._lu_cache: tuple | None = None

    def solve(self, loading: FELoadingMode = 'compression',
              applied_strain: float = -0.01,
              applied_load: float = -10.0,
              verbose: bool = False,
              failure_criterion: Literal['tsai_wu', 'hashin', 'max_stress'] | None = None,
              solver: Literal['direct', 'cg', 'minres'] = 'direct',
              rtol: float = 1e-9,
              diag_scale: bool = False,
              penalty_factor: float = 1e6) -> FieldResults:
        """Solve the static FE problem.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', or 'ilss'.
        applied_strain : float
            Applied nominal strain (negative for compression). Used by the
            displacement-controlled modes ('compression', 'tension',
            'shear').
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
            Linear solver to use for ``K u = F``. ``'direct'`` (default)
            uses :func:`scipy.sparse.linalg.spsolve` (sparse LU). For
            large meshes the LU fill-in dominates RAM; the penalty-modified
            matrix is SPD, so ``'cg'`` (conjugate gradient) with a Jacobi
            preconditioner is a memory-light alternative. ``'minres'``
            is offered for completeness when the matrix is symmetric but
            not strictly positive definite. Auto-switching is intentionally
            *not* performed — callers select the path explicitly
            (issue #57).
        rtol : float
            Relative-residual tolerance for the iterative solvers. Ignored
            when ``solver='direct'``.
        diag_scale : bool, optional
            If ``True``, symmetrically Jacobi-pre-scale the penalty-
            modified system before solving:
            ``(D^{-1/2} K_mod D^{-1/2}) y = D^{-1/2} F_mod``,
            ``u = D^{-1/2} y``, where ``D = diag(K_mod)``. The math is
            unchanged but the diagonal-conditioning ratio is reduced by
            2-3 decades on graded/voided meshes, which improves both LU
            backward error and CG/MINRES convergence. Defaults to
            ``False`` to preserve bit-identical legacy behavior; opt in
            when conditioning is a concern (issue #60).
        penalty_factor : float, optional
            Multiplier on ``max(diag(K))`` used by
            :meth:`BoundaryHandler.apply_penalty` to enforce Dirichlet
            BCs. Lowered from ``1e8`` to ``1e6`` (default) in issue #60
            to keep ``cond(K_mod)`` well below the float64 ceiling while
            still enforcing BCs to six decades. Tune higher only if BC
            slack is a problem; tune lower if conditioning is.

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
            If the iterative solver fails to converge to ``rtol``, or if
            the direct solve produces non-finite values / a residual above
            ``1e-6``.
        """
        t0 = time.perf_counter()
        criterion = failure_criterion if failure_criterion is not None \
            else self.failure_criterion
        if criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )

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

        # 3-4. Penalty/diag-scaling, conditioning diagnostics, solver
        #      dispatch and the solve itself. Returns the unscaled physical
        #      displacement vector ``u``.
        u, _rel_res = self._modify_system_and_solve(
            K, F, constrained, solver=solver, rtol=rtol,
            diag_scale=diag_scale, penalty_factor=penalty_factor,
            verbose=verbose,
        )

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
            self.material, self.porosity_field.void_shape_radii, criterion)

        # 7. Compute knockdown as average-stress ratio (porous / pristine).
        knockdown = self._compute_knockdown(
            loading, stress_global, strain_global)

        displacement = u.reshape(-1, 3)
        reactions, effective_modulus = self._reactions_and_modulus(
            loading, K, u, F, applied_strain)

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

    def _modify_system_and_solve(
        self, K: scipy.sparse.spmatrix, F: np.ndarray, constrained: dict[int, float],
        *, solver: Literal['direct', 'cg', 'minres'] = 'direct',
        rtol: float = 1e-9, diag_scale: bool = False,
        penalty_factor: float = 1e6, verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Apply BCs to the system, condition it, and solve ``K u = F``.

        Enforces Dirichlet BCs via :meth:`BoundaryHandler.apply_penalty`,
        logs the diagonal-conditioning diagnostic (issue #60), optionally
        applies symmetric Jacobi pre-scaling, dispatches to the requested
        linear solver, validates the residual, and unscales the solution.

        Parameters
        ----------
        K : scipy.sparse matrix
            Assembled (pre-penalty) global stiffness matrix.
        F : np.ndarray
            Global force vector.
        constrained : dict[int, float]
            Constrained-DOF map from :meth:`_apply_boundary_conditions`.
        solver : {'direct', 'cg', 'minres'}
            Linear solver to use. See :meth:`solve` for details.
        rtol : float
            Relative-residual tolerance for the iterative solvers.
        diag_scale : bool
            Symmetric Jacobi pre-scaling toggle (issue #60).
        penalty_factor : float
            Penalty multiplier for the Dirichlet enforcement.
        verbose : bool
            Log progress information.

        Returns
        -------
        u : np.ndarray
            Physical displacement vector (already unscaled if
            ``diag_scale`` was applied).
        rel_res : float
            Achieved relative residual of the solve.

        Raises
        ------
        ValueError
            If ``solver`` is not one of 'direct', 'cg', 'minres'.
        RuntimeError
            On a non-positive diagonal for the scaling/preconditioner, a
            non-finite or non-converged solution, or a residual above
            tolerance.
        """
        # 3. Apply penalty
        K_mod, F_mod = BoundaryHandler.apply_penalty(
            K, F, constrained, penalty_factor=penalty_factor,
        )

        # 3a. Conditioning diagnostic (issue #60). The diagonal ratio is
        # an inexpensive proxy for cond(K_mod) — full condest is O(n^2)
        # for sparse matrices and we want this on every solve. Warn the
        # user well before float64's ~1e16 headroom is exhausted.
        _diag = K_mod.diagonal()
        _diag_abs = np.abs(_diag)
        _diag_min = float(_diag_abs[_diag_abs > 0.0].min()) \
            if np.any(_diag_abs > 0.0) else 0.0
        _diag_max = float(_diag_abs.max()) if _diag_abs.size else 0.0
        cond_diag_ratio = (_diag_max / _diag_min) if _diag_min > 0.0 \
            else float('inf')
        logger.info(
            "Matrix conditioning: cond_diag_ratio=%.4e "
            "(penalty_factor=%.2e, diag_scale=%s)",
            cond_diag_ratio, penalty_factor, diag_scale,
        )
        if cond_diag_ratio > 1e12:
            logger.warning(
                "Matrix conditioning near float64 limit "
                "(cond_diag_ratio=%.2e); consider lowering "
                "penalty_factor or enabling diag_scale.",
                cond_diag_ratio,
            )

        # 4. Solve
        if solver not in ('direct', 'cg', 'minres'):
            raise ValueError(
                f"Unknown solver '{solver}'. "
                "Use 'direct', 'cg', or 'minres'."
            )
        if verbose:
            logger.info(
                "Solving system (%d DOFs) with solver='%s'...",
                self.mesh.n_dof, solver,
            )

        # 4a. Optional symmetric Jacobi pre-scaling (issue #60).
        # Replace (K_mod, F_mod) with (K_scaled, F_scaled) for the solve;
        # after solving, unscale y -> u via u = d_inv_sqrt * y.
        if diag_scale:
            _d = K_mod.diagonal()
            if not np.all(_d > 0):
                raise RuntimeError(
                    "Cannot apply diag_scale: K_mod has a non-positive "
                    "diagonal entry. Check assembly / penalty."
                )
            d_inv_sqrt = 1.0 / np.sqrt(_d)
            _D_is = scipy.sparse.diags(d_inv_sqrt)
            K_solve = (_D_is @ K_mod) @ _D_is
            F_solve = d_inv_sqrt * F_mod
            # Log the post-scaling diagonal ratio so the user can see
            # what the rescaling bought them.
            _d_scaled = K_solve.diagonal()
            _d_scaled_abs = np.abs(_d_scaled)
            _ds_min = float(_d_scaled_abs[_d_scaled_abs > 0.0].min()) \
                if np.any(_d_scaled_abs > 0.0) else 0.0
            _ds_max = float(_d_scaled_abs.max()) \
                if _d_scaled_abs.size else 0.0
            cond_diag_ratio_scaled = (_ds_max / _ds_min) \
                if _ds_min > 0.0 else float('inf')
            logger.info(
                "Matrix conditioning after diag_scale: "
                "cond_diag_ratio=%.4e (was %.4e)",
                cond_diag_ratio_scaled, cond_diag_ratio,
            )
        else:
            K_solve = K_mod
            F_solve = F_mod
            d_inv_sqrt = None

        if solver == 'direct':
            y = self._direct_solve(K, K_solve, F_solve, constrained,
                                   penalty_factor, diag_scale)

            # Hygiene checks on the solution vector
            if not np.isfinite(y).all():
                raise RuntimeError(
                    "spsolve produced non-finite values (NaN or Inf) in the solution "
                    "vector. Check matrix conditioning and boundary conditions."
                )
            _r = K_solve @ y - F_solve
            _rel_res = np.linalg.norm(_r) / max(np.linalg.norm(F_solve), 1.0)  # type: ignore[call-overload,operator]
            if _rel_res >= 1e-6:
                raise RuntimeError(
                    f"spsolve residual {_rel_res:.4e} exceeds tolerance 1e-6. "
                    "Check matrix conditioning or penalty factor."
                )
        else:
            # Jacobi (diagonal) preconditioner: K is SPD after penalty,
            # diag(K) is strictly positive.
            diag = K_solve.diagonal()
            if not np.all(diag > 0):
                raise RuntimeError(
                    "Cannot build Jacobi preconditioner: K_mod has a "
                    "non-positive diagonal entry. Check assembly / penalty."
                )
            M = scipy.sparse.diags(1.0 / diag)

            if solver == 'cg':
                y, info = scipy.sparse.linalg.cg(
                    K_solve, F_solve, M=M, rtol=rtol,
                )
            else:  # solver == 'minres'
                y, info = scipy.sparse.linalg.minres(
                    K_solve, F_solve, M=M, rtol=rtol,
                )

            _r = K_solve @ y - F_solve
            _norm_b = float(np.linalg.norm(F_solve))  # type: ignore[call-overload]
            _rel_res = float(
                np.linalg.norm(_r) / _norm_b if _norm_b > 0.0 else 0.0  # type: ignore[call-overload,operator]
            )
            # Compare the achieved relative residual against the user-
            # requested rtol directly. SciPy's iterative solvers can
            # report info=0 while still bouncing off the machine-
            # precision floor — if the user asked for sub-eps tolerance
            # they will (correctly) get a non-convergence error.
            _converged = info == 0 and _rel_res <= rtol * 10.0
            if not _converged:
                raise RuntimeError(
                    f"{solver} failed to converge: info={info}, "
                    f"achieved relative residual {_rel_res:.4e} "
                    f"(requested rtol={rtol:.4e})."
                )
            logger.info(
                "%s converged: relative residual %.4e (rtol=%.4e)",
                solver, _rel_res, rtol,
            )

        # 4b. Unscale if we Jacobi-pre-scaled. ``y`` solves the scaled
        # system; the physical displacement is ``u = D^{-1/2} y``.
        if diag_scale:
            u = d_inv_sqrt * y
        else:
            u = y

        return u, float(_rel_res)

    def _direct_solve(self, K: scipy.sparse.spmatrix, K_solve: scipy.sparse.spmatrix,
                      F_solve: np.ndarray, constrained: dict[int, float],
                      penalty_factor: float, diag_scale: bool) -> np.ndarray:
        """Sparse-LU solve, reusing the last factorization when possible.

        The penalty-modified matrix depends only on the assembled ``K``, the
        *set* of constrained DOFs (not their prescribed values), the penalty
        factor and the diagonal scaling, so e.g. compression and tension on
        the same mesh, or repeat solves of one load case, share a
        factorization. Only the most recent one is kept, bounding memory.
        """
        dofs = np.sort(np.fromiter(constrained.keys(), dtype=np.intp,
                                   count=len(constrained)))
        key = (dofs.tobytes(), float(penalty_factor), bool(diag_scale))
        cached = self._lu_cache
        if cached is not None and cached[0] is K and cached[1] == key:
            lu = cached[2]
        else:
            # K_solve is symmetric (penalty and Jacobi scaling keep it so):
            # a symmetric fill-reducing ordering with diagonal pivoting
            # factors ~20% faster than SuperLU's default COLAMD here.
            lu = scipy.sparse.linalg.splu(
                scipy.sparse.csc_matrix(K_solve),
                permc_spec='MMD_AT_PLUS_A',
                options={'SymmetricMode': True},
            )
            self._lu_cache = (K, key, lu)
        return lu.solve(np.asarray(F_solve, dtype=float))

    def _reactions_and_modulus(
        self, loading: str, K: scipy.sparse.spmatrix, u: np.ndarray,
        F: np.ndarray, applied_strain: float,
    ) -> tuple[np.ndarray, float | None]:
        """Nodal reactions ``K u - F`` and the strain-energy effective modulus.

        For the displacement-controlled modes only the prescribed boundary
        moves work through the reactions, so ``u^T K u = sum(R_i u_i)`` is
        twice the strain energy ``0.5 M strain^2 V``, giving ``M``: axial
        ``E_x = P / (A strain)`` for compression/tension, ``G_xy`` for the
        homogeneous pure-shear BCs.
        """
        Ku = K @ u
        reactions = (Ku - F).reshape(-1, 3)
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
        self, loading: str, stress_global: np.ndarray, strain_global: np.ndarray,
    ) -> float:
        """Compute the stiffness knockdown as a porous/pristine stress ratio.

        Both numerator and denominator use the same 3D FE framework so that
        dimensional/mesh effects cancel. For each element we compute what the
        dominant stress component *would* be with pristine stiffness at the
        same strain, then average. This avoids the CLT-vs-3D mismatch that
        caused knockdown > 1.

        For ILSS short-beam shear the dominant component is ``tau_xz``
        (Voigt index 4); for the other modes it is ``sigma_xx`` (index 0).

        Parameters
        ----------
        loading : str
            Loading mode (selects the dominant stress component).
        stress_global : np.ndarray
            Shape ``(n_elem, n_gp, 6)`` recovered global stresses.
        strain_global : np.ndarray
            Shape ``(n_elem, n_gp, 6)`` recovered global strains.

        Returns
        -------
        knockdown : float
            Modulus ratio ``E_porous / E_pristine``, clamped to ``<= 1.0``.
        """
        if loading == 'ilss':
            comp_idx = 4
        else:
            comp_idx = 0

        avg_sigma = np.mean(stress_global[:, :, comp_idx])

        # Pristine reference: compute the same Voigt component using the
        # rotated pristine stiffness applied to the recovered strain field,
        # with one rotation per distinct ply angle.
        C_base = self.material.get_stiffness_matrix()
        angles, angle_idx = np.unique(
            np.asarray(self.mesh.ply_angles, dtype=float), return_inverse=True)
        rows = np.empty((len(angles), 6))
        for k, angle in enumerate(angles):
            ply_rad = np.radians(angle)
            C_rot = (rotate_stiffness_3d(C_base, ply_rad, axis='z')
                     if abs(ply_rad) > 1e-15 else C_base)
            rows[k] = C_rot[comp_idx, :]
        pristine_sig = np.einsum('ej,egj->eg', rows[angle_idx], strain_global)
        pristine_avg = float(pristine_sig.mean()) if pristine_sig.size else 1.0

        if abs(pristine_avg) > 1e-12:
            knockdown = abs(avg_sigma) / abs(pristine_avg)
        else:
            knockdown = 1.0
        return min(knockdown, 1.0)

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
            self.material, self.porosity_field.void_shape_radii, criterion)

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
