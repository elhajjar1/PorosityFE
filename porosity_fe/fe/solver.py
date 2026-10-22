"""FE solver: :class:`FESolver`, the orchestrator of a linear static solve.

:meth:`FESolver.solve` validates the load case, assembles ``K``
(:class:`~porosity_fe.fe.assembler.GlobalAssembler`), solves the
constrained system and post-processes the field into a
:class:`~porosity_fe.fe.results.FieldResults`. The steps live in focused
modules (IMPROVEMENT_PLAN 5.1):

* :mod:`porosity_fe.fe.results`: :class:`FieldResults`;
* ``_load_cases``: loading modes, input checks, boundary-condition
  dispatch, interior region;
* ``_constrained_solve``: exact elimination of the Dirichlet BCs, sparse
  LU with its per-constrained-set cache, CG / MINRES;
* ``_knockdown``: stiffness measure, pristine reference and its cache;
* ``_thermal``: free-standing unit-``delta_T`` thermal solve;
* ``_postprocess``: stress / strain recovery, reactions and modulus,
  failure metrics;
* :mod:`porosity_fe.fe.failure` and :mod:`porosity_fe.fe.export`:
  failure criteria and result export.

The pre-split names other modules and the tests import from here
(``FieldResults``, ``_PRISTINE_MEASURE_CACHE``, ``_check_delta_T``,
``_minres_deprecation``, ``_resolve_applied_strain``) are re-exported.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections import OrderedDict
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .._ply_angles import _resolve_ply_angles
from .._types import FEFormulation, FELoadingMode
from ..materials import MaterialProperties
from ..mesh import CompositeMesh, check_mesh_quality
from ..porosity_field import PorosityField
from . import _constrained_solve, _knockdown, _postprocess, _thermal, failure
from ._knockdown import _PRISTINE_MEASURE_CACHE as _PRISTINE_MEASURE_CACHE
from ._knockdown import _stiffness_measure
from ._load_cases import _check_delta_T as _check_delta_T
from ._load_cases import _minres_deprecation as _minres_deprecation
from ._load_cases import _resolve_applied_strain as _resolve_applied_strain
from ._load_cases import apply_boundary_conditions, check_solve_arguments, interior_mask
from ._thermal import _ThermalUnitSolution
from .assembler import BoundaryHandler, GlobalAssembler, _ConstraintReduction, _mesh_ties, _ties_token
from .element import _DEFAULT_FORMULATION
from .export import export_results as _export_results
from .results import FieldResults as FieldResults

logger = logging.getLogger("porosity_fe_analysis")


class FESolver:
    """Linear static FE solver for porosity-degraded composite laminates.

    Workflow:
    1. Assemble K via GlobalAssembler
    2. Build BCs via BoundaryHandler
    3. Eliminate the prescribed DOFs: ``K_ff u_f = F_f - K_fc u_c``
    4. Solve the reduced system (sparse LU or CG; MINRES is deprecated)
    5. Recover stresses at Gauss points
    6. Evaluate failure criterion at each GP (Tsai-Wu, Hashin, or max-stress)
    7. Compute knockdown factor

    With ``delta_T`` (``loading='thermal'``, or a mechanical mode plus
    ``delta_T=``) a free-standing thermal solve with 3-2-1 supports gives
    the cure residual stress, which is reported alone or superposed on the
    mechanical solve; see :meth:`solve` and the theory page on thermal and
    cure residual stress.

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
    formulation : {'hex8i', 'hex8'}, optional
        Keyword-only element formulation (see :data:`FEFormulation`).
        ``'hex8i'`` (default) adds nine Wilson-Taylor incompatible modes per
        element, condensed out during assembly, which removes the shear
        locking of the standard brick. ``'hex8'`` is the standard fully
        integrated trilinear brick, the default before the switch, and
        reproduces those earlier results bit for bit. ``'hex8'`` is too
        stiff in bending: the error grows roughly as
        ``(G13 / E11) (dx / h)^2`` with element length ``dx`` and laminate
        thickness ``h`` (about 0.5 % at the production mesh, 8 % at
        ``dx / h = 1.56``, 127 % at ``dx / h = 6.25`` for UD T800). Membrane
        modes are essentially the same with either element; bending
        (``'ilss'``) stiffness, transverse shear stress and the load
        factors differ. Recorded on
        :attr:`FieldResults.formulation` and in the JSON export; the
        stiffness and pristine-reference caches are keyed on it. Unknown
        values raise :class:`ValueError`.

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

    On a mesh with explicit delaminations (``CompositeMesh(...,
    delaminations=...)``) every loading mode ties the bonded part of each
    duplicated interface (:meth:`CompositeMesh.bonded_ties`), prescribes
    no tied duplicate on a face, and measures the knockdown against the
    laminate with no porosity, voids or delamination; see the theory page
    "Delamination".
    """

    #: Supported failure criteria for :meth:`solve`. Used both at runtime
    #: (for validation) and as the documented enumeration.
    SUPPORTED_FAILURE_CRITERIA: tuple[str, ...] = failure.SUPPORTED_FAILURE_CRITERIA

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField,
                 ply_angles: list[float] | str | None = 'QI',
                 failure_criterion: Literal[
                     'tsai_wu', 'hashin', 'max_stress'] = 'tsai_wu',
                 *, formulation: FEFormulation = _DEFAULT_FORMULATION,
                 ) -> None:
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
        self.assembler = GlobalAssembler(mesh, material, porosity_field,
                                         formulation=formulation)
        self.bc_handler = BoundaryHandler(mesh)
        if failure_criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {failure_criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )
        self.failure_criterion = failure_criterion
        # Free-DOF stiffness blocks and their sparse LU factorizations
        # (built lazily by direct solves) of the last _LU_CACHE_SIZE
        # constrained-DOF sets, least recently used first, all for the
        # current assembled K: {partition key: (K, K_ff, SuperLU | None)}.
        # See _free_dof_system.
        self._lu_cache: OrderedDict[bytes, tuple] = OrderedDict()
        # Free-standing thermal solution per unit delta_T, reused while K,
        # the CTEs and the linear solver are unchanged:
        # (K, key, _ThermalUnitSolution).
        self._thermal_cache: tuple | None = None

    @property
    def formulation(self) -> FEFormulation:
        """Element formulation, ``'hex8'`` or ``'hex8i'`` (read-only).

        Held by :attr:`assembler`, whose stiffness cache is keyed on it.
        """
        return self.assembler.formulation

    def solve(self, loading: FELoadingMode = 'compression',
              applied_strain: float | None = None,
              applied_load: float = -10.0,
              verbose: bool = False,
              failure_criterion: Literal['tsai_wu', 'hashin', 'max_stress'] | None = None,
              solver: Literal['direct', 'cg', 'minres'] = 'direct',
              rtol: float = 1e-9,
              diag_scale: bool | None = None,
              penalty_factor: float | None = None,
              *,
              delta_T: float | None = None,
              allow_unresolved_layup: bool = False,
              interior_margin: float | None = None) -> FieldResults:
        """Solve the static FE problem.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', 'ilss', 'flexure_3pt',
            'flexure_4pt' or 'thermal'.
            ``'ilss'`` is the ASTM D2344 short-beam-shear test on the
            specimen of :meth:`CompositeMesh.for_ilss` (supports ``4 h``
            apart under a ``6 h x 2 h`` beam, on rollers, the load spread
            over the loading-nose contact band at mid-span;
            :meth:`BoundaryHandler.ilss_bcs`). A mesh without
            :attr:`CompositeMesh.ilss_geometry` raises :class:`ValueError`
            before anything is assembled. :func:`~porosity_fe.short_beam_shear`
            reports the D2344 apparent shear ``0.75 P / (b h)`` next to the
            FE stresses and failure loads.
            ``'flexure_3pt'`` and ``'flexure_4pt'`` are three- and
            four-point bending on roller supports (ASTM D790 / D7264
            procedures A and B, :meth:`BoundaryHandler.flexure_bcs`) at the
            supports and load lines of :attr:`CompositeMesh.flexure_geometry`
            (build the mesh with :meth:`CompositeMesh.for_flexure`); without
            one the supports are the two bottom end edges. They report the
            flexural modulus as :attr:`FieldResults.effective_modulus`.
            ``'thermal'`` solves the free-standing laminate under the
            uniform temperature change ``delta_T`` (required) with
            statically determinate 3-2-1 supports
            (:meth:`BoundaryHandler.free_bcs`): the cure residual stress.
            ``applied_strain`` and ``applied_load`` are ignored for it.
        applied_strain : float, optional
            Applied nominal strain (negative for compression). Used by the
            displacement-controlled modes ('compression', 'tension',
            'shear'). Defaults to ``-0.01`` for ``'compression'`` and
            ``+0.01`` for ``'tension'`` and ``'shear'``. (Before 2.3 the
            default was ``-0.01`` for every mode, so ``solve('tension')``
            silently ran a compression solve.) A strain whose sign
            contradicts ``'compression'`` / ``'tension'`` logs a warning.
        applied_load : float
            Total load (N, negative = downward) of the force-controlled
            modes: the mid-span load of ``'ilss'`` (ASTM D2344) and
            ``'flexure_3pt'``, the sum of the two loads of
            ``'flexure_4pt'``. Ignored for the other modes.
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
            (:func:`scipy.sparse.linalg.splu`), cached for the last two
            constrained-DOF sets and reused while the mesh, material and
            porosity are unchanged (about 320 MB per factorization on the
            production mesh; see :meth:`_free_dof_system`). ``K_ff`` is
            symmetric positive definite, so
            ``'cg'`` (conjugate gradient with a Jacobi preconditioner) is
            a memory-light alternative; it needs no LU fill-in and is
            the better choice above roughly 40k DOF (about 1.6x the
            25,575 DOF of the default production mesh of a 24-ply laminate,
            30 x 10 x 24), where LU time and memory grow fastest.
            ``'minres'`` is deprecated and will be removed in 2.0: it
            emits a :class:`DeprecationWarning`; use ``'cg'``. Because
            ``K_ff`` is SPD, CG covers every case, while MINRES minimizes
            the residual rather than the energy-norm error, so at the same
            ``rtol`` its displacements are less accurate (4e-7 to 2e-6
            relative to direct on the production mesh at the default
            ``rtol``, against 5e-9 to 1e-8 for CG). It still works: SciPy's
            MINRES stops on a preconditioned residual estimate, so it is
            warm-restarted with a tighter internal tolerance until the
            true relative residual meets ``rtol``. Auto-switching is
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
        delta_T : float, optional
            Keyword-only uniform temperature change from the stress-free
            state, in K (the same number in deg C, since it is a
            difference), negative for a cool-down from cure, e.g.
            ``T_service - T_stress_free``. Required for
            ``loading='thermal'``; never inferred from
            :attr:`MaterialProperties.T_stress_free`. With a mechanical
            ``loading`` it adds the cure residual stress: the free-standing
            thermal solve and the mechanical solve are superposed
            (``sigma = sigma_th + sigma_mech``), because the residual stress
            forms in the free laminate before it is gripped; one solve with
            both loads would model a laminate heated while clamped. The
            first-ply-failure factor then scales only the mechanical part
            (:func:`~porosity_fe.fe.failure.first_ply_failure_load_factor`
            with ``prestress_local``); knockdown, effective modulus and
            reactions are those of the mechanical solve. Needs lamina CTEs
            (``material.has_cte``). ``None`` (default) runs the mechanical
            solve exactly as before.
        allow_unresolved_layup : bool
            Keyword-only. Thermal results depend on the stacking sequence,
            so with ``delta_T`` the solve raises when the element layers do
            not resolve the requested layup
            (:meth:`CompositeMesh.layup_discrepancies`; use
            ``nz = k * n_plies``). ``True`` analyzes the element layup
            anyway, with a warning. No effect without ``delta_T``
            (mechanical solves only warn, once per mesh).
        interior_margin : float, optional
            Keyword-only distance in mm from the lateral faces (``x``/``y``)
            that :attr:`FieldResults.interior_max_failure_index` and
            :attr:`FieldResults.interior_first_ply_failure_load_factor`
            leave out. Defaults to ``max(h, 2 element widths)`` per
            direction, with ``h`` the laminate thickness. For ``'ilss'``
            it is the distance from the support lines, the edges of the
            loading band and the side faces, and for ``'flexure_3pt'`` /
            ``'flexure_4pt'`` the distance from each support and load line
            (in the ``x``-``z`` plane) and from the side faces; both
            default to ``h / 2`` (see
            :attr:`FieldResults.interior_max_failure_index`).

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
            one of ``'direct'``, ``'cg'``, ``'minres'``. With thermal load:
            if ``loading='thermal'`` has no ``delta_T``, ``delta_T`` is not
            a finite number, the material has no CTEs, or the mesh does not
            resolve the layup (unless ``allow_unresolved_layup``). If
            ``interior_margin`` is negative or not finite. For ``'ilss'``,
            if the mesh has no :attr:`CompositeMesh.ilss_geometry`.
        RuntimeError
            If the iterative solver fails to converge to ``rtol``, if the
            direct solve produces non-finite values / a residual above
            ``1e-6``, or if the free-DOF stiffness is singular (boundary
            conditions that leave a rigid-body mode free).
        """
        t0 = time.perf_counter()
        criterion = failure_criterion if failure_criterion is not None \
            else self.failure_criterion
        check_solve_arguments(loading, criterion, solver,
                              supported_criteria=self.SUPPORTED_FAILURE_CRITERIA,
                              penalty_factor=penalty_factor, diag_scale=diag_scale)
        thermal_only = loading == 'thermal'
        dT = _check_delta_T(loading, delta_T, self.material)
        if dT is not None:
            _thermal.check_layup_for_thermal(self.mesh, self.material,
                                             self.formulation, allow_unresolved_layup)
        # For 'ilss' (the ASTM D2344 shear spans) this raises here, before
        # assembly, without a specimen mesh.
        interior = interior_mask(loading, self.mesh, self.bc_handler,
                                 interior_margin)
        if not thermal_only:
            applied_strain = _resolve_applied_strain(loading, applied_strain)

        # 0. Mesh quality check
        check_mesh_quality(self.mesh, verbose=verbose)

        # The knockdown's pristine reference (a cached value, or a direct
        # solve whose factorization is freed on return) comes before the
        # porous model is assembled and factorized, so one factorization is
        # alive at a time, not two.
        pristine = self._pristine_stiffness_measure(loading) \
            if _knockdown.needs_pristine_reference(
                self.mesh, loading, applied_strain, applied_load) else None

        # 1. Global stiffness (re-assembled only if the mesh, material or
        #    porosity changed since the last solve on this solver).
        if verbose:
            logger.info("Assembling global stiffness matrix...")
        K = self.assembler.stiffness(verbose=verbose)

        if verbose:
            t1 = time.perf_counter()
            logger.info("  Assembly time: %.2f s", t1 - t0)

        # 2. Build BCs and the force vector for the requested loading mode.
        constrained: dict[int, float] | None = None
        if not thermal_only:
            constrained, F = self._apply_boundary_conditions(
                loading, applied_strain, applied_load, verbose=verbose)

        # Free-standing thermal (residual) state, scaled from the cached
        # unit-delta_T solution; its factorization is freed before the
        # mechanical one is built.
        thermal: _ThermalUnitSolution | None = None
        residual_global: np.ndarray | None = None
        residual_local: np.ndarray | None = None
        if dT is not None:
            thermal = self._thermal_unit_solution(
                K, solver=solver, rtol=rtol, verbose=verbose,
                next_constrained=constrained)
            residual_global = dT * thermal.stress_global
            residual_local = self._rotate_to_local(residual_global)

        if thermal_only:
            assert thermal is not None and dT is not None
            assert residual_global is not None and residual_local is not None
            u = dT * thermal.u
            stress_global, stress_local = residual_global, residual_local
            strain_global = dT * thermal.strain_global
            strain_local = self._rotate_to_local(strain_global, strain=True)
            load_stress = stress_local
            prestress_local = None
            reactions: np.ndarray = dT * thermal.reactions
            knockdown = float('nan')
            effective_modulus: float | None = None
            basis = 'delta_T'
        else:
            assert constrained is not None
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
            (stress_global, stress_local, strain_global,
             strain_local) = _postprocess.recover_stresses(
                self.assembler.element_batch(), self.mesh, u, verbose=verbose)

            # 7. Knockdown: porous / pristine structural stiffness (1.0 for
            #    a pristine mesh or a zero load), and the reactions and
            #    modulus, all from the mechanical solve.
            knockdown = 1.0 if pristine is None else _knockdown.knockdown(
                loading, _stiffness_measure(loading, K, u, applied_strain,
                                            applied_load), pristine)
            reactions, effective_modulus = self._reactions_and_modulus(
                loading, K, u, F, applied_strain, constrained,
                applied_load=applied_load)
            load_stress = stress_local
            prestress_local = residual_local
            basis = 'mechanical'
            if thermal is not None:
                # Superpose the residual state (linear, both from the
                # stress-free free-standing configuration). The load factor
                # below scales only the mechanical stress.
                assert dT is not None and residual_global is not None
                u = u + dT * thermal.u
                stress_global = stress_global + residual_global
                strain_global = strain_global + dT * thermal.strain_global
                stress_local = self._rotate_to_local(stress_global)
                strain_local = self._rotate_to_local(strain_global, strain=True)
                basis = 'mechanical_with_residual'

        # 6. Failure criterion at each GP, first-ply-failure load factors,
        #    and their residual-stress and interior values.
        fm = _postprocess.failure_metrics(
            self.mesh, self.material, self.porosity_field.void_shape_radii,
            criterion, stress_local=stress_local, load_stress_local=load_stress,
            prestress_local=prestress_local, residual_local=residual_local,
            interior=interior, thermal_only=thermal_only, delta_T=dT)

        displacement = u.reshape(-1, 3)

        if verbose:
            t3 = time.perf_counter()
            logger.info("  Post-processing time: %.2f s", t3 - t1)
            logger.info("Total solve time: %.2f s", t3 - t0)
            logger.info("  Max %s FI: %.4f", criterion, fm.max_fi)
            logger.info("  Knockdown factor: %.4f", knockdown)

        return FieldResults(
            displacement=displacement,
            stress_global=stress_global,
            stress_local=stress_local,
            strain_global=strain_global,
            strain_local=strain_local,
            max_failure_index=fm.max_fi,
            knockdown=knockdown,
            per_element_failure_index=fm.per_element_fi,
            failure_criterion=criterion,
            failure_mode_indices=fm.mode_indices,
            reaction_forces=reactions,
            effective_modulus=effective_modulus,
            first_ply_failure_load_factor=fm.load_factor,
            formulation=self.formulation,
            delta_T=dT,
            residual_stress_local=residual_local,
            residual_max_failure_index=fm.residual_max_fi,
            interior_max_failure_index=fm.interior_max_fi,
            interior_first_ply_failure_load_factor=fm.interior_load_factor,
            load_factor_basis=basis,
            cte_local=(None if thermal is None else
                       (float(thermal.alpha_local[0]), float(thermal.alpha_local[1]),
                        float(thermal.alpha_local[2]))),
            failure_mode_maxima=fm.mode_maxima,
            per_element_mode_failure_index=fm.per_element_mode_fi,
        )

    # Steps of solve(). The work is done in the private step modules; tests
    # and callers use several of these methods, so they stay.

    def _apply_boundary_conditions(
        self, loading: str, applied_strain: float, applied_load: float,
        *, verbose: bool = False,
    ) -> tuple[dict[int, float], np.ndarray]:
        """Dirichlet BCs ``{dof: value}`` and force vector of ``loading``
        (``_load_cases.apply_boundary_conditions``)."""
        constrained, F = apply_boundary_conditions(
            self.bc_handler, loading, applied_strain, applied_load)
        if verbose:
            logger.info("  Applied %d displacement BCs", len(constrained))
        return constrained, F

    #: Warm restarts allowed after the first MINRES call (see
    #: ``_constrained_solve._iterative_solve``).
    _MINRES_MAX_RESTARTS = 3

    #: Constrained-DOF sets whose free-DOF system (``K_ff`` and its LU) a
    #: solver keeps; see :meth:`_free_dof_system`.
    _LU_CACHE_SIZE = 2

    def _solve_constrained(
        self, K: scipy.sparse.spmatrix, F: np.ndarray, constrained: dict[int, float],
        *, solver: Literal['direct', 'cg', 'minres'] = 'direct',
        rtol: float = 1e-9, verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Impose the Dirichlet BCs exactly and solve ``K u = F``: the full
        ``u`` and the relative residual of the reduced system
        (``_constrained_solve.solve_constrained``), direct solves with this
        solver's cached factorizations (:meth:`_free_dof_system`). The
        bonded interface of a delaminated mesh is tied
        (:meth:`CompositeMesh.bonded_ties`)."""
        return _constrained_solve.solve_constrained(
            K, F, constrained, system=self._free_dof_system,
            ties=_mesh_ties(self.mesh), solver=solver,
            rtol=rtol, verbose=verbose, max_restarts=self._MINRES_MAX_RESTARTS)

    def _free_dof_system(
        self, K: scipy.sparse.spmatrix, part: _ConstraintReduction,
        *, factorize: bool,
    ) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.linalg.SuperLU | None]:
        """``K_ff`` and (when ``factorize``) its sparse LU, cached.

        ``self._lu_cache`` keeps the systems of the last
        :attr:`_LU_CACHE_SIZE` (2) constrained-DOF sets, least recently used
        first, so load cases that share supports, or two that alternate,
        factorize once; another set evicts the least recently used before
        its own factorization is built, and a new ``K`` clears them all
        (``_constrained_solve.free_dof_system``). The thermal 3-2-1 set
        takes a slot only while the unit-``delta_T`` thermal solution is
        computed (:meth:`_thermal_unit_solution`). About 320 MB per
        factorization on the 30 x 10 x 24 production mesh.
        """
        return _constrained_solve.free_dof_system(
            self._lu_cache, K, part, factorize=factorize,
            max_entries=self._LU_CACHE_SIZE)

    def _reactions_and_modulus(
        self, loading: str, K: scipy.sparse.spmatrix, u: np.ndarray,
        F: np.ndarray, applied_strain: float, constrained: dict[int, float],
        *, applied_load: float = -10.0,
    ) -> tuple[np.ndarray, float | None]:
        """Nodal reactions and the effective modulus
        (``_postprocess.reactions_and_modulus``)."""
        return _postprocess.reactions_and_modulus(
            loading, K, u, F, applied_strain, constrained,
            applied_load=applied_load, mesh=self.mesh,
            assembler=self.assembler, bc_handler=self.bc_handler)

    def _rotate_to_local(self, field_global: np.ndarray, *,
                         strain: bool = False) -> np.ndarray:
        """A ``(n_elem, n_gp, 6)`` global field in the ply axes
        (``_postprocess.rotate_to_local``)."""
        return _postprocess.rotate_to_local(self.mesh.ply_angles, field_global,
                                            strain=strain)

    def _thermal_unit_solution(
        self, K: scipy.sparse.spmatrix, *,
        solver: Literal['direct', 'cg', 'minres'] = 'direct',
        rtol: float = 1e-9, verbose: bool = False,
        next_constrained: dict[int, float] | None = None,
    ) -> _ThermalUnitSolution:
        """Free-standing thermal solve for ``delta_T = 1`` K, cached.

        ``_thermal.unit_solution``, cached on ``K`` (which changes with the
        mesh, porosity, material and formulation), the CTEs, the linear
        solver and the bonded ties of a delaminated mesh (which change with
        :attr:`CompositeMesh.delaminations` while ``K`` does not). Every
        ``delta_T`` rescales the cached solution, so the 3-2-1 system leaves
        the LU cache as soon as it is solved instead of being held next to
        the mechanical factorization, unless ``next_constrained``, the
        constrained set of the solve that follows, is the same set.
        """
        alpha_local = self.material.cte_vector()
        ties = _mesh_ties(self.mesh)
        key = (tuple(alpha_local.tolist()), solver, float(rtol), _ties_token(ties))
        cached = self._thermal_cache
        if cached is not None and cached[0] is K and cached[1] == key:
            return cached[2]

        supports = self.bc_handler.free_bcs()
        sol = _thermal.unit_solution(
            K, self.assembler.element_batch(), self.mesh, supports,
            alpha_local, verbose=verbose,
            solve=functools.partial(self._solve_constrained, solver=solver,
                                    rtol=rtol, verbose=verbose))
        self._thermal_cache = (K, key, sol)
        thermal_key = _ConstraintReduction.from_constraints(
            K.shape[0], supports[0], ties=ties).key
        if next_constrained is None or thermal_key != _ConstraintReduction.from_constraints(
                K.shape[0], next_constrained, ties=ties).key:
            self._lu_cache.pop(thermal_key, None)
        return sol

    def _pristine_stiffness_measure(self, loading: str) -> float:
        """Stiffness measure of this mesh with no porosity or voids, the
        knockdown's reference, cached across solvers
        (``_knockdown.pristine_stiffness_measure``). :meth:`solve` calls it
        before it assembles the porous model."""
        return _knockdown.pristine_stiffness_measure(
            loading, self.mesh, self.material, self.porosity_field,
            self.formulation, self.bc_handler)

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
        ``UNSTRUCTURED_GRID`` for ParaView / VisIt / PyVista. With
        ``fmt='vtu'`` it delegates to :meth:`FieldResults.to_vtu` with its
        defaults (binary VTK XML, ``Float64``, plus recovered nodal stress
        and strain). The options (precision, exploded output, averaging)
        live on those methods; these two formats are a convenience shim for
        callers that already hold an ``FESolver``.

        Parameters
        ----------
        field_results : FieldResults
            Results from FESolver.solve().
        filename : str or os.PathLike
            Output file path (``.json``, ``.vtk`` or ``.vtu``).
            ``pathlib.Path`` objects are accepted.
        fmt : str
            ``'json'`` (default), ``'vtk'`` or ``'vtu'``.
        mesh : CompositeMesh, optional
            Required when ``fmt`` is ``'vtk'`` or ``'vtu'`` (supplies
            geometry/connectivity).
        include_raw : bool
            When ``True`` (and ``fmt='json'``), also write a sidecar
            ``<filename>.npz`` containing the raw displacement/stress/strain
            arrays so a full audit can re-derive the per-key summary
            statistics. Default ``False`` so existing outputs are not
            bloated (#55).

        Notes
        -----
        The JSON is standard JSON: it never holds ``NaN`` or ``Infinity``,
        which strict parsers reject. A value that is not defined is
        ``null``:

        * ``failure.mode_indices.<mode>``: the criterion does not separate
          that mode (Tsai-Wu: every entry except ``max_fi``; ``NaN`` in
          :attr:`FieldResults.failure_mode_indices`). Max-stress writes 0
          for the modes it does not use.
        * ``failure.knockdown_factor``: no stiffness measure
          (``loading='thermal'``; ``NaN`` in :attr:`FieldResults.knockdown`).
        * ``failure.first_ply_failure_load_factor`` and
          ``failure.interior_first_ply_failure_load_factor``: no point is
          stressed, so the factor is infinite; the interior one is also
          ``null`` when no element is far enough from the lateral faces.
        * ``failure.interior_max_failure_index`` (and the same key in
          ``thermal``): no element is far enough from the lateral faces.
        * ``stiffness.effective_modulus_MPa``: no single modulus (ILSS
          bend, ``loading='thermal'``).

        Any other non-finite value raises :class:`ValueError` naming its
        location, before a file is written.
        :func:`~porosity_fe.load_results_from_json` reads ``null`` as
        ``None``.
        """
        _export_results(field_results, filename, fmt=fmt, mesh=mesh,
                        include_raw=include_raw)
