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
import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .._ply_angles import _resolve_ply_angles
from .._types import FEFormulation, FELoadingMode, PorosityDegradation
from ..homogenization import _DEFAULT_POROSITY_DEGRADATION
from ..materials import MaterialProperties
from ..mesh import CompositeMesh, check_mesh_quality
from ..porosity_field import PorosityField
from ..toughness import ToughnessKnockdownSpec, _resolve_toughness_knockdown
from . import _constrained_solve, _contact, _fracture, _knockdown, _postprocess, _thermal, failure
from ._knockdown import _PRISTINE_MEASURE_CACHE as _PRISTINE_MEASURE_CACHE
from ._knockdown import _stiffness_measure
from ._load_cases import _FRACTURE_TESTS, apply_boundary_conditions, check_solve_arguments, interior_mask
from ._load_cases import _check_delta_T as _check_delta_T
from ._load_cases import _minres_deprecation as _minres_deprecation
from ._load_cases import _resolve_applied_strain as _resolve_applied_strain
from ._thermal import _ThermalUnitSolution
from .assembler import (
    _MESH_TIES,
    BoundaryHandler,
    GlobalAssembler,
    Ties,
    _ConstraintReduction,
    _mesh_ties,
    _ties_token,
)
from .element import _DEFAULT_FORMULATION
from .export import export_results as _export_results
from .results import FieldResults as FieldResults

if TYPE_CHECKING:
    from ..uq import FractureLoadFactorBand
    from .decomposition import KnockdownDecomposition
    from .interlaminar import DelaminationOnset

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
    porosity_degradation : {'isotropic', 'directional'}, optional
        Keyword-only. How the local porosity degrades the ply stiffness and
        strengths (see :data:`PorosityDegradation`). ``'isotropic'``
        (default) projects the Mori-Tanaka tensor of the porous matrix onto
        an isotropic one before the Halpin-Tsai ratios and scales every
        matrix strength by the in-plane ratio ``r_m``, so a flat (penny)
        void weakens ``E33``, ``G13``, ``G23``, ``Z_t`` and ``S_13`` no
        more than ``E22`` and ``Y_t``; every existing result uses it.
        ``'directional'`` keeps the transversely isotropic Mori-Tanaka
        tensor, degrades each ply constant from its own matrix direction
        and scales ``Z_t``, ``S_13`` and ``S_23`` by their own stiffness
        ratios relative to the in-plane one (delamination plan 1.4; theory
        pages on micromechanics and failure). For spherical voids, the
        default shape, the two agree to round-off. For penny voids in the
        ply plane at ``Vp = 0.03`` (T800/epoxy) the directional mapping
        takes ``E33`` to 0.74 of pristine instead of 0.885, ``G13`` and
        ``G23`` to 0.85 instead of 0.90, ``Z_t`` to 0.80 instead of 0.935
        and ``S_13 = S_23`` to 0.87, while ``E22`` and ``G12`` recover to
        0.97; on the production mesh the ILSS knockdown drops from 0.930 to
        0.876 and the compression knockdown rises from 0.985 to 0.995 (flat
        voids parallel to the load). Held by :attr:`assembler` and part of
        its stiffness cache key, so direct, CG and thermal solves all see
        it; the pristine reference is shared (``Vp = 0`` is identical).
        Threaded to the failure criteria, the load factors and the interior
        metrics; pass the same value to
        :func:`~porosity_fe.delamination_onset`, which is post-processing
        and cannot read it from the results (the option is not recorded on
        :class:`FieldResults` or in the JSON export). Not calibrated
        against measured out-of-plane constants of porous laminates, and
        for crowded pennies (local ``Vp`` near 0.1) the Mori-Tanaka
        out-of-plane stiffness is 7-10 % too stiff with no projection to
        soften it. Unknown values raise :class:`ValueError`.

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
    laminate with no porosity, voids or delamination. It keeps the crack
    faces from interpenetrating (frictionless contact, ``crack_contact``),
    reports the virtual-crack-closure energy release rates along the front
    (:attr:`FieldResults.crack_front`) and, when the material has
    interlaminar toughness, the load factor at which the delamination
    starts to grow (:attr:`FieldResults.fracture_load_factor`), with the
    toughness knocked down by the porosity at each front node
    (``solve(..., toughness_knockdown=...)``); see the theory page
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
                 porosity_degradation: PorosityDegradation =
                 _DEFAULT_POROSITY_DEGRADATION,
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
                                         formulation=formulation,
                                         porosity_degradation=porosity_degradation)
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
        # The one free-DOF system (and LU) of the crack-face contact loop's
        # latest active set, kept apart from _lu_cache so the intermediate
        # active sets never evict the load cases' systems; see
        # _contact_solve.
        self._contact_lu_cache: OrderedDict[bytes, tuple] = OrderedDict()
        # The contact active set each recent load case settled on (prescribed
        # DOFs, bonded ties and load parts -> bool per pair), the warm start
        # of its next contact solve.
        self._contact_warm: OrderedDict[bytes, np.ndarray] = OrderedDict()

    @property
    def formulation(self) -> FEFormulation:
        """Element formulation, ``'hex8'`` or ``'hex8i'`` (read-only).

        Held by :attr:`assembler`, whose stiffness cache is keyed on it.
        """
        return self.assembler.formulation

    @property
    def porosity_degradation(self) -> PorosityDegradation:
        """Porosity degradation mapping, ``'isotropic'`` or ``'directional'``
        (read-only; see :data:`PorosityDegradation`).

        Held by :attr:`assembler`, whose stiffness cache is keyed on it;
        the failure criteria and load factors of :meth:`solve` follow it.
        """
        return self.assembler.porosity_degradation

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
              interior_margin: float | None = None,
              crack_contact: bool = True,
              toughness_knockdown: ToughnessKnockdownSpec = 'exponential',
              ) -> FieldResults:
        """Solve the static FE problem.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', 'ilss', 'flexure_3pt',
            'flexure_4pt', 'thermal', 'dcb', 'enf' or 'mmb'.
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
            ``'dcb'``, ``'enf'`` and ``'mmb'`` are the delamination
            fracture tests (ASTM D5528, D7905, D6671) on the specimens of
            :meth:`CompositeMesh.for_dcb`, :meth:`CompositeMesh.for_enf` and
            :meth:`CompositeMesh.for_mmb`, whose
            :attr:`CompositeMesh.fracture_geometry` they need (another mesh
            raises :class:`ValueError` before assembly): hinge loads on the
            two arms (DCB), three-point bending on rollers (ENF), or the
            loads of the MMB lever on rollers
            (:meth:`BoundaryHandler.fracture_bcs`). They report the
            energy release rates along the insert's front
            (:attr:`FieldResults.crack_front`), the critical load factor
            when the material has toughness data, and the load-point
            compliance (:attr:`FieldResults.load_point_compliance`);
            ``effective_modulus`` is ``None`` and the knockdown is the
            compliance of the bonded, pore-free specimen over the solved
            one's (see the theory page "Delamination").
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
            modes: the mid-span load of ``'ilss'`` (ASTM D2344),
            ``'flexure_3pt'`` and ``'enf'``, the sum of the two loads of
            ``'flexure_4pt'``, the lever load of ``'mmb'`` (negative presses
            the lever down, so the hinge opens the crack). For ``'dcb'`` it
            is the hinge load on each arm and only its magnitude is used: the
            arms are always pulled apart. Ignored for the other modes.
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
        crack_contact : bool
            Keyword-only, default ``True``. On a mesh with a delaminated
            node pair, enforce frictionless contact between the crack faces
            (zero initial gap): an active set of normal-only ties on the
            delaminated pairs, re-solved until no pair penetrates and none
            carries a tensile contact force (at most
            :attr:`_CONTACT_MAX_ITERATIONS` solves; a warning if the set
            still changes). Only the ties change between solves, never
            ``K``. With ``delta_T`` the contact is that of the total
            (residual plus mechanical) state, and both solves use its
            active set. ``False`` lets the faces interpenetrate (the
            behaviour before contact existed). Recorded as
            :attr:`FieldResults.crack_contact` with the iterations and the
            number of pairs in contact. No effect without a delaminated
            pair. On a delaminated mesh the solve also reports the
            virtual-crack-closure energy release rates along the front
            (:attr:`FieldResults.crack_front`) and, when the material has
            toughness data, the critical load factor
            (:attr:`FieldResults.fracture_load_factor`); see the theory page
            "Delamination".
        toughness_knockdown : ToughnessKnockdown, str, callable or None
            Keyword-only. How the porosity at each crack-front node knocks
            down the material's interlaminar toughness before the critical
            load factor is computed (:class:`~porosity_fe.ToughnessKnockdown`):
            the default ``'exponential'`` is ``G_c = G_c0 exp(-b Vp)`` with
            the uncalibrated placeholders ``b_I = 6.2`` and
            ``b_II = b_III = 14`` per unit ``Vp``, flagged as
            ``crack_front.toughness_uncalibrated`` (and in
            ``summary().details`` and the JSON ``fracture`` block) where they
            lower a toughness. ``None`` keeps the pristine toughness; a
            :class:`~porosity_fe.ToughnessKnockdown` sets other
            coefficients or the linear law, and a callable ``f(Vp, mode)``
            is used as the factor. Without porosity at the front the
            toughness is the pristine one whatever the law. No effect
            without a delaminated pair or without toughness data; an
            invalid value raises before anything is assembled.

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
            if the mesh has no :attr:`CompositeMesh.ilss_geometry`; for
            ``'dcb'``, ``'enf'`` and ``'mmb'``, if it is not that test's
            specimen (:attr:`CompositeMesh.fracture_geometry`). If
            ``toughness_knockdown`` names an unknown law or has invalid
            coefficients (:class:`TypeError` if it is of the wrong type).
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
        knockdown_law = _resolve_toughness_knockdown(toughness_knockdown)
        compliance: float | None = None
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
        F = np.zeros(0)
        if not thermal_only:
            constrained, F = self._apply_boundary_conditions(
                loading, applied_strain, applied_load, verbose=verbose)

        # A delaminated mesh ties its bonded pairs; with crack-face contact
        # the loop adds the normal ties of the pairs in contact and solves
        # the thermal and mechanical states with them.
        delaminated = _knockdown._is_delaminated(self.mesh)
        ties = _mesh_ties(self.mesh)
        contact: _contact.ContactState | None = None
        u_mech: np.ndarray | None = None
        thermal: _ThermalUnitSolution | None = None
        if delaminated and crack_contact:
            thermal, u_mech, contact = self._contact_solve(
                K, F, constrained, dT, solver=solver, rtol=rtol, verbose=verbose)
            ties = contact.ties
        elif dT is not None:
            # Free-standing thermal (residual) state, scaled from the cached
            # unit-delta_T solution; its factorization is freed before the
            # mechanical one is built.
            thermal = self._thermal_unit_solution(
                K, solver=solver, rtol=rtol, verbose=verbose,
                next_constrained=constrained)
        residual_global: np.ndarray | None = None
        residual_local: np.ndarray | None = None
        if thermal is not None:
            assert dT is not None
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
            if u_mech is None:
                # 3-4. Eliminate the prescribed DOFs and solve the reduced
                #      system.
                u_mech, _rel_res = self._solve_constrained(
                    K, F, constrained, solver=solver, rtol=rtol, verbose=verbose)
                if verbose:
                    t2 = time.perf_counter()
                    logger.info(
                        "  Solve time: %.2f s, residual: %.4e", t2 - t1, _rel_res)
                    t1 = t2
            u = u_mech

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
                applied_load=applied_load, ties=ties)
            if loading in _FRACTURE_TESTS and applied_load != 0.0:
                # Work-conjugate displacement of the load over the load.
                compliance = float(F @ u) / float(applied_load) ** 2
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
            interior=interior, thermal_only=thermal_only, delta_T=dT,
            porosity_degradation=self.porosity_degradation)

        # Delamination: interface forces, virtual crack closure along the
        # front and the critical load factor, from the unfolded residuals.
        fracture: _fracture.FractureOutputs | None = None
        if ties is not None:
            thermal_part = None
            if thermal is not None:
                assert dT is not None and thermal.residual is not None
                thermal_part = (dT * thermal.u, dT * thermal.residual)
            if thermal_only:
                assert thermal_part is not None
                fracture = _fracture.fracture_outputs(
                    self.mesh, self.material, ties, load=thermal_part,
                    toughness_knockdown=knockdown_law)
            else:
                assert u_mech is not None
                fracture = _fracture.fracture_outputs(
                    self.mesh, self.material, ties,
                    load=(u_mech, K @ u_mech - F), prestress=thermal_part,
                    toughness_knockdown=knockdown_law)

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
            crack_front=None if fracture is None else fracture.crack_front,
            fracture_load_factor=None if fracture is None else fracture.load_factor,
            fracture_governing_node=(None if fracture is None
                                     else fracture.governing_node),
            interface_forces=None if fracture is None else fracture.interface_forces,
            crack_contact=bool(crack_contact) if delaminated else None,
            contact_iterations=None if contact is None else contact.iterations,
            contact_pairs=None if contact is None else contact.n_pairs,
            contact_converged=None if contact is None else contact.converged,
            load_point_compliance=compliance,
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

    #: Cap on the linear solves of the crack-face contact loop
    #: (``solve(crack_contact=True)``).
    _CONTACT_MAX_ITERATIONS = 30

    def _solve_constrained(
        self, K: scipy.sparse.spmatrix, F: np.ndarray, constrained: dict[int, float],
        *, solver: Literal['direct', 'cg', 'minres'] = 'direct',
        rtol: float = 1e-9, verbose: bool = False,
        ties: Ties | None | object = _MESH_TIES,
        system: _constrained_solve.FreeDofSystem | None = None,
    ) -> tuple[np.ndarray, float]:
        """Impose the Dirichlet BCs exactly and solve ``K u = F``: the full
        ``u`` and the relative residual of the reduced system
        (``_constrained_solve.solve_constrained``), direct solves with this
        solver's cached factorizations (:meth:`_free_dof_system`, or
        ``system``). ``ties`` default to the bonded interface of a
        delaminated mesh (:meth:`CompositeMesh.bonded_ties`)."""
        tie_set = _mesh_ties(self.mesh) if ties is _MESH_TIES else ties
        assert tie_set is None or isinstance(tie_set, tuple)
        return _constrained_solve.solve_constrained(
            K, F, constrained,
            system=self._free_dof_system if system is None else system,
            ties=tie_set, solver=solver,
            rtol=rtol, verbose=verbose, max_restarts=self._MINRES_MAX_RESTARTS)

    def _contact_solve(
        self, K: scipy.sparse.spmatrix, F: np.ndarray,
        constrained: dict[int, float] | None, delta_T: float | None, *,
        solver: Literal['direct', 'cg', 'minres'] = 'direct',
        rtol: float = 1e-9, verbose: bool = False,
    ) -> tuple[_ThermalUnitSolution | None, np.ndarray | None, _contact.ContactState]:
        """Solve with frictionless crack-face contact (``_contact``).

        The contact is that of the total state: the thermal residual state
        (``delta_T`` times the unit solution) plus the mechanical solve
        (``constrained is None`` for ``loading='thermal'``), both solved
        with the bonded ties and the normal ties of the active pairs. The
        first solve has no pair active and uses :attr:`_lu_cache`, like a
        solve without contact; every later active set is factorized in
        :attr:`_contact_lu_cache`, which holds one system, so at most one
        factorization more than without contact is alive and the load
        cases' systems are never evicted by intermediate sets. The loop
        starts from the active set the same load case (prescribed DOFs,
        bonded ties, thermal and mechanical parts) settled on last time on
        this solver, so repeating a solve, or changing only the load
        magnitude or the porosity, typically takes one solve, and none
        when the last set's system is still in the cache. The active set
        the loop converges to does not depend on where it starts (up to
        pairs whose opening and contact force are both within round-off of
        zero).

        Returns the unit thermal solution (or ``None``), the mechanical
        displacement (or ``None``) and the contact state.
        """
        scratch = functools.partial(_constrained_solve.free_dof_system,
                                    self._contact_lu_cache, max_entries=1)
        prescribed = dict(constrained or {})
        if delta_T is not None:
            prescribed.update(self.bc_handler.free_bcs()[0])
        parts: dict[str, object] = {}

        def state(ties: Ties | None, with_contact: bool
                  ) -> tuple[np.ndarray, np.ndarray]:
            system = scratch if with_contact else None
            u = np.zeros(K.shape[0])
            r = np.zeros(K.shape[0])
            if delta_T is not None:
                th = self._thermal_unit_solution(
                    K, solver=solver, rtol=rtol, verbose=verbose,
                    next_constrained=constrained, ties=ties, system=system)
                assert th.residual is not None
                parts['thermal'] = th
                u = u + delta_T * th.u
                r = r + delta_T * th.residual
            if constrained is not None:
                u_m, _ = self._solve_constrained(
                    K, F, constrained, solver=solver, rtol=rtol,
                    verbose=verbose, ties=ties, system=system)
                parts['mechanical'] = u_m
                u = u + u_m
                r = r + (K @ u_m - F)
            return u, r

        base = _mesh_ties(self.mesh)
        h = hashlib.blake2b(digest_size=16)
        h.update(np.array(sorted(prescribed), dtype=np.int64).tobytes())
        h.update(_ties_token(base))
        h.update(bytes([constrained is None, delta_T is None]))
        warm_key = h.digest()
        rel_tol = 1e-9 if solver == 'direct' else max(1e-9, 100.0 * rtol)
        _, _, contact = _contact.solve_with_contact(
            state, self.mesh, prescribed, base,
            max_iterations=self._CONTACT_MAX_ITERATIONS, rel_tol=rel_tol,
            initial=self._contact_warm.get(warm_key))
        self._contact_warm[warm_key] = contact.active
        self._contact_warm.move_to_end(warm_key)
        while len(self._contact_warm) > 4:
            self._contact_warm.popitem(last=False)
        if verbose:
            logger.info("  Crack-face contact: %d pairs after %d solves%s",
                        contact.n_pairs, contact.iterations,
                        "" if contact.converged else " (not converged)")
        thermal = parts.get('thermal')
        u_mech = parts.get('mechanical')
        assert thermal is None or isinstance(thermal, _ThermalUnitSolution)
        assert u_mech is None or isinstance(u_mech, np.ndarray)
        return thermal, u_mech, contact

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
        *, applied_load: float = -10.0, ties: Ties | None | object = _MESH_TIES,
    ) -> tuple[np.ndarray, float | None]:
        """Nodal reactions and the effective modulus
        (``_postprocess.reactions_and_modulus``); ``ties`` are those of the
        solve, by default the mesh's bonded ties."""
        return _postprocess.reactions_and_modulus(
            loading, K, u, F, applied_strain, constrained,
            applied_load=applied_load, mesh=self.mesh,
            assembler=self.assembler, bc_handler=self.bc_handler, ties=ties)

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
        ties: Ties | None | object = _MESH_TIES,
        system: _constrained_solve.FreeDofSystem | None = None,
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
        ``ties`` default to the mesh's bonded ties (the crack-face contact
        loop passes its own); ``system`` replaces :meth:`_free_dof_system`
        (the contact loop's own cache, which needs no clean-up).
        """
        alpha_local = self.material.cte_vector()
        tie_set = _mesh_ties(self.mesh) if ties is _MESH_TIES else ties
        assert tie_set is None or isinstance(tie_set, tuple)
        key = (tuple(alpha_local.tolist()), solver, float(rtol), _ties_token(tie_set))
        cached = self._thermal_cache
        if cached is not None and cached[0] is K and cached[1] == key:
            return cached[2]

        supports = self.bc_handler.free_bcs()
        sol = _thermal.unit_solution(
            K, self.assembler.element_batch(), self.mesh, supports,
            alpha_local, verbose=verbose, ties=tie_set,
            solve=functools.partial(self._solve_constrained, solver=solver,
                                    rtol=rtol, verbose=verbose, ties=tie_set,
                                    system=system))
        self._thermal_cache = (K, key, sol)
        if system is None:
            thermal_key = _ConstraintReduction.from_constraints(
                K.shape[0], supports[0], ties=tie_set).key
            next_key = None if next_constrained is None else \
                _ConstraintReduction.from_constraints(
                    K.shape[0], next_constrained, ties=tie_set).key
            if thermal_key != next_key:
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
            self.material, self.porosity_field.void_shape_radii, elem_Vp,
            porosity_degradation=self.porosity_degradation)

    def _evaluate_failure(self, stress_local: np.ndarray,
                          criterion: str = 'tsai_wu'
                          ) -> tuple[float, np.ndarray, dict[str, float]]:
        """Failure indices for every Gauss point; see :func:`failure.evaluate_failure`."""
        return failure.evaluate_failure(
            stress_local, self.mesh.porosity, self.mesh.elements,
            self.material, self.porosity_field.void_shape_radii, criterion,
            void_elements=self.mesh.void_elements,
            porosity_degradation=self.porosity_degradation)

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
                       include_raw: bool = False, *,
                       onset: DelaminationOnset | None = None,
                       knockdown_decomposition: KnockdownDecomposition | None = None,
                       fracture_uncertainty: FractureLoadFactorBand | None = None,
                       ) -> None:
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
        onset : DelaminationOnset, optional
            Keyword-only. The stress-based interface onset of these results
            (:meth:`FieldResults.delamination_onset`). With ``fmt='json'``
            its :meth:`~porosity_fe.DelaminationOnset.summary` is written as
            a ``delamination_onset`` block (the criterion, ``x_avg_mm``,
            the maximum index, the critical interface and location, the
            onset load factors and the maximum index of each interface);
            with ``fmt='vtu'`` the per-node index is written as the point
            field ``onset_index`` (see :meth:`FieldResults.to_vtu`). Ignored
            for ``fmt='vtk'``. ``None`` (default) writes neither.
        knockdown_decomposition : KnockdownDecomposition, optional
            Keyword-only. The knockdown of these results split by defect
            (:func:`~porosity_fe.knockdown_decomposition`), written with
            ``fmt='json'`` as a ``knockdown_decomposition`` block
            (``knockdown_total``, ``knockdown_porosity_only``,
            ``knockdown_delamination_only``, ``dominant``). ``None``
            (default) writes none.
        fracture_uncertainty : FractureLoadFactorBand, optional
            Keyword-only. The uncertainty band on the fracture load factor
            of these results
            (:meth:`FieldResults.fracture_load_factor_uncertainty`),
            written with ``fmt='json'`` as a ``fracture_uncertainty``
            block (its :meth:`~porosity_fe.FractureLoadFactorBand.summary`:
            the nominal value, the statistics and percentiles of the
            draws, the inputs and their scatter, the share of draws each
            front node governs, and the share below the required load
            factor when one was given). ``None`` (default) writes none.

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
        * In the ``fracture`` block of a delaminated mesh:
          ``fracture_load_factor`` and the front's ``G_c_N_per_mm`` and
          ``lambda_c`` without toughness data or where infinite (no energy
          released), ``governing_*`` without a governing node, ``psi``
          where ``G_T = 0``, ``front`` without a front, the contact
          counts when ``crack_contact=False``, and ``toughness_knockdown``
          and ``toughness_uncalibrated`` without toughness data.
        * In the ``delamination_onset`` block (with ``onset``): ``Zc_MPa``
          without a compression term, the maximum indices, the critical
          interface and location where no node was evaluated, the onset
          load factors where infinite (the load drives no node to onset) or
          none was evaluated, and an interface's entry of
          ``interface_max_index`` where none of its nodes was evaluated.
        * In the ``fracture_uncertainty`` block (with
          ``fracture_uncertainty``): ``nominal``, an entry of
          ``nominal_per_node`` and the statistics and percentiles where
          infinite (no energy released), ``seed`` when unseeded, and
          ``required_load_factor`` and ``probability_below_required``
          without a required load factor.

        Any other non-finite value raises :class:`ValueError` naming its
        location, before a file is written.
        :func:`~porosity_fe.load_results_from_json` reads ``null`` as
        ``None``.
        """
        _export_results(field_results, filename, fmt=fmt, mesh=mesh,
                        include_raw=include_raw, onset=onset,
                        knockdown_decomposition=knockdown_decomposition,
                        fracture_uncertainty=fracture_uncertainty)
