"""Delamination growth by node release: quasi-static and Paris-law fatigue.

The front of an explicit :class:`~porosity_fe.Delamination` advances one
node pair at a time. Each configuration is solved (:meth:`FESolver.solve`)
and its front evaluated by the virtual crack closure technique
(:mod:`porosity_fe.fe.vcct`); the governing front pairs are then released
(:attr:`CompositeMesh.released_pairs <porosity_fe.CompositeMesh.released_pairs>`)
and the next configuration solved on the same assembled stiffness, with
only the bonded ties changed. Two analyses drive the release:

- :func:`delamination_growth`, quasi-static: every configuration's critical
  load factor ``lambda_c`` (the multiplier on the applied load at which the
  front first reaches its toughness) and the front pairs within
  ``release_tolerance`` of it are released. The sequence of critical loads
  against the crack area is the propagation envelope of linear elastic
  fracture mechanics: a falling critical load is unstable growth under
  load control, a falling critical displacement under displacement control.
- :func:`fatigue_delamination_growth`: a Paris law ``da/dN = C (G_max /
  G_c)^m`` (:class:`ParisLaw`, coefficients from the user: none ship) at
  every front node under the applied load as the cyclic peak; the node that
  needs the fewest cycles to grow by its element length is released, and
  the others keep the share of their length already grown.

Porosity enters through the degraded stiffness and through the toughness at
each front node, knocked down by its local porosity
(:class:`~porosity_fe.ToughnessKnockdown`), so a front that runs into a
porous region grows at a lower load or faster. Both analyses are linear
elastic between releases: no cohesive zone, no geometric nonlinearity, and
the growth is resolved to the element length at the front. See the theory
page "Delamination" (Growth).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import _knockdown
from ._load_cases import _DEFAULT_APPLIED_STRAIN, _FE_LOADINGS, _FORCE_CONTROLLED

if TYPE_CHECKING:
    from .results import FieldResults
    from .solver import FESolver
    from .vcct import CrackFront

logger = logging.getLogger("porosity_fe_analysis")

#: Load cases with a growth analysis: every mechanical one (a delamination
#: front grows under a load the factor scales, not under ``'thermal'``).
GROWTH_LOADINGS: tuple[str, ...] = tuple(m for m in _FE_LOADINGS if m != 'thermal')

#: Why a growth analysis stopped (:attr:`GrowthResult.stop_reason`).
GROWTH_STOP_REASONS: tuple[str, ...] = (
    'max_steps', 'max_area', 'max_cycles', 'not_driven', 'separated',
    'static', 'arrested')

#: Load-factor ties within this relative distance of the smallest always
#: release together (the two nodes of a symmetric front).
_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ParisLaw:
    """Paris law of delamination fatigue growth, normalized by the toughness.

    The growth rate at a front node under a cyclic load whose peak releases
    ``G_max`` is::

        da/dN = C (G_max / G_c)^m        for G_max / G_c >= threshold
        da/dN = 0                        below it

    with ``G_c`` the mixed-mode toughness of the node
    (:func:`~porosity_fe.critical_energy_release_rate`: Benzeggagh-Kenane at
    the node's mode mixity, knocked down by its porosity). Normalizing by
    ``G_c`` is the form of Krueger's fatigue benchmarks and of the
    ``da/dN``-``G_max`` laws of ASTM D6115: the porosity then speeds the
    growth through the toughness alone, at the same ``C`` and ``m``. No
    coefficients ship with PorosityFE: they depend on the material, the
    stress ratio ``R`` and the mode mixity of the tests that fitted them, and
    whether porosity changes ``C`` or ``m`` has no data here.

    Parameters
    ----------
    C : float
        Coefficient, mm per cycle (the rate at ``G_max = G_c``), positive and
        finite.
    m : float
        Exponent, positive and finite (delamination exponents are high, 5 to
        over 10).
    threshold : float
        Ratio ``G_max / G_c`` below which no growth occurs, in ``[0, 1)``;
        default 0 (no threshold).

    Raises
    ------
    ValueError
        For a coefficient that is not a positive finite number or a
        threshold outside ``[0, 1)``.

    Examples
    --------
    >>> law = ParisLaw(C=0.01, m=6.0, threshold=0.1)
    >>> float(law.rate(0.5))
    0.00015625
    >>> float(law.rate(0.05))
    0.0
    """

    C: float
    m: float
    threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in ('C', 'm'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
                    or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"ParisLaw.{name} must be a positive finite number, got {value!r}.")
            object.__setattr__(self, name, float(value))
        t = self.threshold
        if isinstance(t, bool) or not isinstance(t, (int, float, np.number)) \
                or not (0.0 <= float(t) < 1.0):
            raise ValueError(
                f"ParisLaw.threshold must be a ratio G_max / G_c in [0, 1), got {t!r}.")
        object.__setattr__(self, 'threshold', float(t))

    def rate(self, ratio: np.ndarray | float) -> np.ndarray:
        """Growth rate ``da/dN`` (mm per cycle) at the ratios ``G_max / G_c``."""
        r = np.asarray(ratio, dtype=float)
        grows = (r > 0.0) & (r >= self.threshold)
        return np.where(grows, self.C * np.where(grows, r, 0.0) ** self.m, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """The law as a JSON-ready dict."""
        return {'form': 'da/dN = C (G_max / G_c)^m', 'C_mm_per_cycle': self.C,
                'm': self.m, 'threshold': self.threshold}


@dataclass(frozen=True, eq=False)
class GrowthResult:
    """Configurations of a growing delamination, one row per configuration.

    Row ``k`` is the crack after ``k`` release steps: row 0 the delamination
    as given, the last row the grown one. Each was solved under the applied
    load (``applied_load`` or ``applied_strain`` of :meth:`FESolver.solve`),
    and every quantity of a row is at that load.

    Attributes
    ----------
    loading : str
        The load case.
    applied_strain : float or None
        The nominal strain of every solve (strain-controlled load cases);
        ``None`` for the force-controlled ones.
    applied_load : float or None
        The load of every solve (N; force-controlled load cases); ``None``
        for the strain-controlled ones.
    load_factors : numpy.ndarray
        Critical load factor ``lambda_c`` of each configuration, the
        smallest over its front: the applied load times it starts the growth
        of that crack (``inf`` when no front node is driven).
    released_area : numpy.ndarray
        Crack area grown before each configuration (mm^2): the sum of the
        tributary areas (``Delta_a b``,
        :attr:`CrackFront.area <porosity_fe.CrackFront.area>`) of the pairs
        released so far; 0 for row 0.
    released : tuple of numpy.ndarray
        The pairs (indices into :meth:`CompositeMesh.interface_node_pairs
        <porosity_fe.CompositeMesh.interface_node_pairs>`) released after
        each configuration, which give the next one; empty for the last row.
    governing_node : numpy.ndarray
        The front node that governs each configuration: the smallest
        ``lambda_c`` (quasi-static) or the fastest growth (fatigue); ``-1``
        when none does.
    G_T, G_c, psi, Vp : numpy.ndarray
        At the governing node of each configuration, under the applied load:
        total energy release rate and toughness (N/mm), mode mixity and
        crack-plane porosity (fraction); ``nan`` without a governing node.
    compliance : numpy.ndarray or None
        Load-point compliance of each configuration (mm/N) of the fracture
        tests (:attr:`FieldResults.load_point_compliance
        <porosity_fe.FieldResults.load_point_compliance>`); ``None`` for
        other load cases.
    effective_modulus : numpy.ndarray or None
        Effective modulus of each configuration (MPa) of the
        strain-controlled load cases; ``None`` for the others.
    released_pairs : numpy.ndarray
        Every pair released at the end, those of the mesh before the
        analysis included: assign it to :attr:`CompositeMesh.released_pairs
        <porosity_fe.CompositeMesh.released_pairs>` to solve the grown crack.
    stop_reason : str
        Why the analysis stopped (:data:`GROWTH_STOP_REASONS`):
        ``'max_steps'``, ``'max_area'`` or ``'max_cycles'`` (a limit
        reached), ``'not_driven'`` (no front node releases energy),
        ``'separated'`` (the next release would cut the interface through or
        leave the model unsupported), ``'static'`` (fatigue: the peak load
        reaches the static toughness, ``lambda_c <= 1``) or ``'arrested'``
        (fatigue: every front node below the threshold).
    final : FieldResults
        The solve of the last configuration.
    toughness_uncalibrated : bool
        ``True`` when an uncalibrated placeholder coefficient of the
        toughness law lowered a toughness in some configuration
        (:attr:`CrackFront.toughness_uncalibrated
        <porosity_fe.CrackFront.toughness_uncalibrated>`).
    crack_lengths : numpy.ndarray or None
        Mean crack length of each configuration of a fracture specimen
        (mm): its insert (:attr:`FractureSpecimenGeometry.crack_length
        <porosity_fe.FractureSpecimenGeometry.crack_length>`) plus the
        released area over the specimen width. ``None`` for other meshes.
    cycles : numpy.ndarray or None
        Fatigue: cycles to reach each configuration (0 for row 0).
        ``None`` for a quasi-static analysis.
    growth_rate : numpy.ndarray or None
        Fatigue: the largest ``da/dN`` over the front of each configuration
        (mm per cycle). ``None`` for a quasi-static analysis.
    paris : ParisLaw or None
        Fatigue: the growth law. ``None`` for a quasi-static analysis.
    """

    loading: str
    applied_strain: float | None
    applied_load: float | None
    load_factors: np.ndarray
    released_area: np.ndarray
    released: tuple[np.ndarray, ...]
    governing_node: np.ndarray
    G_T: np.ndarray
    G_c: np.ndarray
    psi: np.ndarray
    Vp: np.ndarray
    compliance: np.ndarray | None
    effective_modulus: np.ndarray | None
    released_pairs: np.ndarray
    stop_reason: str
    final: FieldResults
    toughness_uncalibrated: bool
    crack_lengths: np.ndarray | None = None
    cycles: np.ndarray | None = None
    growth_rate: np.ndarray | None = None
    paris: ParisLaw | None = None

    def __len__(self) -> int:
        return int(self.load_factors.size)

    @property
    def kind(self) -> Literal['quasi-static', 'fatigue']:
        """``'fatigue'`` with a :attr:`paris` law, else ``'quasi-static'``."""
        return 'fatigue' if self.paris is not None else 'quasi-static'

    @property
    def n_steps(self) -> int:
        """Release steps taken (rows less one)."""
        return len(self) - 1

    @property
    def critical_load(self) -> np.ndarray | None:
        """Critical load of each configuration (N): ``lambda_c`` times the
        applied load of a force-controlled load case (its magnitude for the
        DCB, whose arms are always pulled apart); ``None`` otherwise."""
        if self.applied_load is None:
            return None
        load = abs(self.applied_load) if self.loading == 'dcb' else self.applied_load
        return self.load_factors * load

    @property
    def critical_displacement(self) -> np.ndarray | None:
        """Critical load-point displacement of each configuration of a
        fracture test (mm, a magnitude): ``lambda_c`` times the compliance
        times the applied load (the DCB's opening of the two arms); ``None``
        for other load cases."""
        if self.compliance is None or self.applied_load is None:
            return None
        return self.load_factors * self.compliance * abs(self.applied_load)

    @property
    def critical_strain(self) -> np.ndarray | None:
        """Critical nominal strain of each configuration: ``lambda_c`` times
        the applied strain of a strain-controlled load case; ``None``
        otherwise."""
        if self.applied_strain is None:
            return None
        return self.load_factors * self.applied_strain

    @property
    def critical_stress(self) -> np.ndarray | None:
        """Critical nominal stress of each configuration (MPa): the critical
        strain times the configuration's effective modulus; ``None`` for the
        force-controlled load cases."""
        strain = self.critical_strain
        if strain is None or self.effective_modulus is None:
            return None
        return strain * self.effective_modulus

    def _levels(self, control: str) -> np.ndarray:
        if control == 'load':
            levels = self.critical_load
            if levels is None:
                levels = self.critical_stress
        elif control == 'displacement':
            levels = self.critical_displacement
            if levels is None:
                levels = self.critical_strain
        else:
            raise ValueError(
                f"control must be 'load' or 'displacement', got {control!r}.")
        if levels is None:
            raise ValueError(
                f"No critical {control} for {self.loading!r}: only the fracture "
                f"tests report a load-point displacement.")
        return np.abs(levels)

    def unstable(self, control: Literal['load', 'displacement'] = 'displacement'
                 ) -> np.ndarray:
        """Whether each release step grows the crack without a load increase.

        Step ``k`` (configuration ``k`` to ``k + 1``) is unstable under
        ``control`` when the next configuration's critical level is not
        above this one's: the crack keeps growing at the load or
        displacement already reached. Under load control the level is the
        critical load (the critical stress of a strain-controlled load
        case), under displacement control the critical load-point
        displacement (the critical strain). A DCB grows stably under
        displacement control and unstably under load control.

        Returns
        -------
        numpy.ndarray
            ``(n_steps,)`` booleans.

        Raises
        ------
        ValueError
            For another ``control``, or displacement control of a
            force-controlled load case that is not a fracture test (no
            load-point displacement).
        """
        levels = self._levels(control)
        return levels[1:] <= levels[:-1]

    def envelope(self, control: Literal['load', 'displacement'] = 'displacement'
                 ) -> np.ndarray:
        """The level at which each configuration is reached under a
        monotonic ``control``: the running maximum of the critical levels
        (magnitudes; see :meth:`unstable`).

        Where a configuration's own critical level is lower, the crack
        reached it in an unstable step, at the level of the last stable
        one. Under load control this is the load a test would record
        against the crack area; between two straight positions of a
        three-dimensional front, the staircase fronts of the node-by-node
        release dip below it and are crossed this way.
        """
        return np.maximum.accumulate(self._levels(control))

    def summary(self) -> dict[str, Any]:
        """The configurations as a JSON-ready dict (lists, ``None`` for
        non-finite values)."""
        def col(values: np.ndarray | None) -> list | None:
            if values is None:
                return None
            return [float(v) if math.isfinite(float(v)) else None
                    for v in np.asarray(values, dtype=float)]

        record: dict[str, Any] = {
            'kind': self.kind,
            'loading': self.loading,
            'applied_strain': self.applied_strain,
            'applied_load_N': self.applied_load,
            'stop_reason': self.stop_reason,
            'n_steps': self.n_steps,
            'toughness_uncalibrated': self.toughness_uncalibrated,
            'load_factors': col(self.load_factors),
            'released_area_mm2': col(self.released_area),
            'released_pairs': [int(p) for p in self.released_pairs],
            'n_released': [int(r.size) for r in self.released],
            'governing_node': [int(n) for n in self.governing_node],
            'G_T_N_per_mm': col(self.G_T),
            'G_c_N_per_mm': col(self.G_c),
            'psi': col(self.psi),
            'Vp': col(self.Vp),
            'crack_length_mm': col(self.crack_lengths),
            'critical_load_N': col(self.critical_load),
            'critical_displacement_mm': col(self.critical_displacement),
            'critical_strain': col(self.critical_strain),
            'critical_stress_MPa': col(self.critical_stress),
        }
        if self.paris is not None:
            record['paris'] = self.paris.to_dict()
            record['cycles'] = col(self.cycles)
            record['growth_rate_mm_per_cycle'] = col(self.growth_rate)
        return record


# --------------------------------------------------------------------------
# The release loop
# --------------------------------------------------------------------------

#: A policy reads a configuration's front and returns the front rows to
#: release, or a stop reason, and the governing row (``None``: none).
_Choice = tuple[np.ndarray | None, str | None, int | None]


def _check_common(fe_solver: FESolver, loading: str, max_steps: object,
                  max_area: object, release_tolerance: object,
                  solve_kwargs: dict[str, Any]) -> None:
    if loading not in GROWTH_LOADINGS:
        raise ValueError(
            f"Delamination growth analyses the mechanical load cases "
            f"{list(GROWTH_LOADINGS)}; got {loading!r}.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) \
            or int(max_steps) < 1:
        raise ValueError(f"max_steps must be an integer >= 1; got {max_steps!r}.")
    if max_area is not None and (
            isinstance(max_area, bool) or not isinstance(max_area, (int, float, np.number))
            or not math.isfinite(float(max_area)) or float(max_area) <= 0.0):
        raise ValueError(f"max_area must be a positive area (mm^2) or None; got {max_area!r}.")
    if isinstance(release_tolerance, bool) \
            or not isinstance(release_tolerance, (int, float, np.number)) \
            or not math.isfinite(float(release_tolerance)) or float(release_tolerance) < 0.0:
        raise ValueError(
            f"release_tolerance must be a finite number >= 0; got {release_tolerance!r}.")
    if not fe_solver.material.has_toughness:
        raise ValueError(
            "Delamination growth needs the interlaminar toughness "
            "(MaterialProperties.G_Ic, G_IIc and bk_eta).")
    if not _knockdown._is_delaminated(fe_solver.mesh):
        raise ValueError(
            "Delamination growth needs a delaminated mesh: build the "
            "CompositeMesh with delaminations= (a front to grow).")


def _grow(fe_solver: FESolver, loading: str, choose: Callable[[CrackFront], _Choice],
          *, max_steps: int, max_area: float | None,
          solve_kwargs: dict[str, Any], extra: Callable[[], None] | None = None,
          ) -> tuple[dict[str, Any], FieldResults]:
    """Solve, choose, release, repeat; the rows and the last solve.

    ``choose`` is the analysis: it reads the front of a configuration and
    returns the rows to release, or a stop reason. The mesh's released
    pairs are put back as they were, whatever happens.
    """
    mesh = fe_solver.mesh
    initial = mesh.released_pairs
    rows: dict[str, list] = {key: [] for key in (
        'load_factor', 'area', 'released', 'node', 'G_T', 'G_c', 'psi', 'Vp',
        'compliance', 'modulus')}
    uncalibrated = False
    area = 0.0
    stop = 'max_steps'
    try:
        results = fe_solver.solve(loading, **solve_kwargs)
        while True:
            front = results.crack_front
            assert front is not None and front.lambda_c is not None
            uncalibrated |= bool(front.toughness_uncalibrated)
            release, reason, row = choose(front)
            rows['load_factor'].append(
                float('inf') if front.load_factor is None else float(front.load_factor))
            rows['area'].append(area)
            if row is None:
                rows['node'].append(-1)
                for key in ('G_T', 'G_c', 'psi', 'Vp'):
                    rows[key].append(float('nan'))
            else:
                assert front.G_c is not None
                rows['node'].append(int(front.node[row]))
                rows['G_T'].append(float(front.G_T[row]))
                rows['G_c'].append(float(front.G_c[row]))
                rows['psi'].append(float(front.psi[row]))
                rows['Vp'].append(float(front.Vp[row]))
            rows['compliance'].append(results.load_point_compliance)
            rows['modulus'].append(results.effective_modulus)
            empty = np.zeros(0, dtype=np.intp)
            if reason is None and len(rows['area']) > max_steps:
                reason = 'max_steps'
            if reason is None and max_area is not None and area >= max_area:
                reason = 'max_area'
            if reason is not None:
                stop = reason
                rows['released'].append(empty)
                break
            assert release is not None
            pairs = front.pair[release]
            previous = mesh.released_pairs
            mesh.released_pairs = np.union1d(previous, pairs)
            try:
                mesh.bonded_pairs()
                grown = fe_solver.solve(loading, **solve_kwargs)
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, ValueError) and 'unconnected parts' not in str(exc):
                    raise
                logger.warning(
                    "Delamination growth stopped: releasing %d more pair(s) "
                    "leaves the laminate in unconnected parts or unsupported "
                    "(%s).", pairs.size, exc)
                mesh.released_pairs = previous
                stop = 'separated'
                rows['released'].append(empty)
                break
            rows['released'].append(np.sort(pairs))
            area += float(np.sum(front.area[release]))
            if extra is not None:
                extra()
            logger.info(
                "Delamination growth step %d: released %d pair(s), area "
                "%.4g mm^2, lambda_c %.4g.", len(rows['area']), pairs.size,
                area, rows['load_factor'][-1])
            results = grown
        final_pairs = mesh.released_pairs
    finally:
        mesh.released_pairs = initial
    record: dict[str, Any] = dict(rows)
    record.update(released_pairs=final_pairs, stop=stop, uncalibrated=uncalibrated)
    return record, results


def _result(fe_solver: FESolver, loading: str, rows: dict[str, Any],
            results: FieldResults, solve_kwargs: dict[str, Any], **fatigue: Any
            ) -> GrowthResult:
    if loading in _FORCE_CONTROLLED:
        applied_load: float | None = float(solve_kwargs.get('applied_load', -10.0))
        applied_strain: float | None = None
    else:
        strain = solve_kwargs.get('applied_strain')
        applied_strain = float(_DEFAULT_APPLIED_STRAIN.get(loading, -0.01)
                               if strain is None else strain)
        applied_load = None
    area = np.asarray(rows['area'], dtype=float)
    geometry = getattr(fe_solver.mesh, 'fracture_geometry', None)
    lengths = None if geometry is None else \
        geometry.crack_length + area / float(fe_solver.mesh.L_y)

    def optional(key: str) -> np.ndarray | None:
        values = rows[key]
        if any(v is None for v in values):
            return None
        return np.asarray(values, dtype=float)

    return GrowthResult(
        loading=loading, applied_strain=applied_strain, applied_load=applied_load,
        load_factors=np.asarray(rows['load_factor'], dtype=float),
        released_area=area, released=tuple(rows['released']),
        governing_node=np.asarray(rows['node'], dtype=np.intp),
        G_T=np.asarray(rows['G_T'], dtype=float), G_c=np.asarray(rows['G_c'], dtype=float),
        psi=np.asarray(rows['psi'], dtype=float), Vp=np.asarray(rows['Vp'], dtype=float),
        compliance=optional('compliance'), effective_modulus=optional('modulus'),
        released_pairs=rows['released_pairs'], stop_reason=rows['stop'], final=results,
        toughness_uncalibrated=rows['uncalibrated'], crack_lengths=lengths, **fatigue)


def _fatigue_release(rate: np.ndarray, delta_a: np.ndarray, share: np.ndarray,
                     tol: float) -> tuple[np.ndarray, float]:
    """The front rows a fatigue step releases, and its cycles.

    Row ``i`` grows its remaining length ``delta_a (1 - share)`` in
    ``delta_a (1 - share) / rate`` cycles (never at ``rate = 0``); the
    fewest are the step's, and every row within ``tol`` of it releases.
    """
    driven = rate > 0.0
    need = np.full(rate.shape, np.inf)
    need[driven] = delta_a[driven] * (1.0 - share[driven]) / rate[driven]
    dN = float(need.min())
    return np.flatnonzero(need <= dN * (1.0 + tol)), dN


def delamination_growth(fe_solver: FESolver, /, loading: str, *, max_steps: int = 50,
                        max_area: float | None = None, release_tolerance: float = 1e-3,
                        **solve_kwargs: Any) -> GrowthResult:
    """Quasi-static growth of the delamination of ``fe_solver``'s mesh.

    Solves the configuration, takes the critical load factor ``lambda_c`` of
    every front node (:attr:`CrackFront.lambda_c
    <porosity_fe.CrackFront.lambda_c>`), releases the nodes within
    ``release_tolerance`` of the smallest, and repeats on the grown crack.
    The critical load of each configuration against its crack area is the
    propagation envelope: under a monotonic load the crack grows along it,
    stably while the critical level rises and unstably where it falls
    (:meth:`GrowthResult.unstable`). A straight front across the width of a
    plane-strain specimen releases a node line per step; a three-dimensional
    front releases its most loaded nodes first and takes its own shape (a
    DCB's front leads at mid-width).

    Parameters
    ----------
    fe_solver : FESolver
        Positional-only. The model, on a mesh with a delamination
        (``CompositeMesh(..., delaminations=...)`` or a fracture specimen);
        its material needs toughness data (``G_Ic``, ``G_IIc``,
        ``bk_eta``). The mesh is left as it was (its
        :attr:`~porosity_fe.CompositeMesh.released_pairs` are put back);
        the analysis starts from the pairs already released there.
    loading : str
        A mechanical load case of :meth:`FESolver.solve` (not
        ``'thermal'``).
    max_steps : int
        Release steps at most (default 50); the result has up to
        ``max_steps + 1`` configurations.
    max_area : float, optional
        Stop once the released area reaches it (mm^2).
    release_tolerance : float
        Release, with the smallest ``lambda_c``, every front node whose
        factor is within this relative distance of it (default 0.1 %); 0
        releases only the governing node and its exact ties (nodes of a
        symmetric front). A larger value takes fewer, coarser steps.
    **solve_kwargs
        Other arguments of :meth:`FESolver.solve`, the same at every step:
        ``applied_load`` or ``applied_strain`` (the reference the factors
        multiply), ``toughness_knockdown``, ``delta_T`` (a residual
        prestress the factor does not scale), ``crack_contact`` and so on.

    Returns
    -------
    GrowthResult

    Raises
    ------
    ValueError
        For ``'thermal'`` or an unknown load case, a mesh without a
        delamination, a material without toughness data, or a limit that is
        not a positive number.

    Notes
    -----
    Every step solves the whole model again with the new ties: one sparse
    factorization per configuration (and per contact iteration), on the
    stiffness assembled once. The growth is resolved to the element length
    at the front; for a crack length to within ``Delta a``, mesh the region
    it grows into with equal elements (a fracture specimen's ``nx``).

    Examples
    --------
    >>> from porosity_fe import (MATERIALS, CompositeMesh, FESolver,
    ...                          PorosityField, delamination_growth)
    >>> mat = MATERIALS['IM7_8552_epoxy']
    >>> pf = PorosityField(mat, 0.02)
    >>> mesh = CompositeMesh.for_dcb(pf, mat, plane_strain=True, ny=1)
    >>> solver = FESolver(mesh, mat, pf, ply_angles='UD')
    >>> growth = delamination_growth(solver, 'dcb', max_steps=10)  # doctest: +SKIP
    >>> growth.critical_load, growth.crack_lengths  # doctest: +SKIP
    """
    _check_common(fe_solver, loading, max_steps, max_area, release_tolerance, solve_kwargs)
    tol = max(float(release_tolerance), _TIE_TOLERANCE)

    def choose(front: CrackFront) -> _Choice:
        lam = front.lambda_c
        assert lam is not None
        finite = np.isfinite(lam)
        if not finite.any():
            return None, 'not_driven', None
        smallest = float(lam[finite].min())
        release = np.flatnonzero(finite & (lam <= smallest * (1.0 + tol)))
        return release, None, front.governing

    rows, results = _grow(fe_solver, loading, choose, max_steps=int(max_steps),
                          max_area=None if max_area is None else float(max_area),
                          solve_kwargs=solve_kwargs)
    return _result(fe_solver, loading, rows, results, solve_kwargs)


def fatigue_delamination_growth(fe_solver: FESolver, /, loading: str, paris: ParisLaw, *,
                                max_steps: int = 50, max_area: float | None = None,
                                max_cycles: float | None = None,
                                release_tolerance: float = 1e-3,
                                **solve_kwargs: Any) -> GrowthResult:
    """Fatigue growth of the delamination of ``fe_solver``'s mesh, by a Paris law.

    The applied load is the peak of a constant-amplitude cycle. At every
    front node the growth rate is ``paris.rate(G_T / G_c)`` (``G_T`` the
    node's energy release rate under the peak, ``G_c`` its mixed-mode
    toughness, knocked down by its porosity), and a node grows its element
    length ``Delta a`` in ``Delta_a (1 - d) / (da/dN)`` cycles, ``d`` the
    share of it already grown. The node needing the fewest cycles is
    released with those within ``release_tolerance`` of it, the cycles are
    counted, every other front node keeps the share it grew meanwhile, and
    the grown crack is solved again.

    Parameters
    ----------
    fe_solver : FESolver
        Positional-only. The model, as for :func:`delamination_growth`.
    loading : str
        A mechanical load case of :meth:`FESolver.solve` (not
        ``'thermal'``).
    paris : ParisLaw
        The growth law, with the user's coefficients.
    max_steps : int
        Release steps at most (default 50).
    max_area : float, optional
        Stop once the released area reaches it (mm^2).
    max_cycles : float, optional
        Stop before a release that would need more cycles in all.
    release_tolerance : float
        Release with the fastest node every node whose cycles to release are
        within this relative distance of its (default 0.1 %).
    **solve_kwargs
        Other arguments of :meth:`FESolver.solve`: ``applied_load`` or
        ``applied_strain`` is the cyclic peak.

    Returns
    -------
    GrowthResult
        With :attr:`~GrowthResult.cycles`, :attr:`~GrowthResult.growth_rate`
        and :attr:`~GrowthResult.paris`. The growth stops with
        ``'static'`` when the peak reaches the static toughness of some
        front node (``lambda_c <= 1``: the crack grows on the first cycle)
        and with ``'arrested'`` when every front node is below the
        threshold.

    Raises
    ------
    ValueError
        As :func:`delamination_growth`, for a ``paris`` that is not a
        :class:`ParisLaw`, or a ``max_cycles`` that is not positive.

    Notes
    -----
    The shares carried between releases accumulate linearly (Miner's rule
    on the element length), the usual treatment of node release in fatigue;
    a node's rate is the one of the configuration it is in, so a coarse
    release step lags a rate that changes quickly with the crack length.
    The rate depends on ``G_max`` only: the stress ratio of the tests the
    coefficients came from is the one assumed.
    """
    _check_common(fe_solver, loading, max_steps, max_area, release_tolerance, solve_kwargs)
    if not isinstance(paris, ParisLaw):
        raise ValueError(f"paris must be a ParisLaw, got {paris!r}.")
    if max_cycles is not None and (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, (int, float, np.number))
            or not math.isfinite(float(max_cycles)) or float(max_cycles) <= 0.0):
        raise ValueError(f"max_cycles must be a positive number or None; got {max_cycles!r}.")
    tol = max(float(release_tolerance), _TIE_TOLERANCE)
    n_pairs = fe_solver.mesh.interface_node_pairs()[0].size
    grown = np.zeros(n_pairs)              # share of Delta_a grown, per pair
    cycles: list[float] = []               # to reach each configuration
    rates: list[float] = []
    total = 0.0
    pending: tuple[float, np.ndarray, np.ndarray] | None = None

    def choose(front: CrackFront) -> _Choice:
        nonlocal pending
        G_c = front.G_c
        assert G_c is not None
        ratio = np.where(G_c > 0.0, np.maximum(front.G_T, 0.0) / np.where(G_c > 0.0, G_c, 1.0),
                         0.0)
        rate = paris.rate(ratio)
        cycles.append(total)
        rates.append(float(rate.max()) if rate.size else 0.0)
        if ratio.size and float(ratio.max()) >= 1.0:
            logger.warning(
                "Fatigue delamination growth stopped: the cyclic peak reaches "
                "the static toughness at a front node (G_max / G_c = %.3g), so "
                "the crack grows on the first cycle.", float(ratio.max()))
            return None, 'static', int(np.argmax(ratio))
        if not (rate > 0.0).any():
            return None, 'arrested', None
        release, dN = _fatigue_release(rate, front.delta_a, grown[front.pair], tol)
        if max_cycles is not None and total + dN > float(max_cycles):
            return None, 'max_cycles', int(np.argmax(rate))
        pending = (dN, front.pair, rate / front.delta_a)
        return release, None, int(np.argmax(rate))

    def advance() -> None:
        # Called once the release is solved: count its cycles and keep the
        # share every other front node grew meanwhile.
        nonlocal total
        assert pending is not None
        dN, pair, share_rate = pending
        grown[pair] += dN * share_rate
        total += dN

    rows, results = _grow(fe_solver, loading, choose, max_steps=int(max_steps),
                          max_area=None if max_area is None else float(max_area),
                          solve_kwargs=solve_kwargs, extra=advance)
    return _result(fe_solver, loading, rows, results, solve_kwargs,
                   cycles=np.asarray(cycles, dtype=float),
                   growth_rate=np.asarray(rates, dtype=float), paris=paris)
