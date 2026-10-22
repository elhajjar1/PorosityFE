"""Mesh-convergence study for the FE solver (IMPROVEMENT_PLAN 3.6, B1).

:func:`refinement_study` solves one load case on a sequence of refined
meshes and reports, per quantity, how the result changes with the mesh:
relative changes between levels, the observed order of convergence and a
Richardson-extrapolated value when the three finest levels converge
monotonically, Roache's grid convergence index, and a converged flag.
Global quantities (knockdown, effective modulus) converge; the
domain-wide maximum failure index and first-ply-failure load factor sit on
corner, free-edge or load-point singularities and in general do not, which
the report flags instead of extrapolating.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np
from scipy.optimize import brentq

from .._types import FEFormulation, FELoadingMode
from ..materials import MaterialProperties
from ..mesh import (
    _D2344_ELEMENTS_PER_THICKNESS,
    _D2344_LENGTH_TO_THICKNESS,
    _D2344_WIDTH_TO_THICKNESS,
    CompositeMesh,
)
from ..pipeline import _default_mesh_resolution
from ..porosity_field import PorosityField
from ._load_cases import _FLEXURE_PROCEDURE, _check_delta_T, _minres_deprecation
from .element import _DEFAULT_FORMULATION, ELEMENT_FORMULATIONS
from .flexure import default_interior_margin
from .solver import FESolver

logger = logging.getLogger("porosity_fe_analysis")

#: Reported quantities in report order, each with its kind: ``'global'``
#: (an integral of the whole field; converges), ``'domain extremum'`` (an
#: extreme over every element; may sit on a singularity) or ``'interior
#: extremum'`` (an extreme over the elements away from the lateral faces).
QUANTITY_KINDS: dict[str, str] = {
    'knockdown': 'global',
    'effective_modulus': 'global',
    'max_failure_index': 'domain extremum',
    'first_ply_failure_load_factor': 'domain extremum',
    'interior_max_failure_index': 'interior extremum',
    'interior_first_ply_failure_load_factor': 'interior extremum',
}

#: Short column labels for the text tables.
_LABELS = {
    'knockdown': 'KD',
    'effective_modulus': 'E_eff MPa',
    'max_failure_index': 'max FI',
    'first_ply_failure_load_factor': 'FPF',
    'interior_max_failure_index': 'int. FI',
    'interior_first_ply_failure_load_factor': 'int. FPF',
}

#: Interior counterpart of each domain extremum.
_INTERIOR_OF = {
    'max_failure_index': 'interior_max_failure_index',
    'first_ply_failure_load_factor': 'interior_first_ply_failure_load_factor',
}

RefineMode = Literal['in_plane', 'all']

#: Safety factor of Roache's grid convergence index for three or more levels.
_GCI_SAFETY = 1.25
#: Bracket for the observed order of convergence.
_P_MIN, _P_MAX = 1e-6, 30.0
#: Smallest observed order for which a Richardson extrapolate is reported.
_P_TRUST = 0.5
#: Relative changes below ``tol * _NOISE_FRACTION`` count as zero.
_NOISE_FRACTION = 0.1

# Peak-memory model of one level, fitted on T800 QI meshes (hex8i, sparse LU
# with MMD_AT_PLUS_A) from 20x8x24 to 45x15x24 and at nz = 48: the nonzeros
# of L + U grow as about 8 n_dof^1.5 (measured 0.85-1.00 of that), held at
# about 12 bytes each, on top of the per-element assembly arrays of
# ``CompositeMesh._FE_BYTES_PER_ELEMENT`` and the interpreter with NumPy and
# SciPy loaded. It was 1.0-1.3 times the measured peak RSS up to 60x20x24.
_BASE_BYTES = 0.15e9
_LU_NNZ_COEFF = 8.0
_LU_NNZ_EXPONENT = 1.5
_LU_BYTES_PER_NNZ = 12.0
#: Fraction of physical memory a study may plan to use by default.
_DEFAULT_MEMORY_FRACTION = 0.8

#: ``'ilss'`` levels are ASTM D2344 specimens (``CompositeMesh.for_ilss``):
#: 6 h long with the supports h in from the ends, so ``nx`` must be a
#: multiple of 6 for the supports and mid-span to stay on node lines.
_ILSS_NX_MULTIPLE = 6
#: Default coarsest ``'ilss'`` level in plane: the ``for_ilss`` default
#: (element length h / 4), 24 x 8 elements.
_ILSS_BASE_NX = int(_D2344_LENGTH_TO_THICKNESS * _D2344_ELEMENTS_PER_THICKNESS)
_ILSS_BASE_NY = int(_D2344_WIDTH_TO_THICKNESS * _D2344_ELEMENTS_PER_THICKNESS)


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementLevel:
    """One mesh of a :func:`refinement_study`, with its results.

    Attributes
    ----------
    resolution : tuple of int
        ``(nx, ny, nz)`` element counts.
    n_elements, n_dof : int
        Mesh size.
    element_size : float
        Characteristic element size (mm) over the refined axes: the
        geometric mean of their element lengths. The refinement ratios of
        the convergence analysis are ratios of this size.
    solve_time_s : float
        Wall time of the level (mesh generation, the pristine reference
        solve when not cached, assembly, factorization, recovery and
        failure evaluation).
    estimated_memory_gb : float
        Planned peak memory of the level (GB); a deliberately generous
        estimate (measured peaks were 0.75 to 1.0 of it).
    knockdown, effective_modulus, max_failure_index,
    first_ply_failure_load_factor, interior_max_failure_index,
    interior_first_ply_failure_load_factor : float or None
        The :class:`~porosity_fe.FieldResults` quantities of the level.
        ``None`` where the solve does not define them (``effective_modulus``
        for ``'ilss'``, interior values without interior elements) or where
        they are not finite (``knockdown`` of a thermal solve).
    """

    resolution: tuple[int, int, int]
    n_elements: int
    n_dof: int
    element_size: float
    solve_time_s: float
    estimated_memory_gb: float
    knockdown: float | None
    effective_modulus: float | None
    max_failure_index: float | None
    first_ply_failure_load_factor: float | None
    interior_max_failure_index: float | None
    interior_first_ply_failure_load_factor: float | None

    @property
    def mesh(self) -> str:
        """The resolution as ``'nx x ny x nz'`` text."""
        return "x".join(str(n) for n in self.resolution)

    def value(self, quantity: str) -> float | None:
        """The value of one of the reported quantities (see ``QUANTITY_KINDS``)."""
        if quantity not in QUANTITY_KINDS:
            raise KeyError(f"Unknown quantity {quantity!r}; use one of {list(QUANTITY_KINDS)}.")
        return getattr(self, quantity)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict; non-finite numbers become ``None``."""
        out: dict[str, Any] = {
            'resolution': list(self.resolution),
            'n_elements': self.n_elements,
            'n_dof': self.n_dof,
            'element_size_mm': _finite_or_none(self.element_size),
            'solve_time_s': _finite_or_none(self.solve_time_s),
            'estimated_memory_gb': _finite_or_none(self.estimated_memory_gb),
        }
        for name in QUANTITY_KINDS:
            out[name] = _finite_or_none(getattr(self, name))
        return out


@dataclass(frozen=True)
class QuantityConvergence:
    """Convergence of one quantity over the levels of a :func:`refinement_study`.

    Attributes
    ----------
    name : str
        The quantity (a :class:`~porosity_fe.FieldResults` attribute).
    kind : str
        ``'global'``, ``'domain extremum'`` or ``'interior extremum'``.
    values : tuple of float or None
        Value per level, coarse to fine.
    relative_changes : tuple of float or None
        Signed change from each level to the next, relative to the finer
        value. For ``knockdown`` it is relative to the stiffness loss
        ``1 - KD`` instead, since ``KD`` close to 1 would make any mesh
        look converged. ``None`` where a value is missing.
    behavior : str
        From the three finest levels: ``'mesh-independent'`` (both changes
        below ``tol / 10``, too small to matter at the tolerance),
        ``'monotone'`` (changes of one sign that shrink faster than the
        logarithm of the element size: monotone convergence),
        ``'non-monotone'`` (the changes alternate in sign), ``'diverging'``
        (they do not shrink: the quantity grows without bound with
        refinement, the signature of a stress singularity, or the mesh is
        still too coarse to show the trend). ``'two levels'`` when only two
        levels were solved and ``'unavailable'`` when a value is missing at
        any level.
    observed_order : float or None
        Observed order of convergence ``p`` (``|error| ~ h^p``) from the
        three finest levels, for ``'monotone'`` only. ``None`` also when the
        finest change is negligible next to the previous one (``p`` above
        30).
    extrapolated : float or None
        Richardson-extrapolated value, ``f1 + (f1 - f2) / (r^p - 1)``, for
        ``'monotone'`` with ``observed_order`` at least 0.5 (below that the
        correction is too large to trust; the finest value when
        ``observed_order`` is ``None``); the finest value for
        ``'mesh-independent'``. With ``refine='in_plane'`` it is the limit
        at the fixed ``nz``.
    gci : float or None
        Roache's grid convergence index of the finest level,
        ``1.25 |f1 - f2| / |f1| / (r^p - 1)`` (relative, as a fraction;
        ``1 - KD`` scale for knockdown), for ``'monotone'``.
    error_estimate : float or None
        Estimated relative error of the finest value that ``converged`` is
        judged on: 0 for ``'mesh-independent'``; the GCI for
        ``'monotone'`` (the last change when ``observed_order`` is
        ``None``); the larger of the last two changes for
        ``'non-monotone'``; the last change for ``'two levels'``; ``None``
        (unbounded) for ``'diverging'`` and ``'unavailable'``.
    converged : bool or None
        ``error_estimate <= tol``; ``None`` for ``'unavailable'``.
    verdict : str
        How to read the quantity: ``'converged'``; ``'mesh-dependent'``
        when it is not converged, its last change exceeds ``tol`` and it
        diverges or converges with an observed order below 0.5, a rate no
        practical refinement settles (a stress singularity: do not use the
        value); ``'not converged'`` otherwise (refine further); ``'n/a'``
        when unavailable.
    """

    name: str
    kind: str
    values: tuple[float | None, ...]
    relative_changes: tuple[float | None, ...]
    behavior: str
    observed_order: float | None
    extrapolated: float | None
    gci: float | None
    error_estimate: float | None
    converged: bool | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict; non-finite numbers become ``None``."""
        return {
            'kind': self.kind,
            'values': [_finite_or_none(v) for v in self.values],
            'relative_changes': [_finite_or_none(v) for v in self.relative_changes],
            'behavior': self.behavior,
            'observed_order': _finite_or_none(self.observed_order),
            'extrapolated': _finite_or_none(self.extrapolated),
            'gci': _finite_or_none(self.gci),
            'error_estimate': _finite_or_none(self.error_estimate),
            'converged': self.converged,
            'verdict': self.verdict,
        }


@dataclass(frozen=True)
class RefinementStudy:
    """Result of :func:`refinement_study`: the levels and per-quantity convergence.

    ``repr()`` prints a short table of the levels; :meth:`summary` adds the
    convergence analysis and notes; :meth:`to_dict` gives a JSON-friendly
    dict.

    Attributes
    ----------
    loading : str
        The load case.
    refine : str
        ``'in_plane'``, ``'all'`` or ``'explicit'`` (``resolutions=``).
    refined_axes : tuple of str
        Axes whose element count changes between levels (``'x'``, ``'y'``,
        ``'z'``); the element size is measured over them.
    formulation, failure_criterion : str
        As used by every level.
    tol : float
        Convergence tolerance of :attr:`QuantityConvergence.converged`.
    interior_margin : float
        Distance (mm) from the lateral faces that the interior quantities
        leave out, the same at every level (for ``'ilss'``, from the
        supports, the loading band and the side faces; for the flexure
        modes, from the support and load lines and the side faces).
    void_volume_fraction : float
        Specimen-average ``Vp`` of the porosity field (fraction).
    applied_strain, applied_load, delta_T : float or None
        Load inputs as passed (``None`` = the solver's default / no thermal
        load).
    levels : tuple of RefinementLevel
        Coarse to fine.
    convergence : dict of str to QuantityConvergence
        Keyed by quantity name, in report order.
    """

    loading: str
    refine: str
    refined_axes: tuple[str, ...]
    formulation: str
    failure_criterion: str
    tol: float
    interior_margin: float
    void_volume_fraction: float
    applied_strain: float | None
    applied_load: float
    delta_T: float | None
    levels: tuple[RefinementLevel, ...]
    convergence: dict[str, QuantityConvergence]

    @property
    def resolutions(self) -> tuple[tuple[int, int, int], ...]:
        """``(nx, ny, nz)`` of every level, coarse to fine."""
        return tuple(lv.resolution for lv in self.levels)

    @property
    def n_elements(self) -> tuple[int, ...]:
        """Element count of every level, coarse to fine."""
        return tuple(lv.n_elements for lv in self.levels)

    @property
    def converged(self) -> dict[str, bool | None]:
        """``{quantity: converged}`` (``None`` for unavailable quantities)."""
        return {k: c.converged for k, c in self.convergence.items()}

    def _available(self) -> list[str]:
        return [k for k, c in self.convergence.items() if c.behavior != 'unavailable']

    def _level_table(self) -> list[str]:
        names = self._available()
        head = (f"{'mesh':>10} {'elements':>9} {'DOF':>8} {'time s':>7}"
                + "".join(f" {_LABELS[n]:>10}" for n in names))
        rows = [head]
        for lv in self.levels:
            rows.append(
                f"{lv.mesh:>10} {lv.n_elements:>9,} {lv.n_dof:>8,} "
                f"{lv.solve_time_s:>7.1f}"
                + "".join(f" {_fmt(lv.value(n)):>10}" for n in names))
        verdicts = {'converged': 'yes', 'mesh-dependent': 'MESH-DEP.',
                    'not converged': 'no', 'n/a': '-'}
        rows.append(f"{'converged':>10} {'':>9} {'':>8} {'':>7}" + "".join(
            f" {verdicts[self.convergence[n].verdict]:>10}" for n in names))
        return rows

    def __repr__(self) -> str:
        title = (f"RefinementStudy({self.loading}, {self.formulation}, "
                 f"refine={self.refine}, tol={self.tol:g})")
        return "\n".join([title, *self._level_table()])

    def summary(self) -> str:
        """Text report: the level table, the convergence analysis and notes."""
        nz = {lv.resolution[2] for lv in self.levels}
        refine = {'in_plane': f"in-plane refinement, nz = {next(iter(nz))} fixed"
                  if len(nz) == 1 else "in-plane refinement",
                  'all': "refinement of all axes", 'explicit': "explicit resolutions"}
        lines = [
            f"Mesh refinement study: loading={self.loading!r}, "
            f"formulation={self.formulation!r}, criterion={self.failure_criterion!r}, "
            f"{refine.get(self.refine, self.refine)} (refined axes: "
            f"{', '.join(self.refined_axes)}), Vp={self.void_volume_fraction:g}"
            + (f", delta_T={self.delta_T:g} K" if self.delta_T is not None else ""),
            f"interior quantities leave out {self.interior_margin:.3g} mm "
            + ("around the supports and the loading band and from the side faces"
               if self.loading == 'ilss' else
               "around the support and load lines and from the side faces"
               if self.loading in _FLEXURE_PROCEDURE else "from the lateral faces")
            + f"; tol = {self.tol:g}",
            "",
            *self._level_table(),
            "",
            f"{'quantity':<38} {'kind':<8} {'last change':>11} {'behavior':>16} "
            f"{'order':>6} {'extrapolated':>12} {'GCI':>9}  verdict",
        ]
        for name, c in self.convergence.items():
            if c.behavior == 'unavailable':
                continue
            last = c.relative_changes[-1] if c.relative_changes else None
            lines.append(
                f"{name:<38} {c.kind.split()[0]:<8} {_pct(last):>11} "
                f"{c.behavior:>16} {_fmt(c.observed_order, 3):>6} "
                f"{_fmt(c.extrapolated):>12} {_pct(c.gci):>9}  {c.verdict}")
        notes = self._notes()
        if notes:
            lines += ["", "Notes:"] + [f"- {n}" for n in notes]
        return "\n".join(lines)

    def _notes(self) -> list[str]:
        notes: list[str] = []
        conv = self.convergence
        if conv['interior_max_failure_index'].behavior == 'unavailable':
            notes.append(
                f"No element lies {self.interior_margin:.3g} mm from the lateral "
                f"faces on every level, so the interior quantities are missing; "
                f"pass a smaller interior_margin or refine the base mesh.")
        for dom, inner in _INTERIOR_OF.items():
            d, i = conv[dom], conv[inner]
            if d.converged is False and i.converged:
                notes.append(
                    f"{dom} does not converge but {inner} does: the domain "
                    f"extreme sits on a stress singularity at a constrained "
                    f"corner, free edge or load point, which grows with "
                    f"refinement. Use {inner}.")
            elif d.converged is False and i.converged is False:
                notes.append(
                    f"Neither {dom} nor {inner} converges: a singularity "
                    f"inside the interior region, a disturbance carried in "
                    f"from the free edges or the edge of the region (in "
                    f"'ilss', next to the excluded loading-nose zone, where "
                    f"the bending moment peaks; in three-point flexure, next "
                    f"to the half-disc under the load), or meshes still too coarse "
                    f"for the trend. Do not read these as mesh-converged "
                    f"values.")
        kd = conv['knockdown']
        if kd.behavior != 'unavailable':
            loss = 1.0 - float(kd.values[-1] or 0.0)
            note = (f"knockdown changes, GCI and tolerance are relative to the "
                    f"stiffness loss 1 - KD ({loss:.3g} on the finest mesh), not "
                    f"to KD")
            if kd.error_estimate is not None and kd.error_estimate > 0.0:
                note += (f"; the estimated error in KD itself is "
                         f"{kd.error_estimate * abs(loss):.2g}")
            notes.append(note + ".")
        if self.refine == 'in_plane':
            notes.append("Extrapolated values are the limit of in-plane "
                         "refinement at the fixed nz.")
        return notes

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (``json.dumps(..., allow_nan=False)`` works).

        Non-finite numbers become ``None``; tuples become lists.
        """
        return {
            'loading': self.loading,
            'refine': self.refine,
            'refined_axes': list(self.refined_axes),
            'formulation': self.formulation,
            'failure_criterion': self.failure_criterion,
            'tol': _finite_or_none(self.tol),
            'interior_margin_mm': _finite_or_none(self.interior_margin),
            'void_volume_fraction': _finite_or_none(self.void_volume_fraction),
            'applied_strain': _finite_or_none(self.applied_strain),
            'applied_load': _finite_or_none(self.applied_load),
            'delta_T': _finite_or_none(self.delta_T),
            'levels': [lv.to_dict() for lv in self.levels],
            'convergence': {k: c.to_dict() for k, c in self.convergence.items()},
        }


# ----------------------------------------------------------------------
# Convergence analysis (pure functions)
# ----------------------------------------------------------------------

def _finite_or_none(x: Any) -> Any:
    """``float(x)`` when finite, else ``None`` (also for ``None``)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fmt(x: float | None, digits: int = 6) -> str:
    return '-' if x is None or not math.isfinite(x) else f"{x:.{digits}g}"


def _pct(x: float | None) -> str:
    return '-' if x is None or not math.isfinite(x) else f"{100 * x:+.3g}%"


def _observed_order(rho: float, r21: float, r32: float) -> float | None:
    """Order ``p`` with ``(r32^p - 1) / (1 - r21^-p) = rho``.

    The left side is the ratio of the coarse to the fine change,
    ``(f3 - f2) / (f2 - f1)``, of ``f(h) = f0 + C h^p`` at element sizes
    ``h3 = r32 h2`` and ``h2 = r21 h1``. It increases with ``p`` from
    ``ln r32 / ln r21`` at ``p -> 0``, so a root exists only for a larger
    ``rho``. With equal ratios it reduces to ``p = ln(rho) / ln(r)``.
    Returns ``None`` above ``_P_MAX`` and ``0.0`` below ``_P_MIN`` (no
    root: the changes do not shrink).
    """
    def g(p: float) -> float:
        return math.expm1(p * math.log(r32)) / -math.expm1(-p * math.log(r21)) - rho

    if g(_P_MAX) < 0.0:
        return None
    if g(_P_MIN) >= 0.0:
        return 0.0
    return float(brentq(g, _P_MIN, _P_MAX, xtol=1e-12, rtol=1e-12))


def _three_level_trend(values: Sequence[float], sizes: Sequence[float],
                       *, noise: float = 0.0,
                       ) -> tuple[str, float | None, float | None]:
    """Classify three levels and Richardson-extrapolate when monotone.

    Parameters
    ----------
    values : sequence of 3 float
        ``(f3, f2, f1)``, coarse to fine.
    sizes : sequence of 3 float
        Element sizes ``(h3, h2, h1)``, strictly decreasing.
    noise : float
        Absolute change at or below which a change counts as zero.

    Returns
    -------
    behavior : str
        ``'mesh-independent'``, ``'monotone'``, ``'non-monotone'`` or
        ``'diverging'`` (see :class:`QuantityConvergence`).
    p : float or None
        Observed order for ``'monotone'`` (``None`` when above 30).
    extrapolated : float or None
        Richardson extrapolate for ``'monotone'`` with ``p`` at least 0.5
        (the finest value when ``p`` is ``None``; below 0.5 the correction
        ``(f1 - f2) / (r^p - 1)`` is too large to trust) and the finest
        value for ``'mesh-independent'``.
    """
    f3, f2, f1 = (float(v) for v in values)
    h3, h2, h1 = (float(h) for h in sizes)
    if not (h3 > h2 > h1 > 0.0):
        raise ValueError(f"element sizes must decrease strictly, got {tuple(sizes)}.")
    e32, e21 = f3 - f2, f2 - f1
    if abs(e32) <= noise and abs(e21) <= noise:
        return 'mesh-independent', None, f1
    if abs(e21) <= noise:          # converged faster than measurable
        return 'monotone', None, f1
    if abs(e32) <= noise:          # was flat, now changes: not converging
        return 'diverging', None, None
    if e32 * e21 < 0.0:
        return 'non-monotone', None, None
    r32, r21 = h3 / h2, h2 / h1
    p = _observed_order(e32 / e21, r21, r32)
    if p is None:
        return 'monotone', None, f1
    if p <= 0.0:
        return 'diverging', None, None
    if p < _P_TRUST:
        return 'monotone', p, None
    return 'monotone', p, f1 - e21 / math.expm1(p * math.log(r21))


def _assess(name: str, values: Sequence[float | None], sizes: Sequence[float],
            tol: float) -> QuantityConvergence:
    """Per-quantity convergence record for :func:`refinement_study`."""
    kind = QUANTITY_KINDS[name]
    vals = tuple(_finite_or_none(v) for v in values)
    if any(v is None for v in vals):
        return QuantityConvergence(
            name=name, kind=kind, values=vals,
            relative_changes=tuple(None for _ in vals[1:]),
            behavior='unavailable', observed_order=None, extrapolated=None,
            gci=None, error_estimate=None, converged=None, verdict='n/a')
    v = np.asarray(vals, dtype=float)
    # Work on the stiffness loss for knockdown: the same p and extrapolate
    # (an affine map), but relative changes against 1 - KD.
    deficit = name == 'knockdown'
    q = 1.0 - v if deficit else v
    floor = 1e-12 * max(float(np.max(np.abs(v))), 1e-300)
    noise_rel = tol * _NOISE_FRACTION

    def rel(a: float, b: float) -> float:
        """Change from ``a`` to ``b`` relative to ``|b|``."""
        d = b - a
        if abs(d) <= floor:
            return 0.0
        return d / abs(b) if abs(b) > floor else math.copysign(math.inf, d)

    changes = tuple(rel(float(q[i]), float(q[i + 1])) for i in range(q.size - 1))
    p: float | None = None
    extrap_q: float | None = None
    gci: float | None = None
    error: float | None
    if q.size < 3:
        if abs(changes[-1]) <= noise_rel:
            behavior, extrap_q, error = 'mesh-independent', float(q[-1]), 0.0
        else:
            behavior, error = 'two levels', abs(changes[-1])
    else:
        scale = max(abs(float(q[-1])), floor)
        behavior, p, extrap_q = _three_level_trend(
            q[-3:], sizes[-3:], noise=noise_rel * scale)
        if behavior == 'mesh-independent':
            error = 0.0
        elif behavior == 'monotone':
            if p is None:
                error = abs(changes[-1])
            else:
                gci = _GCI_SAFETY * abs(changes[-1]) / math.expm1(
                    p * math.log(sizes[-2] / sizes[-1]))
                error = gci
        elif behavior == 'non-monotone':
            error = max(abs(changes[-1]), abs(changes[-2]))
        else:  # diverging
            error = None
    extrapolated = None if extrap_q is None else (1.0 - extrap_q if deficit else extrap_q)
    converged = error is not None and error <= tol
    stalled = behavior == 'diverging' or (p is not None and p < _P_TRUST)
    if converged:
        verdict = 'converged'
    elif stalled and abs(changes[-1]) > tol:
        verdict = 'mesh-dependent'
    else:
        verdict = 'not converged'
    return QuantityConvergence(
        name=name, kind=kind, values=vals,
        relative_changes=tuple(_finite_or_none(c) for c in changes),
        behavior=behavior, observed_order=p, extrapolated=extrapolated,
        gci=gci, error_estimate=error, converged=converged, verdict=verdict)


# ----------------------------------------------------------------------
# Level planning and guards
# ----------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _check_resolution(res: Any, what: str) -> tuple[int, int, int]:
    try:
        nx, ny, nz = res
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be an (nx, ny, nz) triple, got {res!r}.") from None
    out = []
    for n in (nx, ny, nz):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"{what} must hold positive integers, got {res!r}.")
        out.append(int(n))
    return (out[0], out[1], out[2])


def _plan_resolutions(base: tuple[int, int, int], factors: Sequence[float],
                      refine: str, n_plies: int) -> list[tuple[int, int, int]]:
    """Resolutions of the levels for refinement ``factors`` of ``base``.

    ``nx`` and ``ny`` scale by the factor (rounded half up). With
    ``refine='in_plane'`` ``nz`` stays at the base value; with ``'all'`` it
    scales too, rounded to a whole multiple of ``n_plies`` (of the base
    ``nz`` when that is not itself a multiple), so every level resolves the
    same layup.
    """
    nx0, ny0, nz0 = base
    unit = n_plies if nz0 % n_plies == 0 else nz0
    out = []
    for f in factors:
        nx = max(1, _round_half_up(nx0 * f))
        ny = max(1, _round_half_up(ny0 * f))
        nz = nz0 if refine == 'in_plane' else \
            unit * max(1, _round_half_up(nz0 * f / unit))
        out.append((nx, ny, nz))
    return out


def _estimate_memory_bytes(res: tuple[int, int, int], *, n_factorizations: int = 1
                           ) -> float:
    """Generous peak-memory estimate (bytes) of one level; see ``_BASE_BYTES``.

    ``n_factorizations`` is the number of sparse LU factors alive at the
    peak (see :func:`_peak_factorizations`).
    """
    nx, ny, nz = res
    n_el = nx * ny * nz
    n_dof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    lu = _LU_BYTES_PER_NNZ * _LU_NNZ_COEFF * n_dof ** _LU_NNZ_EXPONENT
    return _BASE_BYTES + n_el * CompositeMesh._FE_BYTES_PER_ELEMENT + n_factorizations * lu


def _peak_factorizations(loading: str, solver: str, field: PorosityField) -> int:
    """Sparse LU factors alive at once during a level's solve.

    :meth:`FESolver.solve` holds one at a time: the pristine reference of
    the knockdown (always factorized directly) is freed before the porous
    model is built, and the free-standing thermal problem of a solve with
    ``delta_T`` before the mechanical one. So a direct solve peaks at one,
    and an iterative one at the pristine reference's, if it needs one.
    """
    if solver == 'direct':
        return 1
    porous = field.Vp > 0.0 or bool(field.discrete_voids)
    return 1 if (porous and loading != 'thermal') else 0


def _physical_memory_bytes() -> int | None:
    """Total physical memory, or ``None`` where ``os.sysconf`` cannot tell."""
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        size = os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        return None
    if pages <= 0 or size <= 0:
        return None
    return int(pages) * int(size)


def _resolve_porosity(material: MaterialProperties, porosity: PorosityField | float,
                      porosity_config: Mapping[str, Any] | None) -> PorosityField:
    if isinstance(porosity, PorosityField):
        if porosity_config:
            raise ValueError(
                "porosity_config only applies when porosity is a Vp fraction; "
                "a PorosityField already carries its configuration.")
        if porosity.material is not material and porosity.material != material:
            raise ValueError(
                "The PorosityField was built for a different MaterialProperties "
                "than the one passed to refinement_study.")
        return porosity
    if isinstance(porosity, bool) or not isinstance(porosity, (int, float, np.number)):
        raise TypeError(
            "porosity must be a PorosityField or a void volume fraction Vp in "
            f"[0, 1], got {type(porosity).__name__}.")
    return PorosityField(material, float(porosity), **dict(porosity_config or {}))


def _check_layups(material: MaterialProperties, field: PorosityField,
                  nz_values: Sequence[int], ply_angles: list[float] | str,
                  allow_unresolved_layup: bool) -> None:
    """Refuse (or warn about) levels whose element layers do not resolve the layup."""
    for nz in sorted(set(nz_values)):
        probe = CompositeMesh(field, material, nx=1, ny=1, nz=nz, ply_angles=ply_angles)
        issues = probe.layup_discrepancies()
        if not issues:
            continue
        msg = (f"nz = {nz} does not resolve the {material.n_plies}-ply layup: "
               f"{'. '.join(issues)}. A refinement study on it would measure "
               f"ply aliasing, not discretization error")
        if not allow_unresolved_layup:
            raise ValueError(
                f"{msg}. Use nz = k * n_plies (e.g. nz = {material.n_plies}), or "
                f"pass allow_unresolved_layup=True to study the element layup.")
        logger.warning("%s (allow_unresolved_layup=True).", msg)


# ----------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------

def refinement_study(
    material: MaterialProperties,
    porosity: PorosityField | float,
    loading: FELoadingMode = 'compression',
    *,
    porosity_config: Mapping[str, Any] | None = None,
    ply_angles: list[float] | str = 'QI',
    base_resolution: tuple[int, int, int] | None = None,
    factors: Sequence[float] | None = None,
    resolutions: Sequence[tuple[int, int, int]] | None = None,
    refine: RefineMode = 'in_plane',
    formulation: FEFormulation | None = None,
    failure_criterion: Literal['tsai_wu', 'hashin', 'max_stress'] = 'tsai_wu',
    applied_strain: float | None = None,
    applied_load: float = -10.0,
    delta_T: float | None = None,
    interior_margin: float | None = None,
    tol: float = 0.01,
    solver: Literal['direct', 'cg', 'minres'] = 'direct',
    solver_rtol: float = 1e-9,
    max_elements: int | None = None,
    max_memory_gb: float | None = None,
    allow_unresolved_layup: bool = False,
    verbose: bool = False,
) -> RefinementStudy:
    """Solve one FE load case on refined meshes and assess mesh convergence.

    Each level is a fresh :class:`~porosity_fe.CompositeMesh` and
    :class:`~porosity_fe.FESolver` on the same porosity field, solved
    coarse to fine; only the scalar results are kept, so the memory of a
    level is released before the next one is built. For every reported
    quantity (see :class:`RefinementLevel`) the study gives the relative
    change between successive levels and, from the three finest levels, the
    convergence behavior, the observed order, the Richardson extrapolate,
    Roache's GCI and a converged flag (:class:`QuantityConvergence`).

    The knockdown and effective modulus are integrals of the field and
    converge. The domain maximum failure index and the first-ply-failure
    load factor are extremes over every element: they sit at a constrained
    corner, a free edge or a load point, where the linear-elastic stress is
    singular, so they keep growing (or, for the load factor, shrinking) with
    refinement. The study reports them as ``'diverging'`` /
    ``'mesh-dependent'`` instead of extrapolating them, next to their
    interior counterparts, which leave out a fixed margin along the lateral
    faces and converge when no singularity lies inside that region (on the
    production geometry they do in compression; in ``'ilss'`` the region is
    the specimen's shear spans clear of the supports and the loading nose,
    and in flexure the span clear of the support and load lines).
    The theory page on the finite-element model ("Mesh convergence") has
    measured studies and a recommended resolution.

    Parameters
    ----------
    material : MaterialProperties
        Pristine material.
    porosity : PorosityField or float
        The porosity field, or the specimen-average void volume fraction
        ``Vp`` (a fraction in ``[0, 1]``) to build one from with
        ``porosity_config``. A field must have been built for ``material``.
    loading : {'compression', 'tension', 'shear', 'ilss', 'flexure_3pt', 'flexure_4pt', 'thermal'}
        Load case, as :meth:`FESolver.solve`.
    porosity_config : dict, optional
        :class:`~porosity_fe.PorosityField` keyword arguments
        (``distribution``, ``void_shape``, ...) when ``porosity`` is a
        number, as in :func:`~porosity_fe.build_empirical_pipeline`.
    ply_angles : list of float or {'QI', 'UD'}
        Layup, as :class:`~porosity_fe.CompositeMesh` (default ``'QI'``).
    base_resolution : (int, int, int), optional
        Coarsest ``(nx, ny, nz)``. Default: the production resolution,
        one element layer per ply, ``(30, 10, material.n_plies)``; unlike
        the production default, ``nz`` stays at ``n_plies`` above 48 plies
        (the memory guard below applies instead). For ``'ilss'`` the
        default is the :meth:`~porosity_fe.CompositeMesh.for_ilss` specimen
        mesh, ``(24, 8, material.n_plies)``, and for the flexure modes the
        :meth:`~porosity_fe.CompositeMesh.for_flexure` specimen mesh
        (``(48, 4, 24)`` for the 24-ply presets).
    factors : sequence of float, optional
        Refinement factors of the levels relative to ``base_resolution``,
        strictly increasing (default ``(1, 1.5, 2)``). Three or more levels
        are needed for the observed order and extrapolation.
    resolutions : sequence of (int, int, int), optional
        Explicit levels instead of ``base_resolution`` / ``factors`` /
        ``refine``; each must refine the previous one (no axis coarser,
        more elements).
    refine : {'in_plane', 'all'}
        ``'in_plane'`` (default) scales ``nx`` and ``ny`` and keeps ``nz``:
        bending error of the elements depends on the element length over
        the laminate thickness, not on ``nz`` (see the theory page on the
        finite-element model), and in-plane refinement is far cheaper.
        ``'all'`` also scales ``nz``, rounded to a whole multiple of
        ``material.n_plies`` so the layup stays resolved (the refinement is
        then not geometrically similar, so the observed order is
        approximate).
    formulation : {'hex8i', 'hex8'}, optional
        Element formulation; ``None`` (default) uses the solver default.
    failure_criterion : {'tsai_wu', 'hashin', 'max_stress'}
        Criterion of the failure quantities.
    applied_strain, applied_load, delta_T : float, optional
        Passed to :meth:`FESolver.solve`. ``delta_T`` (K) is required for
        ``loading='thermal'`` and adds the residual stress otherwise.
    interior_margin : float, optional
        Distance (mm) from the lateral faces that the interior quantities
        leave out, held fixed over the levels so they describe one region.
        Default: the solver's default on the coarsest level,
        ``max(h, 2 dx, 2 dy)`` with ``h`` the laminate thickness. (The
        solver's own default shrinks with the elements, which would let the
        interior region creep into the edge singularity.) For ``'ilss'``,
        the distance from the supports, the loading band and the side
        faces, and for the flexure modes from the support and load lines
        and the side faces; both default to ``h / 2`` (as
        :meth:`FESolver.solve`).
    tol : float
        Converged when the estimated relative error of the finest level is
        at most ``tol`` (default 0.01); see
        :attr:`QuantityConvergence.error_estimate`. Changes below
        ``tol / 10`` count as zero.
    solver : {'direct', 'cg', 'minres'}
        Linear solver of the levels, as :meth:`FESolver.solve`. The
        pristine reference of the knockdown is always factorized directly,
        so ``'cg'`` saves memory only for a thermal or pore-free study.
        ``'minres'`` is deprecated (one :class:`DeprecationWarning` per
        call; removal in 2.0): use ``'cg'``.
    solver_rtol : float
        Residual tolerance of the iterative solvers.
    max_elements : int, optional
        Largest element count a level may have; default the mesh cap
        (1,000,000).
    max_memory_gb : float, optional
        Memory budget (GB) for the estimated peak of a level; default 80 %
        of the physical memory where the platform reports it, otherwise no
        limit. ``float('inf')`` disables the check.
    allow_unresolved_layup : bool
        Accept levels whose ``nz`` does not resolve the layup
        (:meth:`CompositeMesh.layup_discrepancies`); they raise by default.
    verbose : bool
        Log the plan and per-level progress at INFO level (DEBUG otherwise).

    Returns
    -------
    RefinementStudy

    Raises
    ------
    ValueError
        Before any solve: for bad options, fewer than two levels, levels
        that do not refine, a level above ``max_elements`` or the mesh caps,
        an estimated peak memory above ``max_memory_gb`` (the message lists
        the estimate of every level), a ``nz`` that does not resolve the
        layup, or invalid thermal inputs.
    TypeError
        If ``porosity`` is neither a field nor a number.

    Notes
    -----
    For ``loading='ilss'`` every level is an ASTM D2344 specimen
    (:meth:`~porosity_fe.CompositeMesh.for_ilss`, default proportions and
    loading band) with the level's ``(nx, ny, nz)``. ``nx`` must be a
    multiple of 6 so the supports and mid-span stay on node lines: generated
    levels round it up, explicit ``resolutions`` that break it raise.

    For ``'flexure_3pt'`` and ``'flexure_4pt'`` every level is an ASTM D7264
    specimen (:meth:`~porosity_fe.CompositeMesh.for_flexure` with the
    level's ``nx``, ``ny`` and ``nz``, default proportions). ``nx`` must be
    even so mid-span stays on a node line: generated levels round it up,
    explicit ``resolutions`` that break it raise. The span is the same at
    every level; the unloaded overhangs take whole elements and can differ
    slightly in length.

    Examples
    --------
    >>> study = refinement_study(MATERIALS['T800_epoxy'], 0.03,
    ...                          'compression')            # doctest: +SKIP
    >>> print(study.summary())                              # doctest: +SKIP
    """
    log = logger.info if verbose else logger.debug
    if loading not in get_args(FELoadingMode):
        raise ValueError(f"Unknown loading {loading!r}; use one of "
                         f"{list(get_args(FELoadingMode))}.")
    formulation = _DEFAULT_FORMULATION if formulation is None else formulation
    if formulation not in ELEMENT_FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}; use one of "
                         f"{list(ELEMENT_FORMULATIONS)}.")
    if failure_criterion not in FESolver.SUPPORTED_FAILURE_CRITERIA:
        raise ValueError(f"Unknown failure_criterion {failure_criterion!r}; use one "
                         f"of {list(FESolver.SUPPORTED_FAILURE_CRITERIA)}.")
    if solver not in ('direct', 'cg', 'minres'):
        raise ValueError(f"Unknown solver {solver!r}; use 'direct', 'cg' or 'minres'.")
    if solver == 'minres':
        warnings.warn(_minres_deprecation('refinement_study'), DeprecationWarning,
                      stacklevel=2)
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.number)) \
            or not (math.isfinite(float(tol)) and float(tol) > 0.0):
        raise ValueError(f"tol must be a positive finite number, got {tol!r}.")
    tol = float(tol)
    if interior_margin is not None and (
            isinstance(interior_margin, bool)
            or not isinstance(interior_margin, (int, float, np.number))
            or not math.isfinite(float(interior_margin)) or interior_margin < 0):
        raise ValueError(f"interior_margin must be a finite length >= 0 in mm, "
                         f"got {interior_margin!r}.")
    _check_delta_T(loading, delta_T, material)
    field = _resolve_porosity(material, porosity, porosity_config)

    # --- plan the levels ------------------------------------------------
    if resolutions is not None:
        if factors is not None or base_resolution is not None:
            raise ValueError("Pass either resolutions= or base_resolution=/"
                             "factors=, not both.")
        plan = [_check_resolution(r, 'each of resolutions') for r in resolutions]
        refine_label = 'explicit'
        if loading == 'ilss' and any(r[0] % _ILSS_NX_MULTIPLE for r in plan):
            raise ValueError(
                f"refinement_study: 'ilss' levels are ASTM D2344 specimens "
                f"(6 h long, supports h in from the ends), so each nx must be a "
                f"multiple of {_ILSS_NX_MULTIPLE} to keep the supports and "
                f"mid-span on node lines; got {[r[0] for r in plan]}.")
        if loading in _FLEXURE_PROCEDURE and any(r[0] % 2 for r in plan):
            raise ValueError(
                f"refinement_study: {loading!r} levels are ASTM D7264 "
                f"specimens, so each nx must be even to keep mid-span on a "
                f"node line; got {[r[0] for r in plan]}.")
    else:
        if refine not in ('in_plane', 'all'):
            raise ValueError(f"refine must be 'in_plane' or 'all', got {refine!r}.")
        if base_resolution is None and loading == 'ilss':
            # The CompositeMesh.for_ilss default specimen mesh.
            base = (_ILSS_BASE_NX, _ILSS_BASE_NY, material.n_plies)
        elif base_resolution is None and loading in _FLEXURE_PROCEDURE:
            # The CompositeMesh.for_flexure default specimen mesh.
            specimen = CompositeMesh.for_flexure(field, material,
                                                 ply_angles=ply_angles)
            base = (specimen.nx, specimen.ny, specimen.nz)
            del specimen
        elif base_resolution is None:
            # The production resolution, but nz = n_plies even above its cap:
            # a study must resolve the layup, and the memory guard applies.
            nx0, ny0, _ = _default_mesh_resolution(material.n_plies)
            base = (nx0, ny0, material.n_plies)
        else:
            base = _check_resolution(base_resolution, 'base_resolution')
        fac = (1.0, 1.5, 2.0) if factors is None else tuple(float(f) for f in factors)
        if not all(math.isfinite(f) and f > 0.0 for f in fac) or \
                any(b <= a for a, b in zip(fac, fac[1:], strict=False)):
            raise ValueError(f"factors must be positive and strictly increasing, "
                             f"got {factors!r}.")
        plan = _plan_resolutions(base, fac, refine, material.n_plies)
        if loading == 'ilss':
            m = _ILSS_NX_MULTIPLE
            plan = [(m * math.ceil(nx / m), ny, nz) for nx, ny, nz in plan]
        elif loading in _FLEXURE_PROCEDURE:
            plan = [(nx + nx % 2, ny, nz) for nx, ny, nz in plan]
        refine_label = refine
    if len(plan) < 2:
        raise ValueError("A refinement study needs at least two levels.")
    for a, b in zip(plan, plan[1:], strict=False):
        if any(nb < na for na, nb in zip(a, b, strict=True)) or b == a:
            raise ValueError(
                f"Each level must refine the previous one (no axis coarser, not "
                f"identical); got {a} then {b}"
                + ("" if resolutions is not None else
                   f" from factors {factors if factors is not None else '(1, 1.5, 2)'}")
                + ".")
    axes = tuple(i for i in range(3) if len({r[i] for r in plan}) > 1)
    refined_axes = tuple('xyz'[i] for i in axes)

    # --- guards, before any solve ----------------------------------------
    cap = CompositeMesh._MAX_TOTAL_ELEMENTS if max_elements is None else int(max_elements)
    n_lu = _peak_factorizations(loading, solver, field)
    est = [_estimate_memory_bytes(r, n_factorizations=n_lu) for r in plan]
    table = "; ".join(f"{r[0]}x{r[1]}x{r[2]}: {r[0] * r[1] * r[2]:,} elements, "
                      f"~{e / 1e9:.2g} GB" for r, e in zip(plan, est, strict=True))
    for r in plan:
        n_el = r[0] * r[1] * r[2]
        if n_el > min(cap, CompositeMesh._MAX_TOTAL_ELEMENTS) or \
                max(r) > CompositeMesh._MAX_ELEMENTS_PER_AXIS:
            raise ValueError(
                f"Level {r[0]}x{r[1]}x{r[2]} has {n_el:,} elements, above the "
                f"limit of {min(cap, CompositeMesh._MAX_TOTAL_ELEMENTS):,} "
                f"(max_elements / mesh caps). Planned levels: {table}.")
    if max_memory_gb is None:
        phys = _physical_memory_bytes()
        budget = None if phys is None else _DEFAULT_MEMORY_FRACTION * phys
    else:
        budget = float(max_memory_gb) * 1e9
    if budget is not None and max(est) > budget:
        raise ValueError(
            f"The finest level needs an estimated {max(est) / 1e9:.2g} GB, above "
            f"the {budget / 1e9:.2g} GB budget (max_memory_gb). Planned levels: "
            f"{table}. Use fewer or smaller factors, refine='in_plane', or "
            f"raise max_memory_gb.")
    _check_layups(material, field, [r[2] for r in plan], ply_angles,
                  allow_unresolved_layup)
    log("refinement_study: %s, %s, %d levels (%s)", loading, formulation,
        len(plan), table)

    # --- solve the levels, coarse to fine ---------------------------------
    margin = interior_margin
    levels: list[RefinementLevel] = []
    for i, (res, est_bytes) in enumerate(zip(plan, est, strict=True)):
        t0 = time.perf_counter()
        if loading == 'ilss':
            mesh = CompositeMesh.for_ilss(field, material, nx=res[0], ny=res[1],
                                          nz=res[2], ply_angles=ply_angles)
        elif loading in _FLEXURE_PROCEDURE:
            mesh = CompositeMesh.for_flexure(field, material, nx=res[0], ny=res[1],
                                             nz=res[2], ply_angles=ply_angles)
        else:
            mesh = CompositeMesh(field, material, nx=res[0], ny=res[1], nz=res[2],
                                 ply_angles=ply_angles)
        dims = (mesh.L_x, mesh.L_y, mesh.L_z)
        if margin is None and (loading == 'ilss' or loading in _FLEXURE_PROCEDURE):
            margin = default_interior_margin(mesh)
        elif margin is None:
            margin = max(mesh.L_z, 2.0 * mesh.L_x / mesh.nx, 2.0 * mesh.L_y / mesh.ny)
        size = float(np.prod([dims[a] / res[a] for a in axes]) ** (1.0 / len(axes)))
        fe = FESolver(mesh, material, field, ply_angles=ply_angles,
                      failure_criterion=failure_criterion, formulation=formulation)
        quiet = warnings.catch_warnings() if solver == 'minres' \
            else contextlib.nullcontext()
        with quiet:
            if solver == 'minres':
                # Reported once above, at the caller, not once per level.
                warnings.filterwarnings('ignore', message=r".*solver='minres'",
                                        category=DeprecationWarning)
            out = fe.solve(loading, applied_strain=applied_strain,
                           applied_load=applied_load, solver=solver,
                           rtol=solver_rtol, delta_T=delta_T,
                           allow_unresolved_layup=allow_unresolved_layup,
                           interior_margin=margin)
        level = RefinementLevel(
            resolution=res, n_elements=mesh.n_elements, n_dof=mesh.n_dof,
            element_size=size, solve_time_s=time.perf_counter() - t0,
            estimated_memory_gb=est_bytes / 1e9,
            knockdown=_finite_or_none(out.knockdown),
            effective_modulus=_finite_or_none(out.effective_modulus),
            max_failure_index=_finite_or_none(out.max_failure_index),
            first_ply_failure_load_factor=_finite_or_none(out.first_ply_failure_load_factor),
            interior_max_failure_index=_finite_or_none(out.interior_max_failure_index),
            interior_first_ply_failure_load_factor=_finite_or_none(
                out.interior_first_ply_failure_load_factor),
        )
        del out, fe, mesh
        gc.collect()
        levels.append(level)
        log("refinement_study: level %d/%d %s done in %.1f s: KD=%s, max FI=%s, "
            "interior max FI=%s", i + 1, len(plan), level.mesh, level.solve_time_s,
            _fmt(level.knockdown), _fmt(level.max_failure_index),
            _fmt(level.interior_max_failure_index))

    sizes = [lv.element_size for lv in levels]
    convergence = {name: _assess(name, [lv.value(name) for lv in levels], sizes, tol)
                   for name in QUANTITY_KINDS}
    assert margin is not None
    return RefinementStudy(
        loading=loading, refine=refine_label, refined_axes=refined_axes,
        formulation=formulation, failure_criterion=failure_criterion, tol=tol,
        interior_margin=float(margin), void_volume_fraction=float(field.Vp),
        applied_strain=None if applied_strain is None else float(applied_strain),
        applied_load=float(applied_load),
        delta_T=None if delta_T is None else float(delta_T),
        levels=tuple(levels), convergence=convergence)
