"""Uncertainty propagation (Monte Carlo / LHS).

:func:`propagate_uncertainty` samples the inputs of the empirical knockdown
and runs each draw through :class:`~porosity_fe.EmpiricalSolver`;
:func:`fracture_load_factor_uncertainty` samples the toughness inputs of the
delamination load factor and re-evaluates it on the front of one FE solve
(the analysis is linear, so the energy release rates do not change). Both
describe scatter with the vocabulary of :mod:`porosity_fe._sampling`.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._sampling import (
    _UQ_METHODS,
    _UQ_NORMAL_DISTS,  # noqa: F401 (re-exported sampling vocabulary)
    _UQ_UNIFORM_DISTS,  # noqa: F401
    _draw_unit_samples,
    _normalize_spec,
    _perturbed_value,
    _unit_to_draw,
)
from .empirical import (  # noqa: F401 (EmpiricalSolver used as forward-ref in _build_solver annotation)
    _LAYUP_WARNING_MARKER,
    Calibration,
    EmpiricalSolver,
)
from .io import _UNITS_UQ, FORMAT_UQ, _finite_or_none, _wrap_envelope, _write_json
from .materials import MATERIALS, MaterialProperties
from .pipeline import build_empirical_pipeline
from .toughness import (
    _TOUGHNESS_LAWS,
    ToughnessKnockdown,
    ToughnessKnockdownSpec,
    _placeholder_applied,
    _resolve_toughness_knockdown,
)

if TYPE_CHECKING:
    from .fe.vcct import CrackFront

# ============================================================
# SECTION 6b: UNCERTAINTY PROPAGATION (MONTE CARLO / LHS)
# ============================================================

# Back-compat alias for the default percentiles surface — canonical
# definition lives on ``Calibration.UQ_DEFAULT_PERCENTILES`` (#121).
_UQ_DEFAULT_PERCENTILES = Calibration.UQ_DEFAULT_PERCENTILES
# Built-in knockdown law -> (EmpiricalSolver override kwarg, QI coefficient table).
_UQ_COEF_TABLES = {
    'judd_wright': ('judd_wright_alpha', Calibration.JUDD_WRIGHT_ALPHA_QI),
    'power_law': ('power_law_n', Calibration.POWER_LAW_N_QI),
    'linear': ('linear_beta', Calibration.LINEAR_BETA_QI),
}


def _material_label(material: MaterialProperties) -> str:
    """Preset name of ``material`` if it equals one, else ``'custom'``."""
    for name, preset in MATERIALS.items():
        if preset == material:
            return name
    return 'custom'


def _normalize_uq_spec(material: MaterialProperties,
                       covs: dict[str, float] | None,
                       spec: dict[str, tuple[str, float]] | None
                       ) -> OrderedDict:
    """Resolve the user-facing uncertainty description into a canonical
    ``OrderedDict{field: (dist, params)}``.

    ``covs`` is the convenience form: ``{field: cov}`` -> truncated-lognormal
    with that coefficient of variation. ``spec`` is the explicit form:
    ``{field: (dist, params)}``. They may both be given (``spec`` wins on a
    field collision). Fields with a non-positive CoV are dropped so a
    zero-CoV request is exactly the deterministic pipeline
    (:func:`porosity_fe._sampling._normalize_spec` over
    ``MaterialProperties.PERTURBABLE_FIELDS``).
    """
    valid = set(MaterialProperties.PERTURBABLE_FIELDS)

    def _check_field(name: str) -> None:
        if name not in valid:
            raise ValueError(
                f"Unknown / non-perturbable material field {name!r}. "
                f"Use one of {sorted(valid)}."
            )

    def _check_set(name: str) -> None:
        if getattr(material, name) is None:
            # The optional toughness fields: a draw scales a set value.
            raise ValueError(
                f"Material field {name!r} is None, so it cannot be "
                f"perturbed; set a nominal value first (e.g. with "
                f"dataclasses.replace)."
            )

    return _normalize_spec(covs, spec, check_field=_check_field, check_set=_check_set)


def propagate_uncertainty(void_volume_fraction: float,
                          material: str | MaterialProperties = 'T800_epoxy',
                          mode: str = 'compression',
                          model: str = 'judd_wright',
                          *,
                          covs: dict[str, float] | None = None,
                          spec: dict[str, tuple[str, float]] | None = None,
                          vp_cov: float = 0.0,
                          coef_cov: float = 0.0,
                          n_samples: int = 1000,
                          method: str = 'monte_carlo',
                          seed: int | None = None,
                          percentiles: tuple[float, ...] = Calibration.UQ_DEFAULT_PERCENTILES,
                          ply_angles: list[float] | str | None = 'QI',
                          config: dict | None = None) -> dict:
    """Propagate input uncertainty through ``EmpiricalSolver.get_failure_load``.

    Perturbs uncertain ``MaterialProperties`` fields (and, optionally, the
    specimen-average porosity ``Vp``) and reports summary statistics of the
    knockdown and failure stress. The base deterministic pipeline is
    untouched; this is a strictly additive wrapper.

    Parameters
    ----------
    void_volume_fraction : float
        Nominal mean porosity fraction in [0, 1].
    material : str or MaterialProperties
        A ``MATERIALS`` preset name or an explicit dataclass instance.
    mode, model : str
        Forwarded to :meth:`EmpiricalSolver.get_failure_load`.
    covs : dict, optional
        Convenience uncertainty spec ``{field: cov}`` -> truncated-lognormal
        with that coefficient of variation (std/mean).
    spec : dict, optional
        Explicit uncertainty spec ``{field: (dist, params)}`` where ``dist``
        is ``'lognormal'`` / ``'normal'`` (params = CoV) or ``'uniform'``
        (params = fractional half-width). Wins over ``covs`` on a collision.
    vp_cov : float
        CoV of the mean porosity itself (truncated-lognormal, clipped to
        [0, 1]). 0.0 (default) holds Vp fixed at ``void_volume_fraction``.
    coef_cov : float
        CoV of the knockdown law's calibration coefficient for ``mode``
        (Judd-Wright ``alpha``, power-law ``n`` or linear ``beta``),
        median-preserving lognormal on the QI value; the layup scaling
        (``EmpiricalSolver.layup_scale``) is applied on top as usual. This is usually the dominant uncertainty:
        with material scatter alone the knockdown does not vary at all.
        0.0 (default) keeps the calibrated value.
    n_samples : int
        Number of draws.
    method : {'monte_carlo', 'lhs'}
        ``'monte_carlo'`` -> ``numpy.random.default_rng``;
        ``'lhs'`` -> ``scipy.stats.qmc.LatinHypercube`` (both seeded from
        ``seed`` so results are reproducible).
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Echoed into the result. With a
        fixed seed the summary is bit-for-bit reproducible.
    percentiles : tuple of float
        Percentiles to report (default 5/50/95).
    ply_angles : list of float, optional
        Forwarded to ``EmpiricalSolver`` (layup scaling).
    config : dict, optional
        Forwarded to ``PorosityField`` (distribution / void_shape / ...).

    Returns
    -------
    dict
        ``{'failure_stress': {'mean','std','min','max','percentiles': {...}},
        'knockdown': {... same ...}, 'nominal': {...},
        'samples': {'failure_stress': np.ndarray, 'knockdown': np.ndarray},
        'seed', 'n_samples', 'method', 'mode', 'model', 'spec', 'vp_cov'}``.
    """
    if isinstance(material, str):
        if material not in MATERIALS:
            raise ValueError(
                f"Unknown material {material!r}. "
                f"Available presets: {sorted(MATERIALS)}."
            )
        material_name = material
        mat = MATERIALS[material]
    else:
        material_name = _material_label(material)
        mat = material

    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(
            f"n_samples must be a positive integer, got {n_samples!r}."
        )
    if method not in _UQ_METHODS:
        raise ValueError(
            f"Unknown sampling method {method!r}. "
            f"Use one of {sorted(_UQ_METHODS)}."
        )
    vp_cov = float(vp_cov)
    if not np.isfinite(vp_cov) or vp_cov < 0.0:
        raise ValueError(
            f"vp_cov must be a finite non-negative number, got {vp_cov!r}."
        )
    coef_cov = float(coef_cov)
    if not np.isfinite(coef_cov) or coef_cov < 0.0:
        raise ValueError(
            f"coef_cov must be a finite non-negative number, got {coef_cov!r}."
        )
    sample_coef = coef_cov > 0.0
    if sample_coef and (model not in _UQ_COEF_TABLES
                        or mode not in _UQ_COEF_TABLES[model][1]):
        raise ValueError(
            f"coef_cov needs a built-in model ({sorted(_UQ_COEF_TABLES)}) "
            f"with a calibrated {mode!r} coefficient; got model={model!r}."
        )
    pcts = tuple(float(p) for p in percentiles)
    if any((not np.isfinite(p)) or p < 0.0 or p > 100.0 for p in pcts):
        raise ValueError(
            f"percentiles must lie in [0, 100], got {percentiles!r}."
        )

    resolved = _normalize_uq_spec(mat, covs, spec)
    field_names = list(resolved.keys())
    # Sampling dimensions: material fields, then the law coefficient (if
    # active), then porosity (if active) last.
    sample_vp = vp_cov > 0.0
    n_vars = len(field_names) + int(sample_coef) + int(sample_vp)

    config = config or {}

    def _build_solver(material_obj: MaterialProperties, vp_value: float,
                      coef: float | None = None) -> EmpiricalSolver:
        # CompositeMesh prints a banner on construction; the sampling loop
        # builds one mesh per draw, so silence it (additive: we do not touch
        # CompositeMesh itself).
        with contextlib.redirect_stdout(io.StringIO()):
            _pf, _mesh, emp = build_empirical_pipeline(
                material_obj,
                vp_value,
                ply_angles=ply_angles,
                mesh_res=(4, 3, 3),
                porosity_config=config,
                solver_kwargs=(None if coef is None
                               else {_UQ_COEF_TABLES[model][0]: {mode: coef}}),
            )
            return emp

    # Deterministic nominal (no perturbation): the mean must land near this.
    nominal = _build_solver(mat, float(void_volume_fraction)).get_failure_load(
        mode, model)

    rng = np.random.default_rng(seed)
    unit = _draw_unit_samples(n_vars, int(n_samples), method, rng)

    fs_samples = np.empty(int(n_samples), dtype=float)
    kd_samples = np.empty(int(n_samples), dtype=float)

    # Pre-compute per-field column index and the unit-variate mapping.
    col_for_field = {name: i for i, name in enumerate(field_names)}
    coef_col = len(field_names) if sample_coef else None
    vp_col = len(field_names) + int(sample_coef) if sample_vp else None
    nominal_vp = float(void_volume_fraction)
    if sample_vp:
        sigma_ln_vp = np.sqrt(np.log1p(vp_cov * vp_cov))
    if sample_coef:
        nominal_coef = float(_UQ_COEF_TABLES[model][1][mode])
        sigma_ln_coef = np.sqrt(np.log1p(coef_cov * coef_cov))

    n_extrapolated = 0
    for s in range(int(n_samples)):
        draws = {}
        for name in field_names:
            dist, _ = resolved[name]
            u = unit[s, col_for_field[name]]
            draws[name] = float(_unit_to_draw(np.array([u]), dist)[0])
        sampled_mat = mat.perturb(draws, resolved) if field_names else mat

        if sample_vp:
            z = float(_unit_to_draw(np.array([unit[s, vp_col]]),
                                    'lognormal')[0])
            vp_value = nominal_vp * np.exp(sigma_ln_vp * z)
            vp_value = float(np.clip(vp_value, 0.0, 1.0))
        else:
            vp_value = nominal_vp

        coef = None
        if sample_coef:
            z = float(_unit_to_draw(np.array([unit[s, coef_col]]),
                                    'lognormal')[0])
            coef = nominal_coef * float(np.exp(sigma_ln_coef * z))

        # Draws scattered past the calibration bound would each warn;
        # count them and warn once after the loop instead.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _build_solver(sampled_mat, vp_value, coef).get_failure_load(
                mode, model)
        for w in caught:
            if "calibration bound" in str(w.message):
                n_extrapolated += 1
            elif _LAYUP_WARNING_MARKER in str(w.message):
                # The nominal solve above already flagged the layup
                # amplification once; the draws share the layup.
                continue
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        fs_samples[s] = res['failure_stress']
        kd_samples[s] = res['knockdown']

    if n_extrapolated:
        warnings.warn(
            f"{n_extrapolated} of {int(n_samples)} uncertainty draws evaluated "
            f"the empirical knockdown beyond its calibration bound "
            f"(Vp <= 0.05); those results are extrapolated.",
            UserWarning, stacklevel=2)

    def _summary(arr: np.ndarray) -> dict:
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'percentiles': {
                f'p{p:g}': float(np.percentile(arr, p)) for p in pcts
            },
        }

    return {
        'failure_stress': _summary(fs_samples),
        'knockdown': _summary(kd_samples),
        'nominal': {
            'failure_stress': float(nominal['failure_stress']),
            'knockdown': float(nominal['knockdown']),
        },
        'samples': {
            'failure_stress': fs_samples,
            'knockdown': kd_samples,
        },
        'seed': seed,
        'n_samples': int(n_samples),
        'method': method,
        'mode': mode,
        'model': model,
        'material': material_name,
        'void_volume_fraction': float(void_volume_fraction),
        'vp_cov': vp_cov,
        'coef_cov': coef_cov,
        'percentiles': list(pcts),
        'spec': {k: list(v) for k, v in resolved.items()},
    }


def save_uq_results_to_json(results: dict[str, dict], filename: str | os.PathLike,
                            include_samples: bool = False) -> None:
    """Write :func:`propagate_uncertainty` outputs as a ``porosity-fe.uq`` JSON.

    Parameters
    ----------
    results : dict
        ``{label: propagate_uncertainty(...)}``, e.g. keyed by loading mode.
    filename : str or os.PathLike
        Output path.
    include_samples : bool, optional
        Keep the raw per-draw ``samples`` arrays (dropped by default; they
        are ``n_samples`` floats per quantity).

    Notes
    -----
    The file is standard JSON (no ``NaN`` / ``Infinity``). Every value of
    a :func:`propagate_uncertainty` result is finite, so a non-finite one
    raises :class:`ValueError` (naming its location) before the file is
    written.
    """
    payload = {}
    for label, res in results.items():
        entry = dict(res)
        if not include_samples:
            entry.pop('samples', None)
        payload[label] = entry
    envelope = _wrap_envelope(FORMAT_UQ, _UNITS_UQ, {'uq': payload})
    # Standard JSON: every value is finite, so a non-finite one raises
    # instead of writing NaN.
    _write_json(envelope, Path(filename))


# ============================================================
# SECTION 6c: UNCERTAINTY BAND ON THE DELAMINATION LOAD FACTOR
# ============================================================

#: Inputs of the fracture load factor band that can scatter: the pristine
#: toughness and the Benzeggagh-Kenane exponent of the material, and the
#: porosity-toughness coefficients of the law. Nothing scatters unless
#: named, and nothing else can: the energy release rates come from one FE
#: solve.
_FRACTURE_UQ_MATERIAL_FIELDS = ('G_Ic', 'G_IIc', 'G_IIIc', 'bk_eta')
_FRACTURE_UQ_LAW_FIELDS = ('b_I', 'b_II', 'b_III')
_FRACTURE_UQ_FIELDS = _FRACTURE_UQ_MATERIAL_FIELDS + _FRACTURE_UQ_LAW_FIELDS


@dataclass(frozen=True)
class FractureLoadFactorBand:
    """Uncertainty band on the delamination load factor ``lambda_c`` from
    toughness scatter; :func:`fracture_load_factor_uncertainty` builds it.

    Every draw re-evaluates the critical load factor of every front node
    on the energy release rates of one FE solve, so the band covers the
    toughness inputs only (see :func:`fracture_load_factor_uncertainty`).

    Attributes
    ----------
    nominal : float
        The deterministic load factor on the nominal inputs: the smallest
        ``lambda_c`` over the front, as :attr:`FieldResults.fracture_load_factor
        <porosity_fe.FieldResults.fracture_load_factor>` (``inf`` when no
        node is driven).
    nominal_per_node : np.ndarray
        ``(n_nodes,)`` deterministic ``lambda_c`` of each front node.
    samples : np.ndarray
        ``(n_samples,)`` load factor of each draw: the smallest
        ``lambda_c`` over the front on that draw's inputs.
    samples_per_node : np.ndarray
        ``(n_samples, n_nodes)`` ``lambda_c`` of every front node on every
        draw.
    governing_fraction : np.ndarray
        ``(n_nodes,)`` share of the draws in which each node governs (has
        the smallest finite ``lambda_c``); the shares sum to 1 unless some
        draw drives no node.
    node : np.ndarray
        Original node id of each front node (``CrackFront.node``).
    mean, std, min, max : float
        Statistics of :attr:`samples` (``inf`` / ``nan`` when a draw is
        infinite).
    percentiles : dict
        ``{'p5': ..., 'p50': ..., 'p95': ...}`` of :attr:`samples` for the
        requested percentiles (linear interpolation; ``inf`` in an
        infinite tail).
    n_samples, method, seed
        The sampling: draws, ``'monte_carlo'`` or ``'lhs'``, and the seed
        (``None`` when unseeded).
    spec : dict
        The resolved scatter, ``{field: (distribution, params)}``; empty
        when nothing scatters.
    inputs : dict
        The nominal values the draws are centred on: ``G_Ic``, ``G_IIc``,
        ``G_IIIc`` (``G_IIc`` when unset), ``bk_eta``, and ``b_I``,
        ``b_II``, ``b_III`` of the law (``None`` for a law without
        coefficients).
    toughness_knockdown : ToughnessKnockdown
        The porosity-toughness law.
    toughness_uncalibrated : bool
        Whether a placeholder coefficient of the law lowered the nominal
        toughness at some front node (the draws are centred on it).
    load_factor_basis : str
        What the load factor multiplies (:attr:`FieldResults.load_factor_basis
        <porosity_fe.FieldResults.load_factor_basis>`).
    required_load_factor : float or None
        The required load factor passed in, for
        :attr:`probability_below_required`; ``None`` when none was.

    Notes
    -----
    :meth:`probability_below` is the share of the draws below a load
    factor, an estimate under the input distributions that were entered.
    It is **not a disposition**: the tool recommends no disposition for a
    delamination, and the band leaves out the porosity, stiffness,
    geometry, load, mesh and model-form uncertainty, and any correlation
    between the inputs (the draws are independent).
    """

    nominal: float
    nominal_per_node: np.ndarray
    samples: np.ndarray
    samples_per_node: np.ndarray
    governing_fraction: np.ndarray
    node: np.ndarray
    mean: float
    std: float
    min: float
    max: float
    percentiles: dict[str, float]
    n_samples: int
    method: str
    seed: int | None
    spec: dict[str, tuple[str, float]]
    inputs: dict[str, float | None]
    toughness_knockdown: ToughnessKnockdown
    toughness_uncalibrated: bool
    load_factor_basis: str
    required_load_factor: float | None = None

    @property
    def n_front_nodes(self) -> int:
        """Number of front nodes."""
        return int(self.node.size)

    def probability_below(self, load_factor: float) -> float:
        """Share of the draws whose load factor is below ``load_factor``.

        An estimate under the entered input scatter, not a disposition (see
        the class notes). ``load_factor`` is a multiple of the load named
        by :attr:`load_factor_basis`, like ``nominal``.

        Raises
        ------
        ValueError
            If ``load_factor`` is not a finite positive number.
        """
        r = float(load_factor)
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError(
                f"load_factor must be a finite positive multiple of the "
                f"analysis load, got {load_factor!r}.")
        return float(np.mean(self.samples < r))

    @property
    def probability_below_required(self) -> float | None:
        """:meth:`probability_below` at :attr:`required_load_factor`;
        ``None`` without one."""
        if self.required_load_factor is None:
            return None
        return self.probability_below(self.required_load_factor)

    def summary(self) -> dict:
        """The band without the draws, as a strict-JSON dict.

        Keys: ``load_factor_basis``, ``nominal``, ``n_samples``,
        ``method``, ``seed``, ``spec`` (``{field: [distribution,
        params]}``), ``inputs``, ``toughness_knockdown`` (a description),
        ``toughness_uncalibrated``, ``mean``, ``std``, ``min``, ``max``,
        ``percentiles``, ``n_front_nodes``, ``node``, ``nominal_per_node``,
        ``governing_fraction``, ``required_load_factor`` and
        ``probability_below_required``. Every non-finite number is
        ``None`` (``null``): an infinite load factor, or the statistics of
        a sample with one. :meth:`FESolver.export_results(...,
        fracture_uncertainty=band) <porosity_fe.FESolver.export_results>`
        writes it as the ``fracture_uncertainty`` block.
        """
        return {
            'load_factor_basis': str(self.load_factor_basis),
            'nominal': _finite_or_none(self.nominal),
            'n_samples': int(self.n_samples),
            'method': str(self.method),
            'seed': None if self.seed is None else int(self.seed),
            'spec': {k: [str(v[0]), float(v[1])] for k, v in self.spec.items()},
            'inputs': {k: (None if v is None else float(v))
                       for k, v in self.inputs.items()},
            'toughness_knockdown': self.toughness_knockdown.describe(),
            'toughness_uncalibrated': bool(self.toughness_uncalibrated),
            'mean': _finite_or_none(self.mean),
            'std': _finite_or_none(self.std),
            'min': _finite_or_none(self.min),
            'max': _finite_or_none(self.max),
            'percentiles': {k: _finite_or_none(v) for k, v in self.percentiles.items()},
            'n_front_nodes': self.n_front_nodes,
            'node': [int(v) for v in self.node],
            'nominal_per_node': [_finite_or_none(float(v)) for v in self.nominal_per_node],
            'governing_fraction': [float(v) for v in self.governing_fraction],
            'required_load_factor': (None if self.required_load_factor is None
                                     else float(self.required_load_factor)),
            'probability_below_required': self.probability_below_required,
        }


def _normalize_fracture_spec(material: MaterialProperties, law: ToughnessKnockdown,
                             covs: dict[str, float] | None,
                             spec: dict[str, tuple[str, float]] | None) -> OrderedDict:
    """:func:`porosity_fe._sampling._normalize_spec` over the toughness
    inputs of the fracture load factor (:data:`_FRACTURE_UQ_FIELDS`)."""

    def _check_field(name: str) -> None:
        if name not in _FRACTURE_UQ_FIELDS:
            raise ValueError(
                f"Unknown field {name!r} for the fracture load factor band: "
                f"only the toughness inputs {list(_FRACTURE_UQ_FIELDS)} can "
                f"scatter. The energy release rates of the front come from "
                f"one FE solve, so the stiffness, strength, porosity, "
                f"geometry and load are fixed; scatter in them changes G and "
                f"needs a solve per draw (propagate_uncertainty samples the "
                f"empirical path).")

    def _check_set(name: str) -> None:
        if name in _FRACTURE_UQ_MATERIAL_FIELDS:
            if getattr(material, name) is None:
                raise ValueError(
                    f"Material field {name!r} is None, so it cannot scatter; "
                    f"set a nominal value first (e.g. with dataclasses.replace). "
                    f"An unset G_IIIc follows G_IIc, draws included.")
        elif not law._has_coefficients:
            raise ValueError(
                f"{name} cannot scatter: the toughness law ({law.describe()}) "
                f"has no coefficient. Use the 'exponential' or 'linear' law.")
        elif name == 'b_III' and law.b_III is None:
            raise ValueError(
                "b_III is not set on the toughness law, so it cannot scatter; "
                "it follows b_II, draws included. Pass "
                "ToughnessKnockdown(b_III=...) to scatter it on its own.")

    return _normalize_spec(covs, spec, check_field=_check_field, check_set=_check_set)


def _toughness_draws(material: MaterialProperties, law: ToughnessKnockdown,
                     Vp: np.ndarray, values: dict[str, np.ndarray]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-mode toughness ``(G_Ic, G_IIc, G_IIIc)`` at every front node for
    every draw, each ``(n_samples, n_nodes)`` or ``(n_nodes,)`` when
    nothing in that mode scatters.

    ``values`` holds the ``(n_samples,)`` draws of the scattered fields;
    every other input keeps its nominal value, through the same
    expressions as :func:`~porosity_fe.fracture_load_factors` (the law's
    ``factor`` at the front porosity times the material value), so an
    unscattered mode is bit for bit the deterministic one. An unset
    ``G_IIIc`` follows ``G_IIc`` and an unset ``b_III`` follows ``b_II``,
    draws included.
    """
    assert material.G_Ic is not None and material.G_IIc is not None
    G: dict[str, np.ndarray | float] = {
        'I': values.get('G_Ic', float(material.G_Ic)),
        'II': values.get('G_IIc', float(material.G_IIc)),
    }
    if 'G_IIIc' in values:
        G['III'] = values['G_IIIc']
    elif material.G_IIIc is not None:
        G['III'] = float(material.G_IIIc)
    else:
        G['III'] = G['II']
    b: dict[str, np.ndarray | None] = {
        'I': values.get('b_I'), 'II': values.get('b_II')}
    if 'b_III' in values:
        b['III'] = values['b_III']
    else:
        b['III'] = None if law.b_III is not None else b['II']
    out = []
    for mode in ('I', 'II', 'III'):
        draws = b[mode]
        if draws is None:
            f = law.factor(Vp, mode)                                  # (n,)
        else:
            assert isinstance(law.law, str)
            f = _TOUGHNESS_LAWS[law.law](Vp[None, :], draws[:, None])  # (m, n)
        g = G[mode]
        out.append((g[:, None] if isinstance(g, np.ndarray) else g) * f)
    return out[0], out[1], out[2]


def _percentile(samples: np.ndarray, p: float) -> float:
    """``numpy.percentile`` (linear) that stays defined with infinite
    samples: ``inf`` when the interpolation reaches the infinite tail."""
    if np.isfinite(samples).all():
        return float(np.percentile(samples, p))
    s = np.sort(samples)
    pos = p / 100.0 * (s.size - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    a, b = float(s[lo]), float(s[hi])
    frac = pos - lo
    if frac == 0.0 or a == b:
        return a
    if not math.isfinite(b):
        return b
    return a + (b - a) * frac


def fracture_load_factor_uncertainty(
        front: CrackFront, material: MaterialProperties, *,
        prestress: CrackFront | None = None,
        toughness_knockdown: ToughnessKnockdownSpec = 'exponential',
        covs: dict[str, float] | None = None,
        spec: dict[str, tuple[str, float]] | None = None,
        n_samples: int = 1000,
        method: str = 'monte_carlo',
        seed: int | None = None,
        percentiles: tuple[float, ...] = Calibration.UQ_DEFAULT_PERCENTILES,
        required_load_factor: float | None = None,
        load_factor_basis: str | None = None,
) -> FractureLoadFactorBand:
    """Uncertainty band on the delamination load factor from toughness scatter.

    The fracture load factor of :func:`~porosity_fe.fracture_load_factors`
    is a point value, though its inputs scatter: the pristine toughness
    ``G_Ic``, ``G_IIc``, ``G_IIIc`` and the Benzeggagh-Kenane exponent
    ``bk_eta`` of the material, and the porosity-toughness coefficients
    ``b_I``, ``b_II``, ``b_III`` of the law (uncalibrated placeholders by
    default). The analysis is linear, so the energy release rates of the
    front from **one** FE solve do not depend on the toughness: every draw
    re-evaluates ``lambda_c = sqrt(G_c / G_T)`` at every front node without
    solving again. With a prestress (``u = u_P + lam u``) each draw takes
    the root of ``G_T(lam) = G_c(psi(lam))`` per node, vectorised over the
    draws, which agrees with :func:`~porosity_fe.fracture_load_factors`
    on the same inputs to about 1e-12; the contact set of the applied state
    is held, as the deterministic factor already assumes.

    **Nothing scatters unless named**: there is no default coefficient of
    variation. Scatter in anything else (stiffness, strength, porosity,
    geometry, load) changes ``G`` and would need a solve per draw, so
    naming such a field raises. The draws are independent between fields
    (no correlation between ``G_Ic`` and ``G_IIc``, for instance). The
    sampling vocabulary is that of :func:`propagate_uncertainty`.

    Parameters
    ----------
    front : CrackFront
        The front the factor scales (the mechanical load), with its
        porosity ``Vp``; usually :attr:`FieldResults.crack_front
        <porosity_fe.FieldResults.crack_front>`, or
        :meth:`FieldResults.energy_release_rates
        <porosity_fe.FieldResults.energy_release_rates>`. A front of a
        combined solve carries its two parts (``CrackFront.parts``), which
        are used when ``prestress`` is ``None``.
        :meth:`FieldResults.fracture_load_factor_uncertainty
        <porosity_fe.FieldResults.fracture_load_factor_uncertainty>` wraps
        this with the solve's law, prestress and basis.
    material : MaterialProperties
        The nominal toughness: ``G_Ic``, ``G_IIc``, ``bk_eta`` and the
        optional ``G_IIIc``.
    prestress : CrackFront, optional
        The fixed part (the thermal residual state) on the same front
        nodes, as for :func:`~porosity_fe.fracture_load_factors`.
    toughness_knockdown : ToughnessKnockdown, str, callable or None
        The porosity law of the toughness (:class:`~porosity_fe.ToughnessKnockdown`);
        its coefficients are the nominal ``b``.
    covs : dict, optional
        ``{field: cov}``: a median-preserving lognormal with that
        coefficient of variation (std / mean) on the field, one of
        ``'G_Ic'``, ``'G_IIc'``, ``'G_IIIc'``, ``'bk_eta'``, ``'b_I'``,
        ``'b_II'``, ``'b_III'``. A zero CoV leaves the field fixed.
    spec : dict, optional
        ``{field: (dist, params)}`` with ``dist`` ``'lognormal'`` or
        ``'normal'`` (``params`` the CoV) or ``'uniform'`` (``params`` the
        fractional half-width); wins over ``covs`` on a collision. A draw
        of a material field is clipped as :meth:`MaterialProperties.perturb
        <porosity_fe.MaterialProperties.perturb>` clips it (a positive
        floor); a coefficient ``b`` is clipped at 0 (the law never raises
        the toughness).
    n_samples : int
        Number of draws.
    method : {'monte_carlo', 'lhs'}
        Plain Monte Carlo or Latin hypercube sampling.
    seed : int, optional
        Seed of ``numpy.random.default_rng``; the band is reproducible from
        it.
    percentiles : tuple of float
        Percentiles to report (default 5 / 50 / 95).
    required_load_factor : float, optional
        A required load factor (a multiple of the analysis load, like
        ``lambda_c``): the band then reports the share of the draws below
        it (:attr:`FractureLoadFactorBand.probability_below_required`), an
        estimate under the entered scatter, not a disposition.
    load_factor_basis : str, optional
        What the load factor multiplies, recorded on the band; default
        ``'mechanical_with_residual'`` with a prestress, ``'mechanical'``
        without.

    Returns
    -------
    FractureLoadFactorBand
        The nominal value, the draws per node and over the front, the
        statistics and percentiles, which node governs how often, the
        inputs used, and :meth:`~FractureLoadFactorBand.summary` for the
        JSON export.

    Raises
    ------
    ValueError
        If the material has no toughness data, a named field cannot
        scatter (not a toughness input, unset, or a coefficient of a law
        without coefficients), ``prestress`` is on other nodes, or a
        sampling argument is invalid.

    Notes
    -----
    Not covered: the porosity field, the stiffness and its knockdown, the
    geometry, the load, the mesh (the front is a staircase on the grid)
    and the model form (linear VCCT with a growth-onset criterion; the
    conservative law ignores the rise of mode I propagation toughness with
    voids). No measured scatter is bundled: the theory page
    "Delamination" lists the few published coefficients of variation as
    starting points. With no scattered field every draw is exactly the
    deterministic value.

    Examples
    --------
    >>> band = results.fracture_load_factor_uncertainty(
    ...     material, covs={'G_Ic': 0.05, 'G_IIc': 0.07}, n_samples=2000,
    ...     method='lhs', seed=0)                                   # doctest: +SKIP
    >>> band.nominal, band.percentiles                             # doctest: +SKIP
    (2.747, {'p5': 2.63, 'p50': 2.75, 'p95': 2.86})
    """
    from .fe import vcct

    law = _resolve_toughness_knockdown(toughness_knockdown)
    eta = vcct._check_toughness(material, "The fracture load factor band")
    moving: CrackFront = front
    fixed: CrackFront | None = prestress
    if prestress is None and front.parts is not None:
        moving, fixed = front.parts
    if fixed is not None:
        vcct._check_same_front(fixed, moving)
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) \
            or n_samples <= 0:
        raise ValueError(
            f"n_samples must be a positive integer, got {n_samples!r}.")
    if method not in _UQ_METHODS:
        raise ValueError(
            f"Unknown sampling method {method!r}. Use one of {sorted(_UQ_METHODS)}.")
    pcts = tuple(float(p) for p in percentiles)
    if any((not np.isfinite(p)) or p < 0.0 or p > 100.0 for p in pcts):
        raise ValueError(
            f"percentiles must lie in [0, 100], got {percentiles!r}.")
    if required_load_factor is not None:
        r = float(required_load_factor)
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError(
                f"required_load_factor must be a finite positive multiple of "
                f"the analysis load, got {required_load_factor!r}.")
        required_load_factor = r
    resolved = _normalize_fracture_spec(material, law, covs, spec)
    basis = (load_factor_basis if load_factor_basis is not None
             else 'mechanical_with_residual' if fixed is not None else 'mechanical')

    # The nominal value is the deterministic function itself.
    nominal_front = vcct.fracture_load_factors(
        moving, material, prestress=fixed, toughness_knockdown=law)
    assert nominal_front.lambda_c is not None
    nominal_per_node = np.array(nominal_front.lambda_c, dtype=float)
    Vp = np.asarray(moving.Vp, dtype=float)
    n_nodes = len(moving)
    inputs = {
        'G_Ic': material.G_Ic, 'G_IIc': material.G_IIc,
        'G_IIIc': material.G_IIIc_eff, 'bk_eta': material.bk_eta,
        'b_I': law.coefficient('I'), 'b_II': law.coefficient('II'),
        'b_III': law.coefficient('III'),
    }

    rng = np.random.default_rng(seed)
    names = list(resolved)
    unit = _draw_unit_samples(len(names), int(n_samples), method, rng)
    if not names:
        # Nothing scatters: every draw is the deterministic value, exactly.
        per_node = np.tile(nominal_per_node, (int(n_samples), 1))
    else:
        values: dict[str, np.ndarray] = {}
        for j, name in enumerate(names):
            dist, params = resolved[name]
            z = _unit_to_draw(unit[:, j], dist)
            if name in _FRACTURE_UQ_MATERIAL_FIELDS:
                nominal = float(getattr(material, name))
                drawn = _perturbed_value(nominal, z, dist, params, name=f"field {name!r}")
                # The floor of MaterialProperties._clip_perturbed.
                values[name] = np.maximum(np.asarray(drawn, dtype=float),
                                          1e-6 * abs(nominal))
            else:
                b_nominal = law.coefficient(name[2:])  # type: ignore[arg-type]
                assert b_nominal is not None
                drawn = _perturbed_value(b_nominal, z, dist, params, name=name)
                values[name] = np.maximum(np.asarray(drawn, dtype=float), 0.0)
        toughness = _toughness_draws(material, law, Vp, values)
        eta_draws: float | np.ndarray = (values['bk_eta'][:, None]
                                         if 'bk_eta' in values else eta)
        if fixed is None:
            # Proportional loading: the same expressions as
            # fracture_load_factors, over (n_samples, n_nodes).
            G_T = np.asarray(moving.G_T, dtype=float)
            with np.errstate(invalid='ignore', divide='ignore'):
                G_c = vcct._bk_toughness(toughness, eta_draws, moving.G_I,
                                         moving.G_II, moving.G_III)
                per_node = np.where(G_T > 0.0,
                                    np.sqrt(G_c / np.where(G_T > 0.0, G_T, 1.0)),
                                    np.inf)
        else:
            per_node = vcct._thermal_factors_batch(fixed, moving, toughness, eta_draws)
        per_node = np.broadcast_to(per_node, (int(n_samples), n_nodes)).copy()

    samples = (np.min(per_node, axis=1) if n_nodes else
               np.full(int(n_samples), np.inf))
    finite = np.isfinite(per_node)
    governing = np.zeros(n_nodes, dtype=float)
    if n_nodes:
        has = finite.any(axis=1)
        rows = np.argmin(np.where(finite, per_node, np.inf), axis=1)
        governing = np.bincount(rows[has], minlength=n_nodes) / float(n_samples)
    with np.errstate(invalid='ignore'):
        stats = {
            'mean': float(np.mean(samples)), 'std': float(np.std(samples)),
            'min': float(np.min(samples)), 'max': float(np.max(samples)),
        }
    return FractureLoadFactorBand(
        nominal=float(np.min(nominal_per_node)) if n_nodes else math.inf,
        nominal_per_node=nominal_per_node,
        samples=samples,
        samples_per_node=per_node,
        governing_fraction=governing,
        node=np.array(moving.node, dtype=int),
        percentiles={f'p{p:g}': _percentile(samples, p) for p in pcts},
        n_samples=int(n_samples),
        method=str(method),
        seed=seed,
        spec={k: (str(v[0]), float(v[1])) for k, v in resolved.items()},
        inputs=inputs,
        toughness_knockdown=law,
        toughness_uncalibrated=_placeholder_applied(law, Vp),
        load_factor_basis=str(basis),
        required_load_factor=required_load_factor,
        **stats,
    )
