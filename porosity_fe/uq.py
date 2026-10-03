"""Uncertainty propagation (Monte Carlo / LHS) over the empirical solver."""

from __future__ import annotations

import contextlib
import io
import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .empirical import (  # noqa: F401 (EmpiricalSolver used as forward-ref in _build_solver annotation)
    _LAYUP_WARNING_MARKER,
    Calibration,
    EmpiricalSolver,
)
from .io import _UNITS_UQ, FORMAT_UQ, _json_default, _wrap_envelope
from .materials import MATERIALS, MaterialProperties
from .pipeline import build_empirical_pipeline

# ============================================================
# SECTION 6b: UNCERTAINTY PROPAGATION (MONTE CARLO / LHS)
# ============================================================

# Back-compat alias for the default percentiles surface — canonical
# definition lives on ``Calibration.UQ_DEFAULT_PERCENTILES`` (#121).
_UQ_DEFAULT_PERCENTILES = Calibration.UQ_DEFAULT_PERCENTILES
_UQ_METHODS = ('monte_carlo', 'lhs')
# Distributions that consume a standard-normal unit draw vs. a U(0,1) draw.
_UQ_NORMAL_DISTS = ('lognormal', 'normal')
_UQ_UNIFORM_DISTS = ('uniform',)
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
    zero-CoV request is exactly the deterministic pipeline.
    """
    resolved: OrderedDict = OrderedDict()
    valid = set(MaterialProperties.PERTURBABLE_FIELDS)

    def _check_field(name: str) -> None:
        if name not in valid:
            raise ValueError(
                f"Unknown / non-perturbable material field {name!r}. "
                f"Use one of {sorted(valid)}."
            )

    for name, cov in (covs or {}).items():
        _check_field(name)
        cov = float(cov)
        if not np.isfinite(cov) or cov < 0.0:
            raise ValueError(
                f"CoV for {name!r} must be a finite non-negative number, "
                f"got {cov!r}."
            )
        if cov > 0.0:
            resolved[name] = ('lognormal', cov)

    for name, ds in (spec or {}).items():
        _check_field(name)
        if not (isinstance(ds, (tuple, list)) and len(ds) == 2):
            raise ValueError(
                f"spec[{name!r}] must be a (distribution, params) pair, "
                f"got {ds!r}."
            )
        dist, params = ds
        if dist not in _UQ_NORMAL_DISTS + _UQ_UNIFORM_DISTS:
            raise ValueError(
                f"spec[{name!r}] has unknown distribution {dist!r}. "
                f"Use one of {sorted(_UQ_NORMAL_DISTS + _UQ_UNIFORM_DISTS)}."
            )
        params = float(params)
        if not np.isfinite(params) or params < 0.0:
            raise ValueError(
                f"spec[{name!r}] params must be a finite non-negative "
                f"number, got {params!r}."
            )
        if params > 0.0:
            resolved[name] = (dist, params)
        else:
            resolved.pop(name, None)
    return resolved


def _draw_unit_samples(n_vars: int, n_samples: int, method: str,
                       rng: np.random.Generator) -> np.ndarray:
    """Return an ``(n_samples, n_vars)`` array of U(0, 1) variates.

    ``method='monte_carlo'`` uses ``rng.random``; ``method='lhs'`` uses
    ``scipy.stats.qmc.LatinHypercube`` seeded from the same ``rng`` so the
    whole helper is reproducible from a single seed.
    """
    if n_vars == 0:
        return np.empty((n_samples, 0))
    if method == 'monte_carlo':
        return rng.random((n_samples, n_vars))
    if method == 'lhs':
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_vars, seed=rng)
        return sampler.random(n=n_samples)
    raise ValueError(
        f"Unknown sampling method {method!r}. Use one of {sorted(_UQ_METHODS)}."
    )


def _unit_to_draw(u: np.ndarray, dist: str) -> np.ndarray:
    """Map U(0, 1) variates to the unit variate the target distribution
    expects: a standard normal for normal/lognormal, the U(0,1) untouched
    (clipped off the 0/1 endpoints) for uniform."""
    if dist in _UQ_NORMAL_DISTS:
        from scipy.stats import norm
        return norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    return u


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
    """
    payload = {}
    for label, res in results.items():
        entry = dict(res)
        if not include_samples:
            entry.pop('samples', None)
        payload[label] = entry
    envelope = _wrap_envelope(FORMAT_UQ, _UNITS_UQ, {'uq': payload})
    with open(Path(filename), 'w', encoding='utf-8') as f:
        json.dump(envelope, f, indent=2, default=_json_default)
