"""Sampling vocabulary shared by the uncertainty helpers.

:func:`~porosity_fe.propagate_uncertainty` (the empirical knockdown) and
:func:`~porosity_fe.fracture_load_factor_uncertainty` (the delamination
load factor) describe scatter the same way and draw it with the same code:

- ``covs={field: cov}`` is a median-preserving truncated lognormal with
  that coefficient of variation; ``spec={field: (dist, params)}`` names
  the distribution, ``'lognormal'`` or ``'normal'`` (``params`` the CoV)
  or ``'uniform'`` (``params`` the fractional half-width);
- :func:`_normalize_spec` resolves the two into one ordered table, dropping
  a field whose scatter is zero, so a zero-CoV request is exactly the
  deterministic calculation;
- :func:`_draw_unit_samples` draws ``U(0, 1)`` variates, plain Monte Carlo
  or Latin hypercube, from one seeded generator;
- :func:`_unit_to_draw` maps them to the unit variate each distribution
  takes, and :func:`_perturbed_value` to the value about its nominal.

:meth:`MaterialProperties._perturbed_value
<porosity_fe.MaterialProperties._perturbed_value>` delegates to
:func:`_perturbed_value` and clips the result to the field's range.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

#: Sampling methods: ``numpy.random.default_rng`` draws, or
#: ``scipy.stats.qmc.LatinHypercube`` seeded from the same generator.
_UQ_METHODS = ('monte_carlo', 'lhs')
#: Distributions that consume a standard-normal unit draw vs. a U(0,1) draw.
_UQ_NORMAL_DISTS = ('lognormal', 'normal')
_UQ_UNIFORM_DISTS = ('uniform',)


def _normalize_spec(covs: dict[str, float] | None,
                    spec: dict[str, tuple[str, float]] | None, *,
                    check_field: Callable[[str], None],
                    check_set: Callable[[str], None]) -> OrderedDict:
    """Resolve the user-facing uncertainty description into a canonical
    ``OrderedDict{field: (dist, params)}``.

    ``covs`` is the convenience form: ``{field: cov}`` -> truncated-lognormal
    with that coefficient of variation. ``spec`` is the explicit form:
    ``{field: (dist, params)}``. They may both be given (``spec`` wins on a
    field collision). Fields with a non-positive CoV are dropped so a
    zero-CoV request is exactly the deterministic pipeline.

    ``check_field(name)`` raises for a field that cannot scatter and
    ``check_set(name)`` for one with no nominal value; both are called
    for every field named, the latter only when its scatter is positive.
    """
    resolved: OrderedDict = OrderedDict()

    for name, cov in (covs or {}).items():
        check_field(name)
        cov = float(cov)
        if not np.isfinite(cov) or cov < 0.0:
            raise ValueError(
                f"CoV for {name!r} must be a finite non-negative number, "
                f"got {cov!r}."
            )
        if cov > 0.0:
            check_set(name)
            resolved[name] = ('lognormal', cov)

    for name, ds in (spec or {}).items():
        check_field(name)
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
            check_set(name)
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


def _perturbed_value(nominal: float, unit_draw: Any, dist: str, params: Any, *,
                     name: str = 'the value') -> Any:
    """Map a unit draw (a standard-normal or U(0,1) variate, or an array of
    them) to the perturbed value about ``nominal``, unclipped.

    ``dist`` is one of:
      - ``'lognormal'``: multiplicative median-preserving lognormal, so a
        positive quantity stays positive. ``params`` is the coefficient of
        variation (CoV, std/mean of the underlying value). ``unit_draw`` is
        a standard-normal variate.
      - ``'normal'``: additive Gaussian. ``params`` is the CoV; the std is
        ``cov * |nominal|``. ``unit_draw`` is a standard-normal variate.
      - ``'uniform'``: ``params`` is the fractional half-width ``h``; the
        value is drawn uniformly on ``nominal * [1 - h, 1 + h]``.
        ``unit_draw`` is a U(0, 1) variate.

    A non-positive ``params`` returns ``nominal`` itself (the draw is not
    consumed). ``name`` only labels the error for an unknown ``dist``.
    """
    if dist == 'lognormal':
        cov = float(params)
        if cov <= 0.0:
            return nominal
        # Median-preserving lognormal with the requested CoV.
        sigma_ln = np.sqrt(np.log1p(cov * cov))
        return nominal * np.exp(sigma_ln * unit_draw)
    if dist == 'normal':
        cov = float(params)
        if cov <= 0.0:
            return nominal
        return nominal + cov * abs(nominal) * unit_draw
    if dist == 'uniform':
        h = float(params)
        if h <= 0.0:
            return nominal
        return nominal * (1.0 - h + 2.0 * h * unit_draw)
    raise ValueError(
        f"Unknown distribution {dist!r} for {name}. "
        f"Use one of 'lognormal', 'normal', 'uniform'."
    )
