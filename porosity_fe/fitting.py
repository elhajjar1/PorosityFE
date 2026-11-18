"""Fit the coefficient of an empirical knockdown law to coupon data.

:func:`fit_knockdown` regresses one of the three knockdown laws of
:class:`~porosity_fe.EmpiricalSolver` on measured strengths against void
content and returns the coefficient with its confidence interval, ready to
pass as an override (``EmpiricalSolver(judd_wright_alpha=...)``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._vp import check_fraction
from .empirical import _KNOCKDOWN_LAWS, EmpiricalSolver

#: Model name -> (coefficient symbol, EmpiricalSolver override argument).
_COEFFICIENT_NAMES: dict[str, tuple[str, str]] = {
    'judd_wright': ('alpha', 'judd_wright_alpha'),
    'power_law': ('n', 'power_law_n'),
    'linear': ('beta', 'linear_beta'),
}


def _ols(x: np.ndarray, y: np.ndarray, intercept: bool
         ) -> tuple[float, float, np.ndarray, int]:
    """Least-squares line ``y = a + b x`` (``a = 0`` without ``intercept``).

    Returns ``(b, a, cov, dof)`` with ``cov`` the 2 x 2 covariance of
    ``(b, a)`` from the residual variance and ``dof`` the residual degrees
    of freedom.
    """
    n = x.size
    if intercept:
        x_mean, y_mean = float(x.mean()), float(y.mean())
        sxx = float(((x - x_mean) ** 2).sum())
        b = float(((x - x_mean) * (y - y_mean)).sum()) / sxx
        a = y_mean - b * x_mean
        dof = n - 2
        s2 = float(((y - a - b * x) ** 2).sum()) / dof
        cov = s2 * np.array([[1.0 / sxx, -x_mean / sxx],
                             [-x_mean / sxx, 1.0 / n + x_mean ** 2 / sxx]])
        return b, a, cov, dof
    sxx = float((x ** 2).sum())
    b = float((x * y).sum()) / sxx
    dof = n - 1
    s2 = float(((y - b * x) ** 2).sum()) / dof
    return b, 0.0, np.array([[s2 / sxx, 0.0], [0.0, 0.0]]), dof


# Each form maps the data to the line it regresses and the line back to
# (coefficient, pristine strength) with their variances. ``sigma_0`` is the
# given pristine strength, or None when it is fitted as the intercept.

def _fit_log(x: np.ndarray, strength: np.ndarray, sigma_0: float | None,
             sign: float) -> tuple[float, float, float, float | None, int]:
    """``ln(strength) = ln(sigma_0) + sign * c * x``: the Judd-Wright law
    (``x = Vp``, ``sign = -1``) and the power law (``x = ln(1 - Vp)``,
    ``sign = +1``). Returns ``(c, var_c, sigma_0, var_ln_sigma_0, dof)``."""
    if sigma_0 is None:
        b, a, cov, dof = _ols(x, np.log(strength), intercept=True)
        return sign * b, float(cov[0, 0]), math.exp(a), float(cov[1, 1]), dof
    b, _, cov, dof = _ols(x, np.log(strength / sigma_0), intercept=False)
    return sign * b, float(cov[0, 0]), sigma_0, None, dof


def _fit_judd_wright(Vp, strength, sigma_0):
    return _fit_log(Vp, strength, sigma_0, -1.0)


def _fit_power_law(Vp, strength, sigma_0):
    if Vp.max() >= 1.0:
        raise ValueError("The power law needs every Vp below 1.")
    return _fit_log(np.log1p(-Vp), strength, sigma_0, 1.0)


def _fit_linear(Vp, strength, sigma_0):
    """``strength = sigma_0 (1 - beta Vp)``. With ``sigma_0`` fitted the
    line is ``a + b Vp`` and ``beta = -b / a``, whose variance is the
    first-order (delta-method) one; the variance returned for the pristine
    strength is that of ``ln(a)``."""
    if sigma_0 is None:
        b, a, cov, dof = _ols(Vp, strength, intercept=True)
        if a <= 0.0:
            raise ValueError(
                "The fitted line has no positive strength at zero porosity; "
                "the linear law cannot be fitted to these data.")
        beta = -b / a
        grad = np.array([-1.0 / a, b / a ** 2])
        return beta, float(grad @ cov @ grad), a, float(cov[1, 1]) / a ** 2, dof
    b, _, cov, dof = _ols(Vp, 1.0 - strength / sigma_0, intercept=False)
    return b, float(cov[0, 0]), sigma_0, None, dof


#: Model name -> the regression of its law (see :func:`fit_knockdown`).
_FIT_FORMS: dict[str, Callable[..., tuple[float, float, float, float | None, int]]] = {
    'judd_wright': _fit_judd_wright,
    'power_law': _fit_power_law,
    'linear': _fit_linear,
}


@dataclass(frozen=True)
class KnockdownFit:
    """A knockdown-law coefficient fitted to coupon data (:func:`fit_knockdown`).

    Attributes
    ----------
    model : str
        ``'judd_wright'``, ``'power_law'`` or ``'linear'``.
    coefficient : float
        The fitted ``alpha``, ``n`` or ``beta`` (:attr:`coefficient_name`).
    standard_error : float
        Standard error of the coefficient.
    interval : tuple of float
        Two-sided confidence interval of the coefficient at
        :attr:`confidence` (Student *t* with :attr:`dof` degrees of
        freedom).
    confidence : float
        The confidence level of the intervals.
    pristine_strength : float
        Strength at zero porosity, in the unit of the data: the value
        given, or the fitted intercept.
    pristine_interval : tuple of float or None
        Confidence interval of a fitted pristine strength; ``None`` when it
        was given.
    n_points : int
        Number of data points.
    dof : int
        Residual degrees of freedom: ``n_points`` less the parameters
        fitted.
    rmse : float
        Root-mean-square error of the fitted strengths, in the unit of the
        data.
    r_squared : float
        Coefficient of determination of the fitted strengths (``nan`` when
        the strengths do not vary).
    max_Vp : float
        Largest void content of the data, a fraction: the fit says nothing
        beyond it.
    mode : str or None
        The loading mode the data belong to, when given.
    """

    model: str
    coefficient: float
    standard_error: float
    interval: tuple[float, float]
    confidence: float
    pristine_strength: float
    pristine_interval: tuple[float, float] | None
    n_points: int
    dof: int
    rmse: float
    r_squared: float
    max_Vp: float
    mode: str | None = None

    @property
    def coefficient_name(self) -> str:
        """``'alpha'``, ``'n'`` or ``'beta'``."""
        return _COEFFICIENT_NAMES[self.model][0]

    @property
    def override_argument(self) -> str:
        """The :class:`~porosity_fe.EmpiricalSolver` keyword the coefficient
        overrides (``'judd_wright_alpha'``, ``'power_law_n'`` or
        ``'linear_beta'``)."""
        return _COEFFICIENT_NAMES[self.model][1]

    @property
    def cov(self) -> float:
        """Coefficient of variation of the coefficient: the ``coef_cov`` of
        :func:`~porosity_fe.propagate_uncertainty`."""
        return self.standard_error / abs(self.coefficient)

    def knockdown(self, Vp: Any) -> np.ndarray:
        """The fitted law at ``Vp`` (a fraction, scalar or array)."""
        law, _ = _KNOCKDOWN_LAWS[self.model]
        return np.asarray(law(check_fraction(Vp), self.coefficient), dtype=float)

    def predict(self, Vp: Any) -> np.ndarray:
        """Fitted strength at ``Vp``: the pristine strength times the law."""
        return self.pristine_strength * self.knockdown(Vp)

    def as_override(self, layup_scale: float = 1.0) -> dict[str, float]:
        """``{mode: coefficient}`` for the solver's :attr:`override_argument`.

        Parameters
        ----------
        layup_scale : float
            The layup scale of the coupons' layup for this mode
            (``EmpiricalSolver.layup_scale[mode]``). The solver multiplies an
            override by the scale of the analysed layup, so a coefficient
            fitted on an amplified layup is divided by its own scale here.
            1 for ``shear``, ``ilss`` and ``transverse_tension``, and for UD,
            cross-ply and quasi-isotropic coupons.

        Raises
        ------
        ValueError
            Without a ``mode``, for a coefficient that is not positive (the
            strength did not fall with porosity) or a scale below 1.
        """
        if self.mode is None:
            raise ValueError(
                "The fit has no loading mode; pass mode= to fit_knockdown.")
        if not self.coefficient > 0.0:
            raise ValueError(
                f"The fitted {self.coefficient_name} = {self.coefficient:g} is "
                f"not positive: the strength did not fall with porosity, and "
                f"the solver takes positive coefficients only.")
        if not layup_scale >= 1.0:
            raise ValueError(f"layup_scale must be at least 1; got {layup_scale!r}.")
        return {self.mode: self.coefficient / float(layup_scale)}

    def summary(self) -> dict:
        """JSON-ready record of the fit."""
        return {
            'model': self.model,
            'mode': self.mode,
            'coefficient_name': self.coefficient_name,
            'coefficient': self.coefficient,
            'standard_error': self.standard_error,
            'interval': list(self.interval),
            'confidence': self.confidence,
            'pristine_strength': self.pristine_strength,
            'pristine_interval': (None if self.pristine_interval is None
                                  else list(self.pristine_interval)),
            'n_points': self.n_points,
            'dof': self.dof,
            'rmse': self.rmse,
            'r_squared': self.r_squared,
            'max_Vp': self.max_Vp,
        }


def fit_knockdown(Vp: Any, strength: Any, model: str = 'judd_wright', *,
                  pristine_strength: float | None = None,
                  mode: str | None = None,
                  confidence: float = 0.95) -> KnockdownFit:
    """Fit a knockdown law to strengths measured against void content.

    Parameters
    ----------
    Vp : array-like
        Void contents of the coupons, **fractions** in ``[0, 1]`` (0.02 for
        2 %).
    strength : array-like
        The strength of each coupon (or panel mean), in any one unit, or
        normalized. All positive.
    model : {'judd_wright', 'power_law', 'linear'}
        The law: ``exp(-alpha Vp)``, ``(1 - Vp)**n`` or ``1 - beta Vp``.
    pristine_strength : float, optional
        The strength at zero porosity, when it is known: the law is then
        fitted through it (one parameter). ``None`` (default) fits it with
        the coefficient (two parameters), which is the right choice when
        the least porous coupons still hold some voids, and for data
        normalized by them. Pass ``1.0`` for data normalized by a void-free
        strength.
    mode : str, optional
        The loading mode of the test (a strength mode of
        :class:`~porosity_fe.EmpiricalSolver`), kept on the result for
        :meth:`KnockdownFit.as_override`.
    confidence : float
        Confidence level of the intervals, in ``(0, 1)``.

    Returns
    -------
    KnockdownFit
        The coefficient, its standard error and confidence interval, the
        pristine strength, and the quality of the fit.

    Raises
    ------
    ValueError
        For an unknown model or mode; void contents that are not fractions
        (with a hint for percents); strengths that are not positive and
        finite; arrays of different lengths; too few points (one more than
        the parameters fitted) or void contents that do not vary; or a
        confidence outside ``(0, 1)``.

    Notes
    -----
    Each law is linear in a transformed variable, so the fit is an ordinary
    least-squares line:

    ===========  ==========================  =====================
    Model        Regressed                   Coefficient
    ===========  ==========================  =====================
    judd_wright  ``ln(strength)`` on ``Vp``  ``alpha = -slope``
    power_law    ``ln(strength)`` on         ``n = slope``
                 ``ln(1 - Vp)``
    linear       ``strength`` on ``Vp``      ``beta = -slope /
                                             intercept``
    ===========  ==========================  =====================

    With ``pristine_strength`` given the line goes through it and has no
    intercept. The two logarithmic fits weigh relative errors, the linear
    one absolute errors. The intervals are Student-*t* intervals of the
    line; for the linear law with a fitted intercept the variance of
    ``beta`` is the first-order (delta-method) one. They describe the
    scatter of these coupons about the law, not the error of the law's
    form, and the fit holds up to :attr:`KnockdownFit.max_Vp` only.

    The exponential porosity-toughness law of
    :class:`~porosity_fe.ToughnessKnockdown` has the Judd-Wright form, so
    its ``b`` is the ``alpha`` fitted to toughness values.

    Examples
    --------
    >>> import numpy as np
    >>> Vp = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    >>> fit = fit_knockdown(Vp, 80.0 * np.exp(-9.0 * Vp), mode='ilss')
    >>> round(fit.coefficient, 6), round(fit.pristine_strength, 6)
    (9.0, 80.0)
    >>> fit.as_override()                       # doctest: +SKIP
    {'ilss': 9.0}
    """
    if model not in _FIT_FORMS:
        raise ValueError(
            f"Unknown knockdown model {model!r}. Use one of {sorted(_FIT_FORMS)}.")
    if mode is not None and mode not in EmpiricalSolver.PRISTINE_STRENGTH_KEY:
        raise ValueError(
            f"Unknown loading mode {mode!r}. Use one of "
            f"{sorted(EmpiricalSolver.PRISTINE_STRENGTH_KEY)}.")
    if not (isinstance(confidence, (int, float)) and 0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie in (0, 1); got {confidence!r}.")
    x = np.atleast_1d(check_fraction(Vp)).astype(float)
    y = np.atleast_1d(np.asarray(strength, dtype=float))
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(
            f"Vp and strength must be one-dimensional and of one length; got "
            f"shapes {x.shape} and {y.shape}.")
    if not (np.isfinite(y).all() and (y > 0.0).all()):
        raise ValueError("Every strength must be a finite number above 0.")
    sigma_0 = None
    if pristine_strength is not None:
        if isinstance(pristine_strength, bool) or \
                not isinstance(pristine_strength, (int, float, np.floating)) or \
                not (math.isfinite(pristine_strength) and pristine_strength > 0.0):
            raise ValueError(
                f"pristine_strength must be a finite number above 0; got "
                f"{pristine_strength!r}.")
        sigma_0 = float(pristine_strength)
    n_params = 1 if sigma_0 is not None else 2
    if x.size < n_params + 1:
        raise ValueError(
            f"A fit of {n_params} parameter{'s' if n_params > 1 else ''} with "
            f"a confidence interval needs at least {n_params + 1} points; got "
            f"{x.size}.")
    varies = np.ptp(x) > 0.0 if sigma_0 is None else x.max() > 0.0
    if not varies:
        raise ValueError(
            "The void contents do not vary: no coefficient can be fitted "
            + ("(a single non-zero Vp at least is needed)."
               if sigma_0 is not None else
               "(two different Vp at least are needed)."))

    coef, var_coef, sigma_fit, var_ln_sigma, dof = _FIT_FORMS[model](x, y, sigma_0)

    from scipy import stats
    t = float(stats.t.ppf(0.5 * (1.0 + confidence), dof))
    se = math.sqrt(max(var_coef, 0.0))
    pristine_interval = None
    if var_ln_sigma is not None:
        half = t * math.sqrt(max(var_ln_sigma, 0.0))
        pristine_interval = (sigma_fit * math.exp(-half), sigma_fit * math.exp(half))

    law, _ = _KNOCKDOWN_LAWS[model]
    residual = y - sigma_fit * np.asarray(law(x, coef), dtype=float)
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return KnockdownFit(
        model=model, coefficient=float(coef), standard_error=se,
        interval=(float(coef - t * se), float(coef + t * se)),
        confidence=float(confidence), pristine_strength=float(sigma_fit),
        pristine_interval=pristine_interval, n_points=int(x.size), dof=int(dof),
        rmse=math.sqrt(ss_res / x.size),
        r_squared=(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else math.nan,
        max_Vp=float(x.max()), mode=mode)
