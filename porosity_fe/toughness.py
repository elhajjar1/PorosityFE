"""Porosity-dependent interlaminar toughness (delamination plan, phase 4).

Voids at the crack plane lower the interlaminar toughness. This module holds
the law that knocks the pristine toughness of
:class:`~porosity_fe.MaterialProperties` (``G_Ic``, ``G_IIc``, ``G_IIIc``,
N/mm) down with the local void fraction ``Vp``::

    G_c(Vp) = G_c0 * f(Vp),    0 <= f(Vp) <= 1,  f(0) = 1

and its selection, :class:`ToughnessKnockdown`. The fracture outputs of the
FE path (:meth:`FESolver.solve(..., toughness_knockdown=...)
<porosity_fe.FESolver.solve>`, :func:`~porosity_fe.fracture_load_factors`,
:func:`~porosity_fe.critical_energy_release_rate`) evaluate it at the
porosity of each crack-front node; :func:`toughness_knockdown_factor`
evaluates it without FE, for screening. The Benzeggagh-Kenane exponent
``bk_eta`` is not changed.

Laws (``Vp`` a fraction in ``[0, 1]``, ``b`` per unit ``Vp``):

- ``'exponential'`` (the default): ``f = exp(-b Vp)``, one ``b`` per mode;
  without coefficients it uses the placeholders below;
- ``'linear'``: ``f = max(1 - b Vp, 0)``; it has no placeholder, so
  ``b_I`` and ``b_II`` are required;
- ``'none'``: ``f = 1``, the pristine toughness;
- a callable ``f(Vp, mode) -> float`` with ``Vp`` a float and ``mode`` one
  of ``'I'``, ``'II'``, ``'III'``; every value it returns must lie in
  ``[0, 1]``.

Mode III follows mode II (``b_III = b_II``) unless ``b_III`` is given.

The law never raises the toughness (``f <= 1``): it is conservative. Mode I
*propagation* toughness can rise with voids through fibre and ply bridging
and multiple crack planes (Carraro, Maragoni and Quaresimin 2015); the law
ignores that rise. The linear VCCT criterion of
:mod:`porosity_fe.fe.vcct` is a growth-onset criterion, so the base values
it knocks down are initiation (or precracked) toughness.

Calibration status: **uncalibrated**. No published ``G_c(Vp)`` relation
exists, and the default exponents are two-point estimates from three
studies (theory page "Delamination", section "Porosity and toughness"):

=====  ======  =========================================================
Mode   ``b``   Source (per unit ``Vp``)
=====  ======  =========================================================
I      6.2     mean of 4.97 (Olivier et al. 2007, ICCM16, T700GC/M21
               DCB: "a void content of 5% induces a decrease by about
               22% in GIC") and
               7.45 (Hakim et al. 2017, Table 1: static ``G_Ic`` 0.52 to
               0.45 kJ/m^2 from 1.49 % to 3.43 % voids)
II     14      Agius and Fox 2015 (abstract): ``G_IIc`` 21 % higher below
               0.3 % voids than at 1.5 %, so ``ln(1.21) / dVp`` with
               ``dVp`` from 1.2 % to 1.5 % gives 12.7 to 15.9; 14.1 at the
               midpoint (0.15 %) of the stated low-void range, rounded
III    = II    no data
=====  ======  =========================================================

Every result that uses a placeholder at a non-zero ``Vp`` is flagged
(``CrackFront.toughness_uncalibrated``, ``summary().details`` and the JSON
``fracture`` block), and the first use logs one INFO line per process. The
studies span ``Vp <= 0.05``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

logger = logging.getLogger("porosity_fe_analysis")

#: Fracture modes of the law: opening, sliding and tearing.
ToughnessMode = Literal['I', 'II', 'III']

#: Names of the built-in laws (``_TOUGHNESS_LAWS`` plus ``'none'``).
ToughnessLaw = Literal['exponential', 'linear', 'none']

_MODES: tuple[ToughnessMode, ...] = ('I', 'II', 'III')


def _f_exponential(Vp: np.ndarray, b: float) -> np.ndarray:
    """``exp(-b Vp)``: 1 at ``Vp = 0``, never 0."""
    return np.exp(-b * Vp)


def _f_linear(Vp: np.ndarray, b: float) -> np.ndarray:
    """``max(1 - b Vp, 0)``, clipped at zero like the empirical linear law."""
    return np.maximum(1.0 - b * Vp, 0.0)


#: Built-in law name -> ``f(Vp, b)``, the one implementation of each form.
#: ``'none'`` (``f = 1``) needs no entry.
_TOUGHNESS_LAWS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    'exponential': _f_exponential,
    'linear': _f_linear,
}

#: Placeholder exponents ``b`` (per unit ``Vp``) of the exponential law, by
#: mode; mode III follows mode II. Uncalibrated two-point estimates (see the
#: module docstring): mode I is the mean of Olivier et al. 2007 (4.97) and
#: Hakim et al. 2017 (7.45); mode II is Agius and Fox
#: 2015 (12.7 to 15.9, 14.1 at the midpoint of its low-void range).
#: ``Calibration.TOUGHNESS_B_PLACEHOLDER`` is the same object.
_TOUGHNESS_B_PLACEHOLDER: dict[str, float] = {'I': 6.2, 'II': 14.0}

#: Set once the uncalibrated-placeholder INFO line has been logged.
_uncalibrated_logged = False


def _check_mode(mode: object) -> ToughnessMode:
    if mode not in _MODES:
        raise ValueError(
            f"Unknown fracture mode {mode!r}. Use one of {list(_MODES)}.")
    return mode


def _check_Vp(Vp: Any) -> np.ndarray:
    """``Vp`` as a float array, checked to be a fraction in ``[0, 1]``."""
    arr = np.asarray(Vp, dtype=float)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all() and arr.min() >= 0.0 and arr.max() <= 1.0:
        return arr
    worst = float(arr[~finite][0]) if not finite.all() else (
        float(arr.max()) if arr.max() > 1.0 else float(arr.min()))
    hint = (f" Did you pass a percent? Use {worst / 100:.4f} instead of {worst:g}."
            if math.isfinite(worst) and worst >= 1.001 else "")
    raise ValueError(
        f"Vp must be a void fraction in [0, 1] (0.02 for 2 %), got {worst!r}.{hint}")


def _check_coefficient(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(
            f"{name} must be a finite number >= 0 (per unit Vp); got {value!r}. "
            "A negative coefficient would raise the toughness with porosity.")
    return v


@dataclass(frozen=True)
class ToughnessKnockdown:
    """How porosity knocks down the interlaminar toughness: ``G_c(Vp) = G_c0 f(Vp)``.

    Selects the law and its coefficients for the fracture outputs
    (:meth:`FESolver.solve(..., toughness_knockdown=...)
    <porosity_fe.FESolver.solve>`, :func:`~porosity_fe.fracture_load_factors`,
    :func:`~porosity_fe.critical_energy_release_rate`,
    :func:`toughness_knockdown_factor`). It lives outside
    :class:`~porosity_fe.MaterialProperties`: the material holds the
    pristine toughness, this holds the porosity dependence. Those functions
    also take ``None`` (no knockdown), a law name or a callable in its
    place.

    Parameters
    ----------
    law : {'exponential', 'linear', 'none'} or callable
        ``'exponential'`` (default): ``f = exp(-b Vp)``. ``'linear'``:
        ``f = max(1 - b Vp, 0)``. ``'none'``: ``f = 1``. A callable
        ``f(Vp, mode) -> float`` receives ``Vp`` as a float fraction and
        ``mode`` as ``'I'``, ``'II'`` or ``'III'`` (return the mode II
        value for ``'III'`` to follow mode II); a value outside ``[0, 1]``
        or not finite raises :class:`ValueError` where it is evaluated.
    b_I, b_II : float, optional
        Coefficients (per unit ``Vp``, ``>= 0``) of modes I and II for the
        exponential and linear laws. ``None`` takes the uncalibrated
        placeholder of the exponential law (6.2 and 14; see the theory
        page "Delamination", section "Porosity and toughness"); the linear
        law has no placeholder and needs both.
    b_III : float, optional
        Mode III coefficient; ``None`` follows mode II.

    Raises
    ------
    ValueError
        For an unknown law name, a negative or non-finite coefficient, a
        coefficient given to ``'none'`` or a callable, or a linear law
        without ``b_I`` and ``b_II``.
    TypeError
        If a coefficient is not a number or ``law`` is neither a name nor a
        callable.

    Examples
    --------
    >>> ToughnessKnockdown().uncalibrated            # the placeholders
    True
    >>> mine = ToughnessKnockdown(b_I=4.0, b_II=10.0)
    >>> mine.uncalibrated, mine.coefficient('III')
    (False, 10.0)
    >>> float(mine.factor(0.02, 'I'))                # exp(-0.08)  # doctest: +SKIP
    0.923...
    """

    law: str | Callable[[float, str], float] = 'exponential'
    b_I: float | None = None
    b_II: float | None = None
    b_III: float | None = None

    def __post_init__(self) -> None:
        law = self.law
        if isinstance(law, str):
            if law not in _TOUGHNESS_LAWS and law != 'none':
                raise ValueError(
                    f"Unknown toughness law {law!r}. Use one of "
                    f"{sorted([*_TOUGHNESS_LAWS, 'none'])} or pass a callable "
                    "f(Vp, mode) -> float in [0, 1].")
        elif not callable(law):
            raise TypeError(
                "law must be a law name or a callable f(Vp, mode) -> float; "
                f"got {type(law).__name__}.")
        given = {name: getattr(self, name) for name in ('b_I', 'b_II', 'b_III')
                 if getattr(self, name) is not None}
        for name, value in given.items():
            object.__setattr__(self, name, _check_coefficient(name, value))
        if not self._has_coefficients and given:
            raise ValueError(
                "Only the 'exponential' and 'linear' laws take coefficients; "
                f"got {', '.join(given)} with {self.describe()}.")
        if law == 'linear' and (self.b_I is None or self.b_II is None):
            raise ValueError(
                "The linear toughness law has no placeholder coefficients: "
                "pass b_I and b_II (per unit Vp).")

    @property
    def _has_coefficients(self) -> bool:
        return isinstance(self.law, str) and self.law in _TOUGHNESS_LAWS

    def coefficient(self, mode: ToughnessMode) -> float | None:
        """Coefficient ``b`` (per unit ``Vp``) used for ``mode``.

        The given value, else mode II's for mode III, else the placeholder;
        ``None`` for ``'none'`` and a callable.
        """
        mode = _check_mode(mode)
        if not self._has_coefficients:
            return None
        if mode == 'III':
            if self.b_III is not None:
                return self.b_III
            mode = 'II'
        value = self.b_I if mode == 'I' else self.b_II
        return _TOUGHNESS_B_PLACEHOLDER[mode] if value is None else value

    def uses_placeholder(self, mode: ToughnessMode) -> bool:
        """Whether ``mode`` takes the uncalibrated placeholder coefficient."""
        mode = _check_mode(mode)
        if self.law != 'exponential':
            return False
        if mode == 'I':
            return self.b_I is None
        if mode == 'III' and self.b_III is not None:
            return False
        return self.b_II is None

    @property
    def uncalibrated(self) -> bool:
        """Whether any mode takes a placeholder coefficient."""
        return any(self.uses_placeholder(m) for m in _MODES)

    def factor(self, Vp: Any, mode: ToughnessMode) -> np.ndarray:
        """Knockdown ``f(Vp) = G_c(Vp) / G_c0`` of ``mode``, in ``[0, 1]``.

        Parameters
        ----------
        Vp : array_like
            Void fraction(s) in ``[0, 1]``.
        mode : {'I', 'II', 'III'}

        Returns
        -------
        np.ndarray
            One factor per ``Vp``, the shape of ``Vp``; exactly 1 at
            ``Vp = 0`` for the built-in laws.

        Raises
        ------
        ValueError
            If ``Vp`` is not a fraction in ``[0, 1]`` (with a hint when it
            looks like a percent), ``mode`` is unknown, or a callable
            returns a value outside ``[0, 1]``.
        """
        mode = _check_mode(mode)
        arr = _check_Vp(Vp)
        if callable(self.law):
            return self._callable_factor(arr, mode)
        if self.law == 'none':
            return np.ones(arr.shape)
        b = self.coefficient(mode)
        assert b is not None
        return _TOUGHNESS_LAWS[self.law](arr, b)

    def _callable_factor(self, Vp: np.ndarray, mode: ToughnessMode) -> np.ndarray:
        law = self.law
        assert callable(law)
        values, inverse = np.unique(Vp.ravel(), return_inverse=True)
        out = np.empty(values.size, dtype=float)
        for i, v in enumerate(values):
            try:
                f = float(law(float(v), mode))
            except Exception as exc:
                raise ValueError(
                    f"The toughness knockdown callable raised "
                    f"{type(exc).__name__} at Vp={v:g}, mode={mode!r}: {exc}") from exc
            if not math.isfinite(f) or f < 0.0 or f > 1.0:
                raise ValueError(
                    f"The toughness knockdown callable returned {f!r} at "
                    f"Vp={v:g}, mode={mode!r}; expected a factor in [0, 1] "
                    "(the toughness must not rise with porosity).")
            out[i] = f
        return out[np.asarray(inverse).reshape(-1)].reshape(Vp.shape)

    def describe(self) -> str:
        """One-line description of the law and its coefficients (exports)."""
        law = self.law
        if callable(law):
            name = getattr(law, '__qualname__', None) or type(law).__name__
            return f"user callable {name}"
        if law == 'none':
            return "none (pristine toughness)"
        form = 'G_c0 exp(-b Vp)' if law == 'exponential' else 'G_c0 max(1 - b Vp, 0)'
        coefs = ', '.join(f"b_{m} = {self.coefficient(m):g}"
                          for m in _MODES)
        held = [f"b_{m}" for m in _MODES if self.uses_placeholder(m)]
        if len(held) == len(_MODES):
            tag = " (placeholder, uncalibrated)"
        elif held:
            tag = f" ({', '.join(held)} placeholder, uncalibrated)"
        else:
            tag = ""
        return f"{law}: G_c = {form}, {coefs}{tag}"


#: A law selection as the API takes it: the dataclass, a law name, a
#: callable ``f(Vp, mode)``, or ``None`` for no knockdown.
ToughnessKnockdownSpec = ToughnessKnockdown | str | Callable[[float, str], float] | None


def _resolve_toughness_knockdown(value: ToughnessKnockdownSpec) -> ToughnessKnockdown:
    """The :class:`ToughnessKnockdown` named by an API argument."""
    if isinstance(value, ToughnessKnockdown):
        return value
    if value is None:
        return ToughnessKnockdown('none')
    if isinstance(value, str) or callable(value):
        return ToughnessKnockdown(value)
    raise TypeError(
        "toughness_knockdown must be a ToughnessKnockdown, a law name "
        "('exponential', 'linear', 'none'), a callable f(Vp, mode) or None; "
        f"got {type(value).__name__}.")


def _placeholder_applied(knockdown: ToughnessKnockdown, Vp: Any,
                         modes: tuple[ToughnessMode, ...] = _MODES) -> bool:
    """Whether a placeholder coefficient of ``modes`` lowered the toughness
    at some ``Vp``, i.e. a placeholder is in use and some ``Vp > 0``."""
    if not any(knockdown.uses_placeholder(m) for m in modes):
        return False
    return bool(np.any(np.asarray(Vp, dtype=float) > 0.0))


def _log_uncalibrated_once(knockdown: ToughnessKnockdown) -> None:
    """One INFO line per process the first time a placeholder is applied."""
    global _uncalibrated_logged
    if _uncalibrated_logged:
        return
    _uncalibrated_logged = True
    logger.info(
        "Interlaminar toughness knocked down by porosity with uncalibrated "
        "placeholder coefficients (%s; two-point literature estimates, data "
        "up to Vp = 0.05). Results that use them carry "
        "toughness_uncalibrated = True. Pass toughness_knockdown="
        "ToughnessKnockdown(b_I=..., b_II=...), another law, or None for the "
        "pristine toughness.", knockdown.describe())


def toughness_knockdown_factor(Vp: Any, mode: ToughnessMode = 'I',
                               toughness_knockdown: ToughnessKnockdownSpec = 'exponential',
                               ) -> float | np.ndarray:
    """Porosity knockdown of the interlaminar toughness, ``G_c(Vp) / G_c0``.

    A screening function without FE, e.g. for a nonconformance report: the
    factor that multiplies the pristine ``G_Ic`` (``mode='I'``), ``G_IIc``
    or ``G_IIIc`` at the void fraction ``Vp`` of the interface. The FE
    fracture outputs apply the same law at every crack-front node. It
    changes no empirical strength result.

    Parameters
    ----------
    Vp : float or array_like
        Void fraction(s) at the interface, in ``[0, 1]`` (``0.02`` for
        2 %).
    mode : {'I', 'II', 'III'}
        Fracture mode. Mode III follows mode II unless ``b_III`` is set.
    toughness_knockdown : ToughnessKnockdown, str, callable or None
        The law (see :class:`ToughnessKnockdown`): default ``'exponential'``
        with the uncalibrated placeholders ``b_I = 6.2``, ``b_II = 14``;
        ``None`` gives 1.

    Returns
    -------
    float or np.ndarray
        The factor in ``[0, 1]``: a float for a scalar ``Vp``, else an
        array of the shape of ``Vp``. The default law logs one INFO line
        per process the first time it lowers a toughness, since its
        coefficients are uncalibrated
        (:attr:`ToughnessKnockdown.uncalibrated`).

    Raises
    ------
    ValueError
        If ``Vp`` is not a fraction in ``[0, 1]``, ``mode`` is unknown or
        the law is invalid.

    Examples
    --------
    >>> toughness_knockdown_factor(0.03, 'I')           # exp(-6.2 * 0.03)  # doctest: +SKIP
    0.830...
    >>> toughness_knockdown_factor(0.03, 'II', ToughnessKnockdown(b_I=5.0, b_II=12.0))  # doctest: +SKIP
    0.697...
    """
    law = _resolve_toughness_knockdown(toughness_knockdown)
    mode = _check_mode(mode)
    f = law.factor(Vp, mode)
    if _placeholder_applied(law, Vp, (mode,)):
        _log_uncalibrated_once(law)
    return float(f) if f.ndim == 0 else f
