"""S-N (fatigue) knockdown model (#59)."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ._vp import check_fraction

# ============================================================
# Fatigue (S-N) knockdown surface (issue #59).
#
# Log-linear (Mandell) fatigue knockdown::
#
#     S_N / S_0 = max(floor, 1 - b * log10(N))
#
# with mode-keyed slopes ``b``. The table below is calibrated against
# representative CFRP S-N data at ``R = 0.1`` (tension-tension): the
# tension/compression slopes follow the Mandell 1991 review value of
# b ~ 0.1 per decade for IM-class CFRP; ILSS / matrix-dominated shear
# is slightly shallower (b ~ 0.08) as reported by Curtis (1989) and the
# WWFE-III fatigue exercise. These are screening-level values; production
# allowables should use a fully populated S-N matrix per CMH-17 Vol. 2.
#
# These are the canonical objects; ``Calibration.FATIGUE_B_QI`` /
# ``Calibration.FATIGUE_KD_FLOOR`` in :mod:`porosity_fe.empirical`
# re-export them so every tuning surface is discoverable from one
# namespace (#121). Update values here; the alias picks them up.
#
# References:
# - Mandell, J. F., "Fatigue Behavior of Fiber-Resin Composites,"
#   Developments in Reinforced Plastics 2, 1991.
# - Curtis, P. T., "The fatigue behaviour of fibrous composite materials,"
#   J. Strain Analysis, 1989.
# ============================================================
# Back-compat alias retained for direct importers (tests, downstream code).
# New code should reference ``Calibration.FATIGUE_B_QI``.
_FATIGUE_B_QI: dict[str, float] = {
    'tension': 0.10,
    'compression': 0.10,
    'shear': 0.08,
    'ilss': 0.08,
    'transverse_tension': 0.10,
}

# Floor clamp for the log-linear formula: at very large N the linear
# extrapolation predicts a negative knockdown. Clamp to 1% of static so
# downstream multiplications stay well-behaved, and emit a warning so the
# caller knows they are off the calibration range.
# Back-compat alias; canonical reference is ``Calibration.FATIGUE_KD_FLOOR``.
_FATIGUE_KD_FLOOR = 0.01
# Stress ratio the S-N slopes were calibrated at (tension-tension).
_FATIGUE_R_CALIBRATION = 0.1

#: Mean-stress corrections :class:`FatigueModel` takes (``None`` is none).
_MEAN_STRESS_CORRECTIONS = ('goodman',)


def _load_ratio(mode: str, R: float) -> float:
    """The ratio of the smaller to the larger load magnitude of a cycle.

    ``R = sigma_min / sigma_max`` is in ``[0, 1]`` for tension-tension (and
    same-sign shear) cycling and ``>= 1`` for compression-compression, where
    the ratio is ``1 / R``. Reversed loading (``R < 0``) needs fatigue data
    at ``R = -1``, which the slopes are not calibrated on.
    """
    if R < 0.0:
        raise ValueError(
            f"FatigueModel: R={R!r} is reversed (tension-compression) loading; "
            f"the 'goodman' correction is anchored at R={_FATIGUE_R_CALIBRATION} "
            f"and the static strength, and needs data at R = -1 for reversed "
            f"cycles.")
    if mode == 'compression':
        if R < 1.0:
            raise ValueError(
                f"FatigueModel: R={R!r} is tension-tension cycling, which a "
                f"compression strength does not see. Compression-compression "
                f"cycling has R = sigma_min / sigma_max >= 1 (e.g. 10).")
        return 1.0 / R
    if R > 1.0:
        raise ValueError(
            f"FatigueModel: R={R!r} is compression-compression cycling; for the "
            f"{mode!r} mode give 0 <= R <= 1 (tension-tension, e.g. 0.1).")
    return R


def _goodman(x: float, r: float) -> float:
    """``S_max / S_0`` at the load ratio ``r`` from its value ``x`` at the
    calibration ratio, on the straight constant-life line through the
    calibration point and the static strength (the Goodman construction).

    In mean stress ``m`` and amplitude ``a`` (both over ``S_0``) the line
    runs from ``(x (1 + r0) / 2, x (1 - r0) / 2)`` to ``(1, 0)``; a cycle at
    ``r`` with maximum ``s`` has ``m = s (1 + r) / 2`` and ``a = s (1 - r) / 2``.
    """
    r0 = _FATIGUE_R_CALIBRATION
    if r == r0:
        return x
    k = x * (1.0 - r0) / (2.0 - x * (1.0 + r0))     # amplitude per unit of 1 - m
    return 2.0 * k / ((1.0 - r) + k * (1.0 + r))


@dataclass
class FatigueModel:
    """S-N (cycles-to-failure) knockdown surface.

    Implements the log-linear (Mandell-style) form::

        S_N / S_0 = max(floor, 1 - b * log10(N))

    where ``b`` is a mode-keyed slope (see :data:`_FATIGUE_B_QI`),
    ``N`` is the number of load cycles, and ``floor`` (default 0.01)
    is a small lower clamp that prevents the linear extrapolation from
    going negative at very large ``N``.

    The default slopes are calibrated for quasi-isotropic CFRP at
    ``R = 0.1`` (typical tension-tension fatigue). By default the ``R``
    argument does not change the result; ``mean_stress='goodman'`` corrects
    the curve to another stress ratio.

    Attributes
    ----------
    b : dict[str, float], optional
        Mode-keyed slope override. Modes absent from the override fall
        back to :data:`_FATIGUE_B_QI`.
    mean_stress : {None, 'goodman'}, optional
        Opt-in mean-stress correction. ``None`` (default): ``R`` does not
        change the result, as before. ``'goodman'``: the maximum stress at
        ``R`` lies on the straight constant-life line through the curve's
        point at the calibration ratio and the static strength, so a ratio
        closer to 1 (less amplitude) gives less knockdown and ``R = 1`` none.
        It covers same-sign cycling: ``0 <= R <= 1`` for the tension, shear,
        ILSS and transverse-tension modes, ``R >= 1`` (compression-
        compression, the ratio ``1 / R``) for compression; reversed loading
        raises. For compression the table records no ratio of its own, and
        ``R = 10`` (the same load-magnitude ratio) is taken as its
        calibration ratio. A linear constant-life line is conservative for
        many carbon-fibre laminates, whose diagrams bulge outwards; it is
        not calibrated against porous-laminate fatigue data here.
    porosity_slope : float, optional
        Opt-in porosity dependence of the slope:
        ``b(Vp) = b * (1 + porosity_slope * Vp)``, ``Vp`` a fraction, so the
        curve falls faster in a more porous laminate. Must be ``>= 0``.
        Default 0 (the slope does not depend on porosity, as before). No
        value is shipped: the validation database holds no fatigue data to
        fit it on, so pass one fitted to your own tests.

    Notes
    -----
    For ``cycles = None`` callers should bypass this class entirely (the
    :meth:`EmpiricalSolver.get_failure_load` path returns ``1.0`` in that
    case). The model is screening-level; production allowables should
    come from a fully populated test matrix per the applicable spec
    (e.g. CMH-17 Vol. 2 fatigue protocols).
    """
    b: dict[str, float] | None = None
    mean_stress: str | None = None
    porosity_slope: float = 0.0

    def __post_init__(self) -> None:
        if self.mean_stress is not None and self.mean_stress not in _MEAN_STRESS_CORRECTIONS:
            raise ValueError(
                f"FatigueModel.mean_stress must be None or one of "
                f"{list(_MEAN_STRESS_CORRECTIONS)}; got {self.mean_stress!r}.")
        slope = self.porosity_slope
        if isinstance(slope, bool) or not isinstance(slope, (int, float, np.integer,
                                                              np.floating)) \
                or not math.isfinite(float(slope)) or float(slope) < 0.0:
            raise ValueError(
                f"FatigueModel.porosity_slope must be a finite number >= 0 (per "
                f"unit void fraction); got {slope!r}.")
        self.porosity_slope = float(slope)

    @property
    def is_default(self) -> bool:
        """Whether the model is the calibrated default: no slope override,
        no mean-stress correction and no porosity dependence."""
        return self.b is None and self.mean_stress is None and self.porosity_slope == 0.0

    def describe(self) -> dict:
        """JSON-ready description of the model's options."""
        return {'b': None if self.b is None else dict(self.b),
                'mean_stress': self.mean_stress,
                'porosity_slope': self.porosity_slope,
                'R_calibration': _FATIGUE_R_CALIBRATION}

    def _slope(self, mode: str) -> float:
        if mode not in _FATIGUE_B_QI:
            raise ValueError(
                f"Unknown fatigue mode {mode!r}. "
                f"Use one of {sorted(_FATIGUE_B_QI)}."
            )
        if self.b is not None and mode in self.b:
            return float(self.b[mode])
        return float(_FATIGUE_B_QI[mode])

    def knockdown_factor(self, mode: str, cycles: float,
                         R: float | None = 0.1, *, Vp: float | None = None) -> float:
        """Multiplicative fatigue knockdown for the given mode.

        Parameters
        ----------
        mode : str
            Loading mode (see :data:`_FATIGUE_B_QI`).
        cycles : float
            Number of load cycles ``N``. Must be a positive finite
            value; ``cycles = 1`` corresponds to the static (one-cycle)
            allowable and returns ``1.0``.
        R : float or None, optional
            Stress ratio ``sigma_min / sigma_max``. Without a
            :attr:`mean_stress` correction it is not used in the result:
            the slopes are calibrated at ``R = 0.1`` (tension-tension), and
            any other value emits a :class:`UserWarning` saying so. With
            ``mean_stress='goodman'`` it corrects the curve (see the
            class). ``None`` is the calibration ratio (``R = 10`` for
            compression under ``'goodman'``).
        Vp : float, optional
            Keyword-only. Void content, a **fraction**; needed (and used)
            only when :attr:`porosity_slope` is set.

        Returns
        -------
        float
            Knockdown factor in ``[floor, 1.0]``. When the linear
            extrapolation would go below the floor (``floor = 0.01``),
            the value is clamped and a :class:`UserWarning` is emitted.
        """
        if R is not None and not np.isfinite(float(R)):
            raise ValueError(
                f"FatigueModel.knockdown_factor: R must be finite, got {R!r}."
            )
        ratio = None
        if self.mean_stress is not None:
            # The load ratio of the correction (None: the calibration one).
            if R is not None:
                ratio = _load_ratio(mode, float(R))
        elif R is not None and not np.isclose(float(R), _FATIGUE_R_CALIBRATION):
            # Without a correction R does not enter the result; say so when
            # it differs from the calibration ratio (IMPROVEMENT_PLAN 2.8).
            warnings.warn(
                f"FatigueModel: R={R!r} does not change the result. The S-N "
                f"slopes are calibrated at R={_FATIGUE_R_CALIBRATION} "
                f"(tension-tension) and no mean-stress correction is applied.",
                UserWarning,
                stacklevel=2,
            )

        N = float(cycles)
        if not np.isfinite(N) or N < 1.0:
            raise ValueError(
                f"FatigueModel.knockdown_factor: cycles must be a finite "
                f"value >= 1, got {cycles!r}."
            )

        b = self._slope(mode)
        if self.porosity_slope:
            if Vp is None:
                raise ValueError(
                    "FatigueModel.knockdown_factor: porosity_slope is set, so "
                    "the void content Vp (a fraction) is needed.")
            b *= 1.0 + self.porosity_slope * float(check_fraction(Vp))
        raw = 1.0 - b * np.log10(N)
        if raw < _FATIGUE_KD_FLOOR:
            warnings.warn(
                f"Fatigue knockdown for mode={mode!r}, cycles={N:.3g}, "
                f"R={R!r} extrapolates to {raw:.3g} (<= floor "
                f"{_FATIGUE_KD_FLOOR}); clamping. The log-linear model is "
                f"off its calibration range — consider a richer S-N model.",
                UserWarning,
                stacklevel=2,
            )
            raw = _FATIGUE_KD_FLOOR
        x = float(min(raw, 1.0))
        if ratio is not None:
            x = _goodman(x, ratio)
        return float(min(x, 1.0))
