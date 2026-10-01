"""Dedicated unit tests for :mod:`porosity_fe.fatigue` (IMPROVEMENT_PLAN 6.6).

``FatigueModel`` was previously exercised only through
``EmpiricalSolver.get_failure_load(cycles=...)`` (tests/test_materials.py)
and the guard-clause checks in tests/test_empirical.py. These tests pin the
model's own contract: the closed-form log-linear surface, the one-cycle
anchor, monotonicity, the ``[floor, 1]`` bounds, slope overrides, and the
informational-only ``R`` argument.
"""

import math
import warnings

import numpy as np
import pytest

from porosity_fe import Calibration, FatigueModel
from porosity_fe.fatigue import _FATIGUE_B_QI, _FATIGUE_KD_FLOOR

MODES = sorted(_FATIGUE_B_QI)


class TestClosedForm:
    """``knockdown_factor`` equals ``1 - b * log10(N)`` inside the valid range."""

    @pytest.mark.parametrize("mode", MODES)
    def test_one_cycle_is_static_allowable(self, mode):
        """N = 1 is the static anchor: log10(1) = 0, so the factor is exactly 1."""
        assert FatigueModel().knockdown_factor(mode, 1) == 1.0

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("decades", [1, 3, 6])
    def test_matches_log_linear_formula(self, mode, decades):
        """At N = 10**k the factor is 1 - b*k with the canonical per-mode slope."""
        b = _FATIGUE_B_QI[mode]
        kd = FatigueModel().knockdown_factor(mode, 10.0 ** decades)
        assert kd == pytest.approx(1.0 - b * decades, rel=1e-12)

    def test_default_slopes_match_documented_table(self):
        """Screening slopes documented in the module: b=0.10 fibre / transverse, 0.08 shear/ILSS."""
        assert _FATIGUE_B_QI == {
            'tension': 0.10, 'compression': 0.10, 'shear': 0.08,
            'ilss': 0.08, 'transverse_tension': 0.10,
        }
        assert _FATIGUE_KD_FLOOR == 0.01

    def test_shear_is_shallower_than_tension(self):
        """Matrix-dominated shear/ILSS retain more strength per decade than tension."""
        fm = FatigueModel()
        for mode in ('shear', 'ilss'):
            assert fm.knockdown_factor(mode, 1e6) > fm.knockdown_factor('tension', 1e6)

    def test_returns_builtin_float(self):
        """The factor is a plain ``float`` (JSON-friendly), not a numpy scalar."""
        kd = FatigueModel().knockdown_factor('tension', 1e4)
        assert type(kd) is float

    def test_accepts_integer_and_numpy_cycles(self):
        """Integer and numpy-scalar cycle counts give the same answer as a float."""
        fm = FatigueModel()
        ref = fm.knockdown_factor('compression', 1e5)
        assert fm.knockdown_factor('compression', 100_000) == pytest.approx(ref, rel=1e-15)
        assert fm.knockdown_factor('compression', np.int64(100_000)) == pytest.approx(ref, rel=1e-15)
        assert fm.knockdown_factor('compression', np.float32(1e5)) == pytest.approx(ref, rel=1e-6)


class TestMonotonicityAndBounds:
    """More cycles never increase strength; the result stays in ``[floor, 1]``."""

    @pytest.mark.parametrize("mode", MODES)
    def test_strictly_decreasing_until_floor(self, mode):
        """Inside the calibrated range each additional decade lowers the factor."""
        fm = FatigueModel()
        cycles = np.logspace(0, 8, 33)
        kds = [fm.knockdown_factor(mode, n) for n in cycles]
        assert all(b < a for a, b in zip(kds, kds[1:], strict=False))

    @pytest.mark.parametrize("mode", MODES)
    def test_non_increasing_over_full_range_including_floor(self, mode):
        """Across the floor transition the sequence is non-increasing and bounded."""
        fm = FatigueModel()
        cycles = np.logspace(0, 20, 41)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            kds = [fm.knockdown_factor(mode, n) for n in cycles]
        assert all(b <= a for a, b in zip(kds, kds[1:], strict=False))
        assert all(_FATIGUE_KD_FLOOR <= kd <= 1.0 for kd in kds)
        assert kds[0] == 1.0
        assert kds[-1] == _FATIGUE_KD_FLOOR

    def test_no_warning_inside_calibrated_range(self):
        """Ordinary cycle counts must not emit the off-calibration warning."""
        fm = FatigueModel()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for mode in MODES:
                fm.knockdown_factor(mode, 1e7)

    def test_just_above_floor_is_not_clamped(self):
        """A raw value slightly above the floor is returned unclamped, silently."""
        b = _FATIGUE_B_QI['tension']
        # raw = 1 - b*log10(N) = 0.02 -> log10(N) = 0.98 / b
        N = 10.0 ** (0.98 / b)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kd = FatigueModel().knockdown_factor('tension', N)
        assert kd == pytest.approx(0.02, rel=1e-9)

    def test_below_floor_clamps_and_warns_with_context(self):
        """Below the floor the value is clamped and the warning names mode and R."""
        b = _FATIGUE_B_QI['ilss']
        N = 10.0 ** (1.5 / b)       # raw = -0.5
        with pytest.warns(UserWarning, match=r"mode='ilss'.*R=0\.5") as rec:
            kd = FatigueModel().knockdown_factor('ilss', N, R=0.5)
        assert kd == _FATIGUE_KD_FLOOR
        # stacklevel=2 attributes the warning to the caller, not fatigue.py.
        assert rec[0].filename == __file__

    def test_upper_clamp_at_one(self):
        """A (non-physical) negative slope override cannot push the factor above 1."""
        fm = FatigueModel(b={'tension': -0.2})
        assert fm.knockdown_factor('tension', 1e6) == 1.0


class TestSlopeOverride:
    """``FatigueModel(b=...)`` is a partial, per-mode override."""

    def test_override_used_for_named_mode(self):
        """An overridden mode uses the supplied slope in the closed form."""
        fm = FatigueModel(b={'compression': 0.05})
        assert fm.knockdown_factor('compression', 1e4) == pytest.approx(1.0 - 0.05 * 4, rel=1e-12)

    def test_unlisted_modes_fall_back_to_defaults(self):
        """Modes absent from the override keep the canonical slope."""
        fm = FatigueModel(b={'compression': 0.05})
        default = FatigueModel()
        for mode in MODES:
            if mode == 'compression':
                continue
            assert fm.knockdown_factor(mode, 1e5) == default.knockdown_factor(mode, 1e5)

    def test_integer_slope_is_coerced(self):
        """An integer slope is coerced to float (b=0 means no fatigue effect)."""
        fm = FatigueModel(b={'shear': 0})
        assert fm.knockdown_factor('shear', 1e9) == 1.0

    def test_override_does_not_mutate_canonical_table(self):
        """Constructing and using an override leaves the module table untouched."""
        before = dict(_FATIGUE_B_QI)
        FatigueModel(b={'tension': 0.3}).knockdown_factor('tension', 10)
        assert _FATIGUE_B_QI == before

    def test_override_cannot_register_new_mode(self):
        """Unknown modes are rejected even if present in the override dict."""
        fm = FatigueModel(b={'peel': 0.1})
        with pytest.raises(ValueError, match="Unknown fatigue mode 'peel'"):
            fm.knockdown_factor('peel', 10)

    def test_unknown_mode_message_lists_valid_modes(self):
        """The error message enumerates the accepted modes."""
        with pytest.raises(ValueError) as exc:
            FatigueModel().knockdown_factor('bending', 10)
        for mode in MODES:
            assert mode in str(exc.value)


class TestStressRatioAndValidation:
    """``R`` is informational today; ``cycles`` must be finite and >= 1."""

    @pytest.mark.parametrize("R", [-1.0, 0.0, 0.1, 0.5, 10.0])
    def test_finite_R_does_not_change_result(self, R):
        """Any finite stress ratio yields the same factor as the default R=0.1."""
        fm = FatigueModel()
        assert fm.knockdown_factor('compression', 1e6, R=R) == fm.knockdown_factor('compression', 1e6)

    @pytest.mark.parametrize("cycles", [0, -5, 0.999])
    def test_cycles_below_one_rejected(self, cycles):
        """Zero, negative and sub-unity cycle counts are rejected."""
        with pytest.raises(ValueError, match="cycles must be a finite value >= 1"):
            FatigueModel().knockdown_factor('tension', cycles)

    def test_non_finite_R_checked_before_cycles(self):
        """A bad ``R`` is reported even when ``cycles`` is also invalid."""
        with pytest.raises(ValueError, match="R must be finite"):
            FatigueModel().knockdown_factor('tension', 0, R=math.nan)

    def test_invalid_cycles_checked_before_mode(self):
        """Cycle validation runs before the mode lookup."""
        with pytest.raises(ValueError, match="cycles"):
            FatigueModel().knockdown_factor('not_a_mode', 0.5)


class TestCalibrationAlias:
    """``Calibration`` re-exports the canonical fatigue objects (#121)."""

    def test_calibration_shares_the_same_objects(self):
        """The alias is the same dict object, so edits in one place show in both."""
        assert Calibration.FATIGUE_B_QI is _FATIGUE_B_QI
        assert Calibration.FATIGUE_KD_FLOOR == _FATIGUE_KD_FLOOR
