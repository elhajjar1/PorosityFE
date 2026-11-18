"""``fit_knockdown``: a knockdown-law coefficient from coupon data."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from scipy import stats

from porosity_fe import (
    MATERIALS,
    Calibration,
    KnockdownFit,
    ToughnessKnockdown,
    build_empirical_pipeline,
    fit_knockdown,
    propagate_uncertainty,
)
from porosity_fe.empirical import _KNOCKDOWN_LAWS
from porosity_fe.fitting import _COEFFICIENT_NAMES, _FIT_FORMS

VP = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
SIGMA_0 = 80.0
EXACT = {
    'judd_wright': (9.0, lambda Vp: np.exp(-9.0 * Vp)),
    'power_law': (7.5, lambda Vp: (1.0 - Vp) ** 7.5),
    'linear': (6.0, lambda Vp: 1.0 - 6.0 * Vp),
}
DATASETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets')


def _noisy(model='judd_wright', seed=0, scatter=0.02):
    _, law = EXACT[model]
    rng = np.random.default_rng(seed)
    return SIGMA_0 * law(VP) * np.exp(rng.normal(0.0, scatter, VP.size))


class TestExactData:
    @pytest.mark.parametrize("model", sorted(EXACT))
    def test_recovers_the_coefficient_and_the_pristine_strength(self, model):
        coef, law = EXACT[model]
        fit = fit_knockdown(VP, SIGMA_0 * law(VP), model)
        assert isinstance(fit, KnockdownFit)
        assert fit.coefficient == pytest.approx(coef, rel=1e-10)
        assert fit.pristine_strength == pytest.approx(SIGMA_0, rel=1e-10)
        assert fit.standard_error == pytest.approx(0.0, abs=1e-9)
        assert fit.interval == pytest.approx((coef, coef), rel=1e-8)
        assert fit.pristine_interval == pytest.approx((SIGMA_0, SIGMA_0), rel=1e-8)
        assert fit.rmse == pytest.approx(0.0, abs=1e-9)
        assert fit.r_squared == pytest.approx(1.0)
        assert (fit.n_points, fit.dof, fit.max_Vp) == (6, 4, 0.05)

    @pytest.mark.parametrize("model", sorted(EXACT))
    def test_through_a_given_pristine_strength(self, model):
        coef, law = EXACT[model]
        fit = fit_knockdown(VP, SIGMA_0 * law(VP), model, pristine_strength=SIGMA_0)
        assert fit.coefficient == pytest.approx(coef, rel=1e-10)
        assert fit.pristine_strength == SIGMA_0
        assert fit.pristine_interval is None
        assert fit.dof == 5

    def test_data_normalized_by_a_porous_baseline_need_the_free_intercept(self):
        """Coupons normalized by a baseline panel with 0.5 % voids: the law
        through 1 at zero porosity is biased, the free intercept is not."""
        Vp = np.array([0.005, 0.01, 0.02, 0.03, 0.04])
        kd = np.exp(-9.0 * Vp) / math.exp(-9.0 * 0.005)
        free = fit_knockdown(Vp, kd)
        assert free.coefficient == pytest.approx(9.0, rel=1e-10)
        assert free.pristine_strength == pytest.approx(math.exp(9.0 * 0.005))
        forced = fit_knockdown(Vp, kd, pristine_strength=1.0)
        assert forced.coefficient < 8.0


class TestAgainstReferenceRegressions:
    @pytest.mark.parametrize("model, x_of", [
        ('judd_wright', lambda Vp: Vp),
        ('power_law', lambda Vp: np.log(1.0 - Vp)),
    ])
    def test_logarithmic_fits_match_linregress(self, model, x_of):
        strength = _noisy(model)
        fit = fit_knockdown(VP, strength, model)
        ref = stats.linregress(x_of(VP), np.log(strength))
        sign = -1.0 if model == 'judd_wright' else 1.0
        assert fit.coefficient == pytest.approx(sign * ref.slope, rel=1e-12)
        assert fit.standard_error == pytest.approx(ref.stderr, rel=1e-9)
        assert fit.pristine_strength == pytest.approx(math.exp(ref.intercept), rel=1e-12)
        t = stats.t.ppf(0.975, 4)
        assert fit.interval == pytest.approx(
            (fit.coefficient - t * ref.stderr, fit.coefficient + t * ref.stderr), rel=1e-9)
        assert fit.pristine_interval == pytest.approx(
            (math.exp(ref.intercept - t * ref.intercept_stderr),
             math.exp(ref.intercept + t * ref.intercept_stderr)), rel=1e-9)

    def test_linear_fit_is_minus_slope_over_intercept_with_the_delta_variance(self):
        strength = _noisy('linear')
        fit = fit_knockdown(VP, strength, 'linear')
        X = np.column_stack([np.ones_like(VP), VP])
        (a, b), res, *_ = np.linalg.lstsq(X, strength, rcond=None)
        cov = float(res[0]) / 4 * np.linalg.inv(X.T @ X)      # of (a, b)
        grad = np.array([b / a ** 2, -1.0 / a])
        assert fit.coefficient == pytest.approx(-b / a, rel=1e-10)
        assert fit.pristine_strength == pytest.approx(a, rel=1e-10)
        assert fit.standard_error == pytest.approx(math.sqrt(grad @ cov @ grad), rel=1e-8)
        t = stats.t.ppf(0.975, 4)
        half = t * math.sqrt(cov[0, 0]) / a
        assert fit.pristine_interval == pytest.approx(
            (a * math.exp(-half), a * math.exp(half)), rel=1e-8)

    def test_through_origin_fit_is_the_closed_form(self):
        strength = _noisy()
        fit = fit_knockdown(VP, strength, pristine_strength=SIGMA_0)
        y = np.log(strength / SIGMA_0)
        alpha = -float(VP @ y) / float(VP @ VP)
        s2 = float(((y + alpha * VP) ** 2).sum()) / 5
        assert fit.coefficient == pytest.approx(alpha, rel=1e-12)
        assert fit.standard_error == pytest.approx(math.sqrt(s2 / float(VP @ VP)), rel=1e-10)

    def test_through_origin_linear_fit(self):
        strength = _noisy('linear')
        fit = fit_knockdown(VP, strength, 'linear', pristine_strength=SIGMA_0)
        y = 1.0 - strength / SIGMA_0
        assert fit.coefficient == pytest.approx(float(VP @ y) / float(VP @ VP), rel=1e-12)

    def test_a_wider_confidence_gives_a_wider_interval(self):
        strength = _noisy()
        narrow = fit_knockdown(VP, strength, confidence=0.90)
        wide = fit_knockdown(VP, strength, confidence=0.99)
        assert narrow.coefficient == wide.coefficient
        assert wide.interval[0] < narrow.interval[0] < narrow.interval[1] < wide.interval[1]
        assert (narrow.confidence, wide.confidence) == (0.90, 0.99)

    def test_fit_quality_is_measured_on_the_strengths(self):
        strength = _noisy()
        fit = fit_knockdown(VP, strength)
        residual = strength - fit.predict(VP)
        assert fit.rmse == pytest.approx(math.sqrt(float((residual ** 2).mean())))
        assert fit.r_squared == pytest.approx(
            1.0 - float((residual ** 2).sum()) / float(((strength - strength.mean()) ** 2).sum()))
        assert 0.9 < fit.r_squared < 1.0


class TestBundledData:
    def _block(self, dataset, prop):
        with open(os.path.join(DATASETS, f'{dataset}.json'), encoding='utf-8') as f:
            block = json.load(f)['properties'][prop]
        return (np.array(block['void_content_pct']) / 100.0,
                np.array(block['normalized_values']))

    @pytest.mark.parametrize("prop, mode, alpha, interval", [
        ('compression_strength', 'compression', 7.213, (6.988, 7.439)),
        ('tensile_strength', 'tension', 4.287, (4.169, 4.406)),
    ])
    def test_the_calibration_dataset(self, prop, mode, alpha, interval):
        """Elhajjar 2025, the QI coupons the shipped coefficients come from:
        the fit lands near the shipped alpha (6.9 and 3.9)."""
        Vp, kd = self._block('elhajjar_2025', prop)
        fit = fit_knockdown(Vp, kd, mode=mode)
        assert fit.coefficient == pytest.approx(alpha, abs=5e-4)
        assert fit.interval == pytest.approx(interval, abs=5e-4)
        assert fit.n_points == 27 and fit.r_squared > 0.99
        shipped = Calibration.JUDD_WRIGHT_ALPHA_QI[mode]
        assert abs(fit.coefficient - shipped) / shipped < 0.12

    def test_the_power_law_exponent_of_the_same_data_is_near_alpha(self):
        """For small Vp the power law is exp(-n Vp), so n is close to alpha.
        The shipped exponent (2.8) is not: see the changelog."""
        Vp, kd = self._block('elhajjar_2025', 'compression_strength')
        fit = fit_knockdown(Vp, kd, 'power_law', mode='compression')
        assert fit.coefficient == pytest.approx(6.849, abs=5e-4)
        assert fit.interval[0] > 2.0 * Calibration.POWER_LAW_N_QI['compression']


class TestUse:
    def test_as_override_feeds_the_solver(self):
        fit = fit_knockdown(VP, _noisy(), mode='ilss')
        assert fit.override_argument == 'judd_wright_alpha'
        assert fit.as_override() == {'ilss': fit.coefficient}
        solver = build_empirical_pipeline(
            MATERIALS['T700_epoxy'], 0.02, mesh_res=(4, 3, 2),
            solver_kwargs={fit.override_argument: fit.as_override()})[2]
        assert solver.get_failure_load('ilss').knockdown == pytest.approx(
            float(fit.knockdown(0.02)), rel=1e-12)

    @pytest.mark.parametrize("model, name, argument", [
        ('judd_wright', 'alpha', 'judd_wright_alpha'),
        ('power_law', 'n', 'power_law_n'),
        ('linear', 'beta', 'linear_beta'),
    ])
    def test_names(self, model, name, argument):
        fit = fit_knockdown(VP, _noisy(model), model)
        assert (fit.coefficient_name, fit.override_argument) == (name, argument)

    def test_a_coefficient_fitted_on_an_amplified_layup_is_divided_by_its_scale(self):
        fit = fit_knockdown(VP, _noisy(), mode='tension')
        assert fit.as_override(layup_scale=2.0) == {'tension': fit.coefficient / 2.0}

    def test_as_override_refusals(self):
        strength = _noisy()
        with pytest.raises(ValueError, match="no loading mode"):
            fit_knockdown(VP, strength).as_override()
        with pytest.raises(ValueError, match="at least 1"):
            fit_knockdown(VP, strength, mode='ilss').as_override(layup_scale=0.5)
        rising = fit_knockdown(VP, SIGMA_0 * np.exp(2.0 * VP), mode='ilss')
        assert rising.coefficient == pytest.approx(-2.0)
        with pytest.raises(ValueError, match="not positive"):
            rising.as_override()

    def test_knockdown_and_predict(self):
        fit = fit_knockdown(VP, SIGMA_0 * np.exp(-9.0 * VP))
        assert fit.knockdown(0.02) == pytest.approx(math.exp(-0.18))
        assert fit.predict([0.0, 0.02]) == pytest.approx(
            [SIGMA_0, SIGMA_0 * math.exp(-0.18)])
        with pytest.raises(ValueError, match="percent"):
            fit.knockdown(2.0)

    def test_cov_is_the_coefficient_cov_of_the_uncertainty_propagation(self):
        fit = fit_knockdown(VP, _noisy(), mode='ilss')
        assert fit.cov == pytest.approx(fit.standard_error / fit.coefficient)
        result = propagate_uncertainty(
            0.02, 'T700_epoxy', 'ilss', 'judd_wright', coef_cov=fit.cov,
            n_samples=50, seed=1)
        assert 0.0 < result['knockdown']['percentiles']['p50'] < 1.0

    def test_summary_is_json_ready(self):
        fit = fit_knockdown(VP, _noisy(), mode='shear')
        summary = fit.summary()
        assert json.loads(json.dumps(summary)) == summary
        assert summary['coefficient_name'] == 'alpha' and summary['mode'] == 'shear'
        assert summary['interval'] == list(fit.interval)
        fixed = fit_knockdown(VP, _noisy(), pristine_strength=SIGMA_0).summary()
        assert fixed['pristine_interval'] is None and fixed['mode'] is None

    def test_the_exponential_toughness_law_has_the_judd_wright_form(self):
        G = 0.3 * np.exp(-6.0 * VP)
        fit = fit_knockdown(VP, G)
        law = ToughnessKnockdown(b_I=fit.coefficient, b_II=fit.coefficient)
        assert float(law.factor(0.03, 'I')) == pytest.approx(float(fit.knockdown(0.03)))

    def test_constant_strengths_give_a_zero_coefficient_and_no_r_squared(self):
        fit = fit_knockdown(VP, np.full(VP.size, 50.0))
        assert fit.coefficient == pytest.approx(0.0, abs=1e-12)
        assert math.isnan(fit.r_squared)


class TestRefusals:
    @pytest.mark.parametrize("kwargs, match", [
        ({'model': 'cubic'}, "Unknown knockdown model"),
        ({'mode': 'peel'}, "Unknown loading mode"),
        ({'mode': 'mode_I_toughness'}, "Unknown loading mode"),
        ({'confidence': 0.0}, "confidence"),
        ({'confidence': 1.0}, "confidence"),
        ({'confidence': '0.9'}, "confidence"),
        ({'pristine_strength': 0.0}, "pristine_strength"),
        ({'pristine_strength': -5.0}, "pristine_strength"),
        ({'pristine_strength': math.nan}, "pristine_strength"),
        ({'pristine_strength': True}, "pristine_strength"),
        ({'pristine_strength': '80'}, "pristine_strength"),
    ])
    def test_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            fit_knockdown(VP, _noisy(), **kwargs)

    @pytest.mark.parametrize("Vp, strength, match", [
        ([1.0, 2.0, 3.0], [1.0, 0.9, 0.8], "percent"),
        ([0.0, 0.01, math.nan], [1.0, 0.9, 0.8], "void fraction"),
        ([0.0, 0.01, 0.02], [1.0, 0.9], "one length"),
        ([[0.0, 0.01], [0.02, 0.03]], [[1.0, 0.9], [0.8, 0.7]], "one-dimensional"),
        ([0.0, 0.01, 0.02], [1.0, 0.0, 0.8], "above 0"),
        ([0.0, 0.01, 0.02], [1.0, -0.9, 0.8], "above 0"),
        ([0.0, 0.01, 0.02], [1.0, math.inf, 0.8], "above 0"),
        ([0.0, 0.01], [1.0, 0.9], "at least 3 points"),
        ([0.02, 0.02, 0.02], [1.0, 0.9, 0.8], "do not vary"),
    ])
    def test_bad_data(self, Vp, strength, match):
        with pytest.raises(ValueError, match=match):
            fit_knockdown(Vp, strength)

    def test_through_a_given_strength_needs_two_points_and_a_non_zero_Vp(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            fit_knockdown([0.02], [0.9], pristine_strength=1.0)
        with pytest.raises(ValueError, match="single non-zero Vp"):
            fit_knockdown([0.0, 0.0], [1.0, 1.0], pristine_strength=1.0)
        # One repeated void content is enough through a given strength.
        fit = fit_knockdown([0.02, 0.02], [0.84, 0.82], pristine_strength=1.0)
        assert fit.coefficient == pytest.approx(-math.log(0.84 * 0.82) / 0.04)

    def test_the_power_law_needs_Vp_below_one(self):
        with pytest.raises(ValueError, match="below 1"):
            fit_knockdown([0.0, 0.5, 1.0], [1.0, 0.5, 0.1], 'power_law')

    def test_a_line_with_no_positive_intercept_cannot_be_a_linear_law(self):
        with pytest.raises(ValueError, match="no positive strength at zero porosity"):
            fit_knockdown([0.5, 0.6, 0.7], [1.0, 3.0, 5.0], 'linear')


def test_every_knockdown_law_has_a_fit_form():
    """A law added to the solver needs its regression here."""
    assert set(_FIT_FORMS) == set(_KNOCKDOWN_LAWS) == set(_COEFFICIENT_NAMES)
    for model, (_, attribute) in _KNOCKDOWN_LAWS.items():
        assert _COEFFICIENT_NAMES[model][1].upper() == attribute
