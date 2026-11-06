"""Empirical toughness screening modes (delamination plan 4.3).

``'mode_I_toughness'`` and ``'mode_II_toughness'`` of
:class:`EmpiricalSolver` return the porosity knockdown of ``G_Ic`` /
``G_IIc`` from the same :class:`ToughnessKnockdown` the FE fracture path
applies at each front node:

- agreement with :func:`toughness_knockdown_factor` for every law, the
  absolute toughness ``G_c0 * KD`` in N/mm, and ``nan`` (with the knockdown
  still defined) when the material has no pristine toughness;
- the ``toughness_uncalibrated`` flag and the one-time INFO line;
- layup independence, no discrete-void SCF, the extrapolation warning, the
  arguments the modes refuse, and that no strength result or default
  sweep changes;
- the FE paths reject them, like ``'transverse_tension'``;
- ``local_sensitivities`` / ``sensitivity_fd`` and ``propagate_uncertainty``
  (with the strict-JSON writer);
- the validation runner's toughness predictions equal the modes' ratios.

Toughness in N/mm; ``Vp`` is a fraction.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import math
import os
import pickle
import warnings
from typing import get_args

import numpy as np
import pytest

import porosity_fe.toughness as toughness_module
from porosity_fe import (
    MATERIALS,
    Calibration,
    CompositeMesh,
    EmpiricalSolver,
    FELoadingMode,
    FESolver,
    LoadingMode,
    PorosityField,
    ToughnessKnockdown,
    VoidGeometry,
    build_empirical_pipeline,
    propagate_uncertainty,
    refinement_study,
    save_uq_results_to_json,
    toughness_knockdown_factor,
)
from porosity_fe.fe._load_cases import _FE_LOADINGS

#: IM7/8552 toughness (N/mm), Krueger NASA/CR-2012-217562, as in
#: tests/test_toughness.py.
TOUGHNESS = dict(G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)
MAT = dataclasses.replace(MATERIALS['T800_epoxy'], **TOUGHNESS)
NO_TOUGHNESS = MATERIALS['T800_epoxy']          # G_Ic = G_IIc = None
MODES = {'mode_I_toughness': ('I', 'G_Ic'), 'mode_II_toughness': ('II', 'G_IIc')}
STRENGTH_MODES = ('compression', 'tension', 'shear', 'ilss', 'transverse_tension')
B_I, B_II = 6.2, 14.0
RES = (4, 3, 3)


def _solver(Vp=0.02, material=MAT, **kwargs) -> EmpiricalSolver:
    ply_angles = kwargs.pop('ply_angles', 'QI')
    config = kwargs.pop('porosity_config', None)
    _pf, _mesh, emp = build_empirical_pipeline(
        material, Vp, ply_angles=ply_angles, mesh_res=RES,
        porosity_config=config, solver_kwargs=kwargs or None)
    return emp


class TestRegistration:
    def test_mode_tables(self):
        assert EmpiricalSolver.PRISTINE_TOUGHNESS_KEY == {
            'mode_I_toughness': 'G_Ic', 'mode_II_toughness': 'G_IIc'}
        # The strength table, which the CLI's --uq and the app iterate, is
        # unchanged.
        assert tuple(EmpiricalSolver.PRISTINE_STRENGTH_KEY) == STRENGTH_MODES
        assert set(get_args(LoadingMode)) == {*STRENGTH_MODES, *MODES}

    def test_not_fe_loadings(self):
        for mode in (*MODES, 'transverse_tension'):
            assert mode not in get_args(FELoadingMode)
            assert mode not in _FE_LOADINGS

    def test_no_strength_calibration_entry(self):
        """The toughness coefficients live in the toughness law, not in
        the strength tables."""
        for table in (Calibration.JUDD_WRIGHT_ALPHA_QI, Calibration.POWER_LAW_N_QI,
                      Calibration.LINEAR_BETA_QI):
            assert not set(MODES) & set(table)
        assert Calibration.TOUGHNESS_B_PLACEHOLDER == {'I': B_I, 'II': B_II}

    def test_default_law(self):
        emp = _solver()
        assert emp.toughness_knockdown == ToughnessKnockdown()
        assert _solver(toughness_knockdown=None).toughness_knockdown == \
            ToughnessKnockdown('none')

    def test_bad_law_raises_at_construction(self):
        with pytest.raises(TypeError, match="toughness_knockdown must be"):
            _solver(toughness_knockdown=3)
        with pytest.raises(ValueError, match="Unknown toughness law"):
            _solver(toughness_knockdown='quadratic')


class TestAgreementWithTheLaw:
    VPS = (0.0, 0.005, 0.02, 0.05)

    @pytest.mark.parametrize("mode", sorted(MODES))
    @pytest.mark.parametrize("Vp", VPS)
    def test_default_law(self, mode, Vp):
        fracture_mode, field = MODES[mode]
        res = _solver(Vp).get_failure_load(mode)
        expected = toughness_knockdown_factor(Vp, fracture_mode)
        assert res.knockdown == expected
        assert res.knockdown == float(np.exp(-Calibration.TOUGHNESS_B_PLACEHOLDER[fracture_mode] * Vp))
        assert res.failure_stress == getattr(MAT, field) * expected
        assert res.model == 'toughness_exponential'
        assert res.details['mode'] == mode
        assert res.details['fracture_mode'] == fracture_mode
        assert res.details['pristine_toughness'] == getattr(MAT, field)
        assert res.details['toughness_knockdown'] == ToughnessKnockdown().describe()
        assert res['critical_location'] == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("law, label", [
        (ToughnessKnockdown(b_I=4.0, b_II=10.0), 'toughness_exponential'),
        (ToughnessKnockdown('linear', b_I=8.0, b_II=20.0), 'toughness_linear'),
        (None, 'toughness_none'),
        ('none', 'toughness_none'),
    ])
    @pytest.mark.parametrize("mode", sorted(MODES))
    def test_other_laws(self, law, label, mode):
        fracture_mode, _ = MODES[mode]
        res = _solver(0.03, toughness_knockdown=law).get_failure_load(mode)
        assert res.knockdown == toughness_knockdown_factor(0.03, fracture_mode, law)
        assert res.model == label

    def test_callable_law(self):
        def half(Vp, mode):
            return 0.5 if mode == 'II' else 0.25

        emp = _solver(toughness_knockdown=half)
        assert emp.get_failure_load('mode_I_toughness').knockdown == 0.25
        res = emp.get_failure_load('mode_II_toughness')
        assert res.knockdown == 0.5 and res.model == 'toughness_half'
        assert res.details['toughness_uncalibrated'] is False

    def test_callable_law_out_of_range_raises(self):
        emp = _solver(toughness_knockdown=lambda Vp, mode: 1.5)
        with pytest.raises(ValueError, match="expected a factor in"):
            emp.get_failure_load('mode_I_toughness')

    def test_model_name_does_not_change_the_result(self):
        emp = _solver()
        ref = emp.get_failure_load('mode_II_toughness')
        for model in ('judd_wright', 'power_law', 'linear'):
            assert emp.get_failure_load('mode_II_toughness', model) == ref

    def test_specimen_average_not_the_distribution(self):
        """Like the strength modes, the result uses the specimen average;
        the per-node field follows the local porosity."""
        uniform = _solver(0.03)
        clustered = _solver(0.03, porosity_config={
            'distribution': 'clustered', 'cluster_location': 'midplane'})
        for mode in MODES:
            assert clustered.get_failure_load(mode) == uniform.get_failure_load(mode)
        clustered.apply_loading('mode_I_toughness')
        nodal = clustered.nodal_knockdown
        np.testing.assert_array_equal(
            nodal, toughness_knockdown_factor(clustered.mesh.porosity, 'I'))
        assert nodal.min() < uniform.get_failure_load('mode_I_toughness').knockdown < nodal.max()

    def test_nodal_field_from_get_failure_load(self):
        emp = _solver(0.02)
        emp.get_failure_load('mode_II_toughness')
        np.testing.assert_array_equal(
            emp.nodal_knockdown, toughness_knockdown_factor(emp.mesh.porosity, 'II'))

    def test_layup_independent(self):
        ref = {m: _solver(0.03).get_failure_load(m) for m in MODES}
        for layup in ('UD', [90, 90, 90, 90], [45, -45, -45, 45]):
            emp = _solver(0.03, ply_angles=layup)
            for mode in MODES:
                assert emp._layup_scale(mode) == 1.0
                assert mode not in emp.layup_scale
                with warnings.catch_warnings():
                    warnings.simplefilter('error')       # no layup warning
                    res = emp.get_failure_load(mode)
                assert res == ref[mode]
                assert 'layup_scale' not in res.details

    def test_no_discrete_void_scf(self):
        void = VoidGeometry(center=(12.5, 20.0 / 3.0, 2.0), radii=(3.0, 3.0, 1.5))
        emp = _solver(0.02, porosity_config={'discrete_voids': [void]})
        porosity = np.clip(emp.mesh.porosity, 0.0, 1.0)
        assert porosity.max() == 1.0                       # a node inside the void
        emp.apply_loading('mode_I_toughness')
        np.testing.assert_array_equal(
            emp.nodal_knockdown, toughness_knockdown_factor(porosity, 'I'))
        # A strength mode does fold the void's stress concentration in.
        emp.apply_loading('ilss')
        assert not np.allclose(emp.nodal_knockdown, np.exp(-10.0 * porosity))

    def test_pickles(self):
        """The sweep's worker returns the solver across processes."""
        emp = pickle.loads(pickle.dumps(_solver()))
        assert emp.get_failure_load('mode_I_toughness') == \
            _solver().get_failure_load('mode_I_toughness')


class TestNonePristineToughness:
    @pytest.mark.parametrize("mode", sorted(MODES))
    def test_knockdown_defined_absolute_value_not(self, mode):
        fracture_mode, _ = MODES[mode]
        res = _solver(0.02, material=NO_TOUGHNESS).get_failure_load(mode)
        assert res.knockdown == toughness_knockdown_factor(0.02, fracture_mode)
        assert math.isnan(res.failure_stress)
        assert res.details['pristine_toughness'] is None
        assert res.details['toughness_uncalibrated'] is True
        # The details stay strict JSON.
        json.dumps(res.details, allow_nan=False)

    def test_one_field_set(self):
        mat = dataclasses.replace(NO_TOUGHNESS, G_Ic=0.3)
        emp = _solver(0.02, material=mat)
        assert emp.get_failure_load('mode_I_toughness').failure_stress == \
            pytest.approx(0.3 * math.exp(-B_I * 0.02), rel=1e-15)
        assert math.isnan(emp.get_failure_load('mode_II_toughness').failure_stress)


class TestUncalibratedFlag:
    def test_placeholder_at_nonzero_vp(self):
        emp = _solver(0.02)
        for mode in MODES:
            assert emp.get_failure_load(mode).details['toughness_uncalibrated'] is True

    def test_zero_porosity_changes_nothing(self):
        emp = _solver(0.0)
        for mode in MODES:
            res = emp.get_failure_load(mode)
            assert res.knockdown == 1.0
            assert res.details['toughness_uncalibrated'] is False

    def test_per_mode(self):
        emp = _solver(0.02, toughness_knockdown=ToughnessKnockdown(b_I=5.0))
        assert emp.get_failure_load('mode_I_toughness').details['toughness_uncalibrated'] is False
        assert emp.get_failure_load('mode_II_toughness').details['toughness_uncalibrated'] is True
        calibrated = _solver(0.02, toughness_knockdown=ToughnessKnockdown(b_I=5.0, b_II=9.0))
        for mode in MODES:
            assert calibrated.get_failure_load(mode).details['toughness_uncalibrated'] is False

    def test_info_logged_once(self, caplog, monkeypatch):
        monkeypatch.setattr(toughness_module, '_uncalibrated_logged', False)
        caplog.set_level(logging.INFO, logger='porosity_fe_analysis')
        _solver(0.0).get_failure_load('mode_I_toughness')
        _solver(0.02, toughness_knockdown=ToughnessKnockdown(b_I=5.0)
                ).get_failure_load('mode_I_toughness')
        assert not [r for r in caplog.records if 'uncalibrated' in r.getMessage()]
        emp = _solver(0.02)
        emp.get_failure_load('mode_II_toughness')
        emp.apply_loading('mode_I_toughness')
        noted = [r for r in caplog.records if 'uncalibrated' in r.getMessage()]
        assert len(noted) == 1 and noted[0].levelno == logging.INFO


class TestRefusalsAndWarnings:
    def test_callable_model_refused(self):
        emp = _solver()
        for call in (emp.get_failure_load, emp.apply_loading):
            with pytest.raises(ValueError, match="toughness_knockdown"):
                call('mode_I_toughness', lambda Vp, mode: 1.0)

    def test_unknown_model_refused(self):
        emp = _solver()
        with pytest.raises(ValueError, match="Unknown knockdown model 'bogus'"):
            emp.get_failure_load('mode_I_toughness', 'bogus')
        with pytest.raises(ValueError, match="Unknown knockdown model 'bogus'"):
            emp.local_sensitivities('mode_I_toughness', 'bogus')

    @pytest.mark.parametrize("kwargs, match", [
        ({'environment': {'T': 80.0, 'M': 1.0}}, "hygrothermal"),
        ({'cycles': 1e6}, "S-N"),
    ])
    def test_environment_and_cycles_refused(self, kwargs, match):
        emp = _solver()
        for mode in MODES:
            for call in (emp.get_failure_load, emp.apply_loading):
                with pytest.raises(ValueError, match=match):
                    call(mode, **kwargs)

    def test_strength_override_with_a_toughness_key(self):
        with pytest.raises(ValueError, match=r"toughness_knockdown=ToughnessKnockdown"):
            _solver(judd_wright_alpha={'mode_I_toughness': 5.0})

    def test_unknown_mode_lists_both_kinds(self):
        with pytest.raises(ValueError, match="Unknown loading mode 'bogus'") as exc:
            _solver().get_failure_load('bogus')
        assert 'mode_I_toughness' in str(exc.value)
        with pytest.raises(ValueError, match="toughness mode"):
            _solver()._get_pristine_strength('mode_I_toughness')

    def test_extrapolation_warning(self):
        with pytest.warns(UserWarning, match="calibration bound"):
            _solver(0.08).get_failure_load('mode_I_toughness')
        with pytest.warns(UserWarning, match="calibration bound"):
            _solver(0.08, toughness_knockdown=ToughnessKnockdown(
                'linear', b_I=5.0, b_II=9.0)).apply_loading('mode_II_toughness')
        for law in (None, lambda Vp, mode: 1.0):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                _solver(0.08, toughness_knockdown=law).get_failure_load('mode_I_toughness')

    def test_local_peak_warning_from_apply_loading(self):
        emp = _solver(0.04, porosity_config={
            'distribution': 'clustered', 'cluster_location': 'midplane'})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            emp.get_failure_load('mode_I_toughness')    # average within the bound
        with pytest.warns(UserWarning, match="local peak"):
            emp.apply_loading('mode_I_toughness')


class TestStrengthResultsUnchanged:
    def test_default_sweep_has_only_strength_modes(self):
        table = _solver().get_all_failure_loads()
        assert tuple(table) == STRENGTH_MODES

    def test_toughness_law_does_not_touch_strength(self):
        default = _solver(0.03)
        other = _solver(0.03, toughness_knockdown=ToughnessKnockdown('linear', b_I=1.0, b_II=1.0))
        a, b = default.get_all_failure_loads(), other.get_all_failure_loads()
        for mode in STRENGTH_MODES:
            for model in a[mode]:
                assert a[mode][model] == b[mode][model]
            assert default.local_sensitivities(mode) == other.local_sensitivities(mode)


class TestFERejects:
    @pytest.mark.parametrize("mode", [*sorted(MODES), 'transverse_tension'])
    def test_solve(self, mode):
        pf = PorosityField(MAT, 0.02)
        solver = FESolver(CompositeMesh(pf, MAT, nx=2, ny=2, nz=2), MAT, pf)
        with pytest.raises(ValueError, match=f"Unknown loading '{mode}'"):
            solver.solve(mode)

    @pytest.mark.parametrize("mode", [*sorted(MODES), 'transverse_tension'])
    def test_refinement_study(self, mode):
        with pytest.raises(ValueError, match=f"Unknown loading '{mode}'"):
            refinement_study(MAT, 0.02, mode)


class TestSensitivities:
    @pytest.mark.parametrize("law", [
        ToughnessKnockdown(),
        ToughnessKnockdown(b_I=4.0, b_II=10.0),
        ToughnessKnockdown('linear', b_I=8.0, b_II=20.0),
    ])
    @pytest.mark.parametrize("mode", sorted(MODES))
    def test_analytic_matches_finite_difference(self, law, mode):
        fracture_mode, _ = MODES[mode]
        emp = _solver(0.02, toughness_knockdown=law)
        s = emp.local_sensitivities(mode)
        b = law.coefficient(fracture_mode)
        assert s['KD'] == emp.get_failure_load(mode).knockdown
        if law.law == 'exponential':
            assert s['dKD_dVp'] == pytest.approx(-b * s['KD'], rel=1e-15)
            assert s['dKD_dcoef'] == pytest.approx(-0.02 * s['KD'], rel=1e-15)
        else:
            assert s['dKD_dVp'] == -b and s['dKD_dcoef'] == -0.02
        assert s['dKD_dVp'] == pytest.approx(emp.sensitivity_fd(mode), rel=1e-6)
        assert s['dKD_dcoef'] == pytest.approx(
            emp.sensitivity_fd(mode, param='coef'), rel=1e-6)
        assert s['toughness_uncalibrated'] is law.uses_placeholder(fracture_mode)

    def test_at_a_given_vp(self):
        s = _solver(0.02).local_sensitivities('mode_II_toughness', Vp=0.04)
        assert s['KD'] == toughness_knockdown_factor(0.04, 'II')

    def test_linear_clipped(self):
        emp = _solver(0.02, toughness_knockdown=ToughnessKnockdown('linear', b_I=60.0, b_II=60.0))
        s = emp.local_sensitivities('mode_I_toughness')
        assert s == {'KD': 0.0, 'dKD_dVp': 0.0, 'dKD_dcoef': 0.0,
                     'toughness_uncalibrated': False}

    def test_none_law(self):
        emp = _solver(0.02, toughness_knockdown=None)
        s = emp.local_sensitivities('mode_I_toughness')
        assert s == {'KD': 1.0, 'dKD_dVp': 0.0, 'dKD_dcoef': 0.0,
                     'toughness_uncalibrated': False}
        assert emp.sensitivity_fd('mode_I_toughness') == 0.0
        assert emp.sensitivity_fd('mode_I_toughness', param='coef') == 0.0

    def test_callable_law(self):
        emp = _solver(0.02, toughness_knockdown=lambda Vp, mode: 1.0 - 2.0 * Vp)
        with pytest.raises(NotImplementedError, match="sensitivity_fd"):
            emp.local_sensitivities('mode_I_toughness')
        assert emp.sensitivity_fd('mode_I_toughness') == pytest.approx(-2.0, rel=1e-9)
        with pytest.raises(ValueError, match="no coefficient"):
            emp.sensitivity_fd('mode_I_toughness', param='coef')

    def test_bad_param(self):
        with pytest.raises(ValueError, match="param must be"):
            _solver().sensitivity_fd('mode_I_toughness', param='b')


class TestUncertainty:
    N = 24

    def test_nominal_and_flag(self):
        r = propagate_uncertainty(0.02, MAT, 'mode_I_toughness', n_samples=8, seed=0)
        kd = toughness_knockdown_factor(0.02, 'I')
        assert r['nominal'] == {'failure_stress': MAT.G_Ic * kd, 'knockdown': kd}
        # Nothing scatters: every draw is the nominal.
        np.testing.assert_array_equal(r['samples']['knockdown'], np.full(8, kd))
        assert r['failure_stress_units'] == 'N/mm'
        assert r['toughness_uncalibrated'] is True
        assert r['toughness_knockdown'] == ToughnessKnockdown().describe()

    def test_pristine_scatter_moves_the_toughness_not_the_knockdown(self):
        r = propagate_uncertainty(0.02, MAT, 'mode_II_toughness', covs={'G_IIc': 0.1},
                                  n_samples=self.N, seed=1, method='lhs')
        kd = toughness_knockdown_factor(0.02, 'II')
        np.testing.assert_array_equal(r['samples']['knockdown'], np.full(self.N, kd))
        assert r['failure_stress']['std'] > 0.0
        assert r['spec'] == {'G_IIc': ['lognormal', 0.1]}

    def test_coefficient_scatter(self):
        r = propagate_uncertainty(0.02, MAT, 'mode_I_toughness', coef_cov=0.2,
                                  n_samples=self.N, seed=2)
        kd = r['samples']['knockdown']
        # Each draw is exp(-b Vp) with a b scattered about the placeholder.
        b = -np.log(kd) / 0.02
        assert np.median(b) == pytest.approx(B_I, rel=0.15)
        assert b.std() > 0.0
        np.testing.assert_allclose(r['samples']['failure_stress'], MAT.G_Ic * kd, rtol=1e-15)
        assert r['toughness_uncalibrated'] is True          # centred on the placeholder
        own = propagate_uncertainty(0.02, MAT, 'mode_I_toughness', coef_cov=0.2,
                                    n_samples=self.N, seed=2,
                                    toughness_knockdown=ToughnessKnockdown(b_I=3.0))
        assert np.median(-np.log(own['samples']['knockdown']) / 0.02) == \
            pytest.approx(3.0, rel=0.15)
        assert own['toughness_uncalibrated'] is False

    def test_coefficient_scatter_needs_coefficients(self):
        for law in (None, lambda Vp, mode: 1.0):
            with pytest.raises(ValueError, match="coef_cov scatters the coefficient b"):
                propagate_uncertainty(0.02, MAT, 'mode_I_toughness', coef_cov=0.1,
                                      n_samples=4, toughness_knockdown=law)

    def test_porosity_scatter_counts_extrapolated_draws(self):
        with pytest.warns(UserWarning, match="uncertainty draws evaluated"):
            propagate_uncertainty(0.045, MAT, 'mode_I_toughness', vp_cov=0.5,
                                  n_samples=self.N, seed=3)

    def test_none_pristine(self, tmp_path):
        r = propagate_uncertainty(0.02, NO_TOUGHNESS, 'mode_II_toughness', vp_cov=0.1,
                                  coef_cov=0.1, n_samples=self.N, seed=4)
        assert r['failure_stress'] is None
        assert r['nominal']['failure_stress'] is None
        assert r['samples']['failure_stress'] is None
        assert np.isfinite(r['samples']['knockdown']).all()
        assert r['knockdown']['mean'] < 1.0
        with pytest.raises(ValueError, match="'G_IIc' is None"):
            propagate_uncertainty(0.02, NO_TOUGHNESS, 'mode_II_toughness',
                                  covs={'G_IIc': 0.1}, n_samples=4)

    def test_json(self, tmp_path):
        strength = propagate_uncertainty(0.02, MAT, 'compression', coef_cov=0.1,
                                         n_samples=8, seed=5)
        toughness = propagate_uncertainty(0.02, NO_TOUGHNESS, 'mode_I_toughness',
                                          coef_cov=0.1, n_samples=8, seed=5)
        path = tmp_path / 'uq.json'
        save_uq_results_to_json({'compression': strength}, path)
        assert json.loads(path.read_text())['units']['failure_stress'] == 'MPa'
        for include_samples in (False, True):
            save_uq_results_to_json({'compression': strength, 'mode_I_toughness': toughness},
                                    path, include_samples=include_samples)
            doc = json.loads(path.read_text(),
                             parse_constant=lambda c: pytest.fail(f"non-standard {c}"))
            assert 'N/mm' in doc['units']['failure_stress']
            assert doc['uq']['mode_I_toughness']['failure_stress'] is None

    def test_strength_mode_unchanged(self):
        kwargs = dict(covs={'sigma_1c': 0.05}, coef_cov=0.1, vp_cov=0.1, n_samples=8, seed=6)
        a = propagate_uncertainty(0.02, MAT, 'compression', **kwargs)
        b = propagate_uncertainty(0.02, MAT, 'compression', toughness_knockdown=None, **kwargs)
        np.testing.assert_array_equal(a['samples']['failure_stress'],
                                      b['samples']['failure_stress'])
        assert 'toughness_uncalibrated' not in a and 'failure_stress_units' not in a


class TestValidationRunnerAgrees:
    """The validation runner predicts the toughness datasets with
    ``toughness_knockdown_factor``; the empirical modes give the same
    normalized values (one law)."""

    # Two datasets reach 5.09 and 5.23 % voids, past the 5 % bound.
    @pytest.mark.filterwarnings("ignore:Empirical knockdown evaluated beyond calibration bound")
    def test_toughness_datasets(self):
        from validation.validate_all import (
            _TOUGHNESS_PROPERTY_TO_MODE,
            load_dataset,
            predict_toughness,
        )
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mode_for = {'G_Ic': 'mode_I_toughness', 'G_IIc': 'mode_II_toughness'}
        checked = 0
        for path in sorted(glob.glob(os.path.join(root, 'validation', 'datasets', '*.json'))):
            data = load_dataset(path)
            base = data.get('baseline_porosity_pct', 0.0) / 100.0
            for key, block in data['properties'].items():
                if key not in _TOUGHNESS_PROPERTY_TO_MODE:
                    continue
                predicted, _band = predict_toughness(data, key, block['void_content_pct'])
                mode = mode_for[key]
                kd_base = (_solver(base).get_failure_load(mode).knockdown
                           if base > 1e-9 else 1.0)
                ours = [_solver(vp / 100.0).get_failure_load(mode).knockdown / kd_base
                        for vp in block['void_content_pct']]
                assert ours == predicted, (path, key)
                checked += 1
        assert checked == 10   # 8 toughness datasets, 10 property entries
