"""The opt-in fatigue options: the Goodman mean-stress correction, the
porosity-dependent S-N slope, a per-mode stress ratio, and the durability
table of the empirical laws (library and CLI)."""

import dataclasses
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import porosity_fe_analysis
from porosity_fe import (
    FORMAT_DURABILITY,
    MATERIALS,
    EmpiricalSolver,
    FatigueModel,
    build_empirical_pipeline,
    load_results_from_json,
)
from porosity_fe.fatigue import _FATIGUE_B_QI, _goodman, _load_ratio
from porosity_fe.reporting import (
    durability_record,
    fatigue_conditions,
    serialise_durability_json,
    write_durability_json,
)

N = 1e6
_TENSION_TYPE = ('tension', 'shear', 'ilss', 'transverse_tension')


def _schema():
    path = Path(__file__).resolve().parent.parent / 'validation' / 'schemas' / \
        'porosity_results_schema.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _solver(Vp=0.03, material=None, **fatigue):
    material = material or dataclasses.replace(MATERIALS['T800_epoxy'], T_g_dry=180.0)
    kwargs = {'solver_kwargs': {'fatigue_model': FatigueModel(**fatigue)}} if fatigue else {}
    return build_empirical_pipeline(material, Vp, **kwargs)[2]


class TestGoodman:
    @pytest.mark.parametrize('mode', sorted(_FATIGUE_B_QI))
    def test_the_calibration_ratio_is_the_calibrated_curve(self, mode):
        R = 10.0 if mode == 'compression' else 0.1
        assert FatigueModel(mean_stress='goodman').knockdown_factor(mode, N, R) == \
            FatigueModel().knockdown_factor(mode, N)
        # R = None is the calibration ratio of the mode.
        assert FatigueModel(mean_stress='goodman').knockdown_factor(mode, N, None) == \
            FatigueModel().knockdown_factor(mode, N)

    def test_closed_form(self):
        """The maximum stress at r on the line through the calibrated point
        (mean m0, amplitude a0) and the static strength (1, 0)."""
        x = 1.0 - 0.10 * 6
        m0, a0 = x * 1.1 / 2, x * 0.9 / 2
        for r in (0.0, 0.3, 0.5, 0.8):
            s = FatigueModel(mean_stress='goodman').knockdown_factor('tension', N, r)
            m, a = s * (1 + r) / 2, s * (1 - r) / 2
            assert a == pytest.approx(a0 * (1 - m) / (1 - m0), rel=1e-12)

    def test_less_amplitude_less_knockdown(self):
        model = FatigueModel(mean_stress='goodman')
        values = [model.knockdown_factor('tension', N, R) for R in (0.0, 0.1, 0.5, 0.9, 1.0)]
        assert values == sorted(values) and values[-1] == 1.0
        comp = [model.knockdown_factor('compression', N, R) for R in (1e6, 10.0, 2.0, 1.0)]
        assert comp == sorted(comp) and comp[-1] == 1.0

    def test_compression_uses_the_load_magnitude_ratio(self):
        model = FatigueModel(mean_stress='goodman')
        assert model.knockdown_factor('compression', N, 2.0) == pytest.approx(
            _goodman(1.0 - 0.10 * 6, 0.5), rel=1e-15)
        assert _load_ratio('compression', 4.0) == 0.25
        assert _load_ratio('shear', 0.3) == 0.3

    @pytest.mark.parametrize('mode, R, match', [
        ('tension', -1.0, 'reversed'), ('compression', -1.0, 'reversed'),
        ('compression', 0.5, 'tension-tension cycling'),
        ('shear', 2.0, 'compression-compression cycling'),
    ])
    def test_ratios_a_mode_cannot_see(self, mode, R, match):
        with pytest.raises(ValueError, match=match):
            FatigueModel(mean_stress='goodman').knockdown_factor(mode, N, R)

    def test_no_warning_for_another_ratio(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            FatigueModel(mean_stress='goodman').knockdown_factor('tension', N, 0.5)

    def test_the_floor_is_corrected_too(self):
        with pytest.warns(UserWarning, match='clamping'):
            low = FatigueModel(mean_stress='goodman').knockdown_factor('tension', 1e12, 0.5)
        assert low == pytest.approx(_goodman(0.01, 0.5))


class TestPorositySlope:
    def test_slope_scales_with_void_content(self):
        model = FatigueModel(porosity_slope=5.0)
        assert model.knockdown_factor('tension', N, Vp=0.0) == FatigueModel().knockdown_factor(
            'tension', N)
        assert model.knockdown_factor('tension', N, Vp=0.04) == pytest.approx(
            1.0 - 0.10 * 1.2 * 6, rel=1e-12)

    def test_needs_the_void_content(self):
        with pytest.raises(ValueError, match='void content Vp'):
            FatigueModel(porosity_slope=2.0).knockdown_factor('tension', N)
        with pytest.raises(ValueError, match='percent'):
            FatigueModel(porosity_slope=2.0).knockdown_factor('tension', N, Vp=3.0)

    def test_combines_with_goodman(self):
        model = FatigueModel(mean_stress='goodman', porosity_slope=5.0)
        x = 1.0 - 0.10 * 1.2 * 6
        assert model.knockdown_factor('tension', N, 0.5, Vp=0.04) == pytest.approx(
            _goodman(x, 0.5), rel=1e-12)


class TestModelOptions:
    def test_defaults(self):
        model = FatigueModel()
        assert model.is_default and model.mean_stress is None and model.porosity_slope == 0.0
        assert model.describe() == {'b': None, 'mean_stress': None, 'porosity_slope': 0.0,
                                    'R_calibration': 0.1}
        assert not FatigueModel(b={'tension': 0.2}).is_default
        assert FatigueModel(porosity_slope=np.int64(3)).porosity_slope == 3.0

    @pytest.mark.parametrize('kwargs, match', [
        ({'mean_stress': 'walker'}, "mean_stress must be None or one of \\['goodman'\\]"),
        ({'porosity_slope': -1.0}, 'porosity_slope must be a finite number >= 0'),
        ({'porosity_slope': float('nan')}, 'porosity_slope'),
        ({'porosity_slope': True}, 'porosity_slope'),
        ({'porosity_slope': '2'}, 'porosity_slope'),
    ])
    def test_refusals(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FatigueModel(**kwargs)

    def test_the_default_model_still_ignores_R(self):
        with pytest.warns(UserWarning, match='does not change the result'):
            assert FatigueModel().knockdown_factor('tension', N, 0.5) == \
                FatigueModel().knockdown_factor('tension', N)
        # None is the calibration ratio, without a warning.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            FatigueModel().knockdown_factor('tension', N, None)


class TestSolver:
    def test_without_a_model_nothing_changes(self):
        solver = _solver()
        result = solver.get_failure_load('tension', cycles=N)
        assert result.details['fatigue_knockdown'] == FatigueModel().knockdown_factor(
            'tension', N)
        assert 'fatigue_model' not in result.details
        assert solver.fatigue_model is None

    def test_a_per_mode_stress_ratio(self):
        solver = _solver(mean_stress='goodman')
        table = solver.get_all_failure_loads(cycles=N, R={'tension': 0.5, 'compression': 2.0})
        model = FatigueModel(mean_stress='goodman')
        assert table['tension']['linear'].details['fatigue_knockdown'] == \
            model.knockdown_factor('tension', N, 0.5)
        assert table['compression']['linear'].details['fatigue_knockdown'] == \
            model.knockdown_factor('compression', N, 2.0)
        # A mode left out takes its calibration ratio.
        assert table['shear']['linear'].details['fatigue_knockdown'] == \
            FatigueModel().knockdown_factor('shear', N)
        assert table['tension']['linear'].details['fatigue_model'] == {
            'b': None, 'mean_stress': 'goodman', 'porosity_slope': 0.0,
            'R_calibration': 0.1, 'R': {'tension': 0.5, 'compression': 2.0}}

    def test_a_single_ratio_for_every_mode_is_refused_where_it_does_not_apply(self):
        with pytest.raises(ValueError, match='tension-tension cycling'):
            _solver(mean_stress='goodman').get_all_failure_loads(cycles=N, R=0.5)

    def test_the_default_model_takes_a_mapping_too(self):
        solver = _solver()
        with pytest.warns(UserWarning, match='R=0.5 does not change the result'):
            result = solver.get_failure_load('tension', cycles=N, R={'tension': 0.5})
        assert result.knockdown == solver.get_failure_load('tension', cycles=N).knockdown

    def test_porosity_slope_uses_the_specimen_average(self):
        solver = _solver(Vp=0.04, porosity_slope=5.0)
        result = solver.get_failure_load('tension', 'linear', cycles=N)
        assert result.details['fatigue_knockdown'] == pytest.approx(1.0 - 0.10 * 1.2 * 6)

    def test_type_check(self):
        mesh = build_empirical_pipeline(MATERIALS['T800_epoxy'], 0.02)[1]
        with pytest.raises(TypeError, match='fatigue_model must be a FatigueModel'):
            EmpiricalSolver(mesh, MATERIALS['T800_epoxy'], fatigue_model={'b': 0.1})


class TestAllowable:
    def test_porosity_slope_lowers_the_allowable(self):
        plain = _solver().allowable_porosity(0.5, 'tension', 'linear', cycles=1e3)
        sloped = _solver(porosity_slope=5.0).allowable_porosity(
            0.5, 'tension', 'linear', cycles=1e3)
        assert 0.0 < sloped.Vp < plain.Vp
        # The fatigue factor is reported at the allowable.
        assert sloped.details['fatigue_knockdown'] == pytest.approx(
            FatigueModel(porosity_slope=5.0).knockdown_factor('tension', 1e3, Vp=sloped.Vp))
        # And a laminate at the allowable meets the requirement, just.
        kd = _solver(Vp=sloped.Vp, porosity_slope=5.0).get_failure_load(
            'tension', 'linear', cycles=1e3).knockdown
        assert kd == pytest.approx(0.5, abs=1e-6) and kd >= 0.5 - 1e-12

    def test_goodman_raises_the_allowable_at_a_ratio_closer_to_one(self):
        base = _solver(mean_stress='goodman').allowable_porosity(0.5, 'tension', cycles=1e3)
        mild = _solver(mean_stress='goodman').allowable_porosity(
            0.5, 'tension', cycles=1e3, R={'tension': 0.8})
        assert mild.Vp > base.Vp

    def test_unachievable_with_a_porosity_slope(self):
        result = _solver(porosity_slope=5.0).allowable_porosity(0.9, 'tension', cycles=N)
        assert not result.achievable and math.isnan(result.Vp)
        assert result.details['fatigue_knockdown'] == FatigueModel().knockdown_factor(
            'tension', N)

    def test_the_table_with_conditions(self):
        table = _solver(mean_stress='goodman').get_all_allowable_porosities(
            0.3, cycles=1e4, R={'compression': 5.0}, environment={'T': 90.0, 'M': 1.0})
        assert set(table) >= {'compression', 'tension'}
        assert table['compression']['judd_wright'].details['fatigue_model']['R'] == {
            'compression': 5.0}


class TestDurabilityRecord:
    def test_entries_are_the_products(self):
        solver = _solver(mean_stress='goodman', porosity_slope=2.0)
        record = durability_record(solver, cycles=N, R={'tension': 0.5},
                                   environment={'T': 80.0, 'M': 1.0})
        entry = record['per_mode']['shear']['judd_wright']
        assert entry['knockdown'] == pytest.approx(
            entry['static_knockdown'] * entry['fatigue_knockdown']
            * entry['environment_knockdown'], rel=1e-12)
        assert entry['failure_stress_MPa'] == pytest.approx(
            entry['knockdown'] * MATERIALS['T800_epoxy'].tau_12, rel=1e-12)
        # The environment leaves the fibre-dominated tension mode alone.
        assert record['per_mode']['tension']['linear']['environment_knockdown'] == 1.0
        assert entry['environment_knockdown'] < 1.0
        lowest = min((e['knockdown'], m, law) for m, row in record['per_mode'].items()
                     for law, e in row.items())
        assert (record['governing']['knockdown'], record['governing']['mode'],
                record['governing']['model']) == lowest
        assert record['conditions'] == {
            'cycles': N, 'R': {'tension': 0.5}, 'mean_stress': 'goodman',
            'porosity_slope': 2.0, 'T_C': 80.0, 'M_wt_pct': 1.0, 'T_g_dry_C': 180.0}
        assert 'screening-level' in record['basis']

    def test_without_conditions_every_factor_is_one(self):
        record = durability_record(_solver())
        for row in record['per_mode'].values():
            for entry in row.values():
                assert entry['fatigue_knockdown'] == entry['environment_knockdown'] == 1.0
                assert entry['knockdown'] == entry['static_knockdown']
        assert fatigue_conditions(_solver(material=MATERIALS['T800_epoxy'])) == {
            'cycles': None, 'R': None, 'mean_stress': None, 'porosity_slope': 0.0,
            'T_C': None, 'M_wt_pct': None, 'T_g_dry_C': None}

    def test_json_envelope_validates(self, tmp_path):
        import jsonschema
        record = durability_record(_solver(), cycles=N, environment={'T': 60.0, 'M': 0.5})
        path = tmp_path / 'd.json'
        write_durability_json(path, record, {'material': 'T800_epoxy', 'n_plies': 24},
                              void_volume_fraction=0.03)
        doc = load_results_from_json(path)
        jsonschema.validate(instance=doc, schema=_schema())
        assert doc['format'] == FORMAT_DURABILITY == 'porosity-fe.durability'
        assert doc['void_volume_fraction'] == 0.03
        assert doc['durability']['conditions']['cycles'] == N
        assert doc['units']['failure_stress_MPa'] == 'MPa'
        assert doc['units']['ply_angles'] == 'degrees'
        bare = json.loads(serialise_durability_json(record))
        assert 'laminate' not in bare and 'void_volume_fraction' not in bare


class TestCLI:
    _CONFIGS = {'uniform_spherical': {'distribution': 'uniform', 'void_shape': 'spherical'}}

    def _main(self, tmp_path, monkeypatch, *argv):
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS', self._CONFIGS)
        return porosity_fe_analysis.main(
            ['--vp', '0.02', '0.04', '--output-dir', str(tmp_path), *argv])

    def test_one_table_per_level(self, tmp_path, monkeypatch, capsys):
        import jsonschema
        assert self._main(tmp_path, monkeypatch, '--cycles', '1e5', '--mean-stress', 'goodman',
                          '--stress-ratio', '0.5', '--compression-stress-ratio', '5',
                          '--fatigue-porosity-slope', '3', '--service-temperature', '90',
                          '--moisture', '1.2', '--tg-dry', '190') == 0
        out = capsys.readouterr().out
        assert 'Fatigue and environment (empirical): Vp = 2.00%' in out
        assert 'Governing: ' in out
        doc = load_results_from_json(tmp_path / 'porosity_durability_4pct.json')
        jsonschema.validate(instance=doc, schema=_schema())
        assert doc['void_volume_fraction'] == 0.04
        assert doc['durability']['conditions'] == {
            'cycles': 1e5, 'mean_stress': 'goodman', 'porosity_slope': 3.0,
            'R': {**{m: 0.5 for m in _TENSION_TYPE}, 'compression': 5.0},
            'T_C': 90.0, 'M_wt_pct': 1.2, 'T_g_dry_C': 190.0}
        material = dataclasses.replace(MATERIALS['T800_epoxy'], T_g_dry=190.0)
        solver = _solver(Vp=0.04, material=material, mean_stress='goodman', porosity_slope=3.0)
        expected = solver.get_failure_load('compression', 'judd_wright', cycles=1e5,
                                           R={'compression': 5.0},
                                           environment={'T': 90.0, 'M': 1.2})
        assert doc['durability']['per_mode']['compression']['judd_wright']['knockdown'] == \
            pytest.approx(expected.knockdown, rel=1e-12)
        assert doc['laminate']['material'] == 'T800_epoxy'
        # The sweep itself is unchanged by the options.
        sweep = load_results_from_json(tmp_path / 'porosity_analysis_results_4pct.json')
        assert sweep['uniform_spherical']['empirical']['compression']['judd_wright'][
            'knockdown'] == pytest.approx(solver.get_failure_load(
                'compression', 'judd_wright').knockdown, rel=1e-12)

    def test_moisture_alone_uses_the_reference_temperature(self, tmp_path, monkeypatch):
        assert self._main(tmp_path, monkeypatch, '--quiet', '--moisture', '1.0',
                          '--tg-dry', '180') == 0
        conditions = json.loads((tmp_path / 'porosity_durability_2pct.json').read_text())[
            'durability']['conditions']
        assert (conditions['T_C'], conditions['M_wt_pct'], conditions['cycles']) == \
            (23.0, 1.0, None)

    def test_the_allowable_holds_at_the_conditions(self, tmp_path, monkeypatch, capsys):
        import jsonschema
        assert self._main(tmp_path, monkeypatch, '--cycles', '1e6',
                          '--required-knockdown', '0.5') == 0
        out = capsys.readouterr().out
        assert 'not achievable' in out
        doc = load_results_from_json(tmp_path / 'porosity_allowable.json')
        jsonschema.validate(instance=doc, schema=_schema())
        assert doc['conditions']['cycles'] == 1e6
        tension = doc['allowable_porosity']['per_mode']['tension']['judd_wright']
        assert tension['allowable_Vp_percent'] is None and not tension['achievable']

    def test_the_allowable_without_conditions_has_no_block(self, tmp_path, monkeypatch):
        assert self._main(tmp_path, monkeypatch, '--quiet', '--required-knockdown', '0.9') == 0
        assert 'conditions' not in json.loads((tmp_path / 'porosity_allowable.json').read_text())
        assert not list(tmp_path.glob('porosity_durability_*'))

    @pytest.mark.parametrize('argv, message', [
        (['--mean-stress', 'goodman'], '--mean-stress needs --cycles'),
        (['--fatigue-porosity-slope', '2'], '--fatigue-porosity-slope needs --cycles'),
        (['--cycles', '1e5', '--stress-ratio', '0.5'], '--stress-ratio needs --mean-stress'),
        (['--cycles', '1e5', '--stress-ratio', '0.5', '--compression-stress-ratio', '5'],
         '--stress-ratio, --compression-stress-ratio need --mean-stress'),
        (['--tg-dry', '180'], '--tg-dry needs --service-temperature or --moisture'),
        (['--service-temperature', '80'], 'need a dry glass-transition temperature'),
        (['--cycles', '0.5'], '--cycles must be a finite number >= 1'),
        (['--cycles', 'nan'], '--cycles must be a finite number >= 1'),
        (['--cycles', '1e5', '--fatigue-porosity-slope', '-1'],
         '--fatigue-porosity-slope must be a finite number >= 0'),
        (['--moisture', '-1', '--tg-dry', '180'], '--moisture must be a finite number >= 0'),
        (['--cycles', '1e5', '--mean-stress', 'goodman', '--compression-stress-ratio', '0.5'],
         '--compression-stress-ratio: FatigueModel: R=0.5 is tension-tension cycling'),
        (['--cycles', '1e5', '--mean-stress', 'goodman', '--stress-ratio', '-1'],
         '--stress-ratio: FatigueModel: R=-1.0 is reversed'),
        (['--cycles', '1e5', '--mean-stress', 'walker'], "invalid choice: 'walker'"),
    ])
    def test_usage_errors(self, tmp_path, monkeypatch, capsys, argv, message):
        with pytest.raises(SystemExit) as exc:
            self._main(tmp_path, monkeypatch, *argv)
        assert exc.value.code == 2
        assert message in capsys.readouterr().err
        assert not list(tmp_path.glob('*.json'))

    @pytest.mark.parametrize('exc, rc, message', [
        (OSError('disk'), 2, 'fatigue and environment for Vp=0.02: disk'),
        (RuntimeError('boom'), 3, 'fatigue and environment for Vp=0.02 failed: boom')])
    def test_failures_keep_the_exit_code_contract(self, tmp_path, monkeypatch, capsys,
                                                  exc, rc, message):
        def boom(*args, **kwargs):
            raise exc
        monkeypatch.setattr('porosity_fe.cli._run_durability', boom)
        assert self._main(tmp_path, monkeypatch, '--quiet', '--cycles', '1e5') == rc
        assert message in capsys.readouterr().err
