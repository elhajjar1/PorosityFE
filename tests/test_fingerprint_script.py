"""The FE result fingerprint and its numeric CI gate (``scripts/fingerprint.py``).

Issue #214. The CI job ``fe_fingerprint`` recomputes the gate set on one
ubuntu cell and compares it with ``scripts/fingerprint_baseline.json``; these
tests cover the comparison itself (what it accepts and what it flags), the
baseline's consistency with the case list, and one cheap case end to end on
every platform at a looser tolerance (other BLAS builds move the last
digits; the CI gate runs on one OS only).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import math
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import porosity_fe

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / 'scripts' / 'fingerprint_baseline.json'


@pytest.fixture(scope='module')
def fp():
    spec = importlib.util.spec_from_file_location('fingerprint', REPO / 'scripts' / 'fingerprint.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def baseline():
    with open(BASELINE, encoding='utf-8') as fh:
        return json.load(fh)


def _errors(fp, a, b, rtol=1e-9, noise=1e-11):
    cmp = fp._Comparison(rtol, noise)
    cmp.case('c', a, b)
    return cmp.errors


def _case(fields=None, summary=None, export=None):
    return {'fields': fields or {}, 'summary': summary or {}, 'export': export or {}}


class TestNumericDigest:
    def test_float_array(self, fp):
        a = np.arange(40.0).reshape(4, 10) - 7.0
        d = fp._numeric(a)
        assert d['__array__'] == '<f8' and d['shape'] == [4, 10]
        assert d['sum'] == a.sum() and d['abs_sum'] == np.abs(a).sum()
        assert d['max'] == 32.0 and d['min'] == -7.0
        assert d['l2'] == pytest.approx(np.linalg.norm(a), rel=1e-15)
        assert len(d['samples']) == fp.N_SAMPLES
        assert d['samples'][0] == -7.0 and d['samples'][-1] == 32.0
        json.dumps(d, allow_nan=False)

    def test_other_arrays_and_scalars(self, fp):
        ints = fp._numeric(np.array([1, 2, 3]))
        assert set(ints) == {'__array__', 'shape', 'sha'}
        assert fp._numeric(np.array([1, 2, 3]))['sha'] != fp._numeric(np.array([1, 2, 4]))['sha']
        assert fp._numeric(float('nan')) == 'nan' and fp._numeric(-math.inf) == '-inf'
        assert fp._numeric(np.float64(0.5)) == 0.5 and fp._numeric(np.int64(3)) == 3
        assert fp._numeric(np.bool_(True)) is True
        assert fp._numeric((1.0, 'x', None)) == [1.0, 'x', None]

    def test_nonfinite_entries(self, fp):
        d = fp._numeric(np.array([1.0, np.nan, 3.0]))
        assert 'nonfinite' in d and d['sum'] == 4.0 and d['max'] == 3.0
        json.dumps(d, allow_nan=False)
        e = fp._numeric(np.array([1.0, np.inf, 3.0]))
        assert e['nonfinite'] != d['nonfinite']


class TestComparison:
    def test_identical(self, fp, baseline):
        name = sorted(baseline['cases'])[0]
        case = baseline['cases'][name]
        assert _errors(fp, case, copy.deepcopy(case)) == []

    def test_scalars_are_relative(self, fp):
        a = _case({'knockdown': 0.99, 'effective_modulus': 56079.2})
        ok = _case({'knockdown': 0.99 * (1 + 3e-10), 'effective_modulus': 56079.2})
        bad = _case({'knockdown': 0.99 * (1 + 1e-8), 'effective_modulus': 56079.2})
        assert _errors(fp, a, ok) == []
        errors = _errors(fp, a, bad)
        assert len(errors) == 1 and 'c.fields.knockdown' in errors[0]

    def test_round_off_zeros_beside_their_siblings(self, fp):
        """A stress mean of 1e-12 MPa beside a maximum of 358 MPa and a
        reaction sum of 1e-9 N beside a modulus of 56 GPa are zero up to
        round-off: they may change by far more than their own size."""
        a = _case(export={'sigma_22': {'min': -209.5, 'max': 358.9, 'mean': 1.8e-12},
                          'stiffness': {'effective_modulus_MPa': 56079.2,
                                        'reaction_force_sum_N': [-3.5e-9, -6.9e-10, 3.6e-9]}})
        b = copy.deepcopy(a)
        b['export']['sigma_22']['mean'] = -2.4e-12
        b['export']['stiffness']['reaction_force_sum_N'] = [1.2e-9, 4.0e-10, -2.0e-9]
        assert _errors(fp, a, b) == []
        b['export']['sigma_22']['mean'] = 1e-6                  # a real change
        assert len(_errors(fp, a, b)) == 1

    def test_arrays(self, fp):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 8, 6)) * 100.0
        a = _case({'stress_local': fp._numeric(x)})
        y = x.copy()
        y.flat[::7] *= 1 + 1e-14                                # round-off
        assert _errors(fp, a, _case({'stress_local': fp._numeric(y)})) == []
        y = x.copy()
        y[0, 0, 0] += 1e-6 * np.abs(x).max()                    # one entry moves
        assert any('stress_local.sum' in e or 'samples[0]' in e
                   for e in _errors(fp, a, _case({'stress_local': fp._numeric(y)})))
        assert any('array layout' in e for e in
                   _errors(fp, a, _case({'stress_local': fp._numeric(x[:49])})))

    def test_structure_strings_and_nonfinite(self, fp):
        a = _case({'failure_criterion': 'hashin', 'knockdown': 'nan',
                   'mode': {'fiber_t': 0.1, 'shear': 1.8}})
        for change, where in (({'failure_criterion': 'tsai_wu'}, 'failure_criterion'),
                              ({'knockdown': 0.5}, 'knockdown'),
                              ({'mode': {'fiber_t': 0.1}}, 'keys')):
            b = copy.deepcopy(a)
            b['fields'].update(change)
            errors = _errors(fp, a, b)
            assert errors and any(where in e for e in errors), (change, errors)
        b = copy.deepcopy(a)
        b['fields']['extra'] = None
        assert any('keys' in e for e in _errors(fp, a, b))

    def test_report_flags_missing_and_new_cases(self, fp, baseline, capsys):
        names = sorted(baseline['cases'])
        cases = {n: baseline['cases'][n] for n in names[1:]}
        cases['new/case'] = baseline['cases'][names[0]]
        assert fp.report(baseline, cases, fp.RTOL, fp.NOISE) == 1
        out = capsys.readouterr().out
        assert f"MISMATCH {names[0]}: missing" in out
        assert "MISMATCH new/case: not in the baseline" in out
        assert fp.report(baseline, baseline['cases'], fp.RTOL, fp.NOISE) == 0
        assert capsys.readouterr().out.rstrip().endswith('OK')


class TestBaseline:
    def test_matches_the_case_list(self, fp, baseline):
        names = [name for name, _thunk in fp._gate_cases(porosity_fe)]
        assert len(names) == len(set(names)) == 30
        assert sorted(names) == sorted(baseline['cases'])
        assert baseline['meta']['n_cases'] == 30
        assert ('hex8i/coupon/shear/direct/hashin' not in names) and \
            ('hex8i/coupon/shear/direct/tsai_wu' in names)

    def test_meta(self, fp, baseline):
        meta = baseline['meta']
        assert meta['rtol'] == fp.RTOL and meta['noise'] == fp.NOISE
        assert meta['environment']['threads']['OPENBLAS_NUM_THREADS'] == '1'
        for case in baseline['cases'].values():
            assert {'fields', 'summary', 'export'} <= set(case)
            assert 'provenance' not in case['export']

    def test_dcb_case_end_to_end(self, fp, baseline):
        """The cheapest gate case (an 8-ply DCB, about a second) recomputed
        here gives the baseline's physical results: the load factor, the
        compliance, the governing node, the energy release rates, load
        factors and openings along the front, and the displacement field.
        At 1e-6: this test runs on every OS, Python and set of wheels of the
        matrix. The DCB's reactions and its mode II / III rates are round-off
        (zero by equilibrium and by symmetry) and other BLAS builds change
        them, so they are left to the CI gate, which runs on the baseline's
        wheels."""
        name = 'defect/dcb/dcb/tsai_wu'
        thunk = dict(fp._gate_cases(porosity_fe))[name]
        logging.disable(logging.WARNING)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res, extras = thunk()
        finally:
            logging.disable(logging.NOTSET)
        with tempfile.TemporaryDirectory() as tmp:
            digest = fp._case_digest(porosity_fe, res, extras, tmp)
        base = baseline['cases'][name]['fields']
        here = digest['fields']
        cmp = fp._Comparison(1e-6, 1e-6)
        for key in ('fracture_load_factor', 'load_point_compliance',
                    'fracture_governing_node', 'displacement'):
            cmp.walk(key, base[key], here[key])
        for key in ('G_I', 'G_T', 'lambda_c', 'opening_local', 'x', 'y', 'z'):
            cmp.walk(f'crack_front.{key}', base['crack_front'][key], here['crack_front'][key])
        assert cmp.errors == []
