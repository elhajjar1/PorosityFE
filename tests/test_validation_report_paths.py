"""Error and edge paths of the validation runner and its report (issue #219).

``tests/test_validation_database.py`` runs the real datasets; this module
covers what they never reach: a schema that will not load, an unknown
modulus property, a toughness property whose arrays disagree in length,
and a report built from results with a dataset that failed to load, a
skipped property, failed properties (strength and toughness), no entry to
average, the default output directory and the per-paper band PNGs.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from validation import validate_all as va

DATASETS = Path(va.__file__).resolve().parent / 'datasets'


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_schema_that_will_not_load(tmp_path, monkeypatch):
    monkeypatch.setattr(va, '_SCHEMA', None)
    monkeypatch.setattr(va, '_SCHEMA_PATH', str(tmp_path / 'missing.json'))
    with pytest.raises(ValueError, match='Failed to load validation schema'):
        va._get_schema()
    (tmp_path / 'broken.json').write_text('{not json', encoding='utf-8')
    monkeypatch.setattr(va, '_SCHEMA_PATH', str(tmp_path / 'broken.json'))
    with pytest.raises(ValueError, match='JSONDecodeError'):
        va._get_schema()


def test_unknown_modulus_property():
    data = va.load_dataset(str(DATASETS / 'liu_2006.json'))
    with pytest.raises(ValueError, match='Unknown modulus property: storage_modulus'):
        va.predict_modulus(data, 'storage_modulus', [1.0])


def test_toughness_arrays_of_different_length(tmp_path):
    doc = json.loads((DATASETS / 'hakim_2017.json').read_text(encoding='utf-8'))
    prop = doc['properties']['G_Ic']
    prop['normalized_values'] = prop['normalized_values'][:-1]
    path = tmp_path / 'short.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    name, results = va._run_one_dataset(str(path))
    assert name == 'short'
    assert results['G_Ic']['error_type'] == 'ValueError'
    assert results['G_Ic']['error'] == '3 void contents for 2 values.'


def _strength(mae, n, band=False):
    entry = {'vp_pcts': [0.0, 2.0, 4.0][:n], 'experimental': [1.0, 0.9, 0.8][:n],
             'predicted': [1.0, 0.92, 0.83][:n], 'mae': mae, 'n_points': n}
    if band:
        entry['predicted_band'] = [(0.99, 1.0), (0.9, 0.94), (0.8, 0.86)][:n]
    return entry


@pytest.fixture
def results():
    """Results in every state the report has to render."""
    toughness = _strength(4.0, 3, band=True)
    toughness.update(category='toughness', stage=['initiation'] * 3,
                     precrack=['insert'] * 3)
    return {
        'good_paper': {'compressive_strength': _strength(3.0, 3, band=True),
                       'flexural_modulus': _strength(5.0, 2),
                       'tensile_strength': {'skipped': 'no model'},
                       'ilss': {'error': 'boom', 'error_type': 'RuntimeError'}},
        'tough_paper': {'G_Ic': toughness,
                        'G_IIc': {'error': 'boom', 'error_type': 'ValueError'}},
        'broken_paper': {'error': 'bad schema', 'error_type': 'ValidationError'},
    }


def test_master_report_renders_every_state(results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot, md = va.generate_master_report(results, emit_band_pngs=True)
    assert Path(plot).parent == tmp_path and Path(md).parent == tmp_path
    text = Path(md).read_text(encoding='utf-8')
    for row in ('| broken_paper | LOAD ERROR | - | - |',
                '| good_paper | tensile_strength | - | SKIPPED (no model) |',
                '| good_paper | ilss | - | ERROR |',
                '| good_paper | compressive_strength | 3 | 3.00 |',
                '| tough_paper | G_IIc | - | - | - | ERROR |',
                '| tough_paper | G_Ic | initiation | insert | 3 | 4.00 |'):
        assert row in text, row
    assert '## Overall MAE' in text and '(n=2 paper-property entries)' in text
    pngs = sorted(p.name for p in tmp_path.glob('validation_*_band.png'))
    assert pngs == ['validation_good_paper_compressive_strength_band.png',
                    'validation_tough_paper_G_Ic_band.png']


def test_report_without_any_entry_to_average(results, tmp_path):
    """Only failures: no overall MAE section, and a toughness section
    with its error row but no averages."""
    failed = copy.deepcopy(results)
    failed['good_paper'] = {'ilss': {'error': 'boom', 'error_type': 'RuntimeError'}}
    failed['tough_paper'] = {'G_IIc': {'error': 'boom', 'error_type': 'ValueError'}}
    assert va.summarize_mae(failed)['n_entries'] == 0
    assert va.summarize_toughness_mae(failed)['n_entries'] == 0
    _plot, md = va.generate_master_report(failed, output_dir=str(tmp_path))
    text = Path(md).read_text(encoding='utf-8')
    assert '## Overall MAE' not in text
    assert '## Interlaminar toughness' in text and 'Property-weighted' not in text
    assert '| tough_paper | G_IIc | - | - | - | ERROR |' in text
    assert not list(tmp_path.glob('validation_*_band.png'))
