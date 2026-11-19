"""The FE path against the validation datasets.

``validation.validate_all.run_all_datasets_fe`` is the opt-in FE
counterpart of ``run_all_datasets``: one FE solve per data point, reported
apart from the empirical headline. The synthetic datasets here have four
plies so each solve takes a few hundredths of a second; the headline over
the bundled datasets takes about a minute and runs with
``POROSITY_FE_VALIDATION=1`` (CI sets it on one test cell).
"""

from __future__ import annotations

import json
import os
import pickle

import pytest

from validation.validate_all import (
    _FE_PROPERTY_LOADING,
    _MODULUS_PROPS,
    _run_one_dataset_fe,
    generate_master_report,
    load_dataset,
    predict_fe,
    predict_modulus,
    predict_strength,
    run_all_datasets,
    run_all_datasets_fe,
    summarize_fe_mae,
    summarize_mae,
)

DATASETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets')
VP = [0.5, 2.0, 4.0]


def _dataset(properties=None, **material):
    """A four-ply UD T700/epoxy dataset with ``properties`` (name -> values)."""
    properties = properties or {'ilss': [1.0, 0.9, 0.8]}
    mat = {"fiber": "T700", "matrix": "epoxy", "fiber_volume_fraction": 0.6,
           "n_plies": 4, "ply_angles": [0, 0, 0, 0]}
    mat.update(material)
    return {"reference": "Synthetic FE validation dataset (2026)",
            "material": mat,
            "baseline_porosity_pct": 0.5,
            "properties": {name: {"void_content_pct": list(VP),
                                  "normalized_values": list(values)}
                           for name, values in properties.items()}}


def _write(tmp_path, name, dataset):
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(dataset), encoding='utf-8')
    return str(path)


class TestPredictFe:
    @pytest.mark.parametrize("prop", sorted(_FE_PROPERTY_LOADING))
    def test_one_at_the_baseline_and_falling_with_porosity(self, prop):
        pred = predict_fe(_dataset(), prop, VP)
        assert pred[0] == 1.0
        assert 1.0 > pred[1] > pred[2] > 0.5

    @pytest.mark.parametrize("prop, tol", [
        ('tensile_modulus', 1e-4),
        ('transverse_tensile_modulus', 1e-4),
        ('shear_modulus', 1e-4),
        ('flexural_modulus', 5e-3),
    ])
    def test_a_modulus_follows_the_laminate_theory_prediction(self, prop, tol):
        """Both use the Mori-Tanaka ply stiffness; a uniform field leaves the
        FE structure nothing else to add."""
        ds = _dataset()
        assert predict_fe(ds, prop, VP) == pytest.approx(
            predict_modulus(ds, prop, VP), abs=tol)

    def test_a_transverse_test_is_a_tension_test_of_the_turned_layup(self):
        turned = _dataset(ply_angles=[90, 90, 90, 90])
        for kind in ('modulus', 'strength'):
            assert predict_fe(_dataset(), f'transverse_tensile_{kind}', VP) == \
                predict_fe(turned, f'tensile_{kind}', VP)

    def test_the_fe_strength_falls_far_less_than_the_coupon_law(self):
        """The finding the section reports: the FE ply strengths follow the
        heuristic sqrt(stiffness retention), which loses a few percent where
        the calibrated Judd-Wright law loses tens."""
        ds = _dataset()
        fe = predict_fe(ds, 'ilss', VP)
        empirical, _ = predict_strength(ds, 'ilss', VP)
        assert 0.93 < fe[2] < 0.99
        assert empirical[2] < 0.75
        fe_ud = predict_fe(ds, 'tensile_strength', VP)
        assert fe_ud[2] > 0.99          # the fiber direction barely notices

    def test_without_interior_elements_the_whole_model_factor_is_used(self, monkeypatch):
        """A coupon mesh too coarse to have interior elements falls back to
        the first-ply-failure factor of the whole model, close to it."""
        import validation.validate_all as va
        fine = predict_fe(_dataset(), 'compression_strength', VP)
        monkeypatch.setitem(va._FE_VALIDATION_MESH, 'coupon', (6, 4))
        coarse = predict_fe(_dataset(), 'compression_strength', VP)
        assert coarse != fine
        assert coarse == pytest.approx(fine, abs=5e-3)

    def test_a_property_without_an_fe_test_raises(self):
        with pytest.raises(ValueError, match="No FE validation test"):
            predict_fe(_dataset(), 'G_Ic', VP)

    def test_every_strength_and_modulus_property_has_a_loading(self):
        schema = json.load(open(os.path.join(
            os.path.dirname(DATASETS), 'schemas', 'validation_dataset_schema.json'),
            encoding='utf-8'))
        # The property names are the alternatives of the schema's patterns.
        names = {name
                 for pattern in schema['properties']['properties']['patternProperties']
                 for name in pattern.strip('^$()').split('|')} - {'G_Ic', 'G_IIc'}
        assert names == set(_FE_PROPERTY_LOADING)
        assert _MODULUS_PROPS < set(_FE_PROPERTY_LOADING)


class TestRunOneDatasetFe:
    def test_entry_of_a_property(self, tmp_path):
        ds = _dataset({'ilss': [1.0, 0.9, 0.8], 'shear_modulus': [1.0, 0.97, 0.93]})
        name, results = _run_one_dataset_fe(_write(tmp_path, 'syn', ds))
        assert name == 'syn'
        ilss = results['ilss']
        assert set(ilss) == {'vp_pcts', 'experimental', 'predicted', 'mae',
                             'n_points', 'n_baseline_points', 'loading', 'quantity'}
        assert ilss['loading'] == 'ilss' and ilss['quantity'] == 'first_ply_failure'
        assert ilss['predicted'] == predict_fe(ds, 'ilss', VP)
        assert ilss['n_points'] == 2 and ilss['n_baseline_points'] == 1
        scored = [abs(p - e) / e * 100 for p, e in
                  zip(ilss['predicted'][1:], [0.9, 0.8], strict=True)]
        assert ilss['mae'] == pytest.approx(sum(scored) / 2)
        assert results['shear_modulus']['loading'] == 'shear'
        assert results['shear_modulus']['quantity'] == 'stiffness'

    def test_toughness_properties_are_left_out(self, tmp_path):
        ds = _dataset()
        ds['properties']['G_Ic'] = {
            "void_content_pct": [0.5, 2.0], "normalized_values": [1.0, 0.9],
            "stage": "initiation", "precrack": "insert"}
        _, results = _run_one_dataset_fe(_write(tmp_path, 'mixed', ds))
        assert set(results) == {'ilss'}

    def test_a_bad_block_is_an_entry_error(self, tmp_path):
        ds = _dataset({'ilss': [1.0, 0.9, 0.8], 'tensile_modulus': [1.0, 0.99, 0.98]})
        ds['properties']['ilss']['normalized_values'] = [1.0]
        ds['properties']['tensile_modulus'] = {
            "void_content_pct": [0.5], "normalized_values": [1.0]}
        _, results = _run_one_dataset_fe(_write(tmp_path, 'bad', ds))
        assert results['ilss']['error'] == "3 void contents for 1 values."
        assert results['ilss']['error_type'] == 'ValueError'
        assert "only baseline points" in results['tensile_modulus']['error']

    def test_an_empty_block_is_an_entry_error(self, tmp_path, monkeypatch):
        """The schema refuses an empty block; the runner still checks, since
        an empty one would give a NaN MAE."""
        import validation.validate_all as va
        ds = _dataset()
        ds['properties']['ilss'] = {"void_content_pct": [], "normalized_values": []}
        monkeypatch.setattr(va, 'load_dataset', lambda path: ds)
        _, results = _run_one_dataset_fe(str(tmp_path / 'empty.json'))
        assert results['ilss']['error'] == "no data points."

    def test_an_unreadable_dataset_is_a_load_error(self, tmp_path):
        path = tmp_path / 'truncated.json'
        path.write_text('{"reference": "trunc', encoding='utf-8')
        name, results = _run_one_dataset_fe(str(path))
        assert name == 'truncated'
        assert results['error_type'] == 'ValidationError'

    def test_the_worker_is_picklable(self):
        assert pickle.loads(pickle.dumps(_run_one_dataset_fe)) is _run_one_dataset_fe


@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory):
    """Empirical and FE results of two synthetic datasets, and their folder."""
    folder = tmp_path_factory.mktemp('fe_datasets')
    _write(folder, 'alpha', _dataset({'ilss': [1.0, 0.9, 0.8],
                                      'shear_modulus': [1.0, 0.97, 0.93]}))
    _write(folder, 'beta', _dataset({'compression_strength': [1.0, 0.88, 0.75]},
                                    ply_angles=[0, 90, 90, 0]))
    return str(folder), run_all_datasets(str(folder)), run_all_datasets_fe(str(folder))


class TestRunAllDatasetsFe:
    def test_shape_follows_the_empirical_run(self, synthetic_runs):
        _, results, fe_results = synthetic_runs
        assert list(fe_results) == list(results) == ['alpha', 'beta']
        assert set(fe_results['alpha']) == {'ilss', 'shear_modulus'}
        assert set(fe_results['beta']) == {'compression_strength'}
        for ds in fe_results.values():
            for entry in ds.values():
                assert entry['mae'] >= 0.0 and entry['n_points'] == 2

    def test_parallel_matches_serial(self, synthetic_runs):
        folder, _, fe_results = synthetic_runs
        assert run_all_datasets_fe(folder, n_jobs=2) == fe_results

    def test_summarize_mae_reads_it(self, synthetic_runs):
        _, _, fe_results = synthetic_runs
        summary = summarize_mae(fe_results)
        assert summary['n_entries'] == 3 and summary['n_points'] == 6

    def test_category_summary_beside_the_empirical_pipeline(self, synthetic_runs):
        _, results, fe_results = synthetic_runs
        summary = summarize_fe_mae(fe_results, results)
        assert summary['overall'] == summarize_mae(fe_results)
        modulus, strength = summary['modulus'], summary['strength']
        assert (modulus['n_entries'], modulus['n_points']) == (1, 2)
        assert (strength['n_entries'], strength['n_points']) == (2, 4)
        assert modulus['fe_mae'] == fe_results['alpha']['shear_modulus']['mae']
        assert modulus['reference_mae'] == results['alpha']['shear_modulus']['mae']
        assert strength['fe_mae'] == pytest.approx(
            (fe_results['alpha']['ilss']['mae']
             + fe_results['beta']['compression_strength']['mae']) / 2)
        assert strength['reference_mae'] == pytest.approx(
            (results['alpha']['ilss']['mae']
             + results['beta']['compression_strength']['mae']) / 2)

    def test_summary_of_nothing(self):
        summary = summarize_fe_mae({'only_toughness': {}}, {})
        assert summary['overall']['n_entries'] == 0
        assert summary['modulus']['n_entries'] == 0
        assert summary['strength']['n_entries'] == 0

    def test_summary_skips_failed_datasets_and_entries(self):
        fe_results = {
            'broken': {'error': 'x', 'error_type': 'ValidationError'},
            'partial': {'ilss': {'error': 'boom', 'error_type': 'ValueError'},
                        'shear_strength': {'mae': 10.0, 'n_points': 3,
                                           'loading': 'shear'}},
        }
        summary = summarize_fe_mae(fe_results, {})
        assert summary['strength']['n_entries'] == 1
        assert summary['strength']['fe_mae'] == 10.0
        # No empirical entry to compare with.
        assert summary['strength']['reference_mae'] != summary['strength']['reference_mae']


class TestReport:
    def test_the_fe_section_follows_the_others(self, synthetic_runs, tmp_path):
        _, results, fe_results = synthetic_runs
        _, md_path = generate_master_report(
            results, output_dir=str(tmp_path), fe_results=fe_results)
        text = open(md_path, encoding='utf-8').read()
        head, marker, tail = text.partition(
            '## FE path (stiffness and first-ply failure)')
        assert marker
        assert '## Overall MAE' in head and '## FE path' not in head
        summary = summarize_fe_mae(fe_results, results)
        assert (f"- Property-weighted: **"
                f"{summary['overall']['property_weighted_mae']:.2f}%** "
                f"(n=3 paper-property entries)") in tail
        assert (f"- Moduli: **{summary['modulus']['fe_mae']:.2f}%** "
                f"property-weighted over 1 entries (CLT: "
                f"{summary['modulus']['reference_mae']:.2f}%)") in tail
        assert "- Strengths: **" in tail and "(empirical Judd-Wright: " in tail
        assert "sqrt(stiffness retention)" in tail
        assert "Not in the overall MAE above." in tail
        fe, emp = fe_results['alpha']['ilss'], results['alpha']['ilss']
        assert (f"| alpha | ilss | ilss | 2 | {fe['mae']:.2f} | "
                f"{emp['mae']:.2f} |") in tail
        assert "| beta | compression_strength | compression | 2 |" in tail

    def test_the_report_is_unchanged_without_fe_results(self, synthetic_runs, tmp_path):
        _, results, fe_results = synthetic_runs
        plain_dir, fe_dir = tmp_path / 'plain', tmp_path / 'fe'
        plain_dir.mkdir()
        fe_dir.mkdir()
        _, plain = generate_master_report(results, output_dir=str(plain_dir))
        _, with_fe = generate_master_report(
            results, output_dir=str(fe_dir), fe_results=fe_results)
        plain_text = open(plain, encoding='utf-8').read()
        assert 'FE path' not in plain_text
        assert open(with_fe, encoding='utf-8').read().startswith(plain_text)

    def test_error_rows(self, synthetic_runs, tmp_path):
        _, results, fe_results = synthetic_runs
        fe_results = dict(fe_results)
        fe_results['alpha'] = dict(fe_results['alpha'],
                                   ilss={'error': 'boom', 'error_type': 'ValueError'})
        fe_results['zeta'] = {'error': 'unreadable', 'error_type': 'ValidationError'}
        _, md_path = generate_master_report(
            results, output_dir=str(tmp_path), fe_results=fe_results)
        text = open(md_path, encoding='utf-8').read()
        assert "| alpha | ilss | - | - | ERROR | - |" in text
        assert "| zeta | LOAD ERROR | - | - | - | - |" in text

    def test_no_section_for_results_without_entries(self, synthetic_runs, tmp_path):
        _, results, _ = synthetic_runs
        _, md_path = generate_master_report(
            results, output_dir=str(tmp_path), fe_results={'alpha': {}})
        assert 'FE path' not in open(md_path, encoding='utf-8').read()


class TestCli:
    def test_fe_flag_adds_the_summary_and_the_section(self, synthetic_runs, tmp_path, capsys):
        from validate_porosity_cli import main
        folder, results, fe_results = synthetic_runs
        assert main(['--datasets', folder, '--output-dir', str(tmp_path), '--fe']) == 0
        printed = capsys.readouterr().out
        summary = summarize_fe_mae(fe_results, results)
        assert "Running the FE solver against the datasets..." in printed
        assert "FE path (stiffness and first-ply failure; not in the overall MAE):" in printed
        assert (f"FE MAE (property-weighted): "
                f"{summary['overall']['property_weighted_mae']:.2f}%  "
                f"(n=3 paper-property entries)") in printed
        assert "Moduli:    FE " in printed and "Strengths: FE " in printed
        text = (tmp_path / 'validation_detail_report.md').read_text(encoding='utf-8')
        assert '## FE path (stiffness and first-ply failure)' in text

    def test_without_the_flag_nothing_changes(self, synthetic_runs, tmp_path, capsys):
        from validate_porosity_cli import main
        folder, _, _ = synthetic_runs
        assert main(['--datasets', folder, '--output-dir', str(tmp_path)]) == 0
        assert "FE" not in capsys.readouterr().out
        text = (tmp_path / 'validation_detail_report.md').read_text(encoding='utf-8')
        assert 'FE path' not in text

    def test_fe_errors_are_printed_unless_quiet(self, tmp_path, capsys):
        from validate_porosity_cli import main
        ds = _dataset({'ilss': [1.0, 0.9, 0.8], 'shear_modulus': [1.0, 0.97, 0.93]})
        ds['properties']['ilss']['normalized_values'] = [1.0]
        folder = tmp_path / 'ds'
        folder.mkdir()
        _write(folder, 'bad', ds)
        assert main(['--datasets', str(folder), '--output-dir', str(tmp_path), '--fe']) == 0
        assert "[FE ERROR] bad / ilss: 3 void contents for 1 values." \
            in capsys.readouterr().out
        assert main(['--datasets', str(folder), '--output-dir', str(tmp_path),
                     '--fe', '--quiet']) == 0
        assert "[FE ERROR]" not in capsys.readouterr().out


class TestBundledDatasets:
    def test_liu_2018_pinned(self):
        """A bundled dataset end to end: UD tension and transverse tension,
        strength and modulus (24 FE solves)."""
        path = os.path.join(DATASETS, 'liu_2018.json')
        _, fe = _run_one_dataset_fe(path)
        assert {k: round(v['mae'], 2) for k, v in fe.items()} == {
            'tensile_strength': 7.12, 'tensile_modulus': 1.03,
            'transverse_tensile_strength': 16.84,
            'transverse_tensile_modulus': 3.31}
        data = load_dataset(path)
        clt = predict_modulus(data, 'tensile_modulus',
                              data['properties']['tensile_modulus']['void_content_pct'])
        assert fe['tensile_modulus']['predicted'] == pytest.approx(clt, abs=1e-4)

    @pytest.mark.skipif(
        os.environ.get("POROSITY_FE_VALIDATION") != "1",
        reason="set POROSITY_FE_VALIDATION=1 to run the FE solver over every "
               "bundled dataset (about a minute)")
    def test_headline_pinned(self):
        """The numbers of the README and the validation docs."""
        results = run_all_datasets(DATASETS)
        fe_results = run_all_datasets_fe(DATASETS)
        summary = summarize_fe_mae(fe_results, results)
        overall = summary['overall']
        assert (overall['n_entries'], overall['n_points']) == (40, 232)
        assert round(overall['property_weighted_mae'], 2) == 16.99
        assert round(overall['point_weighted_mae'], 2) == 19.05
        assert summary['modulus']['n_entries'] == 13
        assert round(summary['modulus']['fe_mae'], 2) == 7.20
        assert round(summary['modulus']['reference_mae'], 2) == 7.27
        assert summary['strength']['n_entries'] == 27
        assert round(summary['strength']['fe_mae'], 2) == 21.71
        assert round(summary['strength']['reference_mae'], 2) == 6.04
        # Toughness-only datasets have no FE entry and no error.
        assert fe_results['hakim_2017'] == {}
        assert not any('error' in entry for ds in fe_results.values()
                       for entry in ds.values())
