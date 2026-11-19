"""A material as a dict and as a JSON file, and a custom material in the sweep."""

import dataclasses
import json

import numpy as np
import pytest

from porosity_fe import (
    FORMAT_MATERIAL,
    MATERIALS,
    MaterialProperties,
    build_empirical_pipeline,
    compare_configurations,
    load_material_from_json,
    load_results_from_json,
    save_material_to_json,
    sweep_configurations,
)

_TINY_CONFIGS = {'uniform_spherical': {'distribution': 'uniform',
                                       'void_shape': 'spherical'}}


@pytest.fixture
def custom():
    """A material that is not a preset, with every option set."""
    return dataclasses.replace(
        MATERIALS['AS4_3501_6_epoxy'], sigma_1c=1100.0, n_plies=8,
        G_Ic=0.2, G_IIc=0.6, bk_eta=2.0, sigma_3t=50.0, tsai_wu_F12=-0.3)


class TestDict:
    @pytest.mark.parametrize('name', sorted(MATERIALS))
    def test_every_preset_round_trips(self, name):
        material = MATERIALS[name]
        record = material.to_dict()
        assert MaterialProperties.from_dict(record) == material
        # JSON-ready: plain numbers and None only.
        assert MaterialProperties.from_dict(json.loads(json.dumps(record))) == material

    def test_every_field_is_written(self, custom):
        record = custom.to_dict()
        assert list(record) == [f.name for f in dataclasses.fields(MaterialProperties)]
        assert record['G_Ic'] == 0.2 and record['alpha_1'] == custom.alpha_1
        assert MATERIALS['T800_epoxy'].to_dict()['G_Ic'] is None

    def test_numpy_values_become_python_numbers(self):
        material = dataclasses.replace(MATERIALS['T800_epoxy'], E11=np.float64(150e3),
                                       n_plies=np.int64(8))
        record = material.to_dict()
        assert type(record['E11']) is float and type(record['n_plies']) is int

    def test_options_may_be_left_out_or_none(self):
        record = {k: v for k, v in MATERIALS['T800_epoxy'].to_dict().items()
                  if v is not None and k not in ('T_ref', 'M_ref', 'fiber_poisson')}
        assert MaterialProperties.from_dict(record) == MATERIALS['T800_epoxy']
        record['alpha_1'] = None
        assert MaterialProperties.from_dict(record).alpha_1 is None

    def test_whole_ply_count_as_a_float(self):
        record = MATERIALS['T800_epoxy'].to_dict()
        record['n_plies'] = 16.0
        assert MaterialProperties.from_dict(record).n_plies == 16

    @pytest.mark.parametrize('change, match', [
        ({'E_11': 1.0}, r"Unknown material field\(s\): 'E_11' \(did you mean 'E11'\?\)"),
        ({'colour': 'black'}, r"Unknown material field\(s\): 'colour'\.$"),
        ({'E11': None}, 'missing E11'),
        ({'E11': '161000'}, r"E11 must be a number, got '161000'"),
        ({'E11': True}, 'E11 must be a number'),
        ({'E11': float('nan')}, 'E11 must be a number'),
        ({'n_plies': 8.5}, 'n_plies must be a whole number'),
        ({'fiber_volume_fraction': 60}, 'not a percent'),
        ({'alpha_2': 26.0}, 'ppm'),
    ])
    def test_refusals(self, change, match):
        record = {**MATERIALS['T800_epoxy'].to_dict(), **change}
        with pytest.raises(ValueError, match=match):
            MaterialProperties.from_dict(record)

    def test_missing_fields_are_all_named(self):
        record = MATERIALS['T800_epoxy'].to_dict()
        for name in ('E11', 'tau_ilss', 'fiber_modulus'):
            del record[name]
        with pytest.raises(ValueError, match='missing E11, tau_ilss, fiber_modulus'):
            MaterialProperties.from_dict(record)

    def test_not_a_mapping(self):
        with pytest.raises(ValueError, match='mapping of its fields; got list'):
            MaterialProperties.from_dict([1, 2])


class TestJsonFile:
    def test_round_trip_with_a_name(self, tmp_path, custom):
        path = tmp_path / 'mat.json'
        save_material_to_json(custom, path, name='  AS4 variant ')
        assert load_material_from_json(path) == custom
        doc = json.loads(path.read_text(encoding='utf-8'))
        assert doc['format'] == FORMAT_MATERIAL == 'porosity-fe.material'
        assert doc['name'] == 'AS4 variant'
        assert doc['units']['t_ply'] == 'mm'
        assert doc['material']['G_Ic'] == 0.2
        assert 'provenance' in doc

    def test_without_a_name(self, tmp_path):
        path = tmp_path / 'mat.json'
        save_material_to_json(MATERIALS['T700_epoxy'], str(path))
        assert json.loads(path.read_text())['name'] is None
        assert load_material_from_json(str(path)) == MATERIALS['T700_epoxy']

    def test_a_bare_object_of_fields(self, tmp_path):
        path = tmp_path / 'by_hand.json'
        path.write_text(json.dumps(MATERIALS['IM7_8552_epoxy'].to_dict()))
        assert load_material_from_json(path) == MATERIALS['IM7_8552_epoxy']

    def test_not_a_results_file(self, tmp_path):
        """A material file is not a results format."""
        path = tmp_path / 'mat.json'
        save_material_to_json(MATERIALS['T800_epoxy'], path)
        with pytest.raises(ValueError, match='unknown format'):
            load_results_from_json(path)

    @pytest.mark.parametrize('content, match', [
        ('{"format": "porosity-fe.ncr", "material": {}}', "'porosity-fe.ncr' is not a material file"),
        ('[1, 2]', 'expected a JSON object'),
        ('{not json', 'not valid JSON'),
        ('{"format": "porosity-fe.material", "material": [1]}', 'mapping of its fields'),
        ('{"format": "porosity-fe.material", "name": 3, "material": {}}', "'name' must be a string"),
    ])
    def test_refusals_name_the_file(self, tmp_path, content, match):
        path = tmp_path / 'bad.json'
        path.write_text(content)
        with pytest.raises(ValueError, match=match) as exc:
            load_material_from_json(path)
        assert 'bad.json' in str(exc.value)

    def test_a_bad_value_names_the_file_and_the_field(self, tmp_path):
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps({**MATERIALS['T800_epoxy'].to_dict(), 'nu12': 0.7}))
        with pytest.raises(ValueError, match=r'bad\.json: MaterialProperties\.nu12'):
            load_material_from_json(path)

    @pytest.mark.parametrize('kwargs, error, match', [
        ({'material': {'E11': 1.0}}, TypeError, 'must be a MaterialProperties'),
        ({'name': '  '}, ValueError, 'non-empty string'),
        ({'name': 5}, ValueError, 'non-empty string'),
    ])
    def test_save_refusals(self, tmp_path, kwargs, error, match):
        args = {'material': MATERIALS['T800_epoxy'], **kwargs}
        with pytest.raises(error, match=match):
            save_material_to_json(args.pop('material'), tmp_path / 'x.json', **args)
        assert not (tmp_path / 'x.json').exists()


class TestSweepMaterial:
    def test_compare_configurations_uses_the_material(self, custom):
        results = compare_configurations(0.03, material_name='my AS4', configs=_TINY_CONFIGS,
                                         material=custom)
        expected = build_empirical_pipeline(custom, 0.03)[2].get_failure_load(
            'compression', 'judd_wright')
        got = results['uniform_spherical']['empirical']['compression']['judd_wright']
        assert got['failure_stress'] == expected.failure_stress
        preset = compare_configurations(0.03, material_name='AS4_3501_6_epoxy',
                                        configs=_TINY_CONFIGS)
        assert got['failure_stress'] != preset['uniform_spherical']['empirical'][
            'compression']['judd_wright']['failure_stress']

    def test_with_a_laminate_and_workers(self, custom):
        """The material travels with each task to a worker process."""
        serial = sweep_configurations([0.02, 0.04], material_name='mine',
                                      configs=_TINY_CONFIGS, material=custom,
                                      ply_angles=[0, 90, 90, 0], t_ply=0.2)
        parallel = sweep_configurations([0.02, 0.04], material_name='mine',
                                        configs=_TINY_CONFIGS, material=custom,
                                        ply_angles=[0, 90, 90, 0], t_ply=0.2, n_jobs=2)
        assert serial == parallel
        laminate = dataclasses.replace(custom, n_plies=4, t_ply=0.2)
        expected = build_empirical_pipeline(laminate, 0.04, ply_angles=[0, 90, 90, 0])[2]
        assert serial[0.04]['uniform_spherical']['empirical']['tension']['linear'][
            'failure_stress'] == expected.get_failure_load('tension', 'linear').failure_stress

    def test_the_name_is_only_a_label(self, custom):
        """With a material, the name need not be a preset."""
        assert compare_configurations(0.01, material_name='not a preset',
                                      configs=_TINY_CONFIGS, material=custom)

    def test_refusals(self):
        with pytest.raises(TypeError, match='must be a MaterialProperties'):
            compare_configurations(0.01, configs=_TINY_CONFIGS, material={'E11': 1})
        with pytest.raises(TypeError, match='must be a MaterialProperties'):
            sweep_configurations([0.01], configs=_TINY_CONFIGS, material='T800_epoxy')
        with pytest.raises(ValueError, match='Unknown material'):
            sweep_configurations([0.01], material_name='nope', configs=_TINY_CONFIGS)

    def test_default_tasks_are_unchanged(self):
        """A preset on the default laminate keeps its six-field tasks."""
        from porosity_fe.pipeline import _laminate_task_args
        assert _laminate_task_args('T800_epoxy', 'QI', None) == ()
        mat = MATERIALS['T800_epoxy']
        assert _laminate_task_args('x', 'QI', None, mat) == ('QI', None, mat)


class TestPipelineMeshKwargs:
    def test_plate_size(self):
        material = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)
        _, mesh, solver = build_empirical_pipeline(
            material, 0.02, mesh_res=(8, 4, 4), mesh_kwargs={'length': 80.0, 'width': 30.0})
        assert (mesh.L_x, mesh.L_y) == (80.0, 30.0)
        _, plain, plain_solver = build_empirical_pipeline(material, 0.02, mesh_res=(8, 4, 4))
        assert (plain.L_x, plain.L_y) == (50.0, 20.0)
        # The empirical failure loads do not depend on the plate.
        assert solver.get_all_failure_loads() == plain_solver.get_all_failure_loads()
