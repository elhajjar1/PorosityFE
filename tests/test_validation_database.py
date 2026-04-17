#!/usr/bin/env python3
"""Tests for validation dataset schema and loader."""

import json
import os
import sys
import tempfile

import jsonschema
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'validation', 'schemas', 'validation_dataset_schema.json')


def test_schema_file_exists():
    assert os.path.exists(SCHEMA_PATH), f"Schema file missing at {SCHEMA_PATH}"


def test_schema_is_valid_jsonschema():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.Draft7Validator.check_schema(schema)


def test_load_dataset_function_exists():
    from validation.validate_all import load_dataset
    assert callable(load_dataset)


def test_load_dataset_rejects_invalid_json():
    from validation.validate_all import load_dataset, ValidationError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"reference": "too short"}, f)  # missing required fields
        tmppath = f.name
    try:
        with pytest.raises(ValidationError):
            load_dataset(tmppath)
    finally:
        os.unlink(tmppath)


def test_elhajjar_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'elhajjar_2025.json')
    data = load_dataset(path)
    assert 'compression_strength' in data['properties']
    assert 'tensile_strength' in data['properties']
    assert data['material']['n_plies'] == 10


def test_liu_2006_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'liu_2006.json')
    data = load_dataset(path)
    assert len(data['properties']) == 5
    assert data['material']['layup_name'] == '[0/90]3s'


def test_stamopoulos_2016_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'stamopoulos_2016.json')
    data = load_dataset(path)
    assert len(data['properties']) == 7


def test_ghiorse_1993_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'ghiorse_1993.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_olivier_1995_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'olivier_1995.json')
    data = load_dataset(path)
    assert 'tensile_strength' in data['properties']


def test_almeida_1994_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'almeida_1994.json')
    data = load_dataset(path)
    assert 'flexural_strength' in data['properties']


def test_tang_1987_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'tang_1987.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_bowles_1992_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'bowles_1992.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_jeong_1997_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'jeong_1997.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_liu_2018_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'liu_2018.json')
    data = load_dataset(path)
    assert 'tensile_strength' in data['properties']


def test_zhang_peek_2025_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'zhang_peek_2025.json')
    data = load_dataset(path)
    assert 'transverse_tensile_strength' in data['properties']


def test_wen_2023_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'wen_2023.json')
    data = load_dataset(path)
    assert 'compression_strength' in data['properties']


def test_wang_2022_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'wang_2022.json')
    data = load_dataset(path)
    assert 'tensile_strength' in data['properties']
