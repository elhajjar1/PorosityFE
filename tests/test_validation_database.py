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
