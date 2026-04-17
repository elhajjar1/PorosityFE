#!/usr/bin/env python3
"""Tests for validation dataset schema and loader."""

import json
import os
import pytest
import jsonschema


SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'validation', 'schemas', 'validation_dataset_schema.json')


def test_schema_file_exists():
    assert os.path.exists(SCHEMA_PATH), f"Schema file missing at {SCHEMA_PATH}"


def test_schema_is_valid_jsonschema():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.Draft7Validator.check_schema(schema)
