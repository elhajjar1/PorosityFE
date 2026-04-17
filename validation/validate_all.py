#!/usr/bin/env python3
"""Master validation runner: loads all datasets and runs model predictions."""

import json
import os
from typing import Dict, Any

import jsonschema


class ValidationError(Exception):
    """Raised when a dataset fails schema validation."""
    pass


_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'schemas', 'validation_dataset_schema.json')
_SCHEMA = None


def _get_schema() -> Dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        with open(_SCHEMA_PATH, encoding='utf-8') as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def load_dataset(path: str) -> Dict[str, Any]:
    """Load and validate a validation dataset JSON file.

    Raises ValidationError if the file fails schema validation.
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    try:
        jsonschema.validate(instance=data, schema=_get_schema())
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Dataset {path} failed schema: {e.message}") from e
    return data
