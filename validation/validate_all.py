#!/usr/bin/env python3
"""Master validation runner: loads all datasets and runs model predictions."""

import json
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


import dataclasses
from porosity_fe_analysis import MATERIALS, MaterialProperties


_FIBER_MATRIX_TO_PRESET = {
    ('T700', 'TDE85 epoxy'): 'T700_epoxy',
    ('T700GC-12K-31E', '#2510 epoxy'): 'T700_epoxy',
    ('T700', 'epoxy'): 'T700_epoxy',
    ('HTA 24k', 'EHkF 420 epoxy'): 'T700_epoxy',
    ('IM7', '8551-7 epoxy'): 'IM7_8551_epoxy',
    ('T300', '924 epoxy'): 'T300_934_epoxy',
    ('T300', '976 epoxy'): 'T300_934_epoxy',
    ('T300', '934 epoxy'): 'T300_934_epoxy',
    ('T300', '914 epoxy'): 'T300_934_epoxy',
    ('Carbon fiber (PEEK-CF60)', 'PEEK (thermoplastic)'): 'CF_PEEK',
    ('AS4', '3501-6 epoxy'): 'T700_epoxy',
    ('AS4 fabric', '3501-6 epoxy'): 'T700_epoxy',
    ('Carbon', 'epoxy'): 'T700_epoxy',
    ('Carbon fiber', 'epoxy'): 'T700_epoxy',
}


def resolve_material(dataset: Dict[str, Any]) -> MaterialProperties:
    """Build a MaterialProperties instance from a dataset's material block.

    Selects the closest preset from MATERIALS based on fiber/matrix, then
    overrides n_plies and fiber_volume_fraction from the dataset.
    """
    m = dataset['material']
    key = (m['fiber'], m['matrix'])
    preset_name = _FIBER_MATRIX_TO_PRESET.get(key, 'T700_epoxy')
    base = MATERIALS[preset_name]
    return dataclasses.replace(
        base,
        n_plies=m['n_plies'],
        fiber_volume_fraction=m['fiber_volume_fraction'],
    )
