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


from porosity_fe_analysis import (
    PorosityField, CompositeMesh, EmpiricalSolver,
)


_PROPERTY_TO_MODE = {
    'compression_strength': 'compression',
    'tensile_strength': 'tension',
    'transverse_tensile_strength': 'tension',
    'flexural_strength': 'compression',
    'shear_strength': 'shear',
    'ilss': 'ilss',
}


def predict_strength(dataset: Dict[str, Any], prop_key: str,
                     vp_pcts) -> list:
    """Predict normalized strength at each porosity level via Judd-Wright.

    Renormalized to the dataset's baseline_porosity_pct.
    """
    mat = resolve_material(dataset)
    ply_angles = dataset['material']['ply_angles']
    mode = _PROPERTY_TO_MODE[prop_key]
    baseline_vp = dataset.get('baseline_porosity_pct', 0.0) / 100.0

    def _kd(vp_frac):
        pf = PorosityField(mat, vp_frac, distribution='uniform',
                           void_shape='spherical')
        mesh = CompositeMesh(pf, mat, nx=10, ny=5,
                             nz=mat.n_plies, ply_angles=ply_angles)
        emp = EmpiricalSolver(mesh, mat, ply_angles=ply_angles)
        return emp.get_failure_load(mode=mode, model='judd_wright')['knockdown']

    kd_base = _kd(baseline_vp) if baseline_vp > 1e-9 else 1.0
    return [float(_kd(vp / 100.0) / kd_base) for vp in vp_pcts]


from porosity_fe_analysis import (
    compute_degraded_clt_moduli,
    compute_degraded_clt_flexural_modulus,
)


def predict_modulus(dataset: Dict[str, Any], prop_key: str,
                    vp_pcts, method: str = 'mori_tanaka') -> list:
    """Predict normalized modulus at each porosity level via CLT.

    For flexural_modulus, uses D-matrix (bending) formulation.
    Otherwise uses A-matrix (membrane).
    """
    mat = resolve_material(dataset)
    ply_angles = dataset['material']['ply_angles']
    baseline_vp = dataset.get('baseline_porosity_pct', 0.0) / 100.0

    if prop_key == 'flexural_modulus':
        def compute_fn(vp):
            return compute_degraded_clt_flexural_modulus(
                mat, ply_angles, vp, method=method)['Ef_x']
    elif prop_key in ('transverse_tensile_modulus', 'shear_modulus',
                      'tensile_modulus'):
        key_map = {
            'tensile_modulus': 'Ex',
            'transverse_tensile_modulus': 'Ey',
            'shear_modulus': 'Gxy',
        }
        extract = key_map[prop_key]

        def compute_fn(vp):
            return compute_degraded_clt_moduli(
                mat, ply_angles, vp, method=method)[extract]
    else:
        raise ValueError(f"Unknown modulus property: {prop_key}")

    base_val = compute_fn(baseline_vp) if baseline_vp > 1e-9 else compute_fn(0.0)
    return [float(compute_fn(vp / 100.0) / base_val) for vp in vp_pcts]
