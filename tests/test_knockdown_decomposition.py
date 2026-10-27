"""Knockdown decomposition by defect and the opt-in onset / decomposition exports.

Delamination plan, phase 5 (decision D6 and the rest of 1.6). Checked here:

- :func:`knockdown_decomposition`: the porosity-only and
  delamination-only knockdowns against the one pristine reference, each
  equal to the knockdown of the matching plain solve; a variant equal to
  the mesh is not solved again; a mesh with one defect only; the mesh is
  left as it was; the thermal and non-finite inputs are refused;
- :meth:`FESolver.export_results` with ``onset=`` and
  ``knockdown_decomposition=``: the ``delamination_onset`` and
  ``knockdown_decomposition`` blocks are strict JSON and validate against
  the schema, and the default export carries neither;
- :meth:`FieldResults.to_vtu` with ``onset=``: the ``onset_index`` point
  field.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    Delamination,
    FESolver,
    KnockdownDecomposition,
    PorosityField,
    knockdown_decomposition,
)
from porosity_fe.fe import decomposition as decomposition_module

#: The small coupons have thin, long elements; their quality warning is noise.
pytestmark = pytest.mark.filterwarnings("ignore:Mesh has:UserWarning")

#: 8-ply AS4/3501-6 (a 1 mm laminate), as in tests/test_delamination.py.
MAT8 = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=8)
MID = 3

_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "validation", "schemas", "porosity_results_schema.json")


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _ilss(vp: float, crack: bool = True):
    """The short-beam specimen with uniform porosity ``vp`` and a mid-plane
    strip delamination from x = 1 to 3 mm (``crack=False``: none)."""
    pf = PorosityField(MAT8, vp, distribution='uniform')
    dl = [Delamination.strip(MID, 1.0, 3.0)] if crack else None
    mesh = CompositeMesh.for_ilss(pf, MAT8, ny=2, delaminations=dl)
    return mesh, pf


def _count_solves(monkeypatch) -> list:
    calls = []
    real = FESolver.solve

    def counting(self, *args, **kwargs):
        calls.append(self.mesh)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(FESolver, 'solve', counting)
    return calls


@pytest.fixture(scope='module')
def porous_delaminated():
    mesh, pf = _ilss(0.03)
    return mesh, pf, knockdown_decomposition(mesh, MAT8, pf, 'ilss')


class TestKnockdownDecomposition:
    def test_both_defects_cost_stiffness_and_combine(self, porous_delaminated):
        _, _, parts = porous_delaminated
        assert isinstance(parts, KnockdownDecomposition)
        assert parts.loading == 'ilss'
        assert 0.0 < parts.total < parts.delamination_only < parts.porosity_only < 1.0
        # Nearly multiplicative on this specimen (not a law: see the Notes).
        assert parts.total == pytest.approx(
            parts.porosity_only * parts.delamination_only, rel=0.01)
        assert parts.dominant == 'delamination'

    def test_parts_are_the_plain_solves(self, porous_delaminated):
        mesh, pf, parts = porous_delaminated
        assert parts.total == FESolver(mesh, MAT8, pf).solve('ilss').knockdown
        # The same porosity on the undelaminated specimen (its bonded
        # duplicate reproduces the plain mesh to round-off).
        plain, plain_pf = _ilss(0.03, crack=False)
        assert parts.porosity_only == pytest.approx(
            FESolver(plain, MAT8, plain_pf).solve('ilss').knockdown, rel=1e-9)
        # The same delamination without porosity.
        dry, dry_pf = _ilss(0.0)
        assert parts.delamination_only == pytest.approx(
            FESolver(dry, MAT8, dry_pf).solve('ilss').knockdown, rel=1e-12)

    def test_total_is_reused_and_mesh_left_as_is(self, monkeypatch):
        mesh, pf = _ilss(0.03)
        porosity = mesh.porosity.copy()
        delams = mesh.delaminations
        total = FESolver(mesh, MAT8, pf).solve('ilss')
        calls = _count_solves(monkeypatch)
        parts = knockdown_decomposition(mesh, MAT8, pf, 'ilss', total=total)
        assert len(calls) == 2 and all(m is not mesh for m in calls)
        assert parts.total == total.knockdown
        assert mesh.delaminations == delams
        np.testing.assert_array_equal(mesh.porosity, porosity)
        assert mesh.void_elements.size == 0

    def test_delamination_alone(self, monkeypatch):
        mesh, pf = _ilss(0.0)
        calls = _count_solves(monkeypatch)
        parts = knockdown_decomposition(mesh, MAT8, pf, 'ilss')
        assert len(calls) == 1                 # the mesh itself, nothing else
        assert parts.porosity_only == 1.0
        assert parts.delamination_only == parts.total < 0.99
        assert parts.dominant == 'delamination'

    def test_porosity_alone(self, monkeypatch):
        mesh, pf = _ilss(0.03, crack=False)
        calls = _count_solves(monkeypatch)
        parts = knockdown_decomposition(mesh, MAT8, pf, 'ilss')
        assert len(calls) == 1
        assert parts.delamination_only == 1.0
        assert parts.porosity_only == parts.total < 1.0
        assert parts.dominant == 'porosity'

    def test_neither_defect(self):
        mesh, pf = _ilss(0.0, crack=False)
        parts = knockdown_decomposition(mesh, MAT8, pf, 'ilss')
        assert (parts.total, parts.porosity_only, parts.delamination_only) == (1.0, 1.0, 1.0)
        assert parts.dominant is None

    def test_membrane_load_on_a_coupon(self):
        """Under in-plane tension (a linear analysis, no buckling) an
        embedded delamination at the mid-plane of the symmetric laminate
        costs no stiffness; a through-width one under the first ply does."""
        pf = PorosityField(MAT8, 0.03, distribution='clustered')
        for crack, costs in (
                (Delamination.rectangle(MID, (25.0, 10.0), (7.0, 4.0)), False),
                (Delamination.strip(1, 20.0, 30.0), True)):
            mesh = CompositeMesh(pf, MAT8, nx=8, ny=4, nz=8, delaminations=[crack])
            parts = knockdown_decomposition(mesh, MAT8, pf, 'tension')
            assert parts.porosity_only < 1.0
            if costs:
                assert parts.total < parts.porosity_only
                assert parts.delamination_only < 1.0 - 1e-6
                assert parts.dominant == 'porosity'
            else:
                assert parts.delamination_only == pytest.approx(1.0, abs=1e-10)
                assert parts.total == pytest.approx(parts.porosity_only, rel=1e-10)

    def test_refuses_thermal_and_non_finite_total(self):
        mesh, pf = _ilss(0.03)
        with pytest.raises(ValueError, match="no stiffness measure"):
            knockdown_decomposition(mesh, MAT8, pf, 'thermal')
        r = FESolver(mesh, MAT8, pf).solve('ilss')
        with pytest.raises(ValueError, match="no finite knockdown"):
            knockdown_decomposition(mesh, MAT8, pf, 'ilss',
                                    total=dataclasses.replace(r, knockdown=math.nan))

    def test_summary_and_dominant_tolerance(self):
        parts = KnockdownDecomposition('tension', 0.9, 0.95, 0.97)
        assert parts.summary() == {
            'loading': 'tension', 'knockdown_total': 0.9,
            'knockdown_porosity_only': 0.95, 'knockdown_delamination_only': 0.97,
            'dominant': 'porosity'}
        # Round-off of a delamination that costs nothing is not a loss.
        assert KnockdownDecomposition('x', 1.0, 1.0, 1.0 + 3e-15).dominant is None
        assert KnockdownDecomposition('x', 0.9, 0.9 + 1e-6, 0.9).dominant == 'delamination'
        assert KnockdownDecomposition('x', 0.9, math.inf, 0.9).summary()[
            'knockdown_porosity_only'] is None

    def test_variants_are_shallow_copies(self):
        mesh, _ = _ilss(0.03)
        bonded = decomposition_module._without_delamination(mesh)
        assert bonded.delaminations == () and mesh.delaminations
        assert bonded.nodes is mesh.nodes and bool(np.all(bonded.bonded_pairs()))
        dry = decomposition_module._without_porosity(mesh)
        assert not dry.porosity.any() and mesh.porosity.any()
        assert dry.delaminations == mesh.delaminations


# --------------------------------------------------------------------------
# Opt-in exports: the onset and the decomposition
# --------------------------------------------------------------------------

def _load_schema():
    with open(_SCHEMA_PATH, encoding='utf-8') as f:
        return json.load(f)


def _strict_load(path):
    """``path`` as JSON, refusing the NaN / Infinity tokens."""
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    with open(path, encoding='utf-8') as f:
        return json.load(f, parse_constant=refuse)


def _vtu_point_field(path, name: str) -> np.ndarray | None:
    piece = ET.parse(path).getroot().find('UnstructuredGrid/Piece/PointData')
    found = [da for da in piece.findall('DataArray') if da.get('Name') == name]
    if not found:
        return None
    return np.frombuffer(base64.b64decode(found[0].text.strip())[8:], '<f8')


@pytest.fixture(scope='module')
def exported(porous_delaminated):
    mesh, pf, parts = porous_delaminated
    result = FESolver(mesh, MAT8, pf).solve('ilss')
    onset = result.delamination_onset(mesh, MAT8)
    return mesh, result, onset, parts


class TestOptInExports:
    def test_default_export_has_neither_block(self, exported, tmp_path):
        _, result, _, _ = exported
        FESolver.export_results(result, tmp_path / 'r.json')
        doc = _strict_load(tmp_path / 'r.json')
        assert 'delamination_onset' not in doc
        assert 'knockdown_decomposition' not in doc

    def test_json_blocks_are_strict_and_validate(self, exported, tmp_path):
        import jsonschema
        mesh, result, onset, parts = exported
        path = tmp_path / 'r.json'
        FESolver.export_results(result, path, onset=onset,
                                knockdown_decomposition=parts)
        doc = _strict_load(path)
        jsonschema.validate(instance=doc, schema=_load_schema())
        assert doc['delamination_onset'] == json.loads(json.dumps(onset.summary()))
        assert doc['delamination_onset']['criterion'] == 'brewer_lagace'
        assert doc['knockdown_decomposition'] == parts.summary()
        # Everything else is the default export.
        base = tmp_path / 'base.json'
        FESolver.export_results(result, base)
        rest = {k: v for k, v in doc.items()
                if k not in ('delamination_onset', 'knockdown_decomposition',
                             'provenance')}
        expected = {k: v for k, v in _strict_load(base).items() if k != 'provenance'}
        assert rest == expected

    def test_onset_nulls_are_json_null(self, exported, tmp_path):
        mesh, result, onset, _ = exported
        none = dataclasses.replace(onset, evaluated=np.zeros_like(onset.evaluated))
        FESolver.export_results(result, tmp_path / 'r.json', onset=none)
        block = _strict_load(tmp_path / 'r.json')['delamination_onset']
        assert block['max_index'] is None and block['onset_load_factor'] is None
        assert block['critical_interface'] is None
        assert set(block['interface_max_index'].values()) == {None}

    def test_vtu_onset_index(self, exported, tmp_path):
        mesh, result, onset, _ = exported
        path = tmp_path / 'r.vtu'
        result.to_vtu(mesh, path, encoding='base64', nodal=False, onset=onset)
        values = _vtu_point_field(path, 'onset_index')
        assert values is not None and values.size == mesh.n_nodes
        ok = onset.evaluated
        np.testing.assert_array_equal(values[onset.node[ok]], onset.index[ok])
        # Both nodes of a duplicated pair carry the value.
        originals, duplicates = mesh.interface_node_pairs()
        np.testing.assert_array_equal(values[duplicates], values[originals])
        others = np.ones(mesh.n_nodes, dtype=bool)
        others[onset.node[ok]] = False
        others[duplicates[np.isin(originals, onset.node[ok])]] = False
        assert np.isnan(values[others]).all()
        # Without onset there is no field, and export_results passes it on.
        result.to_vtu(mesh, tmp_path / 'plain.vtu', encoding='base64', nodal=False)
        assert _vtu_point_field(tmp_path / 'plain.vtu', 'onset_index') is None
        FESolver.export_results(result, tmp_path / 'e.vtu', fmt='vtu', mesh=mesh,
                                onset=onset)
        assert b'Name="onset_index"' in (tmp_path / 'e.vtu').read_bytes()

    def test_vtu_refuses_onset_of_another_mesh(self, exported, tmp_path):
        _, result, onset, _ = exported
        small, small_pf = _ilss(0.03, crack=False)
        small = CompositeMesh(small_pf, MAT8, nx=2, ny=1, nz=8)
        r = FESolver(small, MAT8, small_pf).solve('compression')
        with pytest.raises(ValueError, match="onset has interface node"):
            r.to_vtu(small, tmp_path / 'x.vtu', onset=onset)
