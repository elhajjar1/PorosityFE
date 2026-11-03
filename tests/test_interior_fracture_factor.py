"""The interior fracture load factor (issue #227).

``FieldResults.fracture_load_factor`` is the minimum of ``lambda_c`` over
the front; on the 3-D ENF it sits at a side-face node, where the energy
release rate is mesh dependent. ``interior_fracture_load_factor`` is the
minimum over the front nodes at least the lateral margin from every
lateral face (the rule of the interior failure index): on the ENF it is
next to the mid-width value the NASA benchmark is compared with.
Forces N, lengths mm, rates N/mm.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_fracture_specimens import _enf

from porosity_fe import MATERIALS, CompositeMesh, Delamination, FESolver, PorosityField
from porosity_fe.fe._fracture import _interior_factor
from porosity_fe.fe._load_cases import interior_point_mask, lateral_margins

REPO = Path(__file__).resolve().parents[1]
TOUGH = dict(G_Ic=0.2, G_IIc=0.6, bk_eta=2.0)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope='module')
def enf():
    logging.disable(logging.WARNING)
    try:
        mesh, mat, pf = _enf()
        res = FESolver(mesh, mat, pf).solve('enf')
    finally:
        logging.disable(logging.NOTSET)
    return mesh, res


def _coupon(delamination, *, toughness=True, nx=12, ny=8):
    base = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)
    mat = dataclasses.replace(base, **TOUGH) if toughness else base
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh(pf, mat, nx=nx, ny=ny, nz=4, delaminations=[delamination])
    return mesh, mat, pf


class TestENF:
    def test_minimum_at_the_side_face_interior_near_mid_width(self, enf):
        mesh, res = enf
        front = res.crack_front
        width = float(np.ptp(mesh.nodes[:, 1]))
        row = front.governing
        assert front.y[row] in (pytest.approx(0.0, abs=1e-9), pytest.approx(width))
        interior = res.interior_fracture_load_factor
        assert interior > res.fracture_load_factor * 1.02
        mid = int(np.argmin(np.abs(front.y - width / 2)))
        assert interior == pytest.approx(front.lambda_c[mid], rel=1e-2)
        node = res.interior_fracture_governing_node
        r = int(np.flatnonzero(front.node == node)[0])
        assert front.lambda_c[r] == interior
        _lo, _hi, _mx, my = lateral_margins(mesh)
        assert my <= front.y[r] <= width - my

    def test_reported(self, enf, tmp_path):
        import jsonschema
        mesh, res = enf
        details = res.summary().details
        assert details['interior_fracture_load_factor'] == res.interior_fracture_load_factor
        assert details['interior_fracture_governing_node'] == res.interior_fracture_governing_node
        path = tmp_path / 'enf.json'
        FESolver.export_results(res, str(path))
        doc = json.loads(path.read_text(encoding='utf-8'))
        block = doc['fracture']
        assert block['interior_fracture_load_factor'] == res.interior_fracture_load_factor
        assert block['interior_governing_node'] == res.interior_fracture_governing_node
        schema = json.loads((REPO / 'validation' / 'schemas' / 'porosity_results_schema.json')
                            .read_text(encoding='utf-8'))
        jsonschema.validate(instance=doc, schema=schema)


class TestMargin:
    STRIP = Delamination.strip(1, 15.0, 35.0)

    def test_coupon_front_excludes_the_edge_nodes(self):
        mesh, mat, pf = _coupon(self.STRIP)
        res = FESolver(mesh, mat, pf).solve('compression', applied_strain=-0.01)
        front = res.crack_front
        inside = interior_point_mask(mesh, front.x, front.y)
        assert inside.any() and (~inside).any()
        _lo, hi, _mx, my = lateral_margins(mesh)
        assert np.all(front.y[inside] >= my - 1e-9) and np.all(front.y[inside] <= hi[1] - my + 1e-9)
        assert res.interior_fracture_load_factor == pytest.approx(
            float(np.min(front.lambda_c[inside])), rel=0, abs=0)
        assert res.interior_fracture_load_factor >= res.fracture_load_factor

    def test_explicit_margins(self):
        mesh, mat, pf = _coupon(self.STRIP)
        fe = FESolver(mesh, mat, pf)
        everywhere = fe.solve('compression', applied_strain=-0.01, interior_margin=0.0)
        assert everywhere.interior_fracture_load_factor == everywhere.fracture_load_factor
        assert everywhere.interior_fracture_governing_node == everywhere.fracture_governing_node
        nowhere = fe.solve('compression', applied_strain=-0.01, interior_margin=11.0)
        assert nowhere.interior_fracture_load_factor is None
        assert nowhere.interior_fracture_governing_node is None
        assert nowhere.fracture_load_factor == everywhere.fracture_load_factor

    def test_no_toughness(self):
        mesh, mat, pf = _coupon(self.STRIP, toughness=False)
        res = FESolver(mesh, mat, pf).solve('compression', applied_strain=-0.01)
        assert res.crack_front is not None
        assert res.interior_fracture_load_factor is None
        assert res.interior_fracture_governing_node is None

    def test_no_delamination(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4, **TOUGH)
        pf = PorosityField(mat, 0.0)
        res = FESolver(CompositeMesh(pf, mat, nx=6, ny=3, nz=4), mat, pf).solve('compression')
        assert res.interior_fracture_load_factor is None
        assert res.summary().details['interior_fracture_load_factor'] is None

    def test_undriven_interior(self):
        mesh, mat, pf = _coupon(self.STRIP)
        front = FESolver(mesh, mat, pf).solve('compression', applied_strain=-0.01).crack_front
        undriven = dataclasses.replace(front, lambda_c=np.full(len(front), math.inf))
        assert _interior_factor(mesh, undriven, None) == (math.inf, None)
        assert _interior_factor(mesh, dataclasses.replace(front, lambda_c=None), None) == \
            (None, None)

    def test_point_on_the_margin_line_counts(self):
        mesh, _mat, _pf = _coupon(self.STRIP)
        lo, hi, mx, my = lateral_margins(mesh)
        x = np.array([lo[0] + mx, lo[0] + mx * (1 - 1e-6), 25.0])
        y = np.array([lo[1] + my, 10.0, hi[1] - my])
        np.testing.assert_array_equal(interior_point_mask(mesh, x, y), [True, False, True])
        with pytest.raises(ValueError, match='interior_margin'):
            lateral_margins(mesh, -1.0)
