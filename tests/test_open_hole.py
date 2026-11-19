"""Open-hole coupons: the mapped mesh, the stress concentration and the notch criteria."""

import dataclasses
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    FESolver,
    FEVisualizer,
    OpenHoleAnalysis,
    OpenHoleMesh,
    PorosityField,
    PorosityMap,
    interface_tractions,
    load_results_from_json,
    open_hole_analysis,
    orthotropic_stress_concentration,
)
from porosity_fe.porosity_map import PorosityPatch

E = 70000.0
#: An isotropic "ply" (2 layers of 0.5 mm): the plate of Heywood's formula.
ISO = dataclasses.replace(MATERIALS['T800_epoxy'], E11=E, E22=E, E33=E, G12=E / 2.6,
                          G13=E / 2.6, G23=E / 2.6, nu12=0.3, nu13=0.3, nu23=0.3,
                          n_plies=2, t_ply=0.5)
MAT4 = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)
MAT8 = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=8)
QI8 = [0, 45, -45, 90, 90, -45, 45, 0]
CROSS = [0, 90, 90, 0]


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)   # thin-element aspect ratios
        yield
    logging.disable(logging.NOTSET)


def _solve(material, layup, Vp=0.0, loading='tension', pf=None, **mesh):
    pf = PorosityField(material, Vp) if pf is None else pf
    m = CompositeMesh.for_open_hole(pf, material, ply_angles=layup, **mesh)
    return m, FESolver(m, material, pf, ply_angles=layup).solve(loading)


@pytest.fixture(scope='module')
def isotropic():
    m, r = _solve(ISO, [0, 0], length=72.0)
    return m, r, open_hole_analysis(r, m)


def _quad_areas(xy, quads):
    p = xy[quads]
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


class TestMesh:
    def test_geometry(self, isotropic):
        m, _, _ = isotropic
        g = m.open_hole_geometry
        assert isinstance(m, OpenHoleMesh) and isinstance(m, CompositeMesh)
        assert (g.hole_diameter, g.width, g.length) == (6.0, 36.0, 72.0)
        assert g.center == (36.0, 18.0) and g.radius == 3.0
        assert g.diameter_to_width == pytest.approx(1 / 6)
        xy, quads = m.in_plane_nodes, m.in_plane_quads
        areas = _quad_areas(xy, quads)
        assert np.all(areas > 0.0)                         # counter-clockwise
        # The area is the plate less the 48-gon inscribed in the hole.
        polygon = 0.5 * 48 * 9.0 * math.sin(2 * math.pi / 48)
        assert areas.sum() == pytest.approx(72.0 * 36.0 - polygon, rel=1e-10)
        r = np.hypot(xy[:48, 0] - 36.0, xy[:48, 1] - 18.0)
        np.testing.assert_allclose(r, 3.0)
        assert xy[:, 0].min() == 0.0 and xy[:, 0].max() == pytest.approx(72.0)
        assert xy[:, 1].min() == 0.0 and xy[:, 1].max() == pytest.approx(36.0)
        # No duplicated in-plane node: the blocks share the square's sides.
        assert len(np.unique(np.round(xy, 9), axis=0)) == len(xy)

    def test_extrusion_and_layup(self):
        m = CompositeMesh.for_open_hole(PorosityField(MAT8, 0.02), MAT8, ply_angles=QI8,
                                        length=36.0)
        n2, nq = len(m.in_plane_nodes), len(m.in_plane_quads)
        assert m.n_nodes == n2 * (m.nz + 1) and m.n_elements == nq * m.nz
        assert m.open_hole_geometry.n_end == 0             # length = width
        np.testing.assert_array_equal(m.element_layup, QI8)
        np.testing.assert_allclose(m.porosity, 0.02)
        assert m.n_structured_nodes == m.n_nodes and m.released_pairs.size == 0
        assert 'OpenHoleMesh(D=6 mm, W=36 mm, L=36 mm' in repr(m)

    def test_net_section_and_end_grading(self, isotropic):
        m, _, _ = isotropic
        top, bottom = m.net_section_nodes()
        np.testing.assert_allclose(m.in_plane_nodes[top, 0], 36.0)
        np.testing.assert_allclose(m.in_plane_nodes[bottom, 0], 36.0)
        assert m.in_plane_nodes[top, 1][0] == pytest.approx(21.0)
        assert m.in_plane_nodes[top, 1][-1] == pytest.approx(36.0)
        assert np.all(np.diff(m.in_plane_nodes[top, 1]) > 0.0)
        # The rings widen outward, the first a quarter of the length around.
        widths = np.diff(m.in_plane_nodes[top, 1])
        assert widths[0] == pytest.approx(0.25 * 0.5 * math.pi * 3.0 / 12, rel=1e-6)
        assert np.all(np.diff(widths) > 0.0)
        # The end blocks lengthen away from the hole.
        x = np.unique(np.round(m.in_plane_nodes[m.in_plane_nodes[:, 1] == 0.0, 0], 9))
        right = np.diff(x[x >= 54.0])
        assert right[0] == pytest.approx(3.0, rel=0.3) and np.all(np.diff(right) > 0.0)

    @pytest.mark.parametrize('kwargs, match', [
        ({'hole_diameter': 30.0}, 'too large'), ({'hole_diameter': -1.0}, 'hole_diameter'),
        ({'length': 20.0}, 'at least the width'), ({'n_circumferential': 9}, 'even'),
        ({'n_circumferential': 2}, 'n_circumferential'), ({'n_radial': 1}, 'n_radial'),
        ({'radial_growth': 0.9}, 'radial_growth'), ({'width': 0.0}, 'width')])
    def test_refusals(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CompositeMesh.for_open_hole(PorosityField(MAT4, 0.0), MAT4, **kwargs)

    def test_grid_only_features_refuse(self, isotropic):
        m, r, _ = isotropic
        with pytest.raises(ValueError, match='no mid-y grid section'):
            m.mid_y_section_indices()
        with pytest.raises(ValueError, match='no mid-y grid section'):
            m.mid_y_element_indices()
        with pytest.raises(ValueError, match='rectangular grid'):
            interface_tractions(r, m)
        with pytest.raises(ValueError, match='rectangular grid'):
            FEVisualizer.plot_mesh_3d(m)
        with pytest.raises(ValueError, match='rectangular grid'):
            FEVisualizer.plot_damage_contour(m, None)
        from porosity_fe import Delamination
        m.delaminations = (Delamination.strip(0, 10.0, 20.0),)
        with pytest.raises(ValueError, match='takes no delamination'):
            m.generate_mesh()
        m.delaminations = ()
        m.generate_mesh()


class TestStressConcentration:
    def test_isotropic_plate(self, isotropic):
        """Heywood's finite-width plate: K_T = (2 + (1 - D/W)^3) / (1 - D/W)
        over the gross stress, 3.094 for W / D = 6."""
        m, r, a = isotropic
        lam = 1 / 6
        assert a.reference_stress_concentration == pytest.approx(
            (2 + (1 - lam) ** 3) / (1 - lam), rel=1e-12)
        assert a.stress_concentration == pytest.approx(a.reference_stress_concentration,
                                                       rel=0.02)
        assert a.net_stress_concentration == pytest.approx(
            a.stress_concentration * (1 - lam))
        assert a.far_field_stress == pytest.approx(r.effective_modulus * 0.01, rel=1e-9)
        assert a.loading == 'tension'
        # The stress decays to the edge of the coupon.
        assert a.stress_ratio[0] == pytest.approx(a.stress_concentration)
        assert a.distance[0] == 0.0 and a.distance[-1] == pytest.approx(15.0)
        assert 0.9 < a.stress_ratio[-1] < 1.2
        assert a.laminate_constants['E_x'] == pytest.approx(E, rel=1e-9)

    def test_quasi_isotropic_laminate(self):
        m, r = _solve(MAT8, QI8, length=72.0)
        a = open_hole_analysis(r, m)
        assert a.laminate_constants['E_x'] == pytest.approx(a.laminate_constants['E_y'])
        assert a.reference_stress_concentration == pytest.approx(3.094, abs=0.01)
        assert a.stress_concentration == pytest.approx(a.reference_stress_concentration,
                                                       rel=0.02)

    def test_cross_ply_converges_to_lekhnitskii(self):
        """A cross-ply is orthotropic in the plane (K_T about 5.5 here): the
        peak is steep and converges from below as the mesh is refined, to the
        Lekhnitskii-Tan value; the stress a millimetre from the edge has
        converged on the default mesh."""
        values, points = [], []
        for nc, nr in ((12, 16), (16, 20), (32, 24)):
            m, r = _solve(MAT4, CROSS, length=72.0, n_circumferential=nc, n_radial=nr)
            a = open_hole_analysis(r, m)
            values.append(a.stress_concentration)
            points.append(a.point_stress_factor(1.0))
        ref = a.reference_stress_concentration
        assert values[0] < values[1] < values[2]
        assert values[2] == pytest.approx(ref, rel=0.01)
        assert values[0] > 0.9 * ref
        assert max(points) / min(points) < 1.01

    def test_compression(self):
        m, r = _solve(ISO, [0, 0], loading='compression', length=48.0)
        a = open_hole_analysis(r, m)
        assert a.loading == 'compression' and a.far_field_stress < 0.0
        assert a.stress_concentration == pytest.approx(3.05, rel=0.03)

    def test_reference_formula(self):
        assert orthotropic_stress_concentration(E, E, E / 2.6, 0.3) == pytest.approx(3.0)
        # Tan's correction runs into the infinite-plate value for a small hole.
        k = orthotropic_stress_concentration(140e3, 10e3, 5e3, 0.3, 1e-3)
        assert k == pytest.approx(orthotropic_stress_concentration(140e3, 10e3, 5e3, 0.3),
                                  rel=1e-5)
        assert orthotropic_stress_concentration(140e3, 10e3, 5e3, 0.3, 0.3) > k
        with pytest.raises(ValueError, match='diameter_to_width'):
            orthotropic_stress_concentration(E, E, E / 2.6, 0.3, 1.0)


class TestNotchedStrength:
    def test_point_and_average_stress(self, isotropic):
        _, _, a = isotropic
        ps = a.point_stress_factor(1.0)
        assert ps == pytest.approx(1.0 / np.interp(1.0, a.distance, a.stress_ratio))
        assert 1.0 / a.stress_concentration < ps < 1.0
        # Averaging over a tiny length gives the edge value; over longer
        # lengths the factor rises.
        assert a.average_stress_factor(1e-6) == pytest.approx(1.0 / a.stress_concentration,
                                                              rel=1e-4)
        assert a.average_stress_factor(1.0) < a.average_stress_factor(4.0) < 1.0
        assert a.notched_strength(500.0, d0=1.0) == pytest.approx(500.0 * ps)
        assert a.notched_strength(500.0, a0=4.0) == pytest.approx(
            500.0 * a.average_stress_factor(4.0))

    @pytest.mark.parametrize('call, match', [
        (lambda a: a.notched_strength(500.0), 'Give one of'),
        (lambda a: a.notched_strength(500.0, d0=1.0, a0=4.0), 'Give one of'),
        (lambda a: a.notched_strength(-1.0, d0=1.0), 'unnotched_strength'),
        (lambda a: a.point_stress_factor(20.0), 'past the ligament'),
        (lambda a: a.average_stress_factor(0.0), 'a0')])
    def test_refusals(self, isotropic, call, match):
        with pytest.raises(ValueError, match=match):
            call(isotropic[2])

    def test_needs_an_open_hole_solve(self, isotropic):
        m, r, _ = isotropic
        plain = CompositeMesh(PorosityField(ISO, 0.0), ISO, 6, 2, 2, ply_angles=[0, 0])
        with pytest.raises(ValueError, match='OpenHoleMesh'):
            open_hole_analysis(r, plain)
        with pytest.raises(ValueError, match='reactions'):
            open_hole_analysis(dataclasses.replace(r, reaction_forces=None), m)
        zero = dataclasses.replace(r, reaction_forces=np.zeros_like(r.reaction_forces))
        with pytest.raises(ValueError, match='no end load'):
            open_hole_analysis(zero, m)


class TestPorosity:
    def test_voids_at_the_hole_edge(self):
        """A void-rich ring at the hole (6 % within 5 mm of its centre) against
        the same average spread evenly: the softer ring sheds load, so the
        peak stress drops, but its degraded strength fails it first."""
        pmap = PorosityMap.from_patches(72.0, 36.0, 0.0,
                                        [PorosityPatch((36.0, 18.0), (5.0, 5.0), 0.06)])
        m0, even = _solve(MAT8, QI8, pf=PorosityField(MAT8, pmap.mean), length=72.0)
        m1, ring = _solve(MAT8, QI8, pf=PorosityField.from_map(MAT8, pmap), length=72.0)
        a0, a1 = open_hole_analysis(even, m0), open_hole_analysis(ring, m1)
        assert a1.stress_concentration < a0.stress_concentration
        assert a1.point_stress_factor(1.0) > a0.point_stress_factor(1.0)
        assert ring.first_ply_failure_load_factor < even.first_ply_failure_load_factor
        assert m1.porosity.max() == pytest.approx(0.06)

    def test_uniform_porosity_softens_the_coupon(self):
        m, r = _solve(MAT4, CROSS, Vp=0.04, length=48.0)
        assert r.knockdown < 1.0
        a = open_hole_analysis(r, m)
        assert a.stress_concentration > 0.0


class TestExport:
    def test_json_block_validates(self, tmp_path, isotropic):
        import jsonschema
        m, r, a = isotropic
        path = tmp_path / 'oh.json'
        FESolver.export_results(r, path, open_hole=a)
        doc = load_results_from_json(path)
        schema = json.loads((Path(__file__).resolve().parent.parent / 'validation' /
                             'schemas' / 'porosity_results_schema.json').read_text())
        jsonschema.validate(instance=doc, schema=schema)
        block = doc['open_hole']
        assert block['hole_diameter_mm'] == 6.0 and block['loading'] == 'tension'
        assert block['stress_concentration'] == pytest.approx(a.stress_concentration)
        assert len(block['distance_mm']) == len(block['stress_ratio'])
        FESolver.export_results(r, path)
        assert 'open_hole' not in load_results_from_json(path)

    def test_result_type(self, isotropic):
        assert isinstance(isotropic[2], OpenHoleAnalysis)


class TestCriteriaSummary:
    def test_notch_criteria_block(self, isotropic):
        _, _, a = isotropic
        record = a.summary(d0=1.0, a0=4.0)
        assert record['notch_criteria'] == {
            'point_stress_distance_mm': 1.0, 'point_stress_factor': a.point_stress_factor(1.0),
            'average_stress_distance_mm': 4.0,
            'average_stress_factor': a.average_stress_factor(4.0)}
        assert 'notch_criteria' not in a.summary()


class TestRandomVoidsAroundTheHole:
    def test_study_on_the_open_hole(self):
        from porosity_fe import fe_porosity_uncertainty
        study = fe_porosity_uncertainty(
            MAT4, 0.03, 'tension', cov=0.6, correlation_length=3.0, n_samples=2, seed=1,
            ply_angles=CROSS, length=36.0,
            open_hole={'n_circumferential': 8, 'n_radial': 8})
        assert 'stress_concentration' in study.outputs
        assert study.open_hole == {'hole_diameter_mm': 6.0, 'width_mm': 36.0,
                                   'length_mm': 36.0}
        assert study.field['plate_mm'] == [36.0, 36.0]
        nominal_pf = PorosityField(MAT4, 0.03)
        m = CompositeMesh.for_open_hole(nominal_pf, MAT4, ply_angles=CROSS, length=36.0,
                                        n_circumferential=8, n_radial=8)
        r = FESolver(m, MAT4, nominal_pf, ply_angles=CROSS).solve('tension')
        assert study.nominal['stress_concentration'] == pytest.approx(
            open_hole_analysis(r, m).stress_concentration, rel=1e-12)
        assert study.summary()['open_hole']['hole_diameter_mm'] == 6.0

    @pytest.mark.parametrize('kwargs, match', [
        ({'loading': 'shear'}, 'open-hole study takes'),
        ({'mesh_res': (4, 2, 4)}, 'sets its own mesh'),
        ({'open_hole': {'width': 30.0}}, "cannot set 'width'")])
    def test_refusals(self, kwargs, match):
        from porosity_fe import fe_porosity_uncertainty
        args = {'loading': 'tension', 'open_hole': {}, **kwargs}
        loading = args.pop('loading')
        with pytest.raises(ValueError, match=match):
            fe_porosity_uncertainty(MAT4, 0.03, loading, cov=0.5, correlation_length=3.0,
                                    **args)


class TestMeshOptions:
    def test_uniform_rings_and_explicit_layers(self):
        """A large hole and many rings: equal rings already start fine
        enough, so the growth is 1; nz is taken as given."""
        m = CompositeMesh.for_open_hole(PorosityField(MAT4, 0.0), MAT4, hole_diameter=21.0,
                                        n_circumferential=4, n_radial=8, nz=8, length=36.0)
        g = m.open_hole_geometry
        assert g.radial_growth == 1.0 and m.nz == 8
        top, _ = m.net_section_nodes()
        np.testing.assert_allclose(np.diff(m.in_plane_nodes[top, 1]), (18.0 - 10.5) / 8)
        assert np.all(_quad_areas(m.in_plane_nodes, m.in_plane_quads) > 0.0)

    def test_discrete_voids(self):
        from porosity_fe import VoidGeometry
        void = VoidGeometry(center=(36.0 + 4.0, 18.0, 0.36), radii=(1.0, 1.0, 0.3))
        pf = PorosityField(MAT4, 0.01, discrete_voids=[void])
        m = CompositeMesh.for_open_hole(pf, MAT4, ply_angles=CROSS, length=72.0)
        centers = m.nodes[m.elements[m.void_elements]].mean(axis=1)
        assert len(m.void_elements) > 0
        assert np.all(void.contains(centers[:, 0], centers[:, 1], centers[:, 2]))
