"""Delamination growth by node release: quasi-static and Paris-law fatigue."""

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
    Delamination,
    FESolver,
    GrowthResult,
    ParisLaw,
    PorosityField,
    PorosityMap,
    delamination_growth,
    fatigue_delamination_growth,
    load_results_from_json,
)
from porosity_fe.fe import _knockdown, decomposition
from porosity_fe.fe import growth as growth_module

#: T300/1076 with the T300/914C toughness: Krueger's DCB benchmark material
#: (NASA/TM-2008-215123), as in test_fracture_specimens.py.
T300_1076 = dict(E11=139400.0, E22=10160.0, E33=10160.0, nu12=0.30, nu13=0.30,
                 nu23=0.436, G12=4600.0, G13=4600.0, G23=3540.0)
T300_TOUGHNESS = dict(G_Ic=0.1703, G_IIc=0.4936, bk_eta=1.62)
#: IM7/8552 with the ENF benchmark's toughness (NASA/CR-2011-217305).
IM7_8552 = dict(E11=161000.0, E22=11380.0, E33=11380.0, nu12=0.32, nu13=0.32,
                nu23=0.45, G12=5200.0, G13=5200.0, G23=3900.0)
ENF_TOUGHNESS = dict(G_Ic=0.21, G_IIc=0.78, bk_eta=2.57)

#: The DCB: 3 mm thick (two 1.5 mm arms), 25 mm wide, crack 30 mm, 1 mm
#: elements along x.
DCB_H, DCB_B, DCB_A = 3.0, 25.0, 30.0


def _material(elastic, toughness, thickness, layers):
    return dataclasses.replace(MATERIALS['T800_epoxy'], **elastic, **toughness,
                               t_ply=thickness / layers, n_plies=layers)


DCB_MAT = _material(T300_1076, T300_TOUGHNESS, DCB_H, 8)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)   # plane-strain aspect ratios
        yield
    logging.disable(logging.NOTSET)


def _dcb(crack_length=DCB_A, *, pf=None, length=60.0, ny=1, mat=DCB_MAT, **kw):
    pf = PorosityField(mat, 0.0) if pf is None else pf
    plane = {'plane_strain': True} if ny == 1 else {}
    mesh = CompositeMesh.for_dcb(pf, mat, crack_length=crack_length, length=length,
                                 width=DCB_B, ny=ny, nx=int(length), **plane, **kw)
    return FESolver(mesh, mat, pf, ply_angles='UD')


def _cbt_offset(mat, h):
    """chi h of the corrected beam theory of the DCB (Williams 1989): the
    crack-length correction for the arm's shear and root rotation."""
    gamma = 1.18 * math.sqrt(mat.E11 * mat.E22) / mat.G13
    chi = math.sqrt(mat.E11 / (11.0 * mat.G13) * (3.0 - 2.0 * (gamma / (1.0 + gamma)) ** 2))
    return chi * h


@pytest.fixture(scope='module')
def dcb_growth():
    solver = _dcb()
    return solver, delamination_growth(solver, 'dcb', max_steps=10, applied_load=10.0)


class TestReleasedPairs:
    def test_default_and_assignment(self):
        mesh = _dcb().mesh
        assert mesh.released_pairs.size == 0
        bonded = mesh.bonded_pairs()
        pair = int(np.flatnonzero(bonded)[0])
        mesh.released_pairs = [pair, pair]
        assert list(mesh.released_pairs) == [pair]
        after = mesh.bonded_pairs()
        assert not after[pair] and after.sum() == bonded.sum() - 1
        mesh.released_pairs = None
        assert mesh.bonded_pairs().sum() == bonded.sum()

    @pytest.mark.parametrize('value, match', [
        ([0.5], 'integer indices'), ([[0, 1]], 'integer indices'),
        ([-1], 'must index'), ([10 ** 6], 'must index')])
    def test_bad_values(self, value, match):
        mesh = _dcb().mesh
        with pytest.raises(ValueError, match=match):
            mesh.released_pairs = value

    def test_generate_mesh_clears_it(self):
        mesh = _dcb().mesh
        mesh.released_pairs = [0]
        mesh.generate_mesh()
        assert mesh.released_pairs.size == 0

    def test_releasing_every_pair_separates(self):
        mesh = _dcb().mesh
        mesh.released_pairs = np.arange(mesh.interface_node_pairs()[0].size)
        with pytest.raises(ValueError, match='unconnected parts'):
            mesh.bonded_pairs()

    def test_the_references_bond_every_pair(self):
        mesh = _dcb().mesh
        mesh.released_pairs = [int(np.flatnonzero(mesh.bonded_pairs())[0])]
        assert _knockdown._pristine_mesh(mesh).bonded_pairs().all()
        assert decomposition._without_delamination(mesh).bonded_pairs().all()
        assert mesh.released_pairs.size == 1   # the copies, not the mesh

    def test_a_released_line_is_the_longer_crack(self):
        """Releasing the front line of a strip gives the solve of the strip one
        element longer, to round-off."""
        solver = _dcb()
        front = solver.solve('dcb', applied_load=10.0).crack_front
        solver.mesh.released_pairs = front.pair
        grown = solver.solve('dcb', applied_load=10.0)
        longer = _dcb(DCB_A + 1.0).solve('dcb', applied_load=10.0)
        assert grown.fracture_load_factor == pytest.approx(longer.fracture_load_factor,
                                                           rel=1e-9)
        assert grown.load_point_compliance == pytest.approx(longer.load_point_compliance,
                                                            rel=1e-9)


class TestQuasiStaticDCB:
    def test_rows_are_the_fixed_crack_solves(self, dcb_growth):
        _, g = dcb_growth
        assert g.n_steps == 10 and len(g) == 11 and g.stop_reason == 'max_steps'
        np.testing.assert_allclose(g.crack_lengths, DCB_A + np.arange(11.0), rtol=1e-12)
        for k in (0, 4, 10):
            fixed = _dcb(DCB_A + k).solve('dcb', applied_load=10.0)
            assert g.load_factors[k] == pytest.approx(fixed.fracture_load_factor, rel=1e-9)
            assert g.compliance[k] == pytest.approx(fixed.load_point_compliance, rel=1e-9)
        assert [r.size for r in g.released] == [2] * 10 + [0]   # y = 0 and y = B

    def test_critical_load_follows_corrected_beam_theory(self, dcb_growth):
        """P_c (a + chi h) is constant for the DCB (G = 12 P^2 (a + chi h)^2 /
        (B^2 E h^3), h the arm thickness)."""
        _, g = dcb_growth
        product = g.critical_load * (g.crack_lengths + _cbt_offset(DCB_MAT, DCB_H / 2.0))
        assert np.ptp(product) / product.mean() < 0.005
        # And the level: P_c = B sqrt(G_Ic E11 h^3 / 12) / (a + chi h).
        h = DCB_H / 2.0
        beam = DCB_B * math.sqrt(DCB_MAT.G_Ic * DCB_MAT.E11 * h ** 3 / 12.0)
        assert product.mean() == pytest.approx(beam, rel=0.02)

    def test_stability(self, dcb_growth):
        """Stable under displacement control, unstable under load control."""
        _, g = dcb_growth
        assert not g.unstable('displacement').any()
        assert g.unstable('load').all()
        np.testing.assert_allclose(g.envelope('displacement'), g.critical_displacement)
        np.testing.assert_allclose(g.envelope('load'), np.full(11, g.critical_load[0]))
        with pytest.raises(ValueError, match="'load' or 'displacement'"):
            g.unstable('rotation')

    def test_derived_quantities(self, dcb_growth):
        _, g = dcb_growth
        np.testing.assert_allclose(g.critical_load, 10.0 * g.load_factors)
        np.testing.assert_allclose(g.critical_displacement,
                                   g.load_factors * g.compliance * 10.0)
        assert g.critical_strain is None and g.critical_stress is None
        assert g.effective_modulus is None and g.kind == 'quasi-static'
        assert g.cycles is None and g.growth_rate is None and g.paris is None
        assert np.all(g.psi < 1e-6)                         # mode I
        np.testing.assert_allclose(g.G_T / g.G_c, 1.0 / g.load_factors ** 2, rtol=1e-9)
        np.testing.assert_allclose(g.released_area, np.arange(11.0) * DCB_B)
        assert not g.toughness_uncalibrated                 # no porosity

    def test_mesh_is_left_as_it_was(self, dcb_growth):
        solver, g = dcb_growth
        assert solver.mesh.released_pairs.size == 0
        solver.mesh.released_pairs = g.released_pairs
        try:
            final = solver.solve('dcb', applied_load=10.0)
        finally:
            solver.mesh.released_pairs = ()
        assert final.fracture_load_factor == pytest.approx(g.final.fracture_load_factor,
                                                           rel=1e-12)
        assert g.final.fracture_load_factor == pytest.approx(g.load_factors[-1])

    def test_continues_from_released_pairs(self, dcb_growth):
        solver, g = dcb_growth
        solver.mesh.released_pairs = np.concatenate(g.released[:3])
        try:
            tail = delamination_growth(solver, 'dcb', max_steps=2, applied_load=10.0)
        finally:
            solver.mesh.released_pairs = ()
        np.testing.assert_allclose(tail.load_factors, g.load_factors[3:6], rtol=1e-9)
        np.testing.assert_array_equal(tail.released_pairs,
                                      np.sort(np.concatenate(g.released[:5])))

    def test_max_area(self):
        g = delamination_growth(_dcb(), 'dcb', max_area=60.0, applied_load=10.0)
        assert g.stop_reason == 'max_area'
        assert g.released_area[-1] >= 60.0 > g.released_area[-2]
        assert g.n_steps == 3

    def test_the_load_level_does_not_matter(self, dcb_growth):
        _, g = dcb_growth
        other = delamination_growth(_dcb(), 'dcb', max_steps=2, applied_load=-25.0)
        np.testing.assert_allclose(other.critical_load, g.critical_load[:3], rtol=1e-9)

    def test_the_method_is_the_function(self, dcb_growth):
        _, g = dcb_growth
        again = _dcb().delamination_growth('dcb', max_steps=2, applied_load=10.0)
        np.testing.assert_allclose(again.load_factors, g.load_factors[:3], rtol=1e-12)

    def test_other_errors_propagate(self, monkeypatch):
        """Only a separated laminate stops the growth; another error of a
        solve (a memory refusal, say) is raised, and the mesh put back."""
        solver = _dcb()
        real = solver.solve
        calls = []

        def solve(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise ValueError("The factorization does not fit in memory.")
            return real(*args, **kwargs)

        monkeypatch.setattr(solver, 'solve', solve)
        with pytest.raises(ValueError, match='does not fit'):
            delamination_growth(solver, 'dcb', applied_load=10.0)
        assert solver.mesh.released_pairs.size == 0

    def test_separation_stops_the_growth(self):
        """The ligament runs out at the far end of a short specimen."""
        solver = _dcb(36.0, length=40.0, mat=_material(T300_1076, T300_TOUGHNESS, DCB_H, 4))
        g = delamination_growth(solver, 'dcb', max_steps=10, applied_load=10.0)
        assert g.stop_reason == 'separated'
        assert 0 < g.n_steps < 4 and g.released[-1].size == 0
        assert solver.mesh.released_pairs.size == 0


class TestExport:
    def test_json_block_validates(self, tmp_path, dcb_growth):
        import jsonschema
        _, g = dcb_growth
        path = tmp_path / 'growth.json'
        FESolver.export_results(g.final, path, growth=g)
        doc = load_results_from_json(path)
        schema_path = Path(__file__).resolve().parent.parent / 'validation' / 'schemas' / \
            'porosity_results_schema.json'
        jsonschema.validate(instance=doc, schema=json.loads(schema_path.read_text()))
        block = doc['delamination_growth']
        assert block['kind'] == 'quasi-static' and block['n_steps'] == 10
        assert block['crack_length_mm'] == pytest.approx(list(g.crack_lengths))
        assert block['critical_strain'] is None
        assert block['released_pairs'] == [int(p) for p in g.released_pairs]
        assert 'cycles' not in block
        FESolver.export_results(g.final, path)
        assert 'delamination_growth' not in load_results_from_json(path)

    def test_result_type(self, dcb_growth):
        assert isinstance(dcb_growth[1], GrowthResult)


class TestThreeDimensionalFront:
    @pytest.fixture(scope='class')
    @staticmethod
    def grown():
        mat = _material(T300_1076, T300_TOUGHNESS, DCB_H, 6)
        solver = _dcb(length=40.0, ny=6, mat=mat)
        return solver, delamination_growth(solver, 'dcb', max_steps=12, applied_load=10.0)

    def test_releases_follow_the_front(self, grown):
        """The governing node of the straight front releases first, with its
        mirror image across the width; every release is symmetric about the
        mid-width, and every node line takes the same sequence."""
        solver, g = grown
        orig = solver.mesh.interface_node_pairs()[0]
        front = solver.solve('dcb', applied_load=10.0).crack_front
        assert set(g.released[0]) == set(front.pair[front.lambda_c <= front.lambda_c.min()
                                                     * (1.0 + 1e-6)])
        for pairs in g.released[:-1]:
            y = np.sort(solver.mesh.nodes[orig[pairs], 1])
            np.testing.assert_allclose(y, np.sort(DCB_B - y), atol=1e-9)
        counts = [r.size for r in g.released[:-1]]
        line = counts.index(1) + 1 if 1 in counts else len(counts)
        assert sum(counts[:line]) == 7                    # one node line, ny + 1 nodes
        assert counts[line:2 * line] == counts[:line]

    def test_straight_fronts_match_fixed_cracks(self, grown):
        """Each time a whole node line is released the front is straight
        again and the configuration is the crack one element longer."""
        _, g = grown
        straight = [k for k in range(len(g))
                    if math.isclose(g.crack_lengths[k] % 1.0, 0.0, abs_tol=1e-9)
                    or math.isclose(g.crack_lengths[k] % 1.0, 1.0, abs_tol=1e-9)]
        assert len(straight) >= 2
        mat = _material(T300_1076, T300_TOUGHNESS, DCB_H, 6)
        k = straight[1]
        fixed = _dcb(g.crack_lengths[k], length=40.0, ny=6, mat=mat).solve(
            'dcb', applied_load=10.0)
        assert g.load_factors[k] == pytest.approx(fixed.fracture_load_factor, rel=1e-9)

    def test_staircase_fronts_are_unstable_steps(self, grown):
        """Between straight fronts the critical load dips: those steps are
        unstable under load control, and the envelope stays at the load that
        started the line."""
        _, g = grown
        dips = g.unstable('load')
        assert dips.any()
        envelope = g.envelope('load')
        assert np.all(envelope >= np.abs(g.critical_load) - 1e-12)
        assert np.all(np.diff(envelope) >= 0.0)


class TestENFStability:
    """Beam theory: the ENF grows stably under displacement control only for
    a > 0.69 L (d/da of (2 L^3 + 3 a^3) / a)."""

    @pytest.fixture(scope='class')
    @staticmethod
    def material():
        return _material(IM7_8552, ENF_TOUGHNESS, 4.5, 8)

    def _growth(self, material, a0):
        pf = PorosityField(material, 0.0)
        mesh = CompositeMesh.for_enf(pf, material, crack_length=a0, span=101.6,
                                     width=25.4, plane_strain=True, ny=1)
        solver = FESolver(mesh, material, pf, ply_angles='UD')
        # Four steps stay in the equal elements ahead of the insert.
        return delamination_growth(solver, 'enf', max_steps=4, applied_load=-1000.0)

    def test_short_crack_is_unstable(self, material):
        g = self._growth(material, 0.5 * 50.8)
        assert g.unstable('displacement').all() and g.unstable('load').all()
        assert np.all(g.psi > 0.999)                       # mode II, with contact
        assert np.all(g.critical_load < 0.0)               # the sign of the load

    def test_long_crack_is_stable(self, material):
        g = self._growth(material, 0.8 * 50.8)
        assert not g.unstable('displacement').any()
        assert g.unstable('load').all()


class TestPorosity:
    @pytest.fixture(scope='class')
    @staticmethod
    def runs():
        """A porous band (4 % voids) from x = 33.5 mm ahead of the crack."""
        pmap = PorosityMap([0.0, 33.4, 33.6, 60.0], [0.0, DCB_B],
                           [[0.0, 0.0], [0.0, 0.0], [0.04, 0.04], [0.04, 0.04]])
        pf = PorosityField.from_map(DCB_MAT, pmap)
        clean = delamination_growth(_dcb(), 'dcb', max_steps=6, applied_load=10.0)
        porous = delamination_growth(_dcb(pf=pf), 'dcb', max_steps=6, applied_load=10.0)
        stiffness_only = delamination_growth(_dcb(pf=pf), 'dcb', max_steps=6,
                                             applied_load=10.0, toughness_knockdown=None)
        return clean, porous, stiffness_only

    def test_the_front_slows_into_the_porous_band(self, runs):
        clean, porous, stiffness_only = runs
        ratio = porous.critical_load / clean.critical_load
        assert np.all(ratio[:4] > 0.98)                   # before the band
        assert np.all(ratio[4:] < 0.9)                    # in it
        np.testing.assert_allclose(porous.Vp, [0.0] * 4 + [0.04] * 3)
        assert porous.toughness_uncalibrated and not clean.toughness_uncalibrated
        # Most of the drop is the toughness knocked down at the front.
        assert np.all(stiffness_only.critical_load[4:] > porous.critical_load[4:])
        assert not stiffness_only.toughness_uncalibrated


def _coupon():
    """An 8-ply QI coupon with a 20 x 10 mm delamination under the second ply."""
    mat = dataclasses.replace(MATERIALS['IM7_8552_epoxy'], n_plies=8)
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh(pf, mat, nx=10, ny=4, nz=8, ply_angles='QI',
                         delaminations=[Delamination.rectangle(1, (25, 10), (10, 5))])
    return FESolver(mesh, mat, pf, ply_angles='QI')


class TestStrainControlled:
    def test_tension_coupon(self):
        g = delamination_growth(_coupon(), 'tension', max_steps=2)
        assert g.applied_strain == 0.01 and g.applied_load is None
        assert g.critical_load is None and g.critical_displacement is None
        np.testing.assert_allclose(g.critical_strain, 0.01 * g.load_factors)
        np.testing.assert_allclose(g.critical_stress, g.critical_strain * g.effective_modulus)
        assert g.crack_lengths is None
        np.testing.assert_array_equal(g.unstable('load'),
                                      np.abs(g.critical_stress[1:])
                                      <= np.abs(g.critical_stress[:-1]))
        assert g.unstable('displacement').shape == (2,)

    def test_a_front_the_load_does_not_drive(self):
        """In-plane shear releases no energy at this front: nothing grows."""
        g = delamination_growth(_coupon(), 'shear', max_steps=2)
        assert g.stop_reason == 'not_driven' and g.n_steps == 0
        assert g.load_factors[0] == math.inf and g.governing_node[0] == -1
        assert g.summary()['load_factors'] == [None]

    def test_no_displacement_without_compliance(self, dcb_growth):
        g = dataclasses.replace(dcb_growth[1], compliance=None)
        with pytest.raises(ValueError, match="No critical displacement"):
            g.unstable('displacement')


class TestFatigue:
    LAW = ParisLaw(C=0.05, m=5.0)

    @pytest.fixture(scope='class')
    @staticmethod
    def fatigue():
        return fatigue_delamination_growth(_dcb(), 'dcb', TestFatigue.LAW, max_steps=6,
                                           applied_load=40.0)

    def test_cycles_integrate_the_paris_law(self, fatigue, dcb_growth):
        """With a straight front every step grows the whole line: Delta N =
        Delta a / (C (G_max / G_c)^m), G_max / G_c = (P / P_c)^2."""
        _, static = dcb_growth
        f = fatigue
        assert f.kind == 'fatigue' and f.paris is self.LAW and f.stop_reason == 'max_steps'
        ratio = (40.0 / static.critical_load[:7]) ** 2
        np.testing.assert_allclose(f.growth_rate, 0.05 * ratio ** 5, rtol=1e-9)
        np.testing.assert_allclose(np.diff(f.cycles), 1.0 / f.growth_rate[:-1], rtol=1e-9)
        assert f.cycles[0] == 0.0
        np.testing.assert_allclose(f.load_factors, static.load_factors[:7] / 4.0, rtol=1e-9)

    def test_summary(self, fatigue):
        record = fatigue.summary()
        assert record['paris'] == {'form': 'da/dN = C (G_max / G_c)^m',
                                   'C_mm_per_cycle': 0.05, 'm': 5.0, 'threshold': 0.0}
        assert record['cycles'] == pytest.approx(list(fatigue.cycles))
        assert record['kind'] == 'fatigue'

    def test_a_peak_above_the_static_load_is_static_growth(self):
        f = fatigue_delamination_growth(_dcb(), 'dcb', self.LAW, applied_load=70.0)
        assert f.stop_reason == 'static' and f.n_steps == 0
        assert f.load_factors[0] < 1.0

    def test_threshold_arrests(self):
        law = ParisLaw(C=0.05, m=5.0, threshold=0.5)
        f = fatigue_delamination_growth(_dcb(), 'dcb', law, applied_load=40.0)
        assert f.stop_reason == 'arrested' and f.n_steps == 0
        assert f.governing_node[0] == -1 and math.isnan(f.G_T[0])

    def test_max_cycles(self, fatigue):
        f = fatigue_delamination_growth(_dcb(), 'dcb', self.LAW, applied_load=40.0,
                                        max_cycles=0.5 * (fatigue.cycles[2] + fatigue.cycles[3]))
        assert f.stop_reason == 'max_cycles' and f.n_steps == 2

    def test_porosity_speeds_the_growth(self, fatigue):
        pmap = PorosityMap([0.0, 33.4, 33.6, 60.0], [0.0, DCB_B],
                           [[0.0, 0.0], [0.0, 0.0], [0.04, 0.04], [0.04, 0.04]])
        porous = fatigue_delamination_growth(
            _dcb(pf=PorosityField.from_map(DCB_MAT, pmap)), 'dcb', self.LAW,
            max_steps=6, applied_load=40.0)
        steps = np.diff(porous.cycles) / np.diff(fatigue.cycles)
        assert np.all(steps[4:] < 0.6)                   # in the band
        assert porous.cycles[-1] < fatigue.cycles[-1]

    def test_the_method_is_the_function(self, fatigue):
        again = _dcb().fatigue_delamination_growth('dcb', self.LAW, max_steps=2,
                                                   applied_load=40.0)
        np.testing.assert_allclose(again.cycles, fatigue.cycles[:3], rtol=1e-12)

    def test_shares_carry_over(self):
        """The fastest row releases; the others keep the share they grew."""
        rate = np.array([2.0, 1.0, 0.0, 1.0])
        delta_a = np.ones(4)
        share = np.array([0.0, 0.0, 0.0, 0.6])
        rows, dN = growth_module._fatigue_release(rate, delta_a, share, 1e-9)
        assert dN == pytest.approx(0.4) and list(rows) == [3]
        rows, dN = growth_module._fatigue_release(rate, delta_a, np.zeros(4), 1e-9)
        assert dN == pytest.approx(0.5) and list(rows) == [0]

    def test_three_dimensional_front(self, monkeypatch):
        """Node by node across the width; the share each node grew while
        others released is kept, so the same crack area takes fewer cycles
        than with every share dropped at each release."""
        mat = _material(T300_1076, T300_TOUGHNESS, DCB_H, 6)

        def run():
            f = fatigue_delamination_growth(_dcb(length=40.0, ny=6, mat=mat), 'dcb',
                                            self.LAW, max_steps=10, applied_load=30.0)
            assert np.all(np.diff(f.cycles) > 0.0)
            return f

        kept = run()
        real = growth_module._fatigue_release
        monkeypatch.setattr(growth_module, '_fatigue_release',
                            lambda rate, delta_a, share, tol:
                            real(rate, delta_a, np.zeros_like(share), tol))
        dropped = run()
        assert kept.released_area[-1] == pytest.approx(dropped.released_area[-1])
        assert kept.cycles[-1] < 0.8 * dropped.cycles[-1]


class TestParisLaw:
    @pytest.mark.parametrize('kwargs, match', [
        ({'C': 0.0, 'm': 5.0}, 'ParisLaw.C'), ({'C': -1.0, 'm': 5.0}, 'ParisLaw.C'),
        ({'C': 1.0, 'm': float('inf')}, 'ParisLaw.m'), ({'C': True, 'm': 5.0}, 'ParisLaw.C'),
        ({'C': 1.0, 'm': 5.0, 'threshold': 1.0}, 'threshold'),
        ({'C': 1.0, 'm': 5.0, 'threshold': -0.1}, 'threshold')])
    def test_bad_coefficients(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ParisLaw(**kwargs)

    def test_rate(self):
        law = ParisLaw(C=0.01, m=2, threshold=0.2)
        np.testing.assert_allclose(law.rate([0.0, 0.1, 0.5, 1.0, -0.3]),
                                   [0.0, 0.0, 0.0025, 0.01, 0.0])
        assert law.m == 2.0 and isinstance(law.m, float)


class TestRefusals:
    @pytest.mark.parametrize('kwargs, match', [
        ({'max_steps': 0}, 'max_steps'), ({'max_steps': True}, 'max_steps'),
        ({'max_area': -1.0}, 'max_area'), ({'max_area': float('nan')}, 'max_area'),
        ({'release_tolerance': -0.1}, 'release_tolerance')])
    def test_limits(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            delamination_growth(_dcb(), 'dcb', **kwargs)

    def test_loading(self):
        with pytest.raises(ValueError, match='mechanical load cases'):
            delamination_growth(_dcb(), 'thermal', delta_T=-100.0)

    def test_needs_a_delamination(self):
        mat = dataclasses.replace(MATERIALS['IM7_8552_epoxy'], n_plies=4)
        pf = PorosityField(mat, 0.0)
        solver = FESolver(CompositeMesh(pf, mat, nx=4, ny=2, nz=4), mat, pf)
        with pytest.raises(ValueError, match='delaminated mesh'):
            delamination_growth(solver, 'tension')

    def test_needs_toughness(self):
        mat = dataclasses.replace(DCB_MAT, G_Ic=None, G_IIc=None, bk_eta=None)
        with pytest.raises(ValueError, match='toughness'):
            delamination_growth(_dcb(mat=mat), 'dcb')

    def test_fatigue_arguments(self):
        with pytest.raises(ValueError, match='ParisLaw'):
            fatigue_delamination_growth(_dcb(), 'dcb', (0.05, 5.0))
        with pytest.raises(ValueError, match='max_cycles'):
            fatigue_delamination_growth(_dcb(), 'dcb', ParisLaw(0.05, 5.0), max_cycles=0)
