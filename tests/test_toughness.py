"""Porosity-dependent interlaminar toughness (delamination plan, phase 4).

- The law (:class:`ToughnessKnockdown`, :func:`toughness_knockdown_factor`):
  shapes and limits, input validation, that it never raises the toughness,
  user callables, the placeholder coefficients against the datasets they
  come from, and the one-time INFO line.
- The fracture functions: ``critical_energy_release_rate`` and
  ``fracture_load_factors`` (with and without a prestress) evaluate the
  knocked-down pure-mode toughness at each node's ``Vp``.
- The FE coupling on an end-notched flexure beam: ``G_c`` at the front
  follows the law at the front's porosity, zero porosity is unchanged, an
  interface-clustered field lowers ``G_c`` against uniform porosity of the
  same mean, and the uncalibrated flag reaches ``summary().details`` and
  the JSON ``fracture`` block.

Toughness in N/mm; ``Vp`` is a fraction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os

import numpy as np
import pytest

import porosity_fe.toughness as toughness_module
from porosity_fe import (
    MATERIALS,
    Calibration,
    CompositeMesh,
    CrackFront,
    Delamination,
    FESolver,
    PorosityField,
    ToughnessKnockdown,
    critical_energy_release_rate,
    fracture_load_factors,
    toughness_knockdown_factor,
)
from porosity_fe.toughness import _resolve_toughness_knockdown

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_SCHEMA = os.path.join(_ROOT, 'validation', 'schemas', 'porosity_results_schema.json')
_DATASETS = os.path.join(_ROOT, 'validation', 'datasets')

#: IM7/8552 toughness and B-K exponent (N/mm), Krueger NASA/CR-2012-217562
#: (plan Appendix A), as in tests/test_vcct.py.
TOUGHNESS = dict(G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)
MAT = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=16, **TOUGHNESS)
MID16 = 7          # mid-plane interface of 16 plies
B_I, B_II = 6.2, 14.0


# --------------------------------------------------------------------------
# The law
# --------------------------------------------------------------------------

class TestLaw:
    VP = np.linspace(0.0, 1.0, 101)

    def test_placeholders(self):
        assert Calibration.TOUGHNESS_B_PLACEHOLDER == {'I': B_I, 'II': B_II}
        assert Calibration.TOUGHNESS_B_PLACEHOLDER is toughness_module._TOUGHNESS_B_PLACEHOLDER
        law = ToughnessKnockdown()
        assert law.law == 'exponential' and law.uncalibrated
        assert [law.coefficient(m) for m in ('I', 'II', 'III')] == [B_I, B_II, B_II]
        assert all(law.uses_placeholder(m) for m in ('I', 'II', 'III'))
        assert 'placeholder, uncalibrated' in law.describe()

    def test_exponential_shape(self):
        law = ToughnessKnockdown()
        np.testing.assert_array_equal(law.factor(self.VP, 'I'), np.exp(-B_I * self.VP))
        np.testing.assert_array_equal(law.factor(self.VP, 'II'), np.exp(-B_II * self.VP))
        np.testing.assert_array_equal(law.factor(self.VP, 'III'), law.factor(self.VP, 'II'))
        own = ToughnessKnockdown(b_I=4.0, b_II=10.0, b_III=12.0)
        np.testing.assert_array_equal(own.factor(self.VP, 'III'), np.exp(-12.0 * self.VP))
        assert not own.uncalibrated and 'placeholder' not in own.describe()
        half = ToughnessKnockdown(b_I=4.0)              # mode II stays a placeholder
        assert half.uncalibrated and not half.uses_placeholder('I')
        assert half.uses_placeholder('II') and half.uses_placeholder('III')
        assert '(b_II, b_III placeholder, uncalibrated)' in half.describe()

    def test_linear_shape_and_clip(self):
        law = ToughnessKnockdown('linear', b_I=5.0, b_II=12.0)
        np.testing.assert_array_equal(law.factor(self.VP, 'I'),
                                      np.maximum(1.0 - 5.0 * self.VP, 0.0))
        assert float(law.factor(0.5, 'II')) == 0.0      # clipped, never negative
        assert not law.uncalibrated
        assert law.coefficient('III') == 12.0

    def test_none(self):
        law = _resolve_toughness_knockdown(None)
        assert law.law == 'none' and not law.uncalibrated
        np.testing.assert_array_equal(law.factor(self.VP, 'II'), 1.0)
        assert law.coefficient('I') is None
        assert law.describe() == 'none (pristine toughness)'

    @pytest.mark.parametrize('law', [
        ToughnessKnockdown(),
        ToughnessKnockdown(b_I=0.0, b_II=0.0),
        ToughnessKnockdown(b_I=50.0, b_II=200.0),
        ToughnessKnockdown('linear', b_I=3.0, b_II=30.0),
        ToughnessKnockdown('none'),
    ], ids=lambda k: k.describe()[:24])
    @pytest.mark.parametrize('mode', ['I', 'II', 'III'])
    def test_never_increases(self, law, mode):
        """1 at Vp = 0 exactly, within [0, 1] and non-increasing."""
        f = law.factor(self.VP, mode)
        assert f[0] == 1.0
        assert np.all((f >= 0.0) & (f <= 1.0))
        assert np.all(np.diff(f) <= 0.0)

    def test_zero_coefficient_is_pristine(self):
        np.testing.assert_array_equal(
            ToughnessKnockdown(b_I=0.0, b_II=0.0).factor(self.VP, 'I'), 1.0)

    def test_callable(self):
        calls = []

        def law(Vp, mode):
            calls.append((Vp, mode))
            return 1.0 - (2.0 if mode == 'I' else 3.0) * Vp

        k = ToughnessKnockdown(law)
        Vp = np.array([[0.0, 0.1], [0.1, 0.2]])
        np.testing.assert_allclose(k.factor(Vp, 'I'), 1.0 - 2.0 * Vp)
        assert all(isinstance(v, float) for v, _ in calls)
        assert len(calls) == 3                            # each distinct Vp once
        np.testing.assert_allclose(k.factor(Vp, 'III'), 1.0 - 3.0 * Vp)
        assert not k.uncalibrated and k.coefficient('I') is None
        assert k.describe().startswith('user callable')
        assert _resolve_toughness_knockdown(law).law is law

    @pytest.mark.parametrize('value, match', [
        (1.2, 'must not rise'), (-0.1, r'\[0, 1\]'), (math.nan, r'\[0, 1\]')])
    def test_callable_out_of_range(self, value, match):
        with pytest.raises(ValueError, match=match):
            ToughnessKnockdown(lambda Vp, mode: value).factor(0.02, 'I')

    def test_callable_that_raises(self):
        def broken(Vp, mode):
            raise RuntimeError('no data')
        with pytest.raises(ValueError, match='RuntimeError.*no data'):
            ToughnessKnockdown(broken).factor(0.02, 'II')

    @pytest.mark.parametrize('kwargs, error, match', [
        (dict(law='power'), ValueError, 'Unknown toughness law'),
        (dict(law=3), TypeError, 'law must be'),
        (dict(b_I=-1.0), ValueError, '>= 0'),
        (dict(b_II=math.inf), ValueError, 'finite'),
        (dict(b_I=True), TypeError, 'number'),
        (dict(b_II='14'), TypeError, 'number'),
        (dict(law='none', b_I=1.0), ValueError, 'take coefficients'),
        (dict(law=lambda Vp, mode: 1.0, b_II=1.0), ValueError, 'take coefficients'),
        (dict(law='linear', b_I=1.0), ValueError, 'no placeholder'),
    ])
    def test_invalid_construction(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            ToughnessKnockdown(**kwargs)

    def test_invalid_inputs(self):
        law = ToughnessKnockdown()
        with pytest.raises(ValueError, match='percent.*0.0200'):
            law.factor(2.0, 'I')
        for Vp in (-0.01, math.nan, [0.01, 1.5]):
            with pytest.raises(ValueError, match=r'fraction in \[0, 1\]'):
                law.factor(Vp, 'I')
        with pytest.raises(ValueError, match='Unknown fracture mode'):
            law.factor(0.01, 'IV')
        with pytest.raises(TypeError, match='toughness_knockdown must be'):
            _resolve_toughness_knockdown(3.0)

    def test_coefficients_are_floats(self):
        law = ToughnessKnockdown(b_I=np.int64(5), b_II=12)
        assert type(law.b_I) is float and type(law.b_II) is float


class TestScreeningFunction:
    def test_values_and_types(self):
        assert toughness_knockdown_factor(0.03) == pytest.approx(math.exp(-B_I * 0.03))
        assert isinstance(toughness_knockdown_factor(0.03, 'II'), float)
        assert toughness_knockdown_factor(0.0, 'II') == 1.0
        out = toughness_knockdown_factor([0.0, 0.01, 0.02], 'II')
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.exp(-B_II * np.array([0.0, 0.01, 0.02])))
        assert toughness_knockdown_factor(0.03, 'I', None) == 1.0
        assert toughness_knockdown_factor(
            0.03, 'II', ToughnessKnockdown(b_I=1.0, b_II=2.0)) == pytest.approx(math.exp(-0.06))
        assert toughness_knockdown_factor(
            0.03, 'I', lambda Vp, mode: 0.5) == 0.5

    def test_uncalibrated_info_logged_once(self, caplog, monkeypatch):
        monkeypatch.setattr(toughness_module, '_uncalibrated_logged', False)
        caplog.set_level(logging.INFO, logger='porosity_fe_analysis')
        toughness_knockdown_factor(0.0, 'I')                   # changes nothing
        toughness_knockdown_factor(0.03, 'I', ToughnessKnockdown(b_I=5.0))   # calibrated mode I
        assert not [r for r in caplog.records if 'uncalibrated' in r.getMessage()]
        toughness_knockdown_factor(0.03, 'II')
        toughness_knockdown_factor(0.04, 'I')
        noted = [r for r in caplog.records if 'uncalibrated' in r.getMessage()]
        assert len(noted) == 1 and noted[0].levelno == logging.INFO
        assert 'toughness_uncalibrated' in noted[0].getMessage()


class TestPlaceholderSources:
    """The placeholder exponents come from the two-point estimates of the
    studies (theory page "Delamination", "Porosity and toughness"); the two
    bundled datasets reproduce their share of them."""

    @staticmethod
    def _block(name, key):
        with open(os.path.join(_DATASETS, name), encoding='utf-8') as f:
            return json.load(f)['properties'][key]

    def test_mode_I(self):
        hakim = self._block('hakim_2017.json', 'G_Ic')
        vp = np.asarray(hakim['void_content_pct']) / 100.0
        G = np.asarray(hakim['absolute_values_N_per_mm'])
        b_hakim = -math.log(G[-1] / G[0]) / (vp[-1] - vp[0])          # 7.45
        b_olivier = -math.log(0.78) / 0.05      # -22 % at +5 % voids [S]: 4.97
        assert b_hakim == pytest.approx(7.45, abs=0.01)
        assert 0.5 * (b_hakim + b_olivier) == pytest.approx(B_I, abs=0.05)
        np.testing.assert_allclose(hakim['normalized_values'], G / G[0], atol=1e-4)

    def test_mode_II(self):
        agius = self._block('agius_fox_2015.json', 'G_IIc')
        ratio = agius['normalized_values'][1]
        assert ratio == pytest.approx(1.0 / 1.21, abs=1e-4)
        # Low-void group between 0 and its stated bound of 0.3 %.
        low, high = (math.log(1.21) / ((1.5 - v) / 100.0) for v in (0.0, 0.3))
        assert low == pytest.approx(12.7, abs=0.05) and high == pytest.approx(15.9, abs=0.05)
        assert low < B_II < high
        assert math.log(1.21) / ((1.5 - 0.15) / 100.0) == pytest.approx(B_II, abs=0.15)


# --------------------------------------------------------------------------
# The fracture functions
# --------------------------------------------------------------------------

def _front(Vp, G_I, G_II, G_III=None):
    """A CrackFront stub: pure force/opening products giving the rates."""
    G_I, G_II = np.asarray(G_I, float), np.asarray(G_II, float)
    G_III = np.zeros_like(G_I) if G_III is None else np.asarray(G_III, float)
    n = G_I.size
    area = np.ones(n)
    # G = -F D / (2 area): F = -1, D = 2 G in each local direction.
    force = -np.ones((n, 3))
    opening = np.column_stack([2.0 * G_II, 2.0 * G_I, 2.0 * G_III])
    G_T = G_I + G_II + G_III
    zeros = np.zeros(n)
    return CrackFront(
        node=np.arange(n), duplicate=np.arange(n) + n, pair=np.arange(n),
        interface=np.zeros(n, dtype=int), behind=np.arange(n), x=zeros, y=zeros,
        z=zeros, direction=np.tile([1.0, 0.0], (n, 1)), delta_a=area, width=area,
        force_local=force, opening_local=opening, G_I=G_I, G_II=G_II, G_III=G_III,
        G_T=G_T, psi=(G_II + G_III) / G_T, Vp=np.asarray(Vp, float))


def _bk(G_Ic, G_IIc, G_IIIc, eta, G_I, G_II, G_III):
    psi = (G_II + G_III) / (G_I + G_II + G_III)
    tear = G_III / (G_II + G_III) if G_II + G_III > 0 else 0.0
    return G_Ic + (G_IIc - G_Ic) * psi ** eta + (G_IIIc - G_IIc) * tear * psi ** eta


class TestFractureFunctions:
    def test_pure_modes(self):
        Vp = 0.03
        assert critical_energy_release_rate(MAT, 1.0, 0.0, Vp=Vp) == pytest.approx(
            MAT.G_Ic * math.exp(-B_I * Vp), rel=1e-14)
        assert critical_energy_release_rate(MAT, 0.0, 1.0, Vp=Vp) == pytest.approx(
            MAT.G_IIc * math.exp(-B_II * Vp), rel=1e-14)
        tear = dataclasses.replace(MAT, G_IIIc=1.2)
        assert critical_energy_release_rate(tear, 0.0, 0.0, 1.0, Vp=Vp) == pytest.approx(
            1.2 * math.exp(-B_II * Vp), rel=1e-14)
        own = ToughnessKnockdown(b_I=2.0, b_II=3.0, b_III=9.0)
        assert critical_energy_release_rate(
            tear, 0.0, 0.0, 1.0, Vp=Vp, toughness_knockdown=own) == pytest.approx(
            1.2 * math.exp(-9.0 * Vp), rel=1e-14)

    @pytest.mark.parametrize('G', [(1.0, 1.0, 0.0), (0.3, 1.0, 0.5), (2.0, 0.1, 0.4)])
    def test_mixed_mode_uses_the_knocked_down_pure_modes(self, G):
        """B-K with each pure-mode toughness knocked down; eta unchanged."""
        tear = dataclasses.replace(MAT, G_IIIc=1.1)
        Vp = 0.025
        expected = _bk(tear.G_Ic * math.exp(-B_I * Vp), tear.G_IIc * math.exp(-B_II * Vp),
                       1.1 * math.exp(-B_II * Vp), tear.bk_eta, *G)
        assert critical_energy_release_rate(tear, *G, Vp=Vp) == pytest.approx(expected, rel=1e-13)

    def test_broadcasts_over_Vp(self):
        Vp = np.array([0.0, 0.01, 0.05])
        G_c = critical_energy_release_rate(MAT, 1.0, 0.0, Vp=Vp)
        np.testing.assert_allclose(G_c, MAT.G_Ic * np.exp(-B_I * Vp), rtol=1e-14)
        assert G_c[0] == MAT.G_Ic
        with pytest.raises(ValueError, match='percent'):
            critical_energy_release_rate(MAT, 1.0, 0.0, Vp=3.0)

    def test_load_factor_and_record(self):
        front = _front([0.0, 0.02, 0.04], [0.1, 0.05, 0.0], [0.0, 0.05, 0.2])
        out = fracture_load_factors(front, MAT)
        expected = critical_energy_release_rate(MAT, front.G_I, front.G_II, front.G_III,
                                                Vp=front.Vp)
        np.testing.assert_array_equal(out.G_c, expected)
        np.testing.assert_allclose(out.lambda_c, np.sqrt(expected / front.G_T), rtol=1e-15)
        assert out.toughness_uncalibrated is True
        assert out.toughness_knockdown == ToughnessKnockdown()
        pristine = fracture_load_factors(front, MAT, toughness_knockdown=None)
        assert np.all(out.G_c[1:] < pristine.G_c[1:]) and out.G_c[0] == pristine.G_c[0]
        assert pristine.toughness_uncalibrated is False
        assert pristine.toughness_knockdown.law == 'none'
        own = fracture_load_factors(front, MAT,
                                    toughness_knockdown=ToughnessKnockdown(b_I=5.0, b_II=9.0))
        assert own.toughness_uncalibrated is False
        # Placeholders at zero porosity change nothing, so nothing is flagged.
        dry = fracture_load_factors(dataclasses.replace(front, Vp=np.zeros(3)), MAT)
        assert dry.toughness_uncalibrated is False
        np.testing.assert_array_equal(dry.G_c, pristine.G_c)
        np.testing.assert_array_equal(dry.lambda_c, pristine.lambda_c)

    def test_prestress_uses_the_knocked_down_toughness(self):
        """A prestress c times the load state: lambda_c = sqrt(G_c / G_T) - c
        with G_c knocked down at the node's Vp."""
        front = _front([0.01, 0.03], [0.1, 0.02], [0.05, 0.2])
        lam = fracture_load_factors(front, MAT).lambda_c
        for c in (0.3, -0.4):
            pre = dataclasses.replace(front, force_local=c * front.force_local,
                                      opening_local=c * front.opening_local)
            shifted = fracture_load_factors(front, MAT, prestress=pre)
            np.testing.assert_allclose(shifted.lambda_c, lam - c, rtol=1e-10)
            assert shifted.toughness_uncalibrated is True
        none = fracture_load_factors(front, MAT, prestress=pre, toughness_knockdown=None)
        pristine = fracture_load_factors(front, MAT, toughness_knockdown=None).lambda_c
        np.testing.assert_allclose(none.lambda_c, pristine - c, rtol=1e-10)
        assert np.all(shifted.lambda_c < none.lambda_c)


# --------------------------------------------------------------------------
# FE coupling (end-notched flexure, crack-face contact)
# --------------------------------------------------------------------------

def _enf(Vp, distribution='uniform', **solve_kwargs):
    """ENF beam (span 50 mm, 16 UD plies, 1 mm elements), mid-plane crack
    from the left end to x = 15 mm, 3-point bend under 100 N."""
    pf = PorosityField(MAT, Vp, distribution=distribution)
    mesh = CompositeMesh.for_flexure(
        pf, MAT, span=50.0, width=4.0, overhang=5.0, load_width=0.0,
        nz=16, ny=2, element_length=1.0, ply_angles='UD',
        delaminations=[Delamination.strip(MID16, -1.0, 15.0)])
    solver = FESolver(mesh, MAT, pf, ply_angles='UD')
    return mesh, pf, solver, solver.solve('flexure_3pt', applied_load=-100.0, **solve_kwargs)


@pytest.fixture(scope='module')
def enf():
    out = {}
    for name, Vp, dist in (('dry', 0.0, 'uniform'), ('uniform', 0.02, 'uniform'),
                           ('clustered', 0.02, 'clustered'),
                           ('interface', 0.02, 'interface')):
        mesh, pf, solver, r = _enf(Vp, dist)
        out[name] = dict(mesh=mesh, pf=pf, solver=solver, r=r,
                         pristine=solver.solve('flexure_3pt', applied_load=-100.0,
                                               toughness_knockdown=None))
    return out


class TestFECoupling:
    def test_front_porosity_is_the_field_at_the_crack_plane(self, enf):
        for case in ('uniform', 'clustered', 'interface'):
            front, pf, mesh = enf[case]['r'].crack_front, enf[case]['pf'], enf[case]['mesh']
            np.testing.assert_array_equal(front.Vp, pf.local_porosity(front.x, front.y, front.z))
            np.testing.assert_array_equal(front.Vp, mesh.porosity[front.duplicate])
            assert np.allclose(front.z, 8 * MAT.t_ply)

    @pytest.mark.parametrize('case', ['uniform', 'clustered', 'interface'])
    def test_G_c_follows_the_law_at_the_front(self, enf, case):
        r, pristine = enf[case]['r'], enf[case]['pristine']
        front = r.crack_front
        assert front.Vp.min() > 0.0
        expected = critical_energy_release_rate(
            MAT, front.G_I, front.G_II, front.G_III, Vp=front.Vp)
        np.testing.assert_array_equal(front.G_c, expected)
        # The ENF front is pure mode II (psi = 1 to clipping): the mode II law.
        np.testing.assert_allclose(front.G_c, MAT.G_IIc * np.exp(-B_II * front.Vp), rtol=1e-5)
        np.testing.assert_allclose(front.lambda_c, np.sqrt(front.G_c / front.G_T), rtol=1e-15)
        assert r.fracture_load_factor == pytest.approx(float(np.min(front.lambda_c)), rel=0)
        # The same displacements, energy release rates and contact either way.
        np.testing.assert_array_equal(r.displacement, pristine.displacement)
        np.testing.assert_array_equal(front.G_T, pristine.crack_front.G_T)
        np.testing.assert_allclose(r.fracture_load_factor / pristine.fracture_load_factor,
                                   math.exp(-0.5 * B_II * float(front.Vp[0])), rtol=1e-5)

    def test_zero_porosity_is_unchanged(self, enf):
        r, pristine = enf['dry']['r'], enf['dry']['pristine']
        front = r.crack_front
        np.testing.assert_array_equal(front.Vp, 0.0)
        np.testing.assert_array_equal(front.G_c, pristine.crack_front.G_c)
        np.testing.assert_array_equal(front.lambda_c, pristine.crack_front.lambda_c)
        assert r.fracture_load_factor == pristine.fracture_load_factor
        np.testing.assert_array_equal(
            front.G_c, critical_energy_release_rate(MAT, front.G_I, front.G_II, front.G_III))
        assert front.toughness_uncalibrated is False
        assert r.summary().details['toughness_uncalibrated'] is False

    @pytest.mark.parametrize('clustered', ['clustered', 'interface'])
    def test_interface_clustered_voids_lower_G_c(self, enf, clustered):
        """The same mean Vp, clustered at the crack plane (a midplane bump,
        or Gaussians at every ply interface): more porosity at the front,
        so a lower toughness and load factor than uniform porosity."""
        uni, clu = enf['uniform'], enf[clustered]
        assert clu['pf'].Vp == uni['pf'].Vp == 0.02
        f_u, f_c = uni['r'].crack_front, clu['r'].crack_front
        assert np.all(f_c.Vp > f_u.Vp)
        assert np.all(f_c.G_c < f_u.G_c)
        # Against each one's own pristine toughness (the porosity changes
        # the stiffness too), the clustered front keeps less of it.
        keep_u = f_u.G_c / uni['pristine'].crack_front.G_c
        keep_c = f_c.G_c / clu['pristine'].crack_front.G_c
        assert np.all(keep_c < keep_u)
        # Without the law the distribution changes G_c only through psi.
        np.testing.assert_allclose(clu['pristine'].crack_front.G_c,
                                   uni['pristine'].crack_front.G_c, rtol=1e-5)

    def test_clustered_lowers_the_load_factor_through_the_toughness(self, enf):
        uni, clu = enf['uniform'], enf['clustered']
        ratio = clu['r'].fracture_load_factor / uni['r'].fracture_load_factor
        ratio_pristine = (clu['pristine'].fracture_load_factor
                          / uni['pristine'].fracture_load_factor)
        assert ratio < ratio_pristine

    def test_other_laws_through_the_solve(self, enf):
        solver = enf['uniform']['solver']
        own = solver.solve('flexure_3pt', applied_load=-100.0,
                           toughness_knockdown=ToughnessKnockdown(b_I=3.0, b_II=7.0))
        front = own.crack_front
        np.testing.assert_allclose(front.G_c, MAT.G_IIc * np.exp(-7.0 * front.Vp), rtol=1e-5)
        assert front.toughness_uncalibrated is False
        user = solver.solve('flexure_3pt', applied_load=-100.0,
                            toughness_knockdown=lambda Vp, mode: 0.5)
        np.testing.assert_allclose(user.crack_front.G_c, 0.5 * MAT.G_IIc, rtol=1e-5)
        none = solver.solve('flexure_3pt', applied_load=-100.0, toughness_knockdown='none')
        np.testing.assert_array_equal(none.crack_front.G_c,
                                      enf['uniform']['pristine'].crack_front.G_c)

    def test_invalid_law_raises_before_assembly(self, enf):
        solver = enf['uniform']['solver']
        with pytest.raises(ValueError, match='Unknown toughness law'):
            solver.solve('flexure_3pt', applied_load=-100.0, toughness_knockdown='cubic')
        with pytest.raises(ValueError, match='no placeholder'):
            solver.solve('flexure_3pt', applied_load=-100.0, toughness_knockdown='linear')

    def test_no_toughness_data(self):
        mat = dataclasses.replace(MAT, G_Ic=None, G_IIc=None, bk_eta=None)
        pf = PorosityField(mat, 0.02)
        mesh = CompositeMesh.for_flexure(
            pf, mat, span=50.0, width=4.0, overhang=5.0, load_width=0.0,
            nz=16, ny=2, element_length=1.0, ply_angles='UD',
            delaminations=[Delamination.strip(MID16, -1.0, 15.0)])
        r = FESolver(mesh, mat, pf, ply_angles='UD').solve('flexure_3pt', applied_load=-100.0)
        assert r.crack_front.G_c is None and r.crack_front.toughness_knockdown is None
        assert r.crack_front.toughness_uncalibrated is None
        details = r.summary().details
        assert details['toughness_knockdown'] is None and details['toughness_uncalibrated'] is None


class TestOutputs:
    def test_summary_details(self, enf):
        details = enf['uniform']['r'].summary().details
        assert details['toughness_uncalibrated'] is True
        assert details['toughness_knockdown'] == ToughnessKnockdown().describe()
        pristine = enf['uniform']['pristine'].summary().details
        assert pristine['toughness_uncalibrated'] is False
        assert pristine['toughness_knockdown'] == 'none (pristine toughness)'

    def test_json_fracture_block(self, enf, tmp_path):
        import jsonschema
        r = enf['clustered']['r']
        path = tmp_path / 'enf.json'
        FESolver.export_results(r, path)
        doc = json.loads(path.read_text(encoding='utf-8'))
        with open(_RESULTS_SCHEMA, encoding='utf-8') as f:
            jsonschema.validate(instance=doc, schema=json.load(f))
        block = doc['fracture']
        assert block['toughness_uncalibrated'] is True
        assert 'placeholder, uncalibrated' in block['toughness_knockdown']
        assert block['front']['Vp'] == pytest.approx(list(r.crack_front.Vp))
        assert block['front']['G_c_N_per_mm'] == pytest.approx(list(r.crack_front.G_c))
        FESolver.export_results(enf['clustered']['pristine'], tmp_path / 'none.json')
        none = json.loads((tmp_path / 'none.json').read_text())['fracture']
        assert none['toughness_uncalibrated'] is False
        assert none['toughness_knockdown'] == 'none (pristine toughness)'

    def test_no_delamination_details(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4, **TOUGHNESS)
        pf = PorosityField(mat, 0.02)
        mesh = CompositeMesh(pf, mat, nx=8, ny=4, nz=4)
        mesh.L_x, mesh.L_y = 8.0, 4.0
        mesh.generate_mesh()
        details = FESolver(mesh, mat, pf).solve('compression').summary().details
        assert details['toughness_knockdown'] is None
        assert details['toughness_uncalibrated'] is None

    def test_uncalibrated_info_once_per_process(self, caplog, monkeypatch):
        monkeypatch.setattr(toughness_module, '_uncalibrated_logged', False)
        caplog.set_level(logging.INFO, logger='porosity_fe_analysis')
        mesh, pf, solver, r = _enf(0.01)
        solver.solve('flexure_3pt', applied_load=-50.0)
        noted = [rec for rec in caplog.records if 'placeholder' in rec.getMessage()]
        assert len(noted) == 1 and noted[0].levelno == logging.INFO
        assert r.crack_front.toughness_uncalibrated is True
