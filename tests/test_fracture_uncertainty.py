"""Uncertainty band on the delamination load factor (delamination plan 4.5, #213).

:func:`fracture_load_factor_uncertainty` re-evaluates ``lambda_c`` at every
front node for each draw of the toughness inputs on the energy release
rates of one FE solve. This checks:

- zero scatter gives exactly the deterministic ``lambda_c``, with and
  without a thermal prestress;
- every draw equals :func:`fracture_load_factors` on that draw's inputs
  (material toughness, B-K exponent, law coefficients; exponential and
  linear laws; the unset ``G_IIIc`` / ``b_III`` following mode II);
- on a pure mode I double-cantilever beam a lognormal ``G_Ic`` gives the
  percentiles ``lam0 exp(sigma Phi^-1(p) / 2)``;
- with a prestress each draw takes its own root, as the scalar one does;
- seeds reproduce, the inputs are validated, infinite factors stay
  defined, and the JSON block is strict and validates against the schema;
- ``propagate_uncertainty`` shares the sampling code (its own outputs are
  covered by tests/test_uq.py).

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
from scipy.stats import norm

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    CrackFront,
    Delamination,
    FESolver,
    FractureLoadFactorBand,
    PorosityField,
    ToughnessKnockdown,
    fracture_load_factor_uncertainty,
    fracture_load_factors,
    load_results_from_json,
)
from porosity_fe._sampling import _draw_unit_samples, _perturbed_value, _unit_to_draw
from porosity_fe.uq import _percentile

_SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'validation', 'schemas', 'porosity_results_schema.json')

#: IM7/8552 toughness and B-K exponent (N/mm), Krueger NASA/CR-2012-217562,
#: as in tests/test_vcct.py.
TOUGHNESS = dict(G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)
MAT = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=16, **TOUGHNESS)
MID16 = 7

#: Cross-ply AS4/3501-6 (it has CTEs) of tests/test_vcct.py's thermal case.
TMAT = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=8, G_Ic=0.08,
                           G_IIc=0.6, bk_eta=2.0)
LAYUP = [0, 90, 0, 90, 90, 0, 90, 0]

_MATERIAL_FIELDS = ('G_Ic', 'G_IIc', 'G_IIIc', 'bk_eta')


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope='module')
def enf():
    """ENF beam with interface-clustered voids (front Vp > 0, so the
    placeholder law lowers the toughness), crack to x = 15 mm."""
    pf = PorosityField(MAT, 0.02, distribution='interface')
    mesh = CompositeMesh.for_flexure(
        pf, MAT, span=50.0, width=4.0, overhang=5.0, load_width=0.0, nz=16, ny=2,
        element_length=1.0, ply_angles='UD',
        delaminations=[Delamination.strip(MID16, -1.0, 15.0)])
    solver = FESolver(mesh, MAT, pf, ply_angles='UD')
    return {'mesh': mesh, 'solver': solver,
            'r': solver.solve('flexure_3pt', applied_load=-100.0)}


@pytest.fixture(scope='module')
def combined():
    """Three-point bending with a 100 K cool-down on the cross-ply beam: the
    front carries the mechanical and the residual part."""
    pf = PorosityField(TMAT, 0.02, distribution='interface')
    mesh = CompositeMesh.for_flexure(
        pf, TMAT, span=20.0, width=4.0, overhang=2.0, load_width=0.0, nz=8, ny=2,
        element_length=0.5, ply_angles=LAYUP,
        delaminations=[Delamination.strip(1, -1.0, 8.0)])
    solver = FESolver(mesh, TMAT, pf, ply_angles=LAYUP)
    return solver.solve('flexure_3pt', applied_load=-20.0, delta_T=-100.0)


@pytest.fixture(scope='module')
def dcb():
    """A symmetric UD double-cantilever beam: a pure mode I front."""
    mat = dataclasses.replace(MATERIALS['T800_epoxy'], **TOUGHNESS, t_ply=3.0 / 8, n_plies=8)
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh.for_dcb(pf, mat, crack_length=30.0, length=100.0, width=20.0,
                                 element_length=2.0, ny=2)
    return mat, FESolver(mesh, mat, pf, ply_angles='UD').solve('dcb', applied_load=10.0)


def _draws(band: FractureLoadFactorBand, seed, method='monte_carlo'):
    """The inputs of every draw of ``band``, rebuilt from its seed: one dict
    per draw, the shared sampling helpers and the clip of each field."""
    names = list(band.spec)
    unit = _draw_unit_samples(len(names), band.n_samples, method,
                              np.random.default_rng(seed))
    out = []
    for i in range(band.n_samples):
        values = {}
        for j, name in enumerate(names):
            dist, params = band.spec[name]
            z = float(_unit_to_draw(unit[i:i + 1, j], dist)[0])
            nominal = band.inputs[name]
            floor = 1e-6 * nominal if name in _MATERIAL_FIELDS else 0.0
            values[name] = max(float(_perturbed_value(nominal, z, dist, params)), floor)
        out.append(values)
    return out


def _draw_inputs(material, law, values):
    """The material and law of one draw."""
    mat = dataclasses.replace(material, **{k: v for k, v in values.items()
                                           if k in _MATERIAL_FIELDS})
    coefs = {k: v for k, v in values.items() if k.startswith('b_')}
    if coefs:
        law = dataclasses.replace(law, **coefs)
    return mat, law


def _stub(G_I, G_II, G_III, Vp):
    """A CrackFront stub with these rates (as in tests/test_toughness.py)."""
    G_I, G_II, G_III = (np.asarray(g, float) for g in (G_I, G_II, G_III))
    n = G_I.size
    force = -np.ones((n, 3))
    opening = np.column_stack([2.0 * G_II, 2.0 * G_I, 2.0 * G_III])
    G_T = G_I + G_II + G_III
    with np.errstate(invalid='ignore', divide='ignore'):
        psi = np.where(G_T > 0, (G_II + G_III) / np.where(G_T > 0, G_T, 1.0), np.nan)
    return CrackFront(
        node=np.arange(n), duplicate=np.arange(n) + n, pair=np.arange(n),
        interface=np.zeros(n, dtype=int), behind=np.arange(n), x=np.zeros(n),
        y=np.zeros(n), z=np.zeros(n), direction=np.tile([1.0, 0.0], (n, 1)),
        delta_a=np.ones(n), width=np.ones(n), force_local=force, opening_local=opening,
        G_I=G_I, G_II=G_II, G_III=G_III, G_T=G_T, psi=psi, Vp=np.asarray(Vp, float))


# --------------------------------------------------------------------------
# Exactness and agreement with the deterministic factor
# --------------------------------------------------------------------------

class TestAgainstTheDeterministicFactor:
    def test_zero_scatter_is_exact(self, enf, combined):
        """Nothing named, or every CoV zero: every draw is lambda_c bit for
        bit, with and without a prestress."""
        for r, mat in ((enf['r'], MAT), (combined, TMAT)):
            lam = r.crack_front.lambda_c
            for kwargs in ({}, {'covs': {'G_Ic': 0.0, 'b_II': 0.0}},
                           {'spec': {'bk_eta': ('uniform', 0.0)}}):
                band = r.fracture_load_factor_uncertainty(
                    mat, n_samples=7, seed=3, **kwargs)
                assert band.spec == {}
                np.testing.assert_array_equal(band.samples_per_node, np.tile(lam, (7, 1)))
                np.testing.assert_array_equal(band.samples, np.full(7, r.fracture_load_factor))
                np.testing.assert_array_equal(band.nominal_per_node, lam)
                assert band.nominal == r.fracture_load_factor
                # np.std of equal values rounds to a few ULP, not to 0.
                assert band.std <= 1e-15 * band.nominal
                assert all(v == r.fracture_load_factor for v in band.percentiles.values())

    @pytest.mark.parametrize('scatter', [
        {'covs': {'G_Ic': 0.1, 'G_IIc': 0.07, 'bk_eta': 0.1, 'b_I': 0.2, 'b_II': 0.25}},
        {'spec': {'G_IIc': ('normal', 0.3), 'b_II': ('uniform', 0.5)}},
        {'covs': {'G_IIc': 0.2}, 'method': 'lhs'},
    ])
    def test_every_draw_is_the_deterministic_factor_of_its_inputs(self, enf, scatter):
        r = enf['r']
        front = r.crack_front
        method = scatter.get('method', 'monte_carlo')
        band = r.fracture_load_factor_uncertainty(MAT, n_samples=60, seed=5, **scatter)
        law = front.toughness_knockdown
        for i, values in enumerate(_draws(band, 5, method)):
            mat, law_i = _draw_inputs(MAT, law, values)
            expected = fracture_load_factors(front, mat, toughness_knockdown=law_i).lambda_c
            np.testing.assert_allclose(band.samples_per_node[i], expected, rtol=1e-13)
        np.testing.assert_array_equal(band.samples, band.samples_per_node.min(axis=1))

    def test_unset_mode_III_inputs_follow_mode_II(self):
        """An unset G_IIIc follows the G_IIc draws and an unset b_III the
        b_II draws; set ones scatter on their own or stay fixed."""
        front = _stub([0.1, 0.0, 0.02], [0.0, 0.1, 0.05], [0.0, 0.1, 0.08], [0.01, 0.02, 0.03])
        law = ToughnessKnockdown()
        band = fracture_load_factor_uncertainty(
            front, MAT, covs={'G_IIc': 0.2, 'b_II': 0.3}, n_samples=40, seed=1)
        for i, values in enumerate(_draws(band, 1)):
            mat, law_i = _draw_inputs(MAT, law, values)
            assert mat.G_IIIc is None and law_i.b_III is None      # follow mode II
            np.testing.assert_allclose(
                band.samples_per_node[i],
                fracture_load_factors(front, mat, toughness_knockdown=law_i).lambda_c,
                rtol=1e-13)
        own = dataclasses.replace(MAT, G_IIIc=1.1)
        own_law = ToughnessKnockdown(b_III=9.0)
        band = fracture_load_factor_uncertainty(
            front, own, toughness_knockdown=own_law,
            covs={'G_IIc': 0.2, 'b_II': 0.3, 'G_IIIc': 0.1, 'b_III': 0.2},
            n_samples=40, seed=2)
        assert band.inputs['G_IIIc'] == 1.1 and band.inputs['b_III'] == 9.0
        for i, values in enumerate(_draws(band, 2)):
            mat, law_i = _draw_inputs(own, own_law, values)
            np.testing.assert_allclose(
                band.samples_per_node[i],
                fracture_load_factors(front, mat, toughness_knockdown=law_i).lambda_c,
                rtol=1e-13)

    def test_linear_and_callable_laws(self):
        front = _stub([0.1, 0.02], [0.01, 0.05], [0.0, 0.01], [0.02, 0.04])
        linear = ToughnessKnockdown('linear', b_I=5.0, b_II=12.0)
        band = fracture_load_factor_uncertainty(
            front, MAT, toughness_knockdown=linear, covs={'b_I': 0.3, 'G_Ic': 0.1},
            n_samples=30, seed=4)
        for i, values in enumerate(_draws(band, 4)):
            mat, law_i = _draw_inputs(MAT, linear, values)
            np.testing.assert_allclose(
                band.samples_per_node[i],
                fracture_load_factors(front, mat, toughness_knockdown=law_i).lambda_c,
                rtol=1e-13)
        # A callable law takes material scatter; it has no coefficient.
        def mine(Vp, mode):
            return 1.0 - 2.0 * Vp
        band = fracture_load_factor_uncertainty(front, MAT, toughness_knockdown=mine,
                                                covs={'G_Ic': 0.1}, n_samples=20, seed=0)
        assert band.inputs['b_I'] is None and not band.toughness_uncalibrated
        with pytest.raises(ValueError, match='has no coefficient'):
            fracture_load_factor_uncertainty(front, MAT, toughness_knockdown=mine,
                                             covs={'b_I': 0.1})
        with pytest.raises(ValueError, match='has no coefficient'):
            fracture_load_factor_uncertainty(front, MAT, toughness_knockdown=None,
                                             covs={'b_II': 0.1})

    def test_clipped_tail_draws(self):
        """A wide normal draw is clipped as MaterialProperties.perturb clips
        it (a positive floor on G_Ic) and b at 0 (the law never raises the
        toughness), so every draw is a valid factor of clipped inputs."""
        front = _stub([0.1], [0.05], [0.0], [0.03])
        band = fracture_load_factor_uncertainty(
            front, MAT, spec={'G_Ic': ('normal', 2.0), 'b_I': ('normal', 3.0)},
            n_samples=400, seed=0)
        assert np.all(np.isfinite(band.samples)) and np.all(band.samples > 0.0)
        draws = _draws(band, 0)
        assert any(d['G_Ic'] == 1e-6 * MAT.G_Ic for d in draws)    # floored
        assert any(d['b_I'] == 0.0 for d in draws)                  # clipped at 0
        for i, values in enumerate(draws):
            mat, law_i = _draw_inputs(MAT, ToughnessKnockdown(), values)
            np.testing.assert_allclose(
                band.samples_per_node[i],
                fracture_load_factors(front, mat, toughness_knockdown=law_i).lambda_c,
                rtol=1e-13)
        # The floor is the one of MaterialProperties.perturb.
        assert MAT.perturb({'G_Ic': -10.0}, {'G_Ic': ('normal', 2.0)}).G_Ic == \
            1e-6 * MAT.G_Ic


# --------------------------------------------------------------------------
# Pure mode I: a closed form
# --------------------------------------------------------------------------

class TestPureModeI:
    @pytest.mark.parametrize('cov', [0.1, 0.3])
    def test_lognormal_G_Ic_percentiles(self, dcb, cov):
        """lambda_c = sqrt(G_Ic f / G_T) on a pure mode I front, so a
        median-preserving lognormal G_Ic with sigma = sqrt(ln(1 + CoV^2))
        gives the percentiles lam0 exp(sigma Phi^-1(p) / 2)."""
        mat, r = dcb
        front = r.crack_front
        assert np.max(np.abs(front.psi)) < 1e-12
        lam0 = r.fracture_load_factor
        band = r.fracture_load_factor_uncertainty(
            mat, covs={'G_Ic': cov}, n_samples=4000, method='lhs', seed=0)
        sigma = math.sqrt(math.log1p(cov * cov))
        for p in (5, 50, 95):
            expected = lam0 * math.exp(sigma * norm.ppf(p / 100.0) / 2.0)
            assert band.percentiles[f'p{p}'] == pytest.approx(expected, rel=1e-3)
        # Each draw exactly: lam0 sqrt(G_Ic_draw / G_Ic).
        g = np.array([d['G_Ic'] for d in _draws(band, 0, 'lhs')])
        np.testing.assert_allclose(band.samples, lam0 * np.sqrt(g / mat.G_Ic), rtol=1e-12)
        assert band.load_factor_basis == 'mechanical'
        assert band.inputs['G_Ic'] == mat.G_Ic


# --------------------------------------------------------------------------
# Thermal prestress
# --------------------------------------------------------------------------

class TestPrestress:
    def test_each_draw_takes_its_own_root(self, combined):
        """Under u = u_T + lam u_M the draws are not a rescaling: each is the
        root of G_T(lam) = G_c(psi(lam)) on its inputs (fracture_load_factors
        with prestress, 1e-11)."""
        front = combined.crack_front
        moving, fixed = front.parts
        band = combined.fracture_load_factor_uncertainty(
            TMAT, covs={'G_Ic': 0.2, 'G_IIc': 0.1, 'bk_eta': 0.1, 'b_I': 0.3},
            n_samples=80, seed=4)
        assert band.load_factor_basis == 'mechanical_with_residual'
        assert band.nominal == combined.fracture_load_factor
        law = front.toughness_knockdown
        for i, values in enumerate(_draws(band, 4)):
            mat, law_i = _draw_inputs(TMAT, law, values)
            expected = fracture_load_factors(moving, mat, prestress=fixed,
                                             toughness_knockdown=law_i).lambda_c
            np.testing.assert_allclose(band.samples_per_node[i], expected, rtol=1e-11)

    def test_explicit_prestress_equals_the_stored_parts(self, combined):
        moving, fixed = combined.crack_front.parts
        kwargs = dict(covs={'G_Ic': 0.15}, n_samples=50, seed=8,
                      toughness_knockdown=combined.crack_front.toughness_knockdown)
        a = fracture_load_factor_uncertainty(moving, TMAT, prestress=fixed, **kwargs)
        b = fracture_load_factor_uncertainty(combined.crack_front, TMAT, **kwargs)
        np.testing.assert_array_equal(a.samples_per_node, b.samples_per_node)
        assert a.load_factor_basis == b.load_factor_basis == 'mechanical_with_residual'
        with pytest.raises(ValueError, match='same front nodes'):
            fracture_load_factor_uncertainty(
                moving, TMAT, prestress=_stub([0.1], [0.0], [0.0], [0.0]))


# --------------------------------------------------------------------------
# The band object, sampling and inputs
# --------------------------------------------------------------------------

class TestBand:
    def test_seeds_and_methods(self, enf):
        r = enf['r']
        kw = dict(covs={'G_Ic': 0.1, 'G_IIc': 0.1}, n_samples=50)
        a = r.fracture_load_factor_uncertainty(MAT, seed=11, **kw)
        b = r.fracture_load_factor_uncertainty(MAT, seed=11, **kw)
        c = r.fracture_load_factor_uncertainty(MAT, seed=12, **kw)
        np.testing.assert_array_equal(a.samples_per_node, b.samples_per_node)
        assert a.summary() == b.summary()
        assert a == a and a != b              # identity: the band holds arrays
        assert not np.array_equal(a.samples, c.samples)
        lhs = r.fracture_load_factor_uncertainty(MAT, seed=11, method='lhs', **kw)
        np.testing.assert_array_equal(
            lhs.samples, r.fracture_load_factor_uncertainty(MAT, seed=11, method='lhs',
                                                            **kw).samples)
        assert not np.array_equal(lhs.samples, a.samples)
        assert (a.seed, a.method, lhs.method) == (11, 'monte_carlo', 'lhs')

    def test_contents(self, enf):
        r = enf['r']
        front = r.crack_front
        band = r.fracture_load_factor_uncertainty(
            MAT, covs={'G_IIc': 0.2, 'b_II': 0.3}, n_samples=500, seed=0,
            percentiles=(10, 50, 90), required_load_factor=2.5)
        assert band.n_front_nodes == len(front)
        np.testing.assert_array_equal(band.node, front.node)
        assert band.samples.shape == (500,) and band.samples_per_node.shape == (500, len(front))
        assert set(band.percentiles) == {'p10', 'p50', 'p90'}
        assert band.percentiles['p10'] < band.percentiles['p50'] < band.percentiles['p90']
        assert band.mean == pytest.approx(float(np.mean(band.samples)), rel=1e-15)
        assert band.min <= band.nominal <= band.max
        assert band.governing_fraction.sum() == pytest.approx(1.0)
        assert band.governing_fraction[front.governing] > 0.0
        assert band.toughness_uncalibrated is True               # the placeholders
        assert band.inputs == {'G_Ic': 0.212, 'G_IIc': 0.774, 'G_IIIc': 0.774,
                               'bk_eta': 2.1, 'b_I': 6.2, 'b_II': 14.0, 'b_III': 14.0}
        assert band.spec == {'G_IIc': ('lognormal', 0.2), 'b_II': ('lognormal', 0.3)}
        # The share of draws below a factor: information, not a disposition.
        assert band.probability_below(2.5) == float(np.mean(band.samples < 2.5))
        assert band.probability_below_required == band.probability_below(2.5)
        assert 0.0 < band.probability_below(band.percentiles['p50']) <= 0.5 + 1e-9
        assert band.probability_below(1e-3) == 0.0 and band.probability_below(1e3) == 1.0
        for bad in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match='finite positive'):
                band.probability_below(bad)
        assert r.fracture_load_factor_uncertainty(MAT, n_samples=3) \
            .probability_below_required is None
        assert not any('disposition' in k for k in band.summary())

    def test_infinite_factors_stay_defined(self):
        """A node the load does not drive is inf in every draw and never
        governs; a front with none driven has an infinite band, null in the
        summary."""
        front = _stub([0.1, 0.0], [0.05, 0.0], [0.0, 0.0], [0.01, 0.0])
        band = fracture_load_factor_uncertainty(front, MAT, covs={'G_Ic': 0.1},
                                                n_samples=20, seed=0)
        assert np.all(np.isinf(band.samples_per_node[:, 1]))
        np.testing.assert_array_equal(band.governing_fraction, [1.0, 0.0])
        idle = _stub([0.0], [0.0], [0.0], [0.0])
        band = fracture_load_factor_uncertainty(idle, MAT, covs={'G_Ic': 0.1},
                                                n_samples=10, seed=0, required_load_factor=1.0)
        assert band.nominal == math.inf and np.all(np.isinf(band.samples))
        assert band.probability_below_required == 0.0
        assert band.governing_fraction.sum() == 0.0
        summary = band.summary()
        assert summary['nominal'] is None and summary['mean'] is None
        assert summary['percentiles'] == {'p5': None, 'p50': None, 'p95': None}
        assert summary['nominal_per_node'] == [None]
        json.dumps(summary, allow_nan=False)                     # strict JSON

    def test_percentile_with_an_infinite_tail(self):
        s = np.array([1.0, 2.0, 3.0, math.inf])
        assert _percentile(s, 0) == 1.0 and _percentile(s, 50) == 2.5
        assert _percentile(s, 100) == math.inf and _percentile(s, 95) == math.inf
        finite = np.array([3.0, 1.0, 2.0])
        assert _percentile(finite, 37.0) == float(np.percentile(finite, 37.0))

    def test_invalid_inputs(self, enf):
        r = enf['r']
        front = r.crack_front
        with pytest.raises(ValueError, match='only the toughness inputs'):
            r.fracture_load_factor_uncertainty(MAT, covs={'E11': 0.05})
        with pytest.raises(ValueError, match='only the toughness inputs'):
            r.fracture_load_factor_uncertainty(MAT, spec={'Vp': ('normal', 0.1)})
        with pytest.raises(ValueError, match="'G_IIIc' is None"):
            r.fracture_load_factor_uncertainty(MAT, covs={'G_IIIc': 0.1})
        with pytest.raises(ValueError, match='b_III is not set'):
            r.fracture_load_factor_uncertainty(MAT, covs={'b_III': 0.1})
        with pytest.raises(ValueError, match='CoV'):
            r.fracture_load_factor_uncertainty(MAT, covs={'G_Ic': -0.1})
        with pytest.raises(ValueError, match='unknown distribution'):
            r.fracture_load_factor_uncertainty(MAT, spec={'G_Ic': ('weibull', 0.1)})
        with pytest.raises(ValueError, match='Unknown sampling method'):
            r.fracture_load_factor_uncertainty(MAT, method='sobol')
        for n in (0, -3, 2.5, True):
            with pytest.raises(ValueError, match='n_samples'):
                r.fracture_load_factor_uncertainty(MAT, n_samples=n)
        with pytest.raises(ValueError, match='percentiles'):
            r.fracture_load_factor_uncertainty(MAT, percentiles=(5, 101))
        with pytest.raises(ValueError, match='required_load_factor'):
            r.fracture_load_factor_uncertainty(MAT, required_load_factor=0.0)
        with pytest.raises(ValueError, match='G_Ic'):
            fracture_load_factor_uncertainty(front, MATERIALS['T800_epoxy'])
        # The method keeps the band on the solve's own load factor.
        with pytest.raises(ValueError, match='does not reproduce'):
            r.fracture_load_factor_uncertainty(dataclasses.replace(MAT, G_IIc=0.6))

    def test_method_needs_a_front_and_toughness(self, enf):
        plain = FESolver(CompositeMesh(PorosityField(MAT, 0.0), MAT, nx=4, ny=2, nz=4),
                         MAT, PorosityField(MAT, 0.0)).solve('tension')
        with pytest.raises(ValueError, match='no delamination front'):
            plain.fracture_load_factor_uncertainty(MAT)
        no_tough = dataclasses.replace(MAT, G_Ic=None, G_IIc=None, bk_eta=None)
        solver = FESolver(enf['mesh'], no_tough, enf['mesh'].porosity_field, ply_angles='UD')
        r = solver.solve('flexure_3pt', applied_load=-100.0)
        with pytest.raises(ValueError, match='no fracture load factor'):
            r.fracture_load_factor_uncertainty(MAT)


# --------------------------------------------------------------------------
# JSON export
# --------------------------------------------------------------------------

class TestExport:
    def test_json_block_validates(self, enf, combined, tmp_path):
        import jsonschema
        with open(_SCHEMA, encoding='utf-8') as fh:
            schema = json.load(fh)
        for name, r, mat in (('enf', enf['r'], MAT), ('combined', combined, TMAT)):
            band = r.fracture_load_factor_uncertainty(
                mat, covs={'G_Ic': 0.05, 'G_IIc': 0.07}, n_samples=200, method='lhs',
                seed=1, required_load_factor=1.5)
            path = tmp_path / f'{name}.json'
            FESolver.export_results(r, path, fracture_uncertainty=band)
            text = path.read_text(encoding='utf-8')
            doc = json.loads(text, parse_constant=lambda t: pytest.fail(f'non-standard {t}'))
            jsonschema.validate(doc, schema)
            assert doc['schema_version'] == '1.3'
            block = doc['fracture_uncertainty']
            assert block == json.loads(json.dumps(band.summary()))
            assert block['nominal'] == r.fracture_load_factor
            assert block['spec'] == {'G_Ic': ['lognormal', 0.05], 'G_IIc': ['lognormal', 0.07]}
            assert block['load_factor_basis'] == r.load_factor_basis
            assert block['probability_below_required'] == band.probability_below(1.5)
            assert load_results_from_json(path)['fracture_uncertainty'] == block
            # Opt-in: the default export has no block, and is the export of
            # before (the fracture block included).
            plain = tmp_path / f'{name}_plain.json'
            FESolver.export_results(r, plain)
            plain_doc = json.loads(plain.read_text(encoding='utf-8'))
            assert 'fracture_uncertainty' not in plain_doc
            doc.pop('fracture_uncertainty')
            for d in (doc, plain_doc):
                d.pop('provenance')
            assert doc == plain_doc
