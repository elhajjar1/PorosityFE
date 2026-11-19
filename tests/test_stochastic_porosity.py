"""Random porosity fields, the FE uncertainty study and the sensitivity ranking."""

import dataclasses
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from porosity_fe import (
    FORMAT_FE_UQ,
    MATERIALS,
    Delamination,
    FEUncertainty,
    PorosityField,
    PorosityMap,
    fe_porosity_uncertainty,
    load_results_from_json,
    propagate_uncertainty,
    save_fe_uncertainty_to_json,
    save_uq_results_to_json,
    sensitivity_ranking,
)

MAT4 = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)
CROSS_PLY = [0.0, 90.0, 90.0, 0.0]
#: A small mesh: a draw takes a fraction of a second.
SMALL = (20, 8, 4)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _logs(maps):
    g = np.log(np.array([m.Vp for m in maps]))
    return (g - g.mean(axis=0)) / g.std(axis=0)


class TestRandomMap:
    @pytest.mark.parametrize('kernel, rho', [('squared_exponential', math.exp(-0.5)),
                                             ('exponential', math.exp(-1.0))])
    def test_statistics_of_the_draws(self, kernel, rho):
        """Over 300 raw draws: the mean and CoV point by point, and the
        correlation at one correlation length, are the requested ones."""
        maps = [PorosityMap.random(50.0, 20.0, 0.03, cov=0.5, correlation_length=5.0,
                                   kernel=kernel, seed=s, preserve_mean=False)
                for s in range(300)]
        values = np.array([m.Vp for m in maps])
        assert values.mean() == pytest.approx(0.03, rel=0.03)
        assert (values.std(axis=0) / values.mean(axis=0)).mean() == pytest.approx(0.5, rel=0.05)
        g = _logs(maps)
        lag = round(5.0 / (maps[0].x[1] - maps[0].x[0]))
        assert (g[:, :-lag, :] * g[:, lag:, :]).mean() == pytest.approx(rho, abs=0.03)

    def test_anisotropic_correlation(self):
        maps = [PorosityMap.random(40.0, 40.0, 0.02, cov=0.5, correlation_length=(8.0, 2.0),
                                   resolution=1.0, seed=s) for s in range(200)]
        g = _logs(maps)
        along_x = (g[:, :-2, :] * g[:, 2:, :]).mean()
        along_y = (g[:, :, :-2] * g[:, :, 2:]).mean()
        assert along_x == pytest.approx(math.exp(-0.5 * (2.0 / 8.0) ** 2), abs=0.05)
        assert along_y == pytest.approx(math.exp(-0.5), abs=0.05)

    def test_seeded_and_mean_preserving(self):
        a = PorosityMap.random(50.0, 20.0, 0.02, cov=0.4, correlation_length=5.0, seed=3)
        b = PorosityMap.random(50.0, 20.0, 0.02, cov=0.4, correlation_length=5.0, seed=3)
        c = PorosityMap.random(50.0, 20.0, 0.02, cov=0.4, correlation_length=5.0, seed=4)
        np.testing.assert_array_equal(a.Vp, b.Vp)
        assert not np.array_equal(a.Vp, c.Vp)
        assert a.mean == pytest.approx(0.02, rel=1e-12)
        assert a.extent == (0.0, 50.0, 0.0, 20.0)
        assert a.x[1] - a.x[0] == pytest.approx(1.25)        # a quarter of l
        assert (a.Vp > 0.0).all()

    def test_degenerate_and_capped(self):
        flat = PorosityMap.random(50.0, 20.0, 0.02, cov=0.0, correlation_length=5.0, seed=1)
        assert np.all(flat.Vp == 0.02)
        zero = PorosityMap.random(50.0, 20.0, 0.0, cov=0.5, correlation_length=5.0, seed=1)
        assert np.all(zero.Vp == 0.0)
        fine = PorosityMap.random(50.0, 20.0, 0.02, cov=0.4, correlation_length=0.01, seed=1)
        assert fine.Vp.shape == (401, 401)                   # the cap per axis
        # A very rough field of a high mean hits the cap of 1 and keeps its mean.
        high = PorosityMap.random(50.0, 20.0, 0.4, cov=3.0, correlation_length=2.0, seed=2)
        assert high.max == 1.0 and high.mean == pytest.approx(0.4, rel=1e-9)
        raw = PorosityMap.random(50.0, 20.0, 0.4, cov=3.0, correlation_length=2.0, seed=2,
                                 preserve_mean=False)
        assert raw.max == 1.0 and raw.mean != pytest.approx(0.4, rel=1e-6)

    def test_unreachable_mean(self):
        with pytest.raises(ValueError, match='cannot keep its mean'):
            PorosityMap.random(50.0, 20.0, 0.95, cov=5.0, correlation_length=1.0, seed=1)

    @pytest.mark.parametrize('kwargs, match', [
        ({'cov': -0.1}, 'cov must be'), ({'cov': True}, 'cov must be'),
        ({'kernel': 'matern'}, 'Unknown kernel'),
        ({'correlation_length': 0.0}, 'correlation_length'),
        ({'correlation_length': (1.0, -2.0)}, 'correlation_length'),
        ({'resolution': 0.0}, 'resolution'), ({'mean': 2.0}, 'mean')])
    def test_bad_arguments(self, kwargs, match):
        args = {'cov': 0.4, 'correlation_length': 5.0, 'mean': 0.02, **kwargs}
        mean = args.pop('mean')
        with pytest.raises(ValueError, match=match):
            PorosityMap.random(50.0, 20.0, mean, **args)


class TestRandomField:
    def test_field_from_a_random_map(self):
        pf = PorosityField.random(MAT4, 0.03, cov=0.5, correlation_length=4.0,
                                  length=40.0, width=10.0, seed=11, distribution='clustered',
                                  cluster_location='surface')
        assert pf.seed == 11 and pf.Vp == pytest.approx(0.03, rel=1e-12)
        assert pf.distribution == 'clustered'
        assert pf.in_plane_map.extent == (0.0, 40.0, 0.0, 10.0)
        again = PorosityField.random(MAT4, 0.03, cov=0.5, correlation_length=4.0,
                                     length=40.0, width=10.0, seed=11)
        np.testing.assert_array_equal(pf.in_plane_map.Vp, again.in_plane_map.Vp)


class TestSensitivityRanking:
    def test_linear_model_shares(self):
        """y = 3 a + b with independent standard normals: shares 0.9 and 0.1."""
        rng = np.random.default_rng(0)
        a, b, c = rng.normal(size=(3, 2000))
        r = sensitivity_ranking({'b': b, 'a': a, 'c': c}, 3.0 * a + b, name='y')
        assert r.names[:2] == ('a', 'b') and r.output == 'y'
        assert r.share[0] == pytest.approx(0.9, abs=0.03)
        assert r.share[1] == pytest.approx(0.1, abs=0.03)
        assert abs(r.srrc[2]) < 0.05 and r.r2 > 0.95
        assert r.share.sum() == pytest.approx(1.0)
        assert np.all(np.abs(r.spearman) <= 1.0)

    def test_monotone_not_linear_and_sign(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 3.0, 500)
        r = sensitivity_ranking({'x': x, 'noise': rng.normal(size=500)}, -np.exp(4.0 * x))
        assert r.names[0] == 'x' and r.spearman[0] == pytest.approx(-1.0)
        assert r.r2 > 0.99

    def test_constant_inputs_and_non_finite_outputs(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=50)
        y = x.copy()
        y[:5] = np.inf
        r = sensitivity_ranking({'x': x, 'k': np.full(50, 2.0), 'r': 1.0 + 1e-14 * x}, y)
        assert r.constant == ('k', 'r') and r.n_samples == 45
        record = r.summary()
        assert record['ranking'][0]['input'] == 'x'
        assert record['constant_inputs'] == ['k', 'r']
        json.dumps(record)

    @pytest.mark.parametrize('inputs, output, match', [
        ({'x': [1.0, 2.0]}, [1.0, 2.0, 3.0], 'has 2 draws'),
        ({'x': [1.0, np.nan, 2.0]}, [1.0, 2.0, 3.0], 'non-finite'),
        ({'x': [1.0, 1.0, 1.0]}, [1.0, 2.0, 3.0], 'No input varies'),
        ({'x': [1.0, 2.0, 3.0], 'z': [3.0, 1.0, 2.0]}, [1.0, 2.0, 3.0], 'at least 4'),
        ({'x': [1.0, 2.0, 3.0]}, [5.0, 5.0, 5.0], 'does not vary'),
    ])
    def test_refusals(self, inputs, output, match):
        with pytest.raises(ValueError, match=match):
            sensitivity_ranking(inputs, output)


class TestEmpiricalRanking:
    def test_shares_follow_the_first_order_variances(self):
        """KD = exp(-alpha Vp): the variance of ln KD is (alpha sd_Vp)^2 +
        (Vp sd_alpha)^2, so with CoVs 0.2 on Vp and 0.1 on alpha the shares
        are 0.8 and 0.2; a strength does not move the knockdown."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = propagate_uncertainty(0.03, 'T800_epoxy', 'compression',
                                      covs={'sigma_1c': 0.05}, vp_cov=0.2, coef_cov=0.1,
                                      n_samples=400, seed=1, method='lhs')
        kd = r['sensitivity']['knockdown']
        shares = {row['input']: row['share'] for row in kd['ranking']}
        assert shares['Vp'] == pytest.approx(0.8, abs=0.05)
        assert shares['judd_wright_alpha'] == pytest.approx(0.2, abs=0.05)
        assert shares['sigma_1c'] < 0.01
        fs = r['sensitivity']['failure_stress']
        assert fs['ranking'][0]['input'] == 'sigma_1c'
        inputs = r['samples']['inputs']
        assert set(inputs) == {'sigma_1c', 'judd_wright_alpha', 'Vp'}
        assert inputs['Vp'].shape == (400,)

    def test_unrankable_outputs_are_none(self, tmp_path):
        r = propagate_uncertainty(0.03, 'T800_epoxy', 'compression',
                                  covs={'sigma_1c': 0.05}, n_samples=30, seed=1)
        assert r['sensitivity']['knockdown'] is None       # the knockdown does not vary
        assert r['sensitivity']['failure_stress']['ranking'][0]['input'] == 'sigma_1c'
        path = tmp_path / 'uq.json'
        save_uq_results_to_json({'compression': r}, path, include_samples=True)
        doc = load_results_from_json(path)
        assert doc['uq']['compression']['sensitivity']['knockdown'] is None
        assert len(doc['uq']['compression']['samples']['inputs']['sigma_1c']) == 30

    def test_toughness_mode_coefficient(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], G_Ic=0.3, G_IIc=0.9, bk_eta=2.0)
        r = propagate_uncertainty(0.03, mat, 'mode_I_toughness', coef_cov=0.2, vp_cov=0.1,
                                  n_samples=60, seed=2)
        names = {row['input'] for row in r['sensitivity']['knockdown']['ranking']}
        assert names == {'b_I', 'Vp'}


class TestFEUncertainty:
    @pytest.fixture(scope='class')
    @staticmethod
    def study():
        return fe_porosity_uncertainty(MAT4, 0.03, 'compression', cov=0.8,
                                       correlation_length=3.0, n_samples=12, seed=5,
                                       ply_angles=CROSS_PLY, mesh_res=SMALL)

    def test_outputs_and_inputs(self, study):
        assert isinstance(study, FEUncertainty)
        assert study.outputs == ('knockdown', 'effective_modulus',
                                 'first_ply_failure_load_factor',
                                 'interior_first_ply_failure_load_factor',
                                 'max_failure_index')
        assert study.mesh == SMALL and study.n_samples == 12
        assert all(v.shape == (12,) for v in study.samples.values())
        np.testing.assert_allclose(study.inputs['Vp'], 0.03)
        assert np.all(study.inputs['Vp_peak'] > 0.03)
        assert np.all(study.inputs['Vp_spatial_cov'] > 0.2)

    def test_stiffness_holds_and_strength_spreads(self, study):
        """At a fixed average, rearranging the voids leaves the stiffness
        where it was and spreads the first-ply failure load."""
        kd = study.samples['knockdown']
        assert np.ptp(kd) / kd.mean() < 2e-3
        assert kd.mean() == pytest.approx(study.nominal['knockdown'], rel=2e-3)
        fpf = study.samples['interior_first_ply_failure_load_factor']
        assert np.ptp(fpf) / fpf.mean() > 2e-3
        stats = study.statistics('interior_first_ply_failure_load_factor')
        assert stats['n_finite'] == 12
        assert stats['percentiles']['p5'] <= stats['percentiles']['p50'] <= stats['max']

    def test_reproducible(self, study):
        again = fe_porosity_uncertainty(MAT4, 0.03, 'compression', cov=0.8,
                                        correlation_length=3.0, n_samples=3, seed=5,
                                        ply_angles=CROSS_PLY, mesh_res=SMALL)
        for k in again.outputs:
            np.testing.assert_array_equal(again.samples[k], study.samples[k][:3])

    def test_material_scatter_is_ranked_first(self):
        study = fe_porosity_uncertainty(MAT4, 0.03, 'compression', cov=0.3,
                                        correlation_length=5.0, covs={'sigma_1c': 0.1},
                                        vp_cov=0.3, n_samples=16, seed=2, method='lhs',
                                        ply_angles=[0.0] * 4, mesh_res=SMALL)
        assert study.ranking('first_ply_failure_load_factor').names[0] == 'sigma_1c'
        knockdown = study.ranking('knockdown')
        assert knockdown.names[0] == 'Vp' and knockdown.r2 > 0.9
        assert study.inputs['sigma_1c'].std() > 0.0
        assert study.spec == {'sigma_1c': ['lognormal', 0.1]}

    def test_probability_and_summary(self, study, tmp_path):
        import jsonschema
        fpf = study.samples['first_ply_failure_load_factor']
        assert study.probability_below('first_ply_failure_load_factor', np.median(fpf)) \
            == pytest.approx(0.5, abs=0.1)
        assert study.probability_below('first_ply_failure_load_factor', 0.0) == 0.0
        path = tmp_path / 'fe_uq.json'
        save_fe_uncertainty_to_json(study, path, include_samples=True)
        doc = load_results_from_json(path)
        schema = json.loads((Path(__file__).resolve().parent.parent / 'validation' /
                             'schemas' / 'porosity_results_schema.json').read_text())
        jsonschema.validate(instance=doc, schema=schema)
        assert doc['format'] == FORMAT_FE_UQ and doc['provenance']['seed'] == 5
        block = doc['fe_uq']
        assert block['random_field']['cov'] == 0.8
        assert len(block['samples']['knockdown']) == 12
        assert set(block['sensitivity']) == set(study.outputs)
        save_fe_uncertainty_to_json(study, path)
        assert 'samples' not in load_results_from_json(path)['fe_uq']

    def test_with_a_delamination(self):
        mat = dataclasses.replace(MAT4, G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)
        study = fe_porosity_uncertainty(
            mat, 0.03, 'tension', cov=0.6, correlation_length=4.0, n_samples=3, seed=1,
            ply_angles=CROSS_PLY, mesh_res=SMALL,
            delaminations=[Delamination.strip(0, 20.0, 30.0)])
        assert 'fracture_load_factor' in study.outputs
        assert np.all(np.isfinite(study.samples['fracture_load_factor']))

    def test_unrankable_and_unknown_outputs(self, study):
        record = study.summary()
        assert record['sensitivity']['knockdown'] is not None
        with pytest.raises(ValueError, match='Unknown output'):
            study.statistics('stress')
        empty = dataclasses.replace(study, samples={**study.samples,
                                                    'knockdown': np.full(12, np.inf)})
        assert empty.statistics('knockdown')['mean'] is None
        assert empty.summary()['sensitivity']['knockdown'] is None

    @pytest.mark.parametrize('args, kwargs, match', [
        (('nope', 0.03), {}, 'Unknown material'),
        ((MAT4, 0.03, 'ilss'), {}, 'coupon load cases'),
        ((MAT4, 1.5), {}, 'Vp'),
        ((MAT4, 0.03), {'n_samples': 1}, 'n_samples'),
        ((MAT4, 0.03), {'method': 'sobol'}, 'sampling method'),
        ((MAT4, 0.03), {'kernel': 'matern'}, 'Unknown kernel'),
        ((MAT4, 0.03), {'vp_cov': -1.0}, 'vp_cov'),
        ((MAT4, 0.03), {'config': {'seed': 1}}, "cannot set 'seed'"),
        ((MAT4, 0.03), {'covs': {'nope': 0.1}}, 'non-perturbable'),
    ])
    def test_refusals(self, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            fe_porosity_uncertainty(*args, cov=0.5, correlation_length=5.0,
                                    mesh_res=SMALL, **kwargs)

    def test_preset_by_name_and_raw_draws(self):
        study = fe_porosity_uncertainty('T800_epoxy', 0.02, 'shear', cov=0.5,
                                        correlation_length=5.0, n_samples=2, seed=3,
                                        preserve_mean=False, mesh_res=(6, 2, 4),
                                        ply_angles='QI')
        assert study.material == 'T800_epoxy'
        assert not np.allclose(study.inputs['Vp'], 0.02)

    def test_warnings_of_the_draws_are_raised_once(self):
        """Every draw on a coarse mesh warns of its distorted elements: the
        study raises it once. Without voids every draw is the nominal."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            study = fe_porosity_uncertainty(MAT4, 0.0, 'tension', cov=0.5,
                                            correlation_length=5.0, n_samples=3, seed=1,
                                            ply_angles=CROSS_PLY, mesh_res=(2, 2, 4))
        distorted = [w for w in caught if 'highly distorted' in str(w.message)]
        assert len(distorted) == 1
        np.testing.assert_array_equal(study.inputs['Vp_spatial_cov'], 0.0)
        np.testing.assert_allclose(study.samples['knockdown'], 1.0)


def test_example_map_is_of_the_study():
    study = fe_porosity_uncertainty(MAT4, 0.03, 'tension', cov=0.4, correlation_length=6.0,
                                    n_samples=2, seed=1, ply_angles=CROSS_PLY, mesh_res=SMALL,
                                    length=40.0, width=10.0)
    m = study.example_map()
    assert m.extent == (0.0, 40.0, 0.0, 10.0) and m.mean == pytest.approx(0.03)
    np.testing.assert_array_equal(m.Vp, study.example_map().Vp)
    assert not np.array_equal(m.Vp, study.example_map(seed=5).Vp)
