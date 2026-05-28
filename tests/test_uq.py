#!/usr/bin/env python3
"""Tests for porosity_fe.uq.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import json

from porosity_fe_analysis import (propagate_uncertainty)


class TestPropagateUncertainty:
    """Monte Carlo / LHS uncertainty propagation around get_failure_load."""

    _N = 32  # tiny sample count keeps the suite fast

    def test_result_keys_and_shapes(self):
        r = propagate_uncertainty(
            0.02, 'T800_epoxy', covs={'sigma_1c': 0.08, 'E22': 0.05},
            n_samples=self._N, seed=42)
        for key in ('failure_stress', 'knockdown', 'nominal', 'samples',
                    'seed', 'n_samples', 'method', 'mode', 'model', 'spec'):
            assert key in r
        for stat_key in ('mean', 'std', 'min', 'max', 'percentiles'):
            assert stat_key in r['failure_stress']
            assert stat_key in r['knockdown']
        assert set(r['failure_stress']['percentiles']) == {'p5', 'p50', 'p95'}
        assert r['samples']['failure_stress'].shape == (self._N,)
        assert r['samples']['knockdown'].shape == (self._N,)
        # Echoed metadata.
        assert r['seed'] == 42
        assert r['n_samples'] == self._N
        assert r['method'] == 'monte_carlo'

    def test_same_seed_is_reproducible(self):
        kw = dict(covs={'sigma_1c': 0.08, 'E22': 0.05},
                  n_samples=self._N, seed=123)
        r1 = propagate_uncertainty(0.02, 'T800_epoxy', **kw)
        r2 = propagate_uncertainty(0.02, 'T800_epoxy', **kw)
        np.testing.assert_array_equal(r1['samples']['failure_stress'],
                                      r2['samples']['failure_stress'])
        assert r1['failure_stress']['mean'] == r2['failure_stress']['mean']
        assert r1['failure_stress']['std'] == r2['failure_stress']['std']

    def test_different_seed_differs(self):
        kw = dict(covs={'sigma_1c': 0.08}, n_samples=self._N)
        r1 = propagate_uncertainty(0.02, 'T800_epoxy', seed=1, **kw)
        r2 = propagate_uncertainty(0.02, 'T800_epoxy', seed=2, **kw)
        assert r1['failure_stress']['mean'] != r2['failure_stress']['mean']

    def test_mean_near_deterministic_nominal(self):
        # With a modest CoV the MC mean should be within a few % of the
        # deterministic single-point prediction.
        r = propagate_uncertainty(
            0.02, 'T800_epoxy', covs={'sigma_1c': 0.05},
            n_samples=256, seed=7)
        nominal = r['nominal']['failure_stress']
        assert r['failure_stress']['mean'] == pytest.approx(nominal, rel=0.05)

    def test_zero_cov_gives_zero_std(self):
        r = propagate_uncertainty(
            0.02, 'T800_epoxy', covs={'sigma_1c': 0.0, 'E22': 0.0},
            n_samples=self._N, seed=9)
        assert r['failure_stress']['std'] == 0.0
        assert r['knockdown']['std'] == 0.0
        assert r['failure_stress']['mean'] == pytest.approx(
            r['nominal']['failure_stress'])
        assert r['spec'] == {}  # all-zero CoV collapses to deterministic

    def test_lhs_method_runs_and_is_reproducible(self):
        kw = dict(covs={'sigma_1c': 0.08}, n_samples=self._N, seed=42,
                  method='lhs')
        r1 = propagate_uncertainty(0.02, 'T800_epoxy', **kw)
        r2 = propagate_uncertainty(0.02, 'T800_epoxy', **kw)
        assert r1['method'] == 'lhs'
        assert r1['failure_stress']['std'] > 0.0
        np.testing.assert_array_equal(r1['samples']['failure_stress'],
                                      r2['samples']['failure_stress'])

    def test_explicit_spec_and_vp_cov(self):
        r = propagate_uncertainty(
            0.02, 'T800_epoxy',
            spec={'sigma_1c': ('uniform', 0.1)},
            vp_cov=0.15, n_samples=self._N, seed=3)
        assert r['failure_stress']['std'] > 0.0
        assert r['vp_cov'] == 0.15
        assert r['spec'] == {'sigma_1c': ['uniform', 0.1]}

    def test_percentile_ordering(self):
        r = propagate_uncertainty(
            0.02, 'T800_epoxy', covs={'sigma_1c': 0.08},
            n_samples=128, seed=11)
        p = r['failure_stress']['percentiles']
        assert p['p5'] <= p['p50'] <= p['p95']
        fs = r['failure_stress']
        assert fs['min'] <= p['p5']
        assert p['p95'] <= fs['max']

    def test_unknown_material_rejected(self):
        with pytest.raises(ValueError, match="Unknown material"):
            propagate_uncertainty(0.02, 'not_a_material',
                                  covs={'sigma_1c': 0.08}, n_samples=4)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown sampling method"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'sigma_1c': 0.08}, n_samples=4,
                                  method='sobol')

    def test_non_perturbable_field_rejected(self):
        with pytest.raises(ValueError, match="non-perturbable"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'not_a_field': 0.1}, n_samples=4)

    # Issue #153: pin the input-validation branches in propagate_uncertainty,
    # _normalize_uq_spec, and _draw_unit_samples. These are advanced-user
    # guard rails -- the messages are exactly what someone debugging an
    # unusual UQ input needs, so a silent regression would hurt.

    def test_invalid_field_raises(self):
        with pytest.raises(ValueError, match="non-perturbable"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'not_a_field': 0.1},
                                  n_samples=self._N)

    def test_invalid_sampling_method_raises(self):
        with pytest.raises(ValueError, match="Unknown sampling method"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'sigma_1c': 0.08},
                                  n_samples=self._N,
                                  method='importance_sampling')

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match=r"percentiles must lie in \[0, 100\]"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'sigma_1c': 0.08},
                                  n_samples=self._N,
                                  percentiles=(5, 150))

    def test_non_positive_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'sigma_1c': 0.08},
                                  n_samples=0)
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'sigma_1c': 0.08},
                                  n_samples=-1)

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown material"):
            propagate_uncertainty(0.02, 'nonsense',
                                  covs={'sigma_1c': 0.08},
                                  n_samples=self._N)

    def test_malformed_spec_tuple_raises(self):
        # The (distribution, params) shape check lives on the `spec=` path;
        # passing a 1-tuple there should fail with the documented message.
        with pytest.raises(ValueError,
                           match=r"must be a \(distribution, params\) pair"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  spec={'E11': (0.1,)},
                                  n_samples=self._N)


class TestUQInputValidationBranches:
    """Remaining input-validation guards in _normalize_uq_spec /
    propagate_uncertainty / _draw_unit_samples that lacked direct tests."""

    _N = 8

    def test_negative_cov_rejected(self):
        with pytest.raises(ValueError, match=r"CoV"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  covs={'E11': -0.1}, n_samples=self._N)

    def test_spec_unknown_distribution_rejected(self):
        with pytest.raises(ValueError, match=r"unknown distribution"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  spec={'E11': ('weird', 0.1)},
                                  n_samples=self._N)

    def test_spec_negative_params_rejected(self):
        with pytest.raises(ValueError, match=r"params must be"):
            propagate_uncertainty(0.02, 'T800_epoxy',
                                  spec={'E11': ('normal', -0.1)},
                                  n_samples=self._N)

    def test_spec_zero_params_drops_field(self):
        """A zero-params spec entry is dropped (deterministic), not an
        error — exercises the ``resolved.pop`` else-branch."""
        r = propagate_uncertainty(0.02, 'T800_epoxy',
                                  spec={'E11': ('normal', 0.0)},
                                  n_samples=self._N, seed=0)
        assert 'E11' not in r['spec']

    def test_negative_vp_cov_rejected(self):
        with pytest.raises(ValueError, match=r"vp_cov"):
            propagate_uncertainty(0.02, 'T800_epoxy', covs={'E11': 0.05},
                                  vp_cov=-0.2, n_samples=self._N)

    def test_accepts_material_instance(self):
        """A ``MaterialProperties`` instance (not a preset name) is accepted
        directly (covers the non-string material branch)."""
        from porosity_fe_analysis import MATERIALS
        r = propagate_uncertainty(0.02, MATERIALS['T800_epoxy'],
                                  covs={'E11': 0.05}, n_samples=self._N, seed=0)
        assert r['samples']['knockdown'].shape == (self._N,)

    def test_draw_unit_samples_unknown_method_rejected(self):
        from porosity_fe.uq import _draw_unit_samples
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match=r"Unknown sampling method"):
            _draw_unit_samples(2, 4, 'bogus', rng)


# Issue #65: closed-form local sensitivities + per-point validation bands.


class TestValidationBands:
    """run_all_datasets must propagate a per-point 1-sigma Vp confidence
    band onto every strength prediction."""

    def test_band_present_in_run_all_datasets(self, tmp_path):
        """Synthesize a tiny dataset, run the full pipeline, and check
        that every strength entry carries a ``predicted_band`` whose
        per-point interval straddles the central ``predicted`` value."""
        from validation.validate_all import run_all_datasets

        ds = {
            "reference": "synthetic_uq_test",
            "material": {
                "fiber": "T700",
                "matrix": "TDE85 epoxy",
                "fiber_volume_fraction": 0.60,
                "n_plies": 8,
                "ply_angles": [0, 45, 90, -45, -45, 90, 45, 0],
            },
            "baseline_porosity_pct": 0.0,
            "properties": {
                "tensile_strength": {
                    "void_content_pct": [0.0, 1.0, 2.0, 3.0],
                    "normalized_values": [1.0, 0.95, 0.85, 0.78],
                }
            },
        }
        datasets_dir = tmp_path / 'datasets'
        datasets_dir.mkdir()
        path = datasets_dir / 'synthetic.json'
        path.write_text(json.dumps(ds))
        results = run_all_datasets(datasets_dir=str(datasets_dir), n_jobs=1)
        assert 'synthetic' in results
        prop = results['synthetic']['tensile_strength']
        assert 'predicted_band' in prop, \
            "Per-point 1-sigma band missing from strength prediction (#65)"
        band = prop['predicted_band']
        pred = prop['predicted']
        assert len(band) == len(pred)
        for (lo, hi), p in zip(band, pred):
            # The band must be ordered and must straddle the central point.
            assert lo <= p <= hi, (
                f"Band [{lo}, {hi}] does not straddle central prediction {p}"
            )
