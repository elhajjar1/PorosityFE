"""Dedicated unit tests for :mod:`porosity_fe.pipeline` (IMPROVEMENT_PLAN 6.6).

tests/test_cli.py covers the parallel sweep (serial/parallel agreement,
worker-exception propagation, completion-order independence) and
tests/test_integration.py the ``ConfigResult`` dict shim. This module pins
the serial orchestration contracts that were only exercised incidentally:
the ``build_empirical_pipeline`` factory (including ``solver_kwargs=``),
``_resolve_n_jobs`` edge cases, ``_analyze_one``'s output, what
``compare_configurations`` / ``sweep_configurations`` return, and the
closed-form headline numbers they carry.

Headline knockdowns are checked against the Judd-Wright closed form
``KD = exp(-alpha * Vp)`` with the QI coefficients (the default QI layup
has a layup scale of exactly 1 for every mode).
"""

import logging
import math

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    POROSITY_CONFIGS,
    Calibration,
    CompositeMesh,
    ConfigArtifacts,
    ConfigResult,
    EmpiricalSolver,
    PorosityField,
    build_empirical_pipeline,
    compare_configurations,
    sweep_configurations,
)
from porosity_fe import pipeline
from porosity_fe.pipeline import (
    _DEFAULT_MESH_RES,
    _analyze_one,
    _log_result_summary,
    _resolve_n_jobs,
)

T800 = MATERIALS['T800_epoxy']
TINY = (4, 3, 3)
TWO_CONFIGS = {
    'uniform_spherical': POROSITY_CONFIGS['uniform_spherical'],
    'clustered_midplane': POROSITY_CONFIGS['clustered_midplane'],
}


def _jw(mode, Vp):
    """Closed-form QI Judd-Wright knockdown."""
    return math.exp(-Calibration.JUDD_WRIGHT_ALPHA_QI[mode] * Vp)


class TestBuildEmpiricalPipeline:
    """``build_empirical_pipeline`` wires field -> mesh -> solver consistently."""

    def test_returns_linked_triple(self):
        """The solver wraps the returned mesh, which wraps the returned field."""
        pf, mesh, emp = build_empirical_pipeline(T800, 0.02, mesh_res=TINY)
        assert isinstance(pf, PorosityField)
        assert isinstance(mesh, CompositeMesh)
        assert isinstance(emp, EmpiricalSolver)
        assert mesh.porosity_field is pf
        assert emp.mesh is mesh
        assert pf.Vp == 0.02
        assert pf.material is T800 and mesh.material is T800 and emp.material is T800

    def test_mesh_res_order_is_nx_ny_nz(self):
        """``mesh_res`` maps to (nx, ny, nz) in that order."""
        _, mesh, _ = build_empirical_pipeline(T800, 0.02, mesh_res=(5, 2, 3))
        assert (mesh.nx, mesh.ny, mesh.nz) == (5, 2, 3)
        assert len(mesh.elements) == 5 * 2 * 3
        assert len(mesh.nodes) == 6 * 3 * 4

    def test_default_mesh_is_production_resolution(self):
        """Omitting ``mesh_res`` uses the single-source production default."""
        _, mesh, _ = build_empirical_pipeline(T800, 0.02)
        assert (mesh.nx, mesh.ny, mesh.nz) == _DEFAULT_MESH_RES == (30, 10, 12)

    def test_default_layup_is_qi(self):
        """Default layup is QI: the QI coefficients are used unscaled."""
        _, _, emp = build_empirical_pipeline(T800, 0.03, mesh_res=TINY)
        assert emp.f_md == pytest.approx(0.5)
        assert set(emp.layup_scale.values()) == {1.0}
        r = emp.get_failure_load('compression', 'judd_wright')
        assert r.knockdown == pytest.approx(_jw('compression', 0.03), rel=1e-12)

    def test_ply_angles_reach_mesh_and_solver(self):
        """'UD' is forwarded to both mesh (all-0 plies) and solver (f_md = 0)."""
        _, mesh, emp = build_empirical_pipeline(T800, 0.03, ply_angles='UD', mesh_res=TINY)
        assert np.all(mesh.ply_angles == 0.0)
        assert emp.f_md == 0.0
        assert emp.matrix_energy_fraction == pytest.approx(0.0, abs=1e-12)
        # UD is not less sensitive than QI (IMPROVEMENT_PLAN 2.7): scale 1.
        alpha = Calibration.JUDD_WRIGHT_ALPHA_QI['compression']
        r = emp.get_failure_load('compression', 'judd_wright')
        assert r.knockdown == pytest.approx(math.exp(-alpha * 0.03), rel=1e-12)

    def test_explicit_ply_list(self):
        """An explicit all-90 layup reaches the solver: tension is amplified to
        the transverse-tension alpha (scale 10 / 3.9)."""
        _, mesh, emp = build_empirical_pipeline(T800, 0.02, ply_angles=[90, 90], mesh_res=TINY)
        assert emp.f_md == 1.0
        assert set(np.unique(mesh.ply_angles)) == {90.0}
        with pytest.warns(UserWarning, match="unvalidated layup amplification"):
            r = emp.get_failure_load('tension', 'judd_wright')
        alpha = Calibration.JUDD_WRIGHT_ALPHA_QI['tension'] * (10.0 / 3.9)
        assert r.knockdown == pytest.approx(math.exp(-alpha * 0.02), rel=1e-12)
        assert r.details['layup_scale'] == pytest.approx(10.0 / 3.9, rel=1e-12)

    def test_porosity_config_forwarded(self):
        """``porosity_config`` keys reach the ``PorosityField`` constructor."""
        cfg = {'distribution': 'clustered', 'cluster_location': 'surface', 'void_shape': 'penny'}
        pf, _, _ = build_empirical_pipeline(T800, 0.02, porosity_config=cfg, mesh_res=TINY)
        assert pf.distribution == 'clustered'
        assert pf.cluster_location == 'surface'

    def test_seed_argument_wins_and_config_not_mutated(self):
        """Explicit ``seed=`` overrides a seed in the config without mutating it."""
        cfg = {'distribution': 'uniform', 'seed': 1}
        pf, _, _ = build_empirical_pipeline(T800, 0.02, porosity_config=cfg, seed=99, mesh_res=TINY)
        assert pf.seed == 99
        assert cfg == {'distribution': 'uniform', 'seed': 1}

    def test_seed_from_config_when_argument_absent(self):
        """A seed carried only in ``porosity_config`` is honoured."""
        pf, _, _ = build_empirical_pipeline(T800, 0.02, porosity_config={'seed': 5}, mesh_res=TINY)
        assert pf.seed == 5

    def test_percent_input_rejected_with_hint(self):
        """Vp is a fraction; a percent-looking value fails with a hint."""
        with pytest.raises(ValueError, match="Did you pass a percent"):
            build_empirical_pipeline(T800, 3.0, mesh_res=TINY)

    def test_invalid_porosity_config_key(self):
        """Unknown field kwargs surface as the constructor's TypeError."""
        with pytest.raises(TypeError):
            build_empirical_pipeline(T800, 0.02, porosity_config={'bogus': 1}, mesh_res=TINY)


class TestSolverKwargs:
    """``solver_kwargs=`` forwards coefficient overrides to ``EmpiricalSolver``."""

    def test_judd_wright_alpha_override(self):
        """An alpha override replaces the QI default for that mode only."""
        _, _, emp = build_empirical_pipeline(
            T800, 0.03, mesh_res=TINY, solver_kwargs={'judd_wright_alpha': {'compression': 2.0}})
        assert emp.get_failure_load('compression', 'judd_wright').knockdown == pytest.approx(
            math.exp(-2.0 * 0.03), rel=1e-12)
        assert emp.get_failure_load('ilss', 'judd_wright').knockdown == pytest.approx(
            _jw('ilss', 0.03), rel=1e-12)

    def test_power_law_and_linear_overrides(self):
        """``power_law_n`` and ``linear_beta`` overrides follow their closed forms."""
        _, _, emp = build_empirical_pipeline(
            T800, 0.04, mesh_res=TINY,
            solver_kwargs={'power_law_n': {'shear': 1.5}, 'linear_beta': {'tension': 2.5}})
        assert emp.get_failure_load('shear', 'power_law').knockdown == pytest.approx(
            (1 - 0.04) ** 1.5, rel=1e-12)
        assert emp.get_failure_load('tension', 'linear').knockdown == pytest.approx(
            1 - 2.5 * 0.04, rel=1e-12)

    def test_override_is_layup_scaled(self):
        """Overrides are scaled by the layup like the defaults (all-90
        compression -> x alpha_shear / alpha_compression = 8 / 6.9)."""
        _, _, emp = build_empirical_pipeline(
            T800, 0.02, ply_angles=[90], mesh_res=TINY,
            solver_kwargs={'judd_wright_alpha': {'compression': 2.0}})
        assert emp.JUDD_WRIGHT_ALPHA['compression'] == pytest.approx(2.0 * 8.0 / 6.9)

    def test_none_and_empty_are_equivalent_to_defaults(self):
        """``None`` and ``{}`` both leave the default coefficients in place."""
        coefs = []
        for kw in (None, {}):
            _, _, emp = build_empirical_pipeline(T800, 0.02, mesh_res=TINY, solver_kwargs=kw)
            coefs.append((emp.JUDD_WRIGHT_ALPHA, emp.POWER_LAW_N, emp.LINEAR_BETA))
        assert coefs[0] == coefs[1]

    def test_invalid_override_propagates(self):
        """Validation in ``EmpiricalSolver`` is not swallowed by the factory."""
        with pytest.raises(ValueError, match="unknown mode keys"):
            build_empirical_pipeline(T800, 0.02, mesh_res=TINY,
                                     solver_kwargs={'judd_wright_alpha': {'peel': 1.0}})

    def test_unknown_solver_kwarg_raises_type_error(self):
        """A misspelt keyword is a TypeError, not silently ignored."""
        with pytest.raises(TypeError):
            build_empirical_pipeline(T800, 0.02, mesh_res=TINY, solver_kwargs={'judd_wright_alfa': {}})

    def test_ply_angles_cannot_be_smuggled_twice(self):
        """``ply_angles`` belongs to the factory; repeating it in solver_kwargs is an error."""
        with pytest.raises(TypeError):
            build_empirical_pipeline(T800, 0.02, mesh_res=TINY, solver_kwargs={'ply_angles': 'UD'})


class TestResolveNJobs:
    """Edge cases of ``_resolve_n_jobs`` not covered by tests/test_cli.py."""

    def test_cpu_count_unavailable_falls_back_to_one(self, monkeypatch):
        """If ``os.cpu_count()`` returns None, 'all cores' means one worker."""
        monkeypatch.setattr(pipeline.os, 'cpu_count', lambda: None)
        for n in (None, 0, -1, -8):
            assert _resolve_n_jobs(n) == 1

    def test_any_negative_means_all_cores(self, monkeypatch):
        """Every non-positive value, not just -1, resolves to the core count."""
        monkeypatch.setattr(pipeline.os, 'cpu_count', lambda: 6)
        assert _resolve_n_jobs(-3) == 6

    def test_returns_builtin_int(self):
        """numpy integers are normalised to ``int``."""
        out = _resolve_n_jobs(np.int64(3))
        assert out == 3 and type(out) is int


class TestAnalyzeOne:
    """The picklable per-(Vp, config) worker."""

    def test_output_contract(self):
        """Echoes (Vp, name) and returns the documented dict keys and objects."""
        cfg = POROSITY_CONFIGS['interface_penny']
        Vp, name, result = _analyze_one(0.02, 'any-label', cfg, 'IM7_8551_epoxy', 0.0, 3)
        assert (Vp, name) == (0.02, 'any-label')
        assert set(result) == {'config', 'mesh', 'porosity_field', 'empirical_solver', 'empirical'}
        assert result['config'] is cfg
        assert result['porosity_field'].distribution == 'interface'
        assert result['porosity_field'].seed == 3
        assert result['porosity_field'].material is MATERIALS['IM7_8551_epoxy']
        assert (result['mesh'].nx, result['mesh'].ny, result['mesh'].nz) == _DEFAULT_MESH_RES

    def test_empirical_table_complete_and_closed_form(self):
        """All 5 modes x 3 built-in models; J-W values follow exp(-alpha*Vp)."""
        _, _, result = _analyze_one(0.03, 'u', POROSITY_CONFIGS['uniform_spherical'], 'T800_epoxy', -1500.0)
        emp = result['empirical']
        assert set(emp) == set(Calibration.JUDD_WRIGHT_ALPHA_QI)
        for mode, models in emp.items():
            assert set(models) == {'judd_wright', 'power_law', 'linear'}
            assert models['judd_wright'].knockdown == pytest.approx(_jw(mode, 0.03), rel=1e-12)

    def test_unknown_material_raises_key_error(self):
        """The worker resolves presets by name; validation is the caller's job."""
        with pytest.raises(KeyError):
            _analyze_one(0.02, 'u', {}, 'unobtainium', 0.0)


class TestCompareConfigurationsSerial:
    """Serial ``compare_configurations`` return values."""

    def test_default_configs_are_bundled_presets(self):
        """``configs=None`` sweeps every bundled configuration, in order."""
        results = compare_configurations(0.02)
        assert list(results) == list(POROSITY_CONFIGS)

    def test_result_fields(self):
        """Each ``ConfigResult`` carries Vp, name, config, seed and the J-W headline."""
        results = compare_configurations(0.03, configs=TWO_CONFIGS, seed=11)
        for name, r in results.items():
            assert isinstance(r, ConfigResult)
            assert r.Vp == 0.03
            assert r.config_name == name
            assert r.config is TWO_CONFIGS[name]
            assert r.seed == 11
            assert r.model == 'judd_wright'
            assert r.knockdown == pytest.approx(_jw('compression', 0.03), rel=1e-12)
            assert r.failure_stress == pytest.approx(T800.sigma_1c * r.knockdown, rel=1e-12)
            assert r.knockdown == r.empirical['compression']['judd_wright'].knockdown

    def test_zero_porosity_is_pristine(self):
        """Vp = 0 gives knockdown 1 and the pristine strength in every mode/model."""
        r = compare_configurations(0.0, configs={'u': {}})['u']
        assert r.knockdown == 1.0
        assert r.failure_stress == T800.sigma_1c
        for models in r.empirical.values():
            for fr in models.values():
                assert fr.knockdown == 1.0

    def test_empirical_results_independent_of_distribution(self):
        """Empirical path uses mean Vp, so every distribution gives the same table."""
        results = compare_configurations(0.03)
        tables = [{(m, k): v.knockdown for m, d in r.empirical.items() for k, v in d.items()}
                  for r in results.values()]
        assert all(t == tables[0] for t in tables[1:])

    def test_applied_stress_is_unused(self):
        """Changing ``applied_stress`` does not change the numbers (#132)."""
        a = compare_configurations(0.03, configs=TWO_CONFIGS, applied_stress=-1500.0)
        b = compare_configurations(0.03, configs=TWO_CONFIGS, applied_stress=123.0)
        for name in TWO_CONFIGS:
            assert a[name].failure_stress == b[name].failure_stress

    def test_other_material_changes_pristine_strength(self):
        """The material preset drives the pristine strength, not the knockdown."""
        res = compare_configurations(0.03, material_name='glass_epoxy', configs={'u': {}})['u']
        assert res.knockdown == pytest.approx(_jw('compression', 0.03), rel=1e-12)
        assert res.failure_stress == pytest.approx(MATERIALS['glass_epoxy'].sigma_1c * res.knockdown, rel=1e-12)

    def test_return_artifacts_are_the_live_objects(self):
        """Artifacts align with results by name and hold the objects that produced them."""
        results, artifacts = compare_configurations(0.02, configs=TWO_CONFIGS, return_artifacts=True)
        assert list(artifacts) == list(results) == list(TWO_CONFIGS)
        for name, art in artifacts.items():
            assert isinstance(art, ConfigArtifacts)
            assert art.field_results is None
            assert art.empirical_solver.mesh is art.mesh
            assert art.mesh.porosity_field is art.porosity_field
            assert art.porosity_field.Vp == 0.02
            assert art.porosity_field.distribution == TWO_CONFIGS[name]['distribution']
            assert art.empirical_solver.get_failure_load().knockdown == results[name].knockdown

    def test_single_config_never_builds_a_pool(self, monkeypatch):
        """One task runs inline even with n_jobs > 1 (no ProcessPoolExecutor)."""
        def _no_pool(*args, **kwargs):
            raise AssertionError("ProcessPoolExecutor should not be constructed")

        monkeypatch.setattr(pipeline.concurrent.futures, 'ProcessPoolExecutor', _no_pool)
        results = compare_configurations(0.02, configs={'u': {}}, n_jobs=4)
        assert list(results) == ['u']

    def test_n_jobs_one_never_builds_a_pool(self, monkeypatch):
        """``n_jobs=1`` stays serial for multi-config sweeps."""
        def _no_pool(*args, **kwargs):
            raise AssertionError("ProcessPoolExecutor should not be constructed")

        monkeypatch.setattr(pipeline.concurrent.futures, 'ProcessPoolExecutor', _no_pool)
        assert list(compare_configurations(0.02, configs=TWO_CONFIGS, n_jobs=1)) == list(TWO_CONFIGS)

    def test_serial_logging(self, caplog):
        """Serial runs log a per-config header, the tornado line and the rankings."""
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            compare_configurations(0.02, configs=TWO_CONFIGS)
        text = caplog.text
        for name in TWO_CONFIGS:
            assert f"Configuration: {name}" in text
            assert f"Tornado [{name}]" in text
        assert "RANKINGS" in text
        assert "POROSITY ANALYSIS: Vp = 2.0%" in text


class TestLogResultSummary:
    """Formatting branches of ``_log_result_summary``."""

    @staticmethod
    def _raw():
        _, _, raw = _analyze_one(0.02, 'u', {}, 'T800_epoxy', 0.0)
        return raw

    def test_parallel_single_line(self, caplog):
        """Parallel mode emits one 'done' line quoting both J-W knockdowns."""
        raw = self._raw()
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            _log_result_summary('cfgX', raw, is_parallel=True)
        comp = raw['empirical']['compression']['judd_wright']['knockdown']
        ilss = raw['empirical']['ilss']['judd_wright']['knockdown']
        assert f"Configuration cfgX done — compression KD (J-W) {comp:.3f}, ILSS KD (J-W) {ilss:.3f}" in caplog.text

    def test_missing_solver_degrades_gracefully(self, caplog):
        """Without an ``empirical_solver`` the tornado line explains how to get it."""
        raw = self._raw()
        del raw['empirical_solver']
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            _log_result_summary('cfgY', raw, is_parallel=False)
        assert "sensitivities unavailable" in caplog.text
        assert "return_artifacts=True" in caplog.text


class TestSweepConfigurations:
    """``sweep_configurations`` over several porosity levels (serial path)."""

    def test_unknown_material_rejected(self):
        """Material names are validated before any work, listing the presets."""
        with pytest.raises(ValueError, match="Available presets"):
            sweep_configurations([0.02], material_name='T800epoxy', configs=TWO_CONFIGS)

    def test_order_follows_input_not_sorted(self):
        """Output keys keep the caller's order (no sorting)."""
        out = sweep_configurations([0.04, 0.01, 0.02], configs={'u': {}})
        assert list(out) == [0.04, 0.01, 0.02]

    def test_keys_are_floats_and_duplicates_collapse(self):
        """Keys are coerced to float; numerically equal inputs collapse to one entry."""
        out = sweep_configurations([np.float64(0.03), 0.03, 0], configs={'u': {}})
        assert list(out) == [0.03, 0.0]
        assert all(type(k) is float for k in out)

    def test_accepts_generator(self):
        """Any iterable of fractions works, including a one-shot generator."""
        out = sweep_configurations((v / 100 for v in (1, 2)), configs={'u': {}})
        assert list(out) == [0.01, 0.02]

    def test_matches_compare_configurations(self):
        """Each level equals a standalone serial ``compare_configurations`` call."""
        out = sweep_configurations([0.01, 0.05], configs=TWO_CONFIGS, seed=4)
        for Vp, results in out.items():
            single = compare_configurations(Vp, configs=TWO_CONFIGS, seed=4)
            assert list(results) == list(single)
            for name in results:
                assert results[name].Vp == Vp
                assert results[name].seed == 4
                assert results[name].knockdown == single[name].knockdown
                assert results[name].failure_stress == single[name].failure_stress

    def test_knockdown_decreases_with_porosity(self):
        """Across levels the headline knockdown follows exp(-alpha*Vp), hence decreases."""
        levels = [0.0, 0.01, 0.03, 0.05]
        out = sweep_configurations(levels, configs={'u': {}})
        kds = [out[v]['u'].knockdown for v in levels]
        assert kds == pytest.approx([_jw('compression', v) for v in levels], rel=1e-12)
        assert all(b < a for a, b in zip(kds, kds[1:], strict=False))

    def test_return_artifacts_per_level(self):
        """With ``return_artifacts`` each level maps to a (results, artifacts) pair."""
        out = sweep_configurations([0.01, 0.02], configs=TWO_CONFIGS, return_artifacts=True)
        for Vp, pair in out.items():
            results, artifacts = pair
            assert list(results) == list(artifacts) == list(TWO_CONFIGS)
            assert all(a.porosity_field.Vp == Vp for a in artifacts.values())

    def test_one_task_list_for_all_levels(self, monkeypatch):
        """All (Vp, config) pairs go to a single ``_run_config_tasks`` call."""
        calls = []
        real = pipeline._run_config_tasks

        def spy(tasks, workers):
            calls.append((list(tasks), workers))
            return real(tasks, workers)

        monkeypatch.setattr(pipeline, '_run_config_tasks', spy)
        sweep_configurations([0.01, 0.02, 0.03], configs=TWO_CONFIGS, applied_stress=7.0, seed=2)
        assert len(calls) == 1
        tasks, workers = calls[0]
        assert workers == 1
        assert len(tasks) == 3 * len(TWO_CONFIGS)
        assert [(t[0], t[1]) for t in tasks] == [
            (v, n) for v in (0.01, 0.02, 0.03) for n in TWO_CONFIGS]
        assert all(t[3:] == ('T800_epoxy', 7.0, 2) for t in tasks)

    def test_empty_level_list(self):
        """No porosity levels -> empty result, no work."""
        assert sweep_configurations([], configs=TWO_CONFIGS) == {}



def test_empty_configs_run_nothing():
    """An explicit empty ``configs`` means no configurations, not the
    bundled five (``configs or POROSITY_CONFIGS`` used to swap them in)."""
    from porosity_fe import compare_configurations, sweep_configurations
    assert compare_configurations(0.02, configs={}) == {}
    assert sweep_configurations([0.02, 0.04], configs={}) == {0.02: {}, 0.04: {}}
