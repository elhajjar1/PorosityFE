#!/usr/bin/env python3
"""Tests for porosity_fe.cli.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest
import os

import matplotlib
matplotlib.use('Agg')
import json

from porosity_fe_analysis import (POROSITY_CONFIGS, compare_configurations, load_results_from_json,
                                   JSON_SCHEMA_VERSION, FORMAT_EMPIRICAL_SWEEP)

import porosity_fe_analysis


# Tiny single-config dict keeps the argparse-driver tests fast (#58).
_TINY_CONFIGS = {'uniform_spherical': {'distribution': 'uniform',
                                       'void_shape': 'spherical'}}


class TestValidateCLISmoke:
    """Smoke tests for the validate_porosity CLI entry point (#12)."""

    def test_help_exits_zero(self):
        from validate_porosity_cli import main
        with pytest.raises(SystemExit) as exc:
            main(['--help'])
        assert exc.value.code == 0


def _extract_knockdowns(results: dict) -> dict:
    """Flatten the per-config knockdown numbers for equality checks.

    Picks the numerical scalars the parallel/serial paths must agree on,
    skipping the embedded ``mesh`` / ``porosity_field`` / ``empirical_solver``
    objects (those are different instances per call by construction).
    """
    flat = {}
    for name, r in results.items():
        emp = r['empirical']
        for mode in ('compression', 'tension', 'shear', 'ilss'):
            for model in ('judd_wright', 'power_law', 'linear'):
                key = (name, mode, model)
                flat[key] = emp[mode][model]['knockdown']
    return flat


def _crashing_worker(*args, **kwargs):
    """Module-level stand-in for ``_analyze_one`` used by the parallel
    exception-propagation test (#154).

    Has to live at module scope so ``ProcessPoolExecutor`` can pickle it
    by qualified name when ``compare_configurations`` submits tasks.
    """
    raise RuntimeError("Worker crash")


class TestParallelSweep:
    """Parallel ``compare_configurations`` path (#52)."""

    def test_parallel_matches_serial(self):
        """n_jobs>1 must produce numerically identical results to n_jobs=1.

        The pipeline is deterministic linear algebra (no RNG), so the
        parallel path is expected to be bit-identical, not merely close.
        We assert ``assert_allclose`` with a tight tolerance to allow for
        BLAS reorder noise on multi-threaded platforms.
        """
        configs = {
            'uniform_spherical': POROSITY_CONFIGS['uniform_spherical'],
            'clustered_midplane': POROSITY_CONFIGS['clustered_midplane'],
        }
        serial = compare_configurations(
            0.03, configs=configs, n_jobs=1)
        parallel = compare_configurations(
            0.03, configs=configs, n_jobs=2)

        # Same config-name set, in the same order (deterministic assembly).
        assert list(serial.keys()) == list(parallel.keys())

        s_flat = _extract_knockdowns(serial)
        p_flat = _extract_knockdowns(parallel)
        assert set(s_flat) == set(p_flat)
        for k in s_flat:
            np.testing.assert_allclose(
                p_flat[k], s_flat[k], rtol=1e-10, atol=0.0,
                err_msg=f"Knockdown drift between serial and parallel for {k}")

    def test_resolve_n_jobs_normalises_zero_and_negative(self):
        """0/-1/None all mean "use all cores"."""
        from porosity_fe_analysis import _resolve_n_jobs
        cores = os.cpu_count() or 1
        assert _resolve_n_jobs(None) == cores
        assert _resolve_n_jobs(0) == cores
        assert _resolve_n_jobs(-1) == cores
        assert _resolve_n_jobs(1) == 1
        assert _resolve_n_jobs(4) == 4

    def test_analyze_one_returns_picklable_dict(self):
        """The (Vp, name) -> result tuple must round-trip through pickle.

        Guards the ProcessPoolExecutor contract: if a future refactor
        adds an un-picklable member (lambda, open file handle) the
        parallel path silently degrades to a cryptic worker error. This
        test catches it at the helper level.
        """
        import pickle
        from porosity_fe_analysis import _analyze_one
        Vp, name, result = _analyze_one(
            0.02, 'uniform_spherical',
            POROSITY_CONFIGS['uniform_spherical'],
            'T800_epoxy', -1500.0, None)
        assert Vp == 0.02
        assert name == 'uniform_spherical'
        # Round-trip the whole result dict (mesh + porosity_field +
        # empirical_solver + emp_results all included).
        round_tripped = pickle.loads(pickle.dumps(result))
        assert (round_tripped['empirical']['compression']['judd_wright']
                ['knockdown']
                == result['empirical']['compression']['judd_wright']
                ['knockdown'])

    def test_single_config_does_not_spawn_pool(self):
        """One task should run inline even when n_jobs>1 to avoid
        ProcessPoolExecutor's fork overhead. We can't observe the pool
        directly without monkeypatching, but we can assert the result
        matches the serial path."""
        configs = {'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        serial = compare_configurations(0.03, configs=configs, n_jobs=1)
        also_serial = compare_configurations(0.03, configs=configs, n_jobs=4)
        s_flat = _extract_knockdowns(serial)
        p_flat = _extract_knockdowns(also_serial)
        for k in s_flat:
            assert p_flat[k] == s_flat[k]

    def test_parallel_worker_exception_propagates(self, monkeypatch):
        """A worker exception must re-raise in the parent, not be
        silently swallowed (#154).

        ``compare_configurations`` collects parallel results via
        ``future.result()`` inside an ``as_completed`` loop; that call
        re-raises whatever the worker raised. We pin that contract by
        monkeypatching ``_analyze_one`` on the pipeline module to raise
        ``RuntimeError`` unconditionally. On Linux the
        ``ProcessPoolExecutor`` default start method is fork, so the
        child inherits the patched module attribute and the exception
        crosses the process boundary as expected.
        """
        from porosity_fe import pipeline as _pipeline

        monkeypatch.setattr(_pipeline, "_analyze_one", _crashing_worker)

        # Two configs so we actually take the parallel branch
        # (``len(tasks) > 1`` and ``workers > 1``).
        configs = {
            'uniform_spherical': POROSITY_CONFIGS['uniform_spherical'],
            'clustered_midplane': POROSITY_CONFIGS['clustered_midplane'],
        }
        with pytest.raises(RuntimeError, match="Worker crash"):
            compare_configurations(0.03, configs=configs, n_jobs=2)

    def test_parallel_results_assembled_by_config_name(self):
        """Result-dict key order must match caller-supplied config insertion
        order, never worker completion order (#154).

        ``as_completed`` yields futures in completion order, which is
        non-deterministic, so the assembly loop has to re-iterate the
        original ``configs`` mapping. We run the sweep 10 times with
        distinctive names to make any worker-completion-order leak
        obvious; even a single iteration with reversed keys would fail.
        """
        configs = {
            'cfg_alpha': POROSITY_CONFIGS['uniform_spherical'],
            'cfg_beta': POROSITY_CONFIGS['clustered_midplane'],
        }
        expected_order = list(configs.keys())
        for _ in range(10):
            results = compare_configurations(
                0.03, configs=configs, n_jobs=2)
            assert list(results.keys()) == expected_order


class TestCLIMain:
    """Argparse-driven entry point (issue #58)."""

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main(['--help'])
        assert exc.value.code == 0
        assert 'porosity-analyze' in capsys.readouterr().out

    def test_version_flag(self, capsys):
        """--version prints '<prog> <__version__>' and exits 0 (issue #80)."""
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main(['--version'])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert 'porosity-analyze' in out
        assert porosity_fe_analysis.__version__ in out

    def test_list_materials(self, capsys):
        assert porosity_fe_analysis.main(['--list-materials']) == 0
        out = capsys.readouterr().out
        assert 'T800_epoxy' in out

    def test_unknown_material_errors(self):
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main(['--material', 'unobtainium'])
        assert exc.value.code == 2

    def test_out_of_range_vp_errors(self):
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main(['--vp', '1.5'])
        assert exc.value.code == 2

    def test_single_vp_writes_roundtrippable_json(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        rc = porosity_fe_analysis.main([
            '--material', 'T800_epoxy',
            '--vp', '0.03',
            '--output-dir', str(tmp_path),
            '--quiet',
        ])
        assert rc == 0
        out_file = tmp_path / 'porosity_analysis_results_3pct.json'
        assert out_file.exists()
        data = load_results_from_json(str(out_file))
        assert data['schema_version'] == JSON_SCHEMA_VERSION
        assert data['format'] == FORMAT_EMPIRICAL_SWEEP
        assert 'uniform_spherical' in data

    def test_seed_is_recorded_in_provenance(self, tmp_path, monkeypatch):
        # Regression for #79: --seed must reach provenance, not be dropped.
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        rc = porosity_fe_analysis.main([
            '--vp', '0.03',
            '--seed', '12345',
            '--output-dir', str(tmp_path),
            '--quiet',
        ])
        assert rc == 0
        out_file = tmp_path / 'porosity_analysis_results_3pct.json'
        payload = json.loads(out_file.read_text(encoding='utf-8'))
        assert payload['provenance']['seed'] == 12345

    def test_default_cwd_when_no_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        monkeypatch.chdir(tmp_path)
        rc = porosity_fe_analysis.main(['--vp', '0.02', '--quiet'])
        assert rc == 0
        assert (tmp_path / 'porosity_analysis_results_2pct.json').exists()

    def test_non_integer_vp_label_no_collision(self, tmp_path, monkeypatch):
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        rc = porosity_fe_analysis.main([
            '--vp', '0.025',
            '--output-dir', str(tmp_path),
            '--quiet',
        ])
        assert rc == 0
        assert (tmp_path / 'porosity_analysis_results_2p5pct.json').exists()

    def test_quiet_silences_progress_banner(
            self, tmp_path, monkeypatch, capsys):
        """Regression for #78: --quiet must suppress the analysis banner,
        per-configuration lines, and trailing summary -- not just the
        final 6-line trailer."""
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        rc = porosity_fe_analysis.main([
            '--vp', '0.02',
            '--output-dir', str(tmp_path),
            '--quiet',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        # Banner / per-config / mesh / trailer strings must all be gone.
        assert 'POROSITY ANALYSIS' not in captured.out
        assert 'Configuration:' not in captured.out
        assert 'Mesh generated' not in captured.out
        assert 'RANKINGS' not in captured.out
        assert 'COMPLETE ANALYSIS FINISHED' not in captured.out
        assert 'Saved:' not in captured.out

    def test_default_emits_progress(self, tmp_path, monkeypatch, capsys):
        """Without --quiet, the analysis banner and progress should still
        appear so we don't break the interactive UX."""
        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        rc = porosity_fe_analysis.main([
            '--vp', '0.02',
            '--output-dir', str(tmp_path),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'POROSITY ANALYSIS' in captured.out
        assert 'COMPLETE ANALYSIS FINISHED' in captured.out

    def test_quiet_and_verbose_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main([
                '--vp', '0.02',
                '--output-dir', str(tmp_path),
                '--quiet', '--verbose',
            ])
        assert exc.value.code == 2

    def test_jobs_cli_flag_passed_through(self, tmp_path, monkeypatch):
        """``--jobs N`` from the CLI must thread into compare_configurations
        as ``n_jobs=N`` (#52). We monkeypatch the function with a recording
        shim instead of spinning up real workers — the actual parallel
        sweep is exercised by ``TestParallelSweep`` above."""
        seen = {}
        original = porosity_fe_analysis.compare_configurations

        def _spy(*args, **kwargs):
            seen['n_jobs'] = kwargs.get('n_jobs')
            return original(*args, **kwargs)

        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        monkeypatch.setattr(porosity_fe_analysis,
                            'compare_configurations', _spy)
        rc = porosity_fe_analysis.main([
            '--vp', '0.02',
            '--output-dir', str(tmp_path),
            '--quiet',
            '--jobs', '2',
        ])
        assert rc == 0
        assert seen.get('n_jobs') == 2

    def test_jobs_default_is_serial(self, tmp_path, monkeypatch):
        """Default ``--jobs`` (omitted) must resolve to ``n_jobs=1`` so
        the legacy deterministic behaviour is preserved for unsuspecting
        callers and CI."""
        seen = {}
        original = porosity_fe_analysis.compare_configurations

        def _spy(*args, **kwargs):
            seen['n_jobs'] = kwargs.get('n_jobs')
            return original(*args, **kwargs)

        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        monkeypatch.setattr(porosity_fe_analysis,
                            'compare_configurations', _spy)
        rc = porosity_fe_analysis.main([
            '--vp', '0.02',
            '--output-dir', str(tmp_path),
            '--quiet',
        ])
        assert rc == 0
        assert seen.get('n_jobs') == 1

    # #127: --vp / --vp-pct unit reconciliation -------------------------

    def test_vp_pct_alias_converts_to_fraction(self, tmp_path, monkeypatch):
        """`--vp-pct 2` must run as if `--vp 0.02` were passed."""
        seen = []
        original = porosity_fe_analysis.compare_configurations

        def _spy(*args, **kwargs):
            # First positional arg of compare_configurations is the Vp.
            Vp = args[0] if args else kwargs.get('void_volume_fraction')
            seen.append(float(Vp))
            return original(*args, **kwargs)

        monkeypatch.setattr(porosity_fe_analysis, 'POROSITY_CONFIGS',
                            _TINY_CONFIGS)
        monkeypatch.setattr(porosity_fe_analysis,
                            'compare_configurations', _spy)
        rc = porosity_fe_analysis.main([
            '--vp-pct', '2',
            '--output-dir', str(tmp_path),
            '--quiet',
        ])
        assert rc == 0
        assert seen == [pytest.approx(0.02, rel=1e-12)]

    def test_vp_pct_and_vp_mutually_exclusive(self, tmp_path, capsys):
        """Passing both --vp and --vp-pct must error out cleanly."""
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main([
                '--vp', '0.02', '--vp-pct', '2',
                '--output-dir', str(tmp_path),
            ])
        # argparse exits 2 for invalid usage.
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert 'not allowed with' in err

    def test_vp_percentage_typo_gets_helpful_hint(self, tmp_path, capsys):
        """`--vp 3` (looks like a percentage) must suggest both fixes."""
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main([
                '--vp', '3',
                '--output-dir', str(tmp_path),
            ])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        # Both alternatives surfaced — user gets to pick.
        assert '--vp 0.0300' in err
        assert '--vp-pct 3' in err

    # #147: pin the two uncovered error-handling branches in cli.main so
    # future refactors can't silently drop the user-facing hints.

    def test_cli_main_vp_out_of_range_with_percentage_hint(
            self, tmp_path, capsys):
        """`--vp 5` is in the 1-100 "looks like a percentage" band, so the
        error must include BOTH the fraction-form suggestion (`--vp 0.0500`)
        and the percent-alias suggestion (`--vp-pct 5`). Regression for the
        hint added in #127/#137 — covers cli.py lines ~270-275."""
        with pytest.raises(SystemExit) as exc:
            porosity_fe_analysis.main([
                '--vp', '5',
                '--output-dir', str(tmp_path),
            ])
        # argparse.error() exits with code 2 for invalid usage.
        assert exc.value.code == 2
        err = capsys.readouterr().err
        # Both alternatives must be surfaced so the user can pick.
        assert '--vp 0.0500' in err
        assert '--vp-pct 5' in err

    def test_cli_main_output_dir_permission_error(
            self, tmp_path, monkeypatch, capsys):
        """If output-directory creation raises OSError (e.g. permission
        denied or an invalid path), the CLI must surface the failure on
        stderr and return exit code 2 rather than crashing with a
        traceback. Covers cli.py lines ~284-287."""
        def _raise_permission_error(*args, **kwargs):
            raise PermissionError("denied")

        # The CLI now uses ``Path.mkdir`` (issue #113 modernisation), so
        # patch the directory-creation primitive on the Path class so the
        # call inside main() picks up the raising stub.
        from pathlib import Path as _Path
        monkeypatch.setattr(
            _Path, 'mkdir',
            _raise_permission_error,
        )
        rc = porosity_fe_analysis.main([
            '--vp', '0.02',
            '--output-dir', str(tmp_path / 'nope'),
            '--quiet',
        ])
        assert rc == 2
        err = capsys.readouterr().err
        # Error message should mention the underlying failure so the user
        # knows what went wrong.
        assert 'denied' in err.lower() or 'permission' in err.lower()
