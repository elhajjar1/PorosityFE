#!/usr/bin/env python3
"""Tests for porosity_fe.io.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""

import dataclasses

import numpy as np
import pytest
import os

import matplotlib
matplotlib.use('Agg')
import json

from porosity_fe_analysis import (MATERIALS, PorosityField, POROSITY_CONFIGS, CompositeMesh,
                                   compare_configurations, save_results_to_json,
                                   FESolver, _build_provenance, load_results_from_json,
                                   JSON_SCHEMA_VERSION, FORMAT_EMPIRICAL_SWEEP)


class TestFEExportResults:
    def test_export_creates_file(self, tmp_path):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        path = str(tmp_path / "fe_results.json")
        FESolver.export_results(results, path)
        assert os.path.exists(path)

    def test_export_json_structure(self, tmp_path):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        path = str(tmp_path / "fe_results.json")
        FESolver.export_results(results, path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # Envelope keys
        assert 'schema_version' in data
        assert 'provenance' in data
        # Results merged into the envelope at the top level
        assert 'displacement' in data
        assert 'stress_global' in data
        assert 'failure' in data
        assert 'knockdown_factor' in data['failure']
        assert data['failure']['knockdown_factor'] > 0


def _parse_legacy_vtk(path):
    """Minimal legacy-ASCII VTK UNSTRUCTURED_GRID parser for test assertions.

    Returns a dict with header, n_points, n_cells, the parsed point
    coordinates, the cell connectivity, cell types, and the names of the
    POINT_DATA / CELL_DATA arrays found.
    """
    with open(path, encoding='utf-8') as fh:
        tokens = fh.read().split('\n')
    lines = [ln.strip() for ln in tokens if ln.strip() != '']

    info = {
        'header': lines[0],
        'point_data_arrays': [],
        'cell_data_arrays': [],
    }
    i = 0
    assert lines[2] == 'ASCII'
    assert lines[3] == 'DATASET UNSTRUCTURED_GRID'

    section = None  # None / 'point_data' / 'cell_data'
    while i < len(lines):
        ln = lines[i]
        parts = ln.split()
        if parts[0] == 'POINTS':
            n_points = int(parts[1])
            info['n_points'] = n_points
            pts = []
            for row in lines[i + 1:i + 1 + n_points]:
                pts.append([float(v) for v in row.split()])
            info['points'] = np.array(pts)
            i += 1 + n_points
            continue
        if parts[0] == 'CELLS':
            n_cells = int(parts[1])
            info['n_cells'] = n_cells
            info['cells_total_ints'] = int(parts[2])
            conn = []
            for row in lines[i + 1:i + 1 + n_cells]:
                vals = [int(v) for v in row.split()]
                assert vals[0] == 8  # hex8
                conn.append(vals[1:])
            info['cells'] = np.array(conn)
            i += 1 + n_cells
            continue
        if parts[0] == 'CELL_TYPES':
            n = int(parts[1])
            types = [int(v) for v in lines[i + 1:i + 1 + n]]
            info['cell_types'] = types
            i += 1 + n
            continue
        if parts[0] == 'POINT_DATA':
            section = 'point_data'
            i += 1
            continue
        if parts[0] == 'CELL_DATA':
            section = 'cell_data'
            i += 1
            continue
        if parts[0] in ('SCALARS', 'VECTORS'):
            name = parts[1]
            if section == 'point_data':
                info['point_data_arrays'].append(name)
            elif section == 'cell_data':
                info['cell_data_arrays'].append(name)
            i += 1
            continue
        i += 1
    return info


class TestFEExportVTK:
    """Issue #61: hex mesh + per-element fields written to legacy VTK."""

    def _solve(self):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        # #44: pin to UD so the per-element FI stays non-negative; the new
        # 'QI' default produces a richer multi-axial state that Tsai-Wu can
        # legitimately return small-negative values for in safe regions.
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2, ply_angles='UD')
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        return mesh, results

    def test_to_vtk_creates_file(self, tmp_path):
        mesh, results = self._solve()
        path = str(tmp_path / "fe_results.vtk")
        results.to_vtk(mesh, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_to_vtk_header_and_counts(self, tmp_path):
        mesh, results = self._solve()
        path = str(tmp_path / "fe_results.vtk")
        results.to_vtk(mesh, path)
        info = _parse_legacy_vtk(path)
        assert info['header'].startswith('# vtk DataFile Version')
        assert info['n_points'] == mesh.n_nodes
        assert info['n_cells'] == mesh.n_elements
        # Each hex line is "8 n0..n7" -> 9 ints per cell.
        assert info['cells_total_ints'] == mesh.n_elements * 9
        # All cells must be VTK_HEXAHEDRON (type 12).
        assert info['cell_types'] == [12] * mesh.n_elements

    def test_to_vtk_geometry_matches_mesh(self, tmp_path):
        mesh, results = self._solve()
        path = str(tmp_path / "fe_results.vtk")
        results.to_vtk(mesh, path)
        info = _parse_legacy_vtk(path)
        np.testing.assert_allclose(info['points'], mesh.nodes, rtol=1e-6)
        np.testing.assert_array_equal(info['cells'], mesh.elements)

    def test_to_vtk_has_expected_fields(self, tmp_path):
        mesh, results = self._solve()
        path = str(tmp_path / "fe_results.vtk")
        results.to_vtk(mesh, path)
        info = _parse_legacy_vtk(path)
        assert 'displacement' in info['point_data_arrays']
        assert 'porosity' in info['point_data_arrays']
        for name in ('von_mises', 'sigma_xx', 'tau_xy',
                     'tsai_wu_index', 'Vp_elem', 'is_void'):
            assert name in info['cell_data_arrays'], name

    def test_export_results_fmt_vtk(self, tmp_path):
        mesh, results = self._solve()
        path = str(tmp_path / "via_export.vtk")
        FESolver.export_results(results, path, fmt='vtk', mesh=mesh)
        info = _parse_legacy_vtk(path)
        assert info['n_points'] == mesh.n_nodes
        assert info['n_cells'] == mesh.n_elements

    def test_export_results_vtk_requires_mesh(self, tmp_path):
        _, results = self._solve()
        path = str(tmp_path / "no_mesh.vtk")
        with pytest.raises(ValueError):
            FESolver.export_results(results, path, fmt='vtk')

    def test_export_results_rejects_unknown_format(self, tmp_path):
        _, results = self._solve()
        path = str(tmp_path / "bad.xyz")
        with pytest.raises(ValueError):
            FESolver.export_results(results, path, fmt='nope')

    def test_per_element_failure_index_populated(self):
        mesh, results = self._solve()
        assert results.per_element_failure_index is not None
        assert results.per_element_failure_index.shape == (mesh.n_elements,)
        assert np.all(results.per_element_failure_index >= 0)
        # Scalar max must equal the per-element array's max.
        np.testing.assert_allclose(
            results.max_failure_index,
            float(results.per_element_failure_index.max()))

    def test_json_export_unchanged_back_compatible(self, tmp_path):
        mesh, results = self._solve()
        path = str(tmp_path / "fe_results.json")
        # Default still JSON; explicit fmt='json' also works.
        FESolver.export_results(results, path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        assert 'displacement' in data
        assert 'stress_global' in data
        assert 'failure' in data

    def test_to_vtk_meshio_roundtrip_if_available(self, tmp_path):
        """If meshio happens to be importable, it must parse our file too.

        meshio is NOT a project dependency; this test self-skips when it is
        absent so it never forces the dependency.
        """
        meshio = pytest.importorskip("meshio")
        mesh, results = self._solve()
        path = str(tmp_path / "fe_results.vtk")
        results.to_vtk(mesh, path)
        m = meshio.read(path)
        assert m.points.shape == (mesh.n_nodes, 3)
        total_cells = sum(len(cb.data) for cb in m.cells)
        assert total_cells == mesh.n_elements


class TestResultsSchemaAndReproducibility:
    """#20 (output JSON Schema, numpy serialization) and #55 (__version__,
    seed provenance, determinism contract)."""

    _SCHEMA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'validation', 'schemas', 'porosity_results_schema.json')

    def _one_config_results(self):
        return compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']})

    def test_exported_file_validates_against_results_schema(self, tmp_path):
        import jsonschema
        with open(self._SCHEMA_PATH, encoding='utf-8') as f:
            schema = json.load(f)
        path = str(tmp_path / "schema_check.json")
        save_results_to_json(self._one_config_results(), path)
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        jsonschema.validate(instance=doc, schema=schema)  # raises on drift

    def test_module_has_importable_version(self):
        import porosity_fe_analysis as pfa
        assert isinstance(pfa.__version__, str) and pfa.__version__

    def test_provenance_records_version_and_seed(self, tmp_path):
        results = compare_configurations(
            0.03, seed=4242,
            configs={'uniform_spherical':
                     POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "prov.json")
        save_results_to_json(results, path)
        with open(path, encoding='utf-8') as f:
            prov = json.load(f)['provenance']
        assert prov['porosity_fe_version']  # no longer silently None
        assert prov['seed'] == 4242

    def test_pipeline_is_byte_deterministic(self, tmp_path):
        """Locks in current determinism so any future RNG introduction is
        forced to expose a seed (#55)."""
        p1, p2 = str(tmp_path / "r1.json"), str(tmp_path / "r2.json")
        save_results_to_json(self._one_config_results(), p1)
        save_results_to_json(self._one_config_results(), p2)
        with open(p1, encoding='utf-8') as f:
            d1 = json.load(f)
        with open(p2, encoding='utf-8') as f:
            d2 = json.load(f)
        # Two back-to-back runs in one process differ only by timestamp;
        # strip both the legacy and #55-alias timestamp keys before compare.
        for key in ('timestamp_utc', 'generated_utc'):
            d1['provenance'].pop(key, None)
            d2['provenance'].pop(key, None)
        assert d1 == d2

    def test_json_default_handles_numpy_and_ndarray(self, tmp_path):
        from porosity_fe_analysis import _json_default
        assert _json_default(np.float64(1.5)) == 1.5
        assert _json_default(np.int64(7)) == 7
        assert _json_default(np.array([1.0, 2.0])) == [1.0, 2.0]
        # End-to-end: an ndarray smuggled into the payload must not raise.
        # With #44 the result is a ConfigResult dataclass; mutate a *copy*
        # of its ``config`` dict so the shared POROSITY_CONFIGS entry is
        # not poisoned for other tests, then build a fresh dataclass.
        results = self._one_config_results()
        original = results['uniform_spherical']
        replacement = dataclasses.replace(
            original,
            config={**original.config,
                    'ply_angles': np.array([0.0, 90.0, 45.0])})
        results = {'uniform_spherical': replacement}
        path = str(tmp_path / "np.json")
        save_results_to_json(results, path)  # would TypeError pre-#20
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        assert doc['uniform_spherical']['config']['ply_angles'] == [
            0.0, 90.0, 45.0]


class TestBuildProvenance:
    """Tests for the _build_provenance() reproducibility helper."""

    def test_provenance_returns_dict(self):
        prov = _build_provenance()
        assert isinstance(prov, dict)

    def test_required_keys_present(self):
        prov = _build_provenance()
        for key in ('porosity_fe_version', 'python_version', 'numpy_version',
                    'scipy_version', 'matplotlib_version', 'timestamp_utc',
                    'platform', 'seed', 'git_commit'):
            assert key in prov, f"Missing provenance key: {key}"

    def test_python_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['python_version'], str)
        assert len(prov['python_version']) > 0
        # Should look like "3.X.Y"
        parts = prov['python_version'].split('.')
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_numpy_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['numpy_version'], str)
        assert len(prov['numpy_version']) > 0

    def test_scipy_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['scipy_version'], str)
        assert len(prov['scipy_version']) > 0

    def test_matplotlib_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['matplotlib_version'], str)
        assert len(prov['matplotlib_version']) > 0

    def test_timestamp_utc_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['timestamp_utc'], str)
        assert prov['timestamp_utc'].endswith('Z')
        # Should be parseable as ISO-8601
        import datetime
        ts = prov['timestamp_utc'].rstrip('Z')
        datetime.datetime.fromisoformat(ts)  # raises if malformed

    def test_platform_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['platform'], str)
        assert len(prov['platform']) > 0

    def test_seed_is_none(self):
        # No random seed is used in this codebase; must be null
        prov = _build_provenance()
        assert prov['seed'] is None

    def test_git_commit_is_string_or_none(self):
        prov = _build_provenance()
        assert prov['git_commit'] is None or isinstance(prov['git_commit'], str)

    def test_version_lookup_failure_falls_back_to_package_attr(self, monkeypatch):
        """io.py:101-108: if importlib.metadata.version() raises (source
        checkout not pip-installed), the version fields fall back to the
        package ``__version__`` attribute, never silently None."""
        import importlib.metadata as ilm

        from porosity_fe import __version__ as pkg_version

        def _boom(_dist_name):
            raise ilm.PackageNotFoundError("porosity-fe")

        # _build_provenance does ``import importlib.metadata as _ilm`` then
        # calls ``_ilm.version(...)``; patch the symbol at its real home.
        monkeypatch.setattr(ilm, "version", _boom)
        prov = _build_provenance()
        assert prov['porosity_fe_version'] == pkg_version
        assert prov['package_version'] == pkg_version

    def test_git_subprocess_filenotfound_yields_none_sha(self, monkeypatch):
        """io.py:117-128: a missing ``git`` binary (FileNotFoundError) must
        degrade gracefully to a None git SHA, not propagate."""
        from porosity_fe import io as io_mod

        def _no_git(*_args, **_kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(io_mod.subprocess, "run", _no_git)
        prov = _build_provenance()
        assert prov['git_commit'] is None
        assert prov['git_sha'] is None

    def test_git_subprocess_timeout_yields_none_sha(self, monkeypatch):
        """io.py:117-128: a hung ``git`` (TimeoutExpired) must also degrade
        gracefully to a None git SHA."""
        import subprocess as _subprocess

        from porosity_fe import io as io_mod

        def _timeout(*_args, **_kwargs):
            raise _subprocess.TimeoutExpired(cmd="git rev-parse HEAD",
                                             timeout=5)

        monkeypatch.setattr(io_mod.subprocess, "run", _timeout)
        prov = _build_provenance()
        assert prov['git_commit'] is None
        assert prov['git_sha'] is None


class TestProvenanceInSaveResultsJson:
    """Integration: provenance is present and valid in save_results_to_json output."""

    def test_provenance_in_json_output(self, tmp_path):
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        )
        path = str(tmp_path / "prov_test.json")
        save_results_to_json(results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        prov = data['provenance']
        assert isinstance(prov['python_version'], str) and prov['python_version']
        assert isinstance(prov['numpy_version'], str) and prov['numpy_version']
        assert isinstance(prov['timestamp_utc'], str) and prov['timestamp_utc']
        assert 'porosity_fe_version' in prov

    def test_schema_version_in_json_output(self, tmp_path):
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        )
        path = str(tmp_path / "schema_test.json")
        save_results_to_json(results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Track the current envelope version rather than hard-coding it,
        # so a future additive minor bump doesn't break this assertion (#131).
        assert data['schema_version'] == JSON_SCHEMA_VERSION


class TestJsonEncodingRoundTrip:
    """Regression for #21: JSON I/O must be UTF-8 on every platform.

    Without explicit encoding, Windows opens files in the locale code page
    (cp1252) and silently mangles non-ASCII content. This locks the
    round-trip with characters that are not representable in cp1252.
    """

    def test_non_ascii_round_trips_through_loader(self, tmp_path):
        path = str(tmp_path / "ünïcode_µCT.json")
        payload = {
            "schema_version": JSON_SCHEMA_VERSION,
            "format": FORMAT_EMPIRICAL_SWEEP,
            "note": "µCT scan, σ₁c knockdown — café/naïve ✓",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        loaded = load_results_from_json(path)
        assert loaded["note"] == "µCT scan, σ₁c knockdown — café/naïve ✓"


class TestProvenanceInFEExportResults:
    """Integration: provenance is present and valid in FESolver.export_results output."""

    def test_provenance_in_fe_json_output(self, tmp_path):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        field_results = solver.solve(loading='compression', applied_strain=-0.001)
        path = str(tmp_path / "fe_prov_test.json")
        FESolver.export_results(field_results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        prov = data['provenance']
        assert isinstance(prov['python_version'], str) and prov['python_version']
        assert isinstance(prov['numpy_version'], str) and prov['numpy_version']
        assert isinstance(prov['timestamp_utc'], str) and prov['timestamp_utc']
        assert 'porosity_fe_version' in prov
        assert data['schema_version'] == JSON_SCHEMA_VERSION


class TestIssue55ProvenanceContract:
    """Locks in the #55 reproducibility contract field names and behaviors:
    short-name aliases, opt-in hostname, schema_version inside the block,
    and the include_raw sidecar for FE exports.
    """

    def test_provenance_keys_present(self, tmp_path):
        """All #55 keys (and back-compat aliases) appear in saved JSON."""
        results = compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "keys.json")
        save_results_to_json(results, path)
        with open(path, encoding='utf-8') as f:
            prov = json.load(f)['provenance']
        for key in ('schema_version', 'package_version', 'python', 'numpy',
                    'scipy', 'platform', 'git_sha', 'generated_utc', 'seed'):
            assert key in prov, f"Missing #55 provenance key: {key}"
        # generated_utc must be a non-empty ISO-Z timestamp.
        assert isinstance(prov['generated_utc'], str)
        assert prov['generated_utc'].endswith('Z')

    def test_byte_identical_reruns(self, tmp_path):
        """Two back-to-back runs differ only in the timestamp keys."""
        cfg = {'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        p1 = str(tmp_path / "a.json")
        p2 = str(tmp_path / "b.json")
        save_results_to_json(compare_configurations(0.03, configs=cfg), p1)
        save_results_to_json(compare_configurations(0.03, configs=cfg), p2)
        with open(p1, encoding='utf-8') as f:
            d1 = json.load(f)
        with open(p2, encoding='utf-8') as f:
            d2 = json.load(f)
        for key in ('timestamp_utc', 'generated_utc'):
            d1['provenance'].pop(key, None)
            d2['provenance'].pop(key, None)
        assert d1 == d2

    def test_aliases_match_legacy_keys(self):
        """Short-name aliases mirror the legacy *_version fields exactly."""
        prov = _build_provenance(seed=7)
        assert prov['package_version'] == prov['porosity_fe_version']
        assert prov['python'] == prov['python_version']
        assert prov['numpy'] == prov['numpy_version']
        assert prov['scipy'] == prov['scipy_version']
        assert prov['git_sha'] == prov['git_commit']
        assert prov['generated_utc'] == prov['timestamp_utc']
        assert prov['seed'] == 7
        assert prov['schema_version'] == JSON_SCHEMA_VERSION

    def test_hostname_opt_in_default_off(self, monkeypatch):
        """No hostname unless POROSITY_FE_INCLUDE_HOSTNAME=1."""
        monkeypatch.delenv('POROSITY_FE_INCLUDE_HOSTNAME', raising=False)
        prov = _build_provenance()
        assert 'hostname' not in prov

    def test_hostname_opt_in_when_enabled(self, monkeypatch):
        monkeypatch.setenv('POROSITY_FE_INCLUDE_HOSTNAME', '1')
        prov = _build_provenance()
        assert 'hostname' in prov
        # Either a non-empty string or None on hosts that refuse to report.
        assert prov['hostname'] is None or isinstance(prov['hostname'], str)

    def test_fe_export_include_raw_writes_npz_sidecar(self, tmp_path):
        """include_raw=True emits a sibling .npz with the raw arrays."""
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        field_results = solver.solve(loading='compression',
                                     applied_strain=-0.001)
        json_path = str(tmp_path / "fe_raw.json")
        FESolver.export_results(field_results, json_path, include_raw=True)
        npz_path = json_path + '.npz'
        assert os.path.exists(npz_path)
        loaded = np.load(npz_path)
        for key in ('displacement', 'stress_global', 'stress_local',
                    'strain_global', 'strain_local'):
            assert key in loaded.files
        # Raw arrays should round-trip exactly.
        np.testing.assert_array_equal(loaded['displacement'],
                                      field_results.displacement)

    def test_fe_export_default_no_npz(self, tmp_path):
        """Default behavior must NOT bloat the output with a sidecar."""
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        field_results = solver.solve(loading='compression',
                                     applied_strain=-0.001)
        json_path = str(tmp_path / "fe_nosidecar.json")
        FESolver.export_results(field_results, json_path)
        assert not os.path.exists(json_path + '.npz')

    def test_seed_threaded_through_compare_configurations(self):
        """seed kwarg lands on every PorosityField the pipeline builds."""
        # #44 item 3: pull the porosity_field from the artifacts dict
        # since it's no longer carried on the default ConfigResult.
        _results, artifacts = compare_configurations(
            0.03, seed=99,
            configs={'uniform_spherical':
                     POROSITY_CONFIGS['uniform_spherical']},
            return_artifacts=True)
        pf = artifacts['uniform_spherical'].porosity_field
        assert pf.seed == 99


# Tiny single-config dict keeps the argparse-driver tests fast (#58).
_TINY_CONFIGS = {'uniform_spherical': {'distribution': 'uniform',
                                       'void_shape': 'spherical'}}
