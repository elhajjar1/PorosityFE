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
                                   JSON_SCHEMA_VERSION, FORMAT_EMPIRICAL_SWEEP,
                                   VoidGeometry)


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


_VTU_DTYPES = {'Float64': '<f8', 'Float32': '<f4', 'Int64': '<i8',
               'UInt8': 'u1'}


def _read_vtu(path):
    """Minimal reader for the VTU files ``FieldResults.to_vtu`` writes.

    Handles the two encodings the writer emits: appended raw data and
    inline base64 (``format="binary"``), each with UInt64 byte-count
    headers and no compression. Returns a dict with the piece sizes, the
    ``Points`` / ``Cells`` arrays by name, ``point_data`` / ``cell_data``
    dicts and the VTK type name of every array.
    """
    import base64
    import xml.etree.ElementTree as ET

    raw = open(path, 'rb').read()
    marker = b'<AppendedData encoding="raw">'
    appended = b''
    if marker in raw:
        head, tail = raw.split(marker, 1)
        appended = tail[tail.index(b'_') + 1:]
        root = ET.fromstring(head + marker + b'</AppendedData></VTKFile>')
    else:
        root = ET.fromstring(raw)
    assert root.get('type') == 'UnstructuredGrid'
    assert root.get('byte_order') == 'LittleEndian'
    assert root.get('header_type') == 'UInt64'
    piece = root.find('UnstructuredGrid/Piece')
    out = {'n_points': int(piece.get('NumberOfPoints')),
           'n_cells': int(piece.get('NumberOfCells')),
           'arrays': {}, 'point_data': {}, 'cell_data': {}, 'types': {}}

    def _decode(da):
        dtype = np.dtype(_VTU_DTYPES[da.get('type')])
        if da.get('format') == 'appended':
            off = int(da.get('offset'))
            nbytes = int(np.frombuffer(appended[off:off + 8], '<u8')[0])
            data = appended[off + 8:off + 8 + nbytes]
        else:
            assert da.get('format') == 'binary'
            blob = base64.b64decode(da.text.strip())
            nbytes = int(np.frombuffer(blob[:8], '<u8')[0])
            data = blob[8:]
            assert len(data) == nbytes
        arr = np.frombuffer(data, dtype)
        ncomp = int(da.get('NumberOfComponents', 1))
        return arr.reshape(-1, ncomp) if ncomp > 1 else arr

    for section, key in (('Points', 'arrays'), ('Cells', 'arrays'),
                         ('PointData', 'point_data'),
                         ('CellData', 'cell_data')):
        for da in piece.find(section).findall('DataArray'):
            out[key][da.get('Name')] = _decode(da)
            out['types'][da.get('Name')] = da.get('type')
    return out


@pytest.fixture(scope='module')
def solved():
    """A small QI solve with clustered porosity and explicit void elements."""
    material = MATERIALS['T800_epoxy']
    pf = PorosityField(material, 0.03, distribution='clustered',
                       discrete_voids=[VoidGeometry(
                           center=(25.0, 10.0, 2.2), radii=(6.0, 4.0, 1.0))])
    mesh = CompositeMesh(pf, material, nx=10, ny=4, nz=6, ply_angles='QI')
    assert len(mesh.void_elements) > 0
    results = FESolver(mesh, material, pf).solve(
        loading='compression', applied_strain=-0.001)
    return mesh, results


class TestFEExportVTU:
    """Binary VTK XML export with recovered nodal fields (3.6 E1)."""

    @pytest.mark.parametrize('encoding', ['raw', 'base64'])
    def test_round_trip_is_bit_exact(self, solved, tmp_path, encoding):
        mesh, r = solved
        path = tmp_path / f'fe_{encoding}.vtu'
        r.to_vtu(mesh, path, encoding=encoding)
        d = _read_vtu(path)
        assert d['n_points'] == mesh.n_nodes
        assert d['n_cells'] == mesh.n_elements
        np.testing.assert_array_equal(d['arrays']['Points'], mesh.nodes)
        np.testing.assert_array_equal(
            d['arrays']['connectivity'], mesh.elements.ravel())
        np.testing.assert_array_equal(
            d['arrays']['offsets'], 8 * np.arange(1, mesh.n_elements + 1))
        np.testing.assert_array_equal(d['arrays']['types'], 12)
        assert d['types']['Points'] == 'Float64'

        pd, cd = d['point_data'], d['cell_data']
        np.testing.assert_array_equal(pd['displacement'], r.displacement)
        np.testing.assert_array_equal(pd['porosity'], mesh.porosity)
        np.testing.assert_array_equal(pd['ply_id'], mesh.ply_ids)
        s_nodal, _ = r.nodal_stress(mesh)
        e_nodal, _ = r.nodal_strain(mesh)
        for i, name in enumerate(('sigma_xx', 'sigma_yy', 'sigma_zz',
                                  'tau_yz', 'tau_xz', 'tau_xy')):
            np.testing.assert_array_equal(pd[f'{name}_nodal'], s_nodal[:, i])
            np.testing.assert_array_equal(
                cd[name], np.mean(r.stress_global, axis=1)[:, i])
        for i, name in enumerate(('eps_xx', 'eps_yy', 'eps_zz',
                                  'gamma_yz', 'gamma_xz', 'gamma_xy')):
            np.testing.assert_array_equal(pd[f'{name}_nodal'], e_nodal[:, i])
            np.testing.assert_array_equal(
                cd[name], np.mean(r.strain_global, axis=1)[:, i])
        assert np.all(np.isfinite(pd['von_mises_nodal']))
        assert np.all(pd['von_mises_nodal'] >= 0)
        np.testing.assert_array_equal(cd['tsai_wu_index'],
                                      r.per_element_failure_index)
        np.testing.assert_array_equal(cd['ply_id'], mesh.elem_ply_ids)
        np.testing.assert_array_equal(cd['ply_angle_deg'], mesh.ply_angles)
        is_void = np.zeros(mesh.n_elements)
        is_void[mesh.void_elements] = 1.0
        np.testing.assert_array_equal(cd['is_void'], is_void)
        np.testing.assert_array_equal(cd['knockdown'], r.knockdown)

    def test_has_every_legacy_vtk_field(self, solved, tmp_path):
        mesh, r = solved
        r.to_vtk(mesh, tmp_path / 'legacy.vtk')
        r.to_vtu(mesh, tmp_path / 'new.vtu')
        legacy = _parse_legacy_vtk(str(tmp_path / 'legacy.vtk'))
        d = _read_vtu(tmp_path / 'new.vtu')
        assert legacy['point_data_arrays'] == list(d['point_data'])[:len(
            legacy['point_data_arrays'])]
        assert legacy['cell_data_arrays'] == list(d['cell_data'])

    def test_float32_option(self, solved, tmp_path):
        mesh, r = solved
        r.to_vtu(mesh, tmp_path / 'f64.vtu')
        r.to_vtu(mesh, tmp_path / 'f32.vtu', precision='float32')
        d = _read_vtu(tmp_path / 'f32.vtu')
        assert d['types']['Points'] == 'Float32'
        assert d['types']['sigma_xx_nodal'] == 'Float32'
        assert d['types']['connectivity'] == 'Int64'
        np.testing.assert_array_equal(
            d['point_data']['displacement'],
            r.displacement.astype(np.float32))
        assert (tmp_path / 'f32.vtu').stat().st_size < \
            0.6 * (tmp_path / 'f64.vtu').stat().st_size

    def test_exploded_output_carries_per_element_corners(self, solved, tmp_path):
        mesh, r = solved
        path = tmp_path / 'exploded.vtu'
        r.to_vtu(mesh, path, exploded=True)
        d = _read_vtu(path)
        corners = mesh.elements.ravel()
        assert d['n_points'] == 8 * mesh.n_elements
        np.testing.assert_array_equal(d['arrays']['Points'], mesh.nodes[corners])
        np.testing.assert_array_equal(
            d['arrays']['connectivity'], np.arange(8 * mesh.n_elements))
        np.testing.assert_array_equal(d['point_data']['node_id'], corners)
        np.testing.assert_array_equal(
            d['point_data']['displacement'], r.displacement[corners])
        _, corner = r.nodal_stress(mesh)
        np.testing.assert_array_equal(
            d['point_data']['sigma_xx_nodal'], corner[:, :, 0].ravel())
        _, raw = r.nodal_stress(mesh, average='none')
        r.to_vtu(mesh, path, exploded=True, average='none')
        np.testing.assert_array_equal(
            _read_vtu(path)['point_data']['tau_xz_nodal'], raw[:, :, 4].ravel())

    def test_nodal_false_omits_recovered_fields(self, solved, tmp_path):
        mesh, r = solved
        r.to_vtu(mesh, tmp_path / 'plain.vtu', nodal=False)
        d = _read_vtu(tmp_path / 'plain.vtu')
        assert not [k for k in d['point_data'] if k.endswith('_nodal')]
        assert 'displacement' in d['point_data']

    def test_base64_file_is_well_formed_xml(self, solved, tmp_path):
        import xml.etree.ElementTree as ET
        mesh, r = solved
        r.to_vtu(mesh, tmp_path / 'b64.vtu', encoding='base64')
        root = ET.parse(tmp_path / 'b64.vtu').getroot()
        assert root.tag == 'VTKFile'

    @pytest.mark.parametrize('kwargs', [
        {'precision': 'float16'}, {'encoding': 'zlib'}, {'average': 'mean'}])
    def test_rejects_unknown_options(self, solved, tmp_path, kwargs):
        mesh, r = solved
        with pytest.raises(ValueError):
            r.to_vtu(mesh, tmp_path / 'bad.vtu', **kwargs)

    def test_rejects_mismatched_mesh(self, solved, tmp_path):
        _, r = solved
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03)
        other = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        with pytest.raises(ValueError, match='do not match'):
            r.to_vtu(other, tmp_path / 'bad.vtu')

    def test_export_results_fmt_vtu(self, solved, tmp_path):
        mesh, r = solved
        path = tmp_path / 'via_export.vtu'
        FESolver.export_results(r, path, fmt='vtu', mesh=mesh)
        assert _read_vtu(path)['n_cells'] == mesh.n_elements
        with pytest.raises(ValueError, match='mesh'):
            FESolver.export_results(r, tmp_path / 'x.vtu', fmt='vtu')

    @pytest.mark.parametrize('encoding', ['raw', 'base64'])
    def test_meshio_reads_it_if_available(self, solved, tmp_path, encoding):
        """meshio is not a dependency; the test self-skips without it."""
        meshio = pytest.importorskip("meshio")
        mesh, r = solved
        path = tmp_path / 'fe.vtu'
        r.to_vtu(mesh, path, encoding=encoding)
        m = meshio.read(path)
        np.testing.assert_array_equal(m.points, mesh.nodes)
        np.testing.assert_array_equal(m.cells_dict['hexahedron'], mesh.elements)
        np.testing.assert_array_equal(m.point_data['displacement'],
                                      r.displacement)

    def test_vtk_reads_it_if_available(self, solved, tmp_path):
        """VTK is not a dependency; the test self-skips without it."""
        vtk = pytest.importorskip("vtk")
        from vtk.util.numpy_support import vtk_to_numpy
        mesh, r = solved
        path = tmp_path / 'fe.vtu'
        r.to_vtu(mesh, path)
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        assert reader.GetErrorCode() == 0
        grid = reader.GetOutput()
        assert grid.GetNumberOfCells() == mesh.n_elements
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(grid)
        quality.SetHexQualityMeasureToVolume()
        quality.Update()
        volume = vtk_to_numpy(
            quality.GetOutput().GetCellData().GetArray('Quality')).sum()
        # Positive cell volumes summing to the coupon: hex ordering is right.
        assert volume == pytest.approx(mesh.L_x * mesh.L_y * mesh.L_z, rel=1e-9)


class TestWritePVD:
    def test_writes_relative_series(self, tmp_path):
        import xml.etree.ElementTree as ET
        from porosity_fe import write_pvd
        (tmp_path / 'data').mkdir()
        files = [tmp_path / 'data' / f'vp_{i}.vtu' for i in range(3)]
        write_pvd(tmp_path / 'series.pvd', files, timesteps=[0.0, 0.02, 0.04])
        root = ET.parse(tmp_path / 'series.pvd').getroot()
        assert root.get('type') == 'Collection'
        rows = root.findall('Collection/DataSet')
        assert [r.get('file') for r in rows] == [
            f'data/vp_{i}.vtu' for i in range(3)]
        assert [float(r.get('timestep')) for r in rows] == [0.0, 0.02, 0.04]

    def test_default_timesteps_and_validation(self, tmp_path):
        import xml.etree.ElementTree as ET
        from porosity_fe import write_pvd
        write_pvd(tmp_path / 's.pvd', ['a.vtu', 'b.vtu'])
        rows = ET.parse(tmp_path / 's.pvd').getroot().findall('Collection/DataSet')
        assert [float(r.get('timestep')) for r in rows] == [0.0, 1.0]
        with pytest.raises(ValueError, match='one finite value per file'):
            write_pvd(tmp_path / 's.pvd', ['a.vtu', 'b.vtu'], timesteps=[0.0])
        with pytest.raises(ValueError, match='one finite value per file'):
            write_pvd(tmp_path / 's.pvd', ['a.vtu'], timesteps=[float('nan')])


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


@pytest.fixture(autouse=True)
def _fresh_git_sha_cache():
    """``_git_commit_sha`` is cached per process; tests that mock
    ``subprocess.run`` need a cold cache and must not leak their result."""
    from porosity_fe import io as io_mod
    io_mod._git_commit_sha.cache_clear()
    yield
    io_mod._git_commit_sha.cache_clear()


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

    def test_timestamp_format_is_naive_iso_plus_z(self):
        """The stamp keeps its original ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z``
        shape; a timezone-aware isoformat() would add ``+00:00``."""
        import re
        prov = _build_provenance()
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?Z"
        assert re.fullmatch(pattern, prov['timestamp_utc'])

    def test_no_deprecation_warning(self):
        """``datetime.utcnow()`` is deprecated on Python 3.12+."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            _build_provenance()

    def test_git_is_queried_once_per_process(self, monkeypatch):
        from porosity_fe import io as io_mod
        calls = []
        real_run = io_mod.subprocess.run

        def _counting_run(*args, **kwargs):
            calls.append(args)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(io_mod.subprocess, "run", _counting_run)
        first = _build_provenance()
        second = _build_provenance()
        assert len(calls) == 1
        assert first['git_commit'] == second['git_commit']

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


def test_provenance_aliases_mirror_canonical_keys_and_are_optional():
    """IMPROVEMENT_PLAN 4.7: one canonical spelling; the #55 aliases are still
    written (same values) but the schema no longer requires them."""
    import json as _json
    from pathlib import Path as _Path

    import jsonschema
    from porosity_fe.io import _DEPRECATED_PROVENANCE_ALIASES, _build_provenance
    prov = _build_provenance(seed=3)
    for alias, canonical in _DEPRECATED_PROVENANCE_ALIASES.items():
        assert prov[alias] == prov[canonical]
    schema_path = (_Path(__file__).resolve().parent.parent / 'validation' / 'schemas'
                   / 'porosity_results_schema.json')
    schema = _json.loads(schema_path.read_text(encoding='utf-8'))
    required = set(schema['properties']['provenance']['required'])
    assert required.isdisjoint(_DEPRECATED_PROVENANCE_ALIASES)
    trimmed = {k: v for k, v in prov.items() if k not in _DEPRECATED_PROVENANCE_ALIASES}
    jsonschema.validate({'schema_version': '1.1', 'format': 'porosity-fe.ncr',
                         'provenance': trimmed}, schema)
