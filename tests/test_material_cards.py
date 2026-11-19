"""Solver material cards of the porosity-degraded ply."""

import dataclasses

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    DegradedPly,
    PorosityField,
    PorosityMap,
    VoidGeometry,
    degraded_ply,
    material_cards,
    ply_porosities,
    write_material_cards,
)
from porosity_fe.fe.failure import degraded_strengths
from porosity_fe.homogenization import _degraded_composite_stiffness
from porosity_fe.material_cards import _field16, _nastran_entry, _num
from porosity_fe.void_geometry import VOID_SHAPES


@pytest.fixture
def material():
    return MATERIALS['T800_epoxy']


@pytest.fixture
def laminate():
    """Eight plies with thermal expansion coefficients."""
    return dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=8)


def _data_lines(text, comment):
    return [line for line in text.splitlines() if not line.startswith(comment)]


def _comments(text, comment):
    """The comment lines of a deck as one string (they are wrapped)."""
    return ' '.join(line[len(comment):].strip() for line in text.splitlines()
                    if line.startswith(comment))


def _fields16(line):
    body = line[8:]
    return [body[i:i + 16].strip() for i in range(0, len(body), 16)]


class TestDegradedPly:
    def test_no_porosity_is_the_material(self, material):
        ply = degraded_ply(material, 0.0)
        for name in ('E11', 'E22', 'E33', 'G12', 'G13', 'G23', 'nu12', 'nu13', 'nu23'):
            assert getattr(ply, name) == pytest.approx(getattr(material, name), rel=1e-12)
            assert ply.retention[name] == pytest.approx(1.0, rel=1e-12)
        assert (ply.Xt, ply.Xc, ply.Yt, ply.Yc, ply.S12, ply.S23) == pytest.approx((
            material.sigma_1t, material.sigma_1c, material.sigma_2t,
            material.sigma_2c, material.tau_12, material.tau_ilss))

    def test_constants_rebuild_the_fe_stiffness(self, material):
        """The engineering constants are those of the stiffness an FE
        element gets at this void content."""
        ply = degraded_ply(material, 0.04, void_shape='penny')
        C = _degraded_composite_stiffness(0.04, VOID_SHAPES['penny'], material)
        np.testing.assert_allclose(ply.stiffness, C, rtol=1e-13)
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / ply.E11, 1 / ply.E22, 1 / ply.E33
        S[0, 1] = S[1, 0] = -ply.nu12 / ply.E11
        S[0, 2] = S[2, 0] = -ply.nu13 / ply.E11
        S[1, 2] = S[2, 1] = -ply.nu23 / ply.E22
        S[3, 3], S[4, 4], S[5, 5] = 1 / ply.G23, 1 / ply.G13, 1 / ply.G12
        np.testing.assert_allclose(np.linalg.inv(S), C, rtol=1e-10, atol=1e-6)

    def test_matrix_constants_lose_more_than_the_fibre_one(self, material):
        ply = degraded_ply(material, 0.05)
        assert ply.retention['E11'] > 0.999
        assert ply.retention['E22'] < 0.93
        assert ply.retention['G12'] < 0.93
        assert ply.E22 < material.E22 and ply.G12 < material.G12

    def test_degraded_strengths_are_the_fe_ones(self, material):
        ply = degraded_ply(material, 0.05)
        assert (ply.Xt, ply.Xc, ply.Yt, ply.Yc, ply.S12, ply.S23) == pytest.approx(
            degraded_strengths(material, VOID_SHAPES['spherical'], 0.05))
        assert ply.Yt < material.sigma_2t
        assert ply.strengths == 'degraded'

    def test_pristine_strengths(self, material):
        ply = degraded_ply(material, 0.05, strengths='pristine')
        assert ply.Yt == material.sigma_2t and ply.S12 == material.tau_12
        assert ply.E22 < material.E22          # the stiffness still degrades
        assert ply.strengths == 'pristine'

    @pytest.mark.parametrize('strengths', ['degraded', 'pristine'])
    def test_interlaminar_strengths_only_when_set(self, material, strengths):
        assert degraded_ply(material, 0.03, strengths=strengths).Zt is None
        assert degraded_ply(material, 0.03, strengths=strengths).S13 is None
        mat = dataclasses.replace(material, sigma_3t=60.0, tau_13=95.0)
        ply = degraded_ply(mat, 0.03, strengths=strengths)
        if strengths == 'pristine':
            assert (ply.Zt, ply.S13) == (60.0, 95.0)
        else:
            assert 0.0 < ply.Zt < 60.0 and 0.0 < ply.S13 < 95.0

    def test_directional_keeps_unset_interlaminar_strengths_unset(self, material):
        ply = degraded_ply(material, 0.03, void_shape='penny',
                           porosity_degradation='directional')
        assert ply.Zt is None and ply.S13 is None
        mat = dataclasses.replace(material, sigma_3t=60.0)
        ply = degraded_ply(mat, 0.03, void_shape='penny',
                           porosity_degradation='directional')
        assert ply.Zt is not None and ply.S13 is None

    def test_directional_weakens_the_thickness_direction_more(self, material):
        iso = degraded_ply(material, 0.04, void_shape='penny')
        direc = degraded_ply(material, 0.04, void_shape='penny',
                             porosity_degradation='directional')
        assert iso.E33 == pytest.approx(iso.E22)
        assert direc.E33 < direc.E22

    def test_explicit_radii(self, material):
        named = degraded_ply(material, 0.03, void_shape='cylindrical')
        radii = degraded_ply(material, 0.03, void_shape=(3.0, 1.0, 1.0))
        assert radii.E22 == named.E22

    def test_to_dict_is_json_ready(self, material):
        import json
        record = degraded_ply(material, 0.02).to_dict()
        assert record['Vp'] == 0.02 and record['Zt'] is None
        assert np.shape(record['stiffness']) == (6, 6)
        assert set(record['retention']) == {
            'E11', 'E22', 'E33', 'G12', 'G13', 'G23', 'nu12', 'nu13', 'nu23'}
        json.dumps(record)

    def test_is_a_frozen_dataclass(self, material):
        ply = degraded_ply(material, 0.02)
        assert isinstance(ply, DegradedPly)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ply.E11 = 1.0
        assert 'stiffness' not in repr(ply)

    @pytest.mark.parametrize('Vp, match', [
        (2.0, 'percent'), (-0.1, r'\[0, 1\]'), (0.995, 'no ply'),
        ([0.01, 0.02], 'one void fraction')])
    def test_bad_void_content(self, material, Vp, match):
        with pytest.raises(ValueError, match=match):
            degraded_ply(material, Vp)

    @pytest.mark.parametrize('kwargs, match', [
        ({'void_shape': 'cubic'}, 'Unknown void_shape'),
        ({'void_shape': (1.0, 2.0)}, 'three finite radii'),
        ({'void_shape': (1.0, -2.0, 1.0)}, 'three finite radii'),
        ({'void_shape': 3}, 'three finite radii'),
        ({'strengths': 'none'}, "'degraded' or 'pristine'"),
        ({'porosity_degradation': 'fancy'}, 'porosity_degradation'),
    ])
    def test_bad_options(self, material, kwargs, match):
        with pytest.raises(ValueError, match=match):
            degraded_ply(material, 0.02, **kwargs)


class TestPlyPorosities:
    def test_uniform(self, laminate):
        plies = ply_porosities(PorosityField(laminate, 0.03))
        np.testing.assert_allclose(plies, 0.03, rtol=1e-12)
        assert plies.shape == (8,)

    def test_clustered_profile_keeps_the_mean_and_peaks_at_the_midplane(self, laminate):
        pf = PorosityField(laminate, 0.03, distribution='clustered')
        plies = ply_porosities(pf)
        assert plies.mean() == pytest.approx(0.03, rel=1e-12)
        np.testing.assert_allclose(plies, plies[::-1], rtol=1e-9)
        assert plies[3] > plies[2] > plies[1] > plies[0] > 0.0
        assert plies[3] > 2.0 * 0.03

    def test_surface_cluster_is_one_sided(self, laminate):
        plies = ply_porosities(PorosityField(
            laminate, 0.03, distribution='clustered', cluster_location='surface'))
        assert np.all(np.diff(plies) < 0.0)

    def test_matches_the_profile_integral(self, laminate):
        pf = PorosityField(laminate, 0.02, distribution='interface')
        plies = ply_porosities(pf)
        t = laminate.t_ply
        z = (np.arange(4000) + 0.5) / 4000 * t
        fine = np.array([pf._distributed_porosity(k * t + z).mean() for k in range(8)])
        fine *= 0.02 / fine.mean()
        np.testing.assert_allclose(plies, fine, rtol=1e-5)

    def test_zero_porosity(self, laminate):
        np.testing.assert_array_equal(
            ply_porosities(PorosityField(laminate, 0.0, distribution='clustered')), 0.0)

    def test_in_plane_map_point(self, laminate):
        pmap = PorosityMap([0.0, 50.0], [0.0, 20.0], [[0.01, 0.01], [0.03, 0.03]])
        pf = PorosityField.from_map(laminate, pmap)
        np.testing.assert_allclose(ply_porosities(pf), 0.02, rtol=1e-12)
        np.testing.assert_allclose(ply_porosities(pf, at=(50.0, 10.0)), 0.03, rtol=1e-12)
        np.testing.assert_allclose(ply_porosities(pf, at=(0.0, 10.0)), 0.01, rtol=1e-12)

    def test_point_without_a_map_changes_nothing(self, laminate):
        pf = PorosityField(laminate, 0.03, distribution='clustered')
        np.testing.assert_array_equal(ply_porosities(pf, at=(1.0, 2.0)),
                                      ply_porosities(pf))

    def test_values_are_clipped_at_one(self, laminate):
        pf = PorosityField(laminate, 0.6, distribution='clustered')
        plies = ply_porosities(pf)
        assert plies.max() == 1.0 and plies.min() >= 0.0

    def test_bad_arguments(self, laminate):
        with pytest.raises(TypeError, match='PorosityField'):
            ply_porosities(0.02)
        with pytest.raises(ValueError, match=r'point \(x, y\)'):
            ply_porosities(PorosityField(laminate, 0.02), at=5.0)


class TestAbaqus:
    def test_shell_material(self, material):
        text = material_cards(material, 0.02)
        ply = degraded_ply(material, 0.02)
        lines = text.splitlines()
        assert text.endswith('\n')
        assert '*MATERIAL, NAME=POROUS_PLY' in lines
        elastic = lines[lines.index('*ELASTIC, TYPE=LAMINA') + 1]
        assert [float(v) for v in elastic.split(',')] == pytest.approx(
            [ply.E11, ply.E22, ply.nu12, ply.G12, ply.G13, ply.G23], rel=1e-8)
        # Abaqus takes the compressive limits as negative numbers, then f*.
        limits = lines[lines.index('*FAIL STRESS') + 1]
        assert [float(v) for v in limits.split(',')] == pytest.approx(
            [ply.Xt, -ply.Xc, ply.Yt, -ply.Yc, ply.S12, -0.5], rel=1e-8)
        assert '*EXPANSION, TYPE=ORTHO' not in text and '*DENSITY' not in text

    def test_solid_material(self, laminate):
        text = material_cards(laminate, 0.02, element='solid', density=1.6e-9)
        ply = degraded_ply(laminate, 0.02)
        lines = text.splitlines()
        i = lines.index('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
        first = [float(v) for v in lines[i + 1].rstrip(',').split(',')]
        assert lines[i + 1].endswith(',') and len(first) == 8
        assert first == pytest.approx([ply.E11, ply.E22, ply.E33, ply.nu12, ply.nu13,
                                       ply.nu23, ply.G12, ply.G13], rel=1e-8)
        assert float(lines[i + 2]) == pytest.approx(ply.G23, rel=1e-8)
        assert float(lines[lines.index('*DENSITY') + 1]) == pytest.approx(
            1.6e-9 * 0.98, rel=1e-8)
        cte = lines[lines.index('*EXPANSION, TYPE=ORTHO') + 1]
        assert [float(v) for v in cte.split(',')] == pytest.approx(
            [laminate.alpha_1, laminate.alpha_2, laminate.alpha_2])
        # No plane-stress failure keyword on a solid material: comments.
        assert '*FAIL STRESS' not in _data_lines(text, '**')
        assert any(line.startswith('** Strengths in MPa') for line in lines)

    @pytest.mark.parametrize('kwargs', [{'section': 'LAMINATE'}, {'element': 'solid'}])
    def test_every_line_is_a_comment_a_keyword_or_data(self, laminate, kwargs):
        pf = PorosityField(laminate, 0.03, distribution='clustered')
        text = material_cards(laminate, pf, density=1.6e-9, **kwargs)
        for line in text.splitlines():
            # Comments fit 80 columns; Abaqus reads data lines to 256.
            assert line.strip() and len(line) <= (80 if line.startswith('**') else 256)
            assert line.startswith(('**', '*')) or line[0].isdigit() or line[0] == '-'

    def test_one_material_per_distinct_ply_porosity(self, laminate):
        pf = PorosityField(laminate, 0.03, distribution='clustered')
        text = material_cards(laminate, pf, section='PLATE')
        lines = text.splitlines()
        names = [line.split('=')[1] for line in lines if line.startswith('*MATERIAL')]
        assert names == ['POROUS_PLY_1', 'POROUS_PLY_2', 'POROUS_PLY_3', 'POROUS_PLY_4']
        i = lines.index('*SHELL SECTION, ELSET=PLATE, COMPOSITE')
        rows = [[c.strip() for c in line.split(',')] for line in lines[i + 1:i + 9]]
        assert [r[2] for r in rows] == [
            'POROUS_PLY_1', 'POROUS_PLY_2', 'POROUS_PLY_3', 'POROUS_PLY_4',
            'POROUS_PLY_4', 'POROUS_PLY_3', 'POROUS_PLY_2', 'POROUS_PLY_1']
        assert [float(r[3]) for r in rows] == [0, 90, 45, -45, -45, 45, 90, 0]
        assert all(float(r[0]) == laminate.t_ply and r[1] == '3' for r in rows)
        assert [r[4] for r in rows] == [f'PLY_{k}' for k in range(1, 9)]
        assert len(lines) == i + 9

    def test_ply_table_is_a_comment_without_a_section(self, laminate):
        text = material_cards(laminate, 0.02, ply_angles=[0, 45, -45, 90])
        assert '*SHELL SECTION' not in text
        table = [line for line in text.splitlines() if 'PLY_' in line]
        assert len(table) == 8 and all(line.startswith('**   ') for line in table)
        # The layup is repeated to the ply count, as the mesh lays it up.
        assert [float(line.split(',')[3]) for line in table] == [0, 45, -45, 90] * 2

    def test_strength_options(self, material):
        degraded = material_cards(material, 0.05)
        pristine = material_cards(material, 0.05, strengths='pristine')
        none = material_cards(material, 0.05, strengths='none')
        assert 'not calibrated' in _comments(degraded, '**')
        assert '*FAIL STRESS' in degraded
        assert 'Strengths: pristine' in pristine
        limits = pristine.splitlines()[pristine.splitlines().index('*FAIL STRESS') + 1]
        assert float(limits.split(',')[2]) == material.sigma_2t
        assert '*FAIL STRESS' not in none and 'Strengths: not written.' in none
        assert 'Interlaminar strengths' not in none

    def test_interaction_coefficient_and_interlaminar_strengths(self, material):
        mat = dataclasses.replace(material, tsai_wu_F12=-0.25, sigma_3t=60.0, tau_13=95.0)
        lines = material_cards(mat, 0.0).splitlines()
        assert lines[lines.index('*FAIL STRESS') + 1].endswith(', -0.25')
        assert '** Interlaminar strengths (MPa): S23 = 90., Zt = 60., S13 = 95.' in lines
        solid = material_cards(mat, 0.0, element='solid')
        assert 'Zt = 60., S13 = 95.' in solid
        nastran = material_cards(mat, 0.0, 'nastran', element='solid')
        assert 'Zt = 60., S13 = 95.' in nastran

    def test_header_describes_the_porosity(self, laminate):
        pmap = PorosityMap([0.0, 50.0], [0.0, 20.0], [[0.01, 0.01], [0.03, 0.03]])
        void = VoidGeometry(center=(10.0, 10.0, 0.5), radii=(1.0, 1.0, 0.2))
        pf = PorosityField.from_map(laminate, pmap, distribution='clustered',
                                    cluster_location='surface', void_shape='penny',
                                    discrete_voids=[void])
        header = _comments(material_cards(laminate, pf), '**')
        for phrase in ('clustered profile', 'specimen average Vp = 0.02',
                       'clustered at the surface', 'in-plane map averaged',
                       '1 discrete void(s) not included', 'void shape penny',
                       'isotropic degradation', 'Units: MPa, mm'):
            assert phrase in header
        assert 'at (x, y) = (50, 5) mm' in _comments(
            material_cards(laminate, pf, at=(50.0, 5.0)), '**')

    def test_map_point_changes_the_cards(self, laminate):
        pmap = PorosityMap([0.0, 50.0], [0.0, 20.0], [[0.01, 0.01], [0.03, 0.03]])
        pf = PorosityField.from_map(laminate, pmap)
        assert 'Vp = 0.02:' in material_cards(laminate, pf)
        assert 'Vp = 0.03:' in material_cards(laminate, pf, at=(50.0, 5.0))

    def test_field_void_shape_and_override(self, laminate):
        penny = PorosityField(laminate, 0.03, void_shape='penny')
        assert 'void shape penny' in _comments(material_cards(laminate, penny), '**')
        assert 'void shape spherical' in _comments(
            material_cards(laminate, penny, void_shape='spherical'), '**')
        assert 'void shape radii (2.0, 1.0, 1.0)' in _comments(
            material_cards(laminate, 0.03, void_shape=(2.0, 1.0, 1.0)), '**')
        # A uniform field and its void fraction give the same materials.
        assert _data_lines(material_cards(laminate, penny), '**') == _data_lines(
            material_cards(laminate, 0.03, void_shape='penny'), '**')

    def test_custom_name(self, material):
        assert '*MATERIAL, NAME=T800-VOIDS' in material_cards(
            material, 0.01, name='T800-VOIDS')


class TestNastran:
    def test_mat8(self, laminate):
        text = material_cards(laminate, 0.02, 'nastran', first_id=7, density=1.6e-9)
        ply = degraded_ply(laminate, 0.02)
        data = _data_lines(text, '$')
        assert data[0].startswith('MAT8*   ') and all(d.startswith('*       ')
                                                      for d in data[1:])
        fields = [f for line in data for f in _fields16(line)]
        assert int(fields[0]) == 7
        values = [None if f == '' else float(f) for f in fields[1:]]
        expected = [ply.E11, ply.E22, ply.nu12, ply.G12, ply.G13, ply.G23,
                    1.6e-9 * 0.98, laminate.alpha_1, laminate.alpha_2, None,
                    # Nastran takes the compressive limits as positive numbers.
                    ply.Xt, ply.Xc, ply.Yt, ply.Yc, ply.S12, None,
                    -0.5 / np.sqrt(ply.Xt * ply.Xc * ply.Yt * ply.Yc)]
        assert len(values) == len(expected)
        for got, want in zip(values, expected, strict=True):
            assert got == (None if want is None else pytest.approx(want, rel=1e-9))

    def test_mat8_without_strengths_density_or_cte(self, material):
        data = _data_lines(material_cards(material, 0.02, 'nastran',
                                          strengths='none'), '$')
        fields = [f for line in data for f in _fields16(line)]
        assert len(data) == 2 and len(fields) == 7     # MID ... G2Z

    def test_mat9_is_the_stiffness_in_nastran_order(self, laminate):
        text = material_cards(laminate, 0.03, 'nastran', element='solid')
        ply = degraded_ply(laminate, 0.03)
        data = _data_lines(text, '$')
        assert data[0].startswith('MAT9*   ')
        fields = [f for line in data for f in _fields16(line)]
        G = np.zeros((6, 6))
        G[np.triu_indices(6)] = [float(f) for f in fields[1:22]]
        G = G + np.triu(G, 1).T
        C = ply.stiffness
        # Nastran orders the shears xy, yz, zx; this package 23, 13, 12.
        assert G[3, 3] == pytest.approx(ply.G12, rel=1e-9)
        assert G[4, 4] == pytest.approx(ply.G23, rel=1e-9)
        assert G[5, 5] == pytest.approx(ply.G13, rel=1e-9)
        np.testing.assert_allclose(G[:3, :3], C[:3, :3], rtol=1e-9)
        assert np.count_nonzero(G[:3, 3:]) == 0
        assert fields[22] == ''                          # no density
        assert [float(f) for f in fields[23:29]] == pytest.approx(
            [laminate.alpha_1, laminate.alpha_2, laminate.alpha_2, 0, 0, 0])
        assert any(line.startswith('$ Strengths (MPa): Xt = ') for line in
                   text.splitlines())

    def test_mat9_without_cte(self, material):
        data = _data_lines(material_cards(material, 0.03, 'nastran', element='solid',
                                          density=1.6e-9, strengths='none'), '$')
        fields = [f for line in data for f in _fields16(line)]
        assert len(fields) == 23 and float(fields[22]) == pytest.approx(1.6e-9 * 0.97)

    def test_pcomp(self, laminate):
        pf = PorosityField(laminate, 0.03, distribution='clustered')
        text = material_cards(laminate, pf, 'nastran', section=20, first_id=101)
        data = _data_lines(text, '$')
        start = next(i for i, line in enumerate(data) if line.startswith('PCOMP*'))
        pcomp = data[start:]
        head = _fields16(pcomp[0]) + _fields16(pcomp[1])
        plies = [degraded_ply(laminate, vp) for vp in ply_porosities(pf)]
        assert int(head[0]) == 20 and head[1] == head[2] == ''
        assert float(head[3]) == pytest.approx(min(p.S23 for p in plies), rel=1e-9)
        assert head[4] == 'TSAI'
        rows = [_fields16(line) for line in pcomp[2:]]
        assert len(rows) == 8
        assert [int(r[0]) for r in rows] == [101, 102, 103, 104, 104, 103, 102, 101]
        assert [float(r[1]) for r in rows] == [laminate.t_ply] * 8
        assert [float(r[2]) for r in rows] == [0, 90, 45, -45, -45, 45, 90, 0]
        assert all(r[3] == 'YES' for r in rows)
        assert sum(line.startswith('MAT8*') for line in data) == 4

    def test_pcomp_without_strengths_has_no_failure_theory(self, laminate):
        data = _data_lines(material_cards(laminate, 0.02, 'nastran', section=3,
                                          strengths='none'), '$')
        start = next(i for i, line in enumerate(data) if line.startswith('PCOMP*'))
        head = _fields16(data[start]) + _fields16(data[start + 1])
        assert int(head[0]) == 3 and all(f == '' for f in head[1:])
        assert len(data) == start + 2 + 8

    @pytest.mark.parametrize('kwargs', [{'section': 20}, {'element': 'solid'}])
    def test_lines_fit_the_large_field_format(self, laminate, kwargs):
        pf = PorosityField(laminate, 0.03, distribution='clustered')
        text = material_cards(laminate, pf, 'nastran', density=1.6e-9, **kwargs)
        for line in text.splitlines():
            assert len(line) <= 72 and line.strip()
            assert line.startswith(('$', 'MAT8*', 'MAT9*', 'PCOMP*', '*'))

    def test_ply_table_comment(self, laminate):
        text = material_cards(laminate, 0.02, 'nastran')
        assert 'PCOMP' not in text
        assert '$   1, 1, 0.125, 0.' in text.splitlines()
        assert '$   4, 1, 0.125, -45.' in text.splitlines()

    def test_entry_writer(self):
        assert _field16(None) == ' ' * 16
        assert _field16(12) == f"{12:>16d}"
        assert _field16(np.int64(3)) == f"{3:>16d}"
        assert _field16('TSAI') == 'TSAI' + ' ' * 12
        assert _field16(-161000.0) == '-1.610000000E+05'
        assert _field16(np.float64(0.5)) == ' 5.000000000E-01'
        with pytest.raises(TypeError):
            _field16(True)
        # Eight fields to a logical line, four to a physical one; trailing
        # blanks are dropped, inner ones kept.
        lines = _nastran_entry('MAT1', [1, 2.0, None, 3.0, 4.0, None, None])
        assert lines == ['MAT1*   ' + _field16(1) + _field16(2.0) + ' ' * 16
                         + _field16(3.0), '*       ' + _field16(4.0)]
        assert _nastran_entry('MAT1', [None, None]) == []

    def test_number_text(self):
        assert _num(161000.0) == '161000.'
        assert _num(0.32) == '0.32'
        assert _num(2.6e-05) == '2.6e-05'
        assert _num(-45.0) == '-45.'


class TestMaterialCardsArguments:
    def test_formats_agree_on_the_constants(self, laminate):
        abaqus = material_cards(laminate, 0.04).splitlines()
        nastran = _data_lines(material_cards(laminate, 0.04, 'nastran'), '$')
        e_abaqus = [float(v) for v in
                    abaqus[abaqus.index('*ELASTIC, TYPE=LAMINA') + 1].split(',')]
        fields = [f for line in nastran for f in _fields16(line)]
        assert [float(f) for f in fields[1:7]] == pytest.approx(e_abaqus, rel=1e-8)

    def test_layup_longer_than_the_laminate(self, laminate):
        with pytest.raises(ValueError, match='layup has 10 plies'):
            material_cards(laminate, 0.02, ply_angles=[0] * 10)

    def test_field_of_another_laminate(self, material, laminate):
        with pytest.raises(ValueError, match='same laminate'):
            material_cards(material, PorosityField(laminate, 0.02))

    @pytest.mark.parametrize('kwargs, match', [
        ({'fmt': 'ansys'}, 'fmt must be one of'),
        ({'element': 'beam'}, 'element must be one of'),
        ({'strengths': 'weak'}, 'strengths must be one of'),
        ({'density': -1.0}, 'density must be a positive number'),
        ({'density': float('nan')}, 'density must be a positive number'),
        ({'name': '1PLY'}, 'name must start with a letter'),
        ({'name': 'PLY 1'}, 'name must start with a letter'),
        ({'name': ''}, 'name must start with a letter'),
        ({'at': (1.0, 2.0)}, 'needs a PorosityField'),
        ({'section': 5}, 'element-set name'),
        ({'section': '  '}, 'element-set name'),
        ({'section': 'PLATE', 'element': 'solid'}, 'composite shell section'),
        ({'fmt': 'nastran', 'section': 'PLATE'}, 'property id'),
        ({'fmt': 'nastran', 'section': 0}, 'property id'),
        ({'fmt': 'nastran', 'section': True}, 'property id'),
        ({'fmt': 'nastran', 'section': 4, 'element': 'solid'}, 'PCOMP'),
        ({'fmt': 'nastran', 'first_id': 0}, 'first_id'),
        ({'fmt': 'nastran', 'first_id': 1.5}, 'first_id'),
        ({'fmt': 'nastran', 'first_id': True}, 'first_id'),
        ({'ply_angles': 'XX'}, "'QI' or 'UD'"),
    ])
    def test_bad_options(self, laminate, kwargs, match):
        with pytest.raises(ValueError, match=match):
            material_cards(laminate, 0.02, **kwargs)

    @pytest.mark.parametrize('porosity, match', [
        ([0.01, 0.02], 'void fraction or a PorosityField'),
        (3.0, 'percent'),
        (0.995, 'no ply'),
    ])
    def test_bad_porosity(self, laminate, porosity, match):
        with pytest.raises(ValueError, match=match):
            material_cards(laminate, porosity)


class TestWrite:
    @pytest.mark.parametrize('suffix, marker', [
        ('.inp', '*MATERIAL'), ('.bdf', 'MAT8*'), ('.DAT', 'MAT8*'),
        ('.nas', 'MAT8*'), ('.blk', 'MAT8*')])
    def test_format_from_the_extension(self, tmp_path, material, suffix, marker):
        path = tmp_path / f"cards{suffix}"
        write_material_cards(path, material, 0.02)
        assert marker in path.read_text(encoding='utf-8')

    def test_explicit_format_and_options(self, tmp_path, laminate):
        path = tmp_path / "cards.txt"
        write_material_cards(str(path), laminate, 0.02, 'nastran', element='solid')
        text = path.read_bytes()
        assert b'MAT9*' in text and b'\r' not in text
        assert text.decode() == material_cards(laminate, 0.02, 'nastran',
                                               element='solid')

    def test_unknown_extension(self, tmp_path, material):
        with pytest.raises(ValueError, match="Cannot tell the format"):
            write_material_cards(tmp_path / "cards.txt", material, 0.02)
        assert not (tmp_path / "cards.txt").exists()
