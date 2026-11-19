"""In-plane porosity maps: ``PorosityMap``, ``PorosityPatch`` and the
``in_plane_map`` of ``PorosityField``."""

from __future__ import annotations

import json
import logging
import math
import warnings

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    PorosityField,
    PorosityMap,
    PorosityPatch,
    build_empirical_pipeline,
)
from porosity_fe.io import _dumps_json

MAT = MATERIALS['T700_epoxy']
X = [0.0, 25.0, 50.0]
Y = [0.0, 20.0]
RAMP = [[0.01, 0.01], [0.02, 0.02], [0.03, 0.03]]       # linear in x


def _hot_spot(background=0.01, peak=0.06):
    return PorosityMap.from_patches(
        50.0, 20.0, background, [PorosityPatch((25.0, 10.0), (8.0, 5.0), peak)])


class TestPorosityMap:
    def test_bilinear_inside_and_clamped_outside(self):
        m = PorosityMap(X, Y, RAMP)
        assert m(0.0, 0.0) == 0.01 and m(50.0, 20.0) == 0.03
        assert m(12.5, 7.0) == pytest.approx(0.015)
        assert m(37.5, 20.0) == pytest.approx(0.025)
        assert m(-5.0, 10.0) == 0.01 and m(80.0, -3.0) == 0.03
        out = m(np.array([0.0, 25.0, 50.0]), np.array([0.0, 10.0, 20.0]))
        assert out == pytest.approx([0.01, 0.02, 0.03])
        assert m(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)

    def test_bilinear_in_both_directions(self):
        m = PorosityMap([0.0, 2.0], [0.0, 4.0], [[0.0, 0.04], [0.02, 0.10]])
        assert m(1.0, 2.0) == pytest.approx(0.25 * (0.0 + 0.04 + 0.02 + 0.10))
        assert m(0.5, 1.0) == pytest.approx(
            0.75 * 0.75 * 0.0 + 0.25 * 0.75 * 0.02 + 0.75 * 0.25 * 0.04 + 0.25 * 0.25 * 0.10)

    def test_mean_is_the_area_average_of_the_interpolant(self):
        m = PorosityMap(X, Y, RAMP)
        assert m.mean == pytest.approx(0.02)
        assert (m.max, m.extent) == (0.03, (0.0, 50.0, 0.0, 20.0))
        # A non-uniform grid weighs each cell by its area.
        skewed = PorosityMap([0.0, 10.0, 50.0], Y, [[0.0, 0.0], [0.04, 0.04], [0.04, 0.04]])
        assert skewed.mean == pytest.approx((0.02 * 10.0 + 0.04 * 40.0) / 50.0)
        # Against a fine sampling of the map itself.
        rng = np.random.default_rng(0)
        rough = PorosityMap(np.linspace(0, 50, 6), np.linspace(0, 20, 4),
                            rng.uniform(0.0, 0.05, (6, 4)))
        gx, gy = np.meshgrid(np.linspace(0, 50, 801), np.linspace(0, 20, 401),
                             indexing='ij')
        assert rough.mean == pytest.approx(float(rough(gx, gy).mean()), rel=2e-3)

    @pytest.mark.parametrize("x, y, Vp, match", [
        ([0.0], Y, [[0.01, 0.01]], "x must be at least two"),
        ([0.0, 0.0, 1.0], Y, np.zeros((3, 2)), "strictly increasing"),
        ([1.0, 0.0], Y, np.zeros((2, 2)), "strictly increasing"),
        (X, [0.0, math.nan], np.zeros((3, 2)), "y must be at least two"),
        ([[0.0, 1.0]], Y, np.zeros((2, 2)), "x must be at least two"),
        (X, Y, np.zeros((2, 3)), r"shape \(len\(x\), len\(y\)\) = \(3, 2\)"),
        (X, Y, np.full((3, 2), 2.0), "percent"),
        (X, Y, np.full((3, 2), -0.1), "void fraction"),
    ])
    def test_bad_grids(self, x, y, Vp, match):
        with pytest.raises(ValueError, match=match):
            PorosityMap(x, y, Vp)

    def test_to_dict_and_repr(self):
        m = PorosityMap(X, Y, RAMP)
        assert m.to_dict() == {'x': X, 'y': Y, 'Vp': RAMP}
        assert json.loads(_dumps_json({'config': {'in_plane_map': m}})) == {
            'config': {'in_plane_map': m.to_dict()}}
        assert repr(m) == ("PorosityMap(3 x 2 points, x in [0, 50], y in [0, 20] mm, "
                           "mean Vp=0.0200, max Vp=0.0300)")


class TestPatches:
    def test_patch_shapes(self):
        ellipse = PorosityPatch((10.0, 5.0), (4.0, 2.0), 0.05)
        assert ellipse.shape == 'ellipse' and ellipse.Vp == 0.05
        assert ellipse.contains(10.0, 5.0) and ellipse.contains(14.0, 5.0)
        assert not ellipse.contains(13.5, 6.5)
        rectangle = PorosityPatch((10.0, 5.0), (4.0, 2.0), 0.05, 'rectangle')
        assert rectangle.contains(13.5, 6.5) and not rectangle.contains(14.1, 5.0)
        assert list(rectangle.contains([10.0, 20.0], [5.0, 5.0])) == [True, False]

    @pytest.mark.parametrize("kwargs, match", [
        ({'shape': 'star'}, "Unknown patch shape"),
        ({'center': (1.0,)}, "center must be two numbers"),
        ({'center': (1.0, math.inf)}, "center must be two finite numbers"),
        ({'half_size': (0.0, 1.0)}, "half_size must be two finite numbers above 0"),
        ({'half_size': 'ab'}, "half_size must be two numbers"),
        ({'Vp': 5.0}, "percent"),
    ])
    def test_bad_patch(self, kwargs, match):
        args = {'center': (10.0, 5.0), 'half_size': (4.0, 2.0), 'Vp': 0.05}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            PorosityPatch(**args)

    def test_from_patches_draws_the_zones_on_the_background(self):
        m = _hot_spot()
        assert m.extent == (0.0, 50.0, 0.0, 20.0)
        assert (m.x.size, m.y.size) == (101, 41)
        assert m(25.0, 10.0) == 0.06 and m(2.0, 2.0) == 0.01
        # The area mean: the background plus the ellipse's share.
        expected = 0.01 + 0.05 * math.pi * 8.0 * 5.0 / (50.0 * 20.0)
        assert m.mean == pytest.approx(expected, rel=0.02)

    def test_later_patches_count_where_they_overlap(self):
        m = PorosityMap.from_patches(20.0, 10.0, 0.0, [
            PorosityPatch((10.0, 5.0), (6.0, 3.0), 0.04, 'rectangle'),
            PorosityPatch((10.0, 5.0), (2.0, 2.0), 0.08, 'rectangle')], resolution=1.0)
        assert m(10.0, 5.0) == 0.08 and m(5.0, 5.0) == 0.04 and m(1.0, 1.0) == 0.0
        assert (m.x.size, m.y.size) == (21, 11)

    def test_no_patches_is_a_uniform_map(self):
        m = PorosityMap.from_patches(50.0, 20.0, 0.02, [])
        assert m.mean == pytest.approx(0.02) and m.max == 0.02

    @pytest.mark.parametrize("args, kwargs, error, match", [
        ((0.0, 20.0, 0.01, []), {}, ValueError, "length, width"),
        ((50.0, 20.0, 1.5, []), {}, ValueError, "background"),
        ((50.0, 20.0, 0.01, []), {'resolution': 0.0}, ValueError, "resolution"),
        ((50.0, 20.0, 0.01, []), {'resolution': 'fine'}, ValueError, "resolution"),
        ((50.0, 20.0, 0.01, [((1, 1), (1, 1), 0.02)]), {}, TypeError, "PorosityPatch"),
    ])
    def test_bad_from_patches(self, args, kwargs, error, match):
        with pytest.raises(error, match=match):
            PorosityMap.from_patches(*args, **kwargs)


class TestCsv:
    def _write(self, tmp_path, text):
        path = tmp_path / 'map.csv'
        path.write_text(text, encoding='utf-8')
        return path

    def test_reads_a_grid_in_any_row_order(self, tmp_path):
        path = self._write(tmp_path, (
            "# C-scan of panel 7\n"
            "X, Y, zone, Vp\n"
            "50, 0, b, 0.03\n0, 0, a, 0.01\n25, 20, c, 0.02\n"
            "0, 20, a, 0.01\n25, 0, c, 0.02\n\n50, 20, b, 0.03\n"))
        m = PorosityMap.from_csv(path)
        assert list(m.x) == X and list(m.y) == Y
        assert m.Vp.tolist() == RAMP

    @pytest.mark.parametrize("text, match", [
        ("", "no data"),
        ("# only a comment\n", "no data"),
        ("x,y,Vp\n", "no data rows"),
        ("x,y,porosity\n0,0,0.01\n", "must name the columns x, y and Vp"),
        ("x,y,Vp\n0,0,abc\n", "needs a number"),
        ("x,y,Vp\n0,0\n", "needs a number"),
        ("x,y,Vp\n0,0,0.01\n1,0,0.01\n0,1,0.01\n", "complete rectilinear grid"),
        ("x,y,Vp\n0,0,0.01\n1,0,0.01\n0,1,0.01\n1,1,0.01\n1,1,0.02\n", "no repeat"),
        ("x,y,Vp\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n", "percent"),
        ("x,y,Vp\n0,0,0.01\n0,1,0.01\n", "x must be at least two"),
    ])
    def test_bad_files(self, tmp_path, text, match):
        with pytest.raises(ValueError, match=match):
            PorosityMap.from_csv(self._write(tmp_path, text))


class TestFieldWithMap:
    def test_from_map_keeps_the_absolute_values(self):
        m = PorosityMap(X, Y, RAMP)
        field = PorosityField.from_map(MAT, m)
        assert field.Vp == pytest.approx(0.02) and field.in_plane_map is m
        z = np.full(3, 0.5 * field.Lz)
        assert field.local_porosity(np.array(X), np.zeros(3), z) == pytest.approx(
            [0.01, 0.02, 0.03])

    def test_the_map_gives_the_pattern_and_Vp_the_level(self):
        m = PorosityMap(X, Y, RAMP)
        field = PorosityField(MAT, 0.04, in_plane_map=m)        # twice the map's mean
        z = np.zeros(3)
        assert field.local_porosity(np.array(X), np.zeros(3), z) == pytest.approx(
            [0.02, 0.04, 0.06])

    def test_the_profile_shape_is_kept_through_the_thickness(self):
        m = PorosityMap(X, Y, RAMP)
        mapped = PorosityField.from_map(MAT, m, distribution='clustered')
        plain = PorosityField(MAT, m.mean, distribution='clustered')
        z = np.linspace(0.0, mapped.Lz, 7)
        for x, factor in ((0.0, 0.5), (25.0, 1.0), (50.0, 1.5)):
            assert mapped.local_porosity(np.full(7, x), np.zeros(7), z) == pytest.approx(
                factor * plain.local_porosity(np.zeros(7), np.zeros(7), z))
        z_profile, Vp_profile = mapped.effective_porosity_profile(nz=7, x=50.0, y=0.0)
        assert Vp_profile == pytest.approx(1.5 * plain._distributed_porosity(z_profile))

    def test_local_values_are_clipped_at_one(self):
        m = PorosityMap([0.0, 50.0], Y, [[0.0, 0.0], [1.0, 1.0]])
        field = PorosityField(MAT, 0.9, in_plane_map=m)
        assert field.local_porosity(50.0, 0.0, 0.1) == 1.0

    def test_a_field_without_a_map_is_unchanged(self):
        field = PorosityField(MAT, 0.02, distribution='interface')
        assert field.in_plane_map is None
        x, y, z = np.array([1.0, 30.0]), np.array([2.0, 9.0]), np.array([0.1, 0.4])
        assert np.array_equal(field._distributed(x, y, z), field._distributed_porosity(z))
        assert 'in_plane_map' not in repr(field)

    def test_repr_names_the_map(self):
        field = PorosityField.from_map(MAT, PorosityMap(X, Y, RAMP))
        assert "in_plane_map=PorosityMap(3 x 2 points" in repr(field)

    def test_refusals(self):
        m = PorosityMap(X, Y, RAMP)
        with pytest.raises(TypeError, match="in_plane_map must be a PorosityMap"):
            PorosityField(MAT, 0.02, in_plane_map=RAMP)
        with pytest.raises(TypeError, match="porosity_map must be a PorosityMap"):
            PorosityField.from_map(MAT, RAMP)
        zero = PorosityMap(X, Y, np.zeros((3, 2)))
        with pytest.raises(ValueError, match="zero everywhere"):
            PorosityField(MAT, 0.02, in_plane_map=zero)
        # A void-free laminate with an empty map is consistent.
        field = PorosityField.from_map(MAT, zero)
        assert field.Vp == 0.0 and field.local_porosity(10.0, 5.0, 0.1) == 0.0
        assert PorosityField(MAT, 0.02, in_plane_map=m).in_plane_map is m


class TestMeshAndSolvers:
    def test_nodes_sample_the_map(self):
        m = PorosityMap(X, Y, RAMP)
        field = PorosityField.from_map(MAT, m)
        mesh = CompositeMesh(field, MAT, nx=10, ny=4, nz=MAT.n_plies)
        assert mesh.porosity == pytest.approx(m(mesh.nodes[:, 0], mesh.nodes[:, 1]))
        # A ramp is sampled exactly: the nodal mean is the field's Vp.
        assert mesh.porosity.mean() == pytest.approx(field.Vp)

    def test_the_interface_profile_is_scaled_then_mapped(self):
        m = PorosityMap(X, Y, RAMP)
        mapped = PorosityField.from_map(MAT, m, distribution='interface')
        plain = PorosityField(MAT, m.mean, distribution='interface')
        kwargs = dict(nx=10, ny=4, nz=MAT.n_plies)
        mesh, ref = CompositeMesh(mapped, MAT, **kwargs), CompositeMesh(plain, MAT, **kwargs)
        assert mesh.porosity_renormalization == ref.porosity_renormalization != 1.0
        factor = m(mesh.nodes[:, 0], mesh.nodes[:, 1]) / m.mean
        assert mesh.porosity == pytest.approx(factor * ref.porosity)

    def test_a_mesh_beyond_the_map_warns_and_uses_the_edge_values(self, caplog):
        small = PorosityMap([0.0, 10.0], [0.0, 5.0], [[0.01, 0.01], [0.03, 0.03]])
        field = PorosityField.from_map(MAT, small)
        with caplog.at_level(logging.WARNING, logger='porosity_fe_analysis'):
            mesh = CompositeMesh(field, MAT, nx=10, ny=4, nz=MAT.n_plies)
        assert "extends beyond the porosity map" in caplog.text
        far = mesh.nodes[:, 0] >= 10.0
        assert mesh.porosity[far] == pytest.approx(0.03)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='porosity_fe_analysis'):
            CompositeMesh(PorosityField.from_map(MAT, PorosityMap(X, Y, RAMP)), MAT,
                          nx=10, ny=4, nz=MAT.n_plies)
        assert "porosity map" not in caplog.text

    def test_the_empirical_failure_load_sees_only_the_mean(self):
        m = _hot_spot()
        mapped = EmpiricalSolver(
            CompositeMesh(PorosityField.from_map(MAT, m), MAT, nx=20, ny=8, nz=MAT.n_plies),
            MAT)
        uniform = build_empirical_pipeline(MAT, m.mean, mesh_res=(20, 8, MAT.n_plies))[2]
        for mode in ('compression', 'ilss'):
            assert mapped.get_failure_load(mode).knockdown == \
                uniform.get_failure_load(mode).knockdown
        # The per-node field does show the hot spot.
        with pytest.warns(UserWarning, match="local peak Vp = 0.06"):
            mapped.apply_loading('compression')
        alpha = mapped.JUDD_WRIGHT_ALPHA['compression']
        assert mapped.nodal_knockdown.min() == pytest.approx(math.exp(-alpha * 0.06))
        assert mapped.nodal_knockdown.max() == pytest.approx(math.exp(-alpha * 0.01))

    def test_the_fe_path_resolves_a_hot_spot(self):
        """At one mean void content, a hot spot fails earlier than a uniform
        field: the FE failure index follows the local porosity."""
        m = _hot_spot()
        kwargs = dict(nx=20, ny=8, nz=MAT.n_plies)
        mapped_field = PorosityField.from_map(MAT, m)
        uniform_field = PorosityField(MAT, m.mean)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mapped = FESolver(CompositeMesh(mapped_field, MAT, **kwargs), MAT,
                              mapped_field).solve('compression')
            uniform = FESolver(CompositeMesh(uniform_field, MAT, **kwargs), MAT,
                               uniform_field).solve('compression')
        assert mapped.interior_first_ply_failure_load_factor < \
            uniform.interior_first_ply_failure_load_factor
        assert mapped.interior_max_failure_index > uniform.interior_max_failure_index
        assert mapped.knockdown == pytest.approx(uniform.knockdown, abs=2e-3)

    def test_a_map_goes_through_the_pipeline_factory(self):
        m = PorosityMap(X, Y, RAMP)
        field, mesh, solver = build_empirical_pipeline(
            MAT, m.mean, mesh_res=(10, 4, MAT.n_plies),
            porosity_config={'in_plane_map': m})
        assert field.in_plane_map is m
        assert mesh.porosity.max() == pytest.approx(0.03)
        assert solver.get_failure_load('shear').knockdown == pytest.approx(
            math.exp(-solver.JUDD_WRIGHT_ALPHA['shear'] * 0.02))
