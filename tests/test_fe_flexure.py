#!/usr/bin/env python3
"""Three- and four-point flexure on roller supports (IMPROVEMENT_PLAN 3.6, C1).

Covers the geometry (:class:`FlexureGeometry`,
:meth:`CompositeMesh.for_flexure`), the boundary-condition helpers
(:meth:`BoundaryHandler.roller_supports`, :meth:`~BoundaryHandler.line_load`,
:meth:`~BoundaryHandler.flexure_bcs`) and the ``'flexure_3pt'`` /
``'flexure_4pt'`` solves against classical lamination theory and Timoshenko
beam theory. Most solves use an 8-ply T800 laminate at the D7264
width-to-thickness ratio, so a whole model solves in about a second; the
D7264 default of the 24-ply preset is checked once.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    BoundaryHandler,
    CompositeMesh,
    FESolver,
    FlexureGeometry,
    PorosityField,
    check_mesh_quality,
    compute_degraded_clt_flexural_modulus,
)
from porosity_fe.fe import solver as solver_mod
from porosity_fe.fe.flexure import section_deflection, transverse_shear_stiffness

T800 = MATERIALS['T800_epoxy']
# 8 plies, h = 1.464 mm; width 3 h as in the D7264 standard specimen.
THIN = dataclasses.replace(T800, n_plies=8)
THIN_KW = dict(width=3.0 * THIN.total_thickness)
LOAD = -100.0
MODES = ('flexure_3pt', 'flexure_4pt')


def _clt_flexural_modulus(mesh: CompositeMesh, Vp: float) -> float:
    return compute_degraded_clt_flexural_modulus(
        mesh.material, list(mesh.element_layup), Vp)['Ef_x']


def _mid_span_deflection(mesh, results):
    """Mid-thickness mid-span deflection relative to the supports (mm)."""
    g = mesh.flexure_geometry
    u = results.displacement.ravel()
    return section_deflection(mesh, u, g.midspan) - 0.5 * (
        section_deflection(mesh, u, g.support_x[0])
        + section_deflection(mesh, u, g.support_x[1]))


@pytest.fixture(scope='module')
def thin():
    """``{(layup, Vp): (mesh, {mode: FieldResults})}`` on 8-ply meshes."""
    out = {}
    for layup in ('UD', 'QI'):
        for Vp in (0.0, 0.05):
            pf = PorosityField(THIN, Vp, distribution='uniform')
            mesh = CompositeMesh.for_flexure(pf, THIN, ply_angles=layup, **THIN_KW)
            solver = FESolver(mesh, THIN, pf, ply_angles=layup)
            out[layup, Vp] = (mesh, {m: solver.solve(m, applied_load=LOAD)
                                     for m in MODES})
    return out


class TestFlexureGeometry:

    def test_derived_positions(self):
        g = FlexureGeometry(support_x=(10.0, 110.0), load_span=50.0)
        assert g.span == 100.0
        assert g.midspan == 60.0
        assert g.load_x('3pt') == (60.0,)
        assert g.load_x('4pt') == (35.0, 85.0)
        assert g.shear_span('3pt') == 50.0
        assert g.shear_span('4pt') == 25.0
        assert g.load_width == 0.0

    @pytest.mark.parametrize('kwargs', [
        dict(support_x=(5.0, 5.0), load_span=1.0),
        dict(support_x=(10.0, 0.0), load_span=1.0),
        dict(support_x=(0.0, 10.0), load_span=10.0),
        dict(support_x=(0.0, 10.0), load_span=0.0),
        dict(support_x=(0.0, 10.0), load_span=5.0, load_width=5.0),
        dict(support_x=(0.0, 10.0), load_span=8.0, load_width=2.5),
        dict(support_x=(0.0, 10.0), load_span=5.0, load_width=-0.1),
        dict(support_x=(0.0, float('nan')), load_span=5.0),
        dict(support_x=(0.0,), load_span=5.0),
    ])
    def test_rejects_inconsistent_geometry(self, kwargs):
        with pytest.raises(ValueError):
            FlexureGeometry(**kwargs)

    def test_unknown_procedure_raises(self):
        g = FlexureGeometry(support_x=(0.0, 10.0), load_span=5.0)
        with pytest.raises(ValueError, match="'3pt' or '4pt'"):
            g.load_x('5pt')


class TestForFlexure:

    def test_d7264_default_for_24_ply_preset(self):
        pf = PorosityField(T800, 0.0)
        mesh = CompositeMesh.for_flexure(pf, T800)
        g = mesh.flexure_geometry
        h = T800.total_thickness
        assert (mesh.nx, mesh.ny, mesh.nz) == (48, 4, 24)
        assert g.span / h == pytest.approx(32.0, rel=1e-12)
        assert mesh.L_y == 13.0
        assert mesh.L_z == pytest.approx(h)
        # At least 10 % overhang on each side, centred.
        assert g.support_x[0] >= 0.1 * g.span - 1e-9
        assert g.support_x[0] == pytest.approx(mesh.L_x - g.support_x[1])
        assert g.load_span == pytest.approx(0.5 * g.span)
        # Every ply resolved, aspect ratio within the quality threshold.
        assert not mesh.layup_discrepancies()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            quality = check_mesh_quality(mesh)
        assert quality['max_aspect_ratio'] <= 20.0

    def test_test_lines_fall_on_node_lines(self):
        pf = PorosityField(THIN, 0.0)
        mesh = CompositeMesh.for_flexure(pf, THIN, **THIN_KW)
        g = mesh.flexure_geometry
        x_nodes = np.unique(mesh.nodes[:, 0])
        for x in (*g.support_x, g.midspan, *g.load_x('4pt')):
            assert np.abs(x_nodes - x).min() < 1e-9 * mesh.L_x

    def test_custom_span_and_load_span_align(self):
        pf = PorosityField(THIN, 0.0)
        mesh = CompositeMesh.for_flexure(
            pf, THIN, span=60.0, load_span=20.0, overhang=0.0, width=5.0)
        g = mesh.flexure_geometry
        assert g.span == pytest.approx(60.0)
        assert g.support_x[0] == 0.0 and g.support_x[1] == pytest.approx(mesh.L_x)
        x_nodes = np.unique(mesh.nodes[:, 0])
        for x in g.load_x('4pt'):
            assert np.abs(x_nodes - x).min() < 1e-9 * mesh.L_x

    def test_unaligned_load_span_is_logged(self, caplog):
        pf = PorosityField(THIN, 0.0)
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            mesh = CompositeMesh.for_flexure(pf, THIN, load_span=np.pi * 10.0)
        assert 'between node lines' in caplog.text
        # The loads are still applied in full.
        _, F = BoundaryHandler(mesh).flexure_bcs(LOAD, procedure='4pt')
        assert F.sum() == pytest.approx(LOAD)

    def test_span_to_thickness_and_resolution_options(self):
        pf = PorosityField(THIN, 0.0)
        mesh = CompositeMesh.for_flexure(pf, THIN, span_to_thickness=16.0,
                                         nz=4, ny=2, element_length=1.0)
        g = mesh.flexure_geometry
        assert g.span == pytest.approx(16.0 * THIN.total_thickness)
        assert (mesh.ny, mesh.nz) == (2, 4)
        assert mesh.L_x / mesh.nx <= 1.0 + 1e-12

    @pytest.mark.parametrize('kwargs', [
        dict(span_to_thickness=0.0),
        dict(span=-1.0),
        dict(width=float('inf')),
        dict(overhang=-1.0),
        dict(load_span=0.0),
        dict(element_length=0.0),
        dict(nz=0),
        dict(load_width=100.0),
    ])
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            CompositeMesh.for_flexure(PorosityField(THIN, 0.0), THIN, **kwargs)


class TestFlexureBoundaryConditions:

    @pytest.fixture(scope='class')
    @classmethod
    def mesh(cls):
        pf = PorosityField(THIN, 0.0)
        return CompositeMesh.for_flexure(pf, THIN, **THIN_KW)

    def test_roller_supports_hold_only_uz_plus_three_in_plane_dofs(self, mesh):
        g = mesh.flexure_geometry
        handler = BoundaryHandler(mesh)
        constrained = handler.roller_supports(g.support_x)
        lines = [handler.face_line_nodes('z_min', x) for x in g.support_x]
        assert all(line.size == mesh.ny + 1 for line in lines)
        uz = {3 * int(n) + 2 for line in lines for n in line}
        in_plane = set(constrained) - uz
        assert uz <= set(constrained)
        assert len(in_plane) == 3
        assert sorted(d % 3 for d in in_plane) == [0, 1, 1]
        assert all(v == 0.0 for v in constrained.values())

    def test_roller_supports_need_two_node_lines(self, mesh):
        handler = BoundaryHandler(mesh)
        x0 = mesh.flexure_geometry.support_x[0]
        with pytest.raises(ValueError, match='at least two'):
            handler.roller_supports([x0, x0])
        dx = mesh.L_x / mesh.nx
        with pytest.raises(ValueError, match='node line'):
            handler.roller_supports([x0, x0 + 0.5 * dx])

    def test_line_load_uses_consistent_weights(self, mesh):
        handler = BoundaryHandler(mesh)
        x = mesh.flexure_geometry.midspan
        F = handler.line_load(x, LOAD)
        assert F.sum() == pytest.approx(LOAD)
        Fz = F.reshape(-1, 3)
        assert not Fz[:, :2].any()
        line = handler.face_line_nodes('z_max', x)
        loaded = np.flatnonzero(Fz[:, 2])
        assert set(loaded) == set(line.tolist())
        # Trapezoidal along y: the two edge nodes carry half an interior share.
        shares = Fz[line, 2] / LOAD
        np.testing.assert_allclose(shares, np.r_[0.5, np.ones(mesh.ny - 1), 0.5] / mesh.ny,
                                   rtol=1e-12)

    def test_line_load_between_node_lines_is_shared_linearly(self, mesh):
        handler = BoundaryHandler(mesh)
        dx = mesh.L_x / mesh.nx
        x0 = mesh.flexure_geometry.midspan
        F = handler.line_load(x0 + 0.25 * dx, LOAD).reshape(-1, 3)[:, 2]
        on = F[handler.face_line_nodes('z_max', x0)].sum()
        right = F[handler.face_line_nodes('z_max', x0 + dx)].sum()
        assert on == pytest.approx(0.75 * LOAD)
        assert right == pytest.approx(0.25 * LOAD)

    def test_band_load_is_a_uniform_pressure(self, mesh):
        handler = BoundaryHandler(mesh)
        dx = mesh.L_x / mesh.nx
        x0 = mesh.flexure_geometry.midspan
        F = handler.line_load(x0, LOAD, width=2.0 * dx).reshape(-1, 3)[:, 2]
        per_line = [F[handler.face_line_nodes('z_max', x0 + k * dx)].sum()
                    for k in (-1, 0, 1)]
        np.testing.assert_allclose(per_line, np.array([0.25, 0.5, 0.25]) * LOAD,
                                   rtol=1e-12)
        assert F.sum() == pytest.approx(LOAD)

    @pytest.mark.parametrize('x, width', [(-1.0, 0.0), (1e6, 0.0), (0.1, 1.0)])
    def test_load_outside_the_top_face_raises(self, mesh, x, width):
        with pytest.raises(ValueError, match='outside'):
            BoundaryHandler(mesh).line_load(x, LOAD, width=width)

    @pytest.mark.parametrize('kwargs', [dict(total_load=float('nan')),
                                        dict(total_load=1.0, width=-1.0)])
    def test_line_load_rejects_bad_values(self, mesh, kwargs):
        with pytest.raises(ValueError):
            BoundaryHandler(mesh).line_load(mesh.L_x / 2, **kwargs)

    @pytest.mark.parametrize('procedure, n_lines', [('3pt', 1), ('4pt', 2)])
    def test_flexure_bcs_loads(self, mesh, procedure, n_lines):
        handler = BoundaryHandler(mesh)
        g = mesh.flexure_geometry
        constrained, F = handler.flexure_bcs(LOAD, procedure=procedure)
        assert constrained == handler.roller_supports(g.support_x)
        Fz = F.reshape(-1, 3)[:, 2]
        assert F.sum() == pytest.approx(LOAD)
        for x in g.load_x(procedure):
            assert Fz[handler.face_line_nodes('z_max', x)].sum() == \
                pytest.approx(LOAD / n_lines)

    def test_default_geometry_uses_the_end_edges(self):
        pf = PorosityField(THIN, 0.0)
        mesh = CompositeMesh(pf, THIN, nx=8, ny=2, nz=4)
        assert mesh.flexure_geometry is None
        g = BoundaryHandler(mesh).flexure_geometry()
        assert g.support_x == (0.0, mesh.L_x)
        assert g.load_x('4pt') == pytest.approx((mesh.L_x / 4, 3 * mesh.L_x / 4))


class TestFlexureSolve:

    @pytest.mark.parametrize('Vp', [0.0, 0.05])
    @pytest.mark.parametrize('mode', MODES)
    def test_ud_flexural_modulus_matches_clt(self, thin, mode, Vp):
        mesh, results = thin['UD', Vp]
        ratio = results[mode].effective_modulus / _clt_flexural_modulus(mesh, Vp)
        assert ratio == pytest.approx(1.0, abs=0.005)

    @pytest.mark.parametrize('Vp', [0.0, 0.05])
    @pytest.mark.parametrize('mode', MODES)
    def test_qi_flexural_modulus_matches_clt(self, thin, mode, Vp):
        mesh, results = thin['QI', Vp]
        ratio = results[mode].effective_modulus / _clt_flexural_modulus(mesh, Vp)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_d7264_default_mesh_of_the_24_ply_qi_preset(self):
        # On the 13 mm D7264 specimen the 24-ply QI laminate reads about
        # 1.1 % below CLT: the free-edge effect of the +-45 plies, which
        # shrinks as the width grows (docs/theory/fe.md).
        pf = PorosityField(T800, 0.0)
        mesh = CompositeMesh.for_flexure(pf, T800, ply_angles='QI')
        solver = FESolver(mesh, T800, pf, ply_angles='QI')
        clt = _clt_flexural_modulus(mesh, 0.0)
        for mode in MODES:
            ratio = solver.solve(mode).effective_modulus / clt
            assert 0.98 < ratio < 0.995, (mode, ratio)

    @pytest.mark.parametrize('layup', ['UD', 'QI'])
    def test_knockdown_tracks_the_clt_bending_stiffness_ratio(self, thin, layup):
        mesh0, r0 = thin[layup, 0.0]
        mesh, r = thin[layup, 0.05]
        d_ratio = _clt_flexural_modulus(mesh, 0.05) / _clt_flexural_modulus(mesh0, 0.0)
        for mode in MODES:
            # The compliance knockdown also carries the transverse shear
            # deflection (a few percent of the total at 32:1), which degrades
            # faster than the bending stiffness, so it sits slightly lower.
            assert r[mode].knockdown < d_ratio
            assert r[mode].knockdown == pytest.approx(d_ratio, abs=0.005)
            # The shear-corrected modulus ratio follows D closely.
            ef_ratio = r[mode].effective_modulus / r0[mode].effective_modulus
            assert ef_ratio == pytest.approx(d_ratio, abs=0.002)
            assert r0[mode].knockdown == 1.0

    def test_knockdown_is_the_beam_stiffness_ratio(self, thin):
        mesh, r = thin['QI', 0.05]
        _, r0 = thin['QI', 0.0]
        for mode in MODES:
            ratio = _mid_span_deflection(mesh, r0[mode]) / _mid_span_deflection(mesh, r[mode])
            assert r[mode].knockdown == pytest.approx(ratio, rel=2e-3)

    @pytest.mark.parametrize('mode', MODES)
    def test_rollers_exert_no_horizontal_reaction(self, thin, mode):
        _, results = thin['QI', 0.05]
        R = results[mode].reaction_forces
        assert np.abs(R[:, :2]).max() < 1e-8 * abs(LOAD)
        assert R[:, 2].sum() == pytest.approx(-LOAD, rel=1e-8)

    def test_four_point_inner_span_is_in_constant_moment(self, thin):
        mesh, results = thin['UD', 0.0]
        r = results['flexure_4pt']
        g = mesh.flexure_geometry
        h, b = mesh.L_z, mesh.L_y
        cx = mesh.nodes[mesh.elements, 0].mean(axis=1)
        cz = mesh.nodes[mesh.elements, 2].mean(axis=1)
        x_load = g.load_x('4pt')
        dy, dz = b / mesh.ny, h / mesh.nz
        columns = np.unique(np.round(cx, 9))
        tau = r.stress_global[:, :, 4].mean(axis=1)
        # Section shear force V(x) = integral of tau_xz over the cross-section.
        shear = {c: float(tau[np.abs(cx - c) < 1e-6].sum() * dy * dz) for c in columns}
        inner = [c for c in columns if x_load[0] + 2 * h < c < x_load[1] - 2 * h]
        outer = [c for c in columns if g.support_x[0] + 2 * h < c < x_load[0] - 2 * h]
        assert inner and outer
        for c in inner:
            assert abs(shear[c]) < 1e-6 * abs(LOAD) / 2
        for c in outer:
            assert abs(shear[c]) == pytest.approx(abs(LOAD) / 2, rel=1e-6)
        # Bending stress uniform along x in the inner span (bottom layer).
        sx = r.stress_global[:, :, 0].mean(axis=1)
        bottom = cz < dz
        per_column = np.array([sx[bottom & (np.abs(cx - c) < 1e-6)].mean() for c in inner])
        assert np.ptp(per_column) < 1e-3 * np.abs(per_column).mean()
        # ... and equal to beam theory, M c / I with M = (P / 2) a.
        a = g.shear_span('4pt')
        sigma_beam = 0.5 * abs(LOAD) * a * 0.5 * h / (b * h ** 3 / 12.0)
        # Element mean of the outer layer: the stress at its mid-depth.
        assert per_column.mean() == pytest.approx(sigma_beam * (1 - 1 / mesh.nz), rel=0.01)

    @pytest.mark.parametrize('nz', [8, 16])
    def test_three_point_deflection_matches_timoshenko(self, nz):
        # Span 16 h, where transverse shear is 13 % of the deflection.
        pf = PorosityField(THIN, 0.0)
        mesh = CompositeMesh.for_flexure(pf, THIN, ply_angles='UD',
                                         span_to_thickness=16.0, nz=nz, **THIN_KW)
        r = FESolver(mesh, THIN, pf, ply_angles='UD').solve(
            'flexure_3pt', applied_load=LOAD)
        L, b, h = mesh.flexure_geometry.span, mesh.L_y, mesh.L_z
        bending = abs(LOAD) * L ** 3 / (48.0 * THIN.E11 * b * h ** 3 / 12.0)
        shear = abs(LOAD) * L / (4.0 * (5.0 / 6.0) * THIN.G13 * b * h)
        assert shear / (bending + shear) > 0.1
        ratio = -_mid_span_deflection(mesh, r) / (bending + shear)
        assert ratio == pytest.approx(1.0, abs=0.015)

    def test_hex8_is_stiffer_in_flexure(self):
        # On a coarse mesh (element length 2 h) the standard brick locks;
        # the incompatible-mode element does not.
        pf = PorosityField(THIN, 0.0)
        mesh = CompositeMesh.for_flexure(pf, THIN, ply_angles='UD',
                                         element_length=2.0 * THIN.total_thickness,
                                         **THIN_KW)
        clt = _clt_flexural_modulus(mesh, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # aspect ratio above 20
            ratio = {form: FESolver(mesh, THIN, pf, ply_angles='UD', formulation=form)
                     .solve('flexure_3pt').effective_modulus / clt
                     for form in ('hex8', 'hex8i')}
        assert ratio['hex8i'] == pytest.approx(1.0, abs=0.01), ratio
        assert ratio['hex8'] > 1.08, ratio

    def test_independent_of_load_magnitude_and_sign(self, thin):
        mesh, results = thin['QI', 0.05]
        pf = mesh.porosity_field
        solver = FESolver(mesh, THIN, pf, ply_angles='QI')
        for mode in MODES:
            r = solver.solve(mode, applied_load=250.0)
            assert r.knockdown == pytest.approx(results[mode].knockdown, rel=1e-9)
            assert r.effective_modulus == pytest.approx(
                results[mode].effective_modulus, rel=1e-9)

    def test_zero_load(self, thin):
        mesh, _ = thin['UD', 0.05]
        solver = FESolver(mesh, THIN, mesh.porosity_field, ply_angles='UD')
        r = solver.solve('flexure_3pt', applied_load=0.0)
        assert not r.displacement.any()
        assert r.knockdown == 1.0
        assert r.effective_modulus is None

    def test_pristine_reference_is_keyed_on_the_test_geometry(self):
        # Same nodes, different load span: the four-point pristine reference
        # must not be shared.
        pf = PorosityField(THIN, 0.05)
        mesh = CompositeMesh.for_flexure(pf, THIN, ply_angles='QI', **THIN_KW)
        g = mesh.flexure_geometry
        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        kd_half = FESolver(mesh, THIN, pf).solve('flexure_4pt').knockdown
        mesh.flexure_geometry = dataclasses.replace(g, load_span=0.25 * g.span)
        kd_quarter = FESolver(mesh, THIN, pf).solve('flexure_4pt').knockdown
        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        fresh = FESolver(mesh, THIN, pf).solve('flexure_4pt').knockdown
        assert kd_quarter == pytest.approx(fresh, rel=1e-12)
        assert kd_quarter != pytest.approx(kd_half, rel=1e-9)

    def test_works_on_the_default_coupon(self):
        # Without for_flexure the supports sit at the two bottom end edges.
        pf = PorosityField(THIN, 0.03)
        mesh = CompositeMesh(pf, THIN, nx=16, ny=8, nz=8, ply_angles="UD")
        r = FESolver(mesh, THIN, pf, ply_angles='UD').solve('flexure_3pt')
        assert 0.0 < r.knockdown < 1.0
        assert r.effective_modulus == pytest.approx(
            _clt_flexural_modulus(mesh, 0.03), rel=0.02)
        assert np.abs(r.reaction_forces[:, 0]).max() < 1e-8 * 10.0


def test_shear_stiffness_of_one_orthotropic_layer_is_five_sixths_gh():
    C = T800.get_stiffness_matrix()
    for n in (1, 4):
        z = np.linspace(0.0, 2.0, n + 1)
        Ks = transverse_shear_stiffness(np.repeat(C[None], n, axis=0), z)
        assert Ks == pytest.approx(5.0 / 6.0 * T800.G13 * 2.0, rel=1e-12)


def test_unknown_loading_lists_the_flexure_modes():
    pf = PorosityField(THIN, 0.0)
    mesh = CompositeMesh(pf, THIN, nx=4, ny=2, nz=2)
    with pytest.raises(ValueError, match='flexure_3pt/flexure_4pt'):
        FESolver(mesh, THIN, pf).solve('flexure')
