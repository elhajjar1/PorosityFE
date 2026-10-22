"""ASTM D2344 short-beam shear: the ``'ilss'`` FE loading mode.

IMPROVEMENT_PLAN 3.6, C2: ``'ilss'`` runs the short-beam-strength test on
the specimen of :meth:`CompositeMesh.for_ilss` (support span 4 h, length
6 h, width 2 h, roller supports, the loading nose as a uniform pressure over
its Hertz contact band) instead of a three-point bend of the coupon on
pinned end edges.

Most checks use an 8-ply T800 laminate (h = 1.464 mm), one element layer
per ply, so the specimen meshes stay small. Numbers quoted in comments were
measured on these meshes.
"""

import dataclasses
import math

import numpy as np
import pytest

from porosity_fe import (MATERIALS, BoundaryHandler, CompositeMesh, FESolver,
                         FlexureGeometry, PorosityField, ShortBeamShear,
                         short_beam_shear)
from porosity_fe.fe import solver as solver_mod
from porosity_fe.fe.short_beam import interior_element_mask

T800 = MATERIALS['T800_epoxy']
MAT8 = dataclasses.replace(T800, n_plies=8)
H8 = MAT8.total_thickness
LOAD = -10.0


def _hertz_width(material, radius=3.0):
    """Hertz contact width of the D2344 nose at the pristine short-beam strength."""
    q = 4.0 / 3.0 * material.tau_ilss * material.total_thickness
    e_star = material.E33 / (1.0 - material.nu23 ** 2)
    return 4.0 * math.sqrt(q * radius / (math.pi * e_star))


def _specimen(layup='UD', vp=0.0, *, nz=8, element_length=H8 / 4, material=MAT8,
              **kw):
    pf = PorosityField(material, vp, **({'distribution': 'uniform'} | kw.pop('pf', {})))
    mesh = CompositeMesh.for_ilss(pf, material, nz=nz, element_length=element_length,
                                  ply_angles=layup, **kw)
    return mesh, FESolver(mesh, material, pf, ply_angles=layup)


def _parabola_layer_mean(nz):
    """Mean of a parabolic shear profile over the two mid-plane element
    layers, relative to its peak: what ``midplane_shear`` reads for an exact
    beam-theory profile."""
    return 1.0 - 4.0 / 3.0 / nz ** 2


@pytest.fixture(scope='module')
def ud():
    """Pristine UD specimen (24 x 8 x 8) and its solve."""
    mesh, solver = _specimen('UD')
    return mesh, solver.solve('ilss', applied_load=LOAD)


# ---------------------------------------------------------------------------
# The specimen mesh
# ---------------------------------------------------------------------------

class TestForIlss:

    def test_d2344_specimen_of_the_24_ply_preset(self):
        pf = PorosityField(T800, 0.03)
        mesh = CompositeMesh.for_ilss(pf, T800)
        h = T800.total_thickness
        assert (mesh.nx, mesh.ny, mesh.nz) == (24, 8, 24)
        assert mesh.L_x == pytest.approx(6 * h, rel=1e-12)
        assert mesh.L_y == pytest.approx(2 * h, rel=1e-12)
        assert mesh.L_z == pytest.approx(h, rel=1e-12)
        g = mesh.ilss_geometry
        assert g is mesh.flexure_geometry
        np.testing.assert_allclose(g.support_x, (h, 5 * h), rtol=1e-12)
        assert g.span == pytest.approx(4 * h, rel=1e-12)
        assert g.midspan == pytest.approx(3 * h, rel=1e-12)
        # The default band: the Hertz contact width of the 6 mm nose at the
        # pristine short-beam strength, 0.77 mm (0.18 h).
        assert g.load_width == pytest.approx(_hertz_width(T800), rel=1e-12)
        assert 0.76 < g.load_width < 0.78

    def test_supports_and_midspan_are_node_lines(self):
        pf = PorosityField(MAT8, 0.0)
        for kw in ({}, {'nx': 36}, {'element_length': H8 / 2},
                   {'span_to_thickness': 5.0, 'length_to_thickness': 10.0,
                    'width_to_thickness': 5.0}):
            mesh = CompositeMesh.for_ilss(pf, MAT8, **kw)
            grid = np.unique(mesh.nodes[:, 0])
            g = mesh.ilss_geometry
            for x in (*g.support_x, g.midspan):
                assert np.abs(grid - x).min() < 1e-9, (kw, x)

    def test_options(self):
        pf = PorosityField(MAT8, 0.0)
        iso = CompositeMesh.for_ilss(pf, MAT8, span_to_thickness=5.0,
                                     length_to_thickness=10.0,
                                     width_to_thickness=5.0, nose_radius=5.0)
        assert iso.ilss_geometry.span == pytest.approx(5 * H8, rel=1e-12)
        assert iso.L_x == pytest.approx(10 * H8, rel=1e-12)
        assert iso.L_y == pytest.approx(5 * H8, rel=1e-12)
        assert iso.ilss_geometry.load_width == pytest.approx(
            _hertz_width(MAT8, 5.0), rel=1e-12)
        assert CompositeMesh.for_ilss(pf, MAT8, load_width=0.0).ilss_geometry \
            .load_width == 0.0
        fine = CompositeMesh.for_ilss(pf, MAT8, nx=36)
        assert (fine.nx, fine.ny, fine.nz) == (36, 12, 8)
        coarse = CompositeMesh.for_ilss(pf, MAT8, element_length=H8 / 2, nz=4)
        assert (coarse.nx, coarse.ny, coarse.nz) == (12, 4, 4)

    def test_contact_width_scales_as_hertz(self):
        pf = PorosityField(MAT8, 0.0)
        w3 = CompositeMesh.for_ilss(pf, MAT8).ilss_geometry.load_width
        w6 = CompositeMesh.for_ilss(pf, MAT8, nose_radius=6.0).ilss_geometry.load_width
        assert w6 / w3 == pytest.approx(math.sqrt(2.0), rel=1e-12)
        strong = dataclasses.replace(MAT8, tau_ilss=4 * MAT8.tau_ilss)
        w_strong = CompositeMesh.for_ilss(PorosityField(strong, 0.0), strong) \
            .ilss_geometry.load_width
        assert w_strong / w3 == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.parametrize('kwargs, match', [
        ({'span_to_thickness': 0.0}, 'span_to_thickness'),
        ({'span_to_thickness': float('nan')}, 'span_to_thickness'),
        ({'span_to_thickness': True}, 'span_to_thickness'),
        ({'width_to_thickness': -2.0}, 'width_to_thickness'),
        ({'length_to_thickness': 3.0}, 'at least as long as the span'),
        ({'load_width': -0.1}, 'load_width'),
        ({'load_width': float('inf')}, 'load_width'),
        ({'load_width': 2.5 * H8}, 'load_width'),
        ({'nose_radius': 0.0}, 'nose_radius'),
        ({'nx': 25}, 'multiple of 6'),
        ({'nx': 0}, 'positive integer'),
        ({'element_length': 0.0}, 'element_length'),
    ])
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CompositeMesh.for_ilss(PorosityField(MAT8, 0.0), MAT8, **kwargs)


# ---------------------------------------------------------------------------
# Supports, symmetry, beam theory
# ---------------------------------------------------------------------------

class TestShortBeamMechanics:

    def test_rollers_exert_no_horizontal_reaction(self, ud):
        """The old pinned end edges made a tied arch (horizontal reaction
        about 4 P); the rollers carry none."""
        mesh, r = ud
        R = r.reaction_forces
        assert np.abs(R[:, 0]).max() < 1e-9 * abs(LOAD)
        assert np.abs(R[:, 1]).max() < 1e-9 * abs(LOAD)
        np.testing.assert_allclose(R.sum(axis=0), [0.0, 0.0, -LOAD], atol=1e-9)
        # Each support carries half the load.
        handler = BoundaryHandler(mesh)
        for x in mesh.ilss_geometry.support_x:
            line = handler.face_line_nodes('z_min', x)
            assert R[line, 2].sum() == pytest.approx(-LOAD / 2, rel=1e-9)

    def test_field_is_symmetric_about_midspan(self, ud):
        mesh, r = ud
        nx1, ny1, nz1 = mesh.nx + 1, mesh.ny + 1, mesh.nz + 1
        uz = r.displacement[:, 2].reshape(nz1, ny1, nx1)
        np.testing.assert_allclose(uz, uz[:, :, ::-1], rtol=0,
                                   atol=1e-9 * np.abs(uz).max())
        tau = np.abs(r.stress_global[..., 4].mean(axis=1)).reshape(
            mesh.nz, mesh.ny, mesh.nx)
        np.testing.assert_allclose(tau, tau[:, :, ::-1], rtol=0,
                                   atol=1e-9 * tau.max())

    def test_quarter_span_shear_against_the_d2344_formula(self, ud):
        """The D2344 formula 0.75 P / (b h) is the peak of the parabolic
        beam-theory profile. At span/h = 4 the FE profile half-way between a
        support and the nose is flatter: the mid-plane shear, normalized by
        what the element layers read of an exact parabola, is 0.916 here and
        0.910-0.918 on every mesh tried (element length h/4 to h/8, one or
        two layers per ply, 8 and 24 plies). The band +-0.025 is that spread
        plus margin; a factor-of-2 reference error (as in the old test) or
        a locking element would fall far outside it."""
        mesh, r = ud
        sb = short_beam_shear(r, mesh)
        ratio = sb.midplane_shear / (sb.apparent_shear * _parabola_layer_mean(mesh.nz))
        assert ratio == pytest.approx(0.915, abs=0.025)

    def test_quarter_span_shear_is_mesh_converged(self, ud):
        # h/4 to h/6 elements moves it by 0.17 % (h/8: 0.23 % from h/4).
        mesh, r = ud
        fine_mesh, fine_solver = _specimen('UD', element_length=H8 / 6)
        assert (fine_mesh.nx, fine_mesh.ny) == (36, 12)
        coarse = short_beam_shear(r, mesh)
        fine = short_beam_shear(fine_solver.solve('ilss', applied_load=LOAD), fine_mesh)
        assert fine.midplane_shear == pytest.approx(coarse.midplane_shear, rel=5e-3)

    def test_long_span_recovers_beam_theory(self):
        """The same measurement on a span of 16 thicknesses meets the
        parabola (0.9995 here): the shortfall at 4:1 is the short beam, not
        the element or the measurement."""
        mesh, solver = _specimen('UD', span_to_thickness=16.0,
                                 length_to_thickness=18.0)
        sb = short_beam_shear(solver.solve('ilss', applied_load=LOAD), mesh)
        ratio = sb.midplane_shear / (sb.apparent_shear * _parabola_layer_mean(mesh.nz))
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_section_shear_force_is_half_the_load(self, ud):
        mesh, r = ud
        c = mesh.nodes[mesh.elements].mean(axis=1)
        g = mesh.ilss_geometry
        columns = np.unique(np.round(c[:, 0], 9))
        x = columns[np.argmin(np.abs(columns - (g.support_x[0] + 0.25 * g.span)))]
        in_column = np.abs(c[:, 0] - x) < 1e-9
        dA = (mesh.L_y / mesh.ny) * (mesh.L_z / mesh.nz)
        V = r.stress_global[in_column, :, 4].mean(axis=1).sum() * dA
        assert abs(V) == pytest.approx(abs(LOAD) / 2, rel=2e-3)

    def test_no_effective_modulus(self, ud):
        # At span/h = 4 about half the deflection is shear: a flexural
        # modulus would mostly measure the shear correction.
        assert ud[1].effective_modulus is None

    def test_flexure_3pt_on_the_specimen_is_the_same_test(self):
        mesh, solver = _specimen('QI', 0.03)
        r_ilss = solver.solve('ilss')
        r_flex = solver.solve('flexure_3pt')
        np.testing.assert_array_equal(r_flex.displacement, r_ilss.displacement)
        assert r_flex.knockdown == pytest.approx(r_ilss.knockdown, rel=1e-12)
        assert r_ilss.effective_modulus is None and r_flex.effective_modulus > 0


# ---------------------------------------------------------------------------
# D2344 outputs
# ---------------------------------------------------------------------------

class TestShortBeamShearOutputs:

    @pytest.fixture(scope='class')
    @classmethod
    def porous(cls):
        mesh, solver = _specimen('QI', 0.03, pf={'distribution': 'clustered'})
        return mesh, solver

    def test_values(self, porous):
        mesh, solver = porous
        r = solver.solve('ilss', applied_load=LOAD)
        sb = short_beam_shear(r, mesh)
        assert isinstance(sb, ShortBeamShear)
        b, h = mesh.L_y, mesh.L_z
        assert (sb.width, sb.thickness) == (b, h)
        assert sb.span == pytest.approx(4 * h, rel=1e-12)
        assert sb.load_width == mesh.ilss_geometry.load_width
        assert sb.applied_load == pytest.approx(LOAD, rel=1e-9)
        assert sb.apparent_shear == pytest.approx(0.75 * abs(LOAD) / (b * h), rel=1e-9)
        # Below the formula (0.81 here: midplane-clustered voids and one
        # element layer per ply of this coarse laminate; see the UD checks).
        assert 0.7 < sb.midplane_shear / sb.apparent_shear < 1.0
        assert sb.interior_peak_shear is not None and sb.interior_peak_shear > 0
        lam, lam_int = (r.first_ply_failure_load_factor,
                        r.interior_first_ply_failure_load_factor)
        assert sb.first_ply_failure_load == pytest.approx(lam * abs(LOAD), rel=1e-9)
        assert sb.interior_first_ply_failure_load == pytest.approx(
            lam_int * abs(LOAD), rel=1e-9)
        assert sb.apparent_strength == pytest.approx(
            0.75 * sb.first_ply_failure_load / (b * h), rel=1e-12)
        assert sb.interior_apparent_strength == pytest.approx(
            0.75 * sb.interior_first_ply_failure_load / (b * h), rel=1e-12)
        # The interior leaves out the contact zones, where failure starts.
        assert sb.interior_first_ply_failure_load >= sb.first_ply_failure_load

    def test_independent_of_the_load(self, porous):
        mesh, solver = porous
        a = short_beam_shear(solver.solve('ilss', applied_load=LOAD), mesh)
        b = short_beam_shear(solver.solve('ilss', applied_load=-250.0), mesh)
        assert b.applied_load == pytest.approx(-250.0, rel=1e-9)
        for name in ('midplane_shear', 'interior_peak_shear', 'apparent_shear'):
            assert getattr(b, name) / 25.0 == pytest.approx(getattr(a, name), rel=1e-9)
        for name in ('first_ply_failure_load', 'interior_first_ply_failure_load',
                     'apparent_strength', 'interior_apparent_strength'):
            assert getattr(b, name) == pytest.approx(getattr(a, name), rel=1e-9)

    def test_rejects_unusable_input(self, porous):
        mesh, solver = porous
        r = solver.solve('ilss', applied_load=LOAD)
        coupon = CompositeMesh(PorosityField(MAT8, 0.03), MAT8, nx=4, ny=2, nz=2)
        with pytest.raises(ValueError, match='for_ilss'):
            short_beam_shear(r, coupon)
        coupon.ilss_geometry = FlexureGeometry((12.5, 37.5), load_span=10.0)
        with pytest.raises(ValueError, match='do not belong'):
            short_beam_shear(r, coupon)
        with pytest.raises(ValueError, match='reactions'):
            short_beam_shear(dataclasses.replace(r, reaction_forces=None), mesh)
        with pytest.raises(ValueError, match='no load'):
            short_beam_shear(dataclasses.replace(
                r, reaction_forces=np.zeros_like(r.reaction_forces)), mesh)


# ---------------------------------------------------------------------------
# Interior region
# ---------------------------------------------------------------------------

class TestInteriorRegion:

    def test_interior_is_the_shear_spans_clear_of_the_contacts(self, ud):
        mesh, r = ud
        g = mesh.ilss_geometry
        mask = interior_element_mask(mesh, g)
        assert mask.any()
        c = mesh.nodes[mesh.elements[mask]].mean(axis=1)
        d = 0.5 * H8
        x0, x1 = g.support_x
        assert np.all(c[:, 0] >= x0 + d) and np.all(c[:, 0] <= x1 - d)
        assert np.all(np.abs(c[:, 0] - g.midspan) >= 0.5 * g.load_width + d)
        assert np.all(c[:, 1] >= d) and np.all(c[:, 1] <= mesh.L_y - d)
        # No element of the overhangs or under the nose.
        all_c = mesh.nodes[mesh.elements].mean(axis=1)
        assert not mask[(all_c[:, 0] < x0) | (all_c[:, 0] > x1)].any()
        assert not mask[np.abs(all_c[:, 0] - g.midspan) < 0.5 * g.load_width].any()
        # The solve's interior quantities use exactly this region.
        assert r.interior_max_failure_index == pytest.approx(
            r.per_element_failure_index[mask].max(), rel=1e-12)

    def test_margin_option(self, ud):
        mesh, _ = ud
        g = mesh.ilss_geometry
        wide = interior_element_mask(mesh, g, margin=0.0)
        default = interior_element_mask(mesh, g)
        assert default.sum() < wide.sum()
        assert not (default & ~wide).any()
        assert not interior_element_mask(mesh, g, margin=2.0 * H8).any()
        for bad in (-1.0, float('nan'), True):
            with pytest.raises(ValueError, match='interior_margin'):
                interior_element_mask(mesh, g, margin=bad)

    def test_solve_passes_interior_margin(self):
        mesh, solver = _specimen('QI', 0.03)
        r = solver.solve('ilss', interior_margin=0.0)
        mask = interior_element_mask(mesh, mesh.ilss_geometry, 0.0)
        assert r.interior_max_failure_index == pytest.approx(
            r.per_element_failure_index[mask].max(), rel=1e-12)
        none = solver.solve('ilss', interior_margin=2.0 * H8)
        assert none.interior_max_failure_index is None
        assert none.interior_first_ply_failure_load_factor is None


# ---------------------------------------------------------------------------
# Knockdown
# ---------------------------------------------------------------------------

class TestKnockdown:

    def test_short_beam_knocks_down_more_than_the_old_long_bend(self):
        """The old 'ilss' bend spanned 11.4 thicknesses, so it was mostly
        flexure. At 4:1 about half the compliance is transverse shear, which
        porosity degrades faster than the fibre-dominated bending stiffness:
        uniform 6 %, QI, 0.927 against 0.975 at 11.4:1 on rollers."""
        kd = {}
        for ratio in (4.0, 11.4):
            _, solver = _specimen('QI', 0.06, span_to_thickness=ratio,
                                  length_to_thickness=ratio + 2.0,
                                  element_length=H8 / 2)
            kd[ratio] = solver.solve('ilss').knockdown
        assert 0.0 < kd[4.0] < kd[11.4] - 0.03 < 1.0

    def test_pristine_reference_is_keyed_on_the_specimen_geometry(self):
        """The supports and band are not in the node arrays; two specimens
        with the same nodes and different bands must not share a pristine
        reference."""
        pf = PorosityField(MAT8, 0.04)
        line = CompositeMesh.for_ilss(pf, MAT8, nz=4, element_length=H8 / 2,
                                      load_width=0.0)
        band = CompositeMesh.for_ilss(pf, MAT8, nz=4, element_length=H8 / 2,
                                      load_width=0.5 * H8)
        np.testing.assert_array_equal(line.nodes, band.nodes)

        def kd(mesh):
            return FESolver(mesh, MAT8, pf).solve('ilss').knockdown

        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        kd(line)
        after = kd(band)
        solver_mod._PRISTINE_MEASURE_CACHE.clear()
        fresh = kd(band)
        assert after == pytest.approx(fresh, rel=1e-12)
        assert abs(kd(line) - fresh) > 1e-4


# ---------------------------------------------------------------------------
# Meshes that are not a short-beam specimen
# ---------------------------------------------------------------------------

class TestCouponMeshes:

    def test_production_coupon_raises_with_guidance(self):
        pf = PorosityField(T800, 0.03)
        coupon = CompositeMesh(pf, T800, nx=6, ny=3, nz=4)
        with pytest.raises(ValueError) as exc:
            FESolver(coupon, T800, pf).solve('ilss')
        msg = str(exc.value)
        for text in ('ASTM D2344', '50 x 20 mm', 'CompositeMesh.for_ilss',
                     "loading='flexure_3pt'", 'ilss_geometry'):
            assert text in msg

    def test_flexure_mesh_is_not_a_short_beam(self):
        pf = PorosityField(MAT8, 0.0)
        mesh = CompositeMesh.for_flexure(pf, MAT8, span_to_thickness=16.0)
        with pytest.raises(ValueError, match='for_ilss'):
            FESolver(mesh, MAT8, pf).solve('ilss')

    def test_a_geometry_set_by_hand_is_used(self):
        """Opting in: supports on node lines of a coupon."""
        pf = PorosityField(MAT8, 0.03)
        coupon = CompositeMesh(pf, MAT8, nx=24, ny=4, nz=4)
        dx = coupon.L_x / coupon.nx
        coupon.ilss_geometry = FlexureGeometry((8 * dx, 16 * dx), load_span=4 * dx,
                                               load_width=1.0)
        r = FESolver(coupon, MAT8, pf).solve('ilss')
        assert 0.0 < r.knockdown < 1.0
        assert np.abs(r.reaction_forces[:, 0]).max() < 1e-9 * abs(LOAD)
        sb = short_beam_shear(r, coupon)
        assert sb.span == pytest.approx(8 * dx, rel=1e-12)
