"""Delamination fracture specimens (DCB, ENF, MMB) and the NASA benchmarks.

Delamination plan, phase 3. This checks:

- the builders :meth:`CompositeMesh.for_dcb`, ``for_enf`` and ``for_mmb``:
  the recorded :class:`FractureSpecimenGeometry`, the insert, node lines on
  every load, support and crack tip, the graded mesh at the front and the
  standard defaults;
- the ``'dcb'``, ``'enf'`` and ``'mmb'`` loading modes: statics of the
  loads and reactions, the load-point compliance, the outputs and exports;
- the MMB lever relations (:func:`mmb_lever_length`,
  :func:`mmb_mode_ratio`) against Krueger's lever lengths and the FE mode
  mix;
- the onset loads of Krueger's NASA benchmark specimens:

  - DCB, T300/1076, NASA/TM-2008-215123 (critical opening at a = 30 mm);
  - ENF, IM7/8552, NASA/CR-2011-217305 (Q_crit = 1526 N);
  - MMB, IM7/8552, NASA/CR-2012-217562 (P_crit = 385 N at G_II/G_T = 0.5,
    751 N at 0.8);
  - SLB, C12K/R6376, NASA/TM-2008-215123, behind POROSITY_FE_FRACTURE_BENCH;

- the compliance-based data reduction (:func:`compliance_calibration`)
  against the virtual crack closure, and :func:`refinement_study` on a
  fracture specimen.

Every benchmark number is taken from the reports (table or section cited
next to it). The [0]24 specimens are unidirectional, so their ply count only
sets the element layers: they are modelled as ``n`` layers of thickness
``2h / n``, one element each. The strengths and constituent properties the
benchmark materials take from the T800 preset do not enter: the porosity is
zero and no failure index is checked.

Units: N, mm, MPa; energy release rates and toughness in N/mm.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    ComplianceCalibration,
    CompositeMesh,
    FESolver,
    FractureSpecimenGeometry,
    PorosityField,
    compliance_calibration,
    mmb_lever_length,
    mmb_mode_ratio,
    refinement_study,
)
from porosity_fe.fe._constrained_solve import free_dof_system, solve_constrained
from porosity_fe.fe.assembler import BoundaryHandler, GlobalAssembler, _mesh_ties
from porosity_fe.fe.vcct import fracture_load_factors, vcct_energy_release_rates

#: The heavier mesh study and the SLB benchmark run with this set to 1.
BENCH = os.environ.get('POROSITY_FE_FRACTURE_BENCH') == '1'

# --------------------------------------------------------------------------
# Benchmark materials (MPa, N/mm)
# --------------------------------------------------------------------------

#: T300/1076 unidirectional graphite/epoxy, NASA/TM-2008-215123 Table 1.
T300_1076 = dict(E11=139400.0, E22=10160.0, E33=10160.0, nu12=0.30, nu13=0.30,
                 nu23=0.436, G12=4600.0, G13=4600.0, G23=3540.0)
#: The DCB benchmark's toughness: G_Ic = 170.3 J/m^2 of T300/914C (TM-2008-215123
#: section 5.1 and Fig. 2a, with G_IIc = 493.6 J/m^2 and eta = 1.62).
T300_TOUGHNESS = dict(G_Ic=0.1703, G_IIc=0.4936, bk_eta=1.62)
#: IM7/8552 unidirectional graphite/epoxy, NASA/CR-2011-217305 Table I (the
#: same constants as NASA/CR-2012-217562 Table I).
IM7_8552 = dict(E11=161000.0, E22=11380.0, E33=11380.0, nu12=0.32, nu13=0.32,
                nu23=0.45, G12=5200.0, G13=5200.0, G23=3900.0)
#: The ENF benchmark's toughness, CR-2011-217305 Table II (G_IIc = G_IIIc).
ENF_TOUGHNESS = dict(G_Ic=0.21, G_IIc=0.78, bk_eta=2.57)
#: The MMB benchmark's toughness, CR-2012-217562 Table II (G_IIc = G_IIIc).
MMB_TOUGHNESS = dict(G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)


def _material(elastic: dict, toughness: dict, thickness: float, layers: int):
    """A UD benchmark material, ``thickness`` mm in ``layers`` element layers."""
    return dataclasses.replace(MATERIALS['T800_epoxy'], **elastic, **toughness,
                               t_ply=thickness / layers, n_plies=layers)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)   # plane-strain aspect ratios
        yield
    logging.disable(logging.NOTSET)


def _centre(front, width: float) -> int:
    """Row of the front node at mid-width (the reports' y/B = 0)."""
    return int(np.argmin(np.abs(front.y - 0.5 * width)))


def _x_lines(mesh) -> np.ndarray:
    return np.unique(np.asarray(mesh.nodes)[:mesh.n_structured_nodes, 0])


def _on_line(xs: np.ndarray, x: float) -> bool:
    return bool(np.min(np.abs(xs - x)) <= 1e-9 * max(float(xs[-1]), 1.0))


# --------------------------------------------------------------------------
# Krueger's benchmark specimens
# --------------------------------------------------------------------------

#: DCB: B = 25.0, 2h = 3.0, 2L = 150.0, a = 30.0 mm, [0]24
#: (TM-2008-215123 Fig. 3a).
DCB_B, DCB_2H, DCB_LENGTH, DCB_A = 25.0, 3.0, 150.0, 30.0
#: Critical opening of each arm at a = 30 mm: delta_crit / 2 = 0.75 mm
#: (TM-2008-215123 section 5.2.1). The critical load is only plotted (Fig. 14).
DCB_HALF_OPENING_CRIT = 0.75
#: ENF: B = 25.4, 2h = 4.5, 2L = 101.6, a0 = 25.4 mm, [0]24 (CR-2011-217305
#: Fig. 1); Q_crit = 1526 N, w_crit = 1.42 mm (section 6.3), from 2-D plane
#: strain (CPE4I) models with G_IIc = 0.78 N/mm.
ENF_B, ENF_2H, ENF_SPAN, ENF_A = 25.4, 4.5, 101.6, 25.4
ENF_Q_CRIT = 1526.0
#: MMB: b = 25.4, 2h = 4.5, 2L = 100.8, a0 = 25.4 mm, [0]24 (CR-2012-217562
#: Fig. 1); lever c = 41.3 mm at G_II/G_T = 0.5 and 27.3 mm at 0.8 (Fig. 4);
#: P_crit = 385 N and 751 N (sections 5.3.2, 5.4.2) from 2-D plane-strain
#: (CPE4I) models.
MMB_B, MMB_2H, MMB_SPAN, MMB_A = 25.4, 4.5, 100.8, 25.4
MMB_CASES = {0.5: (41.3, 385.0), 0.8: (27.3, 751.0)}


def _dcb(layers: int = 12, **kw):
    mat = _material(T300_1076, T300_TOUGHNESS, DCB_2H, layers)
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh.for_dcb(pf, mat, crack_length=DCB_A, length=DCB_LENGTH,
                                 width=DCB_B, **kw)
    return mesh, mat, pf


def _enf(layers: int = 12, **kw):
    mat = _material(IM7_8552, ENF_TOUGHNESS, ENF_2H, layers)
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh.for_enf(pf, mat, crack_length=ENF_A, span=ENF_SPAN,
                                 width=ENF_B, **kw)
    return mesh, mat, pf


def _mmb(lever: float, layers: int = 12, **kw):
    mat = _material(IM7_8552, MMB_TOUGHNESS, MMB_2H, layers)
    pf = PorosityField(mat, 0.0)
    mesh = CompositeMesh.for_mmb(pf, mat, crack_length=MMB_A, span=MMB_SPAN,
                                 width=MMB_B, lever_length=lever, **kw)
    return mesh, mat, pf


def _solve(mesh, mat, pf, loading: str, load: float, **kw):
    return FESolver(mesh, mat, pf, ply_angles='UD').solve(loading, applied_load=load, **kw)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

class TestBuilders:
    def test_dcb_geometry_and_insert(self):
        mesh, mat, _ = _dcb(layers=6)
        g = mesh.fracture_geometry
        assert isinstance(g, FractureSpecimenGeometry)
        assert (g.test, g.crack_length, g.load_end, g.interface) == ('dcb', DCB_A, 0.0, 2)
        assert g.span is None and g.lever_length is None and g.support_x is None
        assert (mesh.L_x, mesh.L_y, mesh.L_z) == (DCB_LENGTH, DCB_B, pytest.approx(DCB_2H))
        (crack,) = mesh.delaminations
        assert (crack.shape, crack.interface, crack.x_start, crack.x_end) == \
            ('strip', 2, 0.0, DCB_A)
        xs = _x_lines(mesh)
        assert _on_line(xs, 0.0) and _on_line(xs, DCB_A)
        front = mesh.crack_front_pairs()[0]
        np.testing.assert_allclose(mesh.nodes[front, 0], DCB_A)
        # Open at the loaded end: the pairs on x = 0 are delaminated.
        orig, _ = mesh.interface_node_pairs()
        at_end = np.abs(mesh.nodes[orig, 0]) < 1e-12
        assert not mesh.bonded_pairs()[at_end].any()

    def test_graded_mesh_has_equal_elements_at_the_front(self):
        mesh, mat, _ = _dcb(layers=6)
        xs = _x_lines(mesh)
        assert mesh.x_grid is not None
        np.testing.assert_array_equal(xs, mesh.x_grid)
        i = int(np.argmin(np.abs(xs - DCB_A)))
        dx = np.diff(xs)
        fine = 0.25 * DCB_2H                          # default: a quarter of 2h
        np.testing.assert_allclose(dx[i - 6:i + 6], fine, rtol=1e-12)
        assert dx.max() <= DCB_2H * (1 + 1e-12)       # default largest: 2h
        assert np.all(dx[i + 6:] >= dx[i + 5] * (1 - 1e-9))   # grows away from it
        assert mesh.nx < DCB_LENGTH / fine / 3

    def test_uniform_grid_needs_aligned_nx(self):
        mesh, _, _ = _dcb(layers=6, nx=150)
        assert mesh.x_grid is None and mesh.nx == 150
        with pytest.raises(ValueError, match='multiple of 5'):
            _dcb(layers=6, nx=151)
        with pytest.raises(ValueError, match='element_length'):
            _mmb(41.3, layers=6, nx=100)     # 25.4 and 50.4 on a 100.8 grid

    def test_enf_and_mmb_node_lines(self):
        mesh, _, _ = _enf(layers=6, overhang=5.0)
        g = mesh.fracture_geometry
        assert g.support_x == (5.0, 5.0 + ENF_SPAN) and g.midspan == 5.0 + 0.5 * ENF_SPAN
        assert g.crack_tip_x == 5.0 + ENF_A and mesh.L_x == ENF_SPAN + 10.0
        xs = _x_lines(mesh)
        for x in (0.0, 5.0, 5.0 + ENF_A, g.midspan, 5.0 + ENF_SPAN, mesh.L_x):
            assert _on_line(xs, x)
        mesh, _, _ = _mmb(41.3, layers=6)
        g = mesh.fracture_geometry
        assert (g.test, g.lever_length, g.half_span) == ('mmb', 41.3, 0.5 * MMB_SPAN)
        xs = _x_lines(mesh)
        for x in (0.0, MMB_A, 0.5 * MMB_SPAN, MMB_SPAN):
            assert _on_line(xs, x)

    def test_standard_defaults(self):
        """The plan's reading of the standards (check against the purchased
        ASTM editions): D5528 DCB 125 x 25 mm, a0 = 50 mm; D7905 ENF
        2L = 100 mm, a0 = 30 mm; D6671 MMB L = 50 mm, a0 = 25 mm, with the
        beam-theory lever for G_II/G_T = 0.5."""
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], **MMB_TOUGHNESS)
        pf = PorosityField(mat, 0.0)
        dcb = CompositeMesh.for_dcb(pf, mat).fracture_geometry
        assert (dcb.crack_length,) == (50.0,)
        enf = CompositeMesh.for_enf(pf, mat)
        assert (enf.fracture_geometry.span, enf.fracture_geometry.crack_length,
                enf.L_y, enf.nz) == (100.0, 30.0, 25.0, mat.n_plies)
        mmb = CompositeMesh.for_mmb(pf, mat).fracture_geometry
        assert (mmb.span, mmb.crack_length) == (100.0, 25.0)
        assert mmb.lever_length == pytest.approx(
            mmb_lever_length(mat, 0.5, span=100.0, crack_length=25.0), rel=1e-12)
        assert np.all(CompositeMesh.for_dcb(pf, mat).ply_angles == 0.0)   # 'UD'

    def test_plane_strain_and_bad_inputs(self):
        mesh, _, _ = _enf(layers=6, plane_strain=True, ny=1)
        assert mesh.fracture_geometry.plane_strain and mesh.ny == 1
        mat = _material(IM7_8552, ENF_TOUGHNESS, ENF_2H, 6)
        pf = PorosityField(mat, 0.0)
        with pytest.raises(ValueError, match='inside the'):
            CompositeMesh.for_enf(pf, mat, crack_length=120.0, span=100.0)
        with pytest.raises(ValueError, match='mid-span'):
            CompositeMesh.for_enf(pf, mat, crack_length=50.0, span=100.0)
        with pytest.raises(ValueError, match='not both'):
            CompositeMesh.for_mmb(pf, mat, lever_length=40.0, mode_ratio=0.5)
        with pytest.raises(ValueError, match='multiple of n_plies'):
            CompositeMesh.for_dcb(pf, mat, nz=9)
        with pytest.raises(ValueError, match='shorter'):
            CompositeMesh.for_dcb(pf, mat, crack_length=130.0)
        odd = dataclasses.replace(mat, n_plies=5)
        with pytest.raises(ValueError, match='interface='):
            CompositeMesh.for_dcb(PorosityField(odd, 0.0), odd)
        with pytest.raises(ValueError, match="'mmb' specimen needs"):
            FractureSpecimenGeometry('mmb', 25.0, 2, span=100.0)
        with pytest.raises(ValueError, match='no span'):
            FractureSpecimenGeometry('dcb', 25.0, 2, span=100.0)

    def test_x_grid_must_match_the_mesh(self):
        mesh, _, _ = _dcb(layers=6)
        mesh.L_x = 160.0
        with pytest.raises(ValueError, match='x_grid'):
            mesh.generate_mesh()
        mesh.x_grid = None
        mesh.generate_mesh()                     # evenly spaced again
        np.testing.assert_allclose(np.diff(_x_lines(mesh)), 160.0 / mesh.nx)


# --------------------------------------------------------------------------
# Loading modes
# --------------------------------------------------------------------------

class TestLoadingModes:
    def test_dcb_hinge_loads_are_self_equilibrated(self):
        mesh, mat, pf = _dcb(layers=6)
        r = _solve(mesh, mat, pf, 'dcb', -20.0)        # the sign is ignored
        assert np.abs(r.reaction_forces).max() < 1e-9 * 20.0
        assert r.effective_modulus is None and r.contact_pairs == 0
        _, F = BoundaryHandler(mesh).fracture_bcs(20.0)
        top = np.abs(mesh.nodes[:, 2] - mesh.L_z) < 1e-12
        assert F[2::3][top].sum() == pytest.approx(20.0)
        assert F[2::3].sum() == pytest.approx(0.0, abs=1e-12)
        u = r.displacement
        assert r.load_point_compliance == pytest.approx(float(F @ u.ravel()) / 20.0 ** 2,
                                                        rel=1e-12)
        # The hinge lines open by C P: the upper arm up, the lower down.
        x0 = np.abs(mesh.nodes[:, 0]) < 1e-12
        lower = x0 & (np.abs(mesh.nodes[:, 2]) < 1e-12)
        opening = u[x0 & top, 2].mean() - u[lower, 2].mean()
        assert opening == pytest.approx(r.load_point_compliance * 20.0, rel=1e-3)
        assert r.fracture_load_factor > 0.0
        assert r.summary().details['load_point_compliance'] == r.load_point_compliance

    def test_mmb_reactions_are_the_lever_statics(self):
        mesh, mat, pf = _mmb(41.3, layers=6)
        r = _solve(mesh, mat, pf, 'mmb', -100.0)
        loads = mesh.fracture_geometry.mmb_loads(-100.0)
        x = mesh.nodes[:, 0]
        R = r.reaction_forces
        assert R[np.abs(x) < 1e-9, 2].sum() == pytest.approx(loads['reaction_end'], rel=1e-9)
        assert R[np.abs(x - MMB_SPAN) < 1e-9, 2].sum() == pytest.approx(
            loads['reaction_far'], rel=1e-9)
        np.testing.assert_allclose(R[:, :2].sum(axis=0), 0.0, atol=1e-8)
        # The split into Reeder and Crews' DCB and ENF parts.
        assert loads['mode_I'] > 0.0 and loads['mode_II'] == loads['saddle']
        assert loads['hinge'] == pytest.approx(loads['mode_I'] - loads['mode_II'] / 4)
        assert loads['reaction_end'] == pytest.approx(-loads['mode_I'] - loads['mode_II'] / 4)
        assert loads['hinge'] + loads['reaction_end'] + loads['reaction_far'] == \
            pytest.approx(-loads['saddle'])
        # The work-conjugate displacement is the rigid lever's load point.
        u = r.displacement[:, 2]
        top = np.abs(mesh.nodes[:, 2] - mesh.L_z) < 1e-9
        hinge_up = u[(np.abs(x) < 1e-9) & top].mean()
        saddle_down = -u[(np.abs(x - 0.5 * MMB_SPAN) < 1e-9) & top].mean()
        c, L = 41.3, 0.5 * MMB_SPAN
        lever_point = saddle_down * (c + L) / L + hinge_up * c / L
        assert r.load_point_compliance * 100.0 == pytest.approx(lever_point, rel=2e-3)

    def test_enf_uses_contact_and_is_mode_II(self):
        mesh, mat, pf = _enf(layers=6)
        r = _solve(mesh, mat, pf, 'enf', -1000.0)
        assert r.crack_contact and r.contact_converged and r.contact_pairs > 0
        front = r.crack_front
        assert np.all(front.psi > 0.999)
        assert np.max(front.G_I / front.G_T) < 1e-4
        assert r.fracture_load_factor == pytest.approx(float(np.min(front.lambda_c)))

    def test_knockdown_is_the_bonded_compliance_ratio(self):
        """Pristine reference: the bonded, pore-free specimen with the same
        loads; KD = its compliance over the solved one's."""
        mesh, mat, pf = _enf(layers=6)
        r = _solve(mesh, mat, pf, 'enf', -1000.0)
        bonded = dataclasses.replace(mesh.fracture_geometry)  # same loads
        mesh.delaminations = ()
        mesh.generate_mesh()
        mesh.fracture_geometry = bonded
        ref = _solve(mesh, mat, pf, 'enf', -1000.0)
        assert ref.knockdown == 1.0 and ref.crack_front is None
        assert r.knockdown == pytest.approx(
            ref.load_point_compliance / r.load_point_compliance, rel=1e-9)
        assert 0.0 < r.knockdown < 1.0

    def test_wrong_specimen_raises_before_assembly(self):
        mesh, mat, pf = _dcb(layers=6)
        with pytest.raises(ValueError, match="'enf' on a 'dcb' specimen"):
            _solve(mesh, mat, pf, 'enf', -10.0)
        plain = CompositeMesh(pf, mat, nx=4, ny=2, nz=6)
        with pytest.raises(ValueError, match='for_mmb'):
            FESolver(plain, mat, pf).solve('mmb')

    def test_other_modes_report_no_compliance(self):
        mesh, mat, pf = _enf(layers=6)
        assert _solve(mesh, mat, pf, 'flexure_3pt', -10.0).load_point_compliance is None

    def test_export_and_plane_strain_constraint(self, tmp_path):
        mesh, mat, pf = _mmb(27.3, layers=6, plane_strain=True, ny=1)
        constrained, _ = BoundaryHandler(mesh).fracture_bcs(-10.0)
        tied = np.zeros(mesh.n_nodes, dtype=bool)
        tied[mesh.interface_node_pairs()[1][mesh.bonded_pairs()]] = True
        assert all(constrained.get(3 * n + 1) == 0.0 for n in np.flatnonzero(~tied))
        r = _solve(mesh, mat, pf, 'mmb', -10.0)
        np.testing.assert_allclose(r.displacement[:, 1], 0.0, atol=1e-15)
        path = tmp_path / 'mmb.json'
        FESolver.export_results(r, path)
        doc = json.loads(path.read_text(encoding='utf-8'))
        assert doc['stiffness']['load_point_compliance_mm_per_N'] == \
            pytest.approx(r.load_point_compliance)
        assert doc['stiffness']['effective_modulus_MPa'] is None


# --------------------------------------------------------------------------
# MMB lever relations
# --------------------------------------------------------------------------

class TestMMBLever:
    def test_against_krueger_lever_lengths(self):
        """The corrected beam theory gives 91.96, 40.90 and 27.02 mm for
        G_II/G_T = 0.2, 0.5 and 0.8 on the benchmark specimen; Krueger used
        92.9, 41.3 and 27.3 mm (CR-2012-217562 Figs. 3a and 4), set from the
        experimental study's specimens: 1.0 to 1.1 % longer."""
        mat = _material(IM7_8552, MMB_TOUGHNESS, MMB_2H, 6)
        kw = dict(span=MMB_SPAN, crack_length=MMB_A)
        for ratio, c_krueger in ((0.2, 92.9), (0.5, 41.3), (0.8, 27.3)):
            c = mmb_lever_length(mat, ratio, **kw)
            assert c == pytest.approx(c_krueger, rel=0.015)
            assert mmb_mode_ratio(mat, c, **kw) == pytest.approx(ratio, rel=1e-12)
        assert mmb_lever_length(mat, 1.0, **kw) == pytest.approx(MMB_SPAN / 6)
        assert mmb_mode_ratio(mat, 5.0, **kw) == 1.0          # c < L/3 closes the crack
        with pytest.raises(ValueError, match='mode_ratio'):
            mmb_lever_length(mat, 0.0, **kw)
        with pytest.raises(ValueError, match='pure mode I'):
            mmb_lever_length(mat, 1e-4, **kw)

    def test_beam_theory_without_corrections_is_reeder_and_crews(self):
        """With no shear or root rotation (beta = 1) the ratio is
        (4/3) ((3c - L) / (c + L))^2 (Reeder 1992, TM-104210, Eq. 1)."""
        stiff = dataclasses.replace(_material(IM7_8552, MMB_TOUGHNESS, MMB_2H, 6),
                                    G13=1e12)
        c, L = 41.3, 0.5 * MMB_SPAN
        r = 4.0 / 3.0 * ((3 * c - L) / (c + L)) ** 2
        assert mmb_mode_ratio(stiff, c, span=MMB_SPAN, crack_length=MMB_A) == \
            pytest.approx(1.0 / (1.0 + r), rel=1e-5)

    def test_fe_mode_mix_follows_the_beam_theory(self):
        """FE centre-node G_II/G_T at the beam-theory lever for 0.5 and 0.8
        (12 layers): within 3 % of the target in plane strain and 5 % in the
        three-dimensional specimen (whose centre is slightly more mode I)."""
        for target in (0.5, 0.8):
            mat = _material(IM7_8552, MMB_TOUGHNESS, MMB_2H, 12)
            c = mmb_lever_length(mat, target, span=MMB_SPAN, crack_length=MMB_A)
            for plane_strain, tol in ((True, 0.03), (False, 0.05)):
                mesh, mat, pf = _mmb(c, layers=12, plane_strain=plane_strain,
                                     ny=1 if plane_strain else None)
                front = _solve(mesh, mat, pf, 'mmb', -100.0).crack_front
                psi = float(front.psi[_centre(front, MMB_B)])
                assert psi == pytest.approx(target, rel=tol), (target, plane_strain, psi)
                assert psi < target


# --------------------------------------------------------------------------
# NASA benchmarks (verification gate)
# --------------------------------------------------------------------------

class TestNASABenchmarks:
    """Onset loads of Krueger's benchmark specimens at the centre of the
    width (the reports' y/B = 0), from lambda_c times the applied load. The
    DCB reference comes from a three-dimensional model (C3D8I), the ENF and
    MMB ones from two-dimensional plane-strain models (CPE4I), which the
    plane-strain specimens here match; the three-dimensional specimens are
    checked against the same numbers. 12 element layers (six per arm) and
    the default graded mesh; the measured values and the mesh study are in
    the theory page's benchmark table."""

    def test_dcb_critical_opening(self):
        mesh, mat, pf = _dcb(layers=12)
        r = _solve(mesh, mat, pf, 'dcb', 10.0)
        front = r.crack_front
        lam = float(front.lambda_c[_centre(front, DCB_B)])
        assert np.max(front.psi) < 1e-6                     # pure mode I
        half_opening = 0.5 * r.load_point_compliance * 10.0 * lam
        assert half_opening == pytest.approx(DCB_HALF_OPENING_CRIT, rel=0.03)

    @pytest.mark.parametrize('plane_strain', [True, False])
    def test_enf_critical_load(self, plane_strain):
        mesh, mat, pf = _enf(layers=12, plane_strain=plane_strain,
                             ny=1 if plane_strain else None)
        r = _solve(mesh, mat, pf, 'enf', -1000.0)
        front = r.crack_front
        Q = 1000.0 * float(front.lambda_c[_centre(front, ENF_B)])
        assert Q == pytest.approx(ENF_Q_CRIT, rel=0.03)

    @pytest.mark.parametrize('plane_strain', [True, False])
    @pytest.mark.parametrize('ratio', [0.5, 0.8])
    def test_mmb_critical_load(self, ratio, plane_strain):
        lever, P_crit = MMB_CASES[ratio]
        mesh, mat, pf = _mmb(lever, layers=12, plane_strain=plane_strain,
                             ny=1 if plane_strain else None)
        r = _solve(mesh, mat, pf, 'mmb', -100.0)
        front = r.crack_front
        P = 100.0 * float(front.lambda_c[_centre(front, MMB_B)])
        assert P == pytest.approx(P_crit, rel=0.03)


@pytest.mark.skipif(not BENCH, reason='mesh study: set POROSITY_FE_FRACTURE_BENCH=1')
class TestNASABenchmarkMeshStudy:
    """The benchmarks at 24 element layers (12 per arm), the finest column
    of the theory page's mesh study: the onset loads keep within 3 % of
    the references, and the plane-strain ENF and MMB loads fall from 12 to
    24 layers (the energy release rate rises with the through-thickness
    resolution) while the DCB opening rises."""

    def test_dcb(self):
        values = []
        for layers in (12, 24):
            mesh, mat, pf = _dcb(layers=layers)
            r = _solve(mesh, mat, pf, 'dcb', 10.0)
            lam = float(r.crack_front.lambda_c[_centre(r.crack_front, DCB_B)])
            values.append(0.5 * r.load_point_compliance * 10.0 * lam)
        assert values[1] > values[0]
        assert values[1] == pytest.approx(DCB_HALF_OPENING_CRIT, rel=0.03)

    def test_enf_and_mmb(self):
        cases = [('enf', None, ENF_Q_CRIT, ENF_B)] + \
            [('mmb', lever, P, MMB_B) for lever, P in MMB_CASES.values()]
        for test, lever, reference, width in cases:
            for plane_strain in (True, False):
                loads = []
                for layers in (12, 24):
                    kw = dict(layers=layers, plane_strain=plane_strain,
                              ny=1 if plane_strain else None)
                    mesh, mat, pf = _enf(**kw) if test == 'enf' else _mmb(lever, **kw)
                    front = _solve(mesh, mat, pf, test, -100.0).crack_front
                    loads.append(100.0 * float(front.lambda_c[_centre(front, width)]))
                assert loads[1] < loads[0]
                assert loads[1] == pytest.approx(reference, rel=0.03), (test, lever, loads)


#: SLB: C12K/R6376, D+-30 layup, B = 25.4, t1 = t2 = 2.032, 2L = 177.8,
#: a = 34.29 mm (NASA/TM-2008-215123 Fig. 3b; the SLB model figures, Figs.
#: 9 and 10, label t1 = t2 = 2.0 and a = 34.0 mm). Ply constants Table 1;
#: G_Ic = 340.5, G_IIc = 1285.9 J/m^2, eta = 3.39 (Fig. 2b), the shear modes
#: summed in the criterion (section 3.1; plain B-K here with G_IIIc = G_IIc).
#: Critical centre deflection
#: w_crit = 3.23 mm (section 5.6.1), from geometrically nonlinear 3-D
#: models whose plies away from the delamination are smeared (section 4).
C12K_R6376 = dict(E11=146900.0, E22=10600.0, E33=10600.0, nu12=0.33, nu13=0.33,
                  nu23=0.33, G12=5450.0, G13=5450.0, G23=3990.0)
C12K_TOUGHNESS = dict(G_Ic=0.3405, G_IIc=1.2859, bk_eta=3.39)
#: [+-30/0/-30/0/30/0_4/30/0/-30/0/-30/30/ ! -30/30/0/30/0/-30/0_4/-30/0/30/0/-+30]
#: (TM-2008-215123 section 2; "!" the delamination, after ply 16).
D_PM30 = [30, -30, 0, -30, 0, 30, 0, 0, 0, 0, 30, 0, -30, 0, -30, 30,
          -30, 30, 0, 30, 0, -30, 0, 0, 0, 0, -30, 0, 30, 0, 30, -30]
SLB_B, SLB_T_ARM, SLB_SPAN, SLB_A, SLB_W_CRIT = 25.4, 2.032, 177.8, 34.29, 3.23


@pytest.mark.skipif(not BENCH, reason='SLB benchmark: set POROSITY_FE_FRACTURE_BENCH=1')
class TestSingleLegBendingBenchmark:
    """The multidirectional single-leg-bending specimen: the ENF builder's
    beam, supported under the upper arm at the cracked end (the lower arm
    free) and at the far end, loaded at mid-span (Fig. 9). One element per
    ply (32 layers), the default graded mesh, 8 elements across (73,440
    DOF, about 35 s). Measured (theory page): critical centre deflection
    3.361 mm, 4.0 % above the report's 3.23 mm (P_crit = 313.9 N),
    G_II / G_T = 0.351 at mid-width; with the FE figures' t = 2.0 and
    a = 34.0 mm it would be 3.457 mm (+7.0 %)."""

    def test_critical_deflection(self):
        mat = dataclasses.replace(MATERIALS['T800_epoxy'], **C12K_R6376,
                                  **C12K_TOUGHNESS, t_ply=SLB_T_ARM / 16, n_plies=32)
        pf = PorosityField(mat, 0.0)
        mesh = CompositeMesh.for_enf(pf, mat, crack_length=SLB_A, span=SLB_SPAN,
                                     width=SLB_B, ply_angles=D_PM30, ny=8)
        bh = BoundaryHandler(mesh)
        load = -100.0
        F = bh.line_load(0.5 * SLB_SPAN, load)
        _, duplicates = mesh.interface_node_pairs()
        upper = duplicates[np.abs(mesh.nodes[duplicates, 0]) < 1e-9]   # upper arm, x = 0
        far = bh.face_line_nodes('z_min', SLB_SPAN)
        constrained = {3 * int(n) + 2: 0.0 for n in np.concatenate([upper, far])}
        for line, comps in ((upper, (0, 1)), (far, (1,))):
            mid = int(line[np.argmin(np.abs(mesh.nodes[line, 1] - 0.5 * SLB_B))])
            constrained.update({3 * mid + c: 0.0 for c in comps})
        K = GlobalAssembler(mesh, mat, pf).stiffness()
        system = partial(free_dof_system, OrderedDict(), max_entries=1)
        u, _ = solve_constrained(K, F, constrained, system=system, ties=_mesh_ties(mesh))
        orig, dup = mesh.interface_node_pairs()
        uz = u.reshape(-1, 3)[:, 2]
        assert np.min((uz[dup] - uz[orig])[~mesh.bonded_pairs()]) >= 0.0   # open, no contact
        front = fracture_load_factors(vcct_energy_release_rates(mesh, u, K @ u - F), mat)
        i = _centre(front, SLB_B)
        w_crit = float(F @ u) / abs(load) * float(front.lambda_c[i])   # load-point deflection
        assert 0.3 < float(front.psi[i]) < 0.4
        assert w_crit == pytest.approx(SLB_W_CRIT, rel=0.05)


# --------------------------------------------------------------------------
# Emulated data reduction
# --------------------------------------------------------------------------

class TestDataReduction:
    def test_dcb_mbt_and_compliance_calibration_match_vcct(self):
        """Compliance of the benchmark DCB at a = 26 to 34 mm, reduced by
        D5528 modified beam theory and Berry's compliance calibration: G at
        a = 30 mm within 1 % of the width-averaged VCCT (0.9993 and 0.9994),
        and the MBT correction Delta near the beam-theory chi h (2.67 and
        2.90 mm)."""
        lengths = np.array([26.0, 28.0, 30.0, 32.0, 34.0])
        C, G_vcct = [], None
        for a in lengths:
            mat = _material(T300_1076, T300_TOUGHNESS, DCB_2H, 6)
            pf = PorosityField(mat, 0.0)
            mesh = CompositeMesh.for_dcb(pf, mat, crack_length=float(a),
                                         length=DCB_LENGTH, width=DCB_B, ny=4)
            r = _solve(mesh, mat, pf, 'dcb', 10.0)
            C.append(r.load_point_compliance)
            if a == DCB_A:
                f = r.crack_front
                G_vcct = float(np.sum(f.G_T * f.width) / DCB_B)
        for method in ('mbt', 'power'):
            cal = compliance_calibration(lengths, C, DCB_B, method=method)
            assert isinstance(cal, ComplianceCalibration)
            np.testing.assert_allclose(cal.compliance(lengths), C, rtol=0.01)
            assert float(cal.energy_release_rate(10.0, DCB_A)) == pytest.approx(G_vcct, rel=0.01)
        mbt = compliance_calibration(lengths, C, DCB_B, method='mbt')
        assert 2.0 < mbt.coefficients['delta'] < 3.0     # crack-length correction, mm
        P = mbt.critical_load(mat.G_Ic, DCB_A)
        assert float(mbt.energy_release_rate(P, DCB_A)) == pytest.approx(mat.G_Ic)

    def test_enf_cubic_calibration(self):
        """D7905 cubic fit of the plane-strain ENF at a0 -+ 5.08 mm: G at a0
        within 1 % of VCCT (0.9970)."""
        lengths = np.array([20.32, 22.86, 25.4, 27.94, 30.48])     # a0 -+ 2 x 2.54 mm
        C, G_vcct = [], None
        for a in lengths:
            mat = _material(IM7_8552, ENF_TOUGHNESS, ENF_2H, 6)
            pf = PorosityField(mat, 0.0)
            mesh = CompositeMesh.for_enf(pf, mat, crack_length=float(a), span=ENF_SPAN,
                                         width=ENF_B, plane_strain=True, ny=1)
            r = _solve(mesh, mat, pf, 'enf', -1000.0)
            C.append(r.load_point_compliance)
            if a == ENF_A:
                G_vcct = float(r.crack_front.G_T.mean())
        cal = compliance_calibration(lengths, C, ENF_B, method='cubic')
        assert float(cal.energy_release_rate(1000.0, ENF_A)) == pytest.approx(G_vcct, rel=0.01)

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match='method'):
            compliance_calibration([1, 2], [1, 2], 1.0, method='berry')
        with pytest.raises(ValueError, match='same length'):
            compliance_calibration([1, 2, 3], [1, 2], 1.0, method='mbt')
        with pytest.raises(ValueError, match='two distinct'):
            compliance_calibration([1, 1], [1, 2], 1.0, method='mbt')
        with pytest.raises(ValueError, match='positive'):
            compliance_calibration([1, 2], [-1, 2], 1.0, method='power')


# --------------------------------------------------------------------------
# Refinement study
# --------------------------------------------------------------------------

class TestRefinementStudy:
    def test_enf_levels_keep_the_node_lines(self):
        mat = _material(IM7_8552, ENF_TOUGHNESS, ENF_2H, 6)
        st = refinement_study(
            mat, 0.0, 'enf', ply_angles='UD', resolutions=[(40, 2, 6), (80, 2, 6)],
            applied_load=-1000.0,
            specimen=dict(crack_length=30.0, span=100.0, width=25.0, plane_strain=True))
        assert list(st.convergence)[-2:] == ['load_point_compliance', 'fracture_load_factor']
        assert all(lv.fracture_load_factor > 0 for lv in st.levels)
        assert st.convergence['effective_modulus'].behavior == 'unavailable'
        assert 'lambda_c' in st.summary()
        with pytest.raises(ValueError, match='multiple of 10'):
            refinement_study(mat, 0.0, 'enf', ply_angles='UD',
                             resolutions=[(45, 2, 6), (90, 2, 6)])
        with pytest.raises(ValueError, match='specimen='):
            refinement_study(mat, 0.0, 'compression', specimen={'span': 50.0})

    def test_default_plan_is_aligned(self):
        mat = _material(IM7_8552, MMB_TOUGHNESS, MMB_2H, 6)
        st = refinement_study(mat, 0.0, 'dcb', ply_angles='UD', factors=(1, 2),
                              specimen=dict(length=100.0, crack_length=40.0, width=10.0),
                              applied_load=10.0)
        assert [lv.resolution[0] % 5 for lv in st.levels] == [0, 0]
        assert st.convergence['load_point_compliance'].kind == 'global'
