"""Hybrid laminates: per-ply material and thickness (PlyStack)."""

import dataclasses
import json

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    PlyStack,
    PorosityField,
    compute_clt_effective_modulus,
    compute_degraded_clt_flexural_modulus,
    compute_degraded_clt_moduli,
    obrien_edge_delamination,
    parse_ply_stack,
)
from porosity_fe.fe import failure
from porosity_fe.homogenization import _clt_abd_plies
from porosity_fe.ply_stack import element_materials, material_key, stack_angles

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")

CARBON = MATERIALS['T700_epoxy']
GLASS = MATERIALS['glass_epoxy']
# AS4/3501-6 has CTEs; a glass/epoxy with typical ones for the thermal checks.
AS4 = MATERIALS['AS4_3501_6_epoxy']
GLASS_CTE = dataclasses.replace(GLASS, alpha_1=6.0e-6, alpha_2=22e-6)


def _solve(material, loading='tension', *, Vp=0.0, angles='QI', nx=6, ny=3, nz=None,
           distribution='uniform', criterion='tsai_wu', **kwargs):
    pf = PorosityField(material, Vp, distribution=distribution)
    n = material.n_plies
    mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=n if nz is None else nz,
                         ply_angles=angles)
    solver = FESolver(mesh, material, pf, failure_criterion=criterion)
    return solver.solve(loading, **kwargs), mesh, solver


# ----------------------------------------------------------------------
# The stack
# ----------------------------------------------------------------------

class TestPlyStack:
    def test_materials_and_thicknesses(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS])
        assert stack.n_plies == len(stack) == 4
        assert stack.materials == (GLASS, CARBON)
        assert stack.ply_material_ids.tolist() == [0, 1, 1, 0]
        assert stack.ply_materials == (GLASS, CARBON, CARBON, GLASS)
        t = [GLASS.t_ply, CARBON.t_ply, CARBON.t_ply, GLASS.t_ply]
        np.testing.assert_allclose(stack.thicknesses, t)
        assert stack.total_thickness == pytest.approx(sum(t))
        np.testing.assert_allclose(stack.ply_bounds, np.concatenate(([0.0], np.cumsum(t))))
        np.testing.assert_allclose(stack.interface_z, np.cumsum(t)[:-1])
        assert stack.is_hybrid and not stack.uniform_thickness

    def test_one_thickness_for_every_ply(self):
        stack = PlyStack([CARBON] * 3, thicknesses=0.2)
        assert stack.uniform_thickness and not stack.is_hybrid
        np.testing.assert_allclose(stack.thicknesses, [0.2] * 3)

    def test_equal_materials_share_an_index(self):
        twin = dataclasses.replace(CARBON)
        stack = PlyStack([CARBON, twin, GLASS])
        assert stack.ply_material_ids.tolist() == [0, 0, 1]

    def test_reference(self):
        stack = PlyStack([GLASS, CARBON, GLASS], thicknesses=[0.3, 0.1, 0.3])
        ref = stack.reference
        assert ref.n_plies == 3 and ref.t_ply == pytest.approx(0.7 / 3)
        assert ref.E11 == GLASS.E11
        assert ref.total_thickness == pytest.approx(stack.total_thickness)

    def test_interface_materials(self):
        stack = PlyStack([GLASS, CARBON, CARBON])
        assert stack.interface_materials(0) == (GLASS, CARBON)
        assert stack.interface_materials(1) == (CARBON, CARBON)
        with pytest.raises(ValueError, match="interface must be in"):
            stack.interface_materials(2)

    def test_arrays_are_copies(self):
        stack = PlyStack([CARBON, GLASS])
        stack.thicknesses[0] = 99.0
        stack.ply_material_ids[0] = 5
        assert stack.thicknesses[0] == CARBON.t_ply
        assert stack.ply_material_ids[0] == 0

    def test_equality_hash_and_key(self):
        a = PlyStack([CARBON, GLASS], thicknesses=0.2)
        b = PlyStack([CARBON, GLASS], thicknesses=[0.2, 0.2])
        c = PlyStack([GLASS, CARBON], thicknesses=0.2)
        assert a == b and hash(a) == hash(b) and a != c
        assert a.__eq__(CARBON) is NotImplemented
        assert material_key(a) != material_key(c)
        assert material_key(CARBON) == dataclasses.astuple(CARBON)

    def test_spec_and_repr(self):
        custom = dataclasses.replace(CARBON, E11=1.0e5)
        stack = PlyStack([GLASS, CARBON, CARBON, custom], thicknesses=[0.2, 0.125, 0.125, 0.1])
        assert stack.spec == "glass_epoxy@0.2, T700_epoxy*2@0.125, m2@0.1"
        assert repr(stack) == "PlyStack(4 plies, 0.55 mm: " + stack.spec + ")"
        # The spec of a preset stack reads back to the same stack.
        presets = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=[0.2, 0.1, 0.1, 0.2])
        assert parse_ply_stack(presets.spec, MATERIALS) == presets

    def test_to_dict(self):
        custom = dataclasses.replace(CARBON, E11=1.0e5)
        d = PlyStack([GLASS, custom], thicknesses=0.2).to_dict(MATERIALS)
        assert d['n_plies'] == 2 and d['total_thickness_mm'] == pytest.approx(0.4)
        assert d['thickness_mm'] == [0.2, 0.2] and d['material_index'] == [0, 1]
        assert [m['name'] for m in d['materials']] == ['glass_epoxy', None]
        assert d['materials'][1]['E11'] == 1.0e5
        json.dumps(d)
        assert PlyStack([GLASS]).to_dict()['materials'][0]['name'] is None

    @pytest.mark.parametrize("args, error, match", [
        (([],), ValueError, "at least one ply"),
        ((CARBON,), TypeError, "sequence of MaterialProperties"),
        (("T700_epoxy",), TypeError, "sequence of MaterialProperties"),
        (([CARBON, "glass"],), TypeError, "ply 1 is a str"),
        (([CARBON], 0.0), ValueError, "finite thickness > 0"),
        (([CARBON], float('nan')), ValueError, "finite thickness > 0"),
        (([CARBON], True), TypeError, "must be a number"),
        (([CARBON, GLASS], [0.1]), ValueError, "1 values for 2 plies"),
        (([CARBON, GLASS], [0.1, "x"]), TypeError, r"thicknesses\[1\] must be a number"),
        (([CARBON, GLASS], [0.1, -1.0]), ValueError, r"thicknesses\[1\]"),
    ])
    def test_rejects(self, args, error, match):
        with pytest.raises(error, match=match):
            PlyStack(*args)

    def test_numpy_thicknesses(self):
        stack = PlyStack([CARBON, GLASS], thicknesses=np.array([0.1, 0.3]))
        np.testing.assert_allclose(stack.thicknesses, [0.1, 0.3])


class TestParsePlyStack:
    def test_blocks(self):
        stack = parse_ply_stack(" glass_epoxy@0.2 , T700_epoxy*3, glass_epoxy @ 2e-1", MATERIALS)
        assert stack.ply_material_ids.tolist() == [0, 1, 1, 1, 0]
        np.testing.assert_allclose(stack.thicknesses, [0.2, CARBON.t_ply, CARBON.t_ply,
                                                       CARBON.t_ply, 0.2])

    def test_custom_names(self):
        mine = dataclasses.replace(CARBON, E11=1.0e5)
        stack = parse_ply_stack("mine*2", {'mine': mine})
        assert stack.materials == (mine,)

    @pytest.mark.parametrize("spec, match", [
        ("", "empty"),
        ("   ", "empty"),
        ("T700_epoxy*", "Cannot read the ply block"),
        ("T700_epoxy@abc", "Cannot read the ply block"),
        ("T700_epoxy*0", "must be >= 1"),
        ("unobtainium*2", "Unknown material 'unobtainium'"),
    ])
    def test_rejects(self, spec, match):
        with pytest.raises(ValueError, match=match):
            parse_ply_stack(spec, MATERIALS)

    def test_rejects_a_non_string(self):
        with pytest.raises(ValueError, match="empty"):
            parse_ply_stack(None, MATERIALS)


class TestStackAngles:
    def test_tiles_like_the_mesh(self):
        stack = PlyStack([CARBON] * 6)
        assert stack_angles(stack, [0, 90]) == [0.0, 90.0] * 3
        assert stack_angles(stack, 'QI') == [0.0, 90.0, 45.0, -45.0, -45.0, 45.0]
        assert stack_angles(PlyStack([CARBON] * 2), 'QI') == [0.0, 90.0]

    def test_refuses_a_longer_layup(self):
        with pytest.raises(ValueError, match="layup has 3 plies but the stack has 2"):
            stack_angles(PlyStack([CARBON] * 2), [0, 90, 0])


# ----------------------------------------------------------------------
# Field and mesh
# ----------------------------------------------------------------------

class TestFieldAndMesh:
    def test_interface_voids_sit_at_the_stack_interfaces(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=[0.3, 0.1, 0.1, 0.3])
        pf = PorosityField(stack, 0.03, distribution='interface')
        assert pf.ply_stack is stack and pf.Lz == pytest.approx(0.8)
        z = np.linspace(0.0, 0.8, 8001)
        profile = pf._distributed_porosity(z)
        assert np.mean(profile) == pytest.approx(0.03, rel=2e-3)
        # A Gaussian at each interface, sigma 0.35 times the thinner ply.
        sigmas = [0.035, 0.035, 0.035]
        shape = sum(np.exp(-0.5 * ((z - zi) / s) ** 2)
                    for zi, s in zip(stack.interface_z, sigmas, strict=True))
        np.testing.assert_allclose(profile / profile.max(), shape / shape.max(), rtol=1e-12,
                                   atol=1e-15)

    def test_node_layers_follow_the_plies(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=[0.3, 0.1, 0.1, 0.3])
        pf = PorosityField(stack, 0.0)
        mesh = CompositeMesh(pf, stack, nx=2, ny=1, nz=8, ply_angles=[0, 90, 90, 0])
        z = np.unique(mesh.nodes[:, 2])
        np.testing.assert_allclose(z, [0, 0.15, 0.3, 0.35, 0.4, 0.45, 0.5, 0.65, 0.8],
                                   atol=1e-15)
        assert mesh.L_z == pytest.approx(0.8) and mesh.ply_stack is stack
        assert mesh.material == stack.reference
        layers = mesh.elem_material_ids.reshape(8, -1)[:, 0]
        assert layers.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
        assert mesh.element_layup.tolist() == [0, 0, 90, 90, 90, 90, 0, 0]

    def test_a_layer_may_merge_plies_of_one_material(self):
        stack = PlyStack([GLASS, CARBON, CARBON, CARBON, CARBON, GLASS])
        pf = PorosityField(stack, 0.0)
        # nz = 3 merges plies 0-1? No: layer 0 spans plies 0 and 1 (glass, carbon).
        with pytest.raises(ValueError, match="element layer 0 spans plies 0 to 1"):
            CompositeMesh(pf, stack, nx=2, ny=1, nz=3)
        stack2 = PlyStack([GLASS, GLASS, CARBON, CARBON])
        mesh = CompositeMesh(PorosityField(stack2, 0.0), stack2, nx=2, ny=1, nz=2,
                             ply_angles='UD')
        assert mesh.elem_material_ids.reshape(2, -1)[:, 0].tolist() == [0, 1]

    def test_the_field_must_be_the_mesh_stack(self):
        stack = PlyStack([GLASS, CARBON])
        other = PlyStack([CARBON, GLASS])
        with pytest.raises(ValueError, match="same PlyStack"):
            CompositeMesh(PorosityField(other, 0.0), stack, nx=2, ny=1, nz=2)
        with pytest.raises(ValueError, match="same PlyStack"):
            CompositeMesh(PorosityField(CARBON, 0.0), stack, nx=2, ny=1, nz=2)
        with pytest.raises(ValueError, match="built from a PlyStack"):
            CompositeMesh(PorosityField(stack, 0.0), CARBON, nx=2, ny=1, nz=2)

    def test_element_materials(self):
        stack = PlyStack([GLASS, CARBON])
        mesh = CompositeMesh(PorosityField(stack, 0.0), stack, nx=2, ny=1, nz=2)
        mats, ids = element_materials(mesh, stack)
        assert mats == (GLASS, CARBON) and ids.tolist() == [0, 0, 1, 1]
        assert element_materials(mesh, stack.reference)[0] == (GLASS, CARBON)
        with pytest.raises(ValueError, match="differs from the mesh's"):
            element_materials(mesh, PlyStack([CARBON, GLASS]))
        with pytest.raises(ValueError, match="not a single material"):
            element_materials(mesh, CARBON)
        plain = CompositeMesh(PorosityField(CARBON, 0.0), CARBON, nx=2, ny=1, nz=2)
        with pytest.raises(ValueError, match="built from a single material"):
            element_materials(plain, stack)
        with pytest.raises(ValueError, match="built from a single material"):
            FESolver(plain, stack, plain.porosity_field)

    def test_specimen_builders_take_a_stack(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.5)
        pf = PorosityField(stack, 0.0)
        ilss = CompositeMesh.for_ilss(pf, stack)
        assert ilss.ply_stack is stack and ilss.L_z == pytest.approx(2.0)
        flex = CompositeMesh.for_flexure(pf, stack)
        assert flex.ply_stack is stack
        dcb = CompositeMesh.for_dcb(pf, stack)
        assert dcb.ply_stack is stack
        with pytest.raises(ValueError, match="for a PlyStack give lever_length"):
            CompositeMesh.for_mmb(pf, stack)
        mmb = CompositeMesh.for_mmb(pf, stack, lever_length=40.0)
        assert mmb.ply_stack is stack

    def test_hertz_width_reads_the_top_ply(self):
        from porosity_fe.mesh import _hertz_contact_width
        stack = PlyStack([GLASS, CARBON], thicknesses=1.0)
        top = dataclasses.replace(CARBON, n_plies=2, t_ply=1.0)
        assert _hertz_contact_width(stack) == pytest.approx(_hertz_contact_width(top))


# ----------------------------------------------------------------------
# FE: a stack of identical plies is the single material
# ----------------------------------------------------------------------

class TestIdenticalPliesReproduceTheMaterial:
    MAT = dataclasses.replace(CARBON, n_plies=8)

    @pytest.mark.parametrize("loading, distribution, criterion", [
        ('compression', 'interface', 'tsai_wu'),
        ('tension', 'clustered', 'hashin'),
        ('shear', 'uniform', 'max_stress'),
    ])
    def test_mechanical(self, loading, distribution, criterion):
        stack = PlyStack([self.MAT] * 8)
        a, _, _ = _solve(self.MAT, loading, Vp=0.03, distribution=distribution,
                         criterion=criterion)
        b, _, _ = _solve(stack, loading, Vp=0.03, distribution=distribution,
                         criterion=criterion)
        for name in ('effective_modulus', 'knockdown', 'first_ply_failure_load_factor',
                     'max_failure_index'):
            assert getattr(b, name) == pytest.approx(getattr(a, name), rel=1e-12), name
        np.testing.assert_allclose(b.stress_global, a.stress_global, rtol=1e-12,
                                   atol=1e-12 * np.abs(a.stress_global).max())

    def test_thermal(self):
        mat = dataclasses.replace(AS4, n_plies=4)
        a, _, _ = _solve(mat, 'thermal', angles=[0, 90, 90, 0], delta_T=-120.0)
        b, _, _ = _solve(PlyStack([mat] * 4), 'thermal', angles=[0, 90, 90, 0],
                         delta_T=-120.0)
        np.testing.assert_allclose(b.stress_local, a.stress_local, rtol=1e-12,
                                   atol=1e-10)
        assert b.cte_local == a.cte_local
        assert b.first_ply_failure_load_factor == pytest.approx(
            a.first_ply_failure_load_factor, rel=1e-12)


# ----------------------------------------------------------------------
# FE: hybrid laminates against lamination theory
# ----------------------------------------------------------------------

class TestHybridAgainstLaminationTheory:
    @pytest.mark.parametrize("Vp", [0.0, 0.03])
    def test_unidirectional_carbon_glass(self, Vp):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=[0.3, 0.125, 0.125, 0.3])
        res, _, _ = _solve(stack, 'tension', Vp=Vp, angles='UD', nx=10, ny=4)
        clt = compute_degraded_clt_moduli(stack, [0], Vp)['Ex']
        assert res.effective_modulus == pytest.approx(clt, rel=1e-6)

    def test_thick_and_thin_cross_ply(self):
        stack = PlyStack([CARBON] * 4, thicknesses=[0.25, 0.1, 0.1, 0.25])
        res, _, _ = _solve(stack, 'tension', Vp=0.03, angles=[0, 90, 90, 0], nx=12, ny=6,
                           nz=8)
        clt = compute_degraded_clt_moduli(stack, [0, 90, 90, 0], 0.03)['Ex']
        assert res.effective_modulus == pytest.approx(clt, rel=2e-4)

    def test_pristine_modulus(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=[0.3, 0.125, 0.125, 0.3])
        res, _, _ = _solve(stack, 'tension', angles='UD', nx=10, ny=4)
        assert compute_clt_effective_modulus(stack, [0]) == pytest.approx(
            res.effective_modulus, rel=1e-6)

    def test_iso_strain_stress_split(self):
        """Under tension along the fibres every ply takes the same strain, so
        the fibre stress of carbon over glass is about E11c / E11g."""
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.2)
        res, mesh, _ = _solve(stack, 'tension', angles='UD', nx=10, ny=4)
        s11 = res.stress_local[:, :, 0].mean(axis=1)
        ids = mesh.elem_material_ids
        ratio = s11[ids == 1].mean() / s11[ids == 0].mean()
        assert ratio == pytest.approx(CARBON.E11 / GLASS.E11, rel=0.01)

    def test_stack_clt_of_identical_plies_is_the_material(self):
        mat = dataclasses.replace(CARBON, n_plies=8)
        stack = PlyStack([mat] * 8)
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        for Vp in (0.0, 0.04):
            a = compute_degraded_clt_moduli(mat, angles, Vp)
            b = compute_degraded_clt_moduli(stack, angles, Vp)
            for k in a:
                assert b[k] == pytest.approx(a[k], rel=1e-12)
        assert compute_degraded_clt_flexural_modulus(stack, angles, 0.02)['Ef_x'] == \
            pytest.approx(compute_degraded_clt_flexural_modulus(mat, angles, 0.02)['Ef_x'],
                          rel=1e-12)
        assert compute_clt_effective_modulus(stack, angles) == pytest.approx(
            compute_clt_effective_modulus(mat, angles), rel=1e-12)

    def test_unsymmetric_stack_couples(self):
        C = [m.get_stiffness_matrix() for m in (GLASS, CARBON)]
        _, B, _ = _clt_abd_plies(C, [0, 0], [0.2, 0.2])
        assert np.abs(B).max() > 0.0
        _, B_sym, _ = _clt_abd_plies([C[0], C[1], C[0]], [0, 0, 0], [0.2, 0.4, 0.2])
        np.testing.assert_allclose(B_sym, 0.0, atol=1e-9 * np.abs(B).max())


# ----------------------------------------------------------------------
# FE: failure, thermal and assembly per material
# ----------------------------------------------------------------------

class TestPerMaterial:
    def test_each_element_is_checked_against_its_own_strengths(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.2)
        res, mesh, solver = _solve(stack, 'tension', Vp=0.02, angles='UD', nx=6, ny=3,
                                   criterion='hashin')
        radii = solver.porosity_field.void_shape_radii
        ids = mesh.elem_material_ids
        for i, mat in enumerate(stack.materials):
            others = np.flatnonzero(ids != i)
            _, fi, _ = failure.evaluate_failure(
                res.stress_local, mesh.porosity, mesh.elements, mat, radii, 'hashin',
                others)
            np.testing.assert_allclose(res.per_element_failure_index[ids == i],
                                       fi[ids == i], rtol=1e-12)
        # Carbon has the smaller strain to failure along the fibres.
        assert CARBON.sigma_1t / CARBON.E11 < GLASS.sigma_1t / GLASS.E11
        worst = int(np.argmax(res.per_element_failure_index))
        assert ids[worst] == 1
        modes = res.per_element_mode_failure_index
        assert modes is not None and set(modes) >= {'fiber_t', 'matrix_t'}
        assert modes['fiber_t'][worst] == pytest.approx(res.max_failure_index)

    def test_load_factor_is_the_smallest_over_materials(self):
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.2)
        res, mesh, solver = _solve(stack, 'compression', Vp=0.02, angles=[0, 90, 90, 0])
        radii = solver.porosity_field.void_shape_radii
        lam = failure.first_ply_failure_load_factor(
            res.stress_local, mesh.porosity, mesh.elements, stack, radii,
            element_material_ids=mesh.elem_material_ids)
        assert lam == pytest.approx(res.first_ply_failure_load_factor, rel=1e-12)
        per = [failure.first_ply_failure_load_factor(
            res.stress_local, mesh.porosity, mesh.elements, m, radii,
            void_elements=np.flatnonzero(mesh.elem_material_ids != i))
            for i, m in enumerate(stack.materials)]
        assert lam == pytest.approx(min(per), rel=1e-12)

    def test_failure_needs_element_materials(self):
        stack = PlyStack([GLASS, CARBON])
        stress = np.zeros((2, 8, 6))
        with pytest.raises(ValueError, match="needs element_material_ids"):
            failure.evaluate_failure(stress, np.zeros(12), np.zeros((2, 8), dtype=int),
                                     stack, (1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="1 entries for 2 elements"):
            failure.evaluate_failure(stress, np.zeros(12), np.zeros((2, 8), dtype=int),
                                     stack, (1.0, 1.0, 1.0), element_material_ids=[0])

    def test_a_single_material_stack_takes_the_plain_path(self):
        stack = PlyStack([CARBON] * 2)
        stress = np.zeros((2, 8, 6))
        stress[:, :, 0] = 100.0
        elements = np.zeros((2, 8), dtype=int)
        a = failure.evaluate_failure(stress, np.zeros(12), elements, CARBON, (1, 1, 1))
        b = failure.evaluate_failure(stress, np.zeros(12), elements, stack, (1, 1, 1),
                                     element_material_ids=[0, 0])
        assert a[0] == b[0]
        lam = failure._element_load_factors(stress, np.zeros(12), elements, stack, (1, 1, 1),
                                            element_material_ids=[0, 0])
        np.testing.assert_array_equal(lam, failure._element_load_factors(
            stress, np.zeros(12), elements, CARBON, (1, 1, 1)))

    def test_reference_elements_match_the_batch(self):
        stack = PlyStack([GLASS, CARBON], thicknesses=0.3)
        pf = PorosityField(stack, 0.02)
        mesh = CompositeMesh(pf, stack, nx=2, ny=1, nz=2, ply_angles=[0, 45])
        solver = FESolver(mesh, stack, pf)
        Ke = solver.assembler.element_batch().stiffness_matrices()
        for e in (0, len(mesh.elements) - 1):
            ref = solver.assembler.create_element(e).stiffness_matrix()
            np.testing.assert_allclose(ref, Ke[e], rtol=1e-9, atol=1e-9 * np.abs(Ke[e]).max())

    def test_thermal_residual_stress_matches_lamination_theory(self):
        stack = PlyStack([GLASS_CTE, AS4, AS4, GLASS_CTE], thicknesses=[0.2, 0.15, 0.15, 0.2])
        dT = -150.0
        res, mesh, _ = _solve(stack, 'thermal', angles='UD', nx=20, ny=10, nz=8,
                              delta_T=dT)
        idx = [0, 1, 5]

        def q(m):
            C = m.get_stiffness_matrix()
            return C[np.ix_(idx, idx)] - np.outer(C[idx, 2], C[idx, 2]) / C[2, 2]

        plies = stack.ply_materials
        A, _, _ = _clt_abd_plies([m.get_stiffness_matrix() for m in plies], [0] * 4,
                                 stack.thicknesses)
        N_T = sum(q(m) @ m.cte_vector()[idx] * t for m, t in zip(plies, stack.thicknesses,
                                                                        strict=True))
        eps0 = np.linalg.solve(A, N_T * dT)
        centre = mesh.nodes[mesh.elements].mean(axis=1)
        inner = (abs(centre[:, 0] - 25.0) < 5.0) & (abs(centre[:, 1] - 10.0) < 3.0)
        s = res.stress_local.mean(axis=1)
        for k, m in enumerate(plies):
            expected = q(m) @ (eps0 - m.cte_vector()[idx] * dT)
            sel = inner & (mesh.elem_ply_ids == k)
            assert s[sel, 0].mean() == pytest.approx(expected[0], rel=5e-3)
            assert s[sel, 1].mean() == pytest.approx(expected[1], rel=3e-2)
        assert res.cte_local is None              # the two materials' CTEs differ

    def test_thermal_needs_every_cte(self):
        stack = PlyStack([GLASS, AS4])
        pf = PorosityField(stack, 0.0)
        mesh = CompositeMesh(pf, stack, nx=2, ny=1, nz=2, ply_angles='UD')
        with pytest.raises(ValueError, match="or a material of the PlyStack"):
            FESolver(mesh, stack, pf).solve('thermal', delta_T=-100.0)

    def test_the_pristine_reference_is_keyed_on_the_stack(self):
        a = PlyStack([GLASS, CARBON], thicknesses=0.2)
        b = PlyStack([CARBON, GLASS], thicknesses=0.2)
        ra, _, _ = _solve(a, 'compression', Vp=0.03, angles='UD', nx=4, ny=2)
        rb, _, _ = _solve(b, 'compression', Vp=0.03, angles='UD', nx=4, ny=2)
        assert ra.knockdown < 1.0 and rb.knockdown < 1.0
        # Same geometry, swapped materials: the references differ, so the
        # knockdowns are each their own laminate's.
        ra2, _, _ = _solve(a, 'compression', Vp=0.03, angles='UD', nx=4, ny=2)
        assert ra2.knockdown == ra.knockdown

    def test_single_material_helpers_refuse_a_hybrid(self):
        stack = PlyStack([GLASS, CARBON])
        _, _, solver = _solve(stack, 'tension', angles='UD', nx=2, ny=1)
        with pytest.raises(ValueError, match="reads one material"):
            solver._degraded_strengths(0.0)
        with pytest.raises(ValueError, match="reads one material"):
            solver._evaluate_tsai_wu(np.zeros((8, 6)), (1,) * 6, 0, 0.0)
        fi = solver._evaluate_failure(np.zeros((len(solver.mesh.elements), 8, 6)))
        assert fi[0] == 0.0

    def test_export_and_vtu(self, tmp_path):
        import jsonschema
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.2)
        res, mesh, _ = _solve(stack, 'tension', Vp=0.02, angles='UD', nx=4, ny=2)
        path = tmp_path / 'hybrid.json'
        FESolver.export_results(res, path, mesh=mesh)
        doc = json.loads(path.read_text())
        block = doc['ply_stack']
        assert block['material_index'] == [0, 1, 1, 0]
        assert [m['name'] for m in block['materials']] == ['glass_epoxy', 'T700_epoxy']
        with open('validation/schemas/porosity_results_schema.json') as fh:
            jsonschema.validate(instance=doc, schema=json.load(fh))
        FESolver.export_results(res, tmp_path / 'plain.json')
        assert 'ply_stack' not in json.loads((tmp_path / 'plain.json').read_text())
        vtu = tmp_path / 'hybrid.vtu'
        res.to_vtu(mesh, vtu)
        assert b'Name="material_id"' in vtu.read_bytes()


# ----------------------------------------------------------------------
# Fracture at a mixed interface: the weaker side governs
# ----------------------------------------------------------------------

TOUGH = MATERIALS['IM7_8552_epoxy']
TOUGHER = dataclasses.replace(TOUGH, G_Ic=2.0 * TOUGH.G_Ic, G_IIc=2.0 * TOUGH.G_IIc)


def _dcb(material):
    pf = PorosityField(material, 0.02)
    mesh = CompositeMesh.for_dcb(pf, material, nx=40, ny=2)
    return FESolver(mesh, material, pf).solve('dcb'), mesh


class TestMixedInterface:
    @pytest.fixture(scope='class')
    @staticmethod
    def plain():
        mat = dataclasses.replace(TOUGH, n_plies=8)
        return {'weak': _dcb(mat)[0],
                'strong': _dcb(dataclasses.replace(TOUGHER, n_plies=8))[0]}

    @pytest.mark.parametrize("lower, upper, governs", [
        (TOUGH, TOUGHER, 'weak'), (TOUGHER, TOUGH, 'weak'), (TOUGHER, TOUGHER, 'strong')])
    def test_the_weaker_side_governs(self, plain, lower, upper, governs):
        # Same stiffness on both sides, so only the toughness differs.
        stack = PlyStack([lower] * 4 + [upper] * 4)
        res, _ = _dcb(stack)
        # Equal up to the round-off of the node heights (the stack's ply map).
        assert res.fracture_load_factor == pytest.approx(
            plain[governs].fracture_load_factor, rel=1e-7)
        np.testing.assert_allclose(res.crack_front.G_c, plain[governs].crack_front.G_c,
                                   rtol=1e-7)

    def test_a_material_without_toughness(self):
        stack = PlyStack([TOUGH] * 4 + [GLASS] * 4, thicknesses=0.18)
        res, mesh = _dcb(stack)
        assert res.crack_front is not None and res.fracture_load_factor is None
        solver = FESolver(mesh, stack, mesh.porosity_field)
        with pytest.raises(ValueError, match="every ply material at the delaminated"):
            solver.delamination_growth('dcb', max_steps=1)

    def test_with_a_prestress(self):
        tough = dataclasses.replace(TOUGH, alpha_1=-0.5e-6, alpha_2=30e-6, n_plies=8)
        tougher = dataclasses.replace(tough, G_Ic=2.0 * tough.G_Ic, G_IIc=2.0 * tough.G_IIc)
        stack = PlyStack([tough] * 4 + [tougher] * 4)
        pf = PorosityField(stack, 0.0)
        mesh = CompositeMesh.for_dcb(pf, stack, nx=40, ny=2, ply_angles=[0, 90, 0, 90,
                                                                          90, 0, 90, 0])
        res = FESolver(mesh, stack, pf).solve('dcb', delta_T=-80.0)
        pf0 = PorosityField(tough, 0.0)
        mesh0 = CompositeMesh.for_dcb(pf0, tough, nx=40, ny=2, ply_angles=[0, 90, 0, 90,
                                                                            90, 0, 90, 0])
        ref = FESolver(mesh0, tough, pf0).solve('dcb', delta_T=-80.0)
        assert res.fracture_load_factor == pytest.approx(ref.fracture_load_factor, rel=1e-7)

    def test_the_uncertainty_band_takes_one_material(self, plain):
        stack = PlyStack([TOUGH] * 4 + [TOUGHER] * 4)
        res, _ = _dcb(stack)
        with pytest.raises(ValueError, match="not the PlyStack"):
            res.fracture_load_factor_uncertainty(stack, covs={'G_Ic': 0.1})

    def test_growth_on_a_hybrid(self):
        stack = PlyStack([TOUGH] * 4 + [TOUGHER] * 4)
        pf = PorosityField(stack, 0.0)
        mesh = CompositeMesh.for_dcb(pf, stack, nx=40, ny=2)
        growth = FESolver(mesh, stack, pf).delamination_growth('dcb', max_steps=2)
        assert growth.n_steps >= 1 and np.isfinite(growth.load_factors[0])

    def test_onset_takes_the_weaker_side(self):
        base = dataclasses.replace(CARBON, sigma_3t=50.0, tau_13=80.0)
        strong = dataclasses.replace(base, sigma_3t=100.0, tau_13=160.0)

        def onset(material):
            pf = PorosityField(material, 0.02)
            mesh = CompositeMesh(pf, material, nx=8, ny=4, nz=4, ply_angles=[0, 90, 90, 0])
            res = FESolver(mesh, material, pf).solve('tension')
            return res.delamination_onset(mesh, material, x_avg=0.0)

        weak = onset(dataclasses.replace(base, n_plies=4))
        both = onset(PlyStack([strong] * 4))
        for stack, mixed_at, weak_at in (
                (PlyStack([base, base, strong, strong]), 1, (0, 1)),
                (PlyStack([strong, strong, base, base]), 1, (1, 2))):
            got = onset(stack)
            for k in range(3):
                ref = weak if k in weak_at else both
                sel = got.interface == k
                np.testing.assert_allclose(got.index[sel], ref.index[sel], rtol=1e-9)
                np.testing.assert_allclose(got.Zt[sel], ref.Zt[sel], rtol=1e-12)
            # At the mixed interface the weak side's strengths are reported.
            assert np.all(got.Zt[got.interface == mixed_at] == 50.0 * (
                weak.Zt[weak.interface == mixed_at] / 50.0))
        # S_23 keeps its tau_12 stand-in in both, so the stronger one is
        # lower or level, and lower somewhere.
        sel = weak.evaluated & (weak.index > 0)
        assert np.all(both.index[sel] <= weak.index[sel] * (1 + 1e-12))
        assert np.any(both.index[sel] < 0.99 * weak.index[sel])


# ----------------------------------------------------------------------
# Empirical, O'Brien, open hole, refinement, FE uncertainty
# ----------------------------------------------------------------------

class TestOtherPaths:
    def test_empirical_weights_the_plies_by_thickness(self):
        mat = dataclasses.replace(CARBON, n_plies=4)
        angles = [0, 90, 90, 0]

        def scales(stack):
            pf = PorosityField(stack, 0.03)
            mesh = CompositeMesh(pf, stack, nx=2, ny=1, nz=4, ply_angles=angles)
            return EmpiricalSolver(mesh, stack, ply_angles=angles).layup_scale

        plain = EmpiricalSolver(
            CompositeMesh(PorosityField(mat, 0.03), mat, nx=2, ny=1, nz=4, ply_angles=angles),
            mat, ply_angles=angles).layup_scale
        assert scales(PlyStack([mat] * 4)) == plain
        # Thick 90-degree plies: matrix-dominated, a larger tension scale.
        thick_90 = scales(PlyStack([mat] * 4, thicknesses=[0.05, 0.5, 0.5, 0.05]))
        assert thick_90['tension'] > max(plain['tension'], 1.0)

    def test_empirical_refuses_a_hybrid(self):
        stack = PlyStack([GLASS, CARBON])
        mesh = CompositeMesh(PorosityField(stack, 0.03), stack, nx=2, ny=1, nz=2)
        with pytest.raises(ValueError, match="Use FESolver for a hybrid laminate"):
            EmpiricalSolver(mesh, stack)
        with pytest.raises(ValueError, match="Use FESolver"):
            EmpiricalSolver(mesh, stack.reference)
        one = PlyStack([CARBON] * 2)
        mesh1 = CompositeMesh(PorosityField(one, 0.03), one, nx=2, ny=1, nz=2)
        with pytest.raises(ValueError, match="differs from the mesh's"):
            EmpiricalSolver(mesh1, PlyStack([CARBON] * 2, thicknesses=0.3))

    def test_obrien(self):
        mat = dataclasses.replace(CARBON, n_plies=8)
        angles = [30, -30, 90, 90, 90, 90, -30, 30]
        a = obrien_edge_delamination(mat, angles, [1, 5], 0.004)
        b = obrien_edge_delamination(PlyStack([mat] * 8), angles, [1, 5], 0.004)
        assert b.G == pytest.approx(a.G, rel=1e-12)
        assert b.E_star == pytest.approx(a.E_star, rel=1e-12)
        hybrid = obrien_edge_delamination(PlyStack([GLASS] + [mat] * 6 + [GLASS]),
                                          angles, [1, 5], 0.004, bending_coupling=True)
        assert hybrid.thickness == pytest.approx(2 * GLASS.t_ply + 6 * mat.t_ply)
        assert hybrid.G > 0.0
        with pytest.raises(ValueError, match="t_ply is not taken"):
            obrien_edge_delamination(PlyStack([mat] * 8), angles, [1], 0.004, t_ply=0.1)
        with pytest.raises(ValueError, match="give one angle per ply"):
            obrien_edge_delamination(PlyStack([mat] * 4), angles, [1], 0.004)

    def test_open_hole(self):
        from porosity_fe import open_hole_analysis
        from porosity_fe.open_hole import _laminate_constants
        mat = dataclasses.replace(CARBON, n_plies=4)
        layup = np.array([0.0, 90.0, 90.0, 0.0])
        np.testing.assert_allclose(_laminate_constants(PlyStack([mat] * 4), layup),
                                   _laminate_constants(mat, layup), rtol=1e-12)
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.2)
        pf = PorosityField(stack, 0.0)
        mesh = CompositeMesh.for_open_hole(pf, stack, ply_angles=[0, 90, 90, 0],
                                           n_circumferential=8, n_radial=6)
        assert mesh.ply_stack is stack
        res = FESolver(mesh, stack, pf).solve('tension')
        notch = open_hole_analysis(res, mesh)
        E_x = notch.laminate_constants['E_x']
        assert E_x == pytest.approx(
            compute_degraded_clt_moduli(stack, [0, 90, 90, 0], 0.0)['Ex'], rel=1e-3)
        assert notch.stress_concentration == pytest.approx(
            notch.reference_stress_concentration, rel=0.15)

    def test_material_cards_take_one_material(self):
        from porosity_fe import material_cards
        stack = PlyStack([GLASS, CARBON])
        with pytest.raises(TypeError, match="write the cards of each of stack.materials"):
            material_cards(stack, 0.02)
        with pytest.raises(TypeError, match="plies of one material"):
            material_cards(CARBON, PorosityField(stack, 0.02))

    def test_partition_thicknesses_match_the_plies(self):
        from porosity_fe._layup import _membrane_energy_partition
        with pytest.raises(ValueError, match="thicknesses has 1 values for 2 plies"):
            _membrane_energy_partition(CARBON, [0, 90], thicknesses=[0.1])

    def test_refinement_study_checks_the_field(self):
        from porosity_fe.fe.refinement import _resolve_porosity
        stack = PlyStack([GLASS, CARBON])
        pf = PorosityField(stack, 0.02)
        assert _resolve_porosity(stack, pf, None) is pf
        with pytest.raises(ValueError, match="different MaterialProperties"):
            _resolve_porosity(PlyStack([CARBON, GLASS]), pf, None)
        assert _resolve_porosity(stack, 0.02, None).ply_stack is stack

    def test_fe_uncertainty(self):
        from porosity_fe import fe_porosity_uncertainty
        stack = PlyStack([GLASS, CARBON, CARBON, GLASS], thicknesses=0.2)
        study = fe_porosity_uncertainty(stack, 0.03, 'tension', cov=0.5,
                                        correlation_length=5.0, n_samples=2, seed=3,
                                        ply_angles='UD', mesh_res=(6, 3, 4))
        assert study.material == repr(stack) and study.n_samples == 2
        assert np.all(np.isfinite(study.samples['knockdown']))
        with pytest.raises(ValueError, match="scatters the porosity only"):
            fe_porosity_uncertainty(stack, 0.03, cov=0.5, correlation_length=5.0,
                                    covs={'E11': 0.05})
