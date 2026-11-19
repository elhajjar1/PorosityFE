"""Linear (eigenvalue) buckling of the FE model."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    BucklingResult,
    CompositeMesh,
    Delamination,
    FESolver,
    PorosityField,
    linear_buckling,
    load_results_from_json,
)
from porosity_fe.fe import buckling as buckling_module
from porosity_fe.fe.buckling import geometric_stiffness

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")

#: 8 plies of 0.125 mm: a 1 mm strip, slender over the 50 mm coupon.
MAT8 = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=8)
CROSS_PLY = [0, 90, 0, 90, 90, 0, 90, 0]


def _solver(Vp=0.0, *, material=MAT8, layup='UD', nx=40, ny=4, nz=8, delaminations=None,
            **field):
    pf = PorosityField(material, Vp, **field)
    mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz, ply_angles=layup,
                         **({'delaminations': delaminations} if delaminations else {}))
    return FESolver(mesh, material, pf, ply_angles=layup)


def _euler_strain(material, h=1.0, L=50.0):
    """Fixed-guided column, pi^2 E I / L^2 over E A, with Timoshenko's shear
    correction (shear correction factor 5/6)."""
    eps = math.pi ** 2 * h ** 2 / (12.0 * L ** 2)
    return eps / (1.0 + material.E11 * eps / (5.0 / 6.0 * material.G13))


@pytest.fixture(scope='module')
def column():
    solver = _solver()
    return solver, linear_buckling(solver, 'compression', n_modes=3)


class TestEulerColumn:
    def test_critical_strain(self, column):
        _, result = column
        assert result.critical_strain == pytest.approx(_euler_strain(MAT8), rel=0.006)
        assert result.load_factor == result.load_factors[0] == pytest.approx(
            result.critical_strain / 0.01)
        assert np.all(np.diff(result.load_factors) > 0.0)

    def test_mode_shape(self, column):
        solver, result = column
        x = solver.mesh.nodes[:, 0]
        w = result.modes[0][:, 2]
        shape = (1.0 - np.cos(np.pi * x / 50.0)) / 2.0
        assert abs(np.corrcoef(w, shape)[0, 1]) > 0.999
        assert np.abs(result.modes).max() == pytest.approx(1.0)
        assert result.modes.shape == (3, solver.mesh.n_nodes, 3)

    def test_converges_from_above(self):
        exact = _euler_strain(MAT8)
        coarse = linear_buckling(_solver(nx=20)).critical_strain / exact
        fine = linear_buckling(_solver(nx=80)).critical_strain / exact
        assert 1.0 < fine < coarse < 1.01

    def test_load_factor_scales_with_the_applied_strain(self, column):
        _, result = column
        half = linear_buckling(_solver(), applied_strain=-0.005)
        assert half.load_factor == pytest.approx(2.0 * result.load_factor, rel=1e-8)
        assert half.critical_strain == pytest.approx(result.critical_strain, rel=1e-8)
        assert half.applied_strain == -0.005

    def test_critical_stress_and_load(self, column):
        solver, result = column
        modulus = result.prebuckling.effective_modulus
        assert result.critical_stress == pytest.approx(result.critical_strain * modulus)
        assert result.critical_load(solver.mesh) == pytest.approx(
            result.critical_stress * 20.0 * 1.0)
        assert result.interpenetration is None

    def test_the_method_is_the_function(self, column):
        _, result = column
        again = _solver().buckling('compression', n_modes=3)
        np.testing.assert_allclose(again.load_factors, result.load_factors, rtol=1e-9)


class TestGeometricStiffness:
    def test_energy_of_a_rotation_under_uniform_stress(self):
        """For u = (0, 0, a x) under a uniform sigma_xx the energy u K_G u is
        sigma_xx a^2 V."""
        mesh = _solver(nx=4, ny=2, nz=2).mesh
        n_elem = len(mesh.elements)
        stress = np.zeros((n_elem, 8, 6))
        stress[..., 0] = -100.0
        K_G = geometric_stiffness(mesh, stress)
        u = np.zeros(mesh.n_dof)
        u[2::3] = 0.01 * mesh.nodes[:, 0]
        volume = 50.0 * 20.0 * 1.0
        assert u @ (K_G @ u) == pytest.approx(-100.0 * 0.01 ** 2 * volume, rel=1e-10)
        # A shear stress couples the two in-plane gradients.
        stress[..., 0] = 0.0
        stress[..., 5] = 50.0
        u[2::3] = 0.01 * mesh.nodes[:, 0] + 0.02 * mesh.nodes[:, 1]
        assert u @ (geometric_stiffness(mesh, stress) @ u) == pytest.approx(
            2 * 50.0 * 0.01 * 0.02 * volume, rel=1e-10)

    def test_symmetric(self):
        mesh = _solver(nx=4, ny=2, nz=2).mesh
        rng = np.random.default_rng(0)
        K_G = geometric_stiffness(mesh, rng.normal(size=(len(mesh.elements), 8, 6)))
        assert abs(K_G - K_G.T).max() < 1e-12 * abs(K_G).max()

    def test_shape_check(self):
        mesh = _solver(nx=4, ny=2, nz=2).mesh
        with pytest.raises(ValueError, match='stress_global must have shape'):
            geometric_stiffness(mesh, np.zeros((3, 8, 6)))


class TestPorosity:
    @pytest.fixture(scope='class')
    def knockdowns(self):
        reference = linear_buckling(_solver(layup=CROSS_PLY))
        out = {}
        for name, field in (('uniform', {}),
                            ('midplane', {'distribution': 'clustered',
                                          'cluster_location': 'midplane'}),
                            ('surface', {'distribution': 'clustered',
                                         'cluster_location': 'surface'})):
            result = linear_buckling(_solver(0.04, layup=CROSS_PLY, **field))
            out[name] = result.knockdown(reference)
        return out

    def test_voids_lower_the_buckling_load(self, knockdowns):
        assert all(0.98 < kd < 1.0 for kd in knockdowns.values())

    def test_near_surface_voids_cost_more(self, knockdowns):
        """Same mean void content: the bending stiffness, and the buckling
        load, lose most to voids near the surfaces."""
        assert knockdowns['surface'] < knockdowns['uniform'] < knockdowns['midplane']

    def test_summary_with_a_reference(self):
        reference = linear_buckling(_solver(layup=CROSS_PLY))
        result = linear_buckling(_solver(0.04, layup=CROSS_PLY))
        record = result.summary(reference)
        assert record['knockdown'] == pytest.approx(
            result.critical_stress / reference.critical_stress)
        assert record['reference'] == reference.summary()
        assert 'reference' not in result.summary()


class TestDelamination:
    @pytest.fixture(scope='class')
    def results(self):
        mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=16)
        intact = linear_buckling(_solver(material=mat, nx=50, nz=16), n_modes=2)
        delaminated = linear_buckling(
            _solver(material=mat, nx=50, nz=16,
                    delaminations=[Delamination.strip(1, 15.0, 35.0)]), n_modes=2)
        return intact, delaminated

    def test_the_sublaminate_buckles_first(self, results):
        intact, delaminated = results
        assert delaminated.critical_strain < 0.5 * intact.critical_strain
        # Two 0.125 mm plies over 20 mm, clamped at both ends of the strip:
        # pi^2 t^2 / (3 a^2); the base laminate is not rigid, so a little less.
        estimate = math.pi ** 2 * 0.25 ** 2 / (3.0 * 20.0 ** 2)
        assert 0.9 * estimate < delaminated.critical_strain < estimate

    def test_the_first_mode_opens(self, results):
        _, delaminated = results
        assert delaminated.interpenetration is not None
        assert delaminated.interpenetration.shape == (2,)
        assert delaminated.interpenetration[0] < 0.02
        assert results[0].interpenetration is None

    def test_orientation_and_measure(self):
        """A mode that closes everywhere is flipped to open; one that opens and
        closes keeps the smaller part as its closing."""
        mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4)
        mesh = _solver(material=mat, nx=10, nz=4, ny=2,
                       delaminations=[Delamination.strip(1, 10.0, 40.0)]).mesh
        orig, dup = mesh.interface_node_pairs()
        open_pairs = ~mesh.bonded_pairs()
        mode = np.zeros((1, mesh.n_nodes, 3))
        mode[0, orig[open_pairs], 2] = 1.0              # the faces close
        oriented, closing = buckling_module._orient_and_measure(mesh, mode)
        assert closing[0] == 0.0
        assert np.all(oriented[0, orig[open_pairs], 2] == -1.0)
        mode[0, dup[open_pairs][:1], 2] = 1.5            # one pair opens more
        _, closing = buckling_module._orient_and_measure(mesh, mode)
        assert closing[0] == pytest.approx(0.5 / 1.5)


class TestShear:
    def test_cross_ply_is_symmetric_in_the_sign(self):
        a = linear_buckling(_solver(layup=CROSS_PLY), 'shear')
        b = linear_buckling(_solver(layup=CROSS_PLY), 'shear', applied_strain=-0.01)
        assert a.load_factor == pytest.approx(b.load_factor, rel=1e-6)
        assert a.critical_stress is not None and a.loading == 'shear'

    def test_angle_ply_is_not(self):
        layup = [45, -45, 0, 90, 90, 0, -45, 45]
        a = linear_buckling(_solver(layup=layup), 'shear')
        b = linear_buckling(_solver(layup=layup), 'shear', applied_strain=-0.01)
        assert a.load_factor != pytest.approx(b.load_factor, rel=1e-3)
        assert a.critical_load(_solver(layup=layup).mesh) is None


class TestRefusals:
    @pytest.mark.parametrize('args, kwargs, match', [
        (('tension',), {}, 'in-plane load cases'),
        (('compression', 0), {}, 'n_modes must be an integer'),
        (('compression', True), {}, 'n_modes must be an integer'),
        (('compression',), {'delta_T': -100.0}, 'delta_T is not supported'),
        (('compression',), {'solver': 'cg'}, 'direct solver'),
    ])
    def test_arguments(self, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            linear_buckling(_solver(nx=8), *args, **kwargs)

    def test_a_load_that_does_not_buckle(self, monkeypatch):
        real = buckling_module.geometric_stiffness
        monkeypatch.setattr(buckling_module, 'geometric_stiffness',
                            lambda mesh, stress: -real(mesh, stress))
        with pytest.raises(ValueError, match='does not buckle'):
            linear_buckling(_solver(nx=8))

    def test_the_mesh_supports_are_reused(self):
        """The eigen-solve takes the cached factorization of the prebuckling
        solve's supports: no second one is built."""
        solver = _solver(nx=8)
        linear_buckling(solver)
        assert len(solver._lu_cache) == 1


class TestExport:
    def test_json_block_validates(self, tmp_path, column):
        import jsonschema
        _, result = column
        reference = result
        path = tmp_path / 'b.json'
        FESolver.export_results(result.prebuckling, path, buckling=result,
                                buckling_reference=reference)
        doc = load_results_from_json(path)
        schema_path = Path(__file__).resolve().parent.parent / 'validation' / 'schemas' / \
            'porosity_results_schema.json'
        jsonschema.validate(instance=doc, schema=json.loads(schema_path.read_text()))
        block = doc['buckling']
        assert block['loading'] == 'compression'
        assert block['load_factors'] == pytest.approx(list(result.load_factors))
        assert block['knockdown'] == pytest.approx(1.0)
        assert block['interpenetration'] is None
        FESolver.export_results(result.prebuckling, path)
        assert 'buckling' not in load_results_from_json(path)

    def test_result_type(self, column):
        assert isinstance(column[1], BucklingResult)
