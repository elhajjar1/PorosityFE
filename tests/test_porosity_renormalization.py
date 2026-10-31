"""The ``interface`` profile is scaled to its requested FE volume average.

Delamination plan item 0.4 (#211). The FE interpolates nodal porosity
trilinearly, so its volume average of a through-thickness profile is the
trapezoid integral over the node layers. The ``interface`` Gaussians are
narrow (``sigma = 0.35 t_ply``); with one element layer per ply the nodes
sit on their peaks and the unscaled average was 18 % above the requested
``Vp``. :class:`CompositeMesh` now scales the sampled distributed part so
the FE average is ``Vp``; point values on an interface plane read the
unscaled field (:meth:`CompositeMesh.field_porosity`).
"""

import logging
import warnings

import numpy as np
import pytest

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    PorosityField,
    VoidGeometry,
)

MAT = MATERIALS['T800_epoxy']          # 24 plies
VP = 0.03


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


def _fe_average(mesh: CompositeMesh) -> float:
    """Volume average of the trilinear interpolant of the nodal porosity:
    on box elements, the element-volume-weighted mean of the corner values."""
    corners = mesh.nodes[mesh.elements]
    volume = np.prod(corners.max(axis=1) - corners.min(axis=1), axis=1)
    mean = mesh.porosity[mesh.elements].mean(axis=1)
    return float((mean * volume).sum() / volume.sum())


def _mesh(pf, nz, **kw):
    return CompositeMesh(pf, MAT, nx=6, ny=3, nz=nz, **kw)


class TestInterfaceRenormalization:
    @pytest.mark.parametrize('nz', [24, 48, 96, 12, 36])
    def test_fe_average_is_the_requested_vp(self, nz):
        pf = PorosityField(MAT, VP, distribution='interface', void_shape='penny')
        mesh = _mesh(pf, nz)
        assert _fe_average(mesh) == pytest.approx(VP, rel=1e-12)

    def test_factor_on_one_layer_per_ply(self):
        """The nodes sit on the peaks: unscaled, the FE average was 0.0354."""
        pf = PorosityField(MAT, VP, distribution='interface')
        mesh = _mesh(pf, MAT.n_plies)
        assert mesh.porosity_renormalization == pytest.approx(0.8483, abs=1e-4)
        assert VP / mesh.porosity_renormalization == pytest.approx(0.0354, abs=1e-4)
        # On a finer mesh the profile is resolved and the factor is ~1.
        assert _mesh(pf, 2 * MAT.n_plies).porosity_renormalization == \
            pytest.approx(1.0, abs=2e-3)

    def test_specimens_are_scaled_too(self):
        pf = PorosityField(MAT, VP, distribution='interface')
        for mesh in (CompositeMesh.for_ilss(pf, MAT), CompositeMesh.for_flexure(pf, MAT)):
            assert _fe_average(mesh) == pytest.approx(VP, rel=1e-9)
            assert mesh.porosity_renormalization < 1.0

    @pytest.mark.parametrize('distribution, location', [
        ('uniform', 'midplane'), ('clustered', 'midplane'),
        ('clustered', 'surface'), ('clustered', 'quarter')])
    def test_other_profiles_are_sampled_as_they_are(self, distribution, location):
        pf = PorosityField(MAT, VP, distribution=distribution, cluster_location=location)
        mesh = _mesh(pf, MAT.n_plies)
        assert mesh.porosity_renormalization == 1.0
        x, y, z = mesh.nodes.T
        np.testing.assert_array_equal(mesh.porosity, pf.local_porosity(x, y, z))
        np.testing.assert_array_equal(mesh.stiffness_reduction, 1.0 - mesh.porosity)
        np.testing.assert_array_equal(mesh.field_porosity(np.arange(mesh.n_nodes)),
                                      mesh.porosity)
        # Resolved: the clustered FE average is within 0.2 % of Vp.
        assert _fe_average(mesh) == pytest.approx(VP, rel=2e-3)

    def test_zero_porosity(self):
        pf = PorosityField(MAT, 0.0, distribution='interface')
        mesh = _mesh(pf, MAT.n_plies)
        assert mesh.porosity_renormalization == 1.0
        assert not np.any(mesh.porosity)

    def test_discrete_voids_are_added_after_the_scaling(self):
        void = VoidGeometry(center=(25.0, 10.0, MAT.total_thickness / 2),
                            radii=(4.0, 4.0, 0.6))
        pf = PorosityField(MAT, VP, distribution='interface', discrete_voids=[void])
        mesh = _mesh(pf, MAT.n_plies)
        x, y, z = mesh.nodes.T
        inside = void.contains(x, y, z)
        assert inside.any() and (~inside).any()
        np.testing.assert_array_equal(mesh.porosity[inside], 1.0)
        scaled = pf._distributed_porosity(z) * mesh.porosity_renormalization
        np.testing.assert_array_equal(mesh.porosity[~inside], scaled[~inside])

    def test_field_porosity_is_the_unscaled_field(self):
        """Point values on a plane read the field itself, about 1.25 Vp on an
        interface, whatever the mesh; zeroed porosity stays zero."""
        pf = PorosityField(MAT, VP, distribution='interface')
        mesh = _mesh(pf, MAT.n_plies)
        ids = np.arange(mesh.n_nodes)
        x, y, z = mesh.nodes.T
        np.testing.assert_allclose(mesh.field_porosity(ids), pf.local_porosity(x, y, z),
                                   rtol=1e-14, atol=0)
        fine = _mesh(pf, 2 * MAT.n_plies)
        on_interface = np.isclose(z, 12 * MAT.t_ply)
        on_fine = np.isclose(fine.nodes[:, 2], 12 * MAT.t_ply)
        assert mesh.field_porosity(ids[on_interface]).max() == pytest.approx(
            fine.field_porosity(np.flatnonzero(on_fine)).max(), rel=1e-12)
        assert mesh.field_porosity(ids[on_interface]).max() > 1.2 * VP
        mesh.porosity = np.zeros_like(mesh.porosity)
        assert not np.any(mesh.field_porosity(ids))


class TestResults:
    """What the scaling changes (FE ``interface`` results) and what it does not."""

    @staticmethod
    def _kd(**kw):
        pf = PorosityField(MAT, VP, **kw)
        mesh = CompositeMesh(pf, MAT, nx=12, ny=4, nz=MAT.n_plies)
        return FESolver(mesh, MAT, pf).solve('compression').knockdown

    def test_interface_penny_is_the_penny_shape(self):
        """With the mean honoured, interface penny voids lose about as much
        stiffness as uniform penny voids (0.9842 against 0.9846); before,
        the 18 % excess porosity gave 0.9819."""
        interface = self._kd(distribution='interface', void_shape='penny')
        uniform = self._kd(void_shape='penny')
        assert interface == pytest.approx(0.9842, abs=2e-4)
        assert uniform == pytest.approx(0.9846, abs=2e-4)
        assert 0.0 < uniform - interface < 1e-3

    def test_empirical_is_unchanged(self):
        pf_i = PorosityField(MAT, VP, distribution='interface')
        pf_u = PorosityField(MAT, VP)
        kd = [EmpiricalSolver(CompositeMesh(pf, MAT, nx=6, ny=3, nz=MAT.n_plies), MAT)
              .get_failure_load('compression').knockdown for pf in (pf_i, pf_u)]
        assert kd[0] == kd[1]
