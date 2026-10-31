"""``FESolver(..., porosity_degradation='directional')`` (issue #212,
delamination plan 1.4): the direction-aware strength ratios and their
threading through the FE path. The stiffness mapping itself is tested in
``tests/test_homogenization.py`` (``TestDirectionalDegradation``).

The default ``'isotropic'`` must stay bit-identical, and spherical voids
must give the same results under either option.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from porosity_fe import MATERIALS, CompositeMesh, PorosityField
from porosity_fe.fe import failure

T800 = MATERIALS['T800_epoxy']
SPHERE = (1.0, 1.0, 1.0)
PENNY = (3.0, 3.0, 0.3)
CYL = (3.0, 1.0, 1.0)


# ----------------------------------------------------------------------
# Strength ratios
# ----------------------------------------------------------------------

class TestStrengthRatios:
    def test_isotropic_is_the_default_and_unchanged(self):
        """``degraded_strengths`` and ``degraded_interlaminar_strengths``
        without the option are the isotropic rule, bit for bit, and the
        isotropic rule ignores ``criterion``."""
        for radii in (SPHERE, PENNY, CYL):
            st = failure.degraded_strengths(T800, radii, 0.03)
            assert st == failure.degraded_strengths(T800, radii, 0.03,
                                                    porosity_degradation='isotropic')
            assert failure.degraded_interlaminar_strengths(T800, radii, 0.03) == (None, None)
            assert failure.degraded_interlaminar_strengths(
                T800, radii, 0.03, porosity_degradation='isotropic',
                criterion='hashin') == (None, None)
            r_m, _ = failure._strength_ratios(T800, radii, 0.03)
            assert failure._interlaminar_strength_ratios(T800, radii, 0.03) == (r_m, r_m, r_m)

    def test_spherical_voids_keep_every_ratio(self):
        """Relative scaling: for spheres ``rho_33 = rho_22`` and ``rho_44 =
        rho_55 = rho_66``, so the directional ratios are ``r_m`` and the
        strengths agree to round-off. A literal ``sqrt(rho_55)`` would not
        (0.972 against ``r_m = 0.959`` at 3 %)."""
        for Vp in (0.01, 0.03, 0.1):
            r_m, _ = failure._strength_ratios(T800, SPHERE, Vp)
            r_Zt, r_S13, r_S23 = failure._interlaminar_strength_ratios(
                T800, SPHERE, Vp, porosity_degradation='directional')
            assert (r_Zt, r_S13, r_S23) == pytest.approx((r_m, r_m, r_m), rel=1e-12)
            iso = failure.degraded_strengths(T800, SPHERE, Vp)
            d = failure.degraded_strengths(T800, SPHERE, Vp, porosity_degradation='directional')
            assert d == pytest.approx(iso, rel=1e-12)
        r_m, _ = failure._strength_ratios(T800, SPHERE, 0.03)
        C_m = T800.get_isotropic_matrix_stiffness()
        from porosity_fe.homogenization import _mt_effective_stiffness
        C = _mt_effective_stiffness(C_m, 0.03, SPHERE, T800.matrix_poisson)
        assert r_m == pytest.approx(0.959, abs=5e-4)
        assert np.sqrt(C[4, 4] / C_m[4, 4]) == pytest.approx(0.972, abs=5e-4)

    def test_penny_numbers_t800(self):
        """Issue #212 at Vp = 0.03: ``Z_t`` 0.935 -> 0.800 (the ``C_33`` rule,
        since ``rho_22 = rho_11`` for a penny in the ply plane), ``S_13 =
        S_23`` 0.935 -> 0.873; the in-plane and fibre strengths unchanged."""
        iso = failure.degraded_strengths(T800, PENNY, 0.03)
        d = failure.degraded_strengths(T800, PENNY, 0.03, porosity_degradation='directional')
        assert iso[:5] == d[:5]                                     # Xt, Xc, Yt, Yc, S12
        assert iso[2] / T800.sigma_2t == pytest.approx(0.935, abs=5e-4)
        assert iso[5] / T800.tau_ilss == pytest.approx(0.935, abs=5e-4)
        assert d[5] / T800.tau_ilss == pytest.approx(0.873, abs=5e-4)
        Zt, S13 = failure.degraded_interlaminar_strengths(
            T800, PENNY, 0.03, porosity_degradation='directional', criterion='hashin')
        assert Zt / T800.sigma_2t == pytest.approx(0.800, abs=5e-4)
        assert S13 / T800.tau_ilss == pytest.approx(0.873, abs=5e-4)
        assert S13 == pytest.approx(d[5], rel=1e-12)
        r_m, _ = failure._strength_ratios(T800, PENNY, 0.03)
        r_Zt, r_S13, r_S23 = failure._interlaminar_strength_ratios(
            T800, PENNY, 0.03, porosity_degradation='directional')
        C_m = T800.get_isotropic_matrix_stiffness()
        from porosity_fe.homogenization import _mt_effective_stiffness
        C = _mt_effective_stiffness(C_m, 0.03, PENNY, T800.matrix_poisson)
        rho = np.diag(C) / np.diag(C_m)
        assert r_Zt == pytest.approx(np.sqrt(rho[2]), rel=1e-9)     # C_33 rule
        assert r_S13 == pytest.approx(r_m * np.sqrt(rho[4] / rho[5]), rel=1e-12)
        assert r_S23 == pytest.approx(r_m * np.sqrt(rho[3] / rho[5]), rel=1e-12)
        assert r_S13 == pytest.approx(r_S23, rel=1e-12)

    def test_cylindrical_void_keeps_z_t(self):
        """A cylinder along the fibres leaves the 2-3 plane isotropic:
        ``Z_t`` keeps ``r_m`` and ``S_13`` too, while ``S_23`` (the shear in
        that plane) softens."""
        r_m, _ = failure._strength_ratios(T800, CYL, 0.03)
        r_Zt, r_S13, r_S23 = failure._interlaminar_strength_ratios(
            T800, CYL, 0.03, porosity_degradation='directional')
        assert r_Zt == pytest.approx(r_m, rel=1e-9)
        assert r_S13 == pytest.approx(r_m, rel=1e-9)
        assert r_S23 < r_m

    def test_stand_in_follows_the_criterion(self):
        """With ``tau_13`` unset the directional ``S_13`` starts from each
        criterion's stand-in: ``tau_12`` for Tsai-Wu and maximum stress,
        ``tau_ilss`` for Hashin and the interface criteria; ``Z_t`` from
        ``sigma_2t``. Without a criterion there is no stand-in to pick."""
        r_Zt, r_S13, _ = failure._interlaminar_strength_ratios(
            T800, PENNY, 0.03, porosity_degradation='directional')
        for crit, base in (('tsai_wu', T800.tau_12), ('max_stress', T800.tau_12),
                           ('hashin', T800.tau_ilss), ('brewer_lagace', T800.tau_ilss),
                           ('ye', T800.tau_ilss)):
            Zt, S13 = failure.degraded_interlaminar_strengths(
                T800, PENNY, 0.03, porosity_degradation='directional', criterion=crit)
            assert Zt == pytest.approx(T800.sigma_2t * r_Zt, rel=1e-12)
            assert S13 == pytest.approx(base * r_S13, rel=1e-12)
        with pytest.raises(ValueError, match='needs the criterion'):
            failure.degraded_interlaminar_strengths(
                T800, PENNY, 0.03, porosity_degradation='directional')
        with pytest.raises(ValueError, match='needs the criterion'):
            failure.degraded_interlaminar_strengths(
                T800, PENNY, 0.03, porosity_degradation='directional', criterion='puck')

    def test_set_strengths_take_their_own_ratios(self):
        """``sigma_3t`` and ``tau_13`` set: scaled by ``r_Zt`` and ``r_S13``
        instead of ``r_m``, whatever the criterion; the floor still applies."""
        mat = dataclasses.replace(T800, sigma_3t=60.0, tau_13=95.0)
        r_m, _ = failure._strength_ratios(mat, PENNY, 0.03)
        r_Zt, r_S13, _ = failure._interlaminar_strength_ratios(
            mat, PENNY, 0.03, porosity_degradation='directional')
        assert failure.degraded_interlaminar_strengths(mat, PENNY, 0.03) == \
            pytest.approx((60.0 * r_m, 95.0 * r_m), rel=1e-12)
        for crit in (None, 'tsai_wu', 'hashin'):
            assert failure.degraded_interlaminar_strengths(
                mat, PENNY, 0.03, porosity_degradation='directional', criterion=crit) == \
                pytest.approx((60.0 * r_Zt, 95.0 * r_S13), rel=1e-12)
        assert r_Zt < r_m and r_S13 < r_m
        # At the 0.99 clamp the Mori-Tanaka tensor is nearly empty; the
        # strengths are floored rather than zero or negative.
        from porosity_fe.empirical import Calibration
        Zt, S13 = failure.degraded_interlaminar_strengths(
            mat, PENNY, 0.99, porosity_degradation='directional', criterion='hashin')
        assert Zt >= Calibration.STRENGTH_FLOOR_MPA and S13 >= Calibration.STRENGTH_FLOOR_MPA
        assert all(s >= Calibration.STRENGTH_FLOOR_MPA for s in failure.degraded_strengths(
            mat, PENNY, 0.99, porosity_degradation='directional'))

    def test_pristine_has_no_ratio(self):
        for mapping in ('isotropic', 'directional'):
            assert failure._interlaminar_strength_ratios(
                T800, PENNY, 0.0, porosity_degradation=mapping) == (1.0, 1.0, 1.0)
            st = failure.degraded_strengths(T800, PENNY, 0.0, porosity_degradation=mapping)
            assert st == (T800.sigma_1t, T800.sigma_1c, T800.sigma_2t, T800.sigma_2c,
                          T800.tau_12, T800.tau_ilss)

    def test_unknown_mapping_is_rejected(self):
        for call in (
            lambda: failure.degraded_strengths(T800, PENNY, 0.03, porosity_degradation='x'),
            lambda: failure.degraded_interlaminar_strengths(
                T800, PENNY, 0.03, porosity_degradation='x', criterion='hashin'),
            lambda: failure._interlaminar_strength_ratios(
                T800, PENNY, 0.03, porosity_degradation='x'),
        ):
            with pytest.raises(ValueError, match="Unknown porosity_degradation 'x'"):
                call()


# ----------------------------------------------------------------------
# Failure criteria and load factors
# ----------------------------------------------------------------------

def _random_state(seed=3):
    """Random ply-local stresses on a porous 4-ply mesh, every component
    large enough to govern somewhere, and a random prestress."""
    mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4)
    pf = PorosityField(mat, 0.03, distribution='clustered', seed=1)
    mesh = CompositeMesh(pf, mat, nx=3, ny=2, nz=4, ply_angles=[0, 90, 90, 0])
    rng = np.random.default_rng(seed)
    scale = np.array([600.0, 30.0, 30.0, 40.0, 40.0, 40.0])
    s = rng.normal(size=(mesh.n_elements, 8, 6)) * scale
    pre = rng.normal(size=s.shape) * scale * 0.3
    return mat, mesh, s, pre


def _evaluate(mat, mesh, s, pre, criterion, radii, mapping):
    fields: dict = {}
    out = failure.evaluate_failure(s, mesh.porosity, mesh.elements, mat, radii,
                                   criterion, mode_fields=fields,
                                   porosity_degradation=mapping)
    lam = failure._element_load_factors(s, mesh.porosity, mesh.elements, mat,
                                        radii, criterion, porosity_degradation=mapping)
    lam_p = failure._element_load_factors(s, mesh.porosity, mesh.elements, mat,
                                          radii, criterion, prestress_local=pre,
                                          porosity_degradation=mapping)
    return out, fields, lam, lam_p


class TestCriteria:
    @pytest.mark.parametrize('criterion', ['tsai_wu', 'hashin', 'max_stress'])
    def test_default_is_isotropic_bit_for_bit(self, criterion):
        mat, mesh, s, pre = _random_state()
        a = _evaluate(mat, mesh, s, pre, criterion, PENNY, 'isotropic')
        kw: dict = {}
        out = failure.evaluate_failure(s, mesh.porosity, mesh.elements, mat, PENNY,
                                       criterion, mode_fields=kw)
        lam = failure._element_load_factors(s, mesh.porosity, mesh.elements, mat,
                                            PENNY, criterion)
        lam_p = failure._element_load_factors(s, mesh.porosity, mesh.elements, mat,
                                              PENNY, criterion, prestress_local=pre)
        assert out[0] == a[0][0]
        np.testing.assert_array_equal(out[1], a[0][1])
        assert out[2] == a[0][2] or all(
            np.isnan(v) == np.isnan(a[0][2][k]) for k, v in out[2].items())
        for k in kw:
            np.testing.assert_array_equal(kw[k], a[1][k])
        np.testing.assert_array_equal(lam, a[2])
        np.testing.assert_array_equal(lam_p, a[3])
        assert failure.first_ply_failure_load_factor(
            s, mesh.porosity, mesh.elements, mat, PENNY, criterion) == float(lam.min())

    @pytest.mark.parametrize('criterion', ['tsai_wu', 'hashin', 'max_stress'])
    def test_spherical_voids_identical(self, criterion):
        mat, mesh, s, pre = _random_state()
        a = _evaluate(mat, mesh, s, pre, criterion, SPHERE, 'isotropic')
        b = _evaluate(mat, mesh, s, pre, criterion, SPHERE, 'directional')
        assert b[0][0] == pytest.approx(a[0][0], rel=1e-12)
        np.testing.assert_allclose(b[0][1], a[0][1], rtol=1e-12)
        for k in a[1]:
            np.testing.assert_allclose(b[1][k], a[1][k], rtol=1e-12)
        np.testing.assert_allclose(b[2], a[2], rtol=1e-12)
        np.testing.assert_allclose(b[3], a[3], rtol=1e-12)

    @pytest.mark.parametrize('criterion', ['tsai_wu', 'hashin', 'max_stress'])
    def test_penny_voids_weaken_the_interlaminar_terms(self, criterion):
        """The ``sigma_33``, ``tau_13`` and ``tau_23`` terms change, and they
        grow: an element loaded through the thickness alone has a higher
        index and a lower load factor. One loaded in plane alone is the
        same, except that Hashin's matrix-compression mode divides
        ``sigma_22`` by ``S_23`` too, so that state is kept tensile here
        (the mode's change is checked below)."""
        mat, mesh, s, pre = _random_state()
        radii = PENNY
        in_plane = s.copy()
        in_plane[:, :, 1] = np.abs(in_plane[:, :, 1])
        in_plane[:, :, 2:5] = 0.0
        out_of_plane = np.zeros_like(s)
        out_of_plane[:, :, 2:5] = s[:, :, 2:5]
        for stress, changes in ((in_plane, False), (out_of_plane, True)):
            a = _evaluate(mat, mesh, stress, pre * 0.0, criterion, radii, 'isotropic')
            b = _evaluate(mat, mesh, stress, pre * 0.0, criterion, radii, 'directional')
            if changes:
                assert b[0][0] > a[0][0]
                active = np.isfinite(a[2])
                assert np.all(b[2][active] <= a[2][active]) and b[2][active].min() < a[2][active].min()
            else:
                assert b[0][0] == pytest.approx(a[0][0], rel=1e-12)
                np.testing.assert_allclose(b[2], a[2], rtol=1e-12)

    def test_hashin_matrix_compression_follows_s23(self):
        """Hashin (1980) divides a compressive ``sigma_22`` by ``2 S_23``, the
        transverse shear strength of the 2-3 plane, so the directional
        ``S_23`` changes that mode for a purely in-plane state, where
        Tsai-Wu and maximum stress do not use ``S_23``."""
        mat, mesh, s, pre = _random_state()
        compressive = s.copy()
        compressive[:, :, 1] = -np.abs(compressive[:, :, 1])
        compressive[:, :, 2:5] = 0.0
        iso = _evaluate(mat, mesh, compressive, pre * 0.0, 'hashin', PENNY, 'isotropic')
        d = _evaluate(mat, mesh, compressive, pre * 0.0, 'hashin', PENNY, 'directional')
        np.testing.assert_allclose(d[1]['matrix_t'], iso[1]['matrix_t'], rtol=1e-12)
        np.testing.assert_allclose(d[1]['fiber_t'], iso[1]['fiber_t'], rtol=1e-12)
        assert not np.allclose(d[1]['matrix_c'], iso[1]['matrix_c'], rtol=1e-6)

    @pytest.mark.parametrize('criterion', ['tsai_wu', 'hashin', 'max_stress'])
    def test_load_factor_paths_agree(self, criterion):
        """The closed-form load factor and the prestressed (vectorized) path
        agree with a zero prestress, and the index at the load factor is 1,
        with the directional strengths too."""
        mat, mesh, s, pre = _random_state()
        _, _, lam, lam_zero = _evaluate(mat, mesh, s, pre * 0.0, criterion, PENNY,
                                        'directional')
        np.testing.assert_allclose(lam_zero, lam, rtol=1e-9)
        e = int(np.argmin(lam))
        scaled = s * lam[e]
        fi, per_elem, _ = failure.evaluate_failure(
            scaled, mesh.porosity, mesh.elements, mat, PENNY, criterion,
            porosity_degradation='directional')
        assert per_elem[e] == pytest.approx(1.0, rel=1e-9)
        assert fi == pytest.approx(1.0, rel=1e-9)

    def test_unknown_mapping_is_rejected(self):
        mat, mesh, s, pre = _random_state()
        with pytest.raises(ValueError, match="Unknown porosity_degradation"):
            failure.evaluate_failure(s, mesh.porosity, mesh.elements, mat, PENNY,
                                     'tsai_wu', porosity_degradation='radial')
        with pytest.raises(ValueError, match="Unknown porosity_degradation"):
            failure.first_ply_failure_load_factor(
                s, mesh.porosity, mesh.elements, mat, PENNY, 'tsai_wu',
                porosity_degradation='radial')
