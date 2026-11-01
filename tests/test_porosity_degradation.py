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


# ----------------------------------------------------------------------
# Threading through the FE path
# ----------------------------------------------------------------------

#: A thin 4-ply cross-ply of the AS4/3501-6 preset (it has CTEs), so the
#: thermal path runs too.
MAT4 = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4)
LAYUP = [0.0, 90.0, 90.0, 0.0]


def _coupon(void_shape: str, distribution: str = 'clustered'):
    pf = PorosityField(MAT4, 0.03, distribution=distribution, void_shape=void_shape, seed=1)
    mesh = CompositeMesh(pf, MAT4, nx=6, ny=3, nz=4, ply_angles=LAYUP)
    return pf, mesh


def _assert_results_close(a, b, mesh, rtol, *, thermal=False):
    """Every ``FieldResults`` field of ``a`` and ``b`` agrees to ``rtol``.

    Arrays are compared relative to their largest entry, the failure-index
    fields relative to the maximum failure index (a mode that is zero up to
    round-off, such as delamination under in-plane shear, is noise against
    its own maximum), and strings, ``None`` and flags exactly. Two fields
    are round-off by construction and only bounded: the reactions of the
    statically determinate thermal supports, and the mode breakdown at the
    governing Gauss point, which can move between points whose indices tie
    to round-off (its ``max_fi`` is compared).
    """
    fi_scale = abs(a.max_failure_index)
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == 'failure_mode_indices':
            assert x['max_fi'] == pytest.approx(y['max_fi'], rel=rtol)
        elif f.name == 'reaction_forces' and thermal:
            bound = 1e-8 * float(np.abs(a.stress_global).max()) * mesh.L_y * mesh.L_z
            assert np.abs(x).max() < bound and np.abs(y).max() < bound
        elif f.name in ('per_element_failure_index', 'per_element_mode_failure_index',
                        'failure_mode_maxima'):
            _assert_value_close(x, y, rtol, f.name, scale=fi_scale)
        else:
            _assert_value_close(x, y, rtol, f.name)


def _assert_value_close(x, y, rtol, where, scale=None):
    if x is None or y is None or isinstance(x, (str, bool, np.bool_)):
        assert x == y, where
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), where
        for k in x:
            _assert_value_close(x[k], y[k], rtol, f"{where}[{k}]", scale)
    elif dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            _assert_value_close(getattr(x, f.name), getattr(y, f.name), rtol,
                                f"{where}.{f.name}", scale)
    else:
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert xa.shape == ya.shape, where
        finite = np.isfinite(xa)
        np.testing.assert_array_equal(finite, np.isfinite(ya), err_msg=where)
        np.testing.assert_array_equal(xa[~finite], ya[~finite], err_msg=where)
        if scale is None:
            scale = float(np.abs(xa[finite]).max()) if finite.any() else 0.0
        np.testing.assert_allclose(ya[finite], xa[finite], rtol=rtol,
                                   atol=rtol * scale, err_msg=where)


class TestFEThreading:
    def test_default_property_and_validation(self):
        from porosity_fe import FESolver, GlobalAssembler
        pf, mesh = _coupon('penny')
        fe = FESolver(mesh, MAT4, pf, ply_angles=LAYUP)
        assert fe.porosity_degradation == 'isotropic'
        assert fe.assembler.porosity_degradation == 'isotropic'
        fe_d = FESolver(mesh, MAT4, pf, ply_angles=LAYUP, porosity_degradation='directional')
        assert fe_d.porosity_degradation == 'directional'
        assert fe_d.assembler.create_element(0).porosity_degradation == 'directional'
        assert fe_d.assembler.element_batch().porosity_degradation == 'directional'
        with pytest.raises(AttributeError):
            fe_d.porosity_degradation = 'isotropic'  # type: ignore[misc]
        with pytest.raises(ValueError, match="Unknown porosity_degradation 'radial'"):
            FESolver(mesh, MAT4, pf, porosity_degradation='radial')  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown porosity_degradation"):
            GlobalAssembler(mesh, MAT4, pf, porosity_degradation='Directional')  # type: ignore[arg-type]

    def test_assembler_cache_is_keyed_on_it(self):
        """Changing the mapping on the assembler re-assembles, like a change
        of formulation; the penny laminate is softer through the thickness
        and stiffer in plane."""
        from porosity_fe import GlobalAssembler
        pf, mesh = _coupon('penny')
        asm = GlobalAssembler(mesh, MAT4, pf)
        K_iso = asm.stiffness().copy()
        key_iso = asm._state_key()
        asm.porosity_degradation = 'directional'
        assert asm._state_key() != key_iso
        K_dir = asm.stiffness()
        assert asm.element_batch().porosity_degradation == 'directional'
        assert abs(K_dir - K_iso).max() > 1e-3 * abs(K_iso).max()
        asm.porosity_degradation = 'isotropic'
        assert (asm.stiffness() != K_iso).nnz == 0

    @pytest.mark.parametrize('formulation', ['hex8', 'hex8i'])
    def test_batch_matches_hex8_element(self, formulation):
        """The batched stiffness and stresses reproduce the reference
        ``Hex8Element`` with the directional mapping, on graded porosity."""
        from porosity_fe import GlobalAssembler
        pf, mesh = _coupon('penny')
        asm = GlobalAssembler(mesh, MAT4, pf, formulation=formulation,
                              porosity_degradation='directional')
        batch = asm.element_batch()
        Ke_batch = batch.stiffness_matrices()
        rng = np.random.default_rng(0)
        u = rng.normal(scale=1e-3, size=mesh.n_dof)
        stress = batch.stresses(batch.strains(u))
        for e in range(mesh.n_elements):
            elem = asm.create_element(e)
            Ke = elem.stiffness_matrix()
            Ke = 0.5 * (Ke + Ke.T)
            np.testing.assert_allclose(Ke_batch[e], Ke, rtol=1e-10,
                                       atol=1e-10 * np.abs(Ke).max(), err_msg=f"element {e}")
            sig = elem.stress_at_gauss_points(u[asm.element_dof_indices(e)])
            np.testing.assert_allclose(stress[e], sig, rtol=1e-10,
                                       atol=1e-10 * np.abs(sig).max())

    @pytest.mark.parametrize('loading, kw, criterion', [
        (loading, kw, criterion)
        for loading, kw in (('compression', {}), ('tension', {'delta_T': -157.0}),
                            ('thermal', {'delta_T': -157.0}), ('shear', {}))
        for criterion in ('tsai_wu', 'hashin', 'max_stress')
        # Under in-plane shear sigma_11 is zero up to round-off, and Hashin's
        # fibre- and matrix-tension branches switch on its sign, so their
        # per-element mode fields flip between the two runs.
        if (loading, criterion) != ('shear', 'hashin')
    ])
    def test_spherical_voids_give_identical_results(self, loading, kw, criterion):
        """Every ``FieldResults`` field agrees to round-off for spherical
        voids, with a thermal prestress and each criterion (issue #212). The
        direct solver only: two CG solves differ by their iteration error."""
        from porosity_fe import FESolver
        pf, mesh = _coupon('spherical')
        a = FESolver(mesh, MAT4, pf, ply_angles=LAYUP).solve(
            loading, failure_criterion=criterion, **kw)
        b = FESolver(mesh, MAT4, pf, ply_angles=LAYUP,
                     porosity_degradation='directional').solve(
            loading, failure_criterion=criterion, **kw)
        # The small coupon has elements of aspect ratio up to 66, so its
        # stiffness is ill-conditioned: the stiffness matrices differ by
        # about 1e-15, the scalar results by about 1e-13, the fields by up
        # to 1e-10 of their maximum, and the free-standing thermal solve
        # (3-2-1 supports on a 0.5 mm plate) by up to 2e-8. Under the
        # prestress the residual stress takes most of the strength, so the
        # first-ply-failure factor (about 0.06) is a small difference of
        # large terms: 2e-11 relative on macOS. The scalars therefore take
        # the fields' tolerance; the penny effect they guard is 1e-2.
        thermal = loading == 'thermal'
        _assert_results_close(a, b, mesh, rtol=1e-7 if thermal else 1e-9, thermal=thermal)
        for name in ('knockdown', 'max_failure_index', 'first_ply_failure_load_factor'):
            x, y = getattr(a, name), getattr(b, name)
            if np.isfinite(x):
                assert y == pytest.approx(x, rel=1e-9), name
            else:
                assert repr(x) == repr(y), name           # nan (thermal) or inf

    def test_penny_direct_cg_and_thermal_see_it(self):
        """With penny voids the directional solve differs from the isotropic
        one, the same way for the direct and CG solvers, and the thermal
        residual stress changes with it (its unit solution is cached on K)."""
        from porosity_fe import FESolver
        pf, mesh = _coupon('penny')
        iso = FESolver(mesh, MAT4, pf, ply_angles=LAYUP)
        d = FESolver(mesh, MAT4, pf, ply_angles=LAYUP, porosity_degradation='directional')
        r_iso = iso.solve('compression', delta_T=-157.0, failure_criterion='hashin')
        r_dir = d.solve('compression', delta_T=-157.0, failure_criterion='hashin')
        # CG on the mechanical system (the free-standing thermal one of this
        # coarse coupon stalls near 1e-8).
        r_direct = d.solve('compression', failure_criterion='hashin')
        r_cg = d.solve('compression', failure_criterion='hashin', solver='cg', rtol=1e-10)
        assert r_dir.knockdown != pytest.approx(r_iso.knockdown, rel=1e-6)
        assert r_direct.knockdown == r_dir.knockdown
        assert r_cg.knockdown == pytest.approx(r_direct.knockdown, rel=1e-8)
        np.testing.assert_allclose(r_cg.displacement, r_direct.displacement, rtol=1e-6,
                                   atol=1e-6 * np.abs(r_direct.displacement).max())
        r_cg_iso = iso.solve('compression', failure_criterion='hashin', solver='cg', rtol=1e-10)
        assert r_cg.knockdown != pytest.approx(r_cg_iso.knockdown, rel=1e-6)
        assert not np.allclose(r_dir.residual_stress_local, r_iso.residual_stress_local,
                               rtol=1e-6)
        th_d = d.solve('thermal', delta_T=-157.0)
        np.testing.assert_allclose(th_d.stress_local, r_dir.residual_stress_local,
                                   rtol=1e-9, atol=1e-9 * np.abs(th_d.stress_local).max())

    def test_pristine_reference_is_shared(self):
        """The knockdown reference has no porosity, so both mappings use the
        one cached entry, and the knockdowns compare the porous stiffnesses
        directly."""
        from porosity_fe import FESolver
        from porosity_fe.fe import _knockdown
        pf, mesh = _coupon('penny')
        r_iso = FESolver(mesh, MAT4, pf, ply_angles=LAYUP).solve('compression')
        n = len(_knockdown._PRISTINE_MEASURE_CACHE)
        keys = list(_knockdown._PRISTINE_MEASURE_CACHE)
        r_dir = FESolver(mesh, MAT4, pf, ply_angles=LAYUP,
                         porosity_degradation='directional').solve('compression')
        assert len(_knockdown._PRISTINE_MEASURE_CACHE) == n
        assert set(_knockdown._PRISTINE_MEASURE_CACHE) == set(keys)
        assert r_dir.effective_modulus / r_iso.effective_modulus == pytest.approx(
            r_dir.knockdown / r_iso.knockdown, rel=1e-9)

    def test_failure_metrics_follow_the_solver(self):
        """The solve's failure index and load factor are those of the
        directional strengths evaluated on its stresses."""
        from porosity_fe import FESolver
        pf, mesh = _coupon('penny')
        fe = FESolver(mesh, MAT4, pf, ply_angles=LAYUP, porosity_degradation='directional')
        r = fe.solve('compression', failure_criterion='hashin')
        args = (r.stress_local, mesh.porosity, mesh.elements, MAT4, pf.void_shape_radii,
                'hashin')
        modes: dict = {}
        fi, _, _ = failure.evaluate_failure(*args, void_elements=mesh.void_elements,
                                            mode_fields=modes,
                                            porosity_degradation='directional')
        lam = failure.first_ply_failure_load_factor(
            *args, void_elements=mesh.void_elements, porosity_degradation='directional')
        assert r.max_failure_index == fi
        assert r.first_ply_failure_load_factor == lam
        assert r.failure_mode_maxima['delamination'] == float(modes['delamination'].max())
        assert fe._degraded_strengths(0.03) == failure.degraded_strengths(
            MAT4, pf.void_shape_radii, 0.03, porosity_degradation='directional')
        # The compression maximum is an in-plane mode, which the mapping
        # leaves alone; the delamination mode on the same stresses is
        # higher with the directional strengths.
        modes_iso: dict = {}
        failure.evaluate_failure(*args, void_elements=mesh.void_elements,
                                 mode_fields=modes_iso)
        assert modes['delamination'].max() > modes_iso['delamination'].max()

    def test_delamination_onset(self):
        """``delamination_onset`` and ``FieldResults.delamination_onset``
        take the mapping: the strengths at each interface node are the
        directional ones (``Zc`` with ``Z_t``'s ratio), and spheres give the
        isotropic onset."""
        from porosity_fe import FESolver, delamination_onset
        for shape in ('penny', 'spherical'):
            pf, mesh = _coupon(shape)
            r = FESolver(mesh, MAT4, pf, ply_angles=LAYUP,
                         porosity_degradation='directional').solve('tension')
            iso = delamination_onset(r, mesh, MAT4, Zc=200.0)
            d = delamination_onset(r, mesh, MAT4, Zc=200.0, porosity_degradation='directional')
            d2 = r.delamination_onset(mesh, MAT4, Zc=200.0, porosity_degradation='directional')
            np.testing.assert_array_equal(d.index, d2.index)
            e = d.evaluated
            if shape == 'spherical':
                np.testing.assert_allclose(d.index[e], iso.index[e], rtol=1e-11)
                continue
            vp = float(d.Vp[e][0])
            r_Zt, r_S13, r_S23 = failure._interlaminar_strength_ratios(
                MAT4, pf.void_shape_radii, vp, porosity_degradation='directional')
            i0 = np.flatnonzero(e)[0]
            assert d.Zt[i0] == pytest.approx(MAT4.sigma_2t * r_Zt, rel=1e-12)
            assert d.S13[i0] == pytest.approx(MAT4.tau_ilss * r_S13, rel=1e-12)
            assert d.S23[i0] == pytest.approx(MAT4.tau_ilss * r_S23, rel=1e-12)
            assert np.all(d.Zt[e] < iso.Zt[e]) and np.all(d.S13[e] < iso.S13[e])
            assert d.max_index > iso.max_index
            assert d.onset_load_factor < iso.onset_load_factor
            ye = delamination_onset(r, mesh, MAT4, criterion='ye',
                                    porosity_degradation='directional')
            np.testing.assert_allclose(ye.S13[e], d.S13[e], rtol=1e-12)
        with pytest.raises(ValueError, match="Unknown porosity_degradation"):
            delamination_onset(r, mesh, MAT4, porosity_degradation='radial')  # type: ignore[arg-type]

    def test_refinement_study_records_it(self):
        from porosity_fe import refinement_study
        res = [(4, 2, 4), (6, 3, 4), (8, 4, 4)]
        kw = dict(resolutions=res, ply_angles=LAYUP,
                  porosity_config={'distribution': 'uniform', 'void_shape': 'penny'})
        iso = refinement_study(MAT4, 0.03, 'compression', **kw)
        d = refinement_study(MAT4, 0.03, 'compression', porosity_degradation='directional', **kw)
        assert iso.porosity_degradation == 'isotropic'
        assert iso.to_dict()['porosity_degradation'] == 'isotropic'
        assert 'porosity_degradation' not in iso.summary()
        assert d.porosity_degradation == 'directional'
        assert d.to_dict()['porosity_degradation'] == 'directional'
        assert "porosity_degradation='directional'" in d.summary()
        assert d.levels[-1].knockdown != pytest.approx(iso.levels[-1].knockdown, rel=1e-6)
        with pytest.raises(ValueError, match="Unknown porosity_degradation"):
            refinement_study(MAT4, 0.03, 'compression', porosity_degradation='x', **kw)  # type: ignore[arg-type]

    def test_knockdown_decomposition_passes_it(self, monkeypatch):
        """Every solve of the decomposition uses the mapping, and its
        ``total`` is the directional solve's knockdown."""
        from porosity_fe import Delamination, FESolver, knockdown_decomposition
        pf = PorosityField(MAT4, 0.03, distribution='uniform', void_shape='penny')
        mesh = CompositeMesh(pf, MAT4, nx=6, ny=3, nz=4, ply_angles=LAYUP,
                             delaminations=[Delamination.strip(1, 10.0, 30.0)])
        seen = []
        real = FESolver.solve

        def spy(self, *args, **kwargs):
            seen.append(self.porosity_degradation)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(FESolver, 'solve', spy)
        parts = knockdown_decomposition(mesh, MAT4, pf, 'compression',
                                        porosity_degradation='directional')
        assert seen == ['directional'] * 3
        seen.clear()
        total = FESolver(mesh, MAT4, pf, ply_angles=LAYUP,
                         porosity_degradation='directional').solve(
            'compression', toughness_knockdown=None)
        assert parts.total == pytest.approx(total.knockdown, rel=1e-12)
        iso = knockdown_decomposition(mesh, MAT4, pf, 'compression')
        assert seen[-3:] == ['isotropic'] * 3
        assert iso.porosity_only != pytest.approx(parts.porosity_only, rel=1e-6)
        assert iso.delamination_only == pytest.approx(parts.delamination_only, rel=1e-12)
