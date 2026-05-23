#!/usr/bin/env python3
"""Tests for porosity_fe.homogenization.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (MATERIALS, _mt_effective_stiffness,
                                   _degraded_composite_stiffness,
                                   compute_clt_effective_modulus)


class TestCLTEffectiveModulus:
    def test_all_zero_plies_returns_E11(self):
        """All 0-degree plies should give E_x close to E11."""
        mat = MATERIALS['T800_epoxy']
        E_x = compute_clt_effective_modulus(mat, [0.0] * 24)
        # Should be close to E11 (plane-stress correction makes it slightly different)
        assert abs(E_x - mat.E11) / mat.E11 < 0.02

    def test_quasi_isotropic_lower_than_E11(self):
        """QI layup should have E_x much lower than E11."""
        mat = MATERIALS['T800_epoxy']
        angles = [0, 45, 90, -45] * 6  # 24 plies QI
        E_x = compute_clt_effective_modulus(mat, angles)
        assert E_x < mat.E11
        assert E_x > mat.E22  # Should still be stiffer than transverse

    def test_positive_modulus(self):
        mat = MATERIALS['T800_epoxy']
        E_x = compute_clt_effective_modulus(mat, [0, 90, 0, 90] * 6)
        assert E_x > 0

    def test_symmetric_layup(self):
        """Symmetric layup [0/90]_s should equal [0/90/90/0]."""
        mat = MATERIALS['T800_epoxy']
        E1 = compute_clt_effective_modulus(mat, [0, 90, 90, 0] * 6)
        E2 = compute_clt_effective_modulus(mat, [0, 90] * 12)
        # A-matrix is the same for both (same ply count per angle)
        assert abs(E1 - E2) / E1 < 1e-10


class TestMTEffectiveStiffness:
    def setup_method(self):
        self.mat = MATERIALS['T800_epoxy']
        self.C_m = self.mat.get_isotropic_matrix_stiffness()

    def test_zero_porosity_returns_matrix(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.0, (1, 1, 1), 0.35)
        np.testing.assert_allclose(C_eff, self.C_m, atol=1e-6)

    def test_high_porosity_near_zero(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.99, (1, 1, 1), 0.35)
        assert C_eff[0, 0] < self.C_m[0, 0] * 0.1

    def test_decreasing_stiffness(self):
        C1 = _mt_effective_stiffness(self.C_m, 0.01, (1, 1, 1), 0.35)
        C5 = _mt_effective_stiffness(self.C_m, 0.05, (1, 1, 1), 0.35)
        assert C1[0, 0] > C5[0, 0]

    def test_positive_definite(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.05, (1, 1, 1), 0.35)
        eigenvalues = np.linalg.eigvalsh(C_eff)
        assert np.all(eigenvalues > 0)

    def test_prolate_void_shape(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.03, (3, 1, 1), 0.35)
        assert C_eff.shape == (6, 6)
        assert C_eff[0, 0] < self.C_m[0, 0]

    def test_oblate_void_shape(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.03, (3, 3, 0.3), 0.35)
        assert C_eff.shape == (6, 6)
        assert C_eff[0, 0] < self.C_m[0, 0]

    def test_finite_for_full_Vp_sweep_oblate(self):
        # Oblate voids near Vp -> 1 are the worst case for MT inversion;
        # the pinv fallback + finite check should keep all entries finite.
        for Vp in [0.50, 0.85, 0.95, 0.985]:
            C_eff = _mt_effective_stiffness(self.C_m, Vp, (3, 3, 0.3), 0.35)
            assert np.all(np.isfinite(C_eff)), f"non-finite C_eff at Vp={Vp}"

    def test_penny_void_anisotropy_along_short_axis(self):
        """Regression for #32. A penny-shaped void (3, 3, 0.3) has its
        symmetry axis along x_3 (the short axis), so the effective
        stiffness should show LARGER degradation along the through-disk
        direction (S[2,2]) than along the in-plane directions (S[0,0],
        S[1,1]). The old code treated penny as a prolate cylinder along
        x_1 and degraded the wrong axis."""
        Vp = 0.05
        C_eff = _mt_effective_stiffness(self.C_m, Vp, (3, 3, 0.3), 0.35)
        # Through-thickness (x_3) component degrades more than in-plane (x_1, x_2)
        deg_xx = (self.C_m[0, 0] - C_eff[0, 0]) / self.C_m[0, 0]
        deg_yy = (self.C_m[1, 1] - C_eff[1, 1]) / self.C_m[1, 1]
        deg_zz = (self.C_m[2, 2] - C_eff[2, 2]) / self.C_m[2, 2]
        assert deg_zz > deg_xx, (
            f"penny axis degradation {deg_zz:.4f} should exceed in-plane "
            f"degradation {deg_xx:.4f} (the disk is perpendicular to x_3)"
        )
        # The two in-plane components should be approximately equal
        # (transverse isotropy of an axisymmetric disk).
        assert abs(deg_xx - deg_yy) < 1e-6

    def test_prolate_cylindrical_anisotropy_transverse_to_long_axis(self):
        """Regression for #32. (3, 1, 1) is a prolate cylindrical void
        with its symmetry axis along x_1. Load flows easily along the
        long axis (the void is thin in cross-section), but transverse
        load has to bypass a long obstacle — so transverse degradation
        (deg_yy, deg_zz) should exceed axial degradation (deg_xx)."""
        Vp = 0.05
        C_eff = _mt_effective_stiffness(self.C_m, Vp, (3, 1, 1), 0.35)
        deg_xx = (self.C_m[0, 0] - C_eff[0, 0]) / self.C_m[0, 0]
        deg_yy = (self.C_m[1, 1] - C_eff[1, 1]) / self.C_m[1, 1]
        deg_zz = (self.C_m[2, 2] - C_eff[2, 2]) / self.C_m[2, 2]
        assert deg_yy > deg_xx
        # The two equatorial directions are equivalent (axisymmetric).
        assert abs(deg_yy - deg_zz) < 1e-6

    def test_cache_hit_returns_identical_result(self):
        # #42, #112: a repeated call with the same key must come from the
        # cache and return a numerically identical result (within fp
        # tolerance of the original computation, which here is exact
        # equality since the cache stores the actual array).
        from porosity_fe import _mt_effective_stiffness_cached
        _mt_effective_stiffness_cached.cache_clear()
        first = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        assert _mt_effective_stiffness_cached.cache_info().currsize == 1
        second = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        # Still one entry — no duplication.
        info = _mt_effective_stiffness_cached.cache_info()
        assert info.currsize == 1
        assert info.hits >= 1
        np.testing.assert_array_equal(first, second)

    def test_cache_returns_defensive_copy(self):
        # Callers may mutate the returned array (e.g. callers in the FE
        # path build derived ratios). The cache must not be poisoned by
        # that mutation — the next call must still return the original.
        from porosity_fe import _mt_effective_stiffness_cached
        _mt_effective_stiffness_cached.cache_clear()
        first = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        first[0, 0] = -999.0  # mutate the returned array
        second = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        assert second[0, 0] != -999.0

    def test_cache_distinguishes_materials(self):
        # Two materials with different C_m[0,0] must NOT collide in the
        # cache even at identical (Vp, shape, nu_m).
        from porosity_fe import _mt_effective_stiffness_cached
        _mt_effective_stiffness_cached.cache_clear()
        C_m2 = self.C_m * 2.0  # different fingerprint
        a = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        b = _mt_effective_stiffness(C_m2, 0.04, (1, 1, 1), 0.35)
        assert _mt_effective_stiffness_cached.cache_info().currsize == 2
        # The stiffer matrix should give a stiffer effective stiffness.
        assert b[0, 0] > a[0, 0]


class TestOblateMTValidation:
    """Regression pin for the oblate (penny-void) Eshelby branch (#143).

    Issue #32 fixed an earlier bug where oblate (alpha < 1) voids were
    silently routed to the prolate Eshelby g-function. The current
    implementation in ``_mt_effective_stiffness`` selects the correct
    g-function on ``alpha > 1.0`` vs the ``else`` (alpha <= 1) branch.
    The math is correct; these tests pin numerical snapshots of the
    post-#32 behavior so that future refactors are forced to confirm
    they reproduce the same Mori-Tanaka output.

    Reference: Mura, T. (1987) *Micromechanics of Defects in Solids* §11
    and Nemat-Nasser & Hori (1999) §7.4 describe the prolate/oblate
    Eshelby tensor split. No closed-form analytical value is hand-
    derived here; the snapshots in ``test_oblate_transverse_stiffness_
    regression_pin`` are pinned from the current correct implementation
    (a regression pin, per the issue's note that the math is already
    right).
    """

    def setup_method(self):
        self.mat = MATERIALS['T800_epoxy']
        self.C_m = self.mat.get_isotropic_matrix_stiffness()
        self.nu_m = self.mat.matrix_poisson

    def test_sphere_limit_oblate_matches_prolate(self):
        """As alpha -> 1, the prolate and oblate branches must converge to the
        same sphere result (analytic continuation across alpha = 1).

        Note on tolerance: the issue text proposed ``atol=1e-12``, but at
        ``alpha = 1`` exactly both branches divide by (alpha^2 - 1) and are
        singular. The smallest aspect-ratio offset that still bypasses the
        function's internal sphere shortcut (which fires when all three
        radii are within 1% of each other) is eps ~ 0.011. At that offset
        the prolate(1+eps) and oblate(1-eps) results are two physically
        different configurations and necessarily differ at O(eps), so a
        bit-exact match is mathematically impossible. We pin the looser
        but still tight tolerance that actually holds. Measured gap at
        eps=0.011: |Cp - Co|_inf / |Csph|_inf ~ 3.5e-3.
        """
        eps = 0.011  # just outside the 1% sphere shortcut
        # Compute the sphere baseline via the function's sphere shortcut
        # (radii all equal => short-circuit branch).
        from porosity_fe import _mt_effective_stiffness_cached
        _mt_effective_stiffness_cached.cache_clear()
        C_sphere = _mt_effective_stiffness(
            self.C_m, 0.05, (1.0, 1.0, 1.0), self.nu_m)
        _mt_effective_stiffness_cached.cache_clear()
        C_prolate = _mt_effective_stiffness(
            self.C_m, 0.05, (1.0, 1.0, 1.0 + eps), self.nu_m)
        _mt_effective_stiffness_cached.cache_clear()
        C_oblate = _mt_effective_stiffness(
            self.C_m, 0.05, (1.0, 1.0, 1.0 - eps), self.nu_m)

        scale = np.max(np.abs(C_sphere))
        # Each branch is within ~0.2% of the sphere result at eps=0.011.
        np.testing.assert_allclose(C_prolate, C_sphere, atol=0.005 * scale)
        np.testing.assert_allclose(C_oblate, C_sphere, atol=0.005 * scale)
        # Cross-difference (both branches agree near the sphere limit)
        # to ~0.5% — pins that the formulas are analytic continuations.
        np.testing.assert_allclose(C_prolate, C_oblate, atol=0.01 * scale)

    def test_oblate_transverse_stiffness_regression_pin(self):
        """Snapshot pin for a penny-shaped void (alpha = 0.01) at 5% porosity.

        Regression pin per #143; snapshot of post-#32 behavior. Penny axis
        is along x_3 (radii = (1, 1, 0.01)), so x_1 / x_2 are the in-plane
        (transverse to the disk axis) directions. ``C_eff[1, 1]`` is the
        in-plane stiffness perpendicular to the penny axis.

        These values were generated from the current implementation and
        will catch any silent regression in the oblate g-function or the
        Voigt permutation that maps the canonical-frame Eshelby tensor
        onto the actual axis.
        """
        from porosity_fe import _mt_effective_stiffness_cached
        _mt_effective_stiffness_cached.cache_clear()
        C_eff = _mt_effective_stiffness(
            self.C_m, 0.05, (1.0, 1.0, 0.01), self.nu_m)
        # Snapshot to ~5 significant figures; rtol=1e-4 catches numerically
        # meaningful regressions while tolerating cross-platform FP noise.
        assert C_eff[1, 1] == pytest.approx(5345.6799, rel=1e-4)
        assert C_eff[0, 0] == pytest.approx(5345.6799, rel=1e-4)
        # Off-diagonal in-plane coupling (also pinned)
        assert C_eff[0, 1] == pytest.approx(2884.2736, rel=1e-4)
        # Penny in-plane transverse isotropy: C[0,0] == C[1,1] exactly.
        assert C_eff[0, 0] == pytest.approx(C_eff[1, 1], rel=1e-12)

    def test_oblate_monotonic_in_vp(self):
        """Increasing porosity must monotonically reduce transverse stiffness
        for an oblate (penny) void at fixed aspect ratio.
        """
        from porosity_fe import _mt_effective_stiffness_cached
        Vps = [0.001, 0.01, 0.03, 0.05]
        C22_values = []
        for Vp in Vps:
            _mt_effective_stiffness_cached.cache_clear()
            C = _mt_effective_stiffness(
                self.C_m, Vp, (1.0, 1.0, 0.01), self.nu_m)
            C22_values.append(C[1, 1])
        C22_values = np.array(C22_values)
        assert np.all(np.diff(C22_values) < 0), (
            f"transverse stiffness not strictly decreasing in Vp: {C22_values}"
        )

    def test_oblate_more_severe_than_prolate_at_matched_vp(self):
        """At matched Vp, compare oblate (penny) vs prolate (needle) transverse
        stiffness reduction.

        Note on the issue's expected inequality (#143): the issue text
        predicted that penny (oblate) voids would degrade C[1, 1] *more*
        than prolate (needle) voids at matched Vp. That intuition is
        wrong for the canonical orientation:

        - Penny axis along x_3 (radii = (1, 1, 0.01)) means the disk lies
          IN the x_1-x_2 plane. Loads along x_1 (i.e. C[1, 1]) travel
          along the long in-plane dimension of the disk and barely see
          the void — degradation is mild (ratio ~ 0.952 at Vp = 0.05).
          The brutal direction is C[2, 2] (through-thickness, where the
          load is forced across the penny's short axis); this matches
          the existing ``test_penny_void_anisotropy_along_short_axis``
          regression for #32.
        - Prolate along x_3 (radii = (1, 1, 100)) is a long needle.
          Transverse loads (C[1, 1]) have to flow around the entire
          length of the needle — degradation is severe (ratio ~ 0.833
          at Vp = 0.05).

        So at matched Vp, prolate degrades C[1, 1] MORE than oblate, not
        less. We pin the actual (correct) physics here rather than the
        issue's predicted inequality.
        """
        from porosity_fe import _mt_effective_stiffness_cached
        Vp = 0.05
        _mt_effective_stiffness_cached.cache_clear()
        C_oblate = _mt_effective_stiffness(
            self.C_m, Vp, (1.0, 1.0, 0.01), self.nu_m)
        _mt_effective_stiffness_cached.cache_clear()
        C_prolate = _mt_effective_stiffness(
            self.C_m, Vp, (1.0, 1.0, 100.0), self.nu_m)

        ratio_oblate = C_oblate[1, 1] / self.C_m[1, 1]
        ratio_prolate = C_prolate[1, 1] / self.C_m[1, 1]
        # Actual measured ratios (snapshotted): oblate ~0.952, prolate ~0.833.
        # Prolate is MORE severe on the transverse C[1, 1] component.
        assert ratio_prolate < ratio_oblate, (
            f"expected prolate C[1,1] reduction stronger than oblate; "
            f"got oblate={ratio_oblate:.4f}, prolate={ratio_prolate:.4f}"
        )
        # Pin the numerical values too so the inequality direction can't
        # silently flip without flagging the snapshot.
        assert ratio_oblate == pytest.approx(0.9516, rel=1e-3)
        assert ratio_prolate == pytest.approx(0.8328, rel=1e-3)


class TestDegradedCompositeStiffness:
    """Direct unit tests for _degraded_composite_stiffness (#48).

    Previously exercised only indirectly through Hex8Element._degraded_stiffness;
    the Vp < 1e-12, Vp > 0.99, and the lame-denominator guard branches were
    not covered, and past matrix-modulus fixes lived in this function.
    """

    def setup_method(self):
        self.mat = MATERIALS['T800_epoxy']
        self.pristine = self.mat.get_stiffness_matrix()

    def test_vp_zero_returns_pristine(self):
        C = _degraded_composite_stiffness(0.0, (1, 1, 1), self.mat)
        np.testing.assert_allclose(C, self.pristine, atol=1e-9)

    def test_vp_subepsilon_returns_pristine(self):
        # Below the 1e-12 guard: must take the early-return branch.
        C = _degraded_composite_stiffness(1e-15, (1, 1, 1), self.mat)
        np.testing.assert_allclose(C, self.pristine, atol=1e-9)

    def test_vp_near_one_returns_zeros(self):
        # Above the 0.99 guard: collapsed material is fully degraded.
        C = _degraded_composite_stiffness(0.995, (1, 1, 1), self.mat)
        np.testing.assert_array_equal(C, np.zeros((6, 6)))

    def test_e11_weakly_affected_e22_g12_strongly(self):
        # At 5% porosity the fiber-dominated E11 barely moves while the
        # matrix-dominated E22 and G12 take significant hits. This is the
        # whole reason this helper exists; if a future refactor inverts
        # those rates, this test must fail.
        Vp = 0.05
        C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
        S = np.linalg.inv(C)
        S_pristine = np.linalg.inv(self.pristine)
        # Engineering moduli come straight off the compliance diagonal.
        E11_loss = 1.0 - (1.0 / S[0, 0]) / (1.0 / S_pristine[0, 0])
        E22_loss = 1.0 - (1.0 / S[1, 1]) / (1.0 / S_pristine[1, 1])
        G12_loss = 1.0 - (1.0 / S[5, 5]) / (1.0 / S_pristine[5, 5])
        assert E11_loss < 0.01, f"E11 should be near-pristine, lost {E11_loss:.4f}"
        assert E22_loss > E11_loss * 5, (
            f"E22 loss {E22_loss:.4f} should be much larger than E11 loss {E11_loss:.4f}"
        )
        assert G12_loss > E11_loss * 5, (
            f"G12 loss {G12_loss:.4f} should be much larger than E11 loss {E11_loss:.4f}"
        )

    def test_monotonic_e22_degradation(self):
        Vp_list = [0.01, 0.03, 0.05, 0.08]
        E22_seq = []
        for Vp in Vp_list:
            C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
            S = np.linalg.inv(C)
            E22_seq.append(1.0 / S[1, 1])
        for a, b in zip(E22_seq, E22_seq[1:]):
            assert b < a, f"E22 should drop monotonically with Vp: got {E22_seq}"

    def test_monotonic_degradation(self):
        """E22 AND G12 must monotonically decrease with Vp (issue #48 item 2).

        E22 alone is not enough: a regression that boosts the shear-related
        stiffness terms while degrading the transverse-normal terms would
        slip past an E22-only test. Pin both engineering moduli on the
        exact Vp set called out in the issue."""
        Vp_list = [0.01, 0.03, 0.05, 0.10]
        E22_seq: list = []
        G12_seq: list = []
        for Vp in Vp_list:
            C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
            S = np.linalg.inv(C)
            E22_seq.append(1.0 / S[1, 1])
            G12_seq.append(1.0 / S[5, 5])
        for a, b in zip(E22_seq, E22_seq[1:]):
            assert b < a, f"E22 should drop monotonically with Vp: got {E22_seq}"
        for a, b in zip(G12_seq, G12_seq[1:]):
            assert b < a, f"G12 should drop monotonically with Vp: got {G12_seq}"

    def test_returned_stiffness_positive_definite(self):
        C = _degraded_composite_stiffness(0.05, (1, 1, 1), self.mat)
        eig = np.linalg.eigvalsh(C)
        assert np.all(eig > 0), f"degraded stiffness not positive-definite: {eig}"

    def test_all_finite(self):
        # Sweep through the regime where the lame-denominator guard
        # (lam_eff + mu_eff < 1e-12) could trip; outputs must stay finite.
        for Vp in [1e-10, 0.01, 0.10, 0.50, 0.85, 0.985]:
            C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
            assert np.all(np.isfinite(C)), f"non-finite C at Vp={Vp}"


class TestCLTDegradation:
    """Boundary tests for compute_degraded_clt_moduli (#12)."""

    def setup_method(self):
        from porosity_fe_analysis import compute_degraded_clt_moduli, \
            compute_degraded_clt_flexural_modulus
        self.compute_degraded_clt_moduli = compute_degraded_clt_moduli
        self.compute_degraded_clt_flexural_modulus = compute_degraded_clt_flexural_modulus
        self.material = MATERIALS['T800_epoxy']
        self.layup = [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]

    def test_pristine_at_zero_porosity(self):
        deg = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=0.0)
        # At Vp=0, degraded should be very close to nearly-zero-Vp baseline.
        baseline = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=1e-9)
        for key in ('Ex', 'Ey', 'Gxy'):
            assert abs(deg[key] - baseline[key]) / baseline[key] < 1e-3

    def test_moduli_decrease_with_porosity(self):
        low = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=0.01)
        high = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=0.10)
        for key in ('Ex', 'Ey', 'Gxy'):
            assert high[key] < low[key]

    def test_flexural_modulus_decreases_with_porosity(self):
        f_low = self.compute_degraded_clt_flexural_modulus(
            self.material, self.layup, Vp=0.0
        )['Ef_x']
        f_high = self.compute_degraded_clt_flexural_modulus(
            self.material, self.layup, Vp=0.05
        )['Ef_x']
        assert f_high < f_low
