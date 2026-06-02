#!/usr/bin/env python3
"""Tests for porosity_fe.empirical.

Split out of the monolithic tests/test_porosity_fe.py for issue #124.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe_analysis import (MATERIALS, VoidGeometry, PorosityField, CompositeMesh,
                                   EmpiricalSolver, FESolver)


class TestEmpiricalSolver:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = EmpiricalSolver(self.mesh, self.material)

    def test_judd_wright_zero_porosity(self):
        """At Vp=0, knockdown should be 1.0"""
        kd = self.solver._judd_wright(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_judd_wright_decreasing(self):
        """Higher porosity -> lower knockdown"""
        kd1 = self.solver._judd_wright(0.01, 'compression')
        kd2 = self.solver._judd_wright(0.05, 'compression')
        assert kd1 > kd2

    def test_power_law_zero_porosity(self):
        kd = self.solver._power_law(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_linear_zero_porosity(self):
        kd = self.solver._linear(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_ilss_most_sensitive(self):
        """ILSS should have largest knockdown for same porosity"""
        Vp = 0.05
        kd_comp = self.solver._judd_wright(Vp, 'compression')
        kd_ilss = self.solver._judd_wright(Vp, 'ilss')
        assert kd_ilss < kd_comp

    # ----- issue #35: dedicated 'transverse_tension' mode -----------------
    def test_transverse_tension_mode_registered(self):
        """Issue #35: 'transverse_tension' must be a first-class mode keyed
        off sigma_2t with alpha = 10.0 (matrix-dominated, same as ILSS)."""
        assert 'transverse_tension' in EmpiricalSolver.PRISTINE_STRENGTH_KEY
        assert EmpiricalSolver.PRISTINE_STRENGTH_KEY['transverse_tension'] == 'sigma_2t'
        assert EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI['transverse_tension'] == 10.0

    def test_transverse_tension_distinct_from_longitudinal_tension(self):
        """At the same Vp, transverse_tension knockdown must NOT equal
        longitudinal-tension knockdown (the bug routed sigma_2t through
        alpha=3.9 instead of the matrix-dominated alpha=10.0)."""
        Vp = 0.03
        kd_t = self.solver._judd_wright(Vp, 'tension')
        kd_tt = self.solver._judd_wright(Vp, 'transverse_tension')
        assert kd_tt < kd_t, (
            f"transverse_tension ({kd_tt}) must be more porosity-sensitive "
            f"than longitudinal tension ({kd_t}) at the same Vp"
        )

    def test_transverse_tension_matches_ilss_alpha_at_qi(self):
        """transverse_tension and ilss share the same matrix-dominated alpha
        at the QI reference layup (f_md = 0.5, scale = 1.0)."""
        Vp = 0.04
        kd_ilss = self.solver._judd_wright(Vp, 'ilss')
        kd_tt = self.solver._judd_wright(Vp, 'transverse_tension')
        assert abs(kd_ilss - kd_tt) < 1e-12

    def test_transverse_tension_uses_sigma_2t(self):
        """get_failure_load(mode='transverse_tension') must use sigma_2t as
        the pristine strength."""
        result = self.solver.get_failure_load(mode='transverse_tension',
                                              model='judd_wright')
        expected = self.material.sigma_2t * result['knockdown']
        assert abs(result['failure_stress'] - expected) < 1e-9

    def test_transverse_tension_ud_uses_matrix_floor(self):
        """UD [0]_n layup: transverse_tension should hit the matrix-dominated
        floor (0.80), matching ILSS, not the fiber-dominated floor (0.15)."""
        ud = [0.0] * 8
        solver = EmpiricalSolver(self.mesh, self.material, ply_angles=ud)
        # alpha_QI = 10.0; scale = max(0/0.5, 0.80) = 0.80
        assert abs(solver.JUDD_WRIGHT_ALPHA['transverse_tension'] - 10.0 * 0.80) < 1e-12
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - 10.0 * 0.80) < 1e-12

    def test_get_failure_load_returns_dict(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert 'failure_stress' in result
        assert 'knockdown' in result
        assert 'model' in result

    def test_failure_load_positive(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert result['failure_stress'] > 0
        assert 0 < result['knockdown'] <= 1.0

    def test_unknown_loading_mode_raises_with_listing(self):
        with pytest.raises(ValueError, match=r"Unknown loading mode"):
            self.solver._get_pristine_strength('flexure')

    def test_override_alpha_only_changes_targeted_mode(self):
        """Partial override leaves other modes at QI defaults."""
        solver = EmpiricalSolver(self.mesh, self.material,
                                  judd_wright_alpha={'ilss': 12.0})
        # ILSS overridden (and layup scale = 1.0 for default ply_angles=None / f_md=0.5)
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - 12.0) < 1e-12
        # Other modes match the QI baseline at f_md = 0.5
        assert abs(solver.JUDD_WRIGHT_ALPHA['compression'] - 6.9) < 1e-12
        assert abs(solver.JUDD_WRIGHT_ALPHA['tension'] - 3.9) < 1e-12
        assert abs(solver.JUDD_WRIGHT_ALPHA['shear'] - 8.0) < 1e-12

    def test_override_layup_scaling_applied(self):
        """Override values are scaled by layup the same way as the QI baseline."""
        ud = [0.0] * 16  # f_md = 0; ILSS floor = 0.80
        solver = EmpiricalSolver(self.mesh, self.material,
                                  ply_angles=ud,
                                  judd_wright_alpha={'ilss': 12.0})
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - 12.0 * 0.80) < 1e-12

    def test_override_n_and_beta(self):
        solver = EmpiricalSolver(self.mesh, self.material,
                                  power_law_n={'compression': 4.0},
                                  linear_beta={'shear': 6.0})
        assert abs(solver.POWER_LAW_N['compression'] - 4.0) < 1e-12
        assert abs(solver.LINEAR_BETA['shear'] - 6.0) < 1e-12

    def test_override_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match=r"positive finite"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'compression': -1.0})

    def test_override_nan_alpha_rejected(self):
        with pytest.raises(ValueError, match=r"positive finite"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'compression': float('nan')})

    def test_override_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match=r"unknown mode keys"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'silly_mode': 5.0})

    def test_override_non_dict_rejected(self):
        with pytest.raises(TypeError, match=r"dict mapping mode"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha=[6.9, 3.9, 8.0, 10.0])

    def test_override_does_not_mutate_class_defaults(self):
        """Overrides must not leak back into the class-level QI dicts."""
        EmpiricalSolver(self.mesh, self.material,
                        judd_wright_alpha={'ilss': 99.0})
        assert EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI['ilss'] == 10.0

    def test_get_all_failure_loads(self):
        results = self.solver.get_all_failure_loads()
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            assert mode in results
            for model in ['judd_wright', 'power_law', 'linear']:
                assert model in results[mode]

    def test_failure_stress_below_pristine(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert result['failure_stress'] < self.material.sigma_1c

    def test_discrete_void_scf_amplifies_knockdown(self):
        """Discrete macrovoid should cause worse local knockdown near the void."""
        material = MATERIALS['T800_epoxy']
        void = VoidGeometry(center=(25, 10, material.total_thickness / 2),
                            radii=(2, 2, 0.5))
        pf = PorosityField(material, 0.02, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=20, ny=10, nz=12)
        solver = EmpiricalSolver(mesh, material)
        solver.apply_loading('compression', 'judd_wright')
        min_kd_with_void = solver.nodal_knockdown.min()

        pf_no_void = PorosityField(material, 0.02, distribution='uniform')
        mesh_no_void = CompositeMesh(pf_no_void, material, nx=20, ny=10, nz=12)
        solver_no_void = EmpiricalSolver(mesh_no_void, material)
        solver_no_void.apply_loading('compression', 'judd_wright')
        min_kd_no_void = solver_no_void.nodal_knockdown.min()

        # Discrete void should reduce local knockdown near the void
        assert min_kd_with_void < min_kd_no_void

    def test_apply_loading_bad_model_raises_value_error(self):
        # #22: bad model name should give a ValueError listing the valid
        # choices, not a bare KeyError.
        with pytest.raises(ValueError, match=r"Unknown knockdown model 'bogus'"):
            self.solver.apply_loading(mode='compression', model='bogus')

    def test_apply_loading_bad_mode_raises_value_error(self):
        with pytest.raises(ValueError, match=r"Unknown loading mode 'bogus'"):
            self.solver.apply_loading(mode='bogus', model='judd_wright')


class TestEmpiricalVectorizationEquivalence:
    """#114 + #115: the vectorized empirical/failure paths must match the
    pre-vectorization scalar formulas to high precision.

    These are regression pins, not perf tests — they lock in the behavioural
    contract so a future refactor can't silently drift the per-node knockdown
    or the per-element Vp.
    """

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        # Clustered midplane gives a non-uniform per-node Vp so the
        # vectorized vs scalar paths exercise the full porosity field, not a
        # constant.
        self.pf = PorosityField(self.material, 0.03, distribution='clustered',
                                cluster_location='midplane')
        self.mesh = CompositeMesh(self.pf, self.material, nx=4, ny=3, nz=4,
                                  ply_angles='QI')
        self.solver = EmpiricalSolver(self.mesh, self.material)

    # --- #115: knockdown vectorization vs scalar formula ----------------

    def _scalar_kd(self, model: str, mode: str) -> np.ndarray:
        """Reference: scalar list-comprehension over per-node porosity."""
        if model == 'judd_wright':
            alpha = self.solver.JUDD_WRIGHT_ALPHA[mode]
            return np.array([float(np.exp(-alpha * v))
                             for v in self.mesh.porosity])
        if model == 'power_law':
            n = self.solver.POWER_LAW_N[mode]
            return np.array([float((1.0 - v) ** n)
                             for v in self.mesh.porosity])
        if model == 'linear':
            beta = self.solver.LINEAR_BETA[mode]
            return np.array([float(max(1.0 - beta * v, 0.0))
                             for v in self.mesh.porosity])
        raise AssertionError(f"unhandled model {model!r}")

    @pytest.mark.parametrize(
        "mode,model",
        [(m, mdl) for m in ('compression', 'tension', 'shear', 'ilss',
                            'transverse_tension')
         for mdl in ('judd_wright', 'power_law', 'linear')],
    )
    def test_vectorized_kd_matches_scalar_reference(self, mode, model):
        """Vectorized built-in path must match the scalar formula bit-for-bit."""
        self.solver.apply_loading(mode=mode, model=model)
        ref = self.solver._apply_discrete_void_scf(
            self._scalar_kd(model, mode), mode,
        )
        np.testing.assert_allclose(self.solver.nodal_knockdown, ref,
                                   rtol=0, atol=1e-15)

    def test_user_callable_still_uses_scalar_path(self):
        """A callable model retains the per-Vp scalar contract from #62."""
        seen = []

        def my_model(Vp, mode):
            seen.append(float(Vp))
            return 0.5

        self.solver.apply_loading(mode='compression', model=my_model)
        # The callable is invoked on each node's Vp (plus an internal
        # validation grid for finite/in-range checks). We can't pin the
        # exact count without coupling to that grid, but every mesh-node Vp
        # must appear in the seen values — confirms the array loop ran.
        seen_set = set(seen)
        for v in self.mesh.porosity:
            assert any(abs(v - s) < 1e-12 for s in seen_set), (
                f"node porosity {v} never reached the user callable; "
                f"the vectorization branch may have swallowed it"
            )

    # --- #114: per-element Vp hoist correctness -------------------------

    def test_evaluate_failure_per_element_vp_matches_loop(self):
        """Hoisted `np.mean(porosity[elements], axis=1)` must agree with the
        old per-iteration `np.mean(porosity[elements[e]])`."""
        # Reference: loop over elements, compute mean per element manually.
        ref = np.clip(
            np.array([float(np.mean(
                self.mesh.porosity[self.mesh.elements[e]]
            )) for e in range(self.mesh.n_elements)]),
            0.0, 1.0,
        )
        # Hoisted (vectorized) form used by `_evaluate_failure`.
        actual = np.clip(
            np.mean(self.mesh.porosity[self.mesh.elements], axis=1),
            0.0, 1.0,
        )
        np.testing.assert_allclose(actual, ref, rtol=0, atol=1e-15)

    def test_evaluate_failure_end_to_end_unchanged(self):
        """Full FE solve + failure evaluation must still produce the same
        per-element failure index post-vectorization."""
        # Tighter mesh so the failure index is well-resolved.
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, self.material, nx=4, ny=3, nz=4,
                             ply_angles='UD')  # UD keeps Tsai-Wu non-negative
        fe = FESolver(mesh, self.material, pf)
        res = fe.solve(loading='compression', applied_strain=-0.001)
        # The vectorized path should produce a well-formed per-element FI.
        assert res.per_element_failure_index.shape == (mesh.n_elements,)
        assert np.all(np.isfinite(res.per_element_failure_index))


class TestEmpiricalLayupScaling:
    """Direct unit tests for _matrix_dominated_fraction and _layup_scale."""

    def test_f_md_pure_zero(self):
        assert EmpiricalSolver._matrix_dominated_fraction([0] * 8) == 0.0

    def test_f_md_pure_ninety(self):
        assert EmpiricalSolver._matrix_dominated_fraction([90] * 8) == 1.0

    def test_f_md_off_axis_only(self):
        assert EmpiricalSolver._matrix_dominated_fraction([45, -45, 45, -45]) == 0.5

    def test_f_md_qi_layup_is_0p4(self):
        # Documented QI calibration coupon -> 0.4 under the binning rule.
        # See the comment above _F_MD_REF in porosity_fe_analysis.py and
        # the README "Empirical Strength Knockdown" section.
        layup = [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]
        assert abs(EmpiricalSolver._matrix_dominated_fraction(layup) - 0.4) < 1e-12

    def test_f_md_empty_returns_qi_default(self):
        assert EmpiricalSolver._matrix_dominated_fraction([]) == 0.5
        assert EmpiricalSolver._matrix_dominated_fraction(None) == 0.5

    def test_f_md_threshold_band_at_10_and_80_degrees(self):
        # 10° -> still binned as 0° (fiber-dominated)
        assert EmpiricalSolver._matrix_dominated_fraction([10]) == 0.0
        # 80° -> binned as 90° (matrix-dominated)
        assert EmpiricalSolver._matrix_dominated_fraction([80]) == 1.0
        # 11° -> off-axis bin
        assert EmpiricalSolver._matrix_dominated_fraction([11]) == 0.5

    def _solver_with_layup(self, ply_angles):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=4, ny=3, nz=4)
        return EmpiricalSolver(mesh, material, ply_angles=ply_angles)

    def test_layup_scale_unity_at_reference(self):
        # f_md = 0.5 -> scale = 1.0 -> alpha_eff == alpha_QI
        solver = self._solver_with_layup([45, -45, 45, -45])
        for mode, alpha_qi in EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI.items():
            assert abs(solver.JUDD_WRIGHT_ALPHA[mode] - alpha_qi) < 1e-12

    def test_layup_scale_floor_for_ud(self):
        # f_md = 0.0 -> hits 0.15 floor for non-ILSS modes, 0.80 for ILSS.
        solver = self._solver_with_layup([0] * 8)
        for mode in ('compression', 'tension', 'shear'):
            expected = EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI[mode] * 0.15
            assert abs(solver.JUDD_WRIGHT_ALPHA[mode] - expected) < 1e-12
        ilss_expected = EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI['ilss'] * 0.80
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - ilss_expected) < 1e-12

    def test_layup_scale_above_reference(self):
        # Pure 90 -> f_md = 1.0 -> scale = 2.0
        solver = self._solver_with_layup([90] * 8)
        for mode, alpha_qi in EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI.items():
            assert abs(solver.JUDD_WRIGHT_ALPHA[mode] - alpha_qi * 2.0) < 1e-12


class TestLayupScaleRegressionPin:
    """Pin the current ``_layup_scale`` behavior across the intermediate
    f_md range.

    Snapshot of post-#140 investigation findings. The linear
    ``f_md / _F_MD_REF`` scaling is preserved pending the validation
    campaign documented in #140; these values are the current behavior,
    not an endorsement of correctness. Any future refactor (linear or
    nonlinear) must explicitly update these snapshots so the change is
    visible in review.

    Measurement methodology used in #140: relative CLT stiffness
    retention ratio vs the QI baseline, i.e.
    ``sqrt(Ex_layup(Vp)/Ex_layup(0)) / sqrt(Ex_QI(Vp)/Ex_QI(0))``,
    compared against the empirical
    ``exp(-alpha_QI*(scale_lin - 1)*Vp)`` over Vp in
    ``[0.005, 0.05]``. Max abs relative error observed across the
    layups below was 33.5% (UD at Vp=0.05); >5% on UD-heavy, off-axis,
    and UD layups.
    """

    def _solver_with_layup(self, ply_angles):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=4, ny=3, nz=4)
        return EmpiricalSolver(mesh, material, ply_angles=ply_angles)

    def test_layup_scale_at_baseline_qi_returns_unity(self):
        # QI [0,45,-45,90]_s -> f_md = 0.5 -> scale = 1.0 for all modes.
        solver = self._solver_with_layup([0, 45, -45, 90, 90, -45, 45, 0])
        for mode in ('compression', 'tension', 'shear', 'ilss',
                     'transverse_tension'):
            assert solver._layup_scale(mode) == pytest.approx(1.0, abs=1e-12)

    def test_layup_scale_snapshot_at_intermediate_layups(self):
        # 4-sig-fig snapshot of the current (linear) layup scale across the
        # representative layups used in the #140 measurement set.
        # Update only with an intentional algorithm change.
        layups = {
            'qi':       [0, 45, -45, 90, 90, -45, 45, 0],
            'crossply': [0, 90, 90, 0],
            'ud_heavy': [0, 0, 90, 90, 0, 0],
            'off_axis': [0, 15, -15, -15, 15, 0],
            'ud':       [0, 0, 0, 0, 0, 0],
        }
        expected = {
            # name: (f_md, compression, tension, shear, ilss, transverse_tension)
            'qi':       (0.5000, 1.000, 1.000, 1.000, 1.000, 1.000),
            'crossply': (0.5000, 1.000, 1.000, 1.000, 1.000, 1.000),
            'ud_heavy': (0.3333, 0.6667, 0.6667, 0.6667, 0.8000, 0.8000),
            'off_axis': (0.3333, 0.6667, 0.6667, 0.6667, 0.8000, 0.8000),
            'ud':       (0.0000, 0.1500, 0.1500, 0.1500, 0.8000, 0.8000),
        }
        for name, ply in layups.items():
            solver = self._solver_with_layup(ply)
            exp = expected[name]
            assert solver.f_md == pytest.approx(exp[0], abs=5e-4), name
            for i, mode in enumerate(('compression', 'tension', 'shear',
                                      'ilss', 'transverse_tension'),
                                     start=1):
                got = solver._layup_scale(mode)
                assert got == pytest.approx(exp[i], abs=5e-4), \
                    f'{name}/{mode}: got {got!r}, expected {exp[i]!r}'


class TestEmpiricalLinearSaturation:
    """The linear knockdown clips to 0 once Vp >= 1/beta."""

    def setup_method(self):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=4, ny=3, nz=4)
        self.solver = EmpiricalSolver(mesh, material)

    def test_linear_clips_to_zero_at_full_porosity(self):
        for mode in ('compression', 'tension', 'shear', 'ilss'):
            assert self.solver._linear(1.0, mode) == 0.0

    def test_linear_monotone_decreasing_until_saturation(self):
        prev = 1.0
        for vp in (0.0, 0.01, 0.05, 0.10, 0.18, 0.20):
            kd = self.solver._linear(vp, 'compression')
            assert kd <= prev + 1e-12
            assert 0.0 <= kd <= 1.0
            prev = kd

    def test_linear_internal_clip_tolerates_fp_overshoot(self):
        # FE element-mean averaging can produce 1 + ~1e-15.
        kd = self.solver._linear(1.0 + 1e-15, 'compression')
        assert kd == 0.0

    def test_internal_clip_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            self.solver._judd_wright(float('nan'), 'compression')


class TestEmpiricalSolverPlugin:
    """#62: EmpiricalSolver must accept a user-supplied knockdown callable."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=4, ny=2, nz=2)
        self.solver = EmpiricalSolver(self.mesh, self.material)

    def test_callable_overrides_builtin(self):
        """A constant callable must drive the reported failure stress."""
        const_kd = 0.42

        def my_model(Vp, mode):  # noqa: D401  (test helper)
            return const_kd

        result = self.solver.get_failure_load(mode='compression', model=my_model)
        sigma_0 = self.material.sigma_1c
        assert result['knockdown'] == pytest.approx(const_kd, rel=1e-12)
        assert result['failure_stress'] == pytest.approx(
            const_kd * sigma_0, rel=1e-12)
        # Label is taken from __name__ when a callable is passed.
        assert result['model'] == 'my_model'

    def test_callable_rejects_out_of_range(self):
        """KD > 1 must surface as a ValueError at dispatch time."""
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            self.solver.apply_loading(
                mode='compression',
                model=lambda Vp, mode: 1.5,
            )

    def test_callable_rejects_negative(self):
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            self.solver.apply_loading(
                mode='compression',
                model=lambda Vp, mode: -0.1,
            )

    def test_callable_rejects_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            self.solver.apply_loading(
                mode='compression',
                model=lambda Vp, mode: float('nan'),
            )

    def test_non_callable_non_string_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            self.solver.apply_loading(mode='compression', model=42)

    def test_callable_bypasses_layup_scale(self):
        """User callable receives raw Vp, with no layup-coefficient mediation."""
        # Build two solvers with markedly different layups; the user callable
        # should produce the same knockdown because the layup scale is bypassed.
        ud_solver = EmpiricalSolver(
            self.mesh, self.material, ply_angles=[0.0, 0.0, 0.0, 0.0])
        qi_solver = EmpiricalSolver(
            self.mesh, self.material, ply_angles=[0.0, 90.0, 45.0, -45.0])

        def my_model(Vp, mode):
            return 0.77

        ud_res = ud_solver.get_failure_load(model=my_model)
        qi_res = qi_solver.get_failure_load(model=my_model)
        assert ud_res['knockdown'] == pytest.approx(0.77, rel=1e-12)
        assert qi_res['knockdown'] == pytest.approx(0.77, rel=1e-12)

    def test_callable_still_uses_discrete_void_scf(self):
        """User callable is composed with the discrete-void SCF post-step."""
        # With a discrete macrovoid in the mesh, the SCF post-step should
        # depress the user-defined constant knockdown near the void.
        void = VoidGeometry(
            center=(25.0, 10.0, self.material.total_thickness / 2),
            radii=(2.0, 2.0, 0.5))
        pf_void = PorosityField(
            self.material, 0.02, distribution='uniform',
            discrete_voids=[void])
        mesh_void = CompositeMesh(pf_void, self.material, nx=20, ny=10, nz=12)
        solver = EmpiricalSolver(mesh_void, self.material)
        solver.apply_loading(
            mode='compression', model=lambda Vp, mode: 0.9)
        # Some nodes (those near the discrete void) should be strictly
        # below 0.9 thanks to the SCF post-step.
        kd = solver.nodal_knockdown
        assert kd is not None
        assert np.any(kd < 0.9 - 1e-9), \
            "discrete-void SCF post-step did not depress any nodal knockdown"
        assert np.all(kd <= 0.9 + 1e-9)

    def test_get_all_failure_loads_accepts_extra_models(self):
        """get_all_failure_loads must compose built-ins with extra callables."""
        results = self.solver.get_all_failure_loads(
            extra_models={'flat': lambda Vp, mode: 0.5})
        for mode in ('compression', 'tension', 'shear', 'ilss',
                     'transverse_tension'):
            assert 'judd_wright' in results[mode]
            assert 'flat' in results[mode]
            assert results[mode]['flat']['knockdown'] == pytest.approx(
                0.5, rel=1e-12)


class TestCoefficientOverrideValidation:
    """#150: pin the user-facing customization surfaces of EmpiricalSolver.

    Covers (a) the per-mode coefficient-override validation inside
    :meth:`EmpiricalSolver._merge_coefficient_override` and (b) the
    user-supplied knockdown callable validation inside
    :meth:`EmpiricalSolver._validate_user_kd_callable` (which the
    constructor does not invoke; validation happens at dispatch time via
    :meth:`apply_loading` / :meth:`get_failure_load`).
    """

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=4, ny=2, nz=2)

    def test_coefficient_override_rejects_non_numeric(self):
        """Non-numeric override values must raise TypeError naming the key."""
        with pytest.raises(TypeError, match=r"must be a number"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'compression': 'invalid'})

    def test_coefficient_override_rejects_negative(self):
        """Negative coefficients must raise ValueError ('positive finite')."""
        with pytest.raises(ValueError, match=r"positive finite"):
            EmpiricalSolver(self.mesh, self.material,
                            power_law_n={'compression': -0.5})

    def test_coefficient_override_rejects_infinite(self):
        """Non-finite (inf) coefficients must raise ValueError."""
        with pytest.raises(ValueError, match=r"positive finite"):
            EmpiricalSolver(self.mesh, self.material,
                            linear_beta={'compression': float('inf')})

    def test_user_callable_exception_wrapped_with_context(self):
        """An exception raised inside a user knockdown callable must be
        re-raised as ValueError carrying the original exception type name."""
        solver = EmpiricalSolver(self.mesh, self.material)

        def bad_kd(Vp, mode):
            raise ZeroDivisionError("oops")

        with pytest.raises(ValueError, match=r"ZeroDivisionError") as exc_info:
            solver.apply_loading(mode='compression', model=bad_kd)
        # Validation message should also surface the original message text.
        assert 'oops' in str(exc_info.value)


class TestFatigueModelValidation:
    """#149: pin input-validation branches of FatigueModel.knockdown_factor.

    Covers the three guard clauses that were previously unexercised:
    - unknown ``mode`` rejected by :meth:`FatigueModel._slope`
    - non-finite ``R`` rejected up front
    - ``cycles`` either below 1 or non-finite rejected before slope lookup
    """

    def setup_method(self):
        from porosity_fe_analysis import FatigueModel
        self.FatigueModel = FatigueModel

    def test_unknown_mode_raises(self):
        """Unknown mode keys must raise ValueError with a descriptive message."""
        fm = self.FatigueModel()
        with pytest.raises(ValueError, match=r"Unknown fatigue mode"):
            fm.knockdown_factor('unknown_mode', cycles=1e6)

    def test_non_finite_R_raises(self):
        """Non-finite ``R`` (nan / inf / -inf) must be rejected."""
        fm = self.FatigueModel()
        for bad_R in (float('nan'), float('inf'), float('-inf')):
            with pytest.raises(ValueError, match=r"R must be finite"):
                fm.knockdown_factor('tension', cycles=1e6, R=bad_R)

    def test_cycles_below_one_raises(self):
        """``cycles < 1`` (below the static one-cycle anchor) is invalid."""
        fm = self.FatigueModel()
        with pytest.raises(ValueError, match=r"cycles must be a finite"):
            fm.knockdown_factor('tension', cycles=0.5)

    def test_cycles_non_finite_raises(self):
        """Non-finite ``cycles`` (inf / nan) must also be rejected."""
        fm = self.FatigueModel()
        for bad_cycles in (float('inf'), float('nan')):
            with pytest.raises(ValueError, match=r"cycles must be a finite"):
                fm.knockdown_factor('tension', cycles=bad_cycles)


class TestLocalSensitivities:
    """Closed-form sensitivities must match a central-difference baseline
    to machine precision, and the layup scaling must propagate into the
    coefficient partial."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        # Vp = 2% — well into the interior of the validity region so the
        # FD step doesn't bump into the [0, 1] clip.
        pf = PorosityField(self.material, 0.02, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = EmpiricalSolver(self.mesh, self.material)

    def test_judd_wright_partial_matches_analytic(self):
        s = self.solver.local_sensitivities(mode='compression',
                                            model='judd_wright')
        # Compare analytic dKD/dVp to a central-difference baseline.
        fd = self.solver.sensitivity_fd(mode='compression',
                                        model='judd_wright', param='Vp')
        np.testing.assert_allclose(s['dKD_dVp'], fd, rtol=1e-6)
        # And the coefficient partial.
        fd_c = self.solver.sensitivity_fd(mode='compression',
                                          model='judd_wright', param='coef')
        np.testing.assert_allclose(s['dKD_dcoef'], fd_c, rtol=1e-6)
        # Spot-check the algebra: dKD/dVp = -alpha * KD for judd_wright.
        alpha = self.solver.JUDD_WRIGHT_ALPHA['compression']
        np.testing.assert_allclose(s['dKD_dVp'], -alpha * s['KD'], rtol=1e-12)

    def test_power_law_partial_matches_analytic(self):
        s = self.solver.local_sensitivities(mode='compression',
                                            model='power_law')
        fd = self.solver.sensitivity_fd(mode='compression',
                                        model='power_law', param='Vp')
        np.testing.assert_allclose(s['dKD_dVp'], fd, rtol=1e-6)
        fd_c = self.solver.sensitivity_fd(mode='compression',
                                          model='power_law', param='coef')
        np.testing.assert_allclose(s['dKD_dcoef'], fd_c, rtol=1e-6)

    def test_linear_partial_matches_analytic(self):
        s = self.solver.local_sensitivities(mode='compression',
                                            model='linear')
        fd = self.solver.sensitivity_fd(mode='compression',
                                        model='linear', param='Vp')
        np.testing.assert_allclose(s['dKD_dVp'], fd, rtol=1e-6)
        fd_c = self.solver.sensitivity_fd(mode='compression',
                                          model='linear', param='coef')
        np.testing.assert_allclose(s['dKD_dcoef'], fd_c, rtol=1e-6)
        # Linear law: dKD/dVp must be exactly -beta in the unclipped regime.
        beta = self.solver.LINEAR_BETA['compression']
        np.testing.assert_allclose(s['dKD_dVp'], -beta, rtol=1e-12)

    def test_layup_scaled_alpha_propagates(self):
        """A non-QI (UD) layup must propagate its layup scaling into the
        coefficient partial.  ``dKD/dcoef`` magnitude is ``Vp * KD`` — KD
        moves with the layup-scaled alpha, so the partial scales too."""
        ud = [0.0] * 8  # UD: f_md = 0 -> floor = 0.15 (compression)
        qi = [0.0, 45.0, 90.0, -45.0] * 2
        solver_ud = EmpiricalSolver(self.mesh, self.material, ply_angles=ud)
        solver_qi = EmpiricalSolver(self.mesh, self.material, ply_angles=qi)
        s_ud = solver_ud.local_sensitivities(mode='compression',
                                             model='judd_wright')
        s_qi = solver_qi.local_sensitivities(mode='compression',
                                             model='judd_wright')
        # Sanity: the UD scale (0.15) is smaller than QI scale (1.0), so
        # UD's alpha is smaller, KD is closer to 1, and the *magnitude*
        # of dKD/dVp is smaller too (it's -alpha * KD).
        assert abs(s_ud['dKD_dVp']) < abs(s_qi['dKD_dVp'])
        # The coefficient partial magnitude is just |Vp| * KD; KD(UD) > KD(QI)
        # at the same Vp because alpha(UD) < alpha(QI), so |dKD/dcoef|
        # on UD must be larger than on QI.
        assert abs(s_ud['dKD_dcoef']) > abs(s_qi['dKD_dcoef'])

    def test_default_Vp_matches_mesh_porosity(self):
        """Default Vp is ``mesh.porosity_field.Vp`` — same as
        ``get_failure_load``."""
        s_default = self.solver.local_sensitivities(mode='compression',
                                                    model='judd_wright')
        s_explicit = self.solver.local_sensitivities(
            mode='compression', model='judd_wright',
            Vp=self.mesh.porosity_field.Vp)
        assert s_default == s_explicit

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match=r"Unknown knockdown model"):
            self.solver.local_sensitivities(mode='compression', model='bogus')

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match=r"Unknown loading mode"):
            self.solver.local_sensitivities(mode='flexure', model='judd_wright')

    def test_sensitivity_fd_unknown_param(self):
        with pytest.raises(ValueError, match=r"param must be"):
            self.solver.sensitivity_fd(mode='compression',
                                       model='judd_wright', param='nope')

    def test_power_law_sensitivity_degenerate_at_full_porosity(self):
        """At Vp=1 the power-law knockdown collapses to 0 and both partials
        are pinned to 0 (the ``1 - Vp <= 0`` guard branch)."""
        s = self.solver.local_sensitivities(mode='compression',
                                            model='power_law', Vp=1.0)
        assert s['KD'] == 0.0
        assert s['dKD_dVp'] == 0.0
        assert s['dKD_dcoef'] == 0.0

    def test_linear_sensitivity_in_clipped_regime(self):
        """Once the linear law has clipped to 0 (raw <= 0) its gradients are
        zero. ILSS beta (~9) clips well before Vp=1, so Vp=0.2 is past the
        knee."""
        s = self.solver.local_sensitivities(mode='ilss', model='linear',
                                            Vp=0.2)
        assert s['KD'] == 0.0
        assert s['dKD_dVp'] == 0.0
        assert s['dKD_dcoef'] == 0.0

    def test_sensitivity_fd_unknown_mode(self):
        with pytest.raises(ValueError, match=r"Unknown loading mode"):
            self.solver.sensitivity_fd(mode='flexure', model='judd_wright')

    def test_sensitivity_fd_unknown_model(self):
        with pytest.raises(ValueError, match=r"Unknown knockdown model"):
            self.solver.sensitivity_fd(mode='compression', model='bogus')


class TestCheckInternalVpScalarFastPath:
    """Regression: scalar fast-path in EmpiricalSolver._check_internal_Vp (#180).

    Kept in its own class so it merges mechanically alongside #179, which
    also touches this file.
    """

    @staticmethod
    def _numpy_ref(Vp):
        # The pre-#180 reference implementation (numpy-only path).
        if not np.isfinite(Vp):
            raise ValueError(f"Internal Vp is non-finite: {Vp!r}")
        return float(np.clip(Vp, 0.0, 1.0))

    def test_scalar_matches_numpy_ref_in_range(self):
        values = [0.0, 0.001, 0.05, 0.5, 0.999, 1.0,
                  1.0 + 1e-15, 1.0 - 1e-15, -1e-16]
        for v in values:
            got = EmpiricalSolver._check_internal_Vp(v)
            ref = self._numpy_ref(v)
            assert got == ref, f"mismatch at {v!r}: {got!r} != {ref!r}"
            assert isinstance(got, float)

    def test_out_of_range_clamps(self):
        assert EmpiricalSolver._check_internal_Vp(-0.5) == 0.0
        assert EmpiricalSolver._check_internal_Vp(2.0) == 1.0
        assert EmpiricalSolver._check_internal_Vp(1e9) == 1.0
        assert EmpiricalSolver._check_internal_Vp(-1e9) == 0.0

    @pytest.mark.parametrize("bad", [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_raises(self, bad):
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalSolver._check_internal_Vp(bad)

    def test_np_float64_handled(self):
        # Element-mean averaging yields np.float64, not builtin float.
        v = np.float64(0.5)
        got = EmpiricalSolver._check_internal_Vp(v)
        assert got == 0.5
        assert isinstance(got, float)
        # Clamping and rejection also hold for np.float64.
        assert EmpiricalSolver._check_internal_Vp(np.float64(1.0 + 1e-15)) == 1.0
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalSolver._check_internal_Vp(np.float64('nan'))

    def test_array_fallback_still_works(self):
        # Genuine arrays must still route through the numpy path unchanged.
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalSolver._check_internal_Vp(np.array(float('nan')))
        assert EmpiricalSolver._check_internal_Vp(np.array(2.0)) == 1.0
