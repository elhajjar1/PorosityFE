#!/usr/bin/env python3
"""Tests for porosity_fe.fe.refinement: the mesh-convergence study.

The convergence arithmetic (observed order, Richardson extrapolation, GCI,
classification) is checked on synthetic sequences with known answers; the
study itself on a 4-ply laminate with meshes of a few hundred elements, so
the whole module runs in seconds. Floats are compared with tolerances.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import math
import weakref

import pytest

import porosity_fe
from porosity_fe import (MATERIALS, FESolver, PorosityField, QuantityConvergence,
                         RefinementLevel, RefinementStudy, refinement_study)
from porosity_fe.fe import refinement as refinement_module
from porosity_fe.fe.refinement import (QUANTITY_KINDS, _assess, _estimate_memory_bytes,
                                       _observed_order, _peak_factorizations,
                                       _plan_resolutions, _three_level_trend)

LAYUP = [0.0, 90.0, 90.0, 0.0]
#: A thin 4-ply cross-ply with representative T800/epoxy lamina CTEs, so a
#: ply-resolving mesh has four element layers.
MAT4 = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4, t_ply=0.25,
                           alpha_1=-0.1e-6, alpha_2=31e-6)
#: Coarsest small mesh: elements short enough (aspect ratio <= 20) and an
#: interior region that holds elements at every level.
BASE = (12, 8, 4)


def _study(loading='compression', **kw):
    kw.setdefault('base_resolution', BASE)
    kw.setdefault('ply_angles', LAYUP)
    kw.setdefault('porosity_config', {'distribution': 'clustered'})
    return refinement_study(MAT4, kw.pop('vp', 0.03), loading, **kw)


class _NoSolver(FESolver):
    """Stands in for FESolver where no level may be solved."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("a level was solved")


@pytest.fixture(scope='module')
def compression_study():
    return _study('compression')


# ----------------------------------------------------------------------
# Convergence arithmetic
# ----------------------------------------------------------------------

class TestObservedOrderAndRichardson:
    @pytest.mark.parametrize('p_true', [1.0, 2.0, 3.0])
    def test_known_order_with_unequal_ratios(self, p_true):
        # f(h) = f0 + C h^p on h = 1, 1/1.5, 1/2 (ratios 1.5 then 4/3).
        f0, C = 0.75, 0.2
        h = (1.0, 1.0 / 1.5, 0.5)
        f = [f0 + C * x ** p_true for x in h]
        behavior, p, extrap = _three_level_trend(f, h)
        assert behavior == 'monotone'
        assert p == pytest.approx(p_true, abs=1e-8)
        assert extrap == pytest.approx(f0, rel=1e-10)

    def test_equal_ratios_reduce_to_closed_form(self):
        h = (4.0, 2.0, 1.0)
        f = [3.0 - 0.5 * x ** 2 for x in h]           # approaches from below
        e32, e21 = f[0] - f[1], f[1] - f[2]
        assert _observed_order(e32 / e21, 2.0, 2.0) == pytest.approx(
            math.log(e32 / e21) / math.log(2.0), rel=1e-10)
        behavior, p, extrap = _three_level_trend(f, h)
        assert (behavior, p, extrap) == ('monotone', pytest.approx(2.0),
                                         pytest.approx(3.0))

    def test_oscillating_sequence_is_non_monotone_and_not_extrapolated(self):
        behavior, p, extrap = _three_level_trend([1.0, 0.9, 0.95], (1.0, 0.5, 0.25))
        assert behavior == 'non-monotone'
        assert p is None and extrap is None

    @pytest.mark.parametrize('f', [
        lambda h: 1.0 + h ** -0.5,          # power singularity: changes grow
        lambda h: -math.log(h),             # log singularity: changes constant
    ])
    def test_singular_sequences_diverge(self, f):
        h = (1.0, 0.5, 0.25)
        behavior, p, extrap = _three_level_trend([f(x) for x in h], h)
        assert behavior == 'diverging'
        assert p is None and extrap is None

    def test_constant_sequence_is_mesh_independent(self):
        assert _three_level_trend([2.0, 2.0, 2.0], (1.0, 0.5, 0.25)) == \
            ('mesh-independent', None, 2.0)

    def test_sizes_must_decrease(self):
        with pytest.raises(ValueError, match='decrease'):
            _three_level_trend([1.0, 2.0, 3.0], (0.5, 1.0, 0.25))


class TestAssess:
    H = (1.0, 1.0 / 1.5, 0.5)

    def test_converging_quantity_has_order_extrapolate_and_gci(self):
        f = [10.0 + 2.0 * h ** 2 for h in self.H]
        c = _assess('effective_modulus', f, self.H, tol=0.01)
        assert c.behavior == 'monotone'
        assert c.observed_order == pytest.approx(2.0, abs=1e-8)
        assert c.extrapolated == pytest.approx(10.0, rel=1e-10)
        r21 = self.H[1] / self.H[2]
        gci = 1.25 * abs((f[2] - f[1]) / f[2]) / (r21 ** 2 - 1.0)
        assert c.gci == pytest.approx(gci, rel=1e-8)
        assert c.error_estimate == c.gci
        assert c.converged is (gci <= 0.01)
        assert c.relative_changes == pytest.approx(
            [(f[1] - f[0]) / f[1], (f[2] - f[1]) / f[2]], rel=1e-12)

    def test_knockdown_is_judged_on_the_stiffness_loss(self):
        # 1 - KD = 0.01 + 0.002 h^2: KD changes by < 0.01 %, its deficit by
        # several percent, which is what a 1 % tolerance must see.
        kd = [1.0 - (0.01 + 0.002 * h ** 2) for h in self.H]
        c = _assess('knockdown', kd, self.H, tol=0.01)
        d = [1.0 - k for k in kd]
        assert c.relative_changes[-1] == pytest.approx((d[2] - d[1]) / d[2], rel=1e-9)
        assert c.extrapolated == pytest.approx(1.0 - 0.01, rel=1e-9)
        assert c.converged is False

    def test_oscillation_uses_the_larger_change(self):
        c = _assess('interior_max_failure_index', [1.0, 1.02, 1.01], self.H, tol=0.05)
        assert c.behavior == 'non-monotone'
        assert c.extrapolated is None and c.observed_order is None
        assert c.error_estimate == pytest.approx(0.02 / 1.02, rel=1e-12)
        assert c.converged is True

    def test_diverging_quantity_is_mesh_dependent(self):
        f = [1.0 + h ** -0.5 for h in self.H]
        c = _assess('max_failure_index', f, self.H, tol=0.01)
        assert c.behavior == 'diverging'
        assert c.converged is False and c.error_estimate is None
        assert c.verdict == 'mesh-dependent'

    def test_slow_order_is_not_extrapolated(self):
        f = [1.0 + 0.5 * h ** 0.2 for h in self.H]
        c = _assess('max_failure_index', f, self.H, tol=0.01)
        assert c.behavior == 'monotone'
        assert c.observed_order == pytest.approx(0.2, abs=1e-6)
        assert c.extrapolated is None
        assert c.verdict == 'mesh-dependent'

    def test_slow_but_small_changes_are_not_called_mesh_dependent(self):
        # Changes of 0.5 % and 0.4 %: too slow to extrapolate, so not
        # converged at 1 %, but no sign of a singularity either.
        f = [1.0 + 0.05 * h ** 0.3 for h in self.H]
        c = _assess('knockdown', [1.0 - x for x in f], self.H, tol=0.01)
        assert c.behavior == 'monotone'
        assert c.observed_order == pytest.approx(0.3, abs=1e-6)
        assert c.extrapolated is None
        assert c.gci > 0.01 and c.converged is False
        assert c.verdict == 'not converged'

    def test_changes_below_the_noise_floor_are_mesh_independent(self):
        f = [5.0, 5.0 * (1 + 2e-4), 5.0 * (1 + 7e-4)]    # below tol / 10
        c = _assess('effective_modulus', f, self.H, tol=0.01)
        assert c.behavior == 'mesh-independent'
        assert c.converged is True and c.error_estimate == 0.0

    def test_missing_value_is_unavailable(self):
        c = _assess('effective_modulus', [1.0, None, 1.0], self.H, tol=0.01)
        assert c.behavior == 'unavailable'
        assert c.converged is None and c.verdict == 'n/a'
        c = _assess('knockdown', [math.nan] * 3, self.H, tol=0.01)
        assert c.behavior == 'unavailable' and c.values == (None, None, None)

    def test_two_levels_judge_the_last_change(self):
        c = _assess('effective_modulus', [1.0, 1.005], self.H[1:], tol=0.01)
        assert c.behavior == 'two levels'
        assert c.observed_order is None and c.extrapolated is None
        assert c.converged is True


# ----------------------------------------------------------------------
# Level planning and guards
# ----------------------------------------------------------------------

class TestPlanning:
    def test_in_plane_keeps_nz(self):
        assert _plan_resolutions((30, 10, 24), (1, 1.5, 2), 'in_plane', 24) == \
            [(30, 10, 24), (45, 15, 24), (60, 20, 24)]

    def test_all_scales_nz_in_multiples_of_the_ply_count(self):
        assert _plan_resolutions((8, 4, 4), (1, 1.5, 2, 3), 'all', 4) == \
            [(8, 4, 4), (12, 6, 8), (16, 8, 8), (24, 12, 12)]
        # A base with two layers per ply refines in whole plies.
        assert _plan_resolutions((8, 4, 8), (1, 1.5), 'all', 4) == \
            [(8, 4, 8), (12, 6, 12)]

    def test_memory_estimate_grows_with_lu_fill(self):
        small = _estimate_memory_bytes((30, 10, 24))
        big = _estimate_memory_bytes((60, 20, 24))
        assert 0.5e9 < small < 1.5e9            # measured peak RSS 0.76 GB
        assert 3.4e9 < big < 5.0e9              # measured peak RSS 3.4 GB
        assert _estimate_memory_bytes((60, 20, 24), n_factorizations=0) < big / 3
        assert _estimate_memory_bytes((60, 20, 24), n_factorizations=2) > 1.5 * big

    def test_factorizations_at_the_peak(self):
        porous, solid = PorosityField(MAT4, 0.02), PorosityField(MAT4, 0.0)
        assert _peak_factorizations('compression', 'direct', None, porous) == 1
        assert _peak_factorizations('compression', 'direct', -100.0, porous) == 2
        assert _peak_factorizations('thermal', 'direct', -100.0, porous) == 1
        # The pristine reference is factorized even for an iterative solve.
        assert _peak_factorizations('ilss', 'cg', None, porous) == 1
        assert _peak_factorizations('ilss', 'cg', None, solid) == 0
        assert _peak_factorizations('thermal', 'cg', -100.0, porous) == 0

    def test_element_cap_refuses_before_any_solve(self, monkeypatch):
        monkeypatch.setattr(refinement_module, 'FESolver', _NoSolver)
        with pytest.raises(ValueError, match=r'elements, above the limit of 500') as exc:
            _study(max_elements=500)
        assert '24x16x4: 1,536 elements' in str(exc.value)

    def test_memory_budget_refuses_before_any_solve(self, monkeypatch):
        monkeypatch.setattr(refinement_module, 'FESolver', _NoSolver)
        with pytest.raises(ValueError, match=r'GB budget') as exc:
            _study(max_memory_gb=0.01)
        assert 'Planned levels: 12x8x4' in str(exc.value)

    def test_default_base_is_the_production_resolution(self, monkeypatch):
        """The default base is the production mesh, one element layer per
        ply; nz stays at n_plies past the production cap (the study must
        resolve the layup), so a thick laminate meets the guards instead."""
        monkeypatch.setattr(refinement_module, 'FESolver', _NoSolver)
        with pytest.raises(ValueError, match='above the limit') as exc:
            refinement_study(MATERIALS['T800_epoxy'], 0.03, max_elements=1)
        assert 'Planned levels: 30x10x24: 7,200 elements' in str(exc.value)
        thick = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=60)
        with pytest.raises(ValueError, match='above the limit') as exc:
            refinement_study(thick, 0.03, max_elements=1)
        assert 'Planned levels: 30x10x60: 18,000 elements' in str(exc.value)

    def test_unresolved_layup_is_refused(self):
        with pytest.raises(ValueError, match='does not resolve the 4-ply layup'):
            _study(base_resolution=(12, 8, 2))

    @pytest.mark.parametrize('kwargs, match', [
        (dict(factors=(1.0,)), 'at least two levels'),
        (dict(factors=(1.0, 1.0001)), 'must refine'),
        (dict(factors=(2.0, 1.0)), 'strictly increasing'),
        (dict(base_resolution=None, resolutions=[(12, 8, 4), (10, 10, 4)]),
         'must refine'),
        (dict(base_resolution=None, resolutions=[(12, 8, 4), (16, 8, 4)],
              factors=(1, 2)), 'not both'),
        (dict(resolutions=[(12, 8, 4), (16, 8, 4)]), 'not both'),
        (dict(refine='z'), "refine must be"),
        (dict(tol=0.0), 'tol must be'),
        (dict(interior_margin=-1.0), 'interior_margin'),
        (dict(formulation='hex20'), 'formulation'),
        (dict(failure_criterion='puck'), 'failure_criterion'),
        (dict(solver='lu'), 'solver'),
    ])
    def test_bad_options_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _study(**kwargs)

    def test_bad_loading_and_thermal_inputs_raise(self):
        with pytest.raises(ValueError, match='Unknown loading'):
            _study('bending')
        with pytest.raises(ValueError, match='explicit delta_T'):
            _study('thermal')

    def test_porosity_inputs(self):
        other = PorosityField(MATERIALS['T800_epoxy'], 0.02)
        with pytest.raises(ValueError, match='different MaterialProperties'):
            refinement_study(MAT4, other, base_resolution=BASE, ply_angles=LAYUP)
        own = PorosityField(MAT4, 0.02)
        with pytest.raises(ValueError, match='porosity_config only applies'):
            refinement_study(MAT4, own, porosity_config={'distribution': 'uniform'},
                             base_resolution=BASE, ply_angles=LAYUP)
        with pytest.raises(TypeError, match='PorosityField or a void volume'):
            refinement_study(MAT4, True, base_resolution=BASE, ply_angles=LAYUP)
        with pytest.raises(ValueError, match='percent'):
            refinement_study(MAT4, 3.0, base_resolution=BASE, ply_angles=LAYUP)


# ----------------------------------------------------------------------
# The study on small meshes
# ----------------------------------------------------------------------

class TestCompressionStudy:
    def test_in_plane_default_keeps_nz_and_scales_nx_ny(self, compression_study):
        st = compression_study
        assert st.resolutions == ((12, 8, 4), (18, 12, 4), (24, 16, 4))
        assert st.refine == 'in_plane' and st.refined_axes == ('x', 'y')
        assert st.formulation == 'hex8i' and st.loading == 'compression'
        assert [lv.n_elements for lv in st.levels] == [384, 864, 1536]
        sizes = [lv.element_size for lv in st.levels]
        assert sizes == pytest.approx([math.sqrt(50 / n * 20 / m)
                                       for n, m in ((12, 8), (18, 12), (24, 16))])
        assert all(lv.solve_time_s > 0 for lv in st.levels)

    def test_global_and_interior_quantities_converge(self, compression_study):
        conv = compression_study.convergence
        assert list(conv) == list(QUANTITY_KINDS)
        for name in ('knockdown', 'effective_modulus', 'interior_max_failure_index',
                     'interior_first_ply_failure_load_factor'):
            assert conv[name].converged is True, conv[name]
            assert conv[name].verdict == 'converged'
        assert conv['interior_max_failure_index'].behavior == 'monotone'
        assert conv['interior_max_failure_index'].extrapolated == pytest.approx(
            compression_study.levels[-1].interior_max_failure_index, rel=0.01)

    def test_domain_extrema_are_flagged_not_extrapolated(self, compression_study):
        # They sit at the gripped corners, where the stress is singular.
        for name in ('max_failure_index', 'first_ply_failure_load_factor'):
            c = compression_study.convergence[name]
            assert c.kind == 'domain extremum'
            assert c.converged is False
            assert c.verdict == 'mesh-dependent'
            assert c.extrapolated is None
            assert abs(c.relative_changes[-1]) > 0.01
        summary = compression_study.summary()
        assert 'Use interior_max_failure_index' in summary
        assert 'stress singularity' in summary

    def test_levels_match_a_direct_solve(self, compression_study):
        from porosity_fe import CompositeMesh
        pf = PorosityField(MAT4, 0.03, distribution='clustered')
        mesh = CompositeMesh(pf, MAT4, nx=18, ny=12, nz=4, ply_angles=LAYUP)
        res = FESolver(mesh, MAT4, pf, ply_angles=LAYUP).solve(
            'compression', interior_margin=compression_study.interior_margin)
        lv = compression_study.levels[1]
        assert lv.knockdown == pytest.approx(res.knockdown, rel=1e-10)
        assert lv.max_failure_index == pytest.approx(res.max_failure_index, rel=1e-10)
        assert lv.interior_max_failure_index == pytest.approx(
            res.interior_max_failure_index, rel=1e-10)
        assert lv.effective_modulus == pytest.approx(res.effective_modulus, rel=1e-10)

    def test_interior_margin_is_fixed_from_the_coarsest_level(self, compression_study):
        # max(h, 2 dx, 2 dy) on 12 x 8: 2 * 50 / 12.
        assert compression_study.interior_margin == pytest.approx(100 / 12)

    def test_to_dict_is_strict_json(self, compression_study):
        d = compression_study.to_dict()
        text = json.dumps(d, allow_nan=False)
        back = json.loads(text)
        assert back['levels'][0]['resolution'] == [12, 8, 4]
        assert back['convergence']['knockdown']['converged'] is True
        assert back['convergence']['max_failure_index']['verdict'] == 'mesh-dependent'
        assert set(back['levels'][0]) >= set(QUANTITY_KINDS)

    def test_repr_and_summary_are_text_tables(self, compression_study):
        text = repr(compression_study)
        assert text.startswith('RefinementStudy(compression, hex8i')
        assert '18x12x4' in text and 'converged' in text
        summary = compression_study.summary()
        assert 'nz = 4 fixed' in summary
        for name in QUANTITY_KINDS:
            assert name in summary


class TestOtherLoadCases:
    def test_thermal_study(self):
        st = _study('thermal', delta_T=-150.0, factors=(1, 1.5))
        conv = st.convergence
        assert st.delta_T == -150.0
        # No stiffness measure for a thermal solve.
        assert conv['knockdown'].behavior == 'unavailable'
        assert conv['effective_modulus'].behavior == 'unavailable'
        assert all(lv.knockdown is None for lv in st.levels)
        for name in ('max_failure_index', 'interior_max_failure_index',
                     'first_ply_failure_load_factor'):
            assert conv[name].behavior != 'unavailable'
            assert all(v is not None and v > 0 for v in conv[name].values)
        d = json.loads(json.dumps(st.to_dict(), allow_nan=False))
        assert d['levels'][0]['knockdown'] is None
        assert d['convergence']['knockdown']['converged'] is None

    def test_mechanical_with_residual_stress(self):
        st = _study('tension', delta_T=-150.0, factors=(1, 1.5))
        assert st.levels[0].knockdown is not None
        assert st.convergence['knockdown'].converged is True

    def test_shear_is_mesh_independent(self):
        # Homogeneous pure shear of an in-plane uniform laminate: the patch
        # test makes every level give the same answer.
        st = _study('shear', vp=0.02, porosity_config={'distribution': 'uniform'},
                    factors=(1, 1.5, 2))
        for c in st.convergence.values():
            assert c.behavior == 'mesh-independent', c
            assert c.converged is True

    def test_ilss_levels_use_even_nx(self):
        st = _study('ilss', base_resolution=(11, 4, 4), factors=(1, 1.5),
                    interior_margin=2.0)
        assert [lv.resolution[0] for lv in st.levels] == [12, 18]
        assert st.convergence['effective_modulus'].behavior == 'unavailable'

    def test_refine_all_scales_nz_by_ply_multiples(self):
        st = _study(base_resolution=(10, 6, 4), refine='all', factors=(1, 2))
        assert st.resolutions == ((10, 6, 4), (20, 12, 8))
        assert st.refined_axes == ('x', 'y', 'z')
        assert st.convergence['knockdown'].behavior in ('two levels', 'mesh-independent')
        assert st.levels[1].element_size == pytest.approx(
            (50 / 20 * 20 / 12 * 1.0 / 8) ** (1 / 3))

    def test_explicit_resolutions(self):
        st = _study(base_resolution=None, resolutions=[(12, 8, 4), (16, 8, 4)])
        assert st.refine == 'explicit' and st.refined_axes == ('x',)
        assert st.levels[1].element_size == pytest.approx(50 / 16)

    def test_levels_are_released_before_the_next_one(self, monkeypatch):
        refs: list[weakref.ref] = []

        class Tracking(FESolver):
            def __init__(self, *a, **k):
                gc.collect()
                assert all(r() is None for r in refs), "previous level still alive"
                super().__init__(*a, **k)
                refs.append(weakref.ref(self))
                refs.append(weakref.ref(self.mesh))

        monkeypatch.setattr(refinement_module, 'FESolver', Tracking)
        _study(factors=(1, 1.25, 1.5))
        gc.collect()
        assert len(refs) == 6 and all(r() is None for r in refs)

    def test_verbose_logs_the_plan_and_levels(self, caplog):
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            _study(factors=(1, 1.25), verbose=True)
        text = caplog.text
        assert 'refinement_study: compression, hex8i, 2 levels' in text
        assert 'level 2/2 15x10x4 done' in text


def test_public_api():
    for name in ('refinement_study', 'RefinementStudy', 'RefinementLevel',
                 'QuantityConvergence'):
        assert name in porosity_fe.__all__
    for cls in (RefinementStudy, RefinementLevel, QuantityConvergence):
        assert cls.__module__ == 'porosity_fe.fe.refinement'
