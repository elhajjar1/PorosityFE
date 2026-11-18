"""Allowable porosity: the inverse of the empirical knockdown.

``EmpiricalSolver.allowable_porosity`` returns the largest void content
that still retains a required fraction of the pristine strength; the
reporting helpers turn its table into the ``allowable_porosity`` block of
an NCR.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings

import pytest

from porosity_fe import (
    MATERIALS,
    EmpiricalSolver,
    PorosityAllowable,
    ToughnessKnockdown,
    build_empirical_pipeline,
)
from porosity_fe.reporting import (
    allowable_porosity_record,
    build_ncr_record,
    governing_allowable,
    serialise_ncr_json,
    serialise_ncr_markdown,
    serialise_ncr_pdf,
)

MODES = ('compression', 'tension', 'shear', 'ilss', 'transverse_tension')
MODELS = ('judd_wright', 'power_law', 'linear')
SMALL = (4, 3, 2)


def _solver(material='T700_epoxy', Vp=0.02, **kwargs) -> EmpiricalSolver:
    return build_empirical_pipeline(
        MATERIALS[material], Vp, mesh_res=SMALL, **kwargs)[2]


def _closed_form(solver: EmpiricalSolver, model: str, mode: str, r: float) -> float:
    if model == 'judd_wright':
        return -math.log(r) / solver.JUDD_WRIGHT_ALPHA[mode]
    if model == 'power_law':
        return 1.0 - r ** (1.0 / solver.POWER_LAW_N[mode])
    return (1.0 - r) / solver.LINEAR_BETA[mode]


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return func(*args, **kwargs)


class TestAllowablePorosity:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("model", MODELS)
    def test_matches_the_closed_form_inverse(self, mode, model):
        solver = _solver()
        allowable = _quiet(solver.allowable_porosity, 0.9, mode, model)
        assert isinstance(allowable, PorosityAllowable)
        assert allowable.Vp == pytest.approx(
            _closed_form(solver, model, mode, 0.9), rel=1e-12)
        assert allowable.achievable
        assert allowable.mode == mode and allowable.model == model
        assert allowable.required_knockdown == 0.9
        assert allowable.extrapolated == (allowable.Vp > 0.05)

    @pytest.mark.parametrize("model", MODELS)
    def test_the_allowable_meets_the_requirement_and_the_next_float_does_not(self, model):
        """The result is rounded to the safe side: the forward knockdown at
        it is at least the requirement, and below it one float further."""
        allowable = _solver().allowable_porosity(0.93, 'ilss', model)
        at = _solver(Vp=allowable.Vp).get_failure_load('ilss', model).knockdown
        above = _solver(Vp=math.nextafter(allowable.Vp, 1.0)) \
            .get_failure_load('ilss', model).knockdown
        assert at >= 0.93 > above

    def test_no_loss_allowed_gives_zero_voids(self):
        """To round-off: exp(-alpha Vp) is still exactly 1 at Vp ~ 1e-17."""
        allowable = _solver().allowable_porosity(1.0, 'compression')
        assert 0.0 <= allowable.Vp < 1e-15 and allowable.achievable
        assert not allowable.extrapolated
        assert _solver().allowable_porosity(1.0, 'compression', 'linear').Vp < 1e-15

    def test_docstring_example(self):
        solver = build_empirical_pipeline(MATERIALS['T700_epoxy'], 0.02)[2]
        assert round(solver.allowable_porosity(0.9, 'ilss').Vp, 4) == 0.0105

    def test_does_not_depend_on_the_porosity_of_the_mesh(self):
        a = _solver(Vp=0.0).allowable_porosity(0.9, 'shear')
        b = _solver(Vp=0.04, porosity_config={'distribution': 'clustered'}) \
            .allowable_porosity(0.9, 'shear')
        assert a == b

    def test_required_stress_is_the_knockdown_times_the_pristine_strength(self):
        solver = _solver()
        sigma_0 = MATERIALS['T700_epoxy'].sigma_1c
        by_stress = solver.allowable_porosity(required_stress=0.9 * sigma_0)
        by_fraction = solver.allowable_porosity(0.9)
        assert by_stress.Vp == pytest.approx(by_fraction.Vp, rel=1e-12)
        assert by_stress.details['pristine_strength'] == sigma_0
        assert by_stress.details['required_stress'] == pytest.approx(0.9 * sigma_0)
        assert by_fraction.details['knockdown_at_zero_porosity'] == 1.0

    def test_a_stress_above_the_pristine_strength_is_not_achievable(self):
        allowable = _solver().allowable_porosity(
            required_stress=2.0 * MATERIALS['T700_epoxy'].sigma_1c)
        assert not allowable.achievable and math.isnan(allowable.Vp)
        assert not allowable.extrapolated
        assert allowable.required_knockdown == pytest.approx(2.0)
        # A comparison with nan is false: nothing is "within" it.
        assert not (0.0 <= allowable.Vp)

    def test_fatigue_lowers_the_allowable(self):
        solver = _solver()
        fatigue = solver.get_failure_load('compression', cycles=1e6) \
            .details['fatigue_knockdown']
        assert 0.0 < fatigue < 1.0
        r = 0.9 * fatigue
        allowable = solver.allowable_porosity(r, 'compression', cycles=1e6)
        # KD * fatigue >= r  <=>  KD >= 0.9
        assert allowable.Vp == pytest.approx(
            _closed_form(solver, 'judd_wright', 'compression', 0.9), rel=1e-9)
        assert allowable.details['fatigue_knockdown'] == fatigue
        assert allowable.details['knockdown_at_zero_porosity'] == fatigue
        assert allowable.Vp < _quiet(solver.allowable_porosity, r, 'compression').Vp

    def test_a_requirement_above_the_fatigue_factor_is_not_achievable(self):
        solver = _solver()
        fatigue = solver.get_failure_load('compression', cycles=1e6) \
            .details['fatigue_knockdown']
        allowable = solver.allowable_porosity(
            min(1.0, fatigue * 1.01), 'compression', cycles=1e6)
        assert not allowable.achievable and math.isnan(allowable.Vp)

    def test_environment_lowers_the_allowable(self):
        # No preset sets T_g_dry, without which the environment is a no-op.
        material = dataclasses.replace(MATERIALS['T700_epoxy'], T_g_dry=180.0)
        solver = build_empirical_pipeline(material, 0.02, mesh_res=SMALL)[2]
        env = {'T': 80.0, 'M': 1.0}
        factor = solver.get_failure_load('shear', environment=env) \
            .details['environment_knockdown']
        assert 0.0 < factor < 1.0
        allowable = solver.allowable_porosity(0.9 * factor, 'shear', environment=env)
        assert allowable.details['environment_knockdown'] == factor
        assert allowable.Vp == pytest.approx(
            _closed_form(solver, 'judd_wright', 'shear', 0.9), rel=1e-9)
        dry = _quiet(solver.allowable_porosity, 0.9 * factor, 'shear')
        assert allowable.Vp < dry.Vp
        assert 'environment_knockdown' not in dry.details

    @pytest.mark.parametrize("kwargs, match", [
        ({}, "exactly one"),
        ({'required_knockdown': 0.9, 'required_stress': 100.0}, "exactly one"),
        ({'required_knockdown': 0.0}, "above 0"),
        ({'required_knockdown': -0.5}, "above 0"),
        ({'required_knockdown': math.nan}, "finite"),
        ({'required_knockdown': math.inf}, "finite"),
        ({'required_knockdown': True}, "finite number"),
        ({'required_knockdown': "0.9"}, "finite number"),
        ({'required_knockdown': 90.0}, r"A percent\? Pass 0.9"),
        ({'required_stress': -1.0}, "above 0"),
    ])
    def test_bad_requirement_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _solver().allowable_porosity(**kwargs)

    def test_unknown_mode_and_model_raise_as_the_forward_call(self):
        solver = _solver()
        with pytest.raises(ValueError, match="Unknown loading mode"):
            solver.allowable_porosity(0.9, 'peel')
        with pytest.raises(ValueError, match="Unknown knockdown model"):
            solver.allowable_porosity(0.9, 'shear', 'cubic')

    def test_a_low_requirement_is_flagged_as_extrapolated_with_one_warning(self):
        solver = _solver()
        with pytest.warns(UserWarning, match="beyond the calibration bound") as record:
            allowable = solver.allowable_porosity(0.5, 'compression')
        assert len(record) == 1
        assert "compression / judd_wright" in str(record[0].message)
        assert allowable.extrapolated and allowable.Vp > 0.05

    def test_within_the_calibration_range_nothing_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert not _solver().allowable_porosity(0.95, 'ilss').extrapolated

    def test_a_requirement_every_void_content_meets_gives_one(self):
        # exp(-alpha) of the Judd-Wright law is the knockdown at Vp = 1.
        solver = _solver()
        floor = math.exp(-solver.JUDD_WRIGHT_ALPHA['tension'])
        allowable = _quiet(solver.allowable_porosity, floor * 0.5, 'tension')
        assert allowable.Vp == 1.0 and allowable.extrapolated

    def test_layup_amplification_is_recorded_and_warned(self):
        solver = build_empirical_pipeline(
            MATERIALS['T700_epoxy'], 0.02, mesh_res=SMALL,
            ply_angles=[45.0, -45.0, -45.0, 45.0])[2]
        scale = solver.layup_scale['tension']
        assert scale > 1.0
        with pytest.warns(UserWarning, match="unvalidated layup amplification"):
            allowable = solver.allowable_porosity(0.95, 'tension')
        assert allowable.details['layup_scale'] == scale
        assert allowable.details['layup_extrapolated'] is True
        qi = _solver().allowable_porosity(0.95, 'tension')
        assert allowable.Vp == pytest.approx(qi.Vp / scale, rel=1e-12)
        assert 'layup_scale' not in qi.details

    def test_to_dict_merges_the_details(self):
        d = _solver().allowable_porosity(0.9, 'ilss').to_dict()
        assert d['mode'] == 'ilss' and d['model'] == 'judd_wright'
        assert d['achievable'] is True and d['extrapolated'] is False
        assert d['pristine_strength'] == MATERIALS['T700_epoxy'].tau_ilss
        json.dumps(d)


class TestUserCallable:
    def test_a_step_law_gives_the_last_float_before_the_step(self):
        def step(Vp, mode):
            return 1.0 if Vp < 0.0123 else 0.5

        allowable = _solver().allowable_porosity(0.9, 'compression', step)
        assert allowable.Vp == math.nextafter(0.0123, 0.0)
        assert allowable.model == 'step'
        assert not allowable.extrapolated   # a callable owns its range

    def test_a_dip_that_recovers_stops_at_the_first_crossing(self):
        def dip(Vp, mode):
            return 0.8 if 0.02 <= Vp <= 0.03 else 0.95

        allowable = _solver().allowable_porosity(0.9, 'shear', dip)
        assert allowable.Vp == math.nextafter(0.02, 0.0)

    def test_a_law_that_never_falls_below_gives_one(self):
        allowable = _solver().allowable_porosity(
            0.9, 'compression', lambda Vp, mode: 1.0)
        assert allowable.Vp == 1.0 and allowable.achievable
        assert allowable.model == '<lambda>'

    def test_a_law_below_the_requirement_at_zero_is_not_achievable(self):
        allowable = _solver().allowable_porosity(
            0.9, 'compression', lambda Vp, mode: 0.8)
        assert not allowable.achievable and math.isnan(allowable.Vp)

    def test_the_layup_scale_is_bypassed(self):
        solver = build_empirical_pipeline(
            MATERIALS['T700_epoxy'], 0.02, mesh_res=SMALL,
            ply_angles=[45.0, -45.0, -45.0, 45.0])[2]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            allowable = solver.allowable_porosity(
                0.9, 'tension', lambda Vp, mode: math.exp(-5.0 * Vp))
        assert allowable.Vp == pytest.approx(-math.log(0.9) / 5.0, rel=1e-12)
        assert 'layup_scale' not in allowable.details


class TestToughnessModes:
    def test_the_placeholder_law_is_inverted_and_flagged(self):
        allowable = _solver().allowable_porosity(0.9, 'mode_I_toughness')
        assert allowable.Vp == pytest.approx(-math.log(0.9) / 6.2, rel=1e-12)
        assert allowable.model == 'toughness_exponential'
        assert allowable.details['fracture_mode'] == 'I'
        assert allowable.details['toughness_uncalibrated'] is True
        assert allowable.details['pristine_toughness'] is None
        assert allowable.details['required_stress'] is None

    def test_given_coefficients_are_not_flagged(self):
        solver = _solver(solver_kwargs={
            'toughness_knockdown': ToughnessKnockdown(b_I=4.0, b_II=10.0)})
        allowable = solver.allowable_porosity(0.9, 'mode_II_toughness')
        assert allowable.Vp == pytest.approx(-math.log(0.9) / 10.0, rel=1e-12)
        assert allowable.details['toughness_uncalibrated'] is False

    def test_the_linear_law(self):
        solver = _solver(solver_kwargs={
            'toughness_knockdown': ToughnessKnockdown('linear', b_I=5.0, b_II=8.0)})
        allowable = solver.allowable_porosity(0.8, 'mode_I_toughness')
        assert allowable.Vp == pytest.approx(0.04, rel=1e-12)
        assert allowable.model == 'toughness_linear'

    def test_no_law_allows_every_void_content(self):
        solver = _solver(solver_kwargs={'toughness_knockdown': None})
        allowable = solver.allowable_porosity(0.9, 'mode_I_toughness')
        assert allowable.Vp == 1.0 and not allowable.extrapolated
        assert allowable.details['toughness_uncalibrated'] is False

    def test_a_callable_law_is_bracketed_on_the_grid(self):
        def law(Vp, mode):
            return 1.0 if Vp < 0.031 else 0.6

        solver = _solver(solver_kwargs={'toughness_knockdown': law})
        allowable = solver.allowable_porosity(0.9, 'mode_I_toughness')
        assert allowable.Vp == math.nextafter(0.031, 0.0)
        assert allowable.model == 'toughness_law'

    def test_required_toughness_in_N_per_mm(self):
        material = MATERIALS['IM7_8552_epoxy']
        assert material.G_Ic is not None
        solver = _solver('IM7_8552_epoxy')
        allowable = solver.allowable_porosity(
            mode='mode_I_toughness', required_stress=0.9 * material.G_Ic)
        assert allowable.Vp == pytest.approx(-math.log(0.9) / 6.2, rel=1e-9)
        assert allowable.details['pristine_toughness'] == material.G_Ic

    def test_required_toughness_needs_the_pristine_value(self):
        assert MATERIALS['T700_epoxy'].G_Ic is None
        with pytest.raises(ValueError, match="does not set"):
            _solver().allowable_porosity(
                mode='mode_I_toughness', required_stress=0.2)

    def test_what_the_forward_toughness_call_refuses_is_refused(self):
        solver = _solver()
        with pytest.raises(ValueError, match="toughness_knockdown"):
            solver.allowable_porosity(0.9, 'mode_I_toughness', lambda Vp, m: 1.0)
        with pytest.raises(ValueError, match="environment"):
            solver.allowable_porosity(0.9, 'mode_I_toughness',
                                      environment={'T': 80.0})
        with pytest.raises(ValueError, match="cycles"):
            solver.allowable_porosity(0.9, 'mode_I_toughness', cycles=1e5)


class TestTable:
    def test_one_requirement_covers_the_five_strength_modes(self):
        solver = _solver()
        table = solver.get_all_allowable_porosities(0.95)
        assert tuple(table) == MODES
        for mode in MODES:
            assert tuple(table[mode]) == MODELS
            for model in MODELS:
                assert table[mode][model] == solver.allowable_porosity(0.95, mode, model)

    def test_a_dict_evaluates_the_modes_it_names(self):
        solver = _solver()
        table = solver.get_all_allowable_porosities({'ilss': 0.9, 'compression': 0.95})
        assert tuple(table) == ('compression', 'ilss')    # the table's order
        assert table['ilss']['linear'].required_knockdown == 0.9
        assert table['compression']['linear'].required_knockdown == 0.95

    @pytest.mark.parametrize("bad", [{}, {'mode_I_toughness': 0.9}, {'peel': 0.9}])
    def test_a_dict_must_name_strength_modes(self, bad):
        with pytest.raises(ValueError, match="at least one strength mode"):
            _solver().get_all_allowable_porosities(bad)

    def test_extra_models_are_added(self):
        table = _quiet(
            _solver().get_all_allowable_porosities, 0.9,
            extra_models={'mine': lambda Vp, mode: math.exp(-4.0 * Vp)})
        assert tuple(table['shear']) == (*MODELS, 'mine')
        assert table['shear']['mine'].Vp == pytest.approx(-math.log(0.9) / 4.0)

    def test_the_table_warns_once(self):
        with pytest.warns(UserWarning) as record:
            table = _solver().get_all_allowable_porosities(0.7)
        assert len(record) == 1
        message = str(record[0].message)
        flagged = [a for row in table.values() for a in row.values() if a.extrapolated]
        assert len(flagged) > 1
        assert all(f"{a.mode} / {a.model}" in message for a in flagged)

    def test_factors_are_passed_to_every_entry(self):
        table = _quiet(_solver().get_all_allowable_porosities, 0.5, cycles=1e6)
        assert all('fatigue_knockdown' in a.details
                   for row in table.values() for a in row.values())

    def test_an_amplified_layup_warns_once_for_the_table(self):
        solver = build_empirical_pipeline(
            MATERIALS['T700_epoxy'], 0.02, mesh_res=SMALL,
            ply_angles=[45.0, -45.0, -45.0, 45.0])[2]
        with pytest.warns(UserWarning) as record:
            solver.get_all_allowable_porosities(0.97)
        assert [("layup amplification" in str(w.message)) for w in record] == [True]
        # A table without a fiber-direction mode has nothing amplified.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solver.get_all_allowable_porosities({'ilss': 0.97})


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _entry(Vp, mode='ilss', model='linear', required=0.9, **kwargs):
    achievable = not math.isnan(Vp)
    return PorosityAllowable(Vp=Vp, required_knockdown=required,
                             achievable=achievable,
                             extrapolated=kwargs.get('extrapolated', False),
                             mode=mode, model=model)


class TestGoverningAllowable:
    def test_the_smallest_entry_governs(self):
        table = _quiet(_solver().get_all_allowable_porosities, 0.9)
        worst = governing_allowable(table)
        assert worst.Vp == min(a.Vp for row in table.values() for a in row.values())
        assert (worst.mode, worst.model) == ('ilss', 'judd_wright')

    def test_an_entry_nothing_meets_governs_over_a_smaller_one(self):
        table = {'ilss': {'linear': _entry(0.001)},
                 'shear': {'linear': _entry(math.nan, mode='shear')}}
        assert governing_allowable(table).mode == 'shear'

    def test_an_empty_table_raises(self):
        with pytest.raises(ValueError, match="empty"):
            governing_allowable({})


class TestAllowableRecord:
    def test_record_of_a_table(self):
        table = _quiet(_solver().get_all_allowable_porosities, 0.9)
        record = allowable_porosity_record(table, 2.0)
        assert record['required_knockdown'] == dict.fromkeys(MODES, 0.9)
        assert set(record['per_mode']) == set(MODES)
        ilss = record['per_mode']['ilss']['judd_wright']
        assert ilss == {
            'allowable_Vp_percent': table['ilss']['judd_wright'].Vp * 100.0,
            'achievable': True, 'extrapolated': False}
        assert record['governing'] == {
            'mode': 'ilss', 'model': 'judd_wright', 'required_knockdown': 0.9, **ilss}
        assert record['measured_Vp_percent'] == 2.0
        assert record['within_allowable'] is False
        assert record['statement'] == (
            "Governing allowable void content: 1.05% (ilss / judd_wright at "
            "90% of pristine strength retained). The measured 2.00% is above it.")
        assert "does not replace the porosity allowable" in record['basis']
        json.dumps(record)

    def test_a_measured_value_within_the_allowable(self):
        record = allowable_porosity_record(
            _quiet(_solver().get_all_allowable_porosities, 0.9), 1.0)
        assert record['within_allowable'] is True
        assert record['statement'].endswith("The measured 1.00% is within it.")

    def test_without_a_measured_value(self):
        record = allowable_porosity_record(
            _quiet(_solver().get_all_allowable_porosities, 0.9))
        assert record['measured_Vp_percent'] is None
        assert record['within_allowable'] is None
        assert "measured" not in record['statement']

    def test_an_extrapolated_governing_entry_says_so(self):
        table = {'ilss': {'linear': _entry(0.08, extrapolated=True)}}
        record = allowable_porosity_record(table, 3.0)
        assert "is extrapolated" in record['statement']
        assert record['governing']['extrapolated'] is True
        assert record['within_allowable'] is True

    def test_a_requirement_nothing_meets(self):
        table = {'ilss': {'linear': _entry(math.nan)}}
        record = allowable_porosity_record(table, 0.5)
        assert record['governing']['allowable_Vp_percent'] is None
        assert record['governing']['achievable'] is False
        assert record['within_allowable'] is False
        assert record['statement'].startswith("No void content meets the requirement")
        assert "measured" not in record['statement']


def _analysis_result(Vp_percent=2.0, angles=None):
    """A ``run_analysis``-shaped result with the solver's inputs."""
    material = MATERIALS['T700_epoxy']
    angles = angles or [0.0, 90.0, 45.0, -45.0, -45.0, 45.0, 90.0, 0.0]
    _, mesh, solver = build_empirical_pipeline(
        material, Vp_percent / 100.0, ply_angles=angles, mesh_res=SMALL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        table = solver.get_all_failure_loads()
    empirical = {mode: {model: {'failure_stress': r.failure_stress,
                                'knockdown': r.knockdown}
                        for model, r in row.items()}
                 for mode, row in table.items()}
    cfg = {
        "material_name": "T700_epoxy", "n_plies": material.n_plies,
        "t_ply": material.t_ply, "Vp": Vp_percent, "distribution": "uniform",
        "void_shape": "spherical", "nx": 4, "ny": 3, "nz": 2, "angles": angles,
    }
    return {"config": cfg, "empirical": empirical, "mesh": mesh,
            "material": material}


class TestNcrAllowableBlock:
    META = {"date": "2026-01-01", "prepared_by": "test"}

    def test_absent_without_a_requirement(self):
        ncr = build_ncr_record(_analysis_result(), self.META)
        assert "allowable_porosity" not in ncr["engineering_analysis"]
        assert "Allowable void content" not in serialise_ncr_markdown(ncr)
        assert "allowable_Vp_percent" not in json.loads(serialise_ncr_json(ncr))["units"]

    def test_block_of_a_requirement(self):
        result = _analysis_result()
        ncr = build_ncr_record(result, dict(self.META, required_knockdown=0.9))
        block = ncr["engineering_analysis"]["allowable_porosity"]
        solver = EmpiricalSolver(result["mesh"], result["material"],
                                 ply_angles=result["config"]["angles"])
        assert block == allowable_porosity_record(
            _quiet(solver.get_all_allowable_porosities, 0.9), 2.0)
        assert block["within_allowable"] is False
        # The recommended path is the one without the requirement.
        plain = build_ncr_record(result, self.META)
        assert ncr["recommended_disposition"] == plain["recommended_disposition"]

    def test_a_requirement_per_mode(self):
        ncr = build_ncr_record(_analysis_result(), dict(
            self.META, required_knockdown={'compression': 0.95, 'ilss': 0.8}))
        block = ncr["engineering_analysis"]["allowable_porosity"]
        assert block["required_knockdown"] == {'compression': 0.95, 'ilss': 0.8}
        assert set(block["per_mode"]) == {'compression', 'ilss'}

    def test_the_solver_warnings_are_not_raised(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ncr = build_ncr_record(_analysis_result(),
                                   dict(self.META, required_knockdown=0.6))
        block = ncr["engineering_analysis"]["allowable_porosity"]
        assert any(a["extrapolated"] for row in block["per_mode"].values()
                   for a in row.values())

    def test_markdown_and_pdf_carry_the_block(self):
        ncr = build_ncr_record(_analysis_result(),
                               dict(self.META, required_knockdown=0.6))
        block = ncr["engineering_analysis"]["allowable_porosity"]
        md = serialise_ncr_markdown(ncr)
        assert f"**Allowable void content:** {block['statement']}" in md
        assert "| Mode | Model | Required retained (%) | Allowable Vp (%) |" in md
        assert "| ilss | judd_wright | 60 | 5.11 (extrapolated) |" in md
        assert f"> {block['basis']}" in md
        pdf = serialise_ncr_pdf(ncr)
        assert pdf.startswith(b"%PDF")
        from porosity_fe.reporting import _ncr_text_lines
        text = "\n".join(_ncr_text_lines(ncr))
        assert "Allowable void content: Governing allowable void content" in text
        assert "5.11 (extrapolated)" in text
        assert max(len(line) for line in _ncr_text_lines(ncr)) <= 92

    def test_markdown_of_a_requirement_nothing_meets(self):
        result = _analysis_result()
        block = allowable_porosity_record(
            {'ilss': {'linear': _entry(math.nan)}}, 2.0)
        ncr = build_ncr_record(result, self.META)
        ncr["engineering_analysis"]["allowable_porosity"] = block
        assert "| ilss | linear | 90 | not achievable |" in serialise_ncr_markdown(ncr)

    def test_json_units_and_schema(self):
        ncr = build_ncr_record(_analysis_result(),
                               dict(self.META, required_knockdown=0.9))
        doc = json.loads(serialise_ncr_json(ncr))
        assert doc["units"]["allowable_Vp_percent"] == "%"
        assert doc["units"]["required_knockdown"].startswith("dimensionless")
        assert doc["engineering_analysis"]["allowable_porosity"]["governing"]["mode"] == "ilss"
        jsonschema = pytest.importorskip("jsonschema")
        import pathlib
        schema = json.loads((pathlib.Path(__file__).parent.parent / "validation"
                             / "schemas" / "porosity_results_schema.json").read_text())
        jsonschema.validate(doc, schema)

    @pytest.mark.parametrize("bad", [0.0, 1.5, {}, {'peel': 0.9}])
    def test_a_bad_requirement_raises(self, bad):
        with pytest.raises(ValueError):
            build_ncr_record(_analysis_result(), {"required_knockdown": bad})

    def test_a_result_without_the_solver_inputs_raises(self):
        result = _analysis_result()
        del result["mesh"]
        del result["config"]["angles"]
        with pytest.raises(ValueError, match=r"no mesh, config\['angles'\]"):
            build_ncr_record(result, {"required_knockdown": 0.9})
