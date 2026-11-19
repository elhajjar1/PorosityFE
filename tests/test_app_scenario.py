"""The app's custom material, plate size and saved scenarios (headless).

The Streamlit-driven paths are in ``tests/test_app_streamlit.py``; these
cover the config helpers, ``run_analysis`` and ``porosity_app.scenario``
without a script run (the session state is a plain dict).
"""

import dataclasses
import json

import pytest

pytest.importorskip("streamlit")

from porosity_app import config as app_config
from porosity_app import convergence as app_convergence
from porosity_app import runners as app_runners
from porosity_app import scenario
from porosity_fe import MATERIALS
from porosity_fe.reporting import build_export_payload, download_filename_stem


def _cfg(**overrides) -> dict:
    cfg = {
        "material_name": "T800_epoxy", "angles": [0.0, 90.0, 90.0, 0.0],
        "n_plies": 4, "t_ply": 0.183, "Vp": 3.0, "distribution": "uniform",
        "cluster_location": "midplane", "void_shape": "spherical",
        "loading_mode": "compression", "nx": 8, "ny": 4, "nz": 4,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def custom_fields():
    return dataclasses.replace(MATERIALS["T800_epoxy"], E22=9000.0, G12=4000.0).to_dict()


@pytest.fixture(scope="module")
def custom_result(custom_fields):
    return app_runners.run_analysis(_cfg(material_name="Custom: soft", material=custom_fields,
                                         length=80.0, width=30.0))


@pytest.fixture(scope="module")
def plain_result():
    return app_runners.run_analysis(_cfg())


class TestConfig:
    def test_new_fields_leave_the_default_key_alone(self):
        """A config without a custom material or plate keeps the key (and
        the cached result) of before."""
        default = app_config._config_to_key(_cfg())
        assert app_config._config_to_key(_cfg(material=None, length=None, width=None)) == default
        assert not {k for k, _ in default} & {"material", "length", "width"}

    def test_key_round_trip(self, custom_fields):
        cfg = _cfg(material=custom_fields, length=80.0, width=30.0)
        key = app_config._config_to_key(cfg)
        hash(key)
        back = app_config._key_to_config(key)
        assert back["material"] == custom_fields and back["angles"] == cfg["angles"]
        assert (back["length"], back["width"]) == (80.0, 30.0)
        assert app_config._config_to_key(back) == key

    def test_material_of_a_config(self, custom_fields):
        assert app_config._cfg_material(_cfg()) is MATERIALS["T800_epoxy"]
        custom = app_config._cfg_material(_cfg(material_name="Custom: x",
                                               material=custom_fields))
        assert custom.E22 == 9000.0
        with pytest.raises(ValueError, match="Unknown material 'nope'"):
            app_config._cfg_material(_cfg(material_name="nope"))

    def test_plate_of_a_config(self):
        assert app_config._cfg_plate(_cfg()) == {}
        assert app_config._cfg_plate(_cfg(length=80, width=None)) == {"length": 80.0}


class TestRunAnalysis:
    def test_custom_material_and_plate_reach_the_solves(self, custom_result, plain_result):
        mesh = custom_result["mesh"]
        assert (mesh.L_x, mesh.L_y) == (80.0, 30.0)
        assert custom_result["material"].E22 == 9000.0
        assert custom_result["material"].n_plies == 4
        assert (plain_result["mesh"].L_x, plain_result["mesh"].L_y) == (50.0, 20.0)
        # The softer matrix-direction ply changes the FE stiffness; the plate
        # changes nothing empirical.
        assert custom_result["fe_field"].effective_modulus < \
            plain_result["fe_field"].effective_modulus
        assert custom_result["empirical"]["compression"] == plain_result["empirical"][
            "compression"]

    def test_a_delaminated_plate_keeps_its_size(self, custom_fields):
        from porosity_fe import Delamination
        result = app_runners.run_analysis(_cfg(
            length=60.0, width=24.0, delamination=Delamination.strip(1, 25.0, 35.0)))
        assert (result["mesh"].L_x, result["mesh"].L_y) == (60.0, 24.0)
        assert result["fe_field"] is not None

    def test_the_specimens_ignore_the_plate(self):
        result = app_runners.run_analysis(_cfg(loading_mode="ilss", length=80.0, width=30.0))
        assert result["fe_specimen"] is not None and result["mesh"].L_x != 80.0

    def test_uncertainty_uses_the_custom_material(self, custom_fields):
        key = app_config._config_to_key(_cfg(material_name="Custom: x",
                                             material=custom_fields))
        res, _ = app_runners.run_uq_cached.__wrapped__(key, "shear", 32, 0.1, 0.1, 0.05)
        plain, _ = app_runners.run_uq_cached.__wrapped__(
            app_config._config_to_key(_cfg()), "shear", 32, 0.1, 0.1, 0.05)
        assert res["knockdown"]["percentiles"] == plain["knockdown"]["percentiles"]
        assert res["failure_stress"]["percentiles"] == plain["failure_stress"]["percentiles"]

    def test_convergence_check_refuses_a_custom_plate(self, custom_result, plain_result):
        assert "standard 50 x 20 mm coupon" in \
            app_convergence.convergence_unavailable_reason(custom_result)
        assert app_convergence.convergence_unavailable_reason(plain_result) is None

    def test_convergence_check_uses_the_custom_material(self, custom_fields, monkeypatch):
        seen = {}
        monkeypatch.setattr(app_convergence, "refinement_study",
                            lambda material, *a, **k: seen.setdefault("m", material))
        app_convergence.run_convergence_check(
            _cfg(material_name="Custom: x", material=custom_fields), [(8, 4, 4)])
        assert seen["m"].E22 == 9000.0 and seen["m"].n_plies == 4

    def test_export_payload_records_them(self, custom_result, plain_result):
        payload = build_export_payload(custom_result)
        assert payload["config"]["plate_mm"] == "80x30"
        assert payload["config"]["material_properties"]["E22"] == 9000.0
        assert download_filename_stem(payload).startswith("porosity_Custom__soft_Vp3.0pct_")
        plain = build_export_payload(plain_result)
        assert "plate_mm" not in plain["config"]
        assert "material_properties" not in plain["config"]
        one_side = build_export_payload({**plain_result,
                                         "config": {**plain_result["config"], "width": 40.0}})
        assert one_side["config"]["plate_mm"] == "50x40"


class TestCustomMaterialState:
    def test_options_and_resolution(self):
        state = {}
        assert scenario.material_options(state) == list(MATERIALS)
        material = dataclasses.replace(MATERIALS["T700_epoxy"], E11=1.5e5)
        state[scenario.CUSTOM_MATERIAL_KEY] = scenario.custom_material_record(
            material, "  my   T700 ")
        assert scenario.material_options(state)[-1] == "Custom: my T700"
        assert scenario.resolve_material("Custom: my T700", state) == material
        assert scenario.resolve_material("T800_epoxy", state) is MATERIALS["T800_epoxy"]

    def test_a_blank_name(self):
        record = scenario.custom_material_record(MATERIALS["T700_epoxy"], "   ")
        assert record["name"] == "material"


class TestScenario:
    def _state(self, **inputs):
        state = {"material_select": "T800_epoxy", "layup_input": "[0/90]s",
                 "t_ply_input": 0.2, "vp_input": 2.5, "expert_mode": True,
                 "mesh_nx": 12, "mesh_ny": 6, "mesh_nz": 4,
                 "plate_length": 70.0, "plate_width": 25.0,
                 "_layup_status": ("ok", "x"), "result": None}
        state.update(inputs)
        return state

    def test_document_records_the_inputs_only(self):
        doc = scenario.scenario_document(self._state(), "  run  A ")
        assert doc["format"] == scenario.FORMAT_SCENARIO == "porosity-fe.app-scenario"
        assert doc["name"] == "run A"
        assert doc["inputs"]["vp_input"] == 2.5 and doc["inputs"]["mesh_nx"] == 12
        # Private keys and results are not inputs.
        assert "_layup_status" not in doc["inputs"] and "result" not in doc["inputs"]
        assert doc["material"] is None and doc["summary"] is None
        assert scenario.scenario_document({}, "")["name"] is None

    def test_values_a_widget_would_refuse_are_left_out(self):
        doc = scenario.scenario_document(self._state(vp_input=40.0, mesh_nz=1.5,
                                                     void_shape_select="cubic"))
        assert not {"vp_input", "mesh_nz", "void_shape_select"} & set(doc["inputs"])

    def test_round_trip_through_json(self, plain_result):
        state = self._state(delam_on=True, delam_interface=1, thermal_delta_T=None)
        text = scenario.serialise_scenario(
            scenario.scenario_document(state, "A", plain_result))
        restored = {}
        notes = scenario.apply_scenario(scenario.parse_scenario(text), restored)
        assert notes == []
        for key in ("layup_input", "vp_input", "mesh_nx", "plate_length", "delam_on",
                    "delam_interface", "thermal_delta_T"):
            assert restored[key] == state[key]
        assert type(restored["mesh_nx"]) is int
        # A restored nz is the user's, not the layup default's.
        assert restored["_mesh_nz_default"] is None
        assert restored[scenario.REFERENCE_KEY]["name"] == "A"
        assert restored[scenario.REFERENCE_KEY]["summary"]["Vp_percent"] == 3.0

    def test_custom_material_travels_with_the_scenario(self):
        material = dataclasses.replace(MATERIALS["T700_epoxy"], G12=4100.0)
        record = scenario.custom_material_record(material, "Mine")
        state = self._state(material_select="Custom: Mine")
        state[scenario.CUSTOM_MATERIAL_KEY] = record
        doc = json.loads(scenario.serialise_scenario(scenario.scenario_document(state)))
        assert doc["material"]["fields"]["G12"] == 4100.0
        restored = {}
        assert scenario.apply_scenario(doc, restored) == []
        assert restored["material_select"] == "Custom: Mine"
        assert scenario.resolve_material("Custom: Mine", restored) == material
        # A preset selection does not carry the custom material along.
        state["material_select"] = "T800_epoxy"
        assert scenario.scenario_document(state)["material"] is None

    def test_restore_notes(self):
        doc = {"format": scenario.FORMAT_SCENARIO, "inputs": {
            "vp_input": 99.0, "mesh_nx": "8", "future_input": 1,
            "material_select": "Custom: gone", "layup_input": "[0/91",
            "delam_on": "yes", "t_ply_input": 0.25},
            "material": None, "summary": None}
        state = {scenario.REFERENCE_KEY: {"old": True}}
        notes = scenario.apply_scenario(doc, state)
        assert state == {"t_ply_input": 0.25}          # the reference is dropped too
        text = "\n".join(notes)
        assert "vp_input: 99.0 is outside 0.1 to 15; not restored." in text
        assert "mesh_nx: '8' is not a number" in text
        assert "future_input: not an input of this version; skipped." in text
        assert "material_select: 'Custom: gone' is not a preset" in text
        assert "layup_input: '[0/91' is not a layup" in text
        assert "delam_on: 'yes' is not true or false" in text

    def test_a_broken_custom_material(self):
        doc = {"format": scenario.FORMAT_SCENARIO, "inputs": {},
               "material": {"name": "x", "fields": {"E11": 1.0}}, "summary": None}
        notes = scenario.apply_scenario(doc, {})
        assert notes[0].startswith("The custom material was not restored: The material "
                                   "is missing E22")

    @pytest.mark.parametrize("text, match", [
        ("{nope", "not valid JSON"),
        (b"\xff\xfe", "not valid JSON"),
        ('{"format": "porosity-fe.material"}', "not a saved scenario"),
        ("[1]", "not a saved scenario"),
        ('{"format": "porosity-fe.app-scenario", "inputs": 3}', "has no inputs"),
    ])
    def test_parse_refusals(self, text, match):
        with pytest.raises(ValueError, match=match):
            scenario.parse_scenario(text)

    def test_input_kinds(self):
        spec = scenario.SCENARIO_INPUTS
        assert spec["thermal_delta_T"].problem(None) is None
        assert spec["vp_input"].problem(None) == "is empty"
        assert spec["mesh_nx"].problem(True) == "is not a number"
        assert spec["mesh_nx"].problem(float("inf")) == "is not a number"
        assert spec["mesh_nx"].problem(8.0) is None
        assert spec["mesh_nx"].coerce(8.0) == 8
        assert spec["layup_input"].problem(3) == "is not text"
        assert spec["loading_select"].problem("dcb") is None


class TestComparison:
    def test_rows(self, plain_result, custom_result):
        saved = scenario.scenario_summary(plain_result)
        now = scenario.scenario_summary(custom_result)
        rows = {r["Quantity"]: r for r in scenario.comparison_rows(saved, now)}
        assert rows["Void content Vp (%)"]["Change (%)"] == 0.0
        row = rows["FE effective modulus (MPa), compression"]
        assert row["This run"] < row["Saved run"]
        assert row["Change (%)"] == pytest.approx(
            100.0 * (row["This run"] - row["Saved run"]) / row["Saved run"])
        assert "Knockdown, ilss (linear)" in rows
        # No fracture factor without a delamination: no row for it.
        assert not any("fracture" in q for q in rows)

    def test_fe_rows_need_the_same_loading(self, plain_result):
        saved = scenario.scenario_summary(plain_result)
        now = json.loads(json.dumps(saved))
        now["fe"]["loading"] = "tension"
        assert not any(r["Quantity"].startswith("FE ")
                       for r in scenario.comparison_rows(saved, now))
        now["fe"] = None
        assert not any(r["Quantity"].startswith("FE ")
                       for r in scenario.comparison_rows(saved, now))

    def test_a_zero_saved_value_has_no_change(self):
        rows = scenario.comparison_rows({"Vp_percent": 0.0}, {"Vp_percent": 1.0})
        assert rows == [{"Quantity": "Void content Vp (%)", "Saved run": 0.0,
                         "This run": 1.0, "Change (%)": None}]

    def test_summary_without_an_fe_solve(self, plain_result):
        summary = scenario.scenario_summary({**plain_result, "fe_field": None})
        assert summary["fe"] is None
        json.dumps(summary)
