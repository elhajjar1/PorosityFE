"""Dedicated unit tests for :mod:`porosity_fe.reporting` (IMPROVEMENT_PLAN 6.6).

The happy-path shape of the export writers and the NCR builder is already
pinned in tests/test_integration.py (``TestExportHelpers`` /
``TestNCRExport``). This module covers what those leave out: disposition
bin boundaries and structural-class escalation, meta defaults, filename
sanitising, the in-memory vs on-disk serialiser agreement, the JSON
envelopes, Markdown / PDF rendering details, and building an export /
NCR from a real :class:`EmpiricalSolver` result.
"""

import csv
import datetime
import io
import json
import math
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from porosity_fe import FORMAT_EMPIRICAL_SWEEP, FORMAT_NCR, JSON_SCHEMA_VERSION, MATERIALS, build_empirical_pipeline
from porosity_fe import reporting
from porosity_fe.reporting import (
    STRUCTURAL_CLASSES,
    _format_csv_row,
    _ncr_text_lines,
    _sanitise_filename_component,
    _serialise_payload_csv,
    _serialise_payload_json,
    build_export_payload,
    build_ncr_record,
    download_filename_stem,
    governing_failure,
    recommend_disposition,
    serialise_ncr_json,
    serialise_ncr_markdown,
    serialise_ncr_pdf,
    write_results_csv,
)


def _result(Vp=3.0, empirical=None, **cfg_overrides):
    """A minimal ``run_analysis``-shaped result (Vp in percent, app convention)."""
    cfg = {
        "material_name": "T800_epoxy",
        "n_plies": 16,
        "t_ply": 0.125,
        "Vp": Vp,
        "distribution": "clustered",
        "void_shape": "penny",
        "nx": 4, "ny": 3, "nz": 2,
    }
    cfg.update(cfg_overrides)
    if empirical is None:
        empirical = {
            "compression": {
                "judd_wright": {"failure_stress": 1000.04, "knockdown": 0.91234},
                "linear": {"failure_stress": 950.0, "knockdown": 0.85},
            },
            "shear": {
                "judd_wright": {"failure_stress": 80.0, "knockdown": 0.88},
            },
        }
    return {"config": cfg, "empirical": empirical}


def _ncr(meta=None, **result_kwargs):
    return build_ncr_record(_result(**result_kwargs), meta or {})


# ----------------------------------------------------------------------
# Export payload
# ----------------------------------------------------------------------


class TestBuildExportPayload:
    """``build_export_payload`` flattens a result into the export schema."""

    def test_config_fields_renamed_and_mesh_formatted(self):
        """Config keys are renamed to export names and the mesh becomes 'NXxNYxNZ'."""
        cfg = build_export_payload(_result())["config"]
        assert cfg == {
            "material": "T800_epoxy",
            "n_plies": 16,
            "t_ply": 0.125,
            "Vp_percent": 3.0,
            "distribution": "clustered",
            "void_shape": "penny",
            "mesh": "4x3x2",
        }

    def test_extra_result_keys_are_dropped(self):
        """Unknown config keys and non-empirical result keys are not exported."""
        res = _result(angles=[0, 90])
        res["fe_field"] = object()
        payload = build_export_payload(res)
        assert set(payload) == {"config", "empirical"}
        assert "angles" not in payload["config"]

    def test_empirical_table_renames_failure_stress_and_keeps_every_entry(self):
        """Every (mode, model) pair survives with failure_stress -> failure_stress_MPa."""
        res = _result()
        emp = build_export_payload(res)["empirical"]
        assert {m: set(v) for m, v in emp.items()} == {
            m: set(v) for m, v in res["empirical"].items()}
        for mode, models in res["empirical"].items():
            for model, r in models.items():
                assert emp[mode][model] == {
                    "failure_stress_MPa": r["failure_stress"],
                    "knockdown": r["knockdown"],
                }

    def test_payload_does_not_alias_result(self):
        """Mutating the payload must not write back into the analysis result."""
        res = _result()
        payload = build_export_payload(res)
        payload["empirical"]["compression"]["judd_wright"]["knockdown"] = -1
        assert res["empirical"]["compression"]["judd_wright"]["knockdown"] == 0.91234

    def test_empty_empirical_table(self):
        """A result with no empirical rows gives an empty table, not an error."""
        assert build_export_payload(_result(empirical={}))["empirical"] == {}

    def test_accepts_real_failure_results(self):
        """Works on ``get_all_failure_loads`` output (FailureResult dict shim)."""
        mat = MATERIALS['T800_epoxy']
        _, _, emp = build_empirical_pipeline(mat, 0.02, mesh_res=(4, 3, 3))
        res = _result(Vp=2.0, empirical=emp.get_all_failure_loads())
        payload = build_export_payload(res)
        jw = payload["empirical"]["compression"]["judd_wright"]
        # QI layup -> layup scale 1.0, so KD = exp(-alpha_QI * Vp).
        assert jw["knockdown"] == pytest.approx(math.exp(-6.9 * 0.02), rel=1e-12)
        assert jw["failure_stress_MPa"] == pytest.approx(mat.sigma_1c * jw["knockdown"], rel=1e-12)
        json.dumps(payload)  # plain floats only


class TestCsvSerialisation:
    """CSV row formatting and the file / in-memory writers."""

    def test_format_csv_row_precision(self):
        """Stress to 1 dp and knockdown to 4 dp, with standard rounding."""
        row = _format_csv_row("ilss", "linear", {"failure_stress_MPa": 99.96, "knockdown": 0.123456})
        assert row == ["ilss", "linear", "100.0", "0.1235"]

    def test_format_csv_row_accepts_numpy_scalars(self):
        """numpy floats format identically to Python floats."""
        import numpy as np
        row = _format_csv_row("m", "x", {"failure_stress_MPa": np.float64(1.25), "knockdown": np.float32(0.5)})
        assert row[2:] == ["1.2", "0.5000"]

    def test_file_writer_matches_in_memory_serialiser(self, tmp_path):
        """``write_results_csv`` and ``_serialise_payload_csv`` emit identical text."""
        payload = build_export_payload(_result())
        path = tmp_path / "out.csv"
        write_results_csv(str(path), payload)
        with open(path, encoding="utf-8", newline="") as f:
            assert f.read() == _serialise_payload_csv(payload)

    def test_config_comment_lines_precede_header_in_order(self):
        """One '# key: value' line per config field, in payload order, then the header."""
        payload = build_export_payload(_result())
        lines = _serialise_payload_csv(payload).splitlines()
        n_cfg = len(payload["config"])
        expected = [f"# {k}: {v}" for k, v in payload["config"].items()]
        assert lines[:n_cfg] == expected
        assert lines[n_cfg] == "mode,model,failure_stress_MPa,knockdown"

    def test_values_with_commas_are_quoted(self):
        """Mode/model labels containing commas survive a csv round-trip."""
        payload = build_export_payload(_result(empirical={
            "compression": {"user, custom": {"failure_stress": 1.0, "knockdown": 0.5}}}))
        text = _serialise_payload_csv(payload)
        rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].startswith("#")]
        assert rows[1] == ["compression", "user, custom", "1.0", "0.5000"]

    def test_empty_table_writes_header_only(self):
        """No empirical rows -> comment block plus header, nothing else."""
        text = _serialise_payload_csv(build_export_payload(_result(empirical={})))
        data = [line for line in text.splitlines() if not line.startswith("#")]
        assert data == ["mode,model,failure_stress_MPa,knockdown"]


class TestPayloadJsonEnvelope:
    """``_serialise_payload_json`` wraps the payload in the standard envelope."""

    def test_envelope_fields_and_payload(self):
        """Schema version, format, provenance and units precede the payload keys."""
        payload = build_export_payload(_result())
        data = json.loads(_serialise_payload_json(payload))
        assert data["schema_version"] == JSON_SCHEMA_VERSION
        assert data["format"] == FORMAT_EMPIRICAL_SWEEP
        assert isinstance(data["provenance"], dict)
        assert data["units"]["Vp_percent"] == "%"
        assert data["units"]["failure_stress"] == "MPa"
        assert data["config"] == payload["config"]
        assert data["empirical"] == payload["empirical"]
        assert list(data)[:4] == ["schema_version", "format", "provenance", "units"]

    def test_numpy_values_serialise(self):
        """numpy scalars in the payload are handled by the shared JSON default."""
        import numpy as np
        payload = build_export_payload(_result(empirical={
            "tension": {"judd_wright": {"failure_stress": np.float64(2.5), "knockdown": np.float32(0.75)}}}))
        data = json.loads(_serialise_payload_json(payload))
        assert data["empirical"]["tension"]["judd_wright"] == {"failure_stress_MPa": 2.5, "knockdown": 0.75}


# ----------------------------------------------------------------------
# Filenames
# ----------------------------------------------------------------------


class TestFilenames:
    """Filename-fragment sanitising and the templated download stem."""

    @pytest.mark.parametrize("raw, expected", [
        ("T800/epoxy", "T800_epoxy"),
        ("NCR 12", "NCR_12"),
        ("a / b", "a___b"),
        ("clean-name_1.0", "clean-name_1.0"),
        ("", ""),
    ])
    def test_sanitise_replaces_slashes_and_spaces(self, raw, expected):
        """Forward slashes and spaces become underscores; other characters are kept."""
        assert _sanitise_filename_component(raw) == expected

    def test_sanitise_stringifies_non_strings(self):
        """Non-string inputs are converted with ``str`` first."""
        assert _sanitise_filename_component(12) == "12"
        assert _sanitise_filename_component(None) == "None"

    def test_stem_falls_back_to_fraction_Vp(self):
        """Without ``Vp_percent`` the stem converts a fraction ``Vp`` to percent."""
        stem = download_filename_stem({"config": {"material": "IM7", "Vp": 0.025}})
        assert "_Vp2.5pct_" in stem

    def test_stem_prefers_Vp_percent_over_Vp(self):
        """``Vp_percent`` wins when both keys are present."""
        stem = download_filename_stem({"config": {"material": "IM7", "Vp_percent": 4.0, "Vp": 0.5}})
        assert "_Vp4.0pct_" in stem

    def test_stem_defaults_for_missing_fields(self):
        """Missing material and porosity fall back to 'unknown' and 0.0 %."""
        today = datetime.date.today().strftime("%Y%m%d")
        assert download_filename_stem({"config": {}}) == f"porosity_unknown_Vp0.0pct_{today}"

    def test_stem_rounds_to_one_decimal_and_has_no_extension(self):
        """Vp is shown to one decimal place and no file extension is appended."""
        stem = download_filename_stem({"config": {"material": "m", "Vp_percent": 3.26}})
        assert re.fullmatch(r"porosity_m_Vp3\.3pct_\d{8}", stem)


# ----------------------------------------------------------------------
# Governing failure and disposition
# ----------------------------------------------------------------------


class TestGoverningFailure:
    """``governing_failure`` picks the minimum knockdown across modes and models."""

    def test_min_can_come_from_non_default_model(self):
        """The worst case is not tied to the Judd-Wright model."""
        worst = governing_failure(_result())
        assert (worst["mode"], worst["model"], worst["knockdown"]) == ("compression", "linear", 0.85)
        assert worst["residual_strength_MPa"] == 950.0

    def test_empty_empirical_raises(self):
        """No data at all is an explicit error, not a silent default."""
        with pytest.raises(ValueError, match="no empirical knockdown data"):
            governing_failure(_result(empirical={}))

    def test_modes_without_models_raise(self):
        """Modes present but with empty model tables still count as no data."""
        with pytest.raises(ValueError, match="no empirical knockdown data"):
            governing_failure(_result(empirical={"compression": {}, "ilss": {}}))


class TestRecommendDisposition:
    """Severity bins, their inclusive boundaries, and class-dependent actions."""

    UAI = "Use-As-Is (UAI) — pending MRB concurrence"
    UAI_EE = "Use-As-Is with Engineering Evaluation"
    EE_REPAIR = "Engineering Evaluation / Repair"
    SCRAP = "Repair or Scrap"

    @pytest.mark.parametrize("Vp, kd, expected", [
        (0.0, 1.0, UAI),
        (1.0, 0.95, UAI),            # both bounds inclusive
        (1.0001, 0.99, UAI_EE),      # Vp just over the UAI bound
        (0.5, 0.9499, UAI_EE),       # kd just under the UAI bound
        (2.0, 0.90, UAI_EE),
        (2.0001, 0.95, EE_REPAIR),
        (0.5, 0.8999, EE_REPAIR),
        (5.0, 0.80, EE_REPAIR),
        (5.0001, 0.99, SCRAP),
        (0.1, 0.7999, SCRAP),
    ])
    def test_bins_and_boundaries(self, Vp, kd, expected):
        """Both Vp (percent) and knockdown must satisfy a bin; ties go to the milder bin."""
        assert recommend_disposition(Vp, kd)["path"] == expected

    def test_default_structural_class_is_primary(self):
        """Omitting the class behaves like 'primary'."""
        assert recommend_disposition(1.5, 0.92) == recommend_disposition(1.5, 0.92, "primary")

    def test_record_fields(self):
        """Every recommendation carries the same keys and echoes the class."""
        for cls in STRUCTURAL_CLASSES:
            d = recommend_disposition(3.0, 0.85, cls)
            assert set(d) == {"path", "structural_class", "rationale", "cited_criteria",
                              "required_mrb_actions", "disclaimer"}
            assert d["structural_class"] == cls
            assert "NOT a final disposition" in d["disclaimer"]

    def test_rationale_reports_vp_and_retained_strength(self):
        """The rationale quotes Vp to 2 dp and retained strength as a percent to 1 dp."""
        d = recommend_disposition(1.234, 0.9166)
        assert "(1.23%)" in d["rationale"]
        assert "91.7% retained" in d["rationale"]

    def test_class_specific_actions(self):
        """Primary asks for DER concurrence, non-structural for a function check, secondary neither."""
        acts = {c: recommend_disposition(0.5, 0.99, c)["required_mrb_actions"] for c in STRUCTURAL_CLASSES}
        base = acts["secondary"]
        assert len(acts["primary"]) == len(base) + 1
        assert "DER" in acts["primary"][-1]
        assert len(acts["non-structural"]) == len(base) + 1
        assert "fluid-ingress" in acts["non-structural"][-1]
        assert acts["primary"][:len(base)] == base
        assert acts["non-structural"][:len(base)] == base

    @pytest.mark.parametrize("Vp, kd, has_repair_action", [
        (0.5, 0.99, False),   # UAI
        (1.5, 0.92, False),   # UAI with EE
        (3.0, 0.85, True),    # EE / Repair
        (8.0, 0.50, True),    # Repair or Scrap
    ])
    def test_repair_action_only_for_repair_paths(self, Vp, kd, has_repair_action):
        """The repair-scheme action is added exactly when the path mentions repair."""
        acts = recommend_disposition(Vp, kd, "secondary")["required_mrb_actions"]
        assert any("repair scheme" in a for a in acts) is has_repair_action

    def test_returned_lists_are_fresh(self):
        """Mutating one recommendation's lists must not leak into the next call."""
        d1 = recommend_disposition(0.5, 0.99, "secondary")
        d1["required_mrb_actions"].append("tampered")
        d1["cited_criteria"].clear()
        d2 = recommend_disposition(0.5, 0.99, "secondary")
        assert "tampered" not in d2["required_mrb_actions"]
        assert d2["cited_criteria"]

    @pytest.mark.parametrize("bad", ["Primary", "", "tertiary", None])
    def test_unknown_class_rejected_with_choices(self, bad):
        """Case-sensitive exact match; the message lists the valid classes."""
        with pytest.raises(ValueError, match="non-structural"):
            recommend_disposition(0.5, 0.99, bad)


# ----------------------------------------------------------------------
# NCR record
# ----------------------------------------------------------------------


class TestBuildNcrRecord:
    """``build_ncr_record`` derives the technical content from the result."""

    def test_meta_defaults(self):
        """Empty meta -> blank preparer/reference/note, today's date, primary class, placeholder layup."""
        ncr = _ncr({})
        s = ncr["summary"]
        assert s["prepared_by"] == "" and s["ncr_reference"] == "" and s["note"] == ""
        assert s["date"] == datetime.date.today().isoformat()
        assert s["structural_class"] == "primary"
        assert ncr["nonconformance"]["layup"] == "(see analysis configuration)"
        assert ncr["recommended_disposition"]["structural_class"] == "primary"

    def test_falsy_date_and_layup_fall_back(self):
        """Empty-string date / layup are treated like missing values."""
        ncr = _ncr({"date": "", "layup": ""})
        assert ncr["summary"]["date"] == datetime.date.today().isoformat()
        assert ncr["nonconformance"]["layup"] == "(see analysis configuration)"

    def test_structural_class_threads_into_disposition(self):
        """The meta class drives both the summary and the disposition actions."""
        ncr = _ncr({"structural_class": "non-structural"})
        assert ncr["summary"]["structural_class"] == "non-structural"
        assert any("fluid-ingress" in a for a in ncr["recommended_disposition"]["required_mrb_actions"])

    def test_unknown_structural_class_propagates(self):
        """A typo in the class is rejected, not silently replaced."""
        with pytest.raises(ValueError, match="structural_class"):
            _ncr({"structural_class": "primry"})

    def test_disposition_matches_direct_call(self):
        """The embedded recommendation equals ``recommend_disposition`` on (Vp, worst KD)."""
        ncr = _ncr({"structural_class": "secondary"}, Vp=1.5)
        assert ncr["recommended_disposition"] == recommend_disposition(1.5, 0.85, "secondary")

    def test_nonconformance_mirrors_config(self):
        """Nonconformance fields come straight from the analysis configuration."""
        nc = _ncr({"layup": "[0/90]s"}, Vp=2.5)["nonconformance"]
        assert nc["material"] == "T800_epoxy"
        assert nc["n_plies"] == 16
        assert nc["t_ply_mm"] == 0.125
        assert nc["measured_Vp_percent"] == 2.5
        assert nc["distribution"] == "clustered"
        assert nc["void_shape"] == "penny"
        assert nc["analysis_mesh"] == "4x3x2"
        assert "2.50%" in nc["summary"] and "[0/90]s" in nc["summary"] and "16 plies" in nc["summary"]

    def test_engineering_analysis_uses_worst_case_and_full_table(self):
        """Governing entries come from the worst case; per_mode is the export table."""
        res = _result()
        ea = build_ncr_record(res, {})["engineering_analysis"]
        assert ea["governing_mode"] == "compression"
        assert ea["governing_model"] == "linear"
        assert ea["governing_knockdown"] == 0.85
        assert ea["governing_residual_strength_MPa"] == 950.0
        assert ea["per_mode"] == build_export_payload(res)["empirical"]

    def test_vp_percent_is_coerced_to_float(self):
        """An integer percent in the config is stored as float."""
        nc = _ncr({}, Vp=3)["nonconformance"]
        assert isinstance(nc["measured_Vp_percent"], float)

    def test_from_real_solver_result(self):
        """End to end: real solver output -> NCR whose governing case is the true minimum."""
        mat = MATERIALS['T800_epoxy']
        _, _, emp = build_empirical_pipeline(mat, 0.03, mesh_res=(4, 3, 3))
        table = emp.get_all_failure_loads()
        ncr = build_ncr_record(_result(Vp=3.0, empirical=table), {})
        all_kd = [r.knockdown for models in table.values() for r in models.values()]
        assert ncr["engineering_analysis"]["governing_knockdown"] == min(all_kd)
        json.loads(serialise_ncr_json(ncr))


# ----------------------------------------------------------------------
# NCR serialisers
# ----------------------------------------------------------------------


class TestSerialiseNcrJson:
    """``serialise_ncr_json`` wraps the record in the NCR envelope."""

    def test_envelope_and_round_trip(self):
        """Envelope header fields plus every NCR section survive a JSON round-trip."""
        ncr = _ncr({"prepared_by": "A. B."})
        data = json.loads(serialise_ncr_json(ncr))
        assert data["schema_version"] == JSON_SCHEMA_VERSION
        assert data["format"] == FORMAT_NCR
        assert data["units"]["governing_residual_strength"] == "MPa"
        for key in ("summary", "nonconformance", "engineering_analysis", "recommended_disposition"):
            assert data[key] == ncr[key]

    def test_does_not_mutate_record(self):
        """Serialising must not add envelope keys to the caller's record."""
        ncr = _ncr({})
        keys = set(ncr)
        serialise_ncr_json(ncr)
        assert set(ncr) == keys


class TestSerialiseNcrMarkdown:
    """Markdown attachment rendering."""

    def test_placeholders_for_blank_meta_and_no_note_line(self):
        """Blank preparer / reference render as an em dash; an empty note is omitted."""
        md = serialise_ncr_markdown(_ncr({}))
        assert "- Prepared by: —" in md
        assert "- Parent NCR reference: —" in md
        assert "Engineer note" not in md

    def test_note_rendered_when_present(self):
        """A non-empty engineer note gets its own bullet."""
        md = serialise_ncr_markdown(_ncr({"note": "C-scan hot spot"}))
        assert "- Engineer note: C-scan hot spot" in md

    def test_sections_in_order(self):
        """The five numbered sections appear once each, in order."""
        md = serialise_ncr_markdown(_ncr({}))
        heads = [line for line in md.splitlines() if line.startswith("## ")]
        assert [h.split(".")[0] for h in heads] == ["## 1", "## 2", "## 3", "## 4", "## 5"]

    def test_per_mode_table_rows_and_precision(self):
        """One table row per (mode, model), stress to 1 dp, knockdown to 3 dp."""
        md = serialise_ncr_markdown(_ncr({}))
        assert "| compression | judd_wright | 1000.0 | 0.912 |" in md
        assert "| compression | linear | 950.0 | 0.850 |" in md
        assert "| shear | judd_wright | 80.0 | 0.880 |" in md

    def test_governing_line(self):
        """The governing sentence quotes knockdown, retained percent and residual strength."""
        md = serialise_ncr_markdown(_ncr({}))
        assert ("**Governing (worst-case) case:** compression / linear — knockdown 0.850 "
                "(85.0% of pristine retained), residual strength 950.0 MPa.") in md

    def test_one_checkbox_per_required_action(self):
        """Each required MRB action becomes an unchecked task-list item."""
        ncr = _ncr({"structural_class": "non-structural"})
        md = serialise_ncr_markdown(ncr)
        boxes = [line for line in md.splitlines() if line.startswith("- [ ] ")]
        assert boxes == [f"- [ ] {a}" for a in ncr["recommended_disposition"]["required_mrb_actions"]]

    def test_cited_criteria_listed(self):
        """Every cited criterion is rendered as a bullet."""
        ncr = _ncr({})
        md = serialise_ncr_markdown(ncr)
        for c in ncr["recommended_disposition"]["cited_criteria"]:
            assert f"- {c}" in md

    def test_disclaimer_is_block_quote(self):
        """The disclaimer is rendered as a Markdown block quote."""
        ncr = _ncr({})
        assert f"> {ncr['recommended_disposition']['disclaimer']}" in serialise_ncr_markdown(ncr)


class TestNcrTextLinesAndPdf:
    """Plain-text line layout used by the PDF renderer, and pagination."""

    def test_plain_text_uses_ascii_placeholders(self):
        """The PDF text uses '-' for blank fields (no em dash) and no Markdown markup."""
        lines = _ncr_text_lines(_ncr({}))
        assert "Prepared by:               -" in lines
        assert "Parent NCR reference:      -" in lines
        assert not any(line.startswith(("#", "**", "|")) for line in lines)

    def test_title_upper_cased(self):
        """The first line is the upper-cased title."""
        ncr = _ncr({})
        assert _ncr_text_lines(ncr)[0] == ncr["summary"]["title"].upper()

    def test_long_note_is_wrapped_to_page_width(self):
        """Long free-text fields are wrapped so no line exceeds the PDF width."""
        note = "porosity " * 200
        lines = _ncr_text_lines(_ncr({"note": note}))
        assert max(len(line) for line in lines) <= reporting._PDF_WRAP
        note_lines = [line for line in lines if "porosity porosity" in line]
        assert len(note_lines) > 1

    def test_per_mode_rows_fixed_width(self):
        """Per-mode rows are column-aligned with knockdowns to 3 dp."""
        lines = _ncr_text_lines(_ncr({}))
        row = next(line for line in lines if line.strip().startswith("compression") and "linear" in line)
        assert row == f"  {'compression':<14}{'linear':<14}{'950.0':>14}{'0.850':>12}"

    @staticmethod
    def _page_count(blob: bytes) -> int:
        return len(re.findall(rb"/Type\s*/Page\b(?!s)", blob))

    @pytest.mark.parametrize("note", ["", "void " * 1500], ids=["no-note", "long-note"])
    def test_page_count_matches_line_budget(self, note):
        """Page count equals ceil(lines / lines-per-page), short or long record."""
        ncr = _ncr({"note": note})
        n_lines = len(_ncr_text_lines(ncr))
        expected = math.ceil(n_lines / reporting._PDF_LINES_PER_PAGE)
        assert self._page_count(serialise_ncr_pdf(ncr)) == expected

    def test_long_note_adds_pages(self):
        """A long engineer note produces more pages than the same record without it."""
        short = self._page_count(serialise_ncr_pdf(_ncr({})))
        long_ = self._page_count(serialise_ncr_pdf(_ncr({"note": "void " * 1500})))
        assert long_ > short >= 1

    def test_pdf_render_leaves_no_open_figures(self):
        """Every per-page figure is closed after rendering."""
        plt.close('all')
        serialise_ncr_pdf(_ncr({"note": "void " * 1500}))
        assert plt.get_fignums() == []
