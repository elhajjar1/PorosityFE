#!/usr/bin/env python3
"""Streamlit ``AppTest`` coverage for the app.py UI render paths.

These drive the real Streamlit script (sidebar -> tabs -> render) through
``streamlit.testing.v1.AppTest``, exercising ``_render``,
``_build_sidebar_inputs``, the tab builders, and the NCR export form — the
parts that need a live script-run context and so can't be unit-tested
directly (see tests/test_app.py for the headless helpers).
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(_APP_PATH, default_timeout=120)
    return at.run()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _run_small_analysis(at: AppTest) -> AppTest:
    """Drive a full analysis on a deliberately tiny mesh so the FE solve is
    fast: enable Expert mode to expose the mesh sliders, shrink them, use a
    short layup, then press Run."""
    at.toggle[0].set_value(True).run()          # Expert mode -> mesh sliders
    at.text_input(key="layup_input").set_value("0/90/90/0").run()
    at.slider[0].set_value(8)   # nx
    at.slider[1].set_value(4)   # ny
    at.slider[2].set_value(4)   # nz
    at.run()
    _button(at, "Run analysis").click().run()
    return at


class TestAppRender:
    def test_initial_render_shows_placeholders(self):
        at = _fresh_app()
        assert not at.exception
        # Six tabs, no analysis yet -> placeholder infos and a valid-layup badge.
        assert len(at.tabs) == 6
        assert "result" not in at.session_state or at.session_state["result"] is None
        assert at.session_state["_layup_status"][0] == "ok"
        assert len(at.info) >= 5  # overview note + 5 placeholder tabs

    def test_invalid_layup_disables_run_and_aborts_render(self):
        at = _fresh_app()
        at.text_input(key="layup_input").set_value("garbage ###").run()
        assert at.session_state["_layup_status"][0] == "err"
        # The Run button is disabled and the config build bails out before
        # any tabs are created (sidebar returns None -> _render returns).
        assert _button(at, "Run analysis").disabled is True
        assert len(at.tabs) == 0
        assert len(at.error) >= 1

    def test_run_populates_result_and_tabs(self):
        at = _run_small_analysis(_fresh_app())
        assert not at.exception
        result = at.session_state["result"]
        assert result is not None
        assert result["config"]["nx"] == 8  # the shrunk mesh was used
        assert result["fe_field"] is not None
        # With a result present the Stress tab renders its component selector.
        assert any(sb.label == "Stress component" for sb in at.selectbox)
        # Export tab offers the JSON/CSV downloads and the FE field as VTU,
        # and each of the four figures (profile, mesh, results, stress) a
        # PNG download.
        labels = [b.proto.label for b in at.get("download_button")]
        assert labels.count("Download PNG") == 4
        assert "Download FE results (VTU)" in labels
        assert len(labels) == 7

    def test_ncr_form_submit_generates_summary(self):
        """Submitting the NCR form with a blank parent reference must not
        crash. Regression for a StreamlitDuplicateElementId raised when the
        NCR 'Download JSON' button collided with the export one (both fell
        back to the same auto-generated id)."""
        at = _run_small_analysis(_fresh_app())
        _button(at, "Generate summary").click().run()
        assert not at.exception
        # Recommended-disposition guidance is surfaced...
        assert any("disposition" in w.value.lower() for w in at.warning)
        # ...and all ten download buttons (4 PNG + 3 export + 3 NCR) coexist.
        assert len(at.get("download_button")) == 10


_THERMAL_ONLY = "Thermal only (residual stress, no mechanical load)"
_COMBINED = "Add cure residual stress to the mechanical case"


def _select(at: AppTest, label: str, value) -> AppTest:
    next(s for s in at.selectbox if s.label == label).set_value(value)
    return at.run()


def _thermal_setup(at: AppTest, choice: str, nz: int) -> AppTest:
    """AS4/3501-6 (the CTE preset), the thermal option, the labelled ΔT
    suggestion, and a small [0/90]_s mesh with ``nz`` element layers."""
    _select(at, "Material", "AS4_3501_6_epoxy")
    at.radio(key="thermal_choice").set_value(choice).run()
    # ΔT starts empty (no hidden default) and the run is blocked until set.
    assert at.number_input(key="thermal_delta_T").value is None
    assert _button(at, "Run analysis").disabled is True
    _button(at, "Use suggestion (-150 K)").click().run()
    assert at.number_input(key="thermal_delta_T").value == -150.0
    at.toggle[0].set_value(True).run()          # Expert mode -> mesh sliders
    at.text_input(key="layup_input").set_value("0/90/90/0").run()
    at.slider[0].set_value(8)   # nx
    at.slider[1].set_value(4)   # ny
    at.slider[2].set_value(nz)  # nz
    return at.run()


class TestThermal:
    """IMPROVEMENT_PLAN 3.5 PR 4: the thermal load case in the sidebar."""

    def test_unavailable_without_cte_says_why(self):
        at = _fresh_app()                       # default material: no CTEs
        assert not any(r.label == "FE thermal load" for r in at.radio)
        captions = " ".join(c.value for c in at.caption)
        assert "Unavailable for T800_epoxy" in captions
        assert "AS4_3501_6_epoxy" in captions

    def test_thermal_only_run_shows_residual_results(self):
        at = _thermal_setup(_fresh_app(), _THERMAL_ONLY, nz=4)
        assert not any("element layer" in w.value for w in at.warning)
        _button(at, "Run analysis").click().run()
        assert not at.exception
        result = at.session_state["result"]
        assert result["config"]["thermal_mode"] == "thermal_only"
        assert result["config"]["delta_T"] == -150.0
        fe = result["fe_field"]
        assert fe is not None and fe.load_factor_basis == "delta_T"
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["FE knockdown"] == "n/a"
        assert metrics["Residual max FI"] == f"{fe.residual_max_failure_index:.3f}"
        assert metrics["Critical ΔT (first-ply failure)"] == (
            f"{fe.first_ply_failure_load_factor * -150.0:.0f} K")
        labels = [b.proto.label for b in at.get("download_button")]
        assert "Download FE results (VTU)" in labels

    def test_combined_run_reports_mechanical_basis(self):
        at = _thermal_setup(_fresh_app(), _COMBINED, nz=4)
        _select(at, "Loading mode", "tension")
        _button(at, "Run analysis").click().run()
        assert not at.exception
        result = at.session_state["result"]
        fe = result["fe_field"]
        assert fe.load_factor_basis == "mechanical_with_residual"
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["FE knockdown"] == f"{fe.knockdown:.3f}"
        assert metrics["FPF load factor"] == f"{fe.first_ply_failure_load_factor:.2f}"
        assert any("Without the residual stress" in c.value for c in at.caption)

    def test_unresolved_layup_is_explained_and_nz_offered(self):
        at = _thermal_setup(_fresh_app(), _THERMAL_ONLY, nz=6)
        # Before the run: the sidebar already names the nz that works.
        assert any("Use nz = 8" in w.value for w in at.warning)
        _button(at, "Run analysis").click().run()
        assert not at.exception
        result = at.session_state["result"]
        assert result["fe_field"] is None
        assert result["fe_nz_suggestion"] == 8
        assert "nz = k * n_plies" in result["fe_skipped_reason"]
        assert any("FE solve was skipped" in w.value for w in at.warning)
        # Empirical results still render.
        assert result["empirical"]
        _button(at, "Use nz = 8").click().run()
        assert not at.exception
        assert at.toggle[0].value is True
        assert at.slider[2].value == 8
        assert not any("Use nz = 8" in w.value for w in at.warning)
        _button(at, "Run analysis").click().run()
        assert at.session_state["result"]["fe_field"] is not None


def test_uq_expander_reports_percentiles():
    """IMPROVEMENT_PLAN 3.4: the Results tab runs uncertainty propagation."""
    at = _run_small_analysis(_fresh_app())
    _button(at, "Run uncertainty analysis").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Knockdown p5", "Knockdown p50", "Knockdown p95"]
    p5, p50, p95 = (float(m.value) for m in at.metric)
    assert p5 < p50 < p95
