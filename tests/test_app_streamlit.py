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
        # Export tab offers the JSON/CSV downloads.
        assert len(at.get("download_button")) >= 2

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
        # ...and all five download buttons (2 export + 3 NCR) coexist.
        assert len(at.get("download_button")) == 5
