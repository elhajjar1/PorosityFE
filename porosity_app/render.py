"""The page: ``_render`` lays out the sidebar and the tabs.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import logging

import streamlit as st

from .config import _config_to_key
from .runners import run_analysis_cached
from .sidebar import _DEFAULT_LAYUP, _build_sidebar_inputs, _validate_layup_inline
from .tabs.export import _build_export_tab
from .tabs.figures import _build_mesh_tab, _build_profile_tab
from .tabs.overview import _build_overview_tab
from .tabs.results import _build_results_tab
from .tabs.stress import _build_stress_tab

logger = logging.getLogger(__name__)


def _render():
    st.set_page_config(
        page_title="PorosityFE",
        page_icon=None,
        layout="wide",
    )
    st.title("PorosityFE — Composite Laminate Porosity Analysis")
    st.caption(
        "Predict strength and stiffness knockdown in porosity-degraded "
        "composite laminates. Adjust inputs in the sidebar, then click **Run analysis**."
    )

    # ---- Layup default and initial validation -----------------------------
    # The layup input takes its default only from st.session_state (the
    # widget has no value=), so it never overwrites a value the user typed;
    # the status badge shows immediately for the default.
    st.session_state.setdefault("layup_input", _DEFAULT_LAYUP)
    if "_layup_status" not in st.session_state:
        _validate_layup_inline()

    sidebar = _build_sidebar_inputs()
    if sidebar is None:
        return
    cfg = sidebar["cfg"]
    layup_str = sidebar["layup_str"]

    # ---- Run analysis (only when the button is pressed) --------------------
    if sidebar["run"]:
        with st.spinner("Running porosity analysis…"):
            try:
                st.session_state["result"] = run_analysis_cached(_config_to_key(cfg))
                st.session_state["layup_str"] = layup_str
            except Exception as exc:
                logger.exception("Analysis failed")
                st.session_state["result"] = None
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("result")
    layup_for_title = st.session_state.get("layup_str", layup_str)

    tabs = st.tabs(["Overview", "Profile", "Mesh", "Results", "Stress", "Export"])
    with tabs[0]:
        _build_overview_tab(result, layup_for_title)
    with tabs[1]:
        _build_profile_tab(result)
    with tabs[2]:
        _build_mesh_tab(result)
    with tabs[3]:
        _build_results_tab(result, layup_for_title)
    with tabs[4]:
        _build_stress_tab(result)
    with tabs[5]:
        _build_export_tab(result, layup_for_title)
