"""The Stress tab.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import streamlit as st

from ..plots import _STRESS_COMPONENTS, _STRESS_SLUGS, default_stress_component, plot_stress
from ..tabs.common import _placeholder_tab, _show_figure


def _sync_stress_component(default: str) -> None:
    """Point the Stress tab's component selector at the load case's default.

    Same rule as :func:`_sync_default_nz`: the selector follows
    :func:`default_stress_component` while it still shows the previous
    default (or has no value yet); a component the user picked stays. Must
    run before the selector is created in the script run.
    """
    previous = st.session_state.get("_stress_component_default")
    current = st.session_state.get("stress_component")
    if current is None or current == previous or current not in _STRESS_COMPONENTS:
        st.session_state["stress_component"] = default
    st.session_state["_stress_component_default"] = default


def _build_stress_tab(result: dict | None):
    if result is None:
        _placeholder_tab()
        return
    if result.get("fe_field") is None:
        reason = result.get("fe_skipped_reason")
        st.warning(
            f"FE solve was skipped: {reason}" if reason
            else "No FE field available for this configuration."
        )
        return
    # Opens on the component that governs the load case; a component the
    # user picks stays picked (the default goes through session state).
    _sync_stress_component(default_stress_component(result))
    comp_name = st.selectbox(
        "Stress component",
        options=list(_STRESS_COMPONENTS.keys()),
        key="stress_component",
    )
    slug = _STRESS_SLUGS[_STRESS_COMPONENTS[comp_name][0]]
    _show_figure(plot_stress(result, comp_name), f"stress_{slug}", "dl_png_stress")
    formulation = getattr(result["fe_field"], "formulation", None)
    if formulation:
        st.caption(f"FE element formulation: {formulation}.")
    thermal = result.get("fe_thermal")
    if thermal:
        st.caption(
            f"Thermal residual stress of the free-standing laminate at "
            f"ΔT = {thermal['delta_T']:g} K"
            + (" (no mechanical load)." if thermal["mode"] == "thermal_only"
               else f", plus the {result['config']['loading_mode']} load: the "
                    f"plot shows total stress."))
