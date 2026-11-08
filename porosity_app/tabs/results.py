"""The Results tab and its expanders.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import streamlit as st

from porosity_fe import (
    EmpiricalSolver,
)

from ..config import _FLEXURE_MODES, _config_to_key, calibration_note
from ..convergence import (
    _CONVERGENCE_LEVELS,
    convergence_plan,
    convergence_tables,
    convergence_unavailable_reason,
    run_convergence_cached,
)
from ..plots import plot_fracture_uq, plot_front, plot_results, plot_uq
from ..runners import run_uq_cached
from ..sidebar import _offer_nz_fix
from ..summaries import (
    fe_decomposition_summary,
    fe_delamination_summary,
    fe_failure_summary,
    fe_flexure_summary,
    fe_fracture_test_summary,
    fe_fracture_uq_summary,
    fe_front_rows,
    fe_onset_summary,
    fe_short_beam_summary,
    fe_thermal_summary,
)
from ..tabs.common import _metric_row, _placeholder_tab, _show_figure


def _build_results_tab(result: dict | None, layup_for_title: str):
    if result is None:
        _placeholder_tab()
        return
    extrapolated = calibration_note(result["config"]["Vp"])
    if extrapolated:
        st.warning(f"⚠ {extrapolated}")
    _show_figure(plot_results(result, layup_for_title), "knockdown_results",
                 "dl_png_results")
    _build_fe_failure_results(result)
    _build_short_beam_results(result)
    _build_flexure_results(result)
    _build_thermal_results(result)
    _build_defect_results(result)
    _build_uq_expander(result)
    _build_convergence_expander(result)


def _build_fe_failure_results(result: dict) -> None:
    """FE failure index, governing mode and per-mode maxima (any FE run)."""
    summary = fe_failure_summary(result)
    if summary is None:
        return
    rows, mode_rows, caption = summary
    st.subheader("FE failure")
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    if mode_rows:
        st.dataframe(mode_rows, hide_index=True, use_container_width=True)
    st.caption(caption)


def _build_flexure_results(result: dict) -> None:
    """Metrics of the ASTM D7264 flexure FE solve (nothing for other modes)."""
    summary = fe_flexure_summary(result)
    if summary is None:
        return
    procedure = result["fe_flexure"]["procedure"]
    st.subheader("FE flexure (ASTM D7264 procedure "
                 f"{'A, three-point' if procedure == '3pt' else 'B, four-point'})")
    rows, caption = summary
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    st.caption(caption)


def _build_convergence_expander(result: dict) -> None:
    """Opt-in mesh-convergence check of the analysis' FE solve (refinement_study)."""
    with st.expander("Mesh convergence check"):
        st.caption(
            "Solves the analysis' FE load case again on in-plane refinements "
            "of its mesh (nx and ny scaled, nz fixed) and reports how the "
            "results change. Global results such as the knockdown converge. "
            "The domain-maximum failure index sits at a free edge, corner or "
            "load point, where the linear-elastic stress is singular: it "
            "keeps growing with refinement, so it is mesh-dependent by "
            "nature. Read the interior values, which leave out a fixed margin "
            "along the edges.")
        reason = convergence_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        choice = st.radio("Levels", list(_CONVERGENCE_LEVELS), key="conv_levels",
                          horizontal=True,
                          help="Three levels give an observed order of "
                               "convergence and an extrapolated value; two "
                               "give only the change between them.")
        plan = convergence_plan(result, _CONVERGENCE_LEVELS[choice])
        st.dataframe(
            [{"Level": i,
              "Mesh": " × ".join(str(n) for n in lv["resolution"]),
              "Elements": f"{lv['n_elements']:,}",
              "Est. peak memory": f"{lv['memory_gb']:.2g} GB",
              "Est. time": ("unknown" if lv["time_s"] is None
                            else f"~{lv['time_s']:.0f} s")}
             for i, lv in enumerate(plan, start=1)],
            hide_index=True, use_container_width=True)
        times = [lv["time_s"] for lv in plan]
        total = ("an unknown time" if None in times
                 else f"about {sum(t for t in times if t is not None):.0f} s")
        st.caption(
            f"Before running: {len(plan)} levels, {total} in all (the "
            f"analysis' FE solve time scaled by the DOF ratio to the power "
            f"1.5; a rough guide). The levels run one after another, so the "
            f"peak memory is the finest level's.")
        cfg = result["config"]
        params = (_config_to_key(cfg), tuple(lv["resolution"] for lv in plan))
        if st.button("Check mesh convergence", key="conv_run"):
            st.session_state["conv_params"] = params
        if st.session_state.get("conv_params") != params:
            return
        with st.spinner("Solving the refined meshes..."):
            try:
                study = run_convergence_cached(*params)
            except ValueError as exc:
                st.error(f"The convergence check did not run: {exc}")
                return
        levels, verdicts = convergence_tables(study)
        st.dataframe(levels, hide_index=True, use_container_width=True)
        st.dataframe(verdicts, hide_index=True, use_container_width=True)
        st.caption(
            f"Verdicts at a tolerance of {study.tol:.0%}: 'converged' (the "
            f"estimated error of the finest mesh is within it), 'not "
            f"converged' (refine further), 'mesh-dependent' (grows with "
            f"refinement: a stress singularity, do not use the value). The "
            f"knockdown's change is relative to the stiffness loss 1 − KD. "
            f"Interior quantities leave out {study.interior_margin:.3g} mm "
            + ("around the supports and the loading band and from the side "
               "faces." if study.loading == "ilss" else
               "around the support and load lines and from the side faces."
               if study.loading in _FLEXURE_MODES else "from the lateral faces."))


def _build_short_beam_results(result: dict) -> None:
    """Metrics of the ASTM D2344 short-beam FE solve (nothing for other modes)."""
    summary = fe_short_beam_summary(result)
    if summary is None:
        return
    st.subheader("FE short-beam shear (ASTM D2344)")
    rows, caption = summary
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    st.caption(caption)


def _build_thermal_results(result: dict) -> None:
    """Metrics of the FE thermal load (nothing when it was off)."""
    thermal = result.get("fe_thermal")
    if not thermal:
        return
    what = ("thermal only" if thermal["mode"] == "thermal_only"
            else f"residual stress + {result['config']['loading_mode']}")
    st.subheader(f"FE thermal residual stress ({what}, ΔT = {thermal['delta_T']:g} K)")
    summary = fe_thermal_summary(result)
    if summary is None:
        st.warning(
            f"The thermal FE solve was skipped: {result.get('fe_skipped_reason')}")
        _offer_nz_fix(result.get("fe_nz_suggestion"), key="nz_fix_results")
        return
    rows, caption = summary
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    st.caption(caption)


def _build_defect_results(result: dict) -> None:
    """The Defects block: fracture specimen, delamination front, knockdown
    decomposition and interface onset (nothing when none was requested)."""
    test = fe_fracture_test_summary(result)
    front = fe_delamination_summary(result)
    split = fe_decomposition_summary(result)
    onset = fe_onset_summary(result)
    if test is None and front is None and split is None and onset is None:
        if result.get("delamination") is not None and result.get("fe_field") is None:
            st.warning("The delamination's FE solve was skipped: "
                       f"{result.get('fe_skipped_reason')}")
            _offer_nz_fix(result.get("fe_nz_suggestion"), key="nz_fix_defects")
        return
    st.subheader("Defects: delamination and interface onset")
    if front is not None and front[2]:
        st.warning(
            "⚠ Toughness knockdown uncalibrated: the toughness at the front "
            "was lowered by the porosity there with the placeholder "
            "coefficients b_I = 6.2, b_II = 14 (two-point literature "
            "estimates). The fracture load factor and the critical loads "
            "carry that uncertainty; enter your own coefficients, or none, in "
            "the sidebar.")
    if test is not None:
        st.markdown(f"**FE fracture specimen ({result['fe_specimen']['standard']}, "
                    f"{result['fe_loading'].upper()})**")
        rows, caption = test
        _metric_row(rows)
        st.caption(caption)
    if front is not None:
        st.markdown("**Delamination front (virtual crack closure)**")
        rows, caption, _ = front
        _metric_row(rows)
        st.caption(caption)
        st.dataframe(fe_front_rows(result), hide_index=True,
                     use_container_width=True)
        _show_figure(plot_front(result), "delamination_front", "dl_png_front")
    _build_fracture_uq_block(result)
    if split is not None:
        st.markdown("**Knockdown by defect**")
        rows, caption = split
        _metric_row(rows)
        st.caption(caption)
    if onset is not None:
        st.markdown("**Interface delamination onset (Brewer-Lagace)**")
        rows, table, caption = onset
        _metric_row(rows)
        if table:
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(caption)


def _build_fracture_uq_block(result: dict) -> None:
    """The λ_c uncertainty band of the Defects block (nothing when it was
    not requested)."""
    if fe_fracture_uq_summary(result) is None:
        return
    st.markdown("**λ_c uncertainty band (toughness scatter)**")
    compare = None
    if result.get("fe_fracture_uq") is not None:
        compare = st.number_input(
            "Compare with a load factor (share of draws below)", min_value=0.001,
            max_value=1e6, value=None, step=0.1, format="%.3f",
            key="frac_uq_compare", placeholder="e.g. your required load factor",
            help=("A multiple of the analysis load, like λ_c. The share is an "
                  "estimate under the entered CoVs, not a disposition."))
    rows, caption = fe_fracture_uq_summary(result, compare)
    _metric_row(rows)
    st.caption(caption)
    if result.get("fe_fracture_uq") is not None:
        _show_figure(plot_fracture_uq(result, compare), "lambda_c_band",
                     "dl_png_lambda_band")


def _build_uq_expander(result: dict):
    """Opt-in uncertainty propagation on the analysed laminate (plan 3.4)."""
    with st.expander("Uncertainty (Latin hypercube sampling)"):
        st.caption(
            "Propagates scatter in the Judd-Wright calibration coefficient, "
            "the measured porosity and the pristine strength through the "
            "empirical knockdown. Each CoV is a fraction (0.10 = 10 % of the "
            "value), not a percent of porosity. The defaults are assumptions; "
            "set them to your own data.")
        modes = list(EmpiricalSolver.PRISTINE_STRENGTH_KEY)
        cfg = result["config"]
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox(
                "Mode", modes, key="uq_mode",
                index=modes.index(cfg["loading_mode"])
                if cfg["loading_mode"] in modes else 0)
            n_samples = st.number_input(
                "Draws", min_value=50, max_value=5000, value=500, step=50,
                key="uq_samples")
        with c2:
            coef_cov = st.number_input(
                "Coefficient CoV", min_value=0.0, max_value=1.0, value=0.10,
                step=0.01, key="uq_coef_cov",
                help="Coefficient of variation of the coefficient, as a fraction: 0.10 means 10 %.")
            vp_cov = st.number_input(
                "Porosity CoV", min_value=0.0, max_value=1.0, value=0.10,
                step=0.01, key="uq_vp_cov",
                help="Coefficient of variation of the measured porosity, as a fraction: 0.10 means 10 %.")
        with c3:
            strength_cov = st.number_input(
                "Strength CoV", min_value=0.0, max_value=1.0, value=0.05,
                step=0.01, key="uq_strength_cov",
                help="Coefficient of variation of the pristine strength, as a fraction: 0.10 means 10 %.")
        params = (_config_to_key(cfg), mode, int(n_samples), float(coef_cov),
                  float(vp_cov), float(strength_cov))
        if st.button("Run uncertainty analysis", key="uq_run"):
            st.session_state["uq_params"] = params
        if st.session_state.get("uq_params") != params:
            return
        with st.spinner("Sampling..."):
            uq, messages = run_uq_cached(*params)
        for msg in messages:
            st.warning(msg)
        kd, fs = uq["knockdown"]["percentiles"], uq["failure_stress"]["percentiles"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Knockdown p5", f"{kd['p5']:.3f}", f"{fs['p5']:.0f} MPa",
                  delta_color="off")
        m2.metric("Knockdown p50", f"{kd['p50']:.3f}", f"{fs['p50']:.0f} MPa",
                  delta_color="off")
        m3.metric("Knockdown p95", f"{kd['p95']:.3f}", f"{fs['p95']:.0f} MPa",
                  delta_color="off")
        _show_figure(plot_uq(uq), f"knockdown_uq_{mode}", "dl_png_uq")
