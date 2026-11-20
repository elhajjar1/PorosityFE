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
from ..plots import (
    plot_buckling_mode,
    plot_fe_uq,
    plot_fracture_uq,
    plot_front,
    plot_growth,
    plot_hybrid,
    plot_open_hole,
    plot_results,
    plot_uq,
)
from ..runners import (
    FE_UQ_DEFAULT_SAMPLES,
    FE_UQ_MAX_SAMPLES,
    GROWTH_DEFAULT_STEPS,
    GROWTH_MAX_STEPS,
    OPEN_HOLE_DEFAULT_DIAMETER,
    OPEN_HOLE_DEFAULT_WIDTH,
    buckling_unavailable_reason,
    fe_uq_unavailable_reason,
    growth_unavailable_reason,
    hybrid_default_spec,
    hybrid_unavailable_reason,
    open_hole_unavailable_reason,
    run_buckling_cached,
    run_fe_uq_cached,
    run_growth_cached,
    run_hybrid_cached,
    run_open_hole_cached,
    run_uq_cached,
)
from ..scenario import REFERENCE_KEY, comparison_rows, scenario_summary
from ..sidebar import _offer_nz_fix, _seed
from ..summaries import (
    durability_table,
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
    fe_uq_summary,
    growth_summary,
    growth_table,
    hybrid_summary,
    open_hole_summary,
    sensitivity_caption,
    sensitivity_rows,
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
    _build_scenario_comparison(result)
    _build_durability_expander(result)
    _build_buckling_expander(result)
    _build_open_hole_expander(result)
    _build_hybrid_expander(result)
    _build_growth_expander(result)
    _build_uq_expander(result)
    _build_fe_uq_expander(result)
    _build_convergence_expander(result)


def _build_buckling_expander(result: dict) -> None:
    """Opt-in linear buckling of the analysis' FE model, against the void-free one."""
    with st.expander("Linear buckling (FE)"):
        st.caption(
            "Eigenvalue buckling of the coupon under the analysis' compression or "
            "shear, on its supports: the ends are held straight and the far one "
            "may sway. Linear: no imperfection and no post-buckling path. The "
            "porosity lowers the buckling load through the degraded stiffness of "
            "each element, so voids near the surfaces cost more than voids at "
            "the mid-plane. Two more FE solves and eigen-solves (the model and "
            "the same model without voids).")
        reason = buckling_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        key = _config_to_key(result["config"])
        if st.button("Run buckling analysis", key="buckling_run"):
            st.session_state["buckling_params"] = key
        if st.session_state.get("buckling_params") != key:
            return
        with st.spinner("Solving the buckling eigenproblem..."):
            try:
                out = run_buckling_cached(key, result)
            except (ValueError, RuntimeError) as exc:
                st.error(f"The buckling analysis did not run: {exc}")
                return
        buckled, reference = out["result"], out["reference"]
        knockdown = buckled.knockdown(reference)
        stress = buckled.critical_stress
        c = st.columns(4)
        c[0].metric("Buckling load factor", f"{buckled.load_factor:.4g}",
                    help="Multiple of the analysis' applied strain at buckling.")
        c[1].metric("Critical strain", f"{buckled.critical_strain:.3e}")
        c[2].metric("Critical stress", "n/a" if stress is None else f"{stress:.4g} MPa",
                    help="Average axial (or shear) stress at buckling.")
        c[3].metric("Of the void-free laminate", "n/a" if knockdown is None
                    else f"{knockdown:.4f}",
                    help="Critical stress (and load) over that of the same model "
                         "without voids.")
        rows = [{"Mode": i + 1, "Load factor": f"{v:.4g}"}
                for i, v in enumerate(buckled.load_factors)]
        if buckled.interpenetration is not None:
            for row, closing in zip(rows, buckled.interpenetration, strict=True):
                row["Delamination faces close by"] = f"{closing:.3f}"
            st.caption(
                "With a delamination the eigenproblem has no crack-face contact: "
                "a mode whose faces close (above a few per cent of its largest "
                "displacement) is not physical, and the first mode that opens "
                "is the one to read.")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        _show_figure(plot_buckling_mode(result["mesh"], buckled.modes[0],
                                        buckled.load_factor), "buckling_mode",
                     "dl_png_buckling")


def _build_open_hole_expander(result: dict) -> None:
    """Opt-in open-hole coupon of the analysed laminate (FE)."""
    with st.expander("Open-hole notch (FE)"):
        st.caption(
            "Meshes the open-hole coupon of ASTM D5766 (tension) and D6484 "
            "(compression) around its hole, with the analysis' material, layup, "
            "porosity and load, and reads the stress concentration and the "
            "stress along the net section, which the notched-strength criteria "
            "of Whitney and Nuismer take. One FE solve, a few seconds for a "
            "thin laminate and about a minute for 24 plies.")
        reason = open_hole_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        c = st.columns(4)
        diameter = float(c[0].number_input(
            "Hole diameter (mm)", min_value=0.5, max_value=50.0,
            value=OPEN_HOLE_DEFAULT_DIAMETER, step=0.5, key="oh_diameter"))
        width = float(c[1].number_input(
            "Coupon width (mm)", min_value=2.0, max_value=200.0,
            value=OPEN_HOLE_DEFAULT_WIDTH, step=1.0, key="oh_width"))
        d0 = c[2].number_input(
            "Point-stress distance d0 (mm)", min_value=0.01, max_value=20.0, value=None,
            step=0.1, key="oh_d0", placeholder="from notched tests",
            help="The material's characteristic distance (about 1 mm in Whitney and "
                 "Nuismer's data); no default.")
        strength = c[3].number_input(
            "Unnotched laminate strength (MPa)", min_value=1.0, max_value=1e5, value=None,
            step=10.0, key="oh_strength", placeholder="optional",
            help="Of this laminate at this void content, for the notched strength.")
        if diameter / width > 0.6:
            st.warning("The hole is too large for the mapped mesh: D / W must be at "
                       "most 0.6.")
            return
        key = _config_to_key(result["config"])
        params = (diameter, width)
        if st.button("Run open-hole analysis", key="oh_run"):
            st.session_state["oh_params"] = (key, params)
        if st.session_state.get("oh_params") != (key, params):
            return
        with st.spinner("Solving the open-hole coupon..."):
            try:
                out = run_open_hole_cached(key, params, result)
            except (ValueError, RuntimeError) as exc:
                st.error(f"The open-hole analysis did not run: {exc}")
                return
        rows, caption = open_hole_summary(
            out, result.get("fe_field"), d0=None if d0 is None else float(d0),
            unnotched_strength=None if strength is None else float(strength))
        _metric_row(rows)
        st.caption(caption)
        _show_figure(plot_open_hole(out["analysis"], None if d0 is None else float(d0)),
                     "open_hole_net_section", "dl_png_open_hole")


def _build_hybrid_expander(result: dict) -> None:
    """Opt-in hybrid laminate: the analysis' coupon with its own ply stack (FE)."""
    with st.expander("Hybrid laminate (FE)"):
        st.caption(
            "Gives every ply its own material and thickness and solves the "
            "analysis' coupon, layup, porosity and load on it: every element takes "
            "its ply's stiffness, porosity micromechanics and strengths. Write the "
            "plies bottom to top as comma-separated blocks `NAME*COUNT@MM` (a preset "
            "name, or a custom material by its name; `*COUNT` and `@MM` optional, "
            "the thickness defaulting to the material's ply thickness). The layup "
            "is tiled to the plies. One FE solve.")
        reason = hybrid_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        spec = st.text_input("Ply stack (bottom to top)", value=hybrid_default_spec(result),
                             key="hybrid_spec")
        key = _config_to_key(result["config"])
        if st.button("Run hybrid laminate", key="hybrid_run"):
            st.session_state["hybrid_params"] = (key, spec)
        if st.session_state.get("hybrid_params") != (key, spec):
            return
        with st.spinner("Solving the hybrid laminate..."):
            try:
                out = run_hybrid_cached(key, spec, result)
            except (ValueError, RuntimeError) as exc:
                st.error(f"The hybrid laminate did not run: {exc}")
                return
        rows, caption = hybrid_summary(out, result.get("fe_field"))
        _metric_row(rows)
        st.caption(caption)
        _show_figure(plot_hybrid(out), "hybrid_failure_index", "dl_png_hybrid")


def _build_growth_expander(result: dict) -> None:
    """Opt-in delamination growth of the analysis' crack, by node release."""
    with st.expander("Delamination growth (FE)"):
        st.caption(
            "Grows the analysis' delamination node by node under its load case: "
            "each step releases the front nodes with the smallest critical load "
            "factor and solves the grown crack (one FE solve per step). "
            "Quasi-static growth gives the critical load against the crack; "
            "fatigue growth integrates a Paris law da/dN = C (G_max / G_c)^m with "
            "your coefficients under a cyclic peak. Linear elastic between "
            "releases, no cohesive zone; the porosity enters through the "
            "stiffness and the toughness at each front node.")
        reason = growth_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        c = st.columns(2)
        steps = int(c[0].number_input(
            "Release steps (at most)", min_value=1, max_value=GROWTH_MAX_STEPS,
            value=GROWTH_DEFAULT_STEPS, step=1, key="growth_steps",
            help="Each step is one FE solve of the grown crack."))
        kind = c[1].radio("Growth", ("quasi-static", "fatigue"), horizontal=True,
                          key="growth_kind")
        C = m = threshold = peak = None
        if kind == "fatigue":
            f = st.columns(4)
            C = f[0].number_input("C (mm/cycle)", min_value=1e-12, value=None,
                                  format="%.4g", key="growth_C",
                                  placeholder="your coefficient",
                                  help="The rate at G_max = G_c. No default ships.")
            m = f[1].number_input("m", min_value=0.1, max_value=100.0, value=None,
                                  format="%.3g", key="growth_m",
                                  placeholder="your exponent",
                                  help="Delamination exponents are high, 5 to over 10.")
            threshold = f[2].number_input(
                "Threshold G_max / G_c", min_value=0.0, max_value=0.99, value=0.0,
                step=0.05, key="growth_threshold",
                help="Below it nothing grows; 0 for none.")
            peak = f[3].number_input(
                "Cyclic peak / static critical load", min_value=0.01, max_value=0.99,
                value=0.5, step=0.05, key="growth_peak",
                help="G_max / G_c starts at about its square.")
            if C is None or m is None:
                st.info("Enter the Paris-law coefficients C and m of your material: "
                        "none ship with PorosityFE.")
                return
        key = _config_to_key(result["config"])
        growth_key = (steps, kind, C, m, threshold, peak)
        if st.button("Run growth analysis", key="growth_run"):
            st.session_state["growth_params"] = (key, growth_key)
        if st.session_state.get("growth_params") != (key, growth_key):
            return
        with st.spinner("Growing the delamination..."):
            try:
                growth = run_growth_cached(key, growth_key, result)
            except (ValueError, RuntimeError) as exc:
                st.error(f"The growth analysis did not run: {exc}")
                return
        rows, caption = growth_summary(growth)
        _metric_row(rows)
        st.caption(caption)
        st.dataframe(growth_table(growth), hide_index=True, use_container_width=True)
        _show_figure(plot_growth(growth), "delamination_growth", "dl_png_growth")


def _build_durability_expander(result: dict) -> None:
    """Fatigue and hygrothermal knockdowns of the analysed laminate (empirical)."""
    with st.expander("Fatigue and environment (empirical)"):
        st.caption(
            "Multiplies each empirical knockdown by an S-N fatigue factor "
            "(log-linear, calibrated at R = 0.1) and a Chamis / Springer "
            "hygrothermal factor (matrix-dominated modes only), at the "
            "specimen-average void content. Screening-level models: "
            "production allowables come from a test matrix. The FE solve "
            "does not use them.")
        material = result["material"]
        c1, c2 = st.columns(2)
        with c1:
            _seed("dur_cycles", None)
            cycles = st.number_input(
                "Load cycles N", min_value=1.0, max_value=1e12, step=1000.0,
                format="%.0f", key="dur_cycles", placeholder="no fatigue")
            _seed("dur_slope", 0.0)
            slope = st.number_input(
                "S-N porosity slope K", min_value=0.0, max_value=1000.0, step=1.0,
                key="dur_slope",
                help=("b(Vp) = b (1 + K Vp), Vp a fraction: the curve falls "
                      "faster in a more porous laminate. 0 keeps the slope "
                      "independent of porosity. No calibrated value is shipped."))
            _seed("dur_goodman", False)
            goodman = st.checkbox(
                "Goodman mean-stress correction", key="dur_goodman",
                help=("Moves the S-N curve to the stress ratios below along the "
                      "straight constant-life line through the calibrated point "
                      "and the static strength. Uncalibrated against porous-"
                      "laminate fatigue data."))
            R = R_c = None
            if goodman:
                _seed("dur_R", 0.1)
                R = st.number_input("R of tension, shear and ILSS (0 to 1)",
                                    min_value=0.0, max_value=1.0, step=0.05,
                                    format="%.2f", key="dur_R")
                _seed("dur_R_c", 10.0)
                R_c = st.number_input("R of compression (≥ 1)", min_value=1.0,
                                      max_value=1e6, step=1.0, format="%.1f",
                                      key="dur_R_c")
        with c2:
            _seed("dur_T", None)
            T = st.number_input("Service temperature (°C)", min_value=-100.0,
                                max_value=400.0, step=5.0, format="%.1f",
                                key="dur_T", placeholder="no hygrothermal factor")
            _seed("dur_M", 0.0)
            M = st.number_input("Moisture (wt %)", min_value=0.0, max_value=20.0,
                                step=0.1, format="%.2f", key="dur_M")
            _seed("dur_Tg", None)
            T_g = st.number_input(
                "Dry glass-transition temperature T_g (°C)", min_value=0.0,
                max_value=600.0, step=5.0, format="%.1f", key="dur_Tg",
                placeholder=("the material's" if material.T_g_dry is not None
                             else "required for the hygrothermal factor"))
        if cycles is None and T is None and not M:
            st.info("Enter the load cycles, a service temperature or a moisture "
                    "content.")
            return
        try:
            rows, notes, record = durability_table(
                result, cycles=cycles, mean_stress="goodman" if goodman else None,
                R=R, R_compression=R_c, porosity_slope=float(slope), T=T, M=M,
                T_g_dry=T_g)
        except ValueError as exc:
            st.warning(str(exc))
            return
        gov = record["governing"]
        st.metric("Governing knockdown", f"{gov['knockdown']:.3f}",
                  help=f"{gov['mode']} / {gov['model']}")
        st.dataframe(
            [{**r, **{k: f"{r[k]:.4f}" for k in
                      ("Porosity", "Fatigue", "Environment", "Knockdown")},
              "Strength (MPa)": f"{r['Strength (MPa)']:.1f}"} for r in rows],
            hide_index=True, use_container_width=True)
        for note in notes:
            st.caption(f"⚠ {note}")


def _stop_comparing() -> None:
    """On-click callback: forget the run of the loaded scenario."""
    st.session_state.pop(REFERENCE_KEY, None)


def _build_scenario_comparison(result: dict) -> None:
    """This run next to the run a loaded scenario was saved from."""
    reference = st.session_state.get(REFERENCE_KEY)
    if not reference:
        return
    rows = comparison_rows(reference["summary"], scenario_summary(result))
    if not rows:
        return
    name = reference.get("name")
    st.subheader("Compared with the loaded scenario" + (f": {name}" if name else ""))
    st.dataframe(
        [{"Quantity": r["Quantity"],
          "Saved run": f"{r['Saved run']:.4g}",
          "This run": f"{r['This run']:.4g}",
          "Change (%)": "" if r["Change (%)"] is None else f"{r['Change (%)']:+.2f}"}
         for r in rows],
        hide_index=True, use_container_width=True)
    st.caption(
        "Saved run: the numbers stored in the scenario file. Change: this run "
        "relative to the saved one. An FE result is compared only when both "
        "runs solved the same loading mode.")
    st.button("Stop comparing", key="scenario_stop_compare", on_click=_stop_comparing)


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
        ranking = uq["sensitivity"]["failure_stress"]
        st.markdown("**Which inputs drive the failure stress**")
        rows = sensitivity_rows(ranking)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(sensitivity_caption(ranking))


def _build_fe_uq_expander(result: dict) -> None:
    """Opt-in FE Monte Carlo over random in-plane porosity fields."""
    with st.expander("FE uncertainty (random porosity fields)"):
        st.caption(
            "Solves the analysis' FE model once per draw, each with its own "
            "seeded random map of void content in the plane, of the same "
            "average: the spread of the first-ply-failure load and the "
            "stiffness when the voids are arranged differently, and which "
            "features of the arrangement drive it. One FE solve per draw.")
        reason = fe_uq_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        c = st.columns(3)
        n_samples = int(c[0].number_input(
            "Draws", min_value=2, max_value=FE_UQ_MAX_SAMPLES,
            value=FE_UQ_DEFAULT_SAMPLES, step=1, key="feuq_samples",
            help="Each draw is one FE solve."))
        cov = c[1].number_input(
            "In-plane CoV of Vp", min_value=0.0, max_value=5.0, value=None, step=0.05,
            key="feuq_cov", placeholder="from your C-scans or CT",
            help="Point-to-point coefficient of variation of the void content, a "
                 "fraction (0.5 = 50 % of the local value). No default: it "
                 "describes your part.")
        length = c[2].number_input(
            "Correlation length (mm)", min_value=0.1, max_value=1000.0, value=None,
            step=0.5, key="feuq_length", placeholder="from your C-scans or CT",
            help="Points much closer than this have nearly the same void content.")
        if cov is None or length is None:
            st.info("Enter the in-plane CoV and the correlation length of the void "
                    "content: they describe your part, so there are no defaults.")
            return
        key = _config_to_key(result["config"])
        params = (n_samples, float(cov), float(length))
        if st.button("Run FE uncertainty study", key="feuq_run"):
            st.session_state["feuq_params"] = (key, params)
        if st.session_state.get("feuq_params") != (key, params):
            return
        with st.spinner(f"Solving {n_samples} draws..."):
            try:
                study = run_fe_uq_cached(key, params, result)
            except (ValueError, RuntimeError) as exc:
                st.error(f"The FE uncertainty study did not run: {exc}")
                return
        rows, caption = fe_uq_summary(study)
        _metric_row(rows)
        st.caption(caption)
        _show_figure(plot_fe_uq(study), "fe_uncertainty", "dl_png_fe_uq")
        try:
            ranking = study.ranking("first_ply_failure_load_factor")
        except ValueError:
            ranking = None
        st.markdown("**Which inputs drive the first-ply failure load**")
        table = sensitivity_rows(ranking)
        if table:
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(sensitivity_caption(ranking))
