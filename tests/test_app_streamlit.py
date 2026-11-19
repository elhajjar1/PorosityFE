#!/usr/bin/env python3
"""Streamlit ``AppTest`` coverage for the app.py UI render paths.

These drive the real Streamlit script (sidebar -> tabs -> render) through
``streamlit.testing.v1.AppTest``, exercising ``_render``,
``_build_sidebar_inputs``, the tab builders, and the NCR export form — the
parts that need a live script-run context and so can't be unit-tested
directly (see tests/test_app.py for the headless helpers).
"""

import logging
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
        # Export tab offers the JSON/CSV downloads, the FE field as VTU and
        # the material cards, and each of the four figures (profile, mesh,
        # results, stress) a PNG download; the sidebar the selected material
        # and the scenario.
        labels = [b.proto.label for b in at.get("download_button")]
        assert labels.count("Download PNG") == 4
        assert "Download FE results (VTU)" in labels
        assert "Download material cards" in labels
        assert "Download the selected material (JSON)" in labels
        assert "Download scenario (JSON)" in labels
        assert len(labels) == 10

    def test_ilss_run_shows_the_d2344_specimen(self):
        """ILSS solves the ASTM D2344 short-beam specimen: the sidebar says
        so before the run, the Results tab shows the short-beam metrics."""
        at = _fresh_app()
        next(sb for sb in at.selectbox if sb.label == "Loading mode") \
            .set_value("ilss").run()
        assert any("ASTM D2344 short-beam specimen" in c.value for c in at.caption)
        at = _run_small_analysis(at)
        assert not at.exception
        result = at.session_state["result"]
        assert result["fe_loading"] == "ilss"
        assert result["mesh"].ilss_geometry is not None
        assert any(h.value == "FE short-beam shear (ASTM D2344)" for h in at.subheader)
        assert {m.label for m in at.metric} >= {
            "Mid-plane shear / 0.75P/(bh)", "Apparent ILSS at FPF"}
        assert any("short-beam specimen" in s.value for s in at.success)

    def test_ncr_form_submit_generates_summary(self):
        """Submitting the NCR form with a blank parent reference must not
        crash. Regression for a StreamlitDuplicateElementId raised when the
        NCR 'Download JSON' button collided with the export one (both fell
        back to the same auto-generated id)."""
        at = _run_small_analysis(_fresh_app())
        _button(at, "Generate summary").click().run()
        assert not at.exception
        # Recommended-disposition guidance is surfaced, with the bands it
        # used, labelled illustrative (#229)...
        assert any("disposition" in w.value.lower() for w in at.warning)
        assert any(c.value.startswith("Disposition policy: <= 1 % voids")
                   and "not taken from a cited standard" in c.value for c in at.caption)
        # ...and all thirteen download buttons (2 sidebar + 4 PNG + 4 export
        # + 3 NCR) coexist.
        assert len(at.get("download_button")) == 13
        # No required retained strength was entered: no allowable is shown.
        assert not any("Allowable void content" in i.value for i in at.info)

    def test_material_cards_download(self, monkeypatch):
        """The Export tab offers the degraded ply as solver material cards,
        in either format and element family; a laminate the cards cannot
        describe gets a warning instead of the button."""
        def labels(app):
            return [b.proto.label for b in app.get("download_button")]
        at = _run_small_analysis(_fresh_app())
        assert "Download material cards" in labels(at)
        at.selectbox(key="cards_format").set_value("Nastran (.bdf)")
        at.selectbox(key="cards_element").set_value("solid")
        at.run()
        assert not at.exception
        assert "Download material cards" in labels(at)

        def refuse(*args, **kwargs):
            raise ValueError("no ply to describe")
        monkeypatch.setattr("porosity_app.tabs.export.material_cards", refuse)
        at.run()
        assert not at.exception
        assert "Download material cards" not in labels(at)
        assert any(w.value == "No material cards for this laminate: no ply to describe"
                   for w in at.warning)

    def test_ncr_required_retained_strength_adds_the_allowable(self):
        """The optional required retained strength (percent in the form, a
        fraction in the record) adds the allowable void content."""
        at = _run_small_analysis(_fresh_app())
        at.number_input(key="ncr_required_retained").set_value(90.0)
        _button(at, "Generate summary").click().run()
        assert not at.exception
        shown = [i.value for i in at.info if "Allowable void content" in i.value]
        assert len(shown) == 1
        assert shown[0].startswith(
            "**Allowable void content.** Governing allowable void content: ")
        assert "at 90% of pristine strength retained" in shown[0]
        assert "does not replace the porosity allowable" in shown[0]


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


class TestDefaultMesh:
    """The default nz is one element layer per ply of the layup as entered."""

    @staticmethod
    def _captions(at: AppTest) -> str:
        return " ".join(c.value for c in at.caption)

    def test_default_nz_follows_the_layup_until_moved(self):
        at = _fresh_app()                       # [0/45/-45/90]_3s: 24 plies
        assert "Default mesh: 30 × 10 × 24, one element layer per ply" in \
            self._captions(at)
        at.toggle[0].set_value(True).run()
        assert [s.value for s in at.slider] == [30, 10, 24]
        at.text_input(key="layup_input").set_value("0/90/90/0").run()
        assert at.slider[2].value == 4
        at.slider[2].set_value(6).run()         # the user's choice stays
        at.text_input(key="layup_input").set_value("[0/90]_2s").run()
        assert at.slider[2].value == 6
        assert not at.exception

    def test_thick_layup_caps_nz_and_thermal_offers_the_resolving_nz(self):
        """Above the cap the default no longer resolves the plies, which is
        when the thermal "Use nz = N" hint still appears."""
        at = _select(_fresh_app(), "Material", "AS4_3501_6_epoxy")
        at.text_input(key="layup_input").set_value("[0/45/-45/90]_8s").run()
        assert "30 × 10 × 48, nz capped at 48 for the 64-ply layup" in \
            self._captions(at)
        assert not any("Use nz" in w.value for w in at.warning)
        at.radio(key="thermal_choice").set_value(_THERMAL_ONLY).run()
        assert any("Use nz = 64" in w.value for w in at.warning)
        _button(at, "Use nz = 64").click().run()
        assert at.toggle[0].value is True
        assert at.slider[2].value == 64
        assert not any("Use nz" in w.value for w in at.warning)
        assert not at.exception


    def test_thick_blocked_layup_resolves_within_the_cap(self):
        """#226: 64 plies in blocks of four of one angle resolve with 48
        element layers, so the default keeps the stacking sequence and the
        thermal check is satisfied without changing nz."""
        at = _select(_fresh_app(), "Material", "AS4_3501_6_epoxy")
        at.text_input(key="layup_input").set_value("[0/0/0/0/90/90/90/90]_4s").run()
        assert ("30 × 10 × 48, 48 element layers, each within plies of one angle, "
                "for the 64-ply layup") in self._captions(at)
        at.radio(key="thermal_choice").set_value(_THERMAL_ONLY).run()
        assert not any("Use nz" in w.value for w in at.warning)
        assert not at.exception

def test_hashin_criterion_shows_the_failure_modes():
    """The sidebar criterion reaches the FE solve and the Results tab lists
    each mode's maximum, including delamination onset."""
    at = _fresh_app()
    _select(at, "FE failure criterion", "hashin")
    at = _run_small_analysis(at)
    assert not at.exception
    result = at.session_state["result"]
    assert result["config"]["failure_criterion"] == "hashin"
    assert result["fe_field"].failure_criterion == "hashin"
    assert any(h.value == "FE failure" for h in at.subheader)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Criterion"] == "Hashin"
    table = at.dataframe[0].value
    assert "Delamination onset" in list(table["Mode"])


def test_porosity_degradation_reaches_the_fe_solve():
    """The sidebar's FE porosity degradation (default isotropic) reaches the
    solve and is shown with the element formulation; the directional choice
    is flagged uncalibrated, and moot for spherical voids."""
    def captions(at):
        return " ".join(c.value for c in at.caption)

    at = _fresh_app()
    assert at.selectbox(key="porosity_degradation").value == "isotropic"
    assert "Directional degradation is uncalibrated" not in captions(at)
    at = _run_small_analysis(at)
    assert not at.exception
    assert at.session_state["result"]["fe_field"].porosity_degradation == "isotropic"
    assert "FE element formulation: hex8i; porosity degradation: isotropic." in captions(at)

    at.selectbox(key="porosity_degradation").set_value("directional").run()
    assert "Directional degradation is uncalibrated" in captions(at)
    assert "Spherical voids give the same results" in captions(at)
    _select(at, "Void shape", "penny")
    assert "Spherical voids give the same results" not in captions(at)
    _button(at, "Run analysis").click().run()
    assert not at.exception
    result = at.session_state["result"]
    assert result["config"]["porosity_degradation"] == "directional"
    assert result["fe_field"].porosity_degradation == "directional"
    assert ("FE element formulation: hex8i; porosity degradation: directional "
            "(uncalibrated).") in captions(at)


def test_uq_expander_reports_percentiles():
    """IMPROVEMENT_PLAN 3.4: the Results tab runs uncertainty propagation."""
    at = _run_small_analysis(_fresh_app())
    _button(at, "Run uncertainty analysis").click().run()
    assert not at.exception
    # The FE failure block's metrics come first on the Results tab.
    uq = [m for m in at.metric if m.label.startswith("Knockdown p")]
    assert [m.label for m in uq] == ["Knockdown p5", "Knockdown p50", "Knockdown p95"]
    p5, p50, p95 = (float(m.value) for m in uq)
    assert p5 < p50 < p95


def test_layup_default_comes_from_session_state_only():
    """The layup input takes its default from st.session_state alone, so
    Streamlit no longer warns that it was "created with a default value but
    also had its value set via the Session State API"."""
    messages = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    policy_logger = logging.getLogger("streamlit.elements.lib.policies")
    handler = _Collect()
    policy_logger.addHandler(handler)
    try:
        at = _fresh_app()
        assert at.text_input(key="layup_input").value == "[0/45/-45/90]_3s"
        at.text_input(key="layup_input").set_value("[0/90]_2s").run()
        at.run()                                    # the typed value stays
    finally:
        policy_logger.removeHandler(handler)
    assert not at.exception
    assert at.text_input(key="layup_input").value == "[0/90]_2s"
    assert not any("layup_input" in m for m in messages), messages


class TestStressComponentDefault:
    def test_opens_on_the_load_case_and_keeps_the_users_pick(self):
        """ILSS opens on the interlaminar shear magnitude (τ₁₃ in the 0°
        plies, τ₂₃ in the 90° ones); a component the user picks survives a new
        run with another default (following the default between runs is
        covered headlessly in test_app.py)."""
        at = _select(_fresh_app(), "Loading mode", "ilss")
        at = _run_small_analysis(at)
        assert not at.exception
        assert at.selectbox(key="stress_component").value == "Interlaminar shear √(τ₁₃² + τ₂₃²)"
        at.selectbox(key="stress_component").set_value("Von Mises").run()
        _select(at, "Loading mode", "shear")
        _button(at, "Run analysis").click().run()
        assert not at.exception
        assert at.session_state["result"]["fe_loading"] == "shear"
        assert at.selectbox(key="stress_component").value == "Von Mises"


def _small_flexure_run(at: AppTest, mode: str = "flexure_3pt") -> AppTest:
    """Flexure on 0.5 mm plies, so the 13 mm wide D7264 specimen mesh stays
    small (its nx and ny follow the span; nz comes from the slider)."""
    _select(at, "Loading mode", mode)
    next(n for n in at.number_input if n.label == "Ply thickness (mm)") \
        .set_value(0.5).run()
    return _run_small_analysis(at)


class TestFlexureFlow:
    def test_flexure_3pt_run_shows_fe_and_clt_moduli(self):
        at = _select(_fresh_app(), "Loading mode", "flexure_3pt")
        captions = " ".join(c.value for c in at.caption)
        assert "ASTM D7264 specimen" in captions and "FE only" in captions
        at = _small_flexure_run(at)
        assert not at.exception
        result = at.session_state["result"]
        assert result["fe_loading"] == "flexure_3pt"
        assert result["mesh"].flexure_geometry is not None
        assert result["mesh"].nz == 4
        fe, flex = result["fe_field"], result["fe_flexure"]
        assert fe is not None and fe.effective_modulus > 0
        assert any(h.value == "FE flexure (ASTM D7264 procedure A, three-point)"
                   for h in at.subheader)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["FE flexural modulus E_f"] == f"{fe.effective_modulus / 1000:.1f} GPa"
        assert metrics["CLT E_f (D matrix)"] == f"{flex['clt_E_f'] / 1000:.1f} GPa"
        assert metrics["FE knockdown"] == f"{fe.knockdown:.3f}"
        assert any("three-point flexure specimen" in s.value for s in at.success)
        assert at.selectbox(key="stress_component").value == "σ₁₁ (fiber)"
        # The convergence check refines the D7264 specimen, nx kept even.
        plan = at.dataframe[0].value
        assert list(plan["Mesh"]) == ["40 × 8 × 4", "60 × 12 × 4"]
        labels = [b.proto.label for b in at.get("download_button")]
        assert "Download FE results (VTU)" in labels


def test_convergence_expander_runs_the_study():
    """The Results tab plans the levels before running, then shows the
    level and verdict tables (tiny meshes keep it fast)."""
    at = _run_small_analysis(_fresh_app())
    assert not at.exception
    plan = at.dataframe[0].value
    assert list(plan["Mesh"]) == ["8 × 4 × 4", "12 × 6 × 4"]
    assert list(plan["Elements"]) == ["128", "288"]
    assert all(v.endswith(" GB") for v in plan["Est. peak memory"])
    at.radio(key="conv_levels").set_value("3 levels (×1, ×1.5, ×2)").run()
    meshes = ["8 × 4 × 4", "12 × 6 × 4", "16 × 8 × 4"]
    assert list(at.dataframe[0].value["Mesh"]) == meshes
    assert len(at.dataframe) == 1                  # nothing solved yet
    _button(at, "Check mesh convergence").click().run()
    assert not at.exception
    assert len(at.dataframe) == 3
    levels, verdicts = at.dataframe[1].value, at.dataframe[2].value
    assert list(levels["Mesh"]) == meshes
    assert list(verdicts["Quantity"]) == ["Knockdown", "Interior max FI",
                                          "Domain max FI"]
    assert set(verdicts["Verdict"]) <= {"converged", "not converged",
                                        "mesh-dependent", "n/a"}
    assert any("mesh-dependent" in c.value and "1 − KD" in c.value
               for c in at.caption)


# ----------------------------------------------------------------------
# Defects panel: delamination, toughness, onset, fracture specimens
# (delamination plan, phase 5.1)
# ----------------------------------------------------------------------

def _small_mesh(at: AppTest, layup: str = "0/90/90/0") -> AppTest:
    """Expert mode, a short layup and an 8 x 4 x 4 mesh, not yet run."""
    at.toggle[0].set_value(True).run()
    at.text_input(key="layup_input").set_value(layup).run()
    at.slider[0].set_value(8)
    at.slider[1].set_value(4)
    at.slider[2].set_value(4)
    return at.run()


def _enter_toughness(at: AppTest) -> AppTest:
    at.number_input(key="tough_G_Ic").set_value(0.212).run()
    at.number_input(key="tough_G_IIc").set_value(0.774).run()
    at.number_input(key="tough_bk_eta").set_value(2.1).run()
    return at


def _delamination_run(at: AppTest) -> AppTest:
    """A through-width delamination under the first ply of the small coupon
    in tension, with toughness and the interface onset."""
    at = _small_mesh(at)
    _select(at, "Loading mode", "tension")
    at.checkbox(key="delam_on").check().run()
    at.number_input(key="delam_interface").set_value(0).run()
    at.selectbox(key="delam_shape").set_value("strip").run()
    _enter_toughness(at)
    at.checkbox(key="onset_on").check().run()
    _button(at, "Run analysis").click().run()
    return at


class TestDefectsPanel:
    def test_default_config_has_no_defect(self):
        """Without the new inputs the run's cache key is the one before
        them, and the Results tab has no Defects block."""
        import app
        at = _run_small_analysis(_fresh_app())
        assert not at.exception
        # The cached run rebuilds its config from the key: no defect field
        # was in it.
        cfg = at.session_state["result"]["config"]
        assert not any(k in cfg for k in app._CFG_OPTIONAL_KEYS)
        assert tuple(k for k, _ in app._config_to_key(cfg)) == app._CFG_KEYS
        assert at.session_state["result"]["delamination"] is None
        assert not any(h.value.startswith("Defects") for h in at.subheader)
        assert not any(c.label == "Indication type" for c in at.selectbox)

    def test_delamination_flow(self):
        at = _delamination_run(_fresh_app())
        assert not at.exception
        result = at.session_state["result"]
        fe = result["fe_field"]
        assert result["delamination"].shape == "strip"
        assert fe.crack_front is not None and fe.fracture_load_factor is not None
        assert any(h.value == "Defects: delamination and interface onset"
                   for h in at.subheader)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Fracture load factor λ_c"] == f"{fe.fracture_load_factor:.3g}"
        assert metrics["KD total"] == f"{fe.knockdown:.4f}"
        assert "Max onset index" in metrics
        # The uncalibrated placeholder law is flagged, on the Overview too.
        assert sum("uncalibrated" in w.value for w in at.warning) >= 2
        labels = [b.proto.label for b in at.get("download_button")]
        assert labels.count("Download PNG") == 5          # + the front plot
        assert any("Delamination: through-width strip" in s.value for s in at.success)

    def test_preset_toughness_is_used_without_entries(self):
        """IM7_8552_epoxy carries toughness (#223): with the toughness
        fields left empty the run uses the preset's, so it reports the
        fracture load factor, and the sidebar says so."""
        at = _fresh_app()
        _select(at, "Material", "IM7_8552_epoxy")
        at = _small_mesh(at)
        _select(at, "Loading mode", "tension")
        at.checkbox(key="delam_on").check().run()
        at.number_input(key="delam_interface").set_value(0).run()
        at.selectbox(key="delam_shape").set_value("strip").run()
        assert any("This preset carries interlaminar toughness: G_Ic 0.21" in c.value
                   for c in at.caption)
        assert at.number_input(key="tough_G_Ic").value is None
        _button(at, "Run analysis").click().run()
        assert not at.exception
        fe = at.session_state["result"]["fe_field"]
        assert fe.crack_front is not None and fe.fracture_load_factor is not None
        cfg = at.session_state["result"]["config"]
        assert all(cfg.get(k) is None for k in ("G_Ic", "G_IIc", "bk_eta"))

    def test_ncr_assesses_the_delamination_without_a_path(self):
        at = _delamination_run(_fresh_app())
        assert at.selectbox(key="ncr_indication").value == "combined"
        _button(at, "Generate summary").click().run()
        assert any("required load factor" in e.value for e in at.error)
        at.selectbox(key="ncr_indication").set_value("delamination")
        at.number_input(key="ncr_required_load_factor").set_value(0.5)
        _button(at, "Generate summary").click().run()
        assert not at.exception
        assert not any("Recommended disposition path" in w.value for w in at.warning)
        assert any("no recommended disposition path" in w.value
                   and "fracture margin of safety" in w.value for w in at.warning)
        assert any("does not recommend a disposition" in i.value for i in at.info)
        assert len(at.get("download_button")) == 14       # 2 + 5 PNG + 4 + 3
        # Combined: the porosity's path, labelled as covering the porosity only.
        at.selectbox(key="ncr_indication").set_value("combined")
        _button(at, "Generate summary").click().run()
        assert any("Recommended disposition path (porosity only)" in w.value
                   for w in at.warning)
        assert any("no recommended disposition path" in w.value for w in at.warning)

    def test_fracture_uncertainty_band(self):
        """The opt-in λ_c band: offered once the toughness is in, off by
        default (the key of before), nothing scatters until a CoV is
        entered; the Defects block shows p5 / p50 / p95, the histogram and
        the share of draws below an entered factor."""
        _BAND = "λ_c uncertainty band (toughness scatter)"
        at = _small_mesh(_fresh_app())
        _select(at, "Loading mode", "tension")
        at.checkbox(key="delam_on").check().run()
        at.number_input(key="delam_interface").set_value(0).run()
        at.selectbox(key="delam_shape").set_value("strip").run()
        assert not any(c.label == _BAND for c in at.checkbox)
        _enter_toughness(at)
        assert at.checkbox(key="frac_uq_on").value is False
        at.checkbox(key="frac_uq_on").check().run()
        assert any("nothing scatters until" in i.value for i in at.info)
        at.number_input(key="frac_uq_G_Ic").set_value(0.05).run()
        at.number_input(key="frac_uq_G_IIc").set_value(0.07).run()
        _button(at, "Run analysis").click().run()
        assert not at.exception
        result = at.session_state["result"]
        band = result["fe_fracture_uq"]
        assert result["config"]["fracture_uq"] == [0.05, 0.07, 0.0, 0.0, 0.0]
        assert band.nominal == result["fe_field"].fracture_load_factor
        metrics = {m.label: m.value for m in at.metric}
        for p in ("p5", "p50", "p95"):
            assert metrics[f"λ_c {p}"] == f"{band.percentiles[p]:.3g}"
        assert any("not a disposition" in c.value for c in at.caption)
        labels = [b.proto.label for b in at.get("download_button")]
        assert labels.count("Download PNG") == 6          # + the front, the band
        compare = round(band.percentiles["p50"], 3)
        at.number_input(key="frac_uq_compare").set_value(compare).run()
        metrics = {m.label: m.value for m in at.metric}
        assert metrics[f"Draws below {compare:g}"] == \
            f"{100.0 * band.probability_below(compare):.1f} %"
        assert not at.exception

    def test_inputs_that_block_the_run(self):
        at = _small_mesh(_fresh_app())
        at.checkbox(key="delam_on").check().run()
        at.number_input(key="delam_interface").set_value(5).run()
        assert any("does not exist" in e.value for e in at.error)
        assert _button(at, "Run analysis").disabled is True
        at.number_input(key="delam_interface").set_value(1).run()
        assert _button(at, "Run analysis").disabled is False
        at.selectbox(key="tough_law").set_value("user").run()
        assert _button(at, "Run analysis").disabled is True
        at.number_input(key="tough_b_I").set_value(5.0).run()
        at.number_input(key="tough_b_II").set_value(12.0).run()
        assert _button(at, "Run analysis").disabled is False
        # An unresolved layup is explained before the run, with the fix.
        at.slider[2].set_value(6).run()
        assert any("Use nz = 8" in w.value for w in at.warning)
        _button(at, "Use nz = 8").click().run()
        assert at.slider[2].value == 8
        assert not at.exception

    def test_dcb_specimen_for_test_planning(self):
        at = _select(_fresh_app(), "Loading mode", "dcb")
        captions = " ".join(c.value for c in at.caption)
        assert "for test planning" in captions and "ASTM D5528" in captions
        assert not any(c.label == "Add a delamination (NDI indication)"
                       for c in at.checkbox)
        # 2 mm thick (the widget's largest ply), so the graded mesh is small.
        next(n for n in at.number_input if n.label == "Ply thickness (mm)") \
            .set_value(0.5).run()
        at = _small_mesh(at, layup="0/0/0/0")
        _enter_toughness(at)
        _button(at, "Run analysis").click().run()
        assert not at.exception
        result = at.session_state["result"]
        assert result["fe_loading"] == "dcb"
        assert result["config"]["crack_length"] == 50.0
        assert result["fe_specimen"]["thickness_mm"] == pytest.approx(2.0)
        metrics = {m.label: m.value for m in at.metric}
        assert float(metrics["P_c at mid-width (N)"]) > 0.0
        assert float(metrics["Compliance C (mm/N)"]) > 0.0
        assert any(h.value == "Defects: delamination and interface onset"
                   for h in at.subheader)
        # The NCR does not assess a test specimen.
        assert not any(c.label == "Indication type" for c in at.selectbox)

    def test_mmb_lever_caption(self):
        at = _select(_fresh_app(), "Loading mode", "mmb")
        at.number_input(key="frac_mmb_ratio").set_value(0.8).run()
        assert any(c.value.startswith("Lever length c =") for c in at.caption)
        assert not at.exception


class TestSidebarReviewFixes:
    """Regression tests for the app review of November 2026."""

    def test_one_ply_layup_in_expert_mode(self):
        """nz = 1 (one ply) is the default; the slider started at 2 and
        raised, hiding the rest of the sidebar and the Run button."""
        at = _fresh_app()
        at.text_input(key="layup_input").set_value("0").run()
        at.toggle[0].set_value(True).run()
        assert not at.exception
        assert at.slider(key="mesh_nz").value == 1
        assert any(b.label == "Run analysis" for b in at.button)

    def test_fracture_specimen_mesh_is_described_and_guarded(self):
        """The DCB / ENF / MMB solve meshes the specimen, not the coupon:
        the sidebar now gives its size and warns before a long run."""
        at = _select(_fresh_app(), "Loading mode", "enf")
        at.text_input(key="layup_input").set_value("0/0/0/0").run()
        captions = [c.value for c in at.caption]
        assert any("meshed 151 × 36 × 4 = 21,744 elements" in c for c in captions)
        assert not any(c.startswith("Default mesh:") for c in captions)
        assert any("can take several minutes" in w.value for w in at.warning)

    def test_thermal_only_drops_the_specimen_inputs(self):
        """Thermal only solves the free-standing coupon: the crack length
        and the knockdown split no longer appear."""
        at = _fresh_app()
        _select(at, "Material", "AS4_3501_6_epoxy")
        _select(at, "Loading mode", "dcb")
        assert any(n.key == "frac_a0_dcb" for n in at.number_input)
        at.radio(key="thermal_choice").set_value(
            "Thermal only (residual stress, no mechanical load)").run()
        assert not at.exception
        assert not any(n.key == "frac_a0_dcb" for n in at.number_input)
        assert any("the specimen's inputs do not apply" in c.value for c in at.caption)


def _material_file(**changes) -> bytes:
    import dataclasses

    from porosity_fe import MATERIALS
    from porosity_fe.io import _dumps_json, _material_document
    material = dataclasses.replace(MATERIALS["T800_epoxy"], **changes)
    return _dumps_json(_material_document(material, "My T800")).encode()


def _success(at, start: str) -> list[str]:
    return [s.value for s in at.success if s.value.startswith(start)]


class TestCustomMaterialAndScenario:
    def test_a_material_file_becomes_the_selected_material(self):
        at = _fresh_app()
        at.file_uploader(key="material_upload").upload(
            "my.json", _material_file(E22=9000.0), "application/json").run()
        assert not at.exception
        assert _success(at, "Loaded My T800: E11 = 161000, E22 = 9000")
        material = next(sb for sb in at.selectbox if sb.label == "Material")
        assert material.options[-1] == "Custom: My T800"
        assert material.value == "Custom: My T800"
        # Its own CTEs decide the thermal panel: T800 has none.
        assert any("Unavailable for Custom: My T800: the material has no lamina"
                   in c.value for c in at.caption)
        at = _run_small_analysis(at)
        assert not at.exception
        result = at.session_state["result"]
        assert result["material"].E22 == 9000.0
        assert result["config"]["material"]["E22"] == 9000.0
        assert any(s.value.startswith("Last run: Custom: My T800") for s in at.success)

    def test_a_bad_material_file_changes_nothing(self):
        at = _fresh_app()
        at.file_uploader(key="material_upload").upload(
            "bad.json", b'{"E11": 1}', "application/json").run()
        assert not at.exception
        assert any(e.value.startswith("Not loaded: bad.json: The material is missing E22")
                   for e in at.error)
        material = next(sb for sb in at.selectbox if sb.label == "Material")
        assert "Custom: My T800" not in material.options
        assert material.value == "T800_epoxy"

    def test_remove_the_custom_material(self):
        at = _fresh_app()
        at.file_uploader(key="material_upload").upload(
            "my.json", _material_file(), "application/json").run()
        _button(at, "Remove the custom material").click().run()
        assert not at.exception
        material = next(sb for sb in at.selectbox if sb.label == "Material")
        assert material.options == material.options[:9] and material.value == "T800_epoxy"
        assert not any(b.label == "Remove the custom material" for b in at.button)

    def test_the_selected_material_downloads_as_json(self):
        """The template download is a JSON file (AppTest exposes its media
        URL, not its bytes; the document itself is tested in
        test_material_io.py)."""
        at = _fresh_app()
        button = next(b for b in at.get("download_button")
                      if b.proto.label == "Download the selected material (JSON)")
        assert button.proto.url.endswith(".json")
        at.file_uploader(key="material_upload").upload(
            "my.json", _material_file(), "application/json").run()
        assert any(b.proto.label == "Download the selected material (JSON)"
                   for b in at.get("download_button"))

    def test_plate_size_reaches_the_mesh(self):
        at = _fresh_app()
        at.toggle[0].set_value(True).run()
        at.text_input(key="layup_input").set_value("0/90/90/0").run()
        at.slider(key="mesh_nx").set_value(8)
        at.slider(key="mesh_ny").set_value(4)
        at.slider(key="mesh_nz").set_value(4)
        at.number_input(key="plate_length").set_value(80.0)
        at.number_input(key="plate_width").set_value(30.0)
        at.run()
        assert any(c.value.startswith("Plate 80 × 30 mm: in-plane elements of "
                                      "10.00 × 7.50 mm") for c in at.caption)
        _button(at, "Run analysis").click().run()
        assert not at.exception
        mesh = at.session_state["result"]["mesh"]
        assert (mesh.L_x, mesh.L_y) == (80.0, 30.0)
        assert any("mesh 8×4×4 on a 80 × 30 mm plate" in s.value for s in at.success)

    def test_the_specimen_modes_have_no_plate_size(self):
        at = _fresh_app()
        next(sb for sb in at.selectbox if sb.label == "Loading mode").set_value("ilss").run()
        assert not any(n.label == "Plate length (mm)" for n in at.number_input)
        assert any(c.value.startswith("The plate size does not apply") for c in at.caption)

    def test_save_and_restore_a_scenario(self):
        at = _run_small_analysis(_fresh_app())
        at.number_input(key="plate_length").set_value(60.0).run()
        _button(at, "Run analysis").click().run()
        at.text_input(key="scenario_name").set_value("baseline").run()
        from porosity_app.scenario import scenario_document, serialise_scenario
        text = serialise_scenario(scenario_document(
            at.session_state, "baseline", at.session_state["result"]))
        assert '"summary": {' in text

        bt = _fresh_app()
        bt.file_uploader(key="scenario_upload").upload(
            "baseline.json", text.encode(), "application/json").run()
        assert not bt.exception
        assert any(s.value.startswith("Restored ") and "from baseline.json" in s.value
                   for s in bt.success)
        assert bt.session_state["layup_input"] == "0/90/90/0"
        assert bt.session_state["expert_mode"] is True
        assert (bt.session_state["mesh_nx"], bt.session_state["mesh_nz"]) == (8, 4)
        assert bt.session_state["plate_length"] == 60.0
        assert bt.session_state["scenario_name"] == "baseline"
        bt.number_input(key="vp_input").set_value(5.0).run()
        _button(bt, "Run analysis").click().run()
        assert not bt.exception
        assert bt.session_state["result"]["mesh"].L_x == 60.0
        assert any(h.value == "Compared with the loaded scenario: baseline"
                   for h in bt.subheader)
        table = next(df.value for df in bt.dataframe if "Saved run" in df.value.columns)
        vp = table[table["Quantity"] == "Void content Vp (%)"].iloc[0]
        assert (vp["Saved run"], vp["This run"], vp["Change (%)"]) == ("3", "5", "+66.67")
        _button(bt, "Stop comparing").click().run()
        assert not any(h.value.startswith("Compared with") for h in bt.subheader)

    def test_a_scenario_with_a_custom_material(self):
        at = _fresh_app()
        at.file_uploader(key="material_upload").upload(
            "my.json", _material_file(G12=4000.0), "application/json").run()
        from porosity_app.scenario import scenario_document, serialise_scenario
        text = serialise_scenario(scenario_document(at.session_state))
        bt = _fresh_app()
        bt.file_uploader(key="scenario_upload").upload(
            "s.json", text.encode(), "application/json").run()
        assert not bt.exception
        assert next(sb for sb in bt.selectbox if sb.label == "Material").value == \
            "Custom: My T800"

    def test_a_partly_restored_scenario_says_what_was_skipped(self):
        import json
        doc = {"format": "porosity-fe.app-scenario", "name": None, "material": None,
               "summary": None, "inputs": {"vp_input": 2.0, "mesh_nx": 999,
                                           "removed_input": 1}}
        at = _fresh_app()
        at.file_uploader(key="scenario_upload").upload(
            "old.json", json.dumps(doc).encode(), "application/json").run()
        assert not at.exception
        assert any(w.value.startswith("Restored 1 inputs from old.json") for w in at.warning)
        notes = [c.value for c in at.caption if c.value.startswith("Not restored: ")]
        assert "Not restored: mesh_nx: 999 is outside 2 to 200; not restored." in notes
        assert "Not restored: removed_input: not an input of this version; skipped." in notes
        assert at.session_state["vp_input"] == 2.0

    def test_not_a_scenario_file(self):
        at = _fresh_app()
        at.file_uploader(key="scenario_upload").upload(
            "mat.json", _material_file(), "application/json").run()
        assert not at.exception
        assert any(e.value.startswith("Not loaded: The file is not a saved scenario")
                   for e in at.error)

    def test_a_scenario_restores_a_broken_layup(self):
        """The scenario panel stays with an invalid layup, so a scenario
        can replace it."""
        import json
        at = _fresh_app()
        at.text_input(key="layup_input").set_value("garbage ###").run()
        assert any(e.value.startswith("Invalid layup") for e in at.error)
        doc = {"format": "porosity-fe.app-scenario", "name": None, "material": None,
               "summary": None, "inputs": {"layup_input": "[0/90]s"}}
        at.file_uploader(key="scenario_upload").upload(
            "fix.json", json.dumps(doc).encode(), "application/json").run()
        assert not at.exception
        assert at.session_state["layup_input"] == "[0/90]s"
        assert at.session_state["_layup_status"][0] == "ok"


class TestFatigueAndEnvironment:
    def _table(self, at):
        return next(df.value for df in at.dataframe if "Fatigue" in df.value.columns)

    def test_the_panel(self):
        at = _run_small_analysis(_fresh_app())
        assert any(i.value.startswith("Enter the load cycles") for i in at.info)
        at.number_input(key="dur_cycles").set_value(1e6).run()
        assert not at.exception
        table = self._table(at)
        assert len(table) == 15
        row = table[(table.Mode == "tension") & (table.Law == "judd_wright")].iloc[0]
        assert row["Fatigue"] == "0.4000" and row["Environment"] == "1.0000"
        assert any(m.label == "Governing knockdown" for m in at.metric)

        at.checkbox(key="dur_goodman").check().run()
        at.number_input(key="dur_R").set_value(0.5).run()
        row = self._table(at)
        row = row[(row.Mode == "tension") & (row.Law == "judd_wright")].iloc[0]
        assert row["Fatigue"] == "0.5455"

        at.number_input(key="dur_T").set_value(80.0).run()
        assert any("needs the dry glass-transition temperature" in w.value for w in at.warning)
        at.number_input(key="dur_Tg").set_value(180.0).run()
        table = self._table(at)
        shear = table[(table.Mode == "shear") & (table.Law == "linear")].iloc[0]
        assert float(shear["Environment"]) < 1.0

        at.number_input(key="dur_cycles").set_value(1e12).run()
        assert any(c.value.startswith("⚠ Fatigue knockdown for mode=") for c in at.caption)

    def test_a_scenario_restores_the_panel(self):
        import json
        doc = {"format": "porosity-fe.app-scenario", "name": None, "material": None,
               "summary": None,
               "inputs": {"dur_cycles": 1e5, "dur_goodman": True, "dur_R": 0.3,
                          "layup_input": "0/90/90/0", "expert_mode": True,
                          "mesh_nx": 8, "mesh_ny": 4, "mesh_nz": 4}}
        at = _fresh_app()
        at.file_uploader(key="scenario_upload").upload(
            "s.json", json.dumps(doc).encode(), "application/json").run()
        _button(at, "Run analysis").click().run()
        assert not at.exception
        assert at.number_input(key="dur_cycles").value == 1e5
        assert at.number_input(key="dur_R").value == 0.3
        assert len(self._table(at)) == 15


@pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")
class TestBucklingPanel:
    def test_run_on_demand(self):
        at = _run_small_analysis(_fresh_app())
        assert not any(m.label == "Buckling load factor" for m in at.metric)
        _button(at, "Run buckling analysis").click().run()
        assert not at.exception
        labels = {m.label: m.value for m in at.metric}
        assert float(labels["Buckling load factor"]) > 0.0
        assert 0.9 < float(labels["Of the void-free laminate"]) < 1.0
        table = next(d.value for d in at.dataframe if "Load factor" in d.value.columns)
        assert list(table["Mode"]) == [1, 2, 3]
        assert "Delamination faces close by" not in table.columns
        assert any(d.proto.label == "Download PNG" for d in at.get("download_button"))
        at.run()
        assert any(m.label == "Buckling load factor" for m in at.metric)

    def test_other_loadings_say_why(self):
        at = _fresh_app()
        next(sb for sb in at.selectbox if sb.label == "Loading mode").set_value("tension").run()
        at = _run_small_analysis(at)
        assert any(i.value.startswith("Linear buckling analyses the coupon")
                   for i in at.info)
        assert not any(b.label == "Run buckling analysis" for b in at.button)
