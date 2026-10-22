#!/usr/bin/env python3
"""Tests for the testable (non-Streamlit-runtime) helpers in app.py.

The genuinely pure reporting helpers were extracted to
``porosity_fe.reporting`` (see #155) and are covered elsewhere. This file
exercises what still lives in ``app.py`` but does *not* require a live
Streamlit script-run context:

* ``run_analysis`` — the analysis runner the cached UI entry point wraps;
* the ``plot_*`` figure builders the tabs hand to ``st.pyplot``;
* ``_config_to_key`` — the cache-key encoder;
* ``_validate_layup_inline`` — the layup on-change callback (its only
  Streamlit dependency is ``st.session_state``, which behaves like a dict
  and can be swapped for one).
"""

import dataclasses
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

# app.py imports streamlit at module scope; the CI `test` matrix installs
# only the core deps (streamlit lives in the `web` extra), so skip the whole
# module when it is unavailable rather than erroring at collection.
pytest.importorskip("streamlit")

import app


def _base_cfg(**overrides) -> dict:
    """A valid analysis config mirroring what ``_build_sidebar_inputs``
    produces, with a small mesh so the FE solve stays fast."""
    cfg = {
        "material_name": "T800_epoxy",
        "angles": [0.0, 90.0, 90.0, 0.0],
        "n_plies": 4,
        "t_ply": 0.183,
        "Vp": 3.0,
        "distribution": "uniform",
        "cluster_location": "midplane",
        "void_shape": "spherical",
        "loading_mode": "compression",
        "nx": 10, "ny": 4, "nz": 6,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def ilss_result():
    """One ILSS analysis result (the D2344 specimen FE solve)."""
    return app.run_analysis(_base_cfg(loading_mode="ilss"))


@pytest.fixture(scope="module")
def comp_result():
    """One compression analysis result, reused across the plot tests so the
    FE solve only runs once."""
    return app.run_analysis(_base_cfg())


def _thermal_cfg(**overrides) -> dict:
    """``_base_cfg`` on AS4/3501-6 (the preset with CTEs), one element layer
    per ply, with the FE thermal load at ΔT = -150 K."""
    cfg = _base_cfg(material_name="AS4_3501_6_epoxy", t_ply=0.125,
                    nx=10, ny=6, nz=4, thermal_mode="thermal_only",
                    delta_T=-150.0)
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def thermal_result():
    return app.run_analysis(_thermal_cfg())


@pytest.fixture(scope="module")
def combined_results():
    """Tension with the residual stress, and the same tension without it."""
    return (app.run_analysis(_thermal_cfg(thermal_mode="combined",
                                          loading_mode="tension")),
            app.run_analysis(_thermal_cfg(thermal_mode="off", delta_T=None,
                                          loading_mode="tension")))


class TestConfigToKey:
    def test_lists_become_tuples_and_key_is_hashable(self):
        cfg = _base_cfg()
        key = app._config_to_key(cfg)
        # The ``angles`` list must be encoded as a tuple so the key hashes.
        angles_entry = dict(key)["angles"]
        assert angles_entry == tuple(cfg["angles"])
        assert isinstance(angles_entry, tuple)
        assert hash(key)  # must not raise

    def test_key_covers_every_cfg_field_in_order(self):
        key = app._config_to_key(_base_cfg())
        assert tuple(k for k, _ in key) == app._CFG_KEYS

    def test_scalar_fields_passed_through(self):
        key = dict(app._config_to_key(_base_cfg()))
        assert key["material_name"] == "T800_epoxy"
        assert key["Vp"] == 3.0
        assert key["nx"] == 10

    def test_thermal_fields_default_to_off_and_enter_the_key(self):
        """A config without the thermal fields means no thermal load, and
        each thermal setting gets its own cache entry."""
        key = dict(app._config_to_key(_base_cfg()))
        assert key["thermal_mode"] == "off" and key["delta_T"] is None
        k1 = app._config_to_key(_thermal_cfg())
        k2 = app._config_to_key(_thermal_cfg(delta_T=-100.0))
        k3 = app._config_to_key(_thermal_cfg(thermal_mode="combined"))
        assert len({k1, k2, k3, app._config_to_key(_base_cfg())}) == 4


class TestRunAnalysis:
    def test_compression_returns_full_result(self, comp_result):
        for k in ("config", "material", "porosity_field", "mesh",
                  "empirical", "fe_field", "fe_loading",
                  "fe_skipped_reason", "layup_scale"):
            assert k in comp_result
        assert comp_result["fe_field"] is not None
        assert comp_result["fe_loading"] == "compression"
        assert set(comp_result["layup_scale"]) == {
            "compression", "tension", "shear", "ilss", "transverse_tension"}
        # Empirical table carries the four loading modes.
        for mode in ("compression", "tension", "shear", "ilss"):
            assert mode in comp_result["empirical"]

    def test_ilss_takes_force_controlled_branch(self, ilss_result):
        """``loading_mode='ilss'`` routes through the short-beam-shear
        (force-controlled) solve rather than the applied-strain branch."""
        r = ilss_result
        assert r["fe_loading"] == "ilss"
        assert r["fe_field"] is not None

    def test_ilss_solves_the_d2344_specimen(self, ilss_result):
        """The ILSS FE solve runs on the ASTM D2344 specimen (span 4h,
        6h x 2h) with the configured nz, not on the nx x ny x nz coupon, and
        the result's mesh is the one the field belongs to."""
        from porosity_fe import ShortBeamShear
        r = ilss_result
        mesh = r["mesh"]
        h = 4 * 0.183
        assert mesh.ilss_geometry is not None
        assert (mesh.nx, mesh.ny, mesh.nz) == (24, 8, 6)
        assert r["fe_field"].displacement.shape == (mesh.n_nodes, 3)
        spec = r["fe_specimen"]
        assert spec["standard"] == "ASTM D2344"
        assert spec["length_mm"] == pytest.approx(6 * h, rel=1e-12)
        assert spec["width_mm"] == pytest.approx(2 * h, rel=1e-12)
        assert spec["thickness_mm"] == pytest.approx(h, rel=1e-12)
        assert spec["span_mm"] == pytest.approx(4 * h, rel=1e-12)
        assert spec["load_width_mm"] == mesh.ilss_geometry.load_width
        assert spec["mesh"] == "24x8x6"
        assert isinstance(r["fe_short_beam"], ShortBeamShear)
        # The empirical results do not depend on the mesh.
        coupon = app.run_analysis(_base_cfg())
        assert r["empirical"]["ilss"]["judd_wright"]["knockdown"] == \
            coupon["empirical"]["ilss"]["judd_wright"]["knockdown"]

    def test_short_beam_summary(self, ilss_result, comp_result):
        rows, caption = app.fe_short_beam_summary(ilss_result)
        labels = [label for label, _, _ in rows]
        assert labels == ["FE knockdown", "Mid-plane shear / 0.75P/(bh)",
                          "Interior FPF load", "Apparent ILSS at FPF"]
        values = dict((label, value) for label, value, _ in rows)
        sb = ilss_result["fe_short_beam"]
        assert values["FE knockdown"] == f"{ilss_result['fe_field'].knockdown:.3f}"
        assert values["Mid-plane shear / 0.75P/(bh)"] == \
            f"{sb.midplane_shear / sb.apparent_shear:.3f}"
        assert values["Apparent ILSS at FPF"] == \
            f"{sb.interior_apparent_strength:.1f} MPa"
        assert "ASTM D2344" in caption and "24 × 8 × 6" in caption
        assert app.fe_short_beam_summary(comp_result) is None
        assert comp_result["fe_specimen"] is None
        assert comp_result["fe_short_beam"] is None

    def test_thermal_only_with_ilss_selected_stays_on_the_coupon(self):
        r = app.run_analysis(_thermal_cfg(loading_mode="ilss"))
        assert r["fe_loading"] == "thermal"
        assert r["fe_specimen"] is None and r["fe_short_beam"] is None
        assert (r["mesh"].nx, r["mesh"].ny) == (10, 6)

    def test_combined_thermal_ilss_uses_the_specimen(self):
        r = app.run_analysis(_thermal_cfg(thermal_mode="combined",
                                          loading_mode="ilss"))
        assert r["fe_field"] is not None
        assert r["mesh"].ilss_geometry is not None
        assert r["fe_field"].load_factor_basis == "mechanical_with_residual"
        assert r["fe_mechanical_fpf"] is not None
        assert r["fe_short_beam"] is not None

    def test_ilss_specimen_caption_matches_the_specimen_mesh(self):
        from porosity_fe import MATERIALS, CompositeMesh, PorosityField
        mat = MATERIALS["T800_epoxy"]
        mesh = CompositeMesh.for_ilss(PorosityField(mat, 0.0), mat)
        caption = app._ilss_specimen_caption(mat.n_plies, mat.t_ply, mesh.nz)
        assert f"meshed {mesh.nx} × {mesh.ny} × {mesh.nz}" in caption
        h = mat.total_thickness
        assert f"{6 * h:.1f} × {2 * h:.1f} × {h:.2f} mm" in caption
        assert f"{4 * h:.1f} mm span" in caption
        assert "D2344" in app._ilss_specimen_caption(0, 0.183, 24)

    def test_clustered_distribution_branch(self):
        """A clustered distribution forwards ``cluster_location`` into the
        PorosityField (the conditional pf_kwargs branch)."""
        r = app.run_analysis(_base_cfg(distribution="clustered"))
        assert r["porosity_field"].distribution == "clustered"
        assert r["fe_field"] is not None

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown material"):
            app.run_analysis(_base_cfg(material_name="unobtainium"))

    @pytest.mark.parametrize("cfg, n_lu", [
        (_base_cfg(loading_mode="ilss"), 2),
        (_thermal_cfg(loading_mode="ilss"), 1),
    ], ids=["ilss", "thermal_only"])
    def test_fe_solve_holds_one_factorization_at_a_time(
            self, monkeypatch, lu_alive_at_build, cfg, n_lu):
        """The solver frees the knockdown's pristine reference before it
        factorizes the porous model, so the app holds one factorization at
        a time without solving the reference itself first; a thermal-only
        solve needs no reference."""
        from collections import OrderedDict

        from porosity_fe.fe import _knockdown
        monkeypatch.setattr(_knockdown, "_PRISTINE_MEASURE_CACHE", OrderedDict())
        r = app.run_analysis(cfg)
        assert r["fe_field"] is not None
        assert lu_alive_at_build == [1] * n_lu

    def test_fe_failure_keeps_empirical_results(self, monkeypatch):
        """An exception from the FE path must not discard the empirical
        results; it is reported through ``fe_skipped_reason`` instead."""
        class _ExplodingFESolver:
            def __init__(self, *args, **kwargs):
                pass

            def solve(self, **kwargs):
                raise RuntimeError("singular stiffness matrix")

        monkeypatch.setattr(app, "FESolver", _ExplodingFESolver)
        r = app.run_analysis(_base_cfg(loading_mode="tension"))
        assert r["fe_field"] is None
        assert "RuntimeError" in r["fe_skipped_reason"]
        assert "singular stiffness matrix" in r["fe_skipped_reason"]
        for mode in ("compression", "tension", "shear", "ilss"):
            assert mode in r["empirical"]
        fig = app.plot_results(r, "0/90/90/0")
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert not any(lbl.startswith("FE") for lbl in labels)
        plt.close(fig)

    def test_fe_constructor_failure_is_also_caught(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise ValueError("bad ply angles")

        monkeypatch.setattr(app, "FESolver", _raise)
        r = app.run_analysis(_base_cfg())
        assert r["fe_field"] is None
        assert "ValueError: bad ply angles" in r["fe_skipped_reason"]

    def test_routes_through_build_empirical_pipeline(self, monkeypatch):
        """The app must build field/mesh/solver via the canonical factory so
        mesh-default and ply-angle changes reach the GUI."""
        calls = []
        real_factory = app.build_empirical_pipeline

        def _spy(material, vp, **kwargs):
            calls.append((vp, kwargs))
            return real_factory(material, vp, **kwargs)

        def _no_fe(*args, **kwargs):
            raise RuntimeError("FE not needed for this test")

        monkeypatch.setattr(app, "build_empirical_pipeline", _spy)
        monkeypatch.setattr(app, "FESolver", _no_fe)
        cfg = _base_cfg(distribution="clustered", cluster_location="surface")
        r = app.run_analysis(cfg)

        assert len(calls) == 1
        vp, kwargs = calls[0]
        assert vp == pytest.approx(0.03)
        assert kwargs["mesh_res"] == (10, 4, 6)
        assert kwargs["ply_angles"] == cfg["angles"]
        assert kwargs["porosity_config"]["distribution"] == "clustered"
        assert kwargs["porosity_config"]["cluster_location"] == "surface"
        assert (r["mesh"].nx, r["mesh"].ny, r["mesh"].nz) == (10, 4, 6)


class TestPlots:
    def test_plot_profile(self, comp_result):
        fig = app.plot_profile(comp_result)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh(self, comp_result):
        fig = app.plot_mesh(comp_result)
        assert fig is not None
        plt.close(fig)

    def test_plot_results_with_fe(self, comp_result):
        fig = app.plot_results(comp_result, "0/90/90/0")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize(
        "fe_loading", ["compression", "tension", "shear", "ilss"])
    def test_plot_results_legend_has_every_series(self, comp_result, fe_loading):
        """The FE series is drawn only at its own loading mode; its legend
        entry must survive when that mode is not the first bar group."""
        r = dict(comp_result)
        r["fe_loading"] = fe_loading
        fig = app.plot_results(r, "0/90/90/0")
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert labels == ["Judd-Wright", "Power Law", "Linear",
                          f"FE Stiffness ({fe_loading})"]
        plt.close(fig)

    def test_plot_results_layup_amplification_footnote(self, comp_result):
        """A layup scale above 1 (unvalidated amplification) adds a footnote
        naming the amplified modes; scale 1 adds none."""
        def footnote(result):
            fig = app.plot_results(result, "45/-45/-45/45")
            texts = [t.get_text() for t in fig.axes[0].texts]
            plt.close(fig)
            return "\n".join(texts)

        skewed = dict(comp_result)
        skewed["layup_scale"] = {**comp_result["layup_scale"],
                                 "tension": 1.94, "compression": 1.14}
        note = footnote(skewed)
        assert "Layup amplification (compression x1.14, tension x1.94)" in note
        assert "not validated" in note
        assert "Layup amplification" not in footnote(comp_result)

    def test_plot_stress_component(self, comp_result):
        fig = app.plot_stress(comp_result, "σ₁₁ (fiber)")
        assert fig is not None
        plt.close(fig)

    def test_plot_stress_von_mises(self, comp_result):
        fig = app.plot_stress(comp_result, "Von Mises")
        assert fig is not None
        plt.close(fig)

    def test_ilss_plots_and_vtu_use_the_specimen_mesh(self, ilss_result):
        for fig in (app.plot_mesh(ilss_result),
                    app.plot_stress(ilss_result, "τ₁₃ (interlaminar)"),
                    app.plot_results(ilss_result, "0/90/90/0")):
            assert fig is not None
            plt.close(fig)
        data = app.fe_vtu_bytes(ilss_result)
        n = ilss_result["mesh"].n_nodes
        assert f'NumberOfPoints="{n}"'.encode() in data

    def test_plot_stress_without_fe_field(self):
        """When no FE field is present the stress plot must short-circuit to
        an explanatory placeholder rather than indexing a missing field."""
        fig = app.plot_stress({"fe_field": None}, "Von Mises")
        assert fig is not None
        plt.close(fig)


class TestValidateLayupInline:
    """``_validate_layup_inline`` reads/writes ``st.session_state`` only, so a
    plain dict stands in for the Streamlit session."""

    def test_valid_layup_marks_ok(self, monkeypatch):
        monkeypatch.setattr(app.st, "session_state",
                            {"layup_input": "0/90/90/0"})
        app._validate_layup_inline()
        level, _ = app.st.session_state["_layup_status"]
        assert level == "ok"

    def test_empty_layup_marks_error(self, monkeypatch):
        monkeypatch.setattr(app.st, "session_state",
                            {"layup_input": "   "})
        app._validate_layup_inline()
        level, msg = app.st.session_state["_layup_status"]
        assert level == "err"
        assert "empty" in msg.lower()

    def test_invalid_layup_marks_error(self, monkeypatch):
        monkeypatch.setattr(app.st, "session_state",
                            {"layup_input": "garbage ###"})
        app._validate_layup_inline()
        assert app.st.session_state["_layup_status"][0] == "err"


class TestThermalAnalysis:
    """IMPROVEMENT_PLAN 3.5 PR 4: the FE thermal load case in the app."""

    def test_thermal_only_result(self, thermal_result):
        r = thermal_result
        fe = r["fe_field"]
        assert fe is not None, r["fe_skipped_reason"]
        assert r["fe_loading"] == "thermal"
        assert r["fe_thermal"] == {"mode": "thermal_only", "delta_T": -150.0}
        assert fe.delta_T == -150.0
        assert fe.load_factor_basis == "delta_T"
        assert math.isnan(fe.knockdown)
        assert r["fe_mechanical_fpf"] is None and r["fe_nz_suggestion"] is None
        # The empirical side is unchanged by the thermal option.
        plain = app.run_analysis(_thermal_cfg(thermal_mode="off", delta_T=None))
        assert r["empirical"] == plain["empirical"]

    def test_thermal_only_summary_reads_the_basis(self, thermal_result):
        fe = thermal_result["fe_field"]
        rows, caption = app.fe_thermal_summary(thermal_result)
        values = {label: value for label, value, _ in rows}
        assert list(values) == ["FE knockdown", "Residual max FI",
                                "Interior max FI", "Critical ΔT (first-ply failure)"]
        assert values["FE knockdown"] == "n/a"
        assert values["Residual max FI"] == f"{fe.residual_max_failure_index:.3f}"
        assert fe.interior_max_failure_index is not None   # 10 x 6 has an interior
        assert values["Interior max FI"] == f"{fe.interior_max_failure_index:.3f}"
        lam = fe.first_ply_failure_load_factor
        assert values["Critical ΔT (first-ply failure)"] == f"{lam * -150.0:.0f} K"
        assert "basis 'delta_T'" in caption
        assert f"{lam:.3g} multiplies ΔT = -150 K" in caption

    def test_summary_handles_unstressed_and_missing_values(self, thermal_result):
        fe = dataclasses.replace(thermal_result["fe_field"],
                                 first_ply_failure_load_factor=float("inf"),
                                 interior_max_failure_index=None)
        rows, caption = app.fe_thermal_summary({**thermal_result, "fe_field": fe})
        values = {label: value for label, value, _ in rows}
        assert values["Critical ΔT (first-ply failure)"] == "none"
        assert values["Interior max FI"] == "n/a"
        assert "infinite" in caption

    @pytest.mark.parametrize("value, text", [
        (None, "n/a"), (float("nan"), "n/a"), (float("inf"), "∞"), (0.12345, "0.123")])
    def test_fmt_value(self, value, text):
        assert app._fmt_value(value) == text

    def test_combined_result_and_summary(self, combined_results):
        comb, mech = combined_results
        fe, fe_mech = comb["fe_field"], mech["fe_field"]
        assert comb["fe_loading"] == "tension"
        assert fe.load_factor_basis == "mechanical_with_residual"
        assert fe_mech.load_factor_basis == "mechanical"
        # Knockdown is the mechanical one; the reference factor is the
        # mechanical-only solve's.
        assert fe.knockdown == pytest.approx(fe_mech.knockdown, rel=1e-12)
        assert comb["fe_mechanical_fpf"] == pytest.approx(
            fe_mech.first_ply_failure_load_factor, rel=1e-9)
        # Cool-down puts the 90 deg plies in transverse tension, so the
        # residual stress brings tensile first-ply failure forward.
        assert fe.first_ply_failure_load_factor < comb["fe_mechanical_fpf"]

        rows, caption = app.fe_thermal_summary(comb)
        values = {label: value for label, value, _ in rows}
        assert values["FE knockdown"] == f"{fe.knockdown:.3f}"
        assert values["FPF load factor"] == f"{fe.first_ply_failure_load_factor:.2f}"
        assert "basis 'mechanical_with_residual'" in caption
        assert (f"Without the residual stress the factor is "
                f"{comb['fe_mechanical_fpf']:.2f}") in caption
        assert app.fe_thermal_summary(mech) is None

    def test_unresolved_layup_is_reported_with_the_nz_that_works(self):
        r = app.run_analysis(_thermal_cfg(nz=6))
        assert r["fe_field"] is None
        assert "nz = k * n_plies" in r["fe_skipped_reason"]
        assert r["fe_nz_suggestion"] == 8
        assert app.fe_thermal_summary(r) is None
        for mode in ("compression", "tension", "shear", "ilss"):
            assert mode in r["empirical"]

    def test_material_without_cte_is_reported(self):
        r = app.run_analysis(_thermal_cfg(material_name="T800_epoxy"))
        assert r["fe_field"] is None
        assert "thermal expansion coefficients" in r["fe_skipped_reason"]
        assert r["fe_nz_suggestion"] is None

    @pytest.mark.parametrize("overrides, match", [
        ({"thermal_mode": "sometimes"}, "Unknown thermal_mode"),
        ({"delta_T": None}, "needs a finite delta_T"),
        ({"delta_T": float("nan")}, "needs a finite delta_T"),
        ({"thermal_mode": "combined", "delta_T": "150"}, "needs a finite delta_T"),
    ])
    def test_invalid_thermal_config_raises(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            app.run_analysis(_thermal_cfg(**overrides))

    def test_plot_results_has_no_fe_bar_for_thermal_only(self, thermal_result):
        fig = app.plot_results(thermal_result, "0/90/90/0")
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert labels == ["Judd-Wright", "Power Law", "Linear"]
        note = "\n".join(t.get_text() for t in fig.axes[0].texts)
        assert "thermal-only solve has no stiffness knockdown" in note
        plt.close(fig)

    @pytest.mark.parametrize("nz, n_plies, expected", [
        (12, 24, 24), (6, 4, 8), (8, 4, 8), (30, 24, 48), (2, 4, 4)])
    def test_resolving_nz(self, nz, n_plies, expected):
        assert app._resolving_nz(nz, n_plies) == expected

    def test_delta_T_suggestion(self):
        from porosity_fe import MATERIALS
        mat = MATERIALS["AS4_3501_6_epoxy"]
        assert mat.T_stress_free is None
        value, basis = app._delta_T_suggestion(mat)
        assert value == app._SUGGESTED_DELTA_T_177C_EPOXY
        assert "177 °C-cure epoxy" in basis
        recorded = dataclasses.replace(mat, T_stress_free=180.0)
        assert app._delta_T_suggestion(recorded)[0] == recorded.T_ref - 180.0
        service = dataclasses.replace(recorded, T_service=60.0)
        assert app._delta_T_suggestion(service)[0] == 60.0 - 180.0

    def test_callbacks_set_session_state(self, monkeypatch):
        monkeypatch.setattr(app.st, "session_state", {})
        app._set_delta_T(-150)
        app._apply_nz_fix(24)
        assert app.st.session_state == {"thermal_delta_T": -150.0,
                                        "expert_mode": True, "mesh_nz": 24}


class TestVTUDownload:
    """The Export tab's FE results as VTU bytes (IMPROVEMENT_PLAN 3.6, E1)."""

    def test_bytes_match_the_library_writer(self, comp_result, tmp_path):
        data = app.fe_vtu_bytes(comp_result)
        ref = tmp_path / "ref.vtu"
        comp_result["fe_field"].to_vtu(comp_result["mesh"], ref)
        assert data == ref.read_bytes()
        head = data.split(b"<AppendedData")[0].decode("ascii")
        assert head.startswith('<?xml version="1.0"?>')
        assert '<VTKFile type="UnstructuredGrid"' in head
        n_elem = len(comp_result["mesh"].elements)
        assert f'NumberOfCells="{n_elem}"' in head
        assert 'Name="knockdown"' in head and 'Name="sigma_xx_nodal"' in head

    def test_thermal_bytes_carry_the_residual_stress(self, thermal_result):
        head = app.fe_vtu_bytes(thermal_result).split(b"<AppendedData")[0].decode("ascii")
        for name in ("residual_sigma_11_local", "residual_sigma_22_local",
                     "residual_tau_12_local"):
            assert f'Name="{name}"' in head
        assert 'Name="knockdown"' not in head     # NaN for thermal-only

    def test_cached_wrapper_returns_the_same_bytes(self, comp_result):
        key = app._config_to_key(comp_result["config"])
        assert app.fe_vtu_bytes_cached(key, comp_result) == app.fe_vtu_bytes(comp_result)

    def test_no_fe_field_raises(self):
        with pytest.raises(ValueError, match="no FE field"):
            app.fe_vtu_bytes({"fe_field": None})
