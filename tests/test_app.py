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
import numpy as np
import pytest
from matplotlib.collections import QuadMesh

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


def _flexure_cfg(**overrides) -> dict:
    """``_base_cfg`` for three-point flexure. The D7264 specimen is 13 mm
    wide with elements no longer than the laminate thickness, so 0.5 mm
    plies keep it small (40 x 8 x 4 elements)."""
    cfg = _base_cfg(loading_mode="flexure_3pt", t_ply=0.5, nz=4)
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def flexure_result():
    """One three-point flexure analysis (the D7264 specimen FE solve)."""
    return app.run_analysis(_flexure_cfg())


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

    def test_failure_criterion_defaults_to_tsai_wu_and_enters_the_key(self):
        key = dict(app._config_to_key(_base_cfg()))
        assert key["failure_criterion"] == "tsai_wu"
        assert app._config_to_key(_base_cfg(failure_criterion="hashin")) != \
            app._config_to_key(_base_cfg())


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
        (_thermal_cfg(thermal_mode="combined", loading_mode="ilss"), 3),
    ], ids=["ilss", "thermal_only", "combined"])
    def test_fe_solve_holds_one_factorization_at_a_time(
            self, monkeypatch, lu_alive_at_build, cfg, n_lu):
        """The solver frees the knockdown's pristine reference before it
        factorizes the porous model, and the thermal factorization once
        its unit solution is cached, so the app holds one factorization at
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


# ----------------------------------------------------------------------
# Plot polish: legible sections, colour scales, legend placement
# ----------------------------------------------------------------------

def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _overlap(a, b) -> bool:
    """Whether two display-space bounding boxes intersect."""
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _assert_plain_ticks(cax):
    """A drawn horizontal colorbar: no offset text, and every visible tick
    label reads its own location (no offset or scientific notation)."""
    assert cax.xaxis.get_offset_text().get_text() == ""
    lo, hi = cax.get_xlim()
    shown = [(loc, label.get_text()) for loc, label in
             zip(cax.get_xticks(), cax.get_xticklabels(), strict=True)
             if lo - 1e-9 * abs(lo) <= loc <= hi + 1e-9 * abs(hi)]
    assert len(shown) >= 3
    for loc, text in shown:
        assert "e" not in text.lower()
        assert float(text.replace("−", "-")) == pytest.approx(loc, abs=1e-6)


def _title_text(ax):
    """The (left-aligned) title artist of ``ax``."""
    from matplotlib.text import Text
    title = ax.get_title(loc="left")
    assert title
    return next(t for t in ax.get_children()
                if isinstance(t, Text) and t.get_text() == title)


def _section_parts(fig):
    """The section axes, the colorbar axes and the section's QuadMesh."""
    ax, cax = fig.axes[0], fig.axes[1]
    mesh_artist = next(c for c in ax.collections if isinstance(c, QuadMesh))
    return ax, cax, mesh_artist


class TestSectionPlots:
    """Mesh and Stress tabs: thin sections stay legible (z exaggerated, a
    colorbar sized to the section, titles clear of everything)."""

    @pytest.mark.parametrize("length, thickness, factor", [
        (50.0, 4.392, 4),        # default coupon, 24 x 0.183 mm plies
        (26.352, 4.392, 2),      # ASTM D2344 specimen (6h long)
        (168.6528, 4.392, 12),   # ASTM D7264 specimen (span 32h + overhang)
        (50.0, 0.732, 25),       # 4-ply coupon
        (10.0, 5.0, 1),          # deep enough already
        (0.0, 1.0, 1), (10.0, float("nan"), 1), (-1.0, 1.0, 1),
    ])
    def test_section_exaggeration(self, length, thickness, factor):
        assert app.section_exaggeration(length, thickness) == factor

    def test_exaggerated_factor_reaches_the_target_aspect(self):
        for ratio in (0.3, 0.2, 0.1, 0.05, 0.02, 0.004, 1e-4):
            f = app.section_exaggeration(1.0, ratio)
            assert ratio * f >= app._SECTION_TARGET_ASPECT
            # ... with the smallest factor of the series that does.
            assert f == 1 or ratio * f <= 2.0 * app._SECTION_TARGET_ASPECT

    @pytest.mark.parametrize("which", ["mesh", "stress"])
    def test_section_layout(self, comp_result, which):
        fig = (app.plot_mesh(comp_result) if which == "mesh"
               else app.plot_stress(comp_result, "σ₁₁ (fiber)"))
        renderer = _renderer(fig)
        ax, cax, _ = _section_parts(fig)
        mesh = comp_result["mesh"]
        xmin, xmax = ax.get_xlim()
        zmin, zmax = ax.get_ylim()
        factor = app.section_exaggeration(xmax - xmin, zmax - zmin)
        assert factor > 1                       # the 4-ply coupon is a sliver
        assert ax.get_aspect() == factor
        assert f"z exaggerated ×{factor}" in ax.get_ylabel()
        assert zmax - zmin == pytest.approx(mesh.L_z, rel=1e-9)
        # Horizontal colorbar exactly under the section, as wide as it.
        assert cax.get_position().width == pytest.approx(
            ax.get_position().width, rel=1e-6)
        assert cax.get_position().x0 == pytest.approx(ax.get_position().x0, rel=1e-6)
        assert cax.get_position().y1 < ax.get_position().y0
        # The title clears the subtitle, the section and the colorbar.
        title = _title_text(ax).get_window_extent(renderer)
        subtitle = ax.texts[0].get_window_extent(renderer)
        assert title.width > 0 and subtitle.width > 0
        assert title.y0 >= subtitle.y1 - 1e-6
        for other in (subtitle, ax.get_window_extent(renderer),
                      cax.get_tightbbox(renderer)):
            assert not _overlap(title, other)
        assert not _overlap(subtitle, ax.get_window_extent(renderer))
        # The x-axis label and tick labels sit above the colorbar.
        assert ax.get_tightbbox(renderer).y0 > cax.get_window_extent(renderer).y1
        plt.close(fig)

    def test_uniform_retention_has_a_sane_colour_scale(self, comp_result):
        """Uniform porosity gives one stiffness retention everywhere: the
        scale spans the value +-1 point with plain ticks, not round-off
        with an offset (the old plot showed 1e-12+9.7e1)."""
        fig = app.plot_mesh(comp_result)
        _renderer(fig)
        _, cax, im = _section_parts(fig)
        value = 100.0 * float(np.mean(comp_result["mesh"].stiffness_reduction))
        vmin, vmax = im.get_clim()
        assert vmin < value < vmax
        assert vmax - vmin == pytest.approx(2.0, rel=1e-9)
        _assert_plain_ticks(cax)
        assert f"uniform: {value:.1f} %" in cax.get_xlabel()
        plt.close(fig)

    def test_varying_retention_keeps_100_percent_at_the_top(self):
        r = app.run_analysis(_base_cfg(distribution="clustered", void_shape="penny"))
        fig = app.plot_mesh(r)
        _, cax, im = _section_parts(fig)
        sr = 100.0 * r["mesh"].stiffness_reduction
        vmin, vmax = im.get_clim()
        assert vmax == 100.0 and vmin < float(sr.min())
        assert "uniform" not in cax.get_xlabel()
        plt.close(fig)

    def test_constant_stress_field_has_a_sane_colour_scale(self, comp_result):
        fe = comp_result["fe_field"]
        rng = np.random.default_rng(0)
        flat = np.full_like(fe.stress_local, -35.0)
        flat += 1e-13 * rng.standard_normal(flat.shape)    # round-off
        r = {**comp_result, "fe_field": dataclasses.replace(fe, stress_local=flat)}
        fig = app.plot_stress(r, "σ₁₁ (fiber)")
        _renderer(fig)
        _, cax, qm = _section_parts(fig)
        vmin, vmax = qm.get_clim()
        assert vmin < -35.0 < vmax
        assert vmax - vmin == pytest.approx(2.0, rel=1e-6)
        _assert_plain_ticks(cax)
        assert "uniform: -35 MPa" in cax.get_xlabel()
        # Compression only: the sequential blue map, not the diverging one.
        assert qm.get_cmap() is app._STRESS_CMAP_NEGATIVE
        plt.close(fig)

    @pytest.mark.parametrize("values, kind, clipped", [
        # Symmetric about zero: only the larger (positive) end is clipped.
        (np.linspace(-10.0, 30.0, 50), "diverging", "max"),
        (np.linspace(-38.0, -33.0, 50), "negative", "both"),
        (np.linspace(33.0, 38.0, 50), "positive", "both"),
        (np.linspace(0.0, 38.0, 50), "positive", "both"),
    ])
    def test_stress_colour_scale_choice(self, values, kind, clipped):
        cmap, norm, extend, constant = app._stress_colour_scale(values)
        assert not constant
        if kind == "diverging":
            assert cmap == app._STRESS_CMAP_DIVERGING
            assert norm.vmin == -norm.vmax          # white = true zero (#51)
        else:
            assert cmap is (app._STRESS_CMAP_NEGATIVE if kind == "negative"
                            else app._STRESS_CMAP_POSITIVE)
            # Sequential over the field's own range, not anchored at zero.
            assert values.min() <= norm.vmin < norm.vmax <= values.max()
        # The 2nd-98th percentile range clips the ends: the colorbar says so.
        assert extend == clipped

    def test_unclipped_field_has_no_extend_arrows(self):
        values = np.r_[np.full(10, -5.0), np.full(10, 5.0)]
        _, norm, extend, _ = app._stress_colour_scale(values)
        assert (norm.vmin, norm.vmax, extend) == (-5.0, 5.0, "neither")

    @pytest.mark.parametrize("lo, hi, nearly_uniform", [
        (35.0, 36.0, True), (10.0, 40.0, False), (-36.0, -35.0, True)])
    def test_narrow_single_signed_range_is_called_nearly_uniform(
            self, comp_result, lo, hi, nearly_uniform):
        """A sequential scale over a narrow range would make 1 % look like
        a strong gradient; the label says how small the spread is."""
        fe = comp_result["fe_field"]
        n_elem = fe.stress_local.shape[0]
        ramp = np.linspace(lo, hi, n_elem)[:, None, None]
        field = np.broadcast_to(ramp, fe.stress_local.shape).copy()
        r = {**comp_result, "fe_field": dataclasses.replace(fe, stress_local=field)}
        fig = app.plot_stress(r, "σ₁₁ (fiber)")
        _, cax, qm = _section_parts(fig)
        assert ("nearly uniform" in cax.get_xlabel()) is nearly_uniform
        if nearly_uniform:
            vmin, vmax = qm.get_clim()
            mid, spread = 0.5 * (vmin + vmax), 0.5 * (vmax - vmin) / abs(0.5 * (vmin + vmax))
            assert f"nearly uniform: {mid:.4g} MPa ± {100 * spread:.2g} %" in cax.get_xlabel()
        plt.close(fig)

    def test_von_mises_uses_the_magnitude_map(self, comp_result):
        fig = app.plot_stress(comp_result, "Von Mises")
        _, _, qm = _section_parts(fig)
        assert qm.get_cmap().name == app._STRESS_CMAP_MAGNITUDE
        plt.close(fig)

    def test_single_signed_thermal_stress_is_not_flat(self, thermal_result):
        """Cool-down puts the plies in transverse tension: a positive
        field gets a sequential map over its own range with continuous
        colours, instead of one flat shade of a zero-centred map."""
        fig = app.plot_stress(thermal_result, "σ₂₂ (transverse)")
        _, _, qm = _section_parts(fig)
        data = qm.get_array().compressed()
        lo, hi = np.percentile(data, [2.0, 98.0])
        assert lo > 0.0
        assert qm.get_cmap() is app._STRESS_CMAP_POSITIVE
        assert qm.get_cmap().N >= 256
        vmin, vmax = qm.get_clim()
        assert vmin == pytest.approx(lo, rel=1e-12)
        assert vmax == pytest.approx(hi, rel=1e-12)
        plt.close(fig)

    def test_stress_plot_draws_the_interior_elements(self, comp_result):
        fig = app.plot_stress(comp_result, "τ₁₂ (in-plane shear)")
        ax, _, qm = _section_parts(fig)
        mesh = comp_result["mesh"]
        n_rows, n_cols = qm.get_array().shape
        assert n_rows == mesh.nz
        assert 0 < n_cols < mesh.nx                  # the 10 % ends left out
        xmin, xmax = ax.get_xlim()
        assert xmin > 0.0 and xmax < mesh.L_x
        plt.close(fig)

    def test_results_legend_sits_outside_the_axes(self, comp_result):
        fig = app.plot_results(comp_result, "0/90/90/0")
        renderer = _renderer(fig)
        ax = fig.axes[0]
        legend = ax.get_legend().get_window_extent(renderer)
        axes_box = ax.get_window_extent(renderer)
        assert not _overlap(legend, axes_box)
        assert legend.x0 >= axes_box.x1
        assert legend.x1 <= fig.bbox.x1 + 1.0       # and inside the figure
        title = _title_text(ax).get_window_extent(renderer)
        assert not _overlap(title, legend)
        plt.close(fig)

    def test_uq_legend_outside_and_plain_ticks(self):
        kd = np.full(200, 0.8123)                      # all CoVs zero
        uq = {"samples": {"knockdown": kd}, "mode": "compression",
              "n_samples": 200, "nominal": {"knockdown": 0.8123},
              "knockdown": {"percentiles": {"p5": 0.8123, "p50": 0.8123,
                                            "p95": 0.8123}}}
        fig = app.plot_uq(uq)
        renderer = _renderer(fig)
        ax = fig.axes[0]
        assert not _overlap(ax.get_legend().get_window_extent(renderer),
                            ax.get_window_extent(renderer))
        assert ax.xaxis.get_offset_text().get_text() == ""
        plt.close(fig)


class TestDefaultStressComponent:
    @pytest.mark.parametrize("fe_loading, expected", [
        ("ilss", "τ₁₃ (interlaminar)"),
        ("thermal", "σ₂₂ (transverse)"),
        ("shear", "τ₁₂ (in-plane shear)"),
        ("compression", "σ₁₁ (fiber)"),
        ("tension", "σ₁₁ (fiber)"),
        ("flexure_3pt", "σ₁₁ (fiber)"),
        ("flexure_4pt", "σ₁₁ (fiber)"),
    ])
    def test_default_per_mode(self, fe_loading, expected):
        assert app.default_stress_component({"fe_loading": fe_loading}) == expected
        assert expected in app._STRESS_COMPONENTS

    def test_results_map_to_their_default(self, ilss_result, thermal_result,
                                          comp_result):
        assert app.default_stress_component(ilss_result) == "τ₁₃ (interlaminar)"
        assert app.default_stress_component(thermal_result) == "σ₂₂ (transverse)"
        assert app.default_stress_component(comp_result) == "σ₁₁ (fiber)"

    def test_selection_follows_the_default_until_the_user_picks(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(app.st, "session_state", state)
        app._sync_stress_component("σ₁₁ (fiber)")
        assert state["stress_component"] == "σ₁₁ (fiber)"
        app._sync_stress_component("τ₁₃ (interlaminar)")   # new load case
        assert state["stress_component"] == "τ₁₃ (interlaminar)"
        state["stress_component"] = "Von Mises"             # the user's pick
        app._sync_stress_component("σ₂₂ (transverse)")
        assert state["stress_component"] == "Von Mises"
        state["stress_component"] = "no such component"
        app._sync_stress_component("σ₁₁ (fiber)")
        assert state["stress_component"] == "σ₁₁ (fiber)"

    def test_every_component_has_a_file_slug(self):
        assert {idx for idx, _ in app._STRESS_COMPONENTS.values()} == \
            set(app._STRESS_SLUGS)


# ----------------------------------------------------------------------
# FE-only flexure (ASTM D7264)
# ----------------------------------------------------------------------

class TestFlexure:
    def test_solves_the_d7264_specimen(self, flexure_result):
        r = flexure_result
        mesh = r["mesh"]
        fe = r["fe_field"]
        assert fe is not None, r["fe_skipped_reason"]
        assert r["fe_loading"] == "flexure_3pt"
        assert mesh.flexure_geometry is not None and mesh.ilss_geometry is None
        assert mesh.nz == 4
        assert fe.displacement.shape == (mesh.n_nodes, 3)
        h = 4 * 0.5
        spec = r["fe_specimen"]
        assert spec["standard"] == "ASTM D7264 procedure A"
        assert spec["specimen"] == "three-point flexure specimen"
        assert spec["span_mm"] == pytest.approx(32 * h, rel=1e-9)
        assert spec["span_to_thickness"] == pytest.approx(32.0, rel=1e-9)
        assert spec["width_mm"] == pytest.approx(13.0, rel=1e-12)
        assert spec["thickness_mm"] == pytest.approx(h, rel=1e-12)
        assert spec["load_span_mm"] is None
        assert spec["mesh"] == f"{mesh.nx}x{mesh.ny}x{mesh.nz}"
        assert r["fe_short_beam"] is None

    def test_fe_and_clt_flexural_moduli(self, flexure_result):
        from porosity_fe import MATERIALS, compute_degraded_clt_flexural_modulus
        r = flexure_result
        flex = r["fe_flexure"]
        material = dataclasses.replace(MATERIALS["T800_epoxy"], t_ply=0.5, n_plies=4)
        angles = [0.0, 90.0, 90.0, 0.0]
        clt = compute_degraded_clt_flexural_modulus(material, angles, 0.03)["Ef_x"]
        clt0 = compute_degraded_clt_flexural_modulus(material, angles, 0.0)["Ef_x"]
        assert flex["procedure"] == "3pt"
        assert flex["clt_E_f"] == pytest.approx(clt, rel=1e-12)
        assert flex["clt_E_f_pristine"] == pytest.approx(clt0, rel=1e-12)
        assert flex["clt_knockdown"] == pytest.approx(clt / clt0, rel=1e-12)
        assert flex["clt_void_shape"] == "spherical"
        fe = r["fe_field"]
        # Same porosity model (spherical, uniform), cross-ply: FE and CLT
        # agree closely (the shear-corrected FE modulus).
        assert fe.effective_modulus == pytest.approx(clt, rel=0.02)
        assert 0.0 < fe.knockdown < 1.0

    def test_summary(self, flexure_result):
        rows, caption = app.fe_flexure_summary(flexure_result)
        values = {label: value for label, value, _ in rows}
        assert list(values) == ["FE knockdown", "FE flexural modulus E_f",
                                "CLT E_f (D matrix)", "FE / CLT E_f"]
        fe, flex = flexure_result["fe_field"], flexure_result["fe_flexure"]
        assert values["FE knockdown"] == f"{fe.knockdown:.3f}"
        assert values["FE flexural modulus E_f"] == f"{fe.effective_modulus / 1000:.1f} GPa"
        assert values["CLT E_f (D matrix)"] == f"{flex['clt_E_f'] / 1000:.1f} GPa"
        assert values["FE / CLT E_f"] == f"{fe.effective_modulus / flex['clt_E_f']:.3f}"
        assert "ASTM D7264 procedure A three-point flexure specimen" in caption
        assert "(32:1)" in caption and "loaded at mid-span" in caption
        assert f"CLT knockdown of E_f: {flex['clt_knockdown']:.3f}" in caption
        assert "no empirical flexure knockdown" in caption
        assert "porosity models differ" not in caption      # same model here

    def test_summary_names_the_porosity_model_differences(self, flexure_result):
        cfg = {**flexure_result["config"], "void_shape": "penny",
               "distribution": "clustered", "cluster_location": "surface"}
        _, caption = app.fe_flexure_summary({**flexure_result, "config": cfg})
        assert "spherical voids in CLT, penny voids in FE" in caption
        assert "the clustered (surface) distribution in FE" in caption

    def test_summary_without_fe_field(self, flexure_result):
        rows, _ = app.fe_flexure_summary({**flexure_result, "fe_field": None})
        values = {label: value for label, value, _ in rows}
        assert values["FE knockdown"] == "n/a"
        assert values["FE flexural modulus E_f"] == "n/a"
        assert values["FE / CLT E_f"] == "n/a"
        assert values["CLT E_f (D matrix)"].endswith(" GPa")

    def test_other_modes_have_no_flexure_summary(self, comp_result, ilss_result):
        for r in (comp_result, ilss_result):
            assert r["fe_flexure"] is None
            assert app.fe_flexure_summary(r) is None

    def test_empirical_results_do_not_depend_on_the_mode(self, flexure_result):
        plain = app.run_analysis(_flexure_cfg(loading_mode="compression"))
        assert flexure_result["empirical"] == plain["empirical"]

    def test_four_point(self):
        r = app.run_analysis(_flexure_cfg(loading_mode="flexure_4pt"))
        spec = r["fe_specimen"]
        assert r["fe_field"] is not None and r["fe_field"].effective_modulus > 0
        assert spec["standard"] == "ASTM D7264 procedure B"
        assert spec["load_span_mm"] == pytest.approx(0.5 * spec["span_mm"], rel=1e-9)
        assert r["fe_flexure"]["procedure"] == "4pt"
        _, caption = app.fe_flexure_summary(r)
        assert "four-point flexure specimen" in caption
        assert "two lines" in caption

    def test_thermal_only_with_flexure_selected_stays_on_the_coupon(self):
        r = app.run_analysis(_thermal_cfg(loading_mode="flexure_3pt"))
        assert r["fe_loading"] == "thermal"
        assert r["fe_specimen"] is None and r["fe_flexure"] is None
        assert r["mesh"].flexure_geometry is None

    def test_specimen_mesh_failure_keeps_empirical_results(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise ValueError("too many elements")

        monkeypatch.setattr(app.CompositeMesh, "for_flexure", _raise)
        r = app.run_analysis(_flexure_cfg())
        assert r["fe_field"] is None
        assert "specimen mesh could not be built" in r["fe_skipped_reason"]
        assert r["fe_specimen"] is None
        assert r["fe_flexure"] is not None           # the CLT value still shows
        assert set(r["empirical"]) >= {"compression", "tension", "shear", "ilss"}

    def test_plots(self, flexure_result):
        fig = app.plot_results(flexure_result, "0/90/90/0")
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "COMPRESSION", "TENSION", "SHEAR", "ILSS", "FLEXURE (FE)"]
        assert ax.get_legend_handles_labels()[1] == [
            "Judd-Wright", "Power Law", "Linear", "FE Stiffness (flexure_3pt)"]
        fe_bars = [p for p in ax.patches if p.get_hatch() == "//"]
        assert len(fe_bars) == 1
        bar = fe_bars[0]
        # Alone in its own group, centred on it.
        assert bar.get_x() + bar.get_width() / 2 == pytest.approx(4.0, abs=1e-12)
        assert bar.get_height() == flexure_result["fe_field"].knockdown
        note = "\n".join(t.get_text() for t in ax.texts)
        assert "no empirical flexure model" in note
        plt.close(fig)
        for fig in (app.plot_mesh(flexure_result),
                    app.plot_stress(flexure_result, "σ₁₁ (fiber)")):
            ax, _, _ = _section_parts(fig)
            xmin, xmax = ax.get_xlim()
            zmin, zmax = ax.get_ylim()
            assert ax.get_aspect() == app.section_exaggeration(xmax - xmin, zmax - zmin)
            plt.close(fig)

    def test_bars_centred_in_each_group(self, comp_result):
        """Groups without the FE bar centre their three bars too."""
        fig = app.plot_results(comp_result, "0/90/90/0")
        ax = fig.axes[0]
        for j in range(1, 4):                       # tension, shear, ilss
            centres = sorted(p.get_x() + p.get_width() / 2 for p in ax.patches
                             if abs(p.get_x() + p.get_width() / 2 - j) < 0.5)
            assert len(centres) == 3
            assert np.mean(centres) == pytest.approx(j, abs=1e-12)
        plt.close(fig)

    @pytest.mark.parametrize("mode, text", [
        ("flexure_3pt", "20 N midspan load"),
        ("flexure_4pt", "20 N total load on the two load lines"),
        ("ilss", "20 N midspan load"),
        ("compression", "-2 % strain"),
        ("shear", "2 % shear strain"),
    ])
    def test_applied_load_text(self, mode, text):
        assert app._applied_load_text(mode, 2.0) == text

    def test_sidebar_caption(self):
        caption = app._flexure_specimen_caption("flexure_3pt", 24, 0.183, 24)
        h = 24 * 0.183
        assert "ASTM D7264" in caption and "procedure A" in caption
        assert f"{32 * h:.1f} mm span (32:1)" in caption
        assert "24 element layers" in caption
        assert "no empirical flexure knockdown" in caption
        assert "procedure B" in app._flexure_specimen_caption("flexure_4pt", 0, 0.183, 4)

    def test_loading_choices(self):
        assert list(app._LOADING_LABELS) == [
            "compression", "tension", "shear", "ilss", "flexure_3pt", "flexure_4pt",
            "dcb", "enf", "mmb"]
        for mode in app._FLEXURE_MODES:
            assert "FE only" in app._LOADING_LABELS[mode]
            assert "D7264" in app._LOADING_LABELS[mode]
        for mode, (standard, *_) in app._FRACTURE_MODES.items():
            assert "FE only" in app._LOADING_LABELS[mode]
            assert standard.split()[-1] in app._LOADING_LABELS[mode]


# ----------------------------------------------------------------------
# Mesh-convergence check
# ----------------------------------------------------------------------

class TestConvergenceCheck:
    def test_plan_refines_the_analysis_mesh_in_plane(self, comp_result):
        from porosity_fe.fe.refinement import _estimate_memory_bytes
        plan = app.convergence_plan(comp_result, (1.0, 1.5))
        assert [lv["resolution"] for lv in plan] == [(10, 4, 6), (15, 6, 6)]
        assert [lv["n_elements"] for lv in plan] == [240, 540]
        assert plan[1]["n_dof"] == 3 * 16 * 7 * 7
        for lv in plan:
            assert lv["memory_gb"] == pytest.approx(
                _estimate_memory_bytes(lv["resolution"]) / 1e9, rel=1e-12)
        t0 = comp_result["fe_solve_time_s"]
        assert t0 > 0.0
        assert plan[0]["time_s"] == pytest.approx(t0, rel=1e-12)
        assert plan[1]["time_s"] == pytest.approx(
            t0 * (plan[1]["n_dof"] / plan[0]["n_dof"]) ** 1.5, rel=1e-12)

    def test_plan_without_a_recorded_time(self, comp_result):
        plan = app.convergence_plan({**comp_result, "fe_solve_time_s": None}, (1.0, 2.0))
        assert all(lv["time_s"] is None for lv in plan)

    def test_ilss_plan_keeps_nx_a_multiple_of_six(self, ilss_result):
        plan = app.convergence_plan(ilss_result, (1.0, 1.1, 1.5))
        assert [lv["resolution"] for lv in plan] == [(24, 8, 6), (30, 9, 6), (36, 12, 6)]

    def test_flexure_plan_keeps_nx_even(self, flexure_result):
        assert (flexure_result["mesh"].nx, flexure_result["mesh"].ny) == (40, 8)
        plan = app.convergence_plan(flexure_result, (1.0, 1.025, 1.5))   # 41 -> 42
        assert [lv["resolution"] for lv in plan] == [(40, 8, 4), (42, 8, 4), (60, 12, 4)]

    def test_combined_thermal_plan_counts_two_factorizations(self, combined_results):
        from porosity_fe.fe.refinement import _estimate_memory_bytes
        comb, _ = combined_results
        lv = app.convergence_plan(comb, (1.0,))[0]
        assert lv["memory_gb"] == pytest.approx(
            _estimate_memory_bytes(lv["resolution"], n_factorizations=2) / 1e9,
            rel=1e-12)

    def test_unavailable_reasons(self, comp_result, flexure_result):
        assert app.convergence_unavailable_reason(comp_result) is None
        assert app.convergence_unavailable_reason(flexure_result) is None
        assert "skipped" in app.convergence_unavailable_reason(
            {**comp_result, "fe_field": None})

    def test_runs_refinement_study_on_the_analysis_case(self, monkeypatch):
        """Wiring on a tiny mesh: the study gets the analysis' material,
        layup, porosity, load and the planned levels; the tables read it."""
        calls = []
        real = app.refinement_study

        def spy(material, porosity, loading, **kwargs):
            calls.append((material, porosity, loading, kwargs))
            return real(material, porosity, loading, **kwargs)

        monkeypatch.setattr(app, "refinement_study", spy)
        cfg = _base_cfg(nx=6, ny=2, nz=4, distribution="clustered",
                        cluster_location="surface")
        study = app.run_convergence_check(cfg, [(6, 2, 4), (9, 3, 4)])
        (material, porosity, loading, kwargs), = calls
        assert (material.t_ply, material.n_plies) == (0.183, 4)
        assert porosity == pytest.approx(0.03)
        assert loading == "compression"
        assert kwargs["applied_strain"] == -0.01
        assert kwargs["resolutions"] == [(6, 2, 4), (9, 3, 4)]
        assert kwargs["ply_angles"] == cfg["angles"]
        assert kwargs["porosity_config"] == {
            "distribution": "clustered", "void_shape": "spherical",
            "cluster_location": "surface"}
        assert kwargs["delta_T"] is None
        assert kwargs["allow_unresolved_layup"] is True
        assert study.resolutions == ((6, 2, 4), (9, 3, 4))

        levels, verdicts = app.convergence_tables(study)
        assert [row["Level"] for row in levels] == [1, 2]
        assert [row["Mesh"] for row in levels] == ["6 × 2 × 4", "9 × 3 × 4"]
        assert [row["Elements"] for row in levels] == ["48", "108"]
        assert levels[1]["Knockdown"] == f"{study.levels[1].knockdown:.4f}"
        assert levels[1]["Domain max FI"] == f"{study.levels[1].max_failure_index:.4g}"
        assert [row["Quantity"] for row in verdicts] == [
            "Knockdown", "Interior max FI", "Domain max FI"]
        for row, name in zip(verdicts, ("knockdown", "interior_max_failure_index",
                                        "max_failure_index"), strict=True):
            c = study.convergence[name]
            assert row["Verdict"] == c.verdict
            assert row["Behaviour"] == c.behavior
            assert row["Verdict"] in {"converged", "not converged",
                                      "mesh-dependent", "n/a"}
        change = study.convergence["max_failure_index"].relative_changes[-1]
        assert verdicts[2]["Last change"] == f"{100 * change:+.2f} %"

    @pytest.mark.parametrize("cfg, loading, extra", [
        (_thermal_cfg(), "thermal", {"delta_T": -150.0,
                                     "allow_unresolved_layup": False}),
        (_thermal_cfg(thermal_mode="combined", loading_mode="tension"), "tension",
         {"delta_T": -150.0, "applied_strain": 0.01}),
        (_base_cfg(loading_mode="ilss"), "ilss", {"applied_load": -10.0}),
        (_flexure_cfg(), "flexure_3pt", {"applied_load": -10.0}),
        (_flexure_cfg(loading_mode="flexure_4pt"), "flexure_4pt",
         {"applied_load": -10.0}),
        (_base_cfg(loading_mode="shear"), "shear", {"applied_strain": 0.01}),
    ])
    def test_load_case_wiring(self, monkeypatch, cfg, loading, extra):
        calls = []
        monkeypatch.setattr(app, "refinement_study",
                            lambda m, p, ld, **kw: calls.append((ld, kw)) or "study")
        assert app.run_convergence_check(cfg, [(4, 2, 4), (6, 3, 4)]) == "study"
        (got_loading, kwargs), = calls
        assert got_loading == loading
        for key, value in extra.items():
            assert kwargs[key] == value
        if loading == "thermal":
            assert "applied_strain" not in kwargs and "applied_load" not in kwargs

    def test_cached_wrapper(self, monkeypatch):
        seen = []
        monkeypatch.setattr(app, "run_convergence_check",
                            lambda cfg, res: seen.append((cfg, res)) or "study")
        key = app._config_to_key(_base_cfg(nx=7))
        assert app.run_convergence_cached(key, ((7, 4, 6), (11, 6, 6))) == "study"
        cfg, res = seen[0]
        assert cfg["angles"] == [0.0, 90.0, 90.0, 0.0] and cfg["nx"] == 7
        assert res == ((7, 4, 6), (11, 6, 6))



# ----------------------------------------------------------------------
# FE failure block (criterion selector, governing mode, mode maxima)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def ilss_by_criterion():
    """ILSS on the D2344 specimen with each FE failure criterion."""
    return {c: app.run_analysis(_base_cfg(loading_mode="ilss", failure_criterion=c))
            for c in ("tsai_wu", "hashin", "max_stress")}


class TestFEFailureSummary:
    def test_criterion_reaches_the_solver(self, ilss_by_criterion):
        for crit, result in ilss_by_criterion.items():
            assert result["fe_field"].failure_criterion == crit

    def test_tsai_wu_has_no_mode_table(self, ilss_by_criterion):
        rows, modes, caption = app.fe_failure_summary(ilss_by_criterion["tsai_wu"])
        values = {label: value for label, value, _ in rows}
        assert values["Criterion"] == "Tsai-Wu"
        assert values["Governing mode"] == "not separated (Tsai-Wu)"
        fe = ilss_by_criterion["tsai_wu"]["fe_field"]
        assert values["Max FI (domain)"] == f"{fe.max_failure_index:.4g}"
        assert modes == []
        assert "Pick Hashin" in caption

    def test_hashin_shows_delamination_onset(self, ilss_by_criterion):
        result = ilss_by_criterion["hashin"]
        fe = result["fe_field"]
        rows, modes, caption = app.fe_failure_summary(result)
        values = {label: value for label, value, _ in rows}
        governing = max(fe.failure_mode_maxima, key=fe.failure_mode_maxima.get)
        assert values["Governing mode"] == app._MODE_LABELS[governing]
        assert [m["Mode"] for m in modes] == list(app._MODE_LABELS.values())
        delam = next(m for m in modes if m["Mode"] == "Delamination onset")
        assert delam["Max over model"] == \
            f"{fe.failure_mode_maxima['delamination']:.4g}"
        assert "no explicit delamination" in caption

    def test_max_stress_has_no_delamination_row_value(self, ilss_by_criterion):
        rows, modes, caption = app.fe_failure_summary(ilss_by_criterion["max_stress"])
        delam = next(m for m in modes if m["Mode"] == "Delamination onset")
        assert delam["Max over model"] == "n/a"
        assert dict((label, value) for label, value, _ in rows)[
            "Governing mode"] != "Delamination onset"
        assert "no delamination mode" in caption

    def test_load_factor_row_and_thermal_runs(self, ilss_by_criterion, combined_results):
        rows, _, caption = app.fe_failure_summary(ilss_by_criterion["tsai_wu"])
        assert rows[-1][0] == "FPF load factor (interior)"
        assert "First-ply failure over the whole model at" in caption
        # With a thermal load its own block reports the load factors.
        comb, _ = combined_results
        rows, _, _ = app.fe_failure_summary(comb)
        assert all(label != "FPF load factor (interior)" for label, _, _ in rows)

    def test_none_without_an_fe_field(self, comp_result):
        assert app.fe_failure_summary({**comp_result, "fe_field": None}) is None

    def test_convergence_check_uses_the_criterion(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app, "refinement_study",
                            lambda m, p, ld, **kw: calls.append(kw) or "study")
        app.run_convergence_check(_base_cfg(failure_criterion="hashin"),
                                  [(4, 2, 4), (6, 3, 4)])
        assert calls[0]["failure_criterion"] == "hashin"


# ----------------------------------------------------------------------
# Defects: delamination, toughness, interface onset, fracture specimens
# (delamination plan, phase 5.1)
# ----------------------------------------------------------------------

#: Illustration toughness (N/mm), the IM7/8552 values of the theory page.
_TOUGHNESS = {"G_Ic": 0.212, "G_IIc": 0.774, "bk_eta": 2.1}


def _delam_cfg(**overrides) -> dict:
    """Tension on the small [0/90/90/0] coupon with a through-width
    delamination under the first ply, the toughness, the knockdown split and
    the interface onset."""
    cfg = _base_cfg(loading_mode="tension", nz=4,
                    delamination=app.Delamination.strip(0, 20.0, 30.0),
                    knockdown_split=True, onset=True, **_TOUGHNESS)
    cfg.update(overrides)
    return cfg


def _specimen_cfg(test: str, **overrides) -> dict:
    """A fracture specimen of four 0.75 mm UD plies (a 3 mm specimen keeps
    the graded mesh small), with the toughness."""
    cfg = _base_cfg(loading_mode=test, angles=[0.0] * 4, t_ply=0.75, nz=4,
                    **_TOUGHNESS)
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def delam_result():
    return app.run_analysis(_delam_cfg())


@pytest.fixture(scope="module")
def specimen_results():
    return {test: app.run_analysis(_specimen_cfg(test, mmb_mode_ratio=0.8)
                                   if test == "mmb" else _specimen_cfg(test))
            for test in ("dcb", "enf", "mmb")}


class TestDefectConfigKey:
    def test_default_key_is_unchanged(self):
        """A config without the defect fields, or with each at its default,
        has the key of before: one entry per _CFG_KEYS field."""
        key = app._config_to_key(_base_cfg())
        assert tuple(k for k, _ in key) == app._CFG_KEYS
        explicit = _base_cfg(**{k: app._CFG_DEFAULTS[k] for k in app._CFG_OPTIONAL_KEYS})
        assert app._config_to_key(explicit) == key

    @pytest.mark.parametrize("field, value", [
        ("delamination", app.Delamination.ellipse(1, (25.0, 10.0), (5.0, 3.0))),
        ("knockdown_split", True), ("G_Ic", 0.2), ("G_IIc", 0.7), ("G_IIIc", 0.8),
        ("bk_eta", 2.0), ("toughness_law", "none"), ("toughness_b", [5.0, 12.0]),
        ("onset", True), ("onset_x_avg", 0.5), ("crack_length", 40.0),
        ("mmb_mode_ratio", 0.8),
    ])
    def test_each_defect_field_enters_the_key_when_set(self, field, value):
        key = app._config_to_key(_base_cfg(**{field: value}))
        assert key != app._config_to_key(_base_cfg())
        assert key[:len(app._CFG_KEYS)] == app._config_to_key(_base_cfg())
        entry = dict(key)[field]
        assert entry == (tuple(value) if isinstance(value, list) else value)
        assert hash(key)

    def test_the_cached_runner_restores_the_config(self, monkeypatch):
        """The key round-trips to the config run_analysis sees."""
        seen = {}
        monkeypatch.setattr(app, "run_analysis", lambda cfg: seen.update(cfg) or {})
        cfg = _delam_cfg(toughness_law="user", toughness_b=[5.0, 12.0])
        app.run_analysis_cached.__wrapped__(app._config_to_key(cfg))
        assert seen["delamination"] == cfg["delamination"]
        assert seen["toughness_b"] == [5.0, 12.0]
        assert seen["G_Ic"] == 0.212 and seen["onset"] is True


class TestDefectAnalysis:
    def test_default_run_has_no_defect_results(self, comp_result):
        for k in ("delamination", "fe_decomposition", "fe_decomposition_error",
                  "fe_decomposition_time_s", "fe_onset", "fe_onset_error"):
            assert comp_result[k] is None
        assert comp_result["fe_field"].crack_front is None
        assert app.fe_delamination_summary(comp_result) is None
        assert app.fe_decomposition_summary(comp_result) is None
        assert app.fe_onset_summary(comp_result) is None
        assert app.fe_fracture_test_summary(comp_result) is None
        assert app.fe_front_rows(comp_result) == []
        assert not app.ncr_delamination_available(comp_result)

    def test_default_fields_do_not_change_the_solve(self, comp_result):
        """The same FE numbers with every defect field at its default."""
        explicit = app.run_analysis(_base_cfg(
            **{k: app._CFG_DEFAULTS[k] for k in app._CFG_OPTIONAL_KEYS}))
        a, b = comp_result["fe_field"], explicit["fe_field"]
        assert a.knockdown == b.knockdown
        assert a.max_failure_index == b.max_failure_index
        np.testing.assert_array_equal(a.stress_global, b.stress_global)
        assert comp_result["material"] == explicit["material"]

    def test_delaminated_coupon(self, delam_result):
        r = delam_result
        fe, mesh = r["fe_field"], r["mesh"]
        assert r["delamination"] == r["config"]["delamination"]
        assert mesh.delaminations == (r["delamination"],)
        assert r["material"].has_toughness and r["material"].G_Ic == 0.212
        front = fe.crack_front
        assert len(front) > 0 and fe.fracture_load_factor is not None
        assert front.toughness_uncalibrated is True          # the placeholders
        parts = r["fe_decomposition"]
        assert parts.total == fe.knockdown
        assert parts.total < parts.porosity_only < 1.0
        assert r["fe_decomposition_time_s"] > 0.0
        onset = r["fe_onset"]
        assert onset is not None and onset.x_avg == pytest.approx(0.183)
        # The empirical results do not see the delamination.
        plain = app.run_analysis(_base_cfg(loading_mode="tension"))
        assert r["empirical"] == plain["empirical"]

    def test_summaries(self, delam_result):
        rows, caption, uncalibrated = app.fe_delamination_summary(delam_result)
        fe = delam_result["fe_field"]
        values = {label: value for label, value, _ in rows}
        assert values["Fracture load factor λ_c"] == f"{fe.fracture_load_factor:.3g}"
        assert "Governing node" in values and uncalibrated is True
        assert "Virtual crack closure" in caption and "staircase" in caption
        table = app.fe_front_rows(delam_result)
        assert len(table) == len(fe.crack_front)
        assert set(table[0]) == {"x (mm)", "y (mm)", "G_I (N/mm)", "G_II (N/mm)",
                                 "G_III (N/mm)", "G_T (N/mm)", "ψ", "Vp (%)",
                                 "G_c (N/mm)", "λ_c"}
        assert {row["Vp (%)"] for row in table} == {"3.00"}
        rows, caption = app.fe_decomposition_summary(delam_result)
        assert [label for label, _, _ in rows] == [
            "KD total", "KD porosity only", "KD delamination only",
            "Larger stiffness loss"]
        rows, table, caption = app.fe_onset_summary(delam_result)
        assert [label for label, _, _ in rows][:2] == ["Max onset index",
                                                      "Critical interface"]
        assert [t["Interface"] for t in table] == [0, 1, 2]
        assert "depend on the mesh and on x_avg" in caption

    def test_front_order_and_plot(self, delam_result):
        front = delam_result["fe_field"].crack_front
        segments, s, label = app.front_order(front)
        # A strip inside the plate has two straight fronts across the width.
        assert len(segments) == 2 and label.startswith("y along")
        assert sorted(np.concatenate(segments).tolist()) == list(range(len(front)))
        fig = app.plot_front(delam_result)
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "Mode mixity ψ"
        plt.close(fig)

    def test_elliptical_front_is_ordered_by_angle(self):
        r = app.run_analysis(_base_cfg(
            nz=4, delamination=app.Delamination.ellipse(1, (25.0, 10.0), (12.0, 6.0))))
        segments, s, label = app.front_order(r["fe_field"].crack_front)
        assert len(segments) == 1 and label.startswith("Angle")
        assert np.all(np.diff(s[segments[0]]) >= 0)
        # No toughness: G only, the caption says so, no NCR margin.
        rows, caption, uncalibrated = app.fe_delamination_summary(r)
        assert dict((lab, v) for lab, v, _ in rows)["Fracture load factor λ_c"] == "n/a"
        assert "no material preset carries them" in caption
        assert uncalibrated is False
        assert not app.ncr_delamination_available(r)

    def test_toughness_laws(self):
        user = app.run_analysis(_delam_cfg(toughness_law="user", toughness_b=[5.0, 12.0],
                                           knockdown_split=False, onset=False))
        front = user["fe_field"].crack_front
        assert front.toughness_uncalibrated is False
        assert front.toughness_knockdown.coefficient("I") == 5.0
        none = app.run_analysis(_delam_cfg(toughness_law="none", knockdown_split=False,
                                           onset=False))
        assert none["fe_field"].crack_front.toughness_knockdown.law == "none"
        assert none["fe_field"].fracture_load_factor > \
            user["fe_field"].fracture_load_factor
        with pytest.raises(ValueError, match="needs toughness_b"):
            app.run_analysis(_delam_cfg(toughness_law="user"))
        with pytest.raises(ValueError, match="Unknown toughness_law"):
            app.run_analysis(_delam_cfg(toughness_law="linear"))
        with pytest.raises(ValueError, match="must be a Delamination"):
            app.run_analysis(_base_cfg(delamination=("strip", 0, 20.0, 30.0)))

    def test_unresolved_layup_is_skipped_with_the_nz_fix(self):
        r = app.run_analysis(_delam_cfg(nz=6))
        assert r["fe_field"] is None and r["fe_nz_suggestion"] == 8
        assert r["fe_skipped_reason"].startswith(
            "the delaminated coupon mesh could not be built")
        assert r["fe_decomposition"] is None and r["fe_onset"] is None
        assert r["empirical"]

    def test_delamination_on_the_short_beam(self):
        r = app.run_analysis(_base_cfg(loading_mode="ilss", nz=4,
                                       delamination=app.Delamination.strip(1, 0.5, 2.0),
                                       knockdown_split=True))
        assert r["mesh"].ilss_geometry is not None and r["mesh"].delaminations
        parts = r["fe_decomposition"]
        assert parts.dominant == "delamination" and parts.total < parts.delamination_only

    def test_onset_alone_on_a_default_solve(self, comp_result):
        r = app.run_analysis(_base_cfg(onset=True, onset_x_avg=0.0))
        assert r["fe_onset"].x_avg == 0.0
        assert r["fe_field"].knockdown == comp_result["fe_field"].knockdown
        assert b'Name="onset_index"' in app.fe_vtu_bytes(r)
        assert b'Name="onset_index"' not in app.fe_vtu_bytes(comp_result)
        # Not every interface is a node layer with nz = 3: the reason is kept.
        unresolved = app.run_analysis(_base_cfg(onset=True, nz=3))
        assert unresolved["fe_onset"] is None
        assert "node layer" in unresolved["fe_onset_error"]
        rows, table, caption = app.fe_onset_summary(unresolved)
        assert rows == [] and "not evaluated" in caption

    def test_convergence_check_is_not_offered(self, delam_result, specimen_results):
        assert "delamination" in app.convergence_unavailable_reason(delam_result)
        assert app.convergence_unavailable_reason(specimen_results["dcb"]) is not None


class TestFractureSpecimens:
    def test_specimens_are_built_and_solved(self, specimen_results):
        for test, r in specimen_results.items():
            spec, fe = r["fe_specimen"], r["fe_field"]
            assert r["fe_loading"] == test and r["mesh"].fracture_geometry.test == test
            assert spec["standard"] == app._FRACTURE_MODES[test][0]
            assert spec["crack_length_mm"] == app._FRACTURE_MODES[test][2]
            assert spec["thickness_mm"] == pytest.approx(3.0)
            assert fe.load_point_compliance > 0.0 and fe.fracture_load_factor > 0.0
            assert r["delamination"] is None and r["fe_decomposition"] is None

    def test_mmb_lever_from_the_mode_ratio(self, specimen_results):
        r = specimen_results["mmb"]
        spec = r["fe_specimen"]
        assert spec["mode_ratio_target"] == 0.8
        assert spec["lever_length_mm"] == pytest.approx(app.mmb_lever_length(
            r["material"], 0.8, crack_length=spec["crack_length_mm"]))

    def test_summary_rows(self, specimen_results):
        r = specimen_results["dcb"]
        rows, caption = app.fe_fracture_test_summary(r)
        values = {label: value for label, value, _ in rows}
        front = r["fe_field"].crack_front
        mid = int(np.argmin(np.abs(front.y - 12.5)))
        assert values["P_c at mid-width"] == f"{front.lambda_c[mid] * 10.0:.4g} N"
        assert values["P_c, weakest node"] == \
            f"{r['fe_field'].fracture_load_factor * 10.0:.4g} N"
        assert values["Compliance C"] == \
            f"{r['fe_field'].load_point_compliance:.4g} mm/N"
        assert "ASTM D5528" in caption and "unidirectional" in caption
        # The mode mix at mid-width: mode I for the DCB, mode II for the ENF.
        assert float(values["ψ at mid-width"]) < 0.01
        enf = dict((lab, v) for lab, v, _ in
                   app.fe_fracture_test_summary(specimen_results["enf"])[0])
        assert float(enf["ψ at mid-width"]) > 0.95
        assert "Lever c =" in app.fe_fracture_test_summary(specimen_results["mmb"])[1]

    def test_crack_length_and_bad_nz(self):
        r = app.run_analysis(_specimen_cfg("enf", crack_length=35.0))
        assert r["fe_specimen"]["crack_length_mm"] == 35.0
        bad = app.run_analysis(_specimen_cfg("dcb", nz=6))
        assert bad["fe_field"] is None and bad["fe_nz_suggestion"] == 8
        assert bad["fe_skipped_reason"].startswith("the dcb specimen mesh could not")

    def test_plots_and_stress_default(self, specimen_results):
        r = specimen_results["dcb"]
        fig = app.plot_results(r, "[0]_4")
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert not any(lab.startswith("FE Stiffness") for lab in labels)
        plt.close(fig)
        assert app.default_stress_component(r) == "σ₃₃ (through-thickness)"
        assert app.default_stress_component(specimen_results["enf"]) == \
            "τ₁₃ (interlaminar)"
        fig = app.plot_mesh(r)
        from matplotlib.text import Text
        texts = [t.get_text() for t in fig.axes[0].findobj(Text)]
        assert any("delamination in red" in t for t in texts)
        red = [ln for ln in fig.axes[0].get_lines() if ln.get_color() == "red"]
        insert = r["fe_specimen"]["crack_length_mm"]
        assert red and max(ln.get_xdata().max() for ln in red) > insert - 1e-9
        plt.close(fig)

    def test_toughness_less_specimen_reports_g_only(self):
        r = app.run_analysis(_base_cfg(loading_mode="dcb", angles=[0.0] * 4,
                                       t_ply=0.75, nz=4))
        values = dict((lab, v) for lab, v, _ in app.fe_fracture_test_summary(r)[0])
        assert values["P_c at mid-width"] == "n/a"
        assert values["G_T at mid-width"].endswith("N/mm")
