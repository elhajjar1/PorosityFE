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
def comp_result():
    """One compression analysis result, reused across the plot tests so the
    FE solve only runs once."""
    return app.run_analysis(_base_cfg())


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


class TestRunAnalysis:
    def test_compression_returns_full_result(self, comp_result):
        for k in ("config", "material", "porosity_field", "mesh",
                  "empirical", "fe_field", "fe_loading",
                  "fe_skipped_reason", "f_md"):
            assert k in comp_result
        assert comp_result["fe_field"] is not None
        assert comp_result["fe_loading"] == "compression"
        assert isinstance(comp_result["f_md"], float)
        # Empirical table carries the four loading modes.
        for mode in ("compression", "tension", "shear", "ilss"):
            assert mode in comp_result["empirical"]

    def test_ilss_takes_force_controlled_branch(self):
        """``loading_mode='ilss'`` routes through the short-beam-shear
        (force-controlled) solve rather than the applied-strain branch."""
        r = app.run_analysis(_base_cfg(loading_mode="ilss"))
        assert r["fe_loading"] == "ilss"
        assert r["fe_field"] is not None

    def test_clustered_distribution_branch(self):
        """A clustered distribution forwards ``cluster_location`` into the
        PorosityField (the conditional pf_kwargs branch)."""
        r = app.run_analysis(_base_cfg(distribution="clustered"))
        assert r["porosity_field"].distribution == "clustered"
        assert r["fe_field"] is not None

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown material"):
            app.run_analysis(_base_cfg(material_name="unobtainium"))

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

    def test_plot_results_fiber_dominated_footnote(self, comp_result):
        """A low matrix-dominated fraction (f_md < 0.49) adds the
        layup-scaling footnote to the knockdown chart."""
        skewed = dict(comp_result)
        skewed["f_md"] = 0.3
        fig = app.plot_results(skewed, "0/0/0/0")
        assert fig is not None
        plt.close(fig)

    def test_plot_stress_component(self, comp_result):
        fig = app.plot_stress(comp_result, "σ₁₁ (fiber)")
        assert fig is not None
        plt.close(fig)

    def test_plot_stress_von_mises(self, comp_result):
        fig = app.plot_stress(comp_result, "Von Mises")
        assert fig is not None
        plt.close(fig)

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
