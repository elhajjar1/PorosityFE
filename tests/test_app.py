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
