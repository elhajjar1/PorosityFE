"""The app's delamination growth panel (headless)."""

import math

import numpy as np
import pytest

pytest.importorskip("streamlit")

import matplotlib

matplotlib.use("Agg")

from porosity_app import plots as app_plots
from porosity_app import runners as app_runners
from porosity_app import summaries as app_summaries
from porosity_fe import Delamination, ParisLaw

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")

#: Illustration toughness (N/mm), the IM7/8552 values of the theory page.
_TOUGHNESS = {"G_Ic": 0.212, "G_IIc": 0.774, "bk_eta": 2.1}


def _cfg(**overrides):
    cfg = {"material_name": "T800_epoxy", "angles": [0.0] * 4, "n_plies": 4,
           "t_ply": 0.75, "Vp": 2.0, "distribution": "uniform",
           "cluster_location": "midplane", "void_shape": "spherical",
           "loading_mode": "dcb", "nx": 10, "ny": 4, "nz": 4, **_TOUGHNESS}
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def dcb():
    return app_runners.run_analysis(_cfg())


@pytest.fixture(scope="module")
def growth(dcb):
    return app_runners.run_growth(dcb, max_steps=3)


def test_quasi_static_growth_of_the_specimen(dcb, growth):
    assert app_runners.growth_unavailable_reason(dcb) is None
    assert growth.n_steps == 3 and growth.loading == "dcb"
    assert growth.load_factors[0] == pytest.approx(dcb["fe_field"].fracture_load_factor,
                                                   rel=1e-9)
    assert growth.crack_lengths[0] == pytest.approx(
        dcb["mesh"].fracture_geometry.crack_length)
    assert np.all(growth.critical_load > 0.0)     # the DCB's magnitude
    assert growth.toughness_uncalibrated          # 2 % voids, placeholder law


def test_fatigue_growth_at_a_fraction_of_the_static_load(dcb):
    fatigue = app_runners.run_growth(dcb, max_steps=2, paris=ParisLaw(0.05, 6.0),
                                     peak=0.5)
    assert fatigue.load_factors[0] == pytest.approx(2.0, rel=1e-9)
    assert fatigue.cycles[0] == 0.0 and fatigue.cycles[-1] > 0.0
    cached = app_runners.run_growth_cached(("dcb-test",), (2, "fatigue", 0.05, 6.0, 0.0, 0.5),
                                           dcb)
    np.testing.assert_allclose(cached.cycles, fatigue.cycles, rtol=1e-12)
    quasi = app_runners.run_growth_cached(("dcb-test",), (2, "quasi-static", None, None,
                                                          0.0, None), dcb)
    assert quasi.paris is None and quasi.n_steps == 2
    with pytest.raises(ValueError, match="cyclic peak"):
        app_runners.run_growth(dcb, paris=ParisLaw(0.05, 6.0))


def test_coupon_with_a_residual_stress():
    """A delaminated coupon in tension, with the cure residual stress: the
    growth solves carry the same delta_T."""
    result = app_runners.run_analysis(_cfg(
        material_name="AS4_3501_6_epoxy", loading_mode="tension",
        angles=[0.0, 90.0, 90.0, 0.0], t_ply=0.25,
        delamination=Delamination.strip(0, 20.0, 30.0),
        thermal_mode="combined", delta_T=-100.0))
    growth = app_runners.run_growth(result, max_steps=1)
    assert growth.load_factors[0] == pytest.approx(
        result["fe_field"].fracture_load_factor, rel=1e-9)
    assert growth.critical_strain is not None


def test_unavailable(dcb):
    assert app_runners.growth_unavailable_reason({**dcb, "fe_field": None}) \
        .startswith("The FE solve of this analysis was skipped")
    assert "thermal-only" in app_runners.growth_unavailable_reason(
        {**dcb, "fe_thermal": {"mode": "thermal_only", "delta_T": -100.0}})
    plain = app_runners.run_analysis(_cfg(loading_mode="compression",
                                          angles=[0.0, 90.0, 90.0, 0.0], t_ply=0.25))
    assert "needs a delamination" in app_runners.growth_unavailable_reason(plain)
    no_toughness = app_runners.run_analysis(_cfg(G_Ic=None, G_IIc=None, bk_eta=None))
    assert "interlaminar toughness" in app_runners.growth_unavailable_reason(no_toughness)


def test_summary_and_table(growth):
    rows, caption = app_summaries.growth_summary(growth)
    labels = [r[0] for r in rows]
    assert labels[:3] == ["Release steps", "Crack area grown", "Crack length"]
    assert "Unstable steps (displacement control)" in labels
    assert "the number of steps was reached" in caption
    assert "uncalibrated placeholder" in caption
    table = app_summaries.growth_table(growth)
    assert len(table) == 4 and table[0]["Step"] == 0
    assert {"Crack length (mm)", "λ_c", "Critical load (N)",
            "Critical displacement (mm)", "Vp at front (%)"} <= set(table[0])
    assert table[-1]["Released"] == 0


def test_fatigue_summary_and_table(dcb):
    fatigue = app_runners.run_growth(dcb, max_steps=2,
                                     paris=ParisLaw(0.05, 6.0, threshold=0.1), peak=0.5)
    rows, caption = app_summaries.growth_summary(fatigue)
    labels = [r[0] for r in rows]
    assert "Cycles" in labels and "Largest da/dN" in labels
    assert "C = 0.05 mm/cycle, m = 6, threshold 0.1" in caption
    assert "Cycles" in app_summaries.growth_table(fatigue)[0]
    fig = app_plots.plot_growth(fatigue)
    assert fig.axes[0].get_xlabel() == "Cycles"


def test_strain_controlled_table_and_undriven_rows(growth):
    import dataclasses
    strain = dataclasses.replace(
        growth, applied_load=None, applied_strain=0.01, compliance=None, crack_lengths=None,
        load_factors=np.array([math.inf, 2.0, 2.5, 3.0]), Vp=np.full(4, math.nan),
        effective_modulus=np.full(4, 50000.0))
    table = app_summaries.growth_table(strain)
    assert table[0]["λ_c"] == "∞" and table[0]["Critical strain"] == "∞"
    assert table[0]["Vp at front (%)"] == "n/a"
    assert "Crack length (mm)" not in table[0]
    rows, _ = app_summaries.growth_summary(strain)
    assert "Unstable steps (displacement control)" in [r[0] for r in rows]
    fig = app_plots.plot_growth(strain)
    assert fig.axes[0].get_ylabel() == "Critical strain"
    assert fig.axes[0].get_xlabel() == "Crack area grown (mm²)"


def test_plot(growth):
    fig = app_plots.plot_growth(growth)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Crack length (mm)"
    assert ax.get_ylabel() == "Critical load (N)"
    assert len(ax.lines) == 2                     # the levels and the envelope
