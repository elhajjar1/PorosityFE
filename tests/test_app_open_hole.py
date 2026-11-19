"""The app's open-hole panel (headless)."""

import pytest

pytest.importorskip("streamlit")

import matplotlib

matplotlib.use("Agg")

from porosity_app import plots as app_plots
from porosity_app import runners as app_runners
from porosity_app import summaries as app_summaries

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")


def _cfg(**overrides):
    cfg = {"material_name": "T800_epoxy", "angles": [0.0, 90.0, 90.0, 0.0], "n_plies": 4,
           "t_ply": 0.25, "Vp": 3.0, "distribution": "clustered",
           "cluster_location": "surface", "void_shape": "spherical",
           "loading_mode": "tension", "nx": 10, "ny": 4, "nz": 4}
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def result():
    return app_runners.run_analysis(_cfg())


@pytest.fixture(scope="module")
def out(result):
    return app_runners.run_open_hole(result, 6.0, 36.0)


def test_the_coupon_is_the_analysis_laminate(result, out):
    a, fe = out["analysis"], out["fe"]
    g = a.geometry
    assert (g.hole_diameter, g.width, g.length) == (6.0, 36.0, 108.0)
    assert a.loading == "tension" and fe.knockdown < 1.0           # 3 % voids
    assert a.stress_concentration == pytest.approx(a.reference_stress_concentration,
                                                   rel=0.1)
    assert fe.first_ply_failure_load_factor < \
        result["fe_field"].first_ply_failure_load_factor           # the hole
    cached = app_runners.run_open_hole_cached(("oh-test",), (6.0, 36.0), result)
    assert cached["analysis"].stress_concentration == pytest.approx(
        a.stress_concentration, rel=1e-12)


def test_residual_stress_carries_over():
    result = app_runners.run_analysis(_cfg(material_name="AS4_3501_6_epoxy",
                                           thermal_mode="combined", delta_T=-100.0))
    out = app_runners.run_open_hole(result, 6.0, 36.0)
    assert out["fe"].load_factor_basis == "mechanical_with_residual"


@pytest.mark.parametrize("mode", ["shear", "ilss"])
def test_unavailable(mode):
    assert "compression or tension" in app_runners.open_hole_unavailable_reason(
        app_runners.run_analysis(_cfg(loading_mode=mode)))


def test_summary_and_plot(result, out):
    rows, caption = app_summaries.open_hole_summary(out, result["fe_field"])
    labels = [r[0] for r in rows]
    assert labels == ["K_T (FE)", "K_T (Lekhnitskii-Tan)", "First-ply failure load factor"]
    assert "without the hole" in rows[2][2] and "none ships" in caption
    rows, _ = app_summaries.open_hole_summary(out, None, d0=1.0, unnotched_strength=600.0)
    values = {r[0]: r[1] for r in rows}
    factor = out["analysis"].point_stress_factor(1.0)
    assert values["Point-stress factor at 1 mm"] == f"{factor:.4f}"
    assert values["Notched strength"] == f"{600.0 * factor:.4g} MPa"
    fig = app_plots.plot_open_hole(out["analysis"], 1.0)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "σ_xx / σ_far field" and len(ax.lines) == 4
    assert len(app_plots.plot_open_hole(out["analysis"]).axes[0].lines) == 3
