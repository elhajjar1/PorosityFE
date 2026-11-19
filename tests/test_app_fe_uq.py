"""The app's FE uncertainty panel and its sensitivity tables (headless)."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("streamlit")

import matplotlib

matplotlib.use("Agg")

from porosity_app import plots as app_plots
from porosity_app import runners as app_runners
from porosity_app import summaries as app_summaries
from porosity_fe import Delamination

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")

_TOUGHNESS = {"G_Ic": 0.212, "G_IIc": 0.774, "bk_eta": 2.1}


def _cfg(**overrides):
    cfg = {"material_name": "T800_epoxy", "angles": [0.0, 90.0, 90.0, 0.0], "n_plies": 4,
           "t_ply": 0.25, "Vp": 3.0, "distribution": "clustered",
           "cluster_location": "surface", "void_shape": "spherical",
           "loading_mode": "compression", "nx": 16, "ny": 8, "nz": 4}
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def result():
    return app_runners.run_analysis(_cfg())


@pytest.fixture(scope="module")
def study(result):
    return app_runners.run_fe_uq(result, 4, 0.6, 5.0)


def test_the_study_is_the_analysis_model(result, study):
    mesh = result["mesh"]
    assert study.mesh == (mesh.nx, mesh.ny, mesh.nz) and study.loading == "compression"
    assert study.Vp == pytest.approx(0.03)
    assert study.field["plate_mm"] == [50.0, 20.0]
    assert study.nominal["knockdown"] == pytest.approx(result["fe_field"].knockdown,
                                                       rel=1e-12)
    cached = app_runners.run_fe_uq_cached(("fe-uq-test",), (4, 0.6, 5.0), result)
    for k in study.outputs:
        np.testing.assert_array_equal(cached.samples[k], study.samples[k])


def test_delamination_and_residual_stress_carry_over():
    result = app_runners.run_analysis(_cfg(
        material_name="AS4_3501_6_epoxy", loading_mode="tension",
        delamination=Delamination.strip(0, 20.0, 30.0), thermal_mode="combined",
        delta_T=-100.0, **_TOUGHNESS))
    study = app_runners.run_fe_uq(result, 2, 0.5, 5.0)
    assert "fracture_load_factor" in study.outputs
    assert study.nominal["fracture_load_factor"] == pytest.approx(
        result["fe_field"].fracture_load_factor, rel=1e-9)


@pytest.mark.parametrize("overrides, text", [
    ({"loading_mode": "ilss"}, "compression, tension or shear"),
    ({"thermal_mode": "thermal_only", "delta_T": -100.0,
      "material_name": "AS4_3501_6_epoxy"}, "thermal-only"),
])
def test_unavailable(overrides, text):
    assert text in app_runners.fe_uq_unavailable_reason(
        app_runners.run_analysis(_cfg(**overrides)))


def test_unavailable_without_a_solve(result):
    assert app_runners.fe_uq_unavailable_reason({**result, "fe_field": None}) \
        .startswith("The FE solve of this analysis was skipped")
    assert app_runners.fe_uq_unavailable_reason(result) is None


def test_summary_plot_and_ranking_rows(study):
    rows, caption = app_summaries.fe_uq_summary(study)
    labels = [r[0] for r in rows]
    assert labels == ["First-ply failure p5", "p50", "p95", "Nominal (uniform in-plane)",
                      "Stiffness knockdown mean"]
    assert "CoV 0.6 from point to point" in caption and "uncalibrated" in caption
    fig = app_plots.plot_fe_uq(study)
    assert len(fig.axes) == 3                      # histogram, map, colorbar
    ranking = study.ranking("first_ply_failure_load_factor")
    table = app_summaries.sensitivity_rows(ranking)
    assert {row["Input"] for row in table} <= {"Peak local Vp of the map",
                                                "In-plane CoV of the map"}
    assert "R² =" in app_summaries.sensitivity_caption(ranking)
    assert app_summaries.sensitivity_rows(None) == []
    assert app_summaries.sensitivity_caption(None).startswith("Nothing to rank")


def test_summary_without_finite_draws(study):
    empty = dataclasses.replace(
        study, samples={**study.samples,
                        "first_ply_failure_load_factor": np.full(4, np.inf)},
        nominal={**study.nominal, "first_ply_failure_load_factor": None})
    rows, _ = app_summaries.fe_uq_summary(empty)
    assert rows[0][1] == "n/a" and rows[3][1] == "n/a"
    fig = app_plots.plot_fe_uq(empty)
    assert not fig.axes[0].get_legend()


def test_empirical_ranking_rows():
    record = {"n_samples": 50, "r2": 0.97, "ranking": [
        {"input": "Vp", "spearman": -0.9, "srrc": -0.87, "share": 0.82},
        {"input": "sigma_1c", "spearman": 0.1, "srrc": 0.01, "share": 0.0}]}
    rows = app_summaries.sensitivity_rows(record)
    assert rows[0] == {"Input": "Specimen-average Vp", "Share of the scatter": "82 %",
                       "Spearman ρ": "-0.90", "SRRC": "-0.87"}
    assert rows[1]["Input"] == "sigma_1c"
    assert "over 50 draws" in app_summaries.sensitivity_caption(record)
