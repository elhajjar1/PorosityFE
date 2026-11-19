"""The app's fatigue and environment table (headless)."""

import dataclasses

import pytest

pytest.importorskip("streamlit")

from porosity_app import runners as app_runners
from porosity_app.summaries import durability_table
from porosity_fe import MATERIALS, FatigueModel, build_empirical_pipeline


@pytest.fixture(scope="module")
def result():
    cfg = {"material_name": "T800_epoxy", "angles": [0.0, 90.0, 90.0, 0.0],
           "n_plies": 4, "t_ply": 0.183, "Vp": 3.0, "distribution": "uniform",
           "cluster_location": "midplane", "void_shape": "spherical",
           "loading_mode": "compression", "nx": 8, "ny": 4, "nz": 4}
    return app_runners.run_analysis(cfg)


def _rows(rows, mode, law="judd_wright"):
    return next(r for r in rows if r["Mode"] == mode and r["Law"] == law)


def test_fatigue_only(result):
    rows, notes, record = durability_table(result, cycles=1e6)
    assert len(rows) == 15 and notes == []
    row = _rows(rows, "tension")
    assert row["Fatigue"] == FatigueModel().knockdown_factor("tension", 1e6)
    assert row["Environment"] == 1.0
    assert row["Knockdown"] == pytest.approx(row["Porosity"] * row["Fatigue"])
    assert record["conditions"]["mean_stress"] is None


def test_goodman_ratios_per_mode_family(result):
    rows, _, record = durability_table(result, cycles=1e6, mean_stress="goodman",
                                       R=0.5, R_compression=2.0)
    model = FatigueModel(mean_stress="goodman")
    assert _rows(rows, "shear")["Fatigue"] == model.knockdown_factor("shear", 1e6, 0.5)
    assert _rows(rows, "compression")["Fatigue"] == \
        model.knockdown_factor("compression", 1e6, 2.0)
    assert record["conditions"]["R"]["compression"] == 2.0
    # Without the correction the ratios are not passed on.
    _, _, plain = durability_table(result, cycles=1e6, R=0.5, R_compression=2.0)
    assert plain["conditions"]["R"] is None


def test_porosity_slope(result):
    rows, _, _ = durability_table(result, cycles=1e6, porosity_slope=5.0)
    assert _rows(rows, "tension", "linear")["Fatigue"] == pytest.approx(
        1.0 - 0.10 * (1 + 5.0 * 0.03) * 6)


def test_environment_needs_a_glass_transition(result):
    with pytest.raises(ValueError, match="dry glass-transition temperature T_g"):
        durability_table(result, T=80.0)
    rows, _, record = durability_table(result, T=80.0, M=1.0, T_g_dry=180.0)
    material = dataclasses.replace(MATERIALS["T800_epoxy"], T_g_dry=180.0)
    assert _rows(rows, "shear")["Environment"] == pytest.approx(
        material.environment_knockdown("shear", T=80.0, M=1.0))
    assert _rows(rows, "tension")["Environment"] == 1.0
    assert record["conditions"]["T_g_dry_C"] == 180.0
    # Moisture alone takes the reference temperature.
    _, _, wet = durability_table(result, M=1.0, T_g_dry=180.0)
    assert (wet["conditions"]["T_C"], wet["conditions"]["M_wt_pct"]) == (23.0, 1.0)
    # Zero moisture and no temperature is no hygrothermal request.
    _, _, dry = durability_table(result, cycles=10.0, M=0.0)
    assert dry["conditions"]["T_C"] is None


def test_the_material_glass_transition_is_used(result):
    with_tg = {**result, "material": dataclasses.replace(result["material"], T_g_dry=170.0)}
    _, _, record = durability_table(with_tg, T=60.0)
    assert record["conditions"]["T_g_dry_C"] == 170.0


def test_it_matches_a_direct_solve(result):
    rows, _, _ = durability_table(result, cycles=1e4, T=70.0, T_g_dry=180.0)
    material = dataclasses.replace(result["material"], T_g_dry=180.0)
    solver = build_empirical_pipeline(material, 0.03, ply_angles=[0.0, 90.0, 90.0, 0.0])[2]
    direct = solver.get_failure_load("ilss", "power_law", cycles=1e4,
                                     environment={"T": 70.0, "M": 0.0})
    row = _rows(rows, "ilss", "power_law")
    assert row["Knockdown"] == pytest.approx(direct.knockdown, rel=1e-12)
    assert row["Strength (MPa)"] == pytest.approx(direct.failure_stress, rel=1e-12)


def test_warnings_become_notes(result):
    _, notes, _ = durability_table(result, cycles=1e12)
    assert notes and all("clamping" in n for n in notes)
