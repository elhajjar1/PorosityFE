"""The app's hybrid laminate panel (headless)."""

import pytest

pytest.importorskip("streamlit")

import matplotlib

matplotlib.use("Agg")

from porosity_app import plots as app_plots
from porosity_app import runners as app_runners
from porosity_app import summaries as app_summaries
from porosity_fe import MATERIALS, compute_degraded_clt_moduli

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")


def _cfg(**overrides):
    cfg = {"material_name": "T700_epoxy", "angles": [0.0, 90.0, 90.0, 0.0], "n_plies": 4,
           "t_ply": 0.2, "Vp": 3.0, "distribution": "uniform",
           "cluster_location": "midplane", "void_shape": "spherical",
           "loading_mode": "tension", "nx": 8, "ny": 4, "nz": 4}
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def result():
    return app_runners.run_analysis(_cfg())


@pytest.fixture(scope="module")
def out(result):
    return app_runners.run_hybrid(result, "glass_epoxy@0.2, T700_epoxy*2@0.2, glass_epoxy@0.2")


def test_default_spec(result):
    assert app_runners.hybrid_default_spec(result) == \
        "glass_epoxy, T700_epoxy*2, glass_epoxy"
    glass = app_runners.run_analysis(_cfg(material_name="glass_epoxy", Vp=0.0))
    assert app_runners.hybrid_default_spec(glass) == "glass_epoxy*4"


def test_the_hybrid_is_the_analysis_coupon(result, out):
    stack, fe, mesh = out["stack"], out["fe"], out["mesh"]
    assert stack.n_plies == 4 and stack.is_hybrid
    assert (mesh.nx, mesh.ny, mesh.nz) == (result["mesh"].nx, result["mesh"].ny, 4)
    assert mesh.element_layup.tolist() == [0.0, 90.0, 90.0, 0.0]
    assert fe.knockdown < 1.0                                       # 3 % voids
    assert out["clt_modulus"] == pytest.approx(
        compute_degraded_clt_moduli(stack, [0, 90, 90, 0], 0.03)["Ex"])
    assert fe.effective_modulus == pytest.approx(out["clt_modulus"], rel=0.02)
    cached = app_runners.run_hybrid_cached(("hybrid-test",), "glass_epoxy, T700_epoxy*2, "
                                           "glass_epoxy", result)
    assert cached["stack"].n_plies == 4


def test_a_custom_material_by_its_name():
    import dataclasses
    custom = dataclasses.replace(MATERIALS["T700_epoxy"], E11=1.0e5, n_plies=4, t_ply=0.2)
    result = app_runners.run_analysis(_cfg(material_name="my carbon",
                                           material=custom.to_dict()))
    assert app_runners.hybrid_default_spec(result) == "my carbon*4"
    out = app_runners.run_hybrid(result, "glass_epoxy@0.2, my carbon*2@0.2, glass_epoxy@0.2")
    assert out["stack"].materials[1].E11 == 1.0e5


def test_shear_has_no_modulus_reference():
    result = app_runners.run_analysis(_cfg(loading_mode="shear"))
    out = app_runners.run_hybrid(result, "glass_epoxy, T700_epoxy*2, glass_epoxy")
    assert out["clt_modulus"] is None


def test_residual_stress_carries_over():
    result = app_runners.run_analysis(_cfg(material_name="AS4_3501_6_epoxy",
                                           thermal_mode="combined", delta_T=-100.0))
    out = app_runners.run_hybrid(result, "AS4_3501_6_epoxy*4")
    assert out["fe"].load_factor_basis == "mechanical_with_residual"
    with pytest.raises(ValueError, match="or a material of the PlyStack"):
        app_runners.run_hybrid(result, "glass_epoxy, AS4_3501_6_epoxy*2, glass_epoxy")


def test_rejects_a_longer_layup(result):
    with pytest.raises(ValueError, match="layup has 4 plies but the stack has 2"):
        app_runners.run_hybrid(result, "glass_epoxy, T700_epoxy")


@pytest.mark.parametrize("overrides, text", [
    ({"loading_mode": "ilss"}, "compression, tension or shear"),
    ({"thermal_mode": "thermal_only", "delta_T": -100.0,
      "material_name": "AS4_3501_6_epoxy"}, "thermal-only"),
])
def test_unavailable(overrides, text):
    assert text in app_runners.hybrid_unavailable_reason(
        app_runners.run_analysis(_cfg(**overrides)))


def test_unavailable_without_an_fe_solve_or_with_a_delamination(result):
    assert "skipped" in app_runners.hybrid_unavailable_reason({**result, "fe_field": None})
    assert app_runners.hybrid_unavailable_reason(result) is None

    class Delaminated:
        delaminations = ("d",)
    assert "without the delamination" in app_runners.hybrid_unavailable_reason(
        {**result, "mesh": Delaminated()})


def test_summary_and_plot(result, out):
    rows, caption = app_summaries.hybrid_summary(out, result["fe_field"])
    labels = [r[0] for r in rows]
    assert labels == ["Effective modulus", "Lamination theory", "Stiffness knockdown",
                      "First-ply failure load factor", "Highest failure index in"]
    assert "single-material laminate" in rows[0][2]
    assert rows[-1][1] in ("glass_epoxy", "T700_epoxy")
    assert "glass_epoxy@0.2, T700_epoxy*2@0.2, glass_epoxy@0.2" in caption
    rows, _ = app_summaries.hybrid_summary({**out, "clt_modulus": None}, None)
    assert [r[0] for r in rows][:2] == ["Effective modulus", "Stiffness knockdown"]
    z, fi, ids = app_summaries.hybrid_layer_failure(out)
    assert len(z) == 4 and ids.tolist() == [0, 1, 1, 0]
    fig = app_plots.plot_hybrid(out)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Largest failure index in the layer"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["glass_epoxy", "T700_epoxy"]
