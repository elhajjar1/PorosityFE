"""The app's linear buckling panel (headless)."""

import pytest

pytest.importorskip("streamlit")

from porosity_app import runners as app_runners
from porosity_fe import Delamination

pytestmark = pytest.mark.filterwarnings("ignore:Mesh has .* highly distorted")


def _cfg(**overrides):
    cfg = {"material_name": "AS4_3501_6_epoxy", "angles": [0.0, 90.0, 90.0, 0.0],
           "n_plies": 4, "t_ply": 0.25, "Vp": 3.0, "distribution": "clustered",
           "cluster_location": "surface", "void_shape": "spherical",
           "loading_mode": "compression", "nx": 20, "ny": 4, "nz": 4}
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def result():
    return app_runners.run_analysis(_cfg())


def test_porous_against_void_free(result):
    out = app_runners.run_buckling(result)
    buckled, reference = out["result"], out["reference"]
    assert len(buckled.load_factors) == app_runners._BUCKLING_N_MODES
    assert 0.9 < buckled.knockdown(reference) < 1.0
    assert reference.prebuckling.knockdown == 1.0


def test_the_plate_and_delamination_carry_over():
    result = app_runners.run_analysis(_cfg(length=60.0, width=24.0, nz=4,
                                           delamination=Delamination.strip(0, 20.0, 40.0)))
    out = app_runners.run_buckling(result, n_modes=1)
    assert out["result"].interpenetration is not None
    assert out["reference"].interpenetration is not None
    mesh = result["mesh"]
    assert (mesh.L_x, mesh.L_y) == (60.0, 24.0)


@pytest.mark.parametrize("overrides, start", [
    ({"loading_mode": "tension"}, "Linear buckling analyses the coupon"),
    ({"loading_mode": "ilss"}, "Linear buckling analyses the coupon"),
])
def test_unavailable(overrides, start):
    assert app_runners.buckling_unavailable_reason(
        app_runners.run_analysis(_cfg(**overrides))).startswith(start)


def test_unavailable_without_an_fe_field_or_with_a_residual_stress(result):
    assert app_runners.buckling_unavailable_reason({**result, "fe_field": None}) \
        .startswith("The FE solve of this analysis was skipped")
    assert "thermal residual stress" in app_runners.buckling_unavailable_reason(
        {**result, "fe_thermal": {"mode": "combined", "delta_T": -100.0}})
    assert app_runners.buckling_unavailable_reason(result) is None
