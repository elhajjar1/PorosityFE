"""Streamlit web UI for PorosityFE.

Wraps the porosity_fe_analysis library so engineers can drive analyses
from a browser via Streamlit Community Cloud. The PyQt6 desktop GUI in
porosity_gui.py remains the local entry point; this file is the
hosted-web parallel.

Run locally with:
    streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from porosity_fe_analysis import (
    MATERIALS,
    PorosityField,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
)
from porosity_gui import parse_layup  # pure helper; PyQt6 import is guarded


st.set_page_config(
    page_title="PorosityFE",
    page_icon="🔬",
    layout="wide",
)

st.title("PorosityFE — Composite Porosity Knockdown")
st.caption(
    "Predicts how distributed and discrete voids degrade strength and "
    "stiffness in fiber-reinforced composite laminates. Calibrated against "
    "13 peer-reviewed datasets (Elhajjar 2025, Sci. Rep. 15:25977)."
)

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Material")
    material_name = st.selectbox(
        "Preset",
        list(MATERIALS.keys()),
        help="Override the ply count / thickness below if your laminate differs.",
    )
    n_plies = st.number_input(
        "Number of plies", min_value=1, max_value=128, value=8, step=1,
    )
    t_ply = st.number_input(
        "Ply thickness (mm)",
        min_value=0.05, max_value=1.0, value=0.183, step=0.01, format="%.3f",
    )

    st.header("Layup")
    layup_text = st.text_input(
        "Stacking sequence",
        value="[0/45/-45/90]_3s",
        help=(
            "Slash- or comma-separated angles in degrees. Suffix `_<int>` for "
            "repeats, trailing `s` for symmetric. Examples: `[0]_8`, "
            "`[0/45/-45/90]_3s`, `[90, 0, 90]`."
        ),
    )

    st.header("Porosity")
    Vp_pct = st.number_input(
        "Void volume fraction Vp (%)",
        min_value=0.0, max_value=20.0, value=3.0, step=0.1,
        help="Percent here; converted internally to a fraction (3% → 0.03).",
    )
    distribution = st.selectbox(
        "Distribution", ["uniform", "clustered", "interface"],
    )
    cluster_location = "midplane"
    if distribution == "clustered":
        cluster_location = st.selectbox(
            "Cluster location", ["midplane", "surface", "quarter"],
        )
    void_shape = st.selectbox(
        "Void shape", ["spherical", "cylindrical", "penny"],
    )

    st.header("Loading")
    loading_mode = st.selectbox(
        "Loading mode", ["compression", "tension", "shear", "ilss"],
    )

    st.header("FE solver (optional)")
    run_fe = st.checkbox(
        "Run 3D FE analysis",
        value=False,
        help=(
            "Slow on Streamlit Cloud's shared CPU; minutes for moderate meshes. "
            "ILSS is empirical-only — FE BCs are not implemented (issue #15)."
        ),
    )
    if run_fe:
        c1, c2, c3 = st.columns(3)
        with c1:
            nx = st.number_input("nx", 4, 60, 20, step=2)
        with c2:
            ny = st.number_input("ny", 3, 30, 10, step=1)
        with c3:
            nz = st.number_input("nz", 4, 24, 8, step=1)
    else:
        nx, ny, nz = 10, 5, 6  # cheap mesh for the empirical-only path

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------- gate

if not run_btn:
    st.info("Configure inputs in the sidebar, then click **Run analysis**.")
    st.stop()

# ---------------------------------------------------------------- run

# Parse layup with the same helper the desktop GUI uses (#9)
try:
    angles = parse_layup(layup_text)
except ValueError as exc:
    st.error(f"Layup error: {exc}")
    st.stop()

# Apply ply-count / thickness overrides on top of the preset
material = dataclasses.replace(
    MATERIALS[material_name],
    n_plies=int(n_plies),
    t_ply=float(t_ply),
)

# PorosityField validates Vp at the boundary (#10)
pf_kwargs = {"distribution": distribution, "void_shape": void_shape}
if distribution == "clustered":
    pf_kwargs["cluster_location"] = cluster_location

try:
    porosity_field = PorosityField(material, Vp_pct / 100.0, **pf_kwargs)
except ValueError as exc:
    st.error(f"Porosity field error: {exc}")
    st.stop()

with st.spinner("Building mesh…"):
    mesh = CompositeMesh(
        porosity_field, material,
        nx=int(nx), ny=int(ny), nz=int(nz),
        ply_angles=angles,
    )

with st.spinner("Running empirical solver…"):
    empirical = EmpiricalSolver(mesh, material, ply_angles=angles)
    emp_results = empirical.get_all_failure_loads()

# Optional FE solve. Mirrors the GUI's #9 fix: skip FE for ILSS rather than
# silently substituting compression BCs.
fe_field = None
fe_loading = None
fe_skipped_reason = None
if run_fe:
    if loading_mode in ("compression", "tension", "shear"):
        with st.spinner(f"Running FE solver ({loading_mode}) — this can take a few minutes…"):
            fe_solver = FESolver(mesh, material, porosity_field, ply_angles=angles)
            applied_strain = -0.01 if loading_mode == "compression" else 0.01
            fe_field = fe_solver.solve(
                loading=loading_mode,
                applied_strain=applied_strain,
                verbose=False,
            )
            fe_loading = loading_mode
    else:
        fe_skipped_reason = (
            f"FE solver does not support '{loading_mode}' boundary conditions; "
            f"showing empirical results only."
        )

# ---------------------------------------------------------------- results

left, right = st.columns([2, 3])

with left:
    st.subheader("Empirical knockdown")
    st.caption(f"Layup f_md = {empirical.f_md:.3f}")

    table_rows = []
    for mode in ("compression", "tension", "shear", "ilss"):
        for model in ("judd_wright", "power_law", "linear"):
            r = emp_results[mode][model]
            table_rows.append({
                "mode": mode,
                "model": model,
                "knockdown": round(float(r["knockdown"]), 4),
                "failure_stress_MPa": round(float(r["failure_stress"]), 1),
            })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.subheader("FE result")
    if fe_field is not None:
        m1, m2 = st.columns(2)
        m1.metric(f"FE knockdown ({fe_loading})", f"{fe_field.knockdown:.3f}")
        m2.metric("Max Tsai-Wu FI", f"{fe_field.max_failure_index:.3f}")
    elif fe_skipped_reason:
        st.info(fe_skipped_reason)
    else:
        st.caption("FE solver was not enabled.")

with right:
    st.subheader("Through-thickness porosity profile")
    fig, ax = plt.subplots(figsize=(6, 4))
    z, vp_profile = porosity_field.effective_porosity_profile(nz=200)
    ax.plot(vp_profile * 100, z, lw=2)
    ax.set_xlabel("Vp (%)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"{distribution} / {void_shape}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader(f"Knockdown vs. Vp ({loading_mode} mode)")
    vp_sweep = np.linspace(0.0, 0.10, 25)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for model_name, model_attr in (
        ("Judd-Wright", "_judd_wright"),
        ("Power-Law", "_power_law"),
        ("Linear", "_linear"),
    ):
        kd = [getattr(empirical, model_attr)(vp, loading_mode) for vp in vp_sweep]
        ax2.plot(vp_sweep * 100, kd, lw=2, label=model_name)
    ax2.axvline(Vp_pct, ls="--", color="k", alpha=0.4, label=f"Vp = {Vp_pct}%")
    ax2.set_xlabel("Vp (%)")
    ax2.set_ylabel("Knockdown KD")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ---------------------------------------------------------------- export

st.divider()
export_payload = {
    "schema_version": "1.0",
    "exported_at": datetime.now(timezone.utc).isoformat(),
    "tool": "PorosityFE Streamlit",
    "inputs": {
        "material_name": material_name,
        "n_plies": int(n_plies),
        "t_ply_mm": float(t_ply),
        "layup_string": layup_text,
        "Vp_percent": float(Vp_pct),
        "distribution": distribution,
        "cluster_location": cluster_location if distribution == "clustered" else None,
        "void_shape": void_shape,
        "loading_mode": loading_mode,
        "fe_run": bool(run_fe),
    },
    "f_md": float(empirical.f_md),
    "empirical": {
        mode: {
            model: {
                "knockdown": float(r["knockdown"]),
                "failure_stress_MPa": float(r["failure_stress"]),
            }
            for model, r in models.items()
        }
        for mode, models in emp_results.items()
    },
    "fe": (
        {
            "loading_mode": fe_loading,
            "knockdown": float(fe_field.knockdown),
            "max_failure_index": float(fe_field.max_failure_index),
        }
        if fe_field is not None else None
    ),
}

st.download_button(
    label="Download results (JSON)",
    data=json.dumps(export_payload, indent=2).encode("utf-8"),
    file_name=f"porosity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)
