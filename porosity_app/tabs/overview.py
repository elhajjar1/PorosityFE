"""The Overview tab.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import streamlit as st

from ..config import _README_URL, calibration_note
from ..sidebar import _offer_nz_fix
from ..summaries import _crack_front, delamination_text


def _build_overview_tab(result: dict | None, layup_for_title: str):
    st.markdown(
        """
        **PorosityFE** estimates strength and stiffness knockdown in
        porosity-degraded composite laminates using empirical
        models (Judd–Wright, power law, linear) and a 3D hex finite-element
        solve. Configure the laminate and porosity field in the sidebar and
        press **Run analysis**.

        The empirical knockdowns cover compression, tension, shear and ILSS
        (all four, whichever mode is selected). The selected loading mode
        sets the FE solve: ILSS runs on the ASTM D2344 short-beam specimen,
        and the FE-only flexure modes on the ASTM D7264 three- or
        four-point specimen, with the FE flexural modulus shown next to the
        lamination-theory value. The FE-only DCB, ENF and MMB modes solve
        the ASTM D5528, D7905 and D6671 delamination fracture specimens for
        test planning. An explicit delamination (an NDI indication) can be
        added to the FE solve, with the interlaminar toughness you enter.

        - **Profile** — through-thickness porosity distribution
        - **Mesh** — mid-y cross-section of the FE mesh, coloured by the
          solid fraction 1 − Vp (not a stiffness ratio: the FE degrades the
          stiffness by Mori-Tanaka, which depends on the void shape; thin
          sections are drawn with the thickness exaggerated, and the axis
          label gives the factor)
        - **Results** — empirical knockdown bar chart with the FE stiffness
          knockdown overlaid, the short-beam, flexure and thermal
          residual-stress results of the FE solve, the defect results
          (delamination front, fracture load factor, knockdown by defect,
          interface onset, fracture specimens) when requested, an
          uncertainty analysis and an optional mesh-convergence check
        - **Stress** — FE stress in the ply axes for a chosen component,
          opening on the one that governs the load case
        - **Export** — download the empirical knockdown sweep as JSON or
          CSV, the FE field as a VTU file for ParaView, or generate an NCR
          validation summary (PDF / Markdown / JSON) with a recommended MRB
          disposition path
        """
    )
    if result is None:
        st.info("No results yet. Adjust the sidebar and press **Run analysis**.")
        return
    cfg_r = result["config"]
    thermal = result.get("fe_thermal")
    thermal_note = ""
    if thermal:
        thermal_note = (
            f" FE thermal load: ΔT = {thermal['delta_T']:g} K, "
            + ("thermal only." if thermal["mode"] == "thermal_only"
               else f"residual stress added to {cfg_r['loading_mode']}."))
    spec = result.get("fe_specimen")
    mesh_note = (f"mesh {cfg_r['nx']}×{cfg_r['ny']}×{cfg_r['nz']}" if spec is None
                 else f"FE on the {spec['standard']} {spec['specimen']} "
                      f"({spec['length_mm']:.1f} × {spec['width_mm']:.1f} × "
                      f"{spec['thickness_mm']:.2f} mm, mesh "
                      f"{spec['mesh'].replace('x', '×')})")
    delamination = result.get("delamination")
    delamination_note = ("" if delamination is None else
                         f" Delamination: {delamination_text(delamination)}.")
    st.success(
        f"Last run: {cfg_r['material_name']}, layup {layup_for_title}, "
        f"Vp = {cfg_r['Vp']:.1f}%, {cfg_r['void_shape']}, "
        f"{cfg_r['distribution']}, {mesh_note}."
        + thermal_note + delamination_note
    )
    extrapolated = calibration_note(cfg_r["Vp"])
    if extrapolated:
        st.warning(f"⚠ {extrapolated}")
    front = _crack_front(result)
    if front is not None and front.toughness_uncalibrated:
        st.warning("⚠ The fracture results use the uncalibrated placeholder "
                   "toughness knockdown (Results tab, Defects).")
    if result.get("fe_skipped_reason"):
        reason = result["fe_skipped_reason"]
        st.warning(
            f"⚠ FE solve was skipped: {reason}\n\n"
            f"**Empirical knockdown results are still valid** and are shown in the "
            f"other tabs. FE would have added per-element stress fields; you don't "
            f"need it for the headline knockdown numbers.\n\n"
            f"To retry with FE: adjust the mesh (Expert mode) or pick a different "
            f"loading mode. See the [README]({_README_URL}) for the FE-supported modes."
        )
        _offer_nz_fix(result.get("fe_nz_suggestion"), key="nz_fix_overview")
