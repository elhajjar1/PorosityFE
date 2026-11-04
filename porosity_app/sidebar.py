"""The sidebar inputs.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import dataclasses

import streamlit as st

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    Delamination,
    mmb_lever_length,
)
from porosity_fe.fracture_specimens import (
    _D6671_MODE_RATIO,
)
from porosity_fe.mesh import _layers_merge_angles
from porosity_fe.pipeline import _default_mesh_resolution
from porosity_fe.reporting import (
    parse_layup,
)

from .config import (
    _CRITERION_LABELS,
    _DELAMINATION_SHAPES,
    _FLEXURE_MODES,
    _FRACTURE_MODES,
    _FRACTURE_UQ_FIELDS,
    _FRACTURE_UQ_SAMPLES,
    _LOADING_LABELS,
    _NZ_SLIDER_MAX,
    _THERMAL_OPTIONS,
    _TOUGHNESS_LAW_LABELS,
    _delta_T_suggestion,
    _resolving_nz,
)

# ======================================================================
# Streamlit UI
# ======================================================================

#: Default layup of the sidebar (24-ply quasi-isotropic).
_DEFAULT_LAYUP = "[0/45/-45/90]_3s"


_DISTRIBUTION_OPTIONS = {
    "uniform": ("uniform", "midplane"),
    "clustered (midplane)": ("clustered", "midplane"),
    "clustered (surface)": ("clustered", "surface"),
    "interface": ("interface", "midplane"),
}


def _ilss_specimen_caption(n_plies: int, t_ply: float, nz: int) -> str:
    """Sidebar note: the ILSS FE solve uses the D2344 specimen, not the coupon."""
    if not n_plies:
        return ("ILSS: the FE solve uses the ASTM D2344 short-beam specimen "
                "(span 4h, 6h × 2h).")
    h = n_plies * float(t_ply)
    return (f"ILSS: the FE solve uses the ASTM D2344 short-beam specimen, "
            f"{6 * h:.1f} × {2 * h:.1f} × {h:.2f} mm on a {4 * h:.1f} mm span, "
            f"meshed 24 × 8 × {nz} (nz from this setting; nx and ny apply to "
            f"the other modes).")


def _flexure_specimen_caption(loading: str, n_plies: int, t_ply: float, nz: int) -> str:
    """Sidebar note: a flexure FE solve uses the D7264 specimen, not the coupon."""
    procedure = ("procedure A, three-point: load at mid-span"
                 if _FLEXURE_MODES[loading] == "3pt" else
                 "procedure B, four-point: loads at the quarter points")
    if not n_plies:
        return (f"Flexure (FE only): the FE solve uses the ASTM D7264 specimen "
                f"({procedure}; span 32h, 13 mm wide).")
    h = n_plies * float(t_ply)
    return (f"Flexure (FE only): the FE solve uses the ASTM D7264 specimen "
            f"({procedure}), {h:.2f} mm thick on a {32 * h:.1f} mm span (32:1), "
            f"13 mm wide, about {1.2 * 32 * h:.0f} mm long, with {nz} element "
            f"layers (nz from this setting; nx and ny follow the span). There "
            f"is no empirical flexure knockdown.")


def _validate_layup_inline():
    """On-change callback: validate the layup string and stash a status tuple.

    Stored as st.session_state["_layup_status"] = (level, message) where
    level is "ok" or "err". Called by the layup text input's on_change hook
    so the user gets immediate feedback on typos like '_3z' vs '_3s'
    (issue #126), instead of waiting for a full analysis run.
    """
    raw = st.session_state.get("layup_input", "")
    if not raw.strip():
        st.session_state["_layup_status"] = ("err", "Layup string is empty.")
        return
    try:
        parse_layup(raw)
        st.session_state["_layup_status"] = ("ok", "✓ valid")
    except ValueError as exc:
        st.session_state["_layup_status"] = ("err", str(exc))


def _apply_nz_fix(nz: int) -> None:
    """On-click callback: switch on Expert mode with the ``nz`` slider at ``nz``."""
    st.session_state["expert_mode"] = True
    st.session_state["mesh_nz"] = int(nz)


def _sync_default_nz(nz_default: int) -> None:
    """Keep the Expert-mode ``nz`` slider on the default until the user moves it.

    The default is one element layer per ply, so it changes with the layup.
    The slider follows it while it still shows the previous default (or has
    no value yet); a value the user set, by hand or with "Use nz = ...",
    stays. Must run before the slider is created in the script run.
    """
    previous = st.session_state.get("_mesh_nz_default")
    current = st.session_state.get("mesh_nz")
    if current is None or current == previous:
        st.session_state["mesh_nz"] = min(int(nz_default), _NZ_SLIDER_MAX)
    st.session_state["_mesh_nz_default"] = min(int(nz_default), _NZ_SLIDER_MAX)


def _set_delta_T(value: float) -> None:
    """On-click callback: put ``value`` (K) into the ΔT input."""
    st.session_state["thermal_delta_T"] = float(value)


def _offer_nz_fix(nz_fix: int | None, key: str) -> None:
    """Button that applies ``nz = nz_fix`` (Expert mode), when it fits the slider."""
    if nz_fix is None:
        return
    if nz_fix > _NZ_SLIDER_MAX:
        st.caption(
            f"Resolving this layup needs nz = {nz_fix}, above the nz slider's "
            f"maximum ({_NZ_SLIDER_MAX}); use a layup with fewer plies for "
            f"the thermal solve.")
        return
    st.button(
        f"Use nz = {nz_fix}", key=key, on_click=_apply_nz_fix, args=(nz_fix,),
        help="Switches on Expert mode and sets the nz slider; then run the "
             "analysis again.",
    )


def _build_thermal_inputs(material_name: str) -> tuple[str, float | None]:
    """Sidebar thermal controls; return ``(thermal_mode, delta_T)``.

    Shown only for a material with lamina CTEs (``material.has_cte``);
    otherwise a caption says why the option is unavailable. ``delta_T``
    has no default: the input starts empty and the Run button stays
    disabled until it is filled, by hand or with the labelled suggestion.
    """
    st.subheader("Thermal residual stress (FE)")
    material = MATERIALS[material_name]
    if not material.has_cte:
        with_cte = ", ".join(n for n, m in MATERIALS.items() if m.has_cte) or "none"
        st.caption(
            f"Unavailable for {material_name}: the preset has no lamina "
            f"thermal expansion coefficients (alpha_1, alpha_2), and the "
            f"solver does not guess them. Presets with CTEs: {with_cte}.")
        return "off", None
    choice = st.radio(
        "FE thermal load",
        options=list(_THERMAL_OPTIONS),
        index=0,
        key="thermal_choice",
        help=(
            "Cure residual stress of the free-standing laminate under a "
            "uniform temperature change ΔT. 'Add ... to the mechanical case' "
            "superposes it on the FE solve of the loading mode above (the "
            "residual stress forms before the laminate is gripped); 'Thermal "
            "only' solves it alone. The empirical knockdowns do not change."
        ),
    )
    thermal_mode = _THERMAL_OPTIONS[choice]
    if thermal_mode == "off":
        return "off", None
    delta_T = st.number_input(
        "ΔT (K), negative = cool-down",
        min_value=-1000.0, max_value=1000.0, value=None, step=5.0,
        format="%.1f", key="thermal_delta_T", placeholder="required",
        help=(
            "Temperature change from the stress-free state, "
            "T_service − T_stress_free, in K (the same number as the "
            "difference in °C). Negative for the cool-down after cure. "
            "There is no default."
        ),
    )
    suggestion, basis = _delta_T_suggestion(material)
    st.caption(f"Suggestion, not applied: ΔT ≈ {suggestion:g} K ({basis}).")
    st.button(f"Use suggestion ({suggestion:g} K)", key="thermal_use_suggestion",
              on_click=_set_delta_T, args=(suggestion,))
    if delta_T is None:
        st.info("Enter ΔT to run the analysis with the thermal load.")
    return thermal_mode, (None if delta_T is None else float(delta_T))


def _build_fracture_specimen_inputs(loading_mode: str, material_name: str,
                                    n_plies: int, t_ply: float
                                    ) -> tuple[float | None, float | None]:
    """Crack length (and MMB mode ratio) of a fracture specimen; return
    ``(crack_length, mmb_mode_ratio)``, both ``None`` for the other modes."""
    if loading_mode not in _FRACTURE_MODES:
        return None, None
    standard, specimen, a0_default, load = _FRACTURE_MODES[loading_mode]
    st.caption(
        f"FE only, for test planning: the {standard} {specimen} with the "
        f"laminate above (the standards test unidirectional laminates, e.g. "
        f"[0]_24), the insert at the mid-plane. It reports the compliance and "
        f"the critical {load} P_c = λ_c · P; enter the toughness below. The "
        f"default dimensions are a reading of the standard, not checked "
        f"against it.")
    crack_length = st.number_input(
        "Crack length a0 (mm)", min_value=1.0, max_value=500.0,
        value=float(a0_default), step=1.0, format="%.1f",
        key=f"frac_a0_{loading_mode}",
        help=("Insert length from the load line (DCB), from the support under "
              "the cracked end (ENF) or from the hinge line (MMB)."),
    )
    mode_ratio = None
    if loading_mode == "mmb":
        mode_ratio = st.number_input(
            "Mode ratio G_II / G_T (MMB lever)", min_value=0.05, max_value=1.0,
            value=float(_D6671_MODE_RATIO), step=0.05, format="%.2f",
            key="frac_mmb_ratio",
            help=("Sets the lever length by beam theory "
                  "(mmb_lever_length); the FE mode mix differs by a few per "
                  "cent and is reported with the results."),
        )
        try:
            material = dataclasses.replace(
                MATERIALS[material_name], t_ply=float(t_ply),
                n_plies=int(n_plies) or MATERIALS[material_name].n_plies)
            c = mmb_lever_length(material, float(mode_ratio),
                                 crack_length=float(crack_length))
            st.caption(f"Lever length c = {c:.1f} mm (beam theory, "
                       f"unidirectional arms).")
        except ValueError as exc:
            st.caption(f"No lever for this mode ratio: {exc}")
    return float(crack_length), (None if mode_ratio is None else float(mode_ratio))


def _build_onset_inputs() -> tuple[bool, float | None]:
    """Opt-in interface onset check; return ``(onset, x_avg)``."""
    onset = st.checkbox(
        "Interface delamination onset (Brewer-Lagace)", key="onset_on",
        help=("Adds the stress-based onset check of every ply interface to "
              "the FE solve: the interlaminar tractions averaged over x_avg "
              "against porosity-degraded strengths, with the critical "
              "interface, location and onset load factor. Post-processing "
              "only; it needs nz a multiple of the ply count for every "
              "interface to be a node layer."),
    )
    if not onset:
        return False, None
    x_avg = st.number_input(
        "Averaging length x_avg (mm)", min_value=0.0, max_value=50.0,
        value=None, step=0.05, format="%.3f", key="onset_x_avg",
        placeholder="one ply thickness",
        help=("Side of the square averaging window. Empty: one ply thickness "
              "(Kim & Soni). A calibration parameter: free-edge values depend "
              "on it and on the mesh."),
    )
    return True, None if x_avg is None else float(x_avg)


def _build_delamination_inputs(loading_mode: str, thermal_mode: str,
                               n_plies: int, t_ply: float
                               ) -> tuple[Delamination | None, bool, str | None]:
    """Opt-in explicit delamination (an NDI indication) of the FE mesh.

    Returns ``(delamination, knockdown_split, problem)``: the
    :class:`Delamination` (``None`` when off), whether to split the
    knockdown by defect, and why the run is blocked (``None`` when it is
    not).
    """
    st.subheader("Delamination (FE)")
    if loading_mode in _FRACTURE_MODES and thermal_mode != "thermal_only":
        st.caption("The fracture specimen's insert is its delamination.")
        return None, False, None
    on = st.checkbox(
        "Add a delamination (NDI indication)", key="delam_on",
        help=("An explicit interlaminar crack at one ply interface, separate "
              "from the porosity: the FE solve reports the energy release "
              "rates along its front and, with toughness data, the load "
              "factor at which it starts to grow."),
    )
    if not on:
        return None, False, None
    k_max = max(int(n_plies) - 2, 0)
    interface = st.number_input(
        "Interface k", min_value=0, max_value=200,
        value=max(int(n_plies) // 2 - 1, 0), step=1, key="delam_interface",
        help=("Interface k lies between ply k and ply k + 1 counted from the "
              "bottom, 0-based (plies k + 1 and k + 2 counted from 1). The "
              "mid-plane of an n-ply laminate is n/2 - 1."),
    )
    problem = None
    if n_plies and int(interface) > k_max:
        problem = (f"Interface {int(interface)} does not exist: the "
                   f"{n_plies}-ply laminate has interfaces 0 to {k_max}.")
        st.error(problem)
    elif n_plies:
        h = n_plies * float(t_ply)
        z = (int(interface) + 1) * float(t_ply)
        st.caption(f"Between plies {int(interface) + 1} and {int(interface) + 2} "
                   f"from the bottom: {z:.2f} mm above the bottom face, "
                   f"{h - z:.2f} mm below the top.")
    shape = st.selectbox("Planform", options=list(_DELAMINATION_SHAPES),
                         format_func=_DELAMINATION_SHAPES.__getitem__,
                         key="delam_shape")
    st.caption("Position and size in mm on the FE mesh: the 50 × 20 mm coupon "
               "(x along the load; the ILSS and flexure specimens have their "
               "own size, in the Overview after a run).")
    c1, c2 = st.columns(2)
    try:
        if shape == "strip":
            with c1:
                x0 = st.number_input("x start (mm)", value=20.0, step=1.0,
                                     format="%.2f", key="delam_x_start")
            with c2:
                x1 = st.number_input("x end (mm)", value=30.0, step=1.0,
                                     format="%.2f", key="delam_x_end")
            delamination = Delamination.strip(int(interface), float(x0), float(x1))
        else:
            with c1:
                xc = st.number_input("Centre x (mm)", value=25.0, step=1.0,
                                     format="%.2f", key="delam_xc")
                lx = st.number_input("Length along x (mm)", min_value=0.01,
                                     value=10.0, step=1.0, format="%.2f",
                                     key="delam_lx")
            with c2:
                yc = st.number_input("Centre y (mm)", value=10.0, step=1.0,
                                     format="%.2f", key="delam_yc")
                wy = st.number_input("Width along y (mm)", min_value=0.01,
                                     value=6.0, step=1.0, format="%.2f",
                                     key="delam_wy")
            build = Delamination.ellipse if shape == "ellipse" else Delamination.rectangle
            delamination = build(int(interface), (float(xc), float(yc)),
                                 (0.5 * float(lx), 0.5 * float(wy)))
    except ValueError as exc:
        problem = str(exc)
        st.error(problem)
        return None, False, problem
    split = st.checkbox(
        "Split the knockdown by defect (2 more FE solves)", value=True,
        key="delam_split",
        help=("Also solves the laminate with the porosity but no delamination, "
              "and with the delamination but no porosity, so the Results show "
              "which defect costs the stiffness (knockdown_decomposition)."),
    )
    return delamination, bool(split), problem


def _build_toughness_inputs(material=None) -> tuple[dict, str | None]:
    """Interlaminar toughness and its porosity knockdown; return the config
    fields and why the run is blocked (``None`` when it is not).

    When the selected ``material`` preset carries toughness (today
    ``IM7_8552_epoxy``), an empty field keeps the preset's value: the
    analysis overrides the preset only with the fields entered here."""
    preset = material if material is not None and material.has_toughness else None
    st.subheader("Interlaminar toughness (FE)")
    if preset is None:
        st.caption(
            "This material preset carries no interlaminar toughness: enter "
            "measured initiation (or precracked) values for your material, in "
            "N/mm (= kJ/m²). Without G_Ic, G_IIc and η the FE reports the "
            "energy release rates only, no fracture load factor.")
    else:
        st.caption(
            f"This preset carries interlaminar toughness: G_Ic {preset.G_Ic:g}, "
            f"G_IIc {preset.G_IIc:g}, G_IIIc {preset.G_IIIc_eff:g} N/mm, "
            f"η {preset.bk_eta:g}. Leave a field empty to keep the preset's "
            "value, or enter one measured for your material (N/mm = kJ/m²).")
    required = "required for λ_c"
    c1, c2 = st.columns(2)
    with c1:
        G_Ic = st.number_input("G_Ic, mode I (N/mm)", min_value=0.001,
                               max_value=20.0, value=None, step=0.01,
                               format="%.4f", key="tough_G_Ic",
                               placeholder=required if preset is None
                               else f"preset: {preset.G_Ic:g}")
        G_IIIc = st.number_input("G_IIIc, mode III (N/mm)", min_value=0.001,
                                 max_value=20.0, value=None, step=0.01,
                                 format="%.4f", key="tough_G_IIIc",
                                 placeholder="= G_IIc" if preset is None
                                 else f"preset: {preset.G_IIIc_eff:g}",
                                 help="Optional; empty uses G_IIc"
                                 + ("." if preset is None else " (the preset's G_IIIc)."))
    with c2:
        G_IIc = st.number_input("G_IIc, mode II (N/mm)", min_value=0.001,
                                max_value=20.0, value=None, step=0.01,
                                format="%.4f", key="tough_G_IIc",
                                placeholder=required if preset is None
                                else f"preset: {preset.G_IIc:g}")
        bk_eta = st.number_input("B-K exponent η", min_value=0.01,
                                 max_value=20.0, value=None, step=0.1,
                                 format="%.3f", key="tough_bk_eta",
                                 placeholder=required if preset is None
                                 else f"preset: {preset.bk_eta:g}",
                                 help=("Benzeggagh-Kenane: G_c = G_Ic + "
                                       "(G_IIc - G_Ic) ψ^η, fitted to mixed-"
                                       "mode bending data."))
    # A field left empty on a preset with toughness keeps the preset's value.
    given = [v is not None or preset is not None for v in (G_Ic, G_IIc, bk_eta)]
    if any(given) and not all(given):
        st.info("Enter G_Ic, G_IIc and η for the fracture load factor.")
    law = st.selectbox(
        "Porosity knockdown of the toughness",
        options=list(_TOUGHNESS_LAW_LABELS),
        format_func=_TOUGHNESS_LAW_LABELS.__getitem__, key="tough_law",
        help=("G_c = G_c0 exp(-b Vp) at the porosity of each front node. The "
              "placeholder coefficients are two-point literature estimates, "
              "flagged uncalibrated in the results; the law never raises the "
              "toughness."),
    )
    fields = {"G_Ic": G_Ic, "G_IIc": G_IIc, "G_IIIc": G_IIIc, "bk_eta": bk_eta,
              "toughness_law": law, "toughness_b": None}
    problem = None
    if law == "user":
        c3, c4 = st.columns(2)
        with c3:
            b_I = st.number_input("b_I (per unit Vp)", min_value=0.0,
                                  max_value=1000.0, value=None, step=0.5,
                                  format="%.2f", key="tough_b_I",
                                  placeholder="required")
        with c4:
            b_II = st.number_input("b_II (per unit Vp)", min_value=0.0,
                                   max_value=1000.0, value=None, step=0.5,
                                   format="%.2f", key="tough_b_II",
                                   placeholder="required")
        if b_I is None or b_II is None:
            problem = "Enter b_I and b_II, or pick another toughness knockdown."
            st.info(problem)
        else:
            fields["toughness_b"] = [float(b_I), float(b_II)]
    if all(given):
        fields["fracture_uq"] = _build_fracture_uq_inputs(law)
    return {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in fields.items()}, problem


#: Sidebar labels of the λ_c band's CoVs, by :data:`_FRACTURE_UQ_FIELDS`.
_FRACTURE_UQ_LABELS = {"G_Ic": "CoV of G_Ic", "G_IIc": "CoV of G_IIc",
                       "bk_eta": "CoV of η", "b_I": "CoV of b_I",
                       "b_II": "CoV of b_II"}


def _build_fracture_uq_inputs(law: str) -> list[float] | None:
    """The opt-in λ_c uncertainty band: one CoV (a fraction) per
    :data:`_FRACTURE_UQ_FIELDS` entry, or ``None`` when it is off or
    nothing scatters (there is no default CoV)."""
    on = st.checkbox(
        "λ_c uncertainty band (toughness scatter)", value=False, key="frac_uq_on",
        help=("Re-evaluates the fracture load factor for "
              f"{_FRACTURE_UQ_SAMPLES} draws of the toughness inputs on the "
              "front of the one FE solve (the analysis is linear, so no "
              "further solve)."))
    if not on:
        return None
    st.caption(
        "Coefficients of variation as fractions (0.05 = 5 %), lognormal. "
        "Nothing scatters unless you enter it: there is no default. Measured "
        "scatter is scarce; for IM7/8552, G_Ic 0.015 to 0.055 (Czabaj and "
        "Ratcliffe) and precracked G_IIc 0.065 to 0.076 (O'Brien et al.). "
        "See the theory page, Uncertainty propagation.")
    no_b = law == "none"
    columns = st.columns(2)
    covs = []
    for i, name in enumerate(_FRACTURE_UQ_FIELDS):
        disabled = no_b and name.startswith("b_")
        with columns[i % 2]:
            value = st.number_input(
                _FRACTURE_UQ_LABELS[name], min_value=0.0, max_value=1.0, value=0.0,
                step=0.01, format="%.3f", key=f"frac_uq_{name}", disabled=disabled,
                help=("No coefficient to scatter without a porosity knockdown."
                      if disabled else "Coefficient of variation, a fraction."))
        covs.append(0.0 if disabled else float(value))
    if not any(covs):
        st.info("Enter a CoV for at least one input: nothing scatters until "
                "you do.")
        return None
    return covs


def _build_sidebar_inputs() -> dict | None:
    """Render the sidebar input controls and return the analysis config.

    Returns a dict with keys:
        cfg: the analysis config dict (or None if the layup is invalid),
        layup_str: the raw layup string entered by the user,
        run: bool, True iff the Run button was pressed this rerun.

    Returns None if the layup string fails to parse (an st.error has
    already been emitted, so the caller can `return` immediately).
    """
    with st.sidebar:
        st.header("Inputs")
        # Keyed, with no value= (defaults to off), so the "Use nz = ..."
        # button can switch it on through st.session_state.
        expert = st.toggle(
            "Expert mode",
            key="expert_mode",
            help="Show mesh resolution sliders and other advanced options.",
        )

        st.subheader("Material & laminate")
        material_name = st.selectbox(
            "Material",
            options=list(MATERIALS.keys()),
            index=0,
        )
        # Keyed, with no value=: _render seeds the default through
        # st.session_state (passing both made Streamlit warn).
        layup_str = st.text_input(
            "Layup",
            key="layup_input",
            on_change=_validate_layup_inline,
            help=(
                "Ply angles separated by '/'. Use '_Ns' for N repeats and "
                "trailing 's' for symmetric. Examples: [0/45/-45/90]_3s, "
                "[0/90]_6s, 0/0/0/90/90/90."
            ),
        )
        _layup_status = st.session_state.get("_layup_status")
        if _layup_status:
            _level, _msg = _layup_status
            (st.success if _level == "ok" else st.error)(_msg)
        t_ply = st.number_input(
            "Ply thickness (mm)",
            min_value=0.05, max_value=0.50, value=0.183, step=0.01, format="%.3f",
        )

        st.subheader("Porosity")
        Vp = st.number_input(
            "Void content Vp (%)",
            min_value=0.1, max_value=15.0, value=3.0, step=0.5, format="%.1f",
            help=(
                "Enter a percentage: 3.0 means 3 % voids (a volume fraction "
                "of 0.03). Typical range: 0.5–5 % for autoclave, 2–10 % for OOA."
            ),
        )
        distribution_label = st.selectbox(
            "Distribution",
            options=list(_DISTRIBUTION_OPTIONS.keys()),
            index=0,
            help=(
                "Through-thickness shape of the porosity field. All "
                "options renormalize to the same specimen-average Vp, "
                "so the empirical knockdowns (Judd-Wright / power-law / "
                "linear) collapse to identical numbers across the four "
                "shapes — only the FE solve sees the local peak and "
                "diverges. See the 'Porosity Distribution Choice' "
                "section of README.md for the full rationale (issue "
                "#83).\n\n"
                "uniform: constant Vp at every z — first-pass / NCR "
                "default when only specimen-average Vp is known.\n"
                "clustered (midplane): Gaussian peak at the midplane "
                "(sigma = Lz / 6). Use when X-ray CT shows midplane "
                "concentration.\n"
                "clustered (surface): Gaussian peak at the laminate "
                "surface.\n"
                "interface: comb of Gaussians at every ply-to-ply "
                "interface (sigma = 0.35 * t_ply). Worst case for "
                "ILSS when paired with penny voids.\n\n"
                "Note: there is no preset literally named 'stack' — "
                "the stacked / layered shapes are 'clustered' and "
                "'interface'."
            ),
        )
        void_shape = st.selectbox(
            "Void shape",
            options=["spherical", "cylindrical", "penny"],
            index=0,
            help=(
                "spherical: equiaxed (AR=1)\n"
                "cylindrical: prolate (AR=3)\n"
                "penny: oblate disc (AR=10)"
            ),
        )

        st.subheader("Loading")
        loading_mode = st.selectbox(
            "Loading mode",
            options=list(_LOADING_LABELS),
            format_func=_LOADING_LABELS.__getitem__,
            index=0,
            help=(
                "The empirical knockdowns of compression, tension, shear "
                "and ILSS are always all computed; this selects the FE "
                "solve and its bar in the chart. ILSS solves the ASTM D2344 "
                "short-beam specimen (span 4h, length 6h, width 2h, roller "
                "supports, loading-nose contact band) instead of the coupon "
                "below. flexure_3pt / flexure_4pt are FE only: ASTM D7264 "
                "procedure A (three-point) / B (four-point, load span half "
                "the support span) on a 32:1 specimen 13 mm wide, reporting "
                "the FE flexural modulus next to the lamination-theory one. "
                "There is no empirical flexure knockdown: flexural strength "
                "is outside the empirical validation (README, Limitations). "
                "dcb / enf / mmb are FE only: the ASTM D5528 / D7905 / D6671 "
                "delamination fracture specimens, for test planning "
                "(compliance and critical load with the toughness below)."
            ),
        )

        # Ply count of the laminate as entered (the analysis replaces the
        # preset's n_plies with it); the preset's when the layup is invalid.
        try:
            n_plies = len(parse_layup(layup_str))
        except ValueError:
            n_plies = 0
        crack_length, mmb_mode_ratio = _build_fracture_specimen_inputs(
            loading_mode, material_name, n_plies, t_ply)

        failure_criterion = st.selectbox(
            "FE failure criterion",
            options=list(_CRITERION_LABELS),
            format_func=_CRITERION_LABELS.__getitem__,
            index=0,
            help=(
                "Criterion of the FE failure index and first-ply-failure "
                "load factor. Tsai-Wu gives one interactive index; Hashin "
                "and maximum stress separate the failure modes, and Hashin "
                "includes a delamination-onset mode (interlaminar normal "
                "and shear stress at the ply Gauss points against the "
                "porosity-degraded transverse tensile and interlaminar "
                "shear strengths). It screens initiation inside the plies; "
                "an explicit delamination is the Delamination input below."
            ),
        )
        onset, onset_x_avg = _build_onset_inputs()

        thermal_mode, delta_T = _build_thermal_inputs(material_name)

        delamination, knockdown_split, delam_problem = _build_delamination_inputs(
            loading_mode, thermal_mode, n_plies, t_ply)
        specimen = loading_mode in _FRACTURE_MODES and thermal_mode != "thermal_only"
        toughness, tough_problem = ({}, None)
        if delamination is not None or specimen:
            toughness, tough_problem = _build_toughness_inputs(MATERIALS[material_name])

        try:
            entered_layup = parse_layup(layup_str) if n_plies else None
        except ValueError:
            entered_layup = None
        nx0, ny0, nz0 = _default_mesh_resolution(
            n_plies or MATERIALS[material_name].n_plies, entered_layup)
        if not n_plies or nz0 == n_plies:
            mesh_note = "one element layer per ply"
        elif not _layers_merge_angles(entered_layup, nz0):
            mesh_note = (f"{nz0} element layers, each within plies of one angle, "
                         f"for the {n_plies}-ply layup")
        else:
            mesh_note = f"nz capped at {nz0} for the {n_plies}-ply layup"

        st.subheader("Mesh")
        if expert:
            nx = st.slider("nx", min_value=2, max_value=200, value=nx0, step=1)
            ny = st.slider("ny", min_value=2, max_value=100, value=ny0, step=1)
            # Keyed and initialised through st.session_state (not value=) so
            # the "Use nz = ..." button can set it; it follows the layup's
            # default until the user moves it.
            _sync_default_nz(nz0)
            nz = st.slider("nz", min_value=2, max_value=_NZ_SLIDER_MAX, step=1,
                           key="mesh_nz")
            total_elems = nx * ny * nz
            if total_elems < 100:
                st.warning(
                    f"⚠ Mesh is very coarse ({total_elems} elements). "
                    f"FE results from meshes below ~100 elements are unreliable. "
                    f"The default ({nx0}×{ny0}×{nz0} = {nx0 * ny0 * nz0} "
                    f"elements, {mesh_note}) is recommended."
                )
            elif total_elems > CompositeMesh._MAX_TOTAL_ELEMENTS:
                st.error(
                    f"Mesh has {total_elems:,} elements, above the "
                    f"{CompositeMesh._MAX_TOTAL_ELEMENTS:,}-element cap. "
                    f"Reduce nx, ny or nz to run."
                )
            elif total_elems > 50_000:
                # Very rough time estimate; tune from real benchmarks if available.
                est_min = total_elems / 8000
                st.warning(
                    f"⚠ Mesh is very fine ({total_elems} elements). "
                    f"Estimated solve time: ~{est_min:.0f} minutes. "
                    f"Consider reducing for iteration."
                )
        else:
            nx, ny, nz = nx0, ny0, nz0
            st.caption(f"Default mesh: {nx} × {ny} × {nz}, {mesh_note} "
                       f"(enable Expert mode to change).")
        if loading_mode == "ilss" and thermal_mode != "thermal_only":
            st.caption(_ilss_specimen_caption(n_plies, t_ply, nz))
        elif loading_mode in _FLEXURE_MODES and thermal_mode != "thermal_only":
            st.caption(_flexure_specimen_caption(loading_mode, n_plies, t_ply, nz))

        if thermal_mode != "off":
            # The thermal solve refuses a mesh whose element layers merge
            # plies of different angles (CompositeMesh.layup_discrepancies);
            # say so before the run, with the nz that resolves them.
            merges = (_layers_merge_angles(entered_layup, nz) if entered_layup is not None
                      else bool(n_plies and nz % n_plies))
            if n_plies and merges:
                nz_fix = _resolving_nz(nz, n_plies)
                st.warning(
                    f"Thermal residual stress needs the stacking sequence: "
                    f"nz = {nz} element layers merge plies of different angles "
                    f"in the {n_plies}-ply layup, so the FE solve will refuse "
                    f"it. Use nz = {nz_fix} (a multiple of {n_plies}).")
                _offer_nz_fix(nz_fix, key="nz_fix_sidebar")
        elif (delamination is not None or specimen) and n_plies and nz % n_plies:
            # The same rule for a delamination: every ply interface must be
            # a node layer.
            nz_fix = _resolving_nz(nz, n_plies)
            st.warning(
                f"A delamination needs every ply interface on a node layer: "
                f"nz = {nz} layers do not resolve the {n_plies}-ply layup, so "
                f"the FE solve will be skipped. Use nz = {nz_fix} (a multiple "
                f"of {n_plies}).")
            _offer_nz_fix(nz_fix, key="nz_fix_delamination")

        _layup_bad = st.session_state.get("_layup_status", ("ok", None))[0] == "err"
        _missing_dT = thermal_mode != "off" and delta_T is None
        _defect_problem = delam_problem or tough_problem
        _run_disabled = (
            _layup_bad
            or nx * ny * nz > CompositeMesh._MAX_TOTAL_ELEMENTS
            or _missing_dT
            or _defect_problem is not None
        )
        run = st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
            disabled=_run_disabled,
            help=(
                "Fix the layup string above before running." if _layup_bad
                else "Enter ΔT (K) for the thermal load, or switch it off."
                if _missing_dT
                else _defect_problem if _defect_problem is not None
                else "Reduce the mesh size to run." if _run_disabled
                else None
            ),
        )

    # ---- Build config from sidebar state -----------------------------------
    try:
        angles = parse_layup(layup_str)
    except ValueError as exc:
        st.error(f"Invalid layup: {exc}")
        return None

    distribution, cluster_location = _DISTRIBUTION_OPTIONS[distribution_label]
    cfg = {
        "material_name": material_name,
        "angles": angles,
        "n_plies": len(angles),
        "t_ply": float(t_ply),
        "Vp": float(Vp),
        "distribution": distribution,
        "cluster_location": cluster_location,
        "void_shape": void_shape,
        "loading_mode": loading_mode,
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "thermal_mode": thermal_mode,
        "delta_T": delta_T,
        "failure_criterion": failure_criterion,
        # Defects and fracture (each at its default when not used, so the
        # cache key of a run without them is unchanged).
        "delamination": delamination,
        "knockdown_split": knockdown_split,
        **toughness,
        "onset": onset,
        "onset_x_avg": onset_x_avg,
        "crack_length": crack_length,
        "mmb_mode_ratio": mmb_mode_ratio,
    }
    return {"cfg": cfg, "layup_str": layup_str, "run": run}
