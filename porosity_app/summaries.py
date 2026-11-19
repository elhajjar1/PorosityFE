"""Result tables of the FE solve: failure, short beam, flexure, thermal and defects.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import math

import numpy as np

from porosity_fe import (
    Delamination,
)

from .config import (
    _CRITERION_LABELS,
    _FE_APPLIED_STRAIN,
    _FE_FLEXURE_LOAD_N,
    _FE_FRACTURE_LOAD_N,
    _FE_ILSS_LOAD_N,
    _FLEXURE_MODES,
    _FRACTURE_MODES,
    _MODE_LABELS,
)


def _interior_region_text(fe_loading: str | None) -> str:
    """Where the interior failure metrics of an FE solve of ``fe_loading`` look."""
    if fe_loading == "ilss":
        return ("in the shear spans, at least h/2 (h the laminate thickness) "
                "from the supports, the loading band and the side faces, clear "
                "of the contact stresses")
    if fe_loading in _FLEXURE_MODES:
        return ("at least h/2 (h the laminate thickness) from every support "
                "and load line and from the side faces, clear of the contact "
                "stresses")
    return ("away from the free edges (at least max(h, 2 element widths) in, "
            "h the laminate thickness), where the free-edge stress peak does "
            "not reach")


def _fmt_value(value, fmt: str = "{:.3f}") -> str:
    """``fmt`` for a finite number; ``n/a`` for ``None`` / ``nan``, ``∞`` for ``inf``."""
    if value is None or math.isnan(float(value)):
        return "n/a"
    if math.isinf(float(value)):
        return "∞"
    return fmt.format(float(value))


def _applied_load_text(loading_mode: str, factor: float) -> str:
    """``factor`` times the app's FE load for ``loading_mode``, in words."""
    if loading_mode == "ilss":
        return f"{factor * abs(_FE_ILSS_LOAD_N):.3g} N midspan load"
    if loading_mode in _FLEXURE_MODES:
        where = ("midspan load" if _FLEXURE_MODES[loading_mode] == "3pt"
                 else "total load on the two load lines")
        return f"{factor * abs(_FE_FLEXURE_LOAD_N):.3g} N {where}"
    if loading_mode in _FRACTURE_MODES:
        return (f"{factor * abs(_FE_FRACTURE_LOAD_N):.3g} N "
                f"{_FRACTURE_MODES[loading_mode][3]}")
    sign = "-" if loading_mode == "compression" else ""
    name = "shear strain" if loading_mode == "shear" else "strain"
    return f"{sign}{factor * _FE_APPLIED_STRAIN * 100:.3g} % {name}"


def fe_failure_summary(result: dict) -> tuple[list[tuple[str, str, str]],
                                              list[dict], str] | None:
    """Metric rows, failure-mode rows and a caption for the FE failure block.

    ``None`` without an FE field. Metric rows ``(label, value, help)``: the
    criterion, the domain and interior maximum failure index, the
    governing mode and, unless the analysis has a thermal load (whose block
    reports the load factors), the first-ply-failure load factors. Mode
    rows (``Mode``, ``Max over model``), one per separate mode of the
    Hashin and maximum-stress criteria, from
    :attr:`~porosity_fe.FieldResults.failure_mode_maxima`; empty for
    Tsai-Wu, which does not separate modes. Maximum stress has no
    delamination mode, so that row reads ``n/a``.
    """
    fe = result.get("fe_field")
    if fe is None:
        return None
    crit = fe.failure_criterion
    maxima = fe.failure_mode_maxima
    loading = result.get("fe_loading") or result["config"]["loading_mode"]
    if maxima:
        candidates = {k: v for k, v in maxima.items()
                      if not (crit == "max_stress" and k == "delamination")}
        governing = _MODE_LABELS.get(max(candidates, key=candidates.get),
                                     max(candidates, key=candidates.get))
    else:
        governing = "n/a"
    rows = [
        ("Criterion", _CRITERION_LABELS.get(crit, crit).split(" (")[0],
         "FE failure criterion of this run (sidebar)."),
        ("Max FI (domain)", _fmt_value(fe.max_failure_index, "{:.4g}"),
         "Largest failure index over every element, at the applied load. "
         "Free edges, corners and load points concentrate stress there, so "
         "it depends on the mesh."),
        ("Interior max FI", _fmt_value(fe.interior_max_failure_index, "{:.4g}"),
         f"Maximum failure index of the elements "
         f"{_interior_region_text(result.get('fe_loading'))}; n/a when no "
         f"element is that far in."),
        ("Governing mode", governing,
         "Mode with the largest index over the model (Hashin, maximum "
         "stress). Tsai-Wu's single polynomial does not separate modes."),
    ]
    if not result.get("fe_thermal"):
        rows.append((
            "FPF load factor (interior)",
            _fmt_value(fe.interior_first_ply_failure_load_factor, "{:.3g}"),
            "Multiplier on the applied load at which the first interior "
            "element reaches the criterion (domain value in the caption)."))
    mode_rows = []
    for key, label in _MODE_LABELS.items():
        if not maxima or key not in maxima:
            continue
        value = ("n/a" if crit == "max_stress" and key == "delamination"
                 else _fmt_value(maxima[key], "{:.4g}"))
        mode_rows.append({"Mode": label, "Max over model": value})
    parts = [f"Failure indices at the applied load "
             f"({_applied_load_text(loading, 1.0)})."
             if loading != "thermal" else "Failure indices at the entered ΔT."]
    if not result.get("fe_thermal"):
        lam = fe.first_ply_failure_load_factor
        if lam is not None and math.isfinite(lam):
            parts.append(f"First-ply failure over the whole model at "
                         f"{_applied_load_text(loading, lam)}.")
    if crit == "hashin":
        parts.append(
            "Delamination onset is Hashin's interlaminar mode: "
            "(σ33/Yt)² + (τ13² + τ23²)/S², at the ply Gauss points (inside "
            "the plies, not on the interfaces) against the porosity-"
            "degraded transverse tensile strength and interlaminar shear "
            "strength. The model has no explicit delamination, so it "
            "screens initiation only.")
    elif crit == "max_stress":
        parts.append("Maximum stress has no delamination mode; σ33 and the "
                     "transverse shears enter its matrix and shear modes.")
    else:
        parts.append("Pick Hashin in the sidebar to see the failure modes, "
                     "including delamination onset.")
    return rows, mode_rows, " ".join(parts)


def fe_short_beam_summary(result: dict) -> tuple[list[tuple[str, str, str]], str] | None:
    """Metric rows ``(label, value, help)`` and a caption for an ILSS FE result.

    ``None`` unless the analysis solved the ASTM D2344 short-beam specimen
    (``result["fe_short_beam"]``). The rows give the FE knockdown, the FE
    mid-plane shear at the quarter-span points next to the D2344 formula
    ``0.75 P / (b h)``, and the short-beam strength D2344 would report at the
    first-ply failure of the interior (the shear spans clear of the
    supports and the loading nose); the caption gives the specimen.
    """
    sb = result.get("fe_short_beam")
    spec = result.get("fe_specimen")
    fe = result.get("fe_field")
    if sb is None or spec is None or fe is None:
        return None
    ratio = sb.midplane_shear / sb.apparent_shear
    rows = [
        ("FE knockdown", _fmt_value(fe.knockdown),
         "Porous over pristine stiffness of the short-beam specimen "
         "(P² / uᵀKu), from two solves with the same supports and load."),
        ("Mid-plane shear / 0.75P/(bh)", f"{ratio:.3f}",
         f"FE τxz at mid-thickness, averaged across the width, half-way "
         f"between each support and the nose: {sb.midplane_shear:.4g} MPa at "
         f"{abs(sb.applied_load):g} N, against the D2344 formula "
         f"0.75 P/(b h) = {sb.apparent_shear:.4g} MPa. At a span of four "
         f"thicknesses beam theory is approximate: the profile is flatter "
         f"than the parabola behind the formula, so the ratio is below 1 "
         f"(0.91 for a UD T800 specimen; it depends on the layup)."),
        ("Interior FPF load", _fmt_value(sb.interior_first_ply_failure_load,
                                         "{:,.0f} N"),
         "Load at first-ply failure in the shear spans, at least h/2 from "
         "the supports, the loading band and the side faces. The domain-wide "
         "value sits in the contact stresses under the nose or at a support, "
         "which depend on the mesh."),
        ("Apparent ILSS at FPF", _fmt_value(sb.interior_apparent_strength,
                                            "{:.1f} MPa"),
         "0.75 × (interior first-ply-failure load) / (b h): the short-beam "
         "strength D2344 would report if the specimen failed at first-ply "
         "failure. Compare with the empirical ILSS strength, which comes from "
         "test data."),
    ]
    caption = (
        f"{spec['standard']} short-beam specimen: {spec['length_mm']:.1f} × "
        f"{spec['width_mm']:.1f} × {spec['thickness_mm']:.2f} mm on a "
        f"{spec['span_mm']:.1f} mm span (4h), roller supports, the load spread "
        f"over a {spec['load_width_mm']:.2f} mm loading-nose contact band; "
        f"mesh {spec['mesh'].replace('x', ' × ')}. Domain-wide first-ply failure "
        f"at {_fmt_value(sb.first_ply_failure_load, '{:,.0f}')} N (contact "
        f"zone, mesh-dependent).")
    return rows, caption


def _distribution_text(cfg: dict) -> str:
    """The porosity distribution of ``cfg`` as the sidebar names it."""
    if cfg["distribution"] == "clustered":
        return f"clustered ({cfg.get('cluster_location', 'midplane')})"
    return str(cfg["distribution"])


def fe_flexure_summary(result: dict) -> tuple[list[tuple[str, str, str]], str] | None:
    """Metric rows ``(label, value, help)`` and a caption for a flexure FE result.

    ``None`` unless the analysis ran a flexure mode (``result["fe_flexure"]``).
    The rows give the FE knockdown, the FE flexural modulus (the shear-
    corrected ``effective_modulus``) next to the classical-lamination-theory
    one, and their ratio; FE values read ``n/a`` when the FE solve was
    skipped. The caption describes the specimen, gives the CLT knockdown
    and says when the CLT and FE porosity models differ (the CLT value
    always uses spherical voids spread uniformly through the thickness).
    """
    flex = result.get("fe_flexure")
    if flex is None:
        return None
    cfg = result["config"]
    fe = result.get("fe_field")
    spec = result.get("fe_specimen")
    E_f = None if fe is None else fe.effective_modulus
    clt = flex["clt_E_f"]
    ratio = None if E_f is None else E_f / clt
    rows = [
        ("FE knockdown", _fmt_value(None if fe is None else fe.knockdown),
         "Porous over pristine beam stiffness of the D7264 specimen "
         "(P² / uᵀKu), from two solves with the same supports and load."),
        ("FE flexural modulus E_f",
         _fmt_value(None if E_f is None else E_f / 1000.0, "{:.1f} GPa"),
         "The modulus beam theory needs to reproduce the FE mid-span "
         "deflection once the transverse-shear deflection is taken out "
         "(Timoshenko). Uses the selected void shape and distribution."),
        ("CLT E_f (D matrix)", f"{clt / 1000.0:.1f} GPa",
         "Classical lamination theory, E_f = 12 / (h³ d₁₁) with d = D⁻¹, "
         "every ply degraded by Mori-Tanaka spherical voids at the "
         "specimen-average Vp, uniform through the thickness. The D-matrix "
         "flexural modulus is part of the validation set (README, "
         "Validation)."),
        ("FE / CLT E_f", _fmt_value(ratio),
         "Ratio of the two flexural moduli. With the same porosity model "
         "they agree within about 0.2 % for UD and cross-ply laminates; a "
         "quasi-isotropic one reads about 1 % low in FE at the 13 mm D7264 "
         "width (free-edge effect of the ±45° plies)."),
    ]
    parts = []
    if spec is not None:
        load = ("loaded at mid-span" if flex["procedure"] == "3pt" else
                f"loaded on two lines {spec['load_span_mm']:.1f} mm apart (half "
                f"the span)")
        parts.append(
            f"{spec['standard']} {spec['specimen']}: {spec['length_mm']:.1f} × "
            f"{spec['width_mm']:.1f} × {spec['thickness_mm']:.2f} mm on a "
            f"{spec['span_mm']:.1f} mm span ({spec['span_to_thickness']:.0f}:1), "
            f"roller supports, {load}; mesh "
            f"{spec['mesh'].replace('x', ' × ')}.")
    parts.append(f"CLT knockdown of E_f: {flex['clt_knockdown']:.3f}.")
    differ = []
    if cfg["void_shape"] != flex["clt_void_shape"]:
        differ.append(f"{flex['clt_void_shape']} voids in CLT, {cfg['void_shape']} "
                      f"voids in FE")
    if cfg["distribution"] != "uniform":
        differ.append(f"uniform porosity in CLT, the {_distribution_text(cfg)} "
                      f"distribution in FE (in bending, voids near the surfaces "
                      f"cost more stiffness than voids near the mid-plane)")
    if differ:
        parts.append("The porosity models differ: " + "; ".join(differ) + ".")
    parts.append("There is no empirical flexure knockdown: flexural strength is "
                 "outside the empirical validation (README, Limitations).")
    return rows, " ".join(parts)


def fe_thermal_summary(result: dict) -> tuple[list[tuple[str, str, str]], str] | None:
    """Metric rows ``(label, value, help)`` and a caption for a thermal FE result.

    ``None`` when the analysis had no thermal load or its FE solve was
    skipped. The first-ply-failure factor is read with its
    ``load_factor_basis``: for ``'delta_T'`` (thermal only) it multiplies
    the temperature change, so it is shown as the critical ``delta_T``,
    and the knockdown is ``n/a`` (no load, so no stiffness measure). For
    ``'mechanical_with_residual'`` (combined) it multiplies the applied
    mechanical load with the residual stress held fixed, and the caption
    gives the factor without the residual stress next to it.
    """
    thermal = result.get("fe_thermal")
    fe = result.get("fe_field")
    if not thermal or fe is None:
        return None
    dT = float(thermal["delta_T"])
    crit = fe.failure_criterion
    lam = fe.first_ply_failure_load_factor
    ilam = fe.interior_first_ply_failure_load_factor
    residual_row = (
        "Residual max FI", _fmt_value(fe.residual_max_failure_index),
        f"{crit} failure index of the residual stress alone, maximum over the "
        f"whole model (free-edge peak included). At or above 1 the laminate "
        f"fails on cool-down, before any load.")
    interior_help = (
        f"Maximum failure index of the elements "
        f"{_interior_region_text(result.get('fe_loading'))}; n/a when no "
        f"element is that far in.")

    if fe.load_factor_basis == "delta_T":
        finite = lam is not None and math.isfinite(lam)
        rows = [
            ("FE knockdown", "n/a",
             "A thermal-only solve applies no mechanical load, so it has no "
             "stiffness measure and no knockdown."),
            residual_row,
            ("Interior max FI", _fmt_value(fe.interior_max_failure_index),
             interior_help),
            ("Critical ΔT (first-ply failure)",
             f"{lam * dT:.0f} K" if finite else "none",
             f"Load-factor basis 'delta_T': the first-ply-failure factor "
             f"({_fmt_value(lam, '{:.3g}')}) multiplies ΔT = {dT:g} K."),
        ]
        if finite:
            caption = (
                f"Load-factor basis 'delta_T': the first-ply-failure factor "
                f"{lam:.3g} multiplies ΔT = {dT:g} K, so the first ply fails "
                f"at ΔT ≈ {lam * dT:.0f} K over the whole model")
            if ilam is not None and math.isfinite(ilam):
                caption += f" and ≈ {ilam * dT:.0f} K away from the free edges"
            caption += (". The whole-model values include the free-edge stress "
                        "peak, which depends on the mesh.")
            if lam < 1.0:
                caption += (" The factor is below 1: the laminate fails before "
                            "reaching the entered ΔT.")
        else:
            caption = (f"No point is stressed at ΔT = {dT:g} K, so the first-"
                       f"ply-failure factor is infinite.")
        return rows, caption

    loading = result["config"]["loading_mode"]
    mech = result.get("fe_mechanical_fpf")
    rows = [
        ("FE knockdown", _fmt_value(fe.knockdown),
         "Stiffness knockdown of the mechanical solve. In a linear analysis "
         "the residual stress does not change the stiffness."),
        residual_row,
        ("Interior max FI", _fmt_value(fe.interior_max_failure_index),
         interior_help + " Total stress: residual plus the applied load."),
        ("FPF load factor", _fmt_value(lam, "{:.2f}"),
         "Multiplier on the applied mechanical load at first-ply failure, "
         "with the residual stress held fixed (load-factor basis "
         "'mechanical_with_residual'); 0 when the residual stress alone "
         "fails the laminate."),
    ]
    parts = [f"Load-factor basis '{fe.load_factor_basis}': "]
    if lam is not None and math.isfinite(lam):
        parts.append(f"the first ply fails at {lam:.2f} × the applied load "
                     f"({_applied_load_text(loading, lam)})")
        if ilam is not None and math.isfinite(ilam):
            parts.append(f", {ilam:.2f} × away from the free edges")
    else:
        parts.append("no point is stressed by the applied load")
    if mech is not None and math.isfinite(mech):
        parts.append(f". Without the residual stress the factor is {mech:.2f} "
                     f"({_applied_load_text(loading, mech)})")
    parts.append(".")
    return rows, "".join(parts)


# ----------------------------------------------------------------------
# Defects: delamination front, knockdown decomposition, interface onset,
# fracture specimens (headless; the Results tab's Defects block shows them)
# ----------------------------------------------------------------------

def _crack_front(result: dict):
    """The delamination front of the FE solve of ``result``, or ``None``."""
    fe = result.get("fe_field")
    front = None if fe is None else getattr(fe, "crack_front", None)
    return None if front is None or len(front) == 0 else front


def _load_factor_text(result: dict, factor: float) -> str:
    """``factor`` times the load a load factor of ``result`` multiplies."""
    fe = result["fe_field"]
    if fe.load_factor_basis == "delta_T":
        thermal = result.get("fe_thermal") or {}
        return f"ΔT = {factor * float(thermal.get('delta_T', float('nan'))):.3g} K"
    loading = result.get("fe_loading") or result["config"]["loading_mode"]
    text = _applied_load_text(loading, factor)
    if fe.load_factor_basis == "mechanical_with_residual":
        text += ", with the residual stress held fixed"
    return text


def front_order(front) -> tuple[list[np.ndarray], np.ndarray, str]:
    """The front nodes in order along the front, for plots and tables.

    Returns ``(segments, abscissa, label)``: index arrays into the front
    rows, one per straight front or one for a closed front, the abscissa of
    every row and its axis label. A front across the width (a strip, a
    specimen insert: every node of a segment at one ``x``) runs along ``y``;
    one along the length runs along ``x``; any other front (an ellipse or a
    rectangle, a staircase on the grid) is ordered by its angle around the
    centroid of its nodes, in degrees from ``+x``.
    """
    x = np.asarray(front.x, dtype=float)
    y = np.asarray(front.y, dtype=float)
    tol = 1e-9 * max(float(np.ptp(x)) if x.size else 0.0,
                     float(np.ptp(y)) if y.size else 0.0, 1.0)

    def lines(along: np.ndarray, across: np.ndarray) -> list[np.ndarray] | None:
        keys = np.round(across / tol) if tol > 0 else across
        groups = [np.flatnonzero(keys == k) for k in np.unique(keys)]
        if len(groups) > 2 or any(g.size < 2 for g in groups):
            return None
        return [g[np.argsort(along[g], kind="stable")] for g in groups]

    across_width = lines(y, x)
    if across_width is not None:
        return across_width, y, "y along the front (mm)"
    along_length = lines(x, y)
    if along_length is not None:
        return along_length, x, "x along the front (mm)"
    angle = np.degrees(np.arctan2(y - y.mean(), x - x.mean())) % 360.0
    return [np.argsort(angle, kind="stable")], angle, \
        "Angle around the front (°, from +x)"


def delamination_text(delamination: Delamination) -> str:
    """A delamination's planform, position and interface in words (mm)."""
    k = delamination.interface
    if delamination.shape == "strip":
        planform = (f"through-width strip from x = {delamination.x_start:g} to "
                    f"{delamination.x_end:g} mm")
    else:
        (xc, yc), (ax, ay) = delamination.center, delamination.half_size
        planform = (f"{delamination.shape} {2 * ax:g} × {2 * ay:g} mm centred "
                    f"at ({xc:g}, {yc:g}) mm")
    return f"{planform}, interface {k} (between plies {k + 1} and {k + 2})"


def ncr_delamination_available(result: dict) -> bool:
    """Whether the NCR can assess ``result``'s delamination on a fracture
    margin: a delamination of the part (not a fracture specimen) with a
    front and a fracture load factor."""
    fe = result.get("fe_field")
    return (result.get("delamination") is not None and fe is not None
            and _crack_front(result) is not None
            and fe.fracture_load_factor is not None)


def fe_front_rows(result: dict) -> list[dict]:
    """Rows of the front table: one per front node, in order along the front.

    Columns: position (mm), ``G_I``, ``G_II``, ``G_III``, ``G_T`` (N/mm, at
    the analysis load), the mode mixity ``ψ = (G_II + G_III) / G_T``, the
    porosity at the crack plane (shown in %), the toughness ``G_c`` there
    and the critical load factor ``λ_c`` (``n/a`` without toughness data).
    Empty without a front.
    """
    front = _crack_front(result)
    if front is None:
        return []
    segments, _, _ = front_order(front)
    rows = []
    for seg in segments:
        for i in seg:
            rows.append({
                "x (mm)": f"{front.x[i]:.2f}",
                "y (mm)": f"{front.y[i]:.2f}",
                "G_I (N/mm)": f"{front.G_I[i]:.4g}",
                "G_II (N/mm)": f"{front.G_II[i]:.4g}",
                "G_III (N/mm)": f"{front.G_III[i]:.4g}",
                "G_T (N/mm)": f"{front.G_T[i]:.4g}",
                "ψ": _fmt_value(front.psi[i], "{:.3f}"),
                "Vp (%)": f"{100.0 * front.Vp[i]:.2f}",
                "G_c (N/mm)": ("n/a" if front.G_c is None
                               else _fmt_value(front.G_c[i], "{:.4g}")),
                "λ_c": ("n/a" if front.lambda_c is None
                        else _fmt_value(front.lambda_c[i], "{:.4g}")),
            })
    return rows


def _mode_shares(front, row: int) -> str:
    """``G_I``, ``G_II`` and ``G_III`` of front row ``row`` as shares of ``G_T``."""
    total = float(front.G_T[row])
    if not total > 0.0:
        return "no energy released"
    parts = (("G_I", front.G_I[row]), ("G_II", front.G_II[row]),
             ("G_III", front.G_III[row]))
    return " · ".join(f"{name} {100.0 * float(v) / total:.0f} %" for name, v in parts)


def fe_delamination_summary(result: dict) -> tuple[list[tuple[str, str, str]], str, bool] | None:
    """Metric rows ``(label, value, help)``, a caption and the uncalibrated
    flag of the delamination front; ``None`` without a front.

    The rows give the fracture load factor (the smallest critical load
    factor over the front), the same over the interior front nodes (#227),
    where the smallest governs, its mode mixity and mode shares, and the
    largest ``G_T`` at the analysis load. The flag is
    ``True`` when the toughness was knocked down by porosity with the
    uncalibrated placeholder coefficients.
    """
    front = _crack_front(result)
    if front is None:
        return None
    fe = result["fe_field"]
    lam = fe.fracture_load_factor
    row = front.governing
    focus = row if row is not None else int(np.argmax(front.G_T))
    where = f"{front.x[focus]:.1f}, {front.y[focus]:.1f}"
    interior = getattr(fe, "interior_fracture_load_factor", None)
    rows = [
        ("Fracture load factor λ_c",
         "n/a" if lam is None else _fmt_value(lam, "{:.3g}"),
         "Smallest critical load factor over the front: the multiple of the "
         "analysis load at which the delamination starts to grow (G_T reaches "
         "the Benzeggagh-Kenane toughness G_c(ψ)). n/a without G_Ic, G_IIc "
         "and η; ∞ when the load releases no energy at the front. Where it "
         "sits at a lateral face (a coupon edge the delamination reaches) G "
         "is mesh dependent: compare the interior value."),
        ("Interior λ_c",
         "n/a" if interior is None else _fmt_value(interior, "{:.3g}"),
         "Smallest λ_c over the front nodes at least max(h, 2 element widths) "
         "from every lateral face of the coupon (the margin of the interior "
         "failure index): the value that converges under mesh refinement. "
         "n/a without toughness or without an interior front node."),
        ("Governing node (x, y mm)" if row is not None
         else "Largest G_T at (x, y mm)", where,
         "Front node with the smallest λ_c (without toughness, the largest "
         "G_T)."),
        ("Mode mixity ψ there", _fmt_value(front.psi[focus], "{:.2f}"),
         f"ψ = (G_II + G_III) / G_T at that node: {_mode_shares(front, focus)}."),
        ("Max G_T (N/mm)", f"{float(np.max(front.G_T)):.3g}",
         "Largest total energy release rate over the front at the analysis "
         "load; it scales with the load squared."),
    ]
    parts = [
        f"Virtual crack closure at {len(front)} front nodes "
        f"(Δa = {float(np.median(front.delta_a)):.3g} mm) at the analysis load "
        f"({_load_factor_text(result, 1.0)})."]
    if lam is None:
        parts.append("Enter G_Ic, G_IIc and η in the sidebar for the "
                     "fracture load factor; no material preset carries them.")
    elif math.isfinite(lam):
        parts.append(f"Growth starts at {_load_factor_text(result, lam)}.")
    if front.toughness_knockdown is not None:
        parts.append(f"Toughness: {front.toughness_knockdown.describe()}, at "
                     f"the porosity of each front node.")
    if fe.crack_contact:
        solves = fe.contact_iterations
        parts.append(f"Crack-face contact: {fe.contact_pairs} pairs in contact "
                     f"after {solves} solve{'' if solves == 1 else 's'}.")
    parts.append(
        "Linear elastic fracture mechanics: λ_c is growth onset (the "
        "Delamination growth panel below grows the crack node by node, "
        "quasi-statically or by a Paris law), and the load factor ignores "
        "buckling (the Linear buckling panel below gives the sublaminate's). "
        "A planform front is a "
        "staircase on the grid, so G scatters from node to node, and between "
        "plies of different angles the G_I / G_II split depends on Δa: compare "
        "G_T, and refine nx and ny (Expert mode) to check it.")
    return rows, " ".join(parts), bool(front.toughness_uncalibrated)


def fe_fracture_uq_summary(result: dict, compare: float | None = None
                           ) -> tuple[list[tuple[str, str, str]], str] | None:
    """Metric rows and a caption of the λ_c uncertainty band; ``None`` when
    it was not requested.

    The rows give the nominal λ_c and the p5 / p50 / p95 of the draws and,
    with ``compare`` (a load factor the user enters), the share of the
    draws below it: an estimate under the entered CoVs, not a disposition.
    """
    band = result.get("fe_fracture_uq")
    error = result.get("fe_fracture_uq_error")
    if band is None and error is None:
        return None
    if band is None:
        return [], f"The λ_c uncertainty band was not computed: {error}."
    pct = band.percentiles
    rows = [
        ("λ_c nominal", _fmt_value(band.nominal, "{:.3g}"),
         "The deterministic fracture load factor of the solve."),
        ("λ_c p5", _fmt_value(pct["p5"], "{:.3g}"),
         "5th percentile of the smallest λ_c over the front, over the draws."),
        ("λ_c p50", _fmt_value(pct["p50"], "{:.3g}"), "Median of the draws."),
        ("λ_c p95", _fmt_value(pct["p95"], "{:.3g}"), "95th percentile of the draws."),
    ]
    if compare is not None:
        rows.append((
            f"Draws below {compare:g}", f"{100.0 * band.probability_below(compare):.1f} %",
            "Share of the draws whose λ_c is below the entered factor: an "
            "estimate under the entered CoVs only, not a disposition."))
    scattered = ", ".join(f"{name} {params:g}" for name, (_dist, params) in band.spec.items())
    nominal_row = int(np.argmin(band.nominal_per_node))
    share = float(band.governing_fraction[nominal_row])
    caption = (
        f"{band.n_samples} Latin-hypercube draws (seed 0) of the toughness "
        f"inputs, lognormal with CoV {scattered}, each re-evaluating λ_c at the "
        f"{band.n_front_nodes} front nodes on this solve's energy release "
        f"rates; no further FE solve. The nominal governing node governs "
        f"{100.0 * share:.0f} % of the draws. Not covered: porosity, stiffness, "
        f"geometry, load, mesh and model-form uncertainty, and correlation "
        f"between the inputs (the draws are independent). The share below a "
        f"factor is information for the MRB, not a disposition: the tool "
        f"recommends no disposition for a delamination.")
    if band.toughness_uncalibrated:
        caption += (" The draws are centred on the uncalibrated placeholder "
                    "toughness knockdown.")
    return rows, caption


def fe_decomposition_summary(result: dict) -> tuple[list[tuple[str, str, str]], str] | None:
    """Metric rows and a caption of the knockdown decomposition; ``None``
    when it was not requested or the FE solve was skipped."""
    parts = result.get("fe_decomposition")
    error = result.get("fe_decomposition_error")
    if parts is None and error is None:
        return None
    if parts is None:
        return [], f"The knockdown decomposition failed: {error}"
    dominant = {"porosity": "porosity", "delamination": "delamination",
                None: "neither (equal)"}[parts.dominant]
    rows = [
        ("KD total", _fmt_value(parts.total, "{:.4f}"),
         "Stiffness knockdown with the porosity and the delamination."),
        ("KD porosity only", _fmt_value(parts.porosity_only, "{:.4f}"),
         "The same porosity, the delamination bonded."),
        ("KD delamination only", _fmt_value(parts.delamination_only, "{:.4f}"),
         "The same delamination, no porosity."),
        ("Larger stiffness loss", dominant,
         "The defect whose knockdown alone is the lower."),
    ]
    took = result.get("fe_decomposition_time_s")
    caption = (
        "Each knockdown is the structural stiffness over that of the laminate "
        "with neither defect, under the same load. In a linear analysis an "
        "embedded delamination under in-plane load costs little stiffness "
        "(a linear static solve does not buckle): its effect shows in the "
        "fracture load factor and in the buckling load (Linear buckling "
        "panel), not here."
        + ("" if took is None else f" The two extra FE solves took {took:.0f} s."))
    return rows, caption


def fe_onset_summary(result: dict) -> tuple[list[tuple[str, str, str]], list[dict], str] | None:
    """Metric rows, per-interface rows and a caption of the interface onset;
    ``None`` when it was not requested."""
    onset = result.get("fe_onset")
    error = result.get("fe_onset_error")
    if onset is None and error is None:
        return None
    if onset is None:
        return [], [], f"The interface onset was not evaluated: {error}"
    summ = onset.summary()
    loc = onset.critical_location
    rows = [
        ("Max onset index", _fmt_value(summ["max_index"], "{:.3g}"),
         "Largest Brewer-Lagace index over the interface nodes at the analysis "
         "load (1 = onset)."),
        ("Critical interface",
         "n/a" if summ["critical_interface"] is None
         else f"{summ['critical_interface']} (plies {summ['critical_interface'] + 1}"
              f"/{summ['critical_interface'] + 2})",
         "Interface k lies between ply k and k + 1 from the bottom (0-based); "
         "the plies are counted from 1 here."),
        ("Location (x, y mm)", "n/a" if loc is None else f"{loc[0]:.1f}, {loc[1]:.1f}",
         "Interface node of the largest index."),
        ("Onset load factor", _fmt_value(onset.onset_load_factor, "{:.3g}"),
         "Multiple of the analysis load at which the index first reaches 1."),
        ("Interior onset load factor",
         _fmt_value(onset.interior_onset_load_factor, "{:.3g}"),
         "The same away from the free edges (at least max(h, 2 element "
         "widths) in)."),
    ]
    table = [{"Interface": k, "Plies": f"{k + 1}/{k + 2}",
              "Max onset index": _fmt_value(v, "{:.3g}")}
             for k, v in onset.interface_max_index().items()]
    lam = onset.onset_load_factor
    caption = (
        f"Brewer-Lagace quadratic criterion on the interlaminar tractions "
        f"averaged over a {onset.x_avg:.3g} mm window (x_avg, one ply thickness "
        f"by default: a calibration parameter), against porosity-degraded "
        f"strengths (Z_t = sigma_2t and S_13 = S_23 = tau_ilss unless the "
        f"material sets sigma_3t and tau_13). "
        + ("" if lam is None or not math.isfinite(lam)
           else f"Onset at {_load_factor_text(result, lam)}. ")
        + "The stress at a free edge is singular, so the values there depend "
          "on the mesh and on x_avg; read the interior load factor. A "
          "stress-based initiation screen of an intact interface, not a "
          "fracture analysis.")
    return rows, table, caption


def fe_fracture_test_summary(result: dict) -> tuple[list[tuple[str, str, str]], str] | None:
    """Metric rows and a caption of a fracture specimen solve (DCB, ENF,
    MMB); ``None`` for the other modes or when the FE solve was skipped.

    The rows give the load-point compliance, ``G_T`` and ``ψ`` at mid-width
    (where test reports compare), the critical load ``P_c = λ_c · P`` at
    mid-width and at the weakest front node.
    """
    loading = result.get("fe_loading")
    spec = result.get("fe_specimen")
    front = _crack_front(result)
    if loading not in _FRACTURE_MODES or spec is None or front is None:
        return None
    fe = result["fe_field"]
    P = spec["applied_load_N"]
    mid = int(np.argmin(np.abs(np.asarray(front.y) - 0.5 * spec["width_mm"])))
    lam_mid = None if front.lambda_c is None else float(front.lambda_c[mid])
    lam = fe.fracture_load_factor
    rows = [
        ("Compliance C (mm/N)", _fmt_value(fe.load_point_compliance, "{:.4g}"),
         f"Load-point compliance: the displacement of the load point per unit "
         f"{spec['load']} (work conjugate)."),
        ("G_T at mid-width (N/mm)", f"{float(front.G_T[mid]):.4g}",
         f"At P = {P:g} N; it scales with P²."),
        ("ψ at mid-width", _fmt_value(front.psi[mid], "{:.3f}"),
         "Mode mixity (G_II + G_III) / G_T where the test reports compare."),
        ("P_c at mid-width (N)",
         "n/a" if lam_mid is None else _fmt_value(lam_mid * P, "{:.4g}"),
         "Critical load λ_c · P of the front node at mid-width: growth onset. "
         "Needs G_Ic, G_IIc and η."),
        ("P_c, weakest node (N)",
         "n/a" if lam is None else _fmt_value(lam * P, "{:.4g}"),
         "λ_c · P at the front node that grows first."),
    ]
    parts = [
        f"{spec['standard']} {spec['specimen']}, for test planning: "
        f"{spec['length_mm']:.1f} × {spec['width_mm']:.1f} × "
        f"{spec['thickness_mm']:.2f} mm, insert a0 = {spec['crack_length_mm']:g} mm "
        f"at interface {spec['interface']}"
        + ("" if spec["span_mm"] is None else f", span {spec['span_mm']:g} mm")
        + f", mesh {spec['mesh'].replace('x', ' × ')}; P is the {spec['load']}."]
    if loading == "mmb":
        parts.append(
            f"Lever c = {spec['lever_length_mm']:.1f} mm for G_II/G_T = "
            f"{spec['mode_ratio_target']:g} (beam theory of a unidirectional "
            f"specimen, mmb_lever_length); the FE gives ψ = "
            f"{_fmt_value(front.psi[mid], '{:.3f}')} at mid-width.")
    parts.append(
        "The standards test unidirectional laminates (e.g. [0]_24). The "
        "default dimensions are a reading of the standards not checked against "
        "them. Linear analysis: Krueger's NASA benchmarks are reproduced within "
        "3 % in onset load (README, Limitations).")
    return rows, " ".join(parts)


# ----------------------------------------------------------------------
# Fatigue and environment (empirical)
# ----------------------------------------------------------------------

#: The tension-type modes a single tension stress ratio applies to; the
#: compression mode takes its own (R >= 1).
_TENSION_TYPE_MODES = ("tension", "shear", "ilss", "transverse_tension")


def durability_table(result: dict, *, cycles: float | None = None,
                     mean_stress: str | None = None, R: float | None = None,
                     R_compression: float | None = None, porosity_slope: float = 0.0,
                     T: float | None = None, M: float | None = None,
                     T_g_dry: float | None = None) -> tuple[list[dict], list[str], dict]:
    """The fatigue and hygrothermal knockdowns of an analysis.

    Empirical only: :func:`~porosity_fe.reporting.durability_record` of the
    analysed laminate at its specimen-average void content. ``R`` is the
    stress ratio of the tension-type modes and ``R_compression`` of the
    compression one (each ``None`` for the calibration ratio); they apply
    with ``mean_stress='goodman'`` only. The hygrothermal factor needs a dry
    glass-transition temperature, ``T_g_dry`` or the material's.

    Returns
    -------
    (rows, notes, record)
        Table rows (mode, law, the porosity, fatigue and environment
        factors, their product and the strength in MPa), the warnings the
        models raised, and the record itself.

    Raises
    ------
    ValueError
        For a hygrothermal request without a dry glass-transition
        temperature, or an input the models refuse.
    """
    import dataclasses
    import warnings

    from porosity_fe import EmpiricalSolver, FatigueModel
    from porosity_fe.reporting import durability_record

    material = result["material"]
    if T_g_dry is not None:
        material = dataclasses.replace(material, T_g_dry=float(T_g_dry))
    environment = None
    if T is not None or (M is not None and M > 0.0):
        if material.T_g_dry is None:
            raise ValueError(
                "The hygrothermal factor needs the dry glass-transition "
                "temperature T_g: enter it (no preset sets one).")
        environment = {"T": float(material.T_ref) if T is None else float(T),
                       "M": float(material.M_ref) if M is None else float(M)}
    ratios: dict[str, float] = {}
    if mean_stress is not None:
        if R is not None:
            ratios.update({mode: float(R) for mode in _TENSION_TYPE_MODES})
        if R_compression is not None:
            ratios["compression"] = float(R_compression)
    fatigue = (FatigueModel(mean_stress=mean_stress, porosity_slope=porosity_slope)
               if mean_stress is not None or porosity_slope else None)
    solver = EmpiricalSolver(result["mesh"], material, ply_angles=result["config"]["angles"],
                             fatigue_model=fatigue)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = durability_record(solver, cycles=cycles, R=ratios or None,
                                   environment=environment)
    rows = []
    for mode, models in record["per_mode"].items():
        for model, entry in models.items():
            rows.append({
                "Mode": mode, "Law": model,
                "Porosity": entry["static_knockdown"],
                "Fatigue": entry["fatigue_knockdown"],
                "Environment": entry["environment_knockdown"],
                "Knockdown": entry["knockdown"],
                "Strength (MPa)": entry["failure_stress_MPa"],
            })
    notes = sorted({str(w.message) for w in caught})
    return rows, notes, record


# ----------------------------------------------------------------------
# Delamination growth
# ----------------------------------------------------------------------

#: What each stop reason of a growth analysis means, for the panel.
GROWTH_STOP_TEXT = {
    "max_steps": "the number of steps was reached",
    "max_area": "the area limit was reached",
    "max_cycles": "the cycle limit was reached",
    "not_driven": "no front node is driven by the load",
    "separated": "the next release would leave the laminate in unconnected parts",
    "static": "the cyclic peak reaches the static toughness (λ_c ≤ 1): the "
              "crack grows on the first cycle",
    "arrested": "every front node is below the Paris-law threshold",
}


def _growth_level(growth) -> tuple[str, np.ndarray]:
    """The critical level a growth table and plot show, and its label."""
    if growth.critical_load is not None:
        return "Critical load (N)", np.abs(growth.critical_load)
    return "Critical strain", np.abs(growth.critical_strain)


def growth_summary(growth) -> tuple[list, str]:
    """Metrics and caption of a delamination growth analysis."""
    area = float(growth.released_area[-1])
    rows = [
        ("Release steps", f"{growth.n_steps}",
         f"Stopped because {GROWTH_STOP_TEXT[growth.stop_reason]}."),
        ("Crack area grown", f"{area:.4g} mm²",
         "The tributary areas of the released front nodes."),
    ]
    if growth.crack_lengths is not None:
        rows.append(("Crack length", f"{growth.crack_lengths[0]:.4g} → "
                     f"{growth.crack_lengths[-1]:.4g} mm",
                     "The insert plus the grown area over the specimen width."))
    if growth.paris is None:
        rows.append(("λ_c first → last", f"{growth.load_factors[0]:.4g} → "
                     f"{growth.load_factors[-1]:.4g}",
                     "Critical load factor of the first and last crack: the "
                     "multiple of the analysis load that grows it."))
        control = "displacement" if growth.critical_displacement is not None \
            or growth.critical_strain is not None else "load"
        if growth.n_steps:
            unstable = growth.unstable(control)
            rows.append((f"Unstable steps ({control} control)",
                         f"{int(unstable.sum())} of {unstable.size}",
                         "A step is unstable when the next crack needs no more "
                         "load (or displacement) than this one: it runs on."))
    else:
        rows.append(("Cycles", f"{growth.cycles[-1]:.4g}",
                     "Cycles to grow the last crack, under the cyclic peak."))
        rows.append(("Largest da/dN", f"{float(np.max(growth.growth_rate)):.3g} mm/cycle",
                     "The fastest growth rate over the front, over all steps."))
    parts = [f"Stopped after {growth.n_steps} step"
             f"{'' if growth.n_steps == 1 else 's'}: "
             f"{GROWTH_STOP_TEXT[growth.stop_reason]}."]
    if growth.paris is not None:
        law = growth.paris
        parts.append(
            f"Paris law da/dN = C (G_max / G_c)^m with your C = {law.C:g} mm/cycle, "
            f"m = {law.m:g}" + (f", threshold {law.threshold:g}" if law.threshold
                                 else "") +
            ": no coefficients ship, and G_c is the toughness at each front "
            "node, knocked down by its porosity, so the porosity speeds the "
            "growth through G_c alone.")
    parts.append(
        "Each step releases the governing front nodes and solves the grown "
        "crack: linear elastic between releases, no cohesive zone, resolved to "
        "the element length at the front (VCCT assumes equal elements on both "
        "sides of it). A three-dimensional front advances node by node, so "
        "between straight positions its critical load dips: those steps are "
        "unstable, and the envelope is the load that started the line.")
    if growth.toughness_uncalibrated:
        parts.append("The toughness knockdown is the uncalibrated placeholder law.")
    return rows, " ".join(parts)


def growth_table(growth) -> list[dict]:
    """One row per crack configuration of a growth analysis."""
    label, level = _growth_level(growth)
    rows = []
    for k in range(len(growth)):
        row: dict = {"Step": k}
        if growth.crack_lengths is not None:
            row["Crack length (mm)"] = f"{growth.crack_lengths[k]:.4g}"
        row["Area grown (mm²)"] = f"{growth.released_area[k]:.4g}"
        lam = growth.load_factors[k]
        row["λ_c"] = f"{lam:.4g}" if math.isfinite(lam) else "∞"
        row[label] = f"{level[k]:.4g}" if math.isfinite(level[k]) else "∞"
        if growth.critical_displacement is not None:
            row["Critical displacement (mm)"] = f"{growth.critical_displacement[k]:.4g}"
        if growth.cycles is not None:
            row["Cycles"] = f"{growth.cycles[k]:.4g}"
            row["da/dN (mm/cycle)"] = f"{growth.growth_rate[k]:.3g}"
        vp = growth.Vp[k]
        row["Vp at front (%)"] = f"{100.0 * vp:.3g}" if math.isfinite(vp) else "n/a"
        row["Released"] = int(growth.released[k].size)
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Sensitivity rankings and the FE uncertainty study
# ----------------------------------------------------------------------

#: Readable names of the inputs a sensitivity ranking lists.
SENSITIVITY_INPUT_LABELS = {
    "Vp": "Specimen-average Vp",
    "Vp_peak": "Peak local Vp of the map",
    "Vp_spatial_cov": "In-plane CoV of the map",
    "judd_wright_alpha": "Judd-Wright coefficient α",
    "power_law_n": "Power-law exponent n",
    "linear_beta": "Linear coefficient β",
    "b_I": "Toughness coefficient b_I",
    "b_II": "Toughness coefficient b_II",
}


def sensitivity_rows(ranking) -> list[dict]:
    """Table rows of a ranking: a :class:`SensitivityRanking` or its
    ``summary()``; empty for ``None``."""
    if ranking is None:
        return []
    if not isinstance(ranking, dict):
        ranking = ranking.summary()
    return [{"Input": SENSITIVITY_INPUT_LABELS.get(row["input"], row["input"]),
             "Share of the scatter": f"{100.0 * row['share']:.0f} %",
             "Spearman ρ": f"{row['spearman']:+.2f}",
             "SRRC": f"{row['srrc']:+.2f}"}
            for row in ranking["ranking"]]


def sensitivity_caption(ranking) -> str:
    """How to read a ranking table, with its R²."""
    if ranking is None:
        return ("Nothing to rank: the output does not vary, or too few draws "
                "scatter it.")
    if not isinstance(ranking, dict):
        ranking = ranking.summary()
    return (f"Ranked by the standardized rank regression coefficient (SRRC) over "
            f"{ranking['n_samples']} draws; the share is its square over the sum. "
            f"The sign of ρ is the direction of the effect. R² = {ranking['r2']:.2f}: "
            f"the share of the output's rank scatter the inputs explain together "
            f"(well below 1, the effect is not monotone in one input at a time).")


def fe_uq_summary(study) -> tuple[list, str]:
    """Metrics and caption of an FE uncertainty study."""
    fpf = study.statistics("first_ply_failure_load_factor")
    kd = study.statistics("knockdown")
    nominal = study.nominal.get("first_ply_failure_load_factor")

    def pct(value) -> str:
        return "n/a" if value is None else f"{value:.4g}"

    rows = [
        ("First-ply failure p5", pct(fpf["percentiles"].get("p5")),
         "The 5th percentile of the first-ply-failure load factor over the draws."),
        ("p50", pct(fpf["percentiles"].get("p50")), None),
        ("p95", pct(fpf["percentiles"].get("p95")), None),
        ("Nominal (uniform in-plane)", pct(nominal),
         "The same average void content with no in-plane variation."),
        ("Stiffness knockdown mean", f"{kd['mean']:.4f}",
         f"Nominal {study.nominal['knockdown']:.4f}: at a fixed average a stiffness "
         f"barely moves when the voids are rearranged."),
    ]
    field = study.field
    caption = (
        f"{study.n_samples} FE solves, each with its own random in-plane map of "
        f"void content: lognormal, CoV {field['cov']:g} from point to point, "
        f"correlation length {field['correlation_length_mm']} mm "
        f"({field['kernel'].replace('_', ' ')} kernel), the specimen average kept at "
        f"{100.0 * study.Vp:.3g} %. The through-thickness distribution, layup, mesh "
        f"and load are the analysis'. The FE strength rests on the uncalibrated "
        f"sqrt(stiffness retention) heuristic, so read the spread, not the level. "
        f"Seeded: the same inputs give the same draws.")
    return rows, caption


# ----------------------------------------------------------------------
# Open-hole notch
# ----------------------------------------------------------------------

def open_hole_summary(out: dict, plain_fe=None, *, d0: float | None = None,
                      unnotched_strength: float | None = None) -> tuple[list, str]:
    """Metrics and caption of an open-hole run (:func:`run_open_hole`), with
    the analysis' plain-coupon FE solve ``plain_fe`` to compare."""
    a, fe = out["analysis"], out["fe"]
    g = a.geometry
    rows = [
        ("K_T (FE)", f"{a.stress_concentration:.3f}",
         "Laminate-average axial stress at the hole edge over the gross far-field "
         "stress."),
        ("K_T (Lekhnitskii-Tan)", f"{a.reference_stress_concentration:.3f}",
         "Orthotropic infinite-plate value of the pristine laminate with Tan's "
         "finite-width correction."),
    ]
    fpf = fe.first_ply_failure_load_factor
    if fpf is not None and math.isfinite(fpf):
        plain = getattr(plain_fe, "first_ply_failure_load_factor", None)
        rows.append(("First-ply failure load factor", f"{fpf:.4g}",
                     "Of the coupon with the hole (the uncalibrated FE strength "
                     "heuristic)" + ("" if plain is None else
                                     f"; {plain:.4g} without the hole.")))
    if d0 is not None:
        factor = a.point_stress_factor(d0)
        rows.append((f"Point-stress factor at {d0:g} mm", f"{factor:.4f}",
                     "Notched over unnotched strength by the Whitney-Nuismer "
                     "point-stress criterion."))
        if unnotched_strength is not None:
            rows.append(("Notched strength", f"{factor * unnotched_strength:.4g} MPa",
                         "The unnotched laminate strength you entered times the "
                         "point-stress factor."))
    caption = (
        f"Open-hole coupon {g.width:g} mm wide with a {g.hole_diameter:g} mm hole "
        f"(D/W = {g.diameter_to_width:.3g}), modelled {g.length:g} mm long and gripped "
        f"at the ends, with the analysis' material, layup, porosity and load. The mesh "
        f"is mapped around the hole; the stress concentration of a strongly "
        f"orthotropic laminate converges slowly with it (refine in the library), while "
        f"the stress a millimetre from the edge, which the notch criteria read, has "
        f"converged. The characteristic distance d0 is the material's, fitted to "
        f"notched tests: none ships.")
    return rows, caption


def _stack_names(stack) -> list[str]:
    """The name of each distinct material of a ply stack, as its spec gives it."""
    from porosity_fe import MATERIALS
    return [next((name for name, preset in MATERIALS.items() if preset == m), f"material {i}")
            for i, m in enumerate(stack.materials)]


def hybrid_layer_failure(out: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(z_mid, max failure index, material id)`` of every element layer of a
    hybrid run (:func:`~porosity_app.runners.run_hybrid`), bottom to top."""
    mesh, fe = out["mesh"], out["fe"]
    fi = np.asarray(fe.per_element_failure_index, dtype=float).reshape(mesh.nz, -1)
    z = np.asarray(mesh.nodes, dtype=float)[:mesh.n_structured_nodes, 2]
    layers = np.unique(np.round(z, 12))
    ids = np.asarray(mesh.elem_material_ids).reshape(mesh.nz, -1)[:, 0]
    return 0.5 * (layers[1:] + layers[:-1]), fi.max(axis=1), ids


def hybrid_summary(out: dict, plain_fe=None) -> tuple[list, str]:
    """Metrics and caption of a hybrid laminate run, with the analysis'
    single-material FE solve ``plain_fe`` to compare."""
    stack, fe = out["stack"], out["fe"]
    names = _stack_names(stack)
    rows = []
    modulus = fe.effective_modulus
    if modulus is not None and math.isfinite(modulus):
        plain = getattr(plain_fe, "effective_modulus", None)
        rows.append(("Effective modulus", f"{modulus:,.0f} MPa",
                     "Of the hybrid coupon with its porosity" + (
                         "" if plain is None else f"; {plain:,.0f} MPa for the analysis' "
                         "single-material laminate.")))
    clt = out.get("clt_modulus")
    if clt is not None:
        rows.append(("Lamination theory", f"{clt:,.0f} MPa",
                     "E_x of the same porous stack from lamination theory: the FE value "
                     "sits a little below it, by the free-edge effect of a finite width."))
    rows.append(("Stiffness knockdown", f"{fe.knockdown:.4f}",
                 "Against the same hybrid without voids."))
    fpf = fe.first_ply_failure_load_factor
    if fpf is not None and math.isfinite(fpf):
        plain = getattr(plain_fe, "first_ply_failure_load_factor", None)
        rows.append(("First-ply failure load factor", f"{fpf:.4g}",
                     "Each element against its own material's strengths (the "
                     "uncalibrated FE strength heuristic)" + (
                         "" if plain is None else f"; {plain:.4g} for the analysis' "
                         "single-material laminate.")))
    _, layer_fi, layer_ids = hybrid_layer_failure(out)
    worst = int(np.argmax(layer_fi))
    rows.append(("Highest failure index in", names[int(layer_ids[worst])],
                 f"Element layer {worst + 1} of {len(layer_fi)} from the bottom, "
                 f"index {layer_fi[worst]:.3g}."))
    caption = (
        f"Hybrid laminate of {stack.n_plies} plies, {stack.total_thickness:g} mm: "
        f"{stack.spec} (bottom to top), with the analysis' layup, porosity, load and "
        f"in-plane mesh, one element layer per ply. Every element takes its ply's "
        f"stiffness, porosity micromechanics and strengths. The empirical knockdown "
        f"laws act on one material's strength, so a hybrid has an FE result only.")
    return rows, caption
