"""Reporting helpers shared by ``app.py`` and the test suite.

These functions used to live at module scope inside ``app.py``, but ``app.py``
imports ``streamlit`` at module load. Pulling the streamlit-free helpers into
a dedicated module lets the core test matrix import them (``from
porosity_fe.reporting import build_ncr_record``) without dragging in the
optional ``web`` extra (#155).

Two layers of functionality live here:

* The export payload writers (JSON / CSV) used by the Results / Export tabs.
* The Nonconformance Report (NCR) builder, plus its JSON / Markdown / PDF
  serialisers, which support the MRB disposition workflow. An NCR
  indication is porosity (dispositioned on the void content and the
  governing knockdown), a delamination (on its fracture margin from an FE
  solve with toughness data) or both (:data:`INDICATION_TYPES`).

The module deliberately keeps its top-level imports cheap. ``matplotlib`` is
only needed by :func:`serialise_ncr_pdf`, so it is imported lazily inside
that function.
"""

from __future__ import annotations

import csv
import datetime
import io
import math
import numbers
import re
import textwrap


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_layup(text: str) -> list:
    """Parse a layup string like '[0/45/-45/90]_3s' to a flat angle list.

    Raises ValueError on malformed input rather than silently substituting
    a default — see issue #9.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Layup string is empty.")

    cleaned = text.replace("[", "").replace("]", "")

    symmetric = cleaned.endswith("s")
    if symmetric:
        cleaned = cleaned[:-1]

    repeat = 1
    if "_" in cleaned:
        cleaned, repeat_token = cleaned.rsplit("_", 1)
        try:
            repeat = int(repeat_token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid repeat count {repeat_token!r} in layup {text!r}; "
                f"expected an integer after the '_' (e.g. '[0/90]_3s')."
            ) from exc
        if repeat < 1:
            raise ValueError(
                f"Repeat count must be >= 1, got {repeat} in layup {text!r}."
            )

    sep = "/" if "/" in cleaned else ","
    tokens = [a.strip() for a in cleaned.split(sep) if a.strip()]
    if not tokens:
        raise ValueError(f"No ply angles found in layup {text!r}.")
    try:
        angles = [float(a) for a in tokens]
    except ValueError as exc:
        bad = next((a for a in tokens if not _is_float(a)), tokens[0])
        raise ValueError(
            f"Invalid ply angle {bad!r} in layup {text!r}; "
            f"expected numeric degrees (e.g. '[0/45/-45/90]_3s')."
        ) from exc

    angles = angles * repeat
    if symmetric:
        angles = angles + list(reversed(angles))
    return angles


_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\s\x00-\x1f]')


def _sanitise_filename_component(value: str) -> str:
    """Replace filesystem-unfriendly characters in a filename fragment.

    Slashes and spaces are the common offenders in material codes and
    user-typed NCR references (e.g. ``"T800/epoxy"``, ``"NCR 12"``). They,
    the other characters Windows forbids (``\\ : * ? " < > |``), other
    whitespace and control characters all become underscores, so the
    resulting filename is portable across Windows, macOS and Linux.
    """
    return _UNSAFE_FILENAME_CHARS.sub("_", str(value))


def download_filename_stem(payload: dict) -> str:
    """Build a templated download filename stem from the export payload.

    Encodes material, fibre-volume-porosity and today's date so users who
    run many analyses do not collide on generic names like
    ``porosity_results.json`` (#130). The returned stem has no extension —
    callers append ``.json``/``.csv``/``.pdf`` etc.
    """
    cfg = payload["config"]
    Vp_pct = float(cfg.get("Vp_percent", cfg.get("Vp", 0.0) * 100))
    material = _sanitise_filename_component(cfg.get("material", "unknown"))
    today = datetime.date.today().strftime("%Y%m%d")
    return f"porosity_{material}_Vp{Vp_pct:.1f}pct_{today}"


def build_export_payload(result: dict) -> dict:
    """Flatten an analysis result into the export payload structure.

    Shared by the JSON and CSV writers so both formats describe the same
    fields. ``result`` is the dict produced by ``run_analysis``.
    """
    cfg = result["config"]
    emp = result["empirical"]
    payload = {
        "config": {
            "material": cfg["material_name"],
            "n_plies": cfg["n_plies"],
            "t_ply": cfg["t_ply"],
            "Vp_percent": cfg["Vp"],
            "distribution": cfg["distribution"],
            "void_shape": cfg["void_shape"],
            "mesh": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
        },
        "empirical": {},
    }
    for mode in emp:
        payload["empirical"][mode] = {}
        for model in emp[mode]:
            r = emp[mode][model]
            payload["empirical"][mode][model] = {
                "failure_stress_MPa": r["failure_stress"],
                "knockdown": r["knockdown"],
            }
    return payload


def write_results_json(filepath: str, payload: dict) -> None:
    text = _serialise_payload_json(payload)   # raises before the file is opened
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def _format_csv_row(mode: str, model: str, r: dict) -> list:
    """Format a single empirical row for CSV export with engineering precision.

    Raw Python floats round-trip with full 15-digit precision through
    ``csv.writer``, which produces noisy spreadsheet/script-consumer
    output (#128). Apply explicit per-field precision instead:

    * ``failure_stress_MPa`` -> 1 decimal place (engineering MPa)
    * ``knockdown`` -> 4 decimal places (dimensionless ratio)
    """
    return [
        mode, model,
        f"{r['failure_stress_MPa']:.1f}",
        f"{r['knockdown']:.4f}",
    ]


def write_results_csv(filepath: str, payload: dict) -> None:
    """Write the export payload as a flat CSV.

    Configuration metadata is written as comment lines prefixed with ``#``;
    pandas (``read_csv(comment='#')``) and most CSV viewers handle this
    cleanly, while Excel ignores the comments and treats the table as data.
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        for key, value in payload["config"].items():
            f.write(f"# {key}: {value}\n")
        writer = csv.writer(f)
        writer.writerow(["mode", "model", "failure_stress_MPa", "knockdown"])
        for mode in payload["empirical"]:
            for model in payload["empirical"][mode]:
                r = payload["empirical"][mode][model]
                writer.writerow(_format_csv_row(mode, model, r))


def _serialise_payload_json(payload: dict) -> str:
    """Standard JSON (no NaN / Infinity) of the app's empirical export payload."""
    from porosity_fe import FORMAT_EMPIRICAL_SWEEP
    from porosity_fe.io import _UNITS_STREAMLIT_EMPIRICAL, _dumps_json, _wrap_envelope
    envelope = _wrap_envelope(FORMAT_EMPIRICAL_SWEEP, _UNITS_STREAMLIT_EMPIRICAL, payload)
    return _dumps_json(envelope)


def _serialise_payload_csv(payload: dict) -> str:
    buf = io.StringIO()
    for key, value in payload["config"].items():
        buf.write(f"# {key}: {value}\n")
    writer = csv.writer(buf)
    writer.writerow(["mode", "model", "failure_stress_MPa", "knockdown"])
    for mode in payload["empirical"]:
        for model in payload["empirical"][mode]:
            r = payload["empirical"][mode][model]
            writer.writerow(_format_csv_row(mode, model, r))
    return buf.getvalue()


# ======================================================================
# Nonconformance Report (NCR) — MRB disposition support
# ======================================================================
#
# Lets a field engineer turn a porosity analysis into a structured NCR that
# an MRB can review. The tool produces *analysis and a recommended
# disposition path* — it never issues a final disposition. The governing
# (worst-case) knockdown across all modes/models drives the recommendation,
# because the MRB cares about the most-critical residual strength, not the
# average. A delamination indication is dispositioned on its fracture
# margin instead (the FE growth-onset load factor against the required load
# factor the user enters); no published acceptance numbers exist for it.

STRUCTURAL_CLASSES = ("primary", "secondary", "non-structural")

#: NDI indication types of an NCR: porosity alone, a delamination (a planar
#: interlaminar crack), or a delamination in a porous laminate.
INDICATION_TYPES = ("porosity", "delamination", "combined")

#: Disposition paths, least to most severe. A combined indication takes the
#: more severe of its porosity and delamination paths.
_PATH_SEVERITY = (
    "Use-As-Is (UAI) — pending MRB concurrence",
    "Use-As-Is with Engineering Evaluation",
    "Engineering Evaluation / Repair",
    "Repair or Scrap",
)


def governing_failure(result: dict) -> dict:
    """Most-critical empirical case (lowest knockdown) across modes/models.

    The MRB evaluates the worst-case residual strength, so the
    disposition recommendation keys off the minimum knockdown rather than
    a per-mode average.
    """
    emp = result["empirical"]
    worst = None
    for mode in emp:
        for model in emp[mode]:
            r = emp[mode][model]
            kd = r["knockdown"]
            if worst is None or kd < worst["knockdown"]:
                worst = {
                    "mode": mode,
                    "model": model,
                    "knockdown": kd,
                    "residual_strength_MPa": r["failure_stress"],
                }
    if worst is None:
        raise ValueError("Analysis result has no empirical knockdown data.")
    return worst


def _porosity_path(Vp: float, governing_knockdown: float) -> tuple[str, str]:
    """Disposition path and rationale of a porosity indication: bins on the
    void content (percent) and the governing knockdown."""
    if Vp <= 1.0 and governing_knockdown >= 0.95:
        path = _PATH_SEVERITY[0]
        rationale = (
            f"Measured void content ({Vp:.2f}%) is within the porosity range "
            f"typical of autoclave-grade primary structure, and the predicted "
            f"governing residual strength retains "
            f"{governing_knockdown * 100:.1f}% of pristine. Strength loss is "
            f"minor and likely covered by as-designed margin."
        )
    elif Vp <= 2.0 and governing_knockdown >= 0.90:
        path = _PATH_SEVERITY[1]
        rationale = (
            f"Void content ({Vp:.2f}%) is marginal against typical drawing "
            f"allowables and the governing knockdown "
            f"({governing_knockdown * 100:.1f}% retained) indicates moderate "
            f"strength loss. UAI may be substantiated by a positive "
            f"margin-of-safety check at the knockdown-adjusted allowable."
        )
    elif Vp <= 5.0 and governing_knockdown >= 0.80:
        path = _PATH_SEVERITY[2]
        rationale = (
            f"Void content ({Vp:.2f}%) exceeds porosity allowables typical of "
            f"primary structure and the governing knockdown leaves only "
            f"{governing_knockdown * 100:.1f}% of pristine strength. "
            f"Disposition hinges on the as-designed margin and on whether an "
            f"approved repair can restore a serviceable condition."
        )
    else:
        path = _PATH_SEVERITY[3]
        rationale = (
            f"Void content ({Vp:.2f}%) and the predicted governing knockdown "
            f"({governing_knockdown * 100:.1f}% retained) represent a severe "
            f"degradation. Use-As-Is is not recommended without exceptional, "
            f"test-backed substantiation; repair or scrap is the expected path."
        )
    return path, rationale


_POROSITY_CRITERIA = (
    "Released engineering drawing / part specification porosity allowable "
    "(governing acceptance limit — verify against the controlled drawing).",
    "Process specification cure / void-content requirements for the "
    "applicable material system.",
    "Structural substantiation: laminate margin of safety recomputed at "
    "the knockdown-adjusted allowable for the governing loading mode.",
    "NDI acceptance criteria (ultrasonic C-scan attenuation / void "
    "content) per the applicable NDT specification.",
)

_DELAMINATION_CRITERIA = (
    "Released engineering drawing / part specification acceptance limit for "
    "delaminations and disbonds: size, position and depth of a planar NDI "
    "indication (governing acceptance limit — verify against the controlled "
    "drawing).",
    "Damage-tolerance substantiation: no delamination growth at the "
    "required load factor (linear virtual-crack-closure growth onset, "
    "Benzeggagh-Kenane mixed-mode criterion), with interlaminar toughness "
    "measured for the material system and process.",
    "NDI acceptance criteria for planar indications (ultrasonic C-scan "
    "amplitude and time-of-flight) per the applicable NDT specification.",
)

_MRB_PROCEDURE = (
    "Quality-system MRB procedure for disposition of nonconforming "
    "material."
)

_POROSITY_ACTIONS = (
    "Confirm measured void content by micrograph / acid digestion or a "
    "validated C-scan correlation.",
    "Verify the governing porosity allowable on the released engineering "
    "drawing / specification.",
    "Perform structural substantiation: recompute the margin of safety "
    "using the knockdown-adjusted allowable for the governing mode.",
    "Define the NDI extent and map the affected zone / part region.",
)

_DELAMINATION_ACTIONS = (
    "Confirm the indication by NDI (C-scan amplitude and time-of-flight): "
    "its planform size and position, its interface depth, and that it is a "
    "planar delamination rather than porosity.",
    "Verify the interlaminar toughness used in the analysis (G_Ic, G_IIc, "
    "B-K eta) against data measured for this material system and process; "
    "no material preset carries it.",
    "Confirm the required load factor: it multiplies the same analysis load "
    "as the predicted fracture load factor, and must come from the "
    "structural substantiation of the governing load case.",
    "Assess what the linear growth-onset analysis does not cover: fatigue "
    "growth, sublaminate buckling under compression, environment, and the "
    "mesh dependence of the energy release rates at the front.",
)

_UNCALIBRATED_ACTION = (
    "The porosity knockdown of the toughness used uncalibrated placeholder "
    "coefficients: substantiate the toughness of the porous interface by "
    "test, or rerun the analysis with calibrated coefficients."
)


def _check_load_factor(name: str, value, *, allow_inf: bool) -> float:
    """``value`` as a float load factor (``> 0``; ``inf`` when allowed)."""
    if value is None:
        raise ValueError(
            f"{name} is required for a delamination or combined indication.")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    v = float(value)
    if math.isnan(v) or v < 0.0 or (v == 0.0 and not allow_inf) or (
            math.isinf(v) and not allow_inf):
        bound = "a number >= 0 (inf when nothing drives the front)" if allow_inf \
            else "a finite number > 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}.")
    return v


def _fracture_margin(fracture_load_factor, required_load_factor,
                     toughness_uncalibrated: bool) -> dict:
    """The fracture margin of a delamination: ``lambda_c`` against the
    required load factor (both multiplying the analysis load)."""
    lam = _check_load_factor("fracture_load_factor", fracture_load_factor,
                             allow_inf=True)
    req = _check_load_factor("required_load_factor", required_load_factor,
                             allow_inf=False)
    finite = math.isfinite(lam)
    return {
        "fracture_load_factor": lam if finite else None,
        "required_load_factor": req,
        "margin_of_safety": lam / req - 1.0 if finite else None,
        "meets_required": bool(lam >= req),
        "toughness_uncalibrated": bool(toughness_uncalibrated),
    }


def _delamination_path(margin: dict) -> tuple[str, str]:
    """Disposition path and rationale of a delamination from its fracture
    margin, never from the void content.

    The only threshold is the user's required load factor: a margin that
    meets it is still a crack in the part, analysed for growth onset only,
    so the path is never a plain Use-As-Is.
    """
    lam = margin["fracture_load_factor"]
    req = margin["required_load_factor"]
    ms = margin["margin_of_safety"]
    if margin["meets_required"]:
        path = _PATH_SEVERITY[1]
        if lam is None:
            rationale = (
                f"The analysis load releases no energy at the delamination "
                f"front, so the predicted growth-onset load factor is "
                f"unbounded and meets the required load factor ({req:.3g}).")
        else:
            rationale = (
                f"The predicted delamination growth-onset load factor "
                f"(lambda_c = {lam:.3g}) meets the required load factor "
                f"({req:.3g}): fracture margin of safety {ms:+.2f}.")
        rationale += (
            " The analysis is linear elastic fracture mechanics (growth onset "
            "under static load; no fatigue growth or delamination buckling), "
            "so Use-As-Is needs the MRB's engineering evaluation of what it "
            "does not cover.")
    else:
        path = _PATH_SEVERITY[3]
        rationale = (
            f"The predicted delamination growth-onset load factor "
            f"(lambda_c = {lam:.3g}) is below the required load factor "
            f"({req:.3g}): fracture margin of safety {ms:+.2f}. The "
            f"delamination is predicted to grow before the required load is "
            f"reached; Use-As-Is is not recommended without test-backed "
            f"substantiation, and repair or scrap is the expected path.")
    if margin["toughness_uncalibrated"]:
        rationale += (
            " The toughness at the front was knocked down by the porosity "
            "there with uncalibrated placeholder coefficients.")
    return path, rationale


def recommend_disposition(
    Vp_percent: float, governing_knockdown: float, structural_class: str = "primary",
    *, indication: str = "porosity", fracture_load_factor: float | None = None,
    required_load_factor: float | None = None, toughness_uncalibrated: bool = False,
) -> dict:
    """Recommend (not decide) an MRB disposition path for a porosity NCR.

    The recommendation bins on the measured void content and the governing
    (worst-case) knockdown, then escalates required substantiation by
    structural class. It is deliberately conservative: the released
    engineering drawing / process spec is the governing acceptance
    authority, and the MRB must substantiate against it.

    A delamination indication (``indication='delamination'``) is
    dispositioned on its **fracture margin** instead, never on the void
    content: the predicted growth-onset load factor ``lambda_c`` of an FE
    solve with the delamination (:attr:`FieldResults.fracture_load_factor
    <porosity_fe.FieldResults.fracture_load_factor>`) against the
    ``required_load_factor`` the user supplies. No published acceptance
    numbers exist for delaminations (they are OEM- and part-specific), so
    the one threshold is that margin: ``lambda_c >= required_load_factor``
    recommends "Use-As-Is with Engineering Evaluation", below it "Repair or
    Scrap". ``indication='combined'`` (a delamination in a porous laminate)
    takes the more severe of the porosity path and the delamination path.

    Parameters
    ----------
    Vp_percent : float
        Measured void content **in percent**, as reported on an NCR:
        ``3.0`` means 3 % voids. This is the one porosity input in the
        package that takes a percent rather than a fraction; passing the
        fraction ``0.03`` would be read as 0.03 % and recommend Use-As-Is.
        Validated for every indication; it sets the path of the porosity
        and combined indications only.
    governing_knockdown : float
        Worst-case strength knockdown, a fraction in ``(0, 1]``. Sets the
        path of the porosity and combined indications only.
    structural_class : str
        One of :data:`STRUCTURAL_CLASSES`.
    indication : {'porosity', 'delamination', 'combined'}
        Keyword-only NDI indication type (:data:`INDICATION_TYPES`), default
        ``'porosity'``, which returns exactly what this function returned
        before the other types existed.
    fracture_load_factor : float, optional
        Keyword-only. The predicted growth-onset load factor ``lambda_c``
        (dimensionless, ``>= 0``; ``inf`` when the analysis load releases
        no energy at the front). Required for ``'delamination'`` and
        ``'combined'``, refused for ``'porosity'``.
    required_load_factor : float, optional
        Keyword-only. The load factor the part must reach without
        delamination growth, a finite number ``> 0`` that multiplies the
        **same analysis load** as ``fracture_load_factor`` (for an analysis
        at 1 % strain and a design strain of 0.4 %, 0.4). It comes from the
        structural substantiation; there is no default. Required for
        ``'delamination'`` and ``'combined'``, refused for ``'porosity'``.
    toughness_uncalibrated : bool
        Keyword-only. Whether the toughness behind ``fracture_load_factor``
        was knocked down by porosity with uncalibrated placeholder
        coefficients (:attr:`CrackFront.toughness_uncalibrated
        <porosity_fe.CrackFront.toughness_uncalibrated>`); it adds a
        required MRB action and a sentence to the rationale, and does not
        change the path.

    Returns
    -------
    dict
        ``path``, ``structural_class``, ``rationale``, ``cited_criteria``,
        ``required_mrb_actions`` and ``disclaimer``. For ``'delamination'``
        and ``'combined'`` also ``indication``, ``fracture_margin``
        (``fracture_load_factor`` and ``margin_of_safety =
        lambda_c / required - 1``, both ``None`` when ``lambda_c`` is
        infinite, ``required_load_factor``, ``meets_required``,
        ``toughness_uncalibrated``) and ``delamination_path``; for
        ``'combined'`` also ``porosity_path``.

    Raises
    ------
    ValueError
        If ``Vp_percent`` is not a finite value in ``[0, 100]``,
        ``structural_class`` is not one of :data:`STRUCTURAL_CLASSES`,
        ``indication`` is not one of :data:`INDICATION_TYPES`, a fracture
        input is missing or invalid for a delamination or combined
        indication, or one is given for a porosity indication.
        (An unknown class used to fall back to ``"primary"`` silently,
        which in an MRB record would misstate the substantiation basis.)
    """
    if not (0.0 <= Vp_percent <= 100.0):  # also rejects NaN
        raise ValueError(
            f"Vp_percent must be the void content in percent, in [0, 100]; "
            f"got {Vp_percent!r}."
        )
    Vp = float(Vp_percent)
    if structural_class not in STRUCTURAL_CLASSES:
        raise ValueError(
            f"Unknown structural_class {structural_class!r}; "
            f"expected one of {list(STRUCTURAL_CLASSES)}."
        )
    if indication not in INDICATION_TYPES:
        raise ValueError(
            f"Unknown indication {indication!r}; expected one of "
            f"{list(INDICATION_TYPES)}."
        )
    if indication == "porosity":
        if fracture_load_factor is not None or required_load_factor is not None:
            raise ValueError(
                "fracture_load_factor and required_load_factor apply to a "
                "'delamination' or 'combined' indication; a porosity "
                "indication is dispositioned on Vp_percent and "
                "governing_knockdown.")
        margin = None
        path, rationale = _porosity_path(Vp, governing_knockdown)
    else:
        margin = _fracture_margin(fracture_load_factor, required_load_factor,
                                  toughness_uncalibrated)
        delamination_path, rationale = _delamination_path(margin)
        path = delamination_path
        if indication == "combined":
            porosity_path, porosity_rationale = _porosity_path(
                Vp, governing_knockdown)
            path = max(porosity_path, delamination_path, key=_PATH_SEVERITY.index)
            rationale = (f"Porosity: {porosity_rationale} Delamination: "
                         f"{rationale} The more severe path governs.")

    cited_criteria = []
    required_mrb_actions = []
    if indication in ("porosity", "combined"):
        cited_criteria += _POROSITY_CRITERIA
        required_mrb_actions += _POROSITY_ACTIONS
    if indication in ("delamination", "combined"):
        cited_criteria += _DELAMINATION_CRITERIA
        required_mrb_actions += _DELAMINATION_ACTIONS
        if toughness_uncalibrated:
            required_mrb_actions.append(_UNCALIBRATED_ACTION)
    cited_criteria.append(_MRB_PROCEDURE)

    if structural_class == "primary":
        required_mrb_actions.append(
            "Primary structure: obtain customer / DER engineering concurrence "
            "before approving any Use-As-Is disposition."
        )
    elif structural_class == "non-structural":
        affected = {"porosity": "the porosity", "delamination": "the delamination",
                    "combined": "the porosity or the delamination"}[indication]
        required_mrb_actions.append(
            "Non-structural item: confirm there is no fluid-ingress, "
            f"fatigue, or interface-sealing function affected by {affected}."
        )
    if path.startswith("Repair") or "Repair" in path:
        required_mrb_actions.append(
            "If repair is selected, document the approved repair scheme and "
            "the post-repair re-inspection requirements."
        )

    what = {"porosity": "a predictive porosity-knockdown analysis",
            "delamination": "a predictive delamination fracture analysis",
            "combined": "a predictive porosity-knockdown and delamination "
                        "fracture analysis"}[indication]
    disclaimer = (
        "This recommended disposition path was produced by an automated MRB "
        f"support tool from {what}. It is "
        "NOT a final disposition. A qualified Material Review Board must "
        "independently review, may modify, and must formally approve the "
        "disposition. Final acceptance requires substantiation against the "
        "governing engineering drawing / specification and the applicable "
        "structural margins."
    )

    out: dict = {
        "path": path,
        "structural_class": structural_class,
        "rationale": rationale,
        "cited_criteria": cited_criteria,
        "required_mrb_actions": required_mrb_actions,
        "disclaimer": disclaimer,
    }
    if margin is not None:
        out["indication"] = indication
        out["fracture_margin"] = margin
        out["delamination_path"] = delamination_path
        if indication == "combined":
            out["porosity_path"] = porosity_path
    return out


def _finite_or_none(value) -> float | None:
    """``value`` as a float, ``None`` when missing or not finite (JSON null)."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _planform_record(delamination, mesh) -> dict:
    """Interface depth and planform (mm) of one delamination of ``mesh``.

    The area is that of the planform as given: a strip spans the width of
    the plate, clipped to its length; a rectangle or ellipse is not clipped.
    """
    k = int(delamination.interface)
    n_plies = int(mesh.material.n_plies)
    h = float(mesh.L_z)
    z = (k + 1) * h / n_plies
    rec = {
        "interface": k,
        "between_plies": [k + 1, k + 2],
        "height_from_bottom_mm": z,
        "depth_from_top_mm": h - z,
        "shape": delamination.shape,
    }
    if delamination.shape == "strip":
        x0, x1 = float(delamination.x_start), float(delamination.x_end)
        rec.update({"x_start_mm": x0, "x_end_mm": x1})
        length = max(0.0, min(x1, float(mesh.L_x)) - max(x0, 0.0))
        rec["area_mm2"] = length * float(mesh.L_y)
    else:
        (xc, yc), (ax, ay) = delamination.center, delamination.half_size
        rec.update({"center_x_mm": float(xc), "center_y_mm": float(yc),
                    "length_x_mm": 2.0 * float(ax), "width_y_mm": 2.0 * float(ay)})
        rec["area_mm2"] = (math.pi if delamination.shape == "ellipse" else 4.0) \
            * float(ax) * float(ay)
    return rec


def _governing_front_node(front) -> dict | None:
    """The front node with the smallest critical load factor, or ``None``."""
    row = front.governing
    if row is None:
        return None
    return {
        "interface": int(front.interface[row]),
        "x_mm": float(front.x[row]),
        "y_mm": float(front.y[row]),
        "z_mm": float(front.z[row]),
        "psi": _finite_or_none(front.psi[row]),
        "G_T_N_per_mm": float(front.G_T[row]),
        "G_c_N_per_mm": None if front.G_c is None else _finite_or_none(front.G_c[row]),
        "Vp": float(front.Vp[row]),
    }


def _delamination_inputs(result: dict) -> tuple:
    """The FE field, its crack front and the mesh of a delamination NCR.

    Raises
    ------
    ValueError
        Without an FE solve of a mesh with a delamination front, for a
        fracture test specimen, or without toughness data.
    """
    fe = result.get("fe_field")
    mesh = result.get("mesh")
    front = None if fe is None else getattr(fe, "crack_front", None)
    if fe is None or mesh is None or front is None or len(front) == 0:
        raise ValueError(
            "A delamination or combined NCR needs an FE solve of a mesh with "
            "a delamination front (the app's delamination input).")
    if getattr(mesh, "fracture_geometry", None) is not None:
        raise ValueError(
            "The FE solve is a fracture test specimen (DCB, ENF, MMB), not an "
            "NDI indication; disposition a delamination of the part.")
    if fe.fracture_load_factor is None:
        raise ValueError(
            "The FE solve has no fracture load factor: the material has no "
            "interlaminar toughness (G_Ic, G_IIc, bk_eta), so the delamination "
            "cannot be dispositioned on a fracture margin.")
    return fe, front, mesh


def _delamination_record(result: dict, indication: str,
                         required_load_factor: float) -> dict:
    """The ``delamination`` block of an NCR record (lengths in mm, toughness
    and energy release rates in N/mm, load factors dimensionless)."""
    fe, front, mesh = _delamination_inputs(result)
    material = result.get("material") or mesh.material
    law = front.toughness_knockdown
    decomposition = result.get("fe_decomposition")
    return {
        "indication": indication,
        "delaminations": [_planform_record(d, mesh) for d in mesh.delaminations],
        "analysis_loading": result.get("fe_loading"),
        "load_factor_basis": str(fe.load_factor_basis),
        "toughness_N_per_mm": {
            "G_Ic": _finite_or_none(material.G_Ic),
            "G_IIc": _finite_or_none(material.G_IIc),
            "G_IIIc": _finite_or_none(material.G_IIIc_eff),
            "bk_eta": _finite_or_none(material.bk_eta),
        },
        "toughness_knockdown": None if law is None else law.describe(),
        "toughness_uncalibrated": front.toughness_uncalibrated,
        "fracture_load_factor": _finite_or_none(fe.fracture_load_factor),
        "required_load_factor": float(required_load_factor),
        "n_front_nodes": len(front),
        "governing_front_node": _governing_front_node(front),
        "knockdown_decomposition": (None if decomposition is None
                                    else decomposition.summary()),
    }


def _planform_text(rec: dict) -> str:
    """One planform of a ``delamination`` block in words."""
    if rec["shape"] == "strip":
        shape = (f"through-width strip from x = {rec['x_start_mm']:.1f} to "
                 f"{rec['x_end_mm']:.1f} mm")
    else:
        shape = (f"{rec['shape']} {rec['length_x_mm']:.1f} x "
                 f"{rec['width_y_mm']:.1f} mm centred at "
                 f"({rec['center_x_mm']:.1f}, {rec['center_y_mm']:.1f}) mm")
    return (f"{shape} (area {rec['area_mm2']:.1f} mm²) at interface "
            f"{rec['interface']}, between plies {rec['between_plies'][0]} and "
            f"{rec['between_plies'][1]} from the bottom, "
            f"{rec['depth_from_top_mm']:.2f} mm below the top surface")


def build_ncr_record(result: dict, meta: dict) -> dict:
    """Build an analysis validation summary to attach to an NCR.

    This is the technical attachment, not a full NCR form: it carries the
    porosity analysis, the governing knockdown, and a recommended (not
    final) disposition path. Part/serial/work-order identification lives on
    the parent NCR — ``meta`` only needs who prepared it and an optional
    parent-NCR reference. Everything technical is derived from ``result`` so
    the summary cannot drift from what was actually run.

    ``meta["indication"]`` (default ``"porosity"``, :data:`INDICATION_TYPES`)
    selects the indication type. ``"delamination"`` and ``"combined"`` need
    ``result`` to hold an FE solve of a part with a delamination front and
    toughness data (``result["fe_field"]``, ``result["mesh"]``; the app's
    delamination input) and ``meta["required_load_factor"]``, the load
    factor the part must reach without delamination growth, as a multiple
    of the analysis load (see :func:`recommend_disposition`). They add a
    ``delamination`` block (the interface depth and planform, the
    toughness and its porosity law, the fracture load factor, the governing
    front node and, when ``result["fe_decomposition"]`` holds one, the
    knockdown decomposition) and disposition the delamination on its
    fracture margin. A porosity record is unchanged.
    """
    payload = build_export_payload(result)
    cfg = payload["config"]
    worst = governing_failure(result)
    Vp = float(cfg["Vp_percent"])
    structural_class = meta.get("structural_class", "primary")
    indication = meta.get("indication", "porosity")
    if indication not in INDICATION_TYPES:
        raise ValueError(
            f"Unknown indication {indication!r}; expected one of "
            f"{list(INDICATION_TYPES)}.")

    delamination = None
    if indication == "porosity":
        disposition = recommend_disposition(
            Vp, worst["knockdown"], structural_class
        )
    else:
        required = meta.get("required_load_factor")
        fe, front, _ = _delamination_inputs(result)
        disposition = recommend_disposition(
            Vp, worst["knockdown"], structural_class, indication=indication,
            fracture_load_factor=fe.fracture_load_factor,
            required_load_factor=required,
            toughness_uncalibrated=bool(front.toughness_uncalibrated))
        delamination = _delamination_record(
            result, indication, disposition["fracture_margin"]["required_load_factor"])

    today = datetime.date.today().isoformat()
    layup = meta.get("layup") or "(see analysis configuration)"

    porosity_summary = (
        f"Porosity / void content of {Vp:.2f}% in a "
        f"{cfg['material']} laminate ({cfg['n_plies']} plies, layup "
        f"{layup}); {cfg['distribution']} distribution, "
        f"{cfg['void_shape']} void morphology. Predicted to exceed "
        f"typical drawing porosity allowables and to knock down "
        f"residual strength — see engineering analysis."
    )
    title = "Composite Porosity Analysis — NCR Validation Summary"
    summary = porosity_summary
    if delamination is not None:
        planforms = "; ".join(_planform_text(rec)
                              for rec in delamination["delaminations"])
        delamination_summary = (
            f"Delamination indication: {planforms}. Dispositioned on its "
            f"fracture margin — see the delamination analysis.")
        if indication == "delamination":
            title = "Composite Delamination Analysis — NCR Validation Summary"
            summary = (
                f"{delamination_summary} Laminate: {cfg['material']}, "
                f"{cfg['n_plies']} plies, layup {layup}, analysed with "
                f"{Vp:.2f}% voids ({cfg['distribution']} distribution, "
                f"{cfg['void_shape']} void morphology).")
        else:
            title = ("Composite Porosity and Delamination Analysis — NCR "
                     "Validation Summary")
            summary = f"{porosity_summary} {delamination_summary}"

    record = {
        "summary": {
            "title": title,
            "prepared_by": meta.get("prepared_by", ""),
            "ncr_reference": meta.get("ncr_reference", ""),
            "date": meta.get("date") or today,
            "structural_class": structural_class,
            "note": meta.get("note", ""),
        },
        "nonconformance": {
            "summary": summary,
            "material": cfg["material"],
            "layup": layup,
            "n_plies": cfg["n_plies"],
            "t_ply_mm": cfg["t_ply"],
            "measured_Vp_percent": Vp,
            "distribution": cfg["distribution"],
            "void_shape": cfg["void_shape"],
            "analysis_mesh": cfg["mesh"],
        },
        "engineering_analysis": {
            "governing_mode": worst["mode"],
            "governing_model": worst["model"],
            "governing_knockdown": worst["knockdown"],
            "governing_residual_strength_MPa": worst["residual_strength_MPa"],
            "per_mode": payload["empirical"],
        },
        "recommended_disposition": disposition,
    }
    if delamination is not None:
        record["summary"]["indication"] = indication
        record["delamination"] = delamination
    return record


def serialise_ncr_json(ncr: dict) -> str:
    """Standard JSON (no NaN / Infinity) of an NCR validation summary."""
    from porosity_fe import FORMAT_NCR
    from porosity_fe.io import _UNITS_NCR, _UNITS_NCR_DELAMINATION, _dumps_json, _wrap_envelope
    units = _UNITS_NCR if "delamination" not in ncr \
        else {**_UNITS_NCR, **_UNITS_NCR_DELAMINATION}
    envelope = _wrap_envelope(FORMAT_NCR, units, ncr)
    return _dumps_json(envelope)


def _delamination_rows(ncr: dict) -> list:
    """``(field, value)`` rows of the delamination block, for Markdown and PDF."""
    d = ncr["delamination"]
    margin = ncr["recommended_disposition"]["fracture_margin"]
    t = d["toughness_N_per_mm"]

    def num(v, fmt: str = "{:.3g}") -> str:
        return "n/a" if v is None else fmt.format(v)

    rows = [("Indication type", d["indication"])]
    for i, rec in enumerate(d["delaminations"], start=1):
        label = "Planform" if len(d["delaminations"]) == 1 else f"Planform {i}"
        rows.append((label, _planform_text(rec)))
    rows += [
        ("Analysis loading", f"{d['analysis_loading']} "
                             f"(load factor basis: {d['load_factor_basis']})"),
        ("Toughness G_Ic / G_IIc / G_IIIc (N/mm)",
         f"{num(t['G_Ic'])} / {num(t['G_IIc'])} / {num(t['G_IIIc'])}"),
        ("B-K exponent eta", num(t["bk_eta"])),
        ("Porosity knockdown of the toughness", d["toughness_knockdown"] or "n/a"),
        ("Toughness uncalibrated", "yes" if d["toughness_uncalibrated"] else "no"),
        ("Fracture load factor lambda_c",
         "unbounded (no energy released at the front)"
         if d["fracture_load_factor"] is None else num(d["fracture_load_factor"])),
        ("Required load factor", num(d["required_load_factor"])),
        ("Fracture margin of safety", num(margin["margin_of_safety"], "{:+.2f}")),
    ]
    g = d["governing_front_node"]
    if g is not None:
        rows.append(("Governing front node",
                     f"x = {g['x_mm']:.2f}, y = {g['y_mm']:.2f} mm, psi = "
                     f"{num(g['psi'], '{:.2f}')}, G_T = {g['G_T_N_per_mm']:.4g} "
                     f"N/mm, G_c = {num(g['G_c_N_per_mm'], '{:.4g}')} N/mm, "
                     f"Vp = {g['Vp']:.4f}"))
    kd = d["knockdown_decomposition"]
    if kd is not None:
        rows.append(("Knockdown total / porosity only / delamination only",
                     f"{num(kd['knockdown_total'], '{:.4f}')} / "
                     f"{num(kd['knockdown_porosity_only'], '{:.4f}')} / "
                     f"{num(kd['knockdown_delamination_only'], '{:.4f}')}"))
    return rows


_UNCALIBRATED_NOTE = (
    "Toughness uncalibrated: the toughness at the front was knocked down by "
    "the porosity there with uncalibrated placeholder coefficients."
)


def _per_mode_rows(ea: dict) -> list:
    """Flatten the per-mode/model knockdown table to (mode, model, MPa, kd)."""
    rows = []
    for mode in ea["per_mode"]:
        for model in ea["per_mode"][mode]:
            r = ea["per_mode"][mode][model]
            rows.append((
                mode, model,
                f"{r['failure_stress_MPa']:.1f}",
                f"{r['knockdown']:.3f}",
            ))
    return rows


def serialise_ncr_markdown(ncr: dict) -> str:
    """Render the analysis validation summary as a Markdown attachment."""
    s = ncr["summary"]
    nc = ncr["nonconformance"]
    ea = ncr["engineering_analysis"]
    dp = ncr["recommended_disposition"]

    lines = []
    lines.append(f"# {s['title']}")
    lines.append("")
    lines.append("_Attachment to a Nonconformance Report — analysis validation_")
    lines.append("")
    lines.append(f"- Prepared by: {s['prepared_by'] or '—'}")
    lines.append(f"- Parent NCR reference: {s['ncr_reference'] or '—'}")
    lines.append(f"- Date: {s['date']}")
    lines.append(f"- Structural classification: {s['structural_class']}")
    if s.get("note"):
        lines.append(f"- Engineer note: {s['note']}")
    lines.append("")

    lines.append("## 1. Nonconformance Summary")
    lines.append("")
    lines.append(nc["summary"])
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Material | {nc['material']} |")
    lines.append(f"| Layup | {nc['layup']} |")
    lines.append(f"| Plies | {nc['n_plies']} |")
    lines.append(f"| Ply thickness (mm) | {nc['t_ply_mm']} |")
    lines.append(f"| Measured void content Vp (%) | {nc['measured_Vp_percent']:.2f} |")
    lines.append(f"| Distribution | {nc['distribution']} |")
    lines.append(f"| Void morphology | {nc['void_shape']} |")
    lines.append(f"| Analysis mesh | {nc['analysis_mesh']} |")
    lines.append("")

    lines.append("## 2. Engineering Analysis (predicted porosity knockdown)")
    lines.append("")
    lines.append(
        f"**Governing (worst-case) case:** {ea['governing_mode']} / "
        f"{ea['governing_model']} — knockdown "
        f"{ea['governing_knockdown']:.3f} "
        f"({ea['governing_knockdown'] * 100:.1f}% of pristine retained), "
        f"residual strength "
        f"{ea['governing_residual_strength_MPa']:.1f} MPa."
    )
    lines.append("")
    lines.append("| Mode | Model | Residual strength (MPa) | Knockdown |")
    lines.append("|---|---|---|---|")
    for mode, model, mpa, kd in _per_mode_rows(ea):
        lines.append(f"| {mode} | {model} | {mpa} | {kd} |")
    lines.append("")

    if "delamination" in ncr:
        lines.append("## 2b. Delamination Analysis (fracture margin)")
        lines.append("")
        if ncr["delamination"]["toughness_uncalibrated"]:
            lines.append(f"> **{_UNCALIBRATED_NOTE}**")
            lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for field, value in _delamination_rows(ncr):
            lines.append(f"| {field} | {value} |")
        lines.append("")

    lines.append(
        "## 3. Recommended Disposition Path "
        "(for MRB review — NOT a final disposition)"
    )
    lines.append("")
    lines.append(f"**Recommended path:** {dp['path']}")
    lines.append("")
    if "porosity_path" in dp:
        lines.append(f"- Porosity path: {dp['porosity_path']}")
        lines.append(f"- Delamination path: {dp['delamination_path']}")
        lines.append("")
    lines.append(f"**Rationale:** {dp['rationale']}")
    lines.append("")
    lines.append(f"> {dp['disclaimer']}")
    lines.append("")

    lines.append("## 4. Cited Criteria")
    lines.append("")
    for c in dp["cited_criteria"]:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## 5. Required MRB Actions")
    lines.append("")
    for a in dp["required_mrb_actions"]:
        lines.append(f"- [ ] {a}")
    lines.append("")
    lines.append(
        "_Generated by PorosityFE MRB support tool. This is a predictive "
        "analysis and a recommended disposition path; it does not constitute "
        "MRB approval._"
    )
    lines.append("")
    return "\n".join(lines)


_PDF_WRAP = 92
_PDF_LINES_PER_PAGE = 58


def _ncr_text_lines(ncr: dict) -> list:
    """Plain-text (no Markdown) lines for the PDF, wrapped to a fixed width."""
    s = ncr["summary"]
    nc = ncr["nonconformance"]
    ea = ncr["engineering_analysis"]
    dp = ncr["recommended_disposition"]

    out: list = []

    def para(text: str, indent: str = "") -> None:
        for w in textwrap.wrap(text, _PDF_WRAP - len(indent)) or [""]:
            out.append(indent + w)

    def rule() -> None:
        out.append("-" * _PDF_WRAP)

    out.append(s["title"].upper())
    out.append("Attachment to a Nonconformance Report - analysis validation")
    rule()
    out.append(f"Prepared by:               {s['prepared_by'] or '-'}")
    out.append(f"Parent NCR reference:      {s['ncr_reference'] or '-'}")
    out.append(f"Date:                      {s['date']}")
    out.append(f"Structural classification: {s['structural_class']}")
    if s.get("note"):
        para(f"Engineer note: {s['note']}")
    out.append("")

    out.append("1. NONCONFORMANCE SUMMARY")
    rule()
    para(nc["summary"])
    out.append("")
    out.append(f"  Material:                {nc['material']}")
    out.append(f"  Layup:                   {nc['layup']}")
    out.append(f"  Plies:                   {nc['n_plies']}")
    out.append(f"  Ply thickness (mm):      {nc['t_ply_mm']}")
    out.append(f"  Measured void Vp (%):    {nc['measured_Vp_percent']:.2f}")
    out.append(f"  Distribution:            {nc['distribution']}")
    out.append(f"  Void morphology:         {nc['void_shape']}")
    out.append(f"  Analysis mesh:           {nc['analysis_mesh']}")
    out.append("")

    out.append("2. ENGINEERING ANALYSIS (predicted porosity knockdown)")
    rule()
    para(
        f"Governing (worst-case) case: {ea['governing_mode']} / "
        f"{ea['governing_model']} - knockdown {ea['governing_knockdown']:.3f} "
        f"({ea['governing_knockdown'] * 100:.1f}% of pristine retained), "
        f"residual strength {ea['governing_residual_strength_MPa']:.1f} MPa."
    )
    out.append("")
    out.append(
        f"  {'Mode':<14}{'Model':<14}{'Resid. (MPa)':>14}{'Knockdown':>12}"
    )
    out.append("  " + "-" * 52)
    for mode, model, mpa, kd in _per_mode_rows(ea):
        out.append(f"  {mode:<14}{model:<14}{mpa:>14}{kd:>12}")
    out.append("")

    if "delamination" in ncr:
        out.append("2b. DELAMINATION ANALYSIS (fracture margin)")
        rule()
        if ncr["delamination"]["toughness_uncalibrated"]:
            para(_UNCALIBRATED_NOTE.upper())
            out.append("")
        for field, value in _delamination_rows(ncr):
            para(f"{field}: {value}", indent="  ")
        out.append("")

    out.append("3. RECOMMENDED DISPOSITION PATH")
    out.append("   (for MRB review - NOT a final disposition)")
    rule()
    para(f"Recommended path: {dp['path']}")
    out.append("")
    if "porosity_path" in dp:
        para(f"Porosity path: {dp['porosity_path']}", indent="  ")
        para(f"Delamination path: {dp['delamination_path']}", indent="  ")
        out.append("")
    para(f"Rationale: {dp['rationale']}")
    out.append("")
    para(dp["disclaimer"], indent="  ")
    out.append("")

    out.append("4. CITED CRITERIA")
    rule()
    for c in dp["cited_criteria"]:
        para(f"- {c}", indent="  ")
    out.append("")

    out.append("5. REQUIRED MRB ACTIONS")
    rule()
    for a in dp["required_mrb_actions"]:
        para(f"[ ] {a}", indent="  ")
    out.append("")
    para(
        "Generated by PorosityFE MRB support tool. This is a predictive "
        "analysis and a recommended disposition path; it does not constitute "
        "MRB approval."
    )
    return out


def serialise_ncr_pdf(ncr: dict) -> bytes:
    """Render the validation summary as a paginated US-Letter PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    lines = _ncr_text_lines(ncr)
    pages = [
        lines[i:i + _PDF_LINES_PER_PAGE]
        for i in range(0, len(lines), _PDF_LINES_PER_PAGE)
    ] or [[""]]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for page in pages:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(
                0.07, 0.96, "\n".join(page),
                family="monospace", fontsize=8.5,
                va="top", ha="left",
            )
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


def write_ncr_json(filepath: str, ncr: dict) -> None:
    text = serialise_ncr_json(ncr)   # raises before the file is opened
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def write_ncr_markdown(filepath: str, ncr: dict) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(serialise_ncr_markdown(ncr))


def write_ncr_pdf(filepath: str, ncr: dict) -> None:
    with open(filepath, "wb") as f:
        f.write(serialise_ncr_pdf(ncr))
