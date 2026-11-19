"""The analysis: ``run_analysis``, the FE solve and their Streamlit-cached wrappers.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import tempfile
import time
import warnings
from pathlib import Path

import streamlit as st

from porosity_fe import (
    CompositeMesh,
    Delamination,
    EmpiricalSolver,
    FESolver,
    FEUncertainty,
    GrowthResult,
    ParisLaw,
    ToughnessKnockdown,
    build_empirical_pipeline,
    compute_degraded_clt_flexural_modulus,
    fe_porosity_uncertainty,
    knockdown_decomposition,
    open_hole_analysis,
    propagate_uncertainty,
    short_beam_shear,
)
from porosity_fe.fe.failure import first_ply_failure_load_factor
from porosity_fe.fracture_specimens import (
    _D6671_MODE_RATIO,
)
from porosity_fe.homogenization import _check_porosity_degradation

from .config import (
    _CFG_DEFAULTS,
    _FE_APPLIED_STRAIN,
    _FE_FLEXURE_LOAD_N,
    _FE_FRACTURE_LOAD_N,
    _FE_ILSS_LOAD_N,
    _FLEXURE_MODES,
    _FRACTURE_MODES,
    _FRACTURE_UQ_FIELDS,
    _FRACTURE_UQ_SAMPLES,
    _THERMAL_OPTIONS,
    _TOUGHNESS_LAW_LABELS,
    _cfg_material,
    _cfg_plate,
    _fracture_specimen_mesh,
    _key_to_config,
    _resolving_nz,
)

logger = logging.getLogger(__name__)


# Each entry pickles a mesh and its fields (a few MB), so bound the cache
# instead of letting it grow for the life of the server process.
@st.cache_data(show_spinner=False, max_entries=16)
def run_analysis_cached(cfg_key: tuple) -> dict:
    """Cached wrapper around :func:`run_analysis`. ``cfg_key`` must be hashable."""
    return run_analysis(_key_to_config(cfg_key))


@st.cache_data(show_spinner=False, max_entries=16)
def run_uq_cached(cfg_key: tuple, mode: str, n_samples: int, coef_cov: float,
                  vp_cov: float, strength_cov: float) -> tuple[dict, list[str]]:
    """Latin-hypercube uncertainty propagation for the analysed laminate.

    Returns the :func:`propagate_uncertainty` result and any warning
    messages it raised (e.g. draws beyond the calibration bound).
    """
    cfg = _key_to_config(cfg_key)
    material = dataclasses.replace(
        _cfg_material(cfg), t_ply=cfg["t_ply"], n_plies=cfg["n_plies"])
    strength_field = EmpiricalSolver.PRISTINE_STRENGTH_KEY[mode]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = propagate_uncertainty(
            cfg["Vp"] / 100.0, material, mode, "judd_wright",
            covs={strength_field: strength_cov},
            vp_cov=vp_cov, coef_cov=coef_cov,
            n_samples=n_samples, method="lhs", seed=0,
            ply_angles=cfg["angles"],
        )
    return res, sorted({str(w.message) for w in caught})


# A VTU file of the default mesh is a few MB, so keep only a few.
@st.cache_data(show_spinner=False, max_entries=4)
def fe_vtu_bytes_cached(cfg_key: tuple, _result: dict) -> bytes:
    """Cached :func:`fe_vtu_bytes`, keyed on the analysis config ``cfg_key``.

    ``_result`` must be the analysis of ``cfg_key``; Streamlit does not
    hash arguments whose names start with an underscore.
    """
    return fe_vtu_bytes(_result)


def run_analysis(cfg: dict) -> dict:
    """Run the porosity analysis for one configuration.

    Returns a dict with keys: config, material, porosity_field, mesh,
    empirical, fe_field, fe_loading, fe_skipped_reason, layup_scale,
    fe_thermal, fe_mechanical_fpf, fe_nz_suggestion, fe_specimen,
    fe_short_beam, fe_flexure, fe_solve_time_s, delamination,
    fe_decomposition, fe_decomposition_error, fe_decomposition_time_s,
    fe_onset, fe_onset_error, fe_fracture_uq, fe_fracture_uq_error.

    ``mesh`` is the mesh of the FE solve. For ``loading_mode="ilss"`` (with
    a mechanical load) that is the ASTM D2344 short-beam specimen of
    :meth:`CompositeMesh.for_ilss` with ``cfg["nz"]`` element layers, not
    the ``nx x ny x nz`` coupon: ``fe_specimen`` then describes it
    (dimensions, span, loading band, mesh) and ``fe_short_beam`` holds the
    :func:`short_beam_shear` outputs of the solve (``None`` when it was
    skipped). Both are ``None`` for the other modes.

    The FE-only modes ``"flexure_3pt"`` / ``"flexure_4pt"`` (ASTM D7264
    procedures A / B) solve the :meth:`CompositeMesh.for_flexure` specimen
    (span 32 h, 13 mm wide) with ``cfg["nz"]`` element layers, described by
    ``fe_specimen``; ``fe_flexure`` holds the classical-lamination-theory
    flexural modulus to compare the FE ``effective_modulus`` with (see
    :func:`_clt_flexure_reference`), ``None`` for the other modes. The
    empirical results cover the four empirical modes whatever is selected.

    ``fe_solve_time_s`` is the wall time of the FE solve (s), which the
    mesh-convergence check scales for its time estimate.

    The empirical results are always returned. If the FE solve fails,
    ``fe_field`` is ``None`` and ``fe_skipped_reason`` explains why, so the
    UI can still show the headline knockdown numbers.

    The FE-only fracture modes ``"dcb"``, ``"enf"`` and ``"mmb"`` (ASTM
    D5528, D7905, D6671; test planning) solve the specimen of
    :meth:`CompositeMesh.for_dcb` / ``for_enf`` / ``for_mmb`` with the
    sidebar laminate, ``cfg["nz"]`` element layers and ``cfg["crack_length"]``
    (``cfg["mmb_mode_ratio"]``, the target ``G_II / G_T`` that sets the MMB
    lever), loaded by :data:`_FE_FRACTURE_LOAD_N`; ``fe_specimen``
    describes it.

    ``cfg["delamination"]`` (a :class:`Delamination`, ``None`` when absent)
    adds an explicit delamination to the FE mesh of the other modes (the
    coupon, or the ILSS / flexure specimen), which then reports the crack
    front (``fe_field.crack_front``) and, with the toughness of
    ``cfg["G_Ic"]``, ``cfg["G_IIc"]``, ``cfg["bk_eta"]`` (and optionally
    ``cfg["G_IIIc"]``, N/mm; no preset has them), the fracture load factor,
    the toughness knocked down by porosity with ``cfg["toughness_law"]``.
    ``delamination`` is the delamination of the mesh (``None`` without one,
    and for the fracture modes). With ``cfg["knockdown_split"]`` the
    knockdown is split by defect (:func:`knockdown_decomposition`, two more
    solves; ``fe_decomposition``, its time and any error). With
    ``cfg["onset"]`` the stress-based interface onset of the FE solve is
    added (:meth:`FieldResults.delamination_onset`, averaging length
    ``cfg["onset_x_avg"]``, default one ply; ``fe_onset``, or the reason in
    ``fe_onset_error``). With ``cfg["fracture_uq"]`` (the CoVs of
    :data:`_FRACTURE_UQ_FIELDS`, fractions) the uncertainty band on the
    fracture load factor from toughness scatter is added
    (:meth:`FieldResults.fracture_load_factor_uncertainty`, no further
    solve; ``fe_fracture_uq``, or the reason in ``fe_fracture_uq_error``).
    All are ``None`` when not requested.

    ``cfg["porosity_degradation"]`` (``"isotropic"`` when absent, or the
    opt-in ``"directional"``) is the FE porosity degradation mapping
    (``FESolver(porosity_degradation=)``), used by the FE solve, the
    knockdown split, the interface onset and the mechanical-only load
    factor alike, and recorded on ``fe_field.porosity_degradation``. The
    empirical results do not depend on it.

    ``cfg["thermal_mode"]`` (``"off"`` when absent) selects the FE thermal
    load: ``"combined"`` adds the cure residual stress at
    ``cfg["delta_T"]`` (K, negative for a cool-down) to the mechanical
    solve of ``cfg["loading_mode"]``, ``"thermal_only"`` solves the
    residual stress alone (``fe_loading`` is then ``"thermal"``). Thermal
    results go in ``fe_thermal`` (``{"mode", "delta_T"}``, ``None`` when
    off), ``fe_mechanical_fpf`` (combined: the first-ply-failure factor
    without the residual stress, for comparison) and ``fe_nz_suggestion``
    (the ``nz`` that resolves the layup, when the thermal solve was refused
    because the mesh does not).

    Raises
    ------
    ValueError
        For an unknown material, ``thermal_mode`` or
        ``porosity_degradation``, a thermal mode without
        a finite ``delta_T``, a ``delamination`` that is not a
        :class:`Delamination`, an unknown ``toughness_law``, ``"user"``
        without both coefficients, invalid toughness values, or a
        ``fracture_uq`` that is not one CoV (finite, ``>= 0``) per
        :data:`_FRACTURE_UQ_FIELDS` entry.
    """
    material = _cfg_material(cfg)       # refuses an unknown preset
    plate = _cfg_plate(cfg)
    thermal_mode = cfg.get("thermal_mode", "off")
    delta_T = cfg.get("delta_T")
    if thermal_mode not in _THERMAL_OPTIONS.values():
        raise ValueError(
            f"Unknown thermal_mode {thermal_mode!r}; expected one of "
            f"{sorted(_THERMAL_OPTIONS.values())}."
        )
    if thermal_mode != "off" and (
            not isinstance(delta_T, (int, float)) or isinstance(delta_T, bool)
            or not math.isfinite(delta_T)):
        raise ValueError(
            f"thermal_mode={thermal_mode!r} needs a finite delta_T in K "
            f"(negative for a cool-down); got {delta_T!r}. There is no default."
        )
    fracture_uq_covs = _fracture_uq_covs(cfg)
    porosity_degradation = _check_porosity_degradation(
        cfg.get("porosity_degradation", _CFG_DEFAULTS["porosity_degradation"]))
    material = dataclasses.replace(
        material, t_ply=cfg["t_ply"], n_plies=cfg["n_plies"],
        **_toughness_fields(cfg),
    )
    delamination = cfg.get("delamination")
    if delamination is not None and not isinstance(delamination, Delamination):
        raise ValueError(
            f"cfg['delamination'] must be a Delamination or None, got "
            f"{delamination!r}.")
    toughness_spec = _toughness_knockdown_spec(cfg)

    pf_kwargs = {
        "distribution": cfg["distribution"],
        "void_shape": cfg["void_shape"],
    }
    if cfg["distribution"] == "clustered":
        pf_kwargs["cluster_location"] = cfg["cluster_location"]

    porosity_field, mesh, empirical = build_empirical_pipeline(
        material, cfg["Vp"] / 100.0,
        ply_angles=cfg["angles"],
        mesh_res=(cfg["nx"], cfg["ny"], cfg["nz"]),
        porosity_config=pf_kwargs,
        # Only a plate size that was set: the default coupon is built as before.
        **({"mesh_kwargs": plate} if plate else {}),
    )
    emp_results = empirical.get_all_failure_loads()

    fe_loading = "thermal" if thermal_mode == "thermal_only" else cfg["loading_mode"]
    fe_specimen = None
    specimen_error = None
    # A fracture specimen carries its own insert; on the other meshes the
    # delamination is an input. Without one every mesh is built as before.
    delaminations = [] if delamination is None or fe_loading in _FRACTURE_MODES \
        else [delamination]
    defects = {"delaminations": delaminations} if delaminations else {}
    try:
        if fe_loading == "ilss":
            # ILSS is the ASTM D2344 short-beam test: solve it on the
            # standard specimen (span 4h, 6h x 2h), not on the coupon. The
            # empirical results do not depend on the mesh.
            mesh = CompositeMesh.for_ilss(porosity_field, material, nz=cfg["nz"],
                                          ply_angles=cfg["angles"], **defects)
            fe_specimen = _specimen_info(mesh)
        elif fe_loading in _FLEXURE_MODES:
            # Flexure is FE only: the ASTM D7264 specimen (span 32h, 13 mm
            # wide), one mesh for both procedures.
            mesh = CompositeMesh.for_flexure(porosity_field, material, nz=cfg["nz"],
                                             ply_angles=cfg["angles"], **defects)
            fe_specimen = _flexure_specimen_info(mesh, fe_loading)
        elif fe_loading in _FRACTURE_MODES:
            # The delamination fracture tests are FE only: the ASTM D5528,
            # D7905 or D6671 specimen with the sidebar's laminate.
            mesh = _fracture_specimen_mesh(fe_loading, porosity_field, material, cfg)
            fe_specimen = _fracture_specimen_info(mesh, fe_loading, cfg)
        elif delaminations:
            # The pipeline's coupon, with the delaminated interface's nodes
            # duplicated (the empirical solver keeps the plain one).
            mesh = CompositeMesh(porosity_field, material, nx=cfg["nx"],
                                 ny=cfg["ny"], nz=cfg["nz"],
                                 ply_angles=cfg["angles"], **defects, **plate)
    except ValueError as exc:
        logger.exception("Specimen mesh failed for loading=%s", fe_loading)
        what = (f"the {fe_loading} specimen mesh"
                if fe_loading in ("ilss", *_FLEXURE_MODES, *_FRACTURE_MODES)
                else "the delaminated coupon mesh")
        specimen_error = f"{what} could not be built: {exc}"
    # The delamination outputs (front, toughness law) of a delaminated mesh.
    cracked = bool(delaminations) or fe_loading in _FRACTURE_MODES
    t_solve = time.perf_counter()
    if specimen_error is None:
        fe_field, fe_skipped_reason = _run_fe_solve(
            mesh, material, porosity_field, cfg["angles"], cfg["loading_mode"],
            thermal_mode=thermal_mode, delta_T=delta_T,
            failure_criterion=cfg.get("failure_criterion", "tsai_wu"),
            porosity_degradation=porosity_degradation,
            **({"toughness_knockdown": toughness_spec} if cracked else {}),
        )
    else:
        fe_field, fe_skipped_reason = None, specimen_error
    fe_solve_time_s = time.perf_counter() - t_solve
    fe_short_beam = None
    if fe_loading == "ilss" and fe_specimen is not None and fe_field is not None:
        fe_short_beam = short_beam_shear(fe_field, mesh)
    fe_flexure = None
    if fe_loading in _FLEXURE_MODES:
        fe_flexure = _clt_flexure_reference(material, cfg["angles"], cfg["Vp"] / 100.0,
                                            fe_loading)

    fe_decomposition = fe_decomposition_error = fe_decomposition_time_s = None
    if delaminations and cfg.get("knockdown_split") and fe_field is not None \
            and fe_loading != "thermal":
        # Two more solves: the same porosity without the delamination, and
        # the delamination without porosity, against the same reference.
        t_split = time.perf_counter()
        try:
            fe_decomposition = knockdown_decomposition(
                mesh, material, porosity_field, fe_loading, total=fe_field,
                delta_T=delta_T if thermal_mode == "combined" else None,
                porosity_degradation=porosity_degradation,
                **_fe_load_kwargs(fe_loading))
        except Exception as exc:
            logger.exception("Knockdown decomposition failed for loading=%s",
                             fe_loading)
            fe_decomposition_error = f"{type(exc).__name__}: {exc}"
        fe_decomposition_time_s = time.perf_counter() - t_split
    fe_onset = fe_onset_error = None
    if cfg.get("onset") and fe_field is not None:
        try:
            fe_onset = fe_field.delamination_onset(
                mesh, material, x_avg=cfg.get("onset_x_avg"),
                porosity_degradation=fe_field.porosity_degradation)
        except ValueError as exc:
            fe_onset_error = str(exc)
    fe_fracture_uq = fe_fracture_uq_error = None
    if fracture_uq_covs is not None and fe_field is not None:
        if fe_field.fracture_load_factor is None:
            fe_fracture_uq_error = (
                "the λ_c band needs a delamination front and the toughness "
                "(G_Ic, G_IIc and η)")
        else:
            try:
                # No solve: the band re-evaluates λ_c on this front.
                fe_fracture_uq = fe_field.fracture_load_factor_uncertainty(
                    material, covs=fracture_uq_covs,
                    n_samples=_FRACTURE_UQ_SAMPLES, method="lhs", seed=0)
            except ValueError as exc:
                fe_fracture_uq_error = str(exc)

    fe_thermal = None
    fe_mechanical_fpf = None
    fe_nz_suggestion = None
    if specimen_error is not None and cracked and cfg["nz"] % cfg["n_plies"]:
        # A delamination needs every ply interface on a node layer.
        fe_nz_suggestion = _resolving_nz(cfg["nz"], cfg["n_plies"])
    if thermal_mode != "off":
        fe_thermal = {"mode": thermal_mode, "delta_T": float(delta_T)}
        if fe_field is None:
            if material.has_cte and mesh.layup_discrepancies():
                fe_nz_suggestion = _resolving_nz(cfg["nz"], cfg["n_plies"])
        elif thermal_mode == "combined" and fe_field.residual_stress_local is not None:
            # The same mechanical load without the residual stress. The
            # analysis is linear, so the mechanical part of the stress is
            # the total minus the residual.
            fe_mechanical_fpf = first_ply_failure_load_factor(
                fe_field.stress_local - fe_field.residual_stress_local,
                mesh.porosity, mesh.elements, material,
                porosity_field.void_shape_radii, fe_field.failure_criterion,
                void_elements=mesh.void_elements,
                porosity_degradation=fe_field.porosity_degradation)

    return {
        "config": cfg,
        "material": material,
        "porosity_field": porosity_field,
        "mesh": mesh,
        "empirical": emp_results,
        "fe_field": fe_field,
        "fe_loading": fe_loading,
        "fe_skipped_reason": fe_skipped_reason,
        "layup_scale": dict(empirical.layup_scale),
        "fe_thermal": fe_thermal,
        "fe_mechanical_fpf": fe_mechanical_fpf,
        "fe_nz_suggestion": fe_nz_suggestion,
        "fe_specimen": fe_specimen,
        "fe_short_beam": fe_short_beam,
        "fe_flexure": fe_flexure,
        "fe_solve_time_s": fe_solve_time_s,
        "delamination": delaminations[0] if delaminations else None,
        "fe_decomposition": fe_decomposition,
        "fe_decomposition_error": fe_decomposition_error,
        "fe_decomposition_time_s": fe_decomposition_time_s,
        "fe_onset": fe_onset,
        "fe_onset_error": fe_onset_error,
        "fe_fracture_uq": fe_fracture_uq,
        "fe_fracture_uq_error": fe_fracture_uq_error,
    }


def _fracture_uq_covs(cfg: dict) -> dict[str, float] | None:
    """The ``covs`` of the λ_c band of ``cfg`` (``None`` without one): the
    non-zero entries of ``cfg["fracture_uq"]`` by :data:`_FRACTURE_UQ_FIELDS`
    name. Nothing scatters unless entered, so an all-zero entry is no band."""
    value = cfg.get("fracture_uq")
    if value is None:
        return None
    values = list(value)
    if len(values) != len(_FRACTURE_UQ_FIELDS) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError(
            f"cfg['fracture_uq'] must hold one CoV (a finite fraction >= 0) "
            f"per input {list(_FRACTURE_UQ_FIELDS)}; got {value!r}.")
    covs = {name: float(v) for name, v in zip(_FRACTURE_UQ_FIELDS, values, strict=True)
            if v > 0.0}
    return covs or None


def _toughness_fields(cfg: dict) -> dict:
    """The interlaminar toughness of ``cfg`` that is set, as
    :class:`MaterialProperties` fields (N/mm; ``bk_eta`` dimensionless)."""
    return {k: float(cfg[k]) for k in ("G_Ic", "G_IIc", "G_IIIc", "bk_eta")
            if cfg.get(k) is not None}


def _toughness_knockdown_spec(cfg: dict):
    """The ``toughness_knockdown`` argument of the FE solve for ``cfg``:
    ``'exponential'`` (the placeholders), ``None`` (pristine toughness) or a
    :class:`ToughnessKnockdown` with the user's ``b_I`` and ``b_II``."""
    law = cfg.get("toughness_law", "exponential")
    if law == "exponential":
        return "exponential"
    if law == "none":
        return None
    if law == "user":
        b = cfg.get("toughness_b")
        if b is None or len(b) != 2 or any(v is None for v in b):
            raise ValueError(
                "toughness_law='user' needs toughness_b = [b_I, b_II] (per "
                "unit Vp).")
        return ToughnessKnockdown(b_I=float(b[0]), b_II=float(b[1]))
    raise ValueError(
        f"Unknown toughness_law {law!r}; expected one of "
        f"{sorted(_TOUGHNESS_LAW_LABELS)}.")


def _fe_load_kwargs(loading: str) -> dict:
    """The load arguments of the app's FE solve of ``loading``."""
    if loading == "ilss":
        return {"applied_load": _FE_ILSS_LOAD_N}
    if loading in _FLEXURE_MODES:
        return {"applied_load": _FE_FLEXURE_LOAD_N}
    if loading in _FRACTURE_MODES:
        return {"applied_load": _FE_FRACTURE_LOAD_N}
    return {"applied_strain": (-_FE_APPLIED_STRAIN if loading == "compression"
                               else _FE_APPLIED_STRAIN)}


def _fracture_specimen_info(mesh, loading: str, cfg: dict) -> dict:
    """Dimensions (mm), insert, lever and mesh of a fracture specimen."""
    standard, specimen, _, load = _FRACTURE_MODES[loading]
    geometry = mesh.fracture_geometry
    info = {
        "standard": standard,
        "specimen": specimen,
        "test": loading,
        "length_mm": float(mesh.L_x),
        "width_mm": float(mesh.L_y),
        "thickness_mm": float(mesh.L_z),
        "crack_length_mm": float(geometry.crack_length),
        "interface": int(geometry.interface),
        "span_mm": None if geometry.span is None else float(geometry.span),
        "lever_length_mm": (None if geometry.lever_length is None
                            else float(geometry.lever_length)),
        "mode_ratio_target": None,
        "load": load,
        "applied_load_N": abs(_FE_FRACTURE_LOAD_N),
        "mesh": f"{mesh.nx}x{mesh.ny}x{mesh.nz}",
    }
    if loading == "mmb":
        info["mode_ratio_target"] = float(
            cfg.get("mmb_mode_ratio") or _D6671_MODE_RATIO)
    return info


def _specimen_info(mesh) -> dict:
    """Dimensions (mm) and mesh of a :meth:`CompositeMesh.for_ilss` specimen."""
    geometry = mesh.ilss_geometry
    return {
        "standard": "ASTM D2344",
        "specimen": "short-beam specimen",
        "length_mm": float(mesh.L_x),
        "width_mm": float(mesh.L_y),
        "thickness_mm": float(mesh.L_z),
        "span_mm": float(geometry.span),
        "load_width_mm": float(geometry.load_width),
        "mesh": f"{mesh.nx}x{mesh.ny}x{mesh.nz}",
    }


def _flexure_specimen_info(mesh, loading: str) -> dict:
    """Dimensions (mm) and mesh of a :meth:`CompositeMesh.for_flexure` specimen."""
    geometry = mesh.flexure_geometry
    four_point = _FLEXURE_MODES[loading] == "4pt"
    return {
        "standard": f"ASTM D7264 procedure {'B' if four_point else 'A'}",
        "specimen": f"{'four' if four_point else 'three'}-point flexure specimen",
        "length_mm": float(mesh.L_x),
        "width_mm": float(mesh.L_y),
        "thickness_mm": float(mesh.L_z),
        "span_mm": float(geometry.span),
        "span_to_thickness": float(geometry.span / mesh.L_z),
        "load_span_mm": float(geometry.load_span) if four_point else None,
        "mesh": f"{mesh.nx}x{mesh.ny}x{mesh.nz}",
    }


def _clt_flexure_reference(material, angles, vp: float, loading: str) -> dict:
    """Classical-lamination-theory flexural modulus to compare a flexure FE solve with.

    :func:`compute_degraded_clt_flexural_modulus` (``E_f = 12 / (h^3 d_11)``
    from the D matrix, every ply degraded by Mori-Tanaka spherical voids at
    the specimen-average ``vp``, a fraction) at ``vp`` and at zero porosity.
    It knows neither the void shape nor the through-thickness distribution.
    """
    porous = compute_degraded_clt_flexural_modulus(material, angles, vp)["Ef_x"]
    pristine = compute_degraded_clt_flexural_modulus(material, angles, 0.0)["Ef_x"]
    return {
        "procedure": _FLEXURE_MODES[loading],
        "clt_E_f": float(porous),
        "clt_E_f_pristine": float(pristine),
        "clt_knockdown": float(porous / pristine),
        "clt_void_shape": "spherical",
    }


def _run_fe_solve(mesh, material, porosity_field, angles, loading_mode: str,
                  *, thermal_mode: str = "off", delta_T: float | None = None,
                  failure_criterion: str = "tsai_wu",
                  porosity_degradation: str = _CFG_DEFAULTS["porosity_degradation"],
                  **fracture):
    """Run the FE solve for ``loading_mode``; return ``(field, skipped_reason)``.

    Exactly one of the pair is ``None``. Any exception from the FE path is
    logged and turned into a reason string rather than propagated, because
    the empirical results do not depend on it. That includes the thermal
    solve's refusal of a mesh that does not resolve the layup.

    ``thermal_mode="combined"`` passes ``delta_T`` to the mechanical solve
    (residual stress superposed); ``"thermal_only"`` solves
    ``loading='thermal'`` at ``delta_T`` instead of ``loading_mode``.
    ``failure_criterion`` and ``porosity_degradation`` are the
    ``FESolver``'s. ``fracture`` (``toughness_knockdown=``, for a
    delaminated mesh) goes to the solve.
    The fracture modes ``'dcb'``, ``'enf'`` and ``'mmb'`` are loaded by
    :data:`_FE_FRACTURE_LOAD_N`.
    """
    thermal = {} if thermal_mode == "off" else {"delta_T": delta_T}
    thermal.update(fracture)
    try:
        fe_solver = FESolver(mesh, material, porosity_field, ply_angles=angles,
                             failure_criterion=failure_criterion,
                             porosity_degradation=porosity_degradation)
        if thermal_mode == "thermal_only":
            # Free-standing laminate under delta_T (3-2-1 supports): the
            # cure residual stress, with no mechanical load.
            field = fe_solver.solve(loading="thermal", verbose=False, **thermal)
        elif loading_mode == "ilss":
            # ASTM D2344 short-beam shear on the specimen mesh: roller
            # supports, the load over the loading-nose band, force
            # controlled. The 10 N midspan load is arbitrary: knockdown and
            # field shapes are scale-invariant.
            field = fe_solver.solve(
                loading="ilss", applied_load=_FE_ILSS_LOAD_N, verbose=False,
                **thermal,
            )
        elif loading_mode in _FLEXURE_MODES:
            # ASTM D7264 three- or four-point bending on the flexure
            # specimen mesh: roller supports, force controlled; the solve
            # reports the shear-corrected flexural modulus.
            field = fe_solver.solve(
                loading=loading_mode, applied_load=_FE_FLEXURE_LOAD_N,
                verbose=False, **thermal,
            )
        elif loading_mode in _FRACTURE_MODES:
            # The DCB / ENF / MMB specimen, force controlled: the front's
            # energy release rates scale with the load squared.
            field = fe_solver.solve(
                loading=loading_mode, applied_load=_FE_FRACTURE_LOAD_N,
                verbose=False, **thermal,
            )
        else:
            applied_strain = (-_FE_APPLIED_STRAIN if loading_mode == "compression"
                              else _FE_APPLIED_STRAIN)
            field = fe_solver.solve(
                loading=loading_mode, applied_strain=applied_strain,
                verbose=False, **thermal,
            )
    except Exception as exc:
        logger.exception("FE solve failed for loading=%s, thermal_mode=%s",
                         loading_mode, thermal_mode)
        return None, f"the FE solver raised {type(exc).__name__}: {exc}"
    return field, None


# ======================================================================
# FE result helpers (headless: no Streamlit calls)
# ======================================================================

def fe_vtu_bytes(result: dict) -> bytes:
    """The FE field of ``result`` as binary VTK XML (``.vtu``) bytes.

    Written by :meth:`FieldResults.to_vtu` with its defaults (the legacy
    VTK point and cell fields plus the recovered nodal stress and strain,
    ``Float64``; the front's energy release rates on a delaminated mesh,
    and the onset index when the analysis checked the interface onset) to
    a temporary file, since the writer takes a path.

    Raises
    ------
    ValueError
        If ``result`` has no FE field (the FE solve was skipped).
    """
    fe_field = result.get("fe_field")
    if fe_field is None:
        raise ValueError("The analysis has no FE field to export: the FE "
                         "solve was skipped.")
    onset = result.get("fe_onset")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fe_results.vtu"
        if onset is None:
            fe_field.to_vtu(result["mesh"], path)
        else:
            fe_field.to_vtu(result["mesh"], path, onset=onset)
        return path.read_bytes()


# ----------------------------------------------------------------------
# Linear buckling (on demand, from the Results tab)
# ----------------------------------------------------------------------

#: Load cases the buckling panel analyses (the coupon's in-plane ones).
_BUCKLING_MODES = ("compression", "shear")

#: Modes the panel computes.
_BUCKLING_N_MODES = 3


def buckling_unavailable_reason(result: dict) -> str | None:
    """Why the buckling panel cannot analyse ``result``; ``None`` when it can."""
    if result.get("fe_field") is None:
        return "The FE solve of this analysis was skipped, so there is no model to buckle."
    if result.get("fe_loading") not in _BUCKLING_MODES:
        return ("Linear buckling analyses the coupon under compression or shear: "
                "select one of them as the loading mode.")
    if (result.get("fe_thermal") or {}).get("mode") == "combined":
        return ("Linear buckling does not take the thermal residual stress (it would "
                "be a prestress the load factor does not scale): switch it off.")
    return None


def run_buckling(result: dict, n_modes: int = _BUCKLING_N_MODES) -> dict:
    """The buckling analysis of an analysis' FE model and of the same model
    without voids: ``{"result", "reference"}`` (:class:`BucklingResult`)."""
    from porosity_fe import PorosityField

    cfg = result["config"]
    mesh, material = result["mesh"], result["material"]
    degradation = cfg.get("porosity_degradation", _CFG_DEFAULTS["porosity_degradation"])
    criterion = cfg.get("failure_criterion", "tsai_wu")
    loading = result["fe_loading"]

    def analyse(field, model_mesh):
        solver = FESolver(model_mesh, material, field, ply_angles=cfg["angles"],
                          porosity_degradation=degradation)
        return solver.buckling(loading, n_modes=n_modes, failure_criterion=criterion)

    porous = analyse(result["porosity_field"], mesh)
    void_free = PorosityField(material, 0.0)
    reference_mesh = CompositeMesh(
        void_free, material, nx=mesh.nx, ny=mesh.ny, nz=mesh.nz, ply_angles=cfg["angles"],
        length=float(mesh.L_x), width=float(mesh.L_y),
        **({"delaminations": list(mesh.delaminations)} if mesh.delaminations else {}))
    return {"result": porous, "reference": analyse(void_free, reference_mesh)}


@st.cache_data(show_spinner=False, max_entries=4)
def run_buckling_cached(cfg_key: tuple, _result: dict) -> dict:
    """Cached :func:`run_buckling`; ``_result`` must be the analysis of
    ``cfg_key`` (Streamlit does not hash an argument starting with ``_``)."""
    return run_buckling(_result)


# ----------------------------------------------------------------------
# Delamination growth (on demand, from the Results tab)
# ----------------------------------------------------------------------

#: Default and largest number of release steps of the growth panel.
GROWTH_DEFAULT_STEPS = 20
GROWTH_MAX_STEPS = 200


def growth_unavailable_reason(result: dict) -> str | None:
    """Why the growth panel cannot analyse ``result``; ``None`` when it can."""
    fe = result.get("fe_field")
    if fe is None:
        return "The FE solve of this analysis was skipped, so there is no crack to grow."
    if (result.get("fe_thermal") or {}).get("mode") == "thermal_only":
        return ("Delamination growth needs a mechanical load; the thermal-only "
                "solve has none.")
    if fe.crack_front is None:
        return ("Delamination growth needs a delamination: add one in the Defects "
                "panel of the sidebar, or run a DCB, ENF or MMB specimen.")
    if fe.fracture_load_factor is None:
        return ("Delamination growth needs the interlaminar toughness (G_Ic, "
                "G_IIc, η): enter it in the Defects panel of the sidebar.")
    return None


def run_growth(result: dict, max_steps: int = GROWTH_DEFAULT_STEPS,
               paris: ParisLaw | None = None, peak: float | None = None) -> GrowthResult:
    """The delamination growth of an analysis' FE model, under its load case.

    Quasi-static without ``paris``; with it, fatigue under the cyclic peak
    that is ``peak`` (0 to 1) of the analysis' static critical load. The
    model, toughness law and thermal residual stress are the analysis'.
    """
    cfg = result["config"]
    loading = result["fe_loading"]
    solver = FESolver(result["mesh"], result["material"], result["porosity_field"],
                      ply_angles=cfg["angles"],
                      failure_criterion=cfg.get("failure_criterion", "tsai_wu"),
                      porosity_degradation=cfg.get(
                          "porosity_degradation", _CFG_DEFAULTS["porosity_degradation"]))
    kwargs: dict = {**_fe_load_kwargs(loading),
                    "toughness_knockdown": _toughness_knockdown_spec(cfg)}
    thermal = result.get("fe_thermal") or {}
    if thermal.get("mode") == "combined":
        kwargs["delta_T"] = thermal["delta_T"]
    if paris is None:
        return solver.delamination_growth(loading, max_steps=max_steps, **kwargs)
    lam = result["fe_field"].fracture_load_factor
    if peak is None or lam is None or not math.isfinite(lam):
        raise ValueError("Fatigue growth needs the cyclic peak as a fraction of a "
                         "finite static critical load.")
    name = "applied_load" if "applied_load" in kwargs else "applied_strain"
    kwargs[name] = kwargs[name] * peak * lam
    return solver.fatigue_delamination_growth(loading, paris, max_steps=max_steps,
                                              **kwargs)


@st.cache_data(show_spinner=False, max_entries=4)
def run_growth_cached(cfg_key: tuple, growth_key: tuple, _result: dict) -> GrowthResult:
    """Cached :func:`run_growth`. ``growth_key`` is ``(steps, kind, C, m,
    threshold, peak)`` with ``kind`` ``'quasi-static'`` or ``'fatigue'``;
    ``_result`` must be the analysis of ``cfg_key``."""
    steps, kind, C, m, threshold, peak = growth_key
    if kind == "fatigue":
        return run_growth(_result, steps, ParisLaw(C, m, threshold), peak)
    return run_growth(_result, steps)


# ----------------------------------------------------------------------
# FE uncertainty over random porosity fields (on demand, Results tab)
# ----------------------------------------------------------------------

#: Default and largest number of draws of the FE uncertainty panel.
FE_UQ_DEFAULT_SAMPLES = 20
FE_UQ_MAX_SAMPLES = 200

#: Load cases of the panel (the coupon's in-plane ones).
_FE_UQ_MODES = ("compression", "tension", "shear")


def fe_uq_unavailable_reason(result: dict) -> str | None:
    """Why the FE uncertainty panel cannot study ``result``; ``None`` when it can."""
    if result.get("fe_field") is None:
        return "The FE solve of this analysis was skipped, so there is no model to sample."
    if (result.get("fe_thermal") or {}).get("mode") == "thermal_only":
        return "The FE uncertainty study needs a mechanical load; the thermal-only solve has none."
    if result.get("fe_loading") not in _FE_UQ_MODES:
        return ("The FE uncertainty study takes the coupon under compression, tension "
                "or shear: select one of them as the loading mode.")
    return None


def run_fe_uq(result: dict, n_samples: int, cov: float, correlation_length: float,
              seed: int = 0) -> FEUncertainty:
    """The FE uncertainty study of an analysis: its model with a random
    in-plane porosity map of the same average in every draw."""
    cfg = result["config"]
    mesh, material, pf = result["mesh"], result["material"], result["porosity_field"]
    config = {"distribution": cfg["distribution"], "void_shape": cfg["void_shape"]}
    if cfg["distribution"] == "clustered":
        config["cluster_location"] = cfg["cluster_location"]
    kwargs: dict = {**_fe_load_kwargs(result["fe_loading"]),
                    "failure_criterion": cfg.get("failure_criterion", "tsai_wu")}
    thermal = result.get("fe_thermal") or {}
    if thermal.get("mode") == "combined":
        kwargs["delta_T"] = thermal["delta_T"]
    if mesh.delaminations:
        kwargs["delaminations"] = list(mesh.delaminations)
        kwargs["toughness_knockdown"] = _toughness_knockdown_spec(cfg)
    return fe_porosity_uncertainty(
        material, pf.Vp, result["fe_loading"], cov=cov,
        correlation_length=correlation_length, n_samples=n_samples, seed=seed,
        ply_angles=cfg["angles"], config=config, mesh_res=(mesh.nx, mesh.ny, mesh.nz),
        length=float(mesh.L_x), width=float(mesh.L_y),
        porosity_degradation=cfg.get("porosity_degradation",
                                     _CFG_DEFAULTS["porosity_degradation"]),
        **kwargs)


@st.cache_data(show_spinner=False, max_entries=4)
def run_fe_uq_cached(cfg_key: tuple, params: tuple, _result: dict) -> FEUncertainty:
    """Cached :func:`run_fe_uq`; ``params`` is ``(n_samples, cov,
    correlation_length)`` and ``_result`` the analysis of ``cfg_key``."""
    n_samples, cov, correlation_length = params
    return run_fe_uq(_result, n_samples, cov, correlation_length)


# ----------------------------------------------------------------------
# Open-hole notch (on demand, Results tab)
# ----------------------------------------------------------------------

#: The standards' open-hole coupon (ASTM D5766 / D6484), mm.
OPEN_HOLE_DEFAULT_DIAMETER = 6.0
OPEN_HOLE_DEFAULT_WIDTH = 36.0

#: Load cases of the panel.
_OPEN_HOLE_MODES = ("compression", "tension")


def open_hole_unavailable_reason(result: dict) -> str | None:
    """Why the open-hole panel cannot study ``result``; ``None`` when it can."""
    if result.get("fe_loading") not in _OPEN_HOLE_MODES:
        return ("The open-hole coupon is loaded in compression or tension: select one "
                "of them as the loading mode.")
    return None


def run_open_hole(result: dict, hole_diameter: float, width: float) -> dict:
    """The open-hole coupon of an analysis: its material, layup and porosity
    on :meth:`CompositeMesh.for_open_hole`, under its load. Returns
    ``{"analysis": OpenHoleAnalysis, "fe": FieldResults}``."""
    cfg = result["config"]
    material = result["material"]
    pf_kwargs = {"distribution": cfg["distribution"], "void_shape": cfg["void_shape"]}
    if cfg["distribution"] == "clustered":
        pf_kwargs["cluster_location"] = cfg["cluster_location"]
    from porosity_fe import PorosityField
    pf = PorosityField(material, cfg["Vp"] / 100.0, **pf_kwargs)
    mesh = CompositeMesh.for_open_hole(pf, material, hole_diameter=hole_diameter,
                                       width=width, ply_angles=cfg["angles"])
    kwargs: dict = dict(_fe_load_kwargs(result["fe_loading"]))
    thermal = result.get("fe_thermal") or {}
    if thermal.get("mode") == "combined":
        kwargs["delta_T"] = thermal["delta_T"]
    solver = FESolver(mesh, material, pf, ply_angles=cfg["angles"],
                      failure_criterion=cfg.get("failure_criterion", "tsai_wu"),
                      porosity_degradation=cfg.get(
                          "porosity_degradation", _CFG_DEFAULTS["porosity_degradation"]))
    fe = solver.solve(result["fe_loading"], **kwargs)
    return {"analysis": open_hole_analysis(fe, mesh), "fe": fe}


@st.cache_data(show_spinner=False, max_entries=4)
def run_open_hole_cached(cfg_key: tuple, params: tuple, _result: dict) -> dict:
    """Cached :func:`run_open_hole`; ``params`` is ``(hole_diameter, width)``."""
    return run_open_hole(_result, *params)


# ----------------------------------------------------------------------
# Hybrid laminate (on demand, Results tab)
# ----------------------------------------------------------------------

#: Load cases of the panel (the coupon's in-plane ones).
_HYBRID_MODES = ("compression", "tension", "shear")


def hybrid_default_spec(result: dict) -> str:
    """A starting ply stack for the panel: the analysis' plies with a glass
    face ply on each side (or the plies alone for a glass laminate)."""
    cfg = result["config"]
    name = cfg["material_name"]
    n = int(result["material"].n_plies)
    glass = "glass_epoxy"
    if name == glass or n < 3 or cfg.get("material") is not None:
        return f"{name}*{n}"
    return f"{glass}, {name}*{n - 2}, {glass}"


def hybrid_unavailable_reason(result: dict) -> str | None:
    """Why the hybrid laminate panel cannot study ``result``; ``None`` when it can."""
    if result.get("fe_field") is None:
        return "The FE solve of this analysis was skipped, so there is no model to compare."
    if (result.get("fe_thermal") or {}).get("mode") == "thermal_only":
        return "The hybrid laminate panel needs a mechanical load; the thermal-only solve has none."
    if result.get("fe_loading") not in _HYBRID_MODES:
        return ("The hybrid laminate panel takes the coupon under compression, tension "
                "or shear: select one of them as the loading mode.")
    if result["mesh"].delaminations:
        return "The hybrid laminate panel solves the coupon without the delamination."
    return None


def run_hybrid(result: dict, spec: str) -> dict:
    """The analysis' coupon, porosity and load on the hybrid laminate ``spec``
    (:func:`porosity_fe.parse_ply_stack`; the presets, and a custom material
    by its name). Returns ``{"stack", "fe", "mesh", "clt_modulus"}``: the
    FE solve and, under compression or tension, the lamination-theory
    modulus of the porous stack."""
    from porosity_fe import (
        MATERIALS,
        PlyStack,
        PorosityField,
        compute_degraded_clt_moduli,
        parse_ply_stack,
    )
    from porosity_fe.pipeline import _default_mesh_resolution
    from porosity_fe.ply_stack import stack_angles

    cfg = result["config"]
    names = dict(MATERIALS)
    if cfg.get("material") is not None:
        names[cfg["material_name"]] = _cfg_material(cfg)
    stack: PlyStack = parse_ply_stack(spec, names)
    angles = cfg["angles"]
    stack_angles(stack, angles)  # refuses a layup longer than the stack
    pf_kwargs = {"distribution": cfg["distribution"], "void_shape": cfg["void_shape"]}
    if cfg["distribution"] == "clustered":
        pf_kwargs["cluster_location"] = cfg["cluster_location"]
    Vp = cfg["Vp"] / 100.0
    pf = PorosityField(stack, Vp, **pf_kwargs)
    plain = result["mesh"]
    nz = _default_mesh_resolution(stack.n_plies, angles)[2]
    mesh = CompositeMesh(pf, stack, plain.nx, plain.ny, nz, ply_angles=angles,
                         **_cfg_plate(cfg))
    kwargs: dict = dict(_fe_load_kwargs(result["fe_loading"]))
    thermal = result.get("fe_thermal") or {}
    if thermal.get("mode") == "combined":
        kwargs["delta_T"] = thermal["delta_T"]
    solver = FESolver(mesh, stack, pf, ply_angles=angles,
                      failure_criterion=cfg.get("failure_criterion", "tsai_wu"),
                      porosity_degradation=cfg.get(
                          "porosity_degradation", _CFG_DEFAULTS["porosity_degradation"]))
    fe = solver.solve(result["fe_loading"], **kwargs)
    clt = (compute_degraded_clt_moduli(stack, angles, Vp)["Ex"]
           if result["fe_loading"] in ("compression", "tension") else None)
    return {"stack": stack, "fe": fe, "mesh": mesh, "clt_modulus": clt}


@st.cache_data(show_spinner=False, max_entries=4)
def run_hybrid_cached(cfg_key: tuple, spec: str, _result: dict) -> dict:
    """Cached :func:`run_hybrid`; ``_result`` is the analysis of ``cfg_key``."""
    return run_hybrid(_result, spec)
