"""Porosity FE Analysis — Streamlit web app.

Browser front end for `porosity_fe`. Mirrors the structure used by
WrinkleFE: a sidebar of inputs feeds a cached analysis function whose result
fans out to Profile / Mesh / Results / Stress / Export tabs.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import logging
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter

logger = logging.getLogger(__name__)

# Make sibling modules importable when launched via `streamlit run app.py`
# from a checkout that hasn't been `pip install -e .`'d.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from porosity_fe import (
    LABEL_KNOCKDOWN,
    LABEL_POROSITY_PCT,
    LABEL_STIFFNESS_RETENTION,
    LABEL_X_MM,
    LABEL_Z_MM,
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    RefinementStudy,
    _configure_matplotlib_style,
    build_empirical_pipeline,
    compute_degraded_clt_flexural_modulus,
    propagate_uncertainty,
    refinement_study,
    short_beam_shear,
)
from porosity_fe.fe.failure import first_ply_failure_load_factor
from porosity_fe.fe.refinement import _estimate_memory_bytes
from porosity_fe.pipeline import _default_mesh_resolution

# Re-apply the shared style after Streamlit/matplotlib finished their own
# rcParams nudges, so the Streamlit plots match the static PNGs (#53).
_configure_matplotlib_style()

# Repo README link for in-app guidance (e.g. when FE solve is skipped, #129).
_README_URL = (
    "https://github.com/elhajjar1/PorosityFE"
    "#solver-selection-fe-vs-empirical"
)


# ======================================================================
# Pure helpers (extracted to porosity_fe.reporting so tests can import them
# without pulling in Streamlit — see #155 follow-up).
# ======================================================================

from porosity_fe.reporting import (  # noqa: E402, F401  re-exported for the Streamlit UI
    STRUCTURAL_CLASSES,
    _sanitise_filename_component,
    _serialise_payload_csv,
    _serialise_payload_json,
    build_export_payload,
    build_ncr_record,
    download_filename_stem,
    governing_failure,
    parse_layup,
    recommend_disposition,
    serialise_ncr_json,
    serialise_ncr_markdown,
    serialise_ncr_pdf,
    write_ncr_json,
    write_ncr_markdown,
    write_ncr_pdf,
    write_results_csv,
    write_results_json,
)


# ======================================================================
# Analysis runner (cached)
# ======================================================================

# Tuple ordering for the cache key. Keeping it explicit makes the cache
# invalidate cleanly when a new field is added.
_CFG_KEYS = (
    "material_name", "angles", "n_plies", "t_ply", "Vp",
    "distribution", "cluster_location", "void_shape", "loading_mode",
    "nx", "ny", "nz", "thermal_mode", "delta_T", "failure_criterion",
)

#: Config fields that an older config dict may lack, and the value that
#: absence means (no thermal load, the Tsai-Wu criterion).
_CFG_DEFAULTS = {"thermal_mode": "off", "delta_T": None,
                 "failure_criterion": "tsai_wu"}

#: FE failure criteria offered in the sidebar: key -> label.
_CRITERION_LABELS = {
    "tsai_wu": "Tsai-Wu (one interactive index)",
    "hashin": "Hashin (separate modes, incl. delamination onset)",
    "max_stress": "Maximum stress (separate modes)",
}

#: Failure-mode keys of FieldResults.failure_mode_maxima -> labels.
_MODE_LABELS = {
    "fiber_t": "Fibre tension",
    "fiber_c": "Fibre compression",
    "matrix_t": "Matrix tension",
    "matrix_c": "Matrix compression",
    "shear": "Shear",
    "delamination": "Delamination onset",
}


def _config_to_key(cfg: dict) -> tuple:
    key = []
    for k in _CFG_KEYS:
        v = cfg.get(k, _CFG_DEFAULTS[k]) if k in _CFG_DEFAULTS else cfg[k]
        key.append((k, tuple(v) if isinstance(v, list) else v))
    return tuple(key)


# ----------------------------------------------------------------------
# FE load case
# ----------------------------------------------------------------------

#: Applied load of the app's FE solves: nominal strain magnitude of the
#: displacement-controlled modes (negative for compression), the ILSS
#: midspan force and the total flexure load (N). Knockdown and field shapes
#: are scale-invariant; the first-ply-failure factor multiplies these loads.
_FE_APPLIED_STRAIN = 0.01
_FE_ILSS_LOAD_N = -10.0
_FE_FLEXURE_LOAD_N = -10.0

#: The FE-only flexure modes (ASTM D7264 procedures A and B on the
#: :meth:`CompositeMesh.for_flexure` specimen) and their procedure keys.
_FLEXURE_MODES = {"flexure_3pt": "3pt", "flexure_4pt": "4pt"}

#: Sidebar loading choices: ``cfg["loading_mode"]`` -> label. The four
#: empirical modes are computed whichever is selected; the selection picks
#: the FE solve. The flexure modes have no empirical counterpart.
_LOADING_LABELS = {
    "compression": "compression",
    "tension": "tension",
    "shear": "shear",
    "ilss": "ilss",
    # Short enough to fit the sidebar box; the help text spells them out.
    "flexure_3pt": "flexure_3pt (FE only, D7264 A)",
    "flexure_4pt": "flexure_4pt (FE only, D7264 B)",
}

#: Sidebar label -> ``cfg["thermal_mode"]``. ``"combined"`` superposes the
#: free-standing residual stress on the mechanical solve of the selected
#: loading mode; ``"thermal_only"`` solves the residual stress alone.
_THERMAL_OPTIONS = {
    "Off (mechanical only)": "off",
    "Add cure residual stress to the mechanical case": "combined",
    "Thermal only (residual stress, no mechanical load)": "thermal_only",
}

#: Suggested starting value for delta_T (K) when the preset records no
#: stress-free temperature: a 177 deg C-cure epoxy cooled to room
#: temperature, with the stress-free temperature taken up to about 30 K
#: below the cure temperature (docs/theory/thermal.md). Offered as a
#: labelled suggestion the user must apply; never a hidden default.
_SUGGESTED_DELTA_T_177C_EPOXY = -150.0

#: Largest value of the Expert-mode ``nz`` slider.
_NZ_SLIDER_MAX = 100


def _delta_T_suggestion(material) -> tuple[float, str]:
    """Suggested ``delta_T`` (K) for ``material`` and where it comes from."""
    if material.T_stress_free is not None:
        t_use = material.T_service if material.T_service is not None else material.T_ref
        return (float(t_use) - float(material.T_stress_free),
                f"{t_use:g} °C − the preset's stress-free temperature "
                f"{material.T_stress_free:g} °C")
    return (_SUGGESTED_DELTA_T_177C_EPOXY,
            "typical for a 177 °C-cure epoxy cooled to room temperature; "
            "check it against your cure cycle")


def _resolving_nz(nz: int, n_plies: int) -> int:
    """Smallest multiple of ``n_plies`` that is at least ``nz``.

    With ``nz`` a multiple of the ply count every ply has its own element
    layers, which the thermal solve needs (``CompositeMesh.layup_discrepancies``).
    """
    return int(n_plies) * max(1, math.ceil(int(nz) / int(n_plies)))


# Each entry pickles a mesh and its fields (a few MB), so bound the cache
# instead of letting it grow for the life of the server process.
@st.cache_data(show_spinner=False, max_entries=16)
def run_analysis_cached(cfg_key: tuple) -> dict:
    """Cached wrapper around :func:`run_analysis`. ``cfg_key`` must be hashable."""
    cfg = {}
    for k, v in cfg_key:
        cfg[k] = list(v) if isinstance(v, tuple) else v
    return run_analysis(cfg)


@st.cache_data(show_spinner=False, max_entries=16)
def run_uq_cached(cfg_key: tuple, mode: str, n_samples: int, coef_cov: float,
                  vp_cov: float, strength_cov: float) -> tuple[dict, list[str]]:
    """Latin-hypercube uncertainty propagation for the analysed laminate.

    Returns the :func:`propagate_uncertainty` result and any warning
    messages it raised (e.g. draws beyond the calibration bound).
    """
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg_key}
    material = dataclasses.replace(
        MATERIALS[cfg["material_name"]], t_ply=cfg["t_ply"], n_plies=cfg["n_plies"])
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


# A study keeps only scalars per level (a few kB), but each entry stands
# for minutes of solving; a handful covers going back and forth.
@st.cache_data(show_spinner=False, max_entries=8)
def run_convergence_cached(cfg_key: tuple, resolutions: tuple) -> RefinementStudy:
    """Cached :func:`run_convergence_check` of the analysis config ``cfg_key``."""
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg_key}
    return run_convergence_check(cfg, resolutions)


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
    fe_short_beam, fe_flexure, fe_solve_time_s.

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
        For an unknown material or ``thermal_mode``, or a thermal mode
        without a finite ``delta_T``.
    """
    if cfg["material_name"] not in MATERIALS:
        raise ValueError(
            f"Unknown material {cfg['material_name']!r}. "
            f"Available presets: {sorted(MATERIALS)}."
        )
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
    material = MATERIALS[cfg["material_name"]]
    material = dataclasses.replace(
        material, t_ply=cfg["t_ply"], n_plies=cfg["n_plies"],
    )

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
    )
    emp_results = empirical.get_all_failure_loads()

    fe_loading = "thermal" if thermal_mode == "thermal_only" else cfg["loading_mode"]
    fe_specimen = None
    specimen_error = None
    try:
        if fe_loading == "ilss":
            # ILSS is the ASTM D2344 short-beam test: solve it on the
            # standard specimen (span 4h, 6h x 2h), not on the coupon. The
            # empirical results do not depend on the mesh.
            mesh = CompositeMesh.for_ilss(porosity_field, material, nz=cfg["nz"],
                                          ply_angles=cfg["angles"])
            fe_specimen = _specimen_info(mesh)
        elif fe_loading in _FLEXURE_MODES:
            # Flexure is FE only: the ASTM D7264 specimen (span 32h, 13 mm
            # wide), one mesh for both procedures.
            mesh = CompositeMesh.for_flexure(porosity_field, material, nz=cfg["nz"],
                                             ply_angles=cfg["angles"])
            fe_specimen = _flexure_specimen_info(mesh, fe_loading)
    except ValueError as exc:
        logger.exception("Specimen mesh failed for loading=%s", fe_loading)
        specimen_error = f"the {fe_loading} specimen mesh could not be built: {exc}"
    t_solve = time.perf_counter()
    if specimen_error is None:
        fe_field, fe_skipped_reason = _run_fe_solve(
            mesh, material, porosity_field, cfg["angles"], cfg["loading_mode"],
            thermal_mode=thermal_mode, delta_T=delta_T,
            failure_criterion=cfg.get("failure_criterion", "tsai_wu"),
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

    fe_thermal = None
    fe_mechanical_fpf = None
    fe_nz_suggestion = None
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
                void_elements=mesh.void_elements)

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
    }


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
                  failure_criterion: str = "tsai_wu"):
    """Run the FE solve for ``loading_mode``; return ``(field, skipped_reason)``.

    Exactly one of the pair is ``None``. Any exception from the FE path is
    logged and turned into a reason string rather than propagated, because
    the empirical results do not depend on it. That includes the thermal
    solve's refusal of a mesh that does not resolve the layup.

    ``thermal_mode="combined"`` passes ``delta_T`` to the mechanical solve
    (residual stress superposed); ``"thermal_only"`` solves
    ``loading='thermal'`` at ``delta_T`` instead of ``loading_mode``.
    ``failure_criterion`` is the FE criterion (``FESolver``'s).
    """
    thermal = {} if thermal_mode == "off" else {"delta_T": delta_T}
    try:
        fe_solver = FESolver(mesh, material, porosity_field, ply_angles=angles,
                             failure_criterion=failure_criterion)
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
    ``Float64``) to a temporary file, since the writer takes a path.

    Raises
    ------
    ValueError
        If ``result`` has no FE field (the FE solve was skipped).
    """
    fe_field = result.get("fe_field")
    if fe_field is None:
        raise ValueError("The analysis has no FE field to export: the FE "
                         "solve was skipped.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fe_results.vtu"
        fe_field.to_vtu(result["mesh"], path)
        return path.read_bytes()


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
        governing = "not separated (Tsai-Wu)"
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
# Mesh-convergence check (headless; the Results tab expander drives it)
# ----------------------------------------------------------------------

#: Level sets the convergence check offers: refinement factors of ``nx``
#: and ``ny`` relative to the analysis mesh, ``nz`` fixed. Three levels
#: are needed for an observed order and an extrapolated value.
_CONVERGENCE_LEVELS = {
    "2 levels (×1, ×1.5)": (1.0, 1.5),
    "3 levels (×1, ×1.5, ×2)": (1.0, 1.5, 2.0),
}

#: ``refinement_study`` keeps ``nx`` of an ``'ilss'`` level a multiple of 6
#: so the supports and mid-span of the D2344 specimen stay on node lines.
_ILSS_NX_MULTIPLE = 6

#: Quantities of the convergence tables: (RefinementLevel attribute, label).
_CONVERGENCE_QUANTITIES = (
    ("knockdown", "Knockdown"),
    ("interior_max_failure_index", "Interior max FI"),
    ("max_failure_index", "Domain max FI"),
)


def convergence_unavailable_reason(result: dict) -> str | None:
    """Why the convergence check cannot run on ``result``; ``None`` when it can."""
    if result.get("fe_field") is None:
        return ("The FE solve of this analysis was skipped, so there is no FE "
                "mesh to refine.")
    return None


def convergence_plan(result: dict, factors) -> list[dict]:
    """The levels a convergence check of ``result`` solves, with estimates.

    The levels refine the analysis mesh (the coupon, the D2344 specimen
    for ``'ilss'`` or the D7264 specimen for flexure) in plane, as
    :func:`refinement_study` with ``refine='in_plane'`` does: ``nx`` and
    ``ny`` times each factor, rounded half up (``nx`` then up to a multiple
    of 6 for ``'ilss'``, up to even for flexure), ``nz`` fixed. Each entry
    holds ``resolution``, ``n_elements``, ``n_dof``, ``memory_gb`` (the
    study's own peak-memory estimate, on the generous side) and
    ``time_s``: the analysis FE solve time scaled by the DOF ratio to the
    power 1.5, roughly how the sparse factorization grows (``None`` when
    the result recorded no time). A rough guide only.
    """
    mesh = result["mesh"]
    nx0, ny0, nz0 = int(mesh.nx), int(mesh.ny), int(mesh.nz)
    combined = (result.get("fe_thermal") or {}).get("mode") == "combined"
    t0 = result.get("fe_solve_time_s")
    dof0 = 3 * (nx0 + 1) * (ny0 + 1) * (nz0 + 1)
    plan = []
    for f in factors:
        nx = max(1, math.floor(nx0 * f + 0.5))
        ny = max(1, math.floor(ny0 * f + 0.5))
        if result.get("fe_loading") == "ilss":
            nx = _ILSS_NX_MULTIPLE * math.ceil(nx / _ILSS_NX_MULTIPLE)
        elif result.get("fe_loading") in _FLEXURE_MODES:
            nx += nx % 2
        res = (nx, ny, nz0)
        n_dof = 3 * (nx + 1) * (ny + 1) * (nz0 + 1)
        plan.append({
            "resolution": res,
            "n_elements": nx * ny * nz0,
            "n_dof": n_dof,
            # A combined thermal solve holds two factorizations at its peak.
            "memory_gb": _estimate_memory_bytes(
                res, n_factorizations=2 if combined else 1) / 1e9,
            "time_s": None if not t0 else float(t0) * (n_dof / dof0) ** 1.5,
        })
    return plan


def run_convergence_check(cfg: dict, resolutions) -> RefinementStudy:
    """:func:`refinement_study` of the analysis ``cfg`` on ``resolutions``.

    Same material, layup, porosity field and load case as
    :func:`run_analysis` (thermal-only solves ``'thermal'`` at
    ``cfg["delta_T"]``; ``"combined"`` adds the residual stress), on the
    explicit ``(nx, ny, nz)`` levels of :func:`convergence_plan`. A
    mechanical study accepts an ``nz`` that does not resolve the layup, as
    the analysis does; the thermal solve needs it resolved.

    Raises
    ------
    ValueError
        From :func:`refinement_study`, before any solve: a level above the
        element cap or the memory budget (the message lists every level's
        estimate), or a thermal level whose ``nz`` does not resolve the layup.
    """
    material = dataclasses.replace(
        MATERIALS[cfg["material_name"]], t_ply=cfg["t_ply"], n_plies=cfg["n_plies"])
    pf_kwargs = {"distribution": cfg["distribution"], "void_shape": cfg["void_shape"]}
    if cfg["distribution"] == "clustered":
        pf_kwargs["cluster_location"] = cfg["cluster_location"]
    thermal_mode = cfg.get("thermal_mode", "off")
    loading = "thermal" if thermal_mode == "thermal_only" else cfg["loading_mode"]
    kwargs = {}
    if loading == "ilss":
        kwargs["applied_load"] = _FE_ILSS_LOAD_N
    elif loading in _FLEXURE_MODES:
        kwargs["applied_load"] = _FE_FLEXURE_LOAD_N
    elif loading != "thermal":
        kwargs["applied_strain"] = (-_FE_APPLIED_STRAIN if loading == "compression"
                                    else _FE_APPLIED_STRAIN)
    return refinement_study(
        material, cfg["Vp"] / 100.0, loading,
        porosity_config=pf_kwargs, ply_angles=cfg["angles"],
        resolutions=[tuple(r) for r in resolutions],
        failure_criterion=cfg.get("failure_criterion", "tsai_wu"),
        delta_T=None if thermal_mode == "off" else cfg["delta_T"],
        allow_unresolved_layup=thermal_mode == "off",
        **kwargs,
    )


def convergence_tables(study: RefinementStudy) -> tuple[list[dict], list[dict]]:
    """Rows of the level table and of the per-quantity verdict table.

    Level rows: level, mesh, elements, knockdown, interior and domain
    maximum failure index, solve time. Verdict rows: the quantity, the
    study's verdict (``converged`` / ``mesh-dependent`` / ``not converged``
    / ``n/a``), the behaviour of the levels (``monotone``,
    ``non-monotone``, ``diverging``, ``two levels``, ...) and the last
    relative change (for the knockdown relative to ``1 - KD``).
    """
    levels = []
    for i, lv in enumerate(study.levels, start=1):
        levels.append({
            "Level": i,
            "Mesh": lv.mesh.replace("x", " × "),
            "Elements": f"{lv.n_elements:,}",
            "Knockdown": _fmt_value(lv.knockdown, "{:.4f}"),
            # Significant figures: at the app's arbitrary ILSS / flexure
            # loads the failure indices are of order 0.01.
            "Interior max FI": _fmt_value(lv.interior_max_failure_index, "{:.4g}"),
            "Domain max FI": _fmt_value(lv.max_failure_index, "{:.4g}"),
            "Time (s)": f"{lv.solve_time_s:.1f}",
        })
    verdicts = []
    for name, label in _CONVERGENCE_QUANTITIES:
        c = study.convergence[name]
        last = c.relative_changes[-1] if c.relative_changes else None
        verdicts.append({
            "Quantity": label,
            "Verdict": c.verdict,
            "Behaviour": c.behavior,
            "Last change": _fmt_value(None if last is None else 100.0 * last,
                                      "{:+.2f} %"),
        })
    return levels, verdicts


# ======================================================================
# Plot routines (return a matplotlib Figure for st.pyplot)
# ======================================================================

#: Drawn height over drawn length that the cross-section plots aim for. A
#: coupon is 10 to 40 times longer than it is thick, so at true scale its
#: section is a sliver: the z axis is stretched by the smallest factor of
#: :func:`_exaggeration_factors` that reaches this ratio, and the z axis
#: label says so.
_SECTION_TARGET_ASPECT = 0.3
#: Width of the cross-section figures and the cap on the section's drawn
#: height (inches).
_SECTION_FIG_WIDTH_IN = 9.0
_SECTION_MAX_AXES_HEIGHT_IN = 3.6
#: Margins around the section axes (inches): the title and subtitle above,
#: the x tick labels and axis label between the section and the horizontal
#: colorbar, the colorbar's ticks and label below.
_SECTION_MARGINS_IN = {"left": 1.3, "right": 0.4, "top": 0.95,
                       "gap": 0.75, "cbar": 0.18, "bottom": 0.75}
#: Spread, relative to the magnitude, below which a field is drawn as
#: constant (its colour scale would otherwise span round-off).
_CONSTANT_FIELD_RTOL = 1e-6
#: Half-range of a single-signed stress scale, relative to the mean, below
#: which the colorbar label calls the field nearly uniform.
_NEARLY_UNIFORM_SPREAD = 0.05


def _exaggeration_factors():
    """1, 2, 3, 4, 5, 6, 8, then 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, ..."""
    yield from (1, 2, 3, 4, 5, 6, 8)
    scale = 10
    while True:
        for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
            yield int(round(m * scale))
        scale *= 10


def section_exaggeration(length: float, thickness: float,
                         target: float = _SECTION_TARGET_ASPECT) -> int:
    """Vertical exaggeration of a ``length`` x ``thickness`` section plot.

    The smallest factor of :func:`_exaggeration_factors` that draws the
    section at least ``target`` times as high as it is long; 1 for a
    section that deep already, or when a length is not positive and finite.
    """
    if not (math.isfinite(length) and math.isfinite(thickness)) \
            or length <= 0 or thickness <= 0:
        return 1
    ratio = thickness / length
    for factor in _exaggeration_factors():
        if ratio * factor >= target or factor >= 1_000_000:
            return factor
    return 1   # not reached: the factors never run out


def _z_label(factor: int) -> str:
    """z axis label, naming the vertical exaggeration when there is one."""
    return LABEL_Z_MM if factor == 1 else f"{LABEL_Z_MM}\nz exaggerated ×{factor}"


def _section_figure(x_range, z_range):
    """Figure for a cross-section spanning ``x_range`` x ``z_range`` (mm).

    Returns ``(fig, ax, cax, factor)``: the section axes, stretched along z
    by ``factor`` (:func:`section_exaggeration`) through the data aspect so
    the tick labels stay in mm, and a horizontal colorbar axes of the same
    width under it. The axes are placed in inches, so the colorbar always
    matches the section and the title sits clear of both.
    """
    x0, x1 = (float(v) for v in x_range)
    z0, z1 = (float(v) for v in z_range)
    length, thickness = x1 - x0, z1 - z0
    factor = section_exaggeration(length, thickness)
    drawn = (thickness * factor / length if length > 0 and thickness > 0
             else _SECTION_TARGET_ASPECT)
    m = _SECTION_MARGINS_IN
    fig_w = _SECTION_FIG_WIDTH_IN
    ax_w = fig_w - m["left"] - m["right"]
    ax_h = ax_w * drawn
    if ax_h > _SECTION_MAX_AXES_HEIGHT_IN:
        ax_h = _SECTION_MAX_AXES_HEIGHT_IN
        ax_w = ax_h / drawn
    fig_h = m["top"] + ax_h + m["gap"] + m["cbar"] + m["bottom"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((m["left"] / fig_w, (m["bottom"] + m["cbar"] + m["gap"]) / fig_h,
                       ax_w / fig_w, ax_h / fig_h))
    cax = fig.add_axes((m["left"] / fig_w, m["bottom"] / fig_h,
                        ax_w / fig_w, m["cbar"] / fig_h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    if length > 0 and thickness > 0:
        ax.set_aspect(factor)     # 1 mm of z is drawn as long as `factor` mm of x
    ax.grid(False)
    ax.set_xlabel(LABEL_X_MM)
    ax.set_ylabel(_z_label(factor))
    return fig, ax, cax, factor


def _set_titles(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned bold ``title`` above ``ax``, with a smaller grey ``subtitle``
    line between it and the axes."""
    ax.set_title(title, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, color="0.35")


def _colour_limits(values, *, floor: float, robust: bool = False
                   ) -> tuple[float, float, bool]:
    """Colour-scale limits ``(vmin, vmax, constant)`` of the finite ``values``.

    A field whose spread is at most ``_CONSTANT_FIELD_RTOL`` of its
    magnitude (or 1e-12 absolute) is constant: a scale over its round-off
    would show an offset and meaningless ticks, so the limits are the value
    plus or minus the larger of 1 % of it and ``floor``. Otherwise they are
    the data range or, with ``robust``, its 2nd and 98th percentiles, so a
    few singular peaks do not wash out the rest (the colorbar marks the
    clipped ends). ``(0, 1, False)`` when no value is finite.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0, False
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= max(_CONSTANT_FIELD_RTOL * max(abs(lo), abs(hi)), 1e-12):
        mid = 0.5 * (lo + hi)
        half = max(0.01 * abs(mid), floor)
        return mid - half, mid + half, True
    if robust:
        p_lo, p_hi = (float(p) for p in np.percentile(v, [2.0, 98.0]))
        if p_hi > p_lo:
            lo, hi = p_lo, p_hi
    return lo, hi, False


def _colorbar_extend(values, vmin: float, vmax: float) -> str:
    """Colorbar ``extend`` marking the values outside ``[vmin, vmax]``."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "neither"
    tol = 1e-9 * max(abs(vmin), abs(vmax), 1e-300)
    below, above = bool(v.min() < vmin - tol), bool(v.max() > vmax + tol)
    return {(True, True): "both", (True, False): "min",
            (False, True): "max"}.get((below, above), "neither")


def _finish_colorbar(cb, label: str) -> None:
    """Plain tick labels on ``cb`` (no offset, no scientific notation) and its label."""
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    cb.formatter = formatter
    cb.locator = MaxNLocator(nbins=7)
    cb.update_ticks()
    cb.ax.grid(False)
    cb.set_label(label)


def _mark_uniform_value(cb, value: float) -> None:
    """Mark a constant field's ``value`` on its (horizontal) colorbar."""
    cb.ax.axvline(value, color="k", linewidth=2.0)


def _sequential_cmap(name: str, *, reverse: bool = False) -> ListedColormap:
    """``name`` without its palest fifth, so its low end stays visible on white."""
    colours = matplotlib.colormaps[name](np.linspace(0.2, 1.0, 256))
    return ListedColormap(colours[::-1] if reverse else colours,
                          name=f"{name}{'_r' if reverse else ''}_from_20pct")


#: Stress colormaps. Signed fields crossing zero use RdBu_r on a range
#: symmetric about zero, so its white midpoint is true zero (#51); a field of
#: one sign uses the matching half (blue: compression, darkest = most
#: negative; red: tension), sequential over the field's own range; a
#: magnitude (von Mises) uses the project's cividis (#51).
_STRESS_CMAP_DIVERGING = "RdBu_r"
_STRESS_CMAP_NEGATIVE = _sequential_cmap("Blues", reverse=True)
_STRESS_CMAP_POSITIVE = _sequential_cmap("Reds")
_STRESS_CMAP_MAGNITUDE = "cividis"


def _stress_colour_scale(values, *, magnitude: bool = False):
    """``(cmap, norm, extend, constant)`` for a stress field (MPa).

    The diverging map only when the field crosses zero, otherwise a
    sequential one over the field's range (see ``_STRESS_CMAP_*``), so a
    field of one sign shows its variation instead of one flat colour. The
    range is the 2nd to 98th percentile; a constant field gets its value
    ±1 % (at least ±1 MPa).
    """
    vmin, vmax, constant = _colour_limits(values, floor=1.0, robust=True)
    if magnitude:
        cmap = _STRESS_CMAP_MAGNITUDE
    elif vmin < 0.0 < vmax:
        v = max(-vmin, vmax)
        vmin, vmax = -v, v
        cmap = _STRESS_CMAP_DIVERGING
    elif vmax <= 0.0:
        cmap = _STRESS_CMAP_NEGATIVE
    else:
        cmap = _STRESS_CMAP_POSITIVE
    return (cmap, Normalize(vmin=vmin, vmax=vmax),
            _colorbar_extend(values, vmin, vmax), constant)


def plot_profile(result: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    pf = result["porosity_field"]
    z, Vp = pf.effective_porosity_profile(nz=200)
    ax.plot(Vp * 100, z, "b-", linewidth=2)
    ax.set_xlabel(LABEL_POROSITY_PCT)
    ax.set_ylabel(LABEL_Z_MM)
    ax.set_title("Through-Thickness Porosity Profile")
    ax.set_xlim(left=0)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    fig.tight_layout()
    return fig


def plot_mesh(result: dict):
    """Mid-y cross-section coloured by stiffness retention with void overlays."""
    mesh = result["mesh"]
    section = mesh.mid_y_section_indices()   # (nz + 1, nx + 1)
    X = mesh.nodes[section, 0]
    Z = mesh.nodes[section, 2]
    Sr = mesh.stiffness_reduction[section] * 100.0
    fig, ax, cax, factor = _section_figure((X.min(), X.max()), (Z.min(), Z.max()))

    vmin, vmax, constant = _colour_limits(Sr, floor=1.0)
    if constant:
        vmax = min(vmax, 100.0)      # retention never exceeds 100 %
    else:
        # 100 % (pristine) anchors the top of the scale.
        vmin, vmax = max(0.0, vmin - 1.0), max(100.0, vmax)
    im = ax.pcolormesh(X, Z, Sr, shading="gouraud", cmap="cividis",
                       vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    label = LABEL_STIFFNESS_RETENTION
    if constant:
        value = float(np.mean(Sr))
        label += f" · uniform: {value:.1f} %"
        _mark_uniform_value(cb, value)
    _finish_colorbar(cb, label)

    step_x = max(1, mesh.nx // 20)
    step_z = max(1, mesh.nz // 24)
    for k in range(0, mesh.nz + 1, step_z):
        ax.plot(X[k], Z[k], "k-", linewidth=0.3, alpha=0.4)
    for i in range(0, mesh.nx + 1, step_x):
        ax.plot(X[:, i], Z[:, i], "k-", linewidth=0.3, alpha=0.4)

    n_voids = 0
    void_elems = mesh.void_elements
    if len(void_elems) > 0:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        void_patches = []
        for e_idx in void_elems:
            j_e = (e_idx // mesh.nx) % mesh.ny
            if j_e != mesh.ny // 2:
                continue
            node_coords = mesh.nodes[mesh.elements[e_idx]]
            xz = node_coords[:, [0, 2]]
            unique_xz = np.unique(xz, axis=0)
            if len(unique_xz) < 3:
                continue
            cx_p, cz_p = unique_xz.mean(axis=0)
            angles = np.arctan2(
                unique_xz[:, 1] - cz_p, unique_xz[:, 0] - cx_p,
            )
            order = np.argsort(angles)
            void_patches.append(Polygon(unique_xz[order], closed=True))

        if void_patches:
            pc = PatchCollection(
                void_patches, facecolor="white", edgecolor="red",
                linewidth=1.0, zorder=5, alpha=1.0,
            )
            ax.add_collection(pc)
            n_voids = len(void_patches)

    subtitle = (f"mid-y section · {len(mesh.nodes):,} nodes, "
                f"{len(mesh.elements):,} elements")
    if n_voids:
        subtitle += f" · {n_voids} void elements (white, red outline)"
    _set_titles(ax, "FE mesh: stiffness retention", subtitle)
    return fig


def plot_results(result: dict, layup_str: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    emp = result["empirical"]
    fe_field = result.get("fe_field")
    fe_loading = result.get("fe_loading", "compression")
    cfg = result["config"]
    layup_scale = result.get("layup_scale", {})

    modes = ["compression", "tension", "shear", "ilss"]
    group_labels = [m.upper() for m in modes]
    # (key, legend label, colour, hatch)
    series: list[tuple[str, str, str, str | None]] = [
        ("judd_wright", "Judd-Wright", "#1f77b4", None),
        ("power_law", "Power Law", "#ff7f0e", None),
        ("linear", "Linear", "#2ca02c", None),
    ]

    # A thermal-only solve has no stiffness measure (knockdown NaN): no bar.
    has_fe = fe_field is not None and bool(np.isfinite(fe_field.knockdown))
    flexure = has_fe and fe_loading in _FLEXURE_MODES
    if has_fe:
        series.append(("fe", f"FE Stiffness ({fe_loading})", "#d62728", "//"))
    if flexure:
        # FE only: its own group, after the empirical modes.
        modes.append(fe_loading)
        group_labels.append("FLEXURE (FE)")

    def knockdown(key: str, mode: str) -> float:
        if key == "fe":
            return float(fe_field.knockdown) if mode == fe_loading else float("nan")
        return float(emp[mode][key]["knockdown"]) if mode in emp else float("nan")

    x = np.arange(len(modes))
    width = 0.8 / len(series)
    labelled: set[str] = set()
    for j, mode in enumerate(modes):
        bars = [(s, knockdown(s[0], mode)) for s in series]
        bars = [(s, v) for s, v in bars if np.isfinite(v)]
        # Centre the bars each group has (the FE bar is in one group only).
        offsets = (np.arange(len(bars)) - (len(bars) - 1) / 2.0) * width
        for ((key, label, colour, hatch), value), off in zip(bars, offsets, strict=True):
            ax.bar(x[j] + off, value, width, color=colour, hatch=hatch,
                   edgecolor="white" if hatch is None else "0.3",
                   label="_nolegend_" if key in labelled else label)
            labelled.add(key)
    if flexure:
        ax.axvline(x[-1] - 0.5, color="0.6", linewidth=0.8, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(LABEL_KNOCKDOWN)
    _set_titles(
        ax, "Knockdown factor by loading mode",
        f"Vp = {cfg['Vp']:.1f} %, {cfg['void_shape']} voids, "
        f"{_distribution_text(cfg)} distribution, {layup_str}")
    ax.set_ylim(0, 1.1)
    # Outside the axes, right of the bars, so it never covers one.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    note = ("Solid bars = strength knockdown (at mean Vp); "
            "hatched bar = stiffness knockdown (FE)")
    if fe_field is not None and not has_fe:
        note = ("Solid bars = strength knockdown (at mean Vp); the FE "
                "thermal-only solve has no stiffness knockdown (n/a)")
    if flexure:
        note += ("\nFLEXURE: FE beam-stiffness knockdown only; there is no "
                 "empirical flexure model")
    amplified = {m: s for m, s in layup_scale.items() if s > 1.0}
    if amplified:
        factors = ", ".join(f"{m} x{s:.2f}" for m, s in sorted(amplified.items()))
        note += (f"\nLayup amplification ({factors}): layup more "
                 "matrix-dominated than the QI calibration; not validated")
    # Footnote intentionally smaller than rcParams.font.size (annotation, not
    # primary data); explicit override kept here on purpose. It sits below
    # the tick labels (not inside the axes, where it ran under the legend);
    # tight_layout makes room for it.
    ax.annotate(note, xy=(0.0, 0.0), xycoords="axes fraction",
                xytext=(0, -24), textcoords="offset points",
                fontsize=7, color="0.4", ha="left", va="top")

    fig.tight_layout()
    return fig


_STRESS_COMPONENTS = {
    "σ₁₁ (fiber)": (0, r"$\sigma_{11}$ local (MPa)"),
    "σ₂₂ (transverse)": (1, r"$\sigma_{22}$ local (MPa)"),
    "σ₃₃ (through-thickness)": (2, r"$\sigma_{33}$ local (MPa)"),
    "τ₂₃ (interlaminar)": (3, r"$\tau_{23}$ local (MPa)"),
    "τ₁₃ (interlaminar)": (4, r"$\tau_{13}$ local (MPa)"),
    "τ₁₂ (in-plane shear)": (5, r"$\tau_{12}$ local (MPa)"),
    "Von Mises": (-1, "Von Mises Stress (MPa)"),
}

#: File-name slug of each stress component (by component index).
_STRESS_SLUGS = {0: "s11", 1: "s22", 2: "s33", 3: "t23", 4: "t13", 5: "t12",
                 -1: "von_mises"}


def default_stress_component(result: dict) -> str:
    """The stress component the Stress tab opens on for ``result``.

    The one that governs the load case: τ₁₃ for ILSS (interlaminar shear),
    σ₂₂ for a thermal-only solve (transverse stress, where residual cracking
    starts), τ₁₂ for in-plane shear, σ₁₁ otherwise (fibre stress, also in
    flexure).
    """
    fe_loading = result.get("fe_loading")
    if fe_loading == "ilss":
        return "τ₁₃ (interlaminar)"
    if fe_loading == "thermal":
        return "σ₂₂ (transverse)"
    if fe_loading == "shear":
        return "τ₁₂ (in-plane shear)"
    return "σ₁₁ (fiber)"


def plot_uq(uq: dict):
    """Histogram of sampled knockdowns with the p5 / p50 / p95 band."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    kd = np.asarray(uq["samples"]["knockdown"])
    pct = uq["knockdown"]["percentiles"]
    ax.hist(kd, bins=40, color="0.7", edgecolor="0.4")
    ax.axvspan(pct["p5"], pct["p95"], color="tab:blue", alpha=0.15,
               label=f"p5-p95: {pct['p5']:.3f}-{pct['p95']:.3f}")
    ax.axvline(pct["p50"], color="tab:blue", linewidth=2,
               label=f"p50: {pct['p50']:.3f}")
    ax.axvline(uq["nominal"]["knockdown"], color="k", linestyle="--",
               label=f"nominal: {uq['nominal']['knockdown']:.3f}")
    ax.set_xlabel(LABEL_KNOCKDOWN)
    ax.set_ylabel("Draws")
    ax.set_title(f"Knockdown uncertainty: {uq['mode']}, {uq['n_samples']} draws")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    # Outside the axes, so it never covers the histogram's tails.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False)
    fig.tight_layout()
    return fig


def plot_stress(result: dict, comp_name: str):
    fe_field = result.get("fe_field")
    if fe_field is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No FE results available.",
                transform=ax.transAxes, ha="center", va="center",
                color="0.5")
        ax.set_axis_off()
        return fig

    mesh = result["mesh"]
    stress_local = fe_field.stress_local
    comp_idx, label = _STRESS_COMPONENTS.get(comp_name, (0, comp_name + " (MPa)"))

    if comp_idx == -1:
        s = stress_local.mean(axis=1)
        s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]
        s4, s5, s6 = s[:, 3], s[:, 4], s[:, 5]
        elem_stress = np.sqrt(0.5 * (
            (s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2
            + 6.0 * (s4 ** 2 + s5 ** 2 + s6 ** 2)
        ))
    else:
        elem_stress = stress_local.mean(axis=1)[:, comp_idx]

    # Mid-y element row (nz, nx) and its nodes (nz + 1, nx + 1); the
    # structured mesh makes the nodes the corners of the element cells.
    elems = mesh.mid_y_element_indices()
    section = mesh.mid_y_section_indices()
    X = mesh.nodes[section, 0]
    Z = mesh.nodes[section, 2]
    cx = mesh.nodes[mesh.elements[elems[0]]][:, :, 0].mean(axis=1)
    x_lo, x_hi = float(X.min()), float(X.max())
    # Leave out the 10 % at each end (grips, supports, free-edge peaks).
    cols = np.flatnonzero((cx > x_lo + 0.10 * (x_hi - x_lo))
                          & (cx < x_lo + 0.90 * (x_hi - x_lo)))
    sv = elem_stress[elems][:, cols] if cols.size else np.empty((0, 0))
    if cols.size == 0 or np.isfinite(sv).sum() < 3:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Insufficient interior data for a stress plot.",
                transform=ax.transAxes, ha="center", va="center",
                color="0.5")
        ax.set_axis_off()
        return fig
    i0, i1 = int(cols[0]), int(cols[-1])
    Xc, Zc = X[:, i0:i1 + 2], Z[:, i0:i1 + 2]

    fig, ax, cax, factor = _section_figure((Xc.min(), Xc.max()), (Zc.min(), Zc.max()))
    cmap, norm, extend, constant = _stress_colour_scale(sv, magnitude=comp_idx == -1)
    # One flat cell per element: the element-averaged stress, which jumps
    # between plies (a contour would smear it across the interfaces).
    qm = ax.pcolormesh(Xc, Zc, np.ma.masked_invalid(sv), cmap=cmap, norm=norm,
                       shading="flat")
    cb = fig.colorbar(qm, cax=cax, orientation="horizontal", extend=extend)
    if constant:
        value = float(np.nanmean(sv))
        label += f" · uniform: {value:.4g} MPa"
        _mark_uniform_value(cb, value)
    elif norm.vmin * norm.vmax > 0.0:
        # A sequential scale over a narrow range makes small differences
        # look large: say how small they are.
        mid = 0.5 * (norm.vmin + norm.vmax)
        spread = 0.5 * (norm.vmax - norm.vmin) / abs(mid)
        if spread < _NEARLY_UNIFORM_SPREAD:
            label += f" · nearly uniform: {mid:.4g} MPa ± {100 * spread:.2g} %"
    _finish_colorbar(cb, label)
    _set_titles(ax, f"FE stress: {comp_name}",
                "ply (material) axes · mid-y section, middle 80 % of the "
                "length · element averages")
    return fig


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


def _sync_stress_component(default: str) -> None:
    """Point the Stress tab's component selector at the load case's default.

    Same rule as :func:`_sync_default_nz`: the selector follows
    :func:`default_stress_component` while it still shows the previous
    default (or has no value yet); a component the user picked stays. Must
    run before the selector is created in the script run.
    """
    previous = st.session_state.get("_stress_component_default")
    current = st.session_state.get("stress_component")
    if current is None or current == previous or current not in _STRESS_COMPONENTS:
        st.session_state["stress_component"] = default
    st.session_state["_stress_component_default"] = default


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
                "is outside the empirical validation (README, Limitations)."
            ),
        )

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
                "shear strengths). There is no explicit delamination in "
                "the model: this screens initiation only."
            ),
        )

        thermal_mode, delta_T = _build_thermal_inputs(material_name)

        # Ply count of the laminate as entered (the analysis replaces the
        # preset's n_plies with it); the preset's when the layup is invalid.
        try:
            n_plies = len(parse_layup(layup_str))
        except ValueError:
            n_plies = 0
        nx0, ny0, nz0 = _default_mesh_resolution(
            n_plies or MATERIALS[material_name].n_plies)
        mesh_note = ("one element layer per ply" if not n_plies or nz0 == n_plies
                     else f"nz capped at {nz0} for the {n_plies}-ply layup")

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
            # plies; say so before the run, with the nz that resolves them.
            if n_plies and nz % n_plies:
                nz_fix = _resolving_nz(nz, n_plies)
                st.warning(
                    f"Thermal residual stress needs every ply in its own "
                    f"element layer: nz = {nz} layers do not resolve the "
                    f"{n_plies}-ply layup, so the FE solve will refuse it if "
                    f"the merged plies differ in angle. Use nz = {nz_fix} "
                    f"(a multiple of {n_plies}).")
                _offer_nz_fix(nz_fix, key="nz_fix_sidebar")

        _layup_bad = st.session_state.get("_layup_status", ("ok", None))[0] == "err"
        _missing_dT = thermal_mode != "off" and delta_T is None
        _run_disabled = (
            _layup_bad
            or nx * ny * nz > CompositeMesh._MAX_TOTAL_ELEMENTS
            or _missing_dT
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
    }
    return {"cfg": cfg, "layup_str": layup_str, "run": run}


def _show_figure(fig, file_stem: str, key: str) -> None:
    """Render ``fig`` in the page and offer it as a PNG download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.download_button(
        "Download PNG", data=buf.getvalue(), file_name=f"{file_stem}.png",
        mime="image/png", key=key,
    )


def _placeholder_tab():
    st.info("Run an analysis to populate this tab.")


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
        lamination-theory value.

        - **Profile** — through-thickness porosity distribution
        - **Mesh** — mid-y cross-section of the FE mesh, coloured by
          stiffness retention (thin sections are drawn with the thickness
          exaggerated; the axis label gives the factor)
        - **Results** — empirical knockdown bar chart with the FE stiffness
          knockdown overlaid, the short-beam, flexure and thermal
          residual-stress results of the FE solve, an uncertainty analysis
          and an optional mesh-convergence check
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
    st.success(
        f"Last run: {cfg_r['material_name']}, layup {layup_for_title}, "
        f"Vp = {cfg_r['Vp']:.1f}%, {cfg_r['void_shape']}, "
        f"{cfg_r['distribution']}, {mesh_note}."
        + thermal_note
    )
    if result.get("fe_skipped_reason"):
        reason = result["fe_skipped_reason"]
        st.warning(
            f"⚠ FE solve was skipped: {reason}\n\n"
            f"**Empirical knockdown results are still valid** and are shown in the "
            f"other tabs. FE would have added per-element stress fields; you don't "
            f"need it for the headline knockdown numbers.\n\n"
            f"To retry with FE: adjust the mesh (Expert tab) or pick a different "
            f"loading mode. See the [README]({_README_URL}) for the FE-supported modes."
        )
        _offer_nz_fix(result.get("fe_nz_suggestion"), key="nz_fix_overview")


def _build_profile_tab(result: dict | None):
    if result is None:
        _placeholder_tab()
        return
    _show_figure(plot_profile(result), "porosity_profile", "dl_png_profile")


def _build_mesh_tab(result: dict | None):
    if result is None:
        _placeholder_tab()
        return
    _show_figure(plot_mesh(result), "mesh_section", "dl_png_mesh")


def _build_results_tab(result: dict | None, layup_for_title: str):
    if result is None:
        _placeholder_tab()
        return
    _show_figure(plot_results(result, layup_for_title), "knockdown_results",
                 "dl_png_results")
    _build_fe_failure_results(result)
    _build_short_beam_results(result)
    _build_flexure_results(result)
    _build_thermal_results(result)
    _build_uq_expander(result)
    _build_convergence_expander(result)


def _build_fe_failure_results(result: dict) -> None:
    """FE failure index, governing mode and per-mode maxima (any FE run)."""
    summary = fe_failure_summary(result)
    if summary is None:
        return
    rows, mode_rows, caption = summary
    st.subheader("FE failure")
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    if mode_rows:
        st.dataframe(mode_rows, hide_index=True, use_container_width=True)
    st.caption(caption)


def _build_flexure_results(result: dict) -> None:
    """Metrics of the ASTM D7264 flexure FE solve (nothing for other modes)."""
    summary = fe_flexure_summary(result)
    if summary is None:
        return
    procedure = result["fe_flexure"]["procedure"]
    st.subheader("FE flexure (ASTM D7264 procedure "
                 f"{'A, three-point' if procedure == '3pt' else 'B, four-point'})")
    rows, caption = summary
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    st.caption(caption)


def _build_convergence_expander(result: dict) -> None:
    """Opt-in mesh-convergence check of the analysis' FE solve (refinement_study)."""
    with st.expander("Mesh convergence check"):
        st.caption(
            "Solves the analysis' FE load case again on in-plane refinements "
            "of its mesh (nx and ny scaled, nz fixed) and reports how the "
            "results change. Global results such as the knockdown converge. "
            "The domain-maximum failure index sits at a free edge, corner or "
            "load point, where the linear-elastic stress is singular: it "
            "keeps growing with refinement, so it is mesh-dependent by "
            "nature. Read the interior values, which leave out a fixed margin "
            "along the edges.")
        reason = convergence_unavailable_reason(result)
        if reason is not None:
            st.info(reason)
            return
        choice = st.radio("Levels", list(_CONVERGENCE_LEVELS), key="conv_levels",
                          horizontal=True,
                          help="Three levels give an observed order of "
                               "convergence and an extrapolated value; two "
                               "give only the change between them.")
        plan = convergence_plan(result, _CONVERGENCE_LEVELS[choice])
        st.dataframe(
            [{"Level": i,
              "Mesh": " × ".join(str(n) for n in lv["resolution"]),
              "Elements": f"{lv['n_elements']:,}",
              "Est. peak memory": f"{lv['memory_gb']:.2g} GB",
              "Est. time": ("unknown" if lv["time_s"] is None
                            else f"~{lv['time_s']:.0f} s")}
             for i, lv in enumerate(plan, start=1)],
            hide_index=True, use_container_width=True)
        times = [lv["time_s"] for lv in plan]
        total = ("an unknown time" if None in times
                 else f"about {sum(t for t in times if t is not None):.0f} s")
        st.caption(
            f"Before running: {len(plan)} levels, {total} in all (the "
            f"analysis' FE solve time scaled by the DOF ratio to the power "
            f"1.5; a rough guide). The levels run one after another, so the "
            f"peak memory is the finest level's.")
        cfg = result["config"]
        params = (_config_to_key(cfg), tuple(lv["resolution"] for lv in plan))
        if st.button("Check mesh convergence", key="conv_run"):
            st.session_state["conv_params"] = params
        if st.session_state.get("conv_params") != params:
            return
        with st.spinner("Solving the refined meshes..."):
            try:
                study = run_convergence_cached(*params)
            except ValueError as exc:
                st.error(f"The convergence check did not run: {exc}")
                return
        levels, verdicts = convergence_tables(study)
        st.dataframe(levels, hide_index=True, use_container_width=True)
        st.dataframe(verdicts, hide_index=True, use_container_width=True)
        st.caption(
            f"Verdicts at a tolerance of {study.tol:.0%}: 'converged' (the "
            f"estimated error of the finest mesh is within it), 'not "
            f"converged' (refine further), 'mesh-dependent' (grows with "
            f"refinement: a stress singularity, do not use the value). The "
            f"knockdown's change is relative to the stiffness loss 1 − KD. "
            f"Interior quantities leave out {study.interior_margin:.3g} mm "
            + ("around the supports and the loading band and from the side "
               "faces." if study.loading == "ilss" else
               "around the support and load lines and from the side faces."
               if study.loading in _FLEXURE_MODES else "from the lateral faces."))


def _build_short_beam_results(result: dict) -> None:
    """Metrics of the ASTM D2344 short-beam FE solve (nothing for other modes)."""
    summary = fe_short_beam_summary(result)
    if summary is None:
        return
    st.subheader("FE short-beam shear (ASTM D2344)")
    rows, caption = summary
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    st.caption(caption)


def _build_thermal_results(result: dict) -> None:
    """Metrics of the FE thermal load (nothing when it was off)."""
    thermal = result.get("fe_thermal")
    if not thermal:
        return
    what = ("thermal only" if thermal["mode"] == "thermal_only"
            else f"residual stress + {result['config']['loading_mode']}")
    st.subheader(f"FE thermal residual stress ({what}, ΔT = {thermal['delta_T']:g} K)")
    summary = fe_thermal_summary(result)
    if summary is None:
        st.warning(
            f"The thermal FE solve was skipped: {result.get('fe_skipped_reason')}")
        _offer_nz_fix(result.get("fe_nz_suggestion"), key="nz_fix_results")
        return
    rows, caption = summary
    for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                              strict=True):
        col.metric(label, value, help=help_text)
    st.caption(caption)


def _build_uq_expander(result: dict):
    """Opt-in uncertainty propagation on the analysed laminate (plan 3.4)."""
    with st.expander("Uncertainty (Latin hypercube sampling)"):
        st.caption(
            "Propagates scatter in the Judd-Wright calibration coefficient, "
            "the measured porosity and the pristine strength through the "
            "empirical knockdown. Each CoV is a fraction (0.10 = 10 % of the "
            "value), not a percent of porosity. The defaults are assumptions; "
            "set them to your own data.")
        modes = list(EmpiricalSolver.PRISTINE_STRENGTH_KEY)
        cfg = result["config"]
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox(
                "Mode", modes, key="uq_mode",
                index=modes.index(cfg["loading_mode"])
                if cfg["loading_mode"] in modes else 0)
            n_samples = st.number_input(
                "Draws", min_value=50, max_value=5000, value=500, step=50,
                key="uq_samples")
        with c2:
            coef_cov = st.number_input(
                "Coefficient CoV", min_value=0.0, max_value=1.0, value=0.10,
                step=0.01, key="uq_coef_cov",
                help="Coefficient of variation of the coefficient, as a fraction: 0.10 means 10 %.")
            vp_cov = st.number_input(
                "Porosity CoV", min_value=0.0, max_value=1.0, value=0.10,
                step=0.01, key="uq_vp_cov",
                help="Coefficient of variation of the measured porosity, as a fraction: 0.10 means 10 %.")
        with c3:
            strength_cov = st.number_input(
                "Strength CoV", min_value=0.0, max_value=1.0, value=0.05,
                step=0.01, key="uq_strength_cov",
                help="Coefficient of variation of the pristine strength, as a fraction: 0.10 means 10 %.")
        params = (_config_to_key(cfg), mode, int(n_samples), float(coef_cov),
                  float(vp_cov), float(strength_cov))
        if st.button("Run uncertainty analysis", key="uq_run"):
            st.session_state["uq_params"] = params
        if st.session_state.get("uq_params") != params:
            return
        with st.spinner("Sampling..."):
            uq, messages = run_uq_cached(*params)
        for msg in messages:
            st.warning(msg)
        kd, fs = uq["knockdown"]["percentiles"], uq["failure_stress"]["percentiles"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Knockdown p5", f"{kd['p5']:.3f}", f"{fs['p5']:.0f} MPa",
                  delta_color="off")
        m2.metric("Knockdown p50", f"{kd['p50']:.3f}", f"{fs['p50']:.0f} MPa",
                  delta_color="off")
        m3.metric("Knockdown p95", f"{kd['p95']:.3f}", f"{fs['p95']:.0f} MPa",
                  delta_color="off")
        _show_figure(plot_uq(uq), f"knockdown_uq_{mode}", "dl_png_uq")


def _build_stress_tab(result: dict | None):
    if result is None:
        _placeholder_tab()
        return
    if result.get("fe_field") is None:
        reason = result.get("fe_skipped_reason")
        st.warning(
            f"FE solve was skipped: {reason}" if reason
            else "No FE field available for this configuration."
        )
        return
    # Opens on the component that governs the load case; a component the
    # user picks stays picked (the default goes through session state).
    _sync_stress_component(default_stress_component(result))
    comp_name = st.selectbox(
        "Stress component",
        options=list(_STRESS_COMPONENTS.keys()),
        key="stress_component",
    )
    slug = _STRESS_SLUGS[_STRESS_COMPONENTS[comp_name][0]]
    _show_figure(plot_stress(result, comp_name), f"stress_{slug}", "dl_png_stress")
    formulation = getattr(result["fe_field"], "formulation", None)
    if formulation:
        st.caption(f"FE element formulation: {formulation}.")
    thermal = result.get("fe_thermal")
    if thermal:
        st.caption(
            f"Thermal residual stress of the free-standing laminate at "
            f"ΔT = {thermal['delta_T']:g} K"
            + (" (no mechanical load)." if thermal["mode"] == "thermal_only"
               else f", plus the {result['config']['loading_mode']} load: the "
                    f"plot shows total stress."))


def _build_export_tab(result: dict | None, layup_for_title: str):
    if result is None:
        _placeholder_tab()
        return
    payload = build_export_payload(result)
    export_stem = download_filename_stem(payload)
    payload_json = _serialise_payload_json(payload)
    st.download_button(
        "Download JSON",
        data=payload_json,
        file_name=f"{export_stem}.json",
        mime="application/json",
        use_container_width=True,
        key="dl_export_json",
    )
    st.download_button(
        "Download CSV",
        data=_serialise_payload_csv(payload),
        file_name=f"{export_stem}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_export_csv",
    )
    if result.get("fe_field") is not None:
        fe_slug = result["fe_loading"]
        if (result.get("fe_thermal") or {}).get("mode") == "combined":
            fe_slug += "_residual"
        st.download_button(
            "Download FE results (VTU)",
            data=fe_vtu_bytes_cached(_config_to_key(result["config"]), result),
            file_name=f"{export_stem}_fe_{fe_slug}.vtu",
            mime="application/octet-stream",
            use_container_width=True,
            key="dl_export_vtu",
            help=(
                "Binary VTK XML (.vtu) of the FE mesh for ParaView: "
                "displacement, porosity, element stress and strain, failure "
                "index and ply data, nodal stress and strain recovered "
                "within each ply, and the residual stress of a thermal run."
            ),
        )
    else:
        st.caption("No FE results to download as VTU: the FE solve was skipped.")
    with st.expander("Preview JSON"):
        st.code(payload_json, language="json")

    st.divider()
    st.subheader("NCR validation summary")
    st.caption(
        "Generate a concise analysis summary to **attach to an NCR**. "
        "It carries the porosity validation and a *recommended* "
        "disposition path for the MRB — it is not a full NCR form and "
        "does not issue a final disposition. Part/serial/work-order "
        "identification stays on the parent NCR."
    )

    with st.form("ncr_form"):
        c1, c2 = st.columns(2)
        with c1:
            prepared_by = st.text_input(
                "Prepared by", value="",
                help="Engineer preparing this analysis summary.",
            )
            ncr_reference = st.text_input(
                "Parent NCR reference (optional)", value="",
                help="Cross-reference to the NCR this attaches to.",
            )
        with c2:
            structural_class = st.selectbox(
                "Structural classification",
                options=list(STRUCTURAL_CLASSES),
                index=0,
                help=(
                    "Drives required substantiation. Primary "
                    "structure escalates to customer/DER concurrence "
                    "for any Use-As-Is."
                ),
            )
        note = st.text_area(
            "Engineer note (optional)", value="",
            help="Any observation context to record on the summary.",
        )
        make_ncr = st.form_submit_button(
            "Generate summary", type="primary",
            use_container_width=True,
        )

    if not make_ncr:
        return

    meta = {
        "prepared_by": prepared_by,
        "ncr_reference": ncr_reference,
        "structural_class": structural_class,
        "note": note,
        "date": datetime.date.today().isoformat(),
        "layup": layup_for_title,
    }
    ncr = build_ncr_record(result, meta)
    dp = ncr["recommended_disposition"]
    st.warning(
        f"**Recommended disposition path:** {dp['path']}  \n"
        f"{dp['rationale']}"
    )
    st.info(dp["disclaimer"])

    ncr_md = serialise_ncr_markdown(ncr)
    stem = (
        _sanitise_filename_component(ncr_reference.strip())
        or export_stem
    )
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download PDF",
            data=serialise_ncr_pdf(ncr),
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_ncr_pdf",
        )
    with dl2:
        st.download_button(
            "Download Markdown",
            data=ncr_md,
            file_name=f"{stem}.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_ncr_md",
        )
    with dl3:
        st.download_button(
            "Download JSON",
            data=serialise_ncr_json(ncr),
            file_name=f"{stem}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_ncr_json",
        )
    with st.expander("Preview summary"):
        st.markdown(ncr_md)


def _render():
    st.set_page_config(
        page_title="PorosityFE",
        page_icon=None,
        layout="wide",
    )
    st.title("PorosityFE — Composite Laminate Porosity Analysis")
    st.caption(
        "Predict strength and stiffness knockdown in porosity-degraded "
        "composite laminates. Adjust inputs in the sidebar, then click **Run analysis**."
    )

    # ---- Layup default and initial validation -----------------------------
    # The layup input takes its default only from st.session_state (the
    # widget has no value=), so it never overwrites a value the user typed;
    # the status badge shows immediately for the default.
    st.session_state.setdefault("layup_input", _DEFAULT_LAYUP)
    if "_layup_status" not in st.session_state:
        _validate_layup_inline()

    sidebar = _build_sidebar_inputs()
    if sidebar is None:
        return
    cfg = sidebar["cfg"]
    layup_str = sidebar["layup_str"]

    # ---- Run analysis (only when the button is pressed) --------------------
    if sidebar["run"]:
        with st.spinner("Running porosity analysis…"):
            try:
                st.session_state["result"] = run_analysis_cached(_config_to_key(cfg))
                st.session_state["layup_str"] = layup_str
            except Exception as exc:
                logger.exception("Analysis failed")
                st.session_state["result"] = None
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("result")
    layup_for_title = st.session_state.get("layup_str", layup_str)

    tabs = st.tabs(["Overview", "Profile", "Mesh", "Results", "Stress", "Export"])
    with tabs[0]:
        _build_overview_tab(result, layup_for_title)
    with tabs[1]:
        _build_profile_tab(result)
    with tabs[2]:
        _build_mesh_tab(result)
    with tabs[3]:
        _build_results_tab(result, layup_for_title)
    with tabs[4]:
        _build_stress_tab(result)
    with tabs[5]:
        _build_export_tab(result, layup_for_title)


try:
    from streamlit.runtime import exists as _st_runtime_exists
    _UNDER_STREAMLIT = _st_runtime_exists()
except Exception:
    _UNDER_STREAMLIT = False

if _UNDER_STREAMLIT:
    _render()
