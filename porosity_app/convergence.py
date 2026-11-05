"""The Results tab's mesh-convergence check (``refinement_study`` on the analysis).

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import dataclasses
import math

import streamlit as st

from porosity_fe import (
    MATERIALS,
    RefinementStudy,
    refinement_study,
)
from porosity_fe.fe.refinement import _estimate_memory_bytes, _peak_factorizations

from .config import (
    _CFG_DEFAULTS,
    _FE_APPLIED_STRAIN,
    _FE_FLEXURE_LOAD_N,
    _FE_ILSS_LOAD_N,
    _FLEXURE_MODES,
    _FRACTURE_MODES,
)
from .summaries import _fmt_value


# A study keeps only scalars per level (a few kB), but each entry stands
# for minutes of solving; a handful covers going back and forth.
@st.cache_data(show_spinner=False, max_entries=8)
def run_convergence_cached(cfg_key: tuple, resolutions: tuple) -> RefinementStudy:
    """Cached :func:`run_convergence_check` of the analysis config ``cfg_key``."""
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg_key}
    return run_convergence_check(cfg, resolutions)


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
    if result.get("delamination") is not None or \
            result.get("fe_loading") in _FRACTURE_MODES:
        return ("The convergence check does not cover a delamination or a "
                "fracture specimen yet: refine nx and ny in Expert mode and "
                "compare the front's G_T.")
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
    # The factorizations alive at once, as refinement_study plans them: one,
    # since a combined thermal solve frees its thermal factorization before
    # the mechanical one is built; two for a fracture specimen's contact loop.
    n_factorizations = _peak_factorizations(str(result.get("fe_loading")), "direct")
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
            "memory_gb": _estimate_memory_bytes(
                res, n_factorizations=n_factorizations) / 1e9,
            "time_s": None if not t0 else float(t0) * (n_dof / dof0) ** 1.5,
        })
    return plan


def run_convergence_check(cfg: dict, resolutions) -> RefinementStudy:
    """:func:`refinement_study` of the analysis ``cfg`` on ``resolutions``.

    Same material, layup, porosity field, porosity degradation and load
    case as :func:`run_analysis` (thermal-only solves ``'thermal'`` at
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
        porosity_degradation=cfg.get("porosity_degradation",
                                     _CFG_DEFAULTS["porosity_degradation"]),
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
