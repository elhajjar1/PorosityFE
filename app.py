"""Porosity FE Analysis — Streamlit web app.

Browser front end for `porosity_fe`. Mirrors the structure used by
WrinkleFE: a sidebar of inputs feeds a cached analysis function whose result
fans out to Profile / Mesh / Results / Stress / Export tabs.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import streamlit as st  # noqa: F401  re-exported: tests reach st through app

# Make sibling modules importable when launched via `streamlit run app.py`
# from a checkout that hasn't been `pip install -e .`'d.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# The library names the app used to import here, kept importable from
# ``app`` for scripts that reach them through it.
from porosity_fe import (  # noqa: F401
    LABEL_KNOCKDOWN,
    LABEL_POROSITY_PCT,
    LABEL_STIFFNESS_RETENTION,
    LABEL_X_MM,
    LABEL_Z_MM,
    MATERIALS,
    CompositeMesh,
    Delamination,
    EmpiricalSolver,
    FESolver,
    RefinementStudy,
    ToughnessKnockdown,
    _configure_matplotlib_style,
    build_empirical_pipeline,
    compute_degraded_clt_flexural_modulus,
    knockdown_decomposition,
    mmb_lever_length,
    propagate_uncertainty,
    refinement_study,
    short_beam_shear,
)
from porosity_fe.fe.failure import first_ply_failure_load_factor  # noqa: F401
from porosity_fe.fe.refinement import _estimate_memory_bytes  # noqa: F401
from porosity_fe.fracture_specimens import (  # noqa: F401
    _D5528_CRACK_LENGTH,
    _D6671_CRACK_LENGTH,
    _D6671_MODE_RATIO,
    _D7905_CRACK_LENGTH,
)
from porosity_fe.mesh import _layers_merge_angles  # noqa: F401
from porosity_fe.pipeline import _default_mesh_resolution  # noqa: F401


# ======================================================================
# Pure helpers (extracted to porosity_fe.reporting so tests can import them
# without pulling in Streamlit — see #155 follow-up).
# ======================================================================

from porosity_fe.reporting import (  # noqa: E402, F401  re-exported for the Streamlit UI
    INDICATION_TYPES,
    STRUCTURAL_CLASSES,
    DispositionPolicy,
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
# The app lives in the porosity_app package (#224). Every name it defines
# is re-exported here, so ``import app; app.run_analysis(...)`` and the
# tests keep working; new code imports from ``porosity_app`` directly. A
# monkeypatch must target the module that looks the name up (e.g.
# ``porosity_app.runners.FESolver``), not ``app``.
# ======================================================================

from porosity_app.config import (  # noqa: E402, F401
    _CFG_DEFAULTS,
    _CFG_KEYS,
    _CFG_OPTIONAL_KEYS,
    _config_to_key,
    _CRITERION_LABELS,
    _DELAMINATION_SHAPES,
    _delta_T_suggestion,
    _FE_APPLIED_STRAIN,
    _FE_FLEXURE_LOAD_N,
    _FE_FRACTURE_LOAD_N,
    _FE_ILSS_LOAD_N,
    _FLEXURE_MODES,
    _FRACTURE_MODES,
    _FRACTURE_UQ_FIELDS,
    _FRACTURE_UQ_SAMPLES,
    _LOADING_LABELS,
    _MODE_LABELS,
    _NZ_SLIDER_MAX,
    _POROSITY_DEGRADATION_LABELS,
    _README_URL,
    _resolving_nz,
    _SUGGESTED_DELTA_T_177C_EPOXY,
    _THERMAL_OPTIONS,
    _TOUGHNESS_LAW_LABELS,
)
from porosity_app.runners import (  # noqa: E402, F401
    _clt_flexure_reference,
    _fe_load_kwargs,
    fe_vtu_bytes,
    fe_vtu_bytes_cached,
    _flexure_specimen_info,
    _fracture_specimen_info,
    _fracture_specimen_mesh,
    _fracture_uq_covs,
    run_analysis,
    run_analysis_cached,
    _run_fe_solve,
    run_uq_cached,
    _specimen_info,
    _toughness_fields,
    _toughness_knockdown_spec,
)
from porosity_app.summaries import (  # noqa: E402, F401
    _applied_load_text,
    _crack_front,
    delamination_text,
    _distribution_text,
    fe_decomposition_summary,
    fe_delamination_summary,
    fe_failure_summary,
    fe_flexure_summary,
    fe_fracture_test_summary,
    fe_fracture_uq_summary,
    fe_front_rows,
    fe_onset_summary,
    fe_short_beam_summary,
    fe_thermal_summary,
    _fmt_value,
    front_order,
    _interior_region_text,
    _load_factor_text,
    _mode_shares,
    ncr_delamination_available,
)
from porosity_app.convergence import (  # noqa: E402, F401
    _CONVERGENCE_LEVELS,
    convergence_plan,
    _CONVERGENCE_QUANTITIES,
    convergence_tables,
    convergence_unavailable_reason,
    _ILSS_NX_MULTIPLE,
    run_convergence_cached,
    run_convergence_check,
)
from porosity_app.plots import (  # noqa: E402, F401
    _colorbar_extend,
    _colour_limits,
    _CONSTANT_FIELD_RTOL,
    default_stress_component,
    _draw_delaminations,
    _exaggeration_factors,
    _finish_colorbar,
    _FRONT_COLOURS,
    _mark_uniform_value,
    _NEARLY_UNIFORM_SPREAD,
    plot_fracture_uq,
    plot_front,
    plot_mesh,
    plot_profile,
    plot_results,
    plot_stress,
    plot_uq,
    section_exaggeration,
    _SECTION_FIG_WIDTH_IN,
    _section_figure,
    _SECTION_MARGINS_IN,
    _SECTION_MAX_AXES_HEIGHT_IN,
    _SECTION_TARGET_ASPECT,
    _sequential_cmap,
    _set_titles,
    _STRESS_CMAP_DIVERGING,
    _STRESS_CMAP_MAGNITUDE,
    _STRESS_CMAP_NEGATIVE,
    _STRESS_CMAP_POSITIVE,
    _stress_colour_scale,
    _STRESS_COMPONENTS,
    _STRESS_SLUGS,
    _z_label,
)
from porosity_app.sidebar import (  # noqa: E402, F401
    _apply_nz_fix,
    _build_delamination_inputs,
    _build_fracture_specimen_inputs,
    _build_fracture_uq_inputs,
    _build_onset_inputs,
    _build_sidebar_inputs,
    _build_thermal_inputs,
    _build_toughness_inputs,
    _DEFAULT_LAYUP,
    _DISTRIBUTION_OPTIONS,
    _flexure_specimen_caption,
    _FRACTURE_UQ_LABELS,
    _ilss_specimen_caption,
    _offer_nz_fix,
    _set_delta_T,
    _sync_default_nz,
    _validate_layup_inline,
)
from porosity_app.tabs.common import (  # noqa: E402, F401
    _metric_row,
    _placeholder_tab,
    _show_figure,
)
from porosity_app.tabs.overview import (  # noqa: E402, F401
    _build_overview_tab,
)
from porosity_app.tabs.figures import (  # noqa: E402, F401
    _build_mesh_tab,
    _build_profile_tab,
)
from porosity_app.tabs.results import (  # noqa: E402, F401
    _build_convergence_expander,
    _build_defect_results,
    _build_fe_failure_results,
    _build_flexure_results,
    _build_fracture_uq_block,
    _build_results_tab,
    _build_short_beam_results,
    _build_thermal_results,
    _build_uq_expander,
)
from porosity_app.tabs.stress import (  # noqa: E402, F401
    _build_stress_tab,
    _sync_stress_component,
)
from porosity_app.tabs.export import (  # noqa: E402, F401
    _build_export_tab,
)
from porosity_app.render import (  # noqa: E402, F401
    _render,
)

# Re-apply the shared style after Streamlit/matplotlib finished their own
# rcParams nudges, so the Streamlit plots match the static PNGs (#53).
_configure_matplotlib_style()


try:
    from streamlit.runtime import exists as _st_runtime_exists
    _UNDER_STREAMLIT = _st_runtime_exists()
except Exception:
    _UNDER_STREAMLIT = False

if _UNDER_STREAMLIT:
    _render()
