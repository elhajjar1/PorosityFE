"""Sidebar configuration: its keys, defaults and cache key, labels and load constants.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import math

from porosity_fe.fracture_specimens import (
    _D5528_CRACK_LENGTH,
    _D6671_CRACK_LENGTH,
    _D7905_CRACK_LENGTH,
)

# Repo README link for in-app guidance (e.g. when FE solve is skipped, #129).
_README_URL = (
    "https://github.com/elhajjar1/PorosityFE"
    "#solver-selection-fe-vs-empirical"
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


#: Defect and fracture fields of the config (delamination plan, phase 5),
#: in cache-key order: an explicit delamination (a :class:`Delamination`)
#: and whether to split the knockdown by defect, the interlaminar toughness
#: (N/mm) and its porosity knockdown law (``"exponential"`` with the
#: placeholders, ``"none"``, or ``"user"`` with ``toughness_b = [b_I,
#: b_II]``), the interface onset check and its averaging length (mm), and
#: the crack length (mm) and MMB mode ratio of a fracture specimen, and the
#: CoVs of the λ_c uncertainty band (``fracture_uq``, one per
#: :data:`_FRACTURE_UQ_FIELDS` entry; ``None`` for no band). Unlike
#: :data:`_CFG_KEYS`, each enters the cache key only when it differs from
#: its default, so a config without them keeps the key (and the cached
#: result) it had before they existed.
_CFG_OPTIONAL_KEYS = (
    "delamination", "knockdown_split", "G_Ic", "G_IIc", "G_IIIc", "bk_eta",
    "toughness_law", "toughness_b", "onset", "onset_x_avg",
    "crack_length", "mmb_mode_ratio", "fracture_uq",
)


#: Inputs the app's λ_c uncertainty band can scatter, in the order of
#: ``cfg["fracture_uq"]`` (coefficients of variation, fractions): the
#: toughness and B-K exponent, and the porosity-toughness coefficients.
_FRACTURE_UQ_FIELDS = ("G_Ic", "G_IIc", "bk_eta", "b_I", "b_II")


#: Draws of the band (Latin hypercube, seed 0): it re-evaluates λ_c on the
#: front of the one FE solve, so a few thousand cost milliseconds.
_FRACTURE_UQ_SAMPLES = 2000


#: Config fields that an older config dict may lack, and the value that
#: absence means (no thermal load, the Tsai-Wu criterion, no delamination,
#: no toughness, the default toughness law, no onset check, the standard
#: specimen dimensions).
_CFG_DEFAULTS = {"thermal_mode": "off", "delta_T": None,
                 "failure_criterion": "tsai_wu",
                 "delamination": None, "knockdown_split": False,
                 "G_Ic": None, "G_IIc": None, "G_IIIc": None, "bk_eta": None,
                 "toughness_law": "exponential", "toughness_b": None,
                 "onset": False, "onset_x_avg": None,
                 "crack_length": None, "mmb_mode_ratio": None,
                 "fracture_uq": None}


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
    for k in _CFG_OPTIONAL_KEYS:
        v = cfg.get(k, _CFG_DEFAULTS[k])
        if v != _CFG_DEFAULTS[k]:
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


#: Load of the fracture specimens (N): the hinge load on each arm of the
#: DCB (its sign is not used), the mid-span load of the ENF and the lever
#: load of the MMB (negative presses the lever down and opens the crack).
#: Energy release rates scale with its square, so the critical load is the
#: critical load factor times its magnitude.
_FE_FRACTURE_LOAD_N = -10.0


#: The FE-only flexure modes (ASTM D7264 procedures A and B on the
#: :meth:`CompositeMesh.for_flexure` specimen) and their procedure keys.
_FLEXURE_MODES = {"flexure_3pt": "3pt", "flexure_4pt": "4pt"}


#: The FE-only delamination fracture specimens, for test planning:
#: loading mode -> (standard, specimen, default crack length a0 in mm, what
#: the load P is). The crack lengths are the defaults of
#: :meth:`CompositeMesh.for_dcb`, ``for_enf`` and ``for_mmb``.
_FRACTURE_MODES = {
    "dcb": ("ASTM D5528", "double-cantilever-beam (DCB, mode I) specimen",
            _D5528_CRACK_LENGTH, "hinge load on each arm"),
    "enf": ("ASTM D7905", "end-notched-flexure (ENF, mode II) specimen",
            _D7905_CRACK_LENGTH, "mid-span load"),
    "mmb": ("ASTM D6671", "mixed-mode-bending (MMB) specimen",
            _D6671_CRACK_LENGTH, "lever load"),
}


#: Sidebar choices of the porosity knockdown of the interlaminar toughness:
#: ``cfg["toughness_law"]`` -> label.
_TOUGHNESS_LAW_LABELS = {
    "exponential": "Exponential, placeholders b_I = 6.2, b_II = 14 (uncalibrated)",
    "none": "None: the pristine toughness",
    "user": "Exponential with my coefficients b_I, b_II",
}


#: Sidebar planform choices of the delamination input.
_DELAMINATION_SHAPES = {
    "ellipse": "ellipse (NDI indication)",
    "rectangle": "rectangle",
    "strip": "through-width strip",
}


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
    "dcb": "dcb (FE only, D5528 mode I)",
    "enf": "enf (FE only, D7905 mode II)",
    "mmb": "mmb (FE only, D6671 mixed mode)",
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
