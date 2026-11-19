"""Saved scenarios: the sidebar inputs as a JSON file, and their restore.

A scenario records the value of every sidebar input, a custom material
when one is in use, and the headline numbers of the run it was saved
from. Loading it puts the inputs back, so the run can be repeated, and
keeps those numbers to compare the next run with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from porosity_fe import MATERIALS, MaterialProperties
from porosity_fe.io import _dumps_json, _wrap_envelope
from porosity_fe.reporting import parse_layup

from .config import (
    _CRITERION_LABELS,
    _DELAMINATION_SHAPES,
    _DISTRIBUTION_OPTIONS,
    _FRACTURE_MODES,
    _FRACTURE_UQ_FIELDS,
    _LOADING_LABELS,
    _NZ_SLIDER_MAX,
    _POROSITY_DEGRADATION_LABELS,
    _THERMAL_OPTIONS,
    _TOUGHNESS_LAW_LABELS,
    _VOID_SHAPES,
)

#: ``format`` tag of a scenario file.
FORMAT_SCENARIO = "porosity-fe.app-scenario"

#: Prefix of the Material entry of a custom material.
CUSTOM_MATERIAL_PREFIX = "Custom: "

#: Session keys of the custom material in use (``{"name", "fields"}``) and
#: of the run a loaded scenario was saved from (``{"name", "summary"}``).
CUSTOM_MATERIAL_KEY = "custom_material"
REFERENCE_KEY = "scenario_reference"


@dataclass(frozen=True)
class _Input:
    """One sidebar input a scenario records: its kind and what it accepts."""

    kind: str                       # 'bool', 'text', 'choice', 'number', 'int'
    choices: tuple = ()
    lo: float = -math.inf
    hi: float = math.inf
    optional: bool = False          # may be empty (None)

    def problem(self, value: Any) -> str | None:
        """Why ``value`` cannot be restored, or ``None`` when it can."""
        if value is None:
            return None if self.optional else "is empty"
        if self.kind == 'bool':
            return None if isinstance(value, bool) else "is not true or false"
        if self.kind == 'text':
            return None if isinstance(value, str) else "is not text"
        if self.kind == 'choice':
            return None if value in self.choices else (
                f"is not one of {', '.join(map(str, self.choices))}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            return "is not a number"
        if self.kind == 'int' and float(value) != int(value):
            return "is not a whole number"
        if not self.lo <= value <= self.hi:
            return f"is outside {self.lo:g} to {self.hi:g}"
        return None

    def coerce(self, value: Any) -> Any:
        if value is None or self.kind in ('bool', 'text', 'choice'):
            return value
        return int(value) if self.kind == 'int' else float(value)


def _number(lo: float, hi: float, optional: bool = False) -> _Input:
    return _Input('number', lo=lo, hi=hi, optional=optional)


def _choice(options) -> _Input:
    return _Input('choice', choices=tuple(options))


#: Every sidebar input a scenario records, by its session key. The bounds
#: are the ones of the widget (a value outside them is refused on load,
#: with a note, instead of failing in the widget).
SCENARIO_INPUTS: dict[str, _Input] = {
    "expert_mode": _Input('bool'),
    "material_select": _Input('text'),
    "layup_input": _Input('text'),
    "t_ply_input": _number(0.05, 0.50),
    "vp_input": _number(0.1, 15.0),
    "distribution_select": _choice(_DISTRIBUTION_OPTIONS),
    "void_shape_select": _choice(_VOID_SHAPES),
    "loading_select": _choice(_LOADING_LABELS),
    "criterion_select": _choice(_CRITERION_LABELS),
    "porosity_degradation": _choice(_POROSITY_DEGRADATION_LABELS),
    "plate_length": _number(5.0, 2000.0),
    "plate_width": _number(2.0, 1000.0),
    "mesh_nx": _Input('int', lo=2, hi=200),
    "mesh_ny": _Input('int', lo=2, hi=100),
    "mesh_nz": _Input('int', lo=1, hi=_NZ_SLIDER_MAX),
    "thermal_choice": _choice(_THERMAL_OPTIONS),
    "thermal_delta_T": _number(-1000.0, 1000.0, optional=True),
    "onset_on": _Input('bool'),
    "onset_x_avg": _number(0.0, 50.0, optional=True),
    "delam_on": _Input('bool'),
    "delam_interface": _Input('int', lo=0, hi=200),
    "delam_shape": _choice(_DELAMINATION_SHAPES),
    "delam_x_start": _number(-math.inf, math.inf),
    "delam_x_end": _number(-math.inf, math.inf),
    "delam_xc": _number(-math.inf, math.inf),
    "delam_lx": _number(0.01, math.inf),
    "delam_yc": _number(-math.inf, math.inf),
    "delam_wy": _number(0.01, math.inf),
    "delam_split": _Input('bool'),
    "tough_G_Ic": _number(0.001, 20.0, optional=True),
    "tough_G_IIc": _number(0.001, 20.0, optional=True),
    "tough_G_IIIc": _number(0.001, 20.0, optional=True),
    "tough_bk_eta": _number(0.01, 20.0, optional=True),
    "tough_law": _choice(_TOUGHNESS_LAW_LABELS),
    "tough_b_I": _number(0.0, 1000.0, optional=True),
    "tough_b_II": _number(0.0, 1000.0, optional=True),
    **{f"frac_a0_{mode}": _number(1.0, 500.0) for mode in _FRACTURE_MODES},
    "frac_mmb_ratio": _number(0.05, 1.0),
    "frac_uq_on": _Input('bool'),
    **{f"frac_uq_{name}": _number(0.0, 1.0) for name in _FRACTURE_UQ_FIELDS},
}


def custom_material_label(name: str) -> str:
    """The Material entry of a custom material called ``name``."""
    return f"{CUSTOM_MATERIAL_PREFIX}{name}"


def custom_material_record(material: MaterialProperties, name: str) -> dict:
    """The session record of a custom material: its name and its fields."""
    name = " ".join(str(name).split()) or "material"
    return {"name": name, "fields": material.to_dict()}


def material_options(state) -> list[str]:
    """The Material entries: the presets, then the custom material."""
    options = list(MATERIALS)
    record = state.get(CUSTOM_MATERIAL_KEY)
    if record:
        options.append(custom_material_label(record["name"]))
    return options


def resolve_material(material_name: str, state) -> MaterialProperties:
    """The preset called ``material_name``, or the custom material when it
    is its entry."""
    record = state.get(CUSTOM_MATERIAL_KEY)
    if record and material_name == custom_material_label(record["name"]):
        return MaterialProperties.from_dict(record["fields"])
    return MATERIALS[material_name]


def scenario_summary(result: dict) -> dict:
    """The headline numbers of a run, kept in a scenario for comparison:
    the empirical knockdown of every mode and law, and the FE results."""
    empirical = {mode: {model: float(r["knockdown"]) for model, r in models.items()}
                 for mode, models in result["empirical"].items()}
    fe = None
    field = result.get("fe_field")
    if field is not None:
        def number(value):
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None
        fe = {
            "loading": result["fe_loading"],
            "knockdown": number(field.knockdown),
            "effective_modulus_MPa": number(field.effective_modulus),
            "max_failure_index": number(field.max_failure_index),
            "first_ply_failure_load_factor": number(
                field.first_ply_failure_load_factor),
            "fracture_load_factor": number(
                getattr(field, "fracture_load_factor", None)),
        }
    return {"Vp_percent": float(result["config"]["Vp"]),
            "empirical": empirical, "fe": fe}


def scenario_document(state, name: str = "", result: dict | None = None) -> dict:
    """The scenario of the current sidebar inputs.

    ``state`` is the session state; only the inputs it holds are recorded
    (a panel that is not shown has none). ``result`` adds the headline
    numbers of its run, when the inputs are still the ones it ran with.
    """
    inputs = {}
    for key, spec in SCENARIO_INPUTS.items():
        if key in state and spec.problem(state[key]) is None:
            inputs[key] = spec.coerce(state[key])
    material = None
    record = state.get(CUSTOM_MATERIAL_KEY)
    if record and inputs.get("material_select") == custom_material_label(record["name"]):
        material = {"name": record["name"], "fields": dict(record["fields"])}
    return _wrap_envelope(FORMAT_SCENARIO, None, {
        "name": " ".join(str(name).split()) or None,
        "inputs": inputs,
        "material": material,
        "summary": None if result is None else scenario_summary(result),
    })


def serialise_scenario(document: dict) -> str:
    """A scenario document as JSON text."""
    return _dumps_json(document)


def parse_scenario(text: str | bytes) -> dict:
    """The scenario document in ``text``; ``ValueError`` when it is not one."""
    try:
        document = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"The file is not valid JSON ({exc}).") from None
    if not isinstance(document, dict) or document.get("format") != FORMAT_SCENARIO:
        fmt = document.get("format") if isinstance(document, dict) else None
        raise ValueError(
            f"The file is not a saved scenario (format {fmt!r}; expected "
            f"{FORMAT_SCENARIO!r}). Use the scenario download of this app.")
    if not isinstance(document.get("inputs"), dict):
        raise ValueError("The scenario has no inputs.")
    return document


def apply_scenario(document: dict, state) -> list[str]:
    """Put the inputs of a scenario into the session state.

    Returns notes for the user: one per input that could not be restored
    (unknown, of the wrong type, out of range), each naming the input.
    The custom material and the saved run's numbers are restored first, so
    the Material entry of a custom material is valid.
    """
    notes: list[str] = []
    material = document.get("material")
    custom_label = None
    if material is not None:
        try:
            name = material["name"]
            restored = MaterialProperties.from_dict(material["fields"])
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"The custom material was not restored: {exc}")
        else:
            record = custom_material_record(restored, name)
            state[CUSTOM_MATERIAL_KEY] = record
            custom_label = custom_material_label(record["name"])
    summary = document.get("summary")
    if isinstance(summary, dict):
        state[REFERENCE_KEY] = {"name": document.get("name"), "summary": summary}
    else:
        state.pop(REFERENCE_KEY, None)

    for key, value in document["inputs"].items():
        spec = SCENARIO_INPUTS.get(key)
        if spec is None:
            notes.append(f"{key}: not an input of this version; skipped.")
            continue
        problem = spec.problem(value)
        if problem is None and key == "material_select" \
                and value not in MATERIALS and value != custom_label:
            problem = "is not a preset, and the scenario holds no such custom material"
        if problem is None and key == "layup_input":
            try:
                parse_layup(value)
            except ValueError as exc:
                problem = f"is not a layup ({exc})"
        if problem is not None:
            notes.append(f"{key}: {value!r} {problem}; not restored.")
            continue
        state[key] = spec.coerce(value)
    if "mesh_nz" in document["inputs"] and "mesh_nz" in state:
        # A restored nz is a value the user set: it must not follow the
        # layup's default (see sidebar._sync_default_nz).
        state["_mesh_nz_default"] = None
    return notes


def comparison_rows(reference: dict, current: dict) -> list[dict]:
    """Rows comparing the saved run of a scenario with the current one.

    One row per quantity both summaries hold: its value in each and the
    change in per cent of the saved value.
    """
    def row(label: str, saved, now) -> dict | None:
        if saved is None or now is None:
            return None
        change = None if saved == 0 else 100.0 * (now - saved) / abs(saved)
        return {"Quantity": label, "Saved run": saved, "This run": now,
                "Change (%)": change}

    rows = [row("Void content Vp (%)", reference.get("Vp_percent"),
                current.get("Vp_percent"))]
    saved_emp = reference.get("empirical") or {}
    for mode, models in (current.get("empirical") or {}).items():
        for model, value in models.items():
            rows.append(row(f"Knockdown, {mode} ({model})",
                            (saved_emp.get(mode) or {}).get(model), value))
    saved_fe, now_fe = reference.get("fe"), current.get("fe")
    if saved_fe and now_fe and saved_fe.get("loading") == now_fe.get("loading"):
        loading = now_fe["loading"]
        for key, label in (
                ("knockdown", "FE stiffness knockdown"),
                ("effective_modulus_MPa", "FE effective modulus (MPa)"),
                ("max_failure_index", "FE max failure index"),
                ("first_ply_failure_load_factor", "FE first-ply-failure load factor"),
                ("fracture_load_factor", "FE fracture load factor")):
            rows.append(row(f"{label}, {loading}", saved_fe.get(key), now_fe.get(key)))
    return [r for r in rows if r is not None]
