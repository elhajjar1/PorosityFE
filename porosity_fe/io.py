"""JSON / VTK I/O helpers and provenance."""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .results import ConfigArtifacts

from .results import ConfigResult

logger = logging.getLogger("porosity_fe_analysis")

def _json_default(o):
    """json.dump ``default=`` hook: make numpy scalars/arrays serializable.

    The science payload is already float()-wrapped, but user-supplied
    fields (e.g. ndarray ply_angles, dataclass configs, datetime stamps)
    would otherwise raise TypeError (#20).
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    # Plain dataclass instances (e.g. MaterialProperties) — accept on a
    # best-effort basis so callers can stash a dataclass field in the
    # config dict without an explicit asdict() at the call site.
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(
        f"Object of type {type(o).__name__} is not JSON serializable"
    )


# ----------------------------------------------------------------------
# Strict (standard) JSON
# ----------------------------------------------------------------------
# JSON (RFC 8259) has no NaN or Infinity. Python's json module writes them
# as the bare tokens NaN / Infinity by default, which JavaScript, jq and
# other strict parsers reject. Every writer in the package therefore
# serializes through _dumps_json (allow_nan=False). A field whose
# non-finite value means something (e.g. a Tsai-Wu per-mode index, or the
# knockdown of a thermal-only solve) is mapped to null by its writer with
# _finite_or_none, and the writer documents what null means; any other
# non-finite float is a bug and raises instead of reaching the file.

def _finite_or_none(value: float | None) -> float | None:
    """JSON-safe float: ``None`` (written as ``null``) for missing or non-finite values.

    Only for fields whose writer documents what ``null`` means; every
    other float goes through :func:`_dumps_json` as is and raises if it is
    not finite.
    """
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _non_finite_path(obj, path: str = "$") -> str | None:
    """Location (``$.key[index]``) of the first non-finite float in ``obj``, or ``None``.

    Walks what :func:`_json_default` would serialize, so the error from
    :func:`_dumps_json` can name the offending field.
    """
    if obj is None or isinstance(obj, (str, bool, int, np.integer, np.bool_)):
        return None
    if isinstance(obj, (float, np.floating)):
        return None if math.isfinite(float(obj)) else path
    if isinstance(obj, dict):
        for key, value in obj.items():
            found = _non_finite_path(value, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found = _non_finite_path(value, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "fc":
        bad = np.argwhere(~np.isfinite(obj))
        if bad.size:
            return path + "".join(f"[{int(i)}]" for i in bad[0])
        return None
    try:
        converted = _json_default(obj)
    except TypeError:
        return None
    return _non_finite_path(converted, path)


def _dumps_json(obj) -> str:
    """Serialize a writer payload as standard JSON (no ``NaN`` / ``Infinity``).

    The one serializer behind every JSON writer in the package
    (``indent=2``, numpy and dataclass values via :func:`_json_default`).
    It passes ``allow_nan=False``, so a non-finite float that a writer did
    not map to ``null`` raises instead of producing a file that strict
    parsers reject.

    Raises
    ------
    ValueError
        If ``obj`` holds a non-finite float; the message names its location.
    """
    try:
        return json.dumps(obj, indent=2, default=_json_default, allow_nan=False)
    except ValueError as exc:
        where = _non_finite_path(obj)
        if where is None:
            raise
        raise ValueError(
            f"Refusing to write non-standard JSON: non-finite value at {where}. "
            f"JSON has no NaN or Infinity; a writer maps a non-finite value "
            f"that has a meaning to null explicitly, and any other one is a "
            f"bug in the result it came from."
        ) from exc


def _write_json(obj, filename: str | os.PathLike) -> None:
    """Write ``obj`` to ``filename`` with :func:`_dumps_json`.

    The text is built before the file is opened, so a refused payload
    leaves no partial file behind.
    """
    text = _dumps_json(obj)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


# JSON output schema (#20). Bump the major when an incompatible change
# to the payload structure ships; bump the minor for additive changes.
# 1.1 (#131): added optional top-level ``units`` block documenting the
# physical units of numeric leaves. Purely additive; consumers on 1.0
# loaders can safely ignore the new key.
# 1.2 (IMPROVEMENT_PLAN 3.5): FE-field exports gain an optional ``thermal``
# block, interior and load-factor-basis keys in ``failure``, a nullable
# ``knockdown_factor`` (null for loading='thermal') and ``delta_T_K`` in
# the provenance of thermal solves. Additive. Every writer now emits
# standard JSON: the per-mode ``failure.mode_indices`` a criterion does
# not separate (Tsai-Wu) are null, where 1.1 wrote the invalid token NaN.
# 1.3 (delamination plan, phase 0): FE-field exports gain
# ``failure.mode_maxima`` (each mode's maximum over the model, null for
# Tsai-Wu) and the interlaminar components sigma_33 / tau_23 / tau_13 in the
# ``stress_local`` summary. Also 1.3 (phase 2.4-2.8), additive: an FE-field
# export of a mesh with an explicit delamination gains a ``fracture`` block
# (crack-face contact iterations and pairs, the critical load factor and its
# governing node, and the virtual-crack-closure front table: coordinates,
# G_I / G_II / G_III / G_T in N/mm, mode mixity, toughness and load factor
# per front node), null for every undefined or infinite value. Also 1.3
# (phase 4), additive: the ``fracture`` block gains ``toughness_knockdown``
# (a description of the porosity-toughness law that knocked G_c down at
# each front node's Vp) and ``toughness_uncalibrated`` (true when its
# placeholder coefficients lowered a toughness), null without toughness.
JSON_SCHEMA_VERSION = "1.3"
FORMAT_EMPIRICAL_SWEEP = "porosity-fe.empirical-sweep"
FORMAT_FE_FIELDS = "porosity-fe.fe-fields"
FORMAT_NCR = "porosity-fe.ncr"
FORMAT_UQ = "porosity-fe.uq"
_KNOWN_FORMATS = {FORMAT_EMPIRICAL_SWEEP, FORMAT_FE_FIELDS, FORMAT_NCR, FORMAT_UQ}


# Per-format units descriptors for the self-documenting ``units`` envelope
# block (#131). Keys correspond to the numeric leaves each writer emits,
# so downstream consumers reading only the JSON can interpret ambiguous
# fields like ``knockdown`` without consulting external docs.
_UNITS_EMPIRICAL_SWEEP = {
    "failure_stress": "MPa",
    "knockdown": "dimensionless fraction in (0, 1]",
    "void_volume_fraction": "dimensionless fraction in [0, 1]",
}
_UNITS_UQ = {
    "failure_stress": "MPa",
    "knockdown": "dimensionless fraction in (0, 1]",
    "void_volume_fraction": "dimensionless fraction in [0, 1]",
    "vp_cov": "dimensionless coefficient of variation",
    "coef_cov": "dimensionless coefficient of variation",
    "percentiles": "%",
}
_UNITS_NCR = {
    "failure_stress": "MPa",
    "knockdown": "dimensionless fraction in (0, 1]",
    "governing_residual_strength": "MPa",
    "governing_knockdown": "dimensionless fraction in (0, 1]",
    "measured_Vp_percent": "%",
    "t_ply_mm": "mm",
}
#: Units of the ``delamination`` block of a delamination or combined NCR,
#: added to :data:`_UNITS_NCR` only when the block is written.
_UNITS_NCR_DELAMINATION = {
    "fracture_load_factor": "dimensionless multiple of the analysis load",
    "required_load_factor": "dimensionless multiple of the analysis load",
    "margin_of_safety": "dimensionless (fracture_load_factor / required_load_factor - 1)",
    "psi": "dimensionless mode mixity (G_II + G_III) / G_T",
    "Vp": "dimensionless fraction in [0, 1]",
    "bk_eta": "dimensionless",
}
_UNITS_STREAMLIT_EMPIRICAL = {
    "failure_stress": "MPa",
    "knockdown": "dimensionless fraction in (0, 1]",
    "Vp_percent": "%",
    "t_ply": "mm",
}


@functools.lru_cache(maxsize=1)
def _git_commit_sha() -> str | None:
    """HEAD commit of the checkout containing this module, or ``None``.

    Cached for the life of the process, so each JSON write no longer spawns
    a ``git`` subprocess; the already-imported code cannot change mid-run.
    """
    try:
        # Run git from the directory containing this module so a CLI invoked
        # from somewhere else still resolves the repo SHA. Graceful fallback
        # to ``None`` for wheel/sdist installs or untracked checkouts.
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _wrap_envelope(fmt: str, units: dict | None = None,
                   payload: dict | None = None, *, seed: int | None = None) -> dict:
    """Standard JSON envelope: schema version, format, provenance, units.

    ``payload`` keys follow the envelope keys. ``units`` is omitted when
    ``None``. Every JSON writer builds its envelope here.
    """
    envelope = {
        'schema_version': JSON_SCHEMA_VERSION,
        'format': fmt,
        'provenance': _build_provenance(seed=seed),
    }
    if units is not None:
        # Self-documenting units block (#131): documents the physical units
        # of the numeric leaves so downstream consumers reading only the
        # JSON can interpret values without external docs.
        envelope['units'] = dict(units)
    if payload:
        envelope.update(payload)
    return envelope


#: Deprecated provenance alias -> canonical key. The aliases are still
#: written for back-compat; the schema no longer requires them and they go
#: away in schema 2.0.
_DEPRECATED_PROVENANCE_ALIASES: dict[str, str] = {
    "package_version": "porosity_fe_version",
    "python": "python_version",
    "numpy": "numpy_version",
    "scipy": "scipy_version",
    "generated_utc": "timestamp_utc",
    "git_sha": "git_commit",
}


def _build_provenance(seed: int | None = None) -> dict:
    """Return a provenance metadata dict for JSON output reproducibility.

    Captures software versions, platform, timestamp, optional git commit,
    and the run ``seed`` so that any JSON output can be traced back to the
    exact environment used (#55).

    The canonical keys are ``porosity_fe_version``, ``python_version``,
    ``numpy_version``, ``scipy_version``, ``timestamp_utc`` and
    ``git_commit``. The short aliases from the #55 reproducibility contract
    (``package_version``, ``python``, ``numpy``, ``scipy``,
    ``generated_utc``, ``git_sha``; see :data:`_DEPRECATED_PROVENANCE_ALIASES`)
    are still written but deprecated: the schema no longer requires them and
    they will be dropped in schema 2.0 (IMPROVEMENT_PLAN 4.7).

    The optional ``hostname`` field is opt-in via the
    ``POROSITY_FE_INCLUDE_HOSTNAME`` env var (set to ``1``/``true``/``yes``)
    so the default JSON output does not leak workstation names.
    """
    try:
        import importlib.metadata as _ilm
        pfe_version: str | None = _ilm.version("porosity-fe")
    except Exception:
        # Source checkout not pip-installed: fall back to the package
        # ``__version__`` attribute (defined in ``porosity_fe/__init__.py``).
        from . import __version__ as _pkg_version_attr
        pfe_version = _pkg_version_attr

    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"

    def _pkg_version(module_name: str) -> str | None:
        mod = sys.modules.get(module_name)
        return getattr(mod, "__version__", None) if mod else None

    git_commit = _git_commit_sha()

    numpy_v = _pkg_version("numpy")
    scipy_v = _pkg_version("scipy")
    # Naive-UTC ISO string plus "Z", the format the schema and consumers
    # already expect (not the "+00:00" suffix an aware isoformat() adds).
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    generated_utc = now_utc.isoformat() + "Z"

    prov = {
        # Envelope schema version, repeated inside the provenance block so a
        # consumer holding just the provenance dict can still tell what
        # contract it was emitted under (#55).
        "schema_version": JSON_SCHEMA_VERSION,
        # Existing keys (kept for back-compat with the published JSON schema
        # and downstream consumers).
        "porosity_fe_version": pfe_version,
        "python_version": python_version,
        "platform": platform.platform(),
        "numpy_version": numpy_v,
        "scipy_version": scipy_v,
        "matplotlib_version": _pkg_version("matplotlib"),
        "timestamp_utc": generated_utc,
        "seed": seed,
        "git_commit": git_commit,
    }
    # Deprecated #55 aliases, written until schema 2.0.
    for alias, canonical in _DEPRECATED_PROVENANCE_ALIASES.items():
        prov[alias] = prov[canonical]

    # Hostname is opt-in to avoid leaking workstation names in shared
    # artifacts. Default off (#55).
    if os.environ.get("POROSITY_FE_INCLUDE_HOSTNAME", "").lower() in (
            "1", "true", "yes", "on"):
        try:
            prov["hostname"] = platform.node() or None
        except Exception:
            prov["hostname"] = None

    return prov


def save_results_to_json(results: dict, filename: str | os.PathLike,
                         artifacts: dict[str, ConfigArtifacts] | None = None):
    """Export numerical results to JSON.

    Parameters
    ----------
    results : dict
        Either the new ``Dict[str, ConfigResult]`` returned by
        :func:`compare_configurations`, or the legacy worker-dict shape
        (``Dict[str, dict]``). Both keep the same on-disk JSON shape so
        the published JSON schema is unchanged (#44 item 3).
    filename : str or os.PathLike
        Output JSON path. ``pathlib.Path`` objects are accepted.
    artifacts : dict, optional
        Parallel ``Dict[str, ConfigArtifacts]`` from
        ``compare_configurations(..., return_artifacts=True)``. Used
        only to recover the seed for the provenance block when
        ``results`` is the lightweight ``ConfigResult`` shape (no live
        ``porosity_field`` carried). Optional; the JSON is still written
        if absent, just with ``seed=None`` in provenance.

    Raises
    ------
    ValueError
        If a configuration name collides with an envelope key, or a value
        is a non-finite float. The file is standard JSON (no ``NaN`` /
        ``Infinity``) and no value in a sweep is expected to be undefined,
        so one that is raises (naming its location) before the file is
        written.
    """
    # All configs in a sweep share one seed; record it iff unambiguous.
    # Source the seed from (in priority order):
    #   1. ConfigResult.seed (the new lightweight path),
    #   2. the parallel artifacts dict's porosity_field.seed, or
    #   3. the legacy worker-dict shape (back-compat).
    # (#44 item 3 / #55).
    seeds: set = set()
    for entry in results.values():
        if isinstance(entry, ConfigResult):
            if entry.seed is not None or artifacts is None:
                seeds.add(entry.seed)
            else:
                art = artifacts.get(entry.config_name) if artifacts else None
                pf = getattr(art, 'porosity_field', None) if art is not None else None
                seeds.add(getattr(pf, 'seed', None))
        elif isinstance(entry, dict):
            pf = entry.get('porosity_field')
            if pf is not None:
                seeds.add(getattr(pf, 'seed', None))
    seed = seeds.pop() if len(seeds) == 1 else None

    filename = Path(filename)

    output = _wrap_envelope(FORMAT_EMPIRICAL_SWEEP, _UNITS_EMPIRICAL_SWEEP, seed=seed)
    for name, data in results.items():
        if name in ('schema_version', 'format', 'units'):
            # Defensive: a user-named config that collides with envelope
            # keys would silently overwrite them. Skip with a clear error.
            # (#131 added 'units' to the reserved set.)
            raise ValueError(
                f"Configuration name {name!r} collides with the JSON "
                f"envelope keys ('schema_version', 'format', 'units')."
            )
        # Resolve the void_volume_fraction and config dict for both the
        # legacy dict shape and the new ConfigResult shape. The legacy
        # path reads ``data['porosity_field'].Vp`` and ``data['config']``;
        # the new path reads ``data.Vp`` / ``data.config``.
        if isinstance(data, ConfigResult):
            vp_value = float(data.Vp)
            cfg_dict = data.config
            emp_table = data.empirical
        else:
            vp_value = float(data['porosity_field'].Vp)
            cfg_dict = data['config']
            emp_table = data['empirical']

        entry = {
            'config': cfg_dict,
            'void_volume_fraction': vp_value,
            'empirical': {},
        }
        for mode in emp_table:
            entry['empirical'][mode] = {}
            for model in emp_table[mode]:
                r = emp_table[mode][model]
                # ``r`` is now a FailureResult (with dict-style back-compat
                # shim) for the empirical path, but legacy callers may
                # still hand in raw dicts.
                entry['empirical'][mode][model] = {
                    'failure_stress_MPa': r['failure_stress'],
                    'knockdown': r['knockdown'],
                }
        output[name] = entry

    # Standard JSON only: every value here is finite for a valid sweep, so
    # a non-finite one raises (naming its location) rather than writing NaN.
    _write_json(output, filename)
    logger.info("Saved: %s", filename)


def load_results_from_json(filename: str | os.PathLike) -> dict:
    """Round-trip loader for save_results_to_json / export_results outputs.

    Validates schema_version compatibility and format identifier. Raises
    ValueError on missing or incompatible envelope so callers don't silently
    consume the wrong shape.

    Accepts either a ``str`` path or any ``os.PathLike`` (e.g. ``pathlib.Path``).

    Values that are not defined are ``null`` in the file and ``None`` in
    the returned dict, e.g. the Tsai-Wu per-mode failure indices and the
    knockdown of a thermal-only FE solve (see
    :meth:`FESolver.export_results
    <porosity_fe.fe.solver.FESolver.export_results>` for what each
    ``null`` means). Files written by earlier builds may hold the
    non-standard tokens ``NaN`` / ``Infinity`` instead; they are read as
    ``None`` too, with a logged warning, so old and new files load the
    same way.
    """
    filename = Path(filename)
    legacy_tokens: set[str] = set()

    def _legacy_constant(token: str) -> None:
        legacy_tokens.add(token)
        return None

    with open(filename, encoding='utf-8') as f:
        data = json.load(f, parse_constant=_legacy_constant)
    if legacy_tokens:
        logger.warning(
            "%s: holds the non-standard JSON token(s) %s, written by an "
            "earlier porosity-fe build; read as null (None), which is what "
            "current builds write.", filename, ", ".join(sorted(legacy_tokens)))
    if not isinstance(data, dict):
        raise ValueError(f"{filename}: expected a JSON object at the top level.")
    sv = data.get('schema_version')
    if sv is None:
        raise ValueError(
            f"{filename}: missing 'schema_version'. "
            f"This file was likely written by a pre-1.0 build of porosity-fe."
        )
    major = sv.split('.', 1)[0]
    expected_major = JSON_SCHEMA_VERSION.split('.', 1)[0]
    if major != expected_major:
        raise ValueError(
            f"{filename}: schema_version {sv} is incompatible with this "
            f"loader (expects {expected_major}.x)."
        )
    fmt = data.get('format')
    if fmt not in _KNOWN_FORMATS:
        raise ValueError(
            f"{filename}: unknown format {fmt!r}. "
            f"Known formats: {sorted(_KNOWN_FORMATS)}."
        )
    return data
