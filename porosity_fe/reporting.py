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
  indication is porosity (a recommended disposition path from the void
  content and the governing knockdown), a delamination (a fracture
  assessment from an FE solve with toughness data, with no recommended
  path) or both (:data:`INDICATION_TYPES`).

The module deliberately keeps its top-level imports cheap. ``matplotlib`` is
only needed by :func:`serialise_ncr_pdf`, so it is imported lazily inside
that function.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import io
import math
import numbers
import re
import textwrap

#: One bracketed group and what follows it: ``[0/45/-45/90]_3s``.
_LAYUP_GROUP = re.compile(r'^\[(?P<body>[^\[\]]*)\](?P<suffix>.*)$')
#: A ply token: a decimal angle in degrees (``+``, ``-``, ``±`` or ``∓``
#: prefix) with an optional ply-count subscript, ``0_2`` = two 0° plies.
_PLY_TOKEN = re.compile(
    r'^(?P<sign>[+\-±∓]?)(?P<angle>\d+(?:\.\d*)?|\.\d+)(?:_(?P<count>\d+))?$')
#: After the group: an optional ``_``, a repeat count, a symmetry ``s``.
_GROUP_SUFFIX = re.compile(r'^_?(?P<repeat>[^sS]*?)(?P<sym>[sS])?$')


def _parse_group_suffix(suffix: str, text: str) -> tuple[int, bool]:
    """``(repeat, symmetric)`` of the text after the closing bracket."""
    if not suffix:
        return 1, False
    m = _GROUP_SUFFIX.match(suffix)
    repeat_token = suffix.lstrip('_') if m is None else m['repeat']
    if m is None or (not repeat_token and not m['sym']) or (
            repeat_token and not re.fullmatch(r'-?\d+', repeat_token)):
        raise ValueError(
            f"Invalid repeat count {repeat_token or suffix!r} in layup {text!r}; "
            f"expected an integer and/or 's' after the brackets (e.g. "
            f"'[0/90]_3s', '[0/90]3s' or '[0/90]s').")
    repeat = int(repeat_token) if repeat_token else 1
    if repeat < 1:
        raise ValueError(
            f"Repeat count must be >= 1, got {repeat} in layup {text!r}.")
    return repeat, m['sym'] is not None


def parse_layup(text: str) -> list:
    """Parse a layup string like '[0/45/-45/90]_3s' to a flat angle list.

    Standard laminate notation: ply angles in degrees, separated by ``/``
    (or ``,``), in one bracketed group.

    - A subscript on an angle is a ply count: ``[0_2/90_2]s`` is
      ``[0, 0, 90, 90, 90, 90, 0, 0]``.
    - ``±45`` is ``+45, -45`` (``∓45`` the reverse); a count repeats the
      pair.
    - After the brackets, ``_N`` or ``N`` repeats the group ``N`` times and
      a trailing ``s`` mirrors the result: ``[0/90]_2s``, ``[0/90]2s``,
      ``[0/90]s``.
    - Without brackets the text is a plain ply list (``0/0/90``, ply counts
      allowed, no group suffix).

    Raises ValueError on malformed input rather than silently substituting
    a default (issue #9): an angle that is not a plain decimal number of
    degrees (``nan``, ``1e2`` and digit-group underscores are refused),
    nested groups, or a suffix that is not a repeat count and/or ``s``.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Layup string is empty.")

    if "[" in text or "]" in text:
        m = _LAYUP_GROUP.match(text)
        if m is None:
            raise ValueError(
                f"Malformed layup {text!r}: expected one bracketed group "
                f"such as '[0/45/-45/90]_3s' (nested groups are not "
                f"supported; write the plies out).")
        body, suffix = m["body"], m["suffix"].strip()
    else:
        body, suffix = text, ""
    repeat, symmetric = _parse_group_suffix(suffix, text)

    sep = "/" if "/" in body else ","
    tokens = [a.strip() for a in body.split(sep) if a.strip()]
    if not tokens:
        raise ValueError(f"No ply angles found in layup {text!r}.")
    angles: list[float] = []
    for token in tokens:
        ply = _PLY_TOKEN.match(token.replace("−", "-"))
        if ply is None:
            raise ValueError(
                f"Invalid ply angle {token!r} in layup {text!r}; expected "
                f"degrees, optionally with a ply count (e.g. '45', '-45', "
                f"'±45' or '0_2' for two 0° plies), as in "
                f"'[0_2/±45/90]_3s'.")
        count = int(ply["count"]) if ply["count"] else 1
        if count < 1:
            raise ValueError(
                f"Ply count must be >= 1, got {token!r} in layup {text!r}.")
        theta = float(ply["angle"])
        sign = ply["sign"]
        if sign in ("±", "∓"):
            pair = [theta, -theta] if sign == "±" else [-theta, theta]
            angles += pair * count
        else:
            angles += [-theta if sign == "-" else theta] * count

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
# average. A delamination indication gets no recommended path at all: the
# tool assesses its fracture margin (the FE growth-onset load factor against
# the required load factor the user enters) and leaves the disposition to
# the MRB; no published acceptance numbers exist for it.

STRUCTURAL_CLASSES = ("primary", "secondary", "non-structural")

#: NDI indication types of an NCR: porosity alone, a delamination (a planar
#: interlaminar crack), or a delamination in a porous laminate.
INDICATION_TYPES = ("porosity", "delamination", "combined")


_POLICY_PATHS = (
    "Use-As-Is (UAI) — pending MRB concurrence",
    "Use-As-Is with Engineering Evaluation",
    "Engineering Evaluation / Repair",
    "Repair or Scrap",
)

_POLICY_SOURCE = (
    "Illustrative default bands, not taken from a cited standard or "
    "specification: the released engineering drawing / process "
    "specification porosity allowable governs."
)


@dataclasses.dataclass(frozen=True)
class DispositionPolicy:
    """The bands :func:`recommend_disposition` sorts a porosity NCR into.

    A case falls in band ``i`` (``i = 0, 1, 2``) when its void content is at
    most ``vp_percent_bounds[i]`` **and** its governing knockdown is at least
    ``knockdown_bounds[i]``, taking the first band that holds; otherwise in
    the last band. Each band recommends the path of the same index in
    ``paths``.

    The defaults (1 / 2 / 5 % voids with 0.95 / 0.90 / 0.80 of pristine
    strength retained) are **illustrative**: they are not taken from a
    cited standard or specification (issue #229). The released engineering
    drawing or process specification is the governing acceptance limit; an
    organisation that has its own bands passes them here.

    Parameters
    ----------
    vp_percent_bounds : tuple of 3 floats
        Upper void-content bound of each of the first three bands, in
        **percent**, strictly increasing within ``(0, 100]``.
    knockdown_bounds : tuple of 3 floats
        Lower governing-knockdown bound of each of the first three bands, a
        fraction, strictly decreasing within ``(0, 1]``.
    paths : tuple of 4 str
        The recommended path of each band, the last for a case outside the
        first three.
    source : str
        Where the bands come from, written to the record; the default says
        they are illustrative.
    """

    vp_percent_bounds: tuple[float, float, float] = (1.0, 2.0, 5.0)
    knockdown_bounds: tuple[float, float, float] = (0.95, 0.90, 0.80)
    paths: tuple[str, str, str, str] = _POLICY_PATHS
    source: str = _POLICY_SOURCE

    def __post_init__(self) -> None:
        vp = tuple(float(v) for v in self.vp_percent_bounds)
        kd = tuple(float(k) for k in self.knockdown_bounds)
        if len(vp) != 3 or not all(math.isfinite(v) for v in vp) or \
                not (0.0 < vp[0] < vp[1] < vp[2] <= 100.0):
            raise ValueError(
                "vp_percent_bounds must be three void contents in percent, "
                f"strictly increasing within (0, 100]; got {self.vp_percent_bounds!r}.")
        if len(kd) != 3 or not all(math.isfinite(k) for k in kd) or \
                not (1.0 >= kd[0] > kd[1] > kd[2] > 0.0):
            raise ValueError(
                "knockdown_bounds must be three knockdown fractions, strictly "
                f"decreasing within (0, 1]; got {self.knockdown_bounds!r}.")
        paths = tuple(self.paths)
        if len(paths) != 4 or not all(isinstance(p, str) and p.strip() for p in paths):
            raise ValueError(f"paths must be four non-empty strings; got {self.paths!r}.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must say where the bands come from.")
        object.__setattr__(self, 'vp_percent_bounds', vp)
        object.__setattr__(self, 'knockdown_bounds', kd)
        object.__setattr__(self, 'paths', paths)

    @property
    def is_default(self) -> bool:
        """Whether these are the illustrative default bands."""
        return self == DEFAULT_DISPOSITION_POLICY

    def band(self, Vp_percent: float, governing_knockdown: float) -> int:
        """Index (0 to 3) of the band a case falls in."""
        for i, (vp, kd) in enumerate(zip(self.vp_percent_bounds, self.knockdown_bounds,
                                         strict=True)):
            if Vp_percent <= vp and governing_knockdown >= kd:
                return i
        return 3

    def to_dict(self) -> dict:
        """JSON-ready record of the bands (the ``policy`` block of an NCR)."""
        return {
            "vp_percent_bounds": list(self.vp_percent_bounds),
            "knockdown_bounds": list(self.knockdown_bounds),
            "paths": list(self.paths),
            "source": self.source,
            "default": self.is_default,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DispositionPolicy:
        """The policy of a :meth:`to_dict` record (``default`` is ignored)."""
        return cls(vp_percent_bounds=tuple(data["vp_percent_bounds"]),
                   knockdown_bounds=tuple(data["knockdown_bounds"]),
                   paths=tuple(data.get("paths", _POLICY_PATHS)),
                   source=data.get("source", _POLICY_SOURCE))

    def describe(self) -> str:
        """One line: the bands and where they come from."""
        bands = "; ".join(
            f"<= {vp:g} % voids and >= {kd:g} retained: {path}"
            for vp, kd, path in zip(self.vp_percent_bounds, self.knockdown_bounds,
                                    self.paths, strict=False))
        return f"{bands}; otherwise: {self.paths[3]}. {self.source}"


#: The illustrative default bands of :func:`recommend_disposition`.
DEFAULT_DISPOSITION_POLICY = DispositionPolicy()


def _as_policy(policy) -> DispositionPolicy:
    if policy is None:
        return DEFAULT_DISPOSITION_POLICY
    if isinstance(policy, DispositionPolicy):
        return policy
    if isinstance(policy, dict):
        return DispositionPolicy.from_dict(policy)
    raise TypeError(
        f"policy must be a DispositionPolicy, its to_dict() record or None; "
        f"got {type(policy).__name__}.")


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


def recommend_disposition(
    Vp_percent: float, governing_knockdown: float, structural_class: str = "primary",
    *, policy: DispositionPolicy | dict | None = None,
) -> dict:
    """Recommend (not decide) an MRB disposition path for a porosity NCR.

    The recommendation bins on the measured void content and the governing
    (worst-case) knockdown with the bands of ``policy``, then escalates
    required substantiation by structural class. The default bands are
    **illustrative**, not from a cited standard (:class:`DispositionPolicy`):
    the released engineering drawing / process spec is the governing
    acceptance authority, and the MRB must substantiate against it.

    Parameters
    ----------
    Vp_percent : float
        Measured void content **in percent**, as reported on an NCR:
        ``3.0`` means 3 % voids. This is the one porosity input in the
        package that takes a percent rather than a fraction; passing the
        fraction ``0.03`` would be read as 0.03 % and recommend Use-As-Is.
    governing_knockdown : float
        Worst-case strength knockdown, a fraction in ``(0, 1]``.
    structural_class : str
        One of :data:`STRUCTURAL_CLASSES`.
    policy : DispositionPolicy, dict or None
        The bands; ``None`` (default) for :data:`DEFAULT_DISPOSITION_POLICY`.
        A dict is a :meth:`DispositionPolicy.to_dict` record.

    Returns
    -------
    dict
        ``path``, ``structural_class``, ``rationale``, ``cited_criteria``,
        ``required_mrb_actions``, ``disclaimer`` and ``policy`` (its
        :meth:`~DispositionPolicy.to_dict` record).

    Raises
    ------
    ValueError
        If ``Vp_percent`` is not a finite value in ``[0, 100]``, or
        ``structural_class`` is not one of :data:`STRUCTURAL_CLASSES`.
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

    policy = _as_policy(policy)
    band = policy.band(Vp, governing_knockdown)
    path = policy.paths[band]
    kd_pct = governing_knockdown * 100
    if band < 3:
        bound = (f"band {band + 1} of the disposition policy: at most "
                 f"{policy.vp_percent_bounds[band]:g}% voids with at least "
                 f"{policy.knockdown_bounds[band] * 100:g}% of pristine strength retained")
    if band == 0:
        rationale = (
            f"Measured void content ({Vp:.2f}%) and the predicted governing "
            f"residual strength ({kd_pct:.1f}% of pristine) fall in {bound}. "
            f"Strength loss is minor and likely covered by as-designed margin."
        )
    elif band == 1:
        rationale = (
            f"Void content ({Vp:.2f}%) and the governing knockdown "
            f"({kd_pct:.1f}% retained) fall in {bound}: moderate strength "
            f"loss. UAI may be substantiated by a positive margin-of-safety "
            f"check at the knockdown-adjusted allowable."
        )
    elif band == 2:
        rationale = (
            f"Void content ({Vp:.2f}%) and the governing knockdown, which "
            f"leaves only {kd_pct:.1f}% of pristine strength, fall in {bound}. "
            f"Disposition hinges on the as-designed margin and on whether an "
            f"approved repair can restore a serviceable condition."
        )
    else:
        rationale = (
            f"Void content ({Vp:.2f}%) and the predicted governing knockdown "
            f"({kd_pct:.1f}% retained) fall outside every band of the "
            f"disposition policy: a severe degradation. Use-As-Is is not "
            f"recommended without exceptional, test-backed substantiation; "
            f"repair or scrap is the expected path."
        )

    cited_criteria = [
        "Released engineering drawing / part specification porosity allowable "
        "(governing acceptance limit — verify against the controlled drawing).",
        "Process specification cure / void-content requirements for the "
        "applicable material system.",
        "Structural substantiation: laminate margin of safety recomputed at "
        "the knockdown-adjusted allowable for the governing loading mode.",
        "NDI acceptance criteria (ultrasonic C-scan attenuation / void "
        "content) per the applicable NDT specification.",
        "Quality-system MRB procedure for disposition of nonconforming "
        "material.",
    ]

    required_mrb_actions = [
        "Confirm measured void content by micrograph / acid digestion or a "
        "validated C-scan correlation.",
        "Verify the governing porosity allowable on the released engineering "
        "drawing / specification.",
        "Perform structural substantiation: recompute the margin of safety "
        "using the knockdown-adjusted allowable for the governing mode.",
        "Define the NDI extent and map the affected zone / part region.",
    ]
    if structural_class == "primary":
        required_mrb_actions.append(
            "Primary structure: obtain customer / DER engineering concurrence "
            "before approving any Use-As-Is disposition."
        )
    elif structural_class == "non-structural":
        required_mrb_actions.append(
            "Non-structural item: confirm there is no fluid-ingress, "
            "fatigue, or interface-sealing function affected by the porosity."
        )
    if band >= 2:
        required_mrb_actions.append(
            "If repair is selected, document the approved repair scheme and "
            "the post-repair re-inspection requirements."
        )

    disclaimer = (
        "This recommended disposition path was produced by an automated MRB "
        "support tool from a predictive porosity-knockdown analysis. It is "
        "NOT a final disposition. A qualified Material Review Board must "
        "independently review, may modify, and must formally approve the "
        "disposition. Final acceptance requires substantiation against the "
        "governing engineering drawing / specification and the applicable "
        "structural margins."
    )
    if policy.is_default:
        disclaimer += (
            " The void-content and knockdown bands behind the recommended path "
            "(at most 1 / 2 / 5 % voids with at least 95 / 90 / 80 % of pristine "
            "strength retained) are illustrative defaults, not taken from a "
            "cited standard or specification."
        )

    return {
        "path": path,
        "structural_class": structural_class,
        "rationale": rationale,
        "cited_criteria": cited_criteria,
        "required_mrb_actions": required_mrb_actions,
        "disclaimer": disclaimer,
        "policy": policy.to_dict(),
    }


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
    "Quality-system MRB procedure for disposition of nonconforming "
    "material.",
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
    "Disposition the delamination: this tool recommends no path for it.",
)

_UNCALIBRATED_ACTION = (
    "The porosity knockdown of the toughness used uncalibrated placeholder "
    "coefficients: substantiate the toughness of the porous interface by "
    "test, or rerun the analysis with calibrated coefficients."
)

_DELAMINATION_DISCLAIMER = (
    "PorosityFE does not recommend a disposition for a delamination "
    "indication. This fracture assessment was produced by an automated MRB "
    "support tool from a predictive delamination fracture analysis, as "
    "engineering data for a qualified Material Review Board, which "
    "dispositions the indication against the governing engineering drawing "
    "/ specification and the applicable damage-tolerance substantiation."
)


def _check_load_factor(name: str, value, *, allow_inf: bool) -> float:
    """``value`` as a float load factor (``> 0``; ``inf`` when allowed)."""
    if value is None:
        raise ValueError(f"{name} is required for a delamination assessment.")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    v = float(value)
    if math.isnan(v) or v < 0.0 or (v == 0.0 and not allow_inf) or (
            math.isinf(v) and not allow_inf):
        bound = "a number >= 0 (inf when nothing drives the front)" if allow_inf \
            else "a finite number > 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}.")
    return v


def assess_delamination(
    fracture_load_factor: float, required_load_factor: float,
    structural_class: str = "primary", *, toughness_uncalibrated: bool = False,
) -> dict:
    """Fracture assessment of a delamination indication for the MRB.

    Reports the **fracture margin** of the delamination: the predicted
    growth-onset load factor ``lambda_c`` of an FE solve with the
    delamination (:attr:`FieldResults.fracture_load_factor
    <porosity_fe.FieldResults.fracture_load_factor>`) against the
    ``required_load_factor`` the user supplies, with the criteria and the
    MRB actions that apply. It recommends **no disposition path**: no
    published acceptance numbers exist for delaminations (they are OEM- and
    part-specific), so the MRB dispositions the indication. The void
    content plays no part; porosity enters only through the toughness at
    the front.

    Parameters
    ----------
    fracture_load_factor : float
        The predicted growth-onset load factor ``lambda_c`` (dimensionless,
        ``>= 0``; ``inf`` when the analysis load releases no energy at the
        front).
    required_load_factor : float
        The load factor the part must reach without delamination growth, a
        finite number ``> 0`` that multiplies the **same analysis load** as
        ``fracture_load_factor`` (for an analysis at 1 % strain and a
        design strain of 0.4 %, 0.4). It comes from the structural
        substantiation; there is no default.
    structural_class : str
        One of :data:`STRUCTURAL_CLASSES`.
    toughness_uncalibrated : bool
        Keyword-only. Whether the toughness behind ``fracture_load_factor``
        was knocked down by porosity with uncalibrated placeholder
        coefficients (:attr:`CrackFront.toughness_uncalibrated
        <porosity_fe.CrackFront.toughness_uncalibrated>`); it adds a
        required MRB action and a sentence to the statement.

    Returns
    -------
    dict
        ``structural_class``, ``fracture_margin`` (``fracture_load_factor``
        and ``margin_of_safety = lambda_c / required - 1``, both ``None``
        when ``lambda_c`` is infinite, ``required_load_factor``,
        ``toughness_uncalibrated``), ``statement``, ``cited_criteria``,
        ``required_mrb_actions`` and ``disclaimer``. There is no ``path``.

    Raises
    ------
    ValueError
        If a load factor is missing or invalid, or ``structural_class`` is
        not one of :data:`STRUCTURAL_CLASSES`.
    """
    if structural_class not in STRUCTURAL_CLASSES:
        raise ValueError(
            f"Unknown structural_class {structural_class!r}; "
            f"expected one of {list(STRUCTURAL_CLASSES)}."
        )
    lam = _check_load_factor("fracture_load_factor", fracture_load_factor,
                             allow_inf=True)
    req = _check_load_factor("required_load_factor", required_load_factor,
                             allow_inf=False)
    finite = math.isfinite(lam)
    margin = {
        "fracture_load_factor": lam if finite else None,
        "required_load_factor": req,
        "margin_of_safety": lam / req - 1.0 if finite else None,
        "toughness_uncalibrated": bool(toughness_uncalibrated),
    }
    if finite:
        statement = (
            f"The predicted delamination growth-onset load factor is "
            f"lambda_c = {lam:.3g}, against a required load factor of "
            f"{req:.3g} at the same analysis load: fracture margin of safety "
            f"{margin['margin_of_safety']:+.2f}.")
    else:
        statement = (
            f"The analysis load releases no energy at the delamination front, "
            f"so the predicted growth-onset load factor is unbounded (required "
            f"load factor {req:.3g}).")
    statement += (
        " The analysis is linear elastic fracture mechanics: growth onset "
        "under static load, with no fatigue growth or delamination buckling.")
    if toughness_uncalibrated:
        statement += (
            " The toughness at the front was knocked down by the porosity "
            "there with uncalibrated placeholder coefficients.")

    required_mrb_actions = list(_DELAMINATION_ACTIONS)
    if toughness_uncalibrated:
        required_mrb_actions.append(_UNCALIBRATED_ACTION)
    if structural_class == "primary":
        required_mrb_actions.append(
            "Primary structure: obtain customer / DER engineering concurrence "
            "for the disposition of the delamination."
        )
    elif structural_class == "non-structural":
        required_mrb_actions.append(
            "Non-structural item: confirm there is no fluid-ingress, "
            "fatigue, or interface-sealing function affected by the "
            "delamination."
        )
    return {
        "structural_class": structural_class,
        "fracture_margin": margin,
        "statement": statement,
        "cited_criteria": list(_DELAMINATION_CRITERIA),
        "required_mrb_actions": required_mrb_actions,
        "disclaimer": _DELAMINATION_DISCLAIMER,
    }


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


def _test_specimen_name(mesh) -> str | None:
    """The test specimen an FE mesh models (short-beam or flexure), or ``None``."""
    if getattr(mesh, "ilss_geometry", None) is not None:
        return "ASTM D2344 short-beam specimen"
    if getattr(mesh, "flexure_geometry", None) is not None:
        return "flexure specimen (ASTM D7264 / D790)"
    return None


def _analysis_mesh_label(result: dict, configured: str) -> str:
    """The mesh the FE solve used, ``nx x ny x nz`` and the specimen it
    models; ``configured`` (the coupon mesh of the inputs) when no FE
    field was solved."""
    mesh = result.get("mesh")
    if mesh is None or result.get("fe_field") is None:
        return configured
    label = f"{mesh.nx}x{mesh.ny}x{mesh.nz}"
    specimen = _test_specimen_name(mesh)
    if specimen is None and getattr(mesh, "fracture_geometry", None) is not None:
        specimen = "fracture test specimen"
    return label if specimen is None else f"{label} ({specimen})"


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
            "NDI indication; assess a delamination of the part.")
    specimen = _test_specimen_name(mesh)
    if specimen is not None:
        raise ValueError(
            f"The FE solve is the {specimen}, not the part: its fracture "
            f"margin is the test coupon's. Assess a delamination of the part "
            f"with a coupon loading (compression, tension or shear).")
    if fe.fracture_load_factor is None:
        raise ValueError(
            "The FE solve has no fracture load factor: the material has no "
            "interlaminar toughness (G_Ic, G_IIc, bk_eta), so the delamination "
            "has no fracture margin.")
    return fe, front, mesh


def _delamination_record(result: dict, indication: str,
                         assessment: dict) -> dict:
    """The ``delamination`` block of an NCR record (lengths in mm, toughness
    and energy release rates in N/mm, load factors dimensionless), with the
    fracture ``assessment`` of :func:`assess_delamination`."""
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
        "required_load_factor": assessment["fracture_margin"]["required_load_factor"],
        "n_front_nodes": len(front),
        "governing_front_node": _governing_front_node(front),
        "knockdown_decomposition": (None if decomposition is None
                                    else decomposition.summary()),
        "assessment": assessment,
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
    final) disposition path for the porosity. Part/serial/work-order identification lives on
    the parent NCR — ``meta`` only needs who prepared it and an optional
    parent-NCR reference. Everything technical is derived from ``result`` so
    the summary cannot drift from what was actually run.

    ``meta["indication"]`` (default ``"porosity"``, :data:`INDICATION_TYPES`)
    selects the indication type. ``"delamination"`` and ``"combined"`` need
    ``result`` to hold an FE solve of a part with a delamination front and
    toughness data (``result["fe_field"]``, ``result["mesh"]``; the app's
    delamination input) and ``meta["required_load_factor"]``, the load
    factor the part must reach without delamination growth, as a multiple
    of the analysis load (see :func:`assess_delamination`). They add a
    ``delamination`` block: the interface depth and planform, the
    toughness and its porosity law, the fracture load factor, the governing
    front node, the knockdown decomposition when
    ``result["fe_decomposition"]`` holds one, and the fracture
    ``assessment``. The delamination gets **no recommended path**: for
    ``"delamination"`` ``recommended_disposition`` is ``None``, and for
    ``"combined"`` it is the porosity recommendation, which covers the
    porosity only. A porosity record is unchanged.

    ``meta["policy"]`` (a :class:`DispositionPolicy` or its ``to_dict()``
    record; default :data:`DEFAULT_DISPOSITION_POLICY`, whose bands are
    illustrative) sets the bands of the recommended path; the record keeps
    them under ``recommended_disposition["policy"]``.
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
    disposition = None
    if indication in ("porosity", "combined"):
        disposition = recommend_disposition(
            Vp, worst["knockdown"], structural_class, policy=meta.get("policy")
        )
    if indication in ("delamination", "combined"):
        fe, front, _ = _delamination_inputs(result)
        assessment = assess_delamination(
            fe.fracture_load_factor, meta.get("required_load_factor"),
            structural_class,
            toughness_uncalibrated=bool(front.toughness_uncalibrated))
        delamination = _delamination_record(result, indication, assessment)

    today = datetime.date.today().isoformat()
    layup = meta.get("layup") or "(see analysis configuration)"

    porosity_summary = (
        f"Porosity / void content of {Vp:.2f}% in a "
        f"{cfg['material']} laminate ({cfg['n_plies']} plies, layup "
        f"{layup}); {cfg['distribution']} distribution, "
        f"{cfg['void_shape']} void morphology. Its predicted residual-"
        f"strength knockdown is in the engineering analysis; the released "
        f"drawing's porosity allowable is the acceptance limit."
    )
    title = "Composite Porosity Analysis — NCR Validation Summary"
    summary = porosity_summary
    if delamination is not None:
        planforms = "; ".join(_planform_text(rec)
                              for rec in delamination["delaminations"])
        delamination_summary = (
            f"Delamination indication: {planforms}. Assessed on its "
            f"fracture margin, with no recommended disposition path — see "
            f"the delamination analysis.")
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
            "analysis_mesh": _analysis_mesh_label(result, cfg["mesh"]),
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
    margin = d["assessment"]["fracture_margin"]
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


_NO_DELAMINATION_PATH = (
    "No disposition path is recommended for the delamination: the MRB "
    "dispositions it on the fracture assessment of section 2b against the "
    "governing acceptance criteria."
)


def _ncr_lists(ncr: dict) -> tuple[list, list]:
    """The cited criteria and required MRB actions of an NCR record: the
    porosity recommendation's, then the delamination assessment's, each
    entry once."""
    parts = [ncr["recommended_disposition"],
             ncr.get("delamination", {}).get("assessment")]
    criteria: list = []
    actions: list = []
    for part in parts:
        if part is None:
            continue
        criteria += [c for c in part["cited_criteria"] if c not in criteria]
        actions += [a for a in part["required_mrb_actions"] if a not in actions]
    return criteria, actions


def _policy_text(dp: dict) -> str | None:
    """The bands behind a recommended path, one line; ``None`` for a record
    without them."""
    policy = dp.get("policy")
    if not policy:
        return None
    return DispositionPolicy.from_dict(policy).describe()


def _footer(ncr: dict) -> str:
    """Closing line of the Markdown and PDF attachments."""
    if ncr["recommended_disposition"] is None:
        return ("Generated by PorosityFE MRB support tool. This is a "
                "predictive analysis; it recommends no disposition and does "
                "not constitute MRB approval.")
    return ("Generated by PorosityFE MRB support tool. This is a predictive "
            "analysis and a recommended disposition path; it does not "
            "constitute MRB approval.")


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

    assessment = ncr.get("delamination", {}).get("assessment")
    if dp is not None:
        lines.append(
            "## 3. Recommended Disposition Path "
            "(for MRB review — NOT a final disposition)"
        )
        lines.append("")
        scope = " (porosity only)" if assessment is not None else ""
        lines.append(f"**Recommended path{scope}:** {dp['path']}")
        lines.append("")
        lines.append(f"**Rationale:** {dp['rationale']}")
        lines.append("")
        policy_text = _policy_text(dp)
        if policy_text:
            lines.append(f"**Disposition policy:** {policy_text}")
            lines.append("")
        if assessment is not None:
            lines.append(f"**Delamination:** {_NO_DELAMINATION_PATH}")
            lines.append("")
        lines.append(f"> {dp['disclaimer']}")
        lines.append("")
    else:
        lines.append("## 3. Disposition (by the MRB)")
        lines.append("")
        lines.append(_NO_DELAMINATION_PATH)
        lines.append("")
    if assessment is not None:
        lines.append(f"**Fracture assessment:** {assessment['statement']}")
        lines.append("")
        lines.append(f"> {assessment['disclaimer']}")
        lines.append("")

    criteria, actions = _ncr_lists(ncr)
    lines.append("## 4. Cited Criteria")
    lines.append("")
    for c in criteria:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## 5. Required MRB Actions")
    lines.append("")
    for a in actions:
        lines.append(f"- [ ] {a}")
    lines.append("")
    lines.append(f"_{_footer(ncr)}_")
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

    assessment = ncr.get("delamination", {}).get("assessment")
    if dp is not None:
        out.append("3. RECOMMENDED DISPOSITION PATH")
        out.append("   (for MRB review - NOT a final disposition)")
        rule()
        scope = " (porosity only)" if assessment is not None else ""
        para(f"Recommended path{scope}: {dp['path']}")
        out.append("")
        para(f"Rationale: {dp['rationale']}")
        out.append("")
        policy_text = _policy_text(dp)
        if policy_text:
            para(f"Disposition policy: {policy_text}")
            out.append("")
        if assessment is not None:
            para(f"Delamination: {_NO_DELAMINATION_PATH}")
            out.append("")
        para(dp["disclaimer"], indent="  ")
        out.append("")
    else:
        out.append("3. DISPOSITION (by the MRB)")
        rule()
        para(_NO_DELAMINATION_PATH)
        out.append("")
    if assessment is not None:
        para(f"Fracture assessment: {assessment['statement']}")
        out.append("")
        para(assessment["disclaimer"], indent="  ")
        out.append("")

    criteria, actions = _ncr_lists(ncr)
    out.append("4. CITED CRITERIA")
    rule()
    for c in criteria:
        para(f"- {c}", indent="  ")
    out.append("")

    out.append("5. REQUIRED MRB ACTIONS")
    rule()
    for a in actions:
        para(f"[ ] {a}", indent="  ")
    out.append("")
    para(_footer(ncr))
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
                # Free text (notes, names, part numbers) is printed as
                # typed: '$...$' would otherwise be read as mathtext.
                parse_math=False, usetex=False,
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
