"""argparse-driven CLI (``porosity-analyze``)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .delamination import Delamination
from .homogenization import _DEFAULT_POROSITY_DEGRADATION, POROSITY_DEGRADATIONS
from .io import save_results_to_json
from .materials import MATERIALS
from .pipeline import (
    _laminate_material,
    _resolve_n_jobs,
    compare_configurations,
    sweep_configurations,
)
from .porosity_field import POROSITY_CONFIGS
from .viz import FEVisualizer

logger = logging.getLogger("porosity_fe_analysis")

# Default Vp sweep — preserves the historical hardcoded behavior so that
# `porosity-analyze` with no arguments reproduces today's analysis range.
DEFAULT_POROSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08]


#: Placeholder passed through to compare_configurations, which ignores it.
_DEFAULT_APPLIED_STRESS = -1500.0

#: Preset of a run that names no material.
_DEFAULT_MATERIAL = "T800_epoxy"

#: Defaults of the delamination screening options (``--delamination``),
#: applied only when it is given, so the options can tell "not given".
_DELAM_LOADINGS = ('compression', 'tension', 'shear')
_DELAM_DEFAULT_LOADING = 'compression'
_DELAM_DEFAULT_CONFIG = 'uniform_spherical'
_DELAM_DEFAULT_LAW = 'exponential'
_DELAM_DEFAULT_DEGRADATION = _DEFAULT_POROSITY_DEGRADATION

#: ``--mean-stress`` corrections, and the modes ``--stress-ratio`` applies to
#: (``--compression-stress-ratio`` sets the compression one).
_MEAN_STRESS_CHOICES = ('goodman',)
_TENSION_TYPE_MODES = ('tension', 'shear', 'ilss', 'transverse_tension')

#: ``--material-cards`` formats with their file suffix, and the element
#: families of ``--card-element`` (the first is the default).
_CARD_SUFFIX = {'abaqus': '.inp', 'nastran': '.bdf'}
_CARD_FORMATS = tuple(_CARD_SUFFIX)
_CARD_ELEMENTS = ('shell', 'solid')


def _parse_delamination(spec: str) -> Delamination:
    """``--delamination`` value as a :class:`Delamination`.

    ``strip:K:X_START:X_END`` or ``rectangle|ellipse:K:XC:YC:LX:WY``: the
    interface ``K`` (between ply ``K`` and ``K + 1`` from the bottom,
    0-based) and, in mm on the FE coupon, the strip's ``x`` range or the
    centre and the full length along ``x`` and width along ``y`` (as an NDI
    indication is reported; the half-sizes are half of them).
    """
    parts = [p.strip() for p in str(spec).split(':')]
    shape = parts[0].lower()
    expected = {'strip': 3, 'rectangle': 5, 'ellipse': 5}
    if shape not in expected:
        raise argparse.ArgumentTypeError(
            f"unknown planform {parts[0]!r} in {spec!r}; use "
            f"'strip:K:X_START:X_END', 'rectangle:K:XC:YC:LX:WY' or "
            f"'ellipse:K:XC:YC:LX:WY' (mm).")
    if len(parts) - 1 != expected[shape]:
        raise argparse.ArgumentTypeError(
            f"{shape!r} takes {expected[shape]} numbers after the shape, got "
            f"{len(parts) - 1} in {spec!r}.")
    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"the interface and lengths must be numbers, got {spec!r}.") from None
    k = values[0]
    if not (math.isfinite(k) and k == int(k)):
        raise argparse.ArgumentTypeError(
            f"the interface must be an integer >= 0, got {parts[1]!r}.")
    try:
        if shape == 'strip':
            return Delamination.strip(int(k), values[1], values[2])
        xc, yc, lx, wy = values[1:]
        build = Delamination.ellipse if shape == 'ellipse' else Delamination.rectangle
        return build(int(k), (xc, yc), (0.5 * lx, 0.5 * wy))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


#: FE loadings of ``--fe``: the coupon modes and the ASTM D2344 / D7264
#: specimens. The fracture specimens and the thermal case are not offered
#: (the delamination screening covers a delamination; they need inputs the
#: CLI does not take).
_FE_LOADINGS = ('compression', 'tension', 'shear', 'ilss',
                'flexure_3pt', 'flexure_4pt')


def _parse_layup_option(text: str | None):
    """``--layup`` as the layup the solvers take: ``'QI'`` (also for no
    option), ``'UD'``, or the ply angles of laminate notation."""
    if text is None:
        return 'QI'
    name = text.strip().upper()
    if name in ('QI', 'UD'):
        return name
    from .reporting import parse_layup
    return [float(a) for a in parse_layup(text)]


#: Seed of the ``--uq`` sampling when ``--seed`` is not given: the app's
#: value, so a repeat run gives the same bands.
_UQ_DEFAULT_SEED = 0


def _vp_label(Vp: float) -> str:
    """Stable, filesystem-safe label for a void fraction.

    Integer-percent fractions keep the legacy ``Npct`` form (e.g. 0.03 ->
    ``3pct``); non-integer fractions fall back to a decimal-derived form
    (e.g. 0.025 -> ``2p5pct``) so distinct sweeps never collide.
    """
    pct = Vp * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    return f"{pct:.4f}".rstrip('0').rstrip('.').replace('.', 'p') + "pct"


class _DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Append ``(default: ...)`` only for a real default: an option whose
    default is ``None`` states its effective default in its own help, which
    "(default: None)" contradicted."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse driver for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        prog="porosity-analyze",
        description=(
            "Run the porosity-degraded composite laminate analysis over one "
            "or more void volume fractions and write JSON results."
        ),
        formatter_class=_DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--material",
        default=None,
        help=("Material preset name (validated against the built-in presets; "
              f"the default is {_DEFAULT_MATERIAL})."),
    )
    parser.add_argument(
        "--material-file", default=None, metavar="PATH",
        help=(
            "A material JSON file to analyse instead of a preset, as "
            "save_material_to_json writes it (or a bare JSON object of the "
            "material's fields, in MPa and mm). Its name, or the file's, "
            "labels the run. Not with --material."
        ),
    )
    parser.add_argument(
        "--layup", default=None, metavar="SPEC",
        help=(
            "Layup of the laminate: 'QI' (default: [0/90/45/-45]_s repeated "
            "to the preset's ply count), 'UD', or laminate notation with the "
            "ply angles in degrees, such as '[0/45/-45/90]_2s' or "
            "'[0_2/±45]s'. An explicit layup sets the ply count. It applies "
            "to the sweep, --uq, --fe, --delamination and "
            "--required-knockdown, and is recorded in the JSON."
        ),
    )
    parser.add_argument(
        "--t-ply", type=float, default=None, metavar="MM",
        help="Ply thickness in mm (default: the preset's).",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None, metavar="NAME",
        help=(
            "Porosity configurations to analyse, keys of POROSITY_CONFIGS "
            f"({', '.join(POROSITY_CONFIGS)}). Default: all of them."
        ),
    )
    parser.add_argument(
        "--required-knockdown", type=float, default=None, metavar="R",
        help=(
            "Also report the allowable porosity: the largest void content at "
            "which each strength mode still retains the fraction R of its "
            "pristine strength (0.9 keeps 90%%), for the three knockdown laws, "
            "with the governing (smallest) one. Written to "
            "porosity_allowable.json. It depends on the material and the "
            "layup, not on --vp or the configurations."
        ),
    )
    # #127: the Streamlit app collects Vp as a percentage (0-100) while the
    # CLI historically required a fraction in [0, 1]. Users who used the GUI
    # first naturally type `--vp 3` and get a confusing error. We keep `--vp`
    # as fractions (the documented contract) and add a `--vp-pct` alias for
    # callers who want to think in percent. The two are mutually exclusive.
    vp_group = parser.add_mutually_exclusive_group()
    vp_group.add_argument(
        "--vp",
        type=float,
        nargs="+",
        default=None,
        metavar="VP",
        help=(
            "One or more void volume fractions as a decimal in [0, 1]. "
            "Example: --vp 0.03 for 3%%, --vp 0.01 0.03 0.05 for a sweep. "
            "Use --vp-pct if you'd rather pass percentages (the Streamlit "
            "app accepts percentages; this CLI defaults to fractions)."
        ),
    )
    vp_group.add_argument(
        "--vp-pct",
        type=float,
        nargs="+",
        default=None,
        metavar="VP_PCT",
        help=(
            "One or more void volume fractions as a percentage in [0, 100]. "
            "Internally converted to fractions (--vp-pct 3 is equivalent to "
            "--vp 0.03)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write JSON results into (created if missing).",
    )
    parser.add_argument(
        "--applied-stress",
        type=float,
        default=None,
        help=(
            "Applied stress (MPa), reserved for downstream solver hooks. "
            "Accepted for parity with compare_configurations but currently "
            "unused: the empirical knockdown sweep does not consume it, so "
            "changing this value will not affect results, and passing it "
            "logs a warning. Retained as a stable CLI surface for future "
            "use (#132)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed of the --uq sampling (default 0, as in the app, so repeat "
            "runs agree), recorded in the JSON provenance. The sweep itself "
            "is deterministic (RNG-free): the seed does not alter it."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Also render the heavy matplotlib figures (PNG). Off by default "
            "to keep CI / batch runs fast."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress progress output; only warnings and errors are shown. "
            "Mutually exclusive with --verbose."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Show debug-level progress output in addition to the default "
            "INFO progress. Mutually exclusive with --quiet."
        ),
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="List the available material presets and exit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of worker processes for the per-configuration sweep "
            "in compare_configurations (#52). 1 (default) runs serially "
            "(deterministic, byte-identical to legacy behaviour); N>1 "
            "parallelises the (Vp, config) calls across N processes; 0 or "
            "-1 uses os.cpu_count(). Results are deterministically "
            "re-assembled regardless of N."
        ),
    )
    fe = parser.add_argument_group(
        "finite-element solve",
        "Opt-in: an FE solve of every analysed configuration at each "
        "porosity level, written to porosity_fe_<loading>_<config>_<Vp>.json "
        "(stiffness knockdown, effective modulus, failure indices, "
        "first-ply-failure load factor).")
    fe.add_argument(
        "--fe", choices=_FE_LOADINGS, default=None, metavar="LOADING",
        help=(
            f"FE loading: {', '.join(_FE_LOADINGS)}. The coupon modes solve "
            "the production mesh (30 x 10, one element layer per ply); "
            "'ilss' the ASTM D2344 short beam and the flexure modes the ASTM "
            "D7264 specimen. The FE strength rests on an uncalibrated "
            "heuristic: take the size of a strength knockdown from the sweep."
        ),
    )
    dur = parser.add_argument_group(
        "fatigue and environment (empirical)",
        "Opt-in: the knockdown of every strength mode and law after load "
        "cycles and at a service temperature and moisture, at each porosity "
        "level, written to porosity_durability_<Vp>.json. With "
        "--required-knockdown the allowable porosity holds at the same "
        "conditions. Screening-level models: see the fatigue theory page.")
    dur.add_argument(
        "--cycles", type=float, default=None, metavar="N",
        help="Load cycles of the S-N fatigue factor (log-linear, calibrated "
             "at R = 0.1).")
    dur.add_argument(
        "--mean-stress", choices=_MEAN_STRESS_CHOICES, default=None,
        help="Mean-stress correction of the S-N curve to the stress ratios "
             "below: the straight constant-life line through the calibrated "
             "point and the static strength. Uncalibrated against "
             "porous-laminate fatigue data. Needs --cycles.")
    dur.add_argument(
        "--stress-ratio", type=float, default=None, metavar="R",
        help="sigma_min / sigma_max of the tension, shear, ILSS and "
             "transverse-tension modes, 0 <= R <= 1 (the default is the "
             "calibration ratio, 0.1). Needs --mean-stress.")
    dur.add_argument(
        "--compression-stress-ratio", type=float, default=None, metavar="R",
        help="sigma_min / sigma_max of the compression mode, R >= 1 "
             "(compression-compression; the default is 10). Needs --mean-stress.")
    dur.add_argument(
        "--fatigue-porosity-slope", type=float, default=None, metavar="K",
        help="Porosity dependence of the S-N slope, b(Vp) = b (1 + K Vp) with "
             "Vp a fraction, K >= 0. No calibrated value is shipped. Needs "
             "--cycles.")
    dur.add_argument(
        "--service-temperature", type=float, default=None, metavar="T",
        help="Service temperature in deg C of the hygrothermal factor "
             "(matrix-dominated modes; the default is the reference, 23). "
             "Needs a dry glass-transition temperature.")
    dur.add_argument(
        "--moisture", type=float, default=None, metavar="M",
        help="Moisture content in wt %% of the hygrothermal factor (the "
             "default is the reference, 0). Needs a dry glass-transition "
             "temperature.")
    dur.add_argument(
        "--tg-dry", type=float, default=None, metavar="TG",
        help="Dry glass-transition temperature in deg C for the hygrothermal "
             "factor, replacing the material's T_g_dry (no preset sets one).")
    cards = parser.add_argument_group(
        "material cards",
        "Opt-in: the porosity-degraded ply properties of every analysed "
        "configuration at each porosity level as solver material cards, "
        "written to porosity_cards_<config>_<Vp>.inp (Abaqus) or .bdf "
        "(Nastran), one material per distinct ply void content, with the "
        "ply table.")
    cards.add_argument(
        "--material-cards", choices=_CARD_FORMATS, default=None, metavar="FORMAT",
        help=(
            f"Card format: {', '.join(_CARD_FORMATS)}. The stiffness is the "
            "one the FE path gives a ply at its void content; the strengths "
            "carry the same uncalibrated heuristic as the FE failure indices."
        ),
    )
    cards.add_argument(
        "--card-element", choices=_CARD_ELEMENTS, default=None, metavar="ELEMENT",
        help=(
            f"Element family the cards are for: {', '.join(_CARD_ELEMENTS)} "
            "(the default is shell). Shell: an Abaqus lamina with its failure "
            "stresses, or a Nastran MAT8. Solid: Abaqus engineering constants, "
            "or a Nastran MAT9. Needs --material-cards."
        ),
    )
    uq = parser.add_argument_group(
        "uncertainty quantification",
        "Propagate input scatter through the empirical Judd-Wright knockdown "
        "for every loading mode and write porosity_uq_<Vp>.json per level.")
    uq.add_argument(
        "--uq", action="store_true",
        help="Run the uncertainty propagation after the sweep.")
    uq.add_argument(
        "--uq-samples", type=int, default=500, metavar="N",
        help="Latin-hypercube draws per loading mode.")
    uq.add_argument(
        "--uq-coef-cov", type=float, default=0.10, metavar="COV",
        help=("CoV of the knockdown law's calibration coefficient, as a "
              "fraction (0.10 = 10%%). The default is an assumed value; this "
              "is usually the dominant term."))
    uq.add_argument(
        "--uq-vp-cov", type=float, default=0.10, metavar="COV",
        help=("CoV of the measured mean porosity, as a fraction of it "
              "(0.10 = 10%% of Vp, not 10 percentage points)."))
    uq.add_argument(
        "--uq-strength-cov", type=float, default=0.05, metavar="COV",
        help="CoV of each mode's pristine strength, as a fraction (0.05 = 5%%).")
    delam = parser.add_argument_group(
        "delamination screening (FE)",
        "Opt-in: an FE solve of the coupon (production mesh, the laminate of "
        "--layup) with an explicit delamination at each porosity level, written to "
        "porosity_delamination_<Vp>.json: the FE results with the fracture "
        "block (energy release rates along the front, fracture load factor) "
        "and the knockdown split into porosity-only and delamination-only "
        "parts. Each level takes three FE solves. Lengths in mm, toughness in "
        "N/mm.")
    delam.add_argument(
        "--delamination", type=_parse_delamination, default=None,
        metavar="SPEC",
        help=("The delamination: 'strip:K:X_START:X_END', "
              "'rectangle:K:XC:YC:LX:WY' or 'ellipse:K:XC:YC:LX:WY', with K "
              "the interface (between ply K and K + 1 from the bottom, "
              "0-based) and the position and full sizes in mm on the 50 x 20 "
              "mm coupon, x along the load. Example: --delamination "
              "ellipse:11:25:10:12:8."))
    delam.add_argument(
        "--delam-loading", choices=_DELAM_LOADINGS, default=None,
        help=f"FE loading of the screening (default {_DELAM_DEFAULT_LOADING}).")
    delam.add_argument(
        "--delam-config", default=None, metavar="NAME",
        help=(f"Porosity configuration of the screening, a key of "
              f"POROSITY_CONFIGS (default {_DELAM_DEFAULT_CONFIG})."))
    delam.add_argument(
        "--toughness", type=float, nargs="+", default=None,
        metavar="G",
        help=("Interlaminar toughness G_IC G_IIC BK_ETA [G_IIIC]: mode I and "
              "II toughness (N/mm), the Benzeggagh-Kenane exponent and "
              "optionally the mode III toughness (default G_IIC). Of the "
              "presets only IM7_8552_epoxy carries them (these override it); "
              "for the others, without them the screening reports energy "
              "release rates only, no fracture load factor."))
    delam.add_argument(
        "--toughness-knockdown", choices=('exponential', 'none'), default=None,
        help=("Porosity knockdown of the toughness at the front: "
              "'exponential' (default; the uncalibrated placeholders "
              "b_I = 6.2, b_II = 14, flagged in the output unless "
              "--toughness-b is given) or 'none' (pristine toughness)."))
    delam.add_argument(
        "--toughness-b", type=float, nargs=2, default=None,
        metavar=("B_I", "B_II"),
        help="Your coefficients of the exponential law (per unit Vp).")
    delam.add_argument(
        "--porosity-degradation", choices=POROSITY_DEGRADATIONS, default=None,
        help=("With --delamination or --fe: how the porosity degrades the ply "
              "stiffness and strengths in "
              "the FE solves (FESolver(porosity_degradation=...)): "
              f"'{_DELAM_DEFAULT_DEGRADATION}' (default) or 'directional' "
              "(opt-in, uncalibrated: flat voids weaken E33, G13, G23, Z_t "
              "and S_13 more than the in-plane properties; spherical voids "
              "give the same results). Recorded in the JSON."))
    return parser


def _delamination_options(args) -> dict:
    """The resolved delamination screening options; ``ValueError`` when one
    is given without ``--delamination`` or is invalid."""
    given = [name for name in ('delam_loading', 'delam_config', 'toughness',
                               'toughness_knockdown', 'toughness_b')
             if getattr(args, name) is not None]
    if args.delamination is None:
        if given:
            flags = ", ".join("--" + n.replace('_', '-') for n in given)
            raise ValueError(f"{flags} need --delamination.")
        if (args.porosity_degradation is not None and args.fe is None
                and args.material_cards is None):
            raise ValueError("--porosity-degradation needs --delamination, "
                             "--fe or --material-cards.")
        return {}
    toughness = args.toughness
    if toughness is not None:
        if len(toughness) not in (3, 4):
            raise ValueError(
                f"--toughness takes G_IC G_IIC BK_ETA [G_IIIC], got "
                f"{len(toughness)} values.")
        bad = [v for v in toughness if not (math.isfinite(v) and v > 0.0)]
        if bad:
            raise ValueError(
                f"--toughness values must be positive and finite (N/mm), got "
                f"{bad[0]!r}.")
    from .toughness import ToughnessKnockdown, ToughnessKnockdownSpec

    law = args.toughness_knockdown or _DELAM_DEFAULT_LAW
    knockdown: ToughnessKnockdownSpec = None if law == 'none' else law
    if args.toughness_b is not None:
        if law == 'none':
            raise ValueError("--toughness-b is the exponential law; it cannot "
                             "be combined with --toughness-knockdown none.")
        knockdown = ToughnessKnockdown(b_I=args.toughness_b[0],
                                       b_II=args.toughness_b[1])
    return {
        'delamination': args.delamination,
        'loading': args.delam_loading or _DELAM_DEFAULT_LOADING,
        'config': args.delam_config or _DELAM_DEFAULT_CONFIG,
        'toughness': toughness,
        'toughness_knockdown': knockdown,
        'porosity_degradation': args.porosity_degradation or _DELAM_DEFAULT_DEGRADATION,
    }


def _run_delamination(options: dict, material, porosity_configs: dict,
                      Vp: float, out_path: Path, *, quiet_solver: bool = True,
                      ply_angles='QI') -> str:
    """FE delamination screening at one porosity level, written to ``out_path``.

    One FE solve of the production coupon with the delamination, its
    knockdown decomposition (two more solves) and the FE JSON export with
    the ``fracture`` and ``knockdown_decomposition`` blocks. The solver's
    own progress messages are held back unless ``quiet_solver`` is false
    (``--verbose``). Returns a one-line summary.
    """
    level = logger.level
    if quiet_solver:
        logger.setLevel(max(level, logging.WARNING))
    # Passed only when it is not the default, so the default call is the
    # one it always was.
    layup = {} if isinstance(ply_angles, str) and ply_angles == 'QI' \
        else {'ply_angles': ply_angles}
    try:
        return _delamination_screening(options, material, porosity_configs,
                                       Vp, out_path, **layup)
    finally:
        logger.setLevel(level)


def _run_fe(loading: str, material, ply_angles, name: str, config: dict,
            Vp: float, out_path: Path, *, porosity_degradation: str,
            quiet_solver: bool = True) -> str:
    """One FE solve of a configuration (``--fe``), written to ``out_path``.

    The coupon modes solve the production mesh; ``'ilss'`` and the flexure
    modes their ASTM specimen with its default mesh. The solver's own
    progress messages are held back unless ``quiet_solver`` is false.
    Returns a one-line summary.
    """
    from .fe import FESolver
    from .mesh import CompositeMesh
    from .pipeline import _default_mesh_resolution
    from .porosity_field import PorosityField

    level = logger.level
    if quiet_solver:
        logger.setLevel(max(level, logging.WARNING))
    try:
        pf = PorosityField(material, Vp, **config)
        if loading == 'ilss':
            mesh = CompositeMesh.for_ilss(pf, material, ply_angles=ply_angles)
        elif loading in ('flexure_3pt', 'flexure_4pt'):
            mesh = CompositeMesh.for_flexure(pf, material, ply_angles=ply_angles)
        else:
            mesh = CompositeMesh(
                pf, material,
                *_default_mesh_resolution(material.n_plies, ply_angles),
                ply_angles=ply_angles)
        result = FESolver(mesh, material, pf, ply_angles=ply_angles,
                          porosity_degradation=porosity_degradation).solve(loading)
        FESolver.export_results(result, out_path)
    finally:
        logger.setLevel(level)
    modulus = result.effective_modulus
    return (f"  {loading}, {name}: stiffness knockdown {result.knockdown:.4f}"
            + ("" if modulus is None else f", effective modulus {modulus:.0f} MPa")
            + f", max failure index {result.max_failure_index:.3g}, "
              f"first-ply-failure load factor "
              f"{result.first_ply_failure_load_factor:.4g}"
            + ("" if porosity_degradation == _DELAM_DEFAULT_DEGRADATION
               else f"; porosity degradation {porosity_degradation} (uncalibrated)")
            + f"\n  Saved: {out_path}")


def _run_material_cards(fmt: str, material, ply_angles, name: str, config: dict,
                        Vp: float, out_path: Path, *, element: str,
                        porosity_degradation: str) -> str:
    """The material cards of one configuration (``--material-cards``),
    written to ``out_path``. Returns a one-line summary."""
    from .material_cards import ply_porosities, write_material_cards
    from .porosity_field import PorosityField

    pf = PorosityField(material, Vp, **config)
    write_material_cards(out_path, material, pf, fmt, ply_angles=ply_angles,
                         element=element,
                         porosity_degradation=porosity_degradation)
    plies = ply_porosities(pf)
    n_materials = len(set(np.round(plies, 9).tolist()))
    return (f"  {name}: {n_materials} material{'s' if n_materials != 1 else ''}, "
            f"ply void content {100.0 * plies.min():.2f} to "
            f"{100.0 * plies.max():.2f} %\n  Saved: {out_path}")


def _durability_options(args, material) -> dict:
    """The fatigue and hygrothermal options, resolved; empty when none is
    given. ``ValueError`` names an option given without what it needs."""
    from .fatigue import FatigueModel

    def needs(flags: list[str], what: str) -> None:
        given = [f for f in flags if getattr(args, f.lstrip('-').replace('-', '_')) is not None]
        if given:
            raise ValueError(f"{', '.join(given)} need{'s' if len(given) == 1 else ''} {what}.")

    def finite(flag: str, low: float | None = None) -> float | None:
        value = getattr(args, flag.lstrip('-').replace('-', '_'))
        if value is not None and not (math.isfinite(value) and (low is None or value >= low)):
            bound = "" if low is None else f" >= {low:g}"
            raise ValueError(f"{flag} must be a finite number{bound}; got {value!r}.")
        return value

    if args.cycles is None:
        needs(['--mean-stress', '--stress-ratio', '--compression-stress-ratio',
               '--fatigue-porosity-slope'], '--cycles')
    if args.mean_stress is None:
        needs(['--stress-ratio', '--compression-stress-ratio'],
              '--mean-stress (without a mean-stress correction the S-N curve does '
              'not depend on R)')
    if args.service_temperature is None and args.moisture is None:
        needs(['--tg-dry'], '--service-temperature or --moisture')
    cycles = finite('--cycles', 1.0)
    slope = finite('--fatigue-porosity-slope', 0.0)
    T = finite('--service-temperature')
    M = finite('--moisture', 0.0)
    T_g = finite('--tg-dry')
    finite('--stress-ratio')
    finite('--compression-stress-ratio')
    if cycles is None and T is None and M is None:
        return {}

    R: dict[str, float] = {}
    if args.stress_ratio is not None:
        R.update({mode: float(args.stress_ratio) for mode in _TENSION_TYPE_MODES})
    if args.compression_stress_ratio is not None:
        R['compression'] = float(args.compression_stress_ratio)
    fatigue = None
    if args.mean_stress is not None or slope:
        fatigue = FatigueModel(mean_stress=args.mean_stress, porosity_slope=slope or 0.0)
        for mode, ratio in R.items():
            # A ratio the mode cannot see (e.g. R < 1 for compression).
            try:
                fatigue.knockdown_factor(mode, 10.0, ratio, Vp=0.0)
            except ValueError as exc:
                flag = '--compression-stress-ratio' if mode == 'compression' \
                    else '--stress-ratio'
                raise ValueError(f"{flag}: {exc}") from None
    environment = None
    if T is not None or M is not None:
        if T_g is None and material.T_g_dry is None:
            raise ValueError(
                "--service-temperature and --moisture need a dry glass-transition "
                "temperature: give --tg-dry, or a --material-file that sets "
                "T_g_dry (no preset does).")
        environment = {'T': float(material.T_ref) if T is None else float(T),
                       'M': float(material.M_ref) if M is None else float(M)}
    return {'cycles': cycles, 'R': R or None, 'fatigue_model': fatigue,
            'environment': environment, 'T_g_dry': T_g}


def _durability_solver(material, ply_angles, Vp: float, options: dict):
    """The empirical pipeline of the laminate at ``Vp`` with the fatigue
    model and dry T_g of ``options``: ``(material, mesh, solver)``."""
    from .pipeline import build_empirical_pipeline

    if options.get('T_g_dry') is not None:
        material = dataclasses.replace(material, T_g_dry=float(options['T_g_dry']))
    fatigue = options.get('fatigue_model')
    level = logger.level
    logger.setLevel(max(level, logging.WARNING))    # no mesh report for a table
    try:
        _, mesh, solver = build_empirical_pipeline(
            material, Vp, ply_angles=ply_angles,
            **({'solver_kwargs': {'fatigue_model': fatigue}} if fatigue else {}))
    finally:
        logger.setLevel(level)
    return material, mesh, solver


def _laminate_block(material, material_name: str, mesh) -> dict:
    return {'material': material_name,
            'ply_angles': [float(a) for a in mesh._ply_layup],
            'n_plies': int(material.n_plies),
            't_ply_mm': float(material.t_ply)}


def _run_durability(options: dict, material, material_name: str, ply_angles,
                    Vp: float, out_path: Path) -> list[str]:
    """The fatigue and hygrothermal knockdowns at one porosity level
    (``--cycles``, ``--service-temperature``), written to ``out_path``.
    Returns the lines of its summary."""
    from .reporting import durability_record, write_durability_json

    material, mesh, solver = _durability_solver(material, ply_angles, Vp, options)
    record = durability_record(solver, cycles=options['cycles'], R=options['R'],
                               environment=options['environment'])
    write_durability_json(out_path, record, _laminate_block(material, material_name, mesh),
                          void_volume_fraction=Vp)
    lines = []
    for mode, models in record['per_mode'].items():
        jw = models['judd_wright']
        lines.append(f"  {mode:<20} judd_wright {jw['knockdown']:.4f} (porosity "
                     f"{jw['static_knockdown']:.4f} x fatigue {jw['fatigue_knockdown']:.4f}"
                     f" x environment {jw['environment_knockdown']:.4f})")
    gov = record['governing']
    lines.append(f"  Governing: {gov['mode']} / {gov['model']}, knockdown "
                 f"{gov['knockdown']:.4f}")
    lines.append(f"  Saved: {out_path}")
    return lines


def _run_allowable(required: float, material, material_name: str, ply_angles,
                   out_path: Path, durability: dict | None = None) -> list[str]:
    """The allowable-porosity table of the laminate (``--required-knockdown``),
    at the fatigue and hygrothermal conditions of ``durability`` when given,
    written to ``out_path``. Returns the lines of its summary."""
    import warnings

    from .reporting import (
        allowable_porosity_record,
        fatigue_conditions,
        write_allowable_json,
    )

    options = durability or {}
    material, mesh, solver = _durability_solver(material, ply_angles, 0.0, options)
    conditions = {'cycles': options.get('cycles'), 'R': options.get('R'),
                  'environment': options.get('environment')}
    with warnings.catch_warnings():
        # The record flags an extrapolated entry itself.
        warnings.simplefilter("ignore", UserWarning)
        table = solver.get_all_allowable_porosities(required, **conditions)
    record = allowable_porosity_record(table)
    write_allowable_json(out_path, record, _laminate_block(material, material_name, mesh),
                         fatigue_conditions(solver, **conditions) if options else None)
    lines = [f"  {record['statement']}"]
    for mode, models in record['per_mode'].items():
        # A fatigue or environment factor can put a requirement out of reach.
        cells = [f"{model} " + ("not achievable" if entry['allowable_Vp_percent'] is None
                                else f"{entry['allowable_Vp_percent']:.2f}%")
                 + (" (extrapolated)" if entry['extrapolated'] else "")
                 for model, entry in models.items()]
        lines.append(f"  {mode:<20} " + ", ".join(cells))
    lines.append(f"  {record['basis']}")
    lines.append(f"  Saved: {out_path}")
    return lines


def _delamination_screening(options: dict, material, porosity_configs: dict,
                            Vp: float, out_path: Path, ply_angles='QI') -> str:
    """The solves and export of :func:`_run_delamination`."""
    from .fe import FESolver, knockdown_decomposition
    from .mesh import CompositeMesh
    from .pipeline import _default_mesh_resolution
    from .porosity_field import PorosityField

    config = options['config']
    if config not in porosity_configs:
        raise ValueError(f"Unknown --delam-config {config!r}; use one of "
                         f"{sorted(porosity_configs)}.")
    toughness = options['toughness']
    if toughness is not None:
        G_Ic, G_IIc, eta = toughness[:3]
        G_IIIc = toughness[3] if len(toughness) == 4 else None
        material = dataclasses.replace(material, G_Ic=G_Ic, G_IIc=G_IIc,
                                       G_IIIc=G_IIIc, bk_eta=eta)
    pf = PorosityField(material, Vp, **porosity_configs[config])
    # The 'QI' default is spelled as before, so the default path is unchanged.
    layup = {} if isinstance(ply_angles, str) and ply_angles == 'QI' \
        else {'ply_angles': ply_angles}
    mesh = CompositeMesh(
        pf, material,
        *_default_mesh_resolution(material.n_plies, ply_angles if layup else None),
        delaminations=[options['delamination']], **layup)
    loading = options['loading']
    degradation = options.get('porosity_degradation', _DELAM_DEFAULT_DEGRADATION)
    result = FESolver(mesh, material, pf, porosity_degradation=degradation,
                      **layup).solve(
        loading, toughness_knockdown=options['toughness_knockdown'])
    parts = knockdown_decomposition(mesh, material, pf, loading, total=result,
                                    porosity_degradation=degradation)
    FESolver.export_results(result, out_path, knockdown_decomposition=parts)
    front = result.crack_front
    lam = result.fracture_load_factor
    where = ""
    if front is not None and front.governing is not None:
        row = front.governing
        where = (f" at x = {front.x[row]:.1f}, y = {front.y[row]:.1f} mm, "
                 f"psi = {front.psi[row]:.2f}")
    return (f"  {loading}, {config}: lambda_c = "
            f"{'n/a (no toughness)' if lam is None else f'{lam:.4g}'}{where}; "
            f"knockdown total {parts.total:.4f}, porosity only "
            f"{parts.porosity_only:.4f}, delamination only "
            f"{parts.delamination_only:.4f}"
            + ("; toughness knockdown uncalibrated"
               if front is not None and front.toughness_uncalibrated else "")
            + ("" if degradation == _DELAM_DEFAULT_DEGRADATION
               else f"; porosity degradation {degradation} (uncalibrated)")
            + f"\n  Saved: {out_path}")


def _load_material_file(path: str):
    """The material of ``--material-file`` and the label of its run: the
    name stored in the file, or the file's own."""
    from .io import _material_from_document

    with open(path, encoding='utf-8') as f:
        try:
            document = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc}).") from None
    material, name = _material_from_document(document, path)
    return material, name or Path(path).stem


def _run_uq(args, material, Vp: float, out_path: Path, ply_angles=None) -> dict:
    """Propagate uncertainty for every loading mode at one porosity level.

    ``material`` is a preset name or a :class:`MaterialProperties`;
    ``ply_angles`` the layup when it is not the default.
    """
    from .empirical import EmpiricalSolver
    from .uq import propagate_uncertainty, save_uq_results_to_json

    layup = {} if ply_angles is None else {'ply_angles': ply_angles}
    results = {}
    for mode, strength_field in EmpiricalSolver.PRISTINE_STRENGTH_KEY.items():
        results[mode] = propagate_uncertainty(
            Vp, material, mode, 'judd_wright',
            covs={strength_field: args.uq_strength_cov},
            vp_cov=args.uq_vp_cov,
            coef_cov=args.uq_coef_cov,
            n_samples=args.uq_samples,
            method='lhs',
            seed=_UQ_DEFAULT_SEED if args.seed is None else args.seed,
            **layup,
        )
        kd = results[mode]['knockdown']['percentiles']
        logger.info("  UQ %-18s knockdown p5/p50/p95 = %.3f / %.3f / %.3f",
                    mode, kd['p5'], kd['p50'], kd['p95'])
    save_uq_results_to_json(results, out_path)
    return results


class _DynamicStdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stdout`` on every emit.

    Plain ``StreamHandler(sys.stdout)`` caches the stream reference at
    construction time, which breaks pytest's ``capsys`` fixture (it
    rebinds ``sys.stdout`` per-test). Looking it up lazily keeps the
    formatting identical to the old bare-``print`` output while
    remaining capturable.
    """

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value):
        # logging.StreamHandler.__init__ assigns ``self.stream`` -- accept
        # and ignore so the dynamic property above wins.
        pass


def _configure_cli_logging(*, quiet: bool, verbose: bool) -> None:
    """Wire the module ``logger`` for CLI use.

    Routes progress through the module logger so ``--quiet`` actually
    silences the run (issue #78). Attaches a stdout stream handler the
    first time the CLI is invoked so the output looks like the old
    bare-``print`` style.

    Parameters
    ----------
    quiet : bool
        Suppress INFO/DEBUG; only warnings and errors are surfaced.
    verbose : bool
        Lower the threshold to DEBUG.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Don't propagate to the root logger -- pytest's caplog and library
    # consumers (the Streamlit app) configure their own handlers and we
    # don't want duplicated lines.
    logger.propagate = False
    logger.setLevel(level)

    # Reuse any handler we already attached; otherwise add a simple
    # stdout stream so the formatting matches the old print()-based UX.
    if not any(getattr(h, "_porosity_cli", False) for h in logger.handlers):
        handler = _DynamicStdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._porosity_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def _resolve_via_shim(name: str, fallback):
    """Look up ``name`` on the back-compat ``porosity_fe_analysis`` shim if
    it has been imported (so test ``monkeypatch.setattr`` on the shim is
    honoured), otherwise return ``fallback``.

    The shim re-exports the package's public surface, so attribute lookup
    on it usually returns the same object that lives in the package. Tests
    however historically patch the shim directly (``monkeypatch.setattr(
    porosity_fe_analysis, 'compare_configurations', spy)``) and expect the
    CLI to honour that patch. Going through the shim preserves that
    contract after the #119 split.
    """
    import sys as _sys
    shim = _sys.modules.get('porosity_fe_analysis')
    if shim is None:
        return fallback
    return getattr(shim, name, fallback)


def _write_vp_plots(viz, results: dict, artifacts: dict, output_dir: Path,
                    Vp_label: str) -> None:
    """Write the per-configuration and comparison PNGs for one porosity level."""
    for name in results:
        art = artifacts[name]
        viz.plot_porosity_field(
            art.porosity_field,
            save_path=output_dir / f"porosity_profile_{name}_{Vp_label}.png")
        viz.plot_mesh_3d(
            art.mesh,
            save_path=output_dir / f"porosity_mesh_3d_{name}_{Vp_label}.png")
        viz.plot_mesh_detail(
            art.mesh,
            save_path=output_dir / f"porosity_mesh_detail_{name}_{Vp_label}.png")
        viz.plot_damage_contour(
            art.mesh,
            art.empirical_solver,
            save_path=output_dir / f"porosity_damage_{name}_{Vp_label}.png",
            mode='compression', model='judd_wright')
    viz.plot_model_comparison(
        results,
        save_path=output_dir / f"porosity_comparison_{Vp_label}.png")


def main(argv: list[str] | None = None) -> int:
    """Argparse-driven entry point.

    Returns
    -------
    int
        ``0`` on success, ``2`` on bad input, ``3`` on a solver failure.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    materials = _resolve_via_shim('MATERIALS', MATERIALS)

    if args.list_materials:
        for name in sorted(materials):
            print(name)
        return 0

    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose are mutually exclusive.")

    _configure_cli_logging(quiet=args.quiet, verbose=args.verbose)

    if args.applied_stress is not None:
        logger.warning(
            "--applied-stress %g has no effect: the empirical sweep does not "
            "use an applied stress (IMPROVEMENT_PLAN 4.7).", args.applied_stress)
    else:
        args.applied_stress = _DEFAULT_APPLIED_STRESS

    custom_material = None
    if args.material_file is not None:
        if args.material is not None:
            parser.error("--material-file replaces --material: give one of them.")
        try:
            custom_material, args.material = _load_material_file(args.material_file)
        except (ValueError, OSError) as exc:
            parser.error(f"--material-file: {exc}")
    elif args.material is None:
        args.material = _DEFAULT_MATERIAL
    if custom_material is None and args.material not in materials:
        parser.error(
            f"Unknown material {args.material!r}. "
            f"Available presets: {sorted(materials)}."
        )

    # #127: resolve --vp / --vp-pct into a single list of fractions for the
    # rest of the pipeline. argparse already enforces mutual exclusion, so
    # at most one is non-None.
    if args.vp_pct is not None:
        args.vp = [pct / 100.0 for pct in args.vp_pct]
    elif args.vp is None:
        args.vp = list(DEFAULT_POROSITY_LEVELS)

    try:
        delamination = _delamination_options(args)
        ply_angles = _parse_layup_option(args.layup)
        material = _laminate_material(
            custom_material or materials[args.material], ply_angles, args.t_ply)
        durability = _durability_options(args, material)
    except ValueError as exc:
        parser.error(str(exc))
    # Only what was asked for is passed on, so a default run calls the sweep
    # exactly as before.
    laminate: dict = {}
    if custom_material is not None:
        laminate['material'] = custom_material
    if args.layup is not None:
        laminate['ply_angles'] = ply_angles
    if args.t_ply is not None:
        laminate['t_ply'] = args.t_ply
    if args.card_element is not None and args.material_cards is None:
        parser.error("--card-element needs --material-cards.")
    if args.required_knockdown is not None and not (
            0.0 < args.required_knockdown <= 1.0):  # also rejects NaN
        hint = (f" (a percent? pass {args.required_knockdown / 100.0:g})"
                if 1.0 < args.required_knockdown <= 100.0 else "")
        parser.error(
            f"--required-knockdown {args.required_knockdown!r} must be the "
            f"fraction of the pristine strength to retain, in (0, 1]{hint}.")

    for Vp in args.vp:
        if not (0.0 <= Vp <= 1.0) or Vp != Vp:  # NaN-safe range check
            hint = ""
            if 1.0 < Vp <= 100.0:
                hint = (
                    f" (looks like a percentage; "
                    f"did you mean `--vp {Vp / 100.0:.4f}` or `--vp-pct {Vp}`?)"
                )
            parser.error(
                f"--vp value {Vp!r} is out of range; expected a finite "
                f"float in [0, 1] (a void *fraction*, not a percentage)."
                f"{hint}"
            )
    # Each level's files are named by its label (to 1e-4 %): two distinct
    # levels with one label would overwrite each other's output.
    labelled: dict[str, float] = {}
    for Vp in args.vp:
        label = _vp_label(Vp)
        other = labelled.setdefault(label, Vp)
        if other != Vp:
            parser.error(
                f"--vp values {other!r} and {Vp!r} would write the same output "
                f"files (label {label!r}); levels must differ by at least "
                f"1e-6 as a fraction (1e-4 %).")

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {str(output_dir)!r}: {exc}",
              file=sys.stderr)
        return 2

    # Resolve via the shim so tests that monkeypatch
    # ``porosity_fe_analysis.compare_configurations`` (and POROSITY_CONFIGS)
    # see their patch honoured at call time (#119).
    cmp_fn = _resolve_via_shim('compare_configurations', compare_configurations)
    save_fn = _resolve_via_shim('save_results_to_json', save_results_to_json)
    viz = _resolve_via_shim('FEVisualizer', FEVisualizer)
    porosity_configs = _resolve_via_shim('POROSITY_CONFIGS', POROSITY_CONFIGS)
    if args.configs is not None:
        unknown = [name for name in args.configs if name not in porosity_configs]
        if unknown:
            parser.error(
                f"Unknown --configs {', '.join(map(repr, unknown))}; use any of "
                f"{', '.join(porosity_configs)}.")
        porosity_configs = {name: porosity_configs[name]
                            for name in dict.fromkeys(args.configs)}

    # With several workers and several porosity levels, send every
    # (Vp, config) pair to one process pool (IMPROVEMENT_PLAN 4.6) instead
    # of draining a fresh pool per level.
    precomputed: dict = {}
    if _resolve_n_jobs(args.jobs) > 1 and len(set(args.vp)) > 1:
        sweep_fn = _resolve_via_shim('sweep_configurations', sweep_configurations)
        try:
            precomputed = sweep_fn(
                args.vp,
                material_name=args.material,
                applied_stress=args.applied_stress,
                configs=porosity_configs,
                seed=args.seed,
                n_jobs=args.jobs,
                return_artifacts=True,
                **laminate,
            )
        except ValueError as exc:
            print(f"ERROR: bad input: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface as solver failure
            print(f"ERROR: solver failure: {exc}", file=sys.stderr)
            return 3

    all_results = {}
    for Vp in args.vp:
        Vp_label = _vp_label(Vp)
        if precomputed:
            results, artifacts = precomputed[float(Vp)]
        else:
            try:
                # ``return_artifacts=True`` because the --plots path needs the
                # live mesh / empirical_solver / porosity_field objects for
                # the FEVisualizer calls below (#44 item 3 migration).
                results, artifacts = cmp_fn(
                    Vp,
                    material_name=args.material,
                    applied_stress=args.applied_stress,
                    configs=porosity_configs,
                    seed=args.seed,
                    n_jobs=args.jobs,
                    return_artifacts=True,
                    **laminate,
                )
            except ValueError as exc:
                print(f"ERROR: bad input for Vp={Vp}: {exc}", file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001 - surface as solver failure
                print(f"ERROR: solver failure for Vp={Vp}: {exc}", file=sys.stderr)
                return 3
        all_results[Vp_label] = results

        # Writing outputs keeps the 0/2/3 contract: an unwritable file is
        # an input/environment problem (2), anything else a failure (3).
        try:
            if args.plots:
                _write_vp_plots(viz, results, artifacts, output_dir, Vp_label)
            out_path = output_dir / f"porosity_analysis_results_{Vp_label}.json"
            save_fn(results, out_path, artifacts=artifacts)
        except OSError as exc:
            print(f"ERROR: cannot write output for Vp={Vp}: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            print(f"ERROR: writing output for Vp={Vp} failed: {exc}", file=sys.stderr)
            return 3

    if args.uq:
        if args.uq_samples <= 0:
            print("ERROR: --uq-samples must be a positive integer.", file=sys.stderr)
            return 2
        for Vp in args.vp:
            Vp_label = _vp_label(Vp)
            logger.info("\nUncertainty propagation: Vp = %.2f%%", Vp * 100)
            try:
                if laminate:
                    # The laminate's material; the default layup is 'QI'.
                    _run_uq(args, material, Vp,
                            output_dir / f"porosity_uq_{Vp_label}.json",
                            ply_angles=ply_angles)
                else:
                    _run_uq(args, args.material, Vp,
                            output_dir / f"porosity_uq_{Vp_label}.json")
            except (ValueError, OSError) as exc:
                print(f"ERROR: uncertainty propagation for Vp={Vp}: {exc}",
                      file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001 - surface as a failure
                print(f"ERROR: uncertainty propagation for Vp={Vp} failed: {exc}",
                      file=sys.stderr)
                return 3

    if delamination:
        for Vp in args.vp:
            Vp_label = _vp_label(Vp)
            logger.info("\nDelamination screening (FE): Vp = %.2f%%", Vp * 100)
            try:
                line = _run_delamination(
                    delamination, material, porosity_configs,
                    Vp, output_dir / f"porosity_delamination_{Vp_label}.json",
                    quiet_solver=not args.verbose, ply_angles=ply_angles)
                logger.info("%s", line)
            except (ValueError, OSError) as exc:
                print(f"ERROR: delamination screening for Vp={Vp}: {exc}",
                      file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001 - surface as a failure
                print(f"ERROR: delamination screening for Vp={Vp} failed: {exc}",
                      file=sys.stderr)
                return 3

    if args.fe is not None:
        degradation = args.porosity_degradation or _DELAM_DEFAULT_DEGRADATION
        for Vp in args.vp:
            Vp_label = _vp_label(Vp)
            logger.info("\nFE solve (%s): Vp = %.2f%%", args.fe, Vp * 100)
            for name, config in porosity_configs.items():
                out_path = output_dir / f"porosity_fe_{args.fe}_{name}_{Vp_label}.json"
                try:
                    logger.info("%s", _run_fe(
                        args.fe, material, ply_angles, name, config, Vp, out_path,
                        porosity_degradation=degradation,
                        quiet_solver=not args.verbose))
                except (ValueError, OSError) as exc:
                    print(f"ERROR: FE solve for Vp={Vp}, {name}: {exc}",
                          file=sys.stderr)
                    return 2
                except Exception as exc:  # noqa: BLE001 - surface as a failure
                    print(f"ERROR: FE solve for Vp={Vp}, {name} failed: {exc}",
                          file=sys.stderr)
                    return 3

    if args.material_cards is not None:
        degradation = args.porosity_degradation or _DELAM_DEFAULT_DEGRADATION
        element = args.card_element or _CARD_ELEMENTS[0]
        suffix = _CARD_SUFFIX[args.material_cards]
        for Vp in args.vp:
            Vp_label = _vp_label(Vp)
            logger.info("\nMaterial cards (%s, %s elements): Vp = %.2f%%",
                        args.material_cards, element, Vp * 100)
            for name, config in porosity_configs.items():
                out_path = output_dir / f"porosity_cards_{name}_{Vp_label}{suffix}"
                try:
                    logger.info("%s", _run_material_cards(
                        args.material_cards, material, ply_angles, name, config,
                        Vp, out_path, element=element,
                        porosity_degradation=degradation))
                except (ValueError, OSError) as exc:
                    print(f"ERROR: material cards for Vp={Vp}, {name}: {exc}",
                          file=sys.stderr)
                    return 2
                except Exception as exc:  # noqa: BLE001 - surface as a failure
                    print(f"ERROR: material cards for Vp={Vp}, {name} failed: "
                          f"{exc}", file=sys.stderr)
                    return 3

    if durability:
        for Vp in args.vp:
            Vp_label = _vp_label(Vp)
            logger.info("\nFatigue and environment (empirical): Vp = %.2f%%", Vp * 100)
            try:
                for line in _run_durability(
                        durability, material, args.material, ply_angles, Vp,
                        output_dir / f"porosity_durability_{Vp_label}.json"):
                    logger.info("%s", line)
            except (ValueError, OSError) as exc:
                print(f"ERROR: fatigue and environment for Vp={Vp}: {exc}",
                      file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001 - surface as a failure
                print(f"ERROR: fatigue and environment for Vp={Vp} failed: {exc}",
                      file=sys.stderr)
                return 3

    if args.required_knockdown is not None:
        logger.info("\nAllowable porosity: %g%% of pristine strength retained",
                    args.required_knockdown * 100)
        try:
            for line in _run_allowable(
                    args.required_knockdown, material, args.material, ply_angles,
                    output_dir / "porosity_allowable.json",
                    **({'durability': durability} if durability else {})):
                logger.info("%s", line)
        except (ValueError, OSError) as exc:
            print(f"ERROR: allowable porosity: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            print(f"ERROR: allowable porosity failed: {exc}", file=sys.stderr)
            return 3

    if args.plots and all_results:
        try:
            viz.plot_knockdown_curves(
                all_results,
                save_path=output_dir / "porosity_knockdown_curves.png")
        except OSError as exc:
            print(f"ERROR: cannot write knockdown curves: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            print(f"ERROR: plotting knockdown curves failed: {exc}", file=sys.stderr)
            return 3

    _bar = "=" * 70
    logger.info("\n%s", _bar)
    logger.info("COMPLETE ANALYSIS FINISHED")
    logger.info("%s", _bar)
    logger.info("Material: %s", args.material)
    if laminate:
        logger.info("Laminate: %d plies of %g mm, layup %s", material.n_plies,
                    material.t_ply, args.layup or 'QI')
    logger.info("Porosity levels analyzed: %s",
                [f"{v*100:.2f}%" for v in args.vp])
    logger.info("Configurations: %s", list(porosity_configs.keys()))
    logger.info("Output directory: %s", output_dir.resolve())
    return 0
