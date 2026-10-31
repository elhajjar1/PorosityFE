"""argparse-driven CLI (``porosity-analyze``)."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
from pathlib import Path

from . import __version__
from .delamination import Delamination
from .io import save_results_to_json
from .materials import MATERIALS
from .pipeline import _resolve_n_jobs, compare_configurations, sweep_configurations
from .porosity_field import POROSITY_CONFIGS
from .viz import FEVisualizer

logger = logging.getLogger("porosity_fe_analysis")

# Default Vp sweep — preserves the historical hardcoded behavior so that
# `porosity-analyze` with no arguments reproduces today's analysis range.
DEFAULT_POROSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08]


#: Placeholder passed through to compare_configurations, which ignores it.
_DEFAULT_APPLIED_STRESS = -1500.0

#: Defaults of the delamination screening options (``--delamination``),
#: applied only when it is given, so the options can tell "not given".
_DELAM_LOADINGS = ('compression', 'tension', 'shear')
_DELAM_DEFAULT_LOADING = 'compression'
_DELAM_DEFAULT_CONFIG = 'uniform_spherical'
_DELAM_DEFAULT_LAW = 'exponential'


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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse driver for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        prog="porosity-analyze",
        description=(
            "Run the porosity-degraded composite laminate analysis over one "
            "or more void volume fractions and write JSON results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--material",
        default="T800_epoxy",
        help="Material preset name (validated against the built-in presets).",
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
            "Recorded in JSON provenance for reproducibility. The pipeline "
            "is deterministic (RNG-free), so this does not alter results."
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
        "Opt-in: an FE solve of the coupon (production mesh, 'QI' layup) "
        "with an explicit delamination at each porosity level, written to "
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
              "optionally the mode III toughness (default G_IIC). No preset "
              "has them; without them the screening reports energy release "
              "rates only, no fracture load factor."))
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
    }


def _run_delamination(options: dict, material, porosity_configs: dict,
                      Vp: float, out_path: Path, *, quiet_solver: bool = True) -> str:
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
    try:
        return _delamination_screening(options, material, porosity_configs,
                                       Vp, out_path)
    finally:
        logger.setLevel(level)


def _delamination_screening(options: dict, material, porosity_configs: dict,
                            Vp: float, out_path: Path) -> str:
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
    mesh = CompositeMesh(pf, material, *_default_mesh_resolution(material.n_plies),
                         delaminations=[options['delamination']])
    loading = options['loading']
    result = FESolver(mesh, material, pf).solve(
        loading, toughness_knockdown=options['toughness_knockdown'])
    parts = knockdown_decomposition(mesh, material, pf, loading, total=result)
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
            + f"\n  Saved: {out_path}")


def _run_uq(args, material, Vp: float, out_path: Path) -> dict:
    """Propagate uncertainty for every loading mode at one porosity level."""
    from .empirical import EmpiricalSolver
    from .uq import propagate_uncertainty, save_uq_results_to_json

    results = {}
    for mode, strength_field in EmpiricalSolver.PRISTINE_STRENGTH_KEY.items():
        results[mode] = propagate_uncertainty(
            Vp, material, mode, 'judd_wright',
            covs={strength_field: args.uq_strength_cov},
            vp_cov=args.uq_vp_cov,
            coef_cov=args.uq_coef_cov,
            n_samples=args.uq_samples,
            method='lhs',
            seed=args.seed,
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
            save_path=output_dir / f"porosity_damage_{name}_{Vp_label}.png")
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

    if args.material not in materials:
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
    except ValueError as exc:
        parser.error(str(exc))

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

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {str(output_dir)!r}: {exc}",
              file=sys.stderr)
        return 2

    # Back-compat shim: a few callers may still reference _log. Now that
    # progress is routed through ``logger``, this just forwards to
    # logger.info so --quiet (WARNING level) silences these too.
    def _log(msg: str) -> None:
        logger.info("%s", msg)

    # Resolve via the shim so tests that monkeypatch
    # ``porosity_fe_analysis.compare_configurations`` (and POROSITY_CONFIGS)
    # see their patch honoured at call time (#119).
    cmp_fn = _resolve_via_shim('compare_configurations', compare_configurations)
    save_fn = _resolve_via_shim('save_results_to_json', save_results_to_json)
    viz = _resolve_via_shim('FEVisualizer', FEVisualizer)
    porosity_configs = _resolve_via_shim('POROSITY_CONFIGS', POROSITY_CONFIGS)

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
                    delamination, materials[args.material], porosity_configs,
                    Vp, output_dir / f"porosity_delamination_{Vp_label}.json",
                    quiet_solver=not args.verbose)
                logger.info("%s", line)
            except (ValueError, OSError) as exc:
                print(f"ERROR: delamination screening for Vp={Vp}: {exc}",
                      file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001 - surface as a failure
                print(f"ERROR: delamination screening for Vp={Vp} failed: {exc}",
                      file=sys.stderr)
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
    logger.info("Porosity levels analyzed: %s",
                [f"{v*100:.2f}%" for v in args.vp])
    logger.info("Configurations: %s", list(porosity_configs.keys()))
    logger.info("Output directory: %s", output_dir.resolve())
    return 0
