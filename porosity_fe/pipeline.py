"""Sweep orchestrator: ``_analyze_one`` worker + ``compare_configurations``."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Any, Union

from .empirical import EmpiricalSolver
from .materials import MATERIALS, MaterialProperties
from .mesh import CompositeMesh
from .porosity_field import POROSITY_CONFIGS, PorosityField
from .results import ConfigArtifacts, ConfigResult

logger = logging.getLogger("porosity_fe_analysis")

# Layup descriptor accepted by both ``CompositeMesh`` and ``EmpiricalSolver``:
# either a sentinel string (``'QI'`` / ``'UD'``) or an explicit list of ply
# angles in degrees. ``tuple`` is included for callers that build the layup
# from an immutable sequence.
LayupSpec = Union[str, list[float], tuple[float, ...]]

# Default mesh resolution used by the production sweep (``_analyze_one``) so
# every call site picks up the same defaults.
_DEFAULT_MESH_RES: tuple[int, int, int] = (30, 10, 12)


def build_empirical_pipeline(
    material: MaterialProperties,
    void_volume_fraction: float,
    *,
    ply_angles: LayupSpec = 'QI',
    mesh_res: tuple[int, int, int] = _DEFAULT_MESH_RES,
    porosity_config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[PorosityField, CompositeMesh, EmpiricalSolver]:
    """Factory: ``(material, Vp) -> (PorosityField, CompositeMesh, EmpiricalSolver)``.

    Single point of change for mesh defaults and ply-angle handling. The
    three-step construction recurs in :func:`_analyze_one`, the UQ helper
    in :mod:`porosity_fe.uq`, every script in ``examples/``, and several
    tests; channeling them through this factory means future tweaks to the
    mesh defaults or ply-angle handling happen exactly once. See issue
    #120.

    Parameters
    ----------
    material : MaterialProperties
        Pristine composite material.
    void_volume_fraction : float
        Specimen-average void volume fraction in ``[0, 1]``.
    ply_angles : str or list of float, optional
        Layup descriptor passed to both :class:`CompositeMesh` and
        :class:`EmpiricalSolver`. Defaults to ``'QI'``.
    mesh_res : (int, int, int), optional
        ``(nx, ny, nz)`` element counts. Defaults to the production sweep
        resolution ``(30, 10, 12)``.
    porosity_config : dict, optional
        Extra keyword arguments forwarded to :class:`PorosityField` (e.g.
        ``distribution``, ``void_shape``, ``cluster_location``,
        ``discrete_voids``). ``seed`` may be set here or via the dedicated
        ``seed`` parameter — the explicit ``seed=`` argument wins.
    seed : int, optional
        Recorded into the porosity field for reproducibility provenance.
        Overrides any ``seed`` key in ``porosity_config``.

    Returns
    -------
    (PorosityField, CompositeMesh, EmpiricalSolver)
        The fully-constructed pipeline ready for ``get_failure_load`` /
        ``get_all_failure_loads`` calls.
    """
    pf_kwargs: dict[str, Any] = dict(porosity_config or {})
    if seed is not None:
        pf_kwargs['seed'] = seed
    pf = PorosityField(material, void_volume_fraction, **pf_kwargs)
    nx, ny, nz = mesh_res
    mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz,
                         ply_angles=ply_angles)
    emp = EmpiricalSolver(mesh, material, ply_angles=ply_angles)
    return pf, mesh, emp


# ============================================================
# SECTION 8: ANALYSIS PIPELINE
# ============================================================

def _analyze_one(Vp: float,
                 name: str,
                 config: dict,
                 material_name: str,
                 applied_stress: float,
                 seed: int | None = None) -> tuple[float, str, dict]:
    """Build PorosityField/CompositeMesh/EmpiricalSolver for one (Vp, config).

    Top-level (picklable) helper so this can be dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` from
    :func:`compare_configurations` for a ~Nx speedup on the 5 x 5 sweep
    (#52). Each call is fully independent — no shared mutable state — so
    the result of the parallel execution is order-invariant.

    The returned dict carries both the live solver objects (mesh /
    empirical_solver / porosity_field) and the headline empirical
    knockdown table; the public-facing :func:`compare_configurations`
    splits this into :class:`ConfigResult` / :class:`ConfigArtifacts`
    (#44 item 3). Keeping the worker dict intact preserves the
    parallel-sweep pickle contract (#52).

    Parameters
    ----------
    Vp : float
        Void volume fraction in [0, 1].
    name : str
        Configuration name (key in ``POROSITY_CONFIGS``).
    config : dict
        Porosity-field constructor kwargs.
    material_name : str
        Material preset name. Resolved inside the worker so the parent
        process doesn't need to pickle the :class:`MaterialProperties`
        dataclass across the boundary (it's keyed by name anyway).
    applied_stress : float
        Reserved for downstream solver hooks; **currently unused**.
        Accepted for parity with the ``compare_configurations`` signature
        even though the empirical knockdown does not consume it, so
        changing this value will not affect the worker's output (#132).
    seed : int, optional
        Recorded into the porosity field for reproducibility provenance.

    Returns
    -------
    (Vp, name, result_dict)
        Tuple keyed on ``(Vp, name)`` so the caller can deterministically
        re-assemble results even when the worker pool reorders completion.
    """
    material = MATERIALS[material_name]
    porosity_field, mesh, empirical = build_empirical_pipeline(
        material, Vp, porosity_config=config, seed=seed,
    )
    emp_results = empirical.get_all_failure_loads()

    result = {
        'config': config,
        'mesh': mesh,
        'porosity_field': porosity_field,
        'empirical_solver': empirical,
        'empirical': emp_results,
    }
    return (Vp, name, result)


def _build_config_result(name: str, Vp: float, raw: dict) -> ConfigResult:
    """Distill the worker-dict shape into a lightweight :class:`ConfigResult`.

    Reads the headline compression / Judd-Wright knockdown from the inner
    ``empirical`` table so the convenience scalars on the result match
    what the existing rankings code prints (#44 item 3). The nested
    ``empirical`` dict is carried verbatim so existing callers (JSON
    exporter, plot helpers, tests reading
    ``cfg['empirical']['compression'][model]['knockdown']``) keep working.
    """
    emp = raw['empirical']
    headline = emp['compression']['judd_wright']
    # Carry the seed off the PorosityField so the JSON exporter's
    # provenance block can recover it without holding the live field.
    pf = raw.get('porosity_field')
    seed_val = getattr(pf, 'seed', None) if pf is not None else None
    # Headline is now a FailureResult; the dict-style shim keeps the
    # legacy ``['failure_stress']`` access working too.
    return ConfigResult(
        Vp=float(Vp),
        config_name=str(name),
        config=raw['config'],
        failure_stress=float(headline['failure_stress']),
        knockdown=float(headline['knockdown']),
        model=str(headline['model']),
        empirical=emp,
        seed=seed_val,
    )


def _build_config_artifacts(raw: dict) -> ConfigArtifacts:
    """Bundle the live worker objects into a :class:`ConfigArtifacts`."""
    return ConfigArtifacts(
        mesh=raw['mesh'],
        empirical_solver=raw['empirical_solver'],
        porosity_field=raw['porosity_field'],
        field_results=raw.get('field_results'),
    )


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalise ``n_jobs`` to a positive worker count.

    ``None``/``0``/``-1`` map to ``os.cpu_count() or 1`` so callers can
    request "all cores" without having to look up the count themselves.
    ``1`` preserves the serial path for reproducibility / debugging.
    """
    if n_jobs is None or n_jobs <= 0:
        return os.cpu_count() or 1
    return int(n_jobs)


def _log_result_summary(name: str, result: dict, *, is_parallel: bool) -> None:
    """Emit the per-config result-summary log lines.

    Both branches of :func:`compare_configurations` (serial + parallel)
    surface the same headline numbers — Judd-Wright compression /ILSS
    knockdowns and the closed-form local sensitivities (#65). Keeping the
    formatting in one place means future log-format tweaks happen once
    (#122).

    Parameters
    ----------
    name : str
        Configuration name (used in the message text).
    result : dict
        The raw worker dict returned by :func:`_analyze_one`. Contains the
        nested ``empirical`` table and the live ``empirical_solver``.
    is_parallel : bool
        ``True`` formats a single-line "Configuration X done — ..." line
        suitable for interleaved parallel completions. ``False`` uses the
        indented multi-line serial UX. The tornado line is always emitted
        last when sensitivities are available.
    """
    comp_kd = result['empirical']['compression']['judd_wright']['knockdown']
    ilss_kd = result['empirical']['ilss']['judd_wright']['knockdown']
    if is_parallel:
        logger.info("  Configuration %s done — "
                    "compression KD (J-W) %.3f, ILSS KD (J-W) %.3f",
                    name, comp_kd, ilss_kd)
    else:
        logger.info("    Compression KD (J-W): %.3f", comp_kd)
        logger.info("    ILSS KD (J-W):        %.3f", ilss_kd)

    # Issue #65: surface the closed-form local sensitivities at the same
    # Vp_mean used for the headline KD. The empirical solver only lives
    # on the worker dict (post-#103 it moves to ConfigArtifacts), so when
    # the caller has stripped artifacts we degrade gracefully rather than
    # raise.
    solver = result.get('empirical_solver')
    if solver is None:
        logger.info(
            "    Tornado [%s]: (sensitivities unavailable: "
            "re-run with return_artifacts=True)", name)
        return
    s = solver.local_sensitivities(mode='compression', model='judd_wright')
    logger.info(
        "    Tornado [%s]: dKD/dVp=%.3g, dKD/dcoef=%.3g",
        name, s['dKD_dVp'], s['dKD_dcoef'])


def compare_configurations(void_volume_fraction: float,
                           material_name: str = 'T800_epoxy',
                           applied_stress: float = -1500.0,
                           configs: dict | None = None,
                           seed: int | None = None,
                           n_jobs: int = 1,
                           return_artifacts: bool = False):
    """Main analysis function — loops through porosity configurations.

    Parameters
    ----------
    void_volume_fraction : float
        Specimen-average void volume fraction in [0, 1].
    material_name : str
        Material preset name; validated against :data:`MATERIALS`.
    applied_stress : float
        Reserved for downstream solver hooks; **currently unused**. The
        empirical knockdown sweep does not consume this value, so passing
        a different number will not change the returned results. Retained
        in the signature for forward compatibility (#132).
    configs : dict, optional
        Mapping of configuration name -> :class:`PorosityField` kwargs.
        Defaults to the bundled :data:`POROSITY_CONFIGS`.
    seed : int, optional
        Recorded into provenance and threaded into each
        :class:`PorosityField` for reproducibility (#55). The pipeline is
        deterministic, so this does not alter results today.
    n_jobs : int, optional
        Number of worker processes to use for the per-configuration sweep
        (#52). ``1`` (default) runs serially — bit-for-bit identical to
        the legacy behaviour, useful for tests / debugging. ``N > 1``
        dispatches the (Vp, config) calls to a
        :class:`concurrent.futures.ProcessPoolExecutor` of that size.
        ``0`` / ``-1`` / ``None`` resolve to :func:`os.cpu_count`. Results
        are deterministically re-assembled by ``(Vp, name)`` regardless
        of completion order, so the returned dict is independent of ``N``.
    return_artifacts : bool, optional
        If ``False`` (default), returns ``Dict[str, ConfigResult]`` —
        numbers only, JSON-friendly, safe to retain in long batch loops
        (#44 item 3). If ``True``, returns a tuple
        ``(Dict[str, ConfigResult], Dict[str, ConfigArtifacts])`` so
        callers that need the live ``mesh`` / ``empirical_solver`` /
        ``porosity_field`` objects (plot helpers, the GUI, the
        ``--plots`` CLI path) can still get them. Existing callers that
        accessed ``results[name]['mesh']`` need to switch to the
        artifacts dict; the legacy keys now raise :class:`KeyError` with
        a hint pointing to ``return_artifacts=True``.

    Returns
    -------
    Dict[str, ConfigResult]
        When ``return_artifacts=False`` (default).
    Tuple[Dict[str, ConfigResult], Dict[str, ConfigArtifacts]]
        When ``return_artifacts=True``.
    """
    if material_name not in MATERIALS:
        raise ValueError(
            f"Unknown material {material_name!r}. "
            f"Available presets: {sorted(MATERIALS)}."
        )
    configs = configs or POROSITY_CONFIGS
    workers = _resolve_n_jobs(n_jobs)

    _bar = '=' * 70
    logger.info("\n%s", _bar)
    logger.info("POROSITY ANALYSIS: Vp = %.1f%%", void_volume_fraction * 100)
    logger.info("Material: %s", material_name)
    logger.info("%s", _bar)

    # Build the (Vp, name, config, ...) task list once. We always iterate
    # the original ``configs`` dict so the assembled output preserves the
    # caller's configuration ordering (Python dicts are insertion-ordered)
    # regardless of which worker finishes first.
    tasks = [
        (void_volume_fraction, name, config, material_name, applied_stress, seed)
        for name, config in configs.items()
    ]

    raw_results: dict[tuple[float, str], dict] = {}
    if workers == 1 or len(tasks) <= 1:
        # Serial path — preserves the legacy behaviour byte-for-byte and
        # avoids the ProcessPoolExecutor fork cost for trivially small
        # sweeps. The per-config "Configuration: ..." log lines fire here
        # too, mirroring the original CLI UX.
        for Vp, name, config, mat, stress, sd in tasks:
            logger.info("\n  Configuration: %s", name)
            Vp_out, name_out, result = _analyze_one(
                Vp, name, config, mat, stress, sd)
            raw_results[(Vp_out, name_out)] = result
            _log_result_summary(name_out, result, is_parallel=False)
    else:
        logger.info("Parallel sweep: %d task(s) across %d worker process(es)",
                    len(tasks), workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_analyze_one, *task) for task in tasks]
            for fut in concurrent.futures.as_completed(futures):
                Vp_out, name_out, result = fut.result()
                raw_results[(Vp_out, name_out)] = result
                _log_result_summary(name_out, result, is_parallel=True)

    # Re-assemble in the original config insertion order so callers see a
    # deterministic dict regardless of which worker finished first.
    # Split the worker dict into the public-facing lightweight
    # ConfigResult (numbers + nested empirical table) and the parallel
    # ConfigArtifacts (live mesh / solver / field), per #44 item 3.
    results: dict[str, ConfigResult] = {}
    artifacts: dict[str, ConfigArtifacts] = {}
    for name in configs:
        raw = raw_results[(void_volume_fraction, name)]
        results[name] = _build_config_result(name, void_volume_fraction, raw)
        artifacts[name] = _build_config_artifacts(raw)

    logger.info("\n%s", _bar)
    logger.info("RANKINGS (by compression strength, Judd-Wright)")
    logger.info("%s", _bar)
    ranked = sorted(
        results.keys(),
        key=lambda c: results[c].failure_stress,
        reverse=True,
    )
    for i, name in enumerate(ranked, 1):
        logger.info("  %d. %s: %.1f MPa", i, name, results[name].failure_stress)

    if return_artifacts:
        return results, artifacts
    return results
