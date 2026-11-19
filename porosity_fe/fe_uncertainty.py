"""FE uncertainty over random porosity fields: a Monte Carlo of FE solves.

:func:`fe_porosity_uncertainty` solves the FE coupon once per draw. Each
draw has its own seeded random in-plane porosity map
(:meth:`PorosityField.random <porosity_fe.PorosityField.random>`) and,
optionally, its own specimen-average void content and material constants.
The result has the distribution of the FE knockdown, effective modulus and
first-ply-failure load factor, and ranks which inputs drive them
(:func:`~porosity_fe.sensitivity_ranking`). The empirical
:func:`~porosity_fe.propagate_uncertainty` sees only the specimen average;
here the arrangement of the voids counts. At a fixed average a stiffness
barely moves when the voids are rearranged, but a strength, which a weakest
point sets, does.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._sampling import _UQ_METHODS, _draw_unit_samples, _unit_to_draw
from ._vp import check_fraction
from .delamination import Delamination
from .empirical import Calibration
from .fe import FESolver
from .io import _UNITS_FE_UQ, FORMAT_FE_UQ, _wrap_envelope, _write_json
from .materials import MATERIALS, MaterialProperties
from .mesh import CompositeMesh
from .open_hole import OPEN_HOLE_LOADINGS, OpenHoleMesh, open_hole_analysis
from .pipeline import _default_mesh_resolution
from .porosity_field import PorosityField
from .porosity_map import RANDOM_FIELD_KERNELS, PorosityMap
from .sensitivity import SensitivityRanking, sensitivity_ranking
from .uq import _material_label, _normalize_uq_spec

logger = logging.getLogger("porosity_fe_analysis")

#: Load cases of the FE uncertainty study: the coupon's in-plane ones.
FE_UQ_LOADINGS: tuple[str, ...] = ('compression', 'tension', 'shear')

#: The outputs a draw records, when the nominal solve has them
#: (``fracture_load_factor`` needs a delamination and toughness data,
#: ``stress_concentration`` an open hole).
FE_UQ_OUTPUTS: tuple[str, ...] = (
    'knockdown', 'effective_modulus', 'first_ply_failure_load_factor',
    'interior_first_ply_failure_load_factor', 'max_failure_index',
    'fracture_load_factor', 'stress_concentration')

#: The default output of :meth:`FEUncertainty.ranking`.
_DEFAULT_RANKED = 'first_ply_failure_load_factor'


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


@dataclass(frozen=True, eq=False)
class FEUncertainty:
    """Draws of an FE uncertainty study over random porosity fields.

    Attributes
    ----------
    loading : str
        The load case.
    material : str
        The material's preset name, or a label of a custom one.
    Vp : float
        The nominal specimen-average void content (fraction).
    n_samples : int
        Draws.
    seed : int or None
        The seed of the study: with the same seed the draws repeat.
    method : str
        ``'monte_carlo'`` or ``'lhs'`` (the scalar inputs; the maps are
        always independent draws).
    field : dict
        The random field: ``cov``, ``correlation_length_mm``, ``kernel``,
        ``resolution_mm``, ``preserve_mean`` and the plate it covers,
        ``plate_mm`` (``[length, width]``).
    vp_cov : float
        CoV of the specimen-average void content (0: fixed at :attr:`Vp`).
    spec : dict
        The scattered material fields, ``{field: [distribution, param]}``.
    percentiles : tuple of float
        The percentiles the statistics report.
    nominal : dict
        Each output of the solve with the nominal, uniform in-plane field
        (no random map), ``None`` where the solve has none.
    samples : dict of str to numpy.ndarray
        Each output (:data:`FE_UQ_OUTPUTS` that the nominal solve has) of
        every draw; a load factor is ``inf`` where the load stresses nothing.
    inputs : dict of str to numpy.ndarray
        The inputs of every draw: ``Vp`` (its specimen average), ``Vp_peak``
        (the largest value of its map), ``Vp_spatial_cov`` (the CoV of its
        map's grid values) and each scattered material field.
    mesh : tuple of int
        ``(nx, ny, nz)`` of every solve (with an open hole, the element
        counts along the coupon's long and short edges, and ``nz``).
    open_hole : dict or None
        The open-hole coupon of every solve (``hole_diameter_mm``,
        ``width_mm``, ``length_mm``), ``None`` for the plain coupon.
    """

    loading: str
    material: str
    Vp: float
    n_samples: int
    seed: int | None
    method: str
    field: dict[str, Any]
    vp_cov: float
    spec: dict[str, list]
    percentiles: tuple[float, ...]
    nominal: dict[str, float | None]
    samples: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray]
    mesh: tuple[int, int, int] = (0, 0, 0)
    open_hole: dict[str, float] | None = None

    @property
    def outputs(self) -> tuple[str, ...]:
        """The recorded outputs, in the order of :data:`FE_UQ_OUTPUTS`."""
        return tuple(self.samples)

    def _values(self, output: str) -> np.ndarray:
        if output not in self.samples:
            raise ValueError(
                f"Unknown output {output!r}; this study recorded {list(self.samples)}.")
        return self.samples[output]

    def statistics(self, output: str) -> dict[str, Any]:
        """Mean, standard deviation, extremes and percentiles of ``output``
        over its finite draws (``n_finite`` of them); ``None`` statistics
        when none is finite."""
        values = _finite(self._values(output))
        if values.size == 0:
            return {'n_finite': 0, 'mean': None, 'std': None, 'min': None, 'max': None,
                    'percentiles': {f'p{p:g}': None for p in self.percentiles}}
        return {
            'n_finite': int(values.size),
            'mean': float(values.mean()), 'std': float(values.std()),
            'min': float(values.min()), 'max': float(values.max()),
            'percentiles': {f'p{p:g}': float(np.percentile(values, p))
                            for p in self.percentiles},
        }

    def probability_below(self, output: str, value: float) -> float:
        """Share of the draws whose ``output`` is below ``value`` (an
        infinite load factor counts as above): an estimate under the
        entered scatter, not a disposition."""
        values = self._values(output)
        return float(np.mean(values < float(value)))

    def ranking(self, output: str = _DEFAULT_RANKED) -> SensitivityRanking:
        """Which inputs drive ``output`` (:func:`~porosity_fe.sensitivity_ranking`
        over :attr:`inputs`).

        Raises
        ------
        ValueError
            For an output the study did not record, or one the ranking
            cannot rank (see :func:`~porosity_fe.sensitivity_ranking`).
        """
        return sensitivity_ranking(self.inputs, self._values(output), name=output)

    def example_map(self, seed: int | None = 0) -> PorosityMap:
        """A map of the study's kind (its field, plate and average), drawn
        from ``seed``: an illustration, not one of the draws."""
        f = self.field
        return PorosityMap.random(
            f['plate_mm'][0], f['plate_mm'][1], self.Vp, cov=f['cov'],
            correlation_length=f['correlation_length_mm'], kernel=f['kernel'],
            resolution=f['resolution_mm'], seed=seed, preserve_mean=f['preserve_mean'])

    def summary(self) -> dict[str, Any]:
        """The study as a JSON-ready dict: the settings, the nominal values,
        the statistics of every output and, where it can rank it, the
        ranking of its inputs (``None`` otherwise)."""
        def finite_or_none(v: float | None) -> float | None:
            return None if v is None or not math.isfinite(v) else float(v)

        rankings: dict[str, Any] = {}
        for output in self.samples:
            try:
                rankings[output] = self.ranking(output).summary()
            except ValueError:
                rankings[output] = None
        return {
            'loading': self.loading,
            'material': self.material,
            'Vp': self.Vp,
            'n_samples': self.n_samples,
            'seed': self.seed,
            'method': self.method,
            'random_field': dict(self.field),
            'vp_cov': self.vp_cov,
            'spec': {k: list(v) for k, v in self.spec.items()},
            'mesh': list(self.mesh),
            'open_hole': None if self.open_hole is None else dict(self.open_hole),
            'percentiles': list(self.percentiles),
            'nominal': {k: finite_or_none(v) for k, v in self.nominal.items()},
            'statistics': {k: self.statistics(k) for k in self.samples},
            'sensitivity': rankings,
            'inputs': {k: {'mean': float(v.mean()), 'min': float(v.min()),
                           'max': float(v.max())} for k, v in self.inputs.items()},
        }


def _outputs(result: Any, mesh: CompositeMesh) -> dict[str, float | None]:
    """The :data:`FE_UQ_OUTPUTS` of one FE solve (``None`` where it has none)."""
    out: dict[str, float | None] = {}
    for name in FE_UQ_OUTPUTS:
        value = getattr(result, name, None)
        out[name] = None if value is None else float(value)
    if isinstance(mesh, OpenHoleMesh):
        out['stress_concentration'] = open_hole_analysis(result, mesh).stress_concentration
    return out


def fe_porosity_uncertainty(
    material: str | MaterialProperties, Vp: float, loading: str = 'compression', *,
    cov: float, correlation_length: float | Sequence[float], n_samples: int = 20,
    kernel: str = 'squared_exponential', resolution: float | None = None,
    preserve_mean: bool = True, vp_cov: float = 0.0,
    covs: dict[str, float] | None = None, spec: dict[str, tuple[str, float]] | None = None,
    seed: int | None = None, method: str = 'monte_carlo',
    percentiles: tuple[float, ...] = Calibration.UQ_DEFAULT_PERCENTILES,
    ply_angles: list[float] | str = 'QI', config: dict | None = None,
    mesh_res: tuple[int, int, int] | None = None, length: float | None = None,
    width: float | None = None, delaminations: Sequence[Delamination] | None = None,
    open_hole: Mapping[str, Any] | None = None,
    porosity_degradation: str = 'isotropic', **solve_kwargs: Any,
) -> FEUncertainty:
    """Monte Carlo of FE solves over random in-plane porosity fields.

    Every draw solves the coupon with its own seeded random map of void
    content (:meth:`PorosityField.random <porosity_fe.PorosityField.random>`,
    lognormal with point-to-point CoV ``cov`` and correlation length
    ``correlation_length``) and, when asked, its own specimen-average void
    content (``vp_cov``) and material constants (``covs`` / ``spec``, as
    :func:`~porosity_fe.propagate_uncertainty` takes them). The nominal
    solve has the same average with no in-plane variation.

    Parameters
    ----------
    material : str or MaterialProperties
        A preset name or a material.
    Vp : float
        Nominal specimen-average void content, a **fraction**.
    loading : {'compression', 'tension', 'shear'}
        The FE load case of the coupon (default ``'compression'``).
    cov : float
        Keyword-only, required. Point-to-point CoV of the void content (no
        default: it is the user's description of the part, from C-scans or
        CT, say).
    correlation_length : float or (float, float)
        Keyword-only, required. Correlation length of the field, in mm.
    n_samples : int
        Draws (default 20, at least 2): each is an FE solve.
    kernel, resolution, preserve_mean
        Of the random field (:meth:`PorosityMap.random
        <porosity_fe.PorosityMap.random>`): by default a smooth field on a
        grid of a quarter of the correlation length that keeps the average.
    vp_cov : float
        CoV of the specimen-average void content, lognormal about ``Vp``
        (default 0: fixed).
    covs, spec : dict, optional
        Scatter of material fields, as :func:`~porosity_fe.propagate_uncertainty`.
    seed : int, optional
        Seed of the study: the scalar draws and every map derive from it.
    method : {'monte_carlo', 'lhs'}
        Sampling of the scalar inputs (``vp_cov`` and the material fields).
    percentiles : tuple of float
        Percentiles of the statistics (default 5, 50, 95).
    ply_angles : list of float or str
        The layup (default ``'QI'``).
    config : dict, optional
        Other :class:`~porosity_fe.PorosityField` arguments: the
        through-thickness ``distribution``, ``cluster_location``,
        ``void_shape``.
    mesh_res : (int, int, int), optional
        ``(nx, ny, nz)``; default the production mesh of the laminate.
    length, width : float, optional
        The coupon, in mm: default 50 x 20, or with ``open_hole`` the
        open-hole coupon's (36 wide, three times as long).
    delaminations : sequence of Delamination, optional
        An explicit delamination in every solve; with toughness data the
        ``fracture_load_factor`` is an output.
    open_hole : mapping, optional
        Solve the open-hole coupon (:meth:`CompositeMesh.for_open_hole
        <porosity_fe.CompositeMesh.for_open_hole>`) instead, with these
        arguments of it (``hole_diameter``, ``n_circumferential``,
        ``n_radial``, ``radial_growth``, ``nz``; ``{}`` for its defaults),
        under ``'compression'`` or ``'tension'``: the voids at the hole
        edge count most. ``stress_concentration`` is then an output
        (:func:`~porosity_fe.open_hole_analysis`). Not with ``mesh_res`` or
        ``delaminations``.
    porosity_degradation : str
        Of :class:`~porosity_fe.FESolver` (default ``'isotropic'``).
    **solve_kwargs
        Other arguments of :meth:`FESolver.solve <porosity_fe.FESolver.solve>`
        (``applied_strain``, ``failure_criterion``, ``toughness_knockdown``,
        ...), the same for every draw.

    Returns
    -------
    FEUncertainty

    Raises
    ------
    ValueError
        For an unknown material, load case, method or kernel, a ``Vp`` that
        is not a fraction, fewer than 2 draws, a negative ``cov`` or
        ``vp_cov``, a ``config`` that sets ``seed`` or ``in_plane_map``, or
        a material field that cannot scatter.

    Notes
    -----
    Every draw reassembles the stiffness: the porosity of every element
    changes. A draw costs about one FE solve of the mesh (seconds on the
    production mesh of a 24-ply laminate), so a study of tens of draws is
    a minute or more. The ranking needs more draws than inputs; with only
    the random field scattering it ranks ``Vp_peak`` against
    ``Vp_spatial_cov``.

    Examples
    --------
    >>> from porosity_fe import MATERIALS
    >>> mat = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4)  # doctest: +SKIP
    >>> study = fe_porosity_uncertainty(mat, 0.03, cov=0.5,
    ...                                 correlation_length=5.0, seed=1)  # doctest: +SKIP
    >>> study.statistics('first_ply_failure_load_factor')  # doctest: +SKIP
    """
    if isinstance(material, str):
        if material not in MATERIALS:
            raise ValueError(
                f"Unknown material {material!r}. Available presets: {sorted(MATERIALS)}.")
        mat, label = MATERIALS[material], material
    else:
        mat, label = material, _material_label(material)
    nominal_vp = float(check_fraction(Vp, 'Vp'))
    if loading not in FE_UQ_LOADINGS:
        raise ValueError(
            f"The FE uncertainty study takes the coupon load cases "
            f"{list(FE_UQ_LOADINGS)}; got {loading!r}.")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) \
            or n_samples < 2:
        raise ValueError(f"n_samples must be an integer >= 2; got {n_samples!r}.")
    if method not in _UQ_METHODS:
        raise ValueError(f"Unknown sampling method {method!r}. Use one of {sorted(_UQ_METHODS)}.")
    if kernel not in RANDOM_FIELD_KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Use one of {list(RANDOM_FIELD_KERNELS)}.")
    vp_cov = float(vp_cov)
    if not math.isfinite(vp_cov) or vp_cov < 0.0:
        raise ValueError(f"vp_cov must be a finite number >= 0; got {vp_cov!r}.")
    config = dict(config or {})
    for key in ('seed', 'in_plane_map'):
        if key in config:
            raise ValueError(
                f"config cannot set {key!r}: the study draws the in-plane map of "
                f"every solve from its own seed.")
    pcts = tuple(float(p) for p in percentiles)
    resolved = _normalize_uq_spec(mat, covs, spec)
    names = list(resolved)
    hole: dict[str, Any] | None = None
    if open_hole is not None:
        if loading not in OPEN_HOLE_LOADINGS:
            raise ValueError(
                f"An open-hole study takes {list(OPEN_HOLE_LOADINGS)}; got {loading!r}.")
        if mesh_res is not None or delaminations:
            raise ValueError("open_hole sets its own mesh: give no mesh_res or "
                             "delaminations with it.")
        hole = dict(open_hole)
        for key in ('width', 'length', 'ply_angles', 'porosity_field', 'material'):
            if key in hole:
                raise ValueError(f"open_hole cannot set {key!r}: use the study's own "
                                 f"argument.")
        width = 36.0 if width is None else width
        length = 3.0 * width if length is None else length
        probe = CompositeMesh.for_open_hole(PorosityField(mat, 0.0), mat, width=width,
                                            length=length, ply_angles=ply_angles, **hole)
        nx, ny, nz = probe.nx, probe.ny, probe.nz
        hole_record = {'hole_diameter_mm': probe.open_hole_geometry.hole_diameter,
                       'width_mm': float(width), 'length_mm': float(length)}
    else:
        width = 20.0 if width is None else width
        length = 50.0 if length is None else length
        nx, ny, nz = mesh_res if mesh_res is not None else \
            _default_mesh_resolution(mat.n_plies, ply_angles)
    field_settings = {
        'cov': float(cov), 'kernel': kernel, 'preserve_mean': bool(preserve_mean),
        'correlation_length_mm': (list(map(float, correlation_length))
                                  if isinstance(correlation_length, Sequence)
                                  else float(correlation_length)),
        'resolution_mm': None if resolution is None else float(resolution),
        'plate_mm': [float(length), float(width)],
    }
    defects = {} if not delaminations else {'delaminations': list(delaminations)}

    def solve(material_obj: MaterialProperties, pf: PorosityField):
        if hole is not None:
            mesh = CompositeMesh.for_open_hole(pf, material_obj, width=width, length=length,
                                               ply_angles=ply_angles, **hole)
        else:
            mesh = CompositeMesh(pf, material_obj, nx, ny, nz, ply_angles=ply_angles,
                                 length=length, width=width, **defects)
        solver = FESolver(mesh, material_obj, pf, ply_angles=ply_angles,
                          porosity_degradation=porosity_degradation)
        return _outputs(solver.solve(loading, **solve_kwargs), mesh)

    # Every solve warns alike (a distorted mesh, say): the study collects
    # the warnings and raises each once at the end.
    messages: dict[str, warnings.WarningMessage] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nominal = solve(mat, PorosityField(mat, nominal_vp, **config))
    for w in caught:
        messages.setdefault(str(w.message), w)
    recorded = [k for k in FE_UQ_OUTPUTS if nominal[k] is not None]

    root = np.random.SeedSequence(seed)
    children = root.spawn(int(n_samples) + 1)
    sample_vp = vp_cov > 0.0
    unit = _draw_unit_samples(len(names) + int(sample_vp), int(n_samples), method,
                              np.random.default_rng(children[0]))
    sigma_vp = math.sqrt(math.log1p(vp_cov * vp_cov))
    samples = {k: np.empty(int(n_samples)) for k in recorded}
    inputs = {k: np.empty(int(n_samples)) for k in ('Vp', 'Vp_peak', 'Vp_spatial_cov')}
    inputs.update({k: np.empty(int(n_samples)) for k in names})
    for s in range(int(n_samples)):
        draws = {name: float(_unit_to_draw(np.array([unit[s, i]]), resolved[name][0])[0])
                 for i, name in enumerate(names)}
        drawn = mat.perturb(draws, resolved) if names else mat
        vp = nominal_vp
        if sample_vp:
            z = float(_unit_to_draw(np.array([unit[s, len(names)]]), 'lognormal')[0])
            vp = float(min(nominal_vp * math.exp(sigma_vp * z - 0.5 * sigma_vp ** 2), 1.0))
        map_seed = int(children[s + 1].generate_state(1)[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pf = PorosityField.random(
                drawn, vp, cov=cov, correlation_length=correlation_length,
                length=length, width=width, kernel=kernel, resolution=resolution,
                seed=map_seed, preserve_mean=preserve_mean, **config)
            out = solve(drawn, pf)
        for w in caught:
            messages.setdefault(str(w.message), w)
        for k in recorded:
            value = out[k]
            samples[k][s] = math.inf if value is None else value
        pmap = pf.in_plane_map
        assert pmap is not None
        # With preserve_mean the map's average is vp to round-off.
        inputs['Vp'][s] = vp if preserve_mean else pf.Vp
        inputs['Vp_peak'][s] = pmap.max
        inputs['Vp_spatial_cov'][s] = float(pmap.Vp.std() / pmap.Vp.mean()) \
            if pmap.Vp.mean() > 0.0 else 0.0
        for name in names:
            inputs[name][s] = float(getattr(drawn, name))
        logger.info("FE uncertainty draw %d of %d: %s", s + 1, n_samples,
                    ", ".join(f"{k} {samples[k][s]:.4g}" for k in recorded))
    for w in messages.values():
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return FEUncertainty(
        loading=loading, material=label, Vp=nominal_vp, n_samples=int(n_samples),
        seed=seed, method=method, field=field_settings, vp_cov=vp_cov,
        spec={k: list(v) for k, v in resolved.items()}, percentiles=pcts,
        nominal={k: nominal[k] for k in recorded}, samples=samples, inputs=inputs,
        mesh=(int(nx), int(ny), int(nz)), open_hole=hole_record if hole is not None else None)


def save_fe_uncertainty_to_json(study: FEUncertainty, filename: str | os.PathLike, *,
                                include_samples: bool = False) -> None:
    """Write an :class:`FEUncertainty` as a ``porosity-fe.fe-uq`` JSON
    (:data:`~porosity_fe.FORMAT_FE_UQ`): its :meth:`~FEUncertainty.summary`
    under ``fe_uq``, with the study's seed in the provenance.

    With ``include_samples`` the per-draw ``samples`` and ``inputs`` are
    written too (an infinite load factor as ``null``). Standard JSON: a
    non-finite value anywhere else raises :class:`ValueError` before the
    file is written.
    """
    payload = study.summary()
    if include_samples:
        def column(values: np.ndarray) -> list:
            return [float(v) if math.isfinite(float(v)) else None for v in values]

        payload['samples'] = {k: column(v) for k, v in study.samples.items()}
        payload['input_samples'] = {k: column(v) for k, v in study.inputs.items()}
    envelope = _wrap_envelope(FORMAT_FE_UQ, dict(_UNITS_FE_UQ), {'fe_uq': payload},
                              seed=study.seed)
    _write_json(envelope, Path(filename))
