"""Continuous porosity field with optional discrete-void superposition."""

from __future__ import annotations

import functools

import numpy as np

from ._types import ClusterLocation, Distribution
from .materials import MaterialProperties
from .ply_stack import PlyStack, as_material
from .porosity_map import PorosityMap
from .void_geometry import VOID_SHAPES, VoidGeometry

# ============================================================
# SECTION 3: POROSITY FIELD MODEL
# ============================================================


def _shape_profile(z: np.ndarray, distribution: str, z0: float, Lz: float,
                   t_ply: float, n_plies: int) -> np.ndarray:
    """Un-normalized through-thickness shape for ``clustered`` / ``interface``.

    ``clustered`` is a Gaussian bump at ``z0`` with ``sigma = Lz / 6``;
    ``interface`` is a sum of Gaussians (``sigma = 0.35 t_ply``) at each of
    the ``n_plies - 1`` ply interfaces.
    """
    if distribution == 'clustered':
        sigma = Lz / 6
        return np.exp(-0.5 * ((z - z0) / sigma)**2)
    profile = np.zeros_like(z)
    for k in range(1, n_plies):
        z_int = k * t_ply
        profile += np.exp(-0.5 * ((z - z_int) / (t_ply * 0.35))**2)
    return profile


@functools.lru_cache(maxsize=128)
def _profile_normalization(distribution: str, z0: float, Lz: float,
                           t_ply: float, n_plies: int) -> float:
    """Mean of :func:`_shape_profile` over ``[0, Lz]`` so the field averages to ``Vp``.

    Memoized on the arguments that define the shape, so repeated
    ``local_porosity`` calls on one field (or on fields sharing a laminate)
    reuse it, and a changed attribute simply maps to a new cache entry.
    """
    z_ref = np.linspace(0, Lz, 1000)
    mean_val = np.mean(_shape_profile(z_ref, distribution, z0, Lz, t_ply, n_plies))
    return mean_val if mean_val > 0 else 1.0


def _stack_interface_profile(z: np.ndarray, centres: tuple[float, ...],
                             sigmas: tuple[float, ...]) -> np.ndarray:
    """Un-normalized ``interface`` shape of a :class:`PlyStack`: a Gaussian
    at each interface, ``sigma = 0.35`` times the thinner of its two plies."""
    profile = np.zeros_like(z)
    for z_int, sigma in zip(centres, sigmas, strict=True):
        profile += np.exp(-0.5 * ((z - z_int) / sigma)**2)
    return profile


@functools.lru_cache(maxsize=128)
def _stack_interface_normalization(centres: tuple[float, ...], sigmas: tuple[float, ...],
                                   Lz: float) -> float:
    """Mean of :func:`_stack_interface_profile` over ``[0, Lz]``, sampled as
    :func:`_profile_normalization` samples the single-material shape."""
    z_ref = np.linspace(0, Lz, 1000)
    mean_val = np.mean(_stack_interface_profile(z_ref, centres, sigmas))
    return mean_val if mean_val > 0 else 1.0

POROSITY_CONFIGS = {
    'uniform_spherical': {
        'distribution': 'uniform',
        'void_shape': 'spherical',
    },
    'uniform_cylindrical': {
        'distribution': 'uniform',
        'void_shape': 'cylindrical',
    },
    'clustered_midplane': {
        'distribution': 'clustered',
        'void_shape': 'spherical',
        'cluster_location': 'midplane',
    },
    'clustered_surface': {
        'distribution': 'clustered',
        'void_shape': 'spherical',
        'cluster_location': 'surface',
    },
    'interface_penny': {
        'distribution': 'interface',
        'void_shape': 'penny',
    },
}


class PorosityField:
    """Continuous through-thickness porosity profile plus discrete voids.

    A ``PorosityField`` produces a scalar void volume fraction
    ``Vp(x, y, z) in [0, 1]`` at any point inside the laminate by
    superposing a smooth distributed component (one of three closed-form
    through-thickness profiles) with an optional list of explicit
    :class:`VoidGeometry` ellipsoids. The distributed profile is
    renormalized so its mean over ``z in [0, Lz]`` equals ``Vp``. This is
    the porosity input consumed by :class:`CompositeMesh` (sampled at
    every node) and by :class:`Hex8Element` (the Mori-Tanaka stiffness
    degradation reads the per-node values from here).

    Parameters
    ----------
    material : MaterialProperties or PlyStack
        Composite material — supplies ``t_ply`` and ``n_plies`` (and
        hence ``Lz = t_ply * n_plies``) used by the through-thickness
        profile. A :class:`~porosity_fe.PlyStack` (a hybrid laminate)
        supplies its plies' thicknesses instead: ``Lz`` is their sum and
        the ``'interface'`` Gaussians sit at its interfaces.
    void_volume_fraction : float
        Mean void volume fraction ``Vp`` over the laminate, as a
        fraction in ``[0, 1]`` (e.g. ``0.02`` for 2 %). Passing a percent
        such as ``2.0`` raises ``ValueError`` with a unit hint.
    distribution : {'uniform', 'clustered', 'interface'}, optional
        Through-thickness profile shape (default ``'uniform'``).

        - ``'uniform'`` — constant ``Vp`` at every ``z``.
        - ``'clustered'`` — Gaussian bump centered at
          ``Lz * _CLUSTER_OFFSETS[cluster_location]`` with
          ``sigma = Lz / 6``.
        - ``'interface'`` — sum of Gaussians at each ply interface
          ``z = k * t_ply`` with ``sigma = 0.35 * t_ply``; for a
          :class:`~porosity_fe.PlyStack`, at each of its interfaces with
          ``sigma = 0.35`` times the thinner of the two plies.
    void_shape : str or tuple of 3 float, optional
        Either a key of :data:`VOID_SHAPES`
        (``'spherical'``, ``'cylindrical'`` or ``'penny'``) or an
        explicit ``(a1, a2, a3)`` shape-radii tuple. Used as the Eshelby
        inclusion shape for the Mori-Tanaka homogenization in
        :class:`Hex8Element`. Default ``'spherical'``.
    cluster_location : {'midplane', 'surface', 'quarter'}, optional
        Location of the Gaussian bump for ``distribution='clustered'``,
        expressed as a fraction of ``Lz``. ``'midplane'`` -> ``0.5``,
        ``'surface'`` -> ``0.0``, ``'quarter'`` -> ``0.25``. Ignored for
        the other distributions. Default ``'midplane'``.
    discrete_voids : list of VoidGeometry, optional
        Explicit ellipsoidal voids superposed on the smooth profile
        (their contribution is taken via ``max``, then clipped at 1.0).
    seed : int, optional
        Recorded into provenance. The constructor is RNG-free, so it has no
        effect here: :meth:`random` draws the in-plane map from it and
        records it.
    in_plane_map : PorosityMap, optional
        Keyword-only. In-plane variation of the distributed porosity: at
        ``(x, y)`` the through-thickness profile is multiplied by
        ``in_plane_map(x, y) / in_plane_map.mean``, so the map gives the
        pattern and ``void_volume_fraction`` stays the average over the
        map's extent. :meth:`from_map` takes the average from the map
        itself, which keeps the map's absolute values. ``None`` (default):
        no in-plane variation.

    Attributes
    ----------
    material : MaterialProperties
        Stored reference to the input material (for a
        :class:`~porosity_fe.PlyStack`, its
        :attr:`~porosity_fe.PlyStack.reference`).
    ply_stack : PlyStack or None
        The stack the field was built for, ``None`` for a single material.
    Vp : float
        Mean void volume fraction (after the snap-to-1.0 tolerance).
    distribution : str
        One of the three names listed above.
    cluster_location : str
        One of ``'midplane'``, ``'surface'``, ``'quarter'``.
    void_shape_radii : tuple of 3 float
        Resolved Eshelby ``(a1, a2, a3)`` shape radii.
    discrete_voids : list of VoidGeometry
        Discrete inclusions (empty list if none were supplied).
    Lz : float
        Total laminate thickness (mm), copied from
        ``material.total_thickness`` (the stack's total thickness).
    seed : int or None
        The seed argument, kept for provenance (the seed of the map of
        :meth:`random`).
    in_plane_map : PorosityMap or None
        The in-plane map, when one was given.

    Notes
    -----
    With an ``in_plane_map`` the local porosity is
    ``Vp(x, y, z) = profile(z) * map(x, y) / map.mean`` (clipped at 1), and
    ``Vp`` is the average over the map's extent. The FE path samples it at
    the mesh nodes, in the mesh's coordinates: a mesh larger than the map
    sees the map's edge values outside it. The empirical failure load uses
    the specimen average alone, so a map changes it only through its mean.

    Examples
    --------
    A 2 % uniformly-distributed spherical porosity field:

    >>> from porosity_fe import MATERIALS, PorosityField
    >>> mat = MATERIALS['T800_epoxy']
    >>> field = PorosityField(mat, void_volume_fraction=0.02,
    ...                       distribution='uniform',
    ...                       void_shape='spherical')
    >>> bool(0.0 < field.Vp <= 1.0)
    True

    A midplane-clustered field with penny-shaped voids:

    >>> field = PorosityField(mat, void_volume_fraction=0.03,
    ...                       distribution='clustered',
    ...                       cluster_location='midplane',
    ...                       void_shape='penny')
    """

    _CLUSTER_OFFSETS = {'midplane': 0.5, 'surface': 0.0, 'quarter': 0.25}
    _DISTRIBUTIONS = ('uniform', 'clustered', 'interface')

    def __init__(self, material: MaterialProperties | PlyStack,
                 void_volume_fraction: float,
                 distribution: Distribution = 'uniform', void_shape: str | tuple = 'spherical',
                 cluster_location: ClusterLocation = 'midplane',
                 discrete_voids: list[VoidGeometry] | None = None,
                 seed: int | None = None, *,
                 in_plane_map: PorosityMap | None = None):
        if void_volume_fraction is None:
            raise ValueError("void_volume_fraction is None; expected a finite float in [0, 1].")
        if not isinstance(void_volume_fraction, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"void_volume_fraction must be a numeric type (int, float, or numpy scalar), "
                f"got {type(void_volume_fraction).__name__}."
            )
        material, stack = as_material(material)
        Vp = float(void_volume_fraction)
        # Snap to 1.0 for inputs in (1.0, 1.0 + 1e-9] — accommodates upstream
        # numerical noise (e.g. np.mean across an element) without rejecting it.
        if 1.0 < Vp <= 1.0 + 1e-9:
            Vp = 1.0
        if not np.isfinite(Vp) or not (0.0 <= Vp <= 1.0):
            # Suppress the percent-confusion hint for values just above 1.0
            # (likely numerical noise rather than a percent mistake).
            show_percent_hint = (np.isfinite(Vp) and Vp >= 1.0 + 1e-3)
            hint = (f" Did you pass a percent? Use {Vp / 100:.4f} instead of {Vp}."
                    if show_percent_hint else "")
            raise ValueError(
                f"void_volume_fraction must be a finite fraction in [0, 1], "
                f"got {void_volume_fraction!r}.{hint}"
            )

        if distribution not in self._DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {distribution!r}. "
                f"Use one of {list(self._DISTRIBUTIONS)}."
            )
        if distribution == 'interface' and int(material.n_plies) < 2:
            # One ply has no interface: the profile would be zero everywhere
            # while Vp says otherwise.
            raise ValueError(
                f"distribution='interface' needs at least two plies (it puts "
                f"the voids at the ply interfaces), got n_plies="
                f"{material.n_plies}. Use 'uniform' or 'clustered'.")
        if cluster_location not in self._CLUSTER_OFFSETS:
            raise ValueError(
                f"Unknown cluster_location {cluster_location!r}. "
                f"Use one of {sorted(self._CLUSTER_OFFSETS)}."
            )

        self.material = material
        self.ply_stack = stack
        self.Vp = Vp
        self.distribution = distribution
        self.cluster_location = cluster_location
        self.discrete_voids = discrete_voids or []
        # The constructor is RNG-free; `seed` is recorded into provenance
        # (#55), and PorosityField.random draws its map from it.
        self.seed = seed
        self.Lz = material.total_thickness if stack is None else stack.total_thickness
        if in_plane_map is not None:
            if not isinstance(in_plane_map, PorosityMap):
                raise TypeError(
                    f"in_plane_map must be a PorosityMap or None; got "
                    f"{type(in_plane_map).__name__}.")
            if Vp > 0.0 and not in_plane_map.mean > 0.0:
                raise ValueError(
                    "in_plane_map is zero everywhere: it gives no pattern to "
                    "spread a non-zero void_volume_fraction over.")
        self.in_plane_map = in_plane_map

        # Resolve void shape
        if isinstance(void_shape, str):
            if void_shape not in VOID_SHAPES:
                raise ValueError(
                    f"Unknown void_shape {void_shape!r}. "
                    f"Use one of {sorted(VOID_SHAPES)}."
                )
            self.void_shape_radii = VOID_SHAPES[void_shape]
        else:
            radii = tuple(void_shape)
            if len(radii) != 3 or not all(
                    isinstance(r, (int, float, np.integer, np.floating))
                    and not isinstance(r, bool) and np.isfinite(r) and r > 0
                    for r in radii):
                raise ValueError(
                    f"void_shape must be one of {sorted(VOID_SHAPES)} or three "
                    f"finite radii > 0 (a1, a2, a3; only their ratios matter), "
                    f"got {void_shape!r}.")
            self.void_shape_radii = (float(radii[0]), float(radii[1]), float(radii[2]))

    def _distributed_porosity(self, z: np.ndarray) -> np.ndarray:
        """Through-thickness distributed porosity profile."""
        z = np.asarray(z, dtype=float)
        match self.distribution:
            case 'uniform':
                return np.full_like(z, self.Vp)
            case 'interface' if self.ply_stack is not None:
                t = self.ply_stack.thicknesses
                centres = tuple(self.ply_stack.interface_z.tolist())
                sigmas = tuple((0.35 * np.minimum(t[:-1], t[1:])).tolist())
                profile = _stack_interface_profile(z, centres, sigmas)
                return self.Vp * profile / _stack_interface_normalization(
                    centres, sigmas, float(self.Lz))
            case 'clustered' | 'interface':
                z0 = (self.Lz * self._CLUSTER_OFFSETS[self.cluster_location]
                      if self.distribution == 'clustered' else 0.0)
                shape = (self.distribution, z0, self.Lz,
                         self.material.t_ply, self.material.n_plies)
                profile = _shape_profile(z, *shape)
                return self.Vp * profile / _profile_normalization(*shape)
            case _:
                raise ValueError(
                    f"Unknown distribution {self.distribution!r}. "
                    f"Use one of {list(self._DISTRIBUTIONS)}."
                )

    @classmethod
    def from_map(cls, material: MaterialProperties, porosity_map: PorosityMap,
                 **kwargs) -> PorosityField:
        """A field whose in-plane void content is ``porosity_map``.

        The specimen-average ``Vp`` is the map's area mean, so the
        through-thickness average of the field at ``(x, y)`` is the map's
        value there. ``kwargs`` are the other constructor arguments
        (``distribution``, ``void_shape``, ``cluster_location``,
        ``discrete_voids``, ``seed``).
        """
        if not isinstance(porosity_map, PorosityMap):
            raise TypeError(
                f"porosity_map must be a PorosityMap; got {type(porosity_map).__name__}.")
        return cls(material, porosity_map.mean, in_plane_map=porosity_map, **kwargs)

    @classmethod
    def random(cls, material: MaterialProperties, Vp: float, *, cov: float,
               correlation_length: float | tuple[float, float], length: float = 50.0,
               width: float = 20.0, kernel: str = 'squared_exponential',
               resolution: float | None = None, seed: int | None = None,
               preserve_mean: bool = True, **kwargs) -> PorosityField:
        """A field whose in-plane void content is a seeded random realization.

        Draws a lognormal random map of mean ``Vp`` over the ``length`` x
        ``width`` plate (:meth:`PorosityMap.random
        <porosity_fe.PorosityMap.random>`, which documents ``cov``,
        ``correlation_length``, ``kernel``, ``resolution`` and
        ``preserve_mean``) and returns :meth:`from_map` of it, with ``seed``
        recorded: the same seed gives the same field. ``kwargs`` are the
        other constructor arguments (``distribution``, ``void_shape``,
        ``cluster_location``, ``discrete_voids``), so the through-thickness
        profile is the usual one and the map scales it from point to point.
        Use the plate size of the :class:`~porosity_fe.CompositeMesh` the
        field is for (default 50 x 20 mm).

        Examples
        --------
        >>> from porosity_fe import MATERIALS
        >>> pf = PorosityField.random(MATERIALS['T800_epoxy'], 0.02, cov=0.4,
        ...                           correlation_length=5.0, seed=7)
        >>> round(pf.Vp, 12), pf.seed, pf.in_plane_map.max > 0.02
        (0.02, 7, True)
        """
        porosity_map = PorosityMap.random(
            length, width, Vp, cov=cov, correlation_length=correlation_length,
            kernel=kernel, resolution=resolution, seed=seed, preserve_mean=preserve_mean)
        return cls.from_map(material, porosity_map, seed=seed, **kwargs)

    def _distributed(self, x, y, z) -> np.ndarray:
        """The distributed porosity at ``(x, y, z)``: the through-thickness
        profile, times the in-plane map over its mean when there is one."""
        dist = self._distributed_porosity(z)
        m = self.in_plane_map
        if m is None or not m.mean > 0.0:
            return dist
        return dist * (m(x, y) / m.mean)

    def local_porosity(self, x, y, z) -> np.ndarray:
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return self._with_discrete_voids(self._distributed(x, y, z), x, y, z)

    def _with_discrete_voids(self, Vp_dist: np.ndarray, x, y, z) -> np.ndarray:
        """``Vp_dist`` with the discrete voids added (``1`` inside one),
        clipped at 1."""
        Vp_discrete = np.zeros_like(Vp_dist)
        for void in self.discrete_voids:
            Vp_discrete = np.maximum(Vp_discrete,
                                      void.contains(x, y, z).astype(float))
        return np.minimum(Vp_dist + Vp_discrete, 1.0)

    def local_stiffness_reduction(self, x, y, z) -> np.ndarray:
        Vp_local = self.local_porosity(x, y, z)
        return 1.0 - Vp_local

    def get_void_locations(self) -> list:
        return [(v.center.tolist(), v.radii.tolist()) for v in self.discrete_voids]

    def effective_porosity_profile(self, nz: int = 100, x: float = 25.0,
                                   y: float = 10.0) -> tuple:
        """Through-thickness profile, including discrete voids, at ``(x, y)``.

        Parameters
        ----------
        nz : int
            Number of sample points through the thickness.
        x, y : float
            In-plane sampling location in mm. The defaults are the center of
            the 50 x 20 mm ``CompositeMesh`` domain; pass the coordinates of
            a discrete void to see its contribution.

        Returns
        -------
        (z, Vp) : tuple of np.ndarray
            Thickness coordinates (mm) and porosity fractions.
        """
        z_coords = np.linspace(0, self.Lz, nz)
        Vp_vals = self.local_porosity(np.full(nz, float(x)), np.full(nz, float(y)),
                                      z_coords)
        return z_coords, Vp_vals

    def __repr__(self) -> str:
        mapped = "" if self.in_plane_map is None else f", in_plane_map={self.in_plane_map!r}"
        return (f"PorosityField(Vp={self.Vp:.4f}, "
                f"distribution='{self.distribution}', "
                f"void_shape={self.void_shape_radii}, "
                f"n_discrete_voids={len(self.discrete_voids)}{mapped})")
