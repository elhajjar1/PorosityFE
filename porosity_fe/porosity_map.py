"""In-plane porosity maps: the void content as a function of ``(x, y)``.

A :class:`PorosityMap` holds the through-thickness average void content on
a rectilinear grid, as an ultrasonic C-scan correlation or a CT slice stack
gives it, or as built from a background and a few :class:`PorosityPatch`
zones, or drawn as a seeded realization of a random field
(:meth:`PorosityMap.random`). :class:`~porosity_fe.PorosityField` takes one
as its in-plane variation (``PorosityField.from_map``, ``in_plane_map=``,
``PorosityField.random``).
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._vp import check_fraction

_PATCH_SHAPES = ('ellipse', 'rectangle')

#: Correlation kernels of :meth:`PorosityMap.random`.
RANDOM_FIELD_KERNELS = ('squared_exponential', 'exponential')

#: Grid points per axis of a random map at most: the spacing grows past the
#: requested one rather than exceed it.
_RANDOM_MAX_POINTS = 401


def _pair(name: str, value: Any, *, positive: bool) -> tuple[float, float]:
    """``value`` as two finite floats (both ``> 0`` when ``positive``)."""
    try:
        a, b = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be two numbers (x, y) in mm; got {value!r}.") from None
    if not (math.isfinite(a) and math.isfinite(b)) or (positive and not (a > 0 and b > 0)):
        raise ValueError(
            f"{name} must be two finite numbers"
            f"{' above 0' if positive else ''} (x, y) in mm; got {value!r}.")
    return a, b


@dataclass(frozen=True)
class PorosityPatch:
    """A zone of a porosity map with its own void content.

    Parameters
    ----------
    center : (float, float)
        Centre ``(xc, yc)`` of the zone, in mm.
    half_size : (float, float)
        Semi-axes ``(ax, ay)`` of an ellipse, or the half-sizes of a
        rectangle, in mm.
    Vp : float
        Through-thickness average void content inside the zone, a
        **fraction** in ``[0, 1]``.
    shape : {'ellipse', 'rectangle'}
        The planform (default ``'ellipse'``).
    """

    center: tuple[float, float]
    half_size: tuple[float, float]
    Vp: float
    shape: str = 'ellipse'

    def __post_init__(self) -> None:
        if self.shape not in _PATCH_SHAPES:
            raise ValueError(
                f"Unknown patch shape {self.shape!r}. Use one of {list(_PATCH_SHAPES)}.")
        object.__setattr__(self, 'center', _pair('center', self.center, positive=False))
        object.__setattr__(self, 'half_size',
                           _pair('half_size', self.half_size, positive=True))
        object.__setattr__(self, 'Vp', float(check_fraction(self.Vp, 'PorosityPatch.Vp')))

    def contains(self, x: Any, y: Any) -> np.ndarray:
        """Whether the points ``(x, y)`` lie in the zone (its edge included)."""
        u = (np.asarray(x, dtype=float) - self.center[0]) / self.half_size[0]
        v = (np.asarray(y, dtype=float) - self.center[1]) / self.half_size[1]
        if self.shape == 'ellipse':
            return u * u + v * v <= 1.0
        return (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)


def _area_mean(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> float:
    """Area average over the grid of the bilinear interpolant of ``v``."""
    # Per cell, the mean of its four corners times its area.
    corners = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
    total = float((corners * np.diff(x)[:, None] * np.diff(y)[None, :]).sum())
    return total / float((x[-1] - x[0]) * (y[-1] - y[0]))


def _correlation_factor(t: np.ndarray, length: float, kernel: str) -> np.ndarray:
    """``A`` with ``A A^T`` the correlation matrix of the points ``t``.

    From the symmetric eigendecomposition, with the eigenvalues that
    round-off makes negative set to zero: the squared-exponential matrix of
    a fine grid is singular to machine precision, where a Cholesky factor
    fails.
    """
    d = np.abs(t[:, None] - t[None, :]) / length
    corr = np.exp(-0.5 * d * d) if kernel == 'squared_exponential' else np.exp(-d)
    w, v = np.linalg.eigh(corr)
    return v * np.sqrt(np.clip(w, 0.0, None))


def _axis(name: str, values: Any) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() \
            or not (np.diff(axis) > 0.0).all():
        raise ValueError(
            f"{name} must be at least two finite, strictly increasing "
            f"coordinates in mm.")
    return axis


class PorosityMap:
    """Void content on a rectilinear in-plane grid, interpolated bilinearly.

    The value at ``(x, y)`` is the through-thickness average void content
    of the laminate there. :class:`~porosity_fe.PorosityField` distributes
    it through the thickness with its ``distribution``.

    Parameters
    ----------
    x, y : array-like
        Grid coordinates in mm, in the axes of the mesh the map is used on
        (``x`` along the load of the coupon, from 0), each at least two
        strictly increasing values.
    Vp : array-like
        Void content at the grid points, **fractions** in ``[0, 1]``, of
        shape ``(len(x), len(y))``.

    Attributes
    ----------
    x, y : np.ndarray
        The grid coordinates.
    Vp : np.ndarray
        The grid values, shape ``(len(x), len(y))``.

    Notes
    -----
    Between the grid points the map is bilinear; outside the grid it keeps
    its edge values. :attr:`mean` is the area average over the grid's
    extent, the specimen-average void content of a laminate that the grid
    covers.

    Examples
    --------
    >>> porosity_map = PorosityMap([0.0, 50.0], [0.0, 20.0],
    ...                            [[0.01, 0.01], [0.03, 0.03]])
    >>> round(porosity_map.mean, 6), float(porosity_map(25.0, 10.0))
    (0.02, 0.02)
    """

    def __init__(self, x: Any, y: Any, Vp: Any) -> None:
        self.x = _axis('x', x)
        self.y = _axis('y', y)
        values = check_fraction(Vp, 'PorosityMap.Vp')
        if values.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"Vp must have shape (len(x), len(y)) = "
                f"({self.x.size}, {self.y.size}); got {values.shape}.")
        self.Vp = values

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the grid, in mm."""
        return (float(self.x[0]), float(self.x[-1]),
                float(self.y[0]), float(self.y[-1]))

    @property
    def mean(self) -> float:
        """Area average of the map over its extent (a fraction)."""
        return _area_mean(self.x, self.y, self.Vp)

    @property
    def max(self) -> float:
        """Largest value of the map (a fraction)."""
        return float(self.Vp.max())

    def __call__(self, x: Any, y: Any) -> np.ndarray:
        """The map at the points ``(x, y)`` (mm), clamped to its extent."""
        xq = np.clip(np.asarray(x, dtype=float), self.x[0], self.x[-1])
        yq = np.clip(np.asarray(y, dtype=float), self.y[0], self.y[-1])
        i = np.clip(np.searchsorted(self.x, xq, side='right') - 1, 0, self.x.size - 2)
        j = np.clip(np.searchsorted(self.y, yq, side='right') - 1, 0, self.y.size - 2)
        tx = (xq - self.x[i]) / (self.x[i + 1] - self.x[i])
        ty = (yq - self.y[j]) / (self.y[j + 1] - self.y[j])
        v = self.Vp
        return ((1.0 - tx) * (1.0 - ty) * v[i, j] + tx * (1.0 - ty) * v[i + 1, j]
                + (1.0 - tx) * ty * v[i, j + 1] + tx * ty * v[i + 1, j + 1])

    @classmethod
    def from_patches(cls, length: float, width: float, background: float,
                     patches: Sequence[PorosityPatch], *,
                     resolution: float = 0.5) -> PorosityMap:
        """A map of a uniform background with zones of their own void content.

        Parameters
        ----------
        length, width : float
            The plate along ``x`` and ``y``, in mm: the map covers
            ``[0, length] x [0, width]``.
        background : float
            Void content outside the zones, a fraction.
        patches : sequence of PorosityPatch
            The zones. Where two overlap, the later one counts.
        resolution : float
            Target grid spacing in mm (default 0.5). A zone is drawn on the
            grid points it contains, so its edge is one grid cell wide and
            a zone smaller than the spacing can be missed.
        """
        size = _pair('(length, width)', (length, width), positive=True)
        if not (isinstance(resolution, (int, float)) and math.isfinite(resolution)
                and resolution > 0.0):
            raise ValueError(f"resolution must be a spacing above 0 in mm; got {resolution!r}.")
        base = float(check_fraction(background, 'background'))
        x = np.linspace(0.0, size[0], max(2, round(size[0] / resolution) + 1))
        y = np.linspace(0.0, size[1], max(2, round(size[1] / resolution) + 1))
        gx, gy = np.meshgrid(x, y, indexing='ij')
        values = np.full(gx.shape, base)
        for patch in patches:
            if not isinstance(patch, PorosityPatch):
                raise TypeError(
                    f"patches must be PorosityPatch objects; got {type(patch).__name__}.")
            values[patch.contains(gx, gy)] = patch.Vp
        return cls(x, y, values)

    @classmethod
    def random(cls, length: float, width: float, mean: float, *, cov: float,
               correlation_length: float | Sequence[float],
               kernel: str = 'squared_exponential', resolution: float | None = None,
               seed: int | np.random.SeedSequence | None = None,
               preserve_mean: bool = True) -> PorosityMap:
        """A seeded realization of a lognormal random field of void content.

        A stationary Gaussian field ``g(x, y)`` of zero mean and unit
        variance, with the correlation ``rho(dx, dy) = rho_x(dx) rho_y(dy)``,
        is mapped to the void content

        .. math::

            V_p(x, y) = \\bar V_p \\exp(\\sigma g - \\sigma^2 / 2),
            \\qquad \\sigma^2 = \\ln(1 + \\mathrm{CoV}^2),

        lognormal with mean ``mean`` and coefficient of variation ``cov``
        point by point, so it stays positive. ``'squared_exponential'`` takes
        ``rho(d) = exp(-d^2 / (2 l^2))`` (a smooth field) and
        ``'exponential'`` ``rho(d) = exp(-|d| / l)`` per axis (a rough one).
        The correlation is a product over the axes, so the field is drawn
        as ``A_x Z A_y^T`` from a matrix ``Z`` of standard normal variates
        and the factors of the two axes' correlation matrices: a fine grid
        costs two small eigendecompositions, not one of the whole grid.

        Parameters
        ----------
        length, width : float
            The plate along ``x`` and ``y``, in mm: the map covers
            ``[0, length] x [0, width]`` (the size of the
            :class:`~porosity_fe.CompositeMesh` it is meant for).
        mean : float
            The mean void content, a **fraction** in ``[0, 1]``.
        cov : float
            Coefficient of variation of the void content from point to point
            (``>= 0``; 0 gives a uniform map).
        correlation_length : float or (float, float)
            The length ``l`` of the kernel, in mm, one for both axes or
            ``(l_x, l_y)``. Points much closer than it have nearly the same
            void content; points much farther apart are independent.
        kernel : {'squared_exponential', 'exponential'}
            The correlation function (default ``'squared_exponential'``).
        resolution : float, optional
            Grid spacing in mm; default a quarter of the shorter correlation
            length. At most 401 points per axis: a finer spacing is coarsened.
        seed : int or numpy.random.SeedSequence, optional
            Seed of the draw: the same seed gives the same map.
        preserve_mean : bool
            Scale the realization so its area average (:attr:`mean`) is
            ``mean`` exactly (default): the specimen keeps its measured void
            content and the field only redistributes it. ``False`` keeps the
            raw draw, whose average scatters about ``mean``.

        Returns
        -------
        PorosityMap
            Values above 1 are clipped to 1 (and, with ``preserve_mean``,
            the rest rescaled until the average is ``mean``).

        Raises
        ------
        ValueError
            For a size, correlation length or spacing that is not positive
            and finite, a ``cov`` below 0, an unknown ``kernel``, or a
            ``mean`` the clipped field cannot reach.

        Examples
        --------
        >>> m = PorosityMap.random(50.0, 20.0, 0.02, cov=0.4,
        ...                        correlation_length=5.0, seed=1)
        >>> round(m.mean, 12), m.max > 0.02
        (0.02, True)
        """
        size = _pair('(length, width)', (length, width), positive=True)
        ell = correlation_length if isinstance(correlation_length, Sequence) \
            else (correlation_length, correlation_length)
        ell = _pair('correlation_length', ell, positive=True)
        target = float(check_fraction(mean, 'mean'))
        if isinstance(cov, bool) or not isinstance(cov, (int, float, np.number)) \
                or not math.isfinite(float(cov)) or float(cov) < 0.0:
            raise ValueError(f"cov must be a finite number >= 0; got {cov!r}.")
        if kernel not in RANDOM_FIELD_KERNELS:
            raise ValueError(
                f"Unknown kernel {kernel!r}. Use one of {list(RANDOM_FIELD_KERNELS)}.")
        spacing = 0.25 * min(ell) if resolution is None else resolution
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float, np.number)) \
                or not math.isfinite(float(spacing)) or float(spacing) <= 0.0:
            raise ValueError(f"resolution must be a spacing above 0 in mm; got {resolution!r}.")
        axes = [np.linspace(0.0, extent, min(_RANDOM_MAX_POINTS,
                                             max(2, round(extent / float(spacing)) + 1)))
                for extent in size]
        x, y = axes
        rng = np.random.default_rng(seed)
        if target == 0.0 or float(cov) == 0.0:
            return cls(x, y, np.full((x.size, y.size), target))
        g = _correlation_factor(x, ell[0], kernel) @ rng.standard_normal((x.size, y.size)) \
            @ _correlation_factor(y, ell[1], kernel).T
        sigma = math.sqrt(math.log1p(float(cov) ** 2))
        values = target * np.exp(sigma * g - 0.5 * sigma * sigma)
        if preserve_mean:
            for _ in range(50):
                values = np.minimum(values * (target / _area_mean(x, y, values)), 1.0)
                if abs(_area_mean(x, y, values) - target) <= 1e-12 * target:
                    break
            else:
                raise ValueError(
                    f"A random map of mean {target:g} with cov {cov:g} cannot keep "
                    f"its mean below the cap of 1: lower cov or mean.")
        else:
            values = np.minimum(values, 1.0)
        return cls(x, y, values)

    @classmethod
    def from_csv(cls, path: str | os.PathLike) -> PorosityMap:
        """Read a map from a CSV table of grid points.

        The file has a header row naming the columns ``x``, ``y`` (mm) and
        ``Vp`` (a **fraction**; case does not matter, other columns are
        ignored) and one row per grid point, in any order. Lines starting
        with ``#`` are comments. The points must form a complete
        rectilinear grid: every ``x`` with every ``y``, once.
        """
        with open(path, encoding='utf-8', newline='') as f:
            rows = [row for row in csv.reader(f)
                    if row and not row[0].lstrip().startswith('#')]
        if not rows:
            raise ValueError(f"{path}: no data.")
        header = [cell.strip().lower() for cell in rows[0]]
        try:
            cols = [header.index(name) for name in ('x', 'y', 'vp')]
        except ValueError:
            raise ValueError(
                f"{path}: the header must name the columns x, y and Vp; got "
                f"{rows[0]!r}.") from None
        try:
            points = np.array([[float(row[c]) for c in cols] for row in rows[1:]])
        except (ValueError, IndexError):
            raise ValueError(
                f"{path}: every row needs a number in the x, y and Vp columns.") from None
        if points.size == 0:
            raise ValueError(f"{path}: no data rows.")
        x, ix = np.unique(points[:, 0], return_inverse=True)
        y, iy = np.unique(points[:, 1], return_inverse=True)
        counts = np.zeros((x.size, y.size), dtype=int)
        np.add.at(counts, (ix, iy), 1)
        if (counts != 1).any():
            raise ValueError(
                f"{path}: the points must form a complete rectilinear grid "
                f"(every x with every y, once): {x.size} x values and {y.size} "
                f"y values need {x.size * y.size} rows, with no repeat; got "
                f"{points.shape[0]}.")
        values = np.empty((x.size, y.size))
        values[ix, iy] = points[:, 2]
        return cls(x, y, values)

    def to_dict(self) -> dict:
        """JSON-ready record: ``x``, ``y`` (mm) and ``Vp`` (nested lists)."""
        return {'x': self.x.tolist(), 'y': self.y.tolist(), 'Vp': self.Vp.tolist()}

    def __repr__(self) -> str:
        x0, x1, y0, y1 = self.extent
        return (f"PorosityMap({self.x.size} x {self.y.size} points, "
                f"x in [{x0:g}, {x1:g}], y in [{y0:g}, {y1:g}] mm, "
                f"mean Vp={self.mean:.4f}, max Vp={self.max:.4f})")
