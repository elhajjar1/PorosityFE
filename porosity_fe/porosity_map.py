"""In-plane porosity maps: the void content as a function of ``(x, y)``.

A :class:`PorosityMap` holds the through-thickness average void content on
a rectilinear grid, as an ultrasonic C-scan correlation or a CT slice stack
gives it, or as built from a background and a few :class:`PorosityPatch`
zones. :class:`~porosity_fe.PorosityField` takes one as its in-plane
variation (``PorosityField.from_map``, ``in_plane_map=``).
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
        # The integral of the bilinear interpolant: per cell, the mean of
        # its four corners times its area.
        v = self.Vp
        corners = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
        total = float((corners * np.diff(self.x)[:, None] * np.diff(self.y)[None, :]).sum())
        x0, x1, y0, y1 = self.extent
        return total / ((x1 - x0) * (y1 - y0))

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
