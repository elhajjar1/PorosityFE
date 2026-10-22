"""Explicit delamination defect: a crack at one ply interface.

A :class:`Delamination` is a zero-thickness crack in the plane of the
interface between two plies, with a planform in ``x``-``y``. It is a defect
separate from porosity: the porosity field is still an input of its own, and
a delamination neither adds nor removes voids.

:class:`~porosity_fe.CompositeMesh` represents it by duplicating the node
plane of the interface (``delaminations=``) and the solver bonds the
duplicate to the original node everywhere outside the planform with
master-slave ties; inside it the two crack faces move independently. See
the theory page "Delamination" for the representation, the boundary
conditions and the knockdown reference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Planform shapes of a :class:`Delamination`.
DelaminationShape = Literal['strip', 'rectangle', 'ellipse']

_SHAPES: tuple[str, ...] = ('strip', 'rectangle', 'ellipse')


def _finite_pair(name: str, value: object) -> tuple[float, float]:
    """``value`` as two finite floats, or a :class:`ValueError` naming ``name``."""
    try:
        a, b = (float(v) for v in value)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        raise ValueError(
            f"Delamination.{name} must be two numbers (mm), got {value!r}.") from None
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"Delamination.{name} must be finite (mm), got {value!r}.")
    return a, b


def _finite(name: str, value: object) -> float:
    """``value`` as a finite float, or a :class:`ValueError` naming ``name``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"Delamination.{name} must be a number (mm), got {value!r}.")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"Delamination.{name} must be finite (mm), got {value!r}.")
    return v


@dataclass(frozen=True)
class Delamination:
    """A delamination (interlaminar crack) at one ply interface.

    The crack lies in the plane of interface ``k``, between ply ``k`` and
    ply ``k + 1`` counted from the bottom face (``z = 0``), and covers a
    planform region in ``x``-``y``:

    - ``'strip'``: the band ``x_start < x < x_end`` through the full width
      (all ``y``), the insert of a fracture specimen (DCB, ENF);
    - ``'rectangle'``: ``|x - xc| < ax`` and ``|y - yc| < ay``;
    - ``'ellipse'``: ``((x - xc) / ax)^2 + ((y - yc) / ay)^2 < 1``,

    with ``(xc, yc) = center`` and ``(ax, ay) = half_size``. The region is
    open: a node of the interface strictly inside it is delaminated (its
    crack faces are free), and a node on its boundary stays bonded, as the
    crack front. A ``'strip'`` or ``'rectangle'`` edge that lies on or
    beyond an edge of the plate is not a front: the crack runs out to that
    free edge, and the nodes on the plate edge inside the planform are
    delaminated too (so a strip from ``x_start = 0`` opens at the end of
    the specimen). Several delaminations at one interface act as their
    union. The crack tip is at the first bonded nodes, so a boundary
    between two node lines acts at the next node line outside the region;
    put the boundary on a node line for an exact crack size.

    Build one with :meth:`strip`, :meth:`rectangle` or :meth:`ellipse`, or
    pass the fields directly. Lengths are in mm, in the coordinates of
    :class:`~porosity_fe.CompositeMesh` (the plate spans ``[0, L_x] x
    [0, L_y]``). The mesh checks what needs the plate: ``interface`` below
    ``n_plies - 1``, a region that overlaps the plate, and ``nz`` a
    multiple of ``n_plies`` (so every interface is a node layer).

    Parameters
    ----------
    interface : int
        Interface index ``k >= 0``: the crack lies between ply ``k`` and
        ply ``k + 1`` (0-based, from the bottom). The mid-plane of an
        ``n``-ply laminate (``n`` even) is ``k = n // 2 - 1``.
    shape : {'strip', 'rectangle', 'ellipse'}
        Planform shape, default ``'strip'``.
    x_start, x_end : float, optional
        ``'strip'`` only: the ``x`` range of the crack (mm), finite,
        ``x_end > x_start``. Either end may lie outside the plate.
    center : tuple of float, optional
        ``'rectangle'`` and ``'ellipse'`` only: the centre ``(xc, yc)``
        (mm), finite.
    half_size : tuple of float, optional
        ``'rectangle'`` and ``'ellipse'`` only: the half-sizes (semi-axes
        of the ellipse) ``(ax, ay)`` along ``x`` and ``y`` (mm), positive
        and finite.

    Raises
    ------
    ValueError
        If ``interface`` is not a non-negative integer, ``shape`` is
        unknown, a length is missing for the shape, given for another
        shape, not finite, or a size is not positive.

    See Also
    --------
    CompositeMesh : ``delaminations=`` duplicates the interface nodes.

    Examples
    --------
    A 21 mm insert at the mid-plane of a 16-ply double-cantilever beam,
    from the loaded end:

    >>> d = Delamination.strip(7, x_start=0.0, x_end=21.0)
    >>> d.shape, d.interface
    ('strip', 7)

    A 10 mm x 6 mm elliptical delamination (an NDI indication) under the
    first ply of a coupon:

    >>> d = Delamination.ellipse(0, center=(25.0, 10.0), half_size=(5.0, 3.0))
    >>> bool(d.contains(25.0, 10.0)), bool(d.contains(30.0, 10.0))
    (True, False)
    """

    interface: int
    shape: DelaminationShape = 'strip'
    x_start: float | None = None
    x_end: float | None = None
    center: tuple[float, float] | None = None
    half_size: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        k = self.interface
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 0:
            raise ValueError(
                f"Delamination.interface must be an integer >= 0 (the "
                f"interface between ply k and ply k + 1, from the bottom), "
                f"got {k!r}.")
        object.__setattr__(self, 'interface', int(k))
        if self.shape not in _SHAPES:
            raise ValueError(
                f"Delamination.shape must be one of {list(_SHAPES)}, got "
                f"{self.shape!r}.")
        if self.shape == 'strip':
            if self.center is not None or self.half_size is not None:
                raise ValueError(
                    "A 'strip' delamination is set by x_start and x_end; "
                    "center and half_size are for 'rectangle' and 'ellipse'.")
            if self.x_start is None or self.x_end is None:
                raise ValueError(
                    "A 'strip' delamination needs x_start and x_end (mm).")
            x0, x1 = _finite('x_start', self.x_start), _finite('x_end', self.x_end)
            if not x1 > x0:
                raise ValueError(
                    f"Delamination x_end must be greater than x_start, got "
                    f"x_start={x0:g}, x_end={x1:g} mm.")
            object.__setattr__(self, 'x_start', x0)
            object.__setattr__(self, 'x_end', x1)
            return
        if self.x_start is not None or self.x_end is not None:
            raise ValueError(
                f"A {self.shape!r} delamination is set by center and "
                f"half_size; x_start and x_end are for 'strip'.")
        if self.center is None or self.half_size is None:
            raise ValueError(
                f"A {self.shape!r} delamination needs center (xc, yc) and "
                f"half_size (ax, ay) in mm.")
        center = _finite_pair('center', self.center)
        half = _finite_pair('half_size', self.half_size)
        if not (half[0] > 0.0 and half[1] > 0.0):
            raise ValueError(
                f"Delamination.half_size must be positive (mm), got {half}.")
        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'half_size', half)

    @classmethod
    def strip(cls, interface: int, x_start: float, x_end: float) -> Delamination:
        """Through-width delamination over ``x_start < x < x_end`` (mm)."""
        return cls(interface, 'strip', x_start=x_start, x_end=x_end)

    @classmethod
    def rectangle(cls, interface: int, center: Sequence[float],
                  half_size: Sequence[float]) -> Delamination:
        """Rectangular delamination, centre ``(xc, yc)`` and half-sizes
        ``(ax, ay)`` (mm)."""
        return cls(interface, 'rectangle', center=tuple(center),  # type: ignore[arg-type]
                   half_size=tuple(half_size))  # type: ignore[arg-type]

    @classmethod
    def ellipse(cls, interface: int, center: Sequence[float],
                half_size: Sequence[float]) -> Delamination:
        """Elliptical delamination, centre ``(xc, yc)`` and semi-axes
        ``(ax, ay)`` (mm)."""
        return cls(interface, 'ellipse', center=tuple(center),  # type: ignore[arg-type]
                   half_size=tuple(half_size))  # type: ignore[arg-type]

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the planform (mm); a strip is
        unbounded in ``y``."""
        if self.shape == 'strip':
            assert self.x_start is not None and self.x_end is not None
            return (self.x_start, self.x_end, -math.inf, math.inf)
        assert self.center is not None and self.half_size is not None
        (xc, yc), (ax, ay) = self.center, self.half_size
        return (xc - ax, xc + ax, yc - ay, yc + ay)

    def contains(self, x: np.ndarray | float, y: np.ndarray | float, *,
                 plate: tuple[float, float, float, float] | None = None,
                 tol: float = 0.0) -> np.ndarray:
        """Whether the points ``(x, y)`` lie strictly inside the planform.

        Parameters
        ----------
        x, y : array_like
            In-plane coordinates (mm), broadcast together.
        plate : tuple of float, optional
            ``(x_min, x_max, y_min, y_max)`` of the plate. A ``'strip'`` or
            ``'rectangle'`` edge within ``tol`` of a plate edge, or beyond
            it, is then open: points on that plate edge count as inside.
            Ignored for an ``'ellipse'``, whose boundary is never along a
            plate edge.
        tol : float
            Distance (mm) within which a point counts as on the boundary,
            and so not inside, default 0.

        Returns
        -------
        np.ndarray
            Boolean array of the broadcast shape of ``x`` and ``y``.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.shape == 'ellipse':
            assert self.center is not None and self.half_size is not None
            (xc, yc), (ax, ay) = self.center, self.half_size
            r2 = ((x - xc) / ax) ** 2 + ((y - yc) / ay) ** 2
            return r2 < (1.0 - tol / max(ax, ay)) ** 2
        x_lo, x_hi, y_lo, y_hi = self.bounding_box
        if plate is not None:
            px0, px1, py0, py1 = plate
            x_lo = -math.inf if x_lo <= px0 + tol else x_lo
            x_hi = math.inf if x_hi >= px1 - tol else x_hi
            y_lo = -math.inf if y_lo <= py0 + tol else y_lo
            y_hi = math.inf if y_hi >= py1 - tol else y_hi
        return ((x > x_lo + tol) & (x < x_hi - tol)
                & (y > y_lo + tol) & (y < y_hi - tol))

    def overlaps_plate(self, plate: tuple[float, float, float, float], *,
                       tol: float = 0.0) -> bool:
        """Whether the planform overlaps the plate ``(x_min, x_max, y_min,
        y_max)`` over a positive area (more than ``tol`` mm deep)."""
        px0, px1, py0, py1 = plate
        x_lo, x_hi, y_lo, y_hi = self.bounding_box
        if not (x_lo < px1 - tol and x_hi > px0 + tol
                and y_lo < py1 - tol and y_hi > py0 + tol):
            return False
        if self.shape != 'ellipse':
            return True
        # Scaling x by 1/ax and y by 1/ay maps the ellipse onto the unit
        # circle and the plate onto another axis-aligned rectangle, and
        # clamping commutes with it: the plate point nearest the centre is
        # the clamped centre. The regions overlap when it lies inside.
        assert self.center is not None
        xc, yc = self.center
        px = min(max(xc, px0 + tol), px1 - tol)
        py = min(max(yc, py0 + tol), py1 - tol)
        return bool(self.contains(px, py))
