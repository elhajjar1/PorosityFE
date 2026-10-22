"""ASTM D2344 short-beam shear: the ``'ilss'`` loading mode's geometry and outputs.

:meth:`FESolver.solve(loading='ilss') <porosity_fe.FESolver.solve>` runs the
short-beam-strength test of ASTM D2344/D2344M on a specimen mesh from
:meth:`CompositeMesh.for_ilss <porosity_fe.CompositeMesh.for_ilss>`: roller
supports ``4 h`` apart under a ``6 h x 2 h`` specimen and the loading nose
as a uniform pressure over a contact band at mid-span (see
:class:`~porosity_fe.FlexureGeometry`). This module holds what is specific
to that test:

* :func:`require_ilss_geometry`: the specimen's supports and band, or a
  :class:`ValueError` with guidance for a mesh that is not a short-beam
  specimen (the solver does not move the supports onto some other mesh);
* :func:`interior_element_mask`: the region the solve's interior failure
  quantities use, the shear spans away from the contact zones;
* :func:`short_beam_shear`: the D2344 apparent interlaminar shear
  ``F_sbs = 0.75 P / (b h)`` next to the FE shear stress and the
  first-ply-failure load.

At a span of four thicknesses beam theory is approximate. The
section-average shear force is exactly ``P / 2``, but the shear-stress
profile is not yet the parabola behind ``0.75 P / (b h)``: on the pristine
24-ply T800 specimen the width-averaged mid-plane ``tau_xz`` half-way
between a support and the nose is 0.91 of it for UD and 0.92 for the
``'QI'`` layup, against 0.98 / 0.96 at a span of eight thicknesses and
0.997 / 0.968 at sixteen (the QI limit is its lamination-theory profile).
The ratio depends on the layup and, through the degraded ply stiffness, on
the porosity. Next to the nose and the supports the contact stresses
dominate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..mesh import CompositeMesh, FlexureGeometry
from .flexure import _structured_grid

if TYPE_CHECKING:
    from .results import FieldResults


def require_ilss_geometry(mesh: CompositeMesh) -> FlexureGeometry:
    """The short-beam supports and loading band of ``mesh``.

    :attr:`CompositeMesh.ilss_geometry`, which
    :meth:`CompositeMesh.for_ilss` sets.

    Raises
    ------
    ValueError
        If the mesh has no ILSS geometry. The solver does not snap the
        supports to node lines of another mesh: on the 30 x 10 production
        plate the nearest node lines give a span of 4.17 thicknesses and a
        knockdown 6 % of the stiffness loss off the D2344 specimen's, and
        the snapped span would change with the mesh.
    """
    geometry = getattr(mesh, 'ilss_geometry', None)
    if geometry is not None:
        return geometry
    h = float(mesh.L_z)
    raise ValueError(
        f"loading='ilss' is the ASTM D2344 short-beam-shear test: roller "
        f"supports 4 h apart under a 6 h x 2 h specimen (h = {h:.4g} mm), "
        f"loaded at mid-span. This {mesh.L_x:.4g} x {mesh.L_y:.4g} mm mesh has "
        f"no short-beam geometry (CompositeMesh.ilss_geometry is None). Build "
        f"the specimen with CompositeMesh.for_ilss(porosity_field, material), "
        f"or set mesh.ilss_geometry = FlexureGeometry(...) with the supports on "
        f"node lines. For a three-point bend of this coupon use "
        f"loading='flexure_3pt'.")


def default_interior_margin(mesh: CompositeMesh) -> float:
    """Default contact-zone margin of the short-beam interior region (mm): ``h / 2``."""
    return 0.5 * float(mesh.L_z)


def interior_element_mask(mesh: CompositeMesh, geometry: FlexureGeometry,
                          margin: float | None = None) -> np.ndarray:
    """Elements in the short-beam shear spans, away from the contact zones.

    An element is interior when its centroid lies between the supports, at
    least ``margin`` from both support lines and from the edges of the
    loading band (``|x - x_mid| >= load_width / 2 + margin``), and at least
    ``margin`` from the two side faces (``y``). The overhangs are left out.
    The default margin is ``h / 2`` (``h`` the laminate thickness): the line
    supports and the nose are singular, and their stress concentration
    decays over about that distance. The specimen is only ``2 h`` wide, so
    the interior keeps the middle half of the width.

    Parameters
    ----------
    mesh : CompositeMesh
        The short-beam mesh.
    geometry : FlexureGeometry
        Its supports and band (:func:`require_ilss_geometry`).
    margin : float, optional
        Distance in mm, finite and ``>= 0``; default ``h / 2``.

    Returns
    -------
    np.ndarray
        Boolean mask ``(n_elem,)``.

    Raises
    ------
    ValueError
        If ``margin`` is negative or not finite.
    """
    if margin is not None and (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float, np.number))
            or not math.isfinite(float(margin)) or float(margin) < 0.0):
        raise ValueError(
            f"interior_margin must be a finite length >= 0 in mm, got {margin!r}.")
    d = default_interior_margin(mesh) if margin is None else float(margin)
    nodes = np.asarray(mesh.nodes, dtype=float)
    cent = nodes[mesh.elements].mean(axis=1)
    x, y = cent[:, 0], cent[:, 1]
    x0, x1 = geometry.support_x
    y_lo, y_hi = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    return ((x >= x0 + d) & (x <= x1 - d)
            & (np.abs(x - geometry.midspan) >= 0.5 * geometry.load_width + d)
            & (y >= y_lo + d) & (y <= y_hi - d))


@dataclass(frozen=True)
class ShortBeamShear:
    """ASTM D2344 short-beam-shear quantities of an ``'ilss'`` solve.

    Lengths in mm, forces in N, stresses in MPa. See
    :func:`short_beam_shear`.

    Attributes
    ----------
    span, width, thickness : float
        Support span ``L``, specimen width ``b`` and thickness ``h``.
    load_width : float
        Width of the loading-nose band (``0`` for a line load).
    applied_load : float
        Total load of the solve, signed (negative downward), from the
        support reactions.
    apparent_shear : float
        The D2344 short-beam formula at the applied load,
        ``0.75 |P| / (b h)``: the peak of the parabolic beam-theory shear
        profile for a shear force ``P / 2``.
    midplane_shear : float
        FE ``|tau_xz|`` at mid-thickness, averaged across the width, at the
        quarter-span points (half-way between each support and the nose),
        mean of the two shear spans. It interpolates the element means of
        the two element layers next to the mid-plane, so even an exact
        parabola reads ``1 - (4/3) (dz/h)^2`` of its peak (``dz`` the layer
        thickness: 0.998 at one layer per ply of 24 plies, 0.979 of 8).
        Below ``apparent_shear`` also because at four thicknesses the
        profile is flatter than the parabola: on the pristine 24-ply T800
        specimen (``nz = 24``) 0.91 of it for UD and 0.92 for the ``'QI'``
        layup (layup- and porosity-dependent).
    interior_peak_shear : float or None
        Largest Gauss-point ``|tau_xz|`` of the interior elements
        (:func:`interior_element_mask`); ``None`` when there are none.
    first_ply_failure_load : float
        Load magnitude at first-ply failure anywhere in the model,
        ``first_ply_failure_load_factor * |P|``. It sits under the nose or
        at a support, where the stress is singular, so it depends on the
        mesh.
    interior_first_ply_failure_load : float or None
        The same over the interior elements of the solve
        (``interior_first_ply_failure_load_factor * |P|``).
    apparent_strength : float
        ``0.75 * first_ply_failure_load / (b h)``: the short-beam strength
        D2344 would report if the specimen failed at first-ply failure.
    interior_apparent_strength : float or None
        The same at the interior first-ply-failure load.
    """

    span: float
    width: float
    thickness: float
    load_width: float
    applied_load: float
    apparent_shear: float
    midplane_shear: float
    interior_peak_shear: float | None
    first_ply_failure_load: float
    interior_first_ply_failure_load: float | None
    apparent_strength: float
    interior_apparent_strength: float | None


def _midplane_shear_at(mesh: CompositeMesh, tau_xz: np.ndarray, x: float) -> float:
    """Width-averaged element-mean ``tau_xz`` at mid-thickness and ``x``.

    ``tau_xz`` is ``(n_elem,)`` element means. They are averaged across the
    width (element-width weights), then interpolated linearly between
    element-layer centroids to mid-thickness and between element-column
    centroids to ``x``. (The Gauss-point values of a layer do not vary
    through it in this bending field, so they add nothing here.)
    """
    xg, yg, zg = _structured_grid(mesh)
    t = tau_xz.reshape(mesh.nz, mesh.ny, mesh.nx)
    wy = np.diff(yg) / float(yg[-1] - yg[0])
    t_zx = np.einsum('kji,j->ki', t, wy)
    zc, xc = 0.5 * (zg[:-1] + zg[1:]), 0.5 * (xg[:-1] + xg[1:])
    z_mid = 0.5 * (zg[0] + zg[-1])
    t_x = np.array([np.interp(z_mid, zc, column) for column in t_zx.T])
    return float(np.interp(float(x), xc, t_x))


def short_beam_shear(results: FieldResults, mesh: CompositeMesh, *,
                     interior_margin: float | None = None) -> ShortBeamShear:
    """D2344 apparent interlaminar shear and the FE values beside it.

    For the result of ``FESolver(mesh, ...).solve(loading='ilss')`` on a
    short-beam specimen (:meth:`CompositeMesh.for_ilss`). The load ``P`` is
    recovered from the support reactions, so the result of any
    ``applied_load`` works. Reports the D2344 short-beam formula
    ``0.75 |P| / (b h)``, the FE mid-plane shear at the quarter-span points
    and its interior peak, and the first-ply-failure load converted to an
    apparent short-beam strength.

    Parameters
    ----------
    results : FieldResults
        The ``'ilss'`` solve on ``mesh``.
    mesh : CompositeMesh
        The specimen mesh, with :attr:`CompositeMesh.ilss_geometry`.
    interior_margin : float, optional
        Margin (mm) of the region for ``interior_peak_shear``, as the
        ``interior_margin`` of :meth:`FESolver.solve`; default ``h / 2``.
        The interior first-ply-failure load comes from ``results``, so pass
        the value used for the solve.

    Returns
    -------
    ShortBeamShear

    Raises
    ------
    ValueError
        If the mesh has no ILSS geometry, the results do not belong to the
        mesh, they carry no reactions, or the load is zero.
    """
    geometry = require_ilss_geometry(mesh)
    n_elem = mesh.n_elements
    stress = np.asarray(results.stress_global)
    if np.shape(results.displacement) != (mesh.n_nodes, 3) or stress.ndim != 3 \
            or stress.shape[0] != n_elem:
        raise ValueError(
            f"The results ({np.shape(results.displacement)[0]} nodes, "
            f"{stress.shape[0]} elements) do not belong to this mesh "
            f"({mesh.n_nodes} nodes, {n_elem} elements).")
    if results.reaction_forces is None:
        raise ValueError("short_beam_shear needs the support reactions "
                         "(FieldResults.reaction_forces) to recover the load.")
    load = -float(np.asarray(results.reaction_forces)[:, 2].sum())
    if load == 0.0 or not math.isfinite(load):
        raise ValueError(f"The solve carries no load (sum of the z reactions "
                         f"{-load!r} N).")
    b, h = float(mesh.L_y), float(mesh.L_z)
    P = abs(load)
    x0, x1 = geometry.support_x
    tau_elem = stress[..., 4].mean(axis=1)
    quarter = 0.25 * geometry.span
    midplane = 0.5 * (abs(_midplane_shear_at(mesh, tau_elem, x0 + quarter))
                      + abs(_midplane_shear_at(mesh, tau_elem, x1 - quarter)))
    inner = interior_element_mask(mesh, geometry, interior_margin)
    peak = float(np.abs(stress[inner, :, 4]).max()) if inner.any() else None

    def load_at(factor: float | None) -> float | None:
        return None if factor is None else float(factor) * P

    fpf = load_at(results.first_ply_failure_load_factor)
    fpf_load = float('inf') if fpf is None else fpf
    int_load = load_at(results.interior_first_ply_failure_load_factor)
    return ShortBeamShear(
        span=float(geometry.span), width=b, thickness=h,
        load_width=float(geometry.load_width), applied_load=load,
        apparent_shear=0.75 * P / (b * h),
        midplane_shear=midplane,
        interior_peak_shear=peak,
        first_ply_failure_load=fpf_load,
        interior_first_ply_failure_load=int_load,
        apparent_strength=0.75 * fpf_load / (b * h),
        interior_apparent_strength=None if int_load is None
        else 0.75 * int_load / (b * h),
    )
