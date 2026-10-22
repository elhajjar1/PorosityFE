"""Flexural modulus of the three- and four-point bending FE models.

:meth:`FESolver.solve` with ``loading='flexure_3pt'`` or ``'flexure_4pt'``
reports :attr:`FieldResults.effective_modulus` as the flexural modulus
``E_f`` that Timoshenko beam theory needs to reproduce the FE mid-span
deflection, with the transverse shear deflection taken out:

    delta / P = a (3 L^2 - 4 a^2) / (48 E_f I) + a / (2 b K_s),
    I = b h^3 / 12,

with ``L`` the support span, ``a`` the shear span (``L / 2`` for three-point
loading, ``(L - load_span) / 2`` for four-point), ``b`` and ``h`` the width
and thickness, and ``K_s`` the laminate transverse shear stiffness per unit
width. ``delta`` is the width-averaged mid-thickness deflection at
mid-span relative to the mean over the two support lines, which removes
the local indentation at the supports.

``K_s`` follows from the energy of the transverse shear stresses that
equilibrium gives for the classical-lamination-theory bending stresses of a
narrow beam (``N = 0``, ``M_y = M_xy = 0``), using the element layers'
porosity-degraded stiffness averaged over the span. For one orthotropic
material it is the familiar ``(5/6) G_13 h``.

:func:`interior_element_mask` is the region the interior failure metrics
of the flexure modes are taken over: the span clear of the contact zones
at the supports and under the loads, and of the side faces.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

from ..mesh import CompositeMesh, FlexureGeometry
from .batch import ElementBatch

logger = logging.getLogger("porosity_fe_analysis")

# Three-point Gauss-Legendre rule on [-1, 1]: exact for the quartic shear
# energy density within a layer.
_GAUSS3_X = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GAUSS3_W = np.array([5.0, 8.0, 5.0]) / 9.0


def _structured_grid(mesh: CompositeMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node coordinates along ``x``, ``y`` and ``z`` of the structured mesh."""
    nx1, ny1, nz1 = mesh.nx + 1, mesh.ny + 1, mesh.nz + 1
    nodes = np.asarray(mesh.nodes, dtype=float)
    if nodes.shape[0] != nx1 * ny1 * nz1:
        raise ValueError(
            "The flexural modulus needs the structured CompositeMesh node "
            f"layout ({nx1} x {ny1} x {nz1} nodes), got {nodes.shape[0]} nodes.")
    return (nodes[:nx1, 0], nodes[:ny1 * nx1:nx1, 1], nodes[::ny1 * nx1, 2])


def section_deflection(mesh: CompositeMesh, u: np.ndarray, x: float) -> float:
    """Width-averaged mid-thickness ``u_z`` at ``x`` (mm).

    Averages ``u_z`` across the width with trapezoidal weights (the exact
    mean of the bilinear interpolant), then interpolates linearly to
    mid-thickness and to ``x``.
    """
    xg, yg, zg = _structured_grid(mesh)
    uz = np.asarray(u, dtype=float)[2::3].reshape(zg.size, yg.size, xg.size)
    wy = np.zeros(yg.size)
    if yg.size == 1:
        wy[0] = 1.0
    else:
        seg = np.diff(yg)
        wy[:-1] += 0.5 * seg
        wy[1:] += 0.5 * seg
        wy /= wy.sum()
    w_zx = np.einsum('kji,j->ki', uz, wy)
    z_mid = 0.5 * (zg[0] + zg[-1])
    w_x = np.array([np.interp(z_mid, zg, column) for column in w_zx.T])
    return float(np.interp(float(x), xg, w_x))


def _layer_stiffness(mesh: CompositeMesh, batch: ElementBatch,
                     x_lo: float, x_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Volume-averaged stiffness ``(nz, 6, 6)`` of each element layer between
    ``x_lo`` and ``x_hi``, and the layer boundaries ``(nz + 1,)``."""
    _, _, zg = _structured_grid(mesh)
    n_layer = mesh.nx * mesh.ny
    centre_x = np.asarray(mesh.nodes, dtype=float)[mesh.elements, 0].mean(axis=1)
    inside = (centre_x >= x_lo) & (centre_x <= x_hi)
    w = batch.detJ_w * inside[:, None]
    C_sum = np.einsum('eg,egij->eij', w, batch.C).reshape(mesh.nz, n_layer, 6, 6)
    vol = w.sum(axis=1).reshape(mesh.nz, n_layer).sum(axis=1)
    return C_sum.sum(axis=1) / vol[:, None, None], zg


def transverse_shear_stiffness(C_layers: np.ndarray, z_edges: np.ndarray) -> float:
    """Laminate transverse shear stiffness per unit width ``K_s`` (N/mm).

    For a beam bent about ``y`` with shear force ``V`` per unit width,
    equilibrium of the lamination-theory in-plane stresses gives the
    transverse shear stresses ``tau_xz(z) = -V int s_x dz`` and
    ``tau_yz(z) = -V int s_xy dz``, where ``s`` is the in-plane stress per
    unit moment ``M_x`` (narrow beam: ``N = 0`` and ``M_y = M_xy = 0``).
    Equating their complementary energy ``0.5 V^2 int t^T S t dz`` (``S``
    the inverse of each layer's transverse-shear stiffness block) to
    ``0.5 V^2 / K_s`` gives ``K_s``; for one orthotropic material it is
    ``(5/6) G_13 h``.

    Parameters
    ----------
    C_layers : np.ndarray
        ``(n, 6, 6)`` stiffness of each layer in the beam axes (MPa), Voigt
        order ``[11, 22, 33, 23, 13, 12]``, bottom to top.
    z_edges : np.ndarray
        ``(n + 1,)`` layer boundaries (mm), increasing.
    """
    C = np.asarray(C_layers, dtype=float)
    ip = [0, 1, 5]
    c3 = C[:, ip, 2]
    Q = C[:, ip][:, :, ip] - c3[:, :, None] * c3[:, None, :] / C[:, 2, 2][:, None, None]
    z = np.asarray(z_edges, dtype=float)
    z = z - 0.5 * (z[0] + z[-1])
    zl, zh = z[:-1], z[1:]
    A = np.einsum('k,kij->ij', zh - zl, Q)
    B = np.einsum('k,kij->ij', 0.5 * (zh ** 2 - zl ** 2), Q)
    D = np.einsum('k,kij->ij', (zh ** 3 - zl ** 3) / 3.0, Q)
    abd_inv = np.linalg.inv(np.block([[A, B], [B, D]]))
    eps0, kappa = abd_inv[:3, 3], abd_inv[3:, 3]     # per unit M_x
    s0, s1 = Q @ eps0, Q @ kappa                     # s(z) = s0 + z s1
    increment = (s0 * (zh - zl)[:, None]
                 + s1 * (0.5 * (zh ** 2 - zl ** 2))[:, None])
    F_bottom = np.vstack([np.zeros(3), np.cumsum(increment, axis=0)[:-1]])
    zq = 0.5 * (zl + zh)[:, None] + 0.5 * (zh - zl)[:, None] * _GAUSS3_X
    Fq = (F_bottom[:, None, :]
          + s0[:, None, :] * (zq - zl[:, None])[..., None]
          + s1[:, None, :] * (0.5 * (zq ** 2 - zl[:, None] ** 2))[..., None])
    t = np.stack([Fq[..., 2], Fq[..., 0]], axis=-1)  # (tau_yz, tau_xz) / V
    S = np.linalg.inv(C[:, 3:5, 3:5])
    weights = 0.5 * (zh - zl)[:, None] * _GAUSS3_W
    phi = float(np.einsum('kq,kqi,kij,kqj->', weights, t, S, t))
    return 1.0 / phi


def flexural_modulus(mesh: CompositeMesh, batch: ElementBatch, u: np.ndarray,
                     applied_load: float, geometry: FlexureGeometry,
                     procedure: Literal['3pt', '4pt']) -> float | None:
    """Shear-corrected flexural modulus ``E_f`` (MPa) of a flexure solve.

    See the module docstring for the formula. ``None`` for a zero load, or
    (with a warning) when the shear correction leaves no bending
    deflection.
    """
    if applied_load == 0.0:
        return None
    x0, x1 = geometry.support_x
    L = geometry.span
    a = geometry.shear_span(procedure)
    b, h = float(mesh.L_y), float(mesh.L_z)
    delta = section_deflection(mesh, u, geometry.midspan) - 0.5 * (
        section_deflection(mesh, u, x0) + section_deflection(mesh, u, x1))
    K_s = transverse_shear_stiffness(*_layer_stiffness(mesh, batch, x0, x1))
    bending = delta / applied_load - a / (2.0 * b * K_s)
    if not (np.isfinite(bending) and bending > 0.0):
        logger.warning(
            "Flexural modulus not reported: the mid-span deflection (%.4g mm "
            "at %.4g N) leaves no bending part after the shear correction "
            "(%.4g mm). Check the supports and the load.",
            delta, applied_load, applied_load * a / (2.0 * b * K_s))
        return None
    return float(a * (3.0 * L ** 2 - 4.0 * a ** 2) / (4.0 * b * h ** 3 * bending))


def default_interior_margin(mesh: CompositeMesh) -> float:
    """Default contact-zone margin of the flexure and short-beam interior
    regions (mm): ``h / 2``, with ``h`` the laminate thickness."""
    return 0.5 * float(mesh.L_z)


def interior_element_mask(mesh: CompositeMesh, geometry: FlexureGeometry,
                          procedure: Literal['3pt', '4pt'],
                          margin: float | None = None) -> np.ndarray:
    """Elements of a flexure test away from the contact zones and side faces.

    An element is interior when its centroid lies between the supports, at
    least ``margin`` from the side faces (``y``) and at least ``margin``
    from every contact zone, the distance measured in the ``x``-``z``
    plane: from each support line on the bottom face and from each load
    band (``geometry.load_width`` wide, centred on
    :meth:`FlexureGeometry.load_x`) on the top face. The overhangs are
    left out. The default margin is ``h / 2``, as for the short-beam test
    (:func:`porosity_fe.fe.short_beam.interior_element_mask`).

    Unlike the short-beam region, which drops whole cross-sections next to
    the nose to keep the shear spans, only a half-disc around each contact
    line is left out. In three-point bending the bending stress peaks at
    mid-span, under the load: the bottom (tension) face there stays
    interior, and the top face is interior from ``margin`` either side of
    the load. On the default 24-ply T800 QI specimen the interior maximum
    failure index is 0.99 of the domain maximum (0.95 with whole sections
    dropped); the rest is the free edge.

    Parameters
    ----------
    mesh : CompositeMesh
        The flexure mesh.
    geometry : FlexureGeometry
        Its supports and loads
        (:meth:`~porosity_fe.BoundaryHandler.flexure_geometry`).
    procedure : {'3pt', '4pt'}
        Three- or four-point loading: which load lines are contacts.
    margin : float, optional
        Distance in mm, finite and ``>= 0``; default ``h / 2``.

    Returns
    -------
    np.ndarray
        Boolean mask ``(n_elem,)``.

    Raises
    ------
    ValueError
        If ``margin`` is negative or not finite, or ``procedure`` is
        unknown.
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
    x, y, z = cent[:, 0], cent[:, 1], cent[:, 2]
    z_bot, z_top = float(nodes[:, 2].min()), float(nodes[:, 2].max())
    y_lo, y_hi = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    x0, x1 = geometry.support_x
    mask = (x >= x0) & (x <= x1) & (y >= y_lo + d) & (y <= y_hi - d)
    for xs in (x0, x1):
        mask &= np.hypot(x - xs, z - z_bot) >= d
    half = 0.5 * geometry.load_width
    for xl in geometry.load_x(procedure):
        mask &= np.hypot(np.maximum(np.abs(x - xl) - half, 0.0), z - z_top) >= d
    return mask
