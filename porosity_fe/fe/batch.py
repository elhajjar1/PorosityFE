"""Vectorized per-element, per-Gauss-point quantities for the whole mesh.

:class:`Hex8Element` evaluates one element at a time in Python loops. The
assembler and stress recovery instead need the same quantities for every
element, so :func:`build_element_batch` computes them for all elements at
once with the same formulas:

- ``B`` (strain-displacement) and ``det(J)`` from the shared shape-function
  derivatives, with batched Jacobian inverses;
- the porosity-degraded, ply-rotated stiffness ``C`` at each Gauss point,
  evaluated once per distinct ``(Vp, ply angle)`` pair (a structured mesh has
  only a few dozen) and gathered back per point.

The batch is built during assembly and reused by stress recovery, which
previously re-created every element and recomputed all of this.

For the incompatible-mode formulation (``formulation='hex8i'``) the nine
internal modes of each element are condensed out here, once, and folded
into an *effective* strain-displacement operator
``B_eff = B + G H`` with ``H = -Kaa^-1 Kau``. Stiffness
(``sum B_eff^T C B_eff det(J) w = Kuu - Kua Kaa^-1 Kau``, exactly), strain
and stress recovery, failure evaluation and export then run unchanged on
``ElementBatch.B``.

Thermal loading (IMPROVEMENT_PLAN 3.5) adds an initial strain
``alpha dT``. :func:`thermal_stress_moduli` gives ``beta = C alpha`` per
Gauss point, :meth:`ElementBatch.thermal_loads` the element load vectors
``sum B^T beta dT det(J) w`` and :func:`assemble_load_vector` scatters
them. For ``'hex8i'`` the incompatible modes are loaded too:
``ElementBatch.B`` already holds ``B + G H``, so the same sum is the
condensed load ``f_u + H^T f_a``, and
:meth:`ElementBatch.incompatible_mode_strains` adds the strain of the
internal-mode amplitudes the load drives directly,
``G Kaa^-1 f_a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._types import FEFormulation
from ..gauss import gauss_points_hex
from ..homogenization import _degraded_composite_stiffness
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..transforms import rotate_stiffness_3d, strain_transformation_3d
from .element import (
    _DEFAULT_FORMULATION,
    VP_STIFFNESS_CLAMP,
    Hex8Element,
    _check_formulation,
    _condensation_operator,
    _incompatible_mode_derivatives,
    _strain_operator,
)


@dataclass(frozen=True)
class ElementBatch:
    """Per-element, per-Gauss-point FE quantities for a whole mesh.

    Shapes use ``E`` elements and ``G = 8`` Gauss points.

    Attributes
    ----------
    dofs : np.ndarray
        ``(E, 24)`` global DOF indices, node-major ``[u1x, u1y, u1z, ...]``.
    B : np.ndarray
        ``(E, G, 6, 24)`` strain-displacement matrices (engineering shear).
        For ``formulation='hex8i'`` this is the effective operator
        ``B + G H`` with the statically condensed incompatible modes folded
        in (see :meth:`Hex8Element.strain_operator`).
    C : np.ndarray
        ``(E, G, 6, 6)`` degraded, ply-rotated stiffness (MPa).
    detJ_w : np.ndarray
        ``(E, G)`` Jacobian determinant times quadrature weight.
    formulation : {'hex8i', 'hex8'}
        Element formulation ``B`` was built for (default ``'hex8i'``).
    G_inc : np.ndarray or None
        ``(E, G, 6, 9)`` incompatible-mode strain operators for
        ``'hex8i'`` (``None`` for ``'hex8'``). Kept for loads that act on
        the internal modes directly (thermal strain); the stiffness needs
        only ``B``.
    Kaa : np.ndarray or None
        ``(E, 9, 9)`` internal-mode stiffness ``sum G^T C G det(J) w`` for
        ``'hex8i'`` (``None`` for ``'hex8'``).
    """

    dofs: np.ndarray
    B: np.ndarray
    C: np.ndarray
    detJ_w: np.ndarray
    formulation: FEFormulation = _DEFAULT_FORMULATION
    G_inc: np.ndarray | None = None
    Kaa: np.ndarray | None = None

    def stiffness_matrices(self) -> np.ndarray:
        """Symmetrized element stiffness matrices ``(E, 24, 24)``.

        ``Ke = sum_g B^T C B det(J) w``, then ``0.5 (Ke + Ke^T)`` as the
        assembler has always applied (issue #57).
        """
        CB = np.matmul(self.C, self.B)
        with np.errstate(over='ignore', invalid='ignore'):
            Ke = np.einsum('egki,egkj,eg->eij', self.B, CB, self.detJ_w)
        return 0.5 * (Ke + Ke.transpose(0, 2, 1))

    def strains(self, u: np.ndarray) -> np.ndarray:
        """Global engineering strain ``(E, G, 6)`` from a displacement vector."""
        return np.einsum('egij,ej->egi', self.B, u[self.dofs])

    def stresses(self, strain: np.ndarray) -> np.ndarray:
        """Global stress ``(E, G, 6)`` from :meth:`strains` output."""
        return np.einsum('egij,egj->egi', self.C, strain)

    def thermal_loads(self, beta: np.ndarray) -> np.ndarray:
        """Element load vectors ``(E, 24)`` of an initial (thermal) strain.

        ``f_e = sum_g B^T beta det(J) w``, where ``beta = C alpha dT`` is
        the stress the initial strain ``alpha dT`` would cause if fully
        restrained (``(E, G, 6)``, see :func:`thermal_stress_moduli`).
        For ``'hex8i'``, ``B`` is the condensed ``B + G H`` and this is
        exactly ``f_u + H^T f_a`` with ``f_a = sum G^T beta det(J) w``,
        the load on the incompatible modes condensed onto the nodes.
        """
        return np.einsum('egki,egk,eg->ei', self.B, beta, self.detJ_w)

    def incompatible_mode_strains(self, beta: np.ndarray) -> np.ndarray:
        """Strain ``(E, G, 6)`` of the internal modes an initial strain drives.

        With an initial-strain load the condensed internal-mode amplitudes
        are ``a = H u_e + Kaa^-1 f_a``, so the strain is
        ``B u_e + G a = B_eff u_e + G Kaa^-1 f_a``; this returns the second
        term (zeros for ``'hex8'``). ``f_a`` vanishes when ``beta`` is
        uniform over an element (Taylor's correction makes ``G`` integrate
        to zero), so it matters only where the stiffness varies inside an
        element, e.g. graded porosity.
        """
        if self.G_inc is None or self.Kaa is None:
            return np.zeros(beta.shape, dtype=float)
        f_a = np.einsum('egki,egk,eg->ei', self.G_inc, beta, self.detJ_w)
        # Same fallback as the stiffness condensation: an element whose Kaa
        # is singular or non-finite behaves as a plain hex8 (amplitude 0).
        a = _condensation_operator(self.Kaa, -f_a[:, :, None])[:, :, 0]
        return np.einsum('egij,ej->egi', self.G_inc, a)


def assemble_load_vector(batch: ElementBatch, element_loads: np.ndarray,
                         n_dof: int) -> np.ndarray:
    """Scatter-add element load vectors ``(E, 24)`` into a global ``(n_dof,)``."""
    return np.bincount(batch.dofs.ravel(), weights=element_loads.ravel(),
                       minlength=n_dof)


def global_cte(alpha_local: np.ndarray, angle_deg: float) -> np.ndarray:
    """Lamina CTE vector rotated from ply axes to the laminate frame.

    ``alpha_local`` is ``[a1, a2, a3, 0, 0, 0]`` (1/K, Voigt
    ``[11, 22, 33, 23, 13, 12]``). Thermal strain is an engineering strain,
    and local strain is ``T_eps(theta)`` times global strain (the
    convention of stress recovery), so the global vector is
    ``T_eps(theta)^-1 alpha_local``: ``a_x = a1 c^2 + a2 s^2``,
    ``a_y = a1 s^2 + a2 c^2``, ``a_z = a3`` and the engineering shear
    coefficient ``a_xy = 2 (a1 - a2) s c``.
    """
    alpha_local = np.asarray(alpha_local, dtype=float)
    if abs(angle_deg) <= 1e-15:
        return alpha_local.copy()
    T_eps = strain_transformation_3d(np.radians(angle_deg), axis='z')
    return np.linalg.solve(T_eps, alpha_local)


def thermal_stress_moduli(batch: ElementBatch, mesh: CompositeMesh,
                          alpha_local: np.ndarray) -> np.ndarray:
    """Thermal stress moduli ``beta = C alpha`` (MPa/K) at every Gauss point.

    ``C`` is the batch's porosity-degraded, ply-rotated stiffness and
    ``alpha`` the lamina CTE rotated to the laminate frame
    (:func:`global_cte`). The CTE is held at its pristine value: an empty
    void does not change the free thermal expansion of the matrix around
    it (Levin), so porosity enters only through ``C``. ``beta`` is zero
    in explicit void elements (``mesh.void_elements``).

    Returns
    -------
    np.ndarray
        ``(E, G, 6)``; ``beta * dT`` is the stress the thermal strain
        ``alpha dT`` would cause if fully restrained.
    """
    angles, inverse = np.unique(np.asarray(mesh.ply_angles, dtype=float),
                                return_inverse=True)
    table = np.stack([global_cte(alpha_local, float(a)) for a in angles])
    alpha_e = table[inverse.reshape(-1)]                       # (E, 6)
    beta = np.einsum('egij,ej->egi', batch.C, alpha_e)
    void_idx = np.asarray(mesh.void_elements, dtype=np.intp)
    if void_idx.size:
        beta[void_idx] = 0.0
    return beta


def element_dofs(elements: np.ndarray) -> np.ndarray:
    """Global DOF indices ``(E, 24)`` for hex8 connectivity ``(E, 8)``."""
    elements = np.asarray(elements, dtype=np.intp)
    return (3 * elements[:, :, None] + np.arange(3)).reshape(len(elements), 24)


def _gauss_point_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shape functions ``(G, 8)``, derivatives ``(G, 3, 8)``, points, weights."""
    points, weights = gauss_points_hex(order=2)
    N = np.stack([Hex8Element.shape_functions(*p) for p in points])
    dN = np.stack([Hex8Element.shape_derivatives(*p) for p in points])
    return N, dN, points, weights


def _checked_porosities(node_por: np.ndarray) -> np.ndarray:
    """Validate ``(E, 8)`` nodal porosities as Hex8Element does; clip to [0, 1]."""
    if not np.all(np.isfinite(node_por)):
        raise ValueError(
            "node_porosities must be finite; "
            "received NaN/inf values would propagate as NaN through "
            "the assembled stiffness."
        )
    eps = 1e-9
    too_low = node_por < -eps
    too_high = node_por > 1.0 + eps
    if np.any(too_low) or np.any(too_high):
        bad = node_por[too_low | too_high]
        hint = ""
        if np.any(too_high) and np.max(bad) >= 1.0 + 1e-3:
            hint = " (Pass a fraction in [0, 1], not a percent.)"
        raise ValueError(
            f"node_porosities must be a fraction in [0, 1] (per node), "
            f"got out-of-range values {bad.tolist()}.{hint}"
        )
    return np.clip(node_por, 0.0, 1.0)


def _void_stiffness() -> np.ndarray:
    """Near-zero isotropic stiffness used for explicit void elements."""
    E_void = Hex8Element.VOID_MODULUS
    nu_void = 0.3
    lam = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
    mu = E_void / (2 * (1 + nu_void))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = lam
    C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _incompatible_mode_G(coords: np.ndarray, detJ: np.ndarray,
                         points: np.ndarray) -> np.ndarray:
    """Incompatible-mode strain operators ``(E, G, 6, 9)``.

    Batched :meth:`Hex8Element.G_matrix`: natural bubble-mode derivatives
    mapped with the centroid Jacobian ``J0`` and scaled by
    ``det(J0) / det(J)`` (Taylor's correction).
    """
    dN0 = Hex8Element.shape_derivatives(0.0, 0.0, 0.0)          # (3, 8)
    J0 = np.einsum('ij,ejk->eik', dN0, coords)                  # (E, 3, 3)
    detJ0 = np.linalg.det(J0)
    bad = ~np.isfinite(detJ0) | (detJ0 <= 0.0)
    if np.any(bad):
        raise ValueError(
            f"Element has non-positive Jacobian determinant "
            f"(detJ={detJ0[np.argmax(bad)]!r}) at its centroid; the "
            f"incompatible-mode formulation cannot be built for it."
        )
    dP = _incompatible_mode_derivatives(points)                 # (G, 3, 3)
    dP_dx = np.matmul(np.linalg.inv(J0)[:, None], dP[None])     # (E, G, 3, 3)
    dP_dx = dP_dx * (detJ0[:, None] / detJ)[:, :, None, None]
    return _strain_operator(dP_dx)


def _condense_incompatible_modes(B: np.ndarray, C: np.ndarray,
                                 detJ_w: np.ndarray, G: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Effective operator ``B + G H``, ``H = -Kaa^-1 Kau``, per element.

    ``Kaa = sum G^T C G det(J) w`` and ``Kau = sum G^T C B det(J) w`` over the
    Gauss points. Returns the operator, an array of ``B``'s shape
    ``(E, G, 6, 24)``, and ``Kaa`` ``(E, 9, 9)``.
    """
    n_elem, n_gp = detJ_w.shape
    with np.errstate(over='ignore', invalid='ignore'):
        # Sum over Gauss points and strain components in one matmul:
        # (E, 9, G*6) @ (E, G*6, n) per element.
        CGw = np.matmul(C, G) * detJ_w[:, :, None, None]       # (E, G, 6, 9)
        GtCw = CGw.transpose(0, 3, 1, 2).reshape(n_elem, 9, n_gp * 6)
        Kaa = np.matmul(GtCw, G.reshape(n_elem, n_gp * 6, 9))  # (E, 9, 9)
        Kau = np.matmul(GtCw, B.reshape(n_elem, n_gp * 6, 24))  # (E, 9, 24)
    H = _condensation_operator(Kaa, Kau)
    return B + np.matmul(G, H[:, None]), Kaa


def build_element_batch(mesh: CompositeMesh, material: MaterialProperties,
                        void_shape_radii: tuple, *,
                        formulation: FEFormulation = _DEFAULT_FORMULATION,
                        ) -> ElementBatch:
    """Compute :class:`ElementBatch` for every element of ``mesh``.

    Parameters
    ----------
    mesh, material, void_shape_radii
        Mesh, composite and void shape the stiffness is built from.
    formulation : {'hex8i', 'hex8'}, optional
        Element formulation (see :class:`Hex8Element`), default
        ``'hex8i'``. For ``'hex8i'`` the incompatible modes are condensed
        into ``ElementBatch.B``.

    Raises
    ------
    ValueError
        On an unknown ``formulation``, non-finite or out-of-range nodal
        porosity, or a non-positive Jacobian determinant at any Gauss point
        (same messages as :class:`Hex8Element`).
    """
    _check_formulation(formulation)
    elements = np.asarray(mesh.elements, dtype=np.intp)
    n_elem = len(elements)
    N, dN, points, weights = _gauss_point_tables()

    coords = mesh.nodes[elements]                              # (E, 8, 3)
    J = np.einsum('gij,ejk->egik', dN, coords)                 # (E, G, 3, 3)
    detJ = np.linalg.det(J)
    bad = ~np.isfinite(detJ) | (detJ <= 0.0)
    if np.any(bad):
        e, g = np.argwhere(bad)[0]
        xi, eta, zeta = points[g]
        raise ValueError(
            f"Element has non-positive Jacobian determinant "
            f"(detJ={detJ[e, g]!r}) at Gauss point "
            f"(xi={xi}, eta={eta}, zeta={zeta}). The element is "
            f"degenerate or has inverted node ordering — its "
            f"contribution would silently corrupt the assembled "
            f"stiffness."
        )
    dN_dx = np.matmul(np.linalg.inv(J), dN[None])              # (E, G, 3, 8)
    B = _strain_operator(dN_dx)                                # (E, G, 6, 24)

    # Gauss-point porosity: interpolated from the nodes, except that an
    # element whose nodes all (nearly) agree uses its first node's value,
    # exactly as Hex8Element's uniform-porosity shortcut does.
    node_por = _checked_porosities(mesh.porosity[elements])
    p0 = node_por[:, :1]
    uniform = np.all(np.abs(node_por - p0) <= 1e-12 + 1e-5 * np.abs(p0), axis=1)
    Vp = np.where(uniform[:, None], p0, node_por @ N.T)       # (E, G)
    Vp = np.clip(Vp, 0.0, VP_STIFFNESS_CLAMP)

    # Evaluate the micromechanics once per distinct (Vp, angle) pair.
    angles = np.broadcast_to(
        np.asarray(mesh.ply_angles, dtype=float)[:, None], Vp.shape)
    pairs, inverse = np.unique(
        np.stack([Vp.ravel(), angles.ravel()], axis=1), axis=0,
        return_inverse=True)
    table = np.empty((len(pairs), 6, 6))
    for k, (vp, angle) in enumerate(pairs):
        C_k = _degraded_composite_stiffness(float(vp), void_shape_radii, material)
        ply_rad = np.radians(angle)
        if abs(ply_rad) > 1e-15:
            C_k = rotate_stiffness_3d(C_k, ply_rad, axis='z')
        table[k] = C_k
    C = table[inverse.reshape(-1)].reshape(n_elem, len(weights), 6, 6)

    void_idx = np.asarray(mesh.void_elements, dtype=np.intp)
    if void_idx.size:
        C[void_idx] = _void_stiffness()

    detJ_w = detJ * weights
    G_inc: np.ndarray | None = None
    Kaa: np.ndarray | None = None
    if formulation == 'hex8i':
        G_inc = _incompatible_mode_G(coords, detJ, points)
        B, Kaa = _condense_incompatible_modes(B, C, detJ_w, G_inc)

    return ElementBatch(
        dofs=element_dofs(elements),
        B=B,
        C=C,
        detJ_w=detJ_w,
        formulation=formulation,
        G_inc=G_inc,
        Kaa=Kaa,
    )
