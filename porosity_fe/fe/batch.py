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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..gauss import gauss_points_hex
from ..homogenization import _degraded_composite_stiffness
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..transforms import rotate_stiffness_3d
from .element import Hex8Element

# Upper clamp on the Gauss-point porosity fed to the micromechanics, matching
# Hex8Element._degraded_stiffness.
_VP_CLAMP_MAX = 0.99


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
    C : np.ndarray
        ``(E, G, 6, 6)`` degraded, ply-rotated stiffness (MPa).
    detJ_w : np.ndarray
        ``(E, G)`` Jacobian determinant times quadrature weight.
    """

    dofs: np.ndarray
    B: np.ndarray
    C: np.ndarray
    detJ_w: np.ndarray

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


def build_element_batch(mesh: CompositeMesh, material: MaterialProperties,
                        void_shape_radii: tuple) -> ElementBatch:
    """Compute :class:`ElementBatch` for every element of ``mesh``.

    Raises
    ------
    ValueError
        On non-finite or out-of-range nodal porosity, or a non-positive
        Jacobian determinant at any Gauss point (same messages as
        :class:`Hex8Element`).
    """
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
    dx, dy, dz = dN_dx[:, :, 0], dN_dx[:, :, 1], dN_dx[:, :, 2]
    B = np.zeros((n_elem, len(weights), 6, 24))
    B[:, :, 0, 0::3] = dx
    B[:, :, 1, 1::3] = dy
    B[:, :, 2, 2::3] = dz
    B[:, :, 3, 1::3] = dz
    B[:, :, 3, 2::3] = dy
    B[:, :, 4, 0::3] = dz
    B[:, :, 4, 2::3] = dx
    B[:, :, 5, 0::3] = dy
    B[:, :, 5, 1::3] = dx

    # Gauss-point porosity: interpolated from the nodes, except that an
    # element whose nodes all (nearly) agree uses its first node's value,
    # exactly as Hex8Element's uniform-porosity shortcut does.
    node_por = _checked_porosities(mesh.porosity[elements])
    p0 = node_por[:, :1]
    uniform = np.all(np.abs(node_por - p0) <= 1e-12 + 1e-5 * np.abs(p0), axis=1)
    Vp = np.where(uniform[:, None], p0, node_por @ N.T)       # (E, G)
    Vp = np.clip(Vp, 0.0, _VP_CLAMP_MAX)

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

    return ElementBatch(
        dofs=element_dofs(elements),
        B=B,
        C=C,
        detJ_w=detJ * weights,
    )
