"""Hex8 isoparametric element with porosity degradation."""

from __future__ import annotations

import numpy as np

from .._types import FEFormulation
from ..gauss import gauss_points_hex
from ..homogenization import _degraded_composite_stiffness, _mt_effective_stiffness
from ..materials import MaterialProperties
from ..transforms import rotate_stiffness_3d

# ============================================================
# SECTION 7d: HEX8 ELEMENT WITH POROSITY DEGRADATION
# ============================================================

# Void / degradation thresholds shared by stiffness assembly and failure
# evaluation (IMPROVEMENT_PLAN 2.3). Keep these the only definitions.

#: Upper clamp on the Gauss-point porosity fed to the micromechanics.
VP_STIFFNESS_CLAMP = 0.99

#: Elements whose nodal-mean porosity exceeds this carry no meaningful load
#: and are left out of failure evaluation, as are geometric void elements
#: (``CompositeMesh.void_elements``).
VOID_VP_THRESHOLD = 0.95

#: Element formulations accepted by ``formulation=`` (see
#: :data:`porosity_fe.FEFormulation`). ``'hex8'`` is the standard fully
#: integrated trilinear brick; ``'hex8i'`` adds nine Wilson-Taylor
#: incompatible modes, condensed out per element.
ELEMENT_FORMULATIONS: tuple[str, ...] = ('hex8', 'hex8i')


def _check_formulation(formulation: str) -> str:
    """Return ``formulation`` if it is supported, else raise ``ValueError``."""
    if formulation not in ELEMENT_FORMULATIONS:
        raise ValueError(
            f"Unknown element formulation {formulation!r}. "
            f"Use one of {list(ELEMENT_FORMULATIONS)}."
        )
    return formulation


def _strain_operator(d_dx: np.ndarray) -> np.ndarray:
    """Voigt strain operator from physical derivatives of nodal functions.

    ``d_dx`` has shape ``(..., 3, n)``: ``d_dx[..., i, a]`` is the derivative
    of function ``a`` with respect to ``x_i``. Returns ``(..., 6, 3 n)`` with
    engineering shear, columns ``[a_x, a_y, a_z]`` per function, rows in the
    Voigt order ``[11, 22, 33, 23, 13, 12]`` (the layout of
    :meth:`Hex8Element.B_matrix`).
    """
    dx, dy, dz = d_dx[..., 0, :], d_dx[..., 1, :], d_dx[..., 2, :]
    op = np.zeros(d_dx.shape[:-2] + (6, 3 * d_dx.shape[-1]))
    op[..., 0, 0::3] = dx
    op[..., 1, 1::3] = dy
    op[..., 2, 2::3] = dz
    op[..., 3, 1::3] = dz
    op[..., 3, 2::3] = dy
    op[..., 4, 0::3] = dz
    op[..., 4, 2::3] = dx
    op[..., 5, 0::3] = dy
    op[..., 5, 1::3] = dx
    return op


def _incompatible_mode_derivatives(points: np.ndarray) -> np.ndarray:
    """Natural derivatives of the bubble modes ``1 - xi_m^2`` at ``points``.

    Returns ``(G, 3, 3)``: entry ``[g, i, m]`` is ``d(1 - xi_m^2)/d xi_i`` at
    point ``g``, which is ``-2 xi_m`` on the diagonal and zero elsewhere.
    """
    points = np.asarray(points, dtype=float)
    return -2.0 * points[:, :, None] * np.eye(3)


def _condensation_operator(Kaa: np.ndarray, Kau: np.ndarray) -> np.ndarray:
    """``H = -Kaa^-1 Kau``: internal-mode amplitudes per unit nodal motion.

    Works on one element (``(9, 9)``, ``(9, 24)``) or a batch
    (``(E, 9, 9)``, ``(E, 9, 24)``). An element whose ``Kaa`` is non-finite
    or singular gets ``H = 0``, i.e. falls back to the standard hex8.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        ok = (np.isfinite(Kaa).all(axis=(-2, -1))
              & np.isfinite(Kau).all(axis=(-2, -1)))
    eye = np.broadcast_to(np.eye(Kaa.shape[-1]), Kaa.shape)
    Kaa = np.where(ok[..., None, None], Kaa, eye)
    Kau = np.where(ok[..., None, None], Kau, 0.0)
    try:
        H = -np.linalg.solve(Kaa, Kau)
    except np.linalg.LinAlgError:
        if Kaa.ndim == 2:
            return np.zeros_like(Kau)
        H = np.zeros_like(Kau)
        for e in range(Kaa.shape[0]):
            try:
                H[e] = -np.linalg.solve(Kaa[e], Kau[e])
            except np.linalg.LinAlgError:
                pass
    with np.errstate(invalid='ignore'):
        bad = ~np.isfinite(H).all(axis=(-2, -1))
    if np.any(bad):
        H = np.where(bad[..., None, None], 0.0, H)
    return H


# Natural coordinates of 8 hex nodes
_NODE_COORDS_REF = np.array([
    [-1.0, -1.0, -1.0],  # 0
    [+1.0, -1.0, -1.0],  # 1
    [+1.0, +1.0, -1.0],  # 2
    [-1.0, +1.0, -1.0],  # 3
    [-1.0, -1.0, +1.0],  # 4
    [+1.0, -1.0, +1.0],  # 5
    [+1.0, +1.0, +1.0],  # 6
    [-1.0, +1.0, +1.0],  # 7
], dtype=float)


class Hex8Element:
    """8-node isoparametric hexahedral element with porosity degradation.

    Instead of wrinkle-angle rotation (as in WrinkleFE), this element
    degrades the stiffness matrix at each Gauss point using the local
    porosity via Mori-Tanaka homogenization.

    Parameters
    ----------
    node_coords : np.ndarray
        Shape (8, 3) physical coordinates of the 8 nodes (mm).
    C_base : np.ndarray
        Shape (6, 6) base stiffness matrix (pristine composite).
    ply_angle_deg : float
        Ply orientation angle in degrees.
    node_porosities : np.ndarray
        Shape (8,) porosity volume fraction at each node.
    void_shape_radii : tuple
        (a1, a2, a3) void shape radii for Eshelby tensor.
    nu_m : float
        Matrix Poisson's ratio.
    C_m : np.ndarray
        Shape (6, 6) isotropic matrix stiffness for Mori-Tanaka.
    is_void : bool, optional
        Explicit void element: near-zero isotropic stiffness.
    material : MaterialProperties, optional
        Composite whose constants are degraded component-wise.
    formulation : {'hex8', 'hex8i'}, optional
        ``'hex8'`` (default): standard trilinear brick, full 2x2x2 Gauss
        integration. ``'hex8i'``: the same brick enriched with the nine
        Wilson-Taylor incompatible modes ``(1 - xi^2)``, ``(1 - eta^2)``,
        ``(1 - zeta^2)`` per displacement component, with Taylor's
        centroid-Jacobian correction (so the patch test passes on distorted
        elements), statically condensed out of the element. It removes the
        shear locking that makes ``'hex8'`` too stiff in bending when the
        element length is not small against the laminate thickness. See
        :meth:`G_matrix` and :meth:`strain_operator`.

    Notes
    -----
    For ``'hex8i'`` the condensation is exact: with ``H = -Kaa^-1 Kau`` the
    internal-mode amplitudes are ``alpha = H u_e`` and the element behaves
    as one whose strain operator is ``B_eff = B + G H``
    (:meth:`strain_operator`), so ``Ke = sum B_eff^T C B_eff det(J) w``
    equals ``Kuu - Kua Kaa^-1 Kau``. :meth:`B_matrix` always returns the
    compatible (standard) ``B``.
    """

    # Near-zero stiffness for void elements (Pa, not MPa — ~6 orders softer)
    VOID_MODULUS = 1.0  # MPa (effectively zero vs composite E11 ~ 161,000 MPa)

    def __init__(self, node_coords: np.ndarray, C_base: np.ndarray,
                 ply_angle_deg: float, node_porosities: np.ndarray,
                 void_shape_radii: tuple, nu_m: float,
                 C_m: np.ndarray, is_void: bool = False,
                 material: MaterialProperties = None, *,
                 formulation: FEFormulation = 'hex8') -> None:
        self.node_coords = np.asarray(node_coords, dtype=float)
        if self.node_coords.shape != (8, 3):
            raise ValueError(f"node_coords must be (8,3), got {self.node_coords.shape}.")
        self.C_base = np.asarray(C_base, dtype=float)
        self.ply_angle_deg = ply_angle_deg
        self.node_porosities = np.asarray(node_porosities, dtype=float)
        if self.node_porosities.shape != (8,):
            raise ValueError(f"node_porosities must be (8,), got {self.node_porosities.shape}.")
        if not np.all(np.isfinite(self.node_porosities)):
            raise ValueError(
                "node_porosities must be finite; "
                "received NaN/inf values would propagate as NaN through "
                "the assembled stiffness."
            )
        # Allow a small fp overshoot (~1e-9) and clip back into [0, 1]; reject
        # anything beyond that as a clear unit/percent confusion.
        eps = 1e-9
        too_low = self.node_porosities < -eps
        too_high = self.node_porosities > 1.0 + eps
        if np.any(too_low) or np.any(too_high):
            bad = self.node_porosities[too_low | too_high]
            hint = ""
            if np.any(too_high) and np.max(bad) >= 1.0 + 1e-3:
                hint = " (Pass a fraction in [0, 1], not a percent.)"
            raise ValueError(
                f"node_porosities must be a fraction in [0, 1] (per node), "
                f"got out-of-range values {bad.tolist()}.{hint}"
            )
        self.node_porosities = np.clip(self.node_porosities, 0.0, 1.0)
        self.void_shape_radii = void_shape_radii
        self.nu_m = nu_m
        self.C_m = np.asarray(C_m, dtype=float)
        self.material = material
        self.is_void = is_void
        _check_formulation(formulation)
        self.formulation: FEFormulation = formulation
        # Incompatible-mode condensation H = -Kaa^-1 Kau, built on first use.
        self._H: np.ndarray | None = None

        self._gauss_points, self._gauss_weights = gauss_points_hex(order=2)

        # Pre-compute void stiffness (isotropic, near-zero modulus)
        if self.is_void:
            E_void = self.VOID_MODULUS
            nu_void = 0.3
            lam = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
            mu = E_void / (2 * (1 + nu_void))
            self._void_C = np.zeros((6, 6))
            self._void_C[0, 0] = self._void_C[1, 1] = self._void_C[2, 2] = lam + 2 * mu
            self._void_C[0, 1] = self._void_C[0, 2] = self._void_C[1, 0] = lam
            self._void_C[1, 2] = self._void_C[2, 0] = self._void_C[2, 1] = lam
            self._void_C[3, 3] = self._void_C[4, 4] = self._void_C[5, 5] = mu

        # Cache: if all node porosities are the same, pre-compute C_eff once
        self._uniform_porosity = None
        if not self.is_void and np.allclose(self.node_porosities, self.node_porosities[0], atol=1e-12):
            self._uniform_porosity = float(self.node_porosities[0])

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate 8 trilinear shape functions at natural coordinates.

        Returns
        -------
        np.ndarray
            Shape (8,).
        """
        N = 0.125 * (
            (1.0 + _NODE_COORDS_REF[:, 0] * xi)
            * (1.0 + _NODE_COORDS_REF[:, 1] * eta)
            * (1.0 + _NODE_COORDS_REF[:, 2] * zeta)
        )
        return N

    @staticmethod
    def shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Derivatives of shape functions w.r.t. natural coordinates.

        Returns
        -------
        np.ndarray
            Shape (3, 8): dN[i, j] = dN_j / d(xi_i).
        """
        xi_node = _NODE_COORDS_REF[:, 0]
        eta_node = _NODE_COORDS_REF[:, 1]
        zeta_node = _NODE_COORDS_REF[:, 2]

        dN_dxi = 0.125 * xi_node * (1.0 + eta_node * eta) * (1.0 + zeta_node * zeta)
        dN_deta = 0.125 * (1.0 + xi_node * xi) * eta_node * (1.0 + zeta_node * zeta)
        dN_dzeta = 0.125 * (1.0 + xi_node * xi) * (1.0 + eta_node * eta) * zeta_node

        return np.stack([dN_dxi, dN_deta, dN_dzeta], axis=0)

    def jacobian(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Jacobian matrix (3x3) mapping natural to physical coordinates."""
        dN = self.shape_derivatives(xi, eta, zeta)
        return dN @ self.node_coords

    def B_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Strain-displacement matrix (6x24) in Voigt notation.

        Strain ordering: [eps_11, eps_22, eps_33, gamma_23, gamma_13, gamma_12]
        DOF ordering: [u1x, u1y, u1z, u2x, u2y, u2z, ..., u8x, u8y, u8z]

        Notes
        -----
        Voigt order: ``[11, 22, 33, 23, 13, 12]``. Shear rows produce
        **engineering** strain (``gamma_ij = 2 * eps_ij = du_i/dx_j +
        du_j/dx_i``), which is the convention paired with
        :meth:`MaterialProperties.get_stiffness_matrix` so that ``sigma = C @
        (B @ u)`` is dimensionally consistent. Apply the engineering-strain
        transformation (:func:`strain_transformation_3d`) — not the tensor
        form — when rotating ``B @ u`` between coordinate frames.
        """
        dN_dxi = self.shape_derivatives(xi, eta, zeta)
        J = dN_dxi @ self.node_coords
        J_inv = np.linalg.inv(J)
        dN_dx = J_inv @ dN_dxi  # (3, 8)

        B = np.zeros((6, 24))
        for i in range(8):
            col = 3 * i
            dNi_dx = dN_dx[0, i]
            dNi_dy = dN_dx[1, i]
            dNi_dz = dN_dx[2, i]
            B[0, col] = dNi_dx
            B[1, col + 1] = dNi_dy
            B[2, col + 2] = dNi_dz
            B[3, col + 1] = dNi_dz
            B[3, col + 2] = dNi_dy
            B[4, col] = dNi_dz
            B[4, col + 2] = dNi_dx
            B[5, col] = dNi_dy
            B[5, col + 1] = dNi_dx
        return B

    def G_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Strain operator (6x9) of the nine incompatible modes at a point.

        The modes are ``P_m = 1 - xi_m^2`` (``m`` over ``xi, eta, zeta``) for
        each displacement component; columns are ordered mode-major,
        ``[P_1 x, P_1 y, P_1 z, P_2 x, ...]``, rows in the Voigt order of
        :meth:`B_matrix`. Taylor's correction maps the natural derivatives
        with the centroid Jacobian ``J0`` and scales them by
        ``det(J0) / det(J)``, so that every mode integrates to zero strain
        over the element and the patch test passes on distorted elements.

        Defined for either formulation; only ``'hex8i'`` uses it.
        """
        J0 = self.jacobian(0.0, 0.0, 0.0)
        detJ0 = np.linalg.det(J0)
        detJ = np.linalg.det(self.jacobian(xi, eta, zeta))
        dP = _incompatible_mode_derivatives(np.array([[xi, eta, zeta]]))[0]
        dP_dx = np.linalg.solve(J0, dP) * (detJ0 / detJ)
        return _strain_operator(dP_dx)

    def _incompatible_mode_condensation(self) -> np.ndarray:
        """``H = -Kaa^-1 Kau`` (9x24) for ``'hex8i'``, computed once."""
        if self._H is None:
            Kaa = np.zeros((9, 9))
            Kau = np.zeros((9, 24))
            for gp_idx in range(len(self._gauss_weights)):
                xi, eta, zeta = self._gauss_points[gp_idx]
                w = self._gauss_weights[gp_idx]
                B = self.B_matrix(xi, eta, zeta)
                G = self.G_matrix(xi, eta, zeta)
                C_bar = self._degraded_stiffness(xi, eta, zeta)
                detJ = np.linalg.det(self.jacobian(xi, eta, zeta))
                with np.errstate(over='ignore', invalid='ignore'):
                    CG = C_bar @ G * (detJ * w)
                    Kaa += G.T @ CG
                    Kau += CG.T @ B
            self._H = _condensation_operator(Kaa, Kau)
        return self._H

    def strain_operator(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Strain-displacement operator (6x24) the formulation uses.

        :meth:`B_matrix` for ``'hex8'``. For ``'hex8i'`` the effective
        operator ``B + G H`` with the incompatible modes condensed out
        (``H`` from ``-Kaa^-1 Kau``), so strain at a point is
        ``strain_operator(...) @ u_e`` for both formulations.
        """
        B = self.B_matrix(xi, eta, zeta)
        if self.formulation == 'hex8i':
            B = B + self.G_matrix(xi, eta, zeta) @ self._incompatible_mode_condensation()
        return B

    def _degraded_stiffness(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute porosity-degraded and ply-rotated stiffness at a point.

        Steps:
        1. Interpolate porosity at this point from nodal values.
        2. Degrade individual composite engineering constants (E11, E22, G12, etc.)
           via Mori-Tanaka + micromechanics rule-of-mixtures, so that porosity in
           0-degree plies correctly yields different laminate stiffness reduction
           than porosity in 90-degree plies.
        3. Rotate by ply angle about z-axis.

        Returns
        -------
        np.ndarray
            Shape (6, 6) degraded and rotated stiffness.
        """
        # VOID ELEMENTS: use near-zero isotropic stiffness (explicit inclusion)
        if self.is_void:
            return self._void_C

        # NON-VOID ELEMENTS: degrade by distributed microporosity via Mori-Tanaka
        # 1. Interpolate porosity at this Gauss point
        if self._uniform_porosity is not None:
            Vp = self._uniform_porosity
        else:
            N = self.shape_functions(xi, eta, zeta)
            Vp = float(N @ self.node_porosities)
        Vp = max(0.0, min(Vp, VP_STIFFNESS_CLAMP))

        # 2. Component-wise degradation: degrade E11, E22, G12, etc. individually
        #    This correctly captures that E11 (fiber-dominated) is barely affected
        #    while E22/G12 (matrix-dominated) are strongly reduced by porosity.
        if self.material is not None:
            C_degraded = _degraded_composite_stiffness(
                Vp, self.void_shape_radii, self.material)
        else:
            # Fallback: scalar degradation (legacy behavior)
            C_eff_mt = _mt_effective_stiffness(self.C_m, Vp, self.void_shape_radii, self.nu_m)
            diag_pristine = np.diag(self.C_m)
            diag_degraded = np.diag(C_eff_mt)
            mask = diag_pristine > 1e-12
            if np.any(mask):
                avg_ratio = np.mean(diag_degraded[mask] / diag_pristine[mask])
            else:
                avg_ratio = 1.0
            avg_ratio = max(0.0, min(avg_ratio, 1.0))
            C_degraded = self.C_base * avg_ratio

        # 3. Rotate by ply angle
        ply_rad = np.radians(self.ply_angle_deg)
        if abs(ply_rad) > 1e-15:
            C_degraded = rotate_stiffness_3d(C_degraded, ply_rad, axis='z')

        return C_degraded

    def stiffness_matrix(self) -> np.ndarray:
        """Element stiffness matrix (24x24) via 2x2x2 Gauss quadrature.

        ``Ke = sum over GPs of: B^T @ C_bar @ B * det(J) * w``, with ``B``
        from :meth:`strain_operator` (the condensed operator for
        ``'hex8i'``).

        Raises
        ------
        ValueError
            If the Jacobian determinant is non-positive at any Gauss point —
            this signals a degenerate or inverted element whose contribution
            would corrupt the assembled global stiffness with a wrong-sign
            block. Catching here makes failures legible instead of silent.
        """
        Ke = np.zeros((24, 24))
        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]
            B = self.strain_operator(xi, eta, zeta)
            C_bar = self._degraded_stiffness(xi, eta, zeta)
            J = self.jacobian(xi, eta, zeta)
            detJ = np.linalg.det(J)
            if not np.isfinite(detJ) or detJ <= 0.0:
                raise ValueError(
                    f"Element has non-positive Jacobian determinant "
                    f"(detJ={detJ!r}) at Gauss point "
                    f"(xi={xi}, eta={eta}, zeta={zeta}). The element is "
                    f"degenerate or has inverted node ordering — its "
                    f"contribution would silently corrupt the assembled "
                    f"stiffness."
                )
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                Ke_contrib = (B.T @ C_bar @ B) * detJ * w
            # Protect against overflow in void elements
            if np.any(~np.isfinite(Ke_contrib)):
                Ke_contrib = np.where(np.isfinite(Ke_contrib), Ke_contrib, 0.0)
            Ke += Ke_contrib
        return Ke

    def stress_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute stress at all Gauss points.

        Parameters
        ----------
        u_elem : np.ndarray
            Shape (24,) element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape (n_gp, 6) stress in Voigt notation.
        """
        u_elem = np.asarray(u_elem, dtype=float)
        n_gp = len(self._gauss_weights)
        stresses = np.empty((n_gp, 6))
        for gp_idx in range(n_gp):
            xi, eta, zeta = self._gauss_points[gp_idx]
            B = self.strain_operator(xi, eta, zeta)
            C_bar = self._degraded_stiffness(xi, eta, zeta)
            stresses[gp_idx] = C_bar @ (B @ u_elem)
        return stresses

    def strain_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute strain at all Gauss points.

        Parameters
        ----------
        u_elem : np.ndarray
            Shape (24,) element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape (n_gp, 6) engineering strain in Voigt notation.
        """
        u_elem = np.asarray(u_elem, dtype=float)
        n_gp = len(self._gauss_weights)
        strains = np.empty((n_gp, 6))
        for gp_idx in range(n_gp):
            xi, eta, zeta = self._gauss_points[gp_idx]
            B = self.strain_operator(xi, eta, zeta)
            strains[gp_idx] = B @ u_elem
        return strains

    @property
    def volume(self) -> float:
        """Element volume via Gauss quadrature.

        Uses ``abs(det(J))`` so an inverted-but-otherwise-valid element
        still reports a sensible (positive) volume. ``stiffness_matrix``
        rejects inverted elements at assembly time, so the negative-volume
        case is only reachable via direct ``.volume`` lookup on a degenerate
        element constructed manually.
        """
        vol = 0.0
        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]
            J = self.jacobian(xi, eta, zeta)
            vol += abs(np.linalg.det(J)) * w
        return float(vol)

