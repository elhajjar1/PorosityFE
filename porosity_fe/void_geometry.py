"""Discrete ellipsoidal void geometry."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad

# ============================================================
# SECTION 2: VOID GEOMETRY MODEL
# ============================================================

VOID_SHAPES = {
    'spherical':   (1.0, 1.0, 1.0),
    'cylindrical': (3.0, 1.0, 1.0),
    'penny':       (3.0, 3.0, 0.3),
}


#: Loaded stress component (global frame: x = loading / fiber direction,
#: y = transverse, z = thickness) and remote sign for each loading mode of
#: :meth:`VoidGeometry.stress_concentration_factor`.
_SCF_MODE_LOADS: dict[str, tuple[tuple[int, int], float]] = {
    'compression': ((0, 0), -1.0),
    'tension': ((0, 0), 1.0),
    'shear': ((0, 1), 1.0),
    'ilss': ((0, 2), 1.0),
    'transverse_tension': ((1, 1), 1.0),
}

_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _eshelby_tensor(radii: tuple[float, float, float], nu: float) -> np.ndarray:
    """Interior Eshelby tensor of a general ellipsoid (Voigt 6x6).

    Mura (1987) eqs. 11.16-11.19, with the ``I_i`` / ``I_ij`` integrals
    evaluated by quadrature in ``log(s)`` (smooth and well scaled for any
    aspect ratio). Shear diagonal entries are ``2 S_ijij``, so the matrix
    maps a 6-vector of strain components (tensor or engineering shear,
    consistently) to the same form.
    """
    a = np.asarray(radii, dtype=float)
    a2 = a ** 2
    pref = 2.0 * np.pi * float(np.prod(a))
    lo = float(np.log(a2.min())) - 40.0
    hi = float(np.log(a2.max())) + 40.0
    breaks = sorted(set(np.log(a2).tolist()))

    def integral(*idx: int) -> float:
        def f(x: float) -> float:
            t = np.exp(x)
            denom = np.sqrt(np.prod(a2 + t))
            for i in idx:
                denom *= a2[i] + t
            return t / denom
        return pref * quad(f, lo, hi, points=breaks, limit=400,
                           epsabs=0.0, epsrel=1e-11)[0]

    I1 = [integral(i) for i in range(3)]
    I2 = [[integral(i, j) for j in range(3)] for i in range(3)]
    c = 1.0 / (8.0 * np.pi * (1.0 - nu))
    S = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            if i == j:
                S[i, i] = 3.0 * c * a2[i] * I2[i][i] + (1 - 2 * nu) * c * I1[i]
            else:
                S[i, j] = c * a2[j] * I2[i][j] - (1 - 2 * nu) * c * I1[i]
    for v, (i, j) in zip((3, 4, 5), ((1, 2), (0, 2), (0, 1)), strict=True):
        S[v, v] = 2.0 * ((a2[i] + a2[j]) * I2[i][j]
                         + (1 - 2 * nu) * (I1[i] + I1[j])) / (16.0 * np.pi * (1.0 - nu))
    return S


def _cavity_surface_stress(radii: tuple[float, float, float], nu: float,
                           sigma0: np.ndarray, S: np.ndarray,
                           theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Matrix stress on the surface of an ellipsoidal cavity (local frame).

    Exact linear-elastic solution for a traction-free ellipsoidal cavity in
    an infinite isotropic matrix under the uniform remote stress
    ``sigma0`` (unit shear modulus; only ``nu`` matters). Eshelby's
    equivalent eigenstrain gives the uniform interior strain ``E``; just
    outside, the displacement gradient jumps by ``a (x) n`` with ``a`` fixed
    by the traction-free condition ``sigma . n = 0``. ``theta`` / ``phi``
    parametrize the surface point ``(a1 sin t cos p, a2 sin t sin p,
    a3 cos t)``. Returns stresses of shape ``theta.shape + (3, 3)``.
    """
    a = np.asarray(radii, dtype=float)
    mu = 1.0
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    eye = np.eye(3)
    eps0 = (sigma0 - lam / (2 * mu + 3 * lam) * np.trace(sigma0) * eye) / (2 * mu)
    eps_star = np.linalg.solve(np.eye(6) - S,
                               np.array([eps0[i, j] for i, j in _VOIGT_PAIRS]))
    E = np.zeros((3, 3))
    for k, (i, j) in enumerate(_VOIGT_PAIRS):
        E[i, j] = E[j, i] = eps_star[k]
    sig_E = lam * np.trace(E) * eye + 2 * mu * E

    n = np.stack([np.sin(theta) * np.cos(phi) / a[0],
                  np.sin(theta) * np.sin(phi) / a[1],
                  np.cos(theta) / a[2]], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    acoustic = (lam + mu) * n[..., :, None] * n[..., None, :] + mu * eye
    jump = np.linalg.solve(acoustic, -(n @ sig_E)[..., None])[..., 0]
    a_dot_n = np.sum(jump * n, axis=-1)
    sym = 0.5 * (jump[..., :, None] * n[..., None, :]
                 + n[..., :, None] * jump[..., None, :])
    return (lam * (np.trace(E) + a_dot_n)[..., None, None] * eye
            + 2 * mu * (E + sym))


def _peak_component(radii: tuple[float, float, float], nu: float, S: np.ndarray,
                    Q: np.ndarray, sigma0: np.ndarray, comp: tuple[int, int],
                    theta: np.ndarray, phi: np.ndarray) -> tuple[float, float, float]:
    """Max over the surface points of global ``sigma[comp] / sigma0[comp]``.

    Returns ``(ratio, theta, phi)`` at the maximum. ``Q`` maps global to
    void-local vectors.
    """
    i, j = comp
    sig_loc = _cavity_surface_stress(radii, nu, Q @ sigma0 @ Q.T, S, theta, phi)
    ratio = (Q.T @ sig_loc @ Q)[..., i, j] / sigma0[i, j]
    k = int(np.argmax(ratio))
    return float(ratio.flat[k]), float(theta.flat[k]), float(phi.flat[k])


@lru_cache(maxsize=256)
def _cavity_scfs(radii: tuple[float, float, float], orientation: float,
                 nu: float) -> tuple[tuple[str, float], ...]:
    """Peak loaded-component SCF per mode; see
    :meth:`VoidGeometry.stress_concentration_factor`."""
    S = _eshelby_tensor(radii, nu)
    c, s = np.cos(orientation), np.sin(orientation)
    Q = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # global -> local
    theta, phi = np.meshgrid(np.linspace(0.0, np.pi, 91),
                             np.linspace(0.0, 2 * np.pi, 180, endpoint=False),
                             indexing='ij')
    out = []
    for mode, ((i, j), sign) in _SCF_MODE_LOADS.items():
        sigma0 = np.zeros((3, 3))
        sigma0[i, j] = sigma0[j, i] = sign
        best, t0, p0 = _peak_component(radii, nu, S, Q, sigma0, (i, j), theta, phi)
        # Two local refinements around the running maximum.
        dt, dp = np.pi / 90, 2 * np.pi / 180
        for _ in range(2):
            t_f, p_f = np.meshgrid(
                np.clip(np.linspace(t0 - dt, t0 + dt, 21), 0.0, np.pi),
                np.linspace(p0 - dp, p0 + dp, 21), indexing='ij')
            fine, t_new, p_new = _peak_component(radii, nu, S, Q, sigma0, (i, j),
                                                 t_f, p_f)
            if fine > best:
                best, t0, p0 = fine, t_new, p_new
            dt, dp = dt / 10, dp / 10
        out.append((mode, best))
    return tuple(out)


class VoidGeometry:
    """Single discrete void parameterized as an oriented ellipsoid.

    A void is the locus of points satisfying

    .. math::

        \\left(\\frac{x_\\ell}{a}\\right)^2
        + \\left(\\frac{y_\\ell}{b}\\right)^2
        + \\left(\\frac{z_\\ell}{c}\\right)^2 \\le 1,

    where ``(x_l, y_l, z_l)`` are coordinates in the **void-local** frame:
    coordinates are first translated so the void centroid is at the origin
    and then rotated by ``-orientation`` about the global +z axis. The
    semi-axes ``(a, b, c)`` correspond to the world x / y / z directions
    in that local frame. This class is the porosity-model counterpart of
    ``WrinkleGeometry`` in the WrinkleFE codebase.

    Parameters
    ----------
    center : tuple of 3 float
        Centroid coordinates ``(x, y, z)`` in the global frame, in mm.
        Must be finite.
    radii : tuple of 3 float
        Semi-axes ``(a, b, c)`` of the ellipsoid in mm, ordered along the
        **local** x / y / z axes. All three must be positive and finite
        (they appear as ``1 / r`` in the containment test).
    orientation : float, optional
        Rotation about the global +z axis, in **radians** (default 0).
        Positive values rotate the local x-axis toward the global y-axis.

    Attributes
    ----------
    center : np.ndarray
        Shape ``(3,)`` float array; the void centroid in mm.
    radii : np.ndarray
        Shape ``(3,)`` float array of positive semi-axes ``(a, b, c)``
        in mm.
    orientation : float
        In-plane rotation angle in radians.
    aspect_ratio : float
        Read-only property: ``max(radii) / min(radii)``.

    Examples
    --------
    A 1 mm-radius spherical void at the coupon midpoint:

    >>> v = VoidGeometry(center=(25.0, 10.0, 2.2),
    ...                  radii=(1.0, 1.0, 1.0))
    >>> bool(v.contains(25.0, 10.0, 2.2))
    True
    >>> round(v.volume(), 4)
    4.1888

    A penny-shaped void rotated 30 deg about z:

    >>> import math
    >>> v = VoidGeometry(center=(25.0, 10.0, 2.2),
    ...                  radii=(3.0, 3.0, 0.3),
    ...                  orientation=math.radians(30.0))
    >>> round(v.aspect_ratio, 2)
    10.0
    """

    def __init__(self, center: tuple, radii: tuple, orientation: float = 0.0):
        self.center = np.array(center, dtype=float)
        self.radii = np.array(radii, dtype=float)
        if self.center.shape != (3,):
            raise ValueError(
                f"VoidGeometry.center must have 3 components (x, y, z), "
                f"got shape {self.center.shape}."
            )
        if self.radii.shape != (3,):
            raise ValueError(
                f"VoidGeometry.radii must have 3 components (a, b, c), "
                f"got shape {self.radii.shape}."
            )
        if not np.all(np.isfinite(self.radii)) or np.any(self.radii <= 0):
            raise ValueError(
                f"VoidGeometry.radii must be 3 positive finite numbers "
                f"(used as 1/r in the ellipsoid containment test), "
                f"got {self.radii.tolist()}."
            )
        if not np.all(np.isfinite(self.center)):
            raise ValueError(
                f"VoidGeometry.center must be finite, got {self.center.tolist()}."
            )
        if not np.isfinite(orientation):
            raise ValueError(
                f"VoidGeometry.orientation must be a finite angle (radians), "
                f"got {orientation!r}."
            )
        self.orientation = orientation

    def _to_local(self, x, y, z):
        """Transform world coordinates to void-local (translated + rotated)."""
        dx = np.asarray(x, dtype=float) - self.center[0]
        dy = np.asarray(y, dtype=float) - self.center[1]
        dz = np.asarray(z, dtype=float) - self.center[2]
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        z_loc = dz
        return x_loc, y_loc, z_loc

    def contains(self, x, y, z) -> np.ndarray:
        x_l, y_l, z_l = self._to_local(x, y, z)
        val = (x_l / self.radii[0])**2 + (y_l / self.radii[1])**2 + (z_l / self.radii[2])**2
        return val <= 1.0

    def distance_field(self, x, y, z) -> np.ndarray:
        x_l, y_l, z_l = self._to_local(x, y, z)
        val = np.sqrt((x_l / self.radii[0])**2 + (y_l / self.radii[1])**2 + (z_l / self.radii[2])**2)
        r_eff = np.sqrt(x_l**2 + y_l**2 + z_l**2)
        # At the void center (r_eff ~ 0), val is also ~0 causing 0/0.
        # Return -1.0 (clearly inside) for those points.
        eps = 1e-12
        at_center = r_eff < eps
        r_eff = np.maximum(r_eff, eps)
        val_safe = np.maximum(val, eps)
        result = r_eff * (val - 1.0) / val_safe
        result = np.where(at_center, -1.0, result)
        return result

    def stress_concentration_factor(self, nu_m: float = 0.35) -> dict:
        """Elastic stress concentration factor of this void for each loading mode.

        The void is treated as a traction-free ellipsoidal cavity in an
        infinite isotropic matrix with Poisson's ratio ``nu_m``, and the
        exact (Eshelby) solution gives the stress on its surface. The SCF
        is the peak of the loaded stress component over the surface divided
        by its remote value, in the global frame (``orientation`` is
        honored): ``sigma_xx`` for ``'tension'`` / ``'compression'``,
        ``sigma_yy`` for ``'transverse_tension'``, ``tau_xy`` for
        ``'shear'`` and ``tau_xz`` for ``'ilss'``. Linear elasticity makes
        the tension and compression values equal.

        Reference values: a sphere gives Goodier's
        ``(27 - 15 nu) / (2 (7 - 5 nu))`` in tension and
        ``15 (1 - nu) / (7 - 5 nu)`` in shear; a long elliptic cylinder
        recovers Inglis' ``1 + 2 a / b``. Composite anisotropy is not
        modeled (the voids sit in the matrix). IMPROVEMENT_PLAN 2.6
        replaced uncited piecewise-linear aspect-ratio rules (which gave a
        penny void loaded in its own plane an SCF of 17) and their
        shape-class thresholds.

        Parameters
        ----------
        nu_m : float, optional
            Matrix Poisson's ratio (default 0.35, typical epoxy);
            :class:`EmpiricalSolver` passes ``material.matrix_poisson``.

        Returns
        -------
        dict
            ``{mode: SCF}`` for ``'compression'``, ``'tension'``,
            ``'shear'``, ``'ilss'`` and ``'transverse_tension'``.
        """
        if not (-1.0 < nu_m < 0.5):
            raise ValueError(f"nu_m must be in (-1, 0.5), got {nu_m!r}.")
        radii = tuple(round(float(r), 12) for r in self.radii)
        return dict(_cavity_scfs(radii, round(float(self.orientation), 12),  # type: ignore[arg-type]
                                 round(float(nu_m), 12)))

    def volume(self) -> float:
        return (4.0 / 3.0) * np.pi * self.radii[0] * self.radii[1] * self.radii[2]

    @property
    def aspect_ratio(self) -> float:
        return float(np.max(self.radii) / np.min(self.radii))

    def __repr__(self) -> str:
        return (f"VoidGeometry(center={self.center.tolist()}, "
                f"radii={self.radii.tolist()}, "
                f"orientation={self.orientation:.3f}, "
                f"aspect_ratio={self.aspect_ratio:.2f})")
