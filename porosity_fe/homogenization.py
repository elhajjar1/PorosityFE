"""Mori-Tanaka homogenization, MT-cache, and CLT effective moduli."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._types import PorosityDegradation
from .materials import MaterialProperties
from .transforms import rotate_stiffness_3d
from .void_geometry import VOID_SHAPES

# ============================================================
# SECTION 7d: MT effective stiffness + LRU cache + composite degradation
# ============================================================

_MT_CACHE_MAXSIZE = 4096


def _mt_effective_stiffness(C_m: np.ndarray, Vp: float,
                            void_shape_radii: tuple,
                            nu_m: float) -> np.ndarray:
    """Mori-Tanaka effective stiffness for void inclusions (C_inclusion = 0).

    Memoized via ``functools.lru_cache`` on a content-fingerprint key (#42,
    #112). The cached result is copied before return so callers can mutate
    it freely.

    Standalone Eshelby-tensor-based Mori-Tanaka calculation for use inside
    Hex8Element.

    The Eshelby tensor is computed in a canonical frame (symmetry axis along
    x_1) and then permuted to align with the actual void axis. The previous
    implementation used ``ar = max/min`` as the aspect ratio, which is
    always >= 1, leaving the ``else: # Oblate`` branch unreachable — oblate
    voids (penny shape) were silently routed to the prolate formulas. See
    issue #32.

    Parameters
    ----------
    C_m : np.ndarray
        Shape (6, 6) isotropic matrix stiffness.
    Vp : float
        Void volume fraction (0 to 1).
    void_shape_radii : tuple
        (a1, a2, a3) radii defining void shape for Eshelby tensor.
    nu_m : float
        Matrix Poisson's ratio.

    Returns
    -------
    np.ndarray
        Shape (6, 6) effective stiffness matrix.
    """
    if Vp < 1e-12:
        return C_m.copy()
    if Vp > 0.99:
        return np.zeros((6, 6))

    # Build a hashable content fingerprint (same rounding rules as the
    # pre-#112 manual cache key) and dispatch to the lru_cache'd inner
    # function. An isotropic 6x6 stiffness is fully determined by
    # (lam+2mu, mu); both diagonal slots act as a content fingerprint so
    # different materials never collide.
    radii_key = tuple(round(float(r), 6) for r in void_shape_radii)
    C00 = round(float(C_m[0, 0]), 4)
    C33 = round(float(C_m[3, 3]), 4)
    Vp_key = round(float(Vp), 6)
    nu_key = round(float(nu_m), 8)

    cached = _mt_effective_stiffness_cached(C00, C33, Vp_key, radii_key, nu_key)
    # The cache stores the canonical result; return a defensive copy so
    # callers can mutate freely without poisoning the cache.
    return cached.copy()


def _mori_tanaka_voids(C_m: np.ndarray, S: np.ndarray, Vp: float) -> np.ndarray:
    """Mori-Tanaka stiffness of a matrix with void volume fraction ``Vp``.

    ``S`` is the Eshelby tensor in the engineering-shear Voigt form.
    """
    I6 = np.eye(6)
    inner = I6 - (1 - Vp) * S
    # The Mori-Tanaka concentration tensor `inner` becomes singular as Vp -> 1
    # for high-aspect-ratio (oblate) voids even before the Vp > 0.99 early-out
    # above is reached. Falling back to pinv keeps the result finite instead of
    # propagating NaN/inf through the assembled stiffness.
    try:
        inner_inv = np.linalg.inv(inner)
    except np.linalg.LinAlgError:
        inner_inv = np.linalg.pinv(inner)
    C_eff = C_m @ (I6 - Vp * inner_inv)
    if not np.all(np.isfinite(C_eff)):
        # Last-ditch: return the void-saturated zero stiffness rather than
        # silently emitting NaN/inf from a near-singular inner. The wrapper
        # always returns a defensive copy of whatever this inner returns,
        # so callers stay safe from mutation either way.
        return np.zeros((6, 6))
    return C_eff


@lru_cache(maxsize=_MT_CACHE_MAXSIZE)
def _mt_effective_stiffness_cached(C_m_00: float, C_m_33: float, Vp: float,
                                   void_shape_radii: tuple[float, ...],
                                   nu_m: float) -> np.ndarray:
    """LRU-cached core of :func:`_mt_effective_stiffness` (#112).

    All arguments are hashable (floats / tuples of floats), so this is a
    drop-in for ``functools.lru_cache``. The matrix stiffness ``C_m`` is
    reconstructed from its two independent isotropic components
    (``C[0,0] = lam + 2mu`` and ``C[3,3] = mu``); see the wrapper for the
    fingerprint rationale.
    """
    # Reconstruct the isotropic 6x6 matrix stiffness from its two
    # independent components. C[0,0] = lam + 2mu, C[3,3] = mu, so
    # lam = C[0,0] - 2*C[3,3]; off-diagonal C[i!=j] = lam.
    mu = C_m_33
    lam = C_m_00 - 2.0 * mu
    C_m = np.zeros((6, 6))
    C_m[0, 0] = C_m[1, 1] = C_m[2, 2] = lam + 2.0 * mu
    C_m[0, 1] = C_m[0, 2] = C_m[1, 0] = C_m[1, 2] = C_m[2, 0] = C_m[2, 1] = lam
    C_m[3, 3] = C_m[4, 4] = C_m[5, 5] = mu

    nu = nu_m
    r = list(void_shape_radii)
    sphere_tol = 0.01

    S = np.zeros((6, 6))

    def _close(a, b):
        return abs(a - b) / max(a, b) < sphere_tol

    is_sphere = (max(r) - min(r)) / max(r) < sphere_tol
    if not is_sphere and not (_close(r[1], r[2]) or _close(r[0], r[2])
                              or _close(r[0], r[1])):
        # Triaxial ellipsoid: no closed form. Use the general-ellipsoid
        # tensor from Mura's integrals, already in the engineering-shear
        # Voigt form (IMPROVEMENT_PLAN 2.8; this used to approximate the
        # void as a spheroid about its longest axis).
        from .void_geometry import _eshelby_tensor
        S_eng = _eshelby_tensor((float(r[0]), float(r[1]), float(r[2])), nu)
        return _mori_tanaka_voids(C_m, S_eng, Vp)

    if is_sphere:
        # All three radii within 1% — treat as sphere.
        S[0, 0] = S[1, 1] = S[2, 2] = (7 - 5 * nu) / (15 * (1 - nu))
        S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = \
            (5 * nu - 1) / (15 * (1 - nu))
        S[3, 3] = S[4, 4] = S[5, 5] = (4 - 5 * nu) / (15 * (1 - nu))
    else:
        # Axisymmetric: find the symmetry axis (the radius that differs
        # from the other two equal radii).
        if _close(r[1], r[2]):
            idx_axis = 0  # a_1 is the unique axis
        elif _close(r[0], r[2]):
            idx_axis = 1
        else:
            idx_axis = 2

        a_axis = r[idx_axis]
        a_eq = r[(idx_axis + 1) % 3]  # equatorial radius
        alpha = a_axis / a_eq
        a2 = alpha ** 2

        # g-function with correct branch:
        #   prolate (alpha > 1): cosh-based form
        #   oblate  (alpha < 1): arccos-based form
        # The S-tensor formulas below are identical in structure for both;
        # only the g-function and the sign of (alpha^2 - 1) differ.
        if alpha > 1.0:
            g = alpha / (a2 - 1) ** 1.5 * (
                alpha * np.sqrt(a2 - 1) - np.arccosh(alpha))
        else:
            g = alpha / (1 - a2) ** 1.5 * (
                np.arccos(alpha) - alpha * np.sqrt(1 - a2))

        # Eshelby tensor in canonical frame (symmetry axis along x_1).
        S[0, 0] = (1.0 / (2 * (1 - nu))) * (
            1 - 2 * nu + (3 * a2 - 1) / (a2 - 1) - (1 - 2 * nu + 3 * a2 / (a2 - 1)) * g)
        S[1, 1] = S[2, 2] = (3.0 / (8 * (1 - nu))) * a2 / (a2 - 1) + \
            (1.0 / (4 * (1 - nu))) * (1 - 2 * nu - 9.0 / (4 * (a2 - 1))) * g
        # Voigt row = output strain, column = input strain, so S[0, 1] is
        # S_1122 and S[1, 0] is S_2211 (Tandon & Weng 1984).
        S[0, 1] = S[0, 2] = -(1.0 / (2 * (1 - nu))) * (
            1 - 2 * nu + 1.0 / (a2 - 1)) + (1.0 / (2 * (1 - nu))) * (
            1 - 2 * nu + 3.0 / (2 * (a2 - 1))) * g
        S[1, 0] = S[2, 0] = -(1.0 / (2 * (1 - nu))) * a2 / (a2 - 1) + \
            (1.0 / (4 * (1 - nu))) * (3 * a2 / (a2 - 1) - (1 - 2 * nu)) * g
        S[1, 2] = S[2, 1] = (1.0 / (4 * (1 - nu))) * (
            a2 / (2 * (a2 - 1)) - (1 - 2 * nu + 3.0 / (4 * (a2 - 1))) * g)
        S[3, 3] = (1.0 / (4 * (1 - nu))) * (
            a2 / (2 * (a2 - 1)) + (1 - 2 * nu - 3.0 / (4 * (a2 - 1))) * g)
        S[4, 4] = S[5, 5] = (1.0 / (4 * (1 - nu))) * (
            1 - 2 * nu - (a2 + 1) / (a2 - 1) -
            0.5 * (1 - 2 * nu - 3 * (a2 + 1) / (a2 - 1)) * g)

        # Permute the Voigt tensor to align the symmetry axis with the
        # actual unique-radius axis. For a swap x_1 <-> x_k the Voigt
        # permutation is:
        #   x_1 <-> x_2: [1, 0, 2, 4, 3, 5]
        #   x_1 <-> x_3: [2, 1, 0, 5, 4, 3]
        if idx_axis != 0:
            if idx_axis == 1:
                perm = [1, 0, 2, 4, 3, 5]
            else:
                perm = [2, 1, 0, 5, 4, 3]
            P = np.eye(6)[perm]
            S = P @ S @ P.T

    # The closed forms above are the tensor components S_2323, S_1313,
    # S_1212. The Voigt form used here maps engineering shear strain to
    # engineering shear strain (gamma = 2 eps), so its shear diagonal is
    # twice the tensor component. With this factor the spherical-void
    # result matches the closed-form Mori-Tanaka shear modulus exactly.
    S[3, 3] *= 2.0
    S[4, 4] *= 2.0
    S[5, 5] *= 2.0

    return _mori_tanaka_voids(C_m, S, Vp)


#: Porosity degradation mappings accepted by ``porosity_degradation=`` (see
#: :data:`porosity_fe.PorosityDegradation`). ``'isotropic'`` projects the
#: Mori-Tanaka tensor of the porous matrix onto an isotropic one before the
#: Halpin-Tsai ratios; ``'directional'`` keeps it transversely isotropic and
#: degrades each ply constant from its own matrix direction.
POROSITY_DEGRADATIONS: tuple[str, ...] = ('isotropic', 'directional')

#: Mapping used when ``porosity_degradation=`` is not given, by
#: :class:`~porosity_fe.fe.solver.FESolver`, ``GlobalAssembler``,
#: ``build_element_batch``, ``Hex8Element`` and the failure criteria
#: alike: the historical isotropic projection.
_DEFAULT_POROSITY_DEGRADATION: PorosityDegradation = 'isotropic'


def _check_porosity_degradation(porosity_degradation: str) -> str:
    """Return ``porosity_degradation`` if it is supported, else raise ``ValueError``."""
    if porosity_degradation not in POROSITY_DEGRADATIONS:
        raise ValueError(
            f"Unknown porosity_degradation {porosity_degradation!r}. "
            f"Use one of {list(POROSITY_DEGRADATIONS)}."
        )
    return porosity_degradation


def _isotropic_matrix_constants(C_eff: np.ndarray, nu_m: float
                                ) -> tuple[float, float, float]:
    """``(E_m*, G_m*, nu_m*)`` of the isotropic (Voigt-average) projection
    of the Mori-Tanaka tensor ``C_eff`` (``porosity_degradation='isotropic'``).

    For spherical voids ``C_eff`` is isotropic and this returns its Lame
    constants exactly; cylindrical / penny voids give an anisotropic
    ``C_eff``, whose directions the projection averages away. ``E_m*`` and
    ``G_m*`` are floored at 1 MPa.
    """
    bulk_eff = (np.trace(C_eff[:3, :3])
                + 2.0 * (C_eff[0, 1] + C_eff[0, 2] + C_eff[1, 2])) / 9.0
    mu_eff = (np.trace(C_eff[:3, :3])
              - (C_eff[0, 1] + C_eff[0, 2] + C_eff[1, 2])
              + 3.0 * np.trace(C_eff[3:, 3:])) / 15.0
    lam_eff = bulk_eff - 2.0 * mu_eff / 3.0
    denom = lam_eff + mu_eff
    G_m_eff = max(mu_eff, 1.0)
    E_m_eff = mu_eff * (3.0 * lam_eff + 2.0 * mu_eff) / denom if denom > 1e-12 else 1.0
    E_m_eff = max(E_m_eff, 1.0)
    nu_m_eff = lam_eff / (2.0 * denom) if denom > 1e-12 else nu_m
    return E_m_eff, G_m_eff, nu_m_eff


def _directional_matrix_constants(C_eff: np.ndarray, nu_m: float
                                  ) -> tuple[float, float, float, float,
                                             float, float, float, float]:
    """Engineering constants of the porous matrix read from the Mori-Tanaka
    tensor ``C_eff`` direction by direction
    (``porosity_degradation='directional'``).

    From the normal block of the compliance ``S* = inv(C*[:3, :3])``:
    ``E*_i = 1 / S*_ii``, ``nu*_12 = -S*_12 / S*_11``,
    ``nu*_13 = -S*_13 / S*_11``; the shear moduli are the shear diagonal,
    ``G*_23, G*_13, G*_12 = C*_44, C*_55, C*_66``. Returns ``(E*_1, E*_2,
    E*_3, G*_23, G*_13, G*_12, nu*_12, nu*_13)`` in the void's axes, which
    are the ply's material axes: ``a_1`` along the fibres, ``a_3`` through
    the thickness. Moduli are floored at 1 MPa; a Poisson's ratio that
    cannot be read (singular compliance) falls back to ``nu_m``.
    """
    try:
        S = np.linalg.inv(C_eff[:3, :3])
    except np.linalg.LinAlgError:
        S = np.linalg.pinv(C_eff[:3, :3])
    E = [1.0 / S[i, i] if S[i, i] > 1e-300 else 1.0 for i in range(3)]
    E = [max(e, 1.0) if np.isfinite(e) else 1.0 for e in E]
    ok = S[0, 0] > 1e-300
    nu12 = float(-S[0, 1] / S[0, 0]) if ok else nu_m
    nu13 = float(-S[0, 2] / S[0, 0]) if ok else nu_m
    G = [max(float(C_eff[k, k]), 1.0) for k in (3, 4, 5)]
    return (E[0], E[1], E[2], G[0], G[1], G[2],
            nu12 if np.isfinite(nu12) else nu_m,
            nu13 if np.isfinite(nu13) else nu_m)


def _degraded_composite_stiffness(Vp: float, void_shape_radii: tuple,
                                  mat: MaterialProperties, *,
                                  porosity_degradation: PorosityDegradation =
                                  _DEFAULT_POROSITY_DEGRADATION) -> np.ndarray:
    """Build porosity-degraded composite stiffness via component-wise micromechanics.

    Rather than applying a single scalar degradation ratio, this function:
    1. Uses Mori-Tanaka to get degraded matrix constants (E_m*, G_m*, nu_m*)
    2. Re-applies rule-of-mixtures / Halpin-Tsai with the degraded matrix to
       get degradation *ratios* of the composite engineering constants
       (E11*, E22*, G12*, etc.), applied to the measured ply constants
    3. Builds the full 6x6 stiffness from those degraded constants.

    This correctly captures that porosity (matrix voids) barely affects
    fiber-dominated E11 but strongly degrades matrix-dominated E22 and G12.

    Parameters
    ----------
    Vp : float
        Local void volume fraction (0 to 1).
    void_shape_radii : tuple
        (a1, a2, a3) for Eshelby tensor, in the ply's material axes: ``a1``
        along the fibres, ``a2`` transverse in the ply plane, ``a3``
        through the thickness (a penny void lies in the ply plane, a
        cylindrical one runs along the fibres).
    mat : MaterialProperties
        Material with constituent properties (E_f, E_m, V_f, etc.).
    porosity_degradation : {'isotropic', 'directional'}, optional
        Keyword-only. ``'isotropic'`` (default) takes the isotropic
        (Voigt-average) projection of the Mori-Tanaka tensor, so one
        ``E_m*``, ``G_m*`` and ``nu_m*`` degrade every ply constant:
        ``E33`` by the ratio of ``E22``, ``G13`` and ``G23`` by that of
        ``G12``, ``nu13`` by that of ``nu12``. ``'directional'`` keeps the
        transversely isotropic Mori-Tanaka tensor and degrades each ply
        constant from its own matrix direction: ``E11`` by rule of mixtures
        with ``E*_1``; ``E22`` and ``E33`` by Halpin-Tsai (``xi = 2``) with
        ``E*_2`` and ``E*_3``; ``G12``, ``G13`` and ``G23`` by Halpin-Tsai
        (``xi = 1``) with their own matrix shear modulus; ``nu12`` and
        ``nu13`` by rule of mixtures with ``nu*_12`` and ``nu*_13``;
        ``nu23`` unchanged. For spherical voids the Mori-Tanaka tensor is
        isotropic and the two mappings agree to round-off; for penny
        (oblate) and cylindrical voids only ``'directional'`` weakens the
        out-of-plane constants more than the in-plane ones (see the theory
        page on micromechanics).

    Returns
    -------
    np.ndarray
        Shape (6, 6) degraded composite stiffness in material coordinates.

    Raises
    ------
    ValueError
        On an unknown ``porosity_degradation``, or (``'directional'``
        only) when the degraded ply compliance is not positive definite,
        which no preset reaches for any void shape at ``Vp <= 0.99``.
    """
    if Vp < 1e-12:
        return mat.get_stiffness_matrix()
    if Vp > 0.99:
        return np.zeros((6, 6))
    _check_porosity_degradation(porosity_degradation)

    E_m = mat.matrix_modulus
    nu_m = mat.matrix_poisson
    G_m = E_m / (2.0 * (1.0 + nu_m))
    E_f = mat.fiber_modulus
    Vf = mat.fiber_volume_fraction
    Vm = 1.0 - Vf

    # --- Step 1: Degraded matrix properties from Mori-Tanaka ---
    C_m = mat.get_isotropic_matrix_stiffness()
    C_eff = _mt_effective_stiffness(C_m, Vp, void_shape_radii, nu_m)

    # The porous matrix constants each ply constant is degraded from:
    # (E*_1, E*_2, E*_3, G*_23, G*_13, G*_12, nu*_12, nu*_13). The isotropic
    # projection uses one E_m*, G_m* and nu_m* for every direction.
    if porosity_degradation == 'directional':
        (E1_eff, E2_eff, E3_eff, G23_eff, G13_eff, G12_eff,
         nu12_eff, nu13_eff) = _directional_matrix_constants(C_eff, nu_m)
    else:
        E_m_eff, G_m_eff, nu_m_eff = _isotropic_matrix_constants(C_eff, nu_m)
        E1_eff = E2_eff = E3_eff = E_m_eff
        G23_eff = G13_eff = G12_eff = G_m_eff
        nu12_eff = nu13_eff = nu_m_eff

    # --- Step 2: Compute degradation RATIOS via micromechanics ---
    # Use Halpin-Tsai with pristine and degraded matrix to get ratios,
    # then apply ratios to actual measured composite properties.
    # This avoids mismatch between micromechanics predictions and actual data.

    nu_f = mat.fiber_poisson
    G_f = mat.fiber_shear_modulus_eff

    # E11 ratio (Rule of Mixtures — fiber-dominated, barely affected)
    E11_rom_prist = Vf * E_f + Vm * E_m
    E11_rom_deg = Vf * E_f + Vm * E1_eff
    r_E11 = E11_rom_deg / E11_rom_prist  # ~0.999 for 5% porosity

    # E22, E33 ratios (Halpin-Tsai — matrix-dominated, strongly affected)
    xi_HT = 2.0
    def _halpin_tsai(Ef, Em, xi, vf):
        ratio = Ef / Em
        eta = (ratio - 1.0) / (ratio + xi)
        return Em * (1.0 + xi * eta * vf) / (1.0 - eta * vf)

    E22_HT_prist = _halpin_tsai(E_f, E_m, xi_HT, Vf)
    r_E22 = _halpin_tsai(E_f, E2_eff, xi_HT, Vf) / E22_HT_prist
    r_E33 = _halpin_tsai(E_f, E3_eff, xi_HT, Vf) / E22_HT_prist

    # G12, G13, G23 ratios (Halpin-Tsai — matrix-dominated). The same
    # Halpin-Tsai form (xi = 1, fiber G_f) for all three: with the
    # isotropic projection the three ratios are equal by construction (a
    # G23-specific xi would change results without data to calibrate it
    # against); the directional mapping separates them through the matrix
    # shear moduli only.
    xi_G = 1.0
    G12_HT_prist = _halpin_tsai(G_f, G_m, xi_G, Vf)
    r_G12 = _halpin_tsai(G_f, G12_eff, xi_G, Vf) / G12_HT_prist
    r_G13 = _halpin_tsai(G_f, G13_eff, xi_G, Vf) / G12_HT_prist
    r_G23 = _halpin_tsai(G_f, G23_eff, xi_G, Vf) / G12_HT_prist

    # nu12, nu13 ratios (Rule of Mixtures — weakly affected)
    nu12_rom_prist = Vf * nu_f + Vm * nu_m
    if abs(nu12_rom_prist) > 1e-12:
        r_nu12 = (Vf * nu_f + Vm * nu12_eff) / nu12_rom_prist
        r_nu13 = (Vf * nu_f + Vm * nu13_eff) / nu12_rom_prist
    else:
        r_nu12 = r_nu13 = 1.0

    # --- Step 3: Apply ratios to actual measured properties ---
    E11_deg = mat.E11 * r_E11
    E22_deg = mat.E22 * r_E22
    E33_deg = mat.E33 * r_E33  # the ratio of E22 for 'isotropic'
    G12_deg = mat.G12 * r_G12
    G13_deg = mat.G13 * r_G13  # the ratio of G12 for 'isotropic'
    G23_deg = mat.G23 * r_G23
    nu12_deg = mat.nu12 * r_nu12
    nu13_deg = mat.nu13 * r_nu13  # the ratio of nu12 for 'isotropic'
    nu23_deg = mat.nu23  # weakly affected, keep constant

    # --- Step 4: Build compliance matrix from degraded constants ---
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E11_deg
    S[1, 1] = 1.0 / E22_deg
    S[2, 2] = 1.0 / E33_deg
    S[0, 1] = S[1, 0] = -nu12_deg / E11_deg
    S[0, 2] = S[2, 0] = -nu13_deg / E11_deg
    S[1, 2] = S[2, 1] = -nu23_deg / E22_deg
    S[3, 3] = 1.0 / G23_deg
    S[4, 4] = 1.0 / G13_deg
    S[5, 5] = 1.0 / G12_deg

    if porosity_degradation == 'directional':
        # The directional constants are read from different directions of
        # one tensor, so check that they still make a physical (positive
        # definite) compliance before it is inverted. With nu23 held at its
        # pristine value this needs nu23^2 < E22* / E33*, which only a flat
        # void normal to the transverse in-plane axis (radii like
        # (3, 0.3, 3), a transverse-crack shape) breaks, at Vp >= 0.4; the
        # three preset shapes pass for every preset up to Vp = 0.99.
        try:
            np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"porosity_degradation='directional' gives a ply compliance "
                f"that is not positive definite at Vp={Vp:.4g} for void "
                f"shape radii {tuple(void_shape_radii)}: E11={E11_deg:.4g}, "
                f"E22={E22_deg:.4g}, E33={E33_deg:.4g} MPa, nu12="
                f"{nu12_deg:.4g}, nu13={nu13_deg:.4g}, nu23={nu23_deg:.4g} "
                f"(nu23 is held at its pristine value, so the void must not "
                f"weaken E22 past nu23^2 E33). Use porosity_degradation="
                f"'isotropic' for this void shape at this porosity."
            ) from None

    return np.linalg.inv(S)


# ============================================================
# CLT effective moduli (uses _degraded_composite_stiffness from above)
# ============================================================

def compute_clt_effective_modulus(material: MaterialProperties,
                                  ply_angles: list[float]) -> float:
    """Compute effective laminate longitudinal modulus using CLT (ABD matrix).

    Builds the full A-matrix (in-plane stiffness) from Classical Lamination
    Theory, then computes the effective Ex from the A-matrix inverse.

    Parameters
    ----------
    material : MaterialProperties
        Material with orthotropic ply-level properties.
    ply_angles : list of float
        Ply orientation angles in degrees (one per ply).

    Returns
    -------
    float
        Effective longitudinal modulus E_x (MPa).
    """
    h_total = len(ply_angles) * material.t_ply
    A, _ = _build_clt_abd(material, ply_angles, material.get_stiffness_matrix())

    # Effective modulus: E_x = (A11*A22 - A12^2) / (A22 * h)
    # From a_ij = A_inv, E_x = 1 / (h * a_11)
    A_inv = np.linalg.inv(A)
    E_x = 1.0 / (h_total * A_inv[0, 0])
    return float(E_x)


def _build_clt_abd(material: MaterialProperties, ply_angles: list[float],
                   C_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build CLT A (membrane) and D (bending) matrices from a 6x6 stiffness."""
    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    A_mat = np.zeros((3, 3))
    D_mat = np.zeros((3, 3))

    idx = [0, 1, 5]  # 11, 22, 12 in Voigt
    z_k_prev = -h_total / 2.0

    for angle_deg in ply_angles:
        z_k = z_k_prev + t_ply
        z_mid = (z_k_prev + z_k) / 2.0

        angle_rad = np.radians(float(angle_deg))
        if abs(angle_rad) > 1e-15:
            C_rot = rotate_stiffness_3d(C_base, angle_rad, axis='z')
        else:
            C_rot = C_base

        # Plane-stress reduced stiffness Q_bar
        Q_bar = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ii, jj = idx[i], idx[j]
                if abs(C_rot[2, 2]) > 1e-12:
                    Q_bar[i, j] = C_rot[ii, jj] - C_rot[ii, 2] * C_rot[jj, 2] / C_rot[2, 2]
                else:
                    Q_bar[i, j] = C_rot[ii, jj]

        A_mat += Q_bar * t_ply
        D_mat += Q_bar * (t_ply * z_mid**2 + t_ply**3 / 12.0)

        z_k_prev = z_k

    return A_mat, D_mat


def _clt_abd_full(C_base: np.ndarray, ply_angles: list[float], t_ply: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLT ``A``, ``B`` and ``D`` (3 x 3 each) of a possibly unsymmetric stack.

    The plies (angles in degrees, thickness ``t_ply`` each) are stacked from
    ``z = -h/2`` to ``h/2`` in the order given, about the stack's own
    mid-plane, with the plane-stress reduced stiffness of
    :func:`_build_clt_abd` (``sigma_33 = 0`` condensed out of the rotated
    6 x 6 stiffness). Unlike :func:`_build_clt_abd` it forms the coupling
    matrix ``B``, for the sublaminates of a delaminated laminate.
    """
    idx = [0, 1, 5]  # 11, 22, 12 in Voigt
    h = len(ply_angles) * t_ply
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, angle_deg in enumerate(ply_angles):
        z0 = -h / 2.0 + k * t_ply
        z1 = z0 + t_ply
        angle_rad = np.radians(float(angle_deg))
        C_rot = rotate_stiffness_3d(C_base, angle_rad, axis='z') \
            if abs(angle_rad) > 1e-15 else C_base
        Q_bar = C_rot[np.ix_(idx, idx)] - np.outer(C_rot[idx, 2], C_rot[idx, 2]) / C_rot[2, 2]
        A += Q_bar * (z1 - z0)
        B += Q_bar * (z1 ** 2 - z0 ** 2) / 2.0
        D += Q_bar * (z1 ** 3 - z0 ** 3) / 3.0
    return A, B, D


def compute_degraded_clt_moduli(material: MaterialProperties,
                                ply_angles: list[float],
                                Vp: float,
                                method: str = 'mori_tanaka') -> dict[str, float]:
    """Compute effective laminate in-plane moduli (Ex, Ey, Gxy) with porosity.

    Uses Mori-Tanaka-degraded ply stiffness via _degraded_composite_stiffness,
    then builds the CLT A-matrix to extract effective moduli.

    Parameters
    ----------
    material : MaterialProperties
    ply_angles : list of float  (degrees)
    Vp : float  void volume fraction (0–1)
    method : str  ignored (kept for API symmetry); always uses Mori-Tanaka.

    Returns
    -------
    dict with keys: 'Ex', 'Ey', 'Gxy' (all in MPa)
    """
    void_shape_radii = VOID_SHAPES['spherical']  # spherical default
    C_deg = _degraded_composite_stiffness(Vp, void_shape_radii, material)

    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    A_mat, _ = _build_clt_abd(material, ply_angles, C_deg)

    a_inv = np.linalg.inv(A_mat)
    Ex = 1.0 / (h_total * a_inv[0, 0])
    Ey = 1.0 / (h_total * a_inv[1, 1])
    Gxy = 1.0 / (h_total * a_inv[2, 2])

    return {'Ex': float(Ex), 'Ey': float(Ey), 'Gxy': float(Gxy)}


def compute_degraded_clt_flexural_modulus(material: MaterialProperties,
                                          ply_angles: list[float],
                                          Vp: float,
                                          method: str = 'mori_tanaka') -> dict[str, float]:
    """Compute effective laminate flexural modulus Ef_x with porosity.

    Uses the CLT D-matrix (bending stiffness) to compute the flexural modulus.

    Parameters
    ----------
    material : MaterialProperties
    ply_angles : list of float  (degrees)
    Vp : float  void volume fraction (0–1)
    method : str  ignored; always uses Mori-Tanaka.

    Returns
    -------
    dict with key: 'Ef_x' (MPa)
    """
    void_shape_radii = VOID_SHAPES['spherical']
    C_deg = _degraded_composite_stiffness(Vp, void_shape_radii, material)

    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    _, D_mat = _build_clt_abd(material, ply_angles, C_deg)

    d_inv = np.linalg.inv(D_mat)
    # Flexural modulus: Ef_x = 12 / (h^3 * d_11_inv)
    Ef_x = 12.0 / (h_total**3 * d_inv[0, 0])

    return {'Ef_x': float(Ef_x)}
