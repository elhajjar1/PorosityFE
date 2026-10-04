"""FE failure criteria and porosity-degraded ply strengths.

Pure functions over recovered ply-local stresses, used by
:meth:`FESolver.solve <porosity_fe.fe.solver.FESolver.solve>`; split out of
``fe/solver.py`` (IMPROVEMENT_PLAN 5.1). :class:`FESolver` keeps its
``_degraded_strengths`` / ``_evaluate_*`` methods as thin wrappers.
"""

from __future__ import annotations

import numpy as np

from ..empirical import Calibration
from ..homogenization import _mt_effective_stiffness
from ..materials import MaterialProperties
from .element import VOID_VP_THRESHOLD

#: Failure criteria accepted by :func:`evaluate_failure` and
#: :meth:`FESolver.solve <porosity_fe.fe.solver.FESolver.solve>`.
SUPPORTED_FAILURE_CRITERIA: tuple[str, ...] = ('tsai_wu', 'hashin', 'max_stress')

#: Empty per-mode failure-index dict used when an element is skipped
#: (void) or for criteria that do not populate a particular mode.
EMPTY_MODE_FI: dict[str, float] = {
    'max_fi': 0.0,
    'fiber_t': 0.0,
    'fiber_c': 0.0,
    'matrix_t': 0.0,
    'matrix_c': 0.0,
    'shear': 0.0,
    'delamination': 0.0,
}

def _element_porosity_and_skip(porosity: np.ndarray, elements: np.ndarray,
                               void_elements=None) -> tuple[np.ndarray, np.ndarray]:
    """Nodal-mean element porosity and the mask of elements failure ignores.

    An element is skipped if its mean porosity exceeds
    :data:`~porosity_fe.fe.element.VOID_VP_THRESHOLD` or it is a geometric
    void (``void_elements``, the indices in ``CompositeMesh.void_elements``).
    """
    elem_Vp = np.clip(np.mean(porosity[elements], axis=1), 0.0, 1.0)
    skip = elem_Vp > VOID_VP_THRESHOLD
    if void_elements is not None:
        void_idx = np.asarray(void_elements, dtype=np.intp).ravel()
        if void_idx.size:
            skip[void_idx] = True
    return elem_Vp, skip


def degraded_strengths(material: MaterialProperties, void_shape_radii: tuple,
                       elem_Vp: float
                       ) -> tuple[float, float, float, float, float, float]:
    """Return per-element porosity-degraded ply strengths.

    Implements the strength-degradation block shared by all failure
    criteria. Fiber-direction strengths (``Xt``, ``Xc``) follow the rule-
    of-mixtures fiber ratio (matrix porosity has only a weak indirect
    effect via ``E_m_eff``); transverse and shear strengths
    (``Yt``, ``Yc``, ``S12``, ``S23``) follow the Mori-Tanaka matrix
    stiffness ratio. Strengths are clamped to a small numerical floor so
    the per-criterion polynomial cannot divide by zero.

    Scaling rule
    ------------
    Per-component strengths are scaled by the square root of the
    corresponding stiffness retention ratio. Concretely, for the
    matrix-dominated components::

        r_matrix = sqrt(C_eff[0, 0] / C_pristine[0, 0])
        Yt  = sigma_2t  * r_matrix
        Yc  = sigma_2c  * r_matrix
        S12 = tau_12    * r_matrix
        S23 = tau_ilss  * r_matrix

    and analogously for the fiber-direction components via a rule-of-
    mixtures effective-modulus ratio (also taken under a square root).

    Heuristic correlation: per-component strength scales as the square
    root of the stiffness retention ratio. Loosely motivated by Puck-
    style strength-stiffness coupling (Puck & Schurmann 2002), but not
    directly derived from a published model and not validated against
    experimental degraded-strength data within PorosityFE. Alternative
    scalings exist in the literature -- e.g. linear (``r``), quadratic
    (``r**2``), and Davila-style critical-element forms -- and a future
    kwarg may surface them; today only the sqrt rule is implemented.

    FE vs. empirical divergence
    ---------------------------
    The empirical solver uses calibrated knockdown forms (Judd-Wright,
    power-law, linear) for the same physical effect; the two paths use
    fundamentally different mathematical forms, so FE and empirical
    predictions may diverge for the same ``(layup, Vp)``. This is by
    design -- see the README section on Solver Selection (FE vs.
    empirical) for guidance on which path to trust for which question.

    Parameters
    ----------
    elem_Vp : float
        Element-average void volume fraction in [0, 1].

    Returns
    -------
    (Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s) : tuple of floats
        Floor-clamped degraded strengths in MPa.
    """
    mat = material
    C_m_pristine = mat.get_isotropic_matrix_stiffness()
    if elem_Vp > 1e-12:
        C_eff = _mt_effective_stiffness(
            C_m_pristine, elem_Vp,
            void_shape_radii,
            mat.matrix_poisson)
        # Matrix stiffness degradation ratio (matrix-dominated)
        r_matrix = np.sqrt(max(C_eff[0, 0] / C_m_pristine[0, 0], 0.0))
        # Fiber-direction ratio: scale by ROM ratio (much weaker effect)
        E_m = mat.matrix_modulus
        E_m_eff_approx = E_m * max(C_eff[0, 0] / C_m_pristine[0, 0], 0.0)
        Vf = mat.fiber_volume_fraction
        Vm = 1.0 - Vf
        r_fiber = (Vf * mat.fiber_modulus + Vm * E_m_eff_approx) / \
                  (Vf * mat.fiber_modulus + Vm * E_m)
        r_fiber = np.sqrt(max(r_fiber, 0.0))  # sqrt for strength vs stiffness
    else:
        r_matrix = 1.0
        r_fiber = 1.0

    Xt = mat.sigma_1t * r_fiber
    Xc = mat.sigma_1c * r_fiber
    Yt = mat.sigma_2t * r_matrix
    Yc = mat.sigma_2c * r_matrix
    S12 = mat.tau_12 * r_matrix
    S23 = mat.tau_ilss * r_matrix

    # Strengths approaching zero make the 1/X reciprocals overflow to
    # inf; clamp to a numerical floor so a heavily-degraded element
    # produces a large-but-finite failure index instead of poisoning the
    # global max with inf/NaN. Canonical value in Calibration (#121).
    strength_floor = Calibration.STRENGTH_FLOOR_MPA  # MPa
    return (max(Xt, strength_floor),
            max(Xc, strength_floor),
            max(Yt, strength_floor),
            max(Yc, strength_floor),
            max(S12, strength_floor),
            max(S23, strength_floor))

def evaluate_failure(stress_local: np.ndarray, porosity: np.ndarray,
                     elements: np.ndarray, material: MaterialProperties,
                     void_shape_radii: tuple, criterion: str = 'tsai_wu',
                     void_elements=None
                     ) -> tuple[float, np.ndarray, dict[str, float]]:
    """Evaluate the chosen failure criterion at every Gauss point.

    Dispatches to :func:`evaluate_tsai_wu`, :func:`evaluate_hashin`,
    or :func:`evaluate_max_stress` element by element. Per-element
    strength degradation is computed once via :func:`degraded_strengths`
    and reused by the per-criterion polynomials.

    Parameters
    ----------
    stress_local : np.ndarray
        Shape (n_elem, n_gp, 6) local stresses.
    porosity : np.ndarray
        Shape (n_nodes,) nodal void volume fraction.
    elements : np.ndarray
        Shape (n_elem, 8) hex connectivity.
    material : MaterialProperties
        Pristine ply properties (strengths, constituents, ``tsai_wu_F12``).
    void_shape_radii : tuple
        Void shape used by the Mori-Tanaka strength degradation.
    criterion : {'tsai_wu', 'hashin', 'max_stress'}
        Failure criterion to apply.
    void_elements : array-like of int, optional
        Geometric void elements (``CompositeMesh.void_elements``). They are
        skipped, as are elements whose mean porosity exceeds
        :data:`~porosity_fe.fe.element.VOID_VP_THRESHOLD`.

    Returns
    -------
    max_fi : float
        Overall maximum failure index.
    per_elem_fi : np.ndarray
        Shape (n_elem,) max-over-Gauss-point failure index per element
        (0.0 for skipped void elements).
    mode_indices : dict
        Per-mode breakdown at the element/GP where ``max_fi`` is
        attained. Keys: ``'max_fi'``, ``'fiber_t'``, ``'fiber_c'``,
        ``'matrix_t'``, ``'matrix_c'``, ``'shear'``. For Tsai-Wu the
        per-mode entries are ``NaN`` (the coupled polynomial does not
        separate modes).
    """
    if criterion not in SUPPORTED_FAILURE_CRITERIA:
        raise ValueError(
            f"Unknown failure criterion {criterion!r}. "
            f"Use one of {list(SUPPORTED_FAILURE_CRITERIA)}."
        )

    n_elem, _, _ = stress_local.shape
    per_elem_fi = np.zeros(n_elem, dtype=float)
    # Defense in depth: non-finite porosity silently corrupts elem_Vp.
    if not np.all(np.isfinite(porosity)):  # type: ignore[call-overload]
        raise ValueError(
            f"mesh.porosity contains non-finite values; refusing to evaluate "
            f"{criterion} on a corrupted porosity field."
        )

    # #114: hoist the per-element mean computation outside the loop.
    # The old `np.mean(porosity[elements[e]])` per iteration was an
    # O(n_elem) Python loop where O(1) vectorized NumPy works; this gives
    # ~143x on the inner step and ~1-2 s on a typical 5x5 sweep.
    elem_Vp_all, skip = _element_porosity_and_skip(porosity, elements, void_elements)

    max_fi = 0.0
    best_mode_indices: dict[str, float] = dict(EMPTY_MODE_FI)
    if criterion == 'tsai_wu':
        # Tsai-Wu polynomial couples all components — per-mode breakdown
        # is undefined. Surface NaN sentinels so downstream consumers see
        # "criterion did not separate modes" rather than spurious zeros.
        best_mode_indices = {
            'max_fi': 0.0,
            'fiber_t': float('nan'),
            'fiber_c': float('nan'),
            'matrix_t': float('nan'),
            'matrix_c': float('nan'),
            'shear': float('nan'),
            'delamination': float('nan'),
        }

    for e in range(n_elem):
        elem_Vp = float(elem_Vp_all[e])

        # Skip void elements (carry no meaningful load)
        if skip[e]:
            continue

        strengths = degraded_strengths(material, void_shape_radii, elem_Vp)
        s_all = stress_local[e]  # (n_gp, 6)

        if criterion == 'tsai_wu':
            fi_per_gp = evaluate_tsai_wu(s_all, strengths, e, elem_Vp,
                                         material.tsai_wu_F12)
            elem_max = float(fi_per_gp.max())
            per_elem_fi[e] = elem_max
            if elem_max > max_fi:
                max_fi = elem_max
                best_mode_indices['max_fi'] = elem_max
        else:
            if criterion == 'hashin':
                mode_fi_per_gp = evaluate_hashin(s_all, strengths)
            else:  # 'max_stress'
                mode_fi_per_gp = evaluate_max_stress(s_all, strengths)
            # mode_fi_per_gp is a dict of arrays, each shape (n_gp,).
            fi_per_gp = mode_fi_per_gp['max_fi']
            if not np.all(np.isfinite(fi_per_gp)):
                bad_g = int(np.argmax(~np.isfinite(fi_per_gp)))
                raise ValueError(
                    f"{criterion} failure index is non-finite at element "
                    f"{e}, Gauss point {bad_g} (Vp={elem_Vp:.4f}, "
                    f"stress={s_all[bad_g].tolist()})."
                )
            elem_max = float(fi_per_gp.max())
            per_elem_fi[e] = elem_max
            if elem_max > max_fi:
                max_fi = elem_max
                g_max = int(np.argmax(fi_per_gp))
                best_mode_indices = {
                    'max_fi': elem_max,
                    'fiber_t': float(mode_fi_per_gp['fiber_t'][g_max]),
                    'fiber_c': float(mode_fi_per_gp['fiber_c'][g_max]),
                    'matrix_t': float(mode_fi_per_gp['matrix_t'][g_max]),
                    'matrix_c': float(mode_fi_per_gp['matrix_c'][g_max]),
                    'shear': float(mode_fi_per_gp['shear'][g_max]),
                    'delamination': float(mode_fi_per_gp['delamination'][g_max]),
                }

    return float(max_fi), per_elem_fi, best_mode_indices

def _tsai_wu_coefficients(
        strengths: tuple[float, float, float, float, float, float],
        tsai_wu_F12: float | None) -> tuple[float, ...]:
    """Tsai-Wu coefficients ``(F1, F2, F3, F11, F22, F33, F44, F55, F66,
    F12, F13, F23)`` for one element's degraded strengths."""
    Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        F1 = 1.0 / Xt_s - 1.0 / Xc_s
        F2 = 1.0 / Yt_s - 1.0 / Yc_s
        F3 = F2
        F11 = 1.0 / (Xt_s * Xc_s)
        F22 = 1.0 / (Yt_s * Yc_s)
        F33 = F22
        F44 = 1.0 / S23_s**2
        F55 = 1.0 / S12_s**2
        F66 = 1.0 / S12_s**2
        # F12, F23 use sqrt of a product. Guard against negative
        # products in case future refactors break the F11/F22/F33 sign.
        F11_F22 = max(F11 * F22, 0.0)
        F22_F33 = max(F22 * F33, 0.0)
        # Issue #145: honour a user-supplied interaction coefficient. It is
        # the normalized F*_12 = F_12 / sqrt(F_11 * F_22) (dimensionless,
        # validated to [-1, 0] by MaterialProperties.__post_init__), so it
        # scales with this element's degraded strengths like the default.
        F12_star = -0.5 if tsai_wu_F12 is None else float(tsai_wu_F12)
        F12 = F12_star * np.sqrt(F11_F22)
        F13 = F12
        F23 = -0.5 * np.sqrt(F22_F33)
    return F1, F2, F3, F11, F22, F33, F44, F55, F66, F12, F13, F23


def evaluate_tsai_wu(s_all: np.ndarray,
                     strengths: tuple[float, float, float, float, float, float],
                     e: int, elem_Vp: float,
                     tsai_wu_F12: float | None = None) -> np.ndarray:
    """Tsai-Wu polynomial evaluated for one element's Gauss points.

    The diagonal coefficients ``F1, F2, F11, F22, F33, F44, F55, F66``
    follow directly from the lamina strength allowables. The off-
    diagonal interaction coefficient defaults to::

        F_12 = -0.5 * sqrt(F_11 * F_22)

    which is **Tsai's empirical recommendation** (Tsai & Wu 1971), not
    a first-principles derivation: it produces a closed failure
    envelope that reduces to the von Mises-like ellipse in the
    isotropic limit, but the "true" value varies with the material
    system and ideally comes from biaxial coupon calibration. The
    default may be overridden per-material by setting
    :attr:`MaterialProperties.tsai_wu_F12` to the normalized coefficient
    ``F*_12`` in ``[-1, 0]``; the solver then uses
    ``F_12 = F*_12 * sqrt(F_11 * F_22)``, so ``-0.5`` reproduces the
    default.
    ``F_13`` continues to mirror ``F_12``, and ``F_23`` retains the
    analogous Tsai recommendation ``-0.5 * sqrt(F_22 * F_33)``.

    Bit-identical to the historical implementation when
    ``material.tsai_wu_F12 is None``. Returns the per-GP failure index
    array (shape ``(n_gp,)``).

    References
    ----------
    Tsai, S. W. & Wu, E. M. (1971). "A General Theory of Strength for
    Anisotropic Materials." *J. Composite Materials* 5(1), 58-80.
    """
    (F1, F2, F3, F11, F22, F33, F44, F55, F66,
     F12, F13, F23) = _tsai_wu_coefficients(strengths, tsai_wu_F12)

    # Vectorize across all Gauss points of this element (#41).
    fi_per_gp = (
        F1 * s_all[:, 0] + F2 * s_all[:, 1] + F3 * s_all[:, 2]
        + F11 * s_all[:, 0]**2 + F22 * s_all[:, 1]**2 + F33 * s_all[:, 2]**2
        + F44 * s_all[:, 3]**2 + F55 * s_all[:, 4]**2 + F66 * s_all[:, 5]**2
        + 2 * F12 * s_all[:, 0] * s_all[:, 1]
        + 2 * F13 * s_all[:, 0] * s_all[:, 2]
        + 2 * F23 * s_all[:, 1] * s_all[:, 2]
    )
    if not np.all(np.isfinite(fi_per_gp)):
        bad_g = int(np.argmax(~np.isfinite(fi_per_gp)))
        raise ValueError(
            f"Tsai-Wu failure index is non-finite at element {e}, "
            f"Gauss point {bad_g} (Vp={elem_Vp:.4f}, "
            f"stress={s_all[bad_g].tolist()}). This usually indicates "
            f"a degenerate stiffness or strength matrix; refine the "
            f"mesh or check input bounds."
        )
    return fi_per_gp

def evaluate_hashin(s_all: np.ndarray,
                    strengths: tuple[float, float, float, float, float, float]
                    ) -> dict[str, np.ndarray]:
    """Hashin failure indices for unidirectional plies, plus delamination.

    The four in-plane modes are the Hashin (1980) 2D criterion with
    separate fiber/matrix tension/compression modes, evaluated per Gauss
    point on ``(σ_11, σ_22, τ_12)``:

        Hashin, Z. (1980). "Failure Criteria for Unidirectional Fiber
        Composites." J. Appl. Mech. 47(2), 329-334.

    Because that form ignores ``σ_33``, ``τ_13`` and ``τ_23``, it is blind
    to the interlaminar stresses that govern e.g. the ILSS short-beam
    test. A fifth ``delamination`` mode adds the Brewer & Lagace quadratic
    delamination-initiation criterion, with the through-thickness tensile
    strength taken as ``Y_t`` (transverse isotropy) and both out-of-plane
    shears against the interlaminar shear strength ``S_23``::

        delamination = (<σ_33>/Y_t)^2 + (τ_13^2 + τ_23^2) / S_23^2

    where ``<σ_33> = max(σ_33, 0)`` (through-thickness compression does not
    open a delamination):

        Brewer, J. C. & Lagace, P. A. (1988). "Quadratic Stress Criterion
        for Initiation of Delamination." J. Compos. Mater. 22(12),
        1141-1155.

    ``max_fi`` is the maximum over the five modes. The ``shear`` slot
    returns the in-plane ``(τ_12 / S_12)^2`` contribution for completeness
    and is not itself a mode.

    Returns a dict of per-GP arrays: ``{'max_fi', 'fiber_t', 'fiber_c',
    'matrix_t', 'matrix_c', 'shear', 'delamination'}``.

    Notes
    -----
    "Hashin" is a family of criteria rather than a single canonical form.
    Commercial codes (ANSYS, Abaqus's user-material variants, LS-DYNA's
    ``MAT_LAMINATED_COMPOSITE_FABRIC``, etc.) ship slightly different
    variants of the matrix-compression mode in particular. The two most
    common deviations from the 1980 paper are:

    1. **Matrix-compression denominator.** The 1980 paper uses
       ``(2·S_23)^2`` together with the ``((Y_c / (2·S_23))^2 - 1) ·
       (σ_22 / Y_c)`` cross term; some codes substitute plain ``S_23``,
       others reformulate the denominator entirely (e.g. ``(2·S_23)^2 -
       (S_23 - Y_c)^2`` style expressions).
    2. **Shear strength in the matrix-compression term.** The 1980 paper
       uses ``S_12`` (in-plane shear) in the ``(τ_12 / S_12)^2``
       contribution to matrix compression; some commercial implementations
       swap this for ``S_23`` (through-thickness shear), which changes
       predictions for transversely-loaded plies.

    The PorosityFE implementation matches the 1980 paper exactly:
    matrix-compression uses ``(2·S_23)^2`` as the normal-stress denominator
    and ``S_12`` in the shear term. Users comparing PorosityFE indices
    against an external Hashin reference (commercial solver, textbook,
    or another open-source code) should verify the variant convention
    on the other side before treating any divergence as a bug.
    """
    Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
    sigma_11 = s_all[:, 0]
    sigma_22 = s_all[:, 1]
    tau_12 = s_all[:, 5]

    # Fiber tension: σ_11 >= 0
    ft = (sigma_11 / Xt_s) ** 2 + (tau_12 / S12_s) ** 2
    ft = np.where(sigma_11 >= 0.0, ft, 0.0)

    # Fiber compression: σ_11 < 0
    fc = (sigma_11 / Xc_s) ** 2
    fc = np.where(sigma_11 < 0.0, fc, 0.0)

    # Matrix tension: σ_22 >= 0
    mt = (sigma_22 / Yt_s) ** 2 + (tau_12 / S12_s) ** 2
    mt = np.where(sigma_22 >= 0.0, mt, 0.0)

    # Matrix compression: σ_22 < 0
    mc_term = ((Yc_s / (2.0 * S23_s)) ** 2 - 1.0) * (sigma_22 / Yc_s)
    mc = (sigma_22 / (2.0 * S23_s)) ** 2 + mc_term + (tau_12 / S12_s) ** 2
    mc = np.where(sigma_22 < 0.0, mc, 0.0)

    shear = (tau_12 / S12_s) ** 2

    # Delamination initiation (Brewer & Lagace 1988)
    sigma_33_t = np.maximum(s_all[:, 2], 0.0)
    delam = (sigma_33_t / Yt_s) ** 2 + (s_all[:, 4] ** 2 + s_all[:, 3] ** 2) / S23_s ** 2

    max_fi = np.maximum.reduce([ft, fc, mt, mc, delam])
    return {
        'max_fi': max_fi,
        'fiber_t': ft,
        'fiber_c': fc,
        'matrix_t': mt,
        'matrix_c': mc,
        'shear': shear,
        'delamination': delam,
    }

def evaluate_max_stress(s_all: np.ndarray,
                        strengths: tuple[float, float, float, float, float, float]
                        ) -> dict[str, np.ndarray]:
    """Maximum-stress failure indices.

    ``FI_i = |σ_i| / X_i_allowable`` per component (signed split for
    normals: tensile vs compressive allowable). The reported ``max_fi``
    is the maximum across all five mode/component buckets. Returns the
    same per-GP dict shape as :func:`evaluate_hashin`; unused entries
    are zeroed (rather than NaN) since each mode is well-defined for
    max-stress. ``delamination`` is always zero: max-stress already checks
    ``σ_33`` in the matrix buckets and ``τ_13`` / ``τ_23`` in ``shear``.
    """
    Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
    sigma_11 = s_all[:, 0]
    sigma_22 = s_all[:, 1]
    sigma_33 = s_all[:, 2]
    tau_23 = s_all[:, 3]
    tau_13 = s_all[:, 4]
    tau_12 = s_all[:, 5]

    ft = np.where(sigma_11 >= 0.0, sigma_11 / Xt_s, 0.0)
    fc = np.where(sigma_11 < 0.0, -sigma_11 / Xc_s, 0.0)
    # Matrix uses worst of σ_22 and σ_33 (transverse normals share the
    # same in-plane transverse strength).
    mt_22 = np.where(sigma_22 >= 0.0, sigma_22 / Yt_s, 0.0)
    mt_33 = np.where(sigma_33 >= 0.0, sigma_33 / Yt_s, 0.0)
    mt = np.maximum(mt_22, mt_33)
    mc_22 = np.where(sigma_22 < 0.0, -sigma_22 / Yc_s, 0.0)
    mc_33 = np.where(sigma_33 < 0.0, -sigma_33 / Yc_s, 0.0)
    mc = np.maximum(mc_22, mc_33)
    # Shear: worst of all three engineering shear components against the
    # appropriate allowable (S_23 for the 23 plane, S_12 for 12 / 13).
    shear = np.maximum.reduce([
        np.abs(tau_12) / S12_s,
        np.abs(tau_13) / S12_s,
        np.abs(tau_23) / S23_s,
    ])

    max_fi = np.maximum.reduce([ft, fc, mt, mc, shear])
    return {
        'max_fi': max_fi,
        'fiber_t': ft,
        'fiber_c': fc,
        'matrix_t': mt,
        'matrix_c': mc,
        'shear': shear,
        'delamination': np.zeros_like(max_fi),
    }


def _positive_root(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Smallest positive ``lam`` with ``A lam**2 + B lam = 1``, else ``inf``.

    Written as ``2 / (B + sqrt(B**2 + 4A))`` (the rationalized root), which
    is stable when ``A`` is tiny and reduces to ``1/B`` for ``A = 0``.
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        denom = B + np.sqrt(B * B + 4.0 * A)
        return np.where(denom > 0.0, 2.0 / denom, np.inf)


def _point_load_factors(s_all: np.ndarray,
                        strengths: tuple[float, float, float, float, float, float],
                        criterion: str,
                        tsai_wu_F12: float | None) -> np.ndarray:
    """Per-Gauss-point load factor at which ``criterion`` reaches 1."""
    s0, s1, s2, s3, s4, s5 = (s_all[:, k] for k in range(6))
    if criterion == 'tsai_wu':
        (F1, F2, F3, F11, F22, F33, F44, F55, F66,
         F12, F13, F23) = _tsai_wu_coefficients(strengths, tsai_wu_F12)
        B = F1 * s0 + F2 * s1 + F3 * s2
        A = (F11 * s0**2 + F22 * s1**2 + F33 * s2**2
             + F44 * s3**2 + F55 * s4**2 + F66 * s5**2
             + 2 * F12 * s0 * s1 + 2 * F13 * s0 * s2 + 2 * F23 * s1 * s2)
        return _positive_root(A, B)
    if criterion == 'hashin':
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        zero = np.zeros_like(s0)
        shear = (s5 / S12_s) ** 2
        # Each mode's active branch depends only on stress signs, which a
        # positive load factor preserves.
        modes = [
            np.where(s0 >= 0.0, _positive_root((s0 / Xt_s) ** 2 + shear, zero), np.inf),
            np.where(s0 < 0.0, _positive_root((s0 / Xc_s) ** 2, zero), np.inf),
            np.where(s1 >= 0.0, _positive_root((s1 / Yt_s) ** 2 + shear, zero), np.inf),
            np.where(s1 < 0.0, _positive_root(
                (s1 / (2.0 * S23_s)) ** 2 + shear,
                ((Yc_s / (2.0 * S23_s)) ** 2 - 1.0) * (s1 / Yc_s)), np.inf),
            _positive_root((np.maximum(s2, 0.0) / Yt_s) ** 2
                           + (s4 ** 2 + s3 ** 2) / S23_s ** 2, zero),
        ]
        return np.minimum.reduce(modes)
    fi = evaluate_max_stress(s_all, strengths)['max_fi']
    with np.errstate(divide='ignore'):
        return np.where(fi > 0.0, 1.0 / fi, np.inf)


def _positive_root_offset(A: np.ndarray, B: np.ndarray,
                          c: np.ndarray | float) -> np.ndarray:
    """Smallest positive ``lam`` with ``A lam**2 + B lam = c`` (``c > 0``), else ``inf``.

    The rationalized root ``2 c / (B + sqrt(B**2 + 4 A c))``, stable for
    tiny ``A``; :func:`_positive_root` is the case ``c = 1``. With
    ``c > 0`` it is the smallest positive root for either sign of ``A``,
    and ``inf`` when there is none (negative discriminant, or ``A <= 0``
    with ``B <= 0``).
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        disc = B * B + 4.0 * A * c
        denom = B + np.sqrt(np.maximum(disc, 0.0))
        return np.where((disc >= 0.0) & (denom > 0.0), 2.0 * c / denom, np.inf)


def _tsai_wu_forms(sa: np.ndarray, sb: np.ndarray, co: tuple
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Linear form ``F . sa`` and bilinear form ``sa^T F sb`` of Tsai-Wu.

    ``sa``, ``sb`` are ``(..., 6)``; the coefficients in ``co`` (the
    :func:`_tsai_wu_coefficients` tuple) are scalars or arrays that
    broadcast against ``sa[..., 0]``.
    """
    F1, F2, F3, F11, F22, F33, F44, F55, F66, F12, F13, F23 = co
    a = [sa[..., k] for k in range(6)]
    b = [sb[..., k] for k in range(6)]
    lin = F1 * a[0] + F2 * a[1] + F3 * a[2]
    bil = (F11 * a[0] * b[0] + F22 * a[1] * b[1] + F33 * a[2] * b[2]
           + F44 * a[3] * b[3] + F55 * a[4] * b[4] + F66 * a[5] * b[5]
           + F12 * (a[0] * b[1] + a[1] * b[0])
           + F13 * (a[0] * b[2] + a[2] * b[0])
           + F23 * (a[1] * b[2] + a[2] * b[1]))
    return lin, bil


def _piece_load_factor(s_th: np.ndarray, s_m: np.ndarray,
                       cond: tuple[int, bool] | None,
                       quad: dict, lin: dict) -> np.ndarray:
    """First ``lam >= 0`` at which one branch of a failure mode reaches 1.

    The branch value is ``sum_k quad[k] s_k**2 + sum_k lin[k] s_k`` at
    ``s = s_th + lam s_m``, convex in ``lam`` (every ``quad[k] >= 0``), and
    the branch is active only while its sign condition ``cond = (k,
    tensile)`` holds (``s_k >= 0`` when ``tensile``, ``s_k < 0``
    otherwise; ``None`` means always), i.e. on an interval
    ``[lo, hi]`` of ``lam``. If the branch is already at 1 at ``lo`` (the
    pre-stress alone, or a mode that switches on with a jump at a sign
    change) the answer is ``lo``; otherwise it is the root of the
    quadratic beyond ``lo`` if that lies in the interval, else ``inf``.
    """
    shape = s_th.shape[:-1]
    lo = np.zeros(shape)
    hi = np.full(shape, np.inf)
    empty = np.zeros(shape, dtype=bool)
    if cond is not None:
        k, tensile = cond
        a, b = s_th[..., k], s_m[..., k]
        with np.errstate(divide='ignore', invalid='ignore'):
            cross = -a / b                       # lam where s_k changes sign
        if tensile:   # a + b lam >= 0
            lo = np.where(b > 0.0, np.maximum(cross, 0.0), lo)
            hi = np.where(b < 0.0, cross, hi)
            empty = ((b < 0.0) & (cross < 0.0)) | ((b == 0.0) & (a < 0.0))
        else:         # a + b lam < 0
            lo = np.where(b < 0.0, np.maximum(cross, 0.0), lo)
            hi = np.where(b > 0.0, cross, hi)
            empty = ((b > 0.0) & (cross <= 0.0)) | ((b == 0.0) & (a >= 0.0))
        lo = np.where(empty, 0.0, lo)
    s0 = s_th + lo[..., None] * s_m
    q0 = np.zeros(shape)
    A = np.zeros(shape)
    Bp = np.zeros(shape)
    for k, w in quad.items():
        q0 = q0 + w * s0[..., k] ** 2
        A = A + w * s_m[..., k] ** 2
        Bp = Bp + 2.0 * w * s0[..., k] * s_m[..., k]
    for k, w in lin.items():
        q0 = q0 + w * s0[..., k]
        Bp = Bp + w * s_m[..., k]
    lam = lo + _positive_root_offset(A, Bp, 1.0 - q0)
    lam = np.where(lam <= hi, lam, np.inf)
    lam = np.where(q0 >= 1.0, lo, lam)
    return np.where(empty, np.inf, lam)


def _point_load_factors_prestressed(
        s_th: np.ndarray, s_m: np.ndarray, strengths: tuple,
        criterion: str, tsai_wu_F12: float | None,
        tsai_wu_coefficients: tuple | None = None) -> np.ndarray:
    """Load factor on ``s_m`` with a fixed pre-stress ``s_th``.

    Smallest ``lam >= 0`` at which ``criterion`` evaluated on
    ``s_th + lam * s_m`` reaches 1: ``0`` where ``s_th`` alone has reached
    it, ``inf`` where it is never reached. ``s_th`` and ``s_m`` are
    ``(..., 6)`` ply-local stresses; ``strengths`` is the
    :func:`degraded_strengths` tuple (scalars, or arrays broadcasting
    against ``s_th[..., 0]``). ``tsai_wu_coefficients`` may pass the
    matching :func:`_tsai_wu_coefficients` tuple precomputed.

    - Tsai-Wu: ``FI = F.s + s^T F s`` gives
      ``A lam^2 + B lam = 1 - FI_th`` with ``A = s_m^T F s_m`` and
      ``B = F.s_m + 2 s_th^T F s_m``.
    - Max-stress: each component ``s_th,i + lam s_m,i`` meets the
      allowable on the side it moves towards (linear, with offset).
    - Hashin: a pre-stress can change a stress sign as ``lam`` grows, so
      each mode branch (fiber and matrix, tension and compression, and the
      two branches of ``<sigma_33>`` in delamination) is solved on the
      interval of ``lam`` where its sign condition holds, including onset
      at a sign switch with the branch already at 1.
    """
    if criterion == 'tsai_wu':
        co = tsai_wu_coefficients if tsai_wu_coefficients is not None \
            else _tsai_wu_coefficients(strengths, tsai_wu_F12)
        lin_th, quad_th = _tsai_wu_forms(s_th, s_th, co)
        lin_m, A = _tsai_wu_forms(s_m, s_m, co)
        _, cross = _tsai_wu_forms(s_th, s_m, co)
        c = 1.0 - (lin_th + quad_th)
        lam = _positive_root_offset(A, lin_m + 2.0 * cross, c)
        return np.where(c <= 0.0, 0.0, lam)

    Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
    if criterion == 'max_stress':
        # (component, tensile allowable, compressive allowable)
        bounds = ((0, Xt_s, Xc_s), (1, Yt_s, Yc_s), (2, Yt_s, Yc_s),
                  (3, S23_s, S23_s), (4, S12_s, S12_s), (5, S12_s, S12_s))
        lam = np.full(s_th.shape[:-1], np.inf)
        for k, upper, lower in bounds:
            a, b = s_th[..., k], s_m[..., k]
            with np.errstate(divide='ignore', invalid='ignore'):
                lam_k = np.where(b > 0.0, (upper - a) / b,
                                 np.where(b < 0.0, (-lower - a) / b, np.inf))
            lam_k = np.where((a >= upper) | (a <= -lower), 0.0, lam_k)
            lam = np.minimum(lam, lam_k)
        return lam

    if criterion != 'hashin':
        raise ValueError(
            f"Unknown failure criterion {criterion!r}. "
            f"Use one of {list(SUPPORTED_FAILURE_CRITERIA)}.")
    shear12 = 1.0 / S12_s ** 2
    inter = 1.0 / S23_s ** 2
    pieces: list[tuple[tuple[int, bool], dict, dict]] = [
        ((0, True), {0: 1.0 / Xt_s ** 2, 5: shear12}, {}),            # fiber_t
        ((0, False), {0: 1.0 / Xc_s ** 2}, {}),                       # fiber_c
        ((1, True), {1: 1.0 / Yt_s ** 2, 5: shear12}, {}),            # matrix_t
        ((1, False), {1: 1.0 / (2.0 * S23_s) ** 2, 5: shear12},       # matrix_c
         {1: ((Yc_s / (2.0 * S23_s)) ** 2 - 1.0) / Yc_s}),
        ((2, True), {2: 1.0 / Yt_s ** 2, 3: inter, 4: inter}, {}),    # delam, s33 >= 0
        ((2, False), {3: inter, 4: inter}, {}),                       # delam, s33 < 0
    ]
    return np.minimum.reduce([
        _piece_load_factor(s_th, s_m, cond, quad, lin)
        for cond, quad, lin in pieces])


def _element_load_factors(stress_local: np.ndarray, porosity: np.ndarray,
                          elements: np.ndarray, material: MaterialProperties,
                          void_shape_radii: tuple,
                          criterion: str = 'tsai_wu',
                          void_elements=None, *,
                          prestress_local: np.ndarray | None = None
                          ) -> np.ndarray:
    """Per-element first-failure load factor ``(n_elem,)``, ``inf`` where skipped.

    See :func:`first_ply_failure_load_factor`, which returns the minimum.
    """
    if criterion not in SUPPORTED_FAILURE_CRITERIA:
        raise ValueError(
            f"Unknown failure criterion {criterion!r}. "
            f"Use one of {list(SUPPORTED_FAILURE_CRITERIA)}."
        )
    elem_Vp_all, skip = _element_porosity_and_skip(porosity, elements, void_elements)
    lam_e = np.full(stress_local.shape[0], np.inf)
    active = np.flatnonzero(~skip)
    strengths_by_vp: dict[float, tuple[float, float, float, float, float, float]] = {}

    def strengths_of(vp: float) -> tuple[float, float, float, float, float, float]:
        st = strengths_by_vp.get(vp)
        if st is None:
            st = degraded_strengths(material, void_shape_radii, vp)
            strengths_by_vp[vp] = st
        return st

    if prestress_local is None:
        for e in active:
            lam = _point_load_factors(stress_local[e],
                                      strengths_of(float(elem_Vp_all[e])),
                                      criterion, material.tsai_wu_F12)
            lam_e[e] = float(lam.min())
        return lam_e

    prestress_local = np.asarray(prestress_local, dtype=float)
    if prestress_local.shape != stress_local.shape:
        raise ValueError(
            f"prestress_local has shape {prestress_local.shape}, expected "
            f"{stress_local.shape} (the shape of stress_local).")
    if active.size == 0:
        return lam_e
    # Vectorized over the active elements: per-element strengths (and
    # Tsai-Wu coefficients) gathered from the distinct porosity levels.
    vps, inverse = np.unique(elem_Vp_all[active], return_inverse=True)
    inverse = inverse.reshape(-1)
    table = np.array([strengths_of(float(v)) for v in vps])          # (n_vp, 6)
    strengths = tuple(table[inverse, j][:, None] for j in range(6))
    co = None
    if criterion == 'tsai_wu':
        co_table = np.array([
            _tsai_wu_coefficients(tuple(row), material.tsai_wu_F12)
            for row in table])                                       # (n_vp, 12)
        co = tuple(co_table[inverse, j][:, None] for j in range(co_table.shape[1]))
    lam = _point_load_factors_prestressed(
        prestress_local[active], stress_local[active], strengths, criterion,
        material.tsai_wu_F12, tsai_wu_coefficients=co)
    lam_e[active] = lam.min(axis=1)
    return lam_e


def first_ply_failure_load_factor(stress_local: np.ndarray, porosity: np.ndarray,
                                  elements: np.ndarray, material: MaterialProperties,
                                  void_shape_radii: tuple,
                                  criterion: str = 'tsai_wu',
                                  void_elements=None, *,
                                  prestress_local: np.ndarray | None = None
                                  ) -> float:
    """Load multiplier at which ``criterion`` first reaches 1 anywhere.

    The analysis is linear, so every stress scales with the applied load
    factor ``lam``. Per Gauss point the failure index is then a polynomial
    in ``lam`` (degree 1 for max-stress, degree 2 for Tsai-Wu and the
    Hashin modes), and the returned value is the smallest positive ``lam``
    at which any point of any non-void element reaches an index of 1 (the
    same elements :func:`evaluate_failure` checks). The margin of safety
    is ``lam - 1``. Returns ``inf`` if no point is stressed.

    Parameters
    ----------
    stress_local : np.ndarray
        ``(n_elem, n_gp, 6)`` ply-local stress that ``lam`` scales.
    porosity, elements, material, void_shape_radii, criterion, void_elements
        As for :func:`evaluate_failure`.
    prestress_local : np.ndarray, optional
        Keyword-only ``(n_elem, n_gp, 6)`` ply-local stress held fixed
        while ``lam`` scales ``stress_local``, e.g. a thermal residual
        stress: the criterion is evaluated on
        ``prestress_local + lam * stress_local``, and the result is ``0``
        where the pre-stress alone has reached it. Tsai-Wu gains the
        pre-stress terms in closed form, max-stress becomes linear with an
        offset, and Hashin is solved branch by branch, because a
        pre-stress lets a stress change sign (and a mode switch on) as
        ``lam`` grows. ``None`` (default) is the unchanged path.
    """
    lam_e = _element_load_factors(
        stress_local, porosity, elements, material, void_shape_radii,
        criterion, void_elements, prestress_local=prestress_local)
    return float(lam_e.min()) if lam_e.size else float('inf')
