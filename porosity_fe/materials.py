"""Material properties dataclass and built-in presets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ============================================================
# SECTION 1: MATERIAL PROPERTIES AND CONSTANTS
# ============================================================

@dataclass
class MaterialProperties:
    """Composite material properties with constituent data for micromechanics.

    Bundles the lamina-level orthotropic stiffness, the longitudinal /
    transverse / shear strength allowables (used by the Tsai-Wu failure
    criterion in :class:`Hex8Element`), the ply / laminate geometry and the
    constituent (matrix + fiber) data needed by the Mori-Tanaka
    homogenization for porosity degradation. All stress / modulus inputs are
    in **MPa** and all lengths are in **mm**; Poisson ratios and the fiber
    volume fraction are dimensionless fractions.

    The dataclass is validated by :meth:`__post_init__`: stiffness moduli
    and strengths must be positive finite floats; Poisson ratios must lie
    in ``(-1, 0.5)``; ``n_plies`` must be a positive integer; and
    ``fiber_volume_fraction`` must be a fraction in ``(0, 1)`` (a percent
    such as ``60`` is rejected with a hint). The optional thermal
    expansion coefficients must be finite and given in 1/K (a ppm/K value
    such as ``26`` is rejected with a hint).

    Parameters
    ----------
    E11, E22, E33 : float
        Lamina orthotropic Young's moduli along the fiber (1), transverse
        (2) and through-thickness (3) directions, in MPa.
    G12, G13, G23 : float
        Lamina shear moduli (in-plane, interlaminar and transverse shear),
        in MPa.
    nu12, nu13, nu23 : float
        Major (12), through-thickness (13) and transverse (23) Poisson's
        ratios. Each must lie in ``(-1, 0.5)``.
    sigma_1c, sigma_1t : float
        Longitudinal compression and tension allowables, in MPa.
    sigma_2t, sigma_2c : float
        Transverse tension and compression allowables, in MPa.
        ``sigma_2c`` enters the FE failure criteria (Tsai-Wu ``Y_c``,
        Hashin matrix compression) but has no empirical loading mode: no
        dataset in ``validation/datasets`` measures transverse compression
        against porosity, so there is nothing to calibrate a knockdown
        coefficient on.
    tau_12 : float
        In-plane shear allowable, in MPa.
    tau_ilss : float
        Interlaminar (short-beam) shear allowable, in MPa.
    t_ply : float
        Ply thickness, in mm.
    n_plies : int
        Number of plies in the laminate (positive integer).
    matrix_modulus : float
        Matrix (resin) Young's modulus ``E_m``, in MPa, used by the
        Mori-Tanaka homogenization in :class:`Hex8Element`.
    matrix_poisson : float
        Matrix Poisson's ratio ``nu_m`` (dimensionless, in
        ``(-1, 0.5)``).
    fiber_modulus : float
        Fiber longitudinal Young's modulus ``E_f``, in MPa.
    fiber_volume_fraction : float
        Pristine fiber volume fraction ``V_f``, as a fraction in
        ``(0, 1)`` (e.g. ``0.60`` for a 60 % fiber laminate).
    tsai_wu_F12 : float, optional
        Normalized Tsai-Wu in-plane interaction coefficient
        ``F*_12 = F_12 / sqrt(F_11 * F_22)`` (dimensionless). The FE
        failure check uses ``F_12 = F*_12 * sqrt(F_11 * F_22)`` with each
        element's degraded strengths. ``None`` (default) is Tsai's
        recommendation ``F*_12 = -0.5`` (Tsai & Wu 1971). When provided,
        must lie in ``[-1, 0]`` so the quadratic failure envelope stays
        closed; for critical applications, calibrate against biaxial
        coupon data.
    fiber_poisson : float, optional
        Fiber Poisson's ratio ``nu_f`` used by the micromechanics (default
        0.2, typical of carbon fiber), in ``(-1, 0.5)``.
    fiber_shear_modulus : float, optional
        Fiber axial shear modulus ``G_f`` in MPa for the Halpin-Tsai shear
        ratios. ``None`` (default) uses the isotropic estimate
        ``E_f / (2 (1 + nu_f))``; carbon fibers are anisotropic, so set the
        measured value when it is known.
    alpha_1, alpha_2 : float, optional
        Lamina coefficients of thermal expansion along the fiber (1) and
        transverse (2) directions, in **1/K** (pass ``26e-6``, not ``26``;
        a magnitude of ``1e-3`` /K or more is rejected as a probable
        ppm/K value). ``None`` (default) means no thermal data. Give both
        or neither. ``alpha_1`` may be negative (carbon fibers contract
        axially when heated); ``alpha_2`` must be positive. Read by the FE
        thermal / cure-residual-stress load case
        (``FESolver.solve(loading='thermal', delta_T=...)`` or ``delta_T=``
        on a mechanical mode); without ``delta_T`` they change no result.
        Held at the pristine value: porosity barely changes the ply CTE (an
        empty void does not change the free thermal expansion of the matrix
        around it).
    alpha_3 : float, optional
        Through-thickness CTE in 1/K, positive. ``None`` (default) uses
        ``alpha_2`` (transverse isotropy of a UD tape); see
        :attr:`alpha_3_eff`. Requires ``alpha_1`` and ``alpha_2``.
    T_stress_free : float, optional
        Stress-free temperature in deg C (typically near the cure
        temperature), recorded for the user's ``delta_T = T_service -
        T_stress_free``. The solver does not read it: ``delta_T`` is always
        passed explicitly. ``None`` (default) means unspecified. Must be
        finite and above absolute zero.

    Attributes
    ----------
    E11, E22, E33 : float
        Orthotropic Young's moduli (MPa).
    G12, G13, G23 : float
        Shear moduli (MPa).
    nu12, nu13, nu23 : float
        Poisson's ratios (dimensionless).
    sigma_1c, sigma_1t, sigma_2t, sigma_2c : float
        Normal-direction strengths (MPa).
    tau_12, tau_ilss : float
        Shear strengths (MPa).
    t_ply : float
        Ply thickness (mm).
    n_plies : int
        Number of plies.
    matrix_modulus, matrix_poisson : float
        Constituent matrix elasticity (MPa, dimensionless).
    fiber_modulus, fiber_volume_fraction : float
        Constituent fiber modulus (MPa) and pristine fiber volume
        fraction (dimensionless, in ``(0, 1)``).
    alpha_1, alpha_2, alpha_3 : float or None
        Lamina CTEs (1/K); ``None`` when not supplied.
    T_stress_free : float or None
        Stress-free temperature (deg C); ``None`` when not supplied.
    total_thickness : float
        Read-only property: ``t_ply * n_plies`` (mm). Used as ``L_z`` by
        :class:`CompositeMesh`.

    Examples
    --------
    Build a T800/epoxy ply (the same values are pre-baked in
    :data:`MATERIALS`):

    >>> mat = MaterialProperties(
    ...     E11=161000.0, E22=11380.0, E33=11380.0,
    ...     G12=5170.0, G13=5170.0, G23=3980.0,
    ...     nu12=0.32, nu13=0.32, nu23=0.40,
    ...     sigma_1c=1500.0, sigma_1t=2800.0,
    ...     sigma_2t=80.0, sigma_2c=250.0,
    ...     tau_12=100.0, tau_ilss=90.0,
    ...     t_ply=0.183, n_plies=24,
    ...     matrix_modulus=3500.0, matrix_poisson=0.35,
    ...     fiber_modulus=294000.0, fiber_volume_fraction=0.60,
    ... )
    >>> round(mat.total_thickness, 4)
    4.392
    """
    # Lamina-level orthotropic properties
    E11: float          # Longitudinal modulus (MPa)
    E22: float          # Transverse modulus (MPa)
    E33: float          # Through-thickness modulus (MPa)
    G12: float          # In-plane shear modulus (MPa)
    G13: float          # Interlaminar shear modulus (MPa)
    G23: float          # Transverse shear modulus (MPa)
    nu12: float         # Major Poisson's ratio
    nu13: float         # Through-thickness Poisson's ratio
    nu23: float         # Transverse Poisson's ratio

    # Longitudinal strengths
    sigma_1c: float     # Longitudinal compression strength (MPa)
    sigma_1t: float     # Longitudinal tension strength (MPa)
    # Transverse strengths
    sigma_2t: float     # Transverse tension strength (MPa)
    sigma_2c: float     # Transverse compression strength (MPa)
    # Shear strengths
    tau_12: float       # In-plane shear strength (MPa)
    tau_ilss: float     # Interlaminar shear strength (MPa)

    # Geometric
    t_ply: float        # Ply thickness (mm)
    n_plies: int        # Number of plies

    # Constituent properties (for micromechanics)
    matrix_modulus: float         # E_m (MPa)
    matrix_poisson: float         # nu_m
    fiber_modulus: float          # E_f (MPa)
    fiber_volume_fraction: float  # V_f (pristine)

    # ----------------------------------------------------------------
    # Hygrothermal conditioning (issue #59).
    #
    # All five fields default to "no environmental effect": when any of
    # ``T_service`` / ``M_service`` / ``T_g_dry`` is ``None``,
    # :meth:`environment_knockdown` returns 1.0 (back-compat no-op).
    # ----------------------------------------------------------------
    T_service: float | None = None   # Service temperature (deg C)
    M_service: float | None = None   # Service moisture content (wt %)
    T_ref: float = 23.0                 # Reference / RT (deg C); RTD baseline
    M_ref: float = 0.0                  # Reference moisture (wt %)
    T_g_dry: float | None = None     # Dry glass-transition temperature (deg C)

    # Normalized Tsai-Wu interaction coefficient F*_12 = F_12 /
    # sqrt(F_11 * F_22). None (default) -> Tsai's recommendation -0.5.
    # For critical applications, calibrate against biaxial coupon data.
    tsai_wu_F12: float | None = None

    # Fiber constituent elasticity for the Halpin-Tsai degradation ratios.
    # The defaults reproduce the earlier hard-coded isotropic fiber
    # (nu_f = 0.2, G_f = E_f / (2 (1 + nu_f))); carbon fibers are strongly
    # anisotropic, so set the measured axial shear modulus when known.
    fiber_poisson: float = 0.2
    fiber_shear_modulus: float | None = None

    # Lamina coefficients of thermal expansion (1/K, not ppm/K) and the
    # stress-free temperature (deg C) for the thermal / cure residual-stress
    # load case (IMPROVEMENT_PLAN 3.5). Optional; only FESolver.solve with
    # delta_T reads the CTEs. alpha_3 = None means alpha_3 = alpha_2.
    alpha_1: float | None = None        # Fiber-direction CTE (1/K)
    alpha_2: float | None = None        # Transverse CTE (1/K)
    alpha_3: float | None = None        # Through-thickness CTE (1/K)
    T_stress_free: float | None = None  # Stress-free temperature (deg C)

    # A CTE magnitude at or above this (1/K) is taken to be ppm/K passed by
    # mistake: polymer-matrix plies sit roughly between -5e-6 and 6e-5 /K.
    _CTE_MAX = 1e-3
    _ABSOLUTE_ZERO_C = -273.15

    @property
    def fiber_shear_modulus_eff(self) -> float:
        """``fiber_shear_modulus``, or the isotropic ``E_f / (2 (1 + nu_f))``."""
        if self.fiber_shear_modulus is not None:
            return float(self.fiber_shear_modulus)
        return self.fiber_modulus / (2.0 * (1.0 + self.fiber_poisson))

    @property
    def alpha_3_eff(self) -> float | None:
        """``alpha_3``, or ``alpha_2`` when ``alpha_3`` is ``None`` (1/K)."""
        if self.alpha_3 is not None:
            return float(self.alpha_3)
        return None if self.alpha_2 is None else float(self.alpha_2)

    @property
    def has_cte(self) -> bool:
        """``True`` when the lamina CTEs (``alpha_1``, ``alpha_2``) are set."""
        return self.alpha_1 is not None and self.alpha_2 is not None

    def cte_vector(self) -> np.ndarray:
        """Lamina CTE vector in the material frame, in 1/K.

        Returns
        -------
        numpy.ndarray
            Shape ``(6,)``: ``[alpha_1, alpha_2, alpha_3, 0, 0, 0]`` in the
            Voigt order ``[11, 22, 33, 23, 13, 12]`` of
            :meth:`get_stiffness_matrix`. The shear entries are zero for an
            orthotropic ply in its material axes. ``alpha_3`` falls back to
            ``alpha_2`` (:attr:`alpha_3_eff`).

        Raises
        ------
        ValueError
            If ``alpha_1`` / ``alpha_2`` are not set (:attr:`has_cte` is
            ``False``).
        """
        a1, a2 = self.alpha_1, self.alpha_2
        if a1 is None or a2 is None:
            raise ValueError(
                "MaterialProperties has no thermal expansion coefficients: "
                "set alpha_1 and alpha_2 (in 1/K, e.g. alpha_2=26e-6)."
            )
        a3 = a2 if self.alpha_3 is None else self.alpha_3
        return np.array([a1, a2, a3, 0.0, 0.0, 0.0], dtype=float)

    def __post_init__(self):
        # Stiffness moduli must be positive finite (non-zero for 1/E in compliance).
        for name in ('E11', 'E22', 'E33', 'G12', 'G13', 'G23',
                     'matrix_modulus', 'fiber_modulus'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"MaterialProperties.{name} must be a positive finite number "
                    f"(MPa), got {value!r}."
                )
        # Poisson ratios must be in (-1, 0.5) for an isotropic-stable matrix
        # (the matrix stiffness uses (1 - 2*nu_m) in the denominator) and for
        # well-posed orthotropic compliance entries.
        for name in ('nu12', 'nu13', 'nu23', 'matrix_poisson', 'fiber_poisson'):
            value = getattr(self, name)
            if not np.isfinite(value) or not (-1.0 < value < 0.5):
                raise ValueError(
                    f"MaterialProperties.{name} must be a finite Poisson's ratio "
                    f"in (-1, 0.5), got {value!r}."
                )
        if self.fiber_shear_modulus is not None and (
                not np.isfinite(self.fiber_shear_modulus)
                or self.fiber_shear_modulus <= 0):
            raise ValueError(
                f"MaterialProperties.fiber_shear_modulus must be a positive "
                f"finite number (MPa) or None, got {self.fiber_shear_modulus!r}."
            )
        # Strengths must be positive finite (used as 1/X in Tsai-Wu).
        for name in ('sigma_1c', 'sigma_1t', 'sigma_2t', 'sigma_2c',
                     'tau_12', 'tau_ilss'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"MaterialProperties.{name} must be a positive finite "
                    f"strength (MPa), got {value!r}."
                )
        # Geometry
        if not np.isfinite(self.t_ply) or self.t_ply <= 0:
            raise ValueError(
                f"MaterialProperties.t_ply must be a positive ply thickness (mm), "
                f"got {self.t_ply!r}."
            )
        if not isinstance(self.n_plies, (int, np.integer)) or self.n_plies <= 0:
            raise ValueError(
                f"MaterialProperties.n_plies must be a positive integer, "
                f"got {self.n_plies!r}."
            )
        # Fiber volume fraction
        if (not np.isfinite(self.fiber_volume_fraction)
                or not (0.0 < self.fiber_volume_fraction < 1.0)):
            raise ValueError(
                f"MaterialProperties.fiber_volume_fraction must be a fraction in "
                f"(0, 1), got {self.fiber_volume_fraction!r}. "
                f"(Pass a fraction such as 0.60, not a percent.)"
            )

        # Hygrothermal conditioning (issue #59). Optional scalars; reject
        # only explicit non-finite or nonsensical values so the default
        # ``None`` no-op path is unaffected.
        for name in ('T_service', 'M_service', 'T_g_dry'):
            value = getattr(self, name)
            if value is not None:
                if not np.isfinite(float(value)):
                    raise ValueError(
                        f"MaterialProperties.{name} must be a finite number or "
                        f"None, got {value!r}."
                    )
        for name in ('T_ref', 'M_ref'):
            value = getattr(self, name)
            if not np.isfinite(float(value)):
                raise ValueError(
                    f"MaterialProperties.{name} must be a finite number, "
                    f"got {value!r}."
                )
        if self.M_service is not None and float(self.M_service) < 0.0:
            raise ValueError(
                f"MaterialProperties.M_service (moisture wt%) must be >= 0, "
                f"got {self.M_service!r}."
            )
        if float(self.M_ref) < 0.0:
            raise ValueError(
                f"MaterialProperties.M_ref (moisture wt%) must be >= 0, "
                f"got {self.M_ref!r}."
            )

        # Normalized Tsai-Wu interaction coefficient F*_12 (issue #145).
        # ``None`` defers to Tsai's recommendation F*_12 = -0.5. When the
        # user supplies a value, require a finite number in [-1, 0] so the quadratic
        # failure envelope stays closed (physically meaningful). The exact
        # value within that range is left to user judgement / biaxial
        # calibration data.
        if self.tsai_wu_F12 is not None:
            if (not isinstance(self.tsai_wu_F12, (int, float, np.integer,
                                                  np.floating))
                    or isinstance(self.tsai_wu_F12, bool)
                    or not np.isfinite(float(self.tsai_wu_F12))
                    or not (-1.0 <= float(self.tsai_wu_F12) <= 0.0)):
                raise ValueError(
                    f"MaterialProperties.tsai_wu_F12 must be None or a "
                    f"finite number in [-1, 0] (Tsai-Wu interaction "
                    f"coefficient; values outside this range open the "
                    f"failure envelope), got {self.tsai_wu_F12!r}."
                )

        self._validate_thermal_fields()

    @staticmethod
    def _is_real_number(value) -> bool:
        return (isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                and bool(np.isfinite(float(value))))

    def _validate_thermal_fields(self) -> None:
        """Check the optional CTEs (1/K) and stress-free temperature (deg C).

        Mirrors the percent-vs-fraction hint for ``Vp``: a CTE written in
        ppm/K (``26`` instead of ``26e-6``) is rejected with a hint.
        """
        for name in ('alpha_1', 'alpha_2', 'alpha_3'):
            value = getattr(self, name)
            if value is None:
                continue
            if not self._is_real_number(value):
                raise ValueError(
                    f"MaterialProperties.{name} must be None or a finite "
                    f"thermal expansion coefficient in 1/K, got {value!r}."
                )
            if abs(float(value)) >= self._CTE_MAX:
                raise ValueError(
                    f"MaterialProperties.{name}={value!r} is too large for a "
                    f"thermal expansion coefficient in 1/K (|alpha| must be "
                    f"below {self._CTE_MAX:g} /K). It looks like a ppm/K "
                    f"value: pass the coefficient in 1/K, e.g. 26e-6, not 26."
                )
        # The transverse and through-thickness CTEs of a polymer-matrix ply
        # are matrix-dominated and positive. alpha_1 may be negative (carbon
        # fibers contract axially on heating), so it gets no sign rule.
        for name in ('alpha_2', 'alpha_3'):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError(
                    f"MaterialProperties.{name} must be positive (1/K); "
                    f"got {value!r}. Only alpha_1 (fiber direction) may be "
                    f"negative."
                )
        if (self.alpha_1 is None) != (self.alpha_2 is None):
            raise ValueError(
                "MaterialProperties.alpha_1 and alpha_2 must be given "
                f"together or both left None, got alpha_1={self.alpha_1!r}, "
                f"alpha_2={self.alpha_2!r}."
            )
        if self.alpha_3 is not None and self.alpha_2 is None:
            raise ValueError(
                "MaterialProperties.alpha_3 needs alpha_1 and alpha_2 "
                f"(1/K) as well, got alpha_3={self.alpha_3!r} alone."
            )
        if self.T_stress_free is not None and (
                not self._is_real_number(self.T_stress_free)
                or float(self.T_stress_free) <= self._ABSOLUTE_ZERO_C):
            raise ValueError(
                f"MaterialProperties.T_stress_free must be None or a finite "
                f"temperature in deg C above absolute zero, got "
                f"{self.T_stress_free!r}."
            )

    @property
    def total_thickness(self) -> float:
        return self.t_ply * self.n_plies

    def get_compliance_matrix(self) -> np.ndarray:
        """6x6 compliance matrix [S] for orthotropic material.

        Notes
        -----
        Voigt order: ``[11, 22, 33, 23, 13, 12]`` (normals first, then shears in
        the 23 / 13 / 12 order). The shear rows/columns assume **engineering**
        strain (``gamma_ij = 2 * eps_ij``), i.e. ``S[5, 5] = 1 / G12`` maps a
        single ``tau_12`` directly to ``gamma_12 = tau_12 / G12`` without a
        factor of two. Stress is in MPa; strain is dimensionless.
        """
        S = np.zeros((6, 6))
        S[0, 0] = 1.0 / self.E11
        S[1, 1] = 1.0 / self.E22
        S[2, 2] = 1.0 / self.E33
        S[0, 1] = S[1, 0] = -self.nu12 / self.E11
        S[0, 2] = S[2, 0] = -self.nu13 / self.E11
        S[1, 2] = S[2, 1] = -self.nu23 / self.E22
        S[3, 3] = 1.0 / self.G23
        S[4, 4] = 1.0 / self.G13
        S[5, 5] = 1.0 / self.G12
        return S

    def get_stiffness_matrix(self) -> np.ndarray:
        """6x6 stiffness matrix [C] = [S]^-1.

        Notes
        -----
        Voigt order: ``[11, 22, 33, 23, 13, 12]`` (normals first, then shears in
        the 23 / 13 / 12 order), matching :meth:`get_compliance_matrix`. Shear
        components are **engineering** strain (``gamma_ij = 2 * eps_ij``), so
        ``C[5, 5] = G12`` maps ``gamma_12`` directly to ``tau_12 = G12 *
        gamma_12``. Stress is in MPa; strain is dimensionless.
        """
        return np.linalg.inv(self.get_compliance_matrix())

    def get_isotropic_matrix_stiffness(self) -> np.ndarray:
        """6x6 isotropic stiffness tensor C_m from matrix_modulus and matrix_poisson."""
        E_m = self.matrix_modulus
        nu_m = self.matrix_poisson
        lam = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
        mu = E_m / (2 * (1 + nu_m))
        C_m = np.zeros((6, 6))
        C_m[0, 0] = C_m[1, 1] = C_m[2, 2] = lam + 2 * mu
        C_m[0, 1] = C_m[0, 2] = C_m[1, 0] = C_m[1, 2] = C_m[2, 0] = C_m[2, 1] = lam
        C_m[3, 3] = C_m[4, 4] = C_m[5, 5] = mu
        return C_m

    # Modes whose strength is matrix-/interface-dominated and therefore
    # sensitive to hygrothermal conditioning. Mirrors the analogous frozenset
    # on :class:`EmpiricalSolver` but lives here so callers that only have a
    # ``MaterialProperties`` (no solver) can still query the knockdown.
    _HYGROTHERMAL_MATRIX_DOMINATED_MODES = frozenset({
        'transverse_tension', 'ilss', 'shear', 'compression',
    })

    # Springer / Chamis empirical slope for the dry -> wet shift of the
    # glass-transition temperature: T_g_wet ~= T_g_dry - 25 * M, with M in
    # wt% moisture content. See Springer (1981, "Environmental Effects on
    # Composite Materials") and Chamis (NASA-TM-83320, 1983).
    _SPRINGER_MOISTURE_TG_SLOPE = 25.0  # deg C per wt% moisture

    def environment_knockdown(self, mode: str,
                              T: float | None = None,
                              M: float | None = None) -> float:
        """Hygrothermal (T / M) knockdown factor for the requested mode.

        Implements the standard Chamis / Springer matrix-property ratio::

            F_env = sqrt((T_g_wet - T) / (T_g_dry - T_ref))

        where ``T_g_wet ~= T_g_dry - 25 * M`` (the Springer rule of thumb
        for epoxy matrices, with moisture ``M`` in wt%). The square-root
        form was proposed by Chamis (NASA-TM-83320, 1983) for matrix
        modulus / strength retention as the service temperature approaches
        the wet glass transition.

        Fiber-dominated modes (``'tension'``) are largely insensitive to
        hygrothermal conditioning at engineering relevant temperatures and
        return ``1.0`` unconditionally. Matrix- and matrix/interface-
        dominated modes (``'transverse_tension'``, ``'ilss'``, ``'shear'``)
        get the full Chamis/Springer ratio. ``'compression'`` is
        treated as matrix-dominated here because fiber microbuckling is
        gated by matrix shear stiffness — a defensible aerospace-screening
        choice, but conservative compared to a true fiber-failure mode.

        Parameters
        ----------
        mode : str
            Loading mode name (see
            :attr:`EmpiricalSolver.PRISTINE_STRENGTH_KEY`).
        T : float, optional
            Service temperature in degrees Celsius. Falls back to
            :attr:`T_service` when ``None``.
        M : float, optional
            Service moisture content in wt%. Falls back to
            :attr:`M_service` when ``None``.

        Returns
        -------
        float
            Multiplicative knockdown in ``(0, 1]``. Returns ``1.0`` (no
            effect) whenever any of ``T`` / ``M`` / :attr:`T_g_dry` is
            unspecified — the back-compat no-op path.

        Notes
        -----
        This is a screening-level model. Production design allowables
        should still come from a fully populated test matrix per the
        applicable spec (e.g. CMH-17 Vol. 2 hygrothermal conditioning).
        The factor is clamped to ``[0.01, 1.0]`` to keep downstream
        knockdown composition well-behaved when the service temperature
        is set extremely close to (or above) the wet ``T_g``.
        """
        # Resolve T / M from arguments or attribute defaults. ``None`` from
        # both sides -> graceful no-op.
        T_eff = T if T is not None else self.T_service
        M_eff = M if M is not None else self.M_service
        if T_eff is None or M_eff is None or self.T_g_dry is None:
            return 1.0

        # Fiber-dominated modes are insensitive to T / M at engineering
        # relevant temperatures.
        if mode not in self._HYGROTHERMAL_MATRIX_DOMINATED_MODES:
            return 1.0

        T_eff = float(T_eff)
        M_eff = float(M_eff)
        T_g_dry = float(self.T_g_dry)
        T_ref = float(self.T_ref)

        T_g_wet = T_g_dry - self._SPRINGER_MOISTURE_TG_SLOPE * M_eff
        denom = T_g_dry - T_ref
        if denom <= 0.0:
            # Pathological calibration (T_ref above T_g_dry); the matrix is
            # already above its dry transition at the reference. Refuse to
            # scale rather than divide by zero.
            return 1.0

        numer = T_g_wet - T_eff
        if numer <= 0.0:
            # Service temperature has reached (or exceeded) the wet T_g —
            # matrix has effectively lost its load-carrying capability.
            # Clamp to a small floor so downstream multiplications stay
            # finite and so callers can spot the regime via the value.
            return 0.01

        ratio = numer / denom
        # Square-root form per Chamis; clamp the final factor to <= 1.0 so
        # cool / dry conditioning (numer > denom) does not synthesise
        # strength above the RTD allowable.
        return float(min(np.sqrt(ratio), 1.0))

    # Fields a UQ driver is allowed to perturb. Geometry (t_ply, n_plies) and
    # Poisson ratios are excluded by default: the empirical knockdown models
    # only consume strengths/moduli, and perturbing a bounded Poisson ratio or
    # an integer ply count is rarely the intent. Callers may still target any
    # of these via an explicit `covs`/`spec` key if needed.
    #
    # The thermal fields (alpha_1/2/3, T_stress_free) are deliberately not
    # perturbable: no solver consumes them yet, so a draw would change
    # nothing; alpha_1 can be zero or negative, which the CoV-scaled draws
    # and the positive floor in _clip_perturbed do not handle; and alpha_3 is
    # tied to alpha_2 when unset, so perturbing one alone would break
    # transverse isotropy. Revisit with the thermal solve.
    PERTURBABLE_FIELDS = (
        'E11', 'E22', 'E33', 'G12', 'G13', 'G23',
        'sigma_1c', 'sigma_1t', 'sigma_2t', 'sigma_2c', 'tau_12', 'tau_ilss',
        'matrix_modulus', 'fiber_modulus', 'fiber_volume_fraction',
    )

    def _perturbed_value(self, name: str, unit_draw: float,
                         dist: str, params) -> float:
        """Map one unit draw (a standard-normal or U(0,1) variate) to a
        perturbed value of field ``name``.

        ``dist`` is one of:
          - ``'lognormal'`` (default): multiplicative truncated-lognormal so a
            positive quantity stays positive. ``params`` is the coefficient of
            variation (CoV, std/mean of the underlying value). ``unit_draw`` is
            a standard-normal variate.
          - ``'normal'``: additive Gaussian. ``params`` is the CoV; the std is
            ``cov * |nominal|``. ``unit_draw`` is a standard-normal variate.
          - ``'uniform'``: ``params`` is the fractional half-width ``h``; the
            value is drawn uniformly on ``nominal * [1 - h, 1 + h]``.
            ``unit_draw`` is a U(0, 1) variate.
        """
        nominal = float(getattr(self, name))
        if dist == 'lognormal':
            cov = float(params)
            if cov <= 0.0:
                return nominal
            # Median-preserving lognormal with the requested CoV.
            sigma_ln = np.sqrt(np.log1p(cov * cov))
            return self._clip_perturbed(name, nominal * np.exp(sigma_ln * unit_draw))
        if dist == 'normal':
            cov = float(params)
            if cov <= 0.0:
                return nominal
            return self._clip_perturbed(name, nominal + cov * abs(nominal) * unit_draw)
        if dist == 'uniform':
            h = float(params)
            if h <= 0.0:
                return nominal
            return self._clip_perturbed(name, nominal * (1.0 - h + 2.0 * h * unit_draw))
        raise ValueError(
            f"Unknown distribution {dist!r} for field {name!r}. "
            f"Use one of 'lognormal', 'normal', 'uniform'."
        )

    # Fiber volume fraction cannot exceed hexagonal close packing.
    _VF_MAX = float(np.pi / (2.0 * np.sqrt(3.0)))

    def _clip_perturbed(self, name: str, value: float) -> float:
        """Keep a perturbed draw inside the field's valid range.

        A wide distribution can otherwise push ``fiber_volume_fraction``
        past 1 or a modulus / strength below 0 (``'normal'``), and the
        re-validation in ``__post_init__`` would abort the whole UQ sweep
        (IMPROVEMENT_PLAN 2.8). Clipping moves those rare tail draws to the
        bound instead.
        """
        value = float(value)
        if name == 'fiber_volume_fraction':
            return float(min(max(value, 1e-3), self._VF_MAX))
        # Every other perturbable field is a strictly positive modulus or
        # strength.
        return float(max(value, 1e-6 * abs(float(getattr(self, name)))))

    def perturb(self, draws: dict[str, float],
                spec: dict[str, tuple[str, float]]) -> MaterialProperties:
        """Return a new ``MaterialProperties`` with the fields in ``spec``
        perturbed using the per-field unit draws in ``draws``.

        ``spec`` maps ``field -> (distribution, params)`` (see
        :meth:`_perturbed_value`). ``draws`` maps the same field names to a
        single unit variate. Fields absent from ``spec`` are left at nominal.
        The returned dataclass is re-validated by ``__post_init__``.
        """
        updates = {}
        for name, (dist, params) in spec.items():
            updates[name] = self._perturbed_value(
                name, float(draws[name]), dist, params)
        from dataclasses import replace as _dc_replace
        return _dc_replace(self, **updates)

    def __repr__(self) -> str:
        return (f"MaterialProperties(E11={self.E11}, E22={self.E22}, "
                f"G12={self.G12}, nu12={self.nu12}, "
                f"sigma_1c={self.sigma_1c}, sigma_1t={self.sigma_1t}, "
                f"n_plies={self.n_plies}, t_ply={self.t_ply}, "
                f"Vf={self.fiber_volume_fraction})")


# Thermal expansion coefficients (alpha_1 / alpha_2) are set only on presets
# with a cited lamina value for that material system (currently
# AS4_3501_6_epoxy). The others leave them None rather than carry a proxy or
# micromechanics estimate; pass measured values with dataclasses.replace.
MATERIALS = {
    'T800_epoxy': MaterialProperties(
        E11=161000.0, E22=11380.0, E33=11380.0,
        G12=5170.0, G13=5170.0, G23=3980.0,
        nu12=0.32, nu13=0.32, nu23=0.40,
        sigma_1c=1500.0, sigma_1t=2800.0, sigma_2t=80.0, sigma_2c=250.0,
        tau_12=100.0, tau_ilss=90.0,
        t_ply=0.183, n_plies=24,
        matrix_modulus=3500.0, matrix_poisson=0.35,
        fiber_modulus=294000.0, fiber_volume_fraction=0.60,
    ),
    'T700_epoxy': MaterialProperties(
        E11=132000.0, E22=10300.0, E33=10300.0,
        G12=4700.0, G13=4700.0, G23=3500.0,
        nu12=0.30, nu13=0.30, nu23=0.40,
        sigma_1c=1200.0, sigma_1t=2400.0, sigma_2t=65.0, sigma_2c=200.0,
        tau_12=85.0, tau_ilss=80.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=3200.0, matrix_poisson=0.35,
        fiber_modulus=230000.0, fiber_volume_fraction=0.58,
    ),
    'glass_epoxy': MaterialProperties(
        E11=45000.0, E22=12000.0, E33=12000.0,
        G12=5500.0, G13=5500.0, G23=4000.0,
        nu12=0.28, nu13=0.28, nu23=0.40,
        sigma_1c=600.0, sigma_1t=1100.0, sigma_2t=40.0, sigma_2c=140.0,
        tau_12=70.0, tau_ilss=55.0,
        t_ply=0.200, n_plies=24,
        matrix_modulus=3500.0, matrix_poisson=0.35,
        fiber_modulus=73000.0, fiber_volume_fraction=0.55,
    ),
    'IM7_8551_epoxy': MaterialProperties(
        E11=172000.0, E22=10000.0, E33=10000.0,
        G12=5500.0, G13=5500.0, G23=3800.0,
        nu12=0.30, nu13=0.30, nu23=0.45,
        sigma_1c=1600.0, sigma_1t=3100.0, sigma_2t=90.0, sigma_2c=260.0,
        tau_12=110.0, tau_ilss=100.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=3700.0, matrix_poisson=0.35,
        fiber_modulus=276000.0, fiber_volume_fraction=0.60,
    ),
    'T300_934_epoxy': MaterialProperties(
        E11=131000.0, E22=8500.0, E33=8500.0,
        G12=4600.0, G13=4600.0, G23=3000.0,
        nu12=0.28, nu13=0.28, nu23=0.42,
        sigma_1c=1200.0, sigma_1t=1900.0, sigma_2t=55.0, sigma_2c=200.0,
        tau_12=75.0, tau_ilss=85.0,
        t_ply=0.127, n_plies=16,
        matrix_modulus=3400.0, matrix_poisson=0.35,
        fiber_modulus=230000.0, fiber_volume_fraction=0.60,
    ),
    'CF_PEEK': MaterialProperties(
        E11=140000.0, E22=10000.0, E33=10000.0,
        G12=5200.0, G13=5200.0, G23=3500.0,
        nu12=0.32, nu13=0.32, nu23=0.45,
        sigma_1c=1100.0, sigma_1t=2200.0, sigma_2t=85.0, sigma_2c=180.0,
        tau_12=105.0, tau_ilss=95.0,
        t_ply=0.14, n_plies=8,
        matrix_modulus=3800.0, matrix_poisson=0.38,
        fiber_modulus=240000.0, fiber_volume_fraction=0.60,
    ),
    # AS4/3501-6 (Hercules/Hexcel) — an IM-class carbon/untoughened epoxy
    # system. Nominal lamina properties from Soden, Hinton & Kaddour
    # (Worldwide Failure Exercise, WWFE-I, Compos. Sci. Technol. 1998/2002)
    # and Daniel & Ishai, "Engineering Mechanics of Composite Materials"
    # (2nd ed., 2006, Table A.4). 3501-6 neat-resin modulus from Hexcel
    # technical datasheet (E_m ≈ 4.27 GPa, nu_m ≈ 0.34). AS4 fibre modulus
    # from Hexcel HexTow AS4 datasheet (E_f ≈ 235 GPa). Used for Ghiorse 1993
    # (AS4/3501-6 unidirectional) and Jeong 1997 (AS4 fabric/3501-6).
    'AS4_3501_6_epoxy': MaterialProperties(
        E11=142000.0, E22=10300.0, E33=10300.0,
        G12=7200.0, G13=7200.0, G23=3800.0,
        nu12=0.27, nu13=0.27, nu23=0.40,
        sigma_1c=1440.0, sigma_1t=2280.0, sigma_2t=57.0, sigma_2c=228.0,
        tau_12=71.0, tau_ilss=95.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=4270.0, matrix_poisson=0.34,
        fiber_modulus=235000.0, fiber_volume_fraction=0.60,
        # Lamina CTEs (1/K): alpha_1 = -1.0e-6, alpha_2 = 26e-6 from the
        # WWFE-I lamina-properties table for AS4/3501-6 (Soden, Hinton &
        # Kaddour, Compos. Sci. Technol. 58 (1998) 1011-1022). Daniel &
        # Ishai (2006, Table A.4) give -0.9e-6 / 27e-6 for the same system.
        # alpha_3 = alpha_2 (transverse isotropy). T_stress_free is left
        # unset: the thermal load case should get it from the user.
        alpha_1=-1.0e-6, alpha_2=26.0e-6,
    ),
    # HTA 24k / EHkF 420 epoxy — Tenax HTA (Toho Tenax) high-tenacity
    # carbon fibre with a toughened aerospace epoxy system. Lamina
    # properties from the Stamopoulos et al. (2016) baseline tabulation and
    # the Tenax HTA fibre datasheet (E_f ≈ 238 GPa). The matrix modulus
    # (E_m ≈ 3.4 GPa, nu_m ≈ 0.35) is a typical aerospace toughened-epoxy
    # value used in the absence of an EHkF 420 datasheet entry. Strengths
    # are scaled to a standard HTA/epoxy unidirectional with Vf ≈ 0.60.
    'HTA_EHkF420_epoxy': MaterialProperties(
        E11=130000.0, E22=9000.0, E33=9000.0,
        G12=4500.0, G13=4500.0, G23=3200.0,
        nu12=0.32, nu13=0.32, nu23=0.42,
        sigma_1c=1200.0, sigma_1t=2100.0, sigma_2t=60.0, sigma_2c=200.0,
        tau_12=70.0, tau_ilss=75.0,
        t_ply=0.127, n_plies=16,
        matrix_modulus=3400.0, matrix_poisson=0.35,
        fiber_modulus=238000.0, fiber_volume_fraction=0.60,
    ),
}
