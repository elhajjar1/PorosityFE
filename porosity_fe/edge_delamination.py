"""Free-edge delamination: O'Brien's closed-form strain energy release rate.

A laminate in uniaxial tension that delaminates along one or more
interfaces from its free edges loses axial stiffness in proportion to the
delaminated area. O'Brien (1981, 1982) derived from this rule of mixtures
that the strain energy release rate of the growing edge delamination does
not depend on its size::

    G = (eps^2 t / 2) (E_LAM - E*)

with ``eps`` the nominal axial strain, ``t`` the laminate thickness,
``E_LAM`` the axial modulus of the intact laminate and ``E*`` that of the
laminate completely delaminated along the chosen interfaces,
``E* = sum(E_i t_i) / t`` over the sublaminates the delaminations cut it
into. The moduli come from classical lamination theory. Units: MPa, mm,
``G`` in N/mm (1 N/mm = 1 kJ/m^2).

Source: T. K. O'Brien, "Characterization of delamination onset and growth
in a composite laminate", NASA TM-81940 (January 1981), published in
*Damage in Composite Materials*, ASTM STP 775 (1982). The equation
numbers below are those of the NASA report. See the theory page "Failure
criteria", section "Interlaminar stresses and delamination onset".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._ply_angles import _resolve_ply_angles
from .homogenization import _clt_abd_full
from .materials import MaterialProperties


@dataclass(frozen=True)
class EdgeDelamination:
    """O'Brien's free-edge delamination energy release rate of one laminate.

    Built by :func:`obrien_edge_delamination`. Moduli in MPa, lengths in
    mm, energy release rates in N/mm.

    Attributes
    ----------
    ply_angles : tuple of float
        The laminate, bottom ply first (degrees).
    interfaces : tuple of int
        Delaminated interfaces, increasing (interface ``k`` between ply
        ``k`` and ply ``k + 1`` from the bottom).
    sublaminates : tuple of tuple of int
        Ply indices of each sublaminate the delaminations form, bottom first.
    sublaminate_moduli, sublaminate_thicknesses : tuple of float
        ``E_i`` and ``t_i`` of each sublaminate.
    thickness : float
        Laminate thickness ``t``.
    E_LAM, E_star : float
        Axial modulus of the intact laminate, and of the laminate completely
        delaminated along :attr:`interfaces`, ``sum(E_i t_i) / t``.
    strain : float
        Applied nominal axial strain ``eps`` (dimensionless).
    bending_coupling : bool
        Whether the moduli keep the bending-extension coupling of
        unsymmetric (sub)laminates (O'Brien's eq. B1) or neglect it (eq. 1,
        O'Brien's choice for his prediction).
    G : float
        ``(eps^2 t / 2) (E_LAM - E*)``, O'Brien's eq. (10): one delamination
        growing from each edge.
    """

    ply_angles: tuple[float, ...]
    interfaces: tuple[int, ...]
    sublaminates: tuple[tuple[int, ...], ...]
    sublaminate_moduli: tuple[float, ...]
    sublaminate_thicknesses: tuple[float, ...]
    thickness: float
    E_LAM: float
    E_star: float
    strain: float
    bending_coupling: bool
    G: float

    @property
    def stiffness_ratio(self) -> float:
        """``E* / E_LAM``, the stiffness left once fully delaminated."""
        return self.E_star / self.E_LAM

    @property
    def G_simultaneous(self) -> float:
        """``G / m`` with ``m`` the number of delaminated interfaces.

        The energy release rate of each crack front when the delaminations
        of all ``m`` interfaces grow together at each edge. O'Brien's
        eq. (10) assumes one delamination per edge, moving between the
        interfaces as observed in his specimens; he notes that ``m = 2``
        simultaneous delaminations give ``(eps^2 t / 4) (E_LAM - E*)`` (his
        Appendix A, where his finite-element values are doubled to match
        eq. (10)).
        """
        return self.G / len(self.interfaces)

    def load_factor(self, G_c: float) -> float:
        """Multiplier on :attr:`strain` at which :attr:`G` reaches ``G_c``.

        ``G`` scales with ``eps^2``, so the factor is ``sqrt(G_c / G)``
        (``inf`` when ``G = 0``). ``G_c`` in N/mm, positive.
        """
        G_c = _positive('G_c', G_c)
        return math.inf if self.G <= 0.0 else math.sqrt(G_c / self.G)

    def critical_strain(self, G_c: float) -> float:
        """Nominal strain at which :attr:`G` reaches ``G_c`` (N/mm).

        ``sqrt(2 G_c / (t (E_LAM - E*)))``, O'Brien's onset prediction;
        ``inf`` when ``E* = E_LAM``.
        """
        G_c = _positive('G_c', G_c)
        drop = self.thickness * (self.E_LAM - self.E_star)
        return math.inf if drop <= 0.0 else math.sqrt(2.0 * G_c / drop)


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
            or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return float(value)


def _axial_modulus(C_base: np.ndarray, angles: list[float], t_ply: float,
                   coupling: bool) -> float:
    """``1 / (t a'_11)``: the axial modulus of a (sub)laminate in CLT.

    ``a'`` is ``A^-1`` without coupling (O'Brien's eq. 1) or the in-plane
    block of the inverse of the full ``ABD`` matrix with it (eq. B1), both
    about the stack's own mid-plane.
    """
    A, B, D = _clt_abd_full(C_base, angles, t_ply)
    t = len(angles) * t_ply
    if coupling:
        a11 = np.linalg.inv(np.block([[A, B], [B, D]]))[0, 0]
    else:
        a11 = np.linalg.inv(A)[0, 0]
    return float(1.0 / (t * a11))


def obrien_edge_delamination(material: MaterialProperties,
                             ply_angles: Sequence[float] | str,
                             interfaces: Iterable[int], strain: float, *,
                             bending_coupling: bool = False,
                             t_ply: float | None = None) -> EdgeDelamination:
    """Free-edge delamination energy release rate of a laminate in tension (O'Brien).

    ``G = (eps^2 t / 2) (E_LAM - E*)`` (O'Brien 1981, eq. 10), independent
    of the delamination size. ``E_LAM`` is the axial modulus of the intact
    laminate from classical lamination theory, ``E_LAM = 1 / (t a_11)``
    with ``a = A^-1``. ``E* = sum(E_i t_i) / t`` (his eq. 2) is the modulus
    of the laminate completely delaminated along ``interfaces``, from the
    moduli ``E_i`` of the sublaminates it then falls into, each loaded to
    the same axial strain (but free to take its own transverse strain).

    Parameters
    ----------
    material : MaterialProperties
        Ply properties: ``E11``, ``E22``, ``nu12`` and ``G12`` set the
        plane-stress ply stiffness (pristine; porosity is not applied), and
        ``t_ply`` the ply thickness unless ``t_ply`` is given.
    ply_angles : sequence of float or {'QI', 'UD'}
        The laminate, bottom ply first, in degrees: exactly these plies
        (not tiled to ``material.n_plies``). The sentinels resolve as for
        :class:`~porosity_fe.CompositeMesh`.
    interfaces : iterable of int
        Delaminated interfaces, at least one: interface ``k`` lies between
        ply ``k`` and ply ``k + 1`` (0-based, from the bottom), as for
        :class:`~porosity_fe.Delamination`.
    strain : float
        Nominal axial strain ``eps`` (dimensionless, e.g. ``0.004``; not a
        percent). ``G`` scales with ``eps^2``. O'Brien's derivation is for
        tension, where the edge delamination opens.
    bending_coupling : bool, optional
        Keyword-only. ``False`` (default) computes every modulus from
        ``A^-1`` alone (O'Brien's eq. 1), neglecting the bending-extension
        coupling of unsymmetric sublaminates, as O'Brien did for his
        prediction: it matched the measured stiffness loss of his
        ``[+-30/+-30/90/90bar]s`` specimens (``E* = 0.743 E_LAM``) and his
        finite-element analysis. ``True`` keeps the coupling, the
        in-plane block of the inverse of the full ``ABD`` matrix about each
        stack's mid-plane (his eq. B1), which gives a lower ``E*``
        (``0.69 E_LAM`` for that laminate) and so a higher ``G``.
    t_ply : float, optional
        Ply thickness in mm; default ``material.t_ply``.

    Returns
    -------
    EdgeDelamination
        ``E_LAM``, ``E*``, the sublaminates, ``G`` (one delamination per
        edge, eq. 10) and ``G_simultaneous`` (all delaminated interfaces
        growing together).

    Raises
    ------
    ValueError
        For an empty laminate, an interface outside ``0 .. n - 2``, no
        interface, or a non-finite strain or non-positive ply thickness.

    Notes
    -----
    The rule of mixtures holds away from the delamination front, so ``G``
    is the value the growing delamination approaches once it is a few ply
    thicknesses long; near the free edge O'Brien's own finite-element
    analysis gives less. It is the total energy release rate: the mode
    split needs a finite-element analysis (VCCT). Compare it with a
    toughness measured in the same convention: O'Brien's eq. (10) takes the
    delaminated area as the planform area, one delamination per edge; see
    :attr:`EdgeDelamination.G_simultaneous`.

    Examples
    --------
    O'Brien's edge-delamination specimen ``[+-30/+-30/90/90bar]s`` (11
    plies, the centre 90 not repeated), delaminated at both ``-30/90``
    interfaces:

    >>> layup = [30, -30, 30, -30, 90, 90, 90, -30, 30, -30, 30]
    >>> edt = obrien_edge_delamination(mat, layup, [3, 6], 3470e-6)
    >>> edt.stiffness_ratio, edt.G        # about 0.74, N/mm
    """
    angles = _resolve_ply_angles(ply_angles, caller='obrien_edge_delamination.ply_angles')
    assert angles is not None
    if not angles or not all(math.isfinite(a) for a in angles):
        raise ValueError(
            f"ply_angles must be a non-empty sequence of finite angles in "
            f"degrees, got {ply_angles!r}.")
    n = len(angles)
    try:
        chosen = sorted({int(k) for k in interfaces})
    except TypeError:
        raise ValueError(
            f"interfaces must be an iterable of interface indices, got "
            f"{interfaces!r}.") from None
    if not chosen:
        raise ValueError("interfaces must name at least one delaminated interface.")
    bad = [k for k in chosen if not 0 <= k <= n - 2]
    if bad:
        raise ValueError(
            f"interfaces {bad} do not exist: a {n}-ply laminate has interfaces "
            f"0 to {n - 2} (interface k lies between ply k and ply k + 1, from "
            f"the bottom).")
    if isinstance(strain, bool) or not isinstance(strain, (int, float, np.number)) \
            or not math.isfinite(float(strain)):
        raise ValueError(f"strain must be a finite number, got {strain!r}.")
    h = _positive('t_ply', material.t_ply if t_ply is None else t_ply)

    C_base = material.get_stiffness_matrix()
    coupling = bool(bending_coupling)
    E_lam = _axial_modulus(C_base, angles, h, coupling)
    cuts = [0] + [k + 1 for k in chosen] + [n]
    subs = tuple(tuple(range(a, b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True))
    moduli = tuple(_axial_modulus(C_base, [angles[p] for p in s], h, coupling)
                   for s in subs)
    thick = tuple(len(s) * h for s in subs)
    t = n * h
    E_star = float(sum(E * ti for E, ti in zip(moduli, thick, strict=True)) / t)
    eps = float(strain)
    return EdgeDelamination(
        ply_angles=tuple(angles), interfaces=tuple(chosen), sublaminates=subs,
        sublaminate_moduli=moduli, sublaminate_thicknesses=thick, thickness=t,
        E_LAM=E_lam, E_star=E_star, strain=eps, bending_coupling=coupling,
        G=0.5 * eps * eps * t * (E_lam - E_star))
