"""Solver material cards of the porosity-degraded ply.

:func:`degraded_ply` gives the engineering constants and the strengths of a
ply at one void content, as the FE path of this package degrades them.
:func:`material_cards` writes them for a laminate, one material per
distinct ply void content, as an Abaqus ``*MATERIAL`` block or a Nastran
``MAT8`` / ``MAT9`` entry with the ply table, so that the knocked-down
properties can be carried into another model.

Units are those of :class:`~porosity_fe.MaterialProperties`: MPa and mm.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._ply_angles import _resolve_ply_angles
from ._types import PorosityDegradation
from ._vp import check_fraction
from .homogenization import _degraded_composite_stiffness
from .materials import MaterialProperties
from .porosity_field import PorosityField
from .void_geometry import VOID_SHAPES

CardFormat = Literal['abaqus', 'nastran']
CardElement = Literal['shell', 'solid']
CardStrengths = Literal['degraded', 'pristine', 'none']

_FORMATS = ('abaqus', 'nastran')
_ELEMENTS = ('shell', 'solid')
_STRENGTHS = ('degraded', 'pristine', 'none')
#: File extensions :func:`write_material_cards` reads the format from.
_EXTENSIONS = {'.inp': 'abaqus', '.bdf': 'nastran', '.dat': 'nastran',
               '.nas': 'nastran', '.blk': 'nastran'}
#: Above this void content the micromechanics leaves no stiffness.
_VP_MAX = 0.99
#: Gauss points per ply of the through-thickness average.
_PLY_GAUSS_POINTS = 16
#: Ply void contents closer than this share one material.
_VP_GROUP_DECIMALS = 9
#: Voigt order of this package, ``[11, 22, 33, 23, 13, 12]``, to Nastran's
#: ``[xx, yy, zz, xy, yz, zx]`` of ``MAT9``.
_NASTRAN_VOIGT = (0, 1, 2, 5, 3, 4)

_STRENGTH_NOTES = {
    'degraded': (
        "Strengths: scaled by sqrt(stiffness retention), the heuristic of the "
        "PorosityFE finite-element path. It is not calibrated against coupon "
        "data and loses less strength than the tests show; apply your own "
        "knockdown for an allowable."),
    'pristine': "Strengths: pristine (not degraded by the porosity).",
    'none': "Strengths: not written.",
}


def _void_radii(void_shape: Any) -> tuple[float, float, float]:
    """The Eshelby radii of a named shape or of three radii."""
    if isinstance(void_shape, str):
        if void_shape not in VOID_SHAPES:
            raise ValueError(
                f"Unknown void_shape {void_shape!r}. Use one of {sorted(VOID_SHAPES)}.")
        return VOID_SHAPES[void_shape]
    try:
        radii = tuple(float(r) for r in void_shape)
    except (TypeError, ValueError):
        radii = ()
    if len(radii) != 3 or not all(np.isfinite(r) and r > 0.0 for r in radii):
        raise ValueError(
            f"void_shape must be one of {sorted(VOID_SHAPES)} or three finite "
            f"radii > 0 (a1, a2, a3); got {void_shape!r}.")
    return radii  # type: ignore[return-value]


@dataclass(frozen=True, eq=False)
class DegradedPly:
    """Engineering constants and strengths of a ply at one void content.

    Attributes
    ----------
    Vp : float
        Void content of the ply, a **fraction**.
    E11, E22, E33, G12, G13, G23 : float
        Degraded moduli in MPa (1 along the fibres, 3 through the thickness).
    nu12, nu13, nu23 : float
        Degraded major Poisson's ratios.
    Xt, Xc, Yt, Yc, S12, S23 : float
        Strengths in MPa, all positive: fibre tension and compression,
        transverse tension and compression, in-plane and interlaminar shear.
    Zt, S13 : float or None
        Through-thickness tensile and interlaminar ``13`` shear strength in
        MPa, when the material sets ``sigma_3t`` / ``tau_13``.
    strengths : {'degraded', 'pristine'}
        Whether the strengths carry the porosity (the
        ``sqrt(stiffness retention)`` heuristic of the FE path) or are the
        material's own.
    retention : dict
        Ratio of each degraded constant to its pristine value, keyed like
        the attributes (``'E11'`` ... ``'nu23'``).
    stiffness : numpy.ndarray
        The 6 x 6 stiffness in MPa the constants are read from, in the Voigt
        order ``[11, 22, 33, 23, 13, 12]`` with engineering shear strains.
    """

    Vp: float
    E11: float
    E22: float
    E33: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float
    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    S23: float
    Zt: float | None
    S13: float | None
    strengths: str
    retention: dict = field(repr=False)
    stiffness: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        """JSON-ready record (the stiffness as nested lists)."""
        out: dict = {name: getattr(self, name) for name in (
            'Vp', 'E11', 'E22', 'E33', 'G12', 'G13', 'G23', 'nu12', 'nu13',
            'nu23', 'Xt', 'Xc', 'Yt', 'Yc', 'S12', 'S23', 'Zt', 'S13',
            'strengths')}
        out['retention'] = dict(self.retention)
        out['stiffness'] = self.stiffness.tolist()
        return out


_CONSTANTS = ('E11', 'E22', 'E33', 'G12', 'G13', 'G23', 'nu12', 'nu13', 'nu23')


def degraded_ply(material: MaterialProperties, Vp: float, *,
                 void_shape: str | Sequence[float] = 'spherical',
                 porosity_degradation: PorosityDegradation = 'isotropic',
                 strengths: Literal['degraded', 'pristine'] = 'degraded',
                 ) -> DegradedPly:
    """The ply constants and strengths at the void content ``Vp``.

    The stiffness is the one :class:`~porosity_fe.FESolver` gives an
    element at that void content: Mori-Tanaka voids in the matrix, then the
    ply constants through rule-of-mixtures and Halpin-Tsai ratios. The
    engineering constants are read from its inverse.

    Parameters
    ----------
    material : MaterialProperties
        The pristine ply.
    Vp : float
        Void content, a **fraction** in ``[0, 0.99]``.
    void_shape : str or three floats
        ``'spherical'`` (default), ``'cylindrical'``, ``'penny'`` or the
        Eshelby radii ``(a1, a2, a3)``.
    porosity_degradation : {'isotropic', 'directional'}
        See :data:`~porosity_fe.PorosityDegradation`.
    strengths : {'degraded', 'pristine'}
        ``'degraded'`` (default) scales the strengths by the square root of
        the stiffness retention, as the FE failure criteria do. That
        heuristic is not calibrated: over the validation datasets the FE
        strengths lose less than the coupons. ``'pristine'`` keeps the
        material's strengths.

    Raises
    ------
    ValueError
        For a void content outside ``[0, 0.99]``, an unknown option, or a
        ``'directional'`` degradation that leaves no physical ply.

    Examples
    --------
    >>> from porosity_fe import MATERIALS, degraded_ply
    >>> ply = degraded_ply(MATERIALS['T800_epoxy'], 0.03)
    >>> round(ply.retention['E11'], 4), round(ply.retention['E22'], 4)
    (0.9995, 0.9456)
    """
    if np.ndim(Vp) != 0:
        raise ValueError(f"Vp must be one void fraction; got {Vp!r}.")
    vp = float(check_fraction(Vp))
    if vp > _VP_MAX:
        raise ValueError(
            f"Vp={vp:g} leaves no ply to describe: the micromechanics holds "
            f"up to {_VP_MAX}.")
    if strengths not in ('degraded', 'pristine'):
        raise ValueError(
            f"strengths must be 'degraded' or 'pristine'; got {strengths!r}.")
    radii = _void_radii(void_shape)
    C = _degraded_composite_stiffness(vp, radii, material,
                                      porosity_degradation=porosity_degradation)
    S = np.linalg.inv(C)
    E11, E22, E33 = 1.0 / S[0, 0], 1.0 / S[1, 1], 1.0 / S[2, 2]
    values = {
        'E11': E11, 'E22': E22, 'E33': E33,
        'G12': 1.0 / S[5, 5], 'G13': 1.0 / S[4, 4], 'G23': 1.0 / S[3, 3],
        'nu12': -S[0, 1] * E11, 'nu13': -S[0, 2] * E11, 'nu23': -S[1, 2] * E22,
    }
    values = {name: float(value) for name, value in values.items()}
    retention = {name: values[name] / float(getattr(material, name))
                 for name in _CONSTANTS}

    if strengths == 'degraded':
        # Imported here: the failure module imports the solver stack.
        from .fe.failure import degraded_interlaminar_strengths, degraded_strengths
        Xt, Xc, Yt, Yc, S12, S23 = degraded_strengths(
            material, radii, vp, porosity_degradation=porosity_degradation)
        Zt, S13 = degraded_interlaminar_strengths(
            material, radii, vp, porosity_degradation=porosity_degradation,
            criterion='tsai_wu')
        # Only the strengths the material sets: no stand-in on a card.
        Zt = None if material.sigma_3t is None else Zt
        S13 = None if material.tau_13 is None else S13
    else:
        Xt, Xc = material.sigma_1t, material.sigma_1c
        Yt, Yc = material.sigma_2t, material.sigma_2c
        S12, S23 = material.tau_12, material.tau_ilss
        Zt, S13 = material.sigma_3t, material.tau_13
    return DegradedPly(
        Vp=vp, **values,
        Xt=float(Xt), Xc=float(Xc), Yt=float(Yt), Yc=float(Yc),
        S12=float(S12), S23=float(S23),
        Zt=None if Zt is None else float(Zt),
        S13=None if S13 is None else float(S13),
        strengths=strengths, retention=retention, stiffness=C)


def ply_porosities(porosity_field: PorosityField, *,
                   at: tuple[float, float] | None = None) -> np.ndarray:
    """Mean void content of each ply of a porosity field, bottom ply first.

    The through-thickness profile of the field is averaged over each ply
    and scaled so that the plies average to the field's ``Vp``. Discrete
    voids are local and are left out.

    Parameters
    ----------
    porosity_field : PorosityField
        The field; its material gives the ply count and thickness.
    at : (float, float), optional
        Keyword-only. An in-plane point ``(x, y)`` in mm: with an
        ``in_plane_map`` the plies take the map's value there. ``None``
        (default) is the average over the plane.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_plies,)``, fractions in ``[0, 1]``.
    """
    if not isinstance(porosity_field, PorosityField):
        raise TypeError(
            f"porosity_field must be a PorosityField; got "
            f"{type(porosity_field).__name__}.")
    mat = porosity_field.material
    n, t = int(mat.n_plies), float(mat.t_ply)
    nodes, weights = np.polynomial.legendre.leggauss(_PLY_GAUSS_POINTS)
    z = (np.arange(n)[:, None] + 0.5 * (nodes[None, :] + 1.0)) * t
    values = porosity_field._distributed_porosity(z.ravel()).reshape(n, -1)
    plies = (values * (0.5 * weights)[None, :]).sum(axis=1)
    mean = float(plies.mean())
    if mean > 0.0:
        plies = plies * (porosity_field.Vp / mean)
    if at is not None:
        m = porosity_field.in_plane_map
        try:
            x, y = (float(v) for v in at)
        except (TypeError, ValueError):
            raise ValueError(f"at must be a point (x, y) in mm; got {at!r}.") from None
        if m is not None and m.mean > 0.0:
            plies = plies * (float(m(x, y)) / m.mean)
    return np.clip(plies, 0.0, 1.0)


# ----------------------------------------------------------------------
# The laminate the cards describe
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Laminate:
    angles: np.ndarray              # (n_plies,) degrees, bottom ply first
    t_ply: float
    ply_material: np.ndarray        # (n_plies,) index into ``plies``
    plies: tuple[DegradedPly, ...]  # one per distinct void content
    description: str                # of the porosity, for the header


def _laminate(material: MaterialProperties, porosity: Any, ply_angles: Any,
              void_shape: Any, porosity_degradation: str, strengths: str,
              at: Any) -> _Laminate:
    from .mesh import _expand_layup

    resolved = _resolve_ply_angles(ply_angles, caller='ply_angles')
    assert resolved is not None
    n_plies = int(material.n_plies)
    if len(resolved) > n_plies:
        raise ValueError(
            f"The layup has {len(resolved)} plies but material.n_plies is "
            f"{n_plies}; set n_plies to the laminate's ply count "
            f"(dataclasses.replace(material, n_plies={len(resolved)})).")
    angles = _expand_layup(resolved, n_plies)

    if isinstance(porosity, PorosityField):
        field_material = porosity.material
        if (int(field_material.n_plies) != n_plies
                or float(field_material.t_ply) != float(material.t_ply)):
            raise ValueError(
                f"The porosity field describes {field_material.n_plies} plies "
                f"of {field_material.t_ply} mm, the material {n_plies} of "
                f"{material.t_ply} mm: build both from the same laminate.")
        ply_vp = ply_porosities(porosity, at=at)
        radii = porosity.void_shape_radii if void_shape is None \
            else _void_radii(void_shape)
        description = (f"{porosity.distribution} profile, specimen average "
                       f"Vp = {porosity.Vp:.6g}")
        if porosity.distribution == 'clustered':
            description += f", clustered at the {porosity.cluster_location}"
        if at is not None and porosity.in_plane_map is not None:
            description += f", at (x, y) = ({float(at[0]):g}, {float(at[1]):g}) mm"
        elif porosity.in_plane_map is not None:
            description += ", in-plane map averaged"
        if porosity.discrete_voids:
            description += (f"; {len(porosity.discrete_voids)} discrete "
                            f"void(s) not included")
    else:
        if at is not None:
            raise ValueError("at needs a PorosityField with an in-plane map.")
        if np.ndim(porosity) != 0:
            raise ValueError(
                f"porosity must be a void fraction or a PorosityField; got "
                f"{porosity!r}.")
        vp = float(check_fraction(porosity, 'porosity'))
        ply_vp = np.full(n_plies, vp)
        radii = _void_radii('spherical' if void_shape is None else void_shape)
        description = f"uniform, Vp = {vp:.6g}"
    shape_name = next((name for name, r in VOID_SHAPES.items() if r == radii),
                      f"radii {radii}")
    description += f"; void shape {shape_name}; {porosity_degradation} degradation"

    keys = np.round(ply_vp, _VP_GROUP_DECIMALS)
    distinct = list(dict.fromkeys(keys.tolist()))
    index = {key: i for i, key in enumerate(distinct)}
    ply_strengths = 'pristine' if strengths == 'none' else strengths
    plies = tuple(
        degraded_ply(material, key, void_shape=radii,
                     porosity_degradation=porosity_degradation,  # type: ignore[arg-type]
                     strengths=ply_strengths)                    # type: ignore[arg-type]
        for key in distinct)
    return _Laminate(angles=angles, t_ply=float(material.t_ply),
                     ply_material=np.array([index[key] for key in keys.tolist()]),
                     plies=plies, description=description)


def _header(material: MaterialProperties, lam: _Laminate, strengths: str,
            element: str) -> list[str]:
    from . import __version__
    lines = [
        f"PorosityFE {__version__}: porosity-degraded ply material, "
        f"{element} elements",
        "Units: MPa, mm (density, when given, in the model's own unit)",
        f"Porosity: {lam.description}",
        f"Pristine ply: E11 = {material.E11:g}, E22 = {material.E22:g}, "
        f"G12 = {material.G12:g} MPa, nu12 = {material.nu12:g}",
        "Stiffness: Mori-Tanaka voids in the matrix, ply constants by "
        "rule-of-mixtures and Halpin-Tsai ratios",
    ]
    lines.append(_STRENGTH_NOTES[strengths])
    return [row for text in lines for row in _wrap(text)]


def _wrap(text: str, width: int = 68) -> list[str]:
    """``text`` in comment lines short enough for an 80-column deck."""
    return textwrap.wrap(text, width=width, subsequent_indent='  ')


def _comment(prefix: str, text: str) -> list[str]:
    """``text`` as wrapped comment lines behind ``prefix``."""
    return [f"{prefix} {row}" for row in _wrap(text)]


def _num(value: float) -> str:
    """A real to 9 significant digits, always written as a real."""
    text = f"{float(value):.9g}"
    return text if ('.' in text or 'e' in text or 'n' in text) else text + '.'


def _ply_density(density: float | None, ply: DegradedPly) -> float | None:
    """Density of the porous ply: the void-free density times ``1 - Vp``."""
    return None if density is None else float(density) * (1.0 - ply.Vp)


def _material_names(name: str, count: int) -> list[str]:
    return [name] if count == 1 else [f"{name}_{i + 1}" for i in range(count)]


def _normalized_f12(material: MaterialProperties) -> float:
    return -0.5 if material.tsai_wu_F12 is None else float(material.tsai_wu_F12)


# ----------------------------------------------------------------------
# Abaqus
# ----------------------------------------------------------------------
def _abaqus(material: MaterialProperties, lam: _Laminate, *, element: str,
            strengths: str, name: str, section: Any,
            density: float | None) -> list[str]:
    if section is not None and (not isinstance(section, str) or not section.strip()):
        raise ValueError(
            f"For Abaqus, section is the element-set name of the composite "
            f"shell section; got {section!r}.")
    lines = [f"** {text}" for text in _header(material, lam, strengths, element)]
    names = _material_names(name, len(lam.plies))
    f12 = _normalized_f12(material)
    for mat_name, ply in zip(names, lam.plies, strict=True):
        lines.append("**")
        lines.append(f"** Vp = {ply.Vp:.6g}: E22 retains "
                     f"{100.0 * ply.retention['E22']:.2f} %, G12 "
                     f"{100.0 * ply.retention['G12']:.2f} %")
        lines.append(f"*MATERIAL, NAME={mat_name}")
        if element == 'shell':
            lines.append("*ELASTIC, TYPE=LAMINA")
            lines.append(", ".join(_num(v) for v in (
                ply.E11, ply.E22, ply.nu12, ply.G12, ply.G13, ply.G23)))
        else:
            lines.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
            lines.append(", ".join(_num(v) for v in (
                ply.E11, ply.E22, ply.E33, ply.nu12, ply.nu13, ply.nu23,
                ply.G12, ply.G13)) + ",")
            lines.append(_num(ply.G23))
        rho = _ply_density(density, ply)
        if rho is not None:
            lines.extend(["*DENSITY", _num(rho)])
        if material.has_cte:
            lines.append("*EXPANSION, TYPE=ORTHO")
            lines.append(", ".join(_num(v) for v in material.cte_vector()[:3]))
        if strengths != 'none':
            limits = ", ".join(_num(v) for v in (
                ply.Xt, -ply.Xc, ply.Yt, -ply.Yc, ply.S12, f12))
            if element == 'shell':
                # Abaqus takes the compressive limits as negative numbers and
                # f* as the normalized Tsai-Wu interaction coefficient.
                lines.extend(["*FAIL STRESS", limits])
            else:
                lines.extend(_comment("**", (
                    "Strengths in MPa (*FAIL STRESS is for plane-stress "
                    "elements): Xt, Xc, Yt, Yc, S12, f*")))
                lines.append(f"**   {limits}")
            extra = [f"S23 = {_num(ply.S23)}"]
            if ply.Zt is not None:
                extra.append(f"Zt = {_num(ply.Zt)}")
            if ply.S13 is not None:
                extra.append(f"S13 = {_num(ply.S13)}")
            lines.extend(_comment(
                "**", f"Interlaminar strengths (MPa): {', '.join(extra)}"))
    lines.append("**")
    table = [f"{_num(lam.t_ply)}, 3, {names[m]}, {_num(a)}, PLY_{k + 1}"
             for k, (m, a) in enumerate(zip(lam.ply_material, lam.angles, strict=True))]
    layup = _comment("**", "Layup, bottom ply first: thickness (mm), "
                     "integration points, material, angle (deg), ply name")
    if section is not None:
        if element != 'shell':
            raise ValueError(
                "section writes a composite shell section; with element="
                "'solid' the ply table is written as comments (the plies of "
                "a solid model follow its mesh).")
        lines.extend(layup)
        lines.append(f"*SHELL SECTION, ELSET={section.strip()}, COMPOSITE")
        lines.extend(table)
    else:
        lines.extend(layup)
        lines.extend(f"**   {row}" for row in table)
    return lines


# ----------------------------------------------------------------------
# Nastran (large-field format: 16-character fields keep the precision)
# ----------------------------------------------------------------------
def _field16(value: Any) -> str:
    if value is None:
        return " " * 16
    if isinstance(value, str):
        return f"{value:<16}"
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("a Nastran field is an integer, a real or a string")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):>16d}"
    return f"{float(value):>16.9E}"


def _nastran_entry(name: str, fields: Sequence[Any]) -> list[str]:
    """A bulk-data entry in large-field format.

    ``fields`` are the entry's fields after its name, eight to a logical
    line; trailing blanks are dropped.
    """
    items = list(fields)
    while items and items[-1] is None:
        items.pop()
    lines = []
    for start in range(0, len(items), 4):
        head = f"{name}*" if start == 0 else "*"
        lines.append((f"{head:<8}" + "".join(
            _field16(v) for v in items[start:start + 4])).rstrip())
    return lines


def _nastran(material: MaterialProperties, lam: _Laminate, *, element: str,
             strengths: str, first_id: int, section: Any,
             density: float | None) -> list[str]:
    if isinstance(first_id, bool) or not isinstance(first_id, (int, np.integer)) \
            or first_id < 1:
        raise ValueError(f"first_id must be a material id >= 1; got {first_id!r}.")
    if section is not None and (
            isinstance(section, bool)
            or not isinstance(section, (int, np.integer)) or section < 1):
        raise ValueError(
            f"For Nastran, section is the property id of the PCOMP entry, an "
            f"integer >= 1; got {section!r}.")
    lines = [f"$ {text}" for text in _header(material, lam, strengths, element)]
    ids = [int(first_id) + i for i in range(len(lam.plies))]
    f12_normalized = _normalized_f12(material)
    cte = material.cte_vector() if material.has_cte else None
    for mid, ply in zip(ids, lam.plies, strict=True):
        lines.append("$")
        lines.append(f"$ Material {mid}: Vp = {ply.Vp:.6g}, E22 retains "
                     f"{100.0 * ply.retention['E22']:.2f} %, G12 "
                     f"{100.0 * ply.retention['G12']:.2f} %")
        rho = _ply_density(density, ply)
        if element == 'shell':
            fields: list[Any] = [mid, ply.E11, ply.E22, ply.nu12, ply.G12,
                                 ply.G13, ply.G23, rho]
            fields += [None if cte is None else float(cte[0]),
                       None if cte is None else float(cte[1]), None]
            if strengths != 'none':
                # Nastran takes the compressive limits as positive numbers
                # and F12 as the dimensional Tsai-Wu coefficient (1/MPa^2).
                f12 = f12_normalized / float(
                    np.sqrt(ply.Xt * ply.Xc * ply.Yt * ply.Yc))
                fields += [ply.Xt, ply.Xc, ply.Yt, ply.Yc, ply.S12, None, f12]
            lines.extend(_nastran_entry('MAT8', fields))
        else:
            G = ply.stiffness[np.ix_(_NASTRAN_VOIGT, _NASTRAN_VOIGT)]
            fields = [mid] + [float(G[i, j]) for i in range(6) for j in range(i, 6)]
            fields.append(rho)
            if cte is not None:
                fields += [float(cte[0]), float(cte[1]), float(cte[2]),
                           0.0, 0.0, 0.0]
            lines.extend(_comment("$", (
                "MAT9 in the ply axes (1 fibre, 3 thickness): point CORDM of "
                "the PSOLID at them")))
            lines.extend(_nastran_entry('MAT9', fields))
        if strengths != 'none':
            extra = [f"S23 = {_num(ply.S23)}"]
            if element == 'solid':
                extra = [f"Xt = {_num(ply.Xt)}", f"Xc = {_num(ply.Xc)}",
                         f"Yt = {_num(ply.Yt)}", f"Yc = {_num(ply.Yc)}",
                         f"S12 = {_num(ply.S12)}"] + extra
            if ply.Zt is not None:
                extra.append(f"Zt = {_num(ply.Zt)}")
            if ply.S13 is not None:
                extra.append(f"S13 = {_num(ply.S13)}")
            label = "Strengths" if element == 'solid' else "Interlaminar strengths"
            lines.extend(_comment("$", f"{label} (MPa): {', '.join(extra)}"))
    lines.append("$")
    lines.extend(_comment("$", "Layup, bottom ply first: ply, material id, "
                          "thickness (mm), angle (deg)"))
    rows = [(k + 1, ids[m], lam.t_ply, float(a))
            for k, (m, a) in enumerate(zip(lam.ply_material, lam.angles, strict=True))]
    lines.extend(f"$   {k}, {mid}, {_num(t)}, {_num(a)}" for k, mid, t, a in rows)
    if section is not None:
        if element != 'shell':
            raise ValueError(
                "section writes a PCOMP entry; with element='solid' the ply "
                "table is written as comments (the plies of a solid model "
                "follow its mesh).")
        if strengths == 'none':
            head: list[Any] = [int(section)]
        else:
            # SB: the allowable interlaminar shear of the weakest ply; FT:
            # Tsai-Wu, the criterion F12 is given for.
            head = [int(section), None, None,
                    min(ply.S23 for ply in lam.plies), 'TSAI']
        head += [None] * (8 - len(head))
        plies: list[Any] = []
        for _, mid, t, a in rows:
            plies += [mid, t, a, 'YES']
        lines.extend(_nastran_entry('PCOMP', head + plies))
    return lines


# ----------------------------------------------------------------------
# Public writers
# ----------------------------------------------------------------------
def material_cards(material: MaterialProperties,
                   porosity: float | PorosityField,
                   fmt: CardFormat = 'abaqus', *,
                   ply_angles: Sequence[float] | str = 'QI',
                   element: CardElement = 'shell',
                   strengths: CardStrengths = 'degraded',
                   porosity_degradation: PorosityDegradation = 'isotropic',
                   void_shape: str | Sequence[float] | None = None,
                   name: str = 'POROUS_PLY',
                   first_id: int = 1,
                   section: str | int | None = None,
                   density: float | None = None,
                   at: tuple[float, float] | None = None) -> str:
    """The porosity-degraded plies of a laminate as solver material cards.

    One material is written per distinct ply void content, then the ply
    table (thickness, angle and material of every ply, bottom ply first).

    Parameters
    ----------
    material : MaterialProperties
        The pristine ply; ``n_plies`` and ``t_ply`` give the laminate.
    porosity : float or PorosityField
        A void **fraction** for every ply, or a field: each ply then takes
        the mean of the field's through-thickness profile over it
        (:func:`ply_porosities`), so a clustered profile gives the plies
        different materials.
    fmt : {'abaqus', 'nastran'}
        Abaqus keywords, or Nastran bulk data in large-field format.
    ply_angles : list of float or {'QI', 'UD'}
        The layup, repeated to ``material.n_plies`` plies as
        :class:`~porosity_fe.CompositeMesh` lays it up.
    element : {'shell', 'solid'}
        ``'shell'`` (default): ``*ELASTIC, TYPE=LAMINA`` with ``*FAIL
        STRESS``, or ``MAT8``. ``'solid'``: ``*ELASTIC, TYPE=ENGINEERING
        CONSTANTS``, or ``MAT9`` (the 6 x 6 stiffness in the ply axes),
        with the strengths as comments.
    strengths : {'degraded', 'pristine', 'none'}
        ``'degraded'`` (default): scaled by ``sqrt(stiffness retention)``
        like the FE failure criteria, an uncalibrated heuristic that loses
        less strength than coupon tests show. ``'pristine'``: the
        material's own. ``'none'``: not written.
    porosity_degradation : {'isotropic', 'directional'}
        See :data:`~porosity_fe.PorosityDegradation`.
    void_shape : str or three floats, optional
        The void shape. ``None`` (default) takes the field's, or
        ``'spherical'`` for a void fraction.
    name : str
        Abaqus material name (default ``'POROUS_PLY'``); several materials
        are numbered ``name_1``, ``name_2``, ...
    first_id : int
        Nastran id of the first material (default 1); the others follow.
    section : str or int, optional
        Shell elements only. Abaqus: the element-set name, which adds a
        ``*SHELL SECTION, COMPOSITE`` with the layup. Nastran: the property
        id, which adds a ``PCOMP``. ``None`` (default) writes the ply table
        as comments.
    density : float, optional
        Density of the void-free ply, in the unit of the target model
        (tonne/mm^3 with N, mm and MPa). Each material gets
        ``density * (1 - Vp)``. ``None`` (default) writes no density.
    at : (float, float), optional
        An in-plane point ``(x, y)`` in mm of a field with an
        ``in_plane_map``: the cards describe the laminate there. ``None``
        (default) averages the map.

    Returns
    -------
    str
        The cards, ending with a newline.

    Notes
    -----
    Abaqus takes the compressive limits of ``*FAIL STRESS`` as negative
    numbers and the normalized Tsai-Wu coefficient ``f*``; Nastran takes
    positive limits and the dimensional ``F12 = F*12 / sqrt(Xt Xc Yt Yc)``.
    The thermal expansion coefficients are written when the material sets
    them, at their pristine values.

    Examples
    --------
    >>> from porosity_fe import MATERIALS, material_cards
    >>> text = material_cards(MATERIALS['T800_epoxy'], 0.02)
    >>> '*ELASTIC, TYPE=LAMINA' in text
    True
    """
    if fmt not in _FORMATS:
        raise ValueError(f"fmt must be one of {list(_FORMATS)}; got {fmt!r}.")
    if element not in _ELEMENTS:
        raise ValueError(f"element must be one of {list(_ELEMENTS)}; got {element!r}.")
    if strengths not in _STRENGTHS:
        raise ValueError(
            f"strengths must be one of {list(_STRENGTHS)}; got {strengths!r}.")
    if density is not None and not (np.isfinite(density) and density > 0.0):
        raise ValueError(f"density must be a positive number; got {density!r}.")
    if not isinstance(name, str) or not name or not name[0].isalpha() \
            or not all(c.isalnum() or c in '_-' for c in name):
        raise ValueError(
            f"name must start with a letter and hold letters, digits, '_' "
            f"or '-'; got {name!r}.")
    lam = _laminate(material, porosity, ply_angles, void_shape,
                    porosity_degradation, strengths, at)
    if fmt == 'abaqus':
        lines = _abaqus(material, lam, element=element, strengths=strengths,
                        name=name, section=section, density=density)
    else:
        lines = _nastran(material, lam, element=element, strengths=strengths,
                         first_id=first_id, section=section, density=density)
    return "\n".join(lines) + "\n"


def write_material_cards(filename: str | os.PathLike,
                         material: MaterialProperties,
                         porosity: float | PorosityField,
                         fmt: CardFormat | None = None, **kwargs: Any) -> None:
    """Write :func:`material_cards` to ``filename``.

    Parameters
    ----------
    filename : path
        The file. With ``fmt=None`` its extension gives the format:
        ``.inp`` is Abaqus; ``.bdf``, ``.dat``, ``.nas`` and ``.blk`` are
        Nastran.
    material, porosity, fmt, **kwargs
        As for :func:`material_cards`.
    """
    if fmt is None:
        suffix = os.path.splitext(os.fspath(filename))[1].lower()
        if suffix not in _EXTENSIONS:
            raise ValueError(
                f"Cannot tell the format from the extension {suffix!r}: pass "
                f"fmt='abaqus' or 'nastran', or use one of "
                f"{', '.join(_EXTENSIONS)}.")
        fmt = _EXTENSIONS[suffix]  # type: ignore[assignment]
    text = material_cards(material, porosity, fmt, **kwargs)  # type: ignore[arg-type]
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
