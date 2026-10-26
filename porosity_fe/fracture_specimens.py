"""Delamination fracture specimens: DCB, ENF and MMB geometry, lever and data reduction.

The interlaminar fracture tests of the delamination plan, phase 3:

- the double-cantilever beam (DCB, mode I; ASTM D5528),
- the end-notched flexure beam (ENF, mode II; ASTM D7905),
- the mixed-mode bending specimen (MMB, mode I and II; ASTM D6671).

:meth:`CompositeMesh.for_dcb <porosity_fe.CompositeMesh.for_dcb>`,
:meth:`~porosity_fe.CompositeMesh.for_enf` and
:meth:`~porosity_fe.CompositeMesh.for_mmb` build the specimen meshes, with
the insert as a through-width :class:`~porosity_fe.Delamination` and the
load and support lines on node lines, and record a
:class:`FractureSpecimenGeometry` on the mesh; the ``'dcb'``, ``'enf'`` and
``'mmb'`` loading modes of :meth:`FESolver.solve
<porosity_fe.FESolver.solve>` read it. This module also holds the lever
relations of the MMB test (:func:`mmb_lever_length`, :func:`mmb_mode_ratio`)
and the compliance-based data reduction of the test standards
(:func:`compliance_calibration`), so that FE results can be reduced like
test data.

The ASTM texts were not available while this was written: every default
dimension below is the delamination plan's reading of the standards and
must be checked against the purchased editions. The verification tests use
the specimens of Krueger's NASA benchmark reports instead (see the theory
page "Delamination").

Units: lengths in mm, loads in N, compliance in mm/N, energy release rates
in N/mm (1 N/mm = 1 kJ/m^2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .delamination import Delamination
from .materials import MaterialProperties

if TYPE_CHECKING:  # pragma: no cover
    from .mesh import CompositeMesh
    from .porosity_field import PorosityField

#: The fracture tests, also the names of their loading modes.
FractureTest = Literal['dcb', 'enf', 'mmb']

_FRACTURE_TEST_NAMES: tuple[str, ...] = ('dcb', 'enf', 'mmb')

# Default specimens, as the delamination plan (section 3.1) reads the
# standards. The ASTM texts were not accessed: check every value against
# the purchased editions of D5528, D7905 and D6671.
#: ASTM D5528 DCB: specimen length from the load line (>= 125 mm), width
#: (20 to 25 mm) and insert length a0 from the load line (about 50 mm).
_D5528_LENGTH = 125.0
_D5528_WIDTH = 25.0
_D5528_CRACK_LENGTH = 50.0
#: ASTM D7905 ENF: support span 2L = 100 mm, a0 = 30 mm from the support.
#: The width is not in the plan; 25 mm here.
_D7905_SPAN = 100.0
_D7905_CRACK_LENGTH = 30.0
_D7905_WIDTH = 25.0
#: ASTM D6671 MMB: half-span L = 50 mm (span 100 mm), a0 about 25 mm. The
#: width is not in the plan; 25 mm here.
_D6671_SPAN = 100.0
_D6671_CRACK_LENGTH = 25.0
_D6671_WIDTH = 25.0
#: Default mode ratio G_II / G_T of for_mmb when no lever length is given.
_D6671_MODE_RATIO = 0.5

# In-plane mesh grading towards the crack front (plan decision D5): this
# many elements of exactly the front element length on each side of the
# front, then lengths growing by _GROWTH per element up to the largest.
_FRONT_BAND = 6
_GROWTH = 1.25
# Relative tolerance for a position to lie on a node line.
_ALIGN_TOL = 1e-6


def _finite_positive(name: str, value: object, *, allow_zero: bool = False) -> float:
    """``value`` as a positive (or non-negative) finite float, else a :class:`ValueError`."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    v = float(value)
    if not math.isfinite(v) or v < 0.0 or (v == 0.0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {bound}, got {value!r}.")
    return v


@dataclass(frozen=True)
class FractureSpecimenGeometry:
    """Loads, supports and insert of a delamination fracture specimen (mm).

    Recorded on the mesh by :meth:`CompositeMesh.for_dcb
    <porosity_fe.CompositeMesh.for_dcb>`, :meth:`~porosity_fe.CompositeMesh.for_enf`
    and :meth:`~porosity_fe.CompositeMesh.for_mmb` as
    :attr:`CompositeMesh.fracture_geometry
    <porosity_fe.CompositeMesh.fracture_geometry>`; the loading mode of the
    same name (``test``) of :meth:`FESolver.solve <porosity_fe.FESolver.solve>`
    reads it through :class:`~porosity_fe.BoundaryHandler`. Every load and
    support acts along ``z`` on a node line of the top (``z_max``) or
    bottom (``z_min``) face, spread across the full width with consistent
    nodal weights. With ``P`` the applied load (``applied_load``):

    - ``'dcb'``: hinge loads ``|P|`` on the line ``x = load_end``, ``+z`` on
      the top face (upper arm) and ``-z`` on the bottom face (lower arm),
      equal and opposite, so the specimen is held only by a statically
      determinate set of six constraints at the far end, which carries no
      load.
    - ``'enf'``: three-point bending; roller supports on the bottom face at
      :attr:`support_x` (the first under the cracked end) and ``P`` on the
      top face at :attr:`midspan`.
    - ``'mmb'``: the loads of the MMB lever with the load ``P`` at the
      lever length ``c`` beyond the saddle (see :meth:`load_lines`):
      ``-P c / L`` on the top face at the hinge line ``x = load_end`` (the
      upper arm's hinge) and ``P (c + L) / L`` on the top face at
      :attr:`midspan` (the saddle), on roller supports at :attr:`support_x`
      (the lower arm's hinge and the far roller), ``L`` the half-span.

    Attributes
    ----------
    test : {'dcb', 'enf', 'mmb'}
        The test, and the loading mode that runs it.
    crack_length : float
        Insert length ``a0`` (mm): from the hinge line (DCB, MMB) or the
        support under the cracked end (ENF) to the crack tip, as built.
        Growing the crack by replacing :attr:`CompositeMesh.delaminations
        <porosity_fe.CompositeMesh.delaminations>` does not update it; the
        boundary conditions do not depend on it.
    interface : int
        Ply interface of the insert (between ply ``k`` and ``k + 1`` from
        the bottom).
    load_end : float
        ``x`` (mm) of the line at the cracked end: the DCB hinge line, the
        ENF support under the cracked end, the MMB hinges. The specimen
        starts at ``x = 0``; the insert runs from there to
        :attr:`crack_tip_x`.
    span : float or None
        Support span ``2 L`` (mm) of the ENF and MMB tests; ``None`` for the
        DCB.
    lever_length : float or None
        MMB lever length ``c`` (mm), from the saddle to the lever load;
        ``None`` for the DCB and ENF. ``c = L / 3`` makes the mode I part of
        the beam-theory split zero (see :func:`mmb_mode_ratio`).
    plane_strain : bool
        The specimen in plane strain: ``u_y = 0`` at every node, so no
        section deforms across the width, the counterpart of a
        two-dimensional plane-strain model (the NASA benchmarks of the ENF
        and MMB are such models).

    Raises
    ------
    ValueError
        If ``test`` is unknown, a length is not finite and positive (``c``
        may be 0), the span or lever length is missing for the test or
        given for another, or the crack tip does not lie inside the span.
    """

    test: FractureTest
    crack_length: float
    interface: int
    load_end: float = 0.0
    span: float | None = None
    lever_length: float | None = None
    plane_strain: bool = False

    def __post_init__(self) -> None:
        if self.test not in _FRACTURE_TEST_NAMES:
            raise ValueError(f"FractureSpecimenGeometry.test must be one of "
                             f"{list(_FRACTURE_TEST_NAMES)}, got {self.test!r}.")
        object.__setattr__(self, 'crack_length',
                           _finite_positive('crack_length', self.crack_length))
        object.__setattr__(self, 'load_end',
                           _finite_positive('load_end', self.load_end, allow_zero=True))
        k = self.interface
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 0:
            raise ValueError(f"interface must be an integer >= 0, got {k!r}.")
        object.__setattr__(self, 'interface', int(k))
        object.__setattr__(self, 'plane_strain', bool(self.plane_strain))
        if self.test == 'dcb':
            if self.span is not None or self.lever_length is not None:
                raise ValueError("A 'dcb' specimen has no span or lever_length.")
            return
        if self.span is None:
            raise ValueError(f"A {self.test!r} specimen needs its span (mm).")
        span = _finite_positive('span', self.span)
        object.__setattr__(self, 'span', span)
        if not self.crack_length < span:
            raise ValueError(
                f"The crack tip must lie inside the span: crack_length "
                f"{self.crack_length:g} mm, span {span:g} mm.")
        if self.test == 'enf':
            if self.lever_length is not None:
                raise ValueError("An 'enf' specimen has no lever_length.")
            return
        if self.lever_length is None:
            raise ValueError("An 'mmb' specimen needs its lever_length c (mm).")
        object.__setattr__(self, 'lever_length', _finite_positive(
            'lever_length', self.lever_length, allow_zero=True))

    @property
    def crack_tip_x(self) -> float:
        """``x`` of the crack tip, ``load_end + crack_length`` (mm)."""
        return self.load_end + self.crack_length

    @property
    def half_span(self) -> float | None:
        """Half-span ``L`` (mm) of the ENF and MMB tests; ``None`` for the DCB."""
        return None if self.span is None else 0.5 * self.span

    @property
    def support_x(self) -> tuple[float, float] | None:
        """``x`` of the two roller supports (ENF, MMB), or ``None`` (DCB)."""
        if self.span is None:
            return None
        return (self.load_end, self.load_end + self.span)

    @property
    def midspan(self) -> float | None:
        """``x`` of mid-span (the ENF load, the MMB saddle), or ``None`` (DCB)."""
        if self.span is None:
            return None
        return self.load_end + 0.5 * self.span

    def load_lines(self, load: float) -> tuple[tuple[float, str, float], ...]:
        """The line loads of the applied load ``load`` (N).

        Returns
        -------
        tuple of (x, face, F_z)
            ``x`` of each load line (mm), its face (``'z_max'`` or
            ``'z_min'``) and its total force along ``z`` (N). For the MMB,
            with ``P = load``, the lever length ``c`` and the half-span
            ``L``, the hinge on the upper arm carries ``-P c / L`` and the
            saddle ``P (c + L) / L``; see :meth:`mmb_loads`.
        """
        P = float(load)
        if self.test == 'dcb':
            return ((self.load_end, 'z_max', abs(P)), (self.load_end, 'z_min', -abs(P)))
        assert self.midspan is not None
        if self.test == 'enf':
            return ((self.midspan, 'z_max', P),)
        loads = self.mmb_loads(P)
        return ((self.load_end, 'z_max', loads['hinge']),
                (self.midspan, 'z_max', loads['saddle']))

    def mmb_loads(self, load: float) -> dict[str, float]:
        """Forces of the MMB lever under the lever load ``load`` (N, along ``z``).

        The lever is a rigid bar hinged to the upper arm at the cracked end
        (``x = load_end``), resting on the saddle at mid-span and loaded by
        ``P`` at the lever length ``c`` beyond the saddle. Its equilibrium
        gives the forces it exerts on the specimen; the specimen's gives the
        support reactions (``L`` the half-span, all along ``z``)::

            hinge  = -P c / L            (upper arm, x = load_end)
            saddle =  P (c + L) / L      (top face, mid-span)
            R_end  = -P (L - c) / (2 L)  (lower arm, x = load_end)
            R_far  = -P (c + L) / (2 L)  (far support, x = load_end + 2 L)

        With ``P < 0`` (pressing the lever down) the hinge pulls the upper
        arm up. Reeder and Crews split these into a DCB part, the opening
        pair ``+-P_I`` on the two arms at the hinge line, and an ENF part, a
        mid-span load ``P_II`` whose reaction ``-P_II / 2`` at the cracked
        end the two arms share. As ``z`` components::

            P_I  = -P (3 c - L) / (4 L)   (upper arm; -P_I on the lower)
            P_II =  P (c + L) / L          (the saddle force)

        so the hinge force is ``P_I - P_II / 4`` and the lower arm's support
        reaction ``-P_I - P_II / 4``. Beam theory then gives the mode ratio
        of :func:`mmb_mode_ratio` (``P_I`` vanishes at ``c = L / 3``).

        Returns
        -------
        dict
            ``'hinge'``, ``'saddle'``, ``'reaction_end'``, ``'reaction_far'``,
            ``'mode_I'`` (``P_I``) and ``'mode_II'`` (``P_II``), in N.

        Raises
        ------
        ValueError
            For a specimen other than the MMB.
        """
        if self.test != 'mmb':
            raise ValueError(f"mmb_loads needs an 'mmb' specimen, not {self.test!r}.")
        assert self.lever_length is not None and self.half_span is not None
        P, c, L = float(load), float(self.lever_length), float(self.half_span)
        return {
            'hinge': -P * c / L,
            'saddle': P * (c + L) / L,
            'reaction_end': -P * (L - c) / (2.0 * L),
            'reaction_far': -P * (c + L) / (2.0 * L),
            'mode_I': -P * (3.0 * c - L) / (4.0 * L),
            'mode_II': P * (c + L) / L,
        }

    @property
    def reference_key(self) -> tuple:
        """What the boundary conditions depend on (everything but the
        crack length): the key of the knockdown's pristine reference."""
        return (self.test, self.load_end, self.span, self.lever_length, self.plane_strain)


# ----------------------------------------------------------------------
# MMB lever relations (beam theory)
# ----------------------------------------------------------------------

def _root_rotation_length(material: MaterialProperties, arm_thickness: float) -> float:
    """``chi h``: the crack-length correction of a DCB arm of thickness ``h``.

    ``chi = sqrt(E11 / (11 G13) (3 - 2 (Gamma / (1 + Gamma))^2))`` with
    ``Gamma = 1.18 sqrt(E11 E22) / G13`` (Hashemi, Kinloch and Williams;
    the correction of the DCB checks on the theory page "Delamination").
    """
    E11, E22, G13 = float(material.E11), float(material.E22), float(material.G13)
    gamma = 1.18 * math.sqrt(E11 * E22) / G13
    chi = math.sqrt(E11 / (11.0 * G13) * (3.0 - 2.0 * (gamma / (1.0 + gamma)) ** 2))
    return chi * float(arm_thickness)


def _mmb_beta(material: MaterialProperties, crack_length: float,
              thickness: float | None) -> float:
    """``beta = (a + chi h) / (a + 0.42 chi h)`` of the corrected beam theory."""
    h = 0.5 * (float(material.total_thickness) if thickness is None
               else _finite_positive('thickness', thickness))
    a = _finite_positive('crack_length', crack_length)
    chi_h = _root_rotation_length(material, h)
    return (a + chi_h) / (a + 0.42 * chi_h)


def mmb_mode_ratio(material: MaterialProperties, lever_length: float, *,
                   span: float = _D6671_SPAN, crack_length: float = _D6671_CRACK_LENGTH,
                   thickness: float | None = None) -> float:
    """Mode ratio ``G_II / G_T`` of an MMB specimen from corrected beam theory.

    The lever splits into a DCB part, opening loads of magnitude
    ``P_I = P (3 c - L) / (4 L)``, and an ENF part, a mid-span load
    ``P_II = P (c + L) / L``, with ``P`` the lever load
    (:meth:`FractureSpecimenGeometry.mmb_loads` gives them with signs).
    For a unidirectional specimen with the crack at the mid-plane, arms of
    thickness ``h``, beam theory with the crack-length corrections of the
    DCB (``chi h``) and the ENF (``0.42 chi h``) gives::

        G_I  = 12 P_I^2 (a + chi h)^2 / (b^2 h^3 E11)
        G_II = 9 P_II^2 (a + 0.42 chi h)^2 / (16 b^2 h^3 E11)

        G_I / G_II = (4/3) ((3 c - L) / (c + L))^2 beta^2,
        beta = (a + chi h) / (a + 0.42 chi h)

    With ``beta = 1`` this is Reeder and Crews' relation: its leading
    coefficients, ``36 c^2 - 24 c L + 4 L^2`` in mode I and
    ``3 c^2 + 6 c L + 3 L^2`` in mode II over ``64 L^2 b E11 I``, are
    those of Reeder (1992), NASA TM-104210, Eq. (1). The correction terms
    of that equation are not legible in the copy available here, so the
    corrections are those of the DCB and single-leg checks on the theory
    page "Delamination" (``chi`` from :func:`_root_rotation_length`; the
    mode II factor 0.42 is attributed to Wang and Williams 1992, not
    checked against the paper). For ``c <= L / 3`` the beam-theory mode I
    part is not positive: the hinge closes the crack, and the ratio is 1.

    Parameters
    ----------
    material : MaterialProperties
        Ply constants (``E11``, ``E22``, ``G13``) of the unidirectional arms.
    lever_length : float
        Lever length ``c`` (mm), from the saddle to the lever load.
    span : float
        Support span ``2 L`` (mm), default 100 (D6671, check against the
        standard).
    crack_length : float
        ``a`` (mm), from the hinge line, default 25.
    thickness : float, optional
        Specimen thickness ``2 h`` (mm), default ``material.total_thickness``.

    Returns
    -------
    float
        ``G_II / G_T`` in ``(0, 1]``.

    See Also
    --------
    mmb_lever_length : The inverse.
    """
    c = _finite_positive('lever_length', lever_length, allow_zero=True)
    L = 0.5 * _finite_positive('span', span)
    beta = _mmb_beta(material, crack_length, thickness)
    k = (3.0 * c - L) / (c + L)
    if k <= 0.0:
        return 1.0
    ratio_I_II = 4.0 / 3.0 * (k * beta) ** 2
    return 1.0 / (1.0 + ratio_I_II)


def mmb_lever_length(material: MaterialProperties, mode_ratio: float, *,
                     span: float = _D6671_SPAN, crack_length: float = _D6671_CRACK_LENGTH,
                     thickness: float | None = None) -> float:
    """Lever length ``c`` (mm) of an MMB specimen for a target ``G_II / G_T``.

    The inverse of :func:`mmb_mode_ratio`: with ``r = G_I / G_II =
    (1 - m) / m`` for the target ``m`` and
    ``k = sqrt(3 r / 4) / beta``,

    ``c = L (1 + k) / (3 - k)``, i.e.
    ``c = L (2 beta + sqrt(3 r)) / (6 beta - sqrt(3 r))``,

    which is the form ``(12 beta^2 + 3 r + 8 beta sqrt(3 r)) /
    (36 beta^2 - 3 r) L``. Pure mode II (``m = 1``) gives ``c = L / 3``;
    the mode I limit ``k -> 3`` needs an infinite lever. For IM7/8552
    (NASA/CR-2012-217562 Table I) with ``2 h = 4.5``, ``2 L = 100.8`` and
    ``a = 25.4`` mm it gives 40.9 and 27.0 mm at ``G_II / G_T = 0.5`` and
    0.8, where Krueger's benchmark used 41.3 and 27.3 mm (from the
    experimental study's specimens, Fig. 4 of the report). Check the mode
    mix the FE model gives (:attr:`CrackFront.psi
    <porosity_fe.CrackFront.psi>`): on that benchmark, at Krueger's lever
    lengths, it is 0.99 of this beam theory's in plane strain and 0.96 to
    0.98 at the centre of the three-dimensional specimen.

    Parameters
    ----------
    material : MaterialProperties
        Ply constants of the unidirectional arms.
    mode_ratio : float
        Target ``G_II / G_T`` in ``(0, 1]``.
    span, crack_length, thickness : float
        As :func:`mmb_mode_ratio`.

    Returns
    -------
    float
        ``c`` (mm).

    Raises
    ------
    ValueError
        If ``mode_ratio`` is not in ``(0, 1]`` or is too close to pure mode
        I for a finite lever.
    """
    if isinstance(mode_ratio, bool) or not isinstance(mode_ratio, (int, float, np.number)) \
            or not (0.0 < float(mode_ratio) <= 1.0):
        raise ValueError(f"mode_ratio (G_II / G_T) must lie in (0, 1], got {mode_ratio!r}.")
    m = float(mode_ratio)
    L = 0.5 * _finite_positive('span', span)
    beta = _mmb_beta(material, crack_length, thickness)
    k = math.sqrt(3.0 * (1.0 - m) / (4.0 * m)) / beta
    if k >= 3.0 - 1e-9:
        raise ValueError(
            f"mode_ratio {m:g} is too close to pure mode I for an MMB lever "
            f"(it needs c -> infinity); use the DCB.")
    return L * (1.0 + k) / (3.0 - k)


# ----------------------------------------------------------------------
# Specimen meshes
# ----------------------------------------------------------------------

def _graded_segment(length: float, fine: float, coarse: float, band: int,
                    growth: float) -> list[float]:
    """Element lengths over a segment from its front end: ``band`` of
    exactly ``fine``, then growing by ``growth`` up to ``coarse``, scaled
    after the band to end on the segment's other end."""
    if length <= band * fine * (1.0 + 1e-9):
        n = max(1, round(length / fine))
        return [length / n] * n
    rest = length - band * fine
    tail: list[float] = []
    s = fine
    while sum(tail) < rest * (1.0 - 1e-12):
        s = min(s * growth, coarse)
        tail.append(s)
    if len(tail) > 1 and sum(tail) - rest > 0.5 * tail[-1]:
        tail.pop()
    scale = rest / sum(tail)
    return [fine] * band + [t * scale for t in tail]


def _graded_node_lines(points: Sequence[float], front: int, fine: float,
                      coarse: float, *, band: int = _FRONT_BAND,
                      growth: float = _GROWTH) -> np.ndarray:
    """``x`` of the node lines of a specimen graded towards the crack front.

    Every entry of ``points`` (increasing, the first 0) is a node line. The
    two segments next to ``points[front]`` (the crack tip) start from it
    with ``band`` elements of exactly ``fine`` (fewer, of about ``fine``,
    when the segment is shorter), so the elements either side of the front
    have equal lengths, then grow by ``growth`` per element up to
    ``coarse``; the other segments have uniform elements no longer than
    ``coarse``. Plan decision D5.
    """
    pts = [float(p) for p in points]
    xs = [pts[0]]
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:], strict=True)):
        length = b - a
        if i + 1 == front:
            sizes = _graded_segment(length, fine, coarse, band, growth)[::-1]
        elif i == front:
            sizes = _graded_segment(length, fine, coarse, band, growth)
        else:
            n = max(1, math.ceil(length / coarse - 1e-9))
            sizes = [length / n] * n
        cum = a + np.cumsum(sizes)
        cum[-1] = b
        xs.extend(float(v) for v in cum)
    return np.asarray(xs, dtype=float)


def _aligned_multiple(points: Sequence[float], total: float, limit: int) -> int | None:
    """Smallest element count putting every point on a node line of a
    uniform grid over ``[0, total]`` (``None`` up to ``limit``)."""
    for n in range(1, limit + 1):
        if all(abs(p * n / total - round(p * n / total)) <= _ALIGN_TOL for p in points):
            return n
    return None


@dataclass(frozen=True)
class _SpecimenLayout:
    """What a builder hands to :func:`_build_specimen`."""

    test: str
    points: tuple[float, ...]       # node lines that must exist, 0 first, length last
    front: int                      # index of the crack tip in points
    width: float
    geometry_kwargs: dict[str, Any] = field(default_factory=dict)


def _build_specimen(cls: type[CompositeMesh], porosity_field: PorosityField,
                   material: MaterialProperties, layout: _SpecimenLayout, *,
                   interface: int | None, element_length: float | None,
                   max_element_length: float | None, nx: int | None,
                   ny: int | None, nz: int | None, plane_strain: bool,
                   ply_angles: list[float] | str | None) -> CompositeMesh:
    """Mesh of a fracture specimen (the body of :meth:`CompositeMesh.for_dcb`,
    ``for_enf`` and ``for_mmb``)."""
    caller = f"CompositeMesh.for_{layout.test}"
    h = float(material.total_thickness)
    n_plies = int(material.n_plies)
    nz_f = n_plies if nz is None else nz
    if isinstance(nz_f, bool) or not isinstance(nz_f, (int, np.integer)) or nz_f <= 0:
        raise ValueError(f"{caller}: nz must be a positive integer, got {nz!r}.")
    if nz_f % n_plies:
        raise ValueError(
            f"{caller}: nz must be a multiple of n_plies = {n_plies} so the "
            f"insert's interface is a node layer, got nz = {nz_f}.")
    if interface is None:
        if n_plies % 2:
            raise ValueError(
                f"{caller}: a {n_plies}-ply laminate has no mid-plane interface; "
                f"pass interface= (0 to {n_plies - 2}).")
        interface = n_plies // 2 - 1
    total = layout.points[-1]
    coarse = min(h, 20.0 * h / nz_f) if max_element_length is None else \
        _finite_positive('max_element_length', max_element_length)
    fine = min(0.25 * h, coarse) if element_length is None else \
        _finite_positive('element_length', element_length)
    coarse = max(coarse, fine)
    x_grid: np.ndarray | None
    if nx is not None:
        if isinstance(nx, bool) or not isinstance(nx, (int, np.integer)) or nx <= 0:
            raise ValueError(f"{caller}: nx must be a positive integer, got {nx!r}.")
        if not all(abs(p * nx / total - round(p * nx / total)) <= _ALIGN_TOL
                   for p in layout.points):
            unit = _aligned_multiple(layout.points, total, cls._MAX_ELEMENTS_PER_AXIS)
            hint = (f"use a multiple of {unit}, or leave nx out (element_length=) "
                    f"for the graded mesh" if unit is not None
                    else "leave nx out (element_length=) for the graded mesh")
            positions = ", ".join(f"{p:g}" for p in layout.points)
            raise ValueError(
                f"{caller}: nx={nx} puts a load line, support or the crack tip "
                f"between node lines of the uniform grid (x = {positions} mm on "
                f"a {total:g} mm specimen); {hint}.")
        nx_f = int(nx)
        x_grid = None
    else:
        x_grid = _graded_node_lines(layout.points, layout.front, fine, coarse)
        nx_f = int(x_grid.size - 1)
    if ny is None:
        ny_f = max(2, 2 * math.ceil(layout.width / (2.0 * coarse) - 1e-9))
    else:
        ny_f = ny

    mesh = cls(porosity_field, material, nx=nx_f, ny=ny_f, nz=nz_f, ply_angles=ply_angles)
    mesh.L_x = total
    mesh.L_y = layout.width
    mesh.x_grid = x_grid
    tip = layout.points[layout.front]
    mesh.delaminations = (Delamination.strip(int(interface), 0.0, tip),)
    mesh.generate_mesh()
    mesh.fracture_geometry = FractureSpecimenGeometry(
        test=layout.test,  # type: ignore[arg-type]
        interface=int(interface), plane_strain=plane_strain, **layout.geometry_kwargs)
    return mesh


# ----------------------------------------------------------------------
# Data reduction from compliance
# ----------------------------------------------------------------------

#: Compliance fits of :func:`compliance_calibration`.
ComplianceMethod = Literal['mbt', 'power', 'cubic']


@dataclass(frozen=True)
class ComplianceCalibration:
    """A fit of compliance against crack length, and the ``G`` it implies.

    Built by :func:`compliance_calibration`. Every method gives the energy
    release rate as ``G = P^2 / (2 b) dC/da`` of its fitted ``C(a)``, which
    is how the test standards reduce a measured load, displacement and
    crack length:

    - ``'mbt'`` (ASTM D5528 modified beam theory): ``C^(1/3) = m (a +
      Delta)`` fitted by least squares, so ``G = 3 P delta / (2 b (a +
      Delta))`` with ``delta = C P``; ``Delta`` is the crack-length
      correction, the magnitude of the fitted line's ``x`` intercept;
    - ``'power'`` (ASTM D5528 compliance calibration, Berry's law):
      ``C = a^n / H`` fitted in log-log, ``G = n P delta / (2 b a)``;
    - ``'cubic'`` (ASTM D7905 compliance calibration): ``C = A + m a^3``,
      ``G = 3 m P^2 a^2 / (2 b)``.

    Attributes
    ----------
    method : {'mbt', 'power', 'cubic'}
        The fit.
    width : float
        Specimen width ``b`` (mm).
    crack_lengths, compliances : np.ndarray
        The fitted data (mm, mm/N).
    coefficients : dict
        ``'mbt'``: ``slope`` (``m``, N^(-1/3)) and ``delta`` (``Delta``,
        mm); ``'power'``: ``n`` and ``H``; ``'cubic'``: ``A`` (mm/N) and
        ``m`` (1/(N mm^2)).
    """

    method: ComplianceMethod
    width: float
    crack_lengths: np.ndarray
    compliances: np.ndarray
    coefficients: dict[str, float]

    def compliance(self, crack_length: np.ndarray | float) -> np.ndarray:
        """Fitted compliance ``C(a)`` (mm/N)."""
        a = np.asarray(crack_length, dtype=float)
        c = self.coefficients
        if self.method == 'mbt':
            return (c['slope'] * (a + c['delta'])) ** 3
        if self.method == 'power':
            return a ** c['n'] / c['H']
        return c['A'] + c['m'] * a ** 3

    def compliance_derivative(self, crack_length: np.ndarray | float) -> np.ndarray:
        """``dC/da`` of the fit (1/N)."""
        a = np.asarray(crack_length, dtype=float)
        c = self.coefficients
        if self.method == 'mbt':
            return 3.0 * c['slope'] ** 3 * (a + c['delta']) ** 2
        if self.method == 'power':
            return c['n'] * a ** (c['n'] - 1.0) / c['H']
        return 3.0 * c['m'] * a ** 2

    def energy_release_rate(self, load: np.ndarray | float,
                            crack_length: np.ndarray | float) -> np.ndarray:
        """``G = P^2 / (2 b) dC/da`` (N/mm) at load ``P`` (N) and crack length ``a`` (mm)."""
        P = np.asarray(load, dtype=float)
        return P ** 2 / (2.0 * self.width) * self.compliance_derivative(crack_length)

    def critical_load(self, toughness: float, crack_length: float) -> float:
        """Load (N) at which :meth:`energy_release_rate` reaches ``toughness`` (N/mm)."""
        g = float(self.energy_release_rate(1.0, crack_length))
        return math.sqrt(float(toughness) / g)


def compliance_calibration(crack_lengths: Sequence[float] | np.ndarray,
                           compliances: Sequence[float] | np.ndarray,
                           width: float, *, method: ComplianceMethod
                           ) -> ComplianceCalibration:
    """Fit compliance against crack length the way the test standards do.

    The emulated data reduction of an FE specimen: solve it at several
    crack lengths (one :meth:`CompositeMesh.for_dcb
    <porosity_fe.CompositeMesh.for_dcb>`, ``for_enf`` or ``for_mmb`` mesh
    each), take :attr:`FieldResults.load_point_compliance
    <porosity_fe.FieldResults.load_point_compliance>`, and fit it here, so
    the ``G`` compares like for like with a test report that used the same
    method (see :class:`ComplianceCalibration` for the three fits).

    Parameters
    ----------
    crack_lengths : array_like
        Crack lengths ``a`` (mm), at least two distinct (three or more for a
        meaningful fit), all positive.
    compliances : array_like
        Load-point compliances ``C = delta / P`` (mm/N), positive.
    width : float
        Specimen width ``b`` (mm).
    method : {'mbt', 'power', 'cubic'}
        D5528 modified beam theory, D5528 compliance calibration (Berry's
        power law) or the D7905 cubic compliance calibration.

    Returns
    -------
    ComplianceCalibration

    Raises
    ------
    ValueError
        For an unknown method, mismatched or too few points, or a
        non-positive length, compliance or width.

    Examples
    --------
    >>> cal = compliance_calibration([30, 35, 40], [0.0236, 0.0367, 0.0538],
    ...                              25.0, method='mbt')   # doctest: +SKIP
    >>> cal.energy_release_rate(60.0, 30.0)                 # doctest: +SKIP
    """
    if method not in ('mbt', 'power', 'cubic'):
        raise ValueError(f"method must be 'mbt', 'power' or 'cubic', got {method!r}.")
    a = np.asarray(crack_lengths, dtype=float).ravel()
    C = np.asarray(compliances, dtype=float).ravel()
    b = _finite_positive('width', width)
    if a.size != C.size:
        raise ValueError(
            f"crack_lengths and compliances must have the same length, got "
            f"{a.size} and {C.size}.")
    if np.unique(a).size < 2:
        raise ValueError("compliance_calibration needs at least two distinct crack lengths.")
    if not (np.all(np.isfinite(a)) and np.all(a > 0.0)
            and np.all(np.isfinite(C)) and np.all(C > 0.0)):
        raise ValueError("crack lengths and compliances must be finite and positive.")
    if method == 'mbt':
        slope, intercept = np.polyfit(a, np.cbrt(C), 1)
        coefficients = {'slope': float(slope), 'delta': float(intercept / slope)}
    elif method == 'power':
        n, log_inv_H = np.polyfit(np.log(a), np.log(C), 1)
        coefficients = {'n': float(n), 'H': float(np.exp(-log_inv_H))}
    else:
        m, A = np.polyfit(a ** 3, C, 1)
        coefficients = {'A': float(A), 'm': float(m)}
    return ComplianceCalibration(method=method, width=b, crack_lengths=a,
                                 compliances=C, coefficients=coefficients)
