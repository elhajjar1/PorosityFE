"""Hybrid laminates: the material and thickness of every ply.

A :class:`PlyStack` lists the ply material and ply thickness of a laminate,
bottom (``z = 0``) to top. It stands in for the single
:class:`~porosity_fe.MaterialProperties` (whose ``t_ply`` and ``n_plies``
otherwise give every ply the same material and thickness) wherever the FE
path takes a material: :class:`~porosity_fe.PorosityField`,
:class:`~porosity_fe.CompositeMesh` and :class:`~porosity_fe.FESolver`. The
ply angles stay separate (``ply_angles=``, tiled to the stack's ply count),
as they are for a single material.

Each ply keeps its own stiffness, constituents (the porosity
micromechanics), strengths, CTEs and toughness. A material's own
``n_plies`` is ignored in a stack and its ``t_ply`` is the default thickness
of its plies.

The empirical solver takes a stack of one material (its plies may differ in
thickness: the layup scaling weights them by thickness) and refuses a stack
of several, whose laminate strength the empirical laws do not define.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, replace

import numpy as np

from ._ply_angles import _resolve_ply_angles
from .materials import MaterialProperties

__all__ = ['PlyStack', 'parse_ply_stack']

#: Relative tolerance under which two ply thicknesses count as equal.
_THICKNESS_RTOL = 1e-12

_BLOCK = re.compile(
    r"^\s*(?P<name>[^*@,\s](?:[^*@,]*[^*@,\s])?)\s*"
    r"(?:\*\s*(?P<count>[0-9]+)\s*)?"
    r"(?:@\s*(?P<t>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*)?$")


def _positive_thickness(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{where} must be a number (mm), got {value!r}.")
    t = float(value)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"{where} must be a finite thickness > 0 mm, got {value!r}.")
    return t


class PlyStack:
    """The material and thickness of every ply of a laminate, bottom to top.

    Parameters
    ----------
    materials : sequence of MaterialProperties
        One material per ply, from the bottom (``z = 0``) up. A material's
        own ``n_plies`` is ignored.
    thicknesses : float or sequence of float, optional
        Ply thicknesses (mm): one per ply, or one for every ply. ``None``
        (default) gives each ply its material's ``t_ply``.

    Notes
    -----
    The stack replaces the material where the FE path takes one. The mesh
    puts every interface on a node layer when ``nz`` is a multiple of the
    ply count (each ply then has ``nz / n_plies`` element layers of its
    own thickness), and refuses an element layer that would span plies of
    different materials. :attr:`reference` is the laminate-level material
    the code reads the ply count and total thickness from (the first ply's
    material with ``n_plies`` and the mean ``t_ply`` of the stack): every
    per-ply property is read from :attr:`materials`.

    Examples
    --------
    >>> from porosity_fe import MATERIALS
    >>> carbon, glass = MATERIALS['T700_epoxy'], MATERIALS['glass_epoxy']
    >>> stack = PlyStack([glass] + [carbon] * 4 + [glass], thicknesses=0.15)
    >>> stack.n_plies, round(stack.total_thickness, 6), stack.is_hybrid
    (6, 0.9, True)
    >>> stack.ply_material_ids.tolist()
    [0, 1, 1, 1, 1, 0]
    >>> stack
    PlyStack(6 plies, 0.9 mm: glass_epoxy@0.15, T700_epoxy*4@0.15, glass_epoxy@0.15)
    """

    __slots__ = ('_materials', '_ids', '_t', '_reference')

    def __init__(self, materials: Sequence[MaterialProperties],
                 thicknesses: float | Sequence[float] | np.ndarray | None = None):
        if isinstance(materials, (MaterialProperties, str)) or not isinstance(
                materials, Sequence):
            raise TypeError(
                "PlyStack.materials must be a sequence of MaterialProperties, "
                f"one per ply (bottom to top); got {type(materials).__name__}.")
        plies = list(materials)
        if not plies:
            raise ValueError("PlyStack needs at least one ply.")
        unique: list[MaterialProperties] = []
        ids: list[int] = []
        for k, m in enumerate(plies):
            if not isinstance(m, MaterialProperties):
                raise TypeError(
                    f"PlyStack ply {k} is a {type(m).__name__}, not a "
                    f"MaterialProperties.")
            for i, u in enumerate(unique):
                if u is m or u == m:
                    ids.append(i)
                    break
            else:
                ids.append(len(unique))
                unique.append(m)
        n = len(plies)
        if thicknesses is None:
            t = [float(m.t_ply) for m in plies]
        elif isinstance(thicknesses, (bool, int, float, np.integer, np.floating)):
            t = [_positive_thickness(thicknesses, "PlyStack.thicknesses")] * n
        else:
            values = list(np.asarray(thicknesses, dtype=object).ravel()) \
                if isinstance(thicknesses, np.ndarray) else list(thicknesses)
            if len(values) != n:
                raise ValueError(
                    f"PlyStack.thicknesses has {len(values)} values for {n} "
                    f"plies; give one per ply or a single thickness.")
            t = [_positive_thickness(v, f"PlyStack.thicknesses[{k}]")
                 for k, v in enumerate(values)]
        self._materials = tuple(unique)
        self._ids = np.asarray(ids, dtype=np.intp)
        self._t = np.asarray(t, dtype=float)
        total = float(np.sum(self._t))
        self._reference = replace(unique[0], n_plies=n, t_ply=total / n)

    # ------------------------------------------------------------------
    # The stack
    # ------------------------------------------------------------------

    @property
    def materials(self) -> tuple[MaterialProperties, ...]:
        """The distinct ply materials, in the order they first appear."""
        return self._materials

    @property
    def ply_material_ids(self) -> np.ndarray:
        """``(n_plies,)`` index of each ply's material in :attr:`materials`."""
        return self._ids.copy()

    @property
    def ply_materials(self) -> tuple[MaterialProperties, ...]:
        """The material of each ply, bottom to top."""
        return tuple(self._materials[i] for i in self._ids)

    @property
    def thicknesses(self) -> np.ndarray:
        """``(n_plies,)`` ply thicknesses (mm), bottom to top."""
        return self._t.copy()

    @property
    def n_plies(self) -> int:
        """Number of plies."""
        return int(self._t.size)

    @property
    def total_thickness(self) -> float:
        """Laminate thickness (mm), the sum of the ply thicknesses."""
        return float(np.sum(self._t))

    @property
    def ply_bounds(self) -> np.ndarray:
        """``(n_plies + 1,)`` ply boundaries ``z`` (mm), ``0`` to the top."""
        bounds = np.concatenate(([0.0], np.cumsum(self._t)))
        bounds[-1] = self.total_thickness
        return bounds

    @property
    def interface_z(self) -> np.ndarray:
        """``(n_plies - 1,)`` height of each ply interface (mm); interface
        ``k`` is between plies ``k`` and ``k + 1``."""
        return self.ply_bounds[1:-1].copy()

    @property
    def is_hybrid(self) -> bool:
        """Whether the plies are of more than one material."""
        return len(self._materials) > 1

    @property
    def uniform_thickness(self) -> bool:
        """Whether every ply has the same thickness."""
        t0 = float(self._t[0])
        return bool(np.all(np.abs(self._t - t0) <= _THICKNESS_RTOL * t0))

    @property
    def reference(self) -> MaterialProperties:
        """Laminate-level material: the first ply's material with the stack's
        ``n_plies`` and mean ``t_ply`` (so ``total_thickness`` is the
        stack's). Read only the ply count and thickness from it."""
        return self._reference

    def interface_materials(self, interface: int) -> tuple[MaterialProperties, MaterialProperties]:
        """The materials of the plies below and above ``interface``."""
        if not 0 <= int(interface) <= self.n_plies - 2:
            raise ValueError(
                f"interface must be in [0, {self.n_plies - 2}] for a "
                f"{self.n_plies}-ply stack, got {interface!r}.")
        k = int(interface)
        return self._materials[self._ids[k]], self._materials[self._ids[k + 1]]

    def key(self) -> tuple:
        """Hashable value of the stack (every material field, the
        material of each ply and the thicknesses), for caches."""
        return (tuple(astuple(m) for m in self._materials),
                tuple(self._ids.tolist()), tuple(self._t.tolist()))

    def to_dict(self, names: Mapping[str, MaterialProperties] | None = None) -> dict:
        """JSON-ready description: the thickness and material index of each
        ply, and each distinct material (its name in ``names`` when it is
        one of them, and every field)."""
        def name_of(m: MaterialProperties) -> str | None:
            for name, value in (names or {}).items():
                if value == m:
                    return name
            return None
        return {
            'n_plies': self.n_plies,
            'total_thickness_mm': self.total_thickness,
            'thickness_mm': self._t.tolist(),
            'material_index': self._ids.tolist(),
            'materials': [{'name': name_of(m), **asdict(m)} for m in self._materials],
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlyStack):
            return NotImplemented
        return (self._materials == other._materials
                and np.array_equal(self._ids, other._ids)
                and np.array_equal(self._t, other._t))

    def __hash__(self) -> int:
        return hash(self.key())

    def __len__(self) -> int:
        return self.n_plies

    @property
    def spec(self) -> str:
        """The stack as :func:`parse_ply_stack` reads it, bottom to top:
        blocks ``NAME*COUNT@THICKNESS`` of consecutive like plies, ``NAME``
        the preset name of a :data:`~porosity_fe.MATERIALS` material or
        ``m<index>`` (into :attr:`materials`) for a custom one."""
        from .materials import MATERIALS
        names = []
        for i, m in enumerate(self._materials):
            names.append(next((name for name, preset in MATERIALS.items() if preset == m),
                              f"m{i}"))
        blocks: list[str] = []
        k = 0
        n = self.n_plies
        while k < n:
            j = k
            while j + 1 < n and self._ids[j + 1] == self._ids[k] \
                    and self._t[j + 1] == self._t[k]:
                j += 1
            count = j - k + 1
            blocks.append(names[int(self._ids[k])]
                          + (f"*{count}" if count > 1 else "")
                          + f"@{self._t[k]:g}")
            k = j + 1
        return ", ".join(blocks)

    def __repr__(self) -> str:
        return (f"PlyStack({self.n_plies} plies, {self.total_thickness:g} mm: "
                f"{self.spec})")


def parse_ply_stack(spec: str, materials: Mapping[str, MaterialProperties]
                    ) -> PlyStack:
    """A :class:`PlyStack` from a text description, bottom to top.

    ``spec`` is comma-separated blocks ``NAME[*COUNT][@THICKNESS]``:
    ``COUNT`` plies (default 1) of the material ``materials[NAME]`` (a name
    may hold spaces, not ``*``, ``@`` or ``,``), each
    ``THICKNESS`` mm thick (default the material's ``t_ply``). The command
    line's ``--plies`` and the app use it.

    Examples
    --------
    >>> from porosity_fe import MATERIALS
    >>> stack = parse_ply_stack("glass_epoxy@0.2, T700_epoxy*4, glass_epoxy@0.2",
    ...                         MATERIALS)
    >>> stack.n_plies, stack.ply_material_ids.tolist()
    (6, [0, 1, 1, 1, 1, 0])
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            "The ply stack is empty: give comma-separated blocks "
            "NAME[*COUNT][@THICKNESS], e.g. 'glass_epoxy@0.2,T700_epoxy*4'.")
    plies: list[MaterialProperties] = []
    thicknesses: list[float] = []
    for block in spec.split(','):
        match = _BLOCK.match(block)
        if match is None:
            raise ValueError(
                f"Cannot read the ply block {block.strip()!r}: use "
                f"NAME[*COUNT][@THICKNESS], e.g. 'T700_epoxy*4@0.125'.")
        name = match['name']
        if name not in materials:
            raise ValueError(
                f"Unknown material {name!r} in the ply stack. Use one of "
                f"{sorted(materials)}.")
        count = int(match['count']) if match['count'] is not None else 1
        if count < 1:
            raise ValueError(f"The ply count of block {block.strip()!r} must be >= 1.")
        material = materials[name]
        t = float(match['t']) if match['t'] is not None else float(material.t_ply)
        plies += [material] * count
        thicknesses += [t] * count
    return PlyStack(plies, thicknesses)


def stack_angles(stack: PlyStack, ply_angles: Sequence[float] | str) -> list[float]:
    """The angle of each ply of ``stack``: ``ply_angles`` (a list or the
    ``'QI'`` / ``'UD'`` sentinel) tiled to the stack's plies, as the mesh
    tiles them; a list longer than the stack is refused."""
    angles = _resolve_ply_angles(ply_angles, caller='ply_angles')
    assert angles is not None
    n = stack.n_plies
    if not isinstance(ply_angles, str) and len(angles) > n:
        raise ValueError(
            f"The layup has {len(angles)} plies but the stack has {n}; an "
            f"explicit layup must not be longer than the stack.")
    return [float(angles[k % len(angles)]) for k in range(n)]


def element_materials(mesh, material: MaterialProperties | PlyStack
                      ) -> tuple[tuple[MaterialProperties, ...], np.ndarray]:
    """The distinct materials of ``mesh``'s elements and each element's index.

    A mesh without a ply stack is one material, ``material`` (a
    :class:`PlyStack` is refused); a mesh with one takes its stack's, and
    ``material`` must be that stack or its :attr:`PlyStack.reference`.
    """
    stack = getattr(mesh, 'ply_stack', None)
    n_elem = len(mesh.elements)
    if stack is None:
        if isinstance(material, PlyStack):
            raise ValueError(
                "A PlyStack was given for a mesh built from a single material; "
                "build the PorosityField and CompositeMesh from the PlyStack.")
        return (material,), np.zeros(n_elem, dtype=np.intp)
    if isinstance(material, PlyStack):
        if material != stack:
            raise ValueError(
                "The PlyStack differs from the mesh's; pass the mesh's PlyStack.")
    elif material is not stack.reference and material != stack.reference:
        raise ValueError(
            "The mesh was built from a PlyStack: pass that PlyStack (or "
            "mesh.ply_stack), not a single material.")
    return stack.materials, np.asarray(mesh.elem_material_ids, dtype=np.intp)


def material_key(material: MaterialProperties | PlyStack) -> tuple:
    """Hashable value of a material or a ply stack, for caches."""
    if isinstance(material, PlyStack):
        return ('ply_stack',) + material.key()
    return astuple(material)


def as_material(material: MaterialProperties | PlyStack
                ) -> tuple[MaterialProperties, PlyStack | None]:
    """``(laminate-level material, stack or None)`` of a material argument."""
    if isinstance(material, PlyStack):
        return material.reference, material
    return material, None
