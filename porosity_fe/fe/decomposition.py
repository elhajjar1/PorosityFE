"""Knockdown decomposition of a porous, delaminated laminate (plan decision D6).

The FE stiffness knockdown of a mesh with porosity and an explicit
:class:`~porosity_fe.Delamination` compares it with the laminate free of
both (the pristine reference of :meth:`FESolver.solve
<porosity_fe.FESolver.solve>`; theory page "Delamination", section
"Knockdown reference"). :func:`knockdown_decomposition` adds the two
single-defect knockdowns against the same reference, so the user sees which
defect drives the loss:

- ``total``: porosity and delamination, the knockdown of the solve itself;
- ``porosity_only``: the same porosity (and discrete voids), every
  duplicated interface pair bonded;
- ``delamination_only``: no porosity or voids, the same delamination.

It needs up to two more linear solves, so :meth:`FESolver.solve
<porosity_fe.FESolver.solve>` never runs it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._types import FEFormulation, FELoadingMode
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..porosity_field import PorosityField
from ._knockdown import _is_delaminated
from .element import _DEFAULT_FORMULATION
from .solver import FESolver

if TYPE_CHECKING:
    from .results import FieldResults


@dataclass(frozen=True)
class KnockdownDecomposition:
    """Stiffness knockdown of a porous, delaminated laminate and its two parts.

    Every knockdown is the structural stiffness of a variant over that of
    the same mesh with no porosity, voids or delamination (the pristine
    reference of :attr:`FieldResults.knockdown
    <porosity_fe.FieldResults.knockdown>`), under the same loading and
    boundary conditions. Built by :func:`knockdown_decomposition`.

    Attributes
    ----------
    loading : str
        The loading mode of the solves.
    total : float
        Knockdown with the porosity and the delamination (the solve of the
        mesh as given).
    porosity_only : float
        Knockdown with the same porosity and discrete voids and every
        duplicated interface pair bonded: no delamination. 1 when the mesh
        has no porosity or void.
    delamination_only : float
        Knockdown with the same delamination and no porosity or void. 1 when
        the mesh has no delaminated node pair.

    Notes
    -----
    The two defects need not combine multiplicatively: ``total`` is in
    general neither ``porosity_only * delamination_only`` nor the smaller of
    the two, since porosity near the crack changes how much stiffness the
    crack costs. All three are dimensionless fractions; a value above 1
    (a variant stiffer than the reference) is not clamped, as for
    :attr:`FieldResults.knockdown <porosity_fe.FieldResults.knockdown>`.
    """

    loading: str
    total: float
    porosity_only: float
    delamination_only: float

    #: Knockdowns closer than this are equal for :attr:`dominant`: the round-off
    #: of a linear solve, e.g. a delamination that costs no stiffness.
    _EQUAL_TOL = 1e-9

    @property
    def dominant(self) -> str | None:
        """``'porosity'`` or ``'delamination'``, the defect whose knockdown
        alone is the lower (the larger stiffness loss); ``None`` when the two
        agree to within ``1e-9`` (for instance neither defect costs any
        stiffness)."""
        if abs(self.porosity_only - self.delamination_only) <= self._EQUAL_TOL:
            return None
        return 'porosity' if self.porosity_only < self.delamination_only \
            else 'delamination'

    def summary(self) -> dict:
        """The decomposition as a JSON-ready dict.

        Keys: ``loading``, ``knockdown_total``, ``knockdown_porosity_only``,
        ``knockdown_delamination_only`` (dimensionless; ``None`` if not
        finite) and ``dominant``.
        """
        def num(v: float) -> float | None:
            v = float(v)
            return v if math.isfinite(v) else None

        return {
            'loading': str(self.loading),
            'knockdown_total': num(self.total),
            'knockdown_porosity_only': num(self.porosity_only),
            'knockdown_delamination_only': num(self.delamination_only),
            'dominant': self.dominant,
        }


def _has_porosity(mesh: CompositeMesh) -> bool:
    """Whether ``mesh`` has a porous node or a discrete void element."""
    porosity = np.asarray(mesh.porosity, dtype=float)
    return bool(np.any(porosity > 0.0)) or np.size(mesh.void_elements) > 0


def _without_delamination(mesh: CompositeMesh) -> CompositeMesh:
    """Shallow copy of ``mesh`` with no delamination: the duplicated
    interface nodes stay, every pair bonded (:meth:`CompositeMesh.bonded_pairs`)."""
    variant = copy.copy(mesh)
    variant.delaminations = ()
    return variant


def _without_porosity(mesh: CompositeMesh) -> CompositeMesh:
    """Shallow copy of ``mesh`` with zero porosity and no void element; its
    delaminations stay."""
    variant = copy.copy(mesh)
    variant.porosity = np.zeros_like(mesh.porosity)
    variant.void_elements = np.zeros(0, dtype=np.intp)
    variant.void_element_set = set()
    return variant


def knockdown_decomposition(
    mesh: CompositeMesh, material: MaterialProperties,
    porosity_field: PorosityField, loading: FELoadingMode = 'compression', *,
    applied_strain: float | None = None,
    applied_load: float = -10.0,
    delta_T: float | None = None,
    allow_unresolved_layup: bool = False,
    crack_contact: bool = True,
    formulation: FEFormulation = _DEFAULT_FORMULATION,
    solver: Literal['direct', 'cg'] = 'direct',
    rtol: float = 1e-9,
    total: FieldResults | None = None,
) -> KnockdownDecomposition:
    """Split the FE knockdown of a porous, delaminated laminate by defect.

    Solves ``mesh`` as given, with its porosity but every interface pair
    bonded, and with its delamination but no porosity, each with
    :meth:`FESolver.solve <porosity_fe.FESolver.solve>` under the same
    loading, and returns the three knockdowns against the one pristine
    reference (plan decision D6). A variant that is the mesh itself (no
    delaminated pair, or no porosity) is not solved again, and a variant
    with neither defect has the knockdown 1 exactly, as a solve gives it.

    Parameters
    ----------
    mesh : CompositeMesh
        The mesh with porosity and delaminations, e.g. ``CompositeMesh(...,
        delaminations=[...])``. Not modified: the variants are shallow
        copies.
    material : MaterialProperties
        The ply material.
    porosity_field : PorosityField
        The porosity field of ``mesh`` (its void shape degrades the
        stiffness).
    loading : str
        Any mechanical loading mode of :meth:`FESolver.solve
        <porosity_fe.FESolver.solve>`; not ``'thermal'``, which has no
        stiffness measure.
    applied_strain, applied_load, delta_T, allow_unresolved_layup, crack_contact, solver, rtol
        As :meth:`FESolver.solve <porosity_fe.FESolver.solve>`, for all
        three solves. The knockdown does not depend on the load magnitude;
        ``delta_T`` and ``crack_contact`` change it only through the set of
        crack-face pairs in contact.
    formulation : {'hex8i', 'hex8'}
        Element formulation, as :class:`~porosity_fe.FESolver`.
    total : FieldResults, optional
        The solve of ``mesh`` with the same arguments, when it has already
        been run: its :attr:`~porosity_fe.FieldResults.knockdown` is used
        instead of solving again.

    Returns
    -------
    KnockdownDecomposition
        ``total``, ``porosity_only`` and ``delamination_only``.

    Raises
    ------
    ValueError
        For ``loading='thermal'``, a ``total`` whose knockdown is not
        finite, or anything :meth:`FESolver.solve
        <porosity_fe.FESolver.solve>` rejects.

    Examples
    --------
    >>> crack = Delamination.ellipse(11, center=(25.0, 10.0), half_size=(6.0, 4.0))
    >>> mesh = CompositeMesh(pf, mat, nx=30, ny=10, nz=24, delaminations=[crack])
    >>> parts = knockdown_decomposition(mesh, mat, pf, 'compression')  # doctest: +SKIP
    >>> parts.total, parts.porosity_only, parts.delamination_only  # doctest: +SKIP
    >>> parts.dominant                       # which defect costs more  # doctest: +SKIP
    """
    if loading == 'thermal':
        raise ValueError(
            "loading='thermal' has no stiffness measure, so no knockdown to "
            "decompose; use a mechanical loading mode.")
    def solve(variant: CompositeMesh) -> float:
        result = FESolver(variant, material, porosity_field,
                          formulation=formulation).solve(
            loading, applied_strain=applied_strain, applied_load=applied_load,
            solver=solver, rtol=rtol, delta_T=delta_T,
            allow_unresolved_layup=allow_unresolved_layup,
            crack_contact=crack_contact, toughness_knockdown=None)
        return float(result.knockdown)

    if total is None:
        kd_total = solve(mesh)
    else:
        kd_total = float(total.knockdown)
        if not math.isfinite(kd_total):
            raise ValueError(
                f"total has no finite knockdown ({kd_total!r}): pass the solve "
                f"of a mechanical loading mode, or leave total out.")
    delaminated = _is_delaminated(mesh)
    porous = _has_porosity(mesh)
    # A variant equal to the mesh itself reuses its knockdown; one with
    # neither defect is pristine, whose knockdown a solve sets to 1 exactly.
    if not delaminated:
        porosity_only = kd_total
    elif porous:
        porosity_only = solve(_without_delamination(mesh))
    else:
        porosity_only = 1.0
    if not porous:
        delamination_only = kd_total
    elif delaminated:
        delamination_only = solve(_without_porosity(mesh))
    else:
        delamination_only = 1.0
    return KnockdownDecomposition(
        loading=str(loading), total=kd_total, porosity_only=porosity_only,
        delamination_only=delamination_only)
