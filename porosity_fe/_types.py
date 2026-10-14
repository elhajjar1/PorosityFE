"""Closed-set string type aliases (issue #109).

These ``typing.Literal`` aliases name the small, fixed sets of accepted
string values that recur across the public API (loading modes, knockdown
model names, porosity distribution shapes, mesh faces). Promoting the bare
``str`` annotations on the relevant signatures to these aliases lets static
checkers reject typos and out-of-set values at the call site, without any
runtime behavior change — the runtime validation in each module is the
authoritative guard and is unchanged.

Each alias is the single source of truth for its closed set. The exact
members were read off the runtime validation / dispatch in the owning
module (cross-references in the inline comments below); keep them in lock
step with that validation if the accepted set ever changes.
"""

from __future__ import annotations

from typing import Literal

#: Loading modes accepted by :class:`~porosity_fe.empirical.EmpiricalSolver`
#: (``apply_loading`` / ``get_failure_load`` / ``get_all_failure_loads``).
#: Source of truth: ``EmpiricalSolver.PRISTINE_STRENGTH_KEY`` keys.
LoadingMode = Literal[
    'compression', 'tension', 'shear', 'ilss', 'transverse_tension'
]

#: Loading modes accepted by :meth:`~porosity_fe.fe.solver.FESolver.solve`.
#: The four mechanical modes have boundary-condition builders (no
#: ``'transverse_tension'``); ``'thermal'`` is the free-standing thermal /
#: cure residual-stress solve (needs ``delta_T=`` and lamina CTEs). Source
#: of truth: the ``loading`` dispatch in ``FESolver.solve``.
FELoadingMode = Literal['compression', 'tension', 'shear', 'ilss', 'thermal']

#: Hexahedral element formulations accepted by
#: :class:`~porosity_fe.fe.solver.FESolver` (``formulation=``): ``'hex8i'``
#: (the default), the trilinear brick enriched with nine Wilson-Taylor
#: incompatible modes, and ``'hex8'``, the standard fully integrated
#: trilinear brick. Source of truth:
#: ``porosity_fe.fe.element.ELEMENT_FORMULATIONS``.
FEFormulation = Literal['hex8', 'hex8i']

#: Built-in empirical knockdown model names. Source of truth:
#: ``porosity_fe.empirical._KNOCKDOWN_LAWS``.
KnockdownModel = Literal['judd_wright', 'power_law', 'linear']

#: Through-thickness porosity distribution shapes. Source of truth:
#: ``PorosityField._DISTRIBUTIONS``.
Distribution = Literal['uniform', 'clustered', 'interface']

#: Cluster-bump locations for ``distribution='clustered'``. Source of truth:
#: ``PorosityField._CLUSTER_OFFSETS`` keys.
ClusterLocation = Literal['midplane', 'surface', 'quarter']

#: Mesh face identifiers accepted by
#: :meth:`~porosity_fe.mesh.CompositeMesh.nodes_on_face`.
MeshFace = Literal['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
