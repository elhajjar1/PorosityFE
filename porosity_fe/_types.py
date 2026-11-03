"""Closed-set string type aliases (issue #109).

These ``typing.Literal`` aliases name the small, fixed sets of accepted
string values that recur across the public API (loading modes, knockdown
model names, porosity distribution shapes, mesh faces, the FE element
formulation and porosity degradation). Promoting the bare
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
#: The mechanical modes have boundary-condition builders (no
#: ``'transverse_tension'``): ``'flexure_3pt'`` / ``'flexure_4pt'`` are
#: three- and four-point bending on roller supports (ASTM D790 / D7264);
#: ``'thermal'`` is the free-standing thermal / cure residual-stress solve
#: (needs ``delta_T=`` and lamina CTEs); ``'dcb'``, ``'enf'`` and
#: ``'mmb'`` are the delamination fracture tests (ASTM D5528 / D7905 /
#: D6671) on the specimens of ``CompositeMesh.for_dcb`` / ``for_enf`` /
#: ``for_mmb``. Source of truth:
#: ``porosity_fe.fe._load_cases._FE_LOADINGS`` (the ``loading`` dispatch
#: of ``FESolver.solve``).
FELoadingMode = Literal['compression', 'tension', 'shear', 'ilss',
                        'flexure_3pt', 'flexure_4pt', 'thermal',
                        'dcb', 'enf', 'mmb']

#: Hexahedral element formulations accepted by
#: :class:`~porosity_fe.fe.solver.FESolver` (``formulation=``): ``'hex8i'``
#: (the default), the trilinear brick enriched with nine Wilson-Taylor
#: incompatible modes, and ``'hex8'``, the standard fully integrated
#: trilinear brick. Source of truth:
#: ``porosity_fe.fe.element.ELEMENT_FORMULATIONS``.
FEFormulation = Literal['hex8', 'hex8i']

#: How the FE path turns the local porosity into a degraded ply stiffness
#: and degraded ply strengths (:class:`~porosity_fe.fe.solver.FESolver`,
#: ``porosity_degradation=``): ``'isotropic'`` (the default), which
#: projects the Mori-Tanaka tensor of the porous matrix onto an isotropic
#: one before the Halpin-Tsai ratios and scales every matrix strength by
#: the in-plane ratio ``r_m``, so a void's shape changes the amount of the
#: degradation but not its direction; or ``'directional'``, which keeps
#: the transversely isotropic Mori-Tanaka tensor, degrades each ply
#: constant from its own matrix direction and scales ``Z_t``, ``S_13`` and
#: ``S_23`` by their own stiffness ratios (delamination plan 1.4). Both
#: give the same result for spherical voids. Source of truth:
#: ``porosity_fe.homogenization.POROSITY_DEGRADATIONS``.
PorosityDegradation = Literal['isotropic', 'directional']

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
