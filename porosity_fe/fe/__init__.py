"""FE subpackage: element, assembler, solver, stress recovery and export."""

from .assembler import BoundaryHandler, GlobalAssembler
from .element import _NODE_COORDS_REF as _NODE_COORDS_REF  # noqa: F401
from .element import Hex8Element
from .export import write_pvd
from .recovery import extrapolate_to_nodes
from .refinement import QuantityConvergence, RefinementLevel, RefinementStudy, refinement_study
from .short_beam import ShortBeamShear, short_beam_shear
from .solver import FESolver, FieldResults
from .vcct import (
    CrackFront,
    critical_energy_release_rate,
    fracture_load_factors,
    vcct_energy_release_rates,
)

__all__ = [
    "BoundaryHandler",
    "CrackFront",
    "FESolver",
    "FieldResults",
    "GlobalAssembler",
    "Hex8Element",
    "QuantityConvergence",
    "RefinementLevel",
    "RefinementStudy",
    "ShortBeamShear",
    "critical_energy_release_rate",
    "extrapolate_to_nodes",
    "fracture_load_factors",
    "refinement_study",
    "short_beam_shear",
    "vcct_energy_release_rates",
    "write_pvd",
]
