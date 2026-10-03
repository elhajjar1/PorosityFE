"""Membrane strain-energy partition of a laminate (layup scaling, IMPROVEMENT_PLAN 2.7).

Private helper for :class:`porosity_fe.EmpiricalSolver`'s layup scaling. Under
a unit in-plane load resultant along one laminate direction, the pristine
CLT membrane strain energy is split into the parts stored by the fiber
(``sigma_11 * eps_11``), transverse (``sigma_22 * eps_22``) and in-plane
shear (``tau_12 * gamma_12``) components in each ply's material axes, summed
over plies and normalised to fractions that add up to 1.

Conventions match :func:`porosity_fe.homogenization._build_clt_abd`:

- pristine ply stiffness from :meth:`MaterialProperties.get_stiffness_matrix`
  (MPa), reduced to plane stress, rotated about ``z`` with
  :func:`~porosity_fe.transforms.rotate_stiffness_3d`;
- engineering shear strain, Voigt order ``[11, 22, 12]`` in plane;
- membrane only: the ``B`` and ``D`` matrices are ignored, so an unsymmetric
  layup gets the partition of its membrane response. Equal ply thicknesses
  are assumed (``t_ply`` cancels out of the fractions).

The partition depends only on the A-matrix and the per-ply stiffness, so it
is invariant to stacking-sequence permutation, continuous in ply angle, and
the same for every in-plane isotropic layup (``[0/±45/90]_s`` and
``[0/±60]_s`` give identical fractions).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .materials import MaterialProperties
from .transforms import rotate_stiffness_3d, strain_transformation_3d, stress_transformation_3d

#: Unit membrane resultant per load direction (``[N_x, N_y, N_xy]``).
_LOAD_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    'x': (1.0, 0.0, 0.0),
    'y': (0.0, 1.0, 0.0),
    'xy': (0.0, 0.0, 1.0),
}

_IN_PLANE = np.array([0, 1, 5])  # 11, 22, 12 in 6-component Voigt order


def _plane_stress_reduce(C: np.ndarray) -> np.ndarray:
    """Plane-stress reduced 3x3 stiffness ``[11, 22, 12]`` from a 6x6 ``C``."""
    Q = C[np.ix_(_IN_PLANE, _IN_PLANE)].copy()
    if abs(C[2, 2]) > 1e-12:
        Q -= np.outer(C[_IN_PLANE, 2], C[_IN_PLANE, 2]) / C[2, 2]
    return Q


@lru_cache(maxsize=1024)
def _partition_cached(C_flat: tuple[float, ...], ply_angles: tuple[float, ...],
                      direction: str) -> tuple[float, float, float]:
    C = np.asarray(C_flat, dtype=float).reshape(6, 6)
    q_bars = []
    for angle_deg in ply_angles:
        angle_rad = float(np.radians(angle_deg))
        C_rot = rotate_stiffness_3d(C, angle_rad, axis='z') if abs(angle_rad) > 1e-15 else C
        q_bars.append(_plane_stress_reduce(C_rot))
    A = np.sum(q_bars, axis=0)  # per unit ply thickness; t_ply cancels
    eps0 = np.linalg.solve(A, np.asarray(_LOAD_DIRECTIONS[direction]))
    energy = np.zeros(3)
    for angle_deg, q_bar in zip(ply_angles, q_bars, strict=True):
        angle_rad = float(np.radians(angle_deg))
        T_sig = stress_transformation_3d(angle_rad)[np.ix_(_IN_PLANE, _IN_PLANE)]
        T_eps = strain_transformation_3d(angle_rad)[np.ix_(_IN_PLANE, _IN_PLANE)]
        sig_material = T_sig @ (q_bar @ eps0)
        eps_material = T_eps @ eps0
        energy += sig_material * eps_material
    total = float(np.sum(energy))
    e1, e2, e6 = (float(v) / total for v in energy)
    return e1, e2, e6


def _membrane_energy_partition(material: MaterialProperties,
                               ply_angles: list[float] | tuple[float, ...],
                               direction: str = 'x') -> tuple[float, float, float]:
    """Fiber / transverse / shear fractions of the pristine CLT membrane energy.

    Parameters
    ----------
    material : MaterialProperties
        Ply material; only the pristine stiffness (MPa) is used.
    ply_angles : sequence of float
        Explicit ply angles in degrees (sentinels already resolved).
    direction : {'x', 'y', 'xy'}
        Laminate direction of the unit membrane resultant.

    Returns
    -------
    tuple of float
        ``(e1, e2, e6)``: the fractions of the strain energy stored as
        ``sigma_11*eps_11``, ``sigma_22*eps_22`` and ``tau_12*gamma_12`` in
        the ply material axes. They sum to 1. Each is non-negative for the
        usual layups; Poisson coupling can make one slightly negative (a few
        tenths of a percent) for a single off-axis ply.

    Notes
    -----
    The result is cached on the stiffness values, the angle tuple and the
    direction, because the validation runner builds several hundred solvers
    on a handful of layups.
    """
    if direction not in _LOAD_DIRECTIONS:
        raise ValueError(
            f"Unknown load direction {direction!r}. "
            f"Use one of {sorted(_LOAD_DIRECTIONS)}."
        )
    if len(ply_angles) == 0:
        raise ValueError("ply_angles must contain at least one ply.")
    C = np.asarray(material.get_stiffness_matrix(), dtype=float)
    return _partition_cached(tuple(C.ravel().tolist()),
                             tuple(float(a) for a in ply_angles), direction)
