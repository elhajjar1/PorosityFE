"""Frictionless crack-face contact on a delaminated interface (active set).

A delaminated (untied) node pair of a :class:`~porosity_fe.Delamination` has
two free crack faces, which a linear solve lets interpenetrate under
bending, compression or shear (plan decision D7). This module enforces
frictionless contact with zero initial gap by a primal-dual active set on
the delaminated pairs:

1. solve with the current active set, each active pair held by a
   normal-only tie (``u_z`` of one face equal to the other's; the faces
   slide freely);
2. add every inactive pair whose normal opening
   ``u_z(duplicate) - u_z(original)`` (the interface normal is ``+z``) is
   negative, and release every active pair whose normal contact force is
   tensile;
3. repeat until the set does not change, at most ``max_iterations`` solves.

Only the ties change between iterations, never ``K``; the reduced system
does, so every new active set is factorized again. Split out of the
solver; :meth:`FESolver.solve <porosity_fe.fe.solver.FESolver.solve>`
supplies the solve and holds the factorizations (see its
``_contact_solve``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..mesh import CompositeMesh
from .assembler import Ties

logger = logging.getLogger("porosity_fe_analysis")


@dataclass(frozen=True)
class ContactCandidates:
    """The delaminated pairs that can come into contact, and their normal tie.

    ``pair`` indexes :meth:`CompositeMesh.interface_node_pairs`. The tie is
    ``u[slave] = u[master]`` on the ``z`` DOFs: the duplicate follows the
    original, or the other way round when the duplicate's ``u_z`` is
    prescribed (a pair with both prescribed cannot be tied and is left
    out). ``sign`` turns the residual ``K u - F`` of the slave into the
    normal contact force, compressive positive.
    """

    pair: np.ndarray
    dup_z: np.ndarray
    orig_z: np.ndarray
    slave: np.ndarray
    master: np.ndarray
    sign: np.ndarray


@dataclass(frozen=True)
class ContactState:
    """Outcome of the active-set loop.

    Attributes
    ----------
    active : np.ndarray
        Boolean, one entry per duplicated pair: the pair is in contact.
    iterations : int
        Linear solves of the loop (1 when no pair comes into contact).
    converged : bool
        Whether the active set stopped changing within the iteration cap.
    ties : Ties or None
        The ties of the last solve: the bonded ties plus the normal ties of
        the active pairs.
    """

    active: np.ndarray
    iterations: int
    converged: bool
    ties: Ties | None

    @property
    def n_pairs(self) -> int:
        """Pairs in contact."""
        return int(np.count_nonzero(self.active))


def contact_candidates(mesh: CompositeMesh, constrained: dict[int, float]
                       ) -> ContactCandidates:
    """Delaminated pairs and the normal tie each would take in contact."""
    originals, duplicates = mesh.interface_node_pairs()
    open_ = ~mesh.bonded_pairs()
    dup_z = 3 * duplicates + 2
    orig_z = 3 * originals + 2
    fixed = np.zeros(3 * np.asarray(mesh.nodes).shape[0], dtype=bool)
    if constrained:
        fixed[np.fromiter(constrained.keys(), dtype=np.intp,
                          count=len(constrained))] = True
    d_fixed, o_fixed = fixed[dup_z], fixed[orig_z]
    pair = np.flatnonzero(open_ & ~(d_fixed & o_fixed))
    swap = d_fixed[pair]
    return ContactCandidates(
        pair=pair, dup_z=dup_z[pair], orig_z=orig_z[pair],
        slave=np.where(swap, orig_z[pair], dup_z[pair]),
        master=np.where(swap, dup_z[pair], orig_z[pair]),
        sign=np.where(swap, -1.0, 1.0))


def merge_ties(base: Ties | None, slave: np.ndarray, master: np.ndarray
               ) -> Ties | None:
    """``base`` plus the zero-offset ties ``u[slave] = u[master]``."""
    if slave.size == 0:
        return base
    if base is None:
        return (slave.copy(), master.copy(), np.zeros(slave.size, dtype=float))
    return (np.concatenate([base[0], slave]), np.concatenate([base[1], master]),
            np.concatenate([base[2], np.zeros(slave.size, dtype=float)]))


def solve_with_contact(
    solve: Callable[[Ties | None, bool], tuple[np.ndarray, np.ndarray]],
    mesh: CompositeMesh, constrained: dict[int, float], base_ties: Ties | None,
    *, max_iterations: int, rel_tol: float = 1e-9,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, ContactState]:
    """Frictionless contact of the delaminated pairs by a primal-dual active set.

    Parameters
    ----------
    solve : callable
        ``solve(ties, with_contact) -> (u, r)``: the full displacement and
        the residual ``K u - F`` (tie forces not folded) of the state whose
        contact is enforced, with ``ties`` the bonded ties plus the normal
        ties of the active pairs; ``with_contact`` is false when no pair is
        active (only the bonded ties).
    mesh : CompositeMesh
        The delaminated mesh.
    constrained : dict
        The prescribed DOFs of the solve (a pair whose two ``u_z`` are both
        prescribed is left out).
    base_ties : Ties or None
        The bonded ties.
    max_iterations : int
        Cap on the linear solves; reaching it with a set still changing
        logs a warning and returns the last solve.
    rel_tol : float
        A pair penetrates when its opening is below ``-rel_tol max|u|``, and
        a contact force is tensile below ``-rel_tol max|r|``, so round-off
        does not flip pairs.
    initial : np.ndarray, optional
        Boolean per duplicated pair: the active set to start from (a warm
        start, e.g. the set of a previous solve of the same load case);
        default none active. The result does not depend on it, only the
        number of solves.

    Returns
    -------
    u, r : np.ndarray
        The last solve.
    state : ContactState
        Active set, iterations, convergence and ties.
    """
    cand = contact_candidates(mesh, constrained)
    n_pairs = mesh.interface_node_pairs()[0].size
    active = np.zeros(cand.pair.size, dtype=bool)
    if initial is not None and np.size(initial) == n_pairs:
        active = np.asarray(initial, dtype=bool)[cand.pair]
    ties = base_ties
    u = r = np.zeros(0)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        ties = merge_ties(base_ties, cand.slave[active], cand.master[active])
        u, r = solve(ties, bool(active.any()))
        if cand.pair.size == 0:
            converged = True
            break
        u_scale = float(np.max(np.abs(u))) if u.size else 0.0
        r_scale = float(np.max(np.abs(r))) if r.size else 0.0
        opening = u[cand.dup_z] - u[cand.orig_z]
        pressure = cand.sign * r[cand.slave]
        add = ~active & (opening < -rel_tol * u_scale)
        release = active & (pressure < -rel_tol * r_scale)
        if not (add.any() or release.any()):
            converged = True
            break
        logger.debug("Crack-face contact, solve %d: %d pairs active, +%d "
                     "penetrating, -%d tensile.", iterations,
                     int(active.sum()), int(add.sum()), int(release.sum()))
        if iterations == max_iterations:
            break
        active = (active & ~release) | add
    if not converged:
        logger.warning(
            "Crack-face contact did not settle in %d linear solves: the "
            "active set still changes (%d pairs in contact). The result "
            "uses the last set, so a few crack-face pairs may penetrate or "
            "carry a tensile contact force.", max_iterations, int(active.sum()))
    full = np.zeros(n_pairs, dtype=bool)
    full[cand.pair[active]] = True
    return u, r, ContactState(active=full, iterations=iterations,
                              converged=converged, ties=ties)
