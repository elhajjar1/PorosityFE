"""Constrained linear solve of the FE system, and its factorization cache.

Imposes the Dirichlet BCs exactly (``_ConstraintReduction`` in
:mod:`porosity_fe.fe.assembler`) and solves the reduced SPD system
``K_ff u_f = F_f - K_fc u_c`` by sparse LU (cached per constrained-DOF set)
or Jacobi-preconditioned CG (MINRES is deprecated). Split out of
``fe/solver.py`` (IMPROVEMENT_PLAN 5.1); :class:`FESolver
<porosity_fe.fe.solver.FESolver>` keeps ``_solve_constrained`` and
``_free_dof_system`` as thin wrappers that pass its own cache.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from . import _memory
from .assembler import Ties, _ConstraintReduction

logger = logging.getLogger("porosity_fe_analysis")

#: ``system(K, part, factorize=...) -> (K_ff, lu or None)``, e.g. a bound
#: :meth:`FESolver._free_dof_system <porosity_fe.fe.solver.FESolver._free_dof_system>`.
FreeDofSystem = Callable[..., tuple]


def _relative_residual(A: scipy.sparse.spmatrix, x: np.ndarray,
                       b: np.ndarray) -> float:
    """``||A x - b|| / ||b||``; the absolute residual when ``b = 0``."""
    r = float(np.linalg.norm(A @ x - b))
    norm_b = float(np.linalg.norm(b))
    return r / norm_b if norm_b > 0.0 else r


#: Normwise backward error at which an iterate is accepted even above the
#: residual tolerance: ``x`` then solves a system within about 1000 machine
#: epsilons of ``A x = b``, as well as a direct solve can.
_BACKWARD_ERROR_TOL = 1e3 * float(np.finfo(float).eps)


def _backward_error(A: scipy.sparse.spmatrix, x: np.ndarray,
                    b: np.ndarray) -> float:
    """Normwise backward error ``||A x - b|| / (||A|| ||x|| + ||b||)``
    (infinity norms): the smallest relative perturbation of ``A`` and ``b``
    that ``x`` solves exactly."""
    r = float(np.linalg.norm(A @ x - b, np.inf))
    denom = (float(scipy.sparse.linalg.norm(A, np.inf)) * float(np.linalg.norm(x, np.inf))
             + float(np.linalg.norm(b, np.inf)))
    return r / denom if denom > 0.0 else r


def _iterative_solve(K_ff: scipy.sparse.spmatrix, rhs: np.ndarray,
                     diag: np.ndarray, *, solver: str, rtol: float,
                     max_restarts: int = 3) -> tuple[np.ndarray, float]:
    """Jacobi-preconditioned CG or (deprecated) MINRES on the reduced SPD system.

    A result is accepted when the true relative residual
    ``||K_ff u - rhs|| / ||rhs||`` is at most ``10 * rtol`` (the slack
    absorbs the floating-point floor). SciPy's CG stops on its recurrence
    residual, which tracks the true one, so one call lands at about
    ``rtol``. SciPy's MINRES stops on ``||r||_M / (||A|| ||x||)``, an
    estimate in the preconditioner norm that sits a roughly constant factor
    (10x to 1000x on these meshes) below the true relative residual. So
    while the true residual is above ``rtol``, MINRES is warm-started from
    its last iterate with the internal tolerance scaled by
    ``rtol / achieved``, at most ``max_restarts`` times (one restart
    suffices on the shipped load cases). MINRES minimizes the residual,
    not the energy-norm error, so at the same residual its displacement
    error is larger than CG's; it is deprecated (:meth:`FESolver.solve`
    warns) and stays here until its removal in 2.0.

    Raises
    ------
    RuntimeError
        On a non-positive diagonal (no Jacobi preconditioner), or when the
        solver stops without meeting the tolerance.
    """
    if not np.all(diag > 0.0):
        raise RuntimeError(
            "Cannot build Jacobi preconditioner: the free-DOF stiffness has "
            "a non-positive diagonal entry. Check the assembly.")
    M = scipy.sparse.diags(1.0 / diag)
    maxiter = max(10 * rhs.size, 100)
    accept = 10.0 * rtol
    restarts = 0
    if solver == 'cg':
        u_f, info = scipy.sparse.linalg.cg(
            K_ff, rhs, M=M, rtol=rtol, maxiter=maxiter)
        rel_res = _relative_residual(K_ff, u_f, rhs)
    else:  # solver == 'minres'
        eps = float(np.finfo(float).eps)
        inner = rtol
        x0: np.ndarray | None = None
        while True:
            u_f, info = scipy.sparse.linalg.minres(
                K_ff, rhs, x0=x0, M=M, rtol=inner, maxiter=maxiter)
            rel_res = _relative_residual(K_ff, u_f, rhs)
            if info != 0 or rel_res <= rtol or restarts >= max_restarts:
                break
            # Below eps the estimate cannot tighten further, but a warm
            # restart from the better iterate can still make progress.
            inner = max(inner * rtol / rel_res, eps)
            x0 = u_f
            restarts += 1
    # The relative residual cannot always reach rtol: on a stiff open
    # specimen (the DCB arms) ||K|| ||u|| is far above ||rhs||, and even the
    # exact LU solution's residual sits above 1e-9 from round-off alone. A
    # converged iterate whose backward error is at round-off is as accurate
    # as a direct solve, and is accepted, unless the caller asked for a
    # tolerance below round-off itself, which still raises.
    if info == 0 and rel_res > accept and rtol >= _BACKWARD_ERROR_TOL \
            and _backward_error(K_ff, u_f, rhs) <= _BACKWARD_ERROR_TOL:
        logger.info(
            "%s: relative residual %.4e above %.4e, at the round-off floor "
            "(backward error %.2e); accepted", solver, rel_res, accept,
            _backward_error(K_ff, u_f, rhs))
    elif not (info == 0 and rel_res <= accept):
        raise RuntimeError(
            f"{solver} failed to converge: info={info}, achieved relative "
            f"residual {rel_res:.4e} (requested rtol={rtol:.4e}"
            + (f", after {restarts} warm restarts" if restarts else "")
            + ").")
    logger.info(
        "%s converged: relative residual %.4e (rtol=%.4e, restarts=%d)",
        solver, rel_res, rtol, restarts)
    return u_f, rel_res


def solve_constrained(
    K: scipy.sparse.spmatrix, F: np.ndarray, constrained: dict[int, float],
    *, system: FreeDofSystem, ties: Ties | None = None,
    solver: Literal['direct', 'cg', 'minres'] = 'direct',
    rtol: float = 1e-9, verbose: bool = False, max_restarts: int = 3,
) -> tuple[np.ndarray, float]:
    """Impose the Dirichlet BCs (and ties) exactly and solve ``K u = F``.

    Partitions the DOFs into free (``f``) and constrained (``c``) sets
    (``_ConstraintReduction`` in :mod:`porosity_fe.fe.assembler`), solves
    ``K_ff u_f = F_f - K_fc u_c`` with the requested linear solver and
    returns the full vector with ``u_c`` equal to the prescribed values
    bit for bit. With ``ties`` (the bonded interface of a delaminated
    mesh) the reduced system is ``T^T K T u_r = T^T (F - K g)`` and every
    tied DOF equals its master plus the offset.

    Parameters
    ----------
    K : scipy.sparse matrix
        Assembled global stiffness matrix (not modified).
    F : np.ndarray
        Global force vector.
    constrained : dict[int, float]
        Constrained-DOF map (constrained DOF index -> prescribed value).
    system : callable
        ``system(K, part, factorize=...)`` returning ``K_ff`` and, when
        ``factorize``, its sparse LU (see :func:`free_dof_system`). The
        reduction's ``key`` includes the tie topology, so a cache keyed on
        it never mixes two tie sets.
    ties : (tied, master, offset), optional
        Master-slave ties ``u[tied] = u[master] + offset`` (see
        :meth:`_ConstraintReduction.from_constraints`); ``None`` (default)
        for prescribed values only.
    solver : {'direct', 'cg', 'minres'}
        Linear solver to use. See :meth:`FESolver.solve` for details
        (``'minres'`` is deprecated; :meth:`FESolver.solve` warns).
    rtol : float
        Relative-residual tolerance for the iterative solvers.
    verbose : bool
        Log progress information.
    max_restarts : int
        MINRES warm restarts (see :func:`_iterative_solve`).

    Returns
    -------
    u : np.ndarray
        Full displacement vector.
    rel_res : float
        Achieved relative residual ``||K_ff u_f - rhs|| / ||rhs||``
        of the reduced system (absolute residual when ``rhs = 0``).

    Raises
    ------
    ValueError
        If ``solver`` is not one of 'direct', 'cg', 'minres'.
    RuntimeError
        On a singular free-DOF stiffness, a non-positive diagonal for
        the preconditioner, a non-finite or non-converged solution, or
        a residual above tolerance.
    """
    if solver not in ('direct', 'cg', 'minres'):
        raise ValueError(
            f"Unknown solver '{solver}'. "
            "Use 'direct', 'cg', or 'minres'."
        )
    part = _ConstraintReduction.from_constraints(K.shape[0], constrained, ties=ties)
    if part.free.size == 0:
        return part.lift(), 0.0

    K_ff, lu = system(K, part, factorize=solver == 'direct')
    rhs = part.reduce_rhs(K, F)

    # Cheap conditioning diagnostic: with exact elimination the ratio
    # reflects the physical stiffness contrast (ply anisotropy, void
    # elements), not a boundary-condition artefact.
    diag = K_ff.diagonal()
    positive = diag[diag > 0.0]
    diag_ratio = float(positive.max() / positive.min()) \
        if positive.size == diag.size else float('inf')
    logger.info(
        "Free-DOF stiffness: diag ratio=%.3e (n_free=%d, n_fixed=%d)",
        diag_ratio, part.free.size, part.fixed.size)
    if verbose:
        logger.info(
            "Solving system (%d free of %d DOFs) with solver='%s'...",
            part.free.size, part.n_dof, solver)

    if solver == 'direct':
        assert lu is not None
        u_f = lu.solve(rhs)
        if not np.isfinite(u_f).all():
            raise RuntimeError(
                "The sparse LU solve produced non-finite values (NaN or "
                "Inf). Check the material stiffness and the boundary "
                "conditions.")
        rel_res = _relative_residual(K_ff, u_f, rhs)
        if rel_res >= 1e-6:
            raise RuntimeError(
                f"Sparse LU residual {rel_res:.4e} exceeds tolerance "
                "1e-6. Check the material stiffness and the boundary "
                "conditions.")
    else:
        u_f, rel_res = _iterative_solve(
            K_ff, rhs, diag, solver=solver, rtol=rtol,
            max_restarts=max_restarts)

    return part.expand(u_f), rel_res


def free_dof_system(
    cache: OrderedDict[bytes, tuple], K: scipy.sparse.spmatrix,
    part: _ConstraintReduction, *, factorize: bool, max_entries: int,
    memory_budget: float | None = None, held_bytes: float = 0.0,
) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.linalg.SuperLU | None]:
    """``K_ff`` and (when ``factorize``) its sparse LU, cached in ``cache``.

    ``cache`` maps a constrained-DOF set (``part.key``) to
    ``(K, K_ff, SuperLU | None)``, least recently used first. ``K_ff``
    depends only on the assembled ``K`` and the *set* of constrained DOFs,
    not their prescribed values, so e.g. compression and tension on the
    same mesh, or repeat solves of one load case, share it and its
    factorization. The systems of the last ``max_entries`` constrained sets
    are kept, so alternating two load cases with different supports (e.g.
    shear and ILSS, or compression and ILSS) factorizes each once with two
    entries; a further set evicts the least recently used before its own
    factorization is built, so at most ``max_entries`` are alive at once.
    Every entry belongs to the assembled ``K`` it was built from: a new
    ``K`` (the mesh, material, porosity or formulation changed) clears them
    all. The LU is built only for direct solves; iterative solves cache
    ``K_ff`` alone.

    Memory: one factorization of the production mesh of a 24-ply
    laminate (30 x 10 x 24, 32 M stored entries) takes about 320 MB
    (120 MB on 30 x 10 x 12), so a cache holding two sets holds about
    650 MB. Before a factorization is built, :func:`_memory.check_factorization
    <porosity_fe.fe._memory.check_factorization>` estimates it and raises a
    ``ValueError`` when ``held_bytes`` (what the caller holds outside this
    cache), the factorizations left in the cache and the estimate exceed
    ``memory_budget`` (bytes; ``None`` for no check).

    The factorization is SuperLU's with the symmetric ``MMD_AT_PLUS_A``
    ordering and diagonal pivoting (``SymmetricMode`` with
    ``diag_pivot_thresh = 0``), which is stable for the SPD ``K_ff`` and
    keeps the ordering's fill. With the default threshold of 1 SuperLU
    took an off-diagonal pivot wherever a diagonal was not the largest
    entry of its column, which on the slender DCB arms (251 pivots) cost
    16 % more factor entries and 29 % more time. Measured on the production
    systems (#228), ``COLAMD`` and ``MMD_ATA`` fill the coupon 1.65 and
    1.3 times as much and a geometric nested dissection 1.17 times; none
    is better on every mesh.
    """
    if cache and next(iter(cache.values()))[0] is not K:
        cache.clear()
    entry = cache.get(part.key)
    if entry is None:
        while len(cache) >= max_entries:
            cache.popitem(last=False)
        K_ff, lu = part.reduce_matrix(K), None
    else:
        _, K_ff, lu = entry
    if factorize and lu is None:
        held = held_bytes + sum(_memory.lu_bytes(e[2]) for e in cache.values()
                                if e[2] is not None)
        _memory.check_factorization(part.n_dof, K_ff.shape[0], held, memory_budget)
        # K_ff is SPD: a symmetric fill-reducing ordering with diagonal
        # pivoting (see the docstring).
        try:
            lu = scipy.sparse.linalg.splu(
                K_ff, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                options={'SymmetricMode': True})
        except RuntimeError as exc:
            raise RuntimeError(
                f"Sparse LU factorization of the free-DOF stiffness "
                f"failed ({exc}). A singular matrix usually means the "
                "boundary conditions leave a rigid-body mode "
                "unrestrained.") from exc
    cache[part.key] = (K, K_ff, lu)
    cache.move_to_end(part.key)
    return K_ff, lu
