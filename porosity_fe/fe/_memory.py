"""Peak-memory model of a direct FE solve, and the budget it is held to (#228).

The sparse LU factorization of the free-DOF stiffness dominates the memory
of a direct solve: ``K_ff`` of the 30 x 10 x 24 production coupon has
1.7 M nonzeros, its factors 32 M. :func:`check_factorization` estimates a
factorization before SuperLU builds it and refuses one that would take the
process over its budget, so a too-large :class:`~porosity_fe.FESolver`
call fails at once with the estimate and the alternatives instead of late,
with a ``MemoryError``, a swap storm or the out-of-memory killer.
:func:`porosity_fe.fe.refinement.refinement_study` plans its levels with
the same model.

The model, fitted on T800 QI meshes (hex8i, SuperLU with the
``MMD_AT_PLUS_A`` ordering) from 20 x 8 x 24 to 45 x 15 x 24 and at
``nz = 48``: the nonzeros of ``L + U`` grow as about ``8 n_dof^1.5`` (measured
0.85-1.00 of that; with diagonal pivots, #228, 0.96-0.99 on the AS4/3501-6
coupons, 1.03 on the ILSS specimen, 0.82 on the flexure specimen and
0.58-0.77 on the fracture specimens), held at about 12 bytes each, on top of the per-element assembly arrays of
``CompositeMesh._FE_BYTES_PER_ELEMENT`` and the interpreter with NumPy and
SciPy loaded. It was 1.0-1.3 times the measured peak RSS up to
60 x 20 x 24. A factorization already built counts at its actual size
(``SuperLU.nnz``).
"""

from __future__ import annotations

import math
import os
import sys
from numbers import Real
from typing import Any

#: The interpreter with NumPy and SciPy loaded (bytes).
BASE_BYTES = 0.15e9
#: ``nnz(L + U) ~ LU_NNZ_COEFF * n_dof ** LU_NNZ_EXPONENT``.
LU_NNZ_COEFF = 8.0
LU_NNZ_EXPONENT = 1.5
#: A value (8 bytes) and a row index (4) per stored factor entry.
LU_BYTES_PER_NNZ = 12.0
#: Fraction of physical memory a solve or a study may plan to use by default.
DEFAULT_MEMORY_FRACTION = 0.8


def lu_bytes_estimate(n_dof: int) -> float:
    """Estimated size (bytes) of the sparse LU factors of an ``n_dof`` system."""
    return LU_BYTES_PER_NNZ * LU_NNZ_COEFF * float(n_dof) ** LU_NNZ_EXPONENT


def lu_bytes(lu: Any) -> float:
    """Size (bytes) of a built factorization, from its stored entries."""
    return LU_BYTES_PER_NNZ * float(lu.nnz)


def physical_memory_bytes() -> int | None:
    """Total physical memory, or ``None`` where the platform cannot tell."""
    if sys.platform == 'win32':
        return _windows_physical_memory()
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        size = os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        return None
    if pages <= 0 or size <= 0:
        return None
    return int(pages) * int(size)


def _windows_physical_memory() -> int | None:
    """``GlobalMemoryStatusEx().ullTotalPhys``; ``None`` if the call fails."""
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    windll = getattr(ctypes, 'windll', None)     # Windows only
    if windll is None:
        return None
    try:
        ok = windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except OSError:
        return None
    return int(status.ullTotalPhys) if ok and status.ullTotalPhys > 0 else None


def check_max_memory_gb(value: float | None, name: str = 'max_memory_gb') -> float | None:
    """Validate a memory budget in GB: ``None`` or a number > 0 (``inf`` allowed)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) \
            or math.isnan(value) or value <= 0:
        raise ValueError(
            f"{name} must be a number of GB > 0 (math.inf for no limit) or "
            f"None for {DEFAULT_MEMORY_FRACTION:.0%} of physical memory; got "
            f"{value!r}.")
    return float(value)


def resolve_budget(max_memory_gb: float | None) -> float | None:
    """The budget in bytes: ``max_memory_gb`` GB, or by default
    :data:`DEFAULT_MEMORY_FRACTION` of physical memory, ``None`` (no limit)
    where the platform does not report it."""
    if max_memory_gb is not None:
        return float(max_memory_gb) * 1e9
    phys = physical_memory_bytes()
    return None if phys is None else DEFAULT_MEMORY_FRACTION * phys


def check_factorization(n_dof: int, n_free: int, held_bytes: float,
                        budget: float | None) -> None:
    """Refuse a factorization whose estimated peak exceeds ``budget``.

    ``held_bytes`` is what the process already holds for the solve: the
    interpreter, the mesh's assembly arrays and the factorizations kept
    alive beside the new one. The new one is estimated from ``n_dof``, the
    full system's size, which the model was fitted on (``n_free`` is
    reported).

    Raises
    ------
    ValueError
        When ``held_bytes`` plus the estimate exceed ``budget``.
    """
    if budget is None:
        return
    new = lu_bytes_estimate(n_dof)
    total = held_bytes + new
    if total <= budget:
        return
    raise ValueError(
        f"The sparse LU factorization of the free-DOF stiffness ({n_free:,} "
        f"free of {n_dof:,} DOFs) needs an estimated {new / 1e9:.3g} GB, "
        f"{total / 1e9:.3g} GB with the mesh and the factorizations already "
        f"held, above the {budget / 1e9:.3g} GB budget (FESolver(max_memory_gb=); "
        f"by default {DEFAULT_MEMORY_FRACTION:.0%} of physical memory). Solve "
        f"with solver='cg', which factorizes nothing, use a coarser mesh, or "
        f"raise max_memory_gb if the machine has the memory.")
