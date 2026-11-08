"""The void-fraction check shared by the public entry points."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def check_fraction(Vp: Any, name: str = "Vp") -> np.ndarray:
    """``Vp`` as a float array, checked to be a fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        For a non-finite value or one outside ``[0, 1]``, with a hint when
        the value looks like a percent (at least 1.001).
    """
    arr = np.asarray(Vp, dtype=float)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all() and arr.min() >= 0.0 and arr.max() <= 1.0:
        return arr
    worst = float(arr[~finite][0]) if not finite.all() else (
        float(arr.max()) if arr.max() > 1.0 else float(arr.min()))
    hint = (f" Did you pass a percent? Use {worst / 100:.4f} instead of {worst:g}."
            if math.isfinite(worst) and worst >= 1.001 else "")
    raise ValueError(
        f"{name} must be a void fraction in [0, 1] (0.02 for 2 %), got {worst!r}.{hint}")
