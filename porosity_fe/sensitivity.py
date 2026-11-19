"""Which uncertain inputs drive an output: a ranking from Monte Carlo draws.

:func:`sensitivity_ranking` takes the per-draw values of the scattered
inputs and of one output of a sampling study (:func:`~porosity_fe.propagate_uncertainty`,
:func:`~porosity_fe.fe_porosity_uncertainty`) and ranks the inputs by the
share of the output's scatter each explains. It works on ranks, so it needs
the output to be monotone in each input, not linear: the Spearman rank
correlation of each input with the output, and the standardized rank
regression coefficients (SRRC) of a linear regression of the output's ranks
on all the inputs' ranks at once. For independent inputs the squared SRRC
are the shares of the rank variance each input explains, and they sum to
the regression's ``R^2``; an ``R^2`` well below 1 says the ranking misses
interactions or non-monotone effects (Saltelli et al., *Global Sensitivity
Analysis: The Primer*, 2008, section 1.2.5).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats


@dataclass(frozen=True, eq=False)
class SensitivityRanking:
    """The inputs of a sampling study, most influential first.

    Attributes
    ----------
    output : str
        The output ranked against.
    names : tuple of str
        The inputs that varied, by decreasing ``|srrc|``.
    spearman : numpy.ndarray
        Spearman rank correlation of each input with the output, aligned
        with :attr:`names`: its sign is the direction of the effect.
    srrc : numpy.ndarray
        Standardized rank regression coefficient of each input, aligned
        with :attr:`names`: the effect of the input with the others held.
    share : numpy.ndarray
        ``srrc^2`` over their sum: the share of the explained scatter each
        input takes (sums to 1).
    r2 : float
        Coefficient of determination of the rank regression: how much of
        the output's rank scatter the inputs explain together.
    n_samples : int
        Draws used (those with a finite output).
    constant : tuple of str
        Inputs that did not vary over the draws, left out of the ranking.
    """

    output: str
    names: tuple[str, ...]
    spearman: np.ndarray
    srrc: np.ndarray
    share: np.ndarray
    r2: float
    n_samples: int
    constant: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """The ranking as a JSON-ready dict: one entry per input, in order."""
        return {
            'output': self.output,
            'method': 'spearman and standardized rank regression (SRRC)',
            'n_samples': self.n_samples,
            'r2': self.r2,
            'ranking': [
                {'input': name, 'spearman': float(rho), 'srrc': float(c),
                 'share': float(s)}
                for name, rho, c, s in zip(self.names, self.spearman, self.srrc,
                                           self.share, strict=True)],
            'constant_inputs': list(self.constant),
        }


#: A spread below this fraction of the largest magnitude is round-off: the
#: input (or output) does not vary.
_ROUNDOFF = 1e-9


def _varies(values: np.ndarray) -> bool:
    return values.size > 0 and \
        float(np.ptp(values)) > _ROUNDOFF * float(np.abs(values).max())


def _standardized_ranks(values: np.ndarray) -> np.ndarray:
    r = scipy.stats.rankdata(values)
    return (r - r.mean()) / r.std()


def sensitivity_ranking(inputs: Mapping[str, Any], output: Any, *,
                        name: str = 'output') -> SensitivityRanking:
    """Rank the inputs of a sampling study by their effect on ``output``.

    Parameters
    ----------
    inputs : mapping of str to array_like
        The value of each scattered input in each draw, all the length of
        ``output``. Inputs that do not vary (to round-off, a spread below
        1e-9 of their magnitude) are left out (listed in
        :attr:`SensitivityRanking.constant`).
    output : array_like
        The output of each draw. Draws whose output is not finite (an
        infinite load factor, say) are dropped.
    name : str
        The output's name, recorded on the result.

    Returns
    -------
    SensitivityRanking

    Raises
    ------
    ValueError
        For an input of another length or with a non-finite value, no input
        that varies, an output that does not vary, or fewer finite draws
        than the varying inputs plus two (the regression needs them).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.normal(size=200), rng.normal(size=200)
    >>> ranking = sensitivity_ranking({'a': a, 'b': b}, 3.0 * a + b, name='y')
    >>> ranking.names, bool(ranking.share[0] > 0.8)
    (('a', 'b'), True)
    """
    y = np.asarray(output, dtype=float).ravel()
    columns: dict[str, np.ndarray] = {}
    for key, values in inputs.items():
        column = np.asarray(values, dtype=float).ravel()
        if column.size != y.size:
            raise ValueError(
                f"Input {key!r} has {column.size} draws and the output {y.size}.")
        if not np.isfinite(column).all():
            raise ValueError(f"Input {key!r} has a non-finite value.")
        columns[key] = column
    keep = np.isfinite(y)
    y = y[keep]
    varying = [k for k, c in columns.items() if _varies(c[keep])]
    constant = tuple(k for k in columns if k not in varying)
    if not varying:
        raise ValueError("No input varies over the draws: nothing to rank.")
    if y.size < len(varying) + 2:
        raise ValueError(
            f"Ranking {len(varying)} inputs needs at least {len(varying) + 2} "
            f"draws with a finite {name}; got {y.size}.")
    if not _varies(y):
        raise ValueError(f"The {name} does not vary over the draws: nothing to rank.")
    ry = _standardized_ranks(y)
    rx = np.column_stack([_standardized_ranks(columns[k][keep]) for k in varying])
    spearman = np.clip(rx.T @ ry / y.size, -1.0, 1.0)
    coef, *_ = np.linalg.lstsq(rx, ry, rcond=None)
    residual = ry - rx @ coef
    r2 = float(1.0 - residual @ residual / (ry @ ry))
    order = np.argsort(-np.abs(coef), kind='stable')
    squared = coef[order] ** 2
    total = float(squared.sum())
    share = squared / total if total > 0.0 else np.zeros_like(squared)
    return SensitivityRanking(
        output=name, names=tuple(varying[i] for i in order),
        spearman=spearman[order], srrc=coef[order], share=share,
        r2=r2 if math.isfinite(r2) else 0.0, n_samples=int(y.size), constant=constant)
