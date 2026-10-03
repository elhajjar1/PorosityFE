"""Gauss-point to node recovery of stress and strain fields.

The solver evaluates stress and strain at the ``2 x 2 x 2`` Gauss points of
every element (:class:`~porosity_fe.FieldResults`). The Gauss points sit
inside the element, at ``±1/sqrt(3)`` of the half-width, so their values
under-read a field that peaks on a surface, and the element mean (what the
legacy VTK export writes) under-reads it further. :func:`extrapolate_to_nodes`
recovers values at the element corners and then averages them between
neighbouring elements:

1. **Extrapolation.** Within one element the eight Gauss-point values are
   taken as samples of a trilinear field, the same interpolation the
   element uses for displacement. The corner values ``v_n`` that reproduce
   the samples satisfy ``N_gp v_n = v_gp``, where ``N_gp`` (8 x 8) holds the
   shape functions at the Gauss points, so ``v_n = N_gp^-1 v_gp``. A field
   linear in ``x``, ``y`` and ``z`` is recovered exactly.
2. **Averaging.** The stress in a laminate jumps at ply interfaces, so a
   plain average over every element sharing a node would smear the jump
   into a non-physical value. With ``average='ply'`` (the default) a corner
   is averaged only with corners of elements of the same ply; void
   elements form their own groups. Each element keeps its own
   ``(8, k)`` corner values, so the field stays discontinuous across
   interfaces.

See also ``docs/theory/fe.md`` (Stress recovery).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..mesh import CompositeMesh
from .batch import _gauss_point_tables

NodalAverage = Literal['ply', 'all', 'none']

_N_GP, _, _, _ = _gauss_point_tables()
#: ``(8, G)`` Gauss-point to corner extrapolation matrix, ``inv(N_gp)``.
#: Row ``n`` gives the value at element node ``n`` (VTK hexahedron order)
#: from the ``G = 8`` Gauss-point values in :func:`gauss_points_hex` order.
_GP_TO_NODE = np.linalg.inv(_N_GP)


def _element_groups(mesh: CompositeMesh, average: str) -> np.ndarray:
    """Integer averaging group per element; larger labels win at a node.

    ``'all'``: one group. ``'ply'`` / ``'none'``: the element's ply id
    (``mesh.elem_ply_ids``, which increases upward, so a node on a ply
    interface takes the ply above, as ``mesh.ply_ids`` does), falling back
    to the ply angle when the mesh has no ply ids. Void elements get labels
    below every solid one, so a node shared by void and solid elements
    reports the solid material's stress.
    """
    n_elem = len(mesh.elements)
    if average == 'all':
        return np.zeros(n_elem, dtype=np.int64)
    ply = getattr(mesh, 'elem_ply_ids', None)
    if ply is not None:
        labels = np.asarray(ply, dtype=np.int64).ravel()
    elif getattr(mesh, 'ply_angles', None) is not None:
        labels = np.unique(np.asarray(mesh.ply_angles, dtype=float),
                           return_inverse=True)[1].astype(np.int64).ravel()
    else:
        labels = np.zeros(n_elem, dtype=np.int64)
    if labels.shape[0] != n_elem:
        raise ValueError(
            f"mesh has {labels.shape[0]} element ply ids for {n_elem} elements."
        )
    void_idx = np.asarray(getattr(mesh, 'void_elements', ()),
                          dtype=np.intp).ravel()
    if void_idx.size:
        labels = labels - labels.min()
        labels[void_idx] -= int(labels.max()) + 1
    return labels


def extrapolate_to_nodes(field_gp: np.ndarray, mesh: CompositeMesh, *,
                         average: NodalAverage = 'ply',
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Recover a Gauss-point field at the mesh nodes.

    Extrapolates each element's Gauss-point values to its eight corners
    (exact for fields linear in ``x``, ``y``, ``z``) and averages the corner
    values between elements that share a node, without crossing ply
    interfaces unless asked to.

    Parameters
    ----------
    field_gp : np.ndarray
        ``(n_elem, 8)`` or ``(n_elem, 8, k)`` values at the ``2 x 2 x 2``
        Gauss points, in the order of
        :attr:`FieldResults.stress_global <porosity_fe.FieldResults>` (for
        example a stress or strain array, or one component of it).
    mesh : CompositeMesh
        The mesh the field was computed on (connectivity, ply ids, void
        elements).
    average : {'ply', 'all', 'none'}
        Which elements contribute to the value at a shared node.

        - ``'ply'`` (default): only elements of the same ply; void
          elements are grouped separately. Use this for laminate
          stresses, which jump at ply interfaces.
        - ``'all'``: every element sharing the node. Only appropriate for
          a field that is continuous across elements, such as in-plane
          strain or a homogeneous material.
        - ``'none'``: the corner values are returned unaveraged (the raw
          per-element extrapolation), for example to evaluate a failure
          criterion element by element. ``nodal`` is as for ``'ply'``.

    Returns
    -------
    nodal : np.ndarray
        ``(n_nodes,)`` or ``(n_nodes, k)``: one value per node. With
        ``'ply'`` / ``'none'`` a node on a ply interface reports the
        average over the ply **above** it (the ``mesh.ply_ids``
        convention) and a node shared by void and solid elements reports
        the solid elements' average. ``NaN`` for a node no element uses.
    corner : np.ndarray
        ``(n_elem, 8)`` or ``(n_elem, 8, k)``: per-element corner values
        in VTK hexahedron node order (``mesh.elements``), averaged over the
        node's group (``'ply'`` / ``'all'``) or raw (``'none'``). This is
        the discontinuous field; plot it on an exploded mesh (see
        :meth:`FieldResults.to_vtu <porosity_fe.FieldResults.to_vtu>`).

    Raises
    ------
    ValueError
        If ``average`` is unknown or the array does not match the mesh.

    Notes
    -----
    The extrapolation matrix is the inverse of the ``8 x 8`` matrix of
    trilinear shape functions evaluated at the Gauss points, the same for
    every element because it acts in natural coordinates. Averaging the
    corner values of the stress, rather than the Gauss-point values, is
    what lifts a bending surface stress from the element-mean ``0.50`` of
    the exact value (one element through the thickness of each half) to
    about ``1.02``; see ``docs/theory/fe.md``.

    Examples
    --------
    >>> nodal, corner = extrapolate_to_nodes(result.stress_global, mesh)
    >>> sigma_xx_top = nodal[mesh.nodes[:, 2] == mesh.L_z, 0]
    """
    if average not in ('ply', 'all', 'none'):
        raise ValueError(
            f"average must be 'ply', 'all' or 'none', got {average!r}."
        )
    elements = np.asarray(mesh.elements, dtype=np.int64)
    n_elem = elements.shape[0]
    n_nodes = int(np.asarray(mesh.nodes).shape[0])
    field = np.asarray(field_gp, dtype=float)
    scalar = field.ndim == 2
    if scalar:
        field = field[:, :, None]
    n_gp = _GP_TO_NODE.shape[1]
    if field.ndim != 3 or field.shape[:2] != (n_elem, n_gp) \
            or elements.ndim != 2 or elements.shape[1] != 8:
        raise ValueError(
            f"field_gp must have shape (n_elem, {n_gp}) or "
            f"(n_elem, {n_gp}, k) for an 8-node hex mesh with "
            f"{n_elem} elements; got {np.shape(field_gp)}."
        )
    k = field.shape[2]
    corner_raw = np.einsum('ng,egk->enk', _GP_TO_NODE, field)

    labels = _element_groups(mesh, 'all' if average == 'all' else 'ply')
    uniq_labels, gidx = np.unique(labels, return_inverse=True)
    n_groups = len(uniq_labels)
    # One bucket per (node, group): sorted by node, then by group label.
    key = elements * n_groups + gidx.reshape(-1, 1)
    buckets, inverse = np.unique(key.ravel(), return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse, minlength=len(buckets)).astype(float)
    flat = corner_raw.reshape(-1, k)
    sums = np.stack([np.bincount(inverse, weights=flat[:, c],
                                 minlength=len(buckets)) for c in range(k)],
                    axis=1)
    means = sums / counts[:, None]

    # Per node, keep the bucket with the largest group label (the last one).
    bucket_node = buckets // n_groups
    last = np.flatnonzero(np.r_[bucket_node[1:] != bucket_node[:-1], True])
    nodal = np.full((n_nodes, k), np.nan)
    nodal[bucket_node[last]] = means[last]

    corner = corner_raw if average == 'none' else \
        means[inverse].reshape(n_elem, 8, k)
    if scalar:
        return nodal[:, 0], corner[:, :, 0]
    return nodal, corner
