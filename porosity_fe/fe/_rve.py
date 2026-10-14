"""Periodic unit-cell (RVE) homogenization: a check of the Mori-Tanaka voids.

Internal verification tool (IMPROVEMENT_PLAN 3.6, D1), not public API. It
computes the effective stiffness of a periodic cell numerically, with the
package's own elements, so the analytical Mori-Tanaka estimate
(:func:`porosity_fe.homogenization._mt_effective_stiffness`) can be
checked against a full-field solution of the same micro-structure.

Method
------
The displacement in the cell is split as ``u = E x + w``: ``E`` is a
uniform macroscopic strain (tensor form) and the fluctuation ``w`` is
periodic. Each node on a maximum face (``x = L_x``, ``y = L_y`` or
``z = L_z``) is tied to its image on the opposite minimum face,
``u(x + L) = u(x) + E L`` (:func:`periodic_reduction`, through the
constraint reduction ``u = T u_r + g`` of
:class:`~porosity_fe.fe.assembler._ConstraintReduction`), and the
``(0, 0, 0)`` corner is pinned at ``u = E x`` to remove the three rigid
translations (periodicity already excludes rotations). No other boundary
condition is applied, so the pinned corner carries no force. Because ``w``
is periodic on matching face grids, the volume average of the strain is
exactly ``E``. The six unit macro-strains (engineering shear, Voigt order
``[11, 22, 33, 23, 13, 12]``) give the six columns of the effective
stiffness as volume-averaged stresses (:func:`homogenize`). Periodic
boundary conditions satisfy the Hill-Mandel condition, so the result is
symmetric to round-off and ``E . C_eff E = u^T K u / V``.

Void representation
-------------------
:func:`void_unit_cell` builds a cubic (or box) cell of isotropic matrix
(``matrix_modulus``, ``matrix_poisson``) holding one centred ellipsoidal
void. The void is the set of elements whose centroid lies inside the
ellipsoid (:class:`~porosity_fe.void_geometry.VoidGeometry` and the void
elements of :class:`~porosity_fe.mesh.CompositeMesh`), which get the
near-zero stiffness of the explicit-void path (``Hex8Element.VOID_MODULUS``
= 1 MPa, against ``E_m`` = 3500 MPa for the T800 matrix). Every other
element is pristine matrix: no smeared porosity. The void is therefore a
voxel "staircase" of the ellipsoid. Compare with Mori-Tanaka at the
*voxel* void fraction (:attr:`PeriodicHomogenization.void_fraction`), not at
the nominal ellipsoid volume: on a 12^3 cell the staircase misses it by up
to 8 % for a sphere and 20 % for a thin penny. The remaining discretization
error is the staircase shape plus the FE approximation of the strain
concentration around it; it shrinks, not monotonically, as the cell is
refined, and a void that comes within one element of its periodic image
touches it in the voxel model (see the docs page on micromechanics). The
1 MPa void stiffness raises the effective stiffness by at most 0.02 % for a
sphere and 0.3 % for a crowded penny over an empty void.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg
from scipy.spatial import cKDTree

from .._types import FEFormulation
from ..materials import MaterialProperties
from ..mesh import CompositeMesh
from ..porosity_field import PorosityField
from ..void_geometry import VoidGeometry
from .assembler import GlobalAssembler, _ConstraintReduction
from .element import _DEFAULT_FORMULATION

#: Relative tolerance (of the largest cell dimension) for locating faces
#: and matching opposite-face nodes.
_MATCH_RTOL = 1e-9


def macro_strain_tensor(strain: Sequence[float]) -> np.ndarray:
    """Symmetric 3x3 strain tensor from an engineering-shear Voigt 6-vector.

    ``[e11, e22, e33, g23, g13, g12]`` (``g = 2 eps``) gives
    ``eps_ij`` with ``eps_23 = g23 / 2`` and so on.
    """
    e = np.asarray(strain, dtype=float)
    if e.shape != (6,):
        raise ValueError(f"strain must have 6 Voigt components, got shape {e.shape}.")
    return np.array([[e[0], 0.5 * e[5], 0.5 * e[4]],
                     [0.5 * e[5], e[1], 0.5 * e[3]],
                     [0.5 * e[4], 0.5 * e[3], e[2]]])


def periodic_node_pairs(mesh: CompositeMesh) -> tuple[np.ndarray, np.ndarray, int]:
    """Periodic node ties of a box-shaped mesh.

    Every node on a maximum face is tied to its image on the opposite
    minimum face. A node on several maximum faces (an edge or corner) is
    tied once, across the first such axis in ``x, y, z`` order, to an image
    that is itself tied onward: the ``(L_x, L_y, L_z)`` corner ties to
    ``(0, L_y, L_z)``, which ties to ``(0, 0, L_z)``, which ties to the
    origin corner. The constraint reduction resolves these chains.

    Returns
    -------
    image, source : np.ndarray
        Node indices, ``u[image] = u[source] + E (x[image] - x[source])``.
    corner : int
        The node at the minimum corner, the root of every chain.

    Raises
    ------
    ValueError
        If the node grids of two opposite faces do not match.
    """
    nodes = np.asarray(mesh.nodes, dtype=float)
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    tol = _MATCH_RTOL * float((hi - lo).max())
    on_min = np.abs(nodes - lo) <= tol
    on_max = np.abs(nodes - hi) <= tol
    images, sources = [], []
    for axis in range(3):
        image = np.flatnonzero(on_max[:, axis] & ~on_max[:, :axis].any(axis=1))
        pool = np.flatnonzero(on_min[:, axis])
        target = nodes[image].copy()
        target[:, axis] = lo[axis]
        dist, k = cKDTree(nodes[pool]).query(target)
        if image.size and float(np.max(dist)) > tol:
            raise ValueError(
                f"The node grids of the {'xyz'[axis]}_min and {'xyz'[axis]}_max "
                "faces do not match, so the mesh cannot be made periodic.")
        images.append(image)
        sources.append(pool[k])
    corner = np.flatnonzero(on_min.all(axis=1))
    if corner.size != 1:
        raise ValueError("The mesh has no single node at its minimum corner.")
    return np.concatenate(images), np.concatenate(sources), int(corner[0])


def periodic_reduction(mesh: CompositeMesh, strain: Sequence[float],
                       pairs: tuple[np.ndarray, np.ndarray, int] | None = None,
                       ) -> _ConstraintReduction:
    """Periodic boundary conditions for the macroscopic strain ``strain``.

    ``u = E x + w`` with ``w`` periodic and zero at the minimum corner:
    ties ``u[image] = u[source] + E (x[image] - x[source])`` for each
    component, and the corner pinned at ``u = E x``.

    Parameters
    ----------
    mesh : CompositeMesh
        Box-shaped mesh with matching opposite-face node grids.
    strain : sequence of 6 float
        Macroscopic strain, engineering shear, Voigt order
        ``[11, 22, 33, 23, 13, 12]``.
    pairs : tuple, optional
        Output of :func:`periodic_node_pairs`, to reuse it.

    Returns
    -------
    _ConstraintReduction
        The reduction; its ``key`` does not depend on ``strain``.
    """
    image, source, corner = periodic_node_pairs(mesh) if pairs is None else pairs
    E = macro_strain_tensor(strain)
    nodes = np.asarray(mesh.nodes, dtype=float)
    shift = (nodes[image] - nodes[source]) @ E.T             # (n_pairs, 3)
    comp = np.arange(3)
    tied = (3 * image[:, None] + comp).ravel()
    master = (3 * source[:, None] + comp).ravel()
    u_corner = E @ nodes[corner]
    constrained = {3 * corner + c: float(u_corner[c]) for c in range(3)}
    return _ConstraintReduction.from_constraints(
        mesh.n_dof, constrained, ties=(tied, master, shift.ravel()))


def _grid_indices(points: np.ndarray, tol: float) -> np.ndarray:
    """Integer plane index ``(i, j, k)`` of points on a tensor-product grid."""
    ijk = np.empty(points.shape, dtype=np.intp)
    for axis in range(3):
        planes = np.unique(np.round(points[:, axis] / tol))
        ijk[:, axis] = np.searchsorted(planes, np.round(points[:, axis] / tol))
    return ijk


def _nested_dissection(ijk: np.ndarray, leaf: int = 192) -> np.ndarray:
    """Fill-reducing elimination order for unknowns on a periodic grid.

    Geometric nested dissection: split the index box across its longest
    axis by a separator plane, order both halves recursively, then the
    separator. Along an axis that still wraps around (periodic), one plane
    does not disconnect the grid, so the first cut is two planes, the
    first and the middle one. SuperLU's minimum-degree orderings do poorly
    on the wrapped (toroidal) grid: on a 16^3 cell this order cuts the LU
    fill in half and the factorization time about fourfold.
    """
    order: list[np.ndarray] = []

    def split(idx: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              wrap: np.ndarray) -> None:
        extent = hi - lo + 1
        axis = int(np.argmax(extent))
        if idx.size <= leaf or extent[axis] < 3:
            order.append(idx)
            return
        c = ijk[idx, axis]
        if wrap[axis]:
            mid = (lo[axis] + hi[axis] + 1) // 2
            sep = (c == lo[axis]) | (c == mid)
            parts = ((lo[axis] + 1, mid - 1), (mid + 1, hi[axis]))
        else:
            mid = (lo[axis] + hi[axis]) // 2
            sep = c == mid
            parts = ((lo[axis], mid - 1), (mid + 1, hi[axis]))
        sub_wrap = wrap.copy()
        sub_wrap[axis] = False
        for a, b in parts:
            inside = (c >= a) & (c <= b)
            if inside.any():
                sub_lo, sub_hi = lo.copy(), hi.copy()
                sub_lo[axis], sub_hi[axis] = a, b
                split(idx[inside], sub_lo, sub_hi, sub_wrap)
        order.append(idx[sep])

    split(np.arange(len(ijk)), ijk.min(axis=0), ijk.max(axis=0),
          np.ones(3, dtype=bool))
    return np.concatenate(order)


@dataclass(frozen=True)
class PeriodicHomogenization:
    """Effective stiffness of a periodic cell (:func:`homogenize`).

    Attributes
    ----------
    stiffness : np.ndarray
        ``(6, 6)`` effective stiffness (MPa): column ``j`` is the
        volume-averaged stress under the unit macro-strain ``e_j``
        (engineering shear, Voigt order ``[11, 22, 33, 23, 13, 12]``). Not
        symmetrized; it is symmetric to round-off.
    mean_strain : np.ndarray
        ``(6, 6)`` volume-averaged strain, column ``j`` for load case
        ``j``; the identity to round-off.
    displacements : np.ndarray
        ``(6, n_dof)`` displacement of each load case.
    volume : float
        Cell volume (mm^3), the sum of ``det(J) w``.
    void_fraction : float
        Volume fraction of the void elements.
    n_reduced : int
        Unknowns after the periodic reduction (``3 (n_nodes - n_tied - 1)``).
    formulation : {'hex8i', 'hex8'}
        Element formulation used.
    """

    stiffness: np.ndarray
    mean_strain: np.ndarray
    displacements: np.ndarray
    volume: float
    void_fraction: float
    n_reduced: int
    formulation: FEFormulation


def homogenize(mesh: CompositeMesh, material: MaterialProperties,
               porosity_field: PorosityField, *,
               formulation: FEFormulation = _DEFAULT_FORMULATION,
               ) -> PeriodicHomogenization:
    """Effective stiffness of a periodic cell under six unit macro-strains.

    Assembles the cell with :class:`GlobalAssembler` (any porosity, ply
    angles and void elements the mesh carries), factorizes the periodic
    reduced stiffness once and solves the six load cases of
    :func:`periodic_reduction`.

    Parameters
    ----------
    mesh : CompositeMesh
        Box-shaped mesh with matching opposite-face node grids.
    material, porosity_field
        As for :class:`GlobalAssembler`.
    formulation : {'hex8i', 'hex8'}, optional
        Element formulation, default ``'hex8i'``.

    Raises
    ------
    ValueError
        If the mesh cannot be made periodic.
    RuntimeError
        If the factorization fails or a solve misses the residual check.
    """
    assembler = GlobalAssembler(mesh, material, porosity_field,
                                formulation=formulation)
    K = assembler.stiffness()
    batch = assembler.element_batch()
    pairs = periodic_node_pairs(mesh)
    eye = np.eye(6)
    reductions = [periodic_reduction(mesh, eye[j], pairs) for j in range(6)]
    K_r = reductions[0].reduce_matrix(K)
    # K_r is symmetric positive definite: diagonal pivoting in the order
    # given by the geometric nested dissection.
    nodes = np.asarray(mesh.nodes, dtype=float)
    tol = _MATCH_RTOL * float(np.ptp(nodes, axis=0).max())
    perm = _nested_dissection(_grid_indices(nodes[reductions[0].free // 3], tol))
    try:
        lu = scipy.sparse.linalg.splu(K_r[perm][:, perm].tocsc(), permc_spec='NATURAL',
                                      options={'SymmetricMode': True})
    except RuntimeError as exc:
        raise RuntimeError(
            f"Sparse LU factorization of the periodic cell stiffness failed "
            f"({exc}).") from exc

    detJ_w = batch.detJ_w
    volume = float(detJ_w.sum())
    void_idx = np.asarray(mesh.void_elements, dtype=np.intp)
    void_fraction = float(detJ_w[void_idx].sum()) / volume if void_idx.size else 0.0
    stiffness = np.zeros((6, 6))
    mean_strain = np.zeros((6, 6))
    displacements = np.zeros((6, mesh.n_dof))
    F = np.zeros(mesh.n_dof)
    K_r_norm = float(abs(K_r).sum(axis=1).max())
    for j, red in enumerate(reductions):
        assert red.key == reductions[0].key
        rhs = red.reduce_rhs(K, F)
        u_r = np.empty_like(rhs)
        u_r[perm] = lu.solve(rhs[perm])
        # Normwise backward error: well defined even when the reduced load
        # cancels to round-off (a homogeneous cell, where u = E x exactly).
        scale = K_r_norm * float(np.abs(u_r).max()) + float(np.abs(rhs).max())
        backward = float(np.abs(K_r @ u_r - rhs).max()) / scale if scale > 0 else 0.0
        if not (np.isfinite(u_r).all() and backward < 1e-10):
            raise RuntimeError(
                f"Periodic cell solve {j} failed: backward error {backward:.3e}.")
        u = red.expand(u_r)
        strain = batch.strains(u)
        stress = batch.stresses(strain)
        stiffness[:, j] = np.einsum('egi,eg->i', stress, detJ_w) / volume
        mean_strain[:, j] = np.einsum('egi,eg->i', strain, detJ_w) / volume
        displacements[j] = u
    return PeriodicHomogenization(
        stiffness=stiffness, mean_strain=mean_strain,
        displacements=displacements, volume=volume,
        void_fraction=void_fraction, n_reduced=int(reductions[0].free.size),
        formulation=formulation)


def _box(value: float | Sequence[float], name: str) -> tuple[float, float, float]:
    v = np.broadcast_to(np.asarray(value, dtype=float), (3,))
    if not (np.all(np.isfinite(v)) and np.all(v > 0)):
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return float(v[0]), float(v[1]), float(v[2])


def matrix_material(material: MaterialProperties, thickness: float = 1.0
                    ) -> MaterialProperties:
    """Isotropic copy of ``material`` with its matrix constants as ply constants.

    ``E11 = E22 = E33 = matrix_modulus``, all Poisson's ratios
    ``matrix_poisson`` and ``G = E_m / (2 (1 + nu_m))``, so the pristine
    ply stiffness is ``material.get_isotropic_matrix_stiffness()`` (to
    round-off). One ply of thickness ``thickness`` (mm), the cell height.
    """
    E, nu = material.matrix_modulus, material.matrix_poisson
    G = E / (2.0 * (1.0 + nu))
    return dataclasses.replace(
        material, E11=E, E22=E, E33=E, G12=G, G13=G, G23=G,
        nu12=nu, nu13=nu, nu23=nu, t_ply=float(thickness), n_plies=1)


def void_radii_for(shape_radii: Sequence[float], void_fraction: float,
                   size: float | Sequence[float] = 1.0) -> tuple[float, float, float]:
    """Radii of the given shape whose ellipsoid fills ``void_fraction`` of the cell."""
    shape = np.asarray(shape_radii, dtype=float)
    Lx, Ly, Lz = _box(size, 'size')
    scale = (void_fraction * Lx * Ly * Lz
             / (4.0 / 3.0 * np.pi * float(np.prod(shape)))) ** (1.0 / 3.0)
    r = shape * scale
    return float(r[0]), float(r[1]), float(r[2])


def void_unit_cell(material: MaterialProperties,
                   void_radii: Sequence[float] | None = None, *,
                   n: int | Sequence[int] = 12,
                   size: float | Sequence[float] = 1.0,
                   ) -> tuple[CompositeMesh, MaterialProperties, PorosityField]:
    """Periodic cell of isotropic matrix with one centred ellipsoidal void.

    Parameters
    ----------
    material : MaterialProperties
        Supplies ``matrix_modulus`` and ``matrix_poisson``
        (:func:`matrix_material`).
    void_radii : sequence of 3 float, optional
        Semi-axes ``(a, b, c)`` (mm) along ``x, y, z``; ``None`` for a
        void-free cell. Each must be below half the cell size, so the void
        does not overlap its periodic images.
    n : int or (nx, ny, nz)
        Elements per axis, default 12.
    size : float or (L_x, L_y, L_z)
        Cell size (mm), default 1. The result does not depend on the
        scale, only on the shape.

    Returns
    -------
    mesh, matrix, field
        Pass to :func:`homogenize`. The void elements have ``E = 1`` MPa;
        the nodal porosity is zero everywhere (the discrete void would set
        it to 1 at the nodes inside, smearing a Mori-Tanaka-degraded layer
        around the void).
    """
    Lx, Ly, Lz = _box(size, 'size')
    nx, ny, nz = (int(v) for v in np.broadcast_to(np.asarray(n), (3,)))
    matrix = matrix_material(material, Lz)
    voids = []
    if void_radii is not None:
        radii = np.asarray(void_radii, dtype=float)
        if radii.shape != (3,) or np.any(radii >= 0.5 * np.array([Lx, Ly, Lz])):
            raise ValueError(
                f"void_radii {list(np.atleast_1d(radii))} must be three semi-axes "
                f"below half the cell size {(Lx, Ly, Lz)}; a larger void "
                "overlaps its periodic images.")
        voids.append(VoidGeometry(center=(0.5 * Lx, 0.5 * Ly, 0.5 * Lz),
                                  radii=tuple(radii)))
    field = PorosityField(matrix, 0.0, discrete_voids=voids)
    mesh = CompositeMesh(field, matrix, nx=nx, ny=ny, nz=nz, ply_angles='UD')
    mesh.L_x, mesh.L_y = Lx, Ly              # L_z is the one ply's thickness
    mesh.generate_mesh()
    mesh.porosity[:] = 0.0
    mesh.stiffness_reduction[:] = 1.0
    return mesh, matrix, field


def void_cell_stiffness(material: MaterialProperties,
                        void_radii: Sequence[float] | None = None, *,
                        n: int | Sequence[int] = 12,
                        size: float | Sequence[float] = 1.0,
                        formulation: FEFormulation = _DEFAULT_FORMULATION,
                        ) -> PeriodicHomogenization:
    """:func:`homogenize` of :func:`void_unit_cell` (same arguments)."""
    mesh, matrix, field = void_unit_cell(material, void_radii, n=n, size=size)
    return homogenize(mesh, matrix, field, formulation=formulation)
