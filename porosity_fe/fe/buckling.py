"""Linear (eigenvalue) buckling of the FE model.

The prebuckling state is the linear solve of a load case
(:meth:`FESolver.solve`). Its Gauss-point stress gives the geometric
(initial-stress) stiffness ``K_G``, and the buckling load factors are the
eigenvalues of

.. math::

    (K + \\lambda K_G)\\, \\phi = 0

on the constrained degrees of freedom: the load at which the model loses
stability is ``lambda`` times the applied one. Porosity enters through the
degraded stiffness of every element, so the FE path resolves where the
voids are: near-surface voids lower the bending stiffness, and the
buckling load, more than voids at the mid-plane.

The analysis is linear: no imperfection, no post-buckling path. With an
explicit :class:`~porosity_fe.Delamination` the eigenproblem has no
crack-face contact, so a mode whose faces would interpenetrate is not
physical; :attr:`BucklingResult.interpenetration` says how far each mode
does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .assembler import _ConstraintReduction, _mesh_ties
from .batch import _gauss_point_tables, element_dofs

if TYPE_CHECKING:
    from ..mesh import CompositeMesh
    from .results import FieldResults
    from .solver import FESolver

#: Load cases with a buckling analysis: the displacement-controlled
#: in-plane ones of the coupon.
BUCKLING_LOADINGS = ('compression', 'shear')

#: Iteration cap of the eigen-solve. A load that buckles the model converges
#: in a few dozen; one that does not has no positive load factor to converge
#: on, and the cap keeps that answer quick.
_ARPACK_MAXITER = 1000


@dataclass(frozen=True, eq=False)
class BucklingResult:
    """Buckling load factors and mode shapes of an FE model.

    Attributes
    ----------
    load_factors : numpy.ndarray
        Positive buckling load factors, ascending: the applied load times
        each is a critical load. The first is the buckling load.
    modes : numpy.ndarray
        Mode shapes ``(n_modes, n_nodes, 3)``, each scaled to a largest
        displacement of 1 (sign arbitrary; with a delamination, oriented so
        the faces open).
    loading : str
        The load case, ``'compression'`` or ``'shear'``.
    applied_strain : float
        The nominal strain of the prebuckling solve (negative in
        compression; engineering shear strain for ``'shear'``).
    prebuckling : FieldResults
        The linear solve the geometric stiffness is built from.
    interpenetration : numpy.ndarray or None
        Per mode, with an explicit delamination: the largest closing of a
        delaminated node pair when the mode is oriented so the faces open
        most, over the mode's largest displacement. 0 for a mode whose
        faces only open (or stay together); a mode above a few per cent
        needs crack-face contact the eigenproblem does not have. ``None``
        without a delamination.
    """

    load_factors: np.ndarray
    modes: np.ndarray
    loading: str
    applied_strain: float
    prebuckling: FieldResults
    interpenetration: np.ndarray | None = None

    @property
    def load_factor(self) -> float:
        """The lowest buckling load factor."""
        return float(self.load_factors[0])

    @property
    def critical_strain(self) -> float:
        """Magnitude of the nominal strain at buckling: the load factor
        times ``|applied_strain|``."""
        return self.load_factor * abs(self.applied_strain)

    @property
    def critical_stress(self) -> float | None:
        """Average stress at buckling in MPa (the axial stress for
        ``'compression'``, the shear stress for ``'shear'``): the critical
        strain times the model's effective modulus. ``None`` without one."""
        modulus = self.prebuckling.effective_modulus
        return None if modulus is None else self.critical_strain * float(modulus)

    def critical_load(self, mesh: CompositeMesh) -> float | None:
        """Axial force at buckling in N for ``'compression'``: the critical
        stress times the cross-section ``L_y x L_z`` of ``mesh``. ``None``
        for ``'shear'`` or without an effective modulus."""
        stress = self.critical_stress
        if self.loading != 'compression' or stress is None:
            return None
        return stress * float(mesh.L_y) * float(mesh.L_z)

    def knockdown(self, reference: BucklingResult) -> float | None:
        """Critical stress (and load, on the same mesh) over that of
        ``reference``, e.g. the void-free laminate; ``None`` without an
        effective modulus on either side."""
        own, ref = self.critical_stress, reference.critical_stress
        if own is None or ref is None or ref == 0.0:
            return None
        return own / ref

    def summary(self, reference: BucklingResult | None = None) -> dict:
        """JSON-ready scalars of the analysis (no mode shapes).

        With ``reference`` (the same analysis of the void-free laminate) the
        record adds its ``reference`` summary and the :meth:`knockdown`.
        """
        out: dict = {
            'loading': self.loading,
            'applied_strain': float(self.applied_strain),
            'load_factors': [float(v) for v in self.load_factors],
            'critical_strain': self.critical_strain,
            'critical_stress_MPa': self.critical_stress,
            'interpenetration': (None if self.interpenetration is None
                                 else [float(v) for v in self.interpenetration]),
        }
        if reference is not None:
            out['reference'] = reference.summary()
            out['knockdown'] = self.knockdown(reference)
        return out


def _shape_gradients(mesh: CompositeMesh) -> tuple[np.ndarray, np.ndarray]:
    """``dN/dx`` ``(E, G, 3, 8)`` and ``det(J) w`` ``(E, G)`` of every element."""
    _, dN, _, weights = _gauss_point_tables()
    coords = mesh.nodes[np.asarray(mesh.elements, dtype=np.intp)]   # (E, 8, 3)
    J = np.einsum('gij,ejk->egik', dN, coords)
    detJ = np.linalg.det(J)
    dN_dx = np.matmul(np.linalg.inv(J), dN[None])                  # (E, G, 3, 8)
    return dN_dx, detJ * weights


def geometric_stiffness(mesh: CompositeMesh, stress_global: np.ndarray
                        ) -> scipy.sparse.csc_matrix:
    """Geometric (initial-stress) stiffness of a stress state.

    ``K_G[3a+i, 3b+j] = delta_ij sum_g (dN_a/dx_k) sigma_kl (dN_b/dx_l)
    det(J) w`` over the elements, with the Gauss-point stress
    ``stress_global`` ``(E, G, 6)`` in the Voigt order ``[11, 22, 33, 23,
    13, 12]`` (MPa). Built from the compatible (trilinear) shape functions
    for both element formulations.
    """
    stress = np.asarray(stress_global, dtype=float)
    dN_dx, detJ_w = _shape_gradients(mesh)
    if stress.shape != detJ_w.shape + (6,):
        raise ValueError(
            f"stress_global must have shape {detJ_w.shape + (6,)} (elements, "
            f"Gauss points, 6); got {stress.shape}.")
    s = stress
    S = np.stack([np.stack([s[..., 0], s[..., 5], s[..., 4]], -1),
                  np.stack([s[..., 5], s[..., 1], s[..., 3]], -1),
                  np.stack([s[..., 4], s[..., 3], s[..., 2]], -1)], -2)   # (E, G, 3, 3)
    k = np.einsum('egia,egij,egjb,eg->eab', dN_dx, S, dN_dx, detJ_w)    # (E, 8, 8)
    n_elem = k.shape[0]
    eye = np.eye(3)
    ke = (k[:, :, None, :, None] * eye[None, None, :, None, :]).reshape(n_elem, 24, 24)
    dofs = element_dofs(np.asarray(mesh.elements, dtype=np.intp))
    rows = np.broadcast_to(dofs[:, :, None], ke.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], ke.shape).ravel()
    K_G = scipy.sparse.coo_matrix((ke.ravel(), (rows, cols)),
                                  shape=(mesh.n_dof, mesh.n_dof)).tocsc()
    return 0.5 * (K_G + K_G.T)


def _expand_mode(part: _ConstraintReduction, phi: np.ndarray) -> np.ndarray:
    """``T phi``: a mode on every DOF, zero at the prescribed ones."""
    u = np.zeros(part.n_dof, dtype=float)
    u[part.free] = phi
    if part.transform is not None:
        u[part.tied] = (part.transform @ phi)[part.tied]
    return u


def _orient_and_measure(mesh: CompositeMesh, modes: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray | None]:
    """Orient each mode so its delaminated faces open most, and measure how
    far they still close (over the mode's largest displacement)."""
    orig, dup = mesh.interface_node_pairs()
    if orig.size == 0:
        return modes, None
    open_pairs = ~mesh.bonded_pairs()
    if not open_pairs.any():
        return modes, np.zeros(len(modes))
    out = modes.copy()
    closing = np.zeros(len(modes))
    for m, mode in enumerate(modes):
        gap = mode[dup[open_pairs], 2] - mode[orig[open_pairs], 2]
        opening, closing_part = max(gap.max(), 0.0), max(-gap.min(), 0.0)
        if closing_part > opening:
            out[m] = -mode
            opening, closing_part = closing_part, opening
        scale = float(np.abs(out[m]).max())
        closing[m] = closing_part / scale if scale > 0.0 else 0.0
    return out, closing


def linear_buckling(fe_solver: FESolver, /, loading: str = 'compression',
                    n_modes: int = 1, *, applied_strain: float | None = None,
                    **solve_kwargs: Any) -> BucklingResult:
    """Linear buckling load factors of the model of ``fe_solver``.

    Parameters
    ----------
    fe_solver : FESolver
        Positional-only. The model: mesh, material, porosity field and
        formulation.
    loading : {'compression', 'shear'}
        The displacement-controlled load case of the prebuckling solve
        (:meth:`FESolver.solve`), with its supports. Compression holds the
        end faces straight (``u_x`` uniform over each) and pins one corner
        in ``z``, so the column is clamped against rotation at both ends
        with the far end free to sway: a slender strip buckles at
        ``pi^2 E I / L^2``.
    n_modes : int
        Number of modes (default 1), the lowest load factors (fewer when
        the eigen-solve converges on fewer).
    applied_strain : float, optional
        Keyword-only. Nominal strain of the prebuckling solve (default the
        solve's, ``-0.01`` for compression); the load factors scale with
        its inverse, the critical strain does not.
    **solve_kwargs
        Other arguments of :meth:`FESolver.solve` (``crack_contact``,
        ``interior_margin``, ...). ``delta_T`` is refused: a residual
        stress would be a fixed prestress the load factor does not scale.

    Returns
    -------
    BucklingResult

    Raises
    ------
    ValueError
        For another loading, ``n_modes`` below 1, a ``delta_T``, a direct
        solve that is not possible (``solver='cg'``), or a load under
        which the model does not buckle (the applied load times every load
        factor would be the reverse of it: reverse ``applied_strain``).
    """
    if loading not in BUCKLING_LOADINGS:
        raise ValueError(
            f"Buckling analyses the in-plane load cases {list(BUCKLING_LOADINGS)}; "
            f"got {loading!r}.")
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) \
            or n_modes < 1:
        raise ValueError(f"n_modes must be an integer >= 1; got {n_modes!r}.")
    if solve_kwargs.get('delta_T') is not None:
        raise ValueError(
            "Buckling with delta_T is not supported: the residual stress would "
            "be a prestress the load factor does not scale.")
    if solve_kwargs.get('solver', 'direct') != 'direct':
        raise ValueError("Buckling needs the direct solver (solver='direct').")

    solver = fe_solver
    pre = solver.solve(loading, applied_strain=applied_strain, **solve_kwargs)
    strain = float(applied_strain) if applied_strain is not None else (
        -0.01 if loading == 'compression' else 0.01)
    mesh = solver.mesh
    K = solver.assembler.stiffness()
    constrained, _ = solver._apply_boundary_conditions(loading, strain, -10.0)
    part = _ConstraintReduction.from_constraints(
        mesh.n_dof, constrained, ties=_mesh_ties(mesh))
    K_ff, lu = solver._free_dof_system(K, part, factorize=True)
    assert lu is not None
    KG_ff = part.reduce_matrix(geometric_stiffness(mesh, pre.stress_global))

    # (K + lambda K_G) phi = 0  <=>  -K_G phi = mu K phi with mu = 1 / lambda:
    # the largest mu are the lowest positive load factors, and K is SPD.
    n_free = K_ff.shape[0]
    k = int(min(n_modes, n_free - 2))
    K_inv = scipy.sparse.linalg.LinearOperator((n_free, n_free), matvec=lu.solve,
                                               dtype=float)
    try:
        mu, phi = scipy.sparse.linalg.eigsh(
            -KG_ff, k=k, M=K_ff, Minv=K_inv, which='LA', ncv=max(2 * k + 1, 20),
            maxiter=_ARPACK_MAXITER)
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        # The modes it did converge on, if any.
        mu, phi = exc.eigenvalues, exc.eigenvectors
    scale = float(np.abs(mu).max()) if mu.size else 0.0
    positive = mu > 1e-12 * scale
    if not positive.any():
        raise ValueError(
            f"The model does not buckle under this {loading} load: there is no "
            f"positive load factor. Reverse applied_strain.")
    order = np.argsort(-mu[positive])
    factors = 1.0 / mu[positive][order]
    shapes = phi[:, positive][:, order]
    modes = np.stack([_expand_mode(part, shapes[:, i]).reshape(-1, 3)
                      for i in range(shapes.shape[1])])
    scale = np.abs(modes).max(axis=(1, 2))
    modes = modes / np.where(scale > 0.0, scale, 1.0)[:, None, None]
    modes, closing = _orient_and_measure(mesh, modes)
    if not all(math.isfinite(v) for v in factors):
        raise RuntimeError("The buckling eigenproblem returned a non-finite load factor.")
    return BucklingResult(load_factors=factors, modes=modes, loading=loading,
                          applied_strain=strain, prebuckling=pre,
                          interpenetration=closing)
