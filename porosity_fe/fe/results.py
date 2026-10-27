"""FE field results: the :class:`FieldResults` dataclass.

:meth:`FESolver.solve <porosity_fe.fe.solver.FESolver.solve>` returns one.
Split out of ``fe/solver.py`` (IMPROVEMENT_PLAN 5.1);
:mod:`porosity_fe.fe.solver` re-exports it, so
``from porosity_fe.fe.solver import FieldResults`` keeps working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..mesh import CompositeMesh
from ..results import FailureResult
from .element import _DEFAULT_FORMULATION
from .export import write_vtk, write_vtu
from .recovery import NodalAverage, extrapolate_to_nodes
from .vcct import CrackFront, vcct_energy_release_rates


@dataclass(frozen=True, slots=True)
class FieldResults:
    """Results from a finite element solve.

    Attributes
    ----------
    displacement : np.ndarray
        Shape (n_nodes, 3) nodal displacements.
    stress_global : np.ndarray
        Shape (n_elem, n_gp, 6) stress in global coordinates.
    stress_local : np.ndarray
        Shape (n_elem, n_gp, 6) stress in local (material) coordinates.
    strain_global : np.ndarray
        Shape (n_elem, n_gp, 6) strain in global coordinates.
    strain_local : np.ndarray
        Shape (n_elem, n_gp, 6) strain in local coordinates.
    max_failure_index : float
        Maximum failure index across all Gauss points (criterion-dependent).
    knockdown : float
        Stiffness knockdown factor: porous over pristine structural
        stiffness from a pristine reference solve of the same mesh (``E_x``
        or ``G_xy`` ratio; beam-stiffness ratio for ILSS and flexure).
    per_element_failure_index : np.ndarray or None
        Shape (n_elem,) max-over-Gauss-point failure index per element.
        Optional (defaults to ``None`` for back-compatibility with callers
        that construct ``FieldResults`` directly); populated by
        ``FESolver.solve`` and consumed by the VTK export so failure
        hot-spots can be sliced in ParaView.
    failure_criterion : str
        Which failure criterion was used to produce ``max_failure_index`` and
        ``per_element_failure_index``. One of ``'tsai_wu'`` (default for
        back-compat), ``'hashin'``, or ``'max_stress'``.
    failure_mode_indices : dict or None
        Per-mode breakdown **at the Gauss point of the maximum failure
        index** (not each mode's own maximum, see
        :attr:`failure_mode_maxima`), with keys ``'fiber_t'``,
        ``'fiber_c'``, ``'matrix_t'``, ``'matrix_c'``, ``'shear'``,
        ``'delamination'`` (plus ``'max_fi'``). For Tsai-Wu the per-mode
        entries are ``NaN`` (the polynomial does not separate modes;
        ``null`` in the JSON export); for ``max_stress`` the unused entries
        are zero. Lets the GUI and JSON exporter report the governing
        failure mode, not just severity.
    reaction_forces : np.ndarray or None
        Shape (n_nodes, 3) nodal reaction forces (N): ``K u - F`` at the
        constrained DOFs and exactly zero at the free DOFs. A load applied
        at a constrained DOF is carried by the support and enters its
        reaction with the opposite sign. Summed over a loaded face they
        give the resultant the boundary conditions apply.
    effective_modulus : float or None
        Homogenized modulus of the specimen (MPa). For the displacement-
        controlled modes, ``E_x`` for ``'compression'`` / ``'tension'``
        and ``G_xy`` for ``'shear'``, from the strain energy
        ``u^T K u / (strain^2 * V)``. For ``'flexure_3pt'`` /
        ``'flexure_4pt'``, the flexural modulus ``E_f`` from the mid-span
        deflection ``delta`` with the transverse shear deflection taken
        out (Timoshenko beam theory)::

            delta / P = a (3 L^2 - 4 a^2) / (48 E_f I) + a / (2 b K_s)

        with ``I = b h^3 / 12``, support span ``L``, shear span ``a``
        (``L / 2`` three-point, ``(L - load_span) / 2`` four-point) and
        ``K_s`` the laminate transverse shear stiffness per unit width
        (``(5/6) G_13 h`` for a UD beam); see the finite-element theory
        page. Comparable with the classical-lamination-theory ``E_f``
        (:func:`~porosity_fe.compute_degraded_clt_flexural_modulus`).
        ``None`` for ``'ilss'``: at the short-beam span of four
        thicknesses about half the deflection is shear and the contact
        indentation is not negligible, so a flexural modulus would mostly
        measure the shear correction (use ``'flexure_3pt'`` on a
        :meth:`~porosity_fe.CompositeMesh.for_flexure` specimen for
        ``E_f``). ``None`` for ``'thermal'`` too.
    first_ply_failure_load_factor : float or None
        Multiplier on the applied load at which ``failure_criterion`` first
        reaches 1 at any Gauss point of a non-void element (linear
        scaling); the margin of safety is this value minus 1. ``inf`` if no
        point is stressed.
    formulation : str
        Element formulation of the solve, ``'hex8i'`` (default, also for
        callers that construct ``FieldResults`` directly) or ``'hex8'``;
        see :class:`FESolver`.
    delta_T : float or None
        Temperature change from the stress-free state (K) of a thermal
        (``loading='thermal'``) or combined (mechanical with ``delta_T=``)
        solve; ``None`` for a purely mechanical solve.
    residual_stress_local : np.ndarray or None
        Shape (n_elem, n_gp, 6) ply-local thermal residual stress of the
        free-standing laminate at ``delta_T``, the part of
        :attr:`stress_local` that does not scale with the mechanical load.
        ``None`` without ``delta_T``.
    residual_max_failure_index : float or None
        Maximum failure index of the residual stress alone (``None``
        without ``delta_T``). At or above 1 the laminate fails on cool-down
        and :attr:`first_ply_failure_load_factor` is 0.
    interior_max_failure_index : float or None
        Maximum failure index over the interior elements, whose centroids
        lie at least ``interior_margin`` from every lateral face
        (``x``/``y``; by default ``max(h, 2 element widths)`` with ``h``
        the laminate thickness). It leaves out the free-edge singularity
        that makes the domain maximum mesh-dependent under thermal load.
        For ``'ilss'`` the interior is the two shear spans of the
        short-beam specimen instead: at least ``interior_margin`` (default
        ``h / 2``) from both support lines, the edges of the loading band
        and the side faces, so the contact stresses under the nose and at
        the supports are left out
        (:func:`porosity_fe.fe.short_beam.interior_element_mask`). For
        ``'flexure_3pt'`` and ``'flexure_4pt'`` it is the span between
        the supports, at least ``interior_margin`` (default ``h / 2``)
        from the side faces and, in the ``x``-``z`` plane, from each
        support line and load band, so the bending stress at mid-span
        stays in (:func:`porosity_fe.fe.flexure.interior_element_mask`).
        ``None`` when no element is that far from the edges.
    interior_first_ply_failure_load_factor : float or None
        :attr:`first_ply_failure_load_factor` over the interior elements
        only; ``None`` when there are none.
    load_factor_basis : str
        What :attr:`first_ply_failure_load_factor` multiplies:
        ``'mechanical'`` (default; the applied load), ``'delta_T'``
        (``loading='thermal'``: the temperature change, so the critical
        change is ``factor * delta_T``) or ``'mechanical_with_residual'``
        (the applied load, with the residual stress held fixed).
    cte_local : tuple of float or None
        Lamina CTEs ``(alpha_1, alpha_2, alpha_3)`` (1/K) used with
        ``delta_T``; ``None`` without it.
    failure_mode_maxima : dict or None
        Maximum over the whole model of each separate mode of the
        ``'hashin'`` and ``'max_stress'`` criteria (keys as
        :attr:`failure_mode_indices` without ``'max_fi'``), so a mode that
        does not govern is still visible. The ``'delamination'`` entry is
        the stress-based delamination-initiation index: Hashin's
        interlaminar mode evaluated at the ply Gauss points, which sit
        inside the plies, not on the interfaces; there is no explicit
        delamination in the model. ``None`` for Tsai-Wu, which does not
        separate modes.
    per_element_mode_failure_index : dict or None
        Per-element maximum over the Gauss points of each mode, one
        ``(n_elem,)`` array per key of :attr:`failure_mode_maxima`
        (written to the VTK/VTU exports as ``failure_index_<mode>``);
        ``None`` for Tsai-Wu.
    crack_front : CrackFront or None
        Virtual-crack-closure table of the delamination front of a mesh
        with an explicit :class:`~porosity_fe.Delamination`: per front node
        its coordinates, frame, ``Delta_a`` and tributary width, ``G_I``,
        ``G_II``, ``G_III``, ``G_T`` (N/mm), the mode mixity ``psi`` and,
        when the material has toughness data, the mixed-mode toughness
        ``G_c`` and the critical load factor ``lambda_c``
        (:func:`~porosity_fe.vcct_energy_release_rates`). At the applied
        state, thermal residual included. ``None`` without a delaminated
        node pair.
    fracture_load_factor : float or None
        Smallest ``lambda_c`` over the front: the multiplier on the load
        named by :attr:`load_factor_basis` at which the delamination starts
        to grow (Benzeggagh-Kenane criterion,
        :func:`~porosity_fe.fracture_load_factors`); ``inf`` when the load
        releases no energy at the front. ``None`` without a front or
        without toughness data (``MaterialProperties.G_Ic``, ``G_IIc``,
        ``bk_eta``).
    fracture_governing_node : int or None
        Original node id of the front node with the smallest ``lambda_c``
        (``None`` when :attr:`fracture_load_factor` is ``None`` or
        infinite).
    interface_forces : np.ndarray or None
        ``(n_pairs, 3)`` force (N, global axes) the lower crack face exerts
        on the upper one at every duplicated interface node pair
        (:meth:`CompositeMesh.interface_node_pairs` order): the tie force of
        a bonded pair, the normal contact force of a pair in contact, zero
        for an open pair. ``None`` without duplicated interface nodes.
    crack_contact : bool or None
        Whether crack-face contact was enforced (``FESolver.solve(...,
        crack_contact=...)``); ``None`` without a delaminated node pair.
    contact_iterations : int or None
        Linear solves of the contact active-set loop (1 when no crack-face
        pair came into contact); ``None`` when contact was not enforced.
    contact_pairs : int or None
        Delaminated node pairs in contact at the solution; ``None`` when
        contact was not enforced.
    contact_converged : bool or None
        Whether the contact active set settled within the iteration cap
        (a warning is logged when it did not); ``None`` when contact was
        not enforced.

    Notes
    -----
    Voigt order for both stress and strain: ``[11, 22, 33, 23, 13, 12]``
    (normals first, then 23 / 13 / 12 shears). The last three strain
    components are **engineering** strain (``gamma_ij = 2 * eps_ij``); the
    last three stress components are the matching shear stresses
    ``[tau_23, tau_13, tau_12]``. Sign convention: tensile normals are
    positive, compressive normals are negative — these arrays are signed
    (unlike the empirical ``failure_stress_MPa`` returned by
    :meth:`EmpiricalSolver.apply_loading`, which is a positive magnitude).
    Use :func:`strain_transformation_3d` / :func:`stress_transformation_3d`
    to rotate these arrays between frames.

    With a thermal load (``delta_T`` set) the stress and displacement
    arrays hold totals, residual plus mechanical (superposed solves, see
    :meth:`FESolver.solve`), and the strains are total strains, what a
    gauge bonded in the stress-free state would read: mechanical strain
    plus the free thermal strain ``alpha delta_T``. ``knockdown`` is
    ``nan`` for ``loading='thermal'``, which has no structural stiffness
    measure.
    """
    displacement: np.ndarray
    stress_global: np.ndarray
    stress_local: np.ndarray
    strain_global: np.ndarray
    strain_local: np.ndarray
    max_failure_index: float
    knockdown: float
    per_element_failure_index: np.ndarray | None = None
    failure_criterion: str = 'tsai_wu'
    failure_mode_indices: dict[str, float] | None = None
    reaction_forces: np.ndarray | None = None
    effective_modulus: float | None = None
    first_ply_failure_load_factor: float | None = None
    formulation: str = _DEFAULT_FORMULATION
    delta_T: float | None = None
    residual_stress_local: np.ndarray | None = None
    residual_max_failure_index: float | None = None
    interior_max_failure_index: float | None = None
    interior_first_ply_failure_load_factor: float | None = None
    load_factor_basis: str = 'mechanical'
    cte_local: tuple[float, float, float] | None = None
    failure_mode_maxima: dict[str, float] | None = None
    per_element_mode_failure_index: dict[str, np.ndarray] | None = None
    crack_front: CrackFront | None = None
    fracture_load_factor: float | None = None
    fracture_governing_node: int | None = None
    interface_forces: np.ndarray | None = None
    crack_contact: bool | None = None
    contact_iterations: int | None = None
    contact_pairs: int | None = None
    contact_converged: bool | None = None

    def __repr__(self) -> str:
        n_nodes = self.displacement.shape[0] if self.displacement is not None else 0
        n_elem = self.stress_global.shape[0] if self.stress_global is not None else 0
        return (f"FieldResults(n_nodes={n_nodes}, n_elements={n_elem}, "
                f"max_FI={self.max_failure_index:.4f}, "
                f"knockdown={self.knockdown:.4f})")

    def summary(self, sigma_pristine: float | None = None,
                model_label: str | None = None) -> FailureResult:
        """Distill the field result into a :class:`FailureResult`.

        Unifies the FE return shape with the empirical solver (#44 item 1)
        so callers can treat the two solver outputs polymorphically.

        Parameters
        ----------
        sigma_pristine : float, optional
            Pristine reference stress (MPa) used to compute
            ``failure_stress = knockdown * sigma_pristine``. If ``None``
            (default), the ``failure_stress`` field is set to
            ``knockdown`` itself (unit knockdown — the bare ratio); pass
            the loading-mode-specific pristine strength
            (``material.sigma_1c`` for compression, ``material.tau_ilss``
            for ILSS, etc.) to get a meaningful magnitude.
        model_label : str, optional
            Label used for the ``model`` field. Defaults to
            ``f"fe_{self.failure_criterion}"`` so the FE summary is
            self-describing and distinguishable from the empirical labels.

        Returns
        -------
        FailureResult
            Unified summary with the FE ``knockdown`` (``nan`` for
            ``loading='thermal'``, passed through), derived
            ``failure_stress``, FE-criterion-tagged ``model`` and a
            ``details`` dict carrying ``max_failure_index``,
            ``failure_criterion`` and ``failure_mode_indices`` for
            downstream consumers that need the richer field-result data,
            and the delamination keys (``fracture_load_factor``,
            ``fracture_governing_node``, ``fracture_governing_psi``,
            ``fracture_governing_G_T``, ``crack_front_max_G_T``,
            ``crack_contact``, ``contact_iterations``, ``contact_pairs``,
            ``contact_converged``), ``None`` without a delamination.
        """
        kd = float(self.knockdown)
        sigma_ref = float(sigma_pristine) if sigma_pristine is not None else 1.0
        return FailureResult(
            failure_stress=kd * sigma_ref,
            knockdown=kd,
            model=str(model_label) if model_label is not None
            else f"fe_{self.failure_criterion}",
            details={
                'max_failure_index': float(self.max_failure_index),
                'failure_criterion': self.failure_criterion,
                'failure_mode_indices': (
                    dict(self.failure_mode_indices)
                    if self.failure_mode_indices is not None else None
                ),
                'failure_mode_maxima': (
                    dict(self.failure_mode_maxima)
                    if self.failure_mode_maxima is not None else None
                ),
                'first_ply_failure_load_factor': self.first_ply_failure_load_factor,
                'effective_modulus': self.effective_modulus,
                'formulation': self.formulation,
                'load_factor_basis': self.load_factor_basis,
                'interior_max_failure_index': self.interior_max_failure_index,
                'interior_first_ply_failure_load_factor':
                    self.interior_first_ply_failure_load_factor,
                'delta_T': self.delta_T,
                'residual_max_failure_index': self.residual_max_failure_index,
                **self._fracture_details(),
            },
        )

    def _fracture_details(self) -> dict:
        """Delamination keys of :meth:`summary` (``None`` values without one)."""
        front = self.crack_front
        row = None if front is None else front.governing
        return {
            'fracture_load_factor': self.fracture_load_factor,
            'fracture_governing_node': self.fracture_governing_node,
            'fracture_governing_psi': (
                None if row is None else float(front.psi[row])),  # type: ignore[union-attr]
            'fracture_governing_G_T': (
                None if row is None else float(front.G_T[row])),  # type: ignore[union-attr]
            'crack_front_max_G_T': (
                None if front is None or len(front) == 0
                else float(np.max(front.G_T))),
            'crack_contact': self.crack_contact,
            'contact_iterations': self.contact_iterations,
            'contact_pairs': self.contact_pairs,
            'contact_converged': self.contact_converged,
        }

    def energy_release_rates(self, mesh: CompositeMesh) -> CrackFront | None:
        """Recompute the virtual-crack-closure table from these results.

        :func:`~porosity_fe.vcct_energy_release_rates` on
        :attr:`displacement` and :attr:`interface_forces`, without toughness
        (``G_c`` and ``lambda_c`` are ``None``; see
        :func:`~porosity_fe.fracture_load_factors`). Equal to
        :attr:`crack_front` without them.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results, with its delaminations as
            solved.

        Returns
        -------
        CrackFront or None
            ``None`` when the results have no interface forces (no
            duplicated interface nodes) or the mesh has no front.
        """
        if self.interface_forces is None:
            return None
        return vcct_energy_release_rates(mesh, self.displacement,
                                         self.interface_forces)

    def to_vtk(self, mesh: CompositeMesh,
               filename: str | os.PathLike) -> None:
        """Write the hex mesh and per-element FE fields to a legacy ASCII VTK
        file (``UNSTRUCTURED_GRID``) for inspection in ParaView / VisIt / PyVista.

        The writer is dependency-free: it emits the legacy VTK 3.0 ASCII
        format by hand. The 8-node hex connectivity already stored in
        ``mesh.elements`` follows the standard VTK hexahedron ordering
        (bottom face CCW then top face CCW, see ``_NODE_COORDS_REF``), so the
        cells are written verbatim with cell type 12 (``VTK_HEXAHEDRON``).

        Point data
        -----------
        - ``displacement`` (3-vector), and the scalars ``porosity``,
          ``stiffness_reduction``, ``ply_id`` if present on the mesh.

        Cell data
        ---------
        - element-averaged ``von_mises`` and the six global stress
          (``sigma_xx`` .. ``tau_xy``) and strain (``eps_xx`` .. ``gamma_xy``)
          components (Gauss points reduced by mean),
        - ``failure_index`` (max-over-GP per element of the solve's
          criterion, if available) and its legacy alias ``tsai_wu_index``
          (the same array whatever the criterion),
        - for ``'hashin'`` and ``'max_stress'``, ``failure_index_<mode>``
          per mode (``fiber_t``, ``fiber_c``, ``matrix_t``, ``matrix_c``,
          ``shear``, ``delamination``),
        - ``Vp_elem`` (mean nodal porosity over the 8 corners),
        - ``ply_id``, ``ply_angle_deg``, ``is_void`` and ``knockdown`` where
          available (``knockdown`` only when finite: it is ``nan`` for
          ``loading='thermal'``),
        - with a thermal load, the element-mean ply-local residual stress
          ``residual_sigma_11_local``, ``residual_sigma_22_local`` and
          ``residual_tau_12_local``.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results (supplies geometry,
            connectivity, porosity and ply metadata).
        filename : str or os.PathLike
            Output ``.vtk`` file path. ``pathlib.Path`` objects are accepted.

        See Also
        --------
        to_vtu : Binary VTK XML, smaller and faster, with recovered nodal
            stress and strain.
        """
        write_vtk(self, mesh, filename)

    def to_vtu(self, mesh: CompositeMesh, filename: str | os.PathLike, *,
               precision: Literal['float64', 'float32'] = 'float64',
               encoding: Literal['raw', 'base64'] = 'raw',
               nodal: bool = True,
               exploded: bool = False,
               average: NodalAverage = 'ply') -> None:
        """Write the hex mesh and FE fields to a binary VTK XML file (``.vtu``).

        Opens in ParaView, VisIt, PyVista and meshio. The writer is
        dependency-free. The file holds everything :meth:`to_vtk` writes
        (same names) plus nodal stress and strain recovered from the Gauss
        points (:func:`~porosity_fe.extrapolate_to_nodes`). Binary storage
        is lossless and compact: on the production mesh the
        :meth:`to_vtk` fields take 1.1 MB instead of 2.0 MB, 1.6 MB with
        the nodal fields added, and ``precision='float32'`` brings that to
        0.9 MB. It is also several times faster to write.

        Point data
        ----------
        - ``displacement`` (3-vector), ``porosity``, ``stiffness_reduction``
          and ``ply_id`` as in :meth:`to_vtk`;
        - with ``nodal=True``: ``sigma_xx_nodal`` .. ``tau_xy_nodal``,
          ``eps_xx_nodal`` .. ``gamma_xy_nodal`` (global frame, engineering
          shear strain) and ``von_mises_nodal``;
        - with ``exploded=True``: ``node_id``, the original node of each
          point;
        - on a delaminated mesh with a front (:attr:`crack_front`):
          ``G_I``, ``G_II``, ``G_III`` and ``G_T`` (N/mm) at both nodes of
          every front pair and ``NaN`` at every other node (binary VTK XML
          stores ``NaN``; the legacy ASCII file of :meth:`to_vtk` leaves
          them out, and the JSON export carries the front table).

        Cell data
        ---------
        The :meth:`to_vtk` cell fields: element-mean ``von_mises``,
        ``sigma_xx`` .. ``tau_xy``, ``eps_xx`` .. ``gamma_xy``,
        ``failure_index`` (and ``tsai_wu_index``, ``failure_index_<mode>``),
        ``Vp_elem``, ``ply_id``, ``ply_angle_deg``,
        ``is_void`` and ``knockdown`` where available, and the
        ``residual_*_local`` fields of a thermal load.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results.
        filename : str or os.PathLike
            Output ``.vtu`` file path.
        precision : {'float64', 'float32'}
            Floating-point type of coordinates and fields. ``'float64'``
            (default) is lossless.
        encoding : {'raw', 'base64'}
            ``'raw'`` (default) appends the binary blocks after the XML
            header (VTK's appended format: smallest and fastest). ``'base64'``
            writes them inline as base64 text, which keeps the file
            well-formed XML for tools that require it, at about 4/3 the
            size.
        nodal : bool
            Add the recovered nodal stress and strain (default ``True``).
        exploded : bool
            Give every cell its own eight points (``8 * n_elem`` points),
            so the recovered fields can jump between elements and show the
            stress discontinuity at ply interfaces exactly. With the
            default ``False`` the nodes are shared and each node carries
            one value: at a ply interface, the average over the ply above.
        average : {'ply', 'all', 'none'}
            Averaging of the recovered fields between elements; see
            :func:`~porosity_fe.extrapolate_to_nodes`. ``'none'`` matters
            only with ``exploded=True``.

        Raises
        ------
        ValueError
            For an unknown option, a non-hex mesh, or results that do not
            match the mesh.

        See Also
        --------
        porosity_fe.write_pvd : Group several VTU files into a series.
        """
        write_vtu(self, mesh, filename, precision=precision,
                  encoding=encoding, nodal=nodal, exploded=exploded,
                  average=average)

    def _nodal(self, field_gp: np.ndarray, mesh: CompositeMesh,
               frame: str, average: NodalAverage,
               ) -> tuple[np.ndarray, np.ndarray]:
        if frame not in ('global', 'local'):
            raise ValueError(f"frame must be 'global' or 'local', got {frame!r}.")
        if frame == 'local' and average == 'all' and \
                np.unique(np.asarray(mesh.ply_angles, dtype=float)).size > 1:
            raise ValueError(
                "average='all' would mix ply-local components of plies with "
                "different orientations; use average='ply' or frame='global'."
            )
        return extrapolate_to_nodes(field_gp, mesh, average=average)

    def nodal_stress(self, mesh: CompositeMesh, *,
                     frame: Literal['global', 'local'] = 'global',
                     average: NodalAverage = 'ply',
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Stress recovered at the nodes from the Gauss points.

        Extrapolates each element's eight Gauss-point stresses to its
        corners and averages them between elements of the same ply (by
        default), so a surface or interface peak is not under-read the way
        the element mean or the Gauss-point values are. See
        :func:`~porosity_fe.extrapolate_to_nodes` for the method and the
        ``average`` options.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results.
        frame : {'global', 'local'}
            Recover :attr:`stress_global` (default) or :attr:`stress_local`
            (ply material axes). ``'local'`` cannot be combined with
            ``average='all'`` on a multi-angle laminate.
        average : {'ply', 'all', 'none'}
            Which elements a node averages over (default ``'ply'``).

        Returns
        -------
        nodal : np.ndarray
            ``(n_nodes, 6)`` stress (MPa), Voigt order
            ``[11, 22, 33, 23, 13, 12]``; at a ply interface, the ply above.
        corner : np.ndarray
            ``(n_elem, 8, 6)`` per-element corner stress (discontinuous
            across plies).
        """
        field = self.stress_global if frame == 'global' else self.stress_local
        return self._nodal(field, mesh, frame, average)

    def nodal_strain(self, mesh: CompositeMesh, *,
                     frame: Literal['global', 'local'] = 'global',
                     average: NodalAverage = 'ply',
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Strain recovered at the nodes from the Gauss points.

        Same method and options as :meth:`nodal_stress`, applied to
        :attr:`strain_global` or :attr:`strain_local` (engineering shear
        strain in the last three Voigt slots).

        Returns
        -------
        nodal : np.ndarray
            ``(n_nodes, 6)`` strain.
        corner : np.ndarray
            ``(n_elem, 8, 6)`` per-element corner strain.
        """
        field = self.strain_global if frame == 'global' else self.strain_local
        return self._nodal(field, mesh, frame, average)
