"""FE result export: legacy ASCII VTK and JSON summaries.

Split out of ``fe/solver.py`` (IMPROVEMENT_PLAN 5.1). The public entry
points remain :meth:`FieldResults.to_vtk
<porosity_fe.fe.solver.FieldResults.to_vtk>` and
:meth:`FESolver.export_results
<porosity_fe.fe.solver.FESolver.export_results>`, which delegate here.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..io import FORMAT_FE_FIELDS, JSON_SCHEMA_VERSION, _build_provenance, _json_default
from ..mesh import CompositeMesh

if TYPE_CHECKING:
    from .solver import FieldResults

logger = logging.getLogger("porosity_fe_analysis")


def write_vtk(results: FieldResults, mesh: CompositeMesh,
              filename: str | os.PathLike) -> None:
    """Write ``results`` on ``mesh`` as legacy ASCII VTK; see :meth:`FieldResults.to_vtk`."""
    filename = Path(filename)
    nodes = np.asarray(mesh.nodes, dtype=float)
    elements = np.asarray(mesh.elements, dtype=np.int64)
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

    if elements.shape[1] != 8:
        raise ValueError(
            f"to_vtk only supports 8-node hexahedra; got connectivity "
            f"of width {elements.shape[1]}."
        )
    if results.displacement is not None and \
            results.displacement.shape[0] != n_nodes:
        raise ValueError(
            f"displacement has {results.displacement.shape[0]} rows but the "
            f"mesh has {n_nodes} nodes; results do not match this mesh."
        )

    # Gauss-point-averaged global stress/strain -> (n_elem, 6)
    sig = np.mean(results.stress_global, axis=1)
    eps = np.mean(results.strain_global, axis=1)

    # Element-averaged von Mises from the averaged stress tensor.
    sxx, syy, szz = sig[:, 0], sig[:, 1], sig[:, 2]
    tyz, txz, txy = sig[:, 3], sig[:, 4], sig[:, 5]
    von_mises = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
    )

    def _fmt(values) -> str:
        return "\n".join(repr(float(v)) for v in np.asarray(values).ravel())

    def _fmt_xyz(rows) -> str:
        return "\n".join(
            f"{float(r[0])!r} {float(r[1])!r} {float(r[2])!r}"
            for r in rows
        )

    lines = [
        "# vtk DataFile Version 3.0",
        "PorosityFE results (hex mesh + per-element fields)",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_nodes} float",
    ]
    lines.append(_fmt_xyz(nodes))

    # CELLS: each line is "8 n0 n1 ... n7"; total size = n_elem * 9.
    lines.append(f"CELLS {n_elem} {n_elem * 9}")
    lines.append(
        "\n".join(
            "8 " + " ".join(str(int(i)) for i in conn) for conn in elements
        )
    )
    lines.append(f"CELL_TYPES {n_elem}")
    lines.append("\n".join("12" for _ in range(n_elem)))

    # ---- POINT_DATA ----
    lines.append(f"POINT_DATA {n_nodes}")
    if results.displacement is not None:
        disp = np.asarray(results.displacement, dtype=float)
        lines.append("VECTORS displacement float")
        lines.append(_fmt_xyz(disp))

    def _point_scalar(name: str, arr) -> None:
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.shape[0] != n_nodes:
            return
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(_fmt(arr))

    if getattr(mesh, 'porosity', None) is not None:
        _point_scalar("porosity", mesh.porosity)
    if getattr(mesh, 'stiffness_reduction', None) is not None:
        _point_scalar("stiffness_reduction", mesh.stiffness_reduction)
    if getattr(mesh, 'ply_ids', None) is not None:
        _point_scalar("ply_id", mesh.ply_ids)

    # ---- CELL_DATA ----
    lines.append(f"CELL_DATA {n_elem}")

    def _cell_scalar(name: str, arr) -> None:
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.shape[0] != n_elem:
            return
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(_fmt(arr))

    _cell_scalar("von_mises", von_mises)
    for idx, comp in enumerate(
            ("sigma_xx", "sigma_yy", "sigma_zz",
             "tau_yz", "tau_xz", "tau_xy")):
        _cell_scalar(comp, sig[:, idx])
    for idx, comp in enumerate(
            ("eps_xx", "eps_yy", "eps_zz",
             "gamma_yz", "gamma_xz", "gamma_xy")):
        _cell_scalar(comp, eps[:, idx])

    if results.per_element_failure_index is not None:
        _cell_scalar("tsai_wu_index", results.per_element_failure_index)

    # Element-averaged nodal porosity over the 8 corner nodes.
    if getattr(mesh, 'porosity', None) is not None:
        vp_elem = np.mean(
            np.asarray(mesh.porosity, dtype=float)[elements], axis=1)
        _cell_scalar("Vp_elem", vp_elem)
    if getattr(mesh, 'elem_ply_ids', None) is not None:
        _cell_scalar("ply_id", mesh.elem_ply_ids)
    if getattr(mesh, 'ply_angles', None) is not None:
        _cell_scalar("ply_angle_deg", mesh.ply_angles)
    if getattr(mesh, 'void_elements', None) is not None:
        is_void = np.zeros(n_elem, dtype=float)
        void_idx = np.asarray(mesh.void_elements, dtype=np.int64).ravel()
        if void_idx.size:
            is_void[void_idx] = 1.0
        _cell_scalar("is_void", is_void)

    _cell_scalar(
        "knockdown",
        np.full(n_elem, float(results.knockdown), dtype=float))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        f.write("\n")
    logger.info("Saved FE results (VTK): %s", filename)


def export_results(field_results: FieldResults,
                   filename: str | os.PathLike,
                   fmt: str = 'json',
                   mesh: CompositeMesh | None = None,
                   include_raw: bool = False) -> None:
    """Export FE results as JSON or VTK; see :meth:`FESolver.export_results`."""
    fmt = str(fmt).lower()
    filename = Path(filename)
    if fmt == 'vtk':
        if mesh is None:
            raise ValueError(
                "export_results(fmt='vtk') requires the `mesh` argument "
                "(pass the CompositeMesh used by the solver)."
            )
        field_results.to_vtk(mesh, filename)
        return
    if fmt != 'json':
        raise ValueError(
            f"Unknown export format {fmt!r}. Use 'json' or 'vtk'."
        )

    def _array_stats(arr: np.ndarray) -> dict:
        """Compute summary statistics for an array."""
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
        }

    results_data = {
        'displacement': {
            'n_nodes': int(field_results.displacement.shape[0]),
            'ux': _array_stats(field_results.displacement[:, 0]),
            'uy': _array_stats(field_results.displacement[:, 1]),
            'uz': _array_stats(field_results.displacement[:, 2]),
        },
        'stress_global': {
            'n_elements': int(field_results.stress_global.shape[0]),
            'n_gauss_points': int(field_results.stress_global.shape[1]),
            'sigma_11': _array_stats(field_results.stress_global[:, :, 0]),
            'sigma_22': _array_stats(field_results.stress_global[:, :, 1]),
            'sigma_33': _array_stats(field_results.stress_global[:, :, 2]),
            'tau_23': _array_stats(field_results.stress_global[:, :, 3]),
            'tau_13': _array_stats(field_results.stress_global[:, :, 4]),
            'tau_12': _array_stats(field_results.stress_global[:, :, 5]),
        },
        'stress_local': {
            'sigma_11': _array_stats(field_results.stress_local[:, :, 0]),
            'sigma_22': _array_stats(field_results.stress_local[:, :, 1]),
            'tau_12': _array_stats(field_results.stress_local[:, :, 5]),
        },
        'strain_global': {
            'eps_11': _array_stats(field_results.strain_global[:, :, 0]),
            'eps_22': _array_stats(field_results.strain_global[:, :, 1]),
            'gamma_12': _array_stats(field_results.strain_global[:, :, 5]),
        },
        'failure': {
            'max_tsai_wu_index': float(field_results.max_failure_index),
            'max_failure_index': float(field_results.max_failure_index),
            'criterion': str(getattr(field_results,
                                      'failure_criterion', 'tsai_wu')),
            'mode_indices': (
                {k: float(v) for k, v in field_results.failure_mode_indices.items()}
                if field_results.failure_mode_indices is not None
                else None
            ),
            'knockdown_factor': float(field_results.knockdown),
        },
    }

    output = {
        'schema_version': JSON_SCHEMA_VERSION,
        'format': FORMAT_FE_FIELDS,
        'provenance': _build_provenance(),
        **results_data,
    }
    if include_raw:
        # Sidecar file path lives next to the JSON so users see them
        # together; ``np.savez`` will append ``.npz`` if missing.
        npz_path = filename.with_name(filename.name + ".npz")
        arrays = {
            'displacement': np.asarray(field_results.displacement),
            'stress_global': np.asarray(field_results.stress_global),
            'stress_local': np.asarray(field_results.stress_local),
            'strain_global': np.asarray(field_results.strain_global),
            'strain_local': np.asarray(field_results.strain_local),
        }
        if field_results.per_element_failure_index is not None:
            arrays['per_element_failure_index'] = np.asarray(
                field_results.per_element_failure_index)
        np.savez(npz_path, **arrays)
        output['raw_sidecar'] = npz_path.name
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Saved FE results: %s", filename)
