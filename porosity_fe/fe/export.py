"""FE result export: legacy ASCII VTK, binary VTU / PVD and JSON summaries.

Split out of ``fe/solver.py`` (IMPROVEMENT_PLAN 5.1). The public entry
points are :meth:`FieldResults.to_vtk
<porosity_fe.fe.solver.FieldResults.to_vtk>`, :meth:`FieldResults.to_vtu
<porosity_fe.fe.solver.FieldResults.to_vtu>` and
:meth:`FESolver.export_results
<porosity_fe.fe.solver.FESolver.export_results>`, which delegate here, and
:func:`write_pvd` for a series of VTU files. All writers are
dependency-free: the files are formatted by hand.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from xml.sax.saxutils import quoteattr

import numpy as np

from ..io import FORMAT_FE_FIELDS, _json_default, _wrap_envelope
from ..mesh import CompositeMesh
from .element import _DEFAULT_FORMULATION
from .recovery import NodalAverage, extrapolate_to_nodes

if TYPE_CHECKING:
    from .solver import FieldResults

logger = logging.getLogger("porosity_fe_analysis")

#: Names of the six Voigt components ``[11, 22, 33, 23, 13, 12]`` in the
#: global frame, as written to VTK / VTU.
_STRESS_NAMES = ("sigma_xx", "sigma_yy", "sigma_zz",
                 "tau_yz", "tau_xz", "tau_xy")
_STRAIN_NAMES = ("eps_xx", "eps_yy", "eps_zz",
                 "gamma_yz", "gamma_xz", "gamma_xy")

#: ``VTK_HEXAHEDRON`` cell type.
_VTK_HEXAHEDRON = 12


def _finite_or_none(value: float | None) -> float | None:
    """JSON-safe float: ``None`` for missing or non-finite values."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _von_mises(sig: np.ndarray) -> np.ndarray:
    """Von Mises stress of ``(..., 6)`` Voigt stress ``[11, 22, 33, 23, 13, 12]``."""
    sxx, syy, szz = sig[..., 0], sig[..., 1], sig[..., 2]
    tyz, txz, txy = sig[..., 3], sig[..., 4], sig[..., 5]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
    )


def _checked_geometry(results: FieldResults, mesh: CompositeMesh,
                      writer: str) -> tuple[np.ndarray, np.ndarray]:
    """Nodes ``(N, 3)`` and hex connectivity ``(E, 8)``, checked against ``results``."""
    nodes = np.asarray(mesh.nodes, dtype=float)
    elements = np.asarray(mesh.elements, dtype=np.int64)
    if elements.shape[1] != 8:
        raise ValueError(
            f"{writer} only supports 8-node hexahedra; got connectivity "
            f"of width {elements.shape[1]}."
        )
    if results.displacement is not None and \
            results.displacement.shape[0] != nodes.shape[0]:
        raise ValueError(
            f"displacement has {results.displacement.shape[0]} rows but the "
            f"mesh has {nodes.shape[0]} nodes; results do not match this mesh."
        )
    return nodes, elements


def _point_fields(results: FieldResults,
                  mesh: CompositeMesh) -> list[tuple[str, np.ndarray]]:
    """Per-node arrays shared by the VTK and VTU writers, in write order.

    ``displacement`` is ``(N, 3)``; the scalars are ``(N,)`` and are
    dropped when their length does not match the node count.
    """
    n_nodes = np.asarray(mesh.nodes).shape[0]
    fields: list[tuple[str, np.ndarray]] = []
    if results.displacement is not None:
        fields.append(("displacement",
                       np.asarray(results.displacement, dtype=float)))
    for name, attr in (("porosity", "porosity"),
                       ("stiffness_reduction", "stiffness_reduction"),
                       ("ply_id", "ply_ids")):
        values = getattr(mesh, attr, None)
        if values is None:
            continue
        arr = np.asarray(values, dtype=float).ravel()
        if arr.shape[0] == n_nodes:
            fields.append((name, arr))
    return fields


def _cell_fields(results: FieldResults,
                 mesh: CompositeMesh) -> list[tuple[str, np.ndarray]]:
    """Per-element ``(E,)`` scalars shared by the VTK and VTU writers, in write order.

    Gauss-point stress and strain are reduced by their mean; arrays whose
    length does not match the element count are dropped.
    """
    elements = np.asarray(mesh.elements, dtype=np.int64)
    n_elem = elements.shape[0]
    sig = np.mean(results.stress_global, axis=1)
    eps = np.mean(results.strain_global, axis=1)

    candidates: list[tuple[str, object]] = [("von_mises", _von_mises(sig))]
    candidates += [(name, sig[:, i]) for i, name in enumerate(_STRESS_NAMES)]
    candidates += [(name, eps[:, i]) for i, name in enumerate(_STRAIN_NAMES)]
    if results.per_element_failure_index is not None:
        candidates.append(("tsai_wu_index", results.per_element_failure_index))
    # Element-averaged nodal porosity over the 8 corner nodes.
    if getattr(mesh, 'porosity', None) is not None:
        candidates.append(("Vp_elem", np.mean(
            np.asarray(mesh.porosity, dtype=float)[elements], axis=1)))
    if getattr(mesh, 'elem_ply_ids', None) is not None:
        candidates.append(("ply_id", mesh.elem_ply_ids))
    if getattr(mesh, 'ply_angles', None) is not None:
        candidates.append(("ply_angle_deg", mesh.ply_angles))
    if getattr(mesh, 'void_elements', None) is not None:
        is_void = np.zeros(n_elem, dtype=float)
        void_idx = np.asarray(mesh.void_elements, dtype=np.int64).ravel()
        if void_idx.size:
            is_void[void_idx] = 1.0
        candidates.append(("is_void", is_void))
    candidates.append(
        ("knockdown", np.full(n_elem, float(results.knockdown), dtype=float)))

    fields = []
    for name, values in candidates:
        arr = np.asarray(values, dtype=float).ravel()
        if arr.shape[0] == n_elem:
            fields.append((name, arr))
    return fields


def write_vtk(results: FieldResults, mesh: CompositeMesh,
              filename: str | os.PathLike) -> None:
    """Write ``results`` on ``mesh`` as legacy ASCII VTK; see :meth:`FieldResults.to_vtk`."""
    filename = Path(filename)
    nodes, elements = _checked_geometry(results, mesh, "to_vtk")
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

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
    lines.append("\n".join(str(_VTK_HEXAHEDRON) for _ in range(n_elem)))

    # ---- POINT_DATA ----
    lines.append(f"POINT_DATA {n_nodes}")
    for name, arr in _point_fields(results, mesh):
        if arr.ndim == 2:
            lines.append(f"VECTORS {name} float")
            lines.append(_fmt_xyz(arr))
        else:
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.append(_fmt(arr))

    # ---- CELL_DATA ----
    lines.append(f"CELL_DATA {n_elem}")
    for name, arr in _cell_fields(results, mesh):
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(_fmt(arr))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        f.write("\n")
    logger.info("Saved FE results (VTK): %s", filename)


#: VTK XML type names of the dtypes the VTU writer emits.
_VTU_TYPES = {
    np.dtype('<f8'): "Float64",
    np.dtype('<f4'): "Float32",
    np.dtype('<i8'): "Int64",
    np.dtype('u1'): "UInt8",
}
#: Byte-count header before every binary block (``header_type="UInt64"``).
_VTU_HEADER = np.dtype('<u8')


def _vtu_array(values: np.ndarray, dtype: str) -> np.ndarray:
    """Contiguous little-endian copy of ``values`` as ``dtype``."""
    return np.ascontiguousarray(np.asarray(values).astype(np.dtype(dtype)))


def write_vtu(results: FieldResults, mesh: CompositeMesh,
              filename: str | os.PathLike, *,
              precision: Literal['float64', 'float32'] = 'float64',
              encoding: Literal['raw', 'base64'] = 'raw',
              nodal: bool = True,
              exploded: bool = False,
              average: NodalAverage = 'ply') -> None:
    """Write ``results`` on ``mesh`` as binary VTK XML; see :meth:`FieldResults.to_vtu`."""
    if precision not in ('float64', 'float32'):
        raise ValueError(
            f"precision must be 'float64' or 'float32', got {precision!r}.")
    if encoding not in ('raw', 'base64'):
        raise ValueError(
            f"encoding must be 'raw' or 'base64', got {encoding!r}.")
    filename = Path(filename)
    nodes, elements = _checked_geometry(results, mesh, "to_vtu")
    n_elem = elements.shape[0]
    fdt = '<f8' if precision == 'float64' else '<f4'

    point_fields = _point_fields(results, mesh)
    if nodal:
        s_nodal, s_corner = extrapolate_to_nodes(
            results.stress_global, mesh, average=average)
        e_nodal, e_corner = extrapolate_to_nodes(
            results.strain_global, mesh, average=average)
        if exploded:
            s_pts = s_corner.reshape(-1, 6)
            e_pts = e_corner.reshape(-1, 6)
        else:
            s_pts, e_pts = s_nodal, e_nodal
        point_fields.append(("von_mises_nodal", _von_mises(s_pts)))
        point_fields += [(f"{name}_nodal", s_pts[:, i])
                         for i, name in enumerate(_STRESS_NAMES)]
        point_fields += [(f"{name}_nodal", e_pts[:, i])
                         for i, name in enumerate(_STRAIN_NAMES)]

    corners = elements.ravel()
    if exploded:
        # Eight private points per cell: per-node mesh fields are gathered,
        # recovered fields come from the per-element corner values.
        points = nodes[corners]
        connectivity = np.arange(corners.size, dtype=np.int64)
        point_fields = [
            (name, arr if arr.shape[0] == corners.size else arr[corners])
            for name, arr in point_fields
        ]
    else:
        points = nodes
        connectivity = corners

    # (section, name, array, components), section in Points / Cells /
    # PointData / CellData.
    arrays: list[tuple[str, str, np.ndarray, int]] = [
        ("Points", "Points", _vtu_array(points, fdt), 3),
        ("Cells", "connectivity", _vtu_array(connectivity, '<i8'), 1),
        ("Cells", "offsets",
         _vtu_array(8 * np.arange(1, n_elem + 1), '<i8'), 1),
        ("Cells", "types",
         _vtu_array(np.full(n_elem, _VTK_HEXAHEDRON), 'u1'), 1),
    ]
    for name, arr in point_fields:
        ncomp = 1 if arr.ndim == 1 else arr.shape[1]
        arrays.append(("PointData", name, _vtu_array(arr, fdt), ncomp))
    if exploded:
        arrays.append(("PointData", "node_id", _vtu_array(corners, '<i8'), 1))
    for name, arr in _cell_fields(results, mesh):
        arrays.append(("CellData", name, _vtu_array(arr, fdt), 1))

    offsets: list[int] = []
    offset = 0
    for _, _, arr, _ in arrays:
        offsets.append(offset)
        offset += _VTU_HEADER.itemsize + arr.nbytes

    def _data_array(i: int) -> str:
        _, name, arr, ncomp = arrays[i]
        attrs = f'type="{_VTU_TYPES[arr.dtype]}" Name={quoteattr(name)}'
        if ncomp > 1:  # VTK's convention: omitted for scalars
            attrs += f' NumberOfComponents="{ncomp}"'
        if encoding == 'raw':
            return (f'<DataArray {attrs} format="appended" '
                    f'offset="{offsets[i]}"/>')
        # Uncompressed inline binary: base64 of (UInt64 byte count + data).
        payload = np.array(arr.nbytes, dtype=_VTU_HEADER).tobytes() + arr.tobytes()
        return (f'<DataArray {attrs} format="binary">'
                f'{base64.b64encode(payload).decode("ascii")}</DataArray>')

    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" '
        'byte_order="LittleEndian" header_type="UInt64">',
        '<UnstructuredGrid>',
        f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_elem}">',
    ]
    for section in ("Points", "Cells", "PointData", "CellData"):
        xml.append(f'<{section}>')
        xml += [_data_array(i) for i, a in enumerate(arrays) if a[0] == section]
        xml.append(f'</{section}>')
    xml += ['</Piece>', '</UnstructuredGrid>']

    with open(filename, 'wb') as f:
        if encoding == 'raw':
            # Appended raw data: offsets count from the byte after "_".
            xml.append('<AppendedData encoding="raw">')
            f.write(("\n".join(xml) + "\n_").encode("ascii"))
            for _, _, arr, _ in arrays:
                f.write(np.array(arr.nbytes, dtype=_VTU_HEADER).tobytes())
                f.write(arr.tobytes())
            f.write(b"\n</AppendedData>\n</VTKFile>\n")
        else:
            xml.append('</VTKFile>')
            f.write(("\n".join(xml) + "\n").encode("ascii"))
    logger.info("Saved FE results (VTU): %s", filename)


def write_pvd(filename: str | os.PathLike,
              files: Sequence[str | os.PathLike],
              timesteps: Iterable[float] | None = None) -> None:
    """Write a ParaView collection (``.pvd``) that groups VTU files into a series.

    Use it to step through several results in ParaView, for example one
    :meth:`FieldResults.to_vtu <porosity_fe.FieldResults.to_vtu>` file per
    porosity level or load case, with the porosity (or any other value) as
    the "time" axis.

    Parameters
    ----------
    filename : str or os.PathLike
        Output ``.pvd`` path.
    files : sequence of str or os.PathLike
        The dataset files, in series order. Paths are written relative to
        the ``.pvd`` file's directory where possible, so the folder can be
        moved as a whole.
    timesteps : iterable of float, optional
        One value per file (for example ``Vp`` as a fraction). Defaults to
        ``0, 1, 2, ...``.

    Raises
    ------
    ValueError
        If ``timesteps`` does not have one finite value per file.

    Examples
    --------
    One file per porosity level, stepped through by ``Vp``:

    >>> paths = []
    >>> for vp, result in zip(levels, results):    # your own FE solves
    ...     path = f"vp_{vp:.3f}.vtu"
    ...     result.to_vtu(mesh, path)
    ...     paths.append(path)
    >>> write_pvd("sweep.pvd", paths, timesteps=levels)
    """
    filename = Path(filename)
    paths = [Path(p) for p in files]
    steps = [float(t) for t in (range(len(paths)) if timesteps is None
                                else timesteps)]
    if len(steps) != len(paths) or not all(np.isfinite(steps)):
        raise ValueError(
            f"timesteps must give one finite value per file: got "
            f"{len(steps)} values for {len(paths)} files."
        )
    base = filename.resolve().parent
    rows = []
    for step, path in zip(steps, paths, strict=True):
        try:
            rel = Path(os.path.relpath(path.resolve(), base)).as_posix()
        except ValueError:  # on another drive (Windows): keep it absolute
            rel = path.resolve().as_posix()
        rows.append(f'<DataSet timestep="{step!r}" group="" part="0" '
                    f'file={quoteattr(rel)}/>')
    text = "\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
        '<Collection>',
        *rows,
        '</Collection>',
        '</VTKFile>',
    ]) + "\n"
    filename.write_text(text, encoding="utf-8")
    logger.info("Saved VTU series (PVD): %s", filename)


def export_results(field_results: FieldResults,
                   filename: str | os.PathLike,
                   fmt: str = 'json',
                   mesh: CompositeMesh | None = None,
                   include_raw: bool = False) -> None:
    """Export FE results as JSON, VTK or VTU; see :meth:`FESolver.export_results`."""
    fmt = str(fmt).lower()
    filename = Path(filename)
    if fmt in ('vtk', 'vtu'):
        if mesh is None:
            raise ValueError(
                f"export_results(fmt={fmt!r}) requires the `mesh` argument "
                "(pass the CompositeMesh used by the solver)."
            )
        if fmt == 'vtk':
            field_results.to_vtk(mesh, filename)
        else:
            field_results.to_vtu(mesh, filename)
        return
    if fmt != 'json':
        raise ValueError(
            f"Unknown export format {fmt!r}. Use 'json', 'vtk' or 'vtu'."
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
            'first_ply_failure_load_factor': _finite_or_none(
                field_results.first_ply_failure_load_factor),
        },
    }
    results_data['solver'] = {
        'formulation': str(getattr(field_results, 'formulation',
                                  _DEFAULT_FORMULATION)),
    }
    if field_results.reaction_forces is not None:
        results_data['stiffness'] = {
            'effective_modulus_MPa': (
                float(field_results.effective_modulus)
                if field_results.effective_modulus is not None else None),
            'reaction_force_sum_N': [
                float(v) for v in np.sum(field_results.reaction_forces, axis=0)],
        }

    output = _wrap_envelope(FORMAT_FE_FIELDS, None, results_data)
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
