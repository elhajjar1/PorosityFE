"""Bit-identity fingerprint of FESolver results.

Usage::

    python fingerprint.py <repo_root> <out.json> [--quick]
    python fingerprint.py --compare <a.json> <b.json>

Hashes (sha256) every ``FieldResults`` field (each array's dtype, shape and
raw bytes; each scalar's exact repr; dicts and nested dataclasses
recursively), ``FieldResults.summary()`` and the export JSON with the
provenance block removed (``provenance.delta_T_K`` kept). The cases are the
production meshes of AS4/3501-6 (the preset with CTEs) at 3 % clustered
porosity: the 30 x 10 x 24 coupon (compression, tension, shear, thermal,
and compression / tension with ``delta_T``), ``for_flexure`` (3pt, 4pt, and
3pt with ``delta_T``) and ``for_ilss`` (ILSS, and with ``delta_T``);
formulations ``hex8i`` and ``hex8``; criteria tsai_wu / hashin /
max_stress; solvers direct and cg: 132 cases. ``--quick`` keeps hex8i,
direct and nine loadings: 27 cases. One solver per (mesh, formulation), so
the LU / thermal / pristine caches are exercised the way repeat solves use
them.

Run both sides with the same BLAS thread settings
(``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1``): the thread count changes
the last digits. ``<repo_root>`` is put first on ``sys.path`` so an
exported tree (``git archive``) can be fingerprinted next to an editable
install.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time

DELTA_T = -157.0
VP = 0.03


def _h(obj, hasher) -> None:
    import numpy as np
    if obj is None:
        hasher.update(b'None')
    elif isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        hasher.update(f"nd{a.dtype.str}{a.shape}".encode())
        hasher.update(a.tobytes())
    elif isinstance(obj, (bool, np.bool_)):
        hasher.update(f"b{bool(obj)}".encode())
    elif isinstance(obj, (int, np.integer)):
        hasher.update(f"i{int(obj)}".encode())
    elif isinstance(obj, (float, np.floating)):
        hasher.update(f"f{float(obj)!r}".encode())
    elif isinstance(obj, str):
        hasher.update(f"s{obj}".encode())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            hasher.update(f"k{k}".encode())
            _h(obj[k], hasher)
    elif isinstance(obj, (list, tuple)):
        hasher.update(f"l{len(obj)}".encode())
        for v in obj:
            _h(v, hasher)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            hasher.update(f"d{f.name}".encode())
            _h(getattr(obj, f.name), hasher)
    else:
        hasher.update(f"r{obj!r}".encode())


def _digest(obj) -> str:
    hasher = hashlib.sha256()
    _h(obj, hasher)
    return hasher.hexdigest()[:16]


def _scalars(result) -> dict:
    """A few readable scalars so a diff says what moved."""
    s = result.summary()
    return {
        'knockdown': repr(result.knockdown),
        'E_eff': repr(result.effective_modulus),
        'max_fi': repr(result.max_failure_index),
        'fpf': repr(result.first_ply_failure_load_factor),
        'summary_knockdown': repr(s.knockdown),
    }


def _cases(quick: bool):
    loadings = [
        ('coupon', 'compression', None), ('coupon', 'tension', None),
        ('coupon', 'shear', None), ('coupon', 'thermal', DELTA_T),
        ('coupon', 'compression', DELTA_T), ('coupon', 'tension', DELTA_T),
        ('flexure', 'flexure_3pt', None), ('flexure', 'flexure_4pt', None),
        ('flexure', 'flexure_3pt', DELTA_T),
        ('ilss', 'ilss', None), ('ilss', 'ilss', DELTA_T),
    ]
    formulations = ('hex8i',) if quick else ('hex8i', 'hex8')
    solvers = ('direct',) if quick else ('direct', 'cg')
    if quick:
        loadings = [c for c in loadings if c[1:] not in (('flexure_3pt', DELTA_T), ('ilss', DELTA_T))]
    criteria = ('tsai_wu', 'hashin', 'max_stress')
    for formulation in formulations:
        for mesh_name, loading, dT in loadings:
            for solver in solvers:
                for crit in criteria:
                    yield formulation, mesh_name, loading, dT, solver, crit


def run(repo_root: str, out: str, quick: bool) -> None:
    sys.path.insert(0, os.path.abspath(repo_root))
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.simplefilter('ignore')
    import porosity_fe
    from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
    assert os.path.abspath(porosity_fe.__file__).startswith(os.path.abspath(repo_root)), porosity_fe.__file__

    mat = MATERIALS['AS4_3501_6_epoxy']
    pf = PorosityField(mat, VP, distribution='clustered')
    meshes = {
        'coupon': lambda: CompositeMesh(pf, mat, nx=30, ny=10, nz=mat.n_plies),
        'flexure': lambda: CompositeMesh.for_flexure(pf, mat),
        'ilss': lambda: CompositeMesh.for_ilss(pf, mat),
    }
    solvers: dict = {}
    results = {}
    t_all = time.time()
    tmp = tempfile.mkdtemp()
    for formulation, mesh_name, loading, dT, solver_name, crit in _cases(quick):
        key = (mesh_name, formulation)
        if key not in solvers:
            solvers[key] = FESolver(meshes[mesh_name](), mat, pf, formulation=formulation)
        fe = solvers[key]
        t0 = time.time()
        res = fe.solve(loading, delta_T=dT, solver=solver_name, failure_criterion=crit)
        name = f"{formulation}/{mesh_name}/{loading}{'' if dT is None else f'+dT{dT:g}'}/{solver_name}/{crit}"
        fields = {}
        for f in dataclasses.fields(res):
            fields[f.name] = _digest(getattr(res, f.name))
        fields['summary'] = _digest(res.summary())
        path = os.path.join(tmp, 'x.json')
        FESolver.export_results(res, path)
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
        prov = doc.pop('provenance', {})
        if 'delta_T_K' in prov:
            doc['provenance_delta_T_K'] = prov['delta_T_K']
        fields['export_json'] = hashlib.sha256(
            json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]
        results[name] = {'fields': fields, 'scalars': _scalars(res)}
        print(f"{time.time() - t0:6.2f}s {name} {fields['displacement']}", flush=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    print(f"total {time.time() - t_all:.1f}s, {len(results)} cases -> {out}")


def compare(a: str, b: str) -> int:
    with open(a, encoding='utf-8') as fh:
        A = json.load(fh)
    with open(b, encoding='utf-8') as fh:
        B = json.load(fh)
    common = sorted(set(A) & set(B))
    n_diff = 0
    for name in common:
        fa, fb = A[name]['fields'], B[name]['fields']
        diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        if diff:
            n_diff += 1
            print(f"DIFF {name} {diff} {A[name]['scalars']} {B[name]['scalars']}")
    missing = sorted(set(A) ^ set(B))
    for name in missing:
        print(f"MISSING {name}")
    print(f"{len(common)} common cases, {n_diff + len(missing)} differing/missing")
    return 1 if n_diff or missing else 0


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1], sys.argv[2], quick='--quick' in sys.argv[3:])
