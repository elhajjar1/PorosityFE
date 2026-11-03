"""Fingerprints of the FESolver results: bitwise, and a numeric CI gate.

Usage::

    python scripts/fingerprint.py <repo_root> <out.json> [--quick]
    python scripts/fingerprint.py --compare <a.json> <b.json>
    python scripts/fingerprint.py --check <baseline.json> [--repo <root>]
    python scripts/fingerprint.py --write-baseline <baseline.json> [--repo <root>]

**Bitwise** (the first two forms). Hashes (sha256) every ``FieldResults``
field (each array's dtype, shape and raw bytes; each scalar's exact repr;
dicts and nested dataclasses recursively), ``FieldResults.summary()`` and the
export JSON with the provenance block removed (``provenance.delta_T_K``
kept). The cases are the production meshes of AS4/3501-6 (the preset with
CTEs) at 3 % clustered porosity: the 30 x 10 x 24 coupon (compression,
tension, shear, thermal, and compression / tension with ``delta_T``),
``for_flexure`` (3pt, 4pt, and 3pt with ``delta_T``) and ``for_ilss`` (ILSS,
and with ``delta_T``); formulations ``hex8i`` and ``hex8``; criteria tsai_wu
/ hashin / max_stress; solvers direct and cg: 132 cases. ``--quick`` keeps
hex8i, direct and nine loadings: 27 cases. One solver per (mesh,
formulation), so the LU / thermal / pristine caches are exercised the way
repeat solves use them. Use it to show that a change moves no bit: run both
sides on one machine with the same wheels and the same BLAS thread settings
(``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1``; the thread count changes the
last digits). ``<repo_root>`` is put first on ``sys.path`` so an exported
tree (``git archive``) can be fingerprinted next to an editable install.

**Numeric gate** (the last two forms; issue #214, run by CI). CPUs (the
BLAS kernel OpenBLAS picks for them), thread counts and numpy / scipy
releases move the last digits, so the gate records a numeric digest per
case instead of hashes: every scalar; for every float array its shape, sum,
sum of ``|x|``, 2-norm, max, min and 16 entries at fixed positions; integer
and boolean arrays (as a hash) and strings exactly; the export JSON minus
provenance, leaf by leaf. Two floats ``a``, ``b`` agree when
``|a - b| <= rtol * max(|a|, |b|) + noise * scale``, where ``scale`` is the
magnitude of the values of the same kind beside it: the array's ``max |x|``
for its max, min and entries, its sum of ``|x|`` for its two sums, its
2-norm for the 2-norm; in a dict of results the largest ``|x|`` among its
siblings; in the export JSON the largest ``|x|`` of its section (a stress
component on the scale of the whole stress tensor's statistics, ``uz`` on
that of the displacements). Top-level results and ``summary()`` mix units
and are compared relatively only. The noise term absorbs quantities that
are small by cancellation or zero up to round-off beside others of their
kind: a stress mean of 1e-12 MPa beside a maximum of 358 MPa, ``tau_23``
under in-plane shear. Defaults ``rtol = 1e-9`` and ``noise = 1e-9``
(``--rtol``, ``--noise``).

Some fields are round-off through and through, with nothing of their kind
beside them: the reactions of a free thermal expansion and of the DCB (on
statically determinate supports they are zero by equilibrium), the mode II
/ III rates and mixity of a pure mode I front, and the load factors of front
nodes the load hardly drives (``lambda_c ~ 1e12``). Each case names these in
:data:`_CASE_OPTIONS`, with the reason, and the comparison skips them. The
DCB is also ill-conditioned (slender arms on determinate supports): its
physical results move by up to 2e-8 between BLAS kernels, so it is compared
at 1e-6.

Measured against the single-threaded baseline (made on SkylakeX kernels):
4 BLAS threads, numpy 1.26 / scipy 1.13, and OpenBLAS forced onto the
Haswell and Sandybridge kernels (``OPENBLAS_CORETYPE``) all stay below 0.2
of the tolerance (the flexure knockdowns, 1.5e-10 relative, are the
closest). The CI job installs the numpy and scipy the baseline was
made with (its ``meta.environment``; ``--check`` notes a difference) and
runs the check twice, on the runner's own kernel and on Sandybridge's, so a
new case that depends on the kernel fails on its first PR.

The gate set (30 cases, about two and a half minutes on one thread): hex8i,
direct, the quick set's nine loadings and three criteria except in-plane
shear with Hashin (``sigma_11`` is zero up to round-off there and its sign
switches the fibre mode between machines), and four defect cases:
``interface`` penny voids on the production coupon (the renormalized
profile), the ``'directional'`` degradation on the ILSS specimen, a strip
delamination in a 20 x 6 x 24 coupon under compression with a thermal
prestress (crack-face contact, VCCT, the load factor and its seeded
uncertainty band), and an 8-ply DCB. The fracture cases use fixture
toughness values, not material data (no preset carries toughness).

A change that moves the gate on purpose updates the baseline in the same PR
(``--write-baseline``, single-threaded) and says why in the CHANGELOG.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time

DELTA_T = -157.0
VP = 0.03
RTOL = 1e-9
NOISE = 1e-9
N_SAMPLES = 16

#: Fixture toughness for the gate's fracture cases (N/mm): typical carbon /
#: epoxy magnitudes, not material data.
GATE_TOUGHNESS = {'G_Ic': 0.2, 'G_IIc': 0.6, 'bk_eta': 2.0}


# --------------------------------------------------------------------------
# Bitwise fingerprint
# --------------------------------------------------------------------------

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


_LOADINGS = [
    ('coupon', 'compression', None), ('coupon', 'tension', None),
    ('coupon', 'shear', None), ('coupon', 'thermal', DELTA_T),
    ('coupon', 'compression', DELTA_T), ('coupon', 'tension', DELTA_T),
    ('flexure', 'flexure_3pt', None), ('flexure', 'flexure_4pt', None),
    ('flexure', 'flexure_3pt', DELTA_T),
    ('ilss', 'ilss', None), ('ilss', 'ilss', DELTA_T),
]
_CRITERIA = ('tsai_wu', 'hashin', 'max_stress')


def _quick_loadings():
    return [c for c in _LOADINGS
            if c[1:] not in (('flexure_3pt', DELTA_T), ('ilss', DELTA_T))]


def _cases(quick: bool):
    formulations = ('hex8i',) if quick else ('hex8i', 'hex8')
    solvers = ('direct',) if quick else ('direct', 'cg')
    loadings = _quick_loadings() if quick else _LOADINGS
    for formulation in formulations:
        for mesh_name, loading, dT in loadings:
            for solver in solvers:
                for crit in _CRITERIA:
                    yield formulation, mesh_name, loading, dT, solver, crit


def _case_name(formulation, mesh_name, loading, dT, solver, crit) -> str:
    return (f"{formulation}/{mesh_name}/{loading}{'' if dT is None else f'+dT{dT:g}'}"
            f"/{solver}/{crit}")


def _import(repo_root: str):
    sys.path.insert(0, os.path.abspath(repo_root))
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.simplefilter('ignore')
    import porosity_fe
    assert os.path.abspath(porosity_fe.__file__).startswith(os.path.abspath(repo_root)), \
        porosity_fe.__file__
    return porosity_fe


def _production_meshes(pfe):
    mat = pfe.MATERIALS['AS4_3501_6_epoxy']
    pf = pfe.PorosityField(mat, VP, distribution='clustered')
    meshes = {
        'coupon': lambda: pfe.CompositeMesh(pf, mat, nx=30, ny=10, nz=mat.n_plies),
        'flexure': lambda: pfe.CompositeMesh.for_flexure(pf, mat),
        'ilss': lambda: pfe.CompositeMesh.for_ilss(pf, mat),
    }
    return mat, pf, meshes


def _export_doc(pfe, res, tmp: str) -> dict:
    path = os.path.join(tmp, 'x.json')
    pfe.FESolver.export_results(res, path)
    with open(path, encoding='utf-8') as fh:
        doc = json.load(fh)
    prov = doc.pop('provenance', {})
    if 'delta_T_K' in prov:
        doc['provenance_delta_T_K'] = prov['delta_T_K']
    return doc


def run(repo_root: str, out: str, quick: bool) -> None:
    pfe = _import(repo_root)
    mat, pf, meshes = _production_meshes(pfe)
    solvers: dict = {}
    results = {}
    t_all = time.time()
    tmp = tempfile.mkdtemp()
    for formulation, mesh_name, loading, dT, solver_name, crit in _cases(quick):
        key = (mesh_name, formulation)
        if key not in solvers:
            solvers[key] = pfe.FESolver(meshes[mesh_name](), mat, pf, formulation=formulation)
        fe = solvers[key]
        t0 = time.time()
        res = fe.solve(loading, delta_T=dT, solver=solver_name, failure_criterion=crit)
        name = _case_name(formulation, mesh_name, loading, dT, solver_name, crit)
        fields = {}
        for f in dataclasses.fields(res):
            fields[f.name] = _digest(getattr(res, f.name))
        fields['summary'] = _digest(res.summary())
        doc = _export_doc(pfe, res, tmp)
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


# --------------------------------------------------------------------------
# Numeric gate
# --------------------------------------------------------------------------

def _num(x: float):
    """A float as JSON: itself when finite, else its repr ('nan', 'inf')."""
    x = float(x)
    return x if math.isfinite(x) else repr(x)


def _numeric(obj):
    """A JSON-able numeric digest of a result (see the module docstring)."""
    import numpy as np
    if obj is None or isinstance(obj, (str, bool, np.bool_)):
        return obj if not isinstance(obj, np.bool_) else bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        d: dict = {'__array__': a.dtype.str, 'shape': list(a.shape)}
        if a.dtype.kind != 'f' or a.size == 0:
            d['sha'] = _digest(a)
            return d
        flat = a.ravel().astype(np.float64)
        finite = np.isfinite(flat)
        if not finite.all():
            d['nonfinite'] = _digest(np.where(finite, 0.0, flat))
            flat = np.where(finite, flat, 0.0)
        d.update(sum=float(flat.sum()), abs_sum=float(np.abs(flat).sum()),
                 l2=float(np.sqrt(np.dot(flat, flat))), max=float(flat.max()),
                 min=float(flat.min()))
        idx = np.unique(np.linspace(0, flat.size - 1, N_SAMPLES).round().astype(int))
        d['samples'] = [float(v) for v in flat[idx]]
        return d
    if isinstance(obj, dict):
        return {str(k): _numeric(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_numeric(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return {f.name: _numeric(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return repr(obj)


def _sibling_scale(node) -> float:
    """The largest |x| among the numbers directly in a dict (and in the
    lists directly in it)."""
    vals: list[float] = []
    items = node.values() if isinstance(node, dict) else node
    for v in items:
        if isinstance(v, float):
            vals.append(abs(v))
        elif isinstance(v, list):
            vals.extend(abs(x) for x in v if isinstance(x, float))
    return max(vals, default=0.0)


def _norm(path: str) -> str:
    """A digest path without its list indices: ``parts[0].G_II.samples[3]``
    -> ``parts.G_II.samples``."""
    return re.sub(r'\[\d+\]', '', path)


def _skipped(rel: str, skip) -> bool:
    rel = _norm(rel)
    return any(rel == s or rel.startswith(s + '.') for s in skip)


def _float_scale(node, rel: str, skip) -> float:
    """The largest |x| of the floats in ``node``, recursively, leaving out
    the skipped paths (``rel`` is ``node``'s path within the case)."""
    if _skipped(rel, skip):
        return 0.0
    if isinstance(node, float):
        return abs(node)
    if isinstance(node, dict):
        return max((_float_scale(v, f"{rel}.{k}", skip) for k, v in node.items()), default=0.0)
    if isinstance(node, list):
        return max((_float_scale(v, f"{rel}[{i}]", skip) for i, v in enumerate(node)),
                   default=0.0)
    return 0.0


class _Comparison:
    """Walks two digests; records mismatches and the largest deviation in
    units of the tolerance."""

    def __init__(self, rtol: float, noise: float):
        self.rtol, self.noise = rtol, noise
        self.errors: list[str] = []
        self.worst = (0.0, '')
        self.n_skipped = 0
        self._skip: tuple = ()
        self._root = ''

    def _float(self, path, a, b, scale):
        tol = self.rtol * max(abs(a), abs(b)) + self.noise * scale
        dev = abs(a - b)
        ratio = dev / tol if tol > 0 else (math.inf if dev else 0.0)
        if ratio > self.worst[0]:
            self.worst = (ratio, path)
        if ratio > 1.0:
            self.errors.append(f"{path}: {a!r} -> {b!r} (|diff| {dev:.3g}, tol {tol:.3g})")

    def _array(self, path, a, b):
        if a.keys() != b.keys() or a['__array__'] != b['__array__'] or a['shape'] != b['shape']:
            self.errors.append(f"{path}: array layout {a.get('__array__')}{a.get('shape')} "
                               f"-> {b.get('__array__')}{b.get('shape')}")
            return
        for k in ('sha', 'nonfinite'):
            if a.get(k) != b.get(k):
                self.errors.append(f"{path}.{k}: {a.get(k)} -> {b.get(k)}")
        if 'sum' not in a:
            return
        big = max(abs(a['max']), abs(a['min']), abs(b['max']), abs(b['min']))
        for k, scale in (('sum', a['abs_sum']), ('abs_sum', a['abs_sum']),
                         ('l2', a['l2']), ('max', big), ('min', big)):
            self._float(f"{path}.{k}", a[k], b[k], scale)
        if len(a['samples']) != len(b['samples']):
            self.errors.append(f"{path}.samples: length")
            return
        for i, (x, y) in enumerate(zip(a['samples'], b['samples'], strict=True)):
            self._float(f"{path}.samples[{i}]", x, y, big)

    def walk(self, path, a, b, scale=0.0, siblings=True):
        if self._skip and _skipped(path[len(self._root):], self._skip):
            self.n_skipped += 1
            return
        if isinstance(a, float) and isinstance(b, float):
            self._float(path, a, b, scale)
        elif isinstance(a, dict) and isinstance(b, dict):
            if '__array__' in a or '__array__' in b:
                if '__array__' in a and '__array__' in b:
                    self._array(path, a, b)
                else:
                    self.errors.append(f"{path}: array vs non-array")
                return
            if a.keys() != b.keys():
                self.errors.append(f"{path}: keys {sorted(set(a) ^ set(b))} differ")
            s = max(scale, _sibling_scale(a), _sibling_scale(b)) if siblings else 0.0
            for k in sorted(set(a) & set(b)):
                self.walk(f"{path}.{k}", a[k], b[k], s)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.errors.append(f"{path}: length {len(a)} -> {len(b)}")
                return
            s = max(scale, _sibling_scale(a), _sibling_scale(b))
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                self.walk(f"{path}[{i}]", x, y, s)
        elif isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) \
                and not isinstance(b, bool) and a == b:
            return
        elif a != b or type(a) is not type(b):
            self.errors.append(f"{path}: {a!r} -> {b!r}")

    def case(self, name, a, b, options: dict | None = None):
        """Compare one case. ``options`` (from :func:`_gate_cases`) may set
        its ``rtol`` and ``noise`` and name the paths it ``skip``s."""
        options = options or {}
        rtol, noise = self.rtol, self.noise
        self.rtol, self.noise = options.get('rtol', rtol), options.get('noise', noise)
        self._skip, self._root = tuple(options.get('skip', ())), f"{name}."
        try:
            # Top-level results and summary() mix units: relative only there.
            for part in ('fields', 'summary'):
                pa, pb = a.get(part), b.get(part)
                if isinstance(pa, dict) and isinstance(pb, dict):
                    if pa.keys() != pb.keys():
                        self.errors.append(f"{name}.{part}: keys {sorted(set(pa) ^ set(pb))} differ")
                    for k in sorted(set(pa) & set(pb)):
                        self.walk(f"{name}.{part}.{k}", pa[k], pb[k])
                else:
                    self.walk(f"{name}.{part}", pa, pb, siblings=False)
            # Each export section holds one kind of quantity (the stress
            # tensor's statistics, the displacement components, ...): a
            # component is compared on the scale of its whole section.
            ea, eb = a.get('export'), b.get('export')
            if isinstance(ea, dict) and isinstance(eb, dict):
                if ea.keys() != eb.keys():
                    self.errors.append(f"{name}.export: keys {sorted(set(ea) ^ set(eb))} differ")
                for k in sorted(set(ea) & set(eb)):
                    s = max(_float_scale(ea[k], f"export.{k}", self._skip),
                            _float_scale(eb[k], f"export.{k}", self._skip))
                    self.walk(f"{name}.export.{k}", ea[k], eb[k], s)
            elif ea is not None or eb is not None:
                self.walk(f"{name}.export", ea, eb)
            for part in sorted(set(a) | set(b)):
                if part not in ('fields', 'summary', 'export'):
                    self.walk(f"{name}.{part}", a.get(part), b.get(part))
        finally:
            self.rtol, self.noise = rtol, noise
            self._skip, self._root = (), ''


#: Fields that are zero by equilibrium or symmetry in a case: their digits
#: are round-off, which another BLAS kernel changes by more than their own
#: size, and nothing of the same kind sits beside them to scale them by.
#: Paths are within the case, list indices dropped.
_REACTIONS = ('fields.reaction_forces', 'export.stiffness.reaction_force_sum_N')
_MODE_I_FRONT = ('fields.crack_front.G_II', 'fields.crack_front.G_III', 'fields.crack_front.psi',
                 'fields.crack_front.parts.G_II', 'fields.crack_front.parts.G_III')
_UNDRIVEN_FACTORS = ('fields.crack_front.lambda_c', 'export.fracture.front.lambda_c',
                     'extras.fracture_band.nominal_per_node')

#: Per case: tolerances other than the defaults, the skipped paths, and why.
_CASE_OPTIONS = {
    'thermal': {
        'skip': _REACTIONS,
        'why': "free thermal expansion on the statically determinate (3-2-1) supports: "
               "the reactions are zero by equilibrium"},
    'strip_delamination': {
        'skip': _MODE_I_FRONT + _UNDRIVEN_FACTORS,
        'why': "a through-width strip front is in mode I (its mode II / III rates and "
               "mixity are round-off), and at front nodes the load hardly drives "
               "(G_T ~ 1e-24) lambda_c ~ 1e12 is set by round-off; the governing "
               "factor, G_I, G_T, G_c and the band are compared"},
    'dcb': {
        'skip': _REACTIONS + _MODE_I_FRONT, 'rtol': 1e-6, 'noise': 1e-6,
        'why': "the six supports are statically determinate and carry no load, the front "
               "is in pure mode I, and the slender arms on determinate supports make the "
               "system ill-conditioned: its results move by up to 2e-8 between BLAS "
               "kernels, so it is compared at 1e-6"},
}


def _gate_cases(pfe):
    """Yield ``(name, thunk, options)`` for the gate set; each thunk returns
    ``(FieldResults, extras dict)``, ``options`` is the case's entry of
    :data:`_CASE_OPTIONS` (or empty)."""
    mat, pf, meshes = _production_meshes(pfe)
    solvers: dict = {}

    def production(mesh_name, loading, dT, crit):
        def go():
            if mesh_name not in solvers:
                solvers[mesh_name] = pfe.FESolver(meshes[mesh_name](), mat, pf)
            return solvers[mesh_name].solve(loading, delta_T=dT, failure_criterion=crit), {}
        return go

    for mesh_name, loading, dT in _quick_loadings():
        for crit in _CRITERIA:
            if (loading, crit) == ('shear', 'hashin'):
                continue
            yield _case_name('hex8i', mesh_name, loading, dT, 'direct', crit), \
                production(mesh_name, loading, dT, crit), _CASE_OPTIONS.get(loading, {})

    tough = dataclasses.replace(mat, **GATE_TOUGHNESS)

    def interface_penny():
        pfi = pfe.PorosityField(mat, VP, distribution='interface', void_shape='penny')
        mesh = pfe.CompositeMesh(pfi, mat, nx=30, ny=10, nz=mat.n_plies)
        return pfe.FESolver(mesh, mat, pfi).solve('compression', failure_criterion='tsai_wu'), \
            {'porosity_renormalization': mesh.porosity_renormalization}

    def directional_ilss():
        pfp = pfe.PorosityField(mat, VP, void_shape='penny')
        mesh = pfe.CompositeMesh.for_ilss(pfp, mat)
        return pfe.FESolver(mesh, mat, pfp, porosity_degradation='directional').solve(
            'ilss', failure_criterion='hashin'), {}

    def strip_delamination():
        pfd = pfe.PorosityField(tough, VP, distribution='clustered')
        k = tough.n_plies // 2 - 1
        mesh = pfe.CompositeMesh(pfd, tough, nx=20, ny=6, nz=tough.n_plies,
                                 delaminations=[pfe.Delamination.strip(k, 15.0, 35.0)])
        res = pfe.FESolver(mesh, tough, pfd).solve('compression', delta_T=DELTA_T,
                                                   failure_criterion='tsai_wu')
        band = res.fracture_load_factor_uncertainty(
            tough, covs={'G_Ic': 0.1, 'G_IIc': 0.15, 'bk_eta': 0.1},
            n_samples=500, method='lhs', seed=0)
        return res, {'fracture_band': band.summary()}

    def dcb():
        thin = dataclasses.replace(tough, n_plies=8)
        pf0 = pfe.PorosityField(thin, 0.0)
        mesh = pfe.CompositeMesh.for_dcb(pf0, thin, ny=2)
        return pfe.FESolver(mesh, thin, pf0).solve('dcb', failure_criterion='tsai_wu'), {}

    yield 'defect/interface_penny/compression/tsai_wu', interface_penny, {}
    yield 'defect/directional_penny/ilss/hashin', directional_ilss, {}
    yield ('defect/strip_delamination/compression+dT-157/tsai_wu', strip_delamination,
           _CASE_OPTIONS['strip_delamination'])
    yield 'defect/dcb/dcb/tsai_wu', dcb, _CASE_OPTIONS['dcb']


def _case_digest(pfe, res, extras: dict, tmp: str) -> dict:
    """The numeric digest of one gate case: every ``FieldResults`` field,
    ``summary()``, the export JSON minus provenance, and the extras."""
    digest = {
        'fields': {f.name: _numeric(getattr(res, f.name)) for f in dataclasses.fields(res)},
        'summary': _numeric(res.summary()),
        'export': _numeric(_export_doc(pfe, res, tmp)),
    }
    if extras:
        digest['extras'] = _numeric(extras)
    return digest


def gate_options(pfe) -> dict:
    """``{case name: options}`` of the gate set, without solving."""
    return {name: options for name, _thunk, options in _gate_cases(pfe)}


def _gate_run(repo_root: str, verbose: bool = True) -> dict:
    pfe = _import(repo_root)
    tmp = tempfile.mkdtemp()
    cases = {}
    t_all = time.time()
    for name, thunk, _options in _gate_cases(pfe):
        t0 = time.time()
        res, extras = thunk()
        cases[name] = _case_digest(pfe, res, extras, tmp)
        if verbose:
            print(f"{time.time() - t0:6.2f}s {name}", flush=True)
    if verbose:
        print(f"total {time.time() - t_all:.1f}s, {len(cases)} cases")
    return cases


def _environment() -> dict:
    import numpy
    import scipy
    return {
        'python': platform.python_version(), 'numpy': numpy.__version__,
        'scipy': scipy.__version__, 'platform': platform.platform(),
        'machine': platform.machine(),
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    }


def write_baseline(path: str, repo_root: str) -> int:
    cases = _gate_run(repo_root)
    doc = {'meta': {'environment': _environment(), 'n_cases': len(cases)},
           'cases': cases}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(f"baseline -> {path}")
    return 0


def check(path: str, repo_root: str, rtol: float, noise: float) -> int:
    with open(path, encoding='utf-8') as fh:
        base = json.load(fh)
    cases = _gate_run(repo_root)
    return report(base, cases, rtol, noise, gate_options(sys.modules['porosity_fe']))


def report(base: dict, cases: dict, rtol: float, noise: float,
           options: dict | None = None) -> int:
    """Compare ``cases`` with the baseline ``base`` and print the result;
    ``options`` maps a case name to its :data:`_CASE_OPTIONS` entry.
    Returns the exit code."""
    cmp = _Comparison(rtol, noise)
    options = options or {}
    expected = base['cases']
    for name in sorted(set(expected) | set(cases)):
        if name not in cases or name not in expected:
            cmp.errors.append(f"{name}: {'missing' if name not in cases else 'not in the baseline'}")
            continue
        cmp.case(name, expected[name], cases[name], options.get(name))
    env, here = base['meta']['environment'], _environment()
    print(f"baseline environment: {env}")
    print(f"this environment:     {here}")
    for lib in ('numpy', 'scipy'):
        if env.get(lib) != here[lib]:
            print(f"NOTE: {lib} {here[lib]} here, {env.get(lib)} in the baseline; the "
                  "round-off fields (see the docstring) may differ")
    ratio, where = cmp.worst
    print(f"{len(cases)} cases, rtol {rtol:g}, noise {noise:g} (per-case overrides: "
          f"{sum(1 for o in options.values() if 'rtol' in o)}); {cmp.n_skipped} round-off "
          f"values skipped; largest deviation {ratio:.3g} of the tolerance at {where or '-'}")
    for e in cmp.errors[:200]:
        print(f"MISMATCH {e}")
    if len(cmp.errors) > 200:
        print(f"... {len(cmp.errors) - 200} more")
    if cmp.errors:
        print(f"FAIL: {len(cmp.errors)} mismatches. If the change is intended, regenerate "
              "the baseline in the same PR (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "
              "scripts/fingerprint.py --write-baseline scripts/fingerprint_baseline.json) "
              "and say why in the CHANGELOG.")
        return 1
    print("OK")
    return 0


def _opt(argv: list[str], flag: str, default):
    if flag in argv:
        i = argv.index(flag)
        return type(default)(argv[i + 1]) if default is not None else argv[i + 1]
    return default


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    if argv[0] == '--compare':
        return compare(argv[1], argv[2])
    if argv[0] == '--check':
        return check(argv[1], _opt(argv, '--repo', '.'), _opt(argv, '--rtol', RTOL),
                     _opt(argv, '--noise', NOISE))
    if argv[0] == '--write-baseline':
        return write_baseline(argv[1], _opt(argv, '--repo', '.'))
    run(argv[0], argv[1], quick='--quick' in argv[2:])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
