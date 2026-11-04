"""The register of deprecations removed in 2.0 (#218).

``REGISTER`` lists everything 1.x promises to remove in 2.0, and
``docs/deprecations.md`` lists the same ids for users. The tests hold the
two together and to the code:

* before 2.0 every entry still works, and an entry that warns emits a
  ``DeprecationWarning`` that names 2.0, so 1.x removes nothing early;
* from 2.0 on (``porosity_fe.__version__``) every entry must be gone, so the
  2.0 release cannot ship with one left over;
* every ``DeprecationWarning`` the code emits belongs to an entry (an AST
  count per file), so a new deprecation cannot skip the register;
* importing the package emits none of them.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

import porosity_fe

REPO = Path(__file__).resolve().parents[1]
MAJOR = int(porosity_fe.__version__.split('.')[0])
MAT = dataclasses.replace(porosity_fe.MATERIALS['T800_epoxy'], n_plies=4, t_ply=0.25)


@dataclasses.dataclass(frozen=True)
class Deprecation:
    id: str
    #: The deprecated name or path, as users meet it.
    what: str
    #: ``probe()`` exercises it and returns normally while it exists.
    probe: Callable[[], object]
    #: Whether it warns (the silent ones are private or documented only).
    warns: bool
    #: ``DeprecationWarning`` call sites per file, relative to the repo.
    sites: dict[str, int] = dataclasses.field(default_factory=dict)


def _mesh(**kw):
    pf = porosity_fe.PorosityField(MAT, 0.02)
    return pf, porosity_fe.CompositeMesh(pf, MAT, nx=4, ny=2, nz=4, **kw)


def _solve(**kw):
    pf, mesh = _mesh()
    return porosity_fe.FESolver(mesh, MAT, pf).solve('compression', **kw)


def _f_md_constants():
    from porosity_fe.empirical import Calibration, EmpiricalSolver
    return (Calibration.F_MD_REF, Calibration.F_MD_FLOOR, Calibration.F_MD_FLOOR_ILSS,
            EmpiricalSolver._F_MD_REF)


def _calibration_aliases():
    from porosity_fe import uq
    from porosity_fe.empirical import EmpiricalSolver
    return (EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI, EmpiricalSolver._POWER_LAW_N_QI,
            EmpiricalSolver._LINEAR_BETA_QI, EmpiricalSolver._MATRIX_DOMINATED_MODES,
            uq._UQ_DEFAULT_PERCENTILES, porosity_fe._UQ_DEFAULT_PERCENTILES)


def _apply_penalty():
    from porosity_fe.fe.assembler import BoundaryHandler
    K = scipy.sparse.identity(3, format='csc')
    return BoundaryHandler.apply_penalty(K, np.zeros(3), {0: 0.0})


_PROVENANCE_ALIASES = ('package_version', 'python', 'numpy', 'scipy', 'generated_utc',
                       'git_sha')


def _provenance_aliases():
    from porosity_fe.io import _build_provenance
    prov = _build_provenance()
    if not set(_PROVENANCE_ALIASES) <= set(prov):
        raise KeyError('the provenance aliases are no longer written')
    return prov


def _result_dict_access():
    from porosity_fe.results import FailureResult
    return FailureResult(failure_stress=1.0, knockdown=0.9, model='judd_wright')['knockdown']


def _resolve_material_strict():
    from validation.validate_all import load_dataset, resolve_material
    data = load_dataset(str(REPO / 'validation' / 'datasets' / 'liu_2006.json'))
    return resolve_material(data, strict=True)


def _mesh_dimension_assignment():
    _pf, mesh = _mesh()
    mesh.L_x = 60.0
    return mesh


REGISTER: tuple[Deprecation, ...] = (
    Deprecation('f-md-constants',
                'Calibration.F_MD_REF / F_MD_FLOOR / F_MD_FLOOR_ILSS and the '
                'EmpiricalSolver._F_MD_* aliases', _f_md_constants, True,
                {'porosity_fe/empirical.py': 1}),
    Deprecation('calibration-aliases',
                'EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI / _POWER_LAW_N_QI / _LINEAR_BETA_QI / '
                '_MATRIX_DOMINATED_MODES and porosity_fe._UQ_DEFAULT_PERCENTILES',
                _calibration_aliases, False),
    Deprecation('solver-minres', "FESolver.solve(solver='minres'), "
                "refinement_study(solver='minres')",
                lambda: _solve(solver='minres'), True,
                {'porosity_fe/fe/_load_cases.py': 1, 'porosity_fe/fe/refinement.py': 1}),
    Deprecation('penalty-diag-scale', 'FESolver.solve(penalty_factor=, diag_scale=)',
                lambda: _solve(penalty_factor=1e6), True,
                {'porosity_fe/fe/_load_cases.py': 1}),
    Deprecation('apply-penalty', 'BoundaryHandler.apply_penalty', _apply_penalty, True,
                {'porosity_fe/fe/assembler.py': 1}),
    Deprecation('ply-angles-none', 'ply_angles=None (EmpiricalSolver, CompositeMesh, FESolver)',
                lambda: _mesh(ply_angles=None), True, {'porosity_fe/_ply_angles.py': 1}),
    Deprecation('provenance-aliases',
                'JSON provenance aliases package_version, python, numpy, scipy, '
                'generated_utc, git_sha', _provenance_aliases, False),
    Deprecation('result-dict-access', "FailureResult / ConfigResult dict access "
                "(result['knockdown'])", _result_dict_access, False),
    Deprecation('resolve-material-strict', 'validation.validate_all.resolve_material(strict=)',
                _resolve_material_strict, True, {'validation/validate_all.py': 1}),
    Deprecation('mesh-dimension-assignment',
                'assigning CompositeMesh.L_x / L_y / x_grid after the mesh is generated',
                _mesh_dimension_assignment, True, {'porosity_fe/mesh.py': 1}),
)
IDS = [d.id for d in REGISTER]

#: The files scanned for ``DeprecationWarning`` call sites.
_SCANNED = ('porosity_fe', 'porosity_app', 'validation', 'app.py', 'validate_porosity_cli.py',
            'porosity_fe_analysis.py')


def _probe_is_present(entry: Deprecation) -> bool:
    """Run the probe; ``False`` when the deprecated path is gone (it raises)."""
    try:
        entry.probe()
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


@pytest.mark.parametrize('entry', REGISTER, ids=IDS)
def test_kept_until_2_0_and_gone_from_it(entry):
    if MAJOR < 2:
        if entry.warns:
            with pytest.warns(DeprecationWarning, match=r'2\.0'):
                assert _probe_is_present(entry)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                assert _probe_is_present(entry)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        present = _probe_is_present(entry)
    assert not present and not [w for w in caught
                                if issubclass(w.category, DeprecationWarning)], (
        f"{entry.what} is still in {porosity_fe.__version__}: remove it, with a "
        f"CHANGELOG 'Removed' entry, and its row in docs/deprecations.md.")


def _schema_properties(node, out):
    """Every ``{name: subschema}`` under a ``properties`` key, recursively."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'properties' and isinstance(value, dict):
                out.update(value)
            _schema_properties(value, out)
    elif isinstance(node, list):
        for value in node:
            _schema_properties(value, out)
    return out


def test_provenance_aliases_in_the_schema():
    """Marked deprecated before 2.0; gone from the schema from 2.0 on."""
    schema = json.loads((REPO / 'validation' / 'schemas' / 'porosity_results_schema.json')
                        .read_text(encoding='utf-8'))
    props = _schema_properties(schema, {})
    for alias in _PROVENANCE_ALIASES:
        if MAJOR < 2:
            assert props[alias].get('deprecated') is True, alias
            assert 'schema 2.0' in props[alias]['description'], alias
        else:
            assert alias not in props, alias


def _deprecation_sites(path: Path) -> int:
    """``warnings.warn(..., DeprecationWarning)`` calls in one file."""
    n = 0
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
        if name != 'warn':
            continue
        args = list(node.args) + [k.value for k in node.keywords if k.arg == 'category']
        if any(isinstance(a, ast.Name) and a.id == 'DeprecationWarning' for a in args):
            n += 1
    return n


def test_every_deprecation_warning_is_registered():
    """A new ``DeprecationWarning`` changes a count here: add its entry to
    ``REGISTER`` and its row to ``docs/deprecations.md``."""
    found: Counter[str] = Counter()
    for top in _SCANNED:
        root = REPO / top
        for path in ([root] if root.is_file() else sorted(root.rglob('*.py'))):
            n = _deprecation_sites(path)
            if n:
                found[path.relative_to(REPO).as_posix()] = n
    registered: Counter[str] = Counter()
    for entry in REGISTER:
        registered.update(entry.sites)
    assert dict(found) == dict(registered)


def test_the_docs_list_the_register():
    text = (REPO / 'docs' / 'deprecations.md').read_text(encoding='utf-8')
    listed = re.findall(r'^\| `([a-z0-9-]+)` \|', text, flags=re.M)
    assert sorted(listed) == sorted(IDS)
    assert len(listed) == len(set(listed))


def test_importing_emits_no_deprecation_warning():
    code = ('import porosity_fe, porosity_fe_analysis, porosity_fe.fe.refinement, '
            'validation.validate_all')
    subprocess.run([sys.executable, '-W', 'error::DeprecationWarning', '-c', code],
                   check=True, cwd=REPO)
