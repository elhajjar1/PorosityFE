"""The ``examples/`` scripts run, write what they name, and match the README.

Issue #215: nothing ran the examples, and they had rotted before (an
``interface_penny.py`` that called itself the worst-case ILSS morphology
while running only the empirical solver, fixed in #209). Each script runs
in-process with ``runpy`` (``run_name='__main__'``), the Agg backend, and
its output redirected to ``tmp_path`` by ``POROSITY_FE_EXAMPLES_OUTPUT``.

``distribution_comparison.py`` also checks the two tables the README
quotes from it ("Numbers reproduced by ``python
examples/distribution_comparison.py``"): the empirical Judd-Wright
compression knockdown and the FE Tsai-Wu compression knockdown for each
distribution, at the README's printed precision. A change that moves them
updates the README in the same PR. The script runs at its own default mesh
(12 x 4 x 24, the one the README names), about 20 s.
"""

from __future__ import annotations

import re
import runpy
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / 'examples'

#: script -> (files it writes, a line it prints)
SCRIPTS = {
    'uniform_spherical.py': (['uniform_spherical_profile.png'], 'Knockdown:'),
    'clustered_midplane.py': (['clustered_midplane_profile.png'], 'ILSS knockdown:'),
    'interface_penny.py': (['interface_penny_profile.png'], 'Compression KD:'),
    'discrete_voids.py': (['discrete_voids_scf.png'], 'SCF_compression='),
    'compute_degraded_clt_moduli.py': (['clt_degraded_moduli.png'], 'Ex/Ex0'),
    'distribution_comparison.py': (['distribution_profiles.png',
                                    'distribution_fe_compare.png'], 'Finding:'),
}

LABELS = ('uniform', 'clustered (midplane)', 'clustered (surface)', 'interface')


def _run(script: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('POROSITY_FE_EXAMPLES_OUTPUT', str(tmp_path))
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    monkeypatch.chdir(tmp_path)
    # The scripts trigger the mesh-quality and calibration warnings on purpose.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        runpy.run_path(str(EXAMPLES / script), run_name='__main__')


def test_every_example_is_covered():
    assert sorted(p.name for p in EXAMPLES.glob('*.py')) == sorted(SCRIPTS)


@pytest.mark.parametrize('script', sorted(SCRIPTS))
def test_example_runs(script, tmp_path, monkeypatch, capsys):
    files, marker = SCRIPTS[script]
    _run(script, tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert marker in out
    for name in files:
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0, name
        assert str(path) in out                      # it says where it wrote


def _readme_block(title: str) -> dict[str, tuple[float, int]]:
    """The README's ``label value`` lines under ``title``: value and its
    number of decimals."""
    lines = (REPO / 'README.md').read_text(encoding='utf-8').splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == title)
    values = {}
    for line in lines[start + 1:start + 1 + len(LABELS)]:
        m = re.match(r'\s*(.+?)\s{2,}(\d+\.(\d+))', line)
        assert m, line
        values[m.group(1)] = (float(m.group(2)), len(m.group(3)))
    assert tuple(values) == LABELS
    return values


def _script_row(out: str, section: str, model: str) -> dict[str, float]:
    """The compression row of ``model`` in the script's ``section`` table."""
    tail = out[out.index(section):]
    tail = tail[tail.index('mode = compression'):]
    row = next(line for line in tail.splitlines() if line.split()[:1] == [model])
    return dict(zip(LABELS, (float(v) for v in row.split()[1:]), strict=True))


def test_distribution_comparison_matches_the_readme(tmp_path, monkeypatch, capsys):
    _run('distribution_comparison.py', tmp_path, monkeypatch)
    out = capsys.readouterr().out
    for title, section, model in (
            ('mode = compression, model = judd_wright, Vp_mean = 3 %',
             'Empirical knockdowns', 'judd_wright'),
            ('mode = compression, FE Tsai-Wu, Vp_mean = 3 %',
             'FE knockdowns', 'fe_tsai_wu')):
        readme = _readme_block(title)
        script = _script_row(out, section, model)
        for label in LABELS:
            value, decimals = readme[label]
            assert script[label] == pytest.approx(value, abs=0.5 * 10.0 ** -decimals + 1e-12), \
                f"{title}: {label} prints {script[label]}, the README says {value}"
    # The README's reading of the FE table: the interface case is the lowest.
    fe = _script_row(out, 'FE knockdowns', 'fe_tsai_wu')
    assert min(fe, key=fe.__getitem__) == 'interface'
