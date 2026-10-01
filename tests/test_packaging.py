#!/usr/bin/env python3
"""Packaging consistency (IMPROVEMENT_PLAN 6.7).

The ``requirements*.txt`` files are kept because the executable build, the
security audit and the Streamlit deployment guide use them, but they must
list exactly what ``pyproject.toml`` declares.
"""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parent.parent


def _requirements(name: str) -> set[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


@pytest.mark.parametrize("filename, source", [
    ("requirements.txt", None),
    ("requirements-web.txt", "web"),
    ("requirements-test.txt", "dev"),
])
def test_requirements_files_match_pyproject(project, filename, source):
    declared = project["dependencies"] if source is None \
        else project["optional-dependencies"][source]
    assert _requirements(filename) == set(declared)


def test_all_extra_is_the_union_of_the_others(project):
    assert project["optional-dependencies"]["all"] == ["porosity-fe[web,dev,docs]"]


def test_citation_ships_in_the_sdist():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "include CITATION.cff" in manifest
