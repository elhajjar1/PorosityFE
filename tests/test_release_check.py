#!/usr/bin/env python3
"""Tests for .github/scripts/check_release.py (IMPROVEMENT_PLAN 6.4)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "check_release.py"


@pytest.fixture(scope="module")
def check_release():
    spec = importlib.util.spec_from_file_location("check_release", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_release


def _tree(tmp_path, version="1.3.0", fallback="1.3.0", unreleased="",
          released="1.3.0"):
    (tmp_path / "porosity_fe").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "porosity-fe"\nversion = "{version}"\n')
    (tmp_path / "porosity_fe" / "__init__.py").write_text(
        f'try:\n    pass\nexcept Exception:\n    __version__ = "{fallback}"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n{unreleased}\n"
        f"## [{released}] - 2026-10-01\n\n### Added\n- thing\n")
    return tmp_path


def test_consistent_release_passes(check_release, tmp_path):
    assert check_release("v1.3.0", _tree(tmp_path)) == []


@pytest.mark.parametrize("kwargs, tag, needle", [
    ({}, "v1.3.1", "does not match pyproject"),
    ({"fallback": "1.2.0"}, "v1.3.0", "fallback __version__"),
    ({"unreleased": "\n### Fixed\n- pending\n"}, "v1.3.0", "not empty"),
    ({"released": "1.2.0"}, "v1.3.0", "no '## [1.3.0]' section"),
])
def test_each_inconsistency_is_reported(check_release, tmp_path, kwargs, tag, needle):
    problems = check_release(tag, _tree(tmp_path, **kwargs))
    assert len(problems) == 1
    assert needle in problems[0]


def test_repository_versions_agree(check_release):
    """The real pyproject and fallback version literal must stay in sync
    (tag and changelog aside, which only matter at release time)."""
    problems = check_release("v0.0.0-not-a-release")
    assert not any("fallback" in p for p in problems)
