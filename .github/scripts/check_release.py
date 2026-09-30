#!/usr/bin/env python3
"""Pre-publish release checks, run by .github/workflows/publish.yml on a tag.

Usage: check_release.py vX.Y.Z

Fails (exit 1, one line per problem) unless:
- the tag equals ``v`` + ``[project].version`` in pyproject.toml;
- the source-checkout fallback ``__version__`` in porosity_fe/__init__.py
  matches that version;
- CHANGELOG.md has an empty ``## [Unreleased]`` section and a
  ``## [X.Y.Z]`` section for the release.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_release(tag: str, root: Path = ROOT) -> list[str]:
    """Return the list of problems blocking a release of ``tag``."""
    problems: list[str] = []
    # A regex rather than tomllib, so the script (and its tests) also run on
    # Python 3.10: the version is the first `version = "..."` after [project].
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    project = pyproject.split("[project]", 1)[-1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    if match is None:
        return ["pyproject.toml has no [project] version."]
    version = match.group(1)
    if tag != f"v{version}":
        problems.append(f"tag {tag!r} does not match pyproject version {version!r} "
                        f"(expected 'v{version}').")

    init = (root / "porosity_fe" / "__init__.py").read_text(encoding="utf-8")
    fallback = re.search(r'^\s+__version__ = "([^"]+)"', init, re.MULTILINE)
    if fallback is None or fallback.group(1) != version:
        found = fallback.group(1) if fallback else None
        problems.append(f"porosity_fe/__init__.py fallback __version__ is {found!r}, "
                        f"expected {version!r}.")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)
    unreleased = next((s for s in sections if s.startswith("[Unreleased]")), None)
    if unreleased is not None and unreleased.split("\n", 1)[-1].strip():
        problems.append("CHANGELOG.md [Unreleased] section is not empty; move its "
                        f"entries under '## [{version}] - <date>'.")
    if not any(s.startswith(f"[{version}]") for s in sections):
        problems.append(f"CHANGELOG.md has no '## [{version}]' section.")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    problems = check_release(argv[1])
    for p in problems:
        print(f"release check: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
