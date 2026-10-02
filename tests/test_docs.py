#!/usr/bin/env python3
"""Docs coverage (IMPROVEMENT_PLAN 6.5).

``docs/api.rst`` groups the public API by topic, so it is written by hand;
this keeps it from drifting behind ``porosity_fe.__all__``.
"""

import re
from pathlib import Path

import porosity_fe

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _documented_names() -> set[str]:
    text = (DOCS / "api.rst").read_text(encoding="utf-8")
    names: set[str] = set()
    in_autosummary = False
    for line in text.splitlines():
        if line.startswith(".. autosummary::"):
            in_autosummary = True
            continue
        if in_autosummary:
            if line and not line.startswith(" "):
                in_autosummary = False
            elif re.fullmatch(r"   \w+", line):
                names.add(line.strip())
                continue
        # ``.. py:data:: A`` plus any continuation names aligned under it.
        m = re.match(r"(?:\.\. py:data:: |\s{13})(\w+)\s*$", line)
        if m:
            names.add(m.group(1))
    return names


def test_every_public_name_is_in_the_api_reference():
    missing = sorted(set(porosity_fe.__all__) - _documented_names())
    assert not missing, f"add these to docs/api.rst: {missing}"


def test_api_reference_lists_only_public_names():
    extra = sorted(_documented_names() - set(porosity_fe.__all__))
    assert not extra, f"not in porosity_fe.__all__: {extra}"
