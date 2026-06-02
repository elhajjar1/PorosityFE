# Sphinx configuration for the PorosityFE API reference site.
#
# Build locally with:
#
#     pip install -e ".[docs]"
#     python -m sphinx -b html docs docs/_build/html
#
# The site is also built (and deployed to GitHub Pages on master) by
# .github/workflows/docs.yml.
from __future__ import annotations

import os
import sys
from datetime import date

# Make the top-level porosity_fe package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "PorosityFE"
author = "Rani Elhajjar"
copyright = f"{date.today().year}, {author}"

# Pull the version from the installed package metadata so the docs stay in
# sync with pyproject.toml. Fall back gracefully when the package is not
# importable (e.g. during a fresh clone before `pip install -e .`).
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("porosity-fe")
except Exception:  # pragma: no cover - fallback for an unbuilt checkout
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Generate stub .rst files for entries listed in autosummary directives.
autosummary_generate = True

# Document members in source order (matches the layout of
# porosity_fe: MaterialProperties, VoidGeometry, ...).
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}

# Napoleon -- we use numpydoc style throughout the codebase.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx -- link out to numpy / scipy / matplotlib reference docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Bound how long each inventory fetch may hang. Without this, an
# unreachable remote inventory (e.g. a docs.scipy.org blip) makes the
# default socket wait ~2 minutes before Sphinx gives up -- and under the
# CI ``-W`` flag the resulting warning is a hard build failure, which has
# already reddened the docs build once with no code change. A short
# timeout aborts the request fast so transient outages cause at most a
# brief delay instead of a multi-minute hang.
#
# Supported by ``sphinx.ext.intersphinx`` since Sphinx 2.0; pyproject
# pins ``sphinx>=7.0``, so this is always available here.
#
# Note on the ``-W`` tradeoff: even with a fast timeout, a transient
# unreachable inventory still emits a fetch warning, which ``-W`` would
# escalate to an error. We deliberately do NOT add it to
# ``suppress_warnings`` -- intersphinx logs the "failed to reach any of
# the inventories" message via a plain ``logger.warning`` with no warning
# subtype, so there is no stable category that targets *only* the fetch
# failure. Suppressing it would require a broad filter that could also
# hide genuine cross-reference warnings, defeating ``-W``. The timeout
# alone shrinks the failure window to seconds while keeping every real
# documentation warning fatal.
intersphinx_timeout = 5

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

# Don't fail a CI build over an intersphinx outage.
nitpicky = False
