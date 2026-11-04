"""The Profile and Mesh tabs.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

from ..plots import plot_mesh, plot_profile
from ..tabs.common import _placeholder_tab, _show_figure


def _build_profile_tab(result: dict | None):
    if result is None:
        _placeholder_tab()
        return
    _show_figure(plot_profile(result), "porosity_profile", "dl_png_profile")


def _build_mesh_tab(result: dict | None):
    if result is None:
        _placeholder_tab()
        return
    _show_figure(plot_mesh(result), "mesh_section", "dl_png_mesh")
