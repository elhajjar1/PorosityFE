"""The structure of the ``porosity_app`` package (#224).

``app.py`` is the Streamlit entry point; the app lives in ``porosity_app``.
These tests hold the split together: the modules import each other only
down the layers below (so there is no cycle), ``app.py`` keeps its
bootstrap before the first app import, and every name ``app`` re-exports
is the package's own object, so ``app.X`` and ``porosity_app.<module>.X``
never drift apart.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "porosity_app"

#: Each module may import only modules of a lower layer.
LAYERS = {
    "config": 0, "tabs.common": 0,
    "runners": 1, "summaries": 1, "sidebar": 1,
    "convergence": 2, "plots": 2,
    "tabs.overview": 3, "tabs.figures": 3, "tabs.results": 3, "tabs.stress": 3,
    "tabs.export": 3,
    "render": 4,
}


def _modules() -> list[str]:
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE).with_suffix("").parts
        if parts[-1] != "__init__":
            out.append(".".join(parts))
    return out


def _sibling_imports(module: str) -> set[str]:
    path = PACKAGE / (module.replace(".", "/") + ".py")
    package = "porosity_app" + ("." + module.rsplit(".", 1)[0] if "." in module else "")
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level:
            name = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            found.add(name.removeprefix("porosity_app."))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("porosity_app"):
            found.add(node.module.removeprefix("porosity_app."))
    return found


def test_every_module_has_a_layer():
    assert sorted(_modules()) == sorted(LAYERS)


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_imports_go_down_the_layers(module):
    for dep in _sibling_imports(module):
        assert LAYERS[dep] < LAYERS[module], f"{module} imports {dep}"


def test_app_bootstraps_before_importing_the_package():
    """``matplotlib.use("Agg")`` and the ``sys.path`` insert come first
    (CLAUDE.md, "Streamlit app bootstrap")."""
    src = (REPO / "app.py").read_text(encoding="utf-8")
    first_app_import = src.index("from porosity_app")
    assert src.index('matplotlib.use("Agg")') < src.index("import streamlit") < first_app_import
    assert src.index("sys.path.insert(0") < src.index("from porosity_fe import") < first_app_import
    assert src.index("_configure_matplotlib_style()") > src.rindex("from porosity_app")


def test_app_reexports_the_package_objects():
    import app
    tree = ast.parse((REPO / "app.py").read_text(encoding="utf-8"))
    n = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("porosity_app"):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert getattr(app, alias.asname or alias.name) is getattr(module, alias.name)
                n += 1
    assert n > 100      # 135 names at the split
    # The names the tests monkeypatch are looked up in these modules.
    from porosity_app import convergence, runners
    for name in ("FESolver", "build_empirical_pipeline", "run_analysis"):
        assert hasattr(runners, name)
    for name in ("refinement_study", "run_convergence_check"):
        assert hasattr(convergence, name)
