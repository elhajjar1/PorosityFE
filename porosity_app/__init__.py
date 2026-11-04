"""The PorosityFE Streamlit app, split out of ``app.py`` (#224).

``app.py`` stays the entry point (``streamlit run app.py``): it runs the
bootstrap (the ``Agg`` backend and ``sys.path``), imports these modules,
re-exports their names and calls :func:`porosity_app.render._render` under
Streamlit. The modules, in dependency order:

- ``config``: the sidebar configuration keys, defaults and cache key, labels
  and load constants;
- ``runners``: :func:`~porosity_app.runners.run_analysis`, the FE solve and
  their Streamlit-cached wrappers;
- ``summaries``: the result tables (failure, short beam, flexure, thermal,
  defects);
- ``convergence``: the Results tab's mesh-convergence check;
- ``plots``: the figures;
- ``sidebar``: the sidebar inputs;
- ``tabs``: one module per tab (``overview``, ``figures`` for Profile and
  Mesh, ``results``, ``stress``, ``export``) and ``common`` helpers;
- ``render``: the page layout.

Importing the package selects matplotlib's ``Agg`` backend, as ``app.py``
does, so the modules can be imported without a display.
"""

import matplotlib

matplotlib.use("Agg")
