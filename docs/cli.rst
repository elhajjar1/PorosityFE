Command-line tools
==================

Installing the package puts two commands on ``PATH``
(``[project.scripts]`` in ``pyproject.toml``). The option tables below are
generated from the argument parsers themselves, so they match the installed
version.

Porosity is given in **percent** on the command line (``--vp-pct 2`` means
2 %); the Python API always takes a fraction (``Vp=0.02``).

porosity-analyze
----------------

Runs the empirical analysis over one or more porosity levels and the five
standard porosity configurations (:data:`porosity_fe.POROSITY_CONFIGS`),
and writes JSON results; ``--plots`` adds PNG figures and ``--uq`` an
uncertainty band. Exit code ``0`` on success, ``2`` on invalid input or a
failed analysis.

.. argparse::
   :module: porosity_fe.cli
   :func: _build_arg_parser
   :prog: porosity-analyze

validate_porosity
-----------------

Runs every bundled experimental dataset (``validation/datasets/*.json``)
through the empirical pipeline and writes ``validation_master_report.png``
and ``validation_detail_report.md``. The run summary prints the
property-weighted and point-weighted mean absolute error. A standalone
executable of this command is built from ``ValidatePorosity.spec`` with
PyInstaller and attached to each GitHub release.

Exit codes: ``0`` when the report was written (datasets that failed to
load are listed as ``[ERROR]`` in the summary), ``2`` when the datasets
directory or the validation module could not be found.

.. argparse::
   :module: validate_porosity_cli
   :func: _build_arg_parser
   :prog: validate_porosity
