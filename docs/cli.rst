Command-line tools
==================

Installing the package puts two commands on ``PATH``
(``[project.scripts]`` in ``pyproject.toml``). The option tables below are
generated from the argument parsers themselves, so they match the installed
version.

``--vp`` takes a void **fraction** (``--vp 0.02`` means 2 %), like the
Python API (``Vp=0.02``); ``--vp-pct`` takes percent (``--vp-pct 2``).

porosity-analyze
----------------

Runs the empirical analysis over one or more porosity levels and the five
standard porosity configurations (:data:`porosity_fe.POROSITY_CONFIGS`),
and writes JSON results; ``--plots`` adds PNG figures and ``--uq`` an
uncertainty band. Exit code ``0`` on success, ``2`` on invalid input or an
output that cannot be written, ``3`` on a failed analysis.

``--delamination SPEC`` adds an opt-in FE delamination screening at each
porosity level: the production coupon (``'QI'`` layup) with the
delamination, solved under ``--delam-loading`` with the porosity of
``--delam-config``, and its knockdown split into porosity-only and
delamination-only parts (:func:`~porosity_fe.knockdown_decomposition`;
three FE solves per level). Each level writes
``porosity_delamination_<Vp>.json``, the FE JSON export with the
``fracture`` block (energy release rates along the front, the fracture
load factor, the toughness law and its ``toughness_uncalibrated`` flag) and
the ``knockdown_decomposition`` block, and prints one summary line.
``SPEC`` is ``strip:K:X_START:X_END``, ``rectangle:K:XC:YC:LX:WY`` or
``ellipse:K:XC:YC:LX:WY``: the interface ``K`` (between ply ``K`` and
``K + 1`` from the bottom, 0-based) and lengths in mm on the 50 x 20 mm
coupon, the full sizes ``LX`` and ``WY`` of an NDI indication. Of the
presets only ``IM7_8552_epoxy`` carries interlaminar toughness; for the
others the fracture load factor needs ``--toughness`` (N/mm). ``--porosity-degradation directional`` solves the
screening with the opt-in, uncalibrated direction-aware porosity
degradation (``FESolver(porosity_degradation='directional')``: flat voids
weaken the out-of-plane stiffness and the interlaminar strengths more than
the in-plane ones; see :doc:`theory/micromechanics`) instead of the default
``isotropic``. It applies to all three solves and is recorded in the
``solver`` block of the JSON, and with penny voids
(``--delam-config interface_penny``) it changes the results; spherical
voids give the same results under both. Without ``--delamination`` the run
is unchanged, and its options are refused (see :doc:`theory/delamination`).

.. code-block:: bash

   porosity-analyze --vp 0.02 --delamination ellipse:11:25:10:12:8 \
       --toughness 0.212 0.774 2.1

.. argparse::
   :module: porosity_fe.cli
   :func: _build_arg_parser
   :prog: porosity-analyze

validate_porosity
-----------------

Runs every bundled experimental dataset (``validation/datasets/*.json``)
through the empirical pipeline and writes ``validation_master_report.png``
and ``validation_detail_report.md``. The run summary prints the
property-weighted and point-weighted mean absolute error of the strength
and modulus datasets. The interlaminar toughness datasets (``G_Ic``,
``G_IIc`` blocks) go through the porosity-toughness law instead
(:func:`~porosity_fe.toughness_knockdown_factor`, uncalibrated placeholder
coefficients) and are reported apart: their own summary lines, and their
own section of ``validation_detail_report.md`` with the test stage and
starter crack of each entry. They never enter the headline MAE or the PNG
(see :doc:`theory/delamination`, "Porosity and toughness"). A standalone
executable of this command is built from ``ValidatePorosity.spec`` with
PyInstaller and attached to each GitHub release.

Exit codes: ``0`` when the report was written (datasets that failed to
load are listed as ``[ERROR]`` in the summary), ``2`` when the datasets
directory or the validation module could not be found.

.. argparse::
   :module: validate_porosity_cli
   :func: _build_arg_parser
   :prog: validate_porosity
