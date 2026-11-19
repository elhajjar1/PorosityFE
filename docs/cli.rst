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

``--material-file PATH`` analyses the material of a JSON file instead of a
preset: one written by :func:`~porosity_fe.save_material_to_json`, or a bare
JSON object of the :class:`~porosity_fe.MaterialProperties` fields (MPa,
mm, fractions). The name stored in the file, or the file's own, labels the
run. It replaces ``--material`` and applies to every option below.

The laminate and the scope of a run are options:

- ``--layup SPEC`` and ``--t-ply MM`` set the laminate: ``'QI'`` (the
  default), ``'UD'``, or laminate notation such as ``'[0/45/-45/90]_2s'``
  (the ply angles in degrees; an explicit layup sets the ply count). They
  apply to the sweep and to every option below, and each configuration
  entry of the JSON records its ``laminate`` (``ply_angles``, ``n_plies``,
  ``t_ply_mm``).
- ``--configs NAME [NAME ...]`` analyses only the named configurations.
- ``--fe LOADING`` adds an FE solve of every analysed configuration at
  each porosity level (``compression``, ``tension``, ``shear`` on the
  production coupon mesh; ``ilss`` on the ASTM D2344 short beam;
  ``flexure_3pt`` / ``flexure_4pt`` on the ASTM D7264 specimen), written
  to ``porosity_fe_<loading>_<config>_<Vp>.json`` with one summary line.
  The FE strength rests on an uncalibrated heuristic
  (:doc:`theory/failure`): take the size of a strength knockdown from the
  sweep.
- ``--required-knockdown R`` adds the allowable porosity of the laminate:
  the largest void content at which each strength mode still retains the
  fraction ``R`` of its pristine strength, for the three laws, with the
  governing one (:meth:`~porosity_fe.EmpiricalSolver.allowable_porosity`).
  It is printed and written to ``porosity_allowable.json``. It does not
  depend on ``--vp`` or the configurations.
- ``--buckling LOADING`` (``compression`` or ``shear``) adds a linear
  buckling analysis (:meth:`~porosity_fe.FESolver.buckling`) of every
  analysed configuration at each porosity level on the production coupon,
  and one of the void-free laminate: each writes
  ``porosity_buckling_<loading>_<config>_<Vp>.json``, the FE export of the
  prebuckling solve with a ``buckling`` block (load factors, critical
  strain and stress, the void-free reference and the knockdown of the
  critical stress). ``--porosity-degradation`` applies.
- ``--cycles N`` adds the S-N fatigue factor and ``--service-temperature
  T`` / ``--moisture M`` the hygrothermal one (they need a dry
  glass-transition temperature: ``--tg-dry``, or a ``--material-file``
  that sets ``T_g_dry``). Each level writes ``porosity_durability_<Vp>.json``
  (format ``porosity-fe.durability``): every strength mode and law with the
  porosity knockdown, the fatigue and environment factors and their
  product. ``--required-knockdown`` then holds at the same conditions and
  records them. ``--mean-stress goodman`` corrects the S-N curve to
  ``--stress-ratio R`` (tension, shear, ILSS and transverse tension,
  ``0 <= R <= 1``) and ``--compression-stress-ratio R`` (``R >= 1``), and
  ``--fatigue-porosity-slope K`` makes the slope ``b (1 + K Vp)``; see
  :doc:`theory/fatigue_environment`.
- ``--material-cards FORMAT`` (``abaqus`` or ``nastran``) writes the
  porosity-degraded ply properties of every analysed configuration at each
  porosity level as solver material cards
  (:func:`~porosity_fe.material_cards`): one material per distinct ply
  void content, with the ply table, in
  ``porosity_cards_<config>_<Vp>.inp`` (Abaqus keywords) or ``.bdf``
  (Nastran bulk data). ``--card-element solid`` writes them for solid
  elements (``*ELASTIC, TYPE=ENGINEERING CONSTANTS`` or ``MAT9``) instead
  of shells (``TYPE=LAMINA`` with ``*FAIL STRESS``, or ``MAT8``), and
  ``--porosity-degradation`` selects the stiffness mapping. The strengths
  carry the same uncalibrated heuristic as the FE failure indices.

.. code-block:: bash

   porosity-analyze --vp 0.02 --layup "[0/45/-45/90]_2s" --t-ply 0.2 \
       --configs uniform_spherical interface_penny \
       --fe compression --required-knockdown 0.9

``--delamination SPEC`` adds an opt-in FE delamination screening at each
porosity level: the production coupon (the laminate of ``--layup``) with the
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
is unchanged, and its options are refused (``--porosity-degradation`` is
also taken by ``--fe``; see :doc:`theory/delamination`).

``--delam-growth STEPS`` grows the screening's delamination by node release,
at most ``STEPS`` steps, each one FE solve of the grown crack
(:meth:`~porosity_fe.FESolver.delamination_growth`), and adds a
``delamination_growth`` block to the JSON: the critical load factor, strain
and stress of each crack against the area grown. With ``--paris C M
[THRESHOLD]`` the growth is fatigue instead, by the Paris law ``da/dN = C
(G_max / G_c)^m`` with your coefficients (``C`` in mm per cycle; none ship),
under a cyclic peak that is ``--fatigue-peak`` (a fraction between 0 and 1)
of the delamination's static critical load; the block then has the cycles to
each crack. Both need toughness data (see :doc:`theory/delamination`,
"Growth by node release").

.. code-block:: bash

   porosity-analyze --vp 0.02 --delamination ellipse:11:25:10:12:8 \
       --toughness 0.212 0.774 2.1 --delam-growth 10 \
       --paris 0.05 6 --fatigue-peak 0.5

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

``--fe`` also runs the FE solver against the strength and modulus
datasets, one solve per data point of the test each property comes from
(about a minute for the bundled datasets). It prints the FE mean absolute
error, overall and for the moduli and the strengths beside the empirical
pipeline's, and adds an "FE path" section to
``validation_detail_report.md``. The headline MAE and the PNG do not
change. The FE moduli match the data as laminate theory does; the FE
strengths rest on an uncalibrated heuristic and do not (see
:doc:`theory/failure`, "Degraded strengths").

Exit codes: ``0`` when the report was written (datasets that failed to
load are listed as ``[ERROR]`` in the summary), ``2`` when the datasets
directory or the validation module could not be found.

.. argparse::
   :module: validate_porosity_cli
   :func: _build_arg_parser
   :prog: validate_porosity
