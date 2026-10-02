PorosityFE documentation
========================

**PorosityFE** is a finite-element / micromechanics toolkit for analysing
composite laminates with distributed and discrete porosity. It models the
through-thickness porosity profile, degrades the lamina stiffness via
Mori-Tanaka homogenization, and evaluates Tsai-Wu failure on a structured
8-node hex mesh.

This site documents the Python package ``porosity_fe``: the
:doc:`theory <theory/index>` behind both solvers, the
:doc:`command-line tools <cli>`, and the full :doc:`API reference <api>`.
The Streamlit web app (``streamlit run app.py``) and the validation
database are described in the
`project README <https://github.com/elhajjar1/PorosityFE#readme>`_.

Installation
------------

PorosityFE targets Python 3.10 or newer. The recommended install (from a
fresh clone of the repository) is an editable install with the ``docs``
extra so this site can be rebuilt locally:

.. code-block:: bash

   pip install -e ".[docs]"

For the analysis-only runtime (no docs / web extras):

.. code-block:: bash

   pip install -e .

Quickstart
----------

A 2 %-porosity T800/epoxy laminate, compressed in displacement control:

.. code-block:: python

   from porosity_fe import (
       MATERIALS, PorosityField, CompositeMesh, FESolver,
   )

   mat = MATERIALS["T800_epoxy"]
   field = PorosityField(mat, void_volume_fraction=0.02,
                         distribution="uniform",
                         void_shape="spherical")
   mesh = CompositeMesh(field, mat, nx=20, ny=8, nz=12)
   solver = FESolver(mesh, mat, field)
   result = solver.solve(loading="compression", applied_strain=-0.01)
   print(f"max Tsai-Wu index = {result.max_failure_index:.3f}")
   print(f"stiffness knockdown = {result.knockdown:.3f}")

For the empirical (closed-form) knockdown models, use
:class:`~porosity_fe.EmpiricalSolver`:

.. code-block:: python

   from porosity_fe import MATERIALS, build_empirical_pipeline

   _field, _mesh, emp = build_empirical_pipeline(MATERIALS["T800_epoxy"], 0.02)
   result = emp.get_failure_load("compression", model="judd_wright")
   print(f"compression strength = {result.failure_stress:.0f} MPa "
         f"(knockdown {result.knockdown:.3f})")

``Vp`` is always a fraction in ``[0, 1]`` in the Python API (``0.02`` is
2 %); only the CLI's ``--vp-pct`` and the web app take percent.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   theory/index
   cli
   examples

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   changelog
