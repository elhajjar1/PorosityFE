#!/usr/bin/env python3
"""Performance guards for the FE solver (IMPROVEMENT_PLAN 1.1-1.5, 6.6).

The structural test runs everywhere and is deterministic: a solve must go
through the batched element path, never the per-element Python objects the
old assembly and recovery loops created (~7,000 per production solve).

The timing test needs the production mesh and a quiet machine, so it only
runs when ``POROSITY_FE_BENCHMARK=1`` (CI sets it on one test cell). Its
budgets leave ~4x headroom over the measured times; before these changes a
first solve took 14-24 s.
"""

import logging
import os
import time
import warnings

import pytest

from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
from porosity_fe.fe import element as element_mod


def _solver(nx, ny, nz):
    mat = MATERIALS['T800_epoxy']
    pf = PorosityField(mat, 0.03, distribution='clustered')
    mesh = CompositeMesh(pf, mat, nx=nx, ny=ny, nz=nz)
    return FESolver(mesh, mat, pf)


def test_solve_uses_batched_path_not_per_element_objects(monkeypatch):
    created = []
    real_init = element_mod.Hex8Element.__init__

    def counting_init(self, *args, **kwargs):
        created.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(element_mod.Hex8Element, '__init__', counting_init)
    solver = _solver(6, 3, 4)
    for loading in ('tension', 'compression', 'shear', 'ilss'):
        solver.solve(loading)
    assert created == [], (
        f"{len(created)} Hex8Element objects were built during solve(); the "
        f"batched assembly/recovery path should build none")


@pytest.mark.skipif(
    os.environ.get("POROSITY_FE_BENCHMARK") != "1",
    reason="set POROSITY_FE_BENCHMARK=1 to run the production-mesh timing check",
)
def test_production_mesh_solve_time():
    first_budget = float(os.environ.get("POROSITY_FE_BENCH_FIRST_S", "10"))
    repeat_budget = float(os.environ.get("POROSITY_FE_BENCH_REPEAT_S", "2"))
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = _solver(30, 10, 12)
            t0 = time.perf_counter()
            solver.solve('tension', applied_strain=0.01)
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            solver.solve('compression', applied_strain=-0.01)
            repeat = time.perf_counter() - t0
    finally:
        logging.disable(logging.NOTSET)
    assert first < first_budget, (
        f"first production-mesh solve took {first:.2f} s "
        f"(budget {first_budget} s; ~2.6 s expected, 14-24 s before Phase 2)")
    assert repeat < repeat_budget, (
        f"repeat solve with the same constraints took {repeat:.2f} s "
        f"(budget {repeat_budget} s; ~0.25 s expected when K and its LU "
        f"factorization are reused)")
