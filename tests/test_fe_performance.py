#!/usr/bin/env python3
"""Performance guards for the FE solver (IMPROVEMENT_PLAN 1.1-1.5, 6.6).

The structural test runs everywhere and is deterministic: a solve must go
through the batched element path, never the per-element Python objects the
old assembly and recovery loops created (~7,000 per production solve).

The timing test needs the production mesh and a quiet machine, so it only
runs when ``POROSITY_FE_BENCHMARK=1`` (CI sets it on one test cell). Its
budgets leave ~3-4x headroom over the measured times; before these changes a
first solve took 14-24 s. Since 2.4 the first solve on a new mesh geometry
also runs the pristine reference solve for the knockdown (~3 s, then cached).
"""

import dataclasses
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
    # Thermal and combined solves (IMPROVEMENT_PLAN 3.5) on a ply-resolving
    # mesh of a material with CTEs.
    mat = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=4)
    pf = PorosityField(mat, 0.03, distribution='clustered')
    mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=4, ply_angles=[0, 90, 90, 0])
    thermal = FESolver(mesh, mat, pf)
    thermal.solve('thermal', delta_T=-150.0)
    thermal.solve('tension', delta_T=-150.0)
    assert created == [], (
        f"{len(created)} Hex8Element objects were built during solve(); the "
        f"batched assembly/recovery path should build none")


@pytest.mark.skipif(
    os.environ.get("POROSITY_FE_BENCHMARK") != "1",
    reason="set POROSITY_FE_BENCHMARK=1 to run the production-mesh timing check",
)
def test_production_mesh_solve_time():
    from porosity_fe.fe import solver as solver_mod
    solver_mod._PRISTINE_MEASURE_CACHE.clear()
    first_budget = float(os.environ.get("POROSITY_FE_BENCH_FIRST_S", "20"))
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
        f"(budget {first_budget} s; ~6 s expected: ~2.6 s for the solve plus "
        f"~3 s for the pristine knockdown reference; 14-24 s before Phase 2)")
    assert repeat < repeat_budget, (
        f"repeat solve with the same constraints took {repeat:.2f} s "
        f"(budget {repeat_budget} s; ~0.25 s expected when K and its LU "
        f"factorization are reused)")


@pytest.mark.skipif(
    os.environ.get("POROSITY_FE_BENCHMARK") != "1",
    reason="set POROSITY_FE_BENCHMARK=1 to run the production-mesh timing check",
)
def test_production_mesh_thermal_solve_time():
    """One thermal solve on the production in-plane mesh with every ply resolved.

    ``nz = 24`` for the 24-ply preset (thermal mode refuses the unresolved
    ``nz = 12``), so the mesh has twice the production element count. The
    repeat reuses the cached unit-``delta_T`` solution and only rescales.
    """
    first_budget = float(os.environ.get("POROSITY_FE_BENCH_THERMAL_S", "20"))
    repeat_budget = float(os.environ.get("POROSITY_FE_BENCH_REPEAT_S", "2"))
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = MATERIALS['AS4_3501_6_epoxy']
            pf = PorosityField(mat, 0.03, distribution='clustered')
            mesh = CompositeMesh(pf, mat, nx=30, ny=10, nz=24)
            solver = FESolver(mesh, mat, pf)
            t0 = time.perf_counter()
            solver.solve('thermal', delta_T=-150.0)
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            solver.solve('thermal', delta_T=-100.0)
            repeat = time.perf_counter() - t0
    finally:
        logging.disable(logging.NOTSET)
    assert first < first_budget, (
        f"first production-mesh thermal solve took {first:.2f} s "
        f"(budget {first_budget} s; ~5-6 s expected on 30 x 10 x 24)")
    assert repeat < repeat_budget, (
        f"repeat thermal solve took {repeat:.2f} s (budget {repeat_budget} s; "
        f"~0.5 s expected when the unit-delta_T solution is reused)")
