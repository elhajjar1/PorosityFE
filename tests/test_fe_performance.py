#!/usr/bin/env python3
"""Performance guards for the FE solver (IMPROVEMENT_PLAN 1.1-1.5, 6.6).

The structural test runs everywhere and is deterministic: a solve must go
through the batched element path, never the per-element Python objects the
old assembly and recovery loops created (~7,000 per production solve).

The timing tests need the production mesh (``_default_mesh_resolution``:
30 x 10 x 24 for the 24-ply presets) and a quiet machine, so they only run
when ``POROSITY_FE_BENCHMARK=1`` (CI sets it on one test cell). Their
budgets leave ~2-3x headroom over the measured times. Before the batched
path a first solve took 14-24 s on the 30 x 10 x 12 mesh, and would take
twice that on this one. Since 2.4 the first solve on a new mesh geometry
also runs the pristine reference solve for the knockdown (6-9 s here, then
cached).
"""

import dataclasses
import logging
import os
import time
import warnings

import pytest

from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
from porosity_fe.fe import element as element_mod
from porosity_fe.pipeline import _default_mesh_resolution


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
    for loading in ('tension', 'compression', 'shear'):
        solver.solve(loading)
    mat24 = MATERIALS['T800_epoxy']
    pf24 = PorosityField(mat24, 0.03, distribution='clustered')
    specimen = CompositeMesh.for_ilss(pf24, mat24, nz=4,
                                      element_length=mat24.total_thickness / 2)
    FESolver(specimen, mat24, pf24).solve('ilss')
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
    first_budget = float(os.environ.get("POROSITY_FE_BENCH_FIRST_S", "40"))
    repeat_budget = float(os.environ.get("POROSITY_FE_BENCH_REPEAT_S", "3"))
    resolution = _default_mesh_resolution(MATERIALS['T800_epoxy'].n_plies)
    assert resolution == (30, 10, 24)
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = _solver(*resolution)
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
        f"(budget {first_budget} s; 12-19 s expected on 30 x 10 x 24: 6-10 s "
        f"for the solve plus 6-9 s for the pristine knockdown reference; the "
        f"per-element path before Phase 2 took 14-24 s on 30 x 10 x 12)")
    assert repeat < repeat_budget, (
        f"repeat solve with the same constraints took {repeat:.2f} s "
        f"(budget {repeat_budget} s; 0.7-1 s expected when K and its LU "
        f"factorization are reused, 6-10 s when they are not)")


@pytest.mark.skipif(
    os.environ.get("POROSITY_FE_BENCHMARK") != "1",
    reason="set POROSITY_FE_BENCHMARK=1 to run the production-mesh timing check",
)
def test_production_mesh_thermal_solve_time():
    """One thermal solve on the production mesh (every ply resolved).

    The repeat reuses the cached unit-``delta_T`` solution and only rescales.
    """
    first_budget = float(os.environ.get("POROSITY_FE_BENCH_THERMAL_S", "20"))
    repeat_budget = float(os.environ.get("POROSITY_FE_BENCH_REPEAT_S", "2"))
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = MATERIALS['AS4_3501_6_epoxy']
            pf = PorosityField(mat, 0.03, distribution='clustered')
            nx, ny, nz = _default_mesh_resolution(mat.n_plies)
            assert (nx, ny, nz) == (30, 10, 24)
            mesh = CompositeMesh(pf, mat, nx=nx, ny=ny, nz=nz)
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


@pytest.mark.skipif(
    os.environ.get("POROSITY_FE_BENCHMARK") != "1",
    reason="set POROSITY_FE_BENCHMARK=1 to run the production-mesh timing check",
)
def test_ilss_specimen_solve_time():
    """An ILSS solve on the default ASTM D2344 specimen (24 x 8 x 24 for the
    24-ply presets), pristine reference included, then a repeat."""
    from porosity_fe.fe import solver as solver_mod
    solver_mod._PRISTINE_MEASURE_CACHE.clear()
    first_budget = float(os.environ.get("POROSITY_FE_BENCH_ILSS_S", "25"))
    repeat_budget = float(os.environ.get("POROSITY_FE_BENCH_REPEAT_S", "3"))
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = MATERIALS['T800_epoxy']
            pf = PorosityField(mat, 0.03, distribution='clustered')
            mesh = CompositeMesh.for_ilss(pf, mat)
            assert (mesh.nx, mesh.ny, mesh.nz) == (24, 8, 24)
            solver = FESolver(mesh, mat, pf)
            t0 = time.perf_counter()
            solver.solve('ilss')
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            solver.solve('ilss', applied_load=-20.0)
            repeat = time.perf_counter() - t0
    finally:
        logging.disable(logging.NOTSET)
    assert first < first_budget, (
        f"first ILSS specimen solve took {first:.2f} s (budget {first_budget} s; "
        f"~8.5 s expected on 24 x 8 x 24, pristine reference included)")
    assert repeat < repeat_budget, (
        f"repeat ILSS solve took {repeat:.2f} s (budget {repeat_budget} s; "
        f"~0.6 s expected when K and its LU factorization are reused)")
