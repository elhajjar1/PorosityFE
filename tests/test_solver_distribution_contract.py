#!/usr/bin/env python3
"""The two-solver porosity-distribution contract.

PorosityFE ships two solver paths over the same
``PorosityField`` + ``CompositeMesh`` + ``MaterialProperties`` triple, and they
are *designed to disagree* on the effect of the porosity distribution shape
(see the "Two solver paths, one mesh" section of CLAUDE.md):

* :class:`EmpiricalSolver` applies a closed-form knockdown once at the laminate
  level using the specimen-average ``Vp``. The distribution shape
  (``uniform`` vs ``clustered`` vs ``interface``) has **no effect**, because
  ``get_failure_load`` evaluates the knockdown at ``porosity_field.Vp`` (the
  mean), which is identical across shapes at fixed mean porosity.
* :class:`FESolver` applies stiffness/strength degradation **per element** from
  the local ``Vp(x, y, z)`` field, so a clustered or interface distribution
  concentrates porosity into hot-spots and **does** change the result.

These tests pin that contract so a future refactor can't quietly make the
empirical path shape-sensitive or the FE path shape-blind. They are the
behavioural guard for an architectural claim, not a coverage filler.
"""

import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe import (
    MATERIALS,
    CompositeMesh,
    EmpiricalSolver,
    FESolver,
    PorosityField,
)

MATERIAL = MATERIALS['T800_epoxy']
VP = 0.03  # within the empirical calibration bound (Vp <= 0.05); no extrapolation
DISTRIBUTIONS = ('uniform', 'clustered', 'interface')
EMPIRICAL_MODES = ('compression', 'tension', 'shear', 'ilss', 'transverse_tension')


def _empirical_failure(distribution, mode):
    pf = PorosityField(MATERIAL, VP, distribution=distribution)
    mesh = CompositeMesh(pf, MATERIAL, nx=4, ny=2, nz=4)
    solver = EmpiricalSolver(mesh, MATERIAL)
    return solver.get_failure_load(mode, 'judd_wright')


def _per_node_porosity(distribution):
    pf = PorosityField(MATERIAL, VP, distribution=distribution)
    mesh = CompositeMesh(pf, MATERIAL, nx=8, ny=4, nz=8)
    return np.asarray(mesh.porosity)


def _fe_max_failure_index(distribution):
    pf = PorosityField(MATERIAL, VP, distribution=distribution)
    # nz=8 resolves the through-thickness clustering; the thin elements raise a
    # benign mesh-distortion heads-up that is irrelevant to this contract.
    mesh = CompositeMesh(pf, MATERIAL, nx=3, ny=2, nz=8)
    solver = FESolver(mesh, MATERIAL, pf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = solver.solve(loading='compression', applied_strain=-0.001)
    return results.max_failure_index


# --- Empirical path: distribution-invariant ----------------------------

@pytest.mark.parametrize('mode', EMPIRICAL_MODES)
def test_empirical_failure_is_identical_across_distributions(mode):
    """The empirical headline knockdown/strength must not depend on shape."""
    baseline = _empirical_failure('uniform', mode)
    for distribution in ('clustered', 'interface'):
        result = _empirical_failure(distribution, mode)
        # Same scalar mean Vp -> same code path -> bitwise-identical result.
        assert result.knockdown == baseline.knockdown
        assert result.failure_stress == baseline.failure_stress


# --- The inputs genuinely differ (keeps the invariance non-vacuous) -----

def test_distribution_shapes_produce_distinct_porosity_fields():
    fields = {d: _per_node_porosity(d) for d in DISTRIBUTIONS}

    # Uniform is constant at the mean; the others are graded with peaks that
    # exceed the mean (that is exactly the signal the empirical path ignores).
    assert np.std(fields['uniform']) == pytest.approx(0.0, abs=1e-12)
    for distribution in ('clustered', 'interface'):
        assert np.std(fields[distribution]) > 0.0
        assert np.max(fields[distribution]) > VP
        assert not np.allclose(fields['uniform'], fields[distribution])


# --- FE path: distribution-sensitive -----------------------------------

def test_fe_failure_index_depends_on_distribution():
    fi = {d: _fe_max_failure_index(d) for d in DISTRIBUTIONS}

    mean_fi = np.mean(list(fi.values()))
    spread = (max(fi.values()) - min(fi.values())) / mean_fi
    # The three shapes must produce a materially different worst-case failure
    # index (observed ~0.5%); fp noise on this deterministic solve is ~1e-10.
    assert spread > 1e-3

    # Concentrating porosity (clustered / interface) creates a worse local
    # hot-spot than a uniform field of the same mean Vp.
    assert fi['clustered'] > fi['uniform']
    assert fi['interface'] > fi['uniform']


def test_empirical_blind_but_fe_sensitive_on_the_same_contrast():
    """Capstone: the same uniform->clustered change moves FE but not empirical."""
    emp_uniform = _empirical_failure('uniform', 'compression').knockdown
    emp_clustered = _empirical_failure('clustered', 'compression').knockdown
    assert emp_clustered == emp_uniform  # empirical: no change

    fe_uniform = _fe_max_failure_index('uniform')
    fe_clustered = _fe_max_failure_index('clustered')
    assert fe_clustered != pytest.approx(fe_uniform, rel=1e-3)  # FE: changes
