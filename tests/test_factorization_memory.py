"""The memory budget of the sparse LU factorizations, and their pivoting (#228).

Before SuperLU builds a factorization the solver estimates it
(``fe/_memory.py``, the model ``refinement_study`` plans with) and refuses
one that, with what the solve already holds, exceeds the budget
(``FESolver(max_memory_gb=)``, default 80 % of physical memory), after
freeing the cached factorizations when that makes it fit. The
knockdown's pristine reference follows the requested linear solver, so
``solver='cg'`` factorizes nothing. ``K_ff`` is SPD and is factorized with
diagonal pivots. Sizes in bytes unless named GB.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pytest

from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
from porosity_fe.fe import _constrained_solve, _knockdown, _memory

MAT = dataclasses.replace(MATERIALS['T800_epoxy'], n_plies=4, t_ply=0.25)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    logging.disable(logging.WARNING)
    # A fresh pristine cache, so every test's reference is solved here.
    monkeypatch.setattr(_knockdown, '_PRISTINE_MEASURE_CACHE', OrderedDict())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        yield
    logging.disable(logging.NOTSET)


def _solver(**kw):
    pf = PorosityField(MAT, 0.03, distribution='clustered', seed=3)
    mesh = CompositeMesh(pf, MAT, nx=12, ny=6, nz=4)
    return FESolver(mesh, MAT, pf, **kw)


def _one_factor_budget_gb(fe):
    """A budget that fits the mesh and one factorization, not two."""
    n_dof = fe.mesh.n_dof
    return (fe._held_bytes() + 1.05 * _memory.lu_bytes_estimate(n_dof)) / 1e9


class TestModel:
    def test_estimate_matches_the_study_model(self):
        from porosity_fe.fe.refinement import _estimate_memory_bytes
        res = (30, 10, 24)
        n_dof = 3 * 31 * 11 * 25
        assert _estimate_memory_bytes(res) - _estimate_memory_bytes(res, n_factorizations=0) \
            == pytest.approx(_memory.lu_bytes_estimate(n_dof), rel=1e-12)
        # The 30 x 10 x 24 coupon's factors hold 32.3 M entries; the model
        # says 32.7 M.
        assert _memory.lu_bytes_estimate(n_dof) / _memory.LU_BYTES_PER_NNZ == \
            pytest.approx(32.7e6, rel=0.01)

    @pytest.mark.parametrize('value', [0, -1.0, math.nan, True, '4', [4]])
    def test_invalid_budget(self, value):
        with pytest.raises(ValueError, match='max_memory_gb must be a number of GB > 0'):
            _memory.check_max_memory_gb(value)
        with pytest.raises(ValueError, match='max_memory_gb must be'):
            _solver(max_memory_gb=value)

    def test_budget_resolution(self, monkeypatch):
        assert _memory.check_max_memory_gb(None) is None
        assert _memory.check_max_memory_gb(math.inf) == math.inf
        assert _memory.resolve_budget(2.5) == 2.5e9
        monkeypatch.setattr(_memory, 'physical_memory_bytes', lambda: 10**10)
        assert _memory.resolve_budget(None) == pytest.approx(0.8e10)
        monkeypatch.setattr(_memory, 'physical_memory_bytes', lambda: None)
        assert _memory.resolve_budget(None) is None

    def test_physical_memory_on_this_platform(self):
        """Every CI platform (Linux, macOS, Windows) reports its memory."""
        phys = _memory.physical_memory_bytes()
        assert isinstance(phys, int) and phys > 2**28

    @pytest.mark.skipif(sys.platform == 'win32', reason='os.sysconf path')
    def test_unreported_memory(self, monkeypatch):
        import os

        def broken(name):
            raise ValueError(name)

        monkeypatch.setattr(os, 'sysconf', broken)
        assert _memory.physical_memory_bytes() is None
        monkeypatch.setattr(os, 'sysconf', lambda name: -1)
        assert _memory.physical_memory_bytes() is None
        # Off Windows ctypes has no windll.
        assert _memory._windows_physical_memory() is None

    @pytest.mark.parametrize('content, expected', [
        ('1073741824\n', 2**30),          # cgroup v2 or v1, a 1 GiB limit
        ('max\n', None),                  # v2, no limit
        ('9223372036854771712\n', None),  # v1, no limit
        ('', None),
    ])
    def test_cgroup_memory_limit(self, tmp_path, monkeypatch, content, expected):
        f = tmp_path / 'memory.max'
        f.write_text(content, encoding='ascii')
        monkeypatch.setattr(_memory, '_CGROUP_LIMIT_FILES', (str(f),))
        assert _memory._cgroup_memory_limit() == expected
        monkeypatch.setattr(_memory, '_CGROUP_LIMIT_FILES', (str(tmp_path / 'absent'),))
        assert _memory._cgroup_memory_limit() is None

    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason='cgroup path')
    def test_container_limit_caps_physical_memory(self, monkeypatch):
        """A container's limit below the host's RAM sets the default budget;
        one above it, or none, leaves the host's RAM."""
        import os
        monkeypatch.setattr(os, 'sysconf', lambda name: 2**20 if name == 'SC_PHYS_PAGES' else 4096)
        host = 2**32
        monkeypatch.setattr(_memory, '_cgroup_memory_limit', lambda: 2**30)
        assert _memory.physical_memory_bytes() == 2**30
        assert _memory.resolve_budget(None) == pytest.approx(0.8 * 2**30)
        monkeypatch.setattr(_memory, '_cgroup_memory_limit', lambda: 2**40)
        assert _memory.physical_memory_bytes() == host
        monkeypatch.setattr(_memory, '_cgroup_memory_limit', lambda: None)
        assert _memory.physical_memory_bytes() == host

    def test_check_factorization(self):
        est = _memory.lu_bytes_estimate(30_000)
        _memory.check_factorization(30_000, 29_000, 1e9, None)
        _memory.check_factorization(30_000, 29_000, 1e9, 1e9 + est)
        with pytest.raises(ValueError) as exc:
            _memory.check_factorization(30_000, 29_000, 1e9, 1e9 + 0.99 * est)
        msg = str(exc.value)
        assert '29,000 free of 30,000 DOFs' in msg and "solver='cg'" in msg
        assert 'max_memory_gb' in msg


class TestSolverBudget:
    def test_a_factorization_over_budget_is_refused(self):
        fe = _solver(max_memory_gb=1e-3)
        with pytest.raises(ValueError, match=r'GB budget \(FESolver\(max_memory_gb=\)'):
            fe.solve('compression')
        assert not fe._lu_cache or all(e[2] is None for e in fe._lu_cache.values())

    def test_cg_factorizes_nothing(self, lu_alive_at_build):
        """The pristine reference follows the solver, so a CG solve passes
        any budget and agrees with the direct one."""
        fe = _solver(max_memory_gb=1e-3)
        cg = fe.solve('compression', solver='cg', rtol=1e-10)
        assert lu_alive_at_build == []
        direct = _solver(max_memory_gb=math.inf).solve('compression')
        assert cg.knockdown == pytest.approx(direct.knockdown, rel=1e-8)
        np.testing.assert_allclose(cg.displacement, direct.displacement, rtol=0,
                                   atol=1e-7 * np.abs(direct.displacement).max())
        # The two references are cached apart.
        solvers = {k[4] for k in _knockdown._PRISTINE_MEASURE_CACHE}
        assert solvers == {'direct', ('cg', 1e-10)}

    def test_cached_factorizations_give_way(self, lu_alive_at_build, caplog):
        """One factorization fits the budget: a second constraint set frees
        the cached one (least recently used first) instead of refusing, and
        gets the unbudgeted answer."""
        fe = _solver()
        fe.max_memory_gb = _one_factor_budget_gb(fe)
        fe.solve('compression')                 # pristine, freed, then porous
        assert lu_alive_at_build == [1, 1]
        fe.solve('tension')                     # same constraint set: reused
        assert len(lu_alive_at_build) == 2
        logging.disable(logging.NOTSET)
        with caplog.at_level(logging.INFO, logger='porosity_fe_analysis'):
            shear = fe.solve('shear')           # its pristine frees compression's
        assert lu_alive_at_build == [1, 1, 1, 1]
        assert sum('Freed a cached sparse LU factorization' in r.getMessage()
                   for r in caplog.records) == 1
        assert [e[2] is not None for e in fe._lu_cache.values()] == [True]
        # Back to compression: its pristine reference is cached, so only the
        # porous system is rebuilt, freeing shear's.
        compression = fe.solve('compression')
        assert lu_alive_at_build == [1, 1, 1, 1, 1]
        unbudgeted = _solver(max_memory_gb=math.inf)
        for result, loading in ((shear, 'shear'), (compression, 'compression')):
            assert result.knockdown == pytest.approx(
                unbudgeted.solve(loading).knockdown, rel=1e-12)

    def test_nothing_is_freed_for_a_refusal(self):
        """A factorization that would not fit even with the caches empty is
        refused, and the cached ones are kept."""
        fe = _solver(max_memory_gb=math.inf)
        fe.solve('compression')
        cached = list(fe._lu_cache.values())
        assert [e[2] is not None for e in cached] == [True]
        fe.max_memory_gb = (fe._held_bytes()
                            + 0.5 * _memory.lu_bytes_estimate(fe.mesh.n_dof)) / 1e9
        with pytest.raises(ValueError, match='factorizations already held'):
            fe.solve('shear')
        assert list(fe._lu_cache.values()) == cached

    def test_make_room_order(self):
        """Least recently used first, the new factorization's cache first;
        entries without a factorization stay."""
        class _LU:
            nnz = 1000

        size = _memory.lu_bytes(_LU())
        new = _memory.lu_bytes_estimate(3000)

        def caches():
            own = OrderedDict([(b'a', (None, None, _LU())), (b'b', (None, None, None)),
                               (b'c', (None, None, _LU()))])
            other = OrderedDict([(b'd', (None, None, _LU()))])
            return own, other

        own, other = caches()
        held = _constrained_solve._make_room((own, other), 3000, 1e6,
                                             1e6 + new + 1.5 * size)
        assert held == pytest.approx(1e6 + size)
        assert list(own) == [b'b'] and list(other) == [b'd']
        own, other = caches()
        held = _constrained_solve._make_room((own, other), 3000, 1e6,
                                             1e6 + new + 0.5 * size)
        assert held == pytest.approx(1e6)
        assert list(own) == [b'b'] and list(other) == []
        for budget in (None, 1e6 + new + 3 * size, 1e6 + 0.99 * new):
            own, other = caches()
            held = _constrained_solve._make_room((own, other), 3000, 1e6, budget)
            assert held == pytest.approx(1e6 + 3 * size)
            assert list(own) == [b'a', b'b', b'c'] and list(other) == [b'd']

    def test_default_budget(self, monkeypatch):
        monkeypatch.setattr(_memory, 'physical_memory_bytes', lambda: 10**6)
        with pytest.raises(ValueError, match='0.0008 GB budget'):
            _solver().solve('compression')
        monkeypatch.setattr(_memory, 'physical_memory_bytes', lambda: None)
        assert _solver().solve('compression').knockdown <= 1.0

    def test_refinement_study_passes_its_budget(self, monkeypatch):
        from porosity_fe.fe import refinement
        seen = []
        real = refinement.FESolver

        class Recording(real):
            def __init__(self, *a, **kw):
                seen.append(kw.get('max_memory_gb'))
                super().__init__(*a, **kw)

        monkeypatch.setattr(refinement, 'FESolver', Recording)
        refinement.refinement_study(MAT, 0.02, 'compression', base_resolution=(6, 3, 4),
                                    factors=(1.0, 1.5), ply_angles=[0, 45, -45, 90],
                                    max_memory_gb=50.0)
        assert seen == [50.0, 50.0]
        with pytest.raises(ValueError, match='max_memory_gb must be'):
            refinement.refinement_study(MAT, 0.02, 'compression', base_resolution=(6, 3, 4),
                                        factors=(1.0, 1.5), max_memory_gb=0.0)


class TestDiagonalPivots:
    def test_every_pivot_is_diagonal(self):
        """``K_ff`` is SPD: the factorization pivots on the diagonal, so its
        row and column permutations agree. SuperLU's default threshold of 1
        took off-diagonal pivots on the slender DCB arms (163 here, with 9 %
        more factor entries; 251 and 18 % on the 12-layer benchmark DCB)."""
        import scipy.sparse.linalg as sla
        from test_fracture_specimens import _dcb
        mesh, mat, pf = _dcb(layers=4)
        fe = FESolver(mesh, mat, pf, ply_angles='UD')
        fe.solve('dcb', applied_load=10.0)
        (_, K_ff, lu), = [e for e in fe._lu_cache.values() if e[2] is not None]
        np.testing.assert_array_equal(lu.perm_r, lu.perm_c)
        old = sla.splu(K_ff, permc_spec='MMD_AT_PLUS_A', options={'SymmetricMode': True})
        assert np.count_nonzero(old.perm_r != old.perm_c) > 100
        assert lu.nnz < 0.95 * old.nnz
