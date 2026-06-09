#!/usr/bin/env python3
"""Property tests for ``porosity_fe.gauss`` (Gauss-Legendre quadrature).

The quadrature rules underpin every FE element integral but had no dedicated
test module. These tests pin the defining contract of an ``n``-point
Gauss-Legendre rule: the weights integrate the constant function exactly (sum
to the measure of the domain), the abscissae are symmetric and interior, and
the rule integrates polynomials up to degree ``2n - 1`` exactly while failing
at degree ``2n`` (the property that makes Gauss optimal). The 3D hex rule is
checked as the tensor product of the 1D rule.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe.gauss import gauss_points_1d, gauss_points_hex


def _quad_1d(points, weights, k):
    """Quadrature estimate of integral of x**k over [-1, 1]."""
    return float(np.sum(weights * points ** k))


def _exact_monomial_1d(k):
    """Analytic integral of x**k over [-1, 1]: 0 for odd k, 2/(k+1) for even."""
    return 0.0 if k % 2 else 2.0 / (k + 1)


# --- gauss_points_1d ---------------------------------------------------

@pytest.mark.parametrize('n', [1, 2, 3])
def test_1d_weights_sum_to_interval_measure(n):
    _points, weights = gauss_points_1d(n)
    assert weights.sum() == pytest.approx(2.0, abs=1e-14)
    assert np.all(weights > 0.0)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_1d_points_are_interior_and_symmetric(n):
    points, _weights = gauss_points_1d(n)
    assert points.shape == (n,)
    assert np.all(np.abs(points) < 1.0)
    # Abscissae are symmetric about the origin.
    np.testing.assert_allclose(np.sort(points), -np.sort(points)[::-1], atol=1e-14)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_1d_rule_integrates_up_to_degree_2n_minus_1_exactly(n):
    points, weights = gauss_points_1d(n)
    for k in range(2 * n):  # degrees 0 .. 2n-1
        assert _quad_1d(points, weights, k) == pytest.approx(_exact_monomial_1d(k), abs=1e-12)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_1d_rule_is_not_exact_at_degree_2n(n):
    # The defining optimality bound: an n-point rule cannot integrate the
    # degree-2n monomial exactly.
    points, weights = gauss_points_1d(n)
    k = 2 * n
    assert abs(_quad_1d(points, weights, k) - _exact_monomial_1d(k)) > 1e-3


@pytest.mark.parametrize('n', [0, 4, 5])
def test_1d_rejects_unsupported_point_counts(n):
    with pytest.raises(ValueError, match=r"n=1, 2, 3"):
        gauss_points_1d(n)


# --- gauss_points_hex --------------------------------------------------

def test_hex_default_is_2x2x2_eight_point_rule():
    points, weights = gauss_points_hex()  # default order=2
    assert points.shape == (8, 3)
    assert weights.shape == (8,)
    # Weights integrate the unit constant over the [-1,1]^3 cube (volume 8).
    assert weights.sum() == pytest.approx(8.0, abs=1e-13)
    assert np.all(np.abs(points) < 1.0)


def test_hex_order_one_is_single_centroid_point():
    points, weights = gauss_points_hex(order=1)
    assert points.shape == (1, 3)
    np.testing.assert_allclose(points, [[0.0, 0.0, 0.0]], atol=1e-15)
    assert weights.sum() == pytest.approx(8.0, abs=1e-14)


@pytest.mark.parametrize('a,b,c', [(2, 0, 0), (2, 2, 0), (0, 0, 2), (2, 2, 2), (3, 1, 0)])
def test_hex_integrates_separable_monomials_exactly(a, b, c):
    # A 2-point-per-axis rule is exact for separable polynomials up to degree
    # 3 in each variable; the cube integral factorizes over the 1D integrals.
    points, weights = gauss_points_hex(order=2)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    quad = float(np.sum(weights * x ** a * y ** b * z ** c))
    exact = _exact_monomial_1d(a) * _exact_monomial_1d(b) * _exact_monomial_1d(c)
    assert quad == pytest.approx(exact, abs=1e-13)
