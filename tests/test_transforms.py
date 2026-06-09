#!/usr/bin/env python3
"""Property tests for ``porosity_fe.transforms``.

The 3D rotation helpers (rotation matrix, 6x6 Voigt stress / strain
transforms, and the stiffness rotation) are consumed by ``homogenization``,
``fe.element`` and ``fe.solver`` but had no dedicated test module — they were
exercised only indirectly through the FE path. These tests pin their
mathematical contract directly: orthonormality, the negative-angle inverse,
the Reuter stress/strain duality, and the physical invariants of a rotated
stiffness (round-trip identity, preserved symmetry, isotropic invariance, and
the 90-degree in-plane axis swap).
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from porosity_fe.transforms import (
    rotate_stiffness_3d,
    rotation_matrix_3d,
    strain_transformation_3d,
    stress_transformation_3d,
)

I3 = np.eye(3)
I6 = np.eye(6)
AXES = ['z', 'y']
ANGLES = [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, -np.pi / 5, 1.0, 2.3]


def _isotropic_stiffness(lam: float = 5000.0, mu: float = 3000.0) -> np.ndarray:
    """A Voigt 6x6 isotropic stiffness (MPa) — rotation-invariant by construction."""
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _orthotropic_stiffness() -> np.ndarray:
    """A symmetric orthotropic stiffness with distinct 11/22/33 normal terms."""
    C = np.diag([160000.0, 9000.0, 11000.0, 3500.0, 4500.0, 4600.0]).astype(float)
    C[0, 1] = C[1, 0] = 4000.0
    C[0, 2] = C[2, 0] = 3000.0
    C[1, 2] = C[2, 1] = 2500.0
    return C


# --- rotation_matrix_3d ------------------------------------------------

@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', ANGLES)
def test_rotation_matrix_is_orthonormal_with_unit_determinant(angle, axis):
    R = rotation_matrix_3d(angle, axis=axis)
    np.testing.assert_allclose(R @ R.T, I3, atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('axis', AXES)
def test_rotation_matrix_zero_angle_is_identity(axis):
    np.testing.assert_allclose(rotation_matrix_3d(0.0, axis=axis), I3, atol=1e-15)


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', ANGLES)
def test_rotation_inverse_is_the_negative_angle(angle, axis):
    R = rotation_matrix_3d(angle, axis=axis)
    R_inv = rotation_matrix_3d(-angle, axis=axis)
    np.testing.assert_allclose(R @ R_inv, I3, atol=1e-12)


def test_rotation_matrix_rejects_unknown_axis():
    with pytest.raises(ValueError, match=r"Unsupported axis"):
        rotation_matrix_3d(0.5, axis='x')


# --- stress / strain transformation matrices ---------------------------

@pytest.mark.parametrize('builder', [stress_transformation_3d, strain_transformation_3d])
@pytest.mark.parametrize('axis', AXES)
def test_voigt_transform_zero_angle_is_identity(builder, axis):
    np.testing.assert_allclose(builder(0.0, axis=axis), I6, atol=1e-14)


@pytest.mark.parametrize('builder', [stress_transformation_3d, strain_transformation_3d])
@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', ANGLES)
def test_voigt_transform_inverse_is_the_negative_angle(builder, axis, angle):
    T = builder(angle, axis=axis)
    T_inv = builder(-angle, axis=axis)
    np.testing.assert_allclose(T @ T_inv, I6, atol=1e-10)


@pytest.mark.parametrize('builder', [stress_transformation_3d, strain_transformation_3d])
def test_voigt_transform_rejects_unknown_axis(builder):
    with pytest.raises(ValueError, match=r"Unsupported axis"):
        builder(0.5, axis='x')


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', ANGLES)
def test_stress_and_strain_transforms_are_energy_conjugate(angle, axis):
    # Strain energy sigma:epsilon is frame-invariant, which forces the
    # engineering strain transform to be the inverse-transpose of the stress
    # transform: T_eps^T == T_sig^{-1}. Pin that algebraic duality.
    T_sig = stress_transformation_3d(angle, axis=axis)
    T_eps = strain_transformation_3d(angle, axis=axis)
    np.testing.assert_allclose(T_eps.T, np.linalg.inv(T_sig), atol=1e-10)


# --- rotate_stiffness_3d -----------------------------------------------

def test_rotate_stiffness_rejects_non_6x6():
    with pytest.raises(ValueError, match=r"must be 6x6"):
        rotate_stiffness_3d(np.eye(3), 0.5, axis='z')


@pytest.mark.parametrize('axis', AXES)
def test_rotate_stiffness_zero_angle_is_unchanged(axis):
    C = _orthotropic_stiffness()
    np.testing.assert_allclose(rotate_stiffness_3d(C, 0.0, axis=axis), C, atol=1e-9)


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', [np.pi / 6, np.pi / 4, 1.1, -0.7])
def test_rotate_stiffness_round_trip_recovers_original(angle, axis):
    C = _orthotropic_stiffness()
    rotated = rotate_stiffness_3d(C, angle, axis=axis)
    restored = rotate_stiffness_3d(rotated, -angle, axis=axis)
    np.testing.assert_allclose(restored, C, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', [0.3, np.pi / 4, 1.2])
def test_rotate_stiffness_preserves_symmetry(angle, axis):
    rotated = rotate_stiffness_3d(_orthotropic_stiffness(), angle, axis=axis)
    np.testing.assert_allclose(rotated, rotated.T, atol=1e-6)


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('angle', ANGLES)
def test_isotropic_stiffness_is_rotation_invariant(angle, axis):
    # An isotropic material has no preferred direction, so rotating its
    # stiffness to any frame must return the same matrix.
    C = _isotropic_stiffness()
    np.testing.assert_allclose(rotate_stiffness_3d(C, angle, axis=axis), C, atol=1e-7)


def test_ninety_degree_z_rotation_swaps_in_plane_normal_terms():
    # A 90-degree rotation about z exchanges the 1 and 2 material directions,
    # so C_11 <-> C_22 while the through-thickness C_33 is untouched.
    C = _orthotropic_stiffness()
    C90 = rotate_stiffness_3d(C, np.pi / 2, axis='z')
    assert C90[0, 0] == pytest.approx(C[1, 1], abs=1e-6)
    assert C90[1, 1] == pytest.approx(C[0, 0], abs=1e-6)
    assert C90[2, 2] == pytest.approx(C[2, 2], abs=1e-6)
