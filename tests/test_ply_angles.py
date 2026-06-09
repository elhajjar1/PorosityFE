#!/usr/bin/env python3
"""Direct unit tests for the shared ``_resolve_ply_angles`` helper.

``EmpiricalSolver``, ``CompositeMesh`` and ``FESolver`` all funnel their
``ply_angles`` argument through
``porosity_fe._ply_angles._resolve_ply_angles`` so the three solver paths
can never drift to divergent ``None`` defaults (#44 item 2). Those classes
only ever call the helper with the production ``none_means='QI'``, so the
``'UD'`` / ``'UD_legacy'`` / invalid-``none_means`` branches of its
documented contract were previously reached by no caller and no test. These
tests exercise the helper directly to pin its full contract.
"""

import warnings

import pytest

from porosity_fe._ply_angles import (
    _PLY_ANGLES_QI,
    _PLY_ANGLES_UD,
    _resolve_ply_angles,
)


# --- String sentinels --------------------------------------------------

def test_qi_sentinel_expands_to_canonical_stack():
    assert _resolve_ply_angles('QI') == list(_PLY_ANGLES_QI)


def test_ud_sentinel_expands_to_all_zero_stack():
    assert _resolve_ply_angles('UD') == list(_PLY_ANGLES_UD)


@pytest.mark.parametrize('sentinel', ['qi', 'Qi', 'ud', 'Ud'])
def test_sentinels_are_case_insensitive(sentinel):
    expected = (list(_PLY_ANGLES_QI) if sentinel.upper() == 'QI'
                else list(_PLY_ANGLES_UD))
    assert _resolve_ply_angles(sentinel) == expected


def test_unknown_string_sentinel_raises_value_error():
    with pytest.raises(ValueError, match=r"'QI' or 'UD'"):
        _resolve_ply_angles('nonsense')


def test_bad_sentinel_error_names_the_caller_and_value():
    with pytest.raises(ValueError, match=r"MyCaller.*'oops'"):
        _resolve_ply_angles('oops', caller='MyCaller')


# --- Explicit sequences ------------------------------------------------

def test_explicit_list_returned_verbatim_as_floats():
    result = _resolve_ply_angles([0, 45, 90, -45])
    assert result == [0.0, 45.0, 90.0, -45.0]
    assert all(isinstance(a, float) for a in result)


def test_explicit_tuple_is_converted_to_a_list_of_floats():
    result = _resolve_ply_angles((0, 90))
    assert isinstance(result, list)
    assert result == [0.0, 90.0]


def test_sentinel_returns_a_fresh_list_not_the_module_baseline():
    # Mutating a resolved list must not corrupt the shared baseline for the
    # next caller (the helper returns a fresh ``list(...)``, not the tuple).
    first = _resolve_ply_angles('QI')
    first.append(123.0)
    second = _resolve_ply_angles('QI')
    assert second == list(_PLY_ANGLES_QI)


def test_explicit_inputs_do_not_warn():
    # Only the ``None`` back-compat shim is deprecated; sentinels and
    # explicit sequences must resolve silently.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _resolve_ply_angles('QI') == list(_PLY_ANGLES_QI)
        assert _resolve_ply_angles('UD') == list(_PLY_ANGLES_UD)
        assert _resolve_ply_angles([0, 90]) == [0.0, 90.0]


# --- None back-compat shim + none_means dispatch -----------------------

def test_none_defaults_to_qi_with_deprecation_warning():
    with pytest.warns(DeprecationWarning, match='deprecated'):
        result = _resolve_ply_angles(None)
    assert result == list(_PLY_ANGLES_QI)


def test_none_with_none_means_ud_resolves_to_ud():
    with pytest.warns(DeprecationWarning):
        result = _resolve_ply_angles(None, none_means='UD')
    assert result == list(_PLY_ANGLES_UD)


def test_none_with_ud_legacy_returns_none():
    # CompositeMesh's historical "None means a literal all-zero array"
    # behaviour is preserved as none_means='UD_legacy' -> None.
    with pytest.warns(DeprecationWarning):
        result = _resolve_ply_angles(None, none_means='UD_legacy')
    assert result is None


def test_none_with_unsupported_none_means_raises_internal_error():
    # The shim warns first, then the none_means dispatch rejects an
    # unsupported value with a clear internal-error ValueError.
    with pytest.warns(DeprecationWarning), \
            pytest.raises(ValueError, match='unsupported none_means'):
        _resolve_ply_angles(None, none_means='bogus')


def test_deprecation_message_names_the_caller():
    with pytest.warns(DeprecationWarning, match=r'FESolver\.ply_angles'):
        _resolve_ply_angles(None, caller='FESolver.ply_angles')
