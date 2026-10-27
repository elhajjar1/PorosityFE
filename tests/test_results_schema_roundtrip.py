#!/usr/bin/env python3
"""End-to-end JSON output contract tests (#20).

Exercises the full save -> jsonschema.validate -> load_results_from_json
loop on a minimal one-Vp / one-config sweep so any drift in the envelope,
schema, or loader is caught in a single test file.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from porosity_fe_analysis import (
    FORMAT_EMPIRICAL_SWEEP,
    FORMAT_NCR,
    JSON_SCHEMA_VERSION,
    POROSITY_CONFIGS,
    _json_default,
    compare_configurations,
    load_results_from_json,
    save_results_to_json,
)


_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "validation",
    "schemas",
    "porosity_results_schema.json",
)


def _load_schema():
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _tiny_results():
    """Smallest sweep that still produces a real payload."""
    return compare_configurations(
        0.03,
        configs={"uniform_spherical": POROSITY_CONFIGS["uniform_spherical"]},
    )


def test_save_validate_load_round_trip(tmp_path):
    import jsonschema

    path = str(tmp_path / "round_trip.json")
    save_results_to_json(_tiny_results(), path)

    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)

    # Schema validation: catches envelope drift.
    jsonschema.validate(instance=on_disk, schema=_load_schema())

    # Loader returns the same dict (apart from object identity).
    loaded = load_results_from_json(path)
    assert loaded == on_disk

    # Envelope sanity.
    assert loaded["schema_version"] == JSON_SCHEMA_VERSION
    assert loaded["format"] == FORMAT_EMPIRICAL_SWEEP
    assert "provenance" in loaded
    assert "uniform_spherical" in loaded


def test_ncr_format_validates_against_schema(tmp_path):
    """The NCR exporter is one of the three known top-level formats and
    must validate against the same shared schema (#20)."""
    import jsonschema

    from porosity_fe.reporting import build_ncr_record, write_ncr_json

    result = {
        "config": {
            "material_name": "T800_epoxy",
            "n_plies": 24,
            "t_ply": 0.183,
            "Vp": 3.0,
            "distribution": "uniform",
            "void_shape": "spherical",
            "nx": 30, "ny": 10, "nz": 12,
        },
        "empirical": {
            "compression": {
                "judd_wright": {"failure_stress": 1234.5, "knockdown": 0.823},
            },
            "ilss": {
                "judd_wright": {"failure_stress": 67.0, "knockdown": 0.744},
            },
        },
    }
    meta = {
        "prepared_by": "test",
        "ncr_reference": "NCR-2026-0001",
        "structural_class": "primary",
        "date": "2026-05-19",
        "layup": "[0/90]_s",
    }
    path = str(tmp_path / "ncr.json")
    write_ncr_json(path, build_ncr_record(result, meta))

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    jsonschema.validate(instance=doc, schema=_load_schema())
    assert doc["format"] == FORMAT_NCR

    loaded = load_results_from_json(path)
    assert loaded["format"] == FORMAT_NCR


def test_numpy_default_handler_round_trips_ndarray(tmp_path):
    """An ndarray smuggled into a config dict must serialise via
    _json_default rather than raise TypeError (#20 item 4)."""
    results = _tiny_results()
    # #44: compare_configurations now returns ConfigResult dataclasses, so
    # mutate via dataclasses.replace rather than dict() (the dict-protocol
    # shim drops the live-object keys and would break save_results_to_json).
    original = results["uniform_spherical"]
    replacement = dataclasses.replace(
        original,
        config={
            **original.config,
            "ply_angles_deg": np.array([0.0, 45.0, -45.0, 90.0]),
            "n_plies_np": np.int64(12),
        },
    )
    results = {"uniform_spherical": replacement}

    path = str(tmp_path / "with_ndarray.json")
    save_results_to_json(results, path)

    loaded = load_results_from_json(path)
    assert loaded["uniform_spherical"]["config"]["ply_angles_deg"] == [
        0.0,
        45.0,
        -45.0,
        90.0,
    ]
    assert loaded["uniform_spherical"]["config"]["n_plies_np"] == 12


def test_json_default_handles_dataclass():
    """Plain dataclass instances must be serialisable by _json_default
    (#20 item 4: numpy-type fragility)."""

    @dataclasses.dataclass
    class _Foo:
        a: int
        b: str

    out = _json_default(_Foo(a=1, b="x"))
    assert out == {"a": 1, "b": "x"}


def test_json_default_rejects_unknown_type():
    class _Opaque:
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(_Opaque())


def test_save_results_hostname_opt_in_via_env(tmp_path, monkeypatch):
    """#152: ``provenance.hostname`` is opt-in via
    ``POROSITY_FE_INCLUDE_HOSTNAME``. When the env var is set the field
    is populated with a non-empty string; when unset the key is omitted
    from the provenance block entirely (no workstation-name leakage).
    """
    results = _tiny_results()

    # Opt-in: env var set to a truthy value -> hostname populated.
    monkeypatch.setenv("POROSITY_FE_INCLUDE_HOSTNAME", "1")
    on_path = str(tmp_path / "with_host.json")
    save_results_to_json(results, on_path)
    with open(on_path, encoding="utf-8") as f:
        prov_on = json.load(f)["provenance"]
    assert "hostname" in prov_on
    assert isinstance(prov_on["hostname"], str)
    assert prov_on["hostname"]  # non-empty

    # Default: env var unset -> hostname key absent from the provenance.
    monkeypatch.delenv("POROSITY_FE_INCLUDE_HOSTNAME", raising=False)
    off_path = str(tmp_path / "no_host.json")
    save_results_to_json(results, off_path)
    with open(off_path, encoding="utf-8") as f:
        prov_off = json.load(f)["provenance"]
    assert "hostname" not in prov_off


def test_save_results_config_name_collision_raises(tmp_path):
    """#152: a user-supplied ConfigResult keyed with one of the reserved
    envelope keys (``'schema_version'`` / ``'format'``) must raise
    ValueError rather than silently overwrite the envelope on disk.
    """
    results = _tiny_results()
    original = next(iter(results.values()))
    colliding = {"schema_version": original}

    path = str(tmp_path / "collision.json")
    with pytest.raises(ValueError, match="collides"):
        save_results_to_json(colliding, path)


def test_save_results_includes_units_block(tmp_path):
    """#131: save_results_to_json must emit a top-level ``units`` block so
    consumers reading only the JSON can interpret ambiguous fields like
    ``knockdown``. The block lives in the envelope (after ``provenance``,
    before per-config payloads) and maps payload field names to unit
    strings.
    """
    import jsonschema

    path = str(tmp_path / "with_units.json")
    save_results_to_json(_tiny_results(), path)

    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)

    # New envelope contract: units present, schema bumped to 1.1+.
    assert "units" in on_disk, (
        "Envelope must carry a self-documenting 'units' block (#131)."
    )
    units = on_disk["units"]
    assert isinstance(units, dict) and units, "units must be a non-empty mapping."
    # The empirical-sweep payload writes failure_stress (MPa), knockdown
    # (dimensionless fraction), and void_volume_fraction (dimensionless
    # fraction) — the three numeric leaves the consumer needs labelled.
    assert "failure_stress" in units
    assert units["failure_stress"] == "MPa"
    assert "knockdown" in units
    assert "dimensionless" in units["knockdown"].lower()
    assert "void_volume_fraction" in units
    assert "dimensionless" in units["void_volume_fraction"].lower()

    # Version bump: this envelope shipped as 1.1.
    assert on_disk["schema_version"] == JSON_SCHEMA_VERSION
    assert on_disk["schema_version"] != "1.0"
    assert on_disk["schema_version"].startswith("1.")

    # Whole envelope must still validate against the schema.
    jsonschema.validate(instance=on_disk, schema=_load_schema())


def test_legacy_json_without_units_block_still_loads(tmp_path):
    """#131: the ``units`` block is purely additive — consumers reading a
    file written by an older porosity-fe (no ``units`` key) must still
    load and validate cleanly. The MAJOR version is unchanged, so the
    loader accepts both 1.0 and 1.1 envelopes.
    """
    import jsonschema

    # Hand-craft a minimal valid 1.0-style envelope (no ``units`` key).
    legacy = {
        "schema_version": "1.0",
        "format": FORMAT_EMPIRICAL_SWEEP,
        "provenance": {
            "schema_version": "1.0",
            "porosity_fe_version": "0.0.0",
            "python_version": "3.11.0",
            "platform": "test",
            "numpy_version": "1.26.0",
            "scipy_version": "1.11.0",
            "timestamp_utc": "2026-05-22T00:00:00Z",
            "seed": None,
            "git_commit": None,
            "package_version": "0.0.0",
            "python": "3.11.0",
            "numpy": "1.26.0",
            "scipy": "1.11.0",
            "generated_utc": "2026-05-22T00:00:00Z",
            "git_sha": None,
        },
        "demo_config": {
            "void_volume_fraction": 0.03,
            "empirical": {},
        },
    }
    path = tmp_path / "legacy_no_units.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    # No ``units`` key, but schema validation still passes (additive change).
    jsonschema.validate(instance=legacy, schema=_load_schema())
    # And the loader accepts the older minor version (same MAJOR).
    loaded = load_results_from_json(str(path))
    assert "units" not in loaded
    assert loaded["schema_version"] == "1.0"


def test_save_results_legacy_dict_shape_still_works(tmp_path):
    """#152: the pre-#103 ``Dict[str, dict]`` shape (raw worker-dict with
    ``porosity_field`` / ``config`` / ``empirical`` keys) must still
    round-trip through save_results_to_json so callers that haven't
    migrated to ConfigResult keep working.
    """
    # Borrow real numbers from a tiny sweep, then re-shape them into the
    # legacy worker-dict that predates ConfigResult.
    cr = _tiny_results()["uniform_spherical"]

    class _PFStub:
        # Legacy path reads only ``.Vp`` off the porosity_field object.
        Vp = cr.Vp
        seed = cr.seed

    legacy_results = {
        "uniform_spherical": {
            "porosity_field": _PFStub(),
            "config": cr.config,
            "empirical": cr.empirical,
        },
    }

    path = str(tmp_path / "legacy.json")
    save_results_to_json(legacy_results, path)

    loaded = load_results_from_json(path)
    assert loaded["schema_version"] == JSON_SCHEMA_VERSION
    assert loaded["format"] == FORMAT_EMPIRICAL_SWEEP
    assert "uniform_spherical" in loaded
    assert loaded["uniform_spherical"]["void_volume_fraction"] == float(cr.Vp)
    # Empirical knockdown table survives the round-trip.
    assert (
        loaded["uniform_spherical"]["empirical"]["compression"]["judd_wright"]
        ["knockdown"]
        == cr.empirical["compression"]["judd_wright"]["knockdown"]
    )


def test_config_result_dict_shim_unknown_key_get_and_keys():
    """ConfigResult's dict back-compat shim: an unknown (non-artifact) key
    raises KeyError, ``get`` swallows that into its default, and ``keys``
    enumerates the documented fields (results.py)."""
    r = next(iter(_tiny_results().values()))

    with pytest.raises(KeyError, match="not a known ConfigResult field"):
        r["no_such_field"]

    assert r.get("no_such_field", "fallback") == "fallback"
    assert r.get("Vp") == r["Vp"]

    assert set(r.keys()) == {
        "Vp", "config_name", "config", "failure_stress",
        "knockdown", "model", "empirical", "seed",
    }


# ----------------------------------------------------------------------
# Standard JSON from every writer: no NaN / Infinity tokens
# ----------------------------------------------------------------------
# Python's json module writes NaN / Infinity by default; strict parsers
# (JavaScript, jq, Go, ...) reject them. These tests parse every writer's
# output with a parser that refuses those tokens.

def _reject_constant(token):
    raise ValueError(f"non-standard JSON token {token!r}")


def _strict_loads(text):
    return json.loads(text, parse_constant=_reject_constant)


def _ncr_result():
    return {
        "config": {
            "material_name": "T800_epoxy", "n_plies": 24, "t_ply": 0.183,
            "Vp": 3.0, "distribution": "uniform", "void_shape": "spherical",
            "nx": 30, "ny": 10, "nz": 12,
        },
        "empirical": {
            "compression": {"judd_wright": {"failure_stress": 1234.5, "knockdown": 0.823}},
            "ilss": {"judd_wright": {"failure_stress": 67.0, "knockdown": 0.744}},
        },
    }


def _write_sweep(path):
    save_results_to_json(_tiny_results(), path)


def _write_app_export(path):
    from porosity_fe.reporting import build_export_payload, write_results_json
    write_results_json(path, build_export_payload(_ncr_result()))


def _write_ncr(path):
    from porosity_fe.reporting import build_ncr_record, write_ncr_json
    write_ncr_json(path, build_ncr_record(_ncr_result(), {"prepared_by": "test"}))


def _write_uq(path, include_samples=False):
    from porosity_fe import propagate_uncertainty, save_uq_results_to_json
    res = {"compression": propagate_uncertainty(
        0.03, coef_cov=0.1, vp_cov=0.1, n_samples=20, seed=1)}
    save_uq_results_to_json(res, path, include_samples=include_samples)


def _write_uq_with_samples(path):
    _write_uq(path, include_samples=True)


@pytest.mark.parametrize("writer", [
    _write_sweep, _write_app_export, _write_ncr, _write_uq, _write_uq_with_samples,
], ids=["sweep", "app_export", "ncr", "uq", "uq_samples"])
def test_empirical_writers_emit_standard_json(writer, tmp_path):
    import jsonschema

    path = tmp_path / "out.json"
    writer(str(path))
    doc = _strict_loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=doc, schema=_load_schema())
    assert load_results_from_json(path) == doc


@pytest.fixture(scope="module")
def fe_tsai_wu():
    """The reported case: T800_epoxy, Vp = 0.02, 6x3x24, compression; one
    solve per failure criterion (Tsai-Wu is the default)."""
    from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
    mat = MATERIALS["T800_epoxy"]
    pf = PorosityField(mat, 0.02, seed=0)
    mesh = CompositeMesh(pf, mat, nx=6, ny=3, nz=24)
    solver = FESolver(mesh, mat, pf)
    results = {"tsai_wu": solver.solve("compression")}
    for crit in ("hashin", "max_stress"):
        results[crit] = solver.solve("compression", failure_criterion=crit)
    return results


@pytest.fixture(scope="module")
def fe_thermal():
    """Thermal-only and combined solves of a CTE-carrying [0/90]_s laminate."""
    from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField
    mat = dataclasses.replace(MATERIALS["AS4_3501_6_epoxy"], n_plies=4, t_ply=0.125)
    layup = [0.0, 90.0, 90.0, 0.0]
    pf = PorosityField(mat, 0.02, seed=0)
    mesh = CompositeMesh(pf, mat, nx=6, ny=2, nz=4, ply_angles=layup)
    solver = FESolver(mesh, mat, pf, ply_angles=layup)
    return {"thermal": solver.solve("thermal", delta_T=-150.0),
            "combined": solver.solve("tension", delta_T=-150.0)}


def _export_fe(result, path, **kwargs):
    from porosity_fe import FESolver
    FESolver.export_results(result, path, **kwargs)
    return _strict_loads(path.read_text(encoding="utf-8"))


def test_fe_json_tsai_wu_mode_indices_are_null(fe_tsai_wu, tmp_path):
    """Regression: Tsai-Wu does not separate modes, so its per-mode indices
    are NaN in FieldResults, and the JSON export wrote them as the invalid
    token NaN."""
    import jsonschema

    r = fe_tsai_wu["tsai_wu"]
    assert r.failure_criterion == "tsai_wu"
    assert all(np.isnan(v) for k, v in r.failure_mode_indices.items() if k != "max_fi")

    path = tmp_path / "fe.json"
    doc = _export_fe(r, path, include_raw=True)
    jsonschema.validate(instance=doc, schema=_load_schema())
    modes = doc["failure"]["mode_indices"]
    assert set(modes) == set(r.failure_mode_indices)
    assert modes["max_fi"] == r.failure_mode_indices["max_fi"]
    assert all(v is None for k, v in modes.items() if k != "max_fi")
    # The loader hands the nulls back as None and the rest unchanged.
    loaded = load_results_from_json(path)
    assert loaded == doc
    assert loaded["failure"]["mode_indices"]["fiber_c"] is None


@pytest.mark.parametrize("criterion", ["hashin", "max_stress"])
def test_fe_json_separating_criteria_write_numbers(fe_tsai_wu, criterion, tmp_path):
    import jsonschema

    r = fe_tsai_wu[criterion]
    doc = _export_fe(r, tmp_path / "fe.json")
    jsonschema.validate(instance=doc, schema=_load_schema())
    modes = doc["failure"]["mode_indices"]
    assert modes == {k: float(v) for k, v in r.failure_mode_indices.items()}
    # Schema 1.3: each mode's maximum over the model.
    assert doc["failure"]["mode_maxima"] == r.failure_mode_maxima
    assert max(doc["failure"]["mode_maxima"].values()) == pytest.approx(
        doc["failure"]["max_failure_index"], rel=1e-12)


def test_fe_json_interlaminar_stress_summary(fe_tsai_wu, tmp_path):
    r = fe_tsai_wu["tsai_wu"]
    doc = _export_fe(r, tmp_path / "fe.json")
    assert doc["failure"]["mode_maxima"] is None
    local = doc["stress_local"]
    assert list(local) == ["sigma_11", "sigma_22", "sigma_33", "tau_23",
                           "tau_13", "tau_12"]
    for i, name in enumerate(local):
        component = r.stress_local[:, :, i]
        assert local[name]["max"] == pytest.approx(float(component.max()), rel=1e-12)


def test_fe_json_thermal_exports_are_standard(fe_thermal, tmp_path):
    import jsonschema

    th, comb = fe_thermal["thermal"], fe_thermal["combined"]
    assert np.isnan(th.knockdown)
    doc = _export_fe(th, tmp_path / "th.json", include_raw=True)
    jsonschema.validate(instance=doc, schema=_load_schema())
    assert doc["failure"]["knockdown_factor"] is None
    assert doc["failure"]["load_factor_basis"] == "delta_T"
    assert doc["stiffness"]["effective_modulus_MPa"] is None
    assert doc["thermal"]["delta_T_K"] == -150.0
    loaded = load_results_from_json(tmp_path / "th.json")
    assert loaded["failure"]["knockdown_factor"] is None

    doc = _export_fe(comb, tmp_path / "comb.json")
    jsonschema.validate(instance=doc, schema=_load_schema())
    assert doc["failure"]["knockdown_factor"] == comb.knockdown
    assert doc["failure"]["load_factor_basis"] == "mechanical_with_residual"


def test_fe_json_infinite_load_factor_is_null(fe_tsai_wu, tmp_path):
    """An unstressed model has an infinite first-ply-failure factor: null."""
    r = dataclasses.replace(fe_tsai_wu["hashin"],
                            first_ply_failure_load_factor=float("inf"),
                            interior_first_ply_failure_load_factor=float("inf"))
    fail = _export_fe(r, tmp_path / "fe.json")["failure"]
    assert fail["first_ply_failure_load_factor"] is None
    assert fail["interior_first_ply_failure_load_factor"] is None


def test_fe_json_refuses_unexpected_nan_before_writing(fe_tsai_wu, tmp_path):
    """A NaN in a field with no documented null meaning is a bug upstream:
    the writer raises, names the field, and leaves no file or sidecar."""
    from porosity_fe import FESolver

    bad = dataclasses.replace(fe_tsai_wu["tsai_wu"], max_failure_index=float("nan"))
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError,
                       match=r"non-finite value at \$\.failure\.max_tsai_wu_index"):
        FESolver.export_results(bad, path, include_raw=True)
    assert not path.exists()
    assert not (tmp_path / "bad.json.npz").exists()


def test_sweep_writer_refuses_nan_and_writes_nothing(tmp_path):
    results = _tiny_results()
    original = results["uniform_spherical"]
    bad = dataclasses.replace(
        original, config={**original.config, "Vp_scale": float("nan")})
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError, match=r"\$\.uniform_spherical\.config\.Vp_scale"):
        save_results_to_json({"uniform_spherical": bad}, str(path))
    assert not path.exists()


@pytest.mark.parametrize("payload, where", [
    ({"a": {"b": [1.0, float("nan")]}}, r"\$\.a\.b\[1\]"),
    ({"a": np.float64("inf")}, r"\$\.a"),
    ({"a": np.array([[0.0, 1.0], [2.0, -np.inf]])}, r"\$\.a\[1\]\[1\]"),
    ({"a": (1, 2.5, float("-inf"))}, r"\$\.a\[2\]"),
])
def test_strict_serializer_names_the_non_finite_value(payload, where):
    from porosity_fe.io import _dumps_json
    with pytest.raises(ValueError, match=where):
        _dumps_json(payload)


def test_strict_serializer_keeps_finite_payloads():
    from porosity_fe.io import _dumps_json
    payload = {"x": np.float64(1.5), "y": np.arange(3), "z": None, "s": "NaN"}
    assert _strict_loads(_dumps_json(payload)) == {
        "x": 1.5, "y": [0, 1, 2], "z": None, "s": "NaN"}


def test_loader_reads_legacy_nan_tokens_as_none(tmp_path, caplog):
    """Files written before the fix hold NaN tokens; they load as None, the
    same as a current file, with a warning."""
    import logging

    path = tmp_path / "legacy.json"
    save_results_to_json(_tiny_results(), path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["legacy_fe_block"] = {"fiber_t": float("nan"), "lam": float("inf")}
    path.write_text(json.dumps(doc), encoding="utf-8")   # allow_nan default
    assert "NaN" in path.read_text(encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="porosity_fe_analysis"):
        loaded = load_results_from_json(path)
    assert loaded["legacy_fe_block"] == {"fiber_t": None, "lam": None}
    assert "Infinity, NaN" in caplog.text


def test_every_package_json_call_disallows_nan():
    """Static guard: every json.dump / json.dumps call in the package passes
    allow_nan=False, so a new writer cannot reintroduce NaN tokens."""
    import ast
    from pathlib import Path

    import porosity_fe

    calls = []
    for src in sorted(Path(porosity_fe.__file__).parent.rglob("*.py")):
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("dump", "dumps")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"):
                kw = {k.arg: k.value for k in node.keywords}
                ok = (isinstance(kw.get("allow_nan"), ast.Constant)
                      and kw["allow_nan"].value is False)
                calls.append((f"{src.name}:{node.lineno}", ok))
    assert calls, "expected the shared serializer's json.dumps call"
    assert all(ok for _, ok in calls), [c for c, ok in calls if not ok]
