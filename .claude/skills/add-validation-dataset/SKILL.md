---
name: add-validation-dataset
description: Add a new peer-reviewed experimental porosity dataset to the validation database. Use when the user says things like "add the Smith 2024 dataset", "digitize Fig 3 from Liu 2018 and add it", or "add a new ILSS validation paper" — anything that creates a new JSON file under validation/datasets/ that the validate_porosity CLI and validation/validate_all.py runner will consume.
---

# Add a validation dataset

The validation database is a directory of per-paper JSON files under
`validation/datasets/`, schema-checked against
`validation/schemas/validation_dataset_schema.json`. The
`validate_porosity` CLI iterates the directory and runs each dataset
through the empirical pipeline to produce `validation_master_report.png`
+ `validation_detail_report.md`. The PyInstaller spec
(`ValidatePorosity.spec`) bundles every `*.json` under
`validation/datasets/` into the distributed executable, so a new dataset
ships automatically with the next release build.

The whole `validation/` tree is gitignored (the digitized values come
from published figures and stay out of the repo). New datasets are
shared via the bundled executable or by hand-off, not via git.

## Required information from the user

- **Paper identity**: full citation string, DOI, and the specific figure
  or table the data came from (e.g. "Fig 5a, digitized with
  WebPlotDigitizer" or "Table 2, tabulated").
- **Material**: fiber, matrix, fiber volume fraction (must be in
  `[0.3, 0.8]` — the schema enforces this), layup (`ply_angles` as a
  list of degrees), `n_plies`, and an optional human-readable
  `layup_name` (`"[0/45/90/-45]_s"`).
- **Porosity measurement method**: `acid digestion`, `matrix burnoff`,
  `μCT`, `ultrasonic C-scan`, etc.
- **Baseline porosity**: the void content `pct` (0–15) of the
  near-zero-porosity coupon group used to normalize the data.
- **At least one property block.** The schema accepts these keys (and
  nothing else, by `patternProperties`): `compression_strength`,
  `tensile_strength`, `transverse_tensile_strength`, `ilss`,
  `shear_strength`, `tensile_modulus`, `transverse_tensile_modulus`,
  `flexural_modulus`, `shear_modulus`. Each property block needs:
  - `test_standard` (ASTM / ISO / EN designation),
  - `void_content_pct` (list of numbers in `[0, 15]`),
  - `normalized_values` (list of the same length, values in `[0, 1.2]` —
    `σ(Vp)/σ(baseline)`),
  - and optionally `test_config`, `void_content_uncertainty`,
    `normalized_uncertainty`, `absolute_values_MPa`,
    `digitization_method` (`tabulated` | `webplotdigitizer` | `direct`).

If a property block is missing `void_content_pct` or `normalized_values`,
or their lengths don't match, refuse to write the file — the schema will
reject it.

## Steps

1. **Pick a filename.** Use the convention already in the directory:
   `<firstauthor>_<year>.json` lower-case (e.g. `liu_2018.json`,
   `zhang_peek_2025.json`). If the same first-author/year is already
   present, qualify with a one-word system tag like the existing
   `zhang_peek_2025.json`.

2. **Write the JSON.** Use `validation/datasets/elhajjar_2025.json` as
   the canonical template — it exercises the full schema (two property
   blocks, `test_config`, `digitization_method`, `notes`). Required
   top-level keys: `reference`, `material`, `properties`. Recommended
   extras: `doi`, `data_source`, `porosity_measurement`,
   `baseline_porosity_pct`, `notes`.

3. **Validate the JSON against the schema before saving** (the runner
   will reject it otherwise — see `validation/validate_all.py:load_dataset`):
   ```bash
   python -c "
   import json, jsonschema
   schema = json.load(open('validation/schemas/validation_dataset_schema.json'))
   data   = json.load(open('validation/datasets/<new_file>.json'))
   jsonschema.validate(data, schema)
   print('schema OK')"
   ```

4. **Run the full validation pipeline** to confirm the new dataset
   loads and predicts cleanly end-to-end:
   ```bash
   validate_porosity --output-dir /tmp/poros-val
   # Or in-process:
   python -m validation.validate_all
   ```
   Check `validation_detail_report.md` for the new dataset's
   per-property MAE row.

5. **Run the database test** so the dataset count / discoverability
   stays green:
   ```bash
   python -m pytest tests/test_validation_database.py -v
   ```

6. **Update README.md only if the dataset count changes a headline
   number** — the "13 peer-reviewed experimental datasets" line in
   `README.md` and the per-property MAE table in the "Validation"
   section both depend on the database contents. Re-run
   `validate_porosity` and copy the new aggregate MAEs (property- and
   point-weighted) from its run summary.

## What this skill does NOT cover

- **Adding a new material preset.** Validation datasets describe what
  was tested; the preset registry is independent. Use the
  `add-material-preset` skill if the new paper uses a system that isn't
  already in `MATERIALS`.
- **Schema changes.** If the user has data that doesn't fit any of the
  nine `patternProperties` keys, that's a schema edit, not a dataset
  add — surface it as a follow-up and don't widen the schema silently.
- **Committing or pushing the JSON.** The `validation/` tree is
  gitignored on purpose (digitized data from published figures). The
  dataset travels with the next PyInstaller build, not via git.
