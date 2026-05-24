---
name: add-material-preset
description: Add a new composite material system to the MATERIALS preset registry. Use when the user says things like "add T1100/epoxy as a preset", "register a new material", or "add an HM63 carbon system" — anything that needs a new keyed entry under porosity_fe/materials.py:MATERIALS so it shows up in the Streamlit sidebar dropdown and is selectable by name from compare_configurations / build_empirical_pipeline.
---

# Add a material preset

A preset is a `MaterialProperties(...)` value stored under a string key in
the `MATERIALS` dict in `porosity_fe/materials.py`. The Streamlit sidebar
reads `list(MATERIALS.keys())` (`app.py:479`) and the analysis pipeline
resolves names via `MATERIALS[cfg["material_name"]]` (`app.py:120-125`),
so a single registry entry covers both the UI dropdown and the CLI.

## Required information from the user

Get all of these before writing — `MaterialProperties` has no defaults
for the engineering constants, so a missing field is a hard error:

- **Identity**: short snake_case key (e.g. `T1100_epoxy`), fiber name,
  matrix resin name, and a citation for the lamina properties (Daniel &
  Ishai, WWFE, manufacturer data sheet, etc.).
- **Stiffness (MPa)**: `E11, E22, E33, G12, G13, G23, nu12, nu13, nu23`.
- **Strength (MPa)**: `sigma_1c, sigma_1t, sigma_2t, sigma_2c, tau_12, tau_ilss`.
- **Geometry (mm / count)**: `t_ply`, `n_plies`.
- **Constituents** (used by the FE Eshelby/Mori-Tanaka micromechanics path —
  populate even if the user only plans to run the empirical solver, both
  paths read from the same dataclass): `matrix_modulus, matrix_poisson,
  fiber_modulus, fiber_volume_fraction`.

If the user only has a partial spec, ask for the missing fields before
editing — don't invent values.

## Steps

1. **Sanity-check the inputs.** Confirm:
   - `E11 > E22` (fiber direction stiffer than transverse).
   - `nu12 < 0.5` and the orthotropic compliance is positive-definite
     (rough check: `1 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2*nu21*nu32*nu13 > 0`,
     using `nu21 = nu12 * E22 / E11` etc.).
   - `fiber_volume_fraction` in `[0.3, 0.8]` (the validation-dataset
     schema enforces this bound for a reason — quote that to the user
     if their `Vf` is outside it).

2. **Edit `porosity_fe/materials.py`.** Add the entry to the `MATERIALS`
   dict near line 495, keeping the existing style (multiline block with
   the same field ordering as the surrounding presets). Include a
   one-line citation comment above the entry pointing at the source
   (e.g. `# AS4/3501-6 — Soden, Hinton & Kaddour, WWFE-I, CST 1998`).

3. **Wire it into the existence-check test.** In
   `tests/test_materials.py`, the `TestMATERIALS` block (around line 31)
   asserts each preset key is in `MATERIALS` and pulls representative
   fields. Add an analogous assertion for the new key so a future
   accidental deletion fails CI.

4. **Run the gate locally:**
   ```bash
   python -m pytest tests/test_materials.py -v
   ruff check porosity_fe/materials.py
   mypy porosity_fe/materials.py
   ```
   (Use `python -m pytest` rather than bare `pytest` — the web-container
   `pytest` is uv-isolated and can't see the project deps; see
   `CLAUDE.md`.)

5. **Don't touch `app.py`, `compare_configurations`, or the validation
   datasets.** The Streamlit dropdown picks up new keys automatically
   from `list(MATERIALS.keys())`; `compare_configurations` resolves the
   name through the same dict; validation datasets describe what was
   experimentally tested and aren't tied to the preset registry.

## What this skill does NOT cover

- **Calibrating empirical `alpha`/`n`/`beta` for the new material.** Those
  live on `EmpiricalSolver` (not on `MaterialProperties`) and need
  coupon data to fit. See README "Calibrating `alpha` / `n` for a custom
  material". If the user wants the new preset to ship with non-QI
  coefficients, surface that as a separate follow-up.
- **Adding a validation dataset for the new material.** Use the
  `add-validation-dataset` skill for that.
