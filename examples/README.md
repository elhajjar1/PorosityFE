# PorosityFE examples

Small, runnable Python scripts that exercise the public API end-to-end.
Each script:

- is standalone (no shared helpers) and runs in a few seconds
  (`distribution_comparison.py`, which also runs the FE solver, in about
  20 s),
- uses the `Agg` matplotlib backend so it works headlessly (CI, SSH),
- prints a short summary to stdout,
- writes its PNG to `examples/output/` (gitignored), or to the directory
  named by `POROSITY_FE_EXAMPLES_OUTPUT` when that is set.

`tests/test_examples.py` runs every script in CI, and checks that
`distribution_comparison.py` still prints the knockdowns the top-level
README quotes from it; a change that moves them updates the README in the
same PR.

Run any example from the repo root:

```bash
python examples/uniform_spherical.py
```

Or sweep them all:

```bash
for f in examples/*.py; do python "$f" || echo "FAILED: $f"; done
```

## Scripts

| Script | What it demonstrates | PNG output |
|---|---|---|
| [`uniform_spherical.py`](uniform_spherical.py) | Uniform porosity, spherical voids; empirical compression knockdown (Judd-Wright). | `output/uniform_spherical_profile.png` |
| [`clustered_midplane.py`](clustered_midplane.py) | Gaussian-clustered porosity at the midplane; ILSS knockdown. | `output/clustered_midplane_profile.png` |
| [`interface_penny.py`](interface_penny.py) | Interface-concentrated porosity with penny-shaped voids: the through-thickness profile and the empirical knockdowns, which depend only on the mean `Vp` (the FE comparison is in `distribution_comparison.py`). | `output/interface_penny_profile.png` |
| [`discrete_voids.py`](discrete_voids.py) | Explicit `VoidGeometry` ellipsoids layered on top of a low-uniform background; visualizes the SCF field around one void. | `output/discrete_voids_scf.png` |
| [`compute_degraded_clt_moduli.py`](compute_degraded_clt_moduli.py) | CLT path: laminate effective moduli `(Ex, Ey, Gxy)` vs. `Vp` for a quasi-isotropic layup. | `output/clt_degraded_moduli.png` |
| [`distribution_comparison.py`](distribution_comparison.py) | `uniform` vs `clustered` vs `interface` at matched `Vp_mean`: the empirical knockdowns coincide, the FE knockdowns differ (#83). The README's tables come from it. | `output/distribution_profiles.png`, `output/distribution_fe_compare.png` |

## Conventions used by these scripts

All matrices/tensors that appear in the API use Voigt order
`[11, 22, 33, 23, 13, 12]` with **engineering** shear strain
(`gamma_ij = 2 * eps_ij`). The empirical knockdown solver
(`EmpiricalSolver.apply_loading`) returns the failure stress as a positive
magnitude regardless of mode (compression / tension / shear / ILSS); the
mode is recorded in the result dict, not encoded in the sign of the stress.
The FE path (`FieldResults`) instead stores signed stress and strain
components in the same Voigt order. See the docstring **Notes** blocks on
`MaterialProperties.get_stiffness_matrix`, `Hex8Element.B_matrix`,
`FieldResults`, and `EmpiricalSolver.apply_loading` for the authoritative
statements.
