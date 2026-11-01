# Uncertainty propagation

## Empirical knockdown

{func}`~porosity_fe.propagate_uncertainty` samples uncertain inputs and
runs each draw through
{meth}`EmpiricalSolver.get_failure_load <porosity_fe.EmpiricalSolver.get_failure_load>`.
It reports percentiles (by default 5/50/95) of the knockdown and the failure
stress. The deterministic pipeline is unchanged.

Three kinds of input can vary:

Material fields
: `covs={'sigma_1c': 0.05, ...}` gives a truncated lognormal with that
  coefficient of variation. `spec={field: (dist, param)}` chooses the
  distribution: `'lognormal'`, `'normal'` (param = CoV) or `'uniform'`
  (param = fractional half-width). Draws are clipped to stay physical:
  moduli and strengths stay positive and $V_f$ stays below the hexagonal
  packing limit $\pi / (2\sqrt3)$.

Porosity
: `vp_cov` makes the specimen-average $V_p$ lognormal about the nominal
  value, clipped to $[0, 1]$.

Knockdown coefficient
: `coef_cov` makes the calibrated coefficient for the mode ($\alpha$, $n$
  or $\beta$) a median-preserving lognormal. The layup scaling
  (`EmpiricalSolver.layup_scale`) is applied on top.

The knockdown is a function of $V_p$ and the coefficient only. Scatter in
the material strengths therefore widens the failure-stress band but leaves
the knockdown band at zero width. The coefficient is usually the dominant
uncertainty.

Sampling is plain Monte Carlo (`method='monte_carlo'`) or Latin hypercube
(`'lhs'`). Both are reproducible with `seed=`. Results can be written to
JSON with {func}`~porosity_fe.save_uq_results_to_json`.
`porosity-analyze --uq` and the web app expose the same calculation.

## Delamination load factor

The critical load factor $\lambda_c$ of a delamination
({doc}`delamination`, "Growth criterion and load factor") is a point
value, though its inputs scatter: the pristine toughness $G_{Ic}$,
$G_{IIc}$, $G_{IIIc}$ and the Benzeggagh-Kenane exponent $\eta$ of the
material, and the porosity-toughness coefficients $b_\text{I}$,
$b_\text{II}$, $b_\text{III}$, which are uncalibrated placeholders by
default. {func}`~porosity_fe.fracture_load_factor_uncertainty` and
{meth}`FieldResults.fracture_load_factor_uncertainty <porosity_fe.FieldResults.fracture_load_factor_uncertainty>`
put a band on it.

**One solve.** The analysis is linear, so the energy release rates of the
front from one FE solve do not depend on the toughness. Every draw
re-evaluates the factor of every front node,

$$
\lambda_{c,i} = \sqrt{G_c(\psi_i) / G_{T,i}} ,
$$

with $G_c$ from that draw's inputs, without solving again. Under a thermal
prestress ($\mathbf u = \mathbf u_T + \lambda \mathbf u_M$) the draws are
not a rescaling: each takes the smallest positive root of
$G_T(\lambda) = G_c(\psi(\lambda))$ per node, vectorised over the draws
(bracketed on the grid of the deterministic root, then bisected), which
agrees with {func}`~porosity_fe.fracture_load_factors` on the same inputs
to about $10^{-13}$. The contact set of the applied state is held, as the
deterministic factor already assumes. A combined solve keeps the
mechanical and the residual part of its front (`crack_front.parts`) for
this. The band's value per draw is the smallest factor over the front, and
`governing_fraction` gives the share of the draws in which each node
governs.

```python
r = solver.solve('flexure_3pt', applied_load=-100.0)        # with a Delamination
band = r.fracture_load_factor_uncertainty(
    material, covs={'G_Ic': 0.05, 'G_IIc': 0.07},
    n_samples=2000, method='lhs', seed=0, required_load_factor=1.5)
band.nominal, band.percentiles        # r.fracture_load_factor, {'p5': ..., 'p50': ..., 'p95': ...}
band.probability_below(1.5)           # share of the draws below 1.5
```

The method takes the solve's porosity-toughness law, prestress and
`load_factor_basis`, and refuses a material whose toughness does not
reproduce the solve's load factor.

**Inputs.** The vocabulary is that of the empirical band: `covs={field:
cov}` is a median-preserving lognormal, `spec={field: (dist, param)}` a
lognormal, normal or uniform; Monte Carlo or Latin hypercube, seeded. The
fields are `'G_Ic'`, `'G_IIc'`, `'G_IIIc'`, `'bk_eta'` (clipped to a
positive floor, as {meth}`MaterialProperties.perturb
<porosity_fe.MaterialProperties.perturb>` does) and `'b_I'`, `'b_II'`,
`'b_III'` (clipped at 0, so the law never raises the toughness). An unset
`G_IIIc` follows `G_IIc` and an unset `b_III` follows `b_II`, draws
included; scattering an unset one raises. Naming any other field
(stiffness, strength, porosity) raises: it changes $G$ and would need a
solve per draw. Draws are independent between fields.

**No default scatter.** Nothing scatters unless you name it. Published
scatter is scarce, and what follows are starting points, not defaults:

| Input | CoV | Source |
|---|---|---|
| $G_{Ic}$, IM7/8552 | 1.5 to 5.5 % | Czabaj and Ratcliffe 2013, doi:10.1016/j.compscitech.2013.09.008 |
| $G_{IIc}$, IM7/8552, precracked | 6.5 to 7.6 % | O'Brien et al., NASA/TM-2010-216838 |

Both are one material and one laboratory. The precracking method alone
moves the mean $G_{Ic}$ of IM7/8552 by 13 % (0.206 N/mm fatigue
precracked against 0.237 from the insert), more than the scatter within a
method. The spread of the placeholder coefficients, $b_\text{I}$ from 4.97
to 7.45 and $b_\text{II}$ from 12.7 to 15.9 ({doc}`delamination`, "Porosity
and toughness"), is disagreement between two-point estimates from
different studies, not measured scatter.

**The band.** {class}`~porosity_fe.FractureLoadFactorBand` holds the
nominal value (the deterministic `fracture_load_factor`), the draws per
node and over the front, their mean, standard deviation, extremes and
percentiles (an infinite factor, where a draw drives no node, stays
infinite), `governing_fraction`, the inputs and their scatter, the law and
`load_factor_basis`. `probability_below(r)` is the share of the draws
below a load factor `r` (`probability_below_required` at the
`required_load_factor` passed in): an estimate under the input
distributions that were entered, **not a disposition**. The tool
recommends no disposition for a delamination (the NCR assessment leaves it
to the MRB), and the band does not change that. `summary()` is the
strict-JSON form that
{meth}`FESolver.export_results(..., fracture_uncertainty=band) <porosity_fe.FESolver.export_results>`
writes as the opt-in `fracture_uncertainty` block (schema 1.3, additive).
The web app's sidebar offers the band once the toughness is entered
(CoVs for $G_{Ic}$, $G_{IIc}$, $\eta$, $b_\text{I}$, $b_\text{II}$; 2000
Latin-hypercube draws), and the Results tab's Defects block shows p5, p50
and p95, a histogram and the share of draws below a factor you enter.

**Verification** (`tests/test_fracture_uncertainty.py`). With no scatter
every draw is the deterministic $\lambda_c$ bit for bit, with and without
a prestress. Every draw equals {func}`~porosity_fe.fracture_load_factors`
on that draw's inputs ($10^{-13}$; with a prestress $10^{-11}$). On a pure
mode I front $\lambda_c = \sqrt{G_{Ic} f_\text{I} / G_T}$, so a lognormal
$G_{Ic}$ with $\sigma = \sqrt{\ln(1 + \text{CoV}^2)}$ gives the
percentiles $\lambda_0 \exp(\sigma\,\Phi^{-1}(p) / 2)$. On a 3 mm UD
double-cantilever beam with $\lambda_0 = 5.712$ and 4000 Latin-hypercube
draws:

| CoV of $G_{Ic}$ | p5 / p50 / p95, band | closed form |
|---|---|---|
| 0.1 | 5.263 / 5.712 / 6.200 | 5.262 / 5.712 / 6.200 |
| 0.3 | 4.488 / 5.712 / 7.270 | 4.487 / 5.712 / 7.272 |

**Not covered.** The band is the toughness inputs' share of the
uncertainty only. It leaves out the porosity field and its measurement,
the stiffness and its knockdown, the geometry and planform of the
delamination, the load, the mesh (the front is a staircase on the grid)
and the model form: linear VCCT with a growth-onset criterion, the
B-K interpolation, and the conservative porosity law, which ignores the
rise of mode I propagation toughness with voids through fibre bridging
({doc}`delamination`, "Porosity and toughness"). The inputs are drawn
independently, though $G_{Ic}$ and $G_{IIc}$ of one material are likely
correlated.
