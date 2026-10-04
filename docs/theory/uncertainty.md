# Uncertainty propagation

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
