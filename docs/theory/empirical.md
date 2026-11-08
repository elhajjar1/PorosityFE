# Empirical strength knockdown

{class}`~porosity_fe.EmpiricalSolver` predicts a porous strength as

$$
\sigma_f = \sigma_0 \cdot KD_\text{por}(\bar V_p) \cdot KD_\text{env} \cdot KD_\text{fat},
$$

where $\sigma_0$ is the pristine strength for the loading mode,
$KD_\text{por}$ is a calibrated porosity knockdown evaluated at the
specimen-average porosity $\bar V_p$, and the hygrothermal and fatigue
factors ({doc}`fatigue_environment`) are 1 unless requested.

## Loading modes

| Mode | Pristine strength | Dominated by |
|---|---|---|
| `compression` | $\sigma_{1c}$ | fiber and matrix (microbuckling) |
| `tension` | $\sigma_{1t}$ | fiber |
| `shear` | $\tau_{12}$ | matrix |
| `ilss` | $\tau_\text{ilss}$ (short-beam shear) | matrix and interface |
| `transverse_tension` | $\sigma_{2t}$ | matrix |

There is no transverse-compression mode: no dataset in the validation
database measures $\sigma_{2c}$ against porosity. The FE failure criteria
still use $\sigma_{2c}$.

Two further modes, `mode_I_toughness` and `mode_II_toughness`, return the
porosity knockdown of the interlaminar toughness instead of a strength
(see [Toughness modes](#toughness-modes)). They are optional: no sweep,
report or app view includes them.

## Knockdown laws

Each law takes $V_p$ as a fraction and returns $KD \in (0, 1]$, with
$KD(0) = 1$. All three live in one table,
`porosity_fe.empirical._KNOCKDOWN_LAWS`, which every code path reads.

Judd-Wright
: $KD = e^{-\alpha V_p}$. For small $V_p$, $KD \approx 1 - \alpha V_p$, so
  $\alpha$ is the fractional strength loss per unit $V_p$. Judd & Wright
  (1978) reported ILSS falling about 7 % per 1 % voids, i.e.
  $\alpha \approx 7$.

Power law
: $KD = (1 - V_p)^n$. The form follows Mackenzie's (1950) spherical-void
  elasticity, generalized empirically (Rice 2005). $n = 1$ is a simple
  area reduction; $n > 1$ adds stress concentration.

Linear
: $KD = \max(1 - \beta V_p,\ 0)$. This is the direct reading of Judd &
  Wright's linear data. It reaches zero at $V_p = 1/\beta$, so it is only
  meaningful at low porosity.

## Calibrated coefficients

The coefficients below ({class}`~porosity_fe.Calibration`) come from the
Elhajjar (2025) $[0/45/90/-45/0]_s$ coupons. A least-squares fit of
$\ln KD$ against $V_p$ on that dataset alone gives $\alpha = 6.95$
(compression) and $4.21$ (tension), close to the tabulated values. Every
quasi-isotropic and in-plane isotropic layup uses them unscaled.

| Mode | $\alpha$ (Judd-Wright) | $n$ (power law) | $\beta$ (linear) |
|---|---|---|---|
| `compression` | 6.9 | 2.8 | 5.5 |
| `tension` | 3.9 | 1.8 | 3.5 |
| `shear` | 8.0 | 3.5 | 7.0 |
| `ilss` | 10.0 | 4.5 | 9.0 |
| `transverse_tension` | 10.0 | 4.5 | 9.0 |

`transverse_tension` reuses the ILSS values because both fail by matrix
and interface mechanisms.

**Validity.** The calibration data cover $V_p \lesssim 0.05$. Above that
the solver emits a `UserWarning`. Beyond the bound the compression law is
conservative on the one dataset that goes there: the quasi-isotropic
coupons of Cinquin et al. 2007 (`cinquin_2007_qi`, artificial porosity)
keep 0.79 to 0.83 of their strength at 7.8 to 11 % voids, where
$e^{-6.9 V_p}$ gives 0.46 to 0.58. Within the bound the evidence is
mixed: the Elhajjar (2025) coupons fit $\alpha = 6.95$, while the
quasi-isotropic graphite/epoxy of Uhl et al. 1988 (`uhl_1988`) loses
10.6 % from 0.34 to 4.05 % voids: the fit printed in that paper is
$\alpha = 3.1$. {meth}`~porosity_fe.EmpiricalSolver.get_failure_load`
checks the specimen average. {meth}`~porosity_fe.EmpiricalSolver.apply_loading`
also checks the local peak of a clustered or interface profile, because its
per-node field evaluates the law at the local $V_p$.

## Layup scaling

Only the fiber-direction laminate modes, `tension` and `compression`, are
layup-scaled. `shear`, `ilss` and `transverse_tension` are ply or
interlaminar matrix properties ($\tau_{12}$, $\tau_\text{ilss}$,
$\sigma_{2t}$), so their scale is $s = 1$ for every layup.

For the fiber-direction modes, `porosity_fe._layup._membrane_energy_partition`
loads the pristine laminate with a unit membrane resultant $N_x$. CLT,
membrane only, gives the mid-plane strain $\varepsilon^0 = A^{-1} N$. In
each ply's material axes, the strain energy then splits into fiber,
transverse and shear parts:

$$
(e_1, e_2, e_6) \propto \sum_k \left(\sigma_{11}\varepsilon_{11},\
\sigma_{22}\varepsilon_{22},\ \tau_{12}\gamma_{12}\right)_k,
\qquad e_1 + e_2 + e_6 = 1 .
$$

The fractions weight the calibrated per-mode coefficients:

$$
\alpha_\text{blend} = e_1\,a_\text{fib} + e_2\,a_2 + e_6\,a_6,
\qquad
s = \max\!\left(1,\ \frac{\alpha_\text{blend}}{\alpha_\text{QI}}\right).
$$

- Tension uses $a_2 = \alpha_\text{QI}(\texttt{transverse\_tension})$ and
  $a_6 = \alpha_\text{QI}(\texttt{shear})$.
- Compression uses $a_2 = a_6 = \alpha_\text{QI}(\texttt{shear})$, because
  no transverse-compression mode is calibrated.
- $a_\text{fib}$ is solved so that $\alpha_\text{blend}(\text{QI}) =
  \alpha_\text{QI}$. The rule adds no fitted constant.

The scaled coefficients are $\alpha s$, $\max(n s, 0.1)$ and $\beta s$.
The same $s$, computed from the Judd-Wright QI table, applies to all three
laws.

Properties of the scale:

- It is 1 for UD, cross-ply, QI and every in-plane isotropic layup (for
  example $[0/\pm60]_s$, which has the same $A$ matrix as QI).
- It is never below 1 and never above the matrix anchors,
  $10/3.9 = 2.56$ for tension and $8/6.9 = 1.16$ for compression.
- It is continuous in ply angle and independent of stacking order.

| Layup (T800/epoxy) | $s$ tension | $s$ compression |
|---|---|---|
| $[0]_n$, $[0/90]_s$, $[0_2/90]_s$, $[0/\pm15]_s$, QI, $[0/\pm60]_s$ | 1.00 | 1.00 |
| $[\pm30]_{2s}$ | 1.55 | 1.08 |
| $[\pm45]_{2s}$ | 1.94 | 1.14 |
| $[90]_8$ | 2.56 | 1.16 |

The per-mode values are on the solver's `layup_scale` attribute
({class}`~porosity_fe.EmpiricalSolver`).

```{note}
**Scales above 1 are unvalidated.** None of the 17 bundled strength and
modulus datasets is modelled with an angle-ply, off-axis or 90°-rich layup.
Twelve are UD (including the two ±45 in-plane shear datasets, given in
lamina axes), two are cross-ply and three are QI, and all of them get
$s = 1$. The amplification
therefore rests only on CLT and the calibrated mode alphas.
{meth}`~porosity_fe.EmpiricalSolver.get_failure_load` records any scale
above 1 in `details['layup_scale']` and emits one `UserWarning` per call.

This rule replaced a binned matrix-dominated fraction, $s = \max(f_\text{md}/0.5,
s_\text{floor})$, with floors of 0.15 and 0.80 that had no traceable source
(issues #139 / #140). That rule made UD tension, compression and shear 85 %
less porosity-sensitive than QI. The validation data do not support that:
the best-fit scale for UD coupons averages 0.9 to 1.4 depending on the mode,
so the floor was the largest error source in the database. Replacing it
lowers the
property-weighted validation MAE from 7.05 % to 4.49 % (on the 13 datasets of
the time, with baseline points counted). The study behind the
change is IMPROVEMENT_PLAN item 2.7. `Calibration.F_MD_REF`, `F_MD_FLOOR` and
`F_MD_FLOOR_ILSS` are deprecated, have no effect, and will be removed in 2.0.
```

User overrides `judd_wright_alpha=`, `power_law_n=` and `linear_beta=` are
per-mode dicts. They replace the QI values for the modes given and are
then scaled the same way (the scale itself always comes from the QI table). A user callable `model(Vp, mode) -> KD` bypasses
the table and the scaling entirely; it is checked on a grid for finite
values in $[0, 1]$.

## Specimen average vs. local porosity

The correlations were fitted to specimen-average void contents, so the
failure load uses $\bar V_p$, and the distribution shape has no effect. The
per-node knockdown field from
{meth}`~porosity_fe.EmpiricalSolver.apply_loading` evaluates the law at
each node's local $V_p$. It is meant for visualizing where the porosity
sits, not as a second failure prediction.

## Toughness modes

`mode_I_toughness` and `mode_II_toughness` screen the interlaminar
toughness without FE (delamination plan item 4.3), for example for a
nonconformance report:

$$
KD = \frac{G_c(\bar V_p)}{G_{c0}} = f_\text{I}(\bar V_p) \ \text{or}\
f_\text{II}(\bar V_p),
$$

with $f$ the porosity-toughness law of {doc}`delamination` ("Porosity and
toughness"), the same {class}`~porosity_fe.ToughnessKnockdown` the FE
fracture path applies at each crack-front node. The solver takes it as
`EmpiricalSolver(..., toughness_knockdown=...)`; the default
`'exponential'` law, $e^{-b V_p}$ with $b_\text{I} = 6.2$ and
$b_\text{II} = 14$, has uncalibrated placeholder coefficients.
`KD` equals {func}`~porosity_fe.toughness_knockdown_factor` at the same
$\bar V_p$, bit for bit.

```python
import dataclasses
from porosity_fe import MATERIALS, ToughnessKnockdown, build_empirical_pipeline

# IM7/8552 toughness on the T800 preset, for illustration (N/mm).
mat = dataclasses.replace(MATERIALS["T800_epoxy"], G_Ic=0.212, G_IIc=0.774)
_pf, _mesh, emp = build_empirical_pipeline(mat, 0.02)
res = emp.get_failure_load("mode_II_toughness")
res.knockdown          # exp(-14 * 0.02) = 0.756
res.failure_stress     # G_c = 0.774 * 0.756 = 0.585 N/mm, not a stress
res.details["toughness_uncalibrated"]  # True: the placeholder b_II

_pf, _mesh, own = build_empirical_pipeline(
    mat, 0.02, solver_kwargs={"toughness_knockdown": ToughnessKnockdown(b_I=4.0, b_II=10.0)})
```

The result is a {class}`~porosity_fe.FailureResult` like the strength
modes', with these differences:

- `failure_stress` holds the toughness $G_{c0}\,KD$ in N/mm, with $G_{c0}$
  the material's `G_Ic` or `G_IIc`. Both fields are optional. When the
  one needed is `None`, the knockdown is still returned (the law does not
  use the material), `failure_stress` is `nan` and
  `details['pristine_toughness']` is `None`. No preset except
  `IM7_8552_epoxy` sets them.
- `model` names the law used: `'toughness_exponential'`,
  `'toughness_linear'`, `'toughness_none'` or
  `'toughness_<callable name>'`. The `model` argument selects a strength
  law, so a built-in name is accepted and does not change the result; a
  callable is refused, because the toughness law is given once, as
  `toughness_knockdown`.
- `details['toughness_uncalibrated']` is `True` when the law uses a
  placeholder coefficient for the mode and $\bar V_p > 0$, the rule of
  `CrackFront.toughness_uncalibrated` on the FE path; the first such use
  logs one INFO line per process. `details['toughness_knockdown']`
  describes the law.
- The layup scale is 1: the toughness is an interface property, and the
  modes are not keys of `layup_scale`. The discrete-void stress
  concentration is a stress effect and does not apply.
- `environment=` and `cycles=` raise: no hygrothermal or fatigue law of
  the toughness is calibrated.
- Above $\bar V_p = 0.05$ the `'exponential'` and `'linear'` laws warn as
  the strength laws do (the toughness data span $V_p \le 0.05$); a
  callable law and `'none'` do not.

{meth}`~porosity_fe.EmpiricalSolver.apply_loading` gives the per-node
field of the law at the local $V_p$, and
{meth}`~porosity_fe.EmpiricalSolver.local_sensitivities` the closed-form
partials with respect to $V_p$ and $b$ (those of Judd-Wright for the
exponential law, of the linear law for the linear one, 0 for `'none'`; a
callable law has no closed form and goes through
{meth}`~porosity_fe.EmpiricalSolver.sensitivity_fd`).
{func}`~porosity_fe.propagate_uncertainty` takes the modes too
({doc}`uncertainty`).

The modes are in no default sweep:
{meth}`~porosity_fe.EmpiricalSolver.get_all_failure_loads`,
{func}`~porosity_fe.compare_configurations`, `porosity-analyze` (its
`--uq` included), the app and the NCR disposition, whose governing
knockdown is the lowest strength knockdown, use the five strength modes
only. The FE solver refuses them, like `transverse_tension`: the FE
fracture path computes the toughness at the crack front itself. The
validation runner predicts the toughness datasets with
{func}`~porosity_fe.toughness_knockdown_factor`, which gives the same
values (`tests/test_empirical_toughness.py`).

## Calibrating a new material

1. Make a ladder of coupons spanning $V_p \approx 0$–$5\,\%$, for
   example by varying debulk pressure or cure vacuum.
2. Measure void content (ASTM D2734 or D3171) and cross-check it by
   micro-CT or polished sections.
3. Run the strength test for the mode: D2344 (ILSS), D3039 (tension),
   D6641 (compression), and so on.
4. Normalize by the void-free baseline, $KD = \sigma(V_p)/\sigma(0)$.
5. Regress $\ln KD$ against $V_p$ (slope $-\alpha$) for Judd-Wright, or
   $\ln KD$ against $\ln(1 - V_p)$ (slope $n$) for the power law.

For `shear`, `ilss` and `transverse_tension`, and for `tension` /
`compression` coupons that are UD, cross-ply or quasi-isotropic, the fitted
coefficient is the override as is. For a coupon whose layup scale $s$ is
above 1, divide the fitted coefficient by $s$ before passing it as an
override.
