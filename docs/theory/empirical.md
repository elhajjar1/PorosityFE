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

The coefficients below ({class}`~porosity_fe.Calibration`) come from
Elhajjar (2025). They were tuned with the layup scaling of the next section
already applied, so they are the values at the reference
$f_\text{md} = 0.5$, not raw fits to one coupon layup.

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
the solver emits a `UserWarning`. {meth}`~porosity_fe.EmpiricalSolver.get_failure_load`
checks the specimen average. {meth}`~porosity_fe.EmpiricalSolver.apply_loading`
also checks the local peak of a clustered or interface profile, because its
per-node field evaluates the law at the local $V_p$.

## Layup scaling

Porosity hurts matrix-dominated layups more than fiber-dominated ones. The
solver scales each coefficient with a matrix-dominated fraction
$f_\text{md}$ computed from the ply angles. Each ply contributes:

| Ply angle $\lvert\theta\rvert \bmod 180°$ | Contribution |
|---|---|
| $\le 10°$ | 0 |
| $\ge 80°$ | 1 |
| otherwise | 0.5 |

$f_\text{md}$ is the mean contribution over the plies. The scale factor is

$$
s = \max\!\left(\frac{f_\text{md}}{0.5},\ s_\text{floor}\right),
$$

with $s_\text{floor} = 0.15$ for most modes. ILSS and transverse tension
use $s_\text{floor} = 0.80$, because they stay matrix-dominated whatever
the fiber layup. The scaled coefficients are $\alpha s$, $\max(n s, 0.1)$
and $\beta s$.

| Layup | $f_\text{md}$ | $s$ (compression) | $\alpha_\text{eff}$ (compression) |
|---|---|---|---|
| $[0]_{16}$ | 0.00 | 0.15 (floor) | 1.035 |
| $[0/90/\pm45]_s$ (QI) | 0.50 | 1.00 | 6.90 |
| $[\pm45]_{4s}$ | 0.50 | 1.00 | 6.90 |
| $[90]_8$ | 1.00 | 2.00 | 13.80 |

```{note}
The floors 0.15 and 0.80 are empirical tuning constants with no published
derivation (issue #139). The linear $f_\text{md}/0.5$ rule differs by up to
about 33 % from a CLT stiffness-retention proxy for UD-heavy layups
(issue #140). Recalibrating the layup scaling against layup-varying data is
planned work.
```

User overrides `judd_wright_alpha=`, `power_law_n=` and `linear_beta=` are
per-mode dicts. They replace the QI values for the modes given and are
then scaled the same way. A user callable `model(Vp, mode) -> KD` bypasses
the table and the scaling entirely; it is checked on a grid for finite
values in $[0, 1]$.

## Specimen average vs. local porosity

The correlations were fitted to specimen-average void contents, so the
failure load uses $\bar V_p$, and the distribution shape has no effect. The
per-node knockdown field from
{meth}`~porosity_fe.EmpiricalSolver.apply_loading` evaluates the law at
each node's local $V_p$. It is meant for visualizing where the porosity
sits, not as a second failure prediction.

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

If the coupon layup is not quasi-isotropic, divide the fitted coefficient
by that layup's scale $s$ before passing it as an override.
