# Fatigue and hygrothermal knockdowns

{meth}`~porosity_fe.EmpiricalSolver.get_failure_load` can multiply the
porosity knockdown by two more factors. Both are screening-level models;
design allowables should come from test data (for example CMH-17 Vol. 2
protocols). Both factors are 1 unless asked for, and when active they are
reported separately in `FailureResult.details`. The FE solver does not
apply either one.

## S-N fatigue

{class}`~porosity_fe.FatigueModel` uses the log-linear (Mandell) form

$$
KD_\text{fat} = \max\!\left(0.01,\ 1 - b \log_{10} N\right),
$$

where $N \ge 1$ is the number of cycles (`cycles=`). The default slopes
$b$ ({attr}`Calibration.FATIGUE_B_QI <porosity_fe.Calibration>`) are:

| Mode | $b$ per decade |
|---|---|
| `tension`, `compression`, `transverse_tension` | 0.10 |
| `shear`, `ilss` | 0.08 |

The tension and compression value follows Mandell's (1991) review of
IM-class CFRP. The shallower matrix-dominated slope follows Curtis (1989).
A per-mode dict `FatigueModel(b=...)` overrides them.

The slopes are calibrated at stress ratio $R = 0.1$ (tension-tension). By
default no mean-stress correction is applied: another `R` does not change
the result, and the model warns that it was ignored. Below the 1 % floor
the value is clamped, with a warning.

### Mean-stress correction (opt-in)

`FatigueModel(mean_stress='goodman')` moves the curve to another stress
ratio along a straight constant-life line, the Goodman construction. In
mean stress $m$ and amplitude $a$, both over the static strength $S_0$,
the line runs from the calibrated point at $r_0 = 0.1$,

$$
(m_0, a_0) = \left(x\,rac{1 + r_0}{2},\ x\,rac{1 - r_0}{2}ight),
\qquad x = KD_	ext{fat}(N),
$$

to the static strength $(1, 0)$. A cycle at load ratio $r$ with maximum
$s$ has $m = s(1 + r)/2$ and $a = s(1 - r)/2$, so on the line

$$
KD_	ext{fat}(N, r) = rac{2k}{(1 - r) + k(1 + r)},
\qquad k = rac{a_0}{1 - m_0} = rac{x(1 - r_0)}{2 - x(1 + r_0)} .
$$

At $r = r_0$ it is the calibrated curve, and at $r = 1$ (no cycling) it is
1. $r$ is the ratio of the smaller to the larger load magnitude, so the
correction covers same-sign cycling only:

| Mode | Accepted `R` | $r$ |
|---|---|---|
| `tension`, `shear`, `ilss`, `transverse_tension` | $0 \le R \le 1$ | $R$ |
| `compression` | $R \ge 1$ (compression-compression) | $1/R$ |

Reversed loading ($R < 0$) raises: it needs fatigue data at $R = -1$.
The compression slope has no ratio of its own in the table; $R = 10$, the
same load-magnitude ratio as $R = 0.1$, is taken as its calibration ratio.
No single `R` suits both families, so `R=` also takes a mapping of mode to
`R` (`R={'tension': 0.5, 'compression': 2.0}`); a mode left out takes its
calibration ratio.

A linear constant-life line is conservative for many carbon-fibre
laminates, whose constant-life diagrams bulge outwards (and are often
shifted towards compression). It is not calibrated against porous-laminate
fatigue data here.

### Porosity-dependent slope (opt-in)

`FatigueModel(porosity_slope=K)` steepens the curve with the void content:

$$
b(V_p) = b\,(1 + K\,V_p),
$$

with $V_p$ a fraction. `K = 0` (the default) keeps the slope independent
of porosity. No value of `K` is shipped: the validation database holds no
fatigue data to fit it on, so pass one fitted to your own tests. Pass the
model to the solver as `EmpiricalSolver(..., fatigue_model=...)` (or
`build_empirical_pipeline(..., solver_kwargs={'fatigue_model': ...})`): the
failure load takes the slope at the specimen-average `Vp`, and
{meth}`~porosity_fe.EmpiricalSolver.allowable_porosity` at each trial void
content, so a porosity-dependent slope lowers the allowable.

### The durability table

{func}`porosity_fe.reporting.durability_record` lists every strength mode
and law with the porosity knockdown, the fatigue and hygrothermal factors
and their product. `porosity-analyze --cycles N` (with `--mean-stress`,
`--stress-ratio`, `--compression-stress-ratio`, `--fatigue-porosity-slope`)
and `--service-temperature` / `--moisture` / `--tg-dry` write it to
`porosity_durability_<Vp>.json` and hold the allowable porosity of
`--required-knockdown` at the same conditions; the app's Results tab has a
"Fatigue and environment" panel.

## Hygrothermal conditioning

{meth}`MaterialProperties.environment_knockdown
<porosity_fe.MaterialProperties.environment_knockdown>` implements the
Chamis (1983) matrix-property ratio, with the Springer rule of thumb for
the wet glass transition:

$$
KD_\text{env} = \min\!\left(1,\ \sqrt{\frac{T_{g,\text{wet}} - T}{T_{g,\text{dry}} - T_\text{ref}}}\right),
\qquad
T_{g,\text{wet}} = T_{g,\text{dry}} - 25\,M ,
$$

with temperatures in °C and moisture $M$ in wt%. $T$ and $M$ come from
`environment={'T': ..., 'M': ...}` or from the material's `T_service` /
`M_service`; $T_\text{ref}$ defaults to 23 °C. If $T$, $M$ or
$T_{g,\text{dry}}$ is unset, the factor is 1. When the service temperature
reaches $T_{g,\text{wet}}$, the factor is clamped to 0.01.

Tension is treated as fiber-dominated and always returns 1. Compression,
shear, ILSS and transverse tension get the full ratio. Compression is
included because fiber microbuckling depends on the matrix shear
stiffness; this is conservative for a fiber-failure mode.
