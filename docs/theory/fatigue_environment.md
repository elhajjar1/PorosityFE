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

The slopes are calibrated at stress ratio $R = 0.1$ (tension-tension). No
Goodman or Walker mean-stress correction is applied: another `R` does not
change the result, and the model warns that it was ignored. Below the 1 %
floor the value is clamped, with a warning.

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
