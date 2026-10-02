# Failure criteria

After the FE solve, the stress at every Gauss point is rotated into ply
axes and checked against porosity-degraded ply strengths. Implementation:
`porosity_fe.fe.failure`. Elements whose mean nodal porosity exceeds 0.95,
and void elements, are skipped.

## Degraded strengths

The six ply strengths are scaled by the square root of a stiffness
retention ratio computed from the element's mean porosity. With
$\mathbf C_m$ the pristine matrix stiffness and $\mathbf C^*$ its
Mori-Tanaka degraded counterpart ({doc}`micromechanics`):

$$
r_m = \sqrt{C^*_{11} / C_{m,11}},
\qquad
r_f = \sqrt{\frac{V_f E_f + V_m E_m r_m^2}{V_f E_f + V_m E_m}} .
$$

| Strength | Pristine value | Scaled by |
|---|---|---|
| $X_t$, $X_c$ | $\sigma_{1t}$, $\sigma_{1c}$ | $r_f$ |
| $Y_t$, $Y_c$ | $\sigma_{2t}$, $\sigma_{2c}$ | $r_m$ |
| $S_{12}$ | $\tau_{12}$ | $r_m$ |
| $S_{23}$ | $\tau_\text{ilss}$ | $r_m$ |

Each strength is floored at $10^{-3}$ MPa.

```{note}
The square-root rule is a heuristic: it is loosely motivated by Puck-style
strength–stiffness coupling but is not derived from a published model or
validated against degraded-strength data within PorosityFE. This is the
main reason the FE and empirical strength predictions differ.
```

## Tsai-Wu (default)

With $\sigma_i$ the ply-axis stresses in Voigt order:

$$
FI ={}& F_1 \sigma_1 + F_2 (\sigma_2 + \sigma_3)
 + F_{11} \sigma_1^2 + F_{22} (\sigma_2^2 + \sigma_3^2)
 + F_{44} \sigma_4^2 + F_{55} \sigma_5^2 + F_{66} \sigma_6^2 \\
&+ 2 F_{12} \sigma_1 \sigma_2 + 2 F_{13} \sigma_1 \sigma_3 + 2 F_{23} \sigma_2 \sigma_3 ,
$$

$$
F_1 = \tfrac{1}{X_t} - \tfrac{1}{X_c}, \quad
F_2 = \tfrac{1}{Y_t} - \tfrac{1}{Y_c}, \quad
F_{11} = \tfrac{1}{X_t X_c}, \quad
F_{22} = \tfrac{1}{Y_t Y_c}, \quad
F_{44} = \tfrac{1}{S_{23}^2}, \quad
F_{55} = F_{66} = \tfrac{1}{S_{12}^2} .
$$

The direction-3 terms copy direction 2 (transverse isotropy). The
interaction terms are

$$
F_{12} = F_{13} = F^*_{12} \sqrt{F_{11} F_{22}},
\qquad
F_{23} = -\tfrac12 F_{22},
$$

where $F^*_{12}$ is `MaterialProperties.tsai_wu_F12`. By default it is
$-0.5$, Tsai's recommendation (Tsai & Wu 1971). It must lie in $[-1, 0]$
for the envelope to stay closed. The default is an empirical choice; when
comparing with another code, check its $F_{12}$ convention, and calibrate
$F^*_{12}$ from biaxial tests when they exist. Tsai-Wu does not separate
failure modes, so the per-mode indices are reported as `NaN`.

## Hashin

The four in-plane modes of Hashin (1980), on $(\sigma_1, \sigma_2, \tau_{12})$:

| Mode | Active when | Index |
|---|---|---|
| Fiber tension | $\sigma_1 \ge 0$ | $(\sigma_1/X_t)^2 + (\tau_{12}/S_{12})^2$ |
| Fiber compression | $\sigma_1 < 0$ | $(\sigma_1/X_c)^2$ |
| Matrix tension | $\sigma_2 \ge 0$ | $(\sigma_2/Y_t)^2 + (\tau_{12}/S_{12})^2$ |
| Matrix compression | $\sigma_2 < 0$ | $\left(\frac{\sigma_2}{2S_{23}}\right)^2 + \left[\left(\frac{Y_c}{2S_{23}}\right)^2 - 1\right]\frac{\sigma_2}{Y_c} + \left(\frac{\tau_{12}}{S_{12}}\right)^2$ |

The matrix-compression form matches the 1980 paper; commercial codes ship
variants, so check before comparing. Because these modes ignore
$\sigma_3$, $\tau_{13}$ and $\tau_{23}$, a fifth mode adds the Brewer &
Lagace (1988) delamination-initiation criterion:

$$
FI_\text{delam} = \left(\frac{\langle \sigma_3 \rangle}{Y_t}\right)^2 + \frac{\tau_{13}^2 + \tau_{23}^2}{S_{23}^2},
\qquad \langle \sigma_3 \rangle = \max(\sigma_3, 0).
$$

The reported index is the maximum over the five modes.

## Maximum stress

Each component is compared with its allowable: $\sigma_1$ against $X_t$ or
$X_c$; the worse of $\sigma_2$ and $\sigma_3$ against $Y_t$ or $Y_c$;
$\lvert\tau_{12}\rvert$ and $\lvert\tau_{13}\rvert$ against $S_{12}$; and
$\lvert\tau_{23}\rvert$ against $S_{23}$. The index is the largest ratio.

## First-ply-failure load factor

The analysis is linear, so scaling the applied load by $\lambda$ scales
every stress by $\lambda$. At each Gauss point the failure index becomes a
polynomial in $\lambda$: linear for maximum stress, quadratic for Tsai-Wu
and the Hashin modes. `first_ply_failure_load_factor` is the smallest
positive $\lambda$ at which any checked point reaches an index of 1. The
margin of safety is $\lambda - 1$.
