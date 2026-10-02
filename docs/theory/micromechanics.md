# Micromechanics: Mori-Tanaka with Eshelby voids

The FE solver turns a local porosity $V_p$ into a degraded orthotropic ply
stiffness in three steps. The matrix is degraded by Mori-Tanaka
homogenization with voids as zero-stiffness Eshelby inclusions. The
degraded matrix is pushed through rule-of-mixtures and Halpin-Tsai
estimates to get a degradation *ratio* for each ply constant. The ratios
multiply the measured ply constants of the
{class}`~porosity_fe.MaterialProperties`. Implementation:
`porosity_fe.homogenization._degraded_composite_stiffness`.

## Mori-Tanaka for voids

The matrix is isotropic, with Young's modulus $E_m$ (`matrix_modulus`) and
Poisson's ratio $\nu_m$ (`matrix_poisson`), giving the stiffness
$\mathbf C_m$. Voids have zero stiffness. For a void volume fraction $V_p$
and an Eshelby tensor $\mathbf S$, the Mori-Tanaka effective stiffness
simplifies to

$$
\mathbf C^* = \mathbf C_m \left[\mathbf I - V_p \left(\mathbf I - (1 - V_p)\,\mathbf S\right)^{-1}\right].
$$

For $V_p < 10^{-12}$ the matrix is returned unchanged. For $V_p > 0.99$
the stiffness is zero. If the bracket is numerically singular, a
pseudo-inverse is used.

### Eshelby tensor

$\mathbf S$ depends only on $\nu_m$ and the ratios of the void radii (see
{doc}`porosity`):

- **Sphere** (radii within 1 %): the closed form
  $S_{1111} = \frac{7 - 5\nu}{15(1-\nu)}$,
  $S_{1122} = \frac{5\nu - 1}{15(1-\nu)}$,
  $S_{1212} = \frac{4 - 5\nu}{15(1-\nu)}$.
- **Spheroid** (two radii equal): the closed forms of Tandon & Weng (1984)
  in a frame with the symmetry axis along $x_1$, using the aspect ratio
  $\alpha = a_\text{axis} / a_\text{eq}$. The function $g$ takes the
  prolate ($\alpha > 1$, $\operatorname{arccosh}$) or oblate ($\alpha < 1$,
  $\arccos$) branch. The tensor is then permuted onto the actual
  symmetry axis.
- **General ellipsoid** (three different radii): Mura's (1987) elliptic
  integrals.

In the engineering-shear Voigt form used throughout, the shear diagonal
of $\mathbf S$ is twice the tensor component ($S_{44} = 2 S_{2323}$, and so
on). With this factor the spherical-void result reproduces the closed-form
Mori-Tanaka shear modulus.

## Degraded matrix constants

For non-spherical voids $\mathbf C^*$ is anisotropic. The code takes its
isotropic (Voigt-average) projection, which for spherical voids returns the
Mori-Tanaka constants exactly:

$$
K^* = \frac{\sum_{i} C^*_{ii} + 2\,(C^*_{12} + C^*_{13} + C^*_{23})}{9},
\qquad
\mu^* = \frac{\sum_{i\le 3} C^*_{ii} - (C^*_{12} + C^*_{13} + C^*_{23}) + 3 \sum_{i\ge 4} C^*_{ii}}{15},
$$

with sums over the normal ($i \le 3$) and shear ($i \ge 4$) diagonals.
Then $\lambda^* = K^* - \tfrac23 \mu^*$,
$E_m^* = \mu^*(3\lambda^* + 2\mu^*)/(\lambda^* + \mu^*)$,
$\nu_m^* = \lambda^*/(2(\lambda^* + \mu^*))$ and $G_m^* = \mu^*$.

## Ply degradation ratios

With fiber volume fraction $V_f$, matrix fraction $V_m = 1 - V_f$, fiber
modulus $E_f$, fiber Poisson's ratio $\nu_f$ (`fiber_poisson`, default 0.2)
and fiber shear modulus $G_f$ (`fiber_shear_modulus`, default
$E_f / (2(1 + \nu_f))$), each ratio compares the estimate with the
degraded matrix to the estimate with the pristine matrix:

| Ply constant | Estimate | Ratio applied to |
|---|---|---|
| $E_{11}$ | Rule of mixtures, $V_f E_f + V_m E_m$ | $E_{11}$ |
| $E_{22}$ | Halpin-Tsai, $\xi = 2$, from $E_f, E_m$ | $E_{22}$, $E_{33}$ |
| $G_{12}$ | Halpin-Tsai, $\xi = 1$, from $G_f, G_m$ | $G_{12}$, $G_{13}$, $G_{23}$ |
| $\nu_{12}$ | Rule of mixtures, $V_f \nu_f + V_m \nu_m$ | $\nu_{12}$, $\nu_{13}$ |
| $\nu_{23}$ | (none) | unchanged |

The Halpin-Tsai estimate for a property $P$ is

$$
P = P_m \frac{1 + \xi \eta V_f}{1 - \eta V_f},
\qquad
\eta = \frac{P_f / P_m - 1}{P_f / P_m + \xi}.
$$

The degraded ply compliance is assembled from the degraded engineering
constants and inverted to give the 6×6 ply stiffness in material axes.
The FE solver then rotates it into each ply's orientation.

Two simplifications to keep in mind:

- Matrix porosity barely moves $E_{11}$ (about 0.1 % at $V_p = 0.05$)
  but strongly degrades the matrix-dominated $E_{22}$ and $G_{12}$. This is
  the intended physics.
- $G_{23}$ uses the same ratio as $G_{12}$. A separate Halpin-Tsai
  $\xi$ for $G_{23}$ would need data to calibrate it.

The CLT functions ({doc}`laminate`) use the same degraded ply stiffness
with spherical voids.
