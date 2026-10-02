# Finite-element model

{class}`~porosity_fe.FESolver` solves a linear static problem on a
structured hexahedral mesh of the specimen, with stiffness degraded
element by element from the local porosity.

## Mesh and elements

{class}`~porosity_fe.CompositeMesh` divides the specimen into
$n_x \times n_y \times n_z$ eight-node hexahedra (default production
resolution $30 \times 10 \times 12$). Each element gets the ply angle of
the ply containing its centroid. The porosity field is sampled at every
node. Elements whose centroid lies inside a discrete void are flagged as
void elements. The total element count is capped at $10^6$.

Elements use trilinear shape functions with $2 \times 2 \times 2$ Gauss
quadrature ({class}`~porosity_fe.Hex8Element` is the reference
implementation; assembly runs on batched arrays). At each Gauss point:

1. $V_p$ is interpolated from the element's nodes, then clamped to
   $[0, 0.99]$.
2. The degraded ply stiffness is computed from $V_p$
   ({doc}`micromechanics`).
3. The stiffness is rotated into the ply angle.

Void elements get a near-zero isotropic stiffness.

## Boundary conditions

| Mode | Control | Constraints |
|---|---|---|
| `compression`, `tension` | displacement | $u_x = 0$ on $x_\text{min}$, $u_x = \varepsilon L_x$ on $x_\text{max}$, $u_y = 0$ on $y_\text{min}$, one corner fixed in $z$ |
| `shear` | displacement | $u_x = \tfrac{\gamma}{2} y$, $u_y = \tfrac{\gamma}{2} x$ on the four $x$ and $y$ faces; one corner fixed in $z$ |
| `ilss` | force | bottom-face nodes along both end edges pinned in $x, y, z$; total `applied_load` along $-z$ shared equally by the top-face nodes at mid-span |

Prescribed displacements are imposed by the penalty method: a stiffness of
$\alpha = 10^6 \max_i K_{ii}$ (`penalty_factor`) is added to each
constrained degree of freedom. The default solve is a sparse LU
factorization, which is cached and reused until the mesh, material or
porosity change. Conjugate-gradient and MINRES are available for large
meshes.

```{note}
The ILSS supports pin all three translations, so the model is a beam
pinned at both ends, not one with a sliding roller as in ASTM D2344, and
the absolute bending stiffness is higher than the test's. The knockdown
below compares two solves with the same supports.
```

## Stiffness knockdown

The FE `knockdown` compares the structural stiffness of the porous model
with a pristine reference solve of the same mesh, loading and supports,
with zero porosity and no void elements:

$$
KD_\text{FE} = \frac{\mathcal S_\text{porous}}{\mathcal S_\text{pristine}},
\qquad
\mathcal S =
\begin{cases}
\mathbf u^\mathsf T \mathbf K \mathbf u / \varepsilon^2 & \text{compression, tension, shear} \\
P^2 / \mathbf u^\mathsf T \mathbf K \mathbf u & \text{ILSS}.
\end{cases}
$$

For the displacement-controlled modes $\mathcal S$ is the effective
modulus times the volume, so $KD_\text{FE}$ is the ratio of $E_x$ (or
$G_{xy}$ for shear). For ILSS it is the ratio of beam stiffnesses
(inverse compliances). The pristine solve is cached by mesh geometry, so a
porosity sweep on one mesh pays for it once. A ratio above 1 is not
clipped; a warning is logged. `effective_modulus` reports
$\mathbf u^\mathsf T \mathbf K \mathbf u / (\varepsilon^2 V)$ for the
displacement-controlled modes.

$KD_\text{FE}$ is a *stiffness* knockdown. Strength enters through the
failure criteria on {doc}`failure`.
