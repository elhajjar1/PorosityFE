# Porosity field

{class}`~porosity_fe.PorosityField` gives the void volume fraction
$V_p(x, y, z)$ at any point of the laminate. It is the sum of a smooth
through-thickness profile and, optionally, explicit ellipsoidal voids.

## Through-thickness profiles

The laminate has thickness $L_z = n_\text{plies}\, t_\text{ply}$. Each
profile $f(z)$ is rescaled so that its mean over $[0, L_z]$ equals the
specimen-average porosity $\bar V_p$ (the `void_volume_fraction`
argument):

$$
V_p(z) = \bar V_p \, \frac{f(z)}{\langle f \rangle},
\qquad
\langle f \rangle = \frac{1}{L_z}\int_0^{L_z} f(z)\,dz .
$$

The mean $\langle f \rangle$ is evaluated on 1000 points.

`uniform`
: $f(z) = 1$.

`clustered`
: A Gaussian bump $f(z) = \exp\!\left[-\tfrac12 \left(\tfrac{z - z_0}{\sigma}\right)^2\right]$
  with $\sigma = L_z / 6$. The centre $z_0$ is $0.5 L_z$ (`midplane`),
  $0$ (`surface`) or $0.25 L_z$ (`quarter`).

`interface`
: A Gaussian at each of the $n_\text{plies} - 1$ ply interfaces
  $z_k = k\, t_\text{ply}$, each with $\sigma = 0.35\, t_\text{ply}$.

Because every profile is renormalized to the same mean, the empirical
solver (which uses only $\bar V_p$) gives the same answer for all three.
The FE solver samples $V_p$ at every node and does see the difference.

## Void shapes

The `void_shape` sets the ellipsoid used for the Eshelby tensor in the
micromechanics (see {doc}`micromechanics`). Only the ratios of the radii
matter. The named shapes in {data}`~porosity_fe.VOID_SHAPES` are:

| Name | Radii $(a_1, a_2, a_3)$ | Meaning |
|---|---|---|
| `spherical` | $(1, 1, 1)$ | Equiaxed voids |
| `cylindrical` | $(3, 1, 1)$ | Voids elongated along the fibers ($x$) |
| `penny` | $(3, 3, 0.3)$ | Flat voids in the ply plane, thin through the thickness |

An explicit `(a1, a2, a3)` tuple is also accepted.

## Discrete voids

A {class}`~porosity_fe.VoidGeometry` is an explicit ellipsoid with a
centre, radii and an in-plane orientation. Points inside a discrete void
have $V_p = 1$; elsewhere the smooth profile applies. In the FE mesh
an element whose centroid lies inside a void becomes a void element with
near-zero stiffness and is skipped by the failure check.

### Stress concentration factor

{meth}`VoidGeometry.stress_concentration_factor
<porosity_fe.VoidGeometry.stress_concentration_factor>` treats the void as
a traction-free ellipsoidal cavity in an infinite isotropic matrix with
Poisson's ratio $\nu_m$. The exact Eshelby solution gives the stress on the
cavity surface; the SCF for each loading mode is the peak of the loaded
stress component over the surface divided by its remote value:
$\sigma_{xx}$ for tension and compression, $\sigma_{yy}$ for transverse
tension, $\tau_{xy}$ for shear and $\tau_{xz}$ for ILSS. Two checks on
this solution:

- A sphere gives Goodier's result, $(27 - 15\nu)/(2(7 - 5\nu))$ in
  tension and $15(1 - \nu)/(7 - 5\nu)$ in shear.
- A long elliptic cylinder recovers Inglis' $1 + 2a/b$.

The anisotropy of the surrounding composite is not modelled.

The empirical solver uses the SCF only in its per-node knockdown field. A
node at distance $d$ from the void surface (measured along the ray from the
void centre; $d = 0$ on and inside the surface) has its knockdown
multiplied by

$$
1 - e^{-d / a_\text{max}} \left(1 - \frac{1}{\text{SCF}}\right),
$$

where $a_\text{max}$ is the largest radius. The specimen failure load from
{meth}`~porosity_fe.EmpiricalSolver.get_failure_load` does not include this
factor.
