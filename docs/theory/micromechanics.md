# Micromechanics: Mori-Tanaka with Eshelby voids

The FE solver turns a local porosity $V_p$ into a degraded orthotropic ply
stiffness in three steps. The matrix is degraded by Mori-Tanaka
homogenization with voids as zero-stiffness Eshelby inclusions. The
degraded matrix is pushed through rule-of-mixtures and Halpin-Tsai
estimates to get a degradation *ratio* for each ply constant. The ratios
multiply the measured ply constants of the
{class}`~porosity_fe.MaterialProperties`. Implementation:
`porosity_fe.homogenization._degraded_composite_stiffness`.

By default (`FESolver(..., porosity_degradation='isotropic')`) the
Mori-Tanaka tensor is projected onto an isotropic one before the ratios are
formed, so a void's shape sets *how much* the matrix is degraded but not *in
which direction*. The opt-in `porosity_degradation='directional'` keeps the
transversely isotropic tensor and degrades each ply constant from its own
direction; see [Directional degradation](#directional-degradation-opt-in).

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

For non-spherical voids $\mathbf C^*$ is anisotropic. By default the code
takes its isotropic (Voigt-average) projection, which for spherical voids
returns the Mori-Tanaka constants exactly:

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
with spherical voids, always with the isotropic projection, as does the
empirical solver.

## Directional degradation (opt-in)

The projection above averages away what a flat void does. A penny void
lying in the ply plane (radii $(3, 3, 0.3)$) barely softens the matrix in
the plane but strongly through the thickness, yet after the projection
$E_{22}$ and $E_{33}$ drop by the same factor. With
`FESolver(..., porosity_degradation='directional')` the Mori-Tanaka tensor
is kept as it is, transversely isotropic about the void axis, and each ply
constant is degraded from the matrix constant of its own direction. The
void radii are read in each ply's material axes: $a_1$ along the fibres,
$a_3$ through the thickness, so a penny lies in the ply plane and a
cylindrical void runs along the fibres.

From the normal block of the porous-matrix compliance,
$\mathbf S^* = (\mathbf C^*_{[1:3,1:3]})^{-1}$:

$$
E^*_i = \frac{1}{S^*_{ii}}, \qquad
\nu^*_{12} = -\frac{S^*_{12}}{S^*_{11}}, \qquad
\nu^*_{13} = -\frac{S^*_{13}}{S^*_{11}}, \qquad
G^*_{23}, G^*_{13}, G^*_{12} = C^*_{44}, C^*_{55}, C^*_{66} .
$$

| Ply constant | Estimate | Matrix constant |
|---|---|---|
| $E_{11}$ | Rule of mixtures | $E^*_1$ |
| $E_{22}$, $E_{33}$ | Halpin-Tsai, $\xi = 2$ | $E^*_2$, $E^*_3$ |
| $G_{12}$, $G_{13}$, $G_{23}$ | Halpin-Tsai, $\xi = 1$ | $G^*_{12}$, $G^*_{13}$, $G^*_{23}$ |
| $\nu_{12}$, $\nu_{13}$ | Rule of mixtures | $\nu^*_{12}$, $\nu^*_{13}$ |
| $\nu_{23}$ | (none) | unchanged |

The ratios are formed and applied exactly as in the table above, against the
same pristine estimates. The constants come from different directions of one
tensor, so the degraded ply compliance is checked with a Cholesky
factorization before it is inverted, and a compliance that is not positive
definite raises `ValueError`. With $\nu_{23}$ held at its pristine value
the condition is $\nu_{23}^2 < E_{22}/E_{33}$: it is met by every preset
for the three preset void shapes and four others (the penny normal to the
fibres, the cylinder across them, a $(1, 1, 0.1)$ penny and a
$(2, 1, 0.5)$ ellipsoid) from $V_p = 10^{-4}$ to the 0.99 clamp. Only a
flat void normal to the transverse in-plane axis, $(3, 0.3, 3)$, a
transverse-crack shape, breaks it, from $V_p \approx 0.4$.

For spherical voids $\mathbf C^*$ is isotropic, and the two mappings agree
to round-off: within $10^{-12}$ in the ply stiffness and, on the production
meshes, within $5 \times 10^{-13}$ in the knockdowns, failure indices and
load factors and $2 \times 10^{-11}$ of their maximum in the fields.
T800/epoxy at $V_p = 0.03$, retention of each constant:

| Void | Mapping | $E_{11}$ | $E_{22}$ | $E_{33}$ | $G_{12}$ | $G_{13}$ | $G_{23}$ |
|---|---|---|---|---|---|---|---|
| penny $(3, 3, 0.3)$ | isotropic | 0.9990 | 0.885 | 0.885 | 0.899 | 0.899 | 0.899 |
| penny $(3, 3, 0.3)$ | directional | 0.9997 | 0.968 | 0.741 | 0.968 | 0.847 | 0.847 |
| cylinder $(3, 1, 1)$ | isotropic | 0.9995 | 0.942 | 0.942 | 0.945 | 0.945 | 0.945 |
| cylinder $(3, 1, 1)$ | directional | 0.9997 | 0.932 | 0.932 | 0.949 | 0.949 | 0.935 |
| sphere | either | 0.9995 | 0.946 | 0.946 | 0.948 | 0.948 | 0.948 |

The penny loses more than twice as much through the thickness as the
projection says ($E_{33}$ 26 % against 11.5 %) and less than a third as
much in the plane ($E_{22}$ 3.2 %). The cylinder along the fibres leaves the
$2$-$3$ plane isotropic and softens $G_{23}$, the shear in that plane, more
than $G_{12}$ and $G_{13}$. On the production meshes with uniform penny
porosity at 3 % the short-beam (ILSS) knockdown drops from 0.930 to 0.876
and the compression knockdown rises from 0.985 to 0.995, since flat voids
parallel to the load hardly carry it; the solve costs the same. The
interlaminar strengths follow the same tensor ({doc}`failure`).

**Penny-crack limit.** As the aspect ratio $\gamma = c/a$ of aligned
oblate voids goes to zero at a fixed crack density $\rho = N a^3$ (so
$V_p = \tfrac43 \pi \rho \gamma$), the Mori-Tanaka compliance tends to
that of non-interacting parallel penny cracks (Kachanov 1992, *Appl. Mech.
Rev.* 45(8); Budiansky & O'Connell 1976):

$$
\Delta S_{33} = \frac{16 (1 - \nu^2)\,\rho}{3E}, \qquad
\Delta S_{44} = \Delta S_{55} = \frac{32 (1 - \nu^2)\,\rho}{3 (2 - \nu) E},
$$

with no in-plane change (engineering shear; $E$, $\nu$ of the matrix). For
the T800 matrix at $\rho = 0.05$ the computed over the closed-form
$\Delta S_{33}$ is 1.106, 1.010, 1.001, 1.000 at $\gamma = 10^{-1}$,
$10^{-2}$, $10^{-3}$, $10^{-4}$, and $\Delta S_{55}$ 1.239, 1.023, 1.002,
1.000; the error is first order in $\gamma$, and the in-plane change is of
order $V_p$ (`tests/test_homogenization.py`). These are at the exact void
fraction. The Mori-Tanaka cache rounds $V_p$ to $10^{-6}$, so through it the
same ratios read 1.010, 0.999, 1.003 at $\gamma = 10^{-2}$ to $10^{-4}$,
where $V_p$ is $2 \times 10^{-3}$ to $2 \times 10^{-5}$.

The directional mapping is not calibrated against measured $E_{33}$,
$G_{13}$ or $G_{23}$ of porous laminates, and it is not exposed in the app
or the CLI.

## Verification against a periodic unit cell

Mori-Tanaka is a mean-field estimate: it knows the void volume fraction and
shape, but not how the voids are arranged. To check it, an internal
verification tool (`porosity_fe.fe._rve`, not public API) computes the
stiffness of a periodic array of identical voids with the package's own
elements, and compares it with `_mt_effective_stiffness` for the same
matrix, void shape and void fraction.

### Method

**Cell.** A cube of isotropic matrix (the T800 preset: $E_m = 3500$ MPa,
$\nu_m = 0.35$) with one centred, axis-aligned ellipsoidal void, meshed
with $n^3$ elements. The void is the set of elements whose centroid lies
inside the ellipsoid ({class}`~porosity_fe.VoidGeometry` and the void
elements of {class}`~porosity_fe.CompositeMesh`), with the 1 MPa stiffness
of the explicit-void path; every other element is pristine matrix, with no
smeared porosity. The void is therefore a voxel staircase of the
ellipsoid, and Mori-Tanaka is evaluated at the *voxel* void fraction: on a
$12^3$ cell it differs from the nominal ellipsoid volume by up to 8 % for a
sphere and 20 % for a thin penny. The 1 MPa void raises the cell stiffness
over that of an empty void by at most 0.02 % (sphere) and 0.3 % (crowded
penny).

**Periodic boundary conditions.** The displacement is
$\mathbf u = \bar{\boldsymbol\varepsilon}\,\mathbf x + \mathbf w$ with a
uniform macroscopic strain $\bar{\boldsymbol\varepsilon}$ and a periodic
fluctuation $\mathbf w$. Each node on a maximum face is tied to its image
on the opposite face, $\mathbf u(\mathbf x + \mathbf L) = \mathbf u(\mathbf x)
+ \bar{\boldsymbol\varepsilon}\,\mathbf L$, through the solver's constraint
reduction $\mathbf u = \mathbf T \mathbf u_r + \mathbf g$ ({doc}`fe`, edge
and corner nodes tied in chains that end at the origin corner). That
corner is pinned at $\bar{\boldsymbol\varepsilon}\,\mathbf x$ to remove the
rigid translations; nothing else is constrained, so it carries no force.
The six unit macro-strains (engineering shear, Voigt order
$[11, 22, 33, 23, 13, 12]$) give the six columns of the cell stiffness as
volume-averaged stresses. The tests (`tests/test_rve.py`) check that a
homogeneous cell returns $\mathbf C_m$ to $10^{-14}$ with $\mathbf u =
\bar{\boldsymbol\varepsilon}\,\mathbf x$ at every node, and that a cell
with a void has a mean strain of exactly $\bar{\boldsymbol\varepsilon}$, a
symmetric stiffness equal to $\mathbf u_i^\mathsf{T} \mathbf K \mathbf u_j /
V$ (Hill-Mandel) and no force on the pinned corner, for both element
formulations.

**Cost.** A $24^3$ cell has 41,469 unknowns after the periodic reduction
and takes about 35 s per element formulation. The ties wrap the mesh into
a torus, on which SuperLU's minimum-degree ordering fills poorly; a
geometric nested-dissection ordering cuts the factorization time four- to
sixfold.

### Results

Cell stiffness over Mori-Tanaka, $24^3$ cell, `'hex8i'`. $K$ and $\mu$ are
the isotropic (Voigt-average) projections that the FE path takes from the
Mori-Tanaka tensor (see above); the penny is an oblate spheroid of aspect
5 (radii $1 : 1 : 0.2$), its axis along $z$.

| Void | $V_p$ (voxel) | MT $C_{11}/C_{m,11}$ | MT $C_{33}/C_{m,11}$ | $C_{11}$ | $C_{33}$ | $C_{12}$ | $C_{44}$ | $C_{66}$ | $K$ | $\mu$ |
|---|---|---|---|---|---|---|---|---|---|---|
| sphere | 0.020 | 0.945 | 0.945 | 0.999 | 0.999 | 0.997 | 0.999 | 0.999 | 0.998 | 0.999 |
| sphere | 0.049 | 0.875 | 0.875 | 0.998 | 0.998 | 0.992 | 0.993 | 0.993 | 0.995 | 0.998 |
| sphere | 0.100 | 0.765 | 0.765 | 1.000 | 1.000 | 0.983 | 0.980 | 0.980 | 0.991 | 0.995 |
| penny | 0.018 | 0.943 | 0.854 | 0.997 | 1.007 | 0.995 | 0.992 | 0.999 | 0.999 | 0.998 |
| penny | 0.051 | 0.859 | 0.666 | 0.991 | 0.994 | 0.981 | 0.968 | 0.996 | 0.986 | 0.988 |
| penny | 0.082 | 0.797 | 0.544 | 0.979 | 0.929 | 0.950 | 0.896 | 0.998 | 0.945 | 0.963 |
| penny | 0.098 | 0.769 | 0.495 | 0.961 | 0.811 | 0.907 | 0.792 | 1.002 | 0.880 | 0.925 |

`'hex8'` reads up to 0.5 % stiffer than `'hex8i'` (up to 1.1 % in $C_{33}$
and $C_{13}$ of the crowded pennies).

Refinement of the cell at $V_p \approx 0.05$ (FE / MT, `'hex8i'`, with
`'hex8'` in parentheses):

| Void | Elements | $V_p$ (voxel) | $C_{11}$ | $C_{33}$ | $C_{12}$ | $C_{44}$ |
|---|---|---|---|---|---|---|
| sphere | $12^3$ | 0.0509 | 0.998 (1.000) | 0.998 (1.000) | 0.991 (0.994) | 0.995 (0.997) |
| sphere | $16^3$ | 0.0508 | 0.996 (0.998) | 0.996 (0.998) | 0.988 (0.991) | 0.992 (0.994) |
| sphere | $24^3$ | 0.0486 | 0.998 (0.999) | 0.998 (0.999) | 0.992 (0.993) | 0.993 (0.994) |
| sphere | $32^3$ | 0.0500 | 0.999 (0.999) | 0.999 (0.999) | 0.992 (0.994) | 0.993 (0.993) |
| penny | $12^3$ | 0.0602 | 1.008 (1.011) | 1.135 (1.143) | 1.018 (1.023) | 1.006 (1.010) |
| penny | $16^3$ | 0.0547 | 0.990 (0.993) | 1.009 (1.016) | 0.978 (0.983) | 0.956 (0.960) |
| penny | $24^3$ | 0.0509 | 0.991 (0.993) | 0.994 (0.999) | 0.981 (0.984) | 0.968 (0.970) |
| penny | $24 \times 24 \times 48$ | 0.0498 | 0.993 (0.994) | 0.982 (0.986) | 0.983 (0.985) | 0.969 (0.971) |

The voxel void fraction does not converge monotonically, and neither do the
ratios. For the sphere they settle within 0.1 % from $24^3$ on (a $32^3$
cell takes about three minutes). The thin penny needs more elements through
its thickness: on a $12^3$ cell it is two elements thick and the cell is
13 % too stiff out of plane, and on the $24^3$ cell $C_{33}$ is still about
1 % high, so the out-of-plane gaps of the pennies below are, if anything,
slightly underestimated.

### Conclusions

- **Spherical voids** (the default shape of the FE path): Mori-Tanaka is
  trustworthy over the calibrated range and beyond. The normal
  stiffnesses agree within 0.2 % up to $V_p = 0.10$, and the bulk and shear
  moduli within 1 %. At $V_p = 0.05$ the $C_{11}$ knockdown is 12.5 % from
  Mori-Tanaka and 12.7 % from the cell. The single shear terms differ by
  up to 2 % because a simple cubic array is not isotropic: at
  $V_p = 0.10$, $C_{44}$ is 2.0 % below the Mori-Tanaka shear modulus and
  $(C_{11} - C_{12})/2$ 1.7 % above it, so the isotropic estimate lies
  between the array's two shear moduli. The cell's bulk modulus stays just
  below Mori-Tanaka's (0.5 % at 0.05), consistent with Mori-Tanaka being
  the Hashin-Shtrikman upper bound for spherical voids.
- **Dilute oblate voids**: for the aspect-5 penny the two agree within
  1 % at $V_p = 0.02$ and within 3.5 % at 0.05 (worst $C_{44}$).
- **Crowded oblate voids**: Mori-Tanaka has no notion of void spacing, and
  once neighbouring pennies come close it overestimates the out-of-plane
  stiffness. At $V_p = 0.08$ (ligament between neighbours 8.6 % of the
  cell, two elements) the cell is 7 % softer in $C_{33}$ (9 % on the
  $24 \times 24 \times 48$ cell) and 10 % in $C_{44}$; at $V_p = 0.10$ the
  pennies span 98.5 % of the cell width, the
  ligament is thinner than one element at any affordable resolution, and
  the voxel voids touch their neighbours on the $24^3$ cell (not on the
  $16^3$ one, where the gap is 13 % in $C_{33}$ instead of 19 %). The
  result then depends on how the cell is designed, not only on the
  micromechanics: Mori-Tanaka is not a reliable estimate for oblate voids
  at a local $V_p$ near 0.1, which clustered porosity reaches at a modest
  mean (the `'clustered'` profile peaks at 2.4 times the mean at the
  mid-plane and 4.8 times at the surface). The isotropic projection the FE
  path uses by default softens the error but keeps it: 5.5 % in $K$ and
  3.7 % in $\mu$ at 0.08, 12 % and 7.5 % at 0.10. With
  `porosity_degradation='directional'` nothing softens it: the out-of-plane
  constants come straight from $C^*_{33}$ and $C^*_{44}$, which are 7 to
  10 % too stiff at a local $V_p$ of 0.08 and about 20 % at 0.10, so the
  directional mapping *underestimates* the out-of-plane loss of crowded
  pennies. The in-plane shear $C_{66}$ of the pennies stays within 0.5 % in
  every case.
