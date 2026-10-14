# Finite-element model

{class}`~porosity_fe.FESolver` solves a linear static problem on a
structured hexahedral mesh of the specimen, with stiffness degraded
element by element from the local porosity.

## Mesh and elements

{class}`~porosity_fe.CompositeMesh` divides the specimen into
$n_x \times n_y \times n_z$ eight-node hexahedra (default production
resolution $30 \times 10 \times 12$). Each element gets the ply angle of
the ply containing its centroid; a centroid on a ply interface takes the
ply above. The porosity field is sampled at every node. Elements whose
centroid lies inside a discrete void are flagged as void elements. The
total element count is capped at $10^6$.

```{warning}
An element layer that spans several plies takes only one of their angles,
so unless $n_z$ is a multiple of the ply count the FE laminate is not the
requested one. The default $n_z = 12$ does not resolve the 24-ply presets:
for the T800/epoxy QI laminate it models $[90/{-45}/45/0]_3$, which keeps
the angle fractions but not the stacking sequence or symmetry (pristine
$E_x$ 1.6 % below CLT; $n_z = 24$ is within 0.5 %). Bending, interlaminar
stresses and failure indices are more sensitive to the sequence than the
membrane stiffness is.
{meth}`~porosity_fe.CompositeMesh.layup_discrepancies` lists the
differences, and {class}`~porosity_fe.FESolver` logs them as a warning.
```

Elements use trilinear shape functions, enriched by default with
condensed incompatible modes (see below), and $2 \times 2 \times 2$ Gauss
quadrature ({class}`~porosity_fe.Hex8Element` is the reference
implementation; assembly runs on batched arrays). At each Gauss point:

1. $V_p$ is interpolated from the element's nodes, then clamped to
   $[0, 0.99]$.
2. The degraded ply stiffness is computed from $V_p$
   ({doc}`micromechanics`).
3. The stiffness is rotated into the ply angle.

Void elements get a near-zero isotropic stiffness.

## Element formulations

`FESolver(..., formulation=...)` selects the element:

| `formulation` | Element | Use |
|---|---|---|
| `'hex8i'` (default) | trilinear brick plus nine incompatible modes, condensed per element | every mode; required for bending (`ilss`), coarse in-plane meshes and transverse shear stresses |
| `'hex8'` | the plain trilinear brick, full $2 \times 2 \times 2$ Gauss | reproduces results from before `'hex8i'` became the default, bit for bit |

**Shear locking.** A fully integrated trilinear brick cannot bend without
also shearing: in a bent element, the transverse shear strain at the Gauss
points picks up a spurious term proportional to the curvature, so the
element is too stiff in bending and reports too much transverse shear
stress. The error depends on the element length along the span, $\Delta x$,
relative to the laminate thickness $h$, *not* on the number of elements
through the thickness. For a beam of one material it grows as

$$
\frac{E_\text{bend}^\text{hex8}}{E_\text{bend}} - 1
\approx \frac{G_{13}}{E_{11}} \left(\frac{\Delta x}{h}\right)^2
\approx 0.032 \left(\frac{\Delta x}{h}\right)^2
\quad \text{(T800/epoxy)}.
$$

Composites lock about ten times more than steel at the same
$\Delta x / h$ because $E_{11} / G_{13} \approx 31$. Pure bending of a UD
T800 beam ($50 \times 20 \times 2$ mm, uniform moment) gives:

| Mesh | $\Delta x / h$ | $E_\text{bend} / E_{11}$, `hex8` | `hex8i` |
|---|---|---|---|
| $4 \times 2 \times 2$ | 6.25 | 2.265 | 1.002 |
| $8 \times 4 \times 2$ | 3.13 | 1.323 | 1.002 |
| $16 \times 4 \times 2$ | 1.56 | 1.087 | 1.002 |
| $16 \times 4 \times 8$ | 1.56 | 1.084 | 1.001 |
| $64 \times 4 \times 8$ | 0.39 | 1.010 | 1.001 |

The production mesh has $\Delta x / h = 0.38$ on the default 24-ply
laminate, so the bending stiffness error is only about 0.5 %. The
Gauss-point transverse shear stress is affected far more: in the ILSS
three-point bend at $\Delta x / h = 1.56$, `hex8` reports about twice the
beam-theory peak $0.75\,|P| / (b h)$ at the Gauss points near mid-span,
while its element-mean shear is correct.

**Incompatible modes (`'hex8i'`).** Each displacement component is
enriched with the three bubble functions
$P_m = 1 - \xi_m^2$ ($\xi_m \in \{\xi, \eta, \zeta\}$), the Wilson-Taylor
element (equivalent to the enhanced-assumed-strain EAS-9 brick on
parallelepiped elements):

$$
\boldsymbol\varepsilon = \mathbf B \mathbf u_e + \mathbf G \boldsymbol\alpha,
\qquad
\frac{\partial P_m}{\partial \mathbf x} =
\frac{\det \mathbf J_0}{\det \mathbf J}\,
\mathbf J_0^{-1} \frac{\partial P_m}{\partial \boldsymbol\xi},
$$

with $\mathbf J_0$ the Jacobian at the element centre. Taylor's scaling
makes every mode carry zero mean strain over the element, so the element
passes the patch test on distorted meshes. The nine internal amplitudes
$\boldsymbol\alpha$ belong to one element and are condensed out exactly:

$$
\boldsymbol\alpha = \mathbf H \mathbf u_e,\quad
\mathbf H = -\mathbf K_{\alpha\alpha}^{-1} \mathbf K_{\alpha u},
\qquad
\mathbf K_e = \sum_g \mathbf B_\text{eff}^\mathsf T \mathbf C\,
\mathbf B_\text{eff} \det \mathbf J\, w
= \mathbf K_{uu} - \mathbf K_{u\alpha} \mathbf K_{\alpha\alpha}^{-1}
\mathbf K_{\alpha u},
$$

with $\mathbf B_\text{eff} = \mathbf B + \mathbf G \mathbf H$. The batched
assembly stores $\mathbf B_\text{eff}$ in place of $\mathbf B$, so strain
and stress recovery, failure evaluation and export are unchanged, and the
global $\mathbf K$ has the same size and sparsity. The element has exactly
the six rigid-body zero-energy modes.

**Why `'hex8i'` is the default.** It is what `ilss` and any
bending-dominated case need, as do meshes with $\Delta x / h$ not small and
any result that depends on the transverse shear stresses ($\tau_{13}$,
$\tau_{23}$), such as the ILSS failure index and first-ply-failure load
factor; in membrane loading it agrees with `'hex8'`. Changing the
production geometry (T800 QI, uniform $V_p$ of 3 % and 6 %, midplane
clustered and interface penny voids at 3 %) from `'hex8'` to `'hex8i'`:

| Quantity | $30 \times 10 \times 12$ | $30 \times 10 \times 24$ |
|---|---|---|
| compression, tension, shear knockdown | within $10^{-4}$ | within $10^{-4}$ |
| ILSS knockdown | $+0.0006$ to $+0.0034$ | $+0.0007$ to $+0.0019$ |
| ILSS peak $\lvert\tau_{13}\rvert$ | $-45$ to $-47$ % | $-34$ to $-36$ % |
| ILSS first-ply-failure load factor | $+17$ to $+25$ % | $+13$ to $+15$ % |
| compression maximum failure index | $-13$ to $-14$ % | $-8$ to $-12$ % |

The compression maximum failure index sits at the constrained corners.
Assembly costs more, because the condensation runs once per assembly (cached with
$\mathbf K$): about $+0.1$ s at 3,600 elements and $+0.2$ s at 7,200. A
first solve on a new mesh assembles twice (the porous model and the
pristine reference), so it takes a few percent longer (0 to 7 % measured
on the production meshes); repeat solves and memory are unchanged. The stiffness and pristine-reference caches are
keyed on the formulation (each LU factorization belongs to one assembled
$\mathbf K$), and `FieldResults.formulation` and the JSON export
(`solver.formulation`) record it. Pass `formulation='hex8'` to reproduce
results from before the change bit for bit.

## Boundary conditions

| Mode | Control | Constraints |
|---|---|---|
| `compression`, `tension` | displacement | $u_x = 0$ on $x_\text{min}$, $u_x = \varepsilon L_x$ on $x_\text{max}$, $u_y = 0$ on $y_\text{min}$, one corner fixed in $z$ |
| `shear` | displacement | $u_x = \tfrac{\gamma}{2} y$, $u_y = \tfrac{\gamma}{2} x$ on the four $x$ and $y$ faces; one corner fixed in $z$ |
| `ilss` | force | bottom-face nodes along both end edges pinned in $x, y, z$; total `applied_load` along $-z$ shared equally by the top-face nodes at mid-span |
| `thermal` | temperature change `delta_T` | statically determinate 3-2-1 supports on $z_\text{min}$ only, so the laminate is free; see {doc}`thermal` |

Prescribed displacements are imposed exactly by eliminating the
constrained degrees of freedom. With $f$ the free and $c$ the constrained
set, and $\bar{\mathbf u}_c$ the prescribed values, the solver solves

$$
\mathbf K_{ff}\, \mathbf u_f = \mathbf F_f - \mathbf K_{fc}\, \bar{\mathbf u}_c,
\qquad \mathbf u_c = \bar{\mathbf u}_c ,
$$

so the boundary values hold to the last bit. The reported reactions are
$\mathbf R_c = \mathbf K_{cf} \mathbf u_f + \mathbf K_{cc} \bar{\mathbf u}_c - \mathbf F_c$
at the constrained degrees of freedom and zero elsewhere; a load applied at
a supported node goes into that support's reaction. $\mathbf K_{ff}$ keeps
the conditioning of the physical stiffness: on the production mesh its
diagonal ratio is about 24, where the penalty method used before (a spring
of $10^6 \max_i K_{ii}$ on each constrained degree of freedom) gave
$2.4 \times 10^7$ and left a boundary slack of about $10^{-8}$ of the
displacement.

The default solve is a sparse LU factorization of $\mathbf K_{ff}$, cached
and reused until the mesh, material, porosity or set of constrained degrees
of freedom change. Jacobi-preconditioned conjugate gradients
(`solver='cg'`) agree with it to about $10^{-8}$ at the default
`rtol = 1e-9` and need no fill-in, which pays off above roughly 40,000
degrees of freedom. MINRES (`solver='minres'`) is warm-restarted until its
true residual meets `rtol`, but being a residual minimizer it is less
accurate than CG at the same tolerance (about $10^{-6}$ on the production
mesh).

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

## Stress recovery and export

{class}`~porosity_fe.FieldResults` stores stress and strain at the eight
Gauss points of every element. The Gauss points lie at
$\pm 1/\sqrt{3}$ of the element half-width, so they under-read a field
that peaks on a surface, and the element mean under-reads it further. In
pure bending of a UD beam with one element through each half of the
thickness, the element mean of the surface $\sigma_{xx}$ is 0.50 of
$E_{11}\kappa h/2$ and the largest Gauss-point value is 0.80.

{func}`~porosity_fe.extrapolate_to_nodes` (also
{meth}`FieldResults.nodal_stress <porosity_fe.FieldResults.nodal_stress>`
and {meth}`~porosity_fe.FieldResults.nodal_strain`) recovers nodal values
in two steps:

1. **Extrapolation.** The eight Gauss-point values $\mathbf v_g$ of an
   element are treated as samples of a trilinear field. The corner values
   follow from $\mathbf v_n = \mathbf N_g^{-1}\mathbf v_g$, where
   $(\mathbf N_g)_{ij} = N_j(\boldsymbol\xi_i)$ holds the shape functions
   at the Gauss points. The matrix acts in natural coordinates, so it is
   the same for every element, and it recovers any field linear in $x$,
   $y$ and $z$ exactly.
2. **Averaging.** Laminate stresses jump at ply interfaces, so the corner
   values of neighbouring elements are averaged only within the same ply
   (`average='ply'`, the default); void elements form their own groups.
   The per-element corner values keep the jump. A field stored once per
   node takes, at an interface, the average over the ply above, which is
   the convention `CompositeMesh.ply_ids` uses. `average='all'` averages
   over every element at the node and smears the jump. It is only right
   for a field that is continuous across elements.

| UD pure bending, 16 x 4 x $n_z$ | element mean | Gauss-point max | recovered nodal |
|---|---|---|---|
| $n_z = 2$ | 0.504 | 0.80 | 1.023 |
| $n_z = 4$ | 0.756 | 0.91 | 1.018 |
| $n_z = 8$ | 0.883 | 0.96 | 1.016 |

The values are the largest surface $\sigma_{xx}$ over the mid-span
region, as a ratio to the beam-theory value, for a 50 x 20 x 2 mm beam.
The excess over 1 is not discretization error but the beam's width: the
beam is ten times wider than thick, so it bends partly as a plate and its
free edges read highest. Across the width the recovered value ranges
from 1.001 to 1.016 at $n_z = 8$, and a beam as wide as it is thick gives
1.002–1.003. Peaks at
supports, load points and constrained corners are singular. Recovered
values there grow with refinement just as the Gauss-point values do.

To evaluate a failure criterion at the nodes, apply it to the unaveraged
corner stresses (`average='none'`) in the ply frame (`frame='local'`).
Do not extrapolate the failure index itself: the criteria are nonlinear
in stress.

{meth}`FieldResults.to_vtu <porosity_fe.FieldResults.to_vtu>` writes the
mesh, the {meth}`~porosity_fe.FieldResults.to_vtk` cell fields and the
recovered nodal stress and strain (point arrays `sigma_xx_nodal` …
`gamma_xy_nodal`, `von_mises_nodal`) as binary VTK XML. The data are
stored as `Float64` by default (`Float32` optional), as an appended raw
block or as inline base64. With `exploded=True` each cell gets its own
eight points, so ParaView shows the interface jumps without
interpolating across them. {func}`~porosity_fe.write_pvd` groups several
files into a series. The legacy ASCII {meth}`~porosity_fe.FieldResults.to_vtk`
is unchanged.
