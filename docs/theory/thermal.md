# Thermal and cure residual stress

`FESolver.solve(loading='thermal', delta_T=...)` computes the residual
stress a free-standing laminate develops under a uniform temperature
change from its stress-free state, typically the cool-down from cure.
`delta_T=` on a mechanical loading adds that residual stress to the
mechanical solve. Both need the lamina CTEs
({attr}`~porosity_fe.MaterialProperties.alpha_1`,
{attr}`~porosity_fe.MaterialProperties.alpha_2`, optionally
{attr}`~porosity_fe.MaterialProperties.alpha_3`) and an explicit
`delta_T`. The library never infers `delta_T`, not even from
{attr}`~porosity_fe.MaterialProperties.T_stress_free`, in the same way
that it never guesses $V_p$.

```python
import dataclasses
from porosity_fe import MATERIALS, CompositeMesh, FESolver, PorosityField

mat = MATERIALS['AS4_3501_6_epoxy']              # carries CTEs
pf = PorosityField(mat, 0.02)
mesh = CompositeMesh(pf, mat, nx=30, ny=10, nz=24)   # nz = n_plies
solver = FESolver(mesh, mat, pf)

residual = solver.solve('thermal', delta_T=-150.0)           # cool-down, K
combined = solver.solve('tension', applied_strain=0.005, delta_T=-150.0)
```

## Constitutive law and CTE

With the thermal strain $\boldsymbol\varepsilon_\text{th} = \boldsymbol\alpha\,\Delta T$
the stress is

$$
\boldsymbol\sigma = \mathbf C \left(\boldsymbol\varepsilon - \boldsymbol\alpha\,\Delta T\right),
\qquad
\boldsymbol\alpha_\text{ply} = [\alpha_1, \alpha_2, \alpha_3, 0, 0, 0]^\mathsf T ,
$$

in Voigt order with engineering shear; $\Delta T$ is a temperature
*difference* in K ($T_\text{service} - T_\text{stress-free}$, negative for a
cool-down), and $\alpha_3 = \alpha_2$ unless set. The ply-local strain is
$\mathbf T_\varepsilon(\theta)$ times the laminate strain, so the CTE in
laminate axes is

$$
\boldsymbol\alpha = \mathbf T_\varepsilon(\theta)^{-1} \boldsymbol\alpha_\text{ply}:
\quad
\alpha_x = \alpha_1 c^2 + \alpha_2 s^2,\;
\alpha_y = \alpha_1 s^2 + \alpha_2 c^2,\;
\alpha_z = \alpha_3,\;
\alpha_{xy} = 2(\alpha_1 - \alpha_2)\, s c
$$

({func}`porosity_fe.fe.batch.global_cte`). $\mathbf C$ is the same
porosity-degraded, ply-rotated stiffness the mechanical solve uses.

**Porosity changes the residual stress through $\mathbf C$ only.** The CTE
is held at its pristine value. By Levin's theorem an empty void does not
change the free thermal expansion of the matrix around it (the uniform
field $\boldsymbol\varepsilon = \alpha_m \Delta T\,\mathbf I$ is compatible
and stress-free with the voids present), and at the ply level the
Schapery estimates move $\alpha_1$ by less than 0.1 ppm/K and $\alpha_2$
by less than 2.5 % at $V_p = 0.05$, well inside the scatter of measured
lamina CTEs. Because porosity softens $E_{22}$ and $G_{12}$ far more than
$E_{11}$, the residual stress falls as $V_p$ rises (for the cross-ply
below, $47.6 \to 43.8$ MPa from $V_p = 0$ to $0.05$). Through-thickness
porosity gradients also make a symmetric layup elastically unsymmetric,
so a free laminate with clustered porosity warps slightly.

## Load vector and element formulations

With $\boldsymbol\beta = \mathbf C \boldsymbol\alpha$ (MPa/K, zero in
explicit void elements) at every Gauss point, the thermal load is

$$
\mathbf f_e = \sum_g \mathbf B_g^\mathsf T \boldsymbol\beta_g\, \Delta T\,
\det \mathbf J_g\, w_g ,
$$

scattered into the global vector like the stiffness
({meth}`ElementBatch.thermal_loads <porosity_fe.fe.batch.ElementBatch.thermal_loads>`).

For `formulation='hex8i'` the nine incompatible modes $\mathbf a$ of each
element are loaded too, by
$\mathbf f_a = \sum_g \mathbf G_g^\mathsf T \boldsymbol\beta_g \Delta T \det\mathbf J_g w_g$.
Condensing them out of
$\begin{bmatrix}\mathbf K_{uu} & \mathbf K_{ua}\\ \mathbf K_{au} & \mathbf K_{aa}\end{bmatrix}
\begin{bmatrix}\mathbf u\\ \mathbf a\end{bmatrix} =
\begin{bmatrix}\mathbf f_u\\ \mathbf f_a\end{bmatrix}$ gives

$$
\mathbf a = \mathbf H \mathbf u + \mathbf K_{aa}^{-1} \mathbf f_a,
\qquad
\mathbf f_\text{eff} = \mathbf f_u + \mathbf H^\mathsf T \mathbf f_a ,
\qquad
\mathbf H = -\mathbf K_{aa}^{-1}\mathbf K_{au} .
$$

$\mathbf f_\text{eff}$ is exactly the sum above evaluated with the
condensed operator $\mathbf B_\text{eff} = \mathbf B + \mathbf G \mathbf H$
the hex8i batch already stores, and the strain is
$\mathbf B_\text{eff}\mathbf u + \mathbf G \mathbf K_{aa}^{-1}\mathbf f_a$.
The second term vanishes when $\boldsymbol\beta$ is uniform over an element
(the Taylor-corrected $\mathbf G$ integrates to zero) but not with graded
porosity, where leaving it out changes the stress by several percent
({meth}`ElementBatch.incompatible_mode_strains <porosity_fe.fe.batch.ElementBatch.incompatible_mode_strains>`).
A test solves the same problem with the internal modes kept as unknowns
and matches it to $10^{-13}$.

## Free-standing supports and recovery

`loading='thermal'` uses a statically determinate 3-2-1 set on the
$z_\text{min}$ face ({meth}`~porosity_fe.BoundaryHandler.free_bcs`), which
removes the six rigid-body modes and nothing else:

| corner | constrained |
|---|---|
| $(x_\text{min}, y_\text{min})$ | $u_x, u_y, u_z$ |
| $(x_\text{max}, y_\text{min})$ | $u_y, u_z$ |
| $(x_\text{min}, y_\text{max})$ | $u_z$ |

The thermal load is self-equilibrated, so the reactions are zero (about
$10^{-9}$ N against nodal thermal forces of order 100 N in the tests) and
the laminate expands, contracts and warps freely; a reaction above
$10^{-6}$ of the largest nodal force is logged as a warning. The stress is
recovered as $\boldsymbol\sigma = \mathbf C(\boldsymbol\varepsilon - \boldsymbol\alpha\Delta T)$,
and `strain_global` / `strain_local` hold the *total* strain, mechanical
plus free thermal, which is what a gauge bonded in the stress-free state
reads. The solve is linear in $\Delta T$, so the solver caches the
unit-$\Delta T$ solution and rescales it.

## Combined thermal and mechanical loading

Cure residual stress forms in the free laminate, before it is gripped.
With `delta_T=` on `'compression'`, `'tension'`, `'shear'` or `'ilss'` the
solver therefore superposes two solves that both start from the
stress-free, free-standing configuration:

$$
\boldsymbol\sigma(\lambda) = \boldsymbol\sigma_\text{th} + \lambda\, \boldsymbol\sigma_\text{m},
$$

$\boldsymbol\sigma_\text{th}$ from the free-standing thermal solve and
$\boldsymbol\sigma_\text{m}$ from the mechanical solve with its own
boundary conditions. Gripping a laminate in its cured shape adds no stress.
One solve with both the thermal load and the grip constraints would model
a laminate heated while clamped, a different load case. Stress, strain and
displacement arrays hold the totals and
{attr}`~porosity_fe.FieldResults.residual_stress_local` the thermal part.
Knockdown, effective modulus and reactions come from the mechanical solve
alone: a residual stress does not change a linear stiffness, so they are
identical to the same solve without `delta_T`.

## First-ply failure with a residual stress

Failure indices are evaluated on the total stress. The first-ply-failure
factor $\lambda$ multiplies only the mechanical part, with
$\boldsymbol\sigma_\text{th}$ held fixed
({func}`~porosity_fe.fe.failure.first_ply_failure_load_factor` with
`prestress_local=`); it is 0 when the residual stress alone reaches the
criterion. Scaling the total stress instead would scale the residual stress
with the load and, for the cross-ply below in tension, understate the
first-ply-failure strain by a factor of two.

- **Tsai-Wu.** With $FI(\boldsymbol\sigma) = \mathbf F\cdot\boldsymbol\sigma + \boldsymbol\sigma^\mathsf T \mathbb F \boldsymbol\sigma$,

  $$
  A\lambda^2 + B\lambda = 1 - FI_\text{th},\quad
  A = \boldsymbol\sigma_\text{m}^\mathsf T \mathbb F \boldsymbol\sigma_\text{m},\quad
  B = \mathbf F\cdot\boldsymbol\sigma_\text{m} + 2\boldsymbol\sigma_\text{th}^\mathsf T \mathbb F \boldsymbol\sigma_\text{m},
  $$

  solved by the rationalized root $\lambda = 2c/(B + \sqrt{B^2 + 4Ac})$,
  $c = 1 - FI_\text{th}$.
- **Max-stress.** Each component $\sigma_{\text{th},i} + \lambda\sigma_{\text{m},i}$
  meets the allowable on the side it moves towards (linear with an offset).
- **Hashin.** A pre-stress can change a stress sign as $\lambda$ grows, so
  the sign-switching modes are solved branch by branch: fiber tension and
  compression on the intervals of $\lambda$ where $\sigma_{11} \ge 0$ or
  $< 0$, matrix tension and compression likewise in $\sigma_{22}$, and
  delamination on both sides of $\sigma_{33} = 0$. In each branch the
  first $\lambda$ at which the (convex) mode reaches 1 is either the start
  of the interval, when a mode switches on already at or above 1 (fiber
  tension picks up the shear term $(\tau_{12}/S_{12})^2$ the moment
  $\sigma_{11}$ turns tensile), or the root of its quadratic inside the
  interval.

The closed forms are checked against a scan-and-bisection of
$FI(\boldsymbol\sigma_\text{th} + \lambda\boldsymbol\sigma_\text{m})$ on random
stress pairs and on the FE fields, to $10^{-6}$ relative.

For `loading='thermal'` the factor multiplies $\Delta T$ itself
(`load_factor_basis='delta_T'`): the critical temperature change is
`first_ply_failure_load_factor * delta_T`.

## Results that differ from the mechanical modes

- `knockdown` is `nan` for `loading='thermal'`, which has no structural
  stiffness measure, and `effective_modulus` is `None`. The JSON export
  writes `knockdown_factor: null` and the VTK/VTU files leave out the
  `knockdown` cell field.
- `load_factor_basis` says what the factor multiplies: `'mechanical'`,
  `'delta_T'` or `'mechanical_with_residual'`.
- The JSON export gains a `thermal` block (`delta_T_K`, the CTEs, residual
  and interior failure indices, residual-stress statistics) and records
  `delta_T_K` in its provenance; VTK and VTU files gain the cell fields
  `residual_sigma_11_local`, `residual_sigma_22_local` and
  `residual_tau_12_local`.

## Free edges: interior and domain maximum

At a free edge the laminate develops the interlaminar boundary-layer
stresses ($\sigma_{zz}$, $\tau_{xz}$, $\tau_{yz}$) that are singular in
elasticity, so their peak, and with it the domain-maximum failure index,
depends on the mesh. Every solve therefore also reports
`interior_max_failure_index` and `interior_first_ply_failure_load_factor`
over the elements whose centroids lie at least `interior_margin` from
every lateral face (default: the larger of the laminate thickness and two
element widths). For the cross-ply below (hex8) the interior Tsai-Wu
index is 0.55 against a domain maximum of 0.59, and the interior
$|\sigma_{zz}|$ is 0.005 MPa against 4.4 MPa at the edge.

## Layup resolution

Residual stresses come from the ply-to-ply mismatch, so an element layup
that merges plies gives the wrong laminate: on the production mesh
($n_z = 12$) a 24-ply QI preset becomes unsymmetric and unbalanced, warps
by 0.2 mm on cool-down, and its interior $\sigma_{22}$ ranges from 31 to
46 MPa against 47.6 MPa with every ply resolved. With `delta_T` the solve
therefore raises when
{meth}`~porosity_fe.CompositeMesh.layup_discrepancies` is not empty (use
$n_z = k\, n_\text{plies}$); `allow_unresolved_layup=True` runs the element
layup anyway, with a warning. Mechanical solves keep the existing warning.

## Verification

T800/epoxy stiffness with representative CTEs
$\alpha_1 = -0.1$, $\alpha_2 = \alpha_3 = 31$ ppm/K, $t_\text{ply} = 0.125$ mm,
$\Delta T = -150$ K, plate $50 \times 20$ mm, one element per ply. Error is
the worst interior (30 to 70 % in $x$ and $y$) ply-mean ply-local stress
against CLT, relative to the largest CLT component:

| Laminate | CLT $\sigma_1 / \sigma_2$ (MPa) | mesh | hex8 | hex8i |
|---|---|---|---|---|
| $[0/90]_{2s}$ | $-47.573 / 47.573$ | $40\times16\times8$ | 0.07 % | 0.30 % |
| $[0/90]_{2s}$, $V_p = 0.02$ | $-46.015 / 46.015$ | $40\times16\times8$ | 0.07 % | 0.30 % |
| $[0/90]_{2s}$, $V_p = 0.05$ | $-43.753 / 43.753$ | $40\times16\times8$ | 0.07 % | 0.30 % |
| $[\pm 45]_s$ | $-47.573 / 47.573$ | $40\times16\times4$ | 0.15 % | 0.15 % |
| QI $[0/90/45/{-45}]_s$ | $-47.573 / 47.573$ | $40\times16\times8$ | 0.85 % | 0.61 % |

The cross-ply value matches the closed form
$|\Delta T|(\alpha_2 - \alpha_1)(Q_{11}Q_{22} - Q_{12}^2)/(Q_{11} + Q_{22} + 2Q_{12}) = 47.573$ MPa.
A homogeneous block on a distorted mesh expands freely ($\mathbf u = \alpha\Delta T\,\mathbf x$
and zero stress to round-off) with both formulations, as do a
unidirectional off-axis laminate and any laminate with equal CTEs.

For the cross-ply in tension ($\varepsilon = 0.1$ %, interior, hex8) the
residual stress uses about 55 % of the Tsai-Wu capacity of the 90° plies
($FI_\text{th} = 0.55$), and the first-ply-failure factor drops from 7.0
without it to 2.7 with it.

## Caveats

- **Element formulation and bending.** An unsymmetric laminate warps. The
  `'hex8'` element locks in bending and underpredicts the curvature:
  $[0_4/90_4]$ reaches 0.84 of the linear-CLT curvature on a
  $40 \times 16$ mesh and 0.95 on $80 \times 32$, where `'hex8i'` gives 1.00
  on both. Symmetric laminates do not bend and are unaffected. On coarse
  in-plane meshes of thin plies (element length about 10 ply thicknesses
  and more) `'hex8i'` carries a small element-to-element oscillation in
  from the free edges that decays only over several elements; it shows as
  the 0.3 % interior error above (pointwise up to about 4 %, and an
  interior Tsai-Wu index of 0.59 instead of 0.55 for that cross-ply),
  shrinks with the element length, and is absent with `'hex8'`. Refine
  in-plane, or widen `interior_margin`, before reading interior values
  from `'hex8i'` on such meshes.
- **Linear theory.** The solve is geometrically linear. It is valid for an
  unsymmetric laminate only while the warping deflection stays small next
  to the thickness; the $[0_4/90_4]$ plate above already deflects about
  one thickness, where large-deflection effects (including bistable shapes,
  Hyer 1981) set in. A warning is logged whenever the element layup is
  unsymmetric.
- **CTE data.** Only `AS4_3501_6_epoxy` carries CTEs (WWFE-I lamina data).
  The residual stress is proportional to $\alpha_2 - \alpha_1$, and
  published lamina CTEs scatter by 5 to 10 % on $\alpha_2$; for other
  presets supply verified values.
- **Effective $\Delta T$.** Linear-elastic thermal stress ignores chemical
  shrinkage, viscoelastic relaxation during cool-down and moisture
  swelling (which relieves part of it). Treat $\Delta T$ as an effective
  value; the stress-free temperature is usually taken between the cure
  temperature and about 30 K below it. Only a uniform $\Delta T$ is
  supported.
- **Porosity and strength.** Porosity enters the residual stress through
  the stiffness only, and the FE strengths follow the square root of the
  stiffness retention ({doc}`failure`). Under that rule $Y_t$ falls more
  slowly than the residual $\sigma_{22}$, so porosity can *lower* the
  residual failure index (0.541 to 0.529 from $V_p = 0$ to 0.05 for the
  cross-ply, prototype values), whereas the empirical transverse
  knockdown would raise it.
  Do not read that as porosity being harmless under residual stress.
