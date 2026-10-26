# Delamination

A {class}`~porosity_fe.Delamination` is an explicit interlaminar crack: a
zero-thickness, traction-free surface at one ply interface, with a planform
in $x$-$y$. It is a defect separate from porosity. The porosity field stays
an input of its own, and a delamination neither adds nor removes voids.

The FE model represents the crack and solves the delaminated laminate with
the same linear solver as every other load case. The crack faces are kept
from interpenetrating by frictionless contact. Along the crack front the
solve reports the energy release rates $G_\text{I}$, $G_\text{II}$ and
$G_\text{III}$ by the virtual crack closure technique (VCCT). When the
material has interlaminar toughness data, it also reports the critical load
factor at which the delamination starts to grow, from the
Benzeggagh-Kenane mixed-mode criterion, with the toughness knocked down by
the porosity at the crack plane
([Porosity and toughness](#porosity-and-toughness)). Delamination
*initiation* from an intact interface is screened by stress-based
criteria instead: the Hashin delamination mode at the ply Gauss points,
and the Brewer-Lagace criterion on the interface tractions with O'Brien's
free-edge energy release rate ({doc}`failure`, section "Interlaminar
stresses and delamination onset").

```python
import dataclasses

from porosity_fe import (MATERIALS, CompositeMesh, Delamination, FESolver,
                         PorosityField)

# No preset has toughness data. For illustration only, the IM7/8552 values
# of the table under "Toughness data" on the T800 preset (24 plies); use
# values measured for your material (N/mm).
mat = dataclasses.replace(MATERIALS['T800_epoxy'],
                          G_Ic=0.212, G_IIc=0.774, bk_eta=2.1)
pf = PorosityField(mat, 0.02)
crack = Delamination.ellipse(11, center=(25.0, 10.0), half_size=(6.0, 4.0))
mesh = CompositeMesh(pf, mat, nx=30, ny=10, nz=24, delaminations=[crack])
result = FESolver(mesh, mat, pf).solve('compression')

front = result.crack_front              # one row per front node
front.G_T, front.psi                    # N/mm, mode mixity
front.Vp, front.G_c                     # porosity at the crack, its toughness
front.toughness_uncalibrated            # True: the placeholder knockdown law
result.fracture_load_factor             # growth at this multiple of the load
result.contact_pairs                    # crack-face pairs in contact
```

## Crack representation

Interface $k$ lies between ply $k$ and ply $k + 1$, counted from the bottom
face ($z = 0$) from 0, so an $n$-ply laminate has interfaces 0 to $n - 2$
and its mid-plane is $k = n/2 - 1$. A delamination needs every interface on
a node layer, so $n_z$ must be a multiple of $n_\text{plies}$; interface $k$
is then node layer $(k + 1)\, n_z / n_\text{plies}$.

For every interface with at least one delamination, {class}`~porosity_fe.CompositeMesh`
duplicates the whole node plane of the interface once:

- The copies are appended after the $N_0 = (n_x + 1)(n_y + 1)(n_z + 1)$
  structured nodes ({attr}`~porosity_fe.CompositeMesh.n_structured_nodes`),
  interfaces in increasing order and each plane in structured order, so
  the structured node ids do not change.
- The bottom face of the element layer above the interface is rewired to
  the copies. The original node now belongs to the elements below the
  interface and its duplicate to the elements above.
- Every per-node array (`porosity`, `stiffness_reduction`, `ply_ids`) is
  extended with the original's values. Element arrays (void elements, ply
  ids and angles) are unchanged, since rewiring moves no element centroid.

The duplication runs at the end of
{meth}`~porosity_fe.CompositeMesh.generate_mesh`, so it survives the
builders that resize the plate and generate again
({meth}`~porosity_fe.CompositeMesh.for_flexure`,
{meth}`~porosity_fe.CompositeMesh.for_ilss`), and the delaminations are
checked against the final plate. The assembled stiffness is that of two
sublaminates that share nothing at the interface. It does not depend on the
size or shape of the delamination: that enters only through the ties
below. This is decision D1 of the delamination plan; zero-thickness
interface elements, needed only for cohesive zones, can be added later on
the same duplicated nodes.

Helpers that work on the structured grid (the flexural-modulus deflection,
{meth}`~porosity_fe.CompositeMesh.mid_y_section_indices`, the section
plots of {class}`~porosity_fe.FEVisualizer`) read the first $N_0$ nodes:
on a delaminated interface they see the face below the crack.

## Planform and bonded ties

A delamination covers an open planform region:

| `shape` | Region |
|---|---|
| `'strip'` | $x_\text{start} < x < x_\text{end}$, the full width (a specimen insert) |
| `'rectangle'` | $\lvert x - x_c\rvert < a_x$ and $\lvert y - y_c\rvert < a_y$ |
| `'ellipse'` | $\left(\frac{x - x_c}{a_x}\right)^2 + \left(\frac{y - y_c}{a_y}\right)^2 < 1$ |

A duplicated node pair whose node lies strictly inside a delamination of
its interface is *delaminated*: its two nodes are the two crack faces and
move independently. Every other pair is *bonded*. A node on the boundary of
the region stays bonded; the bonded nodes next to delaminated ones are the
crack front. A strip or rectangle edge that lies on (or beyond) an edge of
the plate is not a front, though: the crack runs out to that free edge, and
the nodes on the plate edge inside the planform are delaminated too, so a
strip from $x_\text{start} = 0$ opens at the end of the specimen. Several
delaminations at one interface act as their union. On the structured grid a
curved front is a staircase of nodes. The crack tip is at the first bonded
nodes, so a boundary between two node lines acts at the next node line
outside the region: the modelled crack is up to one element larger than
the planform. A boundary on a node line is modelled exactly.

The solver bonds each bonded pair with master-slave ties on all three
displacement components,
$\mathbf u_\text{duplicate} = \mathbf u_\text{original}$
({meth}`~porosity_fe.CompositeMesh.bonded_ties`). They go into the same
affine reduction as the prescribed displacements
([Boundary conditions](fe.md#boundary-conditions)):
$\mathbf u = \mathbf T \mathbf u_r + \mathbf g$, solved as
$\mathbf T^\mathsf T \mathbf K \mathbf T\, \mathbf u_r = \mathbf T^\mathsf T (\mathbf F - \mathbf K \mathbf g)$,
directly or by conjugate gradients. With every pair bonded,
$\mathbf T^\mathsf T \mathbf K \mathbf T$ is the free-DOF stiffness of the
mesh without the delamination, up to the order of the floating-point
sums, so the solution is the undelaminated one.

Growing a crack changes only the ties. Replacing
{attr}`~porosity_fe.CompositeMesh.delaminations` by others at the same
interfaces keeps the mesh and the assembled $\mathbf K$; the sparse LU
cache and the cached thermal solution are keyed on the tie set, so the next
solve factorizes the new reduced system without reassembly. A delamination
at a new interface needs {meth}`~porosity_fe.CompositeMesh.generate_mesh`
again. {meth}`~porosity_fe.CompositeMesh.crack_front_pairs` lists the
bonded pairs on the front: those with a delaminated pair among their eight
in-plane neighbours, i.e. in a shared element face. The virtual crack
closure technique reads their tie forces
([Energy release rates](#energy-release-rates-vcct)).

## Boundary conditions

The face and line builders of {class}`~porosity_fe.BoundaryHandler`
prescribe the original nodes and the delaminated duplicates they select.
They leave out the bonded duplicates, which follow their originals through
the ties; prescribing them too would constrain one degree of freedom twice.
The rule applies to every loading mode: the compression, tension and shear
faces, the roller supports and line loads of flexure and ILSS, and the
3-2-1 supports of the thermal solve. A strip from $x = 0$ under compression
therefore holds both crack faces at $u_x = 0$ on $x_\text{min}$.

At a converged solve the residual $(\mathbf K \mathbf u - \mathbf F)$ of a
tied duplicate is the force its tie carries. The reported reactions fold it
onto the original, so a supported original reports the reaction of the
whole bonded node and a bonded duplicate reports zero; a delaminated
duplicate on a face has its own reaction.

Nodal recovery ({func}`~porosity_fe.extrapolate_to_nodes`) averages a
bonded pair as one node and reports the result at both. A delaminated pair
reports the face below the crack at the original and the face above at the
duplicate.

## Knockdown reference

The knockdown's pristine reference
([Stiffness knockdown](fe.md#stiffness-knockdown)) is free of porosity,
voids *and* delamination: it is solved on the same mesh with every
duplicated pair bonded (plan decision D6). So
$KD_\text{FE} = \mathcal S_\text{porous, delaminated} / \mathcal S_\text{pristine, bonded}$,
and a delamination alone, without porosity, gives a knockdown below 1. The
reference cache is keyed on the tie set the reference uses (every pair
bonded), so cracks of different sizes at the same interfaces share one
reference, and it never shares an entry with a solve of other ties. The
knockdown is not yet split into porosity-only and delamination-only parts.

## Crack-face contact

A linear solve lets the two faces of a delaminated pair pass through each
other: under bending, compression or shear the laminate then comes out too
compliant and the opening-mode energy release rate is spurious. The solve
therefore enforces frictionless contact with zero initial gap
({meth}`FESolver.solve(..., crack_contact=True) <porosity_fe.FESolver.solve>`,
the default; plan decision D7). The normal opening of a delaminated pair is
$g = u_z(\text{duplicate}) - u_z(\text{original})$: the interface normal is
$+z$ and the duplicate is the upper face. A primal-dual active set finds the
pairs in contact:

1. Solve with the bonded ties plus, for every active pair, a normal-only tie
   $u_z(\text{duplicate}) = u_z(\text{original})$. It is a tie on the $z$
   degree of freedom alone, so the faces slide freely. When the duplicate's
   $u_z$ is prescribed, the original follows it instead; a pair with both
   prescribed is left out.
2. Add every inactive pair with $g < -\varepsilon \max\lvert u\rvert$, and
   release every active pair whose contact force is tensile,
   $p < -\varepsilon \max\lvert r\rvert$, with $p$ the residual
   $(\mathbf K \mathbf u - \mathbf F)$ of the tied degree of freedom: the
   force the lower face exerts on the upper one, compressive positive.
   $\varepsilon = 10^{-9}$ for the direct solver, $100\,\text{rtol}$ for
   conjugate gradients.
3. Repeat until the set does not change: at most 30 solves, with a warning
   (and `contact_converged = False`) if it still changes.

Only the ties change between solves; $\mathbf K$ is assembled once. Every new
active set is a new reduced matrix $\mathbf T^\mathsf T \mathbf K \mathbf T$,
so it is factorized again. The first solve (no pair active) uses the
solver's LU cache, like a solve without contact. Every later set is
factorized in a separate one-entry cache, so the intermediate sets never
evict the load cases' factorizations and at most one factorization more is
alive. The set a load case settles on is the warm start of its next solve on
the same solver (same prescribed degrees of freedom, bonded ties and load
parts), so repeating a solve or rescaling its load takes one solve.

The results record `crack_contact`, `contact_iterations` (the solves of the
loop), `contact_pairs` and `contact_converged`.
{attr}`~porosity_fe.FieldResults.interface_forces` holds the force every
duplicated pair transmits: the tie force of a bonded pair, the normal contact
force of a pair in contact, zero for an open pair.

**Load scaling.** Under proportional mechanical loading from a zero gap, the
contact problem is positively homogeneous: scaling every prescribed
displacement and force by $\lambda > 0$ scales the solution by $\lambda$, with
the same pairs in contact. The energy release rates then grow as
$\lambda^2$, which the linear load factor below relies on (the tests check
it: twice the load gives the same contact pairs and half the factor). With
a thermal prestress, the contact is that of the total state at the applied
load (residual plus mechanical), and both the thermal unit solve and the
mechanical solve take its active set, so the two still superpose. That set
is not invariant when only the mechanical part is scaled, $\mathbf u_T +
\lambda \mathbf u_M$: the load factor keeps the applied state's set for every
$\lambda$. This is an approximation, exact when no pair changes status
between $\lambda = 1$ and $\lambda_c$. A thermal load alone
(`loading='thermal'`) is proportional in `delta_T`, so its set is invariant.

## Energy release rates (VCCT)

The front nodes are the bonded pairs with a delaminated pair among their
eight in-plane neighbours
({meth}`~porosity_fe.CompositeMesh.crack_front_pairs`). For each front node
$p$:

- **The pair behind**, $q$, is the delaminated neighbour most aligned with
  the mean direction from the node's delaminated edge-sharing neighbours to
  $p$ (from its delaminated diagonal neighbours when it has no edge-sharing
  one). On a straight front it is the neighbour straight behind; at a corner
  of a staircase front it is the diagonal one.
- **The local frame**: $x' = (\mathbf x_p - \mathbf x_q) / \lvert \mathbf
  x_p - \mathbf x_q\rvert$ is the in-plane growth direction, the outward
  normal of the delaminated region: exactly $\pm x$ on a front across the
  width, one of eight grid directions on a staircase. $y' = +z$ is the
  interface normal and $z' = x' \times y'$ runs along the front.
- **$\Delta a$ and $b$**: $\Delta a = \lvert \mathbf x_p - \mathbf
  x_q\rvert$, the element length along $x'$, and $b = A_p / \Delta a$, with
  $A_p$ the node's tributary (dual) cell of the interface on the structured
  grid: half the spacing to each neighbour along $x$ times the same along
  $y$. So $b$ is halved at a plate edge and $A_p$ quartered at a plate
  corner. On a straight front $\Delta a\, b = \Delta x\, \Delta y$, Krueger's
  area with the half width at the edges, and $\sum G_T\, b\, \Delta a$ over
  the front is the closure work of the whole front.
- **Force and opening**: $\mathbf F'$ is the force the tie exerts on the
  upper face (the duplicate) at $p$, the residual $(\mathbf K \mathbf u -
  \mathbf F)$ at the duplicate's degrees of freedom, read before the tie
  forces are folded onto the originals for the reactions.
  $\boldsymbol\Delta'$ is the opening at $q$, upper minus lower face. Both
  are rotated into the frame.

The 8-node brick formulas (Krueger 2004) are then

$$
G_\text{I} = -\frac{F'_y \Delta'_y}{2 \Delta a\, b}, \qquad
G_\text{II} = -\frac{F'_x \Delta'_x}{2 \Delta a\, b}, \qquad
G_\text{III} = -\frac{F'_z \Delta'_z}{2 \Delta a\, b},
$$

$G_T = G_\text{I} + G_\text{II} + G_\text{III}$ and $\psi = (G_\text{II} +
G_\text{III}) / G_T$. In an opening double-cantilever beam the tie holds the
upper face down ($F'_y < 0$) while the faces behind open ($\Delta'_y > 0$),
so $G_\text{I} > 0$. A negative mode I product is set to zero: the faces
behind are closed, and a pair in contact has $\Delta'_y = 0$. The shear
products keep their sign. At a single node they can be negative where the
local field is not self-similar: across the width of a multidirectional
laminate the mode III product changes sign, for instance. Only signed values
make $\sum G_T\, b\, \Delta a$ the energy the extension releases. Zeroing
the negative ones over-predicted $G$ on a QI short-beam specimen with a
mid-plane crack in the shear span: the ratio to the centred compliance
derivative was 1.041 and 1.070 at 0.55 and 0.37 mm elements, against 0.989
and 0.995 signed. The growth criterion takes $\psi$ and the mode III share
clipped to $[0, 1]$, so a negative shear product lowers the node's $G_T$
without inverting its mode mix. $G_T$ does not depend on the frame (it is the dot
product $-\mathbf F' \cdot \boldsymbol\Delta' / (2 \Delta a\, b)$); on a
staircase front only the choice of $q$ matters for it, while the split
between $G_\text{II}$ and $G_\text{III}$ follows the quantized $x'$. At an
interface between plies of different orientation the split depends on
$\Delta a$ (the oscillatory bimaterial singularity) while $G_T$ converges,
so $G_T$ is the quantity to compare (plan decision D5).

{attr}`FieldResults.crack_front <porosity_fe.FieldResults.crack_front>` is
a {class}`~porosity_fe.CrackFront` table with one row per front node.
{func}`~porosity_fe.vcct_energy_release_rates` builds it from a displacement
field and the interface forces
({meth}`FieldResults.energy_release_rates <porosity_fe.FieldResults.energy_release_rates>`
recomputes it from a result), or from the full $\mathbf K \mathbf u -
\mathbf F$ of a solve of your own boundary conditions.

## Growth criterion and load factor

The mixed-mode toughness is Benzeggagh and Kenane's (1996), with Reeder's
(2006; NASA NTRS 20060048260, which interpolates linearly between the mode II
and mode III toughness) term for a separate mode III toughness
({func}`~porosity_fe.critical_energy_release_rate`):

$$
G_c = G_{Ic} + (G_{IIc} - G_{Ic})\, \psi^\eta
      + (G_{IIIc} - G_{IIc}) \frac{G_\text{III}}{G_\text{II} + G_\text{III}}\, \psi^\eta ,
$$

plain B-K when $G_{IIIc} = G_{IIc}$, which is the default. It gives
$G_{Ic}$ at $\psi = 0$, $G_{IIc}$ in pure mode II and $G_{IIIc}$ in pure mode
III. The inputs are the optional {class}`~porosity_fe.MaterialProperties`
fields `G_Ic`, `G_IIc`, `G_IIIc` (default `G_IIc`) and `bk_eta`, in N/mm. No
preset sets them (see [Toughness data](#toughness-data)). Without them the
solve still reports the energy release rates, logs one INFO line, and the
load factor is `None`. Each pure-mode toughness is the material's, knocked
down by the porosity at the front node
([Porosity and toughness](#porosity-and-toughness)); $\eta$ does not
change.

**Mechanical load.** Every energy release rate scales with $\lambda^2$ and
$\psi$ does not change, so node $i$ starts to grow at

$$
\lambda_{c,i} = \sqrt{G_c(\psi_i) / G_{T,i}}
$$

($\infty$ where $G_T = 0$). The smallest over the front is
{attr}`~porosity_fe.FieldResults.fracture_load_factor`, at the node
{attr}`~porosity_fe.FieldResults.fracture_governing_node`
(`summary().details` adds its $\psi$ and $G_T$). It multiplies the load named
by `load_factor_basis`: the applied load, or `delta_T` for
`loading='thermal'`.

**Thermal prestress** (a mechanical load with `delta_T`, basis
`'mechanical_with_residual'`). The front state is
$\mathbf F' = \mathbf F'_T + \lambda \mathbf F'_M$,
$\boldsymbol\Delta' = \boldsymbol\Delta'_T + \lambda \boldsymbol\Delta'_M$,
so each mode's closure work is quadratic in $\lambda$ and the mix changes
with it. $\lambda_c$ is the smallest positive root of
$G_T(\lambda) = G_c(\psi(\lambda))$ at each node
({func}`~porosity_fe.fracture_load_factors` with `prestress=`): bracketed on
a geometric grid of 12 % steps around the mechanical estimate and refined
with `scipy.optimize.brentq`. It is 0 where the residual state alone reaches
$G_c$ and $\infty$ where the load releases no energy. The contact set of the
applied state is held for every $\lambda$ (see
[Crack-face contact](#crack-face-contact)).

## Outputs and exports

- {class}`~porosity_fe.FieldResults`: `crack_front`, `fracture_load_factor`,
  `fracture_governing_node`, `interface_forces`, `crack_contact`,
  `contact_iterations`, `contact_pairs` and `contact_converged`, all `None`
  without a delaminated pair; `summary().details` carries them, with
  `toughness_knockdown` (a description of the porosity-toughness law) and
  `toughness_uncalibrated`, which the front table holds as
  `crack_front.toughness_knockdown` and `crack_front.toughness_uncalibrated`.
- JSON ({meth}`FESolver.export_results <porosity_fe.FESolver.export_results>`,
  schema 1.3): a `fracture` block with the contact counts, the load factor,
  the governing node, its $\psi$ and $G_T$, `toughness_knockdown` and
  `toughness_uncalibrated`, and the front table (one array per column,
  unit-suffixed keys such as `G_T_N_per_mm`; `Vp` is the porosity at the
  crack plane). Undefined or infinite values are `null`. A mesh without a
  delaminated pair writes no block.
- VTU ({meth}`FieldResults.to_vtu <porosity_fe.FieldResults.to_vtu>`): point
  data `G_I`, `G_II`, `G_III` and `G_T` at both nodes of every front pair,
  `NaN` elsewhere. The legacy ASCII VTK leaves them out, since its readers do
  not parse `NaN` reliably.

## Verification

- **Fully bonded equivalence.** A mesh with two duplicated interfaces and
  every pair bonded (a strip between two node lines contains no node)
  reproduces the undelaminated solution in compression, tension, shear,
  `thermal`, tension with `delta_T`, `flexure_3pt` and `ilss`, with both
  element formulations: displacements, stresses, reactions, knockdown,
  effective modulus, failure indices and load factors agree within
  $5 \times 10^{-11}$ relative (`tests/test_delamination.py`).
- **Double-cantilever beam compliance.** A 16-ply UD T800/epoxy beam,
  60 x 4 mm, with 1 mm elements, a mid-plane strip from the loaded end, the
  hinge lines of the two arms opened by $\pm\delta$ and the far end clamped.
  With $h$ the arm thickness, root-rotation-corrected beam theory gives
  $C = 8 (a + \chi h)^3 / (E_{11} b h^3)$ and
  $G = 12 P^2 (a + \chi h)^2 / (E_{11} b^2 h^3)$, with
  $\chi = \sqrt{\frac{E_{11}}{11 G_{13}}\left(3 - 2\left(\frac{\Gamma}{1 + \Gamma}\right)^2\right)}$
  and $\Gamma = 1.18 \sqrt{E_{11} E_{22}} / G_{13}$ ($\chi h = 2.87$ mm).
  At $a = 21$ mm the FE compliance is 0.983 of $C$, and the energy release
  rate from the compliance derivative over a one-element growth,
  $G = \frac{P^2}{2b} \frac{\Delta C}{\Delta a} = 0.311$ N/mm
  (311 J/m²), is 0.989 of the beam-theory $G$ at the mean crack length,
  21.5 mm. Uncorrected beam theory ($\chi = 0$) gives 0.245 N/mm there.

The fracture-mechanics checks (`tests/test_vcct.py`, measured values;
"centred" is the difference over one element either side of $a$, which
estimates the derivative at $a$ like VCCT, while the one-element forward
difference is centred half an element ahead):

| Check | Model | Reference | Measured | Test tolerance |
|---|---|---|---|---|
| DCB $G_\text{I}$ | the beam above, $a = 21$ mm | $\frac{P^2}{2b}\frac{dC}{da}$, one-element forward / centred | 0.959 / 0.9991 (width average 0.2986 N/mm) | 5 % / 0.5 % |
| DCB $G_\text{I}$ | same | corrected beam theory at $a$ | 0.988 | 3 % |
| DCB mode purity | same | $G_\text{II} = G_\text{III} = 0$ | $G_\text{II}/G_\text{I}$, $G_\text{III}/G_\text{I} < 10^{-24}$ | $10^{-3}$ |
| Crack closure identity | one-element extension, fixed grips | $U(a) - U(a + \Delta a)$ | $2 \times 10^{-12}$ relative | $10^{-9}$ |
| Energy balance | $\sum G_T\, b\, \Delta a$ (one-step VCCT) | $[U(a - \Delta a) - U(a + \Delta a)] / 2$ | 0.9938 (1.085 against the forward extension alone) | 2 % |
| Width, no Poisson coupling | $\nu = 0$, 4 mm wide, 5 front nodes | uniform | spread $1.7 \times 10^{-12}$ | $10^{-6}$ |
| Width, T800 | 20 mm wide, 2 mm elements | uniform away from the edges | 1.031 to 1.043 of the mean from 4 to 16 mm (1.2 % spread); 0.736 at the free edges | 2 % |
| Convergence | $a = 20$ mm, $\Delta a$ = 2, 1, 0.5 (0.25) mm | corrected beam theory | 0.9792, 0.9878, 0.9931 (0.9961) | monotone, 1 % at 0.5 mm |
| Single-leg $\psi$ | upper arm loaded, $a = 21$ mm | Williams with root rotation ($\chi h$ in mode I, $0.42\,\chi h$ in mode II, attributed to Wang and Williams 1992; that citation is not checked against the paper): 0.3936; $3/7 = 0.4286$ for a long crack | 0.3897 (0.990) | 5 % |
| Single-leg $G_T$ | same | centred / forward compliance derivative | 0.9991 / 0.957 | 5 % |
| ENF $G_\text{II}$ | three-point bend, span 50 mm, 0.96 mm elements, $a = 15.4$ mm (0.6 of the half span), with contact | centred / forward compliance derivative | 0.9983 / 0.943; $G_\text{II}$ is 99.5 % of $G_T$ | 2 % |
| ENF contact | same | $G_\text{I} = 0$, no penetration | $G_\text{I} \le 1.1 \times 10^{-6}\, G_T$; least opening 0; 15 pairs in contact after 4 solves | $10^{-4}\, G_T$ |
| ENF without contact | same, `crack_contact=False` | | the faces overlap by up to 0.86 mm (deflection 0.63 mm); compliance 1.49 times | a visible overlap |
| Load scaling | ENF at twice the load | $\lambda_c / 2$ | ratio 0.5 to $10^{-9}$, the same 15 pairs, 1 solve (warm start) | $10^{-9}$ |
| Thermal $\lambda_c$ | cross-ply $[0/90/0/90]_s$ AS4/3501-6, 0/90 interface, $\Delta T = -100$ K, no contact | root of $G_T(\lambda) = G_c(\psi(\lambda))$ from the separate thermal and mechanical solves | identical; 0.2148 against 0.2771 without the residual state | $10^{-10}$ |
| QI shear front | QI short-beam specimen, mid-plane crack in the shear span, $48 \times 4 \times 24$, no contact | centred compliance derivative | 0.989 (0.995 at $72 \times 4$); the mode III product changes sign across the width (1.041 / 1.070 if negative products are zeroed) | 2 % |
| Penny crack | isotropic block, quarter model, $4a \times 4a \times 8a$, $h = a/8$, remote tension | $G = (1 - \nu^2) K_I^2 / E$, $K_I = 2\sigma\sqrt{a/\pi}$ | front mean 1.061 at the nominal $a$, 1.003 at the mean front radius (1.058 $a$); 0.785 to 1.434 around the front | 10 % / 5 % |

The penny crack's front nodes lie between $a$ and $a + \sqrt 2\, h$ from
the centre: the modelled crack is the staircase, larger than the circle, and
the closed form scales with $a$. Outside the test suite, the same model
gave a front mean of 1.131, 1.061, 1.046 and 1.035 of the closed form at the
nominal radius for $h = a/4$, $a/8$, $a/12$ and $a/16$ (1.032, 1.003, 1.010
and 1.003 at the mean front radius), and 1.038 (0.981) with the block
enlarged to $6a \times 6a \times 12a$ at $h = a/8$. The node-to-node scatter
around the staircase does not shrink with refinement (0.73 to 1.43 at
$h = a/16$); the front average converges.

## Toughness data

No preset carries toughness. Sourced values from the delamination plan's
literature brief (N/mm; [P] read in the primary source, [S] from a
secondary source, [NV] not verified):

| Material | $G_{Ic}$ | $G_{IIc}$ | B-K $\eta$ | Source |
|---|---|---|---|---|
| IM7/8552 (benchmark reference) | 0.212 | 0.774 | 2.1 | Krueger, NASA/CR-2012-217562 [P] |
| IM7/8552 | 0.206 (fatigue precracked) to 0.237 (insert) | 0.739 to 0.807 (precracked) | | Czabaj and Ratcliffe 2013, doi:10.1016/j.compscitech.2013.09.008; O'Brien et al., NASA/TM-2010-216838 [P] |
| AS4/3501-6 | 0.077 to 0.091 | 0.54 to 0.71 | NV | Reeder 1992, NASA TM-104210 [P] |
| AS4/PEEK (APC-2) | 0.969 | 1.719 | 2.284 | Camanho and Dávila, NASA/TM-2002-211737 [P] (Reeder: 0.78 to 0.84 / 1.16 to 1.25) |
| E-glass/epoxy (M10) | 0.118 | 2.906 | 2.6 [S] | Kenane and Benzeggagh 1997, doi:10.1016/S0266-3538(97)00021-3 [P] |
| IM7/8551-7 | 0.61 to 0.63 (initiation) | NV | NV | Naik, Crews and Shivakumar, NASA TM-101677 [P] |
| T300/934 | about 0.088 | NV | NV | Funk and Deaton, NASA TP-2950 [P] |
| T700/EPON 828 | 0.52 (at 1.49 % voids) | NV | NV | Hakim et al. 2017, doi:10.4236/msa.2017.82011 [P] |
| T800H/3900-2 | 0.564 (initiation) | NV | NV | Zhang and Fox 2007, doi:10.1080/10426910701385432 [P, abstract] |
| T800S/M21, HTA/EHkF420 | NV | NV | NV | No verified values found |
| T300/914C (DCB benchmark) | 0.1703 | 0.4936 | 1.62 | Krueger, NASA/TM-2008-215123 [P] |
| C12K/R6376 (SLB benchmark) | 0.3405 | 1.286 | 3.39 | Krueger, NASA/TM-2008-215123 [P] |

Use precracked or initiation values as the base that the porosity
knockdown applies to; values measured from the insert run high.

## Porosity and toughness

Voids at the crack plane lower the interlaminar toughness. The toughness at
each front node is the material's pristine value times a porosity factor
for each mode ({class}`~porosity_fe.ToughnessKnockdown`):

$$
G_{Ic}(V_p) = G_{Ic}\, f_\text{I}(V_p), \qquad
G_{IIc}(V_p) = G_{IIc}\, f_\text{II}(V_p), \qquad
G_{IIIc}(V_p) = G_{IIIc}\, f_\text{III}(V_p),
$$

and the B-K criterion above takes these, with $\eta$ unchanged. The law
lives outside {class}`~porosity_fe.MaterialProperties`, which holds the
pristine toughness, and is chosen per solve:
{meth}`FESolver.solve(..., toughness_knockdown=...) <porosity_fe.FESolver.solve>`,
and the same keyword on {func}`~porosity_fe.fracture_load_factors` and
{func}`~porosity_fe.critical_energy_release_rate`.
{func}`~porosity_fe.toughness_knockdown_factor` evaluates the factor without
FE, for screening; no empirical strength mode uses it.

| `toughness_knockdown` | $f(V_p)$ |
|---|---|
| `'exponential'` (default) | $e^{-b V_p}$ with the placeholders $b_\text{I} = 6.2$, $b_\text{II} = b_\text{III} = 14$ |
| `ToughnessKnockdown(b_I=..., b_II=..., b_III=...)` | $e^{-b V_p}$ with your coefficients; one left `None` keeps its placeholder |
| `ToughnessKnockdown('linear', b_I=..., b_II=...)` | $\max(1 - b V_p, 0)$; it has no placeholder, so both are required |
| a callable `f(Vp, mode)` | your factor at each `Vp` (a float) and `mode` (`'I'`, `'II'` or `'III'`) |
| `None` or `'none'` | 1, the pristine toughness |

$b$ is per unit $V_p$ (a fraction) and must be finite and $\ge 0$. Mode III
follows mode II unless `b_III` is given. Every law gives $0 \le f \le 1$ and
$f(0) = 1$: the toughness never rises with porosity, and a callable value
outside $[0, 1]$ raises. Without porosity at the front, $G_c$ and
$\lambda_c$ are exactly the pristine ones, whatever the law.

**Local porosity.** $V_p$ at a front node (`CrackFront.Vp`) is the porosity
field at the crack plane: the mean of
{attr}`CompositeMesh.porosity <porosity_fe.CompositeMesh.porosity>` at the
pair's original and duplicate node, which hold the same value, the field
sampled at the interface node. This is where the porosity *distribution*
matters for delamination: interface-clustered voids lower the toughness
where the crack runs. On the end-notched flexure beam of
`tests/test_toughness.py` (16 UD plies of T800/epoxy with the IM7/8552
toughness of the table above, a mid-plane crack, 2 % mean porosity, pure
mode II at the front):

| Porosity field | $V_p$ at the front | $G_c$ (N/mm) | $\lambda_c$ |
|---|---|---|---|
| `'uniform'`, no knockdown (`None`) | 0.020 | 0.774 | 3.160 |
| `'uniform'` | 0.020 | 0.585 | 2.747 |
| `'interface'` (Gaussians at every ply interface) | 0.025 | 0.544 | 2.649 |
| `'clustered'` at the mid-plane | 0.048 | 0.395 | 2.255 |

The field values come from the profiles as they are normalized today, so
the local $V_p$ of the `'interface'` profile depends on plan item 0.4 (its
renormalization). A discrete {class}`~porosity_fe.VoidGeometry` that
contains a front node sets $V_p = 1$ there, far beyond the data.

**Calibration status: uncalibrated.** No published $G_c(V_p)$ relation
exists. About ten studies measured both, and they disagree (Mehdikhani et
al. 2019, doi:10.1177/0021998318772152): initiation $G_{Ic}$ falls or does
not change, $G_{IIc}$ falls, and propagation $G_{Ic}$ can *rise* through
fibre and ply bridging and multiple crack planes. The placeholders are
two-point estimates $b = -\ln(G_c / G_{c,\text{ref}}) / \Delta V_p$ ([P] read
in the primary source, [S] from a secondary source):

| Study | Material | $V_p$ | Finding | $b$ |
|---|---|---|---|---|
| Hakim et al. 2017 [P], doi:10.4236/msa.2017.82011 | T700/EPON 828 UD, DCB | 1.49, 1.77, 3.43 % | static $G_{Ic}$ 0.52, 0.44, 0.45 N/mm (Table 1) | 7.45 from the end points (4.39 by least squares over all three) |
| Olivier et al. 2007 [P], ICCM16 (open access, Zenodo record 15414374) | T700GC/M21 $[0]_{16}$, DCB (ISO 15024, Berry's method) | 0.1, 5.09, 15.7 % | "a void content of 5% induces a decrease by about 22% in GIC" (§4.3), read from R-curves to 140 mm of crack (Fig. 8); cited as a propagation value by Carraro et al. 2015 | 4.97 |
| Agius and Fox 2015 [P, abstract], doi:10.1016/j.compositesa.2015.02.023 | out-of-autoclave prepreg, not named in the abstract | < 0.3 / 1.5 % | $G_{IIc}$ 21 % higher for the low-void laminates | 12.7 to 15.9 for a low-void value of 0 to 0.3 %; 14.1 at 0.15 % |
| Carraro, Maragoni and Quaresimin 2015 [P], doi:10.1179/2055035914Y.0000000004 | STS40 NCF/EC157, DCB | 0 / 1.9 % (0° interface), 0 / 1.23 % (±45°) | initiation unchanged (0°) or slightly higher (±45°), propagation slightly higher; values only in figures | not used |
| Asp and Brandt 1997 [S], ICCM-11 | UD carbon/epoxy | 0 / about 1 % | initiation $G_c$ slightly down (mode I) or unchanged (mixed mode), mode II slightly down; the original was not read | not used |

$b_\text{I} = 6.2$ is the mean of 4.97 and 7.45, and $b_\text{II} = 14$ the
Agius and Fox estimate at the midpoint of its low-void range, rounded. Mode
III has no data. Every result that a placeholder changes is flagged:
`crack_front.toughness_uncalibrated` (and the same key in
`summary().details` and the JSON `fracture` block) is `True` when the law
uses a placeholder and some front node has $V_p > 0$, and the first such
use logs one INFO line per process. `toughness_knockdown` describes the law
in every case.

**Validation.** `validate_porosity` runs the bundled toughness datasets
(`hakim_2017`, `agius_fox_2015`) through the law, normalized to each
baseline group, and reports them in a section of their own, apart from the
strength and modulus MAE: 6.20 % and 1.15 % (3.67 % property-weighted,
4.18 % point-weighted over 5 points). The placeholders come in part from
the same studies, so this is a consistency check, not an independent
validation. Hakim's 1.77 % and 3.43 % groups have about the same $G_{Ic}$,
which no monotone law follows.

**Limits.**

- Conservative by construction: the law ignores the rise of mode I
  propagation toughness with voids. VCCT with the load factor is a
  growth-onset criterion, so the pristine values should be initiation or
  precracked ones (see [Toughness data](#toughness-data)).
- The data span $V_p \le 0.05$; beyond it the law extrapolates.
- One porosity per front node, the field at the interface. Void size,
  shape and position relative to the crack plane are not modelled, though
  they matter: voids on the crack plane ease growth, voids off it deflect
  the crack and add bridging (Carraro et al. 2015). Hakim et al. measured
  more porosity in the layers around the insert than the specimen average.
- The coefficient uncertainty is not propagated
  ({func}`~porosity_fe.propagate_uncertainty` does not perturb $b$; plan
  item 4.5).

## Limitations

- Linear elastic fracture mechanics: $\lambda_c$ is the load at which growth
  *starts*. There is no propagation, R-curve, cohesive zone or unstable
  growth, and no delamination buckling.
- The crack size is quantized to node lines, and a planform front is a
  staircase whose nodes lie up to one diagonal element outside the
  planform. On a staircase $G$ scatters from node to node (0.79 to 1.43 of
  the closed form around the penny crack at $h = a/8$), so read the front
  average, or refine; $G_\text{II}$ and $G_\text{III}$ are split along one of
  eight grid directions.
- Where the front meets a free edge, the edge node reads a lower $G$ (0.74
  of the mean on the 20 mm UD beam) that depends on the mesh, the corner
  singularity of the free surface.
- At an interface between plies of different orientation the mode split
  depends on $\Delta a$; compare $G_T$.
- Contact is frictionless, node to node on the matching duplicated nodes
  (small sliding), from a zero gap. With a thermal prestress the load factor
  holds the contact set of the applied state.
- No preset has toughness data. The porosity knockdown of the toughness has
  uncalibrated placeholder coefficients and ignores the rise of mode I
  propagation toughness with voids
  ([Porosity and toughness](#porosity-and-toughness)). The stiffness
  knockdown is not split into porosity and delamination parts.
- The stresses at the front are singular, so stresses and failure indices
  next to it depend on the mesh.
