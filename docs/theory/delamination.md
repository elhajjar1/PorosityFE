# Delamination

A {class}`~porosity_fe.Delamination` is an explicit interlaminar crack: a
zero-thickness, traction-free surface at one ply interface, with a planform
in $x$-$y$. It is a defect separate from porosity. The porosity field stays
an input of its own, and a delamination neither adds nor removes voids.

The FE model represents the crack and solves the delaminated laminate with
the same linear solver as every other load case. Energy release rates
(virtual crack closure), crack-face contact and a fracture criterion are
the next steps; until then, delamination *initiation* is screened by the
stress-based Hashin delamination mode ({doc}`failure`).

```python
from porosity_fe import (MATERIALS, CompositeMesh, Delamination, FESolver,
                         PorosityField)

mat = MATERIALS['T800_epoxy']                       # 24 plies
pf = PorosityField(mat, 0.02)
crack = Delamination.ellipse(11, center=(25.0, 10.0), half_size=(6.0, 4.0))
mesh = CompositeMesh(pf, mat, nx=30, ny=10, nz=24, delaminations=[crack])
result = FESolver(mesh, mat, pf).solve('compression')
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
in-plane neighbours, i.e. in a shared element face. They are where the
virtual crack closure technique will read the tie forces.

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
reference, and it never shares an entry with a solve of other ties.
Splitting the knockdown into porosity-only and delamination-only parts
comes with the fracture outputs.

## Verification

- **Fully bonded equivalence.** A mesh with two duplicated interfaces and
  every pair bonded (a strip between two node lines contains no node)
  reproduces the undelaminated solution in compression, tension, shear,
  `thermal`, tension with `delta_T`, `flexure_3pt` and `ilss`, with both
  element formulations: displacements, stresses, reactions, knockdown,
  effective modulus, failure indices and load factors agree within
  $5 \times 10^{-11}$ relative (`tests/test_delamination.py`).
- **Double-cantilever beam.** A 16-ply UD T800/epoxy beam, 60 x 4 mm, with
  1 mm elements, a mid-plane strip from the loaded end, the hinge lines of
  the two arms opened by $\pm\delta$ and the far end clamped. With
  $h$ the arm thickness, root-rotation-corrected beam theory gives
  $C = 8 (a + \chi h)^3 / (E_{11} b h^3)$ and
  $G = 12 P^2 (a + \chi h)^2 / (E_{11} b^2 h^3)$, with
  $\chi = \sqrt{\frac{E_{11}}{11 G_{13}}\left(3 - 2\left(\frac{\Gamma}{1 + \Gamma}\right)^2\right)}$
  and $\Gamma = 1.18 \sqrt{E_{11} E_{22}} / G_{13}$ ($\chi h = 2.87$ mm).
  At $a = 21$ mm the FE compliance is 0.983 of $C$, and the energy release
  rate from the compliance derivative over a one-element growth,
  $G = \frac{P^2}{2b} \frac{\Delta C}{\Delta a} = 0.311$ N/mm
  (311 J/m²), is 0.989 of the beam-theory $G$ at the mean crack length,
  21.5 mm. Uncorrected beam theory ($\chi = 0$) gives 0.245 N/mm there.

## Limitations

- No energy release rates or fracture criterion yet, so no critical load
  for growth; that is the next step (virtual crack closure,
  Benzeggagh-Kenane).
- No crack-face contact: under compression, shear or bending the faces of
  a delamination can interpenetrate, which makes the laminate too
  compliant (end-notched flexure needs contact).
- The stresses at the front are singular, so stresses and failure indices
  next to it depend on the mesh. Planform fronts are staircases on the
  structured grid.
- Linear and static: no cohesive zone, no progressive growth, no
  delamination buckling.
