# Finite-element model

{class}`~porosity_fe.FESolver` solves a linear static problem on a
structured hexahedral mesh of the specimen, with stiffness degraded
element by element from the local porosity.

## Mesh and elements

{class}`~porosity_fe.CompositeMesh` divides the specimen into
$n_x \times n_y \times n_z$ eight-node hexahedra (default production
resolution $30 \times 10 \times n_\text{plies}$, one element layer per ply,
with $n_z$ capped at 48). Each element gets the ply angle of
the ply containing its centroid; a centroid on a ply interface takes the
ply above. The porosity field is sampled at every node. Elements whose
centroid lies inside a discrete void are flagged as void elements. The
total element count is capped at $10^6$. An explicit delamination
(`delaminations=`) duplicates the node plane of its ply interface and
appends the copies after the structured nodes; see {doc}`delamination`.

```{warning}
An element layer that spans several plies takes only one of their angles,
so unless $n_z$ is a multiple of the ply count the FE laminate is not the
requested one. $n_z = 12$, the production default before it became the
ply count, does not resolve the 24-ply presets: for the T800/epoxy QI
laminate it models $[90/{-45}/45/0]_3$, which keeps the angle fractions but
not the stacking sequence or symmetry (pristine $E_x$ 1.6 % below CLT;
$n_z = 24$ is within 0.5 %). Bending, interlaminar stresses and failure
indices are more sensitive to the sequence than the membrane stiffness is
(see "Production resolution" below). Above 48 plies the default $n_z$
stops at 48 and no longer resolves the plies; pass `mesh_res` (or build the
mesh) with $n_z = n_\text{plies}$ for such a laminate.
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
Gauss-point transverse shear stress is affected far more: in a
three-point bend of a UD beam ($50 \times 20 \times 2$ mm) at
$\Delta x / h = 1.56$, `hex8` reports about twice the beam-theory peak
$0.75\,|P| / (b h)$ at the Gauss points near mid-span, while its
element-mean shear is correct.

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
| compression maximum failure index | $-13$ to $-14$ % | $-8$ to $-12$ % |

The compression maximum failure index sits at the constrained corners. On
the default ASTM D2344 specimen of the same cases ($24 \times 8 \times 24$,
element length $h/4$; see [Short-beam shear](#short-beam-shear-ilss)) the
ILSS knockdown moves by $-0.0001$ to $+0.0003$, the peak
$\lvert\tau_{13}\rvert$ drops by 32 to 35 % (the parasitic Gauss-point
shear of `'hex8'`), the domain first-ply-failure load factor falls by 6 to
7 % and the interior one rises by 1.4 to 1.6 %; the mid-plane shear at the
quarter span changes by 0.2 %. The elements are short there, so locking
costs little stiffness.
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
| `ilss` | force | ASTM D2344 short beam on the {meth}`CompositeMesh.for_ilss <porosity_fe.CompositeMesh.for_ilss>` specimen: the `flexure_3pt` rollers $4h$ apart, `applied_load` as a uniform pressure over the loading-nose contact band at mid-span; see [Short-beam shear](#short-beam-shear-ilss) |
| `flexure_3pt` | force | roller supports: $u_z = 0$ on the bottom-face node lines at the two supports, plus $u_x$ at one node and $u_y$ at two (no horizontal reaction); `applied_load` as a consistent line load across the top face at mid-span; see [Flexure](#flexure) |
| `flexure_4pt` | force | the same rollers; half of `applied_load` on each of two top-face lines `load_span` apart, centred on mid-span |
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
diagonal ratio is about 31 (T800/epoxy QI, compression), where the penalty
method used before (a spring of $10^6 \max_i K_{ii}$ on each constrained
degree of freedom) gives $3.2 \times 10^7$ and left a boundary slack of
about $10^{-8}$ of the displacement.

The reduction is one affine map, $\mathbf u = \mathbf T \mathbf u_r + \mathbf g$,
with the reduced system $\mathbf T^\mathsf{T} \mathbf K \mathbf T\, \mathbf u_r
= \mathbf T^\mathsf{T} (\mathbf F - \mathbf K \mathbf g)$. For prescribed
values alone $\mathbf T$ selects the free degrees of freedom, which gives the
system above (computed by slicing, as before). The same map also takes
master-slave ties $u_s = u_m + c$, chained or ending on a prescribed degree of
freedom. A delaminated mesh ties the bonded part of each duplicated
interface, in every loading mode ({doc}`delamination`); without a
delamination no loading mode uses ties. The periodic unit cell that checks
the Mori-Tanaka void model does too ({doc}`micromechanics`).

The default solve is a sparse LU factorization of $\mathbf K_{ff}$. A
solver keeps the factorizations of the last two sets of constrained degrees
of freedom (about 320 MB each on the production mesh), so load cases that
share supports, or two that alternate, factorize once; all are rebuilt when
the mesh, material or porosity change. Jacobi-preconditioned conjugate gradients
(`solver='cg'`) agree with it to about $10^{-8}$ at the default
`rtol = 1e-9` and need no fill-in, which pays off above roughly 40,000
degrees of freedom. MINRES (`solver='minres'`) is deprecated and will be
removed in 2.0; it emits a `DeprecationWarning`. $\mathbf K_{ff}$ is
symmetric positive definite, so CG covers every case, and MINRES, a
residual minimizer, is less accurate at the same tolerance: on the
production mesh its displacements are $4 \times 10^{-7}$ to
$2 \times 10^{-6}$ from the direct solve in compression and shear (CG:
$5 \times 10^{-9}$ to $10^{-8}$). Until then it still works, warm-restarted
until its true residual meets `rtol`.

```{note}
Before the ASTM D2344 rework, `ilss` was a three-point bend of whatever
coupon the mesh was, with both bottom end edges pinned in $x, y, z$: on
the 50 mm coupon a span of 11.4 thicknesses, mostly flexure, and a tied
arch (horizontal reaction about $4P$, deflection about half the simply
supported one). The closest model now is `'flexure_3pt'` on
`CompositeMesh.for_flexure(field, material, span=50.0, width=20.0,
overhang=0.0)`, on rollers instead of pinned edges.
```

## Stiffness knockdown

The FE `knockdown` compares the structural stiffness of the porous model
with a pristine reference solve of the same mesh, loading and supports,
with zero porosity, no void elements and, on a delaminated mesh, every
interface node pair bonded ({doc}`delamination`):

$$
KD_\text{FE} = \frac{\mathcal S_\text{porous}}{\mathcal S_\text{pristine}},
\qquad
\mathcal S =
\begin{cases}
\mathbf u^\mathsf T \mathbf K \mathbf u / \varepsilon^2 & \text{compression, tension, shear} \\
P^2 / \mathbf u^\mathsf T \mathbf K \mathbf u & \text{ILSS, flexure}.
\end{cases}
$$

For the displacement-controlled modes $\mathcal S$ is the effective
modulus times the volume, so $KD_\text{FE}$ is the ratio of $E_x$ (or
$G_{xy}$ for shear). For ILSS and flexure it is the ratio of beam
stiffnesses (inverse compliances). The pristine solve is cached by mesh
geometry (for flexure and ILSS also by the support and load lines), so a
porosity sweep on one mesh pays for it once. A ratio above 1 is not
clipped; a warning is logged. `effective_modulus` reports
$\mathbf u^\mathsf T \mathbf K \mathbf u / (\varepsilon^2 V)$ for the
displacement-controlled modes and the flexural modulus for the flexure
modes (below); for ILSS it is `None` (see
[Short-beam shear](#short-beam-shear-ilss)).

$KD_\text{FE}$ is a *stiffness* knockdown. Strength enters through the
failure criteria on {doc}`failure`.

## Flexure

`loading='flexure_3pt'` and `'flexure_4pt'` model the flexure tests of
ASTM D7264 (polymer-matrix composites; procedure A three-point,
procedure B four-point with the load span half the support span) and
ASTM D790 (three-point). Both are force controlled by `applied_load`.

**Geometry.** {meth}`CompositeMesh.for_flexure <porosity_fe.CompositeMesh.for_flexure>`
builds the specimen and records the support and load lines as
{attr}`~porosity_fe.CompositeMesh.flexure_geometry`
({class}`~porosity_fe.FlexureGeometry`). Its defaults follow D7264: a
support span $L$ of 32 laminate thicknesses (`span_to_thickness`; D7264
also allows 16, 20, 40 and 60, D790 uses 16 by default and 32 to 60 for
high-modulus composites), a 13 mm width, 10 % of the span overhanging each
support, and a four-point load span of $L/2$ (loads at the quarter
points). Every ply gets its own element layer ($n_z$ = ply count), and the
element length is at most $\min(h, 20 h / n_z)$, which keeps the aspect
ratio at or below 20 and $\Delta x / h \le 1$. $n_x$ is chosen so that the
supports, mid-span and the four-point load lines fall on node lines. For
the 24-ply T800 preset this gives a 140.5 mm span and $48 \times 4 \times 24$
elements (18,375 degrees of freedom). `nx=` sets the element count along
the length instead (even, so mid-span is a node line): the span keeps its
length and gets the most elements that leave each overhang at least 10 %
of it, two fewer when that puts the four-point load lines on node lines,
and the overhangs take the rest. Solving both procedures takes
about 3 to 4 s on a pristine mesh and 7 to 8 s on a porous one, which
adds the pristine reference solves (four cores); the two procedures share
the supports and so the factorization. On a mesh without
`flexure_geometry` the supports are the two bottom end edges. The web
app offers both procedures as FE-only loading modes on this specimen
(with $n_z$ from its mesh setting), and its mesh-convergence check refines
the specimen.

**Supports and loads.**
{meth}`BoundaryHandler.roller_supports <porosity_fe.BoundaryHandler.roller_supports>`
holds $u_z = 0$ along each support line of the bottom face. Three more
constraints remove the in-plane rigid-body motions: $u_x$ at the
mid-width node of the first support and $u_y$ at the mid-width nodes of
both. They are statically determinate in the plane, so the supports
exert no horizontal reaction (below $2 \times 10^{-9}$ N at
$P = 100$ N) and the beam shortens and contracts across its width
freely. {meth}`BoundaryHandler.line_load <porosity_fe.BoundaryHandler.line_load>`
applies each load across the top face with consistent nodal forces:
trapezoidal weights across the width (the edge nodes carry half an
interior share, where equal shares would give them more) and the linear
shape functions along $x$, so a load between two node lines is shared in
proportion to the distance. `load_width` spreads it as a uniform pressure
over a band instead of a line.

**Flexural modulus.** `effective_modulus` is the modulus $E_f$ that
Timoshenko beam theory needs to reproduce the mid-span deflection
$\delta$, with the transverse shear deflection taken out:

$$
\frac{\delta}{P} = \frac{a\,(3L^2 - 4a^2)}{48\,E_f I} + \frac{a}{2\,b\,K_s},
\qquad I = \frac{b h^3}{12},
$$

with $a$ the shear span ($L/2$ for three-point loading, $(L - L_\text{load})/2$
for four-point), $b$ the width and $h$ the thickness. Without the shear
term these are the D790 / D7264 formulas $E_f = L^3 m / (4 b h^3)$ and,
for $a = L/4$, $E_f = 0.17\,L^3 m / (b h^3)$, with $m = P / \delta$.
$\delta$ is the width-averaged mid-thickness deflection at mid-span minus
its mean over the two support lines, which removes the local indentation
at the supports. $K_s$ is the laminate transverse shear stiffness per unit
width. Equilibrium gives the transverse shear stresses of the
lamination-theory bending stresses of a narrow beam ($N = 0$,
$M_y = M_{xy} = 0$), per unit shear force $V$:

$$
\mathbf t(z) = \begin{pmatrix} \tau_{yz} \\ \tau_{xz} \end{pmatrix} \Big/ V
= -\int_{-h/2}^{z} \begin{pmatrix} s_{xy} \\ s_{x} \end{pmatrix} \mathrm dz',
\qquad
\mathbf s(z) = \bar{\mathbf Q}(z)\,(\boldsymbol\varepsilon^0 + z\,\boldsymbol\kappa),
$$

with $(\boldsymbol\varepsilon^0, \boldsymbol\kappa)$ the mid-plane strains
and curvatures per unit $M_x$ from the inverse $ABD$ matrix. Equating
their complementary energy to $V^2 / (2 K_s)$ gives

$$
\frac{1}{K_s} = \int_{-h/2}^{h/2} \mathbf t^\mathsf T\, \mathbf S(z)\, \mathbf t\, \mathrm dz,
$$

with $\mathbf S$ the inverse of each layer's transverse shear stiffness
block. The layer stiffnesses are those of the model: the element layers'
porosity-degraded, rotated $\mathbf C$, averaged over the span. For one
orthotropic material $K_s = \tfrac56 G_{13} h$. At 32:1 transverse shear is
1.6 % (QI) to 3.6 % (UD) of the T800 deflection, so the correction is small
and its uncertainty smaller still. The D7264 and D790 moduli omit it and
read lower by that fraction. The model compares with the
classical-lamination-theory modulus
$E_f = 12 / (h^3 d_{11})$, $d = D^{-1}$
({func}`~porosity_fe.compute_degraded_clt_flexural_modulus`). On the
default D7264 meshes of the 24-ply T800 preset (`'hex8i'`, uniform
spherical porosity):

| Laminate | $V_p$ | $E_f / E_f^\text{CLT}$, 3-point | 4-point | knockdown | $E_f / E_{f,0}$ | CLT $D_{11}^*$ ratio |
|---|---|---|---|---|---|---|
| UD | 0 | 1.0015 | 1.0010 | 1 | 1 | 1 |
| UD | 0.05 | 1.0015 | 1.0010 | 0.9959 | 0.9993 | 0.9993 |
| QI | 0 | 0.9888 | 0.9884 | 1 | 1 | 1 |
| QI | 0.05 | 0.9876 | 0.9872 | 0.9869 | 0.9882 | 0.9894 |
| $[0/90/0/90]_{s3}$ | 0 | 1.0012 | 1.0008 | 1 | 1 | 1 |

The knockdown is a compliance ratio, so it also carries the shear
deflection, which degrades faster with porosity than the bending stiffness
($K_s$ drops about 8.5 % at $V_p = 0.05$); it sits a little below the bending
stiffness ratio, which the modulus ratio $E_f / E_{f,0}$ follows.

The QI shortfall is the free edge, not the mesh or the shear correction.
Near the free edges of the $\pm 45°$ plies the in-plane stresses relax
and interlaminar stresses appear, over a band about one thickness wide,
so the deficit shrinks with the width-to-thickness ratio $b/h$: QI
3-point $E_f / E_f^\text{CLT}$ is 0.963 at $b = h$, 0.989 at the D7264
13 mm ($b \approx 3h$) and 0.996 at $b \approx 6h$, while the cross-ply
laminate stays at 1.001 at every width. Refining the mesh to
$102 \times 8 \times 24$ moves the QI value by 0.2 %.

**Checks.** The UD three-point mid-span deflection matches Timoshenko
beam theory ($E_{11}$, $G_{13}$, $\kappa = 5/6$) to 0.2 % at 32:1
and 0.7 % at 16:1 (one element layer per ply), where shear is 13 % of the deflection. In shorter
beams the FE deflection falls further below it (2.6 % at 8:1), because
the load and supports restrain the warping of the cross-sections, so the
shear-corrected modulus reads slightly high there (0.6 % for the
cross-ply at 16:1). Prefer 32:1 for moduli. In the four-point inner span
the section shear force is zero (below $10^{-9}$ of $P/2$) and the
bending stress is uniform along $x$ to 0.1 %. The element-level
$\tau_{xz}$ there is not zero in a QI laminate: free-edge interlaminar
shear reaches about a third of the shear-span beam value.

**Element formulation.** The plain `'hex8'` brick locks in bending
(see [Element formulations](#element-formulations)), so it reads the
flexural modulus high, increasingly so as the elements lengthen:

| $\Delta x / h$ (mesh) | UD `hex8i` | UD `hex8` | QI `hex8i` | QI `hex8` |
|---|---|---|---|---|
| 0.5 ($78 \times 6 \times 24$) | 1.0011 | 1.0099 | 0.9875 | 1.0087 |
| 0.8 ($48 \times 4 \times 24$, default) | 1.0015 | 1.0235 | 0.9888 | 1.0396 |
| 2.0 ($20 \times 2 \times 24$) | 1.0047 | 1.1387 | 0.9988 | 1.2885 |

(three-point $E_f / E_f^\text{CLT}$; the coarse QI row has only two
elements across the width, which under-resolves the free-edge band.)

**Peak stresses.** Line loads and line supports are singular, so the
stresses right under them grow with refinement. At 32:1 first-ply failure
is governed by the bending stress next to the load line, and the effect
is small: the three-point first-ply-failure load of the QI preset is
585, 582 and 580 N at element lengths of 3.5, 2.2 and 1.1 mm, and 588 to
581 N with the load spread over a 2 mm band (`load_width=2.0`). Shorter
spans concentrate more of the failure index under the load; use a band
there.

**Interior region.** The interior failure quantities of the flexure modes
leave out the contact zones and the side faces: an element is interior
when its centroid lies between the supports, at least `interior_margin`
(default $h/2$) from both side faces and, measured in the $x$-$z$ plane,
at least that far from each support line on the bottom face and from each
load band on the top face
({func}`porosity_fe.fe.flexure.interior_element_mask`). Only a half-disc
around each contact line goes, not the whole cross-section as in the
short-beam region: in three-point bending the bending stress peaks at
mid-span, so the tension face under the load must stay in. On the default
specimen of the 24-ply T800 QI laminate ($V_p = 0.03$) the interior
maximum failure index is 0.993 of the domain maximum in both procedures;
dropping whole cross-sections within $h/2$ of the load would leave 0.949
in three-point bending. The rest is the free-edge band of the QI
laminate (about one thickness wide, see above), where the bottom-ply
stress rises toward the side faces. The margin keeps the edge itself out;
refining across the width resolves more of the band's tail, so the
interior maximum creeps up by about half a percent per level: 0.1241,
0.1257, 0.1263 and 0.1269 at $P = 100$ N, three-point, on
$48 \times 4$, $72 \times 6$, $96 \times 8$ and $120 \times 10$ (all
$\times 24$), and 0.0616, 0.0621, 0.0623 and 0.0625 four-point. A side
margin of $h$ instead of $h/2$ does not converge faster (the value jumps
whenever a column of elements crosses the margin), so the flexure default
matches the short-beam one.

## Short-beam shear (ILSS)

`loading='ilss'` is the short-beam-strength test of ASTM D2344/D2344M,
force controlled by `applied_load`. Its outputs are the stiffness
knockdown, the failure quantities and, from
{func}`~porosity_fe.short_beam_shear`, the D2344 apparent interlaminar
shear next to the FE stresses.

**Specimen.** {meth}`CompositeMesh.for_ilss <porosity_fe.CompositeMesh.for_ilss>`
builds it with the D2344 proportions in laminate thicknesses $h$: support
span $4h$, length $6h$, width $2h$ (`span_to_thickness`,
`length_to_thickness`, `width_to_thickness`; ISO 14130 uses 5, 10 and 5).
It is built on `for_flexure`, so the supports and mid-span are node lines,
and it records them as {attr}`~porosity_fe.CompositeMesh.ilss_geometry`.
Every ply gets its own element layer and the in-plane element length is
$h/4$ by default: $24 \times 8 \times n_\text{plies}$ elements, for the
24-ply T800 preset a $26.4 \times 8.8 \times 4.39$ mm specimen on a 17.6 mm
span (16,875 degrees of freedom; a first solve with the pristine reference
takes about 8.5 s, a repeat 0.6 s, against 17 s and 0.9 s for the former
bend of the $30 \times 10 \times 24$ coupon).

**Supports and loading nose.** The supports (3 mm cylinders in D2344) are
the rollers of `flexure_3pt` ([Flexure](#flexure)): no horizontal reaction
(below $10^{-11}$ N at $P = 10$ N). The loading nose (a 6 mm cylinder) is a
uniform pressure over a band across the top face at mid-span, with
consistent nodal forces. A line load would be singular, with a
logarithmically divergent compliance, so refinement would not converge.
The default band is the Hertz contact width of the nose at the pristine
short-beam strength, the load per unit width
$q = P/b = \tfrac43 \tau_\text{ilss} h$ that D2344 relates to the
strength:

$$
w = 2a = 2\sqrt{\frac{4 q R}{\pi E^*}},
\qquad E^* = \frac{E_{33}}{1 - \nu_{23}^2},
$$

with $R$ the nose radius (`nose_radius`, 3 mm) and a rigid nose. That is
0.77 mm ($0.18h$) for the T800 preset and 0.42 to 0.70 mm for the other
presets. It is an estimate (the laminate is stiffer in plane than
$E_{33}$, which narrows the contact; damage under the nose near failure
widens it); `load_width` sets it explicitly. On the default mesh the band
is narrower than an element, so it loads the nearest node lines. Its width
matters little: $w = 0$, 0.77 and 2.2 mm give knockdowns of 0.9295, 0.9314
and 0.9339 at uniform $V_p = 0.06$.

**Beam theory is approximate at four thicknesses.** D2344 reports the
short-beam strength $F_\text{sbs} = 0.75\,P_\text{max}/(b h)$, the peak of
the parabolic beam-theory shear profile for the shear force $P/2$. The FE
section shear force is $P/2$ (to $10^{-5}$), but the profile has not settled
into the parabola: the width-averaged mid-plane $\tau_{xz}$ half-way
between a support and the nose is below the formula, and approaches it as
the span grows (pristine 24-ply T800, `short_beam_shear(...).midplane_shear`
over `apparent_shear`):

| span / $h$ | UD | QI |
|---|---|---|
| 4 (D2344) | 0.908 | 0.915 |
| 8 | 0.980 | 0.961 |
| 16 | 0.997 | 0.968 |

The QI limit is its lamination-theory profile; the measurement reads the
element means next to the mid-plane, $1 - \tfrac43 (\Delta z/h)^2 = 0.998$
of an exact parabola here. At 4:1 the value is mesh-converged: element
lengths $h/4$, $h/6$ and $h/8$ give 0.908, 0.909 and 0.910 for UD. Porosity
moves it a little (0.908 at uniform $V_p = 0.06$, 0.898 for midplane
clustering, QI). Next to the nose and the supports the contact stresses
dominate.

**Interior region.** The line supports and the nose concentrate stress, so
the domain maximum failure index and first-ply-failure load factor sit in
the contact zones and depend on the mesh. For `ilss` the interior
quantities use the two shear spans instead of the lateral margin of the
axial modes: elements at least `interior_margin` (default $h/2$) from both
support lines, from the edges of the loading band and from the side faces,
with the overhangs left out
({func}`porosity_fe.fe.short_beam.interior_element_mask`). On the QI
preset the interior first ply to fail is a 90° ply near the bottom face
under transverse tension, at the edge of the region nearest the nose, where
the bending moment peaks; the interior load factor moves by a few percent
with the mesh (373, 403 and 389 at uniform $V_p = 0.03$ for element
lengths $h/4$, $h/6$ and $h/8$; see also "Mesh convergence") because the
region's edge falls on different element centroids. With Hashin,
delamination governs on the default mesh.

**Outputs.** {func}`~porosity_fe.short_beam_shear` returns a
{class}`~porosity_fe.ShortBeamShear`: the D2344 formula at the applied load
(`apparent_shear`), the FE mid-plane shear at the quarter-span points
(`midplane_shear`) and its interior Gauss-point peak, the first-ply-failure
loads (domain and interior) and the short-beam strengths D2344 would report
at them ($0.75 \times$ load $/ (b h)$). `effective_modulus` is `None`: at
this span about half the deflection is shear and the contact indentation
is not negligible, so a flexural modulus would mostly measure the shear
correction (use `flexure_3pt` on a `for_flexure` specimen).

**Other meshes raise.** `'ilss'` on a mesh without `ilss_geometry` (the
production coupon, a `for_flexure` specimen) raises a `ValueError` that
says how to build the specimen, before anything is assembled. Snapping the
supports to node lines of the coupon would give a different test: on the
$30 \times 10$ production coupon the nearest node lines make the span
$4.17h$ and the knockdown at uniform $V_p = 0.06$ 0.9359 instead of 0.9314
(6 % of the stiffness loss), and the snapped span would change with the
mesh. To run the test on another mesh, set `mesh.ilss_geometry` to a
{class}`~porosity_fe.FlexureGeometry` with the supports on node lines. The
app, the examples and {func}`~porosity_fe.refinement_study` build the
specimen themselves.

**Result change.** T800 QI, the four production cases, from the former
pinned bend of the $30 \times 10 \times 24$ coupon to the D2344 specimen:

| Case | knockdown | first-ply-failure load factor (interior), $P = 10$ N | apparent $F_\text{sbs}$ at interior first-ply failure (MPa) |
|---|---|---|---|
| uniform $V_p = 0.03$ | 0.9812 → 0.9654 | 174 (383) → 248 (373) | 32.7 → 72.6 |
| uniform $V_p = 0.06$ | 0.9625 → 0.9314 | 169 (376) → 243 (370) | 32.1 → 72.0 |
| midplane clustered 0.03 | 0.9829 → 0.9666 | 178 (386) → 252 (371) | 32.9 → 72.2 |
| interface penny 0.03 | 0.9573 → 0.9202 | 173 (374) → 249 (378) | 31.9 → 73.5 |

The short beam is shear dominated, and porosity degrades the transverse
shear stiffness faster than the fibre-dominated bending stiffness, so the
knockdown drops by 0.016 to 0.037. The old bend failed first at its pinned
edges and load line, far below the shear the specimen carries now; the
peak $\lvert\tau_{13}\rvert$ goes from about $5.9$ to $1.5$ times
$0.75 P / (b h)$.

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

## Mesh convergence

{func}`~porosity_fe.refinement_study` solves one load case on a sequence of
refined meshes and reports which results have converged:

```python
from porosity_fe import MATERIALS, refinement_study

study = refinement_study(MATERIALS["T800_epoxy"], 0.03, "compression",
                         porosity_config={"distribution": "clustered"})
print(study.summary())          # tables and notes; study.to_dict() for JSON
```

The web app's Results tab runs the same study on the analysis mesh
("Mesh convergence check": two or three in-plane levels, $n_z$ fixed).

**What is refined.** By default the levels are $30 \times 10 \times
n_\text{plies}$ refined in plane by the factors 1, 1.5 and 2, with $n_z$
held at one element layer per ply. Three reasons:

- An $n_z$ that does not resolve the plies analyzes a different laminate
  (see the warning under "Mesh and elements"), so a study that crosses
  such values measures ply aliasing, not discretization error. The study
  refuses those levels unless `allow_unresolved_layup=True`.
- The element's bending error is set by $\Delta x / h$, not by $n_z$
  ("Element formulations"), and one `'hex8i'` element per ply is already
  fine through the thickness: on the production geometry below, doubling
  $n_z$ to 48 at $30 \times 10$ moves the compression knockdown by
  $4 \times 10^{-5}$ and the interior failure index by 0.3 %.
- Cost. The sparse LU fill grows as about $8\, n_\text{dof}^{1.5}$, so
  every refined axis is expensive.

`refine='all'` also scales $n_z$, rounded to whole multiples of the ply
count, and `resolutions=` takes explicit levels. For `ilss` every level
is an ASTM D2344 specimen (`CompositeMesh.for_ilss`, default base
$24 \times 8 \times n_\text{plies}$) whose $n_x$ is a multiple of 6, so
the supports and mid-span stay on node lines; the interior margin is the
short-beam default $h/2$. For `flexure_3pt` and `flexure_4pt` every level
is an ASTM D7264 specimen (`CompositeMesh.for_flexure` with the level's
`nx`, default base the specimen's own mesh, $48 \times 4 \times 24$ for the
24-ply presets) whose $n_x$ is even; the span is the same on every level,
only the unloaded overhangs can change length by part of an element, and
the interior margin is again $h/2$.
The interior quantities use one `interior_margin` for every level (by
default the coarsest level's $\max(h, 2\Delta x, 2\Delta y)$); the
solver's own default shrinks with the elements, which would let the
interior region creep into the edge singularity.

**Metrics.** Each quantity gets the relative change from level to level,
measured against the finer value; for the knockdown it is measured
against the stiffness loss $1 - KD$, since $KD$ close to 1 would make any
mesh look converged. The three finest levels, with values $f_3, f_2, f_1$
(coarse to fine) at element sizes $h_3 > h_2 > h_1$ (the geometric mean
element length over the refined axes), are classified:

| Behavior | Condition | Error estimate of $f_1$ |
|---|---|---|
| mesh-independent | both changes below `tol / 10` | 0 |
| monotone | changes of one sign, $p > 0$ below | Roache's GCI |
| non-monotone | the changes alternate in sign | larger of the last two changes |
| diverging | changes of one sign that do not shrink ($p \le 0$) | none (unbounded) |

The observed order $p$ solves

$$
\frac{f_3 - f_2}{f_2 - f_1} = \frac{r_{32}^{\,p} - 1}{1 - r_{21}^{-p}},
\qquad r_{32} = \frac{h_3}{h_2},\; r_{21} = \frac{h_2}{h_1},
$$

which reduces to $p = \ln\!\big[(f_3 - f_2)/(f_2 - f_1)\big] / \ln r$ for
equal ratios. The right side grows with $p$ from $\ln r_{32} / \ln r_{21}$,
so a smaller ratio of changes means the changes do not shrink faster than
$\ln h$: diverging. For a monotone sequence the Richardson extrapolate and
the grid convergence index of the finest level are

$$
f_\text{ext} = f_1 + \frac{f_1 - f_2}{r_{21}^{\,p} - 1},
\qquad
\mathrm{GCI} = \frac{1.25\, \lvert (f_1 - f_2)/f_1 \rvert}{r_{21}^{\,p} - 1}.
$$

The extrapolate is reported only for $p \ge 0.5$; below that the
correction is too large to trust. A quantity is converged when its error
estimate is at most `tol` (default 1 %). It is `'mesh-dependent'` when it
is not converged, its last change exceeds `tol`, and it diverges or
converges with $p < 0.5$: it keeps moving at a rate no practical
refinement settles, which is what a stress singularity looks like.
Three levels cannot always tell a singularity from meshes still too
coarse for the asymptotic trend; when the interior counterpart converges,
the singular reading is the right one.

**What converges and what does not.** T800/epoxy QI (24 plies,
$h = 4.39$ mm), midplane-clustered $V_p = 0.03$, `'hex8i'`, Tsai-Wu,
default study ($n_z = 24$, interior margin 4.39 mm; wall times on a 4-core
Linux machine). Compression at
$\varepsilon = -1$ %:

| Quantity | $30 \times 10$ | $45 \times 15$ | $60 \times 20$ | Study |
|---|---|---|---|---|
| knockdown | 0.992437 | 0.992410 | 0.992393 | changes 0.36 %, 0.22 % of $1 - KD$, $p = 0.38$: not converged at 1 % ($KD \pm 1.8 \times 10^{-4}$) |
| effective modulus (MPa) | 60,803 | 60,765 | 60,738 | changes below 0.1 %: converged |
| maximum failure index | 1.229 | 1.388 | 1.739 | +11 %, +20 %: diverging, mesh-dependent |
| interior maximum failure index | 1.195 | 1.145 | 1.133 | $p = 3.2$, extrapolate 1.126, GCI 0.8 %: converged |
| first-ply-failure load factor | 0.873 | 0.836 | 0.731 | diverging, mesh-dependent |
| interior first-ply-failure load factor | 0.890 | 0.916 | 0.922 | $p = 3.1$, extrapolate 0.926, GCI 0.6 %: converged |
| wall time (s) | 15 | 63 | 252 | |
| peak memory, estimated / measured (GB) | 0.76 / 0.76 | 1.9 / 1.5 | 3.9 / 3.4 | |

The domain maximum sits at the gripped corners, where the stress is
singular; it grows faster with each refinement and has no mesh-converged
value. The interior values converge from above, so a coarse mesh is
conservative for them here. The knockdown and modulus barely move: they are
integrals of the field.

ILSS at $P = -10$ N, on the D2344 specimen
([Short-beam shear](#short-beam-shear-ilss); default base
$24 \times 8 \times 24$, interior margin $h/2 = 2.2$ mm around the supports
and the loading band and from the side faces):

| Quantity | $24 \times 8$ | $36 \times 12$ | $48 \times 16$ | Study |
|---|---|---|---|---|
| knockdown | 0.96657 | 0.96720 | 0.96750 | changes −1.9 %, −0.92 % of $1 - KD$, $p = 1.1$, extrapolate 0.96828, GCI 3 %: not converged at 1 % ($KD \pm 0.001$) |
| maximum failure index ($10^{-3}$) | 2.588 | 2.450 | 2.885 | non-monotone, not converged |
| interior maximum failure index ($10^{-3}$) | 1.652 | 1.486 | 1.551 | non-monotone, 4.2 %: not converged |
| first-ply-failure load factor | 251.8 | 272.6 | 229.5 | non-monotone, not converged |
| interior first-ply-failure load factor | 371.2 | 401.8 | 387.3 | non-monotone, 3.7 %: not converged |
| wall time (s) | 8.6 | 41 | 123 | |

The loading band takes the load-line singularity out of the problem, but
the supports stay line supports, whose indentation compliance converges
slowly, so the knockdown still moves by about $3 \times 10^{-4}$ per level
(0.0017 between the default mesh and the extrapolate, 5 % of the stiffness
loss). The domain extremes sit under the nose and at the supports. The
interior region no longer contains a singularity, but its governing
element is a 90° ply near the bottom face at the edge of the region next
to the nose, where the bending moment peaks; which centroid falls just
outside the margin changes with the mesh, so the interior values move by
about 4 % without a trend. Read them as values to about ±5 %. (On the
former pinned bend of the coupon the interior first-ply-failure load
factor fell from 385 to 323 over the same kind of study, diverging.)

Three- and four-point flexure at $P = -10$ N, on the D7264 specimen
([Flexure](#flexure); default base $48 \times 4 \times 24$, interior margin
$h/2 = 2.2$ mm around the support and load lines and from the side faces;
about two minutes per procedure):

| Quantity | $48 \times 4$ | $72 \times 6$ | $96 \times 8$ | Study |
|---|---|---|---|---|
| 3-point flexural modulus (MPa) | 65,157 | 65,075 | 65,042 | changes below 0.13 %: converged |
| 3-point maximum failure index ($10^{-3}$) | 11.645 | 11.718 | 11.769 | $p \approx 0$: not converged |
| 3-point interior maximum failure index ($10^{-3}$) | 11.564 | 11.718 | 11.769 | $p = 2.1$, extrapolate 11.831, GCI 0.7 %: converged |
| 3-point interior first-ply-failure load factor | 58.76 | 58.32 | 58.09 | $p = 0.8$, GCI 1.9 %: not converged |
| 4-point flexural modulus (MPa) | 65,132 | 65,055 | 65,024 | changes below 0.12 %: converged |
| 4-point interior maximum failure index ($10^{-3}$) | 5.929 | 5.989 | 6.003 | $p = 3.2$, extrapolate 6.012, GCI 0.2 %: converged |
| 4-point interior first-ply-failure load factor | 113.94 | 113.70 | 113.57 | $p = 0.85$, GCI 0.5 %: converged |

The knockdown moves by about $2 \times 10^{-5}$ in both procedures (0.992039,
0.992032, 0.992016 three-point), 0.2 % of the stiffness loss and below the
tolerance; the study still reports it as not converged because its
second change is larger than its first. Beyond the coarsest level the
domain and interior maxima coincide: the peak is on the tension face
near mid-span (three-point) or between the load lines (four-point),
inside the region, and the domain maximum on the coarsest mesh is the
element beside the side face. The three-point interior load factor still
drifts by about 0.4 % per level; read it to about ±2 %.

Thermal residual stress (AS4/3501-6 QI, $h = 3$ mm,
$\Delta T = -157$ K, the compression meshes): the domain maximum failure index grows
at the free edges (0.735, 0.776, 0.801: mesh-dependent), and the interior
one is non-monotone (0.685, 0.706, 0.669: not converged). These meshes
have elements 13 to 7 ply thicknesses long, where `'hex8i'` carries an
oscillation in from the free edges ({doc}`thermal`, "Caveats").

**Reading the report.** `summary()` prints the levels, then one row per
quantity with the last change, behavior, observed order, extrapolate, GCI
and verdict, then notes: which domain extremes to replace by their
interior counterparts, the knockdown error in $KD$ itself, and that with
`refine='in_plane'` the extrapolate is the limit at the fixed $n_z$.
Every field is also on {class}`~porosity_fe.RefinementStudy`,
{class}`~porosity_fe.RefinementLevel` and
{class}`~porosity_fe.QuantityConvergence`.

**Production resolution.** The production default is
$30 \times 10 \times n_\text{plies}$, one element layer per ply ($n_z$
capped at 48). It was $30 \times 10 \times 12$, which does not resolve the
24-ply presets: in-plane refinement at $n_z = 12$ converges the compression
interior failure index (1.477, 1.472, 1.472), but to the value of a
different laminate:

| T800 QI, $V_p = 0.03$ clustered | $30 \times 10 \times 12$ | $30 \times 10 \times 24$ | $60 \times 20 \times 24$ (extrapolate) |
|---|---|---|---|
| compression knockdown | 0.99218 | 0.99244 | 0.99239 |
| compression interior maximum failure index | 1.477 | 1.195 | 1.133 (1.126) |
| compression maximum failure index | 2.13 | 1.23 | 1.74 (mesh-dependent) |
| first compression solve, incl. pristine reference | 4.7 s | 14.5 s | 252 s |

One element layer per ply at the production in-plane size puts the
knockdown within $2 \times 10^{-4}$ of the finest mesh and the interior
failure index 6 % above its extrapolate, at three times the cost of
$n_z = 12$; $45 \times 15 \times n_\text{plies}$ brings the interior index
within 2 % (one minute, 1.5 GB). Read failure from the interior
quantities: the domain maximum is mesh-dependent at every resolution.
(`ilss` does not use this mesh: it solves the short-beam specimen, whose
default resolution is discussed above.)

**Cost and guards.** Levels are solved coarse to fine and each is released
before the next is built. Each level's pristine reference is solved before
its porous model is assembled, so only one factorization is alive at a
time (at $45 \times 15 \times 24$ the peak is 1.5 GB instead of 2.9 GB).
Before solving, the study refuses a level above `max_elements` (default
the $10^6$ mesh cap) or one whose estimated peak memory exceeds
`max_memory_gb` (default 80 % of physical memory where the platform
reports it), and the error lists the estimate of every level. The
estimate is $0.15\ \text{GB} + 30\ \text{kB}$ per element plus 12 bytes
per LU nonzero; it was 1.0 to 1.3 times the measured peak on meshes from
$20 \times 8 \times 24$ to $60 \times 20 \times 24$. With `solver='cg'`
the compression study above gives the same values to the printed digits
in 193 s instead of 329 s, at the same 3.4 GB peak: the knockdown's
pristine reference is always factorized directly, so CG saves memory only
in thermal or pore-free studies.
