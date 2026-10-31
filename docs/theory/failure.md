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
| $Z_t$ (optional) | `sigma_3t` | $r_m$ |
| $S_{13}$ (optional) | `tau_13` | $r_m$ |

Each strength is floored at $10^{-3}$ MPa.

The through-thickness tensile strength $Z_t$ (`MaterialProperties.sigma_3t`)
and the interlaminar shear strength $S_{13}$ (`tau_13`) are optional and
`None` by default; no preset sets them. When they are `None`, every
criterion keeps the stand-in it always used: $Y_t$ for $Z_t$, and for
$S_{13}$ either $S_{12}$ or $S_{23}$ depending on the criterion (see
[Strengths and their fallbacks](#strengths-and-their-fallbacks)). When they
are set, every criterion uses them for $\sigma_3$ in tension and for
$\tau_{13}$, degraded by the same $r_m$ as the other matrix strengths.
There is no through-thickness compressive strength input: $Y_c$ stays the
allowable of $\sigma_3$ in compression.

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

When `sigma_3t` is set, $Z_t$ replaces $Y_t$ in the direction-3 terms:
$F_3 = 1/Z_t - 1/Y_c$, $F_{33} = 1/(Z_t Y_c)$, and through them
$F_{13} = F^*_{12}\sqrt{F_{11}F_{33}}$ and
$F_{23} = -\tfrac12\sqrt{F_{22}F_{33}}$. When `tau_13` is set,
$F_{55} = 1/S_{13}^2$.

where $F^*_{12}$ is `MaterialProperties.tsai_wu_F12`. By default it is
$-0.5$, Tsai's recommendation (Tsai & Wu 1971). It must lie in $[-1, 0]$
for the envelope to stay closed. The default is an empirical choice; when
comparing with another code, check its $F_{12}$ convention, and calibrate
$F^*_{12}$ from biaxial tests when they exist. Tsai-Wu does not separate
failure modes, so the per-mode indices are reported as `NaN` (`null` in
the JSON export).

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
FI_\text{delam} = \left(\frac{\langle \sigma_3 \rangle}{Z_t}\right)^2
 + \left(\frac{\tau_{13}}{S_{13}}\right)^2 + \left(\frac{\tau_{23}}{S_{23}}\right)^2,
\qquad \langle \sigma_3 \rangle = \max(\sigma_3, 0),
$$

with $Z_t = Y_t$ and $S_{13} = S_{23}$ unless `sigma_3t` and `tau_13` are
set, so by default
$FI_\text{delam} = (\langle\sigma_3\rangle/Y_t)^2 + (\tau_{13}^2 + \tau_{23}^2)/S_{23}^2$.

The reported index is the maximum over the five modes.

```{note}
The delamination mode is a stress-based *initiation* screen and nothing
more. It is evaluated at the ply Gauss points, which sit about
$0.21\,t_\text{ply}$ inside each ply with one element per ply, not on the
interfaces, and without the averaging distance of the original criterion.
It knows nothing of an explicit delamination ({doc}`delamination`), whose
growth is judged by energy release rates instead. Unless `sigma_3t` and
`tau_13` are set, $Y_t$ (from $\sigma_{2t}$) stands in for $Z_t$, and both
interlaminar shears are checked against the apparent short-beam strength
$\tau_\text{ilss}$. All are degraded by the same in-plane $r_m$ as the
other matrix strengths, whatever the void shape. The interface criteria of
[Interlaminar stresses and delamination onset](#interlaminar-stresses-and-delamination-onset)
evaluate the same quadratic form on the interface tractions, averaged.
```

## Reported modes

`FieldResults.failure_mode_indices` is the breakdown of the modes **at the
Gauss point of the overall maximum**. `FieldResults.failure_mode_maxima`
is each mode's own maximum over the model, so a mode that does not govern,
such as delamination onset in a compression solve, is still visible. The
VTK and VTU exports write each mode's per-element maximum as
`failure_index_<mode>` next to `failure_index`, and the JSON export writes
`failure.mode_maxima`. Both are `None` (`null`) for Tsai-Wu. The web app's
Results tab shows the governing mode and, for Hashin and maximum stress,
the per-mode maxima.

## Maximum stress

Each component is compared with its allowable: $\sigma_1$ against $X_t$ or
$X_c$; the worse of $\sigma_2$ and $\sigma_3$ against $Y_t$ or $Y_c$ ($Z_t$
for $\sigma_3$ in tension when `sigma_3t` is set);
$\lvert\tau_{12}\rvert$ against $S_{12}$, $\lvert\tau_{13}\rvert$ against
$S_{12}$ (or $S_{13}$ when `tau_13` is set); and
$\lvert\tau_{23}\rvert$ against $S_{23}$. The index is the largest ratio.
The interlaminar components fall into the matrix and shear modes, so the
`delamination` entry is always 0.

## First-ply-failure load factor

The analysis is linear, so scaling the applied load by $\lambda$ scales
every stress by $\lambda$. At each Gauss point the failure index becomes a
polynomial in $\lambda$: linear for maximum stress, quadratic for Tsai-Wu
and the Hashin modes. `first_ply_failure_load_factor` is the smallest
positive $\lambda$ at which any checked point reaches an index of 1. The
margin of safety is $\lambda - 1$.

## Interlaminar stresses and delamination onset

The criteria above check the ply stresses at the Gauss points, which sit
inside the plies. Delamination starts at a ply interface, so
{func}`~porosity_fe.interface_tractions` recovers the stress vector on the
interfaces themselves, and {func}`~porosity_fe.delamination_onset` checks it
with the stress-based onset criteria of the literature. Both are
post-processing of a solve: nothing {meth}`FESolver.solve
<porosity_fe.FESolver.solve>` computes changes. Implementation:
`porosity_fe.fe.interlaminar`. An explicit delamination, once present, is
judged by its energy release rates instead ({doc}`delamination`).

```python
from porosity_fe import (MATERIALS, CompositeMesh, FESolver, PorosityField,
                         delamination_onset)

mat = MATERIALS['T800_epoxy']
pf = PorosityField(mat, 0.02)
mesh = CompositeMesh(pf, mat, nx=30, ny=10, nz=24)
result = FESolver(mesh, mat, pf).solve('tension')

tr = result.interface_tractions(mesh)      # per interface node
tr.traction, tr.jump                       # (tau_xz, tau_yz, sigma_zz), MPa
onset = delamination_onset(result, mesh, mat)
onset.max_index, onset.critical_interface, onset.critical_location
onset.onset_load_factor                    # onset at this multiple of the load
onset.interior_onset_load_factor           # away from the free edges
onset.grid(onset.index, onset.critical_interface)   # (ny+1, nx+1) map
```

### Interface tractions

Interface $k$ lies between ply $k$ and ply $k + 1$ from the bottom face. It
is a node layer of the mesh when $(k+1)\,n_z/n_\text{plies}$ is an integer:
every interface is when $n_z$ is a multiple of $n_\text{plies}$, the
production resolution. At each such interface the traction is

$$
\mathbf t = \boldsymbol\sigma \cdot \mathbf e_z = (\tau_{xz},\ \tau_{yz},\ \sigma_{zz})
$$

in the laminate frame. It is the stress vector acting on the interface
plane, the same physical vector whichever ply it is computed from, so its
two sides can be compared directly.

Each side is recovered from its own element layer. The eight Gauss-point
stresses of an element are extrapolated to its corners with the trilinear
extrapolation of {func}`~porosity_fe.extrapolate_to_nodes`,
$\mathbf v_n = \mathbf N_{gp}^{-1}\mathbf v_{gp}$. On the face the element
shares with the interface, that gives a bilinear traction field. At each
interface node the face values of each side are averaged, giving
$\mathbf t^-$ from below and $\mathbf t^+$ from above. The result reports
both, their average $\bar{\mathbf t} = (\mathbf t^- + \mathbf t^+)/2$ and the
jump $\mathbf t^+ - \mathbf t^-$.

The exact traction is continuous across a bonded interface (equilibrium),
so the jump measures how well the mesh resolves the interlaminar stress.
On a 1 mm $[0/90]_s$ T800 laminate in tension, the RMS jump at the $0/90$
interface, 0.5 to 2 mm in from the free edge, falls from 90 % to 58 % to
31 % of the RMS traction as the mesh goes from 10 to 20 to 40 elements
across the 5 mm width and from 1 to 2 to 4 element layers per ply
(`tests/test_interlaminar.py`). At the free edge itself the stress is
singular: the jump does not vanish there, and the edge value grows under
refinement.

The recovered tractions satisfy equilibrium with lamination theory. On the
$[45/-45]_s$ laminate in tension, $\tau_{xz}$ at the $45/-45$ interface is
$-\partial N_{xy}/\partial y$ of the $+45$ ply, so its integral from
mid-width to the free edge equals the ply's in-plane shear force of
lamination theory. The finest of the three meshes above is 12 % low, the
error halves with each refinement (first-order convergence near the
singularity), and the Richardson extrapolation of the two finest meshes is
within 0.3 %. The same holds for $\tau_{yz}$ and the transverse force
$N_{yy}$ of the $0°$ ply of the $[0/90]_s$ laminate.

On a delaminated interface only the faces whose four corner pairs are
bonded are used. The crack faces are traction free, or carry a contact
pressure (in {attr}`FieldResults.interface_forces
<porosity_fe.FieldResults.interface_forces>`), and the extrapolated stress
next to the front is not an interface traction. Nodes inside a delamination
are therefore reported as not valid (`NaN`), and so are nodes whose faces
all touch a void element or an element the criteria skip (mean porosity
above 0.95).

### Averaging

At a free edge the interlaminar stress is singular, so a criterion on the
point stress depends on the mesh. Brewer & Lagace (1988) evaluated their
criterion on stresses averaged over a characteristic distance from the
edge, $x_\text{avg}$; the default here is one ply thickness, the averaging
distance usually attributed to Kim & Soni (1984). It is a calibration
parameter: the averaged edge stress, and with it the predicted onset load,
depends on it.

`delamination_onset` averages the two-sided traction over a square in-plane
window of side $x_\text{avg}$. The window is centred on each node and
shifted inward where it would cross a plate edge, so at a straight free
edge it spans exactly $x_\text{avg}$ from the edge, Brewer & Lagace's
averaging distance. The average is the exact integral of the bilinear face
fields over the part of the window on valid faces. `x_avg=0` uses the nodal
traction without averaging. The averaged stress converges under refinement
only once the elements are smaller than $x_\text{avg}$.

### Onset criteria

`criterion='brewer_lagace'` (default) is the quadratic criterion of Brewer &
Lagace (1988) on the averaged traction, in the laminate frame:

$$
FI = \left(\frac{\langle\bar\sigma_{zz}\rangle}{Z_t}\right)^2
 \left[+ \left(\frac{\langle-\bar\sigma_{zz}\rangle}{Z_c}\right)^2\right]
 + \left(\frac{\bar\tau_{xz}}{S_{13}}\right)^2 + \left(\frac{\bar\tau_{yz}}{S_{23}}\right)^2 ,
$$

with $\langle\cdot\rangle$ the positive part and onset at $FI = 1$. The
compression term is included only when `Zc` (an interlaminar compressive
strength, MPa) is passed; by default through-thickness compression does not
drive onset.

`criterion='ye'` is the quadratic criterion usually attributed to Ye (1988),
$(\langle\sigma_{33}\rangle/Z_t)^2 + (\tau_{13}/S_{13})^2 + (\tau_{23}/S_{23})^2$,
without a compression term. PorosityFE evaluates it with the averaged
shears rotated into the material axes of each of the two plies at the
interface and takes the larger index. With the default strengths
($S_{13} = S_{23} = \tau_\text{ilss}$) the shear term does not depend on the
in-plane axes, and the two criteria coincide.

```{note}
We did not re-read Brewer & Lagace (1988), Kim & Soni (1984) or Ye (1988)
for this implementation. The forms above are the ones these papers are
cited for, and the evaluation of Ye's criterion in the ply axes, the window
and its edge rule are PorosityFE's choices.
```

### Strengths and their fallbacks

The interface criteria use $Z_t$ = `sigma_3t`, $S_{13}$ = `tau_13` and
$S_{23}$ = `tau_ilss`, degraded by the matrix ratio $r_m$ of
[Degraded strengths](#degraded-strengths) at the porosity of the interface
node (the local $V_p$ at the interface). When the optional strengths are
`None` each criterion falls back as before:

| Criterion | $\sigma_{33}$ in tension | $\tau_{13}$ | $\tau_{23}$ |
|---|---|---|---|
| Tsai-Wu | `sigma_3t`, else `sigma_2t` | `tau_13`, else `tau_12` | `tau_ilss` |
| Maximum stress | `sigma_3t`, else `sigma_2t` | `tau_13`, else `tau_12` | `tau_ilss` |
| Hashin delamination mode | `sigma_3t`, else `sigma_2t` | `tau_13`, else `tau_ilss` | `tau_ilss` |
| Interface (Brewer-Lagace, Ye) | `sigma_3t`, else `sigma_2t` | `tau_13`, else `tau_ilss` | `tau_ilss` |

The defaults are inconsistent: Tsai-Wu and maximum stress check $\tau_{13}$
against the in-plane shear strength $\tau_{12}$, while Hashin and the
interface criteria check it against the short-beam strength
$\tau_\text{ilss}$. They are kept so that no existing result changes;
setting `tau_13` makes every criterion use the same value. In the
Brewer-Lagace criterion $\bar\tau_{xz}$ meets $S_{13}$ and $\bar\tau_{yz}$
meets $S_{23}$, with $x$ the $0°$ direction of the laminate; with the
defaults both are $\tau_\text{ilss}$.

Every interlaminar strength is degraded by the same $r_m$, the square root
of the in-plane matrix stiffness ratio, whatever the void shape. A
direction-aware rule (the $C_{33}$ ratio for $Z_t$, $C_{55}$ for $S_{13}$)
would make flat voids weaken the interface more; it changes existing
results and is a pending decision (delamination plan, item 1.4).

### Onset load factor

The analysis is linear, so the averaged traction at a multiple $\lambda$ of
the load is $\bar{\mathbf t}_\text{th} + \lambda\,\bar{\mathbf t}_m$. The
prestress $\bar{\mathbf t}_\text{th}$ is the thermal residual traction of a
combined solve (`load_factor_basis='mechanical_with_residual'`, recovered
from {attr}`FieldResults.residual_stress_local
<porosity_fe.FieldResults.residual_stress_local>`), and zero for a
mechanical or a thermal-only solve. The onset index is then quadratic in
$\lambda$:

- without a prestress the normal term keeps its branch for $\lambda > 0$,
  so $\lambda = 1/\sqrt{FI}$;
- with one, $\sigma_{zz}$ can change sign as $\lambda$ grows, so each branch
  of the normal term is solved on the interval of $\lambda$ where its sign
  holds, with the rationalized root of `_positive_root_offset`, as for the
  Hashin load factor above. The factor is 0 where the prestress alone has
  reached onset.

`onset_load_factor` is the smallest factor over the evaluated nodes and
`critical_location` the node of the largest index. The interior variant
(`interior_max_index`, `interior_onset_load_factor`) leaves out the nodes
closer than `interior_margin` to a lateral face, by default
$\max(h, 2\ \text{element widths})$ per in-plane direction as for the
interior failure metrics of the in-plane loading modes. For the short-beam
and flexure specimens pass `interior_margin` explicitly: the shear-span
interior of their ply criteria is not applied here.

### Free-edge delamination (O'Brien)

A laminate in tension that delaminates from its free edges loses axial
stiffness in proportion to the delaminated area. From this rule of
mixtures, O'Brien derived a strain energy release rate independent of the
delamination size (his eq. 10),

$$
G = \frac{\varepsilon^2 t}{2}\left(E_\text{LAM} - E^*\right),
\qquad
E^* = \frac{1}{t}\sum_i E_i\, t_i ,
$$

with $\varepsilon$ the nominal strain, $t$ the laminate thickness,
$E_\text{LAM}$ the axial modulus of the intact laminate and $E^*$ that of
the laminate completely delaminated along the chosen interfaces, from the
moduli $E_i$ and thicknesses $t_i$ of the sublaminates it then falls into
(each at the same axial strain, free to take its own transverse strain).
{func}`~porosity_fe.obrien_edge_delamination` computes it from lamination
theory with the pristine ply stiffness:

- By default every modulus is $1/(t\,a_{11})$ with $\mathbf a = \mathbf
  A^{-1}$ (O'Brien's eq. 1). This neglects the bending-extension coupling
  of the unsymmetric sublaminates, as O'Brien did for his predictions.
- `bending_coupling=True` uses the in-plane block of the inverse of the full
  $\mathbf{ABD}$ matrix of each stack about its own mid-plane (his
  eq. B1), which lowers $E^*$ and raises $G$.

Equation (10) assumes one delamination growing from each edge, moving
between the interfaces as O'Brien observed. With $m$ interfaces delaminating
together, each crack front releases $G/m$
(`EdgeDelamination.G_simultaneous`); O'Brien states the $m = 2$ case in his
Appendix A, where he doubles his finite-element values to compare them
with eq. (10). Compare $G$ with a toughness in the same convention. The
rule of mixtures holds once the delamination has grown a short distance
from the edge, below which his finite-element $G$ is smaller; $G$ is the
total, without a mode split.

O'Brien's edge-delamination specimen is $[\pm30/\pm30/90/\overline{90}]_s$:
11 plies, the centre $90°$ ply not repeated, delaminated at both $-30/90$
interfaces (interfaces 3 and 6 from the bottom). With his T300/5208 ply
($E_{11} = 138$ GPa, $E_{22} = 15$ GPa, $G_{12} = 5.9$ GPa,
$\nu_{12} = 0.21$) and 1.51 mm thickness:

| Quantity | O'Brien (NASA TM-81940) | `obrien_edge_delamination` |
|---|---|---|
| $E^*/E_\text{LAM}$, no coupling (eq. 1) | 0.743 | 0.745 |
| $E^*/E_\text{LAM}$, with coupling (eq. B1) | 0.69 | 0.693 |
| $G$ at the measured onset strain $3470\ \mu\varepsilon$ | 137 J/m² ($G_c$) | 135 J/m² |

His measured stiffness loss extrapolated to $E^* = 0.742\,E_0$, which the
uncoupled value matched.

The FE model gives a cross-check. A $3 \times 6$ mm coupon of the specimen
in tension, one element per ply and 0.125 mm elements across the width,
with explicit delaminations 1.5 mm in from the free edge in both $-30/90$
interfaces along the whole length ({doc}`delamination`):

- releases 0.9 % less energy per crack front than $G/2$, the value for two
  interfaces growing together (from the effective modulus at two
  delamination widths, fixed grip);
- has a VCCT $G_T$ at mid-length 5 % below $G/2$, with
  $G_\text{I}/G_T = 0.58$ and $G_\text{III}$ negligible.

```python
# t300: a MaterialProperties with the ply constants above, t_ply = 1.51 / 11
layup = [30, -30, 30, -30, 90, 90, 90, -30, 30, -30, 30]
edt = obrien_edge_delamination(t300, layup, [3, 6], 3470e-6)
edt.E_star / edt.E_LAM, edt.G              # 0.745, 0.135 N/mm
edt.critical_strain(0.137)                 # onset strain for G_c = 0.137 N/mm
```

Source: T. K. O'Brien, "Characterization of delamination onset and growth
in a composite laminate", NASA TM-81940, January 1981 (NTRS 19810009574),
for the equations and numbers above. It was published in *Damage in
Composite Materials*, ASTM STP 775 (1982), which we did not access.

### Limits

- The model is linear with smeared porosity. The onset criteria are
  stress-based screens, not fracture mechanics, and their free-edge values
  depend on $x_\text{avg}$ and, until the elements are smaller than
  $x_\text{avg}$, on the mesh. Check the jump and refine.
- The strength degradation is the heuristic square-root rule of
  [Degraded strengths](#degraded-strengths), the same in every direction.
- `sigma_2t` and `tau_ilss` stand in for the interlaminar strengths unless
  `sigma_3t` and `tau_13` are set, and no preset sets them.
- The interface onset is computed on request, never by the solve: in the
  app with the sidebar option "Interface delamination onset"
  (Brewer-Lagace, its averaging length an input), in the JSON export as a
  `delamination_onset` block (`FESolver.export_results(..., onset=onset)`)
  and in the VTU as the point field `onset_index`
  (`FieldResults.to_vtu(..., onset=onset)`). The default exports do not
  carry it.
- O'Brien's $G$ uses the pristine ply stiffness, the laminate as given (not
  tiled to `n_plies`) and a uniaxial tensile strain.
