# Classical lamination theory

The laminate-level moduli used by the validation suite (tensile, shear and
flexural modulus against porosity) come from classical lamination theory
(CLT) applied to the degraded ply stiffness of {doc}`micromechanics`.
Implementation: {func}`~porosity_fe.compute_clt_effective_modulus`
(pristine), {func}`~porosity_fe.compute_degraded_clt_moduli` and
{func}`~porosity_fe.compute_degraded_clt_flexural_modulus`.

## Reduced stiffness

Each ply's 6×6 stiffness $\mathbf C$ is rotated about $z$ by its ply angle
$\theta$ ({func}`~porosity_fe.rotate_stiffness_3d`). The plane-stress
reduced stiffness is then obtained by condensing out $\sigma_{33} = 0$:

$$
\bar Q_{ij} = \bar C_{ij} - \frac{\bar C_{i3}\,\bar C_{j3}}{\bar C_{33}},
\qquad i, j \in \{1, 2, 6\},
$$

where $\bar{\mathbf C}$ is the rotated stiffness and index 6 is the
in-plane shear (Voigt slot 12).

## A and D matrices

Plies of thickness $t$ are stacked from $z = -h/2$ to $z = h/2$ in the
order given, with $h = n\,t$. With $z_k$ the mid-plane of ply $k$:

$$
\mathbf A = \sum_k \bar{\mathbf Q}_k\, t,
\qquad
\mathbf D = \sum_k \bar{\mathbf Q}_k \left(t\, z_k^2 + \frac{t^3}{12}\right).
$$

The coupling matrix $\mathbf B$ is not formed, so the moduli below assume
a symmetric layup.

## Effective moduli

With $\mathbf a = \mathbf A^{-1}$ and $\mathbf d = \mathbf D^{-1}$:

$$
E_x = \frac{1}{h\, a_{11}}, \qquad
E_y = \frac{1}{h\, a_{22}}, \qquad
G_{xy} = \frac{1}{h\, a_{66}}, \qquad
E^{f}_x = \frac{12}{h^3\, d_{11}} .
$$

The degraded variants evaluate the ply stiffness at the specimen-average
$V_p$ with spherical voids; the `method` argument is accepted for API
symmetry and ignored.
