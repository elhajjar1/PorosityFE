# Theory

These pages state the models PorosityFE implements, with the equations as
they appear in the code and the places where the code makes a modelling
choice that a user should know about. Each page names the function or
class that implements it.

PorosityFE answers one question two ways:

- The **empirical path** ({class}`~porosity_fe.EmpiricalSolver`) applies a
  calibrated closed-form knockdown to the pristine strength, once, at the
  specimen-average porosity. The layup enters only the fiber-direction
  modes, through a CLT strain-energy blend of the calibrated coefficients.
  The porosity distribution shape does not change the result.
- The **finite-element path** ({class}`~porosity_fe.FESolver`) degrades the
  stiffness of every element from its local porosity with Mori-Tanaka
  micromechanics, solves a linear static problem, and evaluates a failure
  criterion with porosity-degraded strengths. The distribution shape does
  change the result.

The two paths are independent implementations of the same physical claim
and are not expected to agree number for number. The README section
"Solver selection: FE vs. empirical" says which to use for which question.

Conventions used on every page:

- $V_p$ is the void volume fraction, a number in $[0, 1]$ (2 % porosity is
  $V_p = 0.02$).
- Stiffnesses and strengths are in MPa, lengths in mm.
- Voigt order is $[11, 22, 33, 23, 13, 12]$; the last three strain
  components are engineering shear strains $\gamma_{ij} = 2\varepsilon_{ij}$.
- The laminate axes are $x$ (fiber direction of a 0° ply, the loading
  direction), $y$ (in-plane transverse) and $z$ (through the thickness).

```{toctree}
:maxdepth: 1

porosity
micromechanics
laminate
empirical
fe
failure
fatigue_environment
uncertainty
```
