# Changelog

All notable changes to PorosityFE will be documented in this file.

## [Unreleased]

### Added
- **Interlaminar stresses and delamination onset at the ply interfaces**
  (delamination plan, phase 1.1-1.3 and 1.5). Post-processing and
  lamination theory only: existing results are unchanged unless the new
  strengths are set.
  - `MaterialProperties.sigma_3t` (through-thickness tensile strength
    `Z_t`) and `tau_13` (`S_13`), in MPa, default `None`, validated
    positive and finite, perturbable when set; no preset sets them. With
    `None` every criterion keeps its stand-in: `sigma_2t` for `Z_t`; for
    `tau_13`, `tau_12` in Tsai-Wu and maximum stress and `tau_ilss` in
    Hashin. When set, Tsai-Wu, Hashin's delamination mode, maximum stress,
    their load factors and the interface criteria use them, degraded by
    the same matrix ratio `r_m` (`fe.failure.degraded_interlaminar_strengths`).
  - `interface_tractions(results, mesh)` (also `FieldResults.interface_tractions`):
    the traction `(tau_xz, tau_yz, sigma_zz)` at every ply interface that
    is a node layer, extrapolated to the shared face from the element
    layer below and the one above with the trilinear extrapolation of
    `extrapolate_to_nodes`. `InterfaceTractions` reports both sides, their
    average and their jump, a resolution indicator; on a delaminated
    interface only bonded faces count.
  - `delamination_onset(results, mesh, material)` (also
    `FieldResults.delamination_onset`): the Brewer-Lagace quadratic
    criterion on the traction averaged over a square window of side
    `x_avg` (default one ply thickness; a calibration parameter), with an
    optional compression term (`Zc`), or Ye's criterion in the ply axes
    (`criterion='ye'`). Strengths are porosity-degraded at the interface
    node. `DelaminationOnset` gives the per-node index and onset load
    factor (closed form, or the quadratic root with a thermal prestress),
    the critical interface and location, an interior variant and a
    JSON-ready `summary()`.
  - `obrien_edge_delamination(material, ply_angles, interfaces, strain)`:
    O'Brien's free-edge delamination energy release rate
    `G = (eps^2 t / 2) (E_LAM - E*)` from lamination theory, with or
    without the bending-extension coupling of the sublaminates
    (`EdgeDelamination`). It reproduces O'Brien's
    `[+-30/+-30/90/90bar]s` specimen (NASA TM-81940): `E*/E_LAM` 0.745
    (0.743) without coupling, 0.693 (0.69) with it, and G = 135 J/m² at
    his onset strain (his G_c = 137 J/m²).
  - The failure theory page has a section "Interlaminar stresses and
    delamination onset": the recovery, the averaging, the criteria, the
    strength fallbacks and their `tau_13` inconsistency, the load factor,
    the free-edge G and the limits. The app and the exports do not carry
    the interface onset yet.
- **Failure modes made visible, and the docs say what delamination
  coverage exists** (delamination plan, phase 0). The default Tsai-Wu
  results, the knockdowns and the validation MAE are unchanged.
  - `FieldResults.failure_mode_maxima`: each separate mode's maximum over
    the model, for Hashin and maximum stress (`None` for Tsai-Wu).
    - `failure_mode_indices` is the breakdown at the Gauss point of the
      overall maximum only, so a mode that does not govern used to be
      invisible: for example, Hashin's delamination-onset mode in a
      compression solve.
    - `FieldResults.per_element_mode_failure_index` holds each mode's
      per-element maximum.
    - `porosity_fe.fe.failure.evaluate_failure` takes an optional
      `mode_fields=` dict to fill. `MODE_KEYS` lists the modes.
  - Exports:
    - VTK/VTU write `failure_index` (the criterion's index) and, for Hashin
      and maximum stress, `failure_index_<mode>` per mode. The legacy
      `tsai_wu_index` stays as an alias, written whatever the criterion.
    - JSON schema 1.3 adds `failure.mode_maxima` and the interlaminar
      components `sigma_33`, `tau_23` and `tau_13` to the `stress_local`
      summary, which carried only the in-plane ones.
  - App:
    - A sidebar "FE failure criterion" selector: Tsai-Wu (default), Hashin
      or maximum stress. It also applies to the convergence check.
    - A Results-tab **FE failure** block: domain and interior maximum
      index, the interior first-ply-failure load factor, the governing
      mode, and each mode's maximum for Hashin and maximum stress.
      Maximum stress has no delamination mode, so its row reads n/a.
  - Docs:
    - The README limitation now says what exists: Hashin's stress-based
      delamination-onset index at the ply Gauss points, and no explicit
      delamination, energy release rate or growth.
    - The README states the 3D Tsai-Wu polynomial the code evaluates,
      instead of the in-plane one labelled "Full 3D".
    - `examples/interface_penny.py` no longer calls interface penny
      porosity the "worst-case ILSS morphology": its empirical knockdowns
      depend only on the mean `Vp`.
    - The README and the porosity theory page note two things about the
      FE knockdown of interface penny porosity at 3 %. Most of its
      reduction is the penny shape: uniform penny voids give 0.9846, and
      interface penny voids 0.9819. And with one element layer per ply
      the `interface` profile's FE volume average is 0.0354, not 0.03.
    - The failure theory page describes the delamination mode's limits and
      the reported mode quantities.
- **Explicit delaminations in the FE model** (delamination plan, phase
  2.1-2.3). Results without a delamination are unchanged bit for bit.
  - `Delamination` (public): a crack at ply interface `k`, between ply `k`
    and ply `k + 1` from the bottom, with a `'strip'` (through the width,
    `x_start` to `x_end`), `'rectangle'` or `'ellipse'` planform (`center`,
    `half_size`, mm). The region is open: nodes on its boundary stay
    bonded and form the crack front; a strip or rectangle edge on the plate
    edge opens to it. Several delaminations at one interface act as their
    union.
  - `CompositeMesh(..., delaminations=...)`, also on `for_flexure` and
    `for_ilss`, duplicates the node plane of every delaminated interface
    at the end of `generate_mesh()`: the copies follow the structured
    nodes, whose ids are unchanged, and the element layer above is rewired
    to them. Needs `nz` a multiple of `n_plies`. New
    `n_structured_nodes`, `delaminated_interfaces`,
    `interface_node_pairs()`, `bonded_pairs()`, `bonded_ties()` and
    `crack_front_pairs()`. Replacing `mesh.delaminations` at the same
    interfaces changes only the ties, so a crack grows without reassembly.
  - `FESolver` ties each bonded duplicate to its original, in every loading
    mode, direct or CG, and in the thermal unit solve; the LU and thermal
    caches are keyed on the tie set. Boundary conditions prescribe the
    original nodes and the delaminated duplicates on a face, never a tied
    duplicate. Reactions fold each tie force onto its original. The
    knockdown's pristine reference has every pair bonded (no porosity,
    voids or delamination), so a delamination alone gives a knockdown
    below 1; its cache key holds that tie set.
  - `extrapolate_to_nodes` averages a bonded pair as one node and reports
    the two crack faces of a delaminated pair separately. The
    flexural-modulus deflection and the structured section helpers read
    the structured nodes.
  - Verified: with every pair bonded the solution equals the undelaminated
    one within 5e-11 relative in every loading mode and both formulations; on a
    16-ply UD double-cantilever beam the compliance is 0.983 of
    root-rotation-corrected beam theory and the energy release rate from
    the compliance derivative is 0.311 N/mm, 0.989 of beam theory. Theory
    page "Delamination".
- **Crack-face contact, energy release rates and a growth criterion for
  explicit delaminations** (delamination plan, phase 2.4-2.8). Results
  without a delaminated node pair are unchanged: on the 132-case FE
  fingerprint every existing `FieldResults` field and the JSON export are
  identical, and only `summary().details` gains the new (`None`) keys.
  - Crack-face contact: `FESolver.solve(..., crack_contact=True)` (the
    default) holds the delaminated pairs that would interpenetrate with
    normal-only ties, frictionless from a zero gap, by a primal-dual active
    set (at most 30 solves, a warning if it does not settle). Only the ties
    change, never `K`; the intermediate active sets are factorized in a
    separate one-entry cache, and a load case's last set is the warm start
    of its next solve. With `delta_T` the contact is that of the total
    state. `crack_contact=False` lets the faces overlap, as before.
  - Virtual crack closure (`porosity_fe/fe/vcct.py`, public
    `vcct_energy_release_rates` and `CrackFront`): `G_I`, `G_II`, `G_III`,
    `G_T` (N/mm) and the mode mixity at every front node, from the tie
    force of the front pair (read before the reactions fold it) and the
    opening of the pair one element behind (Krueger's 8-node brick
    formulas). The growth direction is exactly `+-x` on a straight front
    and one of eight grid directions on a staircase; the tributary width
    is half at a plate edge.
    - A negative mode I product (closed faces) counts as zero.
    - The shear products keep their sign, so the front sums to the
      energy the extension releases. On a QI short-beam specimen the
      mode III product changes sign across the width: zeroing the
      negative ones over-predicted `G` by 4–7 % against the compliance
      derivative, while the signed sum is within 1.1 %.
  - Growth criterion (public `critical_energy_release_rate`,
    `fracture_load_factors`): Benzeggagh-Kenane, with Reeder's term for a
    separate mode III toughness; critical load factor
    `sqrt(G_c / G_T)` per node, or, with a thermal residual state, the
    root of the quadratic `G_T(lam) = G_c(psi(lam))` by `brentq`. The
    toughness hook takes the local porosity; the porosity-toughness law is
    the phase 4 entry below.
  - New optional `MaterialProperties` fields `G_Ic`, `G_IIc`, `G_IIIc`
    (default `G_IIc`) and `bk_eta`, in N/mm, validated and perturbable,
    `None` by default; no preset sets them. Without them the solve reports
    the energy release rates only. The theory page lists sourced values.
  - New optional `FieldResults` fields `crack_front`,
    `fracture_load_factor`, `fracture_governing_node`, `interface_forces`,
    `crack_contact`, `contact_iterations`, `contact_pairs` and
    `contact_converged`, also in `summary().details`;
    `FieldResults.energy_release_rates(mesh)` recomputes the front table.
  - Exports: a `fracture` block in the FE JSON (schema 1.3, additive,
    `null` for undefined or infinite values) and `G_I` to `G_T` point data
    on the front nodes in the VTU (`NaN` elsewhere; not in the legacy VTK).
  - Verified (`tests/test_vcct.py`): on the 16-ply UD double-cantilever
    beam VCCT `G_I` is 0.9991 of the centred compliance derivative (0.959
    of the one-element forward one) and 0.988 of corrected beam theory,
    with `G_II / G_I` and `G_III / G_I` below 1e-24; the one-step VCCT
    energy is 0.9938 of the centred energy release; `G_I` is uniform
    across the width without Poisson coupling and converges under
    refinement (0.9792, 0.9878, 0.9931 of beam theory at 2, 1 and 0.5 mm).
    End-notched flexure with contact gives `G_II` 0.9983 of the centred
    compliance derivative and `G_I` about 1e-6 of `G_T`, while without
    contact the faces overlap. A single-leg beam's mode mix is 0.3897
    against 0.3936 from corrected beam theory. The load factor is checked
    by hand, under load scaling, with a prestress and through a thermal
    solve. A penny crack in an isotropic block (`h = a/8`) gives a front
    mean of `G_T` 1.061 of `(1 - nu^2) K_I^2 / E` at the nominal radius and
    1.003 at the mean radius of the staircase front, with 0.785 to 1.434
    around it. Theory page "Delamination": contact, VCCT, criterion, load
    factor, verification table and toughness data.
- **Porosity-dependent interlaminar toughness, and toughness validation
  data** (delamination plan, phase 4). Results without a delaminated node
  pair are unchanged: on the 27-case quick FE fingerprint every
  `FieldResults` field and the JSON export are identical, and only
  `summary().details` gains two `None` keys. The empirical strength results
  and the validation headline (4.49 % / 3.89 %) are unchanged.
  - The law (`porosity_fe/toughness.py`): `G_c(Vp) = G_c0 f(Vp)` for each
    mode, with `ToughnessKnockdown` choosing `f` outside
    `MaterialProperties`: `'exponential'` (`exp(-b Vp)`, the default),
    `'linear'` (`max(1 - b Vp, 0)`, coefficients required), `'none'` /
    `None`, or a callable `f(Vp, mode)`. Coefficients are per unit `Vp`
    and `>= 0`; mode III follows mode II; the law never raises the
    toughness, and a callable value outside `[0, 1]` raises.
    `toughness_knockdown_factor(Vp, mode)` screens without FE.
  - Placeholder exponents `b_I = 6.2`, `b_II = 14`, **uncalibrated**
    two-point estimates (decision D4): `b_I` the mean of Olivier et al.
    2007 (4.97) and Hakim et al. 2017 (7.45), `b_II`
    from Agius and Fox 2015 (12.7 to 15.9). `Calibration` mirrors them as
    `TOUGHNESS_B_PLACEHOLDER`. The rise of mode I propagation toughness
    with voids that some studies report is ignored (conservative).
  - FE coupling: the toughness at each crack-front node is knocked down by
    `CrackFront.Vp`, the porosity field at the crack plane (the mean of
    the pair's two nodes), so interface-clustered voids lower `G_c` where
    the crack runs. `FESolver.solve`, `fracture_load_factors` and
    `critical_energy_release_rate` take `toughness_knockdown=` (default
    `'exponential'`); with zero porosity at the front `G_c` and `lambda_c`
    are bit-identical to before. On an end-notched flexure beam with 2 %
    mean porosity, a mid-plane cluster puts 4.8 % at the front and lowers
    `G_c` from 0.585 to 0.395 N/mm against uniform porosity.
  - Flags: `CrackFront.toughness_knockdown` and `toughness_uncalibrated`
    (true when a placeholder lowered a toughness), the same keys in
    `summary().details` and the JSON `fracture` block (schema 1.3,
    additive), and one INFO line per process at the first such use.
  - Validation: the dataset schema gains `G_Ic` and `G_IIc` blocks (N/mm),
    each tagged with its stage (initiation / propagation / unspecified)
    and starter crack (insert / precracked / unspecified); a
    toughness-only dataset needs only fiber and matrix.
    `validation/validate_all.py` maps the keys (they raised `KeyError`),
    predicts them with the law (`predict_toughness`) and summarizes them
    apart (`summarize_toughness_mae`, a report section and a
    `validate_porosity` summary line); `summarize_mae` leaves them out.
    New datasets `hakim_2017` (static `G_Ic` at 1.49, 1.77 and 3.43 %
    voids, Table 1) and `agius_fox_2015` (`G_IIc` ratio from the
    abstract): toughness MAE 3.67 % property-weighted, 4.18 %
    point-weighted, a consistency check since the placeholders come in
    part from the same studies. Carraro et al. 2015 give values only in
    figures and are not included.
  - Docs: theory page "Delamination", section "Porosity and toughness"
    (the law, local porosity, sources and calibration status, limits);
    README Validation and Limitations; CLI page.
- **Fracture specimens: DCB, ENF and MMB, verified against the NASA
  benchmarks** (delamination plan, phase 3). Existing results are
  unchanged: on the quick FE fingerprint every existing `FieldResults`
  field and the JSON export are identical, and `summary().details` only
  gains the `load_point_compliance` key (`None`).
  - `CompositeMesh.for_dcb`, `for_enf` and `for_mmb` build the
    double-cantilever-beam (ASTM D5528), end-notched-flexure (D7905) and
    mixed-mode-bending (D6671) specimens: the insert as a through-width
    `Delamination` strip at the mid-plane (or `interface=`) open at the
    loaded end; the hinges, supports, load lines and crack tip on node
    lines; and a public `FractureSpecimenGeometry` recorded as
    `mesh.fracture_geometry`. The default dimensions are the plan's reading
    of the standards, whose text was not available: check them against
    the purchased editions.
  - The in-plane mesh is graded towards the front (plan decision D5): six
    elements of exactly `element_length` either side of it, then 25 %
    growth up to `max_element_length`. The node lines are the new
    `CompositeMesh.x_grid` (`None`, evenly spaced, everywhere else);
    `nx=` gives a uniform grid that must align the key positions.
    `plane_strain=True` holds `u_y = 0`, the counterpart of a 2-D
    plane-strain model.
  - Loading modes `'dcb'` (equal and opposite hinge line loads, six
    statically determinate constraints at the far end), `'enf'`
    (three-point bending on `roller_supports`, crack-face contact) and
    `'mmb'` (the forces of the statically determinate MMB lever, on
    rollers), force controlled by `applied_load`, through
    `BoundaryHandler.fracture_bcs`; `line_load(..., face='z_min')` loads
    the bottom face. New `FieldResults.load_point_compliance` (mm/N, these
    modes only; also in the JSON `stiffness` block). `effective_modulus` is
    `None`; the knockdown keeps the reference rule (bonded, pore-free
    specimen).
  - MMB lever relations `mmb_lever_length` and `mmb_mode_ratio`: Reeder
    and Crews' split into DCB and ENF parts with crack-length corrected
    beam theory. They give 40.90 and 27.02 mm for `G_II / G_T` = 0.5 and
    0.8 on the NASA specimen, where Krueger used 41.3 and 27.3 mm.
  - Data reduction: `compliance_calibration` (public, with
    `ComplianceCalibration`) fits compliance against crack length by
    D5528 modified beam theory, D5528 compliance calibration or the D7905
    cubic fit, and gives `G = P^2 / (2 b) dC/da` of the fit.
  - `refinement_study` runs the three modes on uniform grids aligned with
    the key positions (`specimen=` for the dimensions) and adds
    `load_point_compliance` and `fracture_load_factor` to their reported
    quantities.
  - Verified against Krueger's NASA benchmarks with the reports'
    materials, geometry and toughness (`tests/test_fracture_specimens.py`,
    12 element layers, mid-width onset): DCB T300/1076
    (NASA/TM-2008-215123) critical opening 0.7289 mm against 0.75 mm
    (−2.8 %); ENF IM7/8552 (NASA/CR-2011-217305) 1523.0 N in plane strain
    and 1537.6 N in 3-D against 1526 N; MMB IM7/8552
    (NASA/CR-2012-217562) 383.5 and 738.0 N in plane strain and 377.0 and
    732.0 N in 3-D against 385 and 751 N. The multidirectional SLB
    (C12K/R6376, TM-2008-215123; `POROSITY_FE_FRACTURE_BENCH=1`) gives a
    critical deflection 4.0 % above the report's with the specimen
    dimensions of its Fig. 3, but 7.0 % above with the 2.0 mm arms and
    34.0 mm crack its FE model figures label. Theory page "Delamination":
    specimens, loads, the MMB lever relations with their sources, the
    benchmark table, the mesh study and mesh guidance.
- **Flexure modes and a mesh-convergence check in the app.** Empirical
  results and the existing loading modes are unchanged.
  - The loading choices gain `flexure_3pt` and `flexure_4pt`, labelled
    FE only (ASTM D7264 procedures A and B). The FE solve runs on the
    `CompositeMesh.for_flexure` specimen (span 32 thicknesses, 13 mm
    wide, `nz` from the mesh setting, as for ILSS); the sidebar describes
    it before the run. The Results tab shows the FE knockdown, the FE
    flexural modulus (`effective_modulus`, shear corrected) next to the
    lamination-theory one (`compute_degraded_clt_flexural_modulus`, same
    layup and `Vp`) and their ratio. The caption says when the porosity
    models differ: the CLT value always uses spherical voids spread
    uniformly through the thickness. The knockdown chart draws the FE
    bar in its own "FLEXURE (FE)" group; the empirical bars are as
    before, since there is no empirical flexure model.
  - The Results tab has a "Mesh convergence check" expander. It runs
    `refinement_study` on the analysis' material, porosity, layup and
    load case, refining its mesh in plane (2 levels, x1 and x1.5, or 3
    levels up to x2; `nz` fixed). Before the run it lists each level's
    element count, estimated peak memory and a rough time (the
    analysis' FE solve time scaled by the DOF ratio to the power 1.5).
    After the run it shows the level table (elements, knockdown, interior
    and domain maximum failure index) and each quantity's verdict and
    behaviour, and says that the domain maximum at free edges and corners
    is mesh-dependent and the interior values are the ones to read.
    Results are cached (`max_entries=8`). For the flexure modes it
    refines the D7264 specimen (see the next entry).
- **Interior failure metrics and mesh-convergence studies for flexure.**
  Knockdowns, moduli and the domain failure quantities are unchanged.
  - The flexure modes have their own interior region
    (`porosity_fe.fe.flexure.interior_element_mask`). It is the span
    between the supports, at least `interior_margin` (default `h / 2`)
    from the side faces and, in the `x`-`z` plane, from each support
    line and load band. Only a half-disc around each contact line is
    left out, so in three-point bending the tension face at mid-span,
    where the bending stress peaks, stays in.
    - Before, the flexure modes used the coupon's lateral-face margin,
      `max(h, 2 element widths)`. That left no element on the 13 mm wide,
      4-element D7264 specimen, so `interior_max_failure_index` and
      `interior_first_ply_failure_load_factor` were `None`.
    - On the default 24-ply T800 QI specimen the interior maximum failure
      index is now 0.993 of the domain maximum, in three- and four-point
      bending. Dropping whole cross-sections next to the load, as the
      short-beam region does, would give 0.95.
  - `CompositeMesh.for_flexure(nx=...)` sets the element count along the
    length instead of `element_length`. `nx` must be even.
    - The span gets the most elements that leave each overhang at least
      the requested length, or two fewer when that puts the four-point
      load lines on node lines. The overhangs take whole elements, so they
      can come out longer than requested.
    - `nx` equal to the default mesh's rebuilds it node for node.
  - `refinement_study(loading='flexure_3pt' | 'flexure_4pt')` refines the
    D7264 specimen. It builds every level with `for_flexure`, using the
    level's `nx`, `ny` and `nz`.
    - The default base is the specimen's own mesh: 48 x 4 x 24 for the
      24-ply presets.
    - Generated levels round `nx` up to even, and explicit levels with an
      odd `nx` raise.
    - The interior margin defaults to `h / 2`, as for ILSS. The span is
      the same at every level; only the unloaded overhangs can differ.
  - The app's convergence check works for flexure. Its level table now
    gives the failure indices to four significant figures, since at the
    app's ILSS and flexure loads they are around 0.01. The interior-index
    help text describes the region of the case that was run: free edges,
    the D2344 shear spans, or the flexure contact zones.
- **Three- and four-point flexure on roller supports (IMPROVEMENT_PLAN
  3.6, C1).** The CLI and pipeline do not expose it yet (the app does,
  see the previous entry), and the `'ilss'` mode is unchanged. Existing
  loading modes are bit-identical (checked on the production mesh, both
  element formulations, all five modes).
  - `FESolver.solve(loading='flexure_3pt' | 'flexure_4pt',
    applied_load=...)` (new `FELoadingMode` members): ASTM D790 / D7264
    procedure A (mid-span load) and D7264 procedure B (two loads, the load
    span half the support span by default). Force controlled, like ILSS;
    `knockdown` is the porous over pristine beam stiffness `P^2 / u^T K u`,
    with the pristine reference also keyed on the support and load lines.
  - `effective_modulus` is the flexural modulus `E_f` from the mid-span
    deflection with the transverse shear deflection removed (Timoshenko:
    `delta / P = a (3 L^2 - 4 a^2) / (48 E_f I) + a / (2 b K_s)`, with the
    laminate shear stiffness `K_s` from the lamination-theory shear-stress
    energy, `(5/6) G13 h` for a UD beam). On the D7264 meshes of the
    24-ply T800 preset it matches the CLT `E_f`
    (`compute_degraded_clt_flexural_modulus`) within 0.15 % for UD and
    cross-ply laminates, at `Vp = 0` and `0.05`; QI reads 1.1-1.3 % low,
    the free-edge effect of the +-45 plies at the 13 mm width (0.4 % at
    twice that width). The UD deflection matches Timoshenko beam theory
    within 0.2 % at 32:1 and 0.7 % at 16:1. `'hex8'` reads 2-5 % stiffer
    on the default mesh and 13-29 % at an element length of 2 h.
  - `CompositeMesh.for_flexure(...)` builds the specimen: span 32
    thicknesses (`span_to_thickness`, or `span`), 13 mm wide, 10 %
    overhang, one element layer per ply, element length up to
    `min(h, 20 h / nz)`, and `nx` chosen so the supports, mid-span and
    four-point load lines are node lines (48 x 4 x 24 elements for the
    24-ply T800 preset). It records them as `mesh.flexure_geometry`, a
    `FlexureGeometry` (new, public): support positions, load span and an
    optional `load_width` that spreads each load over a band.
  - `BoundaryHandler.roller_supports` (`u_z = 0` on bottom-face support
    lines plus three statically determinate in-plane constraints, so the
    horizontal reaction is zero to round-off), `BoundaryHandler.line_load`
    (consistent nodal forces: trapezoidal across the width, linear shape
    functions along `x`, optional uniform-pressure band),
    `BoundaryHandler.flexure_bcs`, `BoundaryHandler.flexure_geometry` and
    `BoundaryHandler.face_line_nodes` (new).
- **Mesh-convergence study `refinement_study` (IMPROVEMENT_PLAN 3.6, B1).**
  No existing result changes; the app runs it from the Results tab (see
  the first entry). `refinement_study(material,
  Vp_or_field, loading, ...)` solves one FE load case (any mode, with or
  without `delta_T`) on a sequence of refined meshes, coarse to fine, and
  returns a `RefinementStudy`: per level the resolution, element and DOF
  counts, wall time, estimated peak memory, knockdown, effective modulus,
  domain and interior maximum failure index and first-ply-failure load
  factor (`RefinementLevel`); per quantity the relative changes, the
  convergence behavior of the three finest levels (`'monotone'`,
  `'non-monotone'`, `'diverging'`, `'mesh-independent'`), the observed
  order, a Richardson extrapolate (monotone only, observed order at least
  0.5), Roache's GCI and a converged flag against `tol` (default 1 %;
  knockdown judged on `1 - KD`) (`QuantityConvergence`). `summary()`
  prints the level table, the convergence table and notes;
  `to_dict()` is strict JSON (non-finite values become `null`).
  - Levels default to `(30, 10, n_plies)` refined in plane by factors
    1, 1.5 and 2 with `nz` fixed; `refine='all'` also scales `nz` in whole
    multiples of the ply count, and `resolutions=` takes explicit levels.
    A level whose `nz` does not resolve the layup is refused (ply aliasing
    would swamp the discretization error). For `'ilss'` the generated
    levels use an even `nx`, so the load line stays on one node column.
  - Interior quantities use one `interior_margin` for every level (by
    default the coarsest level's `max(h, 2 dx, 2 dy)`), so they describe
    one region.
  - Before solving, a level above `max_elements` (default the mesh cap)
    or above the memory budget (`max_memory_gb`, default 80 % of physical
    memory where the platform reports it) raises `ValueError` listing the
    estimate of every level. Each level is released before the next is
    built, and its pristine reference is solved first, so the peak memory
    is that of one model rather than two (1.5 GB instead of 2.9 GB at
    45 x 15 x 24).
  - On T800/epoxy QI (24 plies, midplane-clustered `Vp = 0.03`, `'hex8i'`)
    from 30 x 10 x 24 to 60 x 20 x 24 (15 s, 63 s and 252 s; peak 3.4 GB),
    compression: the knockdown moves by 4e-5 in all, the interior maximum
    failure index converges (1.195, 1.145, 1.133; extrapolate 1.126, GCI
    0.8 %), and the domain maximum grows at the gripped corners (1.23,
    1.39, 1.74) and is flagged mesh-dependent. In `'ilss'` the mid-span
    line load is singular, so its failure indices and load factors are
    mesh-dependent at any resolution. The former production default
    30 x 10 x 12, which does not resolve the 24-ply layup, converges in
    plane to a different laminate (interior index 1.48). Method, tables and
    a production-resolution recommendation are in the theory page
    (finite-element model, "Mesh convergence"); the production default now
    follows it (see Changed).
- **Nodal stress recovery and binary VTU export (IMPROVEMENT_PLAN 3.6,
  E1).** Results and existing exports do not change.
  - `extrapolate_to_nodes` (new, also `FieldResults.nodal_stress` and
    `FieldResults.nodal_strain`) extrapolates the 2 x 2 x 2 Gauss-point
    values of each element to its corners, then averages them only
    within the same ply, so ply-interface jumps are not smeared. Void
    elements are grouped separately. In UD pure bending with two
    elements through the thickness, the surface `sigma_xx` goes from
    0.50 of beam theory (element mean, what `to_vtk` writes) to 1.02.
    `average='all'` and `average='none'` (raw per-element corners) are
    also available.
  - `FieldResults.to_vtu` writes binary VTK XML (`.vtu`) without new
    dependencies. It carries the `to_vtk` point and cell fields plus the
    recovered nodal stress and strain. Options: `Float64` (default) or
    `Float32`; appended raw (default) or inline base64; `exploded=True`
    for per-element corner values. On the production mesh (3,600
    elements) it writes 1.6 MB in 14 ms with the nodal fields, or the
    `to_vtk` fields alone in 1.1 MB and 3 ms. The legacy ASCII file is
    2.0 MB and takes 57 ms.
    `FESolver.export_results(fmt='vtu')` delegates to it. `write_pvd`
    (new) groups VTU files into a ParaView series. `to_vtk` and its
    output are unchanged and remain the default for `fmt='vtk'`.
- **Periodic unit-cell check of the Mori-Tanaka void model
  (IMPROVEMENT_PLAN 3.6, D1).** An internal verification tool, not public
  API (`porosity_fe.fe._rve`, no `__all__` change); no result changes.
  - The constraint reduction of the FE solver
    (`porosity_fe.fe.assembler._ConstraintReduction`, formerly
    `_DirichletPartition`) is now the affine map
    `u = T u_r + g`: besides prescribed values it takes master-slave ties
    `u_s = u_m + c`, chained or ending on a prescribed degree of freedom,
    and reduces to `T^T K T u_r = T^T (F - K g)`. Without ties it slices
    exactly as before: every loading mode is bit-identical on 30 x 10 x 12
    and 30 x 10 x 24, both element formulations.
  - The tool builds a cube of isotropic matrix holding one centred
    ellipsoidal void (the elements whose centroid lies inside, at the
    1 MPa explicit-void stiffness), applies periodic boundary conditions
    `u(x + L) = u(x) + E L` with one corner pinned, and returns the 6 x 6
    effective stiffness from the volume-averaged stress of six unit
    macro-strains, for either element formulation. A 24^3 cell takes about
    35 s (a geometric nested-dissection ordering makes the LU four to six
    times faster than SuperLU's minimum-degree ordering on the wrapped
    grid).
  - Against `_mt_effective_stiffness` at the voxel void fraction (24^3,
    `'hex8i'`, T800 matrix): spherical voids agree within 0.2 % on the
    normal stiffnesses and 1 % on the isotropic bulk and shear moduli up
    to `Vp = 0.10`; the individual shear terms differ by up to 2 % because
    a simple cubic array is not isotropic (its two shear moduli bracket the
    Mori-Tanaka value). Oblate voids of aspect 5 agree within 1 % at
    `Vp = 0.02` and within 3.5 % at 0.05, but once neighbours crowd, the cell
    is much softer out of plane: Mori-Tanaka overestimates `C33` and `C44`
    by 7 % and 10 % at 0.08 and by about 20 % at 0.10. Table, mesh study
    and conclusions are in the theory page (micromechanics, "Verification
    against a periodic unit cell"). `tests/test_rve.py` checks the method
    on cells up to 12^3; `POROSITY_FE_RVE_SWEEP=1` runs the 24^3 cases.
- **Thermal expansion inputs (IMPROVEMENT_PLAN 3.5, first part).**
  `MaterialProperties` gains optional `alpha_1`, `alpha_2`, `alpha_3`
  (lamina CTEs in 1/K) and `T_stress_free` (deg C), all defaulting to
  `None`; `alpha_3=None` means `alpha_3 = alpha_2`. A CTE of `1e-3` /K or
  more is rejected as a probable ppm/K value ("pass 26e-6, not 26"),
  `alpha_1` and `alpha_2` must be given together, and only `alpha_1` may be
  negative. `has_cte` and `cte_vector()` (Voigt `[a1, a2, a3, 0, 0, 0]`)
  are there for the solver. The thermal load case below is the only
  reader; without `delta_T` every existing result is unchanged. Only
  `AS4_3501_6_epoxy` carries CTEs
  (`alpha_1 = -1.0e-6`, `alpha_2 = 26e-6` /K, WWFE-I lamina data, Soden,
  Hinton & Kaddour 1998); the other presets leave them `None` until
  sourced values are confirmed. No preset sets `T_stress_free`. The CTE
  fields are not UQ-perturbable.
- **Thermal / cure residual-stress load case (IMPROVEMENT_PLAN 3.5, PRs 2
  and 3).** The app exposes it too (see the next entry). Mechanical
  solves without `delta_T` are bit-identical to before (checked on the
  production mesh, both element formulations, all four modes and three
  criteria).
  - `FESolver.solve(loading='thermal', delta_T=...)` solves the
    free-standing laminate under a uniform temperature change (K, negative
    for a cool-down) with statically determinate 3-2-1 supports
    (`BoundaryHandler.free_bcs`, new), whose reactions come out at about
    1e-9 N. The load `sum B^T C alpha dT det(J) w` uses the porosity-
    degraded, ply-rotated stiffness and the pristine lamina CTE (an empty
    void does not change the free thermal expansion); for
    `formulation='hex8i'` the incompatible modes are loaded and condensed
    consistently (`f_u + H^T f_a`, strain `B_eff u + G Kaa^-1 f_a`), and
    match a solve with the internal modes kept as unknowns to 1e-13. Stress
    is `C (B u - alpha dT)`; strains are totals. On a [0/90]_2s T800 plate
    the interior stresses match CLT (47.573 MPa) to 0.07 % with `'hex8'`
    and 0.3 % with `'hex8i'`; a distorted homogeneous block expands freely
    with zero stress to round-off. `knockdown` is `nan` (`null` in JSON),
    `effective_modulus` is `None`, and `first_ply_failure_load_factor`
    multiplies `delta_T` (the critical temperature change).
  - `delta_T=` on `'compression'`, `'tension'`, `'shear'` or `'ilss'`
    superposes that residual state on the mechanical solve (two solves:
    the residual stress forms before the laminate is gripped). Knockdown,
    effective modulus and reactions stay those of the mechanical solve.
    The first-ply-failure factor scales only the mechanical stress:
    `first_ply_failure_load_factor(..., prestress_local=...)` (new
    keyword) solves Tsai-Wu in closed form with the residual terms,
    max-stress as linear with an offset, and Hashin branch by branch
    because a residual stress lets a stress change sign as the load grows
    (checked against bisection to 1e-6). For the cross-ply in tension the
    factor drops from 7.0 to 2.7; scaling the total stress would have
    given 1.3.
  - `delta_T` is never inferred (not from `T_stress_free`); a material
    without CTEs, a non-finite `delta_T`, or a mesh whose element layers
    do not resolve the requested layup (`nz` not a multiple of `n_plies`)
    raises `ValueError`, the last unless `allow_unresolved_layup=True`.
    An unsymmetric element layup logs a warning (linear theory; `'hex8'`
    underpredicts its curvature, 0.84 of CLT where `'hex8i'` gives 1.00).
  - `FieldResults` gains `delta_T`, `residual_stress_local`,
    `residual_max_failure_index`, `load_factor_basis`, `cte_local`, and,
    for every solve, `interior_max_failure_index` and
    `interior_first_ply_failure_load_factor` over the elements at least
    `interior_margin` from the lateral faces, away from the free-edge
    singularity. The JSON export adds a `thermal` block, these keys in
    `failure` and `delta_T_K` in the provenance (`JSON_SCHEMA_VERSION`
    1.2, additive); VTK/VTU files add `residual_sigma_11_local`,
    `residual_sigma_22_local` and `residual_tau_12_local` cell fields and
    leave out the `knockdown` field when it is `nan`. `FELoadingMode`
    gains `'thermal'`. Theory: `docs/theory/thermal.md`.
  - Caveats documented there: CTE data (one preset), effective `delta_T`
    (no chemical shrinkage, relaxation or moisture), geometric linearity,
    and that under the square-root strength rule porosity can lower the
    residual failure index.
- **Thermal load case and VTU download in the app (IMPROVEMENT_PLAN 3.5
  PR 4, 3.6 E1).** Runs with the thermal option off are unchanged.
  - The sidebar has a new "Thermal residual stress (FE)" section. It
    appears only for a material with lamina CTEs (`has_cte`, today only
    `AS4_3501_6_epoxy`); for the others a caption says why the option is
    unavailable. The choices are off, "add cure residual stress to the
    mechanical case" (superposed on the FE solve of the selected loading
    mode) and "thermal only".
  - The "ΔT (K), negative = cool-down" input starts empty, with no hidden
    default, and the Run button stays disabled until it is filled. A
    labelled suggestion (−150 K for a 177 °C-cure epoxy cooled to room
    temperature, or the preset's `T_stress_free` when one is recorded) is
    applied only by its button.
  - The thermal solve refuses a mesh whose element layers merge plies.
    The sidebar warns before the run when `nz` is not a multiple of the
    ply count. If the solve refuses, the app shows the solver's message
    instead of crashing, and keeps the empirical results. Both places
    offer a "Use nz = ..." button that switches on Expert mode with the
    `nz` that resolves the layup.
  - The Results tab shows the knockdown (n/a for thermal-only), the
    residual and interior maximum failure indices, and the first-ply-
    failure factor read with its `load_factor_basis`. For thermal-only
    that factor is the critical ΔT (factor × ΔT). For the combined case
    it multiplies the applied load, and the factor without the residual
    stress is shown next to it. The knockdown chart has no FE bar for a
    thermal-only run.
  - The Export tab has a "Download FE results (VTU)" button
    (`FieldResults.to_vtu`, binary, with the recovered nodal stress and
    strain and the residual-stress fields of a thermal run). The bytes
    are cached with `st.cache_data(max_entries=4)`.
  - `run_analysis` takes `thermal_mode` and `delta_T` in its config. An
    older config dict without them means no thermal load. The result
    gains `fe_thermal`, `fe_mechanical_fpf` and `fe_nz_suggestion`.
- **Incompatible-mode hex8 element (IMPROVEMENT_PLAN 3.6, step A1).** `FESolver(..., formulation='hex8i')` enriches each brick with the
  nine Wilson-Taylor incompatible modes (with Taylor's centroid-Jacobian
  correction), condensed out per element and folded into an effective
  `B = B + G H` in the batched path, so stiffness, stress recovery,
  failure and export run unchanged. The standard hex8 locks in bending
  with an error of about `(G13 / E11) (dx / h)^2` (element length over
  laminate thickness): on a UD T800 beam `E_bend / E11` is 2.265 at 4x2x2
  and 1.084 at 16x4x8 with `'hex8'`, 1.002 and 1.001 with `'hex8i'`. The
  new element passes the patch test on distorted meshes and has exactly
  six zero-energy modes. On the production mesh (T800 QI, 30x10x12) it
  leaves the compression, tension and shear knockdowns within 1e-4, moves
  the ILSS knockdown by +0.0006 to +0.0034, lowers the peak ILSS
  `|tau_13|` by about 45 % and raises the ILSS first-ply-failure load
  factor by 17-25 %. The option is validated, recorded as
  `FieldResults.formulation`, in `summary().details` and as
  `solver.formulation` in the JSON export, and is part of the assembler's
  `K` cache key and the shared pristine-reference cache key, so the two
  formulations never share a cached result. `GlobalAssembler`,
  `build_element_batch` and `Hex8Element` take the same keyword
  (`Hex8Element.G_matrix` / `strain_operator` expose the enrichment); the
  new `FEFormulation` type alias names the accepted values. It is now the
  default (see "Changed"); `formulation='hex8'` reproduces the earlier
  results bit for bit. See "Element formulations" in `docs/theory/fe.md`.
- **Documentation: theory pages, CLI reference, full API reference
  (IMPROVEMENT_PLAN 6.5).** New `docs/theory/` pages state the porosity
  field, Mori-Tanaka/Eshelby micromechanics, CLT, the empirical knockdown
  laws and their calibration, the FE model and knockdown definition, the
  failure criteria, the fatigue and hygrothermal factors, and uncertainty
  propagation, as implemented. `docs/cli.rst` renders both commands'
  options from their argument parsers (new `sphinx-argparse` docs
  dependency). `docs/api.rst` now covers every name in
  `porosity_fe.__all__` (61, up from 10), and `tests/test_docs.py` fails if
  the two drift apart. The CHANGELOG is part of the site. The README
  per-property MAE table is brought back in line with the current
  validation run (ILSS 4.9 %, tensile strength 8.2 %; it showed 4.3 % and
  6.9 %).
- **Fiber constituent inputs.** `MaterialProperties` gains
  `fiber_poisson` (default 0.2) and `fiber_shear_modulus` (default `None`,
  meaning the isotropic `E_f / (2 (1 + nu_f))`). They replace a hard-coded
  isotropic fiber in the Halpin-Tsai degradation ratios. The defaults
  reproduce the earlier results bit for bit.
- **PyPI release automation (IMPROVEMENT_PLAN 6.4).** A new
  `.github/workflows/publish.yml` runs on `v*` tags:
  - `.github/scripts/check_release.py` checks that the tag matches both
    version literals, that `[Unreleased]` is empty and that the release
    has its own CHANGELOG section;
  - the sdist and wheel are built and pass `twine check --strict`, and
    the wheel is smoke-tested;
  - the packages are published through PyPI trusted publishing (no
    stored token).

  A `.pre-commit-config.yaml` runs the CI lint checks (ruff, mypy)
  locally. The license metadata moves to an SPDX expression
  (`license = "MIT"`, setuptools >= 77), which removes the deprecation
  that breaks builds after 2027-02-18.
- **Property-based tests (IMPROVEMENT_PLAN 6.6).** `tests/test_properties.py`
  uses `hypothesis` (new dev dependency) to check that:
  - stiffness rotation round-trips;
  - stress transforms compose as a group;
  - work is frame-invariant, which guards the engineering-shear factor 2;
  - the Tsai-Wu index is frame-invariant;
  - every knockdown law stays a monotone fraction on `Vp in [0, 0.05]`,
    at QI and after layup scaling.

  The headline validation MAE (35 entries, 239 points) is now pinned. It
  was 7.05% property-weighted and 6.53% point-weighted, and is 4.49% /
  3.89% after the layup-scaling change under "Changed".
- **Uncertainty propagation in the CLI and app (IMPROVEMENT_PLAN 3.4).**
  - `porosity-analyze --uq` writes `porosity_uq_<Vp>.json` (new
    `porosity-fe.uq` format, `FORMAT_UQ`) for every loading mode. The
    sampled inputs are set with `--uq-samples`, `--uq-coef-cov`,
    `--uq-vp-cov` and `--uq-strength-cov`.
  - The app's Results tab gains an **Uncertainty** expander showing the
    p5/p50/p95 metrics and a knockdown histogram.
  - `propagate_uncertainty(coef_cov=...)` perturbs the knockdown law's
    calibration coefficient. Before this, material scatter alone left
    the knockdown with zero spread.
  - A passed `MaterialProperties` instance now reports its preset name
    (or `"custom"`) instead of `"MaterialProperties"`.
  - Draws beyond the calibration bound produce one summary warning
    instead of one per draw.
  - New `save_uq_results_to_json()`, and a `solver_kwargs=` hook on
    `build_empirical_pipeline` for the coefficient overrides.
- **`sweep_configurations(void_volume_fractions, ...)`** runs
  `compare_configurations` for several porosity levels with every
  `(Vp, config)` pair in one task list, so `n_jobs > 1` keeps one process
  pool busy across levels. `porosity-analyze --jobs N` uses it when more
  than one `--vp` is given. Results equal the per-level calls.
- **PNG download per figure in the Streamlit app** (profile, mesh,
  results, stress tabs).
- **First-ply-failure load factor.** `FieldResults.first_ply_failure_load_factor`
  is the multiplier on the applied load at which the selected criterion
  first reaches 1 (linear scaling; margin of safety = factor - 1), solved
  exactly per Gauss point for Tsai-Wu, Hashin and max-stress. Also in
  `summary().details` and the JSON export.
- **Reaction forces and effective modulus from FE solves.** `FieldResults`
  now carries `reaction_forces` (N, per node) and `effective_modulus`
  (MPa): `E_x` for compression/tension and `G_xy` for shear, from the
  strain energy `u^T K u / (strain^2 V)`; `None` for the force-controlled
  ILSS bend. A pristine UD specimen recovers `E11` and `G12` exactly, and
  a quasi-isotropic one matches CLT `G_xy` exactly and `E_x` to ~1%. The
  JSON export gains a `stiffness` block.
- **Self-documenting `units` block in JSON envelopes** (`save_results_to_json`,
  `write_results_json` / `_serialise_payload_json`, `serialise_ncr_json`) plus
  per-field `description` entries on numeric leaves in
  `validation/schemas/porosity_results_schema.json` so downstream consumers
  reading only the JSON can tell whether `knockdown` is a fraction, a
  percentage, or a multiplier. `JSON_SCHEMA_VERSION` bumped to `1.1`
  (additive, backwards-compatible — 1.0 files still load and validate). (#131)
- **`--jobs N` flag on the `porosity-analyze` CLI** parallelises the per-
  configuration sweep in `compare_configurations` over a
  `concurrent.futures.ProcessPoolExecutor`. `N=1` (default) preserves
  the deterministic serial path byte-for-byte; `N>1` dispatches the
  `(Vp, config)` calls across worker processes; `0`/`-1` resolve to
  `os.cpu_count()`. Results are deterministically re-assembled by
  `(Vp, name)` so the returned dict is independent of `N`. Measured
  ~2x wall-clock speedup on the 5-Vp × 5-config sweep with 4 workers
  on a 4-core box. (#52)
- **`n_jobs` kwarg on `compare_configurations`** exposes the same knob
  to library callers (including a future GUI batch sweep).
- **Top-level `_analyze_one(Vp, name, config, material, applied_stress,
  seed)` helper** factors the per-configuration build/solve out of the
  inner loop so it can be pickled to a worker process. The returned
  `(Vp, name, result_dict)` tuple includes the mesh / porosity_field /
  empirical_solver instances (all three pickle cleanly today; no
  rebuild on the main process is needed).

### Removed
- **The PyQt6 desktop GUI is no longer shipped** (it was removed before
  this changelog's tracked history begins). Older entries that mention the
  `porosity-fe` console script, the `[gui]` extra, PyQt6 or
  `PorosityFE.spec` describe that GUI. The supported interfaces are the
  Streamlit app (`streamlit run app.py`), the `porosity-analyze` and
  `validate_porosity` commands, and the `porosity_fe` library; the
  executable build is `ValidatePorosity.spec`.

### Fixed
- **App figures were hard to read.** No result changes.
  - Mesh and Stress tabs: a laminate section about 50 x 4.4 mm was drawn
    to scale, a sliver under a colorbar about ten times its height that
    ran over the title. The z axis is now exaggerated by the smallest
    factor of 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, ... that draws the
    section at least 0.3 times as high as it is long (x4 for the default
    coupon); the z axis label gives the factor, the ticks stay in mm.
    The colorbar is horizontal under the section and as wide as it, and
    the title is a short line with a subtitle, clear of both.
  - A constant field (uniform porosity: stiffness retention 97 %
    everywhere) got a colorbar over its round-off, with an offset
    `1e-12+9.7e1` and ticks 0.995 / 0.000 / -1.009. Colorbars now have
    plain tick labels (no offset or scientific notation), and a constant
    field gets its value +-1 % (at least +-1 unit), a mark at the value
    and "uniform: 97.0 %" in the label.
  - The stress map was always the diverging `RdBu_r` centred on zero, so
    a field of one sign (the thermal σ11, -33 to -38 MPa) was one flat
    blue. The diverging map is now used only when the field crosses zero;
    a field of one sign gets the matching half (blue for compression, red
    for tension) as a sequential map over its own 2nd to 98th percentile,
    von Mises gets cividis, and the colorbar marks clipped ends. Each
    element is drawn as one cell with continuous colours instead of a
    20-level contour of the element centroids, which smeared the jumps
    between plies.
  - The Results chart's legend sat on the compression bars; it is now
    right of the axes. Each group centres the bars it has. The
    uncertainty histogram's legend moved outside as well.
  - The Stress tab opens on the component that governs the load case:
    τ13 for ILSS, σ22 for thermal only (transverse stress, where residual
    cracking starts), τ12 for shear, σ11 otherwise. A component the user
    picks stays picked.
  - Streamlit warned that the layup input was "created with a default
    value but also had its value set via Session State"; its default now
    comes from session state only.
- **The FE JSON export wrote invalid JSON.** Tsai-Wu does not separate
  failure modes, so `FieldResults.failure_mode_indices` holds `NaN` for
  every mode except `max_fi`, and `FESolver.export_results` wrote them
  as the bare token `NaN`. Python reads that, but strict parsers
  (JavaScript `JSON.parse`, jq, `json.loads(..., parse_constant=...)`)
  reject the whole file. Every JSON writer (`export_results`,
  `save_results_to_json`, the app's export and NCR JSON,
  `save_uq_results_to_json`) now goes through one serializer with
  `allow_nan=False`:
  - A value that is not defined is written as `null`, and the schema and
    the `export_results` docstring say what `null` means for each field.
    The Tsai-Wu per-mode indices are `null` where they were `NaN`. The
    nullable fields that were already `null` keep their meaning:
    `knockdown_factor` (thermal-only), the first-ply-failure factors (no
    stressed point), the interior values (no interior element) and
    `effective_modulus_MPa` (ILSS, thermal).
  - Any other non-finite value raises `ValueError` naming its location
    (e.g. `$.failure.max_tsai_wu_index`) before any file is written.
    Before, it was written silently. The JSON text is built before the
    file is opened, so a refused export leaves no partial file and no
    `.npz` sidecar.
  - `load_results_from_json` reads `null` as `None`. It reads the `NaN` /
    `Infinity` tokens of files written by earlier builds as `None` too,
    with a logged warning, so old and new files load the same way.
  - `validation/schemas/porosity_results_schema.json` declares the FE
    `failure`, `stiffness` and `thermal` fields, with `null` allowed only
    where it has a meaning.
  - Tests parse every writer's output with a parser that rejects the
    `NaN` / `Infinity` tokens, including the thermal and Tsai-Wu cases. A
    static check fails if any `json.dump` call in the package omits
    `allow_nan=False`.
- **`FESolver.solve(solver='cg')` and `solver='minres'` returned wrong
  answers at the default `rtol` (result change for iterative-solver
  users).** The penalty rows put `alpha * v ~ 1e11 * v` into the
  right-hand side of the displacement-controlled modes, so `||F||` was
  huge and the relative-residual test passed as soon as the constrained
  DOFs were right, with the interior unconverged. On a 20x8x12 clustered
  `Vp = 0.02` mesh, compression CG was 24 % off in displacement and 16 % in
  stress, shear CG 17 % in the failure index, and compression MINRES 8 % in
  displacement, all reported as converged. ILSS MINRES always failed its
  residual check, because SciPy's MINRES stops on a preconditioned estimate
  about 1,000x below the true residual. With the boundary conditions
  eliminated (see Changed), CG agrees with the direct solve to within
  4e-8 in displacement, stress and failure index on the production mesh.
  MINRES is warm-restarted, at most 3 times, until its true residual
  meets `rtol` (one restart on every shipped load case, ILSS included).
  It agrees to within 2e-6, since a residual minimizer is less accurate
  than CG at the same tolerance. Iterative solves are slower because they
  now actually converge: CG takes about 1,000 iterations (about 1.3 s) on
  the production mesh instead of stopping after about 85.
- **ILSS beam-theory test reference.** `TestILSSBeamTheoryValidation`
  compared the peak FE `|tau_xz|` with `1.5 |F| / (b h)`, but in a
  three-point bend each half-span carries `V = F / 2`, so the beam-theory
  peak is `0.75 |F| / (b h)` (the ASTM D2344 short-beam-strength formula).
  The test passed only because hex8 shear locking roughly doubles the
  Gauss-point `tau_xz` near mid-span. The test now uses the correct
  reference, requires `'hex8i'` to match it (within 10 % at mid-width in
  the shear spans; measured -7 %), checks the section-mean shear against
  the parabolic profile for both elements, pins the known hex8 overshoot
  (about 2.1x in the old band), and adds simply supported (Timoshenko,
  within 2 %) and pinned-support (tied-arch) deflection checks. Library
  results are unchanged.
- **`MaterialProperties(tsai_wu_F12=...)` is now the normalized
  coefficient it was documented as.** The docstring called it a
  dimensionless value in `[-1, 0]`, but the FE Tsai-Wu check used it as
  the raw `F_12` (units MPa^-2), so any value a user would choose (e.g.
  `-0.3`) swamped `F_11` and `F_22` (about 1e-7 MPa^-2) and opened the
  failure envelope. It is now `F*_12 = F_12 / sqrt(F_11 * F_22)`: the
  solver uses `F_12 = F*_12 * sqrt(F_11 * F_22)` with each element's
  degraded strengths, and `-0.5` reproduces the default. Results with
  `tsai_wu_F12=None` (every built-in material) are unchanged.
- **FE caches key on every material field.** The shared pristine-reference
  cache (`FESolver` knockdown) and the assembler's `K` cache keyed the
  material by its `repr`, which omits `E33`, `G13`, `G23`, `nu13`, `nu23`
  and the other non-printed fields. Two materials differing only there
  shared one pristine reference: a T800 ILSS solve followed by a copy with
  softer `G13` / `G23` / `E33` gave a knockdown of 0.613 instead of 0.952.
  An in-place edit of such a field also left a solver on its stale `K`.
  Both caches now key on `dataclasses.astuple(material)`. **Result change**
  only for callers that hit the collision.
- **Element ply assignment is exact, and the FE solver warns when the
  mesh does not resolve the layup.** Each element takes the angle of the
  ply containing its centroid. With fewer element layers than plies the
  centroid often sits exactly on a ply interface, and floating-point
  division put it on either side depending on the ply thickness. On the
  production mesh (`nz = 12`) the 24-ply T800 `'QI'` laminate became
  `[90, -45, 45, 90, 90, -45, 45, 90, 90, -45, -45, 0]`, keeping one of
  its six 0-degree plies, while T700 (same ply count) got
  `[90, -45, 45, 0]` three times. Ply ids are now computed in integer
  arithmetic, and an interface centroid (or node, for `ply_ids`) always
  takes the ply above it.
  - New `CompositeMesh.element_layup` gives the element angle of each
    layer.
  - New `CompositeMesh.layup_discrepancies()` lists how that layup differs
    from the requested one: element layers that merge plies of different
    angles (any `nz` that is not a multiple of `n_plies`, unless the
    merged plies share an angle), the resulting change in each angle's
    thickness fraction, and lost symmetry or balance.
  - `FESolver` logs these findings as one warning per mesh. The empirical
    path does not use the element layup, so it does not warn.

  **Result change:** FE results change where rounding used to pick the
  lower ply: T800 at `nz` 6, 12, 18, 20, 30 and 36; CF/PEEK at 12, 28 and
  44; glass/epoxy at 42; T300/934 and HTA/EHkF420 at 40. Element layups
  with `nz` a multiple of `n_plies` are unchanged. On the production mesh
  T800 is now `[90, -45, 45, 0] x 3` like the other 24-ply presets, and
  CF/PEEK goes from `[0, 90, 90, 45, -45, -45, -45, -45, 45, 90, 90, 0]`
  to `[0, 90, 90, 45, -45, -45, -45, 45, 45, 90, 0, 0]`. The README FE
  distribution table (T800, `nz = 12`, compression, `Vp = 0.03`) moves
  from 0.9844 / 0.9839 / 0.9846 / 0.9638 to 0.9922 / 0.9923 / 0.9914 /
  0.9821 (uniform / clustered midplane / clustered surface / interface).
  The production mesh still resolves no preset: each element layer spans
  two plies of a 24-ply preset and 4/3 of a 16-ply one, so the default FE
  laminate is not the requested one, and none of the presets keeps its
  symmetry. Resolving the plies by default (`nz = n_plies`) or giving a
  multi-ply element the thickness-averaged stiffness of its plies awaits
  a maintainer decision.
- **Total mesh size is capped.** `CompositeMesh` rejected more than
  10 000 elements per axis but still admitted 10 000^3 in total, so an
  oversized mesh failed with an out-of-memory error partway through FE
  assembly. It now rejects more than 1 000 000 elements up front and
  states the estimated FE memory in the message. The app shows the error
  under the mesh sliders and disables Run.
- **Triaxial voids use the exact Eshelby tensor.** Voids with three
  different radii were treated as a spheroid about their longest axis; the
  Mori-Tanaka stiffness now uses the general-ellipsoid tensor (Mura's
  integrals). **Result change** only for triaxial `void_shape_radii`; the
  preset void shapes are axisymmetric and unchanged.
- **UQ sweeps survive extreme draws.** A wide distribution could push
  `fiber_volume_fraction` past 1, or a modulus or strength below 0 with
  `'normal'`, and abort the sweep in validation. Perturbed values are now
  clipped: fiber volume fraction to the hexagonal packing limit (0.907),
  moduli and strengths to stay positive.
- **Mesh quality check matches assembly.** `check_mesh_quality` tested the
  Jacobian only at the element center, while assembly rejects a
  non-positive determinant at any of the 8 Gauss points. It now checks the
  same points, so a corner-inverted element is reported.
- **`FatigueModel` says when `R` is ignored.** The S-N slopes are calibrated
  at `R = 0.1` and no mean-stress correction is applied, so other values of
  `R` do not change the result. A non-calibration `R` now emits a
  `UserWarning` saying so.
- **Knockdown-curve plot paired the wrong points.** `FEVisualizer.plot_knockdown_curves`
  sorted the x values numerically but read the y values in label text
  order. With `--vp 0.02 0.10 --plots` the 10% knockdowns were drawn at
  2% and the reverse. Labels such as `2p5pct` (from `--vp 0.025`) also
  crashed the plot. Both are now parsed and ordered one way.
- **Download filenames are Windows-safe as documented.**
  `_sanitise_filename_component` now also replaces `\ : * ? " < > |`,
  other whitespace and control characters, not just `/` and spaces.
- **`compare_configurations(configs={})` / `sweep_configurations(configs={})`
  run nothing.** An empty mapping used to fall back to the five bundled
  configurations; only `None` does now.
- **`porosity-analyze` output failures honor the exit-code contract.**
  An `OSError` while writing a JSON result or a plot exited with a
  traceback (code 1); it now returns 2, and any other output failure
  returns 3, as documented.
- **Bounded app result cache.** `run_analysis_cached` kept every analysis
  (mesh and fields included) for the life of the server; it now holds at
  most 16. The export JSON is also serialized once per rerun, not twice.
- **Void stress concentration factors from elasticity.**
  `VoidGeometry.stress_concentration_factor` used uncited
  piecewise-linear rules in aspect ratio, with shape classes that
  disagreed with the micromechanics' (and jumped at aspect ratio 1.2 and
  `radii[1] = radii[0]/2`). A penny void loaded in its own plane got
  SCF 17, and orientation was ignored. It now uses the exact solution for
  a traction-free ellipsoidal cavity in an isotropic matrix: Eshelby's
  interior field plus the traction-free jump condition, maximized over the
  surface. That reproduces Goodier's sphere values, Kirsch's 3 and Inglis'
  `1 + 2a/b`, honors `orientation`, and takes the matrix Poisson's ratio
  (`nu_m`, default 0.35; `EmpiricalSolver` passes
  `material.matrix_poisson`). **Result change:** only the discrete-void
  term of `EmpiricalSolver.nodal_knockdown` (and the SCF plot) changes.
  For example, the preset penny void goes from SCF 17 / 14 (tension /
  ILSS) to 1.16 / 6.08, and a sphere from 2.0 / 1.8 to 2.07 / 1.86.
  Specimen-level failure loads and the validation MAE are unaffected.
- **FE knockdown is now a real stiffness ratio.** `FieldResults.knockdown`
  was the ratio of signed domain-mean stresses (porous vs pristine
  stiffness applied to the porous strain field), silently clamped to 1.
  It read `sigma_xx` for the shear mode and the sign-changing `tau_xz`
  field for ILSS, so both came out as exactly 1.0 at any porosity. The
  knockdown is now porous over pristine structural stiffness from a
  second solve of the same mesh and boundary conditions with no porosity
  or void elements: the `E_x` / `G_xy` ratio for compression, tension and
  shear, and the beam-stiffness (inverse compliance) ratio for ILSS.
  Values above 1 are logged, not clamped. The pristine result is cached
  by mesh geometry and material, so a porosity sweep pays for it once
  per loading mode; the first solve on a new production mesh takes about
  3 s longer. **Result change:** shear and ILSS FE knockdowns drop below
  1 (e.g. 0.985 and 0.946 for uniform QI at `Vp = 0.06`), and the
  compression values shift slightly (README comparison table updated).
- **`FESolver.solve('tension')` now pulls.** `applied_strain` defaulted
  to `-0.01` for every mode and `tension_bcs` forwarded it unchanged, so
  a tension solve without an explicit strain was a compression solve
  (identical failure indices to `'compression'`). The default is now
  mode-dependent (`-0.01` compression, `+0.01` tension and shear), and a
  strain whose sign contradicts `'compression'` / `'tension'` logs a
  warning. **Result change** only for callers relying on the default;
  explicit strains (as the app and pipeline pass) are unaffected.
- **One definition of a "void element" in the FE path.** Failure
  evaluation and the first-ply-failure load factor now skip geometric
  void elements (`CompositeMesh.void_elements`, assembled with ~1 MPa
  stiffness) as well as elements above the porosity threshold. Before,
  a geometric void with low nodal porosity went through the failure
  polynomial with near-pristine strengths. The thresholds are shared
  constants in `porosity_fe.fe.element` (`VOID_VP_THRESHOLD = 0.95`,
  `VP_STIFFNESS_CLAMP = 0.99`). Geometric voids never governed the
  maximum index in the cases checked, so only their per-element entries
  change (to 0).
- **Mori-Tanaka Eshelby tensor corrected.** Three errors in
  `_mt_effective_stiffness`, now checked against Mura's integrals and the
  closed-form spherical-void Mori-Tanaka moduli:
  (1) the shear diagonal lacked the factor 2 of the engineering-shear
  Voigt form, so every void shape under-degraded shear stiffness;
  (2) in the axisymmetric (cylindrical / penny) branch `S_1122` and
  `S_2211` were swapped and the `S_2211` expression was wrong, which also
  made `C_eff` asymmetric (11% for cylinders, 76% for pennies);
  (3) `_degraded_composite_stiffness` read the matrix moduli from single
  entries `C_eff[3,3]` / `C_eff[0,1]`, which for an anisotropic `C_eff`
  picks one plane's values depending on the void orientation. It now uses
  the isotropic (Voigt-average) projection, which is exact for spheres.
  **Result change:** FE knockdowns at `Vp = 0.04` move by about one
  percentage point (e.g. uniform QI 0.963 -> 0.950, penny interface
  0.991 -> 0.979), with matching shifts in stresses and failure indices.
  CLT modulus predictions change too: validation MAE goes from 7.09% to
  7.05% property-weighted and 6.56% to 6.53% point-weighted, and only
  modulus entries move (liu_2018 transverse modulus 3.29% -> 2.76%,
  stamopoulos_2016 shear modulus 15.39% -> 14.73%, stamopoulos_2016
  transverse modulus 1.02% -> 1.49%). The penny-void regression pins in
  `test_homogenization.py` were re-derived: at high crack density the
  in-plane `C_11` now correctly approaches its plane-stress limit.
- **FE Hashin criterion now sees interlaminar stresses.** It used only
  `sigma_11`, `sigma_22`, `tau_12`, so `loading='ilss'` with
  `failure_criterion='hashin'` returned an index blind to the governing
  `tau_13` / `tau_23`. A fifth `delamination` mode (Brewer & Lagace 1988:
  `(<sigma_33>/Y_t)^2 + (tau_13^2 + tau_23^2)/S_23^2`) is part of
  `max_fi` and the load factor. **Result change:** Hashin ILSS indices are
  now governed by delamination; Hashin results for the other load cases
  in the regression fingerprint, and all Tsai-Wu / max-stress results,
  are unchanged. Every criterion's mode breakdown now has a
  `delamination` key (NaN for Tsai-Wu, 0 for max-stress, which already
  checks those components).
- **Empirical extrapolation warning now reports the right value and sees
  local peaks.** The message labelled the specimen-average `Vp` as
  "max Vp"; it now says "specimen-average Vp". `apply_loading()` also
  warns when the mean is within the `Vp <= 0.05` calibration bound but a
  `clustered` / `interface` distribution's local peak is not, since the
  per-node knockdown field is then extrapolated near the peak.
  `get_failure_load()` still checks only the mean, because its result uses
  the mean. Nodes inside discrete voids (`Vp = 1.0`, handled by the SCF
  step) are excluded from the peak. The warning is now attributed to the
  caller's line for both entry points.
- **App: the FE legend entry no longer disappears** for tension, shear,
  and ILSS runs. It was attached only to a bar in the first (compression)
  group, where the FE series is not drawn.
- **App: an FE solver failure no longer discards the empirical results.**
  Only the FE solve is guarded; on failure `fe_field` is `None` and
  `fe_skipped_reason` names the exception, which activates the existing
  "FE solve was skipped" notice.
- **App: `run_analysis` builds the field, mesh, and empirical solver
  through `build_empirical_pipeline`**, so changes to mesh defaults or
  ply-angle handling reach the GUI.
- **`FEVisualizer` closes figures after saving them.** The CLI `--plots`
  sweep no longer accumulates 100+ open figures. The `Figure` is still
  returned; without `save_path` it is left open for the caller as before.
- **Provenance timestamps no longer use the deprecated
  `datetime.utcnow()`** (Python 3.12+). The `...Z` string format is
  unchanged.
- **`validate_porosity --version` falls back to `porosity_fe.__version__`**
  instead of its own hard-coded version string, leaving
  `porosity_fe/__init__.py` as the only literal to bump at release.

### Changed
- **`fe/solver.py` split into focused modules, and lower FE peak memory
  (IMPROVEMENT_PLAN 5.1).** No result changes: every `FieldResults` field
  (reactions and mode indices included) and the JSON export are bit
  for bit the same, checked by sha256 fingerprints on the production meshes
  of AS4/3501-6 for compression, tension, shear, thermal, combined
  `delta_T`, three- and four-point flexure and ILSS, both element
  formulations, all three failure criteria, direct and CG solves.
  - `fe/solver.py` (1,768 lines) keeps `FESolver` as the orchestrator
    (about 750 lines, half of them the public docstrings). `FieldResults`
    moves to `fe/results.py`; the steps of a solve to private modules:
    `fe/_load_cases.py` (loading modes, argument and `delta_T` checks,
    boundary-condition dispatch, interior region), `fe/_constrained_solve.py`
    (exact Dirichlet elimination, the per-constrained-set LU cache, CG /
    MINRES), `fe/_knockdown.py` (stiffness measure, pristine reference and
    its cache), `fe/_thermal.py` (layup check, unit-`delta_T` free-standing
    solve) and `fe/_postprocess.py` (stress / strain recovery, reactions and
    modulus, failure metrics). `from porosity_fe.fe.solver import FESolver,
    FieldResults` and the private names other modules and the tests use
    still work.
  - `FESolver.solve` solves (or looks up) the knockdown's pristine
    reference before it assembles the porous model, so the reference's
    factorization is freed before the porous one is built: peak RSS of a
    30 x 10 x 24 compression solve (T800, `[0/45/-45/90]_3s`) drops from
    1.22 GB to 0.74 GB. The app and `refinement_study` no longer call the
    private `_pristine_stiffness_measure` first to get that.
  - The unit-`delta_T` thermal solution is cached and rescaled, so its
    3-2-1 factorization now leaves the LU cache once the solution is cached
    instead of staying next to the mechanical one (the mechanical entries
    keep their LRU behaviour). Peak RSS on the same mesh (AS4/3501-6,
    `delta_T = -157` K): a combined compression solve 1.55 GB to 0.74 GB;
    the app's combined residual-stress run 1.06 GB to 0.76 GB (ILSS
    specimen: 0.66 GB to 0.50 GB). `refinement_study`'s memory estimate
    counts one factorization for direct solves with `delta_T` (was two).
- **The FE `'ilss'` mode is the ASTM D2344 short-beam test (IMPROVEMENT_PLAN
  3.6, C2).** It was a three-point bend of whatever coupon the mesh was,
  with both bottom end edges pinned in x, y and z and the load split
  equally over the mid-span top nodes: on the 50 mm production coupon a
  span of 11.4 thicknesses (mostly flexure; shear was 12 % of the
  deflection) and a tied arch (horizontal reaction about 4 P, deflection
  about half the simply supported one). Now it runs on a D2344 specimen.
  - `CompositeMesh.for_ilss(field, material, ...)` (new) builds it: support
    span 4 h, length 6 h, width 2 h (`span_to_thickness`,
    `length_to_thickness`, `width_to_thickness`; ISO 14130 is 5 / 10 / 5),
    one element layer per ply, in-plane element length h / 4
    (24 x 8 x `n_plies`; `nx`, `ny`, `nz`, `element_length` override it),
    built on `for_flexure` so the supports and mid-span are node lines. It
    records them as `mesh.ilss_geometry` (new attribute, a
    `FlexureGeometry`; also `mesh.flexure_geometry`, so `'flexure_3pt'` on
    the specimen runs the same test).
  - Supports: the `flexure_3pt` rollers, no horizontal reaction (below
    1e-11 N at 10 N). Loading nose: a uniform pressure over a band at
    mid-span with consistent nodal forces, by default the Hertz contact
    width of the 6 mm D2344 nose at the pristine short-beam strength
    (`2 sqrt(4 q R / (pi E*))`, `q = (4/3) tau_ilss h`,
    `E* = E33 / (1 - nu23^2)`: 0.77 mm, 0.18 h, for the T800 preset);
    `load_width=` and `nose_radius=` set it. A line load would be
    singular and keep the compliance from converging.
  - `BoundaryHandler.ilss_bcs(applied_load, *, geometry=None)` keeps its
    positional signature and delegates to `flexure_bcs(procedure='3pt')` at
    the mesh's `ilss_geometry` (or `geometry=`); `BoundaryHandler.
    ilss_geometry()` is new. **Behaviour change:** on a mesh without an ILSS
    geometry, `ilss_bcs` and `FESolver.solve('ilss')` raise a `ValueError`
    saying how to build the specimen (before anything is assembled), instead
    of bending the coupon. Snapping the supports onto the coupon's node
    lines was rejected: on the 30 x 10 production coupon the nearest lines
    give a span of 4.17 h and a knockdown 0.0044 above the specimen's at
    uniform 6 % (6 % of the stiffness loss), and the span would change with
    the mesh. Setting `mesh.ilss_geometry` by hand opts a mesh in. The old
    bend is closest to `'flexure_3pt'` on
    `CompositeMesh.for_flexure(field, material, span=50.0, width=20.0,
    overhang=0.0)`, on rollers instead of pinned edges.
  - The interior failure quantities of an `'ilss'` solve use the specimen's
    shear spans: at least `interior_margin` (default h / 2) from both
    support lines, the band edges and the side faces, overhangs excluded,
    so the contact stresses under the nose and at the supports are left
    out. `effective_modulus` stays `None` (about half the deflection is
    shear at this span).
  - `short_beam_shear(results, mesh)` / `ShortBeamShear` (new, public):
    the D2344 formula `0.75 P / (b h)` at the applied load, the FE
    width-averaged mid-plane `tau_xz` at the quarter-span points and its
    interior peak, the first-ply-failure loads (domain, interior) and the
    short-beam strengths D2344 would report at them. At a span of four
    thicknesses the profile is flatter than the parabola behind the
    formula: the mid-plane shear is 0.908 (UD) and 0.915 (QI) of it on the
    pristine T800 specimen, converged to 0.2 % between element lengths h / 4
    and h / 8, against 0.997 (UD) at 16 thicknesses.
  - The app solves ILSS on the specimen (`nz` from the mesh setting; the
    result's `mesh` is the specimen, with new `fe_specimen` and
    `fe_short_beam` entries), says so in the sidebar and the overview, and
    shows the short-beam metrics in the Results tab. `refinement_study`
    refines specimens (default base 24 x 8 x `n_plies`, `nx` a multiple of
    6; explicit levels that break that raise). The distribution example
    solves ILSS on the specimen.
  - **Result change** for every FE `'ilss'` solve. T800 QI, from the pinned
    bend of the 30 x 10 x 24 coupon to the default specimen
    (24 x 8 x 24), `P = -10 N`:

    | case | knockdown | first-ply-failure load factor (interior) | apparent `F_sbs` at interior first-ply failure | peak `abs(tau_13)` |
    |---|---|---|---|---|
    | uniform 3 % | 0.9812 -> 0.9654 (-0.0158) | 174 (383) -> 248 (373) | 32.7 -> 72.6 MPa | 0.502 -> 0.298 MPa |
    | uniform 6 % | 0.9625 -> 0.9314 (-0.0311) | 169 (376) -> 243 (370) | 32.1 -> 72.0 MPa | 0.495 -> 0.300 MPa |
    | midplane-clustered 3 % | 0.9829 -> 0.9666 (-0.0163) | 178 (386) -> 252 (371) | 32.9 -> 72.2 MPa | 0.507 -> 0.299 MPa |
    | interface penny 3 % | 0.9573 -> 0.9202 (-0.0371) | 173 (374) -> 249 (378) | 31.9 -> 73.5 MPa | 0.498 -> 0.311 MPa |

    The short beam is shear dominated and porosity degrades the transverse
    shear stiffness faster than the bending stiffness, so the knockdown
    drops (the planning estimate was about -0.04). The load factors belong
    to different specimens (8.8 mm instead of 20 mm wide), so compare the
    apparent strength `0.75 x load / (b h)`: the old bend failed first at
    its pinned edges and load line, far below the shear the specimen now
    carries. The domain maximum failure index at 10 N rises 78-93 %
    (narrower specimen); the peak `abs(tau_13)` drops from 5.9 to 1.5 times
    `0.75 P / (b h)`.
  - Cost: a first ILSS solve with the pristine reference takes 8.5 s
    instead of 17 s (16,875 instead of 25,575 degrees of freedom), a repeat
    0.6 s instead of 0.9 s. A timing guard for it joins
    `tests/test_fe_performance.py` (budget 25 s / 3 s).
  - Unchanged: every other FE mode, bit for bit (displacement, stress,
    strain, reactions and failure fields hashed before and after for
    compression, tension, shear, flexure on coupon and `for_flexure`
    meshes, thermal and combined, both elements); the empirical path and
    the validation MAE (4.49 % / 3.89 %); the README distribution table
    (compression).
  - Tests: `TestILSSBeamTheoryValidation` is replaced by
    `tests/test_fe_ilss.py` (specimen geometry, rollers, symmetry, the
    quarter-span shear against `0.75 P / (b h)` with a tolerance from the
    mesh study, the long-span limit, the interior region, the outputs, the
    pristine-reference key, coupon meshes raising); generic solver tests
    that used `'ilss'` as a second load case on a coupon run it on a small
    specimen or use `'flexure_3pt'`.
- **The production FE mesh resolves every ply: 30 x 10 x `n_plies`.**
  `_DEFAULT_MESH_RES` is now `(30, 10, None)`; `nz = None` means one element
  layer per ply, resolved by `_default_mesh_resolution(n_plies)` and capped
  at 48 layers (14,400 elements; a direct FE solve on it peaks at 1.3 to
  2.3 GB), so a thick custom laminate does not get an unbounded default
  mesh; above the cap the plies are not resolved and an INFO line says so.
  `build_empirical_pipeline`, `compare_configurations`,
  `sweep_configurations`, `porosity-analyze`, the app, the examples and
  `refinement_study` (whose default base keeps `nz = n_plies` past the cap,
  under its memory guard) all take it from there. The 24-ply presets get
  30 x 10 x 24 (7,200 elements, was 3,600), the 16-ply presets
  30 x 10 x 16 and `CF_PEEK` 30 x 10 x 8. The former `nz = 12` resolved no
  preset, so the FE laminate was not the requested one: the 24-ply QI
  layup became `[90/-45/45/0]_3`, which is not symmetric.
  - App: the default `nz`, and the Expert-mode slider's starting value, is
    the ply count of the layup as entered, and follows layup edits until
    the user moves the slider. The thermal "Use nz = N" hint now appears
    only for a hand-set `nz` or a layup above 48 plies. The app solves the
    knockdown's pristine reference before the porous model, so only one
    factorization is alive at a time: an analysis on the default mesh
    peaks at about 0.75 GB instead of 1.2 GB (0.6 GB on the former
    30 x 10 x 12 mesh), and at about 1.05 GB with the thermal residual
    stress, which holds two.
  - **Result change** for FE solves on the default mesh (the app, or
    `FESolver` on a mesh from `build_empirical_pipeline` without
    `mesh_res`). T800 QI (uniform `Vp` 3 % and 6 %, midplane-clustered and
    interface-penny 3 %), 30 x 10 x 12 to 30 x 10 x 24:

    | quantity | change |
    |---|---|
    | compression / tension knockdown | -4e-5 to +7e-4 |
    | compression / tension `effective_modulus` | +1.1 to +1.2 % |
    | shear knockdown | -1.0e-3 to 0 |
    | ILSS knockdown | -0.0011 to +0.0114 |
    | compression max failure index (interior) | -42 to -43 % (-19 to -20 %) |
    | tension max failure index (interior) | -25 to -30 % (-3 to -6 %) |
    | compression / tension first-ply-failure load factor | +40 to +42 % / +23 to +30 % |
    | ILSS max failure index | -52 to -56 % |
    | ILSS first-ply-failure load factor (interior) | +5 to +8 % (+62 to +66 %) |

    The new values are closer to the mesh-converged ones: in the
    refinement study of the theory page ("Mesh convergence", "Production
    resolution") the 30 x 10 x 24 knockdowns are within 2e-4 of the finest
    mesh and the interior compression failure index (1.195) is 6 % above
    its extrapolate (1.126), against 1.477 on 30 x 10 x 12. On the 16-ply
    `T300_934_epoxy` (uniform 3 %) the compression modulus drops 13.8 %
    (its 12 element layers were 33 % 0 deg plies instead of 25 %) and the
    knockdown by 9e-4; on `CF_PEEK` (8 plies) the modulus rises 0.8 %.
  - Cost: a first FE solve on the default mesh, pristine reference
    included, takes about three times as long (12.5-19 s instead of
    4-7 s on two four-core machines), a repeat solve 0.7-1 s instead of
    0.25-0.4 s, and one sparse LU factorization holds about 320 MB instead
    of 120 MB.
  - Unchanged: every empirical result. The failure loads use the
    specimen-average `Vp`; only the per-node `nodal_knockdown` field (for
    plots) is sampled on the finer mesh. The validation MAE stays
    4.49 % / 3.89 % (`validate_porosity` builds its own
    10 x 5 x `n_plies` meshes).
  - The FE timing guard (`tests/test_fe_performance.py`,
    `POROSITY_FE_BENCHMARK=1`) measures the default mesh; its budgets go
    from 20 s / 2 s to 40 s / 3 s (first / repeat solve).
  - The README FE distribution table is regenerated: its example
    (`examples/distribution_comparison.py`) now uses 12 x 4 x 24, one
    element layer per ply (compression knockdowns 0.9924, 0.9925, 0.9928,
    0.9819).
- **The incompatible-mode element `'hex8i'` is the default FE formulation
  (IMPROVEMENT_PLAN 3.6, step A2).** `FESolver`, `GlobalAssembler`,
  `build_element_batch` and `Hex8Element` now default to
  `formulation='hex8i'`, so the pipeline, the Streamlit app (which shows the
  formulation under the stress plot) and the examples pick it up; the plain
  brick stays available as `formulation='hex8'`, which reproduces the
  previous results bit for bit. `FieldResults` built directly, and the JSON
  export of a result without the field, record `'hex8i'`. **Result change**
  for every FE solve that did not pass `formulation=`. Measured on T800 QI
  (uniform `Vp` 3 % and 6 %, midplane-clustered and interface-penny 3 %),
  `'hex8'` to `'hex8i'`:

  | quantity | 30x10x12 (production) | 30x10x24 |
  |---|---|---|
  | compression / tension / shear knockdown | -2e-5 to 0 | -5e-5 to 0 |
  | ILSS knockdown | +0.0006 to +0.0034 | +0.0007 to +0.0019 |
  | ILSS peak `abs(tau_13)` | -45 to -47 % | -34 to -36 % |
  | ILSS first-ply-failure load factor | +17 to +25 % | +13 to +15 % |
  | compression max failure index | -13.3 to -13.7 % | -7.7 to -11.5 % |

  The tension maximum failure index drops by 1.4-2.8 % (30x10x12) and
  3.1-3.5 % (30x10x24); `effective_modulus` in compression and tension by
  about 0.1 %; shear is unchanged. Cost: the per-element condensation adds
  about 0.06 s per assembly at 3,600 elements and 0.18 s at 7,200; a first
  solve on a new mesh assembles twice (porous and pristine reference), so
  it takes 0-7 % longer depending on the machine (4.1 s for both elements
  here at 30x10x12, best of five); repeat solves and memory are unchanged.
  The README FE distribution table (12x4x12) is regenerated: uniform
  0.9922 -> 0.9921, interface 0.9821 -> 0.9820. The empirical path,
  including the validation MAE, is unchanged. Tests: the Hashin ILSS
  delamination check and the distribution-contract FE checks now use
  ply-resolving meshes (`nz = 24`), because on the earlier ply-merging
  meshes the lock-free element changes which failure mode governs (matrix
  tension in the outer 90 deg element layer) and shrinks the
  uniform-to-clustered contrast in the corner failure index below the
  test's 0.1 % threshold.
- **Dirichlet boundary conditions are eliminated exactly instead of by a
  penalty (IMPROVEMENT_PLAN 1.6).** `FESolver` solves
  `K_ff u_f = F_f - K_fc u_c` with the prescribed values set exactly, in
  place of adding `1e6 * max(diag K)` to each constrained DOF.
  - Boundary values hold bit for bit (the penalty left a slack of about
    1e-8 of the displacement), and the free-DOF stiffness keeps its
    physical conditioning: the diagonal ratio on the production mesh drops
    from 2.4e7 to 24.
  - Direct-solver results move very little. On the 30x10x12 production
    mesh (clustered, uniform and interface porosity, QI and UD, all four
    loadings) the changes are:
    - displacements: at most 4e-8 of `max|u|`;
    - stresses: at most 3e-7 of `max|sigma|`;
    - `effective_modulus`: +7e-9 to +8e-8, always up, because the penalty
      springs absorbed part of the prescribed displacement;
    - `knockdown`: at most 6e-9;
    - `first_ply_failure_load_factor`: at most 1.3e-7;
    - `max_failure_index`: at most 2e-7 (1.1e-6 for the near-zero UD
      tension index).

    The README tables, the JSON schema and the validation MAE (7.05 % /
    6.53 %) are unchanged.
  - `reaction_forces` are `K u - F` at the constrained DOFs and exactly
    zero at the free DOFs, where they used to carry about 1e-10 N of solver
    residual.
  - A load at a constrained DOF now enters that support's reaction.
    `apply_penalty` overwrote `F` there, silently dropping the load; none
    of the shipped load cases does this.
  - The LU factorization is cached per constrained-DOF set. CG and MINRES
    reuse the cached free-DOF matrix and never factorize.
  - A singular free-DOF stiffness (boundary conditions that leave a
    rigid-body mode free) raises an error that says so.
  - The `cond_diag_ratio` INFO line and the "conditioning near float64
    limit" warning are replaced by one INFO line on the free-DOF diagonal
    ratio.
- **`FESolver` keeps two sparse LU factorizations instead of one.** The
  free-DOF systems (`K_ff` and its LU) of the last two constrained-DOF sets
  are cached, least recently used evicted first, so alternating load cases
  with different supports on one solver (shear and ILSS, compression and
  ILSS) factorize each once instead of on every switch: on the production
  mesh each alternating shear / ILSS solve takes 0.7-1.0 s instead of
  4.7-5.4 s. A third set evicts the least recently used before its own
  factorization is built, so at most two are alive at once, and a new
  stiffness matrix (mesh, material, porosity or formulation changed) clears
  the cache as before. Memory: one factorization is about 320 MB on the
  production mesh (30 x 10 x 24; 120 MB on 30 x 10 x 12), so a solver that
  alternates two sets now holds about 650 MB while it lives. Results are
  unchanged.
- **`solver='minres'` is deprecated.** `FESolver.solve(solver='minres')`
  and `refinement_study(solver='minres')` (once per study, not per level)
  emit a `DeprecationWarning` that points to `solver='cg'`; MINRES still
  works and will be removed in 2.0. With the prescribed DOFs eliminated,
  the reduced stiffness `K_ff` is symmetric positive definite, so CG covers
  every case, and MINRES, which minimizes the residual rather than the
  energy-norm error, is less accurate at the same `rtol`. On the
  production mesh (30 x 10 x 24, T800 QI, clustered `Vp = 0.03`, default
  `rtol`) the displacements of MINRES are 3.6e-7 (compression), 1.7e-6
  (shear) and 7e-10 (ILSS) from the direct solve, those of CG 5e-9, 1e-8
  and 1e-11. The tests that cover MINRES now also assert the warning.
- **`FESolver.solve(penalty_factor=..., diag_scale=...)` are deprecated
  and have no effect.** Both now default to `None`, and passing any value
  emits a `DeprecationWarning`. `BoundaryHandler.apply_penalty` still
  returns the penalty-modified system but warns. Use the new
  `BoundaryHandler.apply_elimination(K, F, constrained)`, which returns
  `(K_ff, rhs, free_dofs)`. All three will be removed in 2.0.
- **Empirical layup scaling replaced (results change; IMPROVEMENT_PLAN 2.7,
  substance of #139 / #140).** The binned matrix-dominated fraction rule
  (`alpha_QI × f_md / 0.5`, floored at 0.15, or 0.80 for ILSS and
  transverse tension) is gone.
  - `shear`, `ilss` and `transverse_tension` are now layup-independent
    (scale 1). They are ply or interlaminar matrix properties.
  - `tension` and `compression` use `scale = max(1, alpha_blend /
    alpha_QI)`. `alpha_blend` weights the calibrated per-mode alphas by the
    fiber / transverse / shear split of the pristine CLT membrane strain
    energy under a unit `N_x` load (new private `porosity_fe._layup`).
    The rule adds no fitted constant.
  - The scale applies to all three laws, to user overrides and to
    `local_sensitivities`.
  - The default QI layup gives bit-identical results for every mode and
    law, and so does every in-plane isotropic layup.
  - UD, cross-ply, the Elhajjar `[0/45/90/-45/0]_s` coupon, `[0_2/90]_s`
    and `[0/±15]_s` now use the QI coefficients unscaled. This is the
    conservative direction: UD tension KD at `Vp = 0.05` drops from 0.971
    to 0.823, and UD compression from 0.950 to 0.708.
  - Layups more matrix-dominated than QI under an x load are amplified.
    `[±45]_2s` tension is ×1.94 and compression ×1.14; `[90]_8` tension is
    ×2.56 and compression ×1.16. Their matrix modes are no longer
    amplified (they were ×2.0 for `[90]_8`).

  The amplification has no coupon data behind it. It is recorded in
  `FailureResult.details` (`'layup_scale'`, `'layup_extrapolated'`) and
  flagged with one `UserWarning` per `get_failure_load` /
  `apply_loading` / `get_all_failure_loads` call.
  `propagate_uncertainty` reports it once, from the nominal solve, rather
  than once per draw. New `EmpiricalSolver.layup_scale` and
  `EmpiricalSolver.matrix_energy_fraction` attributes expose the values.
  The app's knockdown chart footnote shows the amplification in place of
  `f_md`; the result dict carries `layup_scale` instead of `f_md`.

  Validation MAE:
  - Property-weighted: 7.05% → 4.49%.
  - Point-weighted: 6.53% → 3.89%.
  - Worst entry: 22.64% → 16.22%.
  - By property: ILSS 4.9% → 4.1%, tensile strength 8.2% → 1.9%,
    compression 11.4% → 2.6%, shear strength 13.5% → 4.5%, transverse
    tensile strength 7.4% → 6.6%. Moduli are unchanged.

  20 of 35 entries move. 17 improve, for example
  `wen_2023:shear_strength` 22.64% → 6.10% and
  `olivier_1995:tensile_strength` 15.67% → 1.19%. Three get worse:
  - `olivier_1995:ilss`: 1.55% → 9.21%;
  - `stamopoulos_2016:transverse_tensile_strength`: 2.49% → 4.44%;
  - `stamopoulos_2016:ilss`: 2.52% → 3.67%.

  These two papers imply a scale of 0.56 to 0.76, below every other UD
  paper. The MAE pins in `tests/test_validation_database.py` are updated.
- **Deprecated: `Calibration.F_MD_REF`, `F_MD_FLOOR`, `F_MD_FLOOR_ILSS`**
  and the `EmpiricalSolver._F_MD_*` aliases. They keep their old values
  but no longer have any effect. Reading one emits a `DeprecationWarning`,
  and they will be removed in 2.0. `EmpiricalSolver.f_md` stays as a
  legacy descriptor and no longer drives the scaling.
  `Calibration.MATRIX_DOMINATED_MODES` is unused but kept.
  `Calibration.FIBER_DIRECTION_MODES` and
  `Calibration.LAYUP_MATRIX_ANCHORS` describe the new rule.
- **Provenance short aliases are deprecated (IMPROVEMENT_PLAN 4.7).** The
  canonical keys are `porosity_fe_version`, `python_version`,
  `numpy_version`, `scipy_version`, `timestamp_utc` and `git_commit`. The
  #55 short aliases (`package_version`, `python`, `numpy`, `scipy`,
  `generated_utc`, `git_sha`) are still written with the same values, so
  output is unchanged. The results schema no longer requires them, marks
  them `deprecated`, and they will be removed in schema 2.0.
- **`porosity-analyze --applied-stress` warns that it has no effect.** The
  empirical sweep never used it; passing it explicitly now logs a warning.
- **Packaging (IMPROVEMENT_PLAN 6.7).**
  - The `all` extra is now `porosity-fe[web,dev,docs]` instead of a
    hand-copied list.
  - A test fails if `requirements.txt`, `requirements-web.txt` or
    `requirements-test.txt` drifts from `pyproject.toml`. The files stay
    because the executable build, the security audit and the Streamlit
    deployment guide use them.
  - `CITATION.cff` ships in the sdist.
  - `ValidatePorosity.spec` refuses to build when it finds no validation
    datasets, instead of producing an executable with none, and bundles
    every `porosity_fe` submodule.
- **`effective_porosity_profile(nz, x=25.0, y=10.0)`** takes the in-plane
  sampling location, which was hard-coded to the domain center.
- **G23 degradation shares the G12 ratio explicitly.** Both used the same
  Halpin-Tsai form with identical inputs; the duplicate computation is
  removed and the equality is documented. Results are unchanged.
- **Percent inputs are labelled as percent.** The only porosity inputs that
  take a percent now say so where the value is asked for:
  - the app field is "Void content Vp (%)", with help "3.0 means 3 % voids";
  - `recommend_disposition`'s first parameter is renamed `Vp` ->
    `Vp_percent`. It is documented and rejects values outside `[0, 100]`;
    positional callers are unaffected, but `Vp=` keyword callers must switch
    to `Vp_percent=`;
  - the README conventions section lists all three percent inputs
    (the app field, `--vp-pct`, `recommend_disposition`).

  The uncertainty CoV inputs (CLI and app) now state that they are
  fractions of the value (0.10 = 10 % of Vp, not 10 percentage points).
- **Dedup and cleanup (IMPROVEMENT_PLAN 5.4).** All five JSON writers
  build their envelope through one `io._wrap_envelope()` (output is
  unchanged). `compute_clt_effective_modulus` reuses `_build_clt_abd`
  (bit-identical). The new `CompositeMesh.mid_y_section_indices()` /
  `mid_y_element_indices()` replace three copies of the cross-section
  index loop in the app and `FEVisualizer`. `validate_all.py` has one
  import block and no longer carries the empty
  `_UNSUPPORTED_STRENGTH_PROPS` guards. `FEVisualizer` `save_path`
  parameters are annotated `str | os.PathLike | None`.
- **`recommend_disposition` rejects an unknown `structural_class`** with
  `ValueError`. It used to substitute `"primary"` silently, which in an
  MRB record misstates the substantiation basis.
- **`resolve_material(strict=...)` is deprecated.** It had been a no-op
  since #34; passing it now emits a `DeprecationWarning`.
- **`sigma_2c` scope documented (IMPROVEMENT_PLAN 3.1).** The plan asked
  to add a `transverse_compression` empirical mode or document why not.
  There is no porosity dataset for transverse compression to calibrate a
  coefficient on, so the mode is left out. `MaterialProperties`, the
  README scope table and `EmpiricalSolver.PRISTINE_STRENGTH_KEY` now say
  so, and note that the FE criteria already use `sigma_2c` (Tsai-Wu `Y_c`,
  Hashin matrix compression).
- **FE solves are ~5x faster, and repeat solves ~60x faster.** On the
  production mesh (30x10x12, 3,600 elements) a first solve went from
  14 s (clustered porosity) / 9.6 s (uniform) to ~2.6 s / ~2.1 s, now
  bound by the sparse LU factorization. Assembly and stress recovery use
  batched per-element arrays instead of per-element Python objects, and
  recovery reuses what assembly computed. The FE knockdown, penalty BCs
  and mesh-quality check are vectorized. The assembled stiffness and its
  factorization are reused across solves with the same constraints
  (e.g. compression after tension: ~0.23 s), and are rebuilt
  automatically if the mesh, material or porosity change. Results agree
  with the previous implementation to ~1e-11 relative.
- **The validation datasets are tracked in git like any other file.**
  `.gitignore` used to ignore the whole `validation/` tree even though the
  datasets were committed, so `git add` silently skipped a newly added
  dataset. The blanket rule is removed; the generated
  `validation_*_report` files stay ignored.
- **`elhajjar1/PorosityFE` is the canonical repository.** The PyPI project
  URLs (homepage, repository, issues, documentation), the app's README
  link, the docs and the Streamlit deployment guide now point there, in
  line with the README, `CITATION.cff` and `CONTRIBUTING.md`.
- **CI installs the package** (`pip install -e ".[dev]"`) in the lint, test
  and Streamlit jobs, so tests exercise the real packaging, and the
  repo-root `conftest.py` `sys.path` shim is removed. Run tests after an
  editable install.
- **CI measures coverage** on the ubuntu / Python 3.12 test cell and uploads
  `coverage.xml` as a workflow artifact. `pytest-cov` joins the `dev` extra.
- **CI caches pip downloads** in the test, security and executable-build
  workflows, and the executable build installs its runtime dependencies
  from `requirements.txt` instead of a hand-written list.
- **Internal: one implementation per empirical knockdown law.** The
  Judd-Wright, power-law, and linear forms were each written out in four
  places in `EmpiricalSolver`; they now live in one table, and every entry
  point raises the same "Unknown knockdown model" message. Results are
  bit-identical.
- **Internal: each `apply_loading()` / `get_failure_load()` call validates
  a user knockdown callable, and evaluates the hygrothermal and fatigue
  factors, once** instead of twice.
- **Internal: the porosity profile normalization is memoized**, and the git
  commit used in JSON provenance is looked up once per process instead of
  once per file written.

## [1.2.0] - 2026-05-11

### Fixed
- **FE: penny / oblate voids were silently routed to the prolate Eshelby
  formula** because `ar = max/min ≥ 1` made the oblate branch unreachable.
  `_mt_effective_stiffness` now detects the axis of symmetry, uses the
  correct g-function branch (prolate vs oblate), and permutes the Voigt
  tensor to align with the actual void axis. Penny-shaped voids
  (`VOID_SHAPES['penny']`) now produce physically-correct anisotropy
  with through-disk degradation exceeding in-plane degradation. (#32)
- **FE: stiffness assembly now rejects inverted elements** (non-positive
  `det(J)`) at quadrature time rather than letting a wrong-sign block
  silently corrupt the assembled global K. (#33)
- **FE: engineering strain in post-processing is now rotated with
  `strain_transformation_3d`** (was `stress_transformation_3d`), so
  `FieldResults.strain_local` shear components are no longer off by 2x.
  Stress / Tsai-Wu paths were already correct and are unchanged. (#38)
- **GUI: `porosity-fe` console script** now prints a friendly stderr
  message and exits non-zero when PyQt6 is missing, instead of leaking
  a Python traceback. Error text mentions the discoverable
  `pip install porosity-fe[gui]` form. (#46)

### Added
- **CSV export** alongside the existing JSON export in the GUI's
  *File → Export Results* menu. Format is picked from the typed
  extension (`.csv` / `.json`) or the selected filter. CSV layout:
  `#`-prefixed config metadata, then a flat
  `mode,model,failure_stress_MPa,knockdown` table. (#30)
- **`EmpiricalSolver` constructor overrides** for `judd_wright_alpha`,
  `power_law_n`, and `linear_beta` — partial-merge with QI defaults,
  layup-scaled the same way, replacing the documented subclass
  workaround. (#16)
- **Input validation** on `MaterialProperties`, `CompositeMesh`,
  `VoidGeometry`, and `Hex8Element` constructors (positive moduli /
  strengths, Poisson ratios in `(-1, 0.5)`, positive geometry, fraction
  range on `node_porosities`). (#13)
- **Validation reporting** now publishes both property-weighted and
  point-weighted MAE — the published 7.7% is the property-weighted form;
  point-weighted is ~7.0%. (#36) New `summarize_mae()` helper exposes
  both for downstream tools.
- **GUI threading hardening**: `closeEvent` waits for the worker to
  exit; `_stop_requested` uses `threading.Event` for cross-core memory
  fences; the Run button is disabled before `_build_config` runs;
  `_on_stop` warns when the worker doesn't terminate within 2s. (#17)
- **Numerical-stability guards** in Mori-Tanaka inversion (pinv
  fallback + finite check) and Tsai-Wu evaluation (strength-floor clamp,
  non-finite check, fp clip on `elem_Vp`). (#14)
- **Error-message polish** on bad material / void-shape / mode /
  cluster_location names; UTF-8 encoding on all file I/O; floating-point
  noise clip on `Vp ≈ 1.0`; string `Vp` rejected with `TypeError`. (#21, #22, #23)
- **PyInstaller spec parity**: `PorosityFE.spec` now mirrors
  `ValidatePorosity.spec`'s dataset bundling and adds commonly-missed
  hidden imports (`PyQt6.sip`, `PyQt6.QtSvg`,
  `scipy.sparse.csgraph._validation`, `scipy.special.cython_special`). (#24)
- **Documentation**: README "Inputs and Conventions" section clarifies
  Vp as a fraction in [0, 1] with do/don't table; "Defining a custom
  material" worked example; per-ply vs. specimen-average porosity behavior
  for empirical vs. FE paths. (#1, #2, #4)

### Changed
- **Version source of truth**: `validate_porosity --version` now reads
  from `importlib.metadata` with a hard-coded fallback. Version aligned
  across `pyproject.toml`, `CHANGELOG.md`, `CITATION.cff`, README BibTeX,
  and `PorosityFE.spec`. (#45)

## [1.1.1] - 2026-04-19

### Changed
- **Removed `flexural_strength` property** from the validation schema and all
  datasets. The property consistently showed high MAE (8-40%) because 3-point
  bend failure in cross-ply and UD laminates involves mixed compression +
  interlaminar shear mechanisms that the current Judd-Wright mode mapping
  (`compression` proxy) cannot capture. Removing it focuses the validation
  database on properties the model can predict well.
- Overall validation MAE: **9.76% → 7.69%** (35 property-dataset pairs,
  down from 41)
- Affected datasets (6): Almeida 1994, Ghiorse 1993, Liu 2006, Olivier 1995,
  Stamopoulos 2016, Tang 1987 — all retain their other properties

### Added
- CI badge for "Build Executables" workflow in README
- Explicit documentation of model scope and property coverage

## [1.1.0] - 2026-04-19

### Added
- Expanded validation database from 3 to 13 peer-reviewed experimental papers
- Unified JSON Schema (Draft-07) for validation datasets
- 3 new material presets: IM7/8551, T300/934, CF/PEEK
- 3 CLT helper functions: `compute_degraded_clt_moduli`,
  `compute_degraded_clt_flexural_modulus`, `_build_clt_abd`
- Master validation runner `validation/validate_all.py` with strength
  (Judd-Wright) and modulus (CLT) prediction, aggregated MAE report
- Cross-platform CLI executable `validate_porosity` (Linux/macOS/Windows)
- GitHub Actions workflow `build-executables.yml` that builds and releases
  the CLI on Ubuntu, macOS, and Windows runners
- 28 new tests bringing total to 186

### Classical validation datasets added
- Ghiorse (1993) SAMPE Quarterly — AS4/3501-6
- Almeida & Nogueira Neto (1994) Compos. Struct. — 0-10% void range
- Tang, Lee & Springer (1987) J. Comp. Mater. — T300/976
- Bowles & Frimpong (1992) J. Comp. Mater. — IM7/8551-7
- Jeong (1997) J. Comp. Mater. — AS4 fabric
- Olivier, Cottu & Ferret (1995) Composites — T300/914

### Recent validation datasets added
- Liu et al. (2018) J. Comp. Mater. — T300/924, 6 porosity levels
- Zhang et al. (2025) Polymers — CF/PEEK thermoplastic matrix
- Wen et al. (2023) J. Reinf. Plast. Compos. — T700/epoxy + temperature
- Wang et al. (2022) J. Comp. Mater. — CF/epoxy + micro-CT damage evolution

## [1.0.0] - 2026-04-03

### Added
- PyQt6 desktop GUI with interactive porosity analysis
- Empirical strength models: Judd-Wright (exponential) and Power Law correlations
- 3D finite element solver with Eshelby-based stiffness degradation
- 3 porosity distribution types: uniform, clustered (midplane/surface/quarter), interface-concentrated
- 3 void morphologies: spherical, cylindrical (prolate), penny-shaped (oblate)
- 4 loading modes: compression, tension, shear, ILSS
- 3 built-in material presets: T800/epoxy, E-glass/epoxy, T700/epoxy
- Discrete void modeling with stress concentration factors
- Tsai-Wu failure criterion for multiaxial states
- Visualization: porosity fields, 3D meshes, damage contours, knockdown curves
- JSON export of analysis results
- PyInstaller macOS app bundle
- Comprehensive test suite
