# PorosityFE delamination plan: separating delamination from porosity with fracture mechanics

Prepared 2026-10-22 from two investigations:
- a read-only code audit of branch head `2686d62` (PR #208), where all file:line references below point;
- a sourced literature brief.

I reproduced the key audit numbers myself.

**Units throughout:** MPa and mm, so energy release rates and toughness are in N/mm (1 N/mm = 1 kJ/m²).

---

## Status (2026-10-26)

The plan below is kept as written on 2026-10-22; its file:line references
point at commit 2686d62. Progress by pull request:

| Phase | State | PR |
|---|---|---|
| 0.1–0.3 | Done: mode maxima, exports, app criterion selector, docs | #209 |
| 0.4 | Pending your decision (changes `interface` results) | — |
| 0.5 | Folded into phase 1.1 (optional `tau_13` with today's behaviour as default) | — |
| 1.1–1.3, 1.5 | Done: optional `sigma_3t` / `tau_13`, interface traction recovery, Brewer–Lagace / Ye onset with load factor, O'Brien free-edge G | #210 |
| 1.4 | Pending your decision (changes penny results) | — |
| 1.6 | Done: `FieldResults` methods; on request the JSON `delamination_onset` block (`export_results(..., onset=)`), the VTU `onset_index` field and the app's onset check | #210 |
| 2.1–2.8 | Done: `Delamination`, duplicated interface nodes and ties, crack-face contact, VCCT, B-K criterion and load factor | #209 |
| 3.1–3.3 | Done: DCB / ENF / MMB builders and loading modes, MMB lever, compliance calibration, NASA benchmarks within 3 % (SLB 4 %, behind a flag) | #210 |
| 4.1, 4.2, 4.4 | Done: `ToughnessKnockdown` (uncalibrated placeholders, flagged), front toughness from the local porosity, `hakim_2017` and `agius_fox_2015` reported apart | #210 |
| 4.3 | Partly: `toughness_knockdown_factor` screens without FE; no empirical `LoadingMode` | — |
| 4.5 | Not started (uncertainty of `b`) | — |
| 5 | Done: app Defects panel (delamination, toughness and its law, onset, DCB / ENF / MMB for test planning), knockdown decomposition (D6, promised in 2.7), NCR indication types with a fracture assessment of a delamination and no recommended path for it (your decision), `porosity-analyze --delamination`, docs | #210 |
| 6 | Deferred (cohesive zones) | — |

Decisions taken with the recommended option because they change no
existing result: D1 (node duplication and ties), D2, D3, D4 (the
exponential toughness law with flagged placeholder coefficients, on by
default), D5, D6, D7, the optional `tau_13` default, benchmark materials inside the tests, cohesive
zones deferred. Still open, because they change existing results: 0.4
(`interface` profile renormalization), 1.4 (direction-aware degradation /
transversely isotropic Mori–Tanaka tensor) and unifying the `tau_13`
strength.

Two corrections to the text below, from the primary sources: the ENF
benchmark toughness is `G_IIc = 0.78`, `eta = 2.57` (NASA/CR-2011-217305
Table II; 0.774 and 2.1 are the MMB report's), and the DCB benchmark uses
T300/1076 elastic constants with the T300/914C toughness.

One change to the design in Phase 2.5 below: only the mode I product is
clipped at zero; the shear products keep their sign so the front sums to
the released energy (see `docs/theory/delamination.md`).

---

## 1. Short answer: does the software tell delamination from porosity?

**No.** PorosityFE models porosity only. It has no crack, no interface entity, no traction-free surface, and no fracture-mechanics quantity. The README says so in one line: "Delamination initiation/propagation is not explicitly simulated", README.md:668.

Several features look like delamination; none of them is one:

| What looks like delamination | What it really is | Evidence |
|---|---|---|
| **Hashin "delamination" mode** | A stress-based *initiation index*, `(<σ33>/Yt)² + (τ13²+τ23²)/S23²`, evaluated at ply Gauss points. It uses Yt = `sigma_2t` (there is no Zt) and S23 = `tau_ilss`. It is reported only at the Gauss point of the global maximum index, not as a maximum over the model. The app always uses Tsai-Wu, so it never shows this mode; the pipeline and CLI are empirical-only. | fe/failure.py:451-453, 130-133, 263-274; app.py:495 |
| **`interface` profile with `penny` voids** | Smeared porosity: Gaussian peaks at ply interfaces, Mori-Tanaka with oblate (3, 3, 0.3) voids. The Mori-Tanaka tensor is projected onto an isotropic one before the Halpin-Tsai ratios are applied, so **the penny's out-of-plane weakening is averaged away**: at Vp = 0.03 E22 and E33 drop by the same factor, 0.885 (reproduced). On the production mesh (nz = n_plies) the nodes sit on the Gaussian peaks, so the FE element-mean Vp is **0.0354 for a nominal 0.03, +18 %** (reproduced). | porosity_field.py:18-33; homogenization.py:265-328 |
| **Discrete `VoidGeometry`** | A thin interface disc either sets Vp = 1 at nodes (half-degrading the two adjacent plies) or, once it is thicker than t_ply/2, turns two plies into a 1 MPa soft hole. Neither is a zero-thickness crack, and the behaviour jumps at c = t_ply/2. | mesh.py:804-813; element.py:26, 177 |
| **Empirical ILSS knockdown** | A porosity-only strength law. Distribution and void shape have no effect: interface-penny and uniform-spherical both give 0.7408 at 3 %. | empirical.py:113-124, 296-300 |
| **Validation database** | 13 datasets: 9 ILSS, 7 tensile, 5 flexural modulus, and others. **None has fracture-toughness or delamination data.** | validation/datasets/*.json |
| **NCR disposition** | Bins only on Vp % and knockdown; it never separates a delamination NDI indication from porosity. | reporting.py:249-358 |
| **Free edges** | The classic free-edge delamination site is deliberately excluded from the interior metrics, as noise. | fe/_load_cases.py:132-157 |

**What a delamination needs that a porosity model cannot give:**
- a zero-thickness traction-free surface at a specific interface;
- a crack-tip singularity, and so energy release rates G_I, G_II and G_III;
- a growth criterion against mode-mixity-dependent toughness G_c(ψ);
- sensitivity to the delamination's size, depth (which interface) and the adjacent ply angles.

In this plan, porosity's role becomes twofold: it **modifies the toughness** of the interface, and voids act as **initiators**.

**Good news from the audit.** The existing constraint machinery already supports a linear fracture-mechanics (LEFM) path. `_ConstraintReduction` (u = T·u_r + g, with master-slave ties) can bond a duplicated interface everywhere except over the delamination. A throwaway probe script did exactly this with **no library edits** on a 16-ply UD DCB, 9.9k DOF, in 1.8 s:
- VCCT G_I = 299 J/m²;
- compliance derivative over one element = 311 J/m² (4 % apart; reproduced).

Growing the crack only changes ties, so K stays the same, and the LU cache already keys on ties (assembler.py:362-364; fe/_constrained_solve.py:231-254).

---

## 2. Goals and non-goals

**Goals**
1. **Explicit delamination defect**, separate from porosity: which interface, planform shape and size, position (typically from an NDI indication).
2. **Energy release rates by VCCT** (G_I, G_II, G_III along the front) on the existing linear solver, and a **critical load factor** from a mixed-mode toughness criterion (Benzeggagh-Kenane).
3. **Fracture test specimens** DCB, ENF and MMB (D5528, D7905, D6671), verified against published NASA benchmarks.
4. **Porosity-toughness coupling**: a G_c(Vp) knockdown, using the *local* interface porosity under the front, with honest uncertainty.
5. **Better stress-based onset**: interface tractions instead of ply-interior Gauss points, proper Zt and S13 inputs, the Brewer-Lagace quadratic criterion with an averaging length, exposed in the app and exports.
6. **Disposition**: separate porosity and delamination indications in the NCR workflow, including the combined case (a delamination in a porous laminate).
7. **Verification and validation**: benchmark suite, toughness-vs-Vp datasets, mesh-convergence support.

**Non-goals for this workstream (deferred, see Phase 6)**
- Cohesive-zone nonlinear propagation: R-curves, unstable growth, snap-back.
- Fatigue delamination growth: the porosity data are thin and conflicting.
- Delamination buckling under compression (sublaminate buckling, compression after impact): needs at least a linear eigen-buckling capability.
- Impact damage.

---

## 3. Design decisions (recommendation first)

**D1. Crack representation.**
- **(A, recommended)** Duplicate the full plane of interface nodes once, append the duplicates after the structured nodes, and rewire the element layer above. The delamination is then "the set of untied node pairs"; the bonded region is tied through `_ConstraintReduction`.
  - Topology and K are the same for every crack size, so only ties change.
  - It stays linear and reuses the LU and tie machinery, as the probe showed.
- (B) Zero-thickness interface elements. These are needed only for cohesive zones (Phase 6). Design (A) so that (B) can slot in later.

**D2. Structured-index helpers.** Several helpers assume `len(nodes) == (nx+1)(ny+1)(nz+1)`: `flexure._structured_grid`, `mesh.ply_ids`, `mid_y_section_indices`, and the `viz` slices.
- **Recommended:** keep the original nodes at indices 0..N0−1 and append the duplicates, as the probe does. Add `CompositeMesh.n_structured_nodes` and have the helpers slice by it.
- Builders that call `generate_mesh()` again (`for_flexure`, mesh.py:585-587) must apply the duplication *after* generation.

**D3. Toughness data model.**
- New optional `MaterialProperties` fields, default `None`, like the CTEs: `G_Ic`, `G_IIc`, `G_IIIc` (defaults to `G_IIc`), `bk_eta`, plus `sigma_3t` (Zt) and `tau_13` (S13).
- A fracture analysis without toughness raises with guidance, as `delta_T` does without CTEs.
- Add the new fields to the `__post_init__` checks and `PERTURBABLE_FIELDS` (materials.py:304-312, 611-615).
- Populate presets only from sources (Appendix A). Leave the rest `None`.

**D4. Porosity-toughness law.**
- The literature gives **no published Vp-G relation**. Only about ten studies measured both, and they disagree:
  - initiation G_Ic falls;
  - G_IIc falls;
  - propagation G_Ic can *rise* through bridging (Mehdikhani et al. 2019 review).
- **Recommended:** an exponential knockdown G = G₀·exp(−b·Vp), separate for initiation and propagation and for each mode.
  - Defaults are literature-derived placeholders: b_I ≈ 5–7.5, b_II ≈ 13–16. These are two-point estimates from Hakim 2017, Olivier 2007 and Agius & Fox 2015, flagged **uncalibrated**.
  - The law never increases G (conservative).
  - Users can override b.
  - Calibrate as datasets arrive (Phase 4).
- Alternative: no default law until there are data, so the toughness knockdown is user-supplied only.

**D5. Mesh at the crack front.**
- Krueger's VCCT benchmarks use C3D8I bricks, the same family as `hex8i`, with Δa ≈ 0.25–1.0 mm and equal element lengths either side of the front. One element per ply through the thickness is adequate.
- The production in-plane mesh is far too coarse at a front: Δx ≈ 1.7 mm on the 50 mm coupon.
- **Recommended:** optional *structured grading*, i.e. non-uniform x and y node coordinates with a refined band around the front, which keeps every structured helper working. Specimens (Phase 3) use uniform fine meshes.
- Default Δa = one ply thickness at interfaces between dissimilar plies (oscillatory singularity: G_I and G_II depend on Δa while G_T converges, Raju et al. 1988). Document it and report G_T.

**D6. Knockdown semantics with a delamination.**
- The pristine reference is free of both porosity and delamination.
- Report the decomposition KD_total, KD_porosity-only and KD_delamination-only, so users see which defect drives the loss.

**D7. Crack-face contact.**
- A linear solve can let the crack faces interpenetrate (ENF, compression, shear).
- **Recommended:** a small active-set contact loop on the untied pairs, a few linear solves, with **no new factorization of K** (only T changes, which already carries a cache key).
  1. Tie u_z wherever the opening goes negative.
  2. Iterate until no pair is in tension-tie or penetration.
- Faces slide frictionlessly.

---

## 4. Phased plan

Each phase is one or two PRs. Every PR runs the established gate: full suite, numpy-2 suite, timing guard, ruff, mypy, `sphinx -W`, validation MAE, and the 132-case FE fingerprint against master, so result changes are always intentional and listed.

### Phase 0: Correctness and honesty fixes (S, 1 PR, no new physics)

| # | Change | Result changes? |
|---|---|---|
| 0.1 | Docs and examples say exactly what exists: Hashin onset index only, not in the app. `examples/interface_penny.py` claims "worst-case ILSS morphology" but runs only the empirical solver, where shape and distribution do nothing: run the FE or reword it. Stop calling penny voids "crack density" (CHANGELOG:709, test_homogenization.py:235). Fix the "Full 3D Tsai-Wu" label on a 2D formula (README.md:615-618). | No |
| 0.2 | Report the **maximum of every failure mode over the model** (`FieldResults.failure_mode_maxima`) and a per-element delamination index in the VTU. Rename the VTK field `tsai_wu_index` → `failure_index`, keeping the old name as an alias. Add a JSON `max_failure_index` next to the legacy `max_tsai_wu_index`. | Additive only |
| 0.3 | App: failure-criterion selector (Tsai-Wu, Hashin, max-stress), so the delamination mode is visible, plus a "governing mode" readout. | No (default unchanged) |
| 0.4 | **`interface` profile inflation (+18 % on nz = n_plies).** Renormalize the sampled field so the FE volume average equals the requested Vp, or sample element averages. The distribution-contract test then has to assert the physics, not the sampling artefact. | **Yes**, `interface` configs. Needs your go-ahead |
| 0.5 | τ13 strength inconsistency: Tsai-Wu and max-stress check τ13 against `tau_12`, Hashin against `tau_ilss`. Add optional `tau_13`; when it is `None`, keep today's behaviour for each criterion and log the inconsistency once. | No by default |

### Phase 1: Stress-based delamination onset done right (M, 1–2 PRs)

- **1.1 Strength inputs.** Optional `sigma_3t` (Zt) and `tau_13` (S13), with documented fallbacks (Zt ← `sigma_2t`; S13 ← `tau_ilss` for the interlaminar criteria).
- **1.2 Interface traction recovery** (`fe/interlaminar.py`, new). At every ply interface:
  - Extrapolate the Gauss-point stresses of both adjacent elements to the shared face and form t = σ·n = (τ13, τ23, σ33).
  - Average the two sides, and report the jump as a resolution indicator.
  - This replaces today's per-ply nodal extrapolation, which never reconciles the two sides (fe/recovery.py:169-187).
- **1.3 Onset criteria on interfaces.**
  - Brewer-Lagace quadratic criterion, Σ(σ̄ᵢⱼ/Zᵢⱼ)² = 1, with separate tension and compression allowables. The tractions are averaged over a length x_avg, default one ply thickness (Kim & Soni), stated as a calibration parameter. The Ye criterion is an option.
  - Outputs: maps of the onset index per interface, the critical interface and location, and an onset load factor. The factor is closed form for a mechanical load; with thermal prestress it is quadratic in λ, through the existing `_positive_root_offset`.
- **1.4 Direction-aware porosity degradation of interlaminar strengths.** Today every matrix strength scales by √(C11*/Cm11), which ignores direction. Use the C33 ratio for Zt and C55 for S13.
  - This connects to a separate decision: keeping the transversely isotropic Mori-Tanaka tensor instead of the isotropic projection, so penny voids degrade E33, G13 and G23 more than in-plane properties.
  - That changes results for every penny configuration, so it is listed as a decision, not a default.
- **1.5 Free-edge delamination.**
  - O'Brien's closed form, G = (ε²·t/2)·(E_LAM − E*), in the CLT/empirical path. It is independent of delamination size; O'Brien found G_I controls static onset.
  - An FE cross-check: free-edge interface tractions, and the energy release rate once Phase 2 lands.
  - The O'Brien edge-delamination specimen is a ready verification case.
- **1.6 Outputs.** New optional `FieldResults` fields (interface-traction summary, onset index per interface, onset load factor). JSON schema 1.2 → 1.3, keeping strict JSON with finite values only. VTU interface fields. An app panel "Interlaminar stresses and delamination onset".
- **Verification:**
  - traction continuity at interfaces under refinement;
  - free-edge σ33 and τ13 profiles against published free-edge solutions;
  - onset load factor against hand calculations on a two-ply case;
  - `refinement_study` quantity kinds for the onset index.

### Phase 2: Explicit delamination and VCCT (L, 2–3 PRs; the core)

- **2.1 `Delamination` defect (public).**
  - Interface index k (between plies k and k+1).
  - Planform: through-width strip (2-D-like, for specimens), ellipse or rectangle in x-y, given by centre and size.
  - Validation: inside the plate, nz a multiple of n_plies so the interface is a node layer, no overlap with other delaminations at the same interface.
  - Lives beside `VoidGeometry` in the defect inputs; porosity stays a separate input.
- **2.2 Mesh support.**
  - Duplicate the interface plane, append the duplicates and rewire the elements above (D1, D2).
  - Copy the porosity to the duplicates.
  - Duplication survives `generate_mesh`, and `n_structured_nodes` keeps the helpers working.
  - Optional structured grading around fronts (D5).
- **2.3 Ties in `FESolver`.**
  - Thread `ties` through `apply_boundary_conditions` → `solve_constrained` → the pristine reference (`fe/_knockdown.py`) and the thermal unit solve (solver.py:624-635). Today the reduction is built without ties (fe/_constrained_solve.py:162).
  - Resolve the face-BC conflict: tied duplicates on a prescribed face raise "both tied and prescribed" (assembler.py:397-400). Prescribe the originals only, and let the ties carry the duplicates.
- **2.4 Crack-face contact** (D7): active-set normal ties on untied pairs. It is needed for ENF and for any compressive or shear loading, and is reported in the results (number of contact iterations, contact area).
- **2.5 VCCT post-processing** (`fe/vcct.py`, new).
  - Front detection: tied pairs adjacent to untied ones.
  - A local frame per front node: x′ = in-plane growth direction normal to the front, y′ = interface normal, z′ = along the front.
  - Tie forces F′ at the front pairs, from (K·u) at the duplicate DOFs. This needs its own extraction path, because `reactions_and_modulus` zeros non-constrained residuals (fe/_postprocess.py:101-111).
  - Openings Δ′ at the pairs one row behind.
  - The 8-node brick formulas (Krueger 2004):
    - G_I = −F′_y·Δv′ / (2·Δa·b)
    - G_II = −F′_x·Δu′ / (2·Δa·b)
    - G_III = −F′_z·Δw′ / (2·Δa·b)
  - Tributary width b per front node, with corners and edges handled.
  - Outputs per front node: G_I, G_II, G_III, G_T and mode mixity ψ = (G_II + G_III)/G_T.
  - Planform fronts are staircases on a structured grid. Use the local front normal and the tributary area, and verify against the closed-form circular crack.
- **2.6 Fracture criterion and critical load factor.**
  - B-K: G_c(ψ) = G_Ic + (G_IIc − G_Ic)·ψ^η.
  - Mechanical load only: G scales with λ², so **λ_c = √(G_c(ψ)/G_T)** at each front node, with ψ independent of λ. Report the minimum over the front, its location and its ψ.
  - With thermal prestress: u = u_T + λ·u_M, so each mode's G(λ) is quadratic, ψ depends on λ, and λ_c comes from a scalar root per node (`brentq`, already used in `refinement.py`).
  - Toughness at each front node: G_c from `MaterialProperties`, knocked down by the local interface Vp (Phase 4).
- **2.7 Knockdown decomposition** (D6) and results.
  - New optional `FieldResults` fields: front table (x, y, G_I, G_II, G_III, ψ, G_c, λ_c), `fracture_load_factor`, governing front node, contact info.
  - JSON 1.3 fields and VTU front data.
- **2.8 Verification gate** (all of these become tests):
  - The probe as a test: DCB VCCT within 5 % of the FE compliance derivative over Δa.
  - Corrected beam theory for DCB, with the root-rotation correction: the probe's uncorrected beam theory reads 212 J/m², and the correction should close the gap to within about 3 %.
  - Energy balance G·ΔA = ΔU for a one-element extension.
  - G_III ≈ 0 and G_II ≈ 0 for a symmetric DCB.
  - The isotropic circular crack: G = K_I²(1 − ν²)/E with K_I = 2σ·√(a/π), checked on the staircase front against mesh refinement.
  - Mesh convergence of G_T through `refinement_study`.
  - Bit-identity of every existing mode when no delamination is present (fingerprint).

### Phase 3: Fracture specimens and NASA benchmarks (M, 1–2 PRs)

- **3.1 Builders** `CompositeMesh.for_dcb`, `for_enf` and `for_mmb`, with an SLB option and ISO 15114 C-ELS later. Each has a geometry dataclass in the `FlexureGeometry` pattern, node-line alignment like `for_flexure`'s `aligned()`, and the insert as a through-width `Delamination`.
  - Default dimensions: D5528 DCB (≥ 125 mm long, 20–25 mm wide, a₀ ≈ 50 mm); D7905 ENF (2L = 100 mm, a₀ = 30 mm); D6671 MMB (half-span 50 mm, a₀ ≈ 25 mm).
  - **We could not access the ASTM text** (the ASTM site prohibits AI use of its standards), so you need to check these defaults against the purchased editions.
- **3.2 Loading modes** `'dcb'`, `'enf'` and `'mmb'` in `FELoadingMode` / `_FE_LOADINGS`, all force controlled.
  - DCB: hinge line loads.
  - ENF: `roller_supports` + `line_load`, reused (not the `flexure_3pt` mode, whose `flexural_modulus` needs a structured node count).
  - MMB: lever kinematics, either as an equivalent split of the two loads (Reeder's lever relation for a target G_II/G_T) or as a multi-point constraint lever.
  - Outputs: compliance, front G distribution, critical load P_c = λ_c·P.
  - Emulated data reduction (MBT and compliance calibration from FE compliance at several a), so results compare like-for-like with test reports.
- **3.3 Benchmarks** (verification gate; Krueger's NASA reports):

  | Benchmark | Material | Geometry | Reference | Target |
  |---|---|---|---|---|
  | DCB | T300/1076 elastic, G_Ic = 0.1703 N/mm | B = 25, 2h = 3, 2L = 150, a₀ = 30 mm | Critical load–displacement path, NASA/TM-2008-215123 | Onset load within 3 % |
  | ENF | IM7/8552, G_IIc = 0.774 | B = 25.4, 2h = 4.5, 2L = 101.6, a₀ = 25.4 mm | Q_crit = 1526 N, NASA/CR-2011-217305 | Within 3 % |
  | MMB | IM7/8552, η = 2.1 | 2L = 100.8, a₀ = 25.4 mm | P_crit = 385 N at G_II/G_T = 0.5; 751 N at 0.8, NASA/CR-2012-217562 | Within 3 % |
  | SLB | C12K/R6376 multidirectional | B = 25.4, t₁ = t₂ = 2.032, 2L = 177.8, a = 34.29 mm | NASA/TM-2008-215123 | Within 5 % |

  Benchmark materials are defined in the tests. Adding an `IM7_8552_epoxy` preset is a separate decision; it needs a complete, sourced property set.
- **3.4 `refinement_study`.** New quantity kinds (G_T at the governing front node, λ_c, ψ), a builder dispatch and a front-alignment multiple, in the same pattern as ILSS and flexure.

### Phase 4: Porosity-toughness coupling and validation (M, 1–2 PRs)

- **4.1 Toughness knockdown laws.**
  - `_TOUGHNESS_LAWS` (exponential, linear, user-supplied), keyed by mode (I, II) and stage (initiation, propagation), with the placeholder coefficients from D4.
  - Every result carries an `uncalibrated` flag until datasets back the coefficients.
  - The law never increases G.
- **4.2 Local toughness in the FE.** G_c at each front node comes from the interface's local Vp, sampled from the porosity field at the front.
  - This is where the porosity *distribution* finally matters for delamination: interface-clustered voids degrade toughness exactly where the crack runs.
  - It depends on the Phase 0.4 renormalization, so local Vp means what the user asked for.
- **4.3 Empirical screening path.** Optional empirical modes `mode_I_toughness` and `mode_II_toughness` (G_c knockdown, no FE), plus O'Brien free-edge G from Phase 1.5, for quick NCR screening.
  - They need `LoadingMode`, `Calibration`, `PRISTINE_STRENGTH_KEY` and `_KNOCKDOWN_LAWS`-style registration (CLAUDE.md "Adding empirical coefficients").
- **4.4 Validation datasets.** Add schema keys `G_Ic` and `G_IIc` and extend `_PROPERTY_TO_MODE` (validate_all.py:179-190, 237; a new key currently raises KeyError).
  - Digitize the open-access studies first:
    - Hakim 2017: T700/EPON 828, three Vp levels, static G_Ic and fatigue-onset curves;
    - Carraro 2015: STS40/EC157, 0 / 1.9 % and 0 / 1.23 %, R-curves;
    - Agius & Fox 2015: G_IIc at < 0.3 / 1.5 %.
  - Then the rest: Asp & Brandt 1997; Mouritz 2000 (glass/polyester, 3–30 %); Hou, Ye & Mai 1996 (CF/PEI, 0–6 %); Olivier 2007.
  - Tag each point as initiation or propagation, and as from the insert or precracked. The precracking method shifts G as much as porosity does: an insert gives 45–65 % higher ENF values on IM7/8552.
  - Report the toughness MAE **separately** from the strength headline.
- **4.5 Uncertainty.** Propagate b and G₀ through the existing `propagate_uncertainty`, and report bands on λ_c. Document that mode I propagation toughness can *increase* with voids, which the conservative law ignores.

### Phase 5: Product integration: defects, disposition, docs (M, 1 PR)

- **5.1 App "Defects" panel.**
  - Porosity, as today, plus a delamination input: interface, planform, size and position from NDI.
  - Results: the G_T and ψ along the front, the governing mode, the fracture load factor, and the knockdown decomposition.
  - DCB, ENF and MMB as test-planning specimens.
- **5.2 NCR disposition.** `recommend_disposition` gains indication types:
  - porosity: Vp % or attenuation;
  - delamination: planform size and interface depth;
  - combined.
  - A delamination is dispositioned on a **fracture margin** (λ_c against the required load factor), not on Vp bins.
  - Acceptance thresholds are OEM- and part-specific; no published numbers were found, so they stay user inputs.
- **5.3 CLI and validation tooling.** A delamination option on `porosity-analyze`; toughness datasets in `validate_porosity`.
- **5.4 Docs.**
  - A theory page "Delamination and fracture mechanics": VCCT, B-K, onset criteria, porosity-toughness, limits, benchmarks.
  - README Limitations and Inputs table; CLAUDE.md architecture; CHANGELOG.

### Phase 6 (optional, gated): cohesive zones and nonlinear growth (L)

Start this only if users need R-curves, unstable growth or progressive delamination. It needs:
- a zero-thickness 8-node interface element outside the hex batch;
- a bilinear traction-separation law (Camanho & Dávila, NASA/TM-2002-211737) with B-K;
- penalty stiffness K = α·E₃/t with α ≥ 50;
- a mesh finer than l_cz/3, where l_cz = M·E₂·G_c/τ₀² (Turon et al. 2007). In their example that means elements under 0.1 mm, and above 0.5 mm the peak load was over-predicted by more than 2×;
- Newton iteration with arc length.

It breaks the linear load-factor framework, the LU cache and the single-state `FieldResults`. The linear VCCT λ_c covers growth *onset*, which is what disposition needs, so the recommendation is to defer.

---

## 5. Verification and validation matrix

| Check | Reference | Tolerance | Phase |
|---|---|---|---|
| Interface traction continuity | Mesh refinement | Jump → 0 | 1 |
| Free-edge interlaminar stresses | Published free-edge solutions | Qualitative profile, then convergence | 1 |
| O'Brien edge delamination G | Closed form | 5 % | 1, 2 |
| VCCT vs compliance derivative | FE dC/da | 5 % (probe: 4 %) | 2 |
| DCB vs corrected beam theory | Root-rotation-corrected beam | 3 % | 2 |
| Energy balance | G·ΔA = ΔU | 2 % | 2 |
| Circular crack, isotropic | K_I = 2σ√(a/π) | 5 % at the finest mesh | 2 |
| NASA DCB, ENF, MMB, SLB | Krueger reports | 3–5 % on critical load | 3 |
| G_T mesh convergence | `refinement_study` | GCI < 2 % at Δa ≤ 0.5 mm | 2, 3 |
| Toughness vs Vp | Hakim, Carraro, Agius & Fox, … | Reported, exploratory | 4 |
| Existing modes unchanged | 132-case fingerprint | Bit-identical unless intended | All |

---

## 6. Risks and mitigations

- **Toughness data are scarce and contradictory.** Placeholder laws carry an uncalibrated flag and stay out of the headline MAE; uncertainty bands are shown; user overrides are allowed.
- **Mode partition at bimaterial interfaces.** Fix Δa (default one ply), report G_T, and document the dependence.
- **Mesh resolution at fronts.** Structured grading (D5), the memory guard and a `refinement_study` quantity. Specimens get their own fine meshes.
- **Staircase fronts for planform delaminations.** Local front normals and tributary areas, verified against the circular-crack solution.
- **Crack-face interpenetration.** Active-set contact (D7), with no new factorization.
- **Standards text not accessed.** You check the specimen defaults against purchased editions.
- **Result changes from Phase 0.4 and 1.4.** Separate decisions, fingerprint-diffed and recorded in the CHANGELOG.
- **Scope creep into cohesive zones.** Gated as Phase 6.

---

## 7. Sequencing and effort

| PR | Contents | Depends on | Size | Result changes |
|---|---|---|---|---|
| A | Phase 0 (0.1–0.3, 0.5) | — | S | None (additive) |
| A′ | Phase 0.4 interface renormalization | A | S | `interface` configs (decision) |
| B | Phase 1 onset, tractions, Zt/S13, free edge | A | M | Additive; 1.4 is a decision |
| C | Phase 2.1–2.3 crack mesh and ties in `FESolver` | A | M | None without a delamination |
| D | Phase 2.4–2.8 contact, VCCT, λ_c, outputs | C | M–L | Additive |
| E | Phase 3 specimens and benchmarks | D | M | Additive |
| F | Phase 4 toughness laws and datasets | D (+A′) | M | Additive |
| G | Phase 5 app, NCR, CLI, docs | D–F | M | Additive |
| (H) | Phase 6 cohesive zones | Decision | L | — |

PR B and PR C can run in parallel after A. The critical path is A → C → D → E/F → G.

---

## 8. Decisions needed from you

1. **Crack representation:** node duplication with ties (recommended) or interface elements now.
2. **Phase 0.4:** renormalize the `interface` profile so FE Vp equals the requested Vp. This changes `interface` results by removing the +18 % inflation.
3. **Phase 1.4:** keep the transversely isotropic Mori-Tanaka tensor so penny voids weaken out-of-plane properties more. This changes every penny configuration.
4. **τ13 strength:** add optional `tau_13` with today's behaviour as the default (recommended), or unify on one strength now (changes results).
5. **Porosity-toughness default:** a conservative placeholder exponential flagged uncalibrated (recommended), or no default until data exist.
6. **Benchmark material:** add a sourced `IM7_8552_epoxy` preset, or keep benchmark materials inside the tests.
7. **Cohesive zones:** defer (recommended) or plan now.

---

## Appendix A: Baseline toughness (literature brief)

**Flags:** [P] read in the primary source; [S] from a secondary source; [NV] not verified. Values in N/mm.

| Material | G_Ic | G_IIc | B-K η | Source |
|---|---|---|---|---|
| IM7/8552 (benchmark reference) | 0.212 | 0.774 | 2.1 | Krueger, NASA/CR-2012-217562 [P] |
| IM7/8552 | 0.206 (fatigue precracked) – 0.237 (insert) | 0.739–0.807 (precracked) | – | Czabaj & Ratcliffe 2013, doi:10.1016/j.compscitech.2013.09.008; O'Brien et al., NASA/TM-2010-216838 [P] |
| AS4/3501-6 | 0.077–0.091 | 0.54–0.71 | NV | Reeder 1992, NASA TM-104210 [P] |
| AS4/PEEK (APC-2) | 0.969 | 1.719 | 2.284 | Camanho & Dávila, NASA/TM-2002-211737 [P] (Reeder: 0.78–0.84 / 1.16–1.25) |
| E-glass/epoxy (M10) | 0.118 | 2.906 | 2.6 [S] | Kenane & Benzeggagh 1997, doi:10.1016/S0266-3538(97)00021-3 [P] |
| IM7/8551-7 | 0.61–0.63 (initiation) | NV | NV | Naik, Crews & Shivakumar, NASA TM-101677 [P] |
| T300/934 | ≈ 0.088 | NV | NV | Funk & Deaton, NASA TP-2950 [P] |
| T700/EPON 828 | 0.52 (at 1.49 % voids) | NV | NV | Hakim et al. 2017, doi:10.4236/msa.2017.82011 [P] |
| T800H/3900-2 | 0.564 (initiation) | NV | NV | Zhang & Fox 2007, doi:10.1080/10426910701385432 [P, abstract] |
| T800S/M21, HTA/EHkF420 | NV | NV | NV | No verified values found |
| T300/914C (DCB benchmark) | 0.1703 | 0.4936 | 1.62 | Krueger, NASA/TM-2008-215123 [P] |
| C12K/R6376 (SLB benchmark) | 0.3405 | 1.286 | 3.39 | Krueger, NASA/TM-2008-215123 [P] |

Use precracked or initiation values as the base the porosity knockdown applies to. Values measured from the insert run high.

## Appendix B: Porosity vs toughness studies

| Study | Material | Vp | Finding |
|---|---|---|---|
| Hakim et al. 2017 [P] | T700/EPON 828 UD | 1.49, 1.77, 3.43 % | G_Ic 0.52 → 0.45 (about −7 % per 1 % voids); mode I fatigue onset G_Imax at 10⁶ cycles −50 % |
| Carraro et al. 2015 [P], doi:10.1179/2055035914Y.0000000004 | STS40 NCF/EC157 | 0 / 1.9 % (0°), 0 / 1.23 % (±45°) | G_Ic slightly *higher* with voids (bridging); fatigue Paris curves unchanged |
| Agius & Fox 2015 [P, abstract], doi:10.1016/j.compositesa.2015.02.023 | OoA carbon/epoxy | < 0.3 / 1.5 % | G_IIc 21 % higher without voids |
| Olivier et al. 2007 [S] | UD carbon/epoxy | up to 5 % | G_Ic −22 % for +5 % voids |
| Asp & Brandt 1997 [S] | UD carbon/epoxy | 0 / 0.9 % | Initiation down or unchanged; propagation G_Ic +75 % (ply splits); mode II slightly down |
| Mouritz 2000 [S], doi:10.1177/002199830003400303 | Woven glass/polyester | 3–30 % | G_Ic (initiation and propagation) falls steeply to about 15 %, then plateaus |
| Hou, Ye & Mai 1996 [S], doi:10.1177/073168449601501104 | Woven CF/PEI | 0–6 % | Propagation G_Ic up; initiation G_Ic, G_IIc down |
| Mehdikhani et al. 2019 review, doi:10.1177/0021998318772152 | — | — | No consensus; competing mechanisms; fatigue amplifies void effects |

**Implementer references.**
- Krueger 2004, Appl. Mech. Rev. 57:109 (VCCT), with the NASA benchmark reports above.
- Camanho & Dávila 2002 (B-K, decohesion element).
- Turon et al. 2007, EFM 74:1665 (cohesive-zone sizing).
- Brewer & Lagace 1988, JCM 22:1141 (quadratic onset).
- O'Brien 1982, ASTM STP 775 (edge delamination).
- Reeder 1992, NASA TM-104210 (mixed-mode criteria).
- Mehdikhani et al. 2019 (void effects).
