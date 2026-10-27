# Repository requirements

Read [research status](docs/research_status.md), [task requirements](docs/task_requirements.md),
[model specification](docs/current_model_spec.md), and [code map](prototype/README.md).
Directory ownership, naming and publication rules are in [README](README.md).
Historical notes do not authorize new runs or establish current process state.

## Current scope and mainline

- Local HTML workbench for common rubber, blade wedge/nose sharpness, explicit
  Coulomb friction/lubrication settings and inclination. The parent integration
  workspace now enables conditional video comparison.
  Entry: `python -m tools.workbench --background --open`. Configuration catalogs
  are in `configs/`; architecture/schema in `docs/workbench.md`; evidence and
  research interpretation in `docs/research_guide.md`.
- Intact finite-strain isochoric Maxwell, native P1/improved-max IPC with full
  EV/VV, body self-contact and CCD. The intact baseline remains unchanged.
  The 2026-10-28 user task authorizes the separate `cohesive_cut` extension and
  saved-state onset diagnostics. The extension is a prescribed finite-gap,
  radial mode-I mechanism demonstrator, not general mixed-mode or calibrated
  fracture prediction. Speed is fixed at 0.2 mm/s; travel must not exceed 2 mm.
  See parent `docs/fracture_implementation.md` and `docs/research_status.md`.
- Common reference material: Ginf=0.14 MPa, Gv=1.24 MPa, tau=1.7 s,
  K=23.34333333333331 MPa, mu=0.35. Existing experimental angles are 0/33/57
  degrees. Workbench angle changes and research configurations are explicit inputs,
  not changes to the reference baseline. Balanced/q25 are research candidates,
  not independently calibrated materials.
- Baseline definition: `prototype/intact_2mm_baseline.json`. The three complete
  201-state paths are local `outputs/baseline/angle0`, `angle33`, `angle57`.
  The 0/33 paths are the latest completed replay; 57 is the earlier complete
  common-material path. Computational completion is not experimental calibration.
- Workbench geometry supports a symmetric wedge with a circular nose, not arbitrary
  CAD or a flat chamfer. Lubrication currently means a condition label plus an
  explicitly selected positive Coulomb mu, not an oil-film model. Public Fy maps
  solver `reaction_x_n`; Fyz is the in-plane resultant, not another component.

## Numerical and operational invariants

- Keep force residual 1e-4 N, positive J, full CCD, boundaries, dissipation and
  exact material/friction histories. Failed trials never commit histories.
- Cohesive histories also commit only after whole-step acceptance. Finite seam
  spacing, zero linear shear stiffness of radial bonds, reference area and
  damage-disabled split controls must remain explicit. A split elastic control
  is not the original intact mesh; never claim the two are equivalent.
- Reference recovery is secant then same-increment affine, 500 iterations per
  attempt. Affine has no 40-stagnation early stop. Original minimum increment is
  0.000625 mm. Do not confuse this with retired fixed-step candidate controllers.
- New physical settings start from their own consistent initial history. Do not
  transplant internal variables across materials, angles, mesh/contact geometries
  or performance policies, or bypass restore-consistency checks.
- Failure snapshots must pair failed positions with the actual attempt's
  `failed_previous`, accepted material metric, physical dt and attempted depth;
  rolled-back `previous` can refer to an earlier state.
- Independent path audits determine qualification. Standard result bundles mask
  unqualified quantities and omit unaudited fields. Exporting saved fields is
  postprocessing, not a new FE run or a fresh full-path equilibrium audit.
- The earlier q50 common2mm investigation remains paused. Its 57-degree path
  reached 2 mm mechanically but is continuously qualified only to 1.25 mm because
  of the 1.26 mm material-metric reproduction failure. Do not promote later states
  or relax the gate. This is separate from the complete original-material baseline.
- Original/fine-blade timing and material batching/PARDISO comparisons are local
  measurements, not full-path speed guarantees. Short-term residual stagnation is
  not proof that no equilibrium root exists. Retained evidence and failed routes
  are recorded in `docs/research_status.md` and `docs/research_history.md`.
- UI startup does not start FE. A user launch creates one explicitly configured,
  bounded job; server restart does not restart a queue or retry an old job.
  Long runs require explicit scope, resource budget and stopping conditions.
  Check startup once, then leave detached. No automatic extensions or old queues.
- Do not edit numerical source used by a live solve. Record version, parameters,
  source/input hashes and necessary checkpoints; do not create frozen source trees.
  Do not start loading merely to clean or organize files.

## Data and interpretation

- Same rubber across angles; no angle-specific material, effective-width or force
  scaling. Preserve raw inputs, units and trial identities.
- Authorized force inputs are local `docs/1008n0a2__Fz__curve_1.csv`,
  `docs/1008n33a1__Fz__curve_2.csv`, `docs/1014n57a1__Fz__curve_0.csv`.
  Comparison targets subtract the initial Fz: 57 starts at 0.51 N raw and 0 N in
  the derived target. Do not overwrite the originals or shift depth. Do not use
  old V01 force channels. Do not apply experimental zeroing to simulated forces.
- Later experimental data include fracture. The 0-degree ~1 mm estimate refers
  to a1, not the current a2; 33 ~1.2--1.3 mm is approximate; 57 onset is unknown.
  Numerical failures and declining slopes alone are not fracture evidence.
- Optical flow remains auxiliary. Follow each delivery schema in
  `docs/optical_flow_handoff.md`: old CSV requires valid=1 and reference_frame_id=0
  for initial comparisons; the 20261024 fixed-ID NPZ instead uses contact IDs,
  endpoint/continuous support and inferred/ambiguous/occluded flags. Do not assume
  laboratory calibration, mechanical first contact or force-video synchronization.
- Distinguish reference-material contact offset, current displacement and accumulated
  slip. Missing/ambiguous contact remains missing/an interval. IPC force components
  are not initial-surface projections; force share is not dissipated-energy share.
- Prior numerical failures do not prove model-family insufficiency. Separate
  literature evidence, actual numerical observations and unverified hypotheses.

## Maintenance and publication

- Keep core code, reusable project regressions, original papers plus one
  `references/README.md`, reproductions, experimental inputs, complete baselines
  and necessary current research evidence. Remove obsolete runners, one-off tests,
  duplicate logs and temporary failed outputs only after checking dependencies.
- Before recursive deletion, verify resolved absolute paths remain within this
  repository and check live dependencies. Preserve accurate histories/fingerprints;
  never pretend deleted source copies or retired evidence remain available.
- Status/logs/analysis use HTML plus JSON/CSV. Update existing state/history documents
  instead of creating competing status summaries. Run relevant reusable regressions
  after cleanup; tests must not silently start long loading runs.
- Publish only code, text documentation and reviewed configuration. PDFs, media,
  raw data, outputs, caches, local settings and runtimes stay local. Preserve source
  experimental filenames; do not rename raw inputs to satisfy code naming rules.
