# Repository requirements

## Current user task and ownership (2026-11-10)

The only current task is [research_prototype_task](../../researchtopic/research_prototype_task.md).
Read it and [Git workflow](../../docs/git_workflow.md) before acting. The first paper is experiments
plus explanatory forward simulation; learning, differentiable solvers and PINNs are possible second-paper
work, not implementation tasks now. Today the user authorizes SERVER-ONLY execution. The local agent
preserves its a39ce7 results and workspace; no new local FE, fit, code edits or duplicate packaging.
The server reads that fixed handoff without waiting for the local raw package or merging its branch.

The b30aa743 pilot/A/B experiment and resume guard are completed at 4730067a; the event windows failed
and do not qualify connected fracture or entry. Do not restart those tasks, their stopped parents or
historical C1/C2 controllers. Today's task is the bounded matched fine-body continuum/off/on comparison,
with a minimal opt-in pre-commit physical-chord gate in the existing step path. No new dynamic mode,
controller, solver, friction/material sweep, mesh platform or learning pipeline is authorized today.
Preserve previous results and failure labels; budgets do not roll over.

Read [research status](docs/research_status.md), [model specification](docs/current_model_spec.md),
[code map](prototype/README.md) and [project conventions](README.md) as evidence of implemented behavior.
They do not make an unimplemented task an existing capability. Root AGENTS and publication gates still apply.

## Physical scope and interpretation

- Reuse finite-strain isochoric Maxwell, P1 plane strain and established blade/contact machinery.
  Preserve reference configurations and historical model identities; this does not forbid explicitly
  authorized new shared inputs, a real damage-off mode or a declared interface formulation.
- Common reference material, measured blade data and source qualifications live in the current task;
  do not maintain a competing parameter table here. Speed is 0.2 mm/s and travel does not exceed 2 mm.
  The original s=7 mm geometry and the symmetric s=15 mm control are distinct experiments in the model.
- The workbench represents a symmetric wedge/circular nose, not arbitrary CAD. Lubrication is a
  condition label plus explicit Coulomb mu, not an oil-film model. Public Fy maps solver reaction_x_n;
  Fyz is a resultant. Force components, surface projections and energy shares are different quantities.
- Today all material/fracture parameters are frozen to isolate body/surface discretization and interface effects.
  General common-parameter calibration within defensible intact windows remains a later option.
  The fracture agent must not tune strength/Gc,
  angle-specific material, effective width or force factors to make a desired cutting curve.
- Treat the second-paper data foundation as source-traceable configurations, histories and observables,
  not permission to create a learning platform, differentiable solver or generic data pipeline.

## Numerical and operational invariants

- Keep physical force/momentum residual 1e-4 N, positive J, collision/path checks, actual boundaries,
  true material time, dissipation and exact accepted histories. Stricter existing solver targets are
  numerical settings, not a universal definition of scientific validity. Never retroactively accept a
  failed state or loosen a gate merely to extend a curve.
- Damage/friction/material histories commit only after whole-step acceptance; failed attempts roll back.
  Record reference interface area and count Gc once. No mass scaling, invented damping or hidden
  smoothing may manufacture a fracture result. Numerical work/defects are not physical dissipation.
- Exact tied-interface off, finite-stiffness cohesive off and continuous intact are different controls.
  Measure their differences; do not claim equivalence by name. Any approximate closure treatment must
  disclose its penetration/compliance and collision scope; do not label it exact IPC or full qualification.
- Preserve formulation-specific history identity: current full-domain intrinsic single-seam and mixed
  network histories use two Gauss points per segment. The separate historical extrinsic face route may
  use four or another explicitly configured order. Whole-face failure requires every applicable point,
  not a point count borrowed from another route. Network metric floors are globalization only, not physical energy.
- New blade/material/friction/mesh/interface settings start from their own consistent initial history.
  The previous QS-seeded inertia experiment retains its conditional identity and cannot upgrade its source
  path; today does not extend it. Ordinary restore-consistency checks remain. Documentation-only commits
  do not require rerunning paths whose actual numerical sources and inputs still match. Record actual imports/interpreter.
- Failure snapshots pair failed positions with that attempt's failed_previous, accepted material metric,
  physical dt and attempted depth. Do not confuse rolled-back previous with the attempted parent.
- Independent audits qualify the accepted prefix. Saved-state export is postprocessing, not a new solve
  or a new equilibrium audit. Do not export rejected points or mask later failures as completed curves.
- The paused q50 path at 57 degrees remains qualified only through 1.25 mm; its 1.26 mm metric failure
  is not erased by mechanical completion at 2 mm. It is distinct from the original-material baseline.
- Keep original recovery/iteration policies within their actual route. Numerical stagnation does not
  prove no root exists. The current task does not authorize a sweep of controllers or universal tolerances.
- Do not edit numerical code used by a live solve. New work uses an isolated worktree/version and bounded
  outputs. UI startup or server restart never starts FE, tracking, an old queue or automatic continuation.
  Use the current task's budget and stricter machine scheduler/resource constraints; no automatic extensions.

## Experimental data

- Preserve raw filenames, units and trial identities. Authorized existing forces are local
  docs/1008n0a2__Fz__curve_1.csv, docs/1008n33a1__Fz__curve_2.csv and
  docs/1014n57a1__Fz__curve_0.csv. Derived experimental targets subtract initial force (57 starts at 0.51 N);
  never overwrite raw data, shift depth to improve agreement or zero the simulated force in the same way.
- No angle-specific materials or width/force multipliers. Later 0/33 degree records may include fracture.
  The approximate 0 degree 1 mm label belongs to a1, not current a2; 33 degree 1.2--1.3 mm is approximate.
  Repeated 57 degree tests uncracked to 2 mm are a condition-level right-censored observation, not an event
  at 2 mm or trial-matched validation. A diagnostic stress crossing and a numerical stop are not fracture.
- Optical-flow data retain their own schema/support/reference flags. Do not assume laboratory scale,
  first-contact registration or force-video synchronization. Distinguish material displacement, contact
  material migration and actual relative slip; missing or ambiguous measurements remain missing/conditional.
- Separate simulation completion, numerical sensitivity, mechanism explanation and experimental prediction.
  Report measured facts, numerical observations and untested hypotheses separately.

## Maintenance and publishing

- One current task, short role entry pages and one existing current-state section. Update in place;
  use Git history for older text, not another dated task/review/README or frozen source tree.
- Reuse existing functions and entry points. Add a module only for a clearly new physical responsibility,
  explaining what it replaces; do not extend inheritance/dynamic class-swapping or runner/audit wrappers.
- Preserve source hashes, live dependencies, complete baselines and necessary failure evidence. Before
  cleanup, check references and replay identity; do not bulk-delete legacy code or unowned dirty files.
  Resolve absolute paths before any recursive deletion and keep them within the intended repository.
- One-off probes/logs/plots stay in ignored run directories. Git receives code, text documentation and
  reviewed configuration only, not raw data, state arrays, media, runtimes, credentials or local settings.
- Run required root checks and directly relevant reusable subproject regressions. Tests must not silently
  launch research runs. More files, audits and controller layers are not scientific progress.
