# Repository requirements

Read [research status](docs/research_status.md), [task requirements](docs/task_requirements.md),
[model specification](docs/current_model_spec.md) and [code map](prototype/README.md).
Historical notes do not authorize new runs.

## Scope

- One common rubber material for0/33/57 degrees; identical properties are a
  user-confirmed premise. Predict intact force, surface motion and reference
  contact migration. No fracture/damage/CZM in this stage; no angle-specific fit.
- Match force amplitude AND shape across angles.57 participated in parameter
  selection; its agreement is not independent validation.
- Authorized local Fz CSVs:1008n0a2,1008n33a1,1014n57a1. Preserve N values and
  exported zeros; convert um to mm only.57 starts at.51 N. OldV01/raw channels
  are excluded. Do not borrow one angle's experiment to validate another.
- Later experiment is fractured.33 onset~1.2--1.3 mm is approximate;0 onset~1 mm
  referred to a1, not current a2.57 unknown. Early windows are provisional.
  Solver failure is not fracture detection. Missing contact identity stays missing.

## Current state, 2026-10-20

Native P1/improved-max IPC + finite isochoric Maxwell is the working model.
AL/P2, MINI, analytic quadrature and GCP are diagnostic alternatives, not defaults.
The common-material queue ENDED17:31:41 JST. Six0/33 cases completed.5 mm;
soft/fast/partition57 failed.10/.20/.09, accepted.09/.19/.08. All345 accepted
records passed automatic exact-history audits. No set passed the joint gate;
no combination proposal or.8 extension ran. No loading process remains.

soft improves amplitude but not0/33 slope evolution; partition raises force.
Two57 rejected fields localize residual near the nose; a specific contact/
Jacobian defect is NOT proved. Next priority: bounded failed-increment diagnosis,
then shared lower instantaneous stiffness with less relaxation hypothesis.
This is a research priority, not authorization to start a new loading queue.

## Execution and preservation

- Preserve original physical residual/J/CCD/dissipation acceptance and exact
  material AND friction histories. Do not replace acceptance with projected
  residuals, increase budgets indefinitely, or fit parameters separately by angle.
- New calculations need independent outputs, frozen source/inputs and bounded
  budgets. Resume only identical own histories. Never weaken hash protection
  to resume across a source refactor; old frozen runners remain locally available.
- Use HTML for status/logs/analysis with JSON/CSV evidence. Check a detached
  launch once, then leave it to logs unless asked for status.
- Speed and accuracy both matter. Prefer equivalent assembly/linear algebra;
  quantify approximations' effects on force shape and contact identity.
- Keep recoverable3 mm baseline, current three-angle2 mm paths, current study
  histories and useful failure evidence. Retire obsolete outputs with explicit
  manifests and honest replay limits. Preserve user data, papers, videos, runtimes.
- Remote: code and Markdown only, plus approved small model configs. No raw
  CSV, media, PDF, results, private fixtures or binaries. Run check_publish.py.

Latest authorization: clean obsolete results/code/docs, document research state,
verify, commit and push. No new loading study requested by this cleanup task.
