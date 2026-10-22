# Repository requirements

Read [research status](docs/research_status.md), [task requirements](docs/task_requirements.md),
[model specification](docs/current_model_spec.md), and [code map](prototype/README.md).
Historical notes do not authorize new runs.

## Current mainline

- Intact finite-strain isochoric Maxwell, native P1/improved-max IPC with full EV/VV,
  body self-contact and CCD. No fracture, damage, CZM or post-fracture prediction.
- Shared Ginf=0.14 MPa, Gv=1.24 MPa, tau=1.7 s, K=23.34333333333331 MPa,
  mu=0.35; 0/33/57 degrees, vertical speed0.2 mm/s. Target2 mm only, never3 mm.
- Baseline definition: prototype/intact_2mm_baseline.json. Three complete201-state
  paths are in outputs/baseline/angle0, angle33, angle57.0/33 are the latest
  completed replay;57 is the earlier complete common-material path.
  Computational completion is not experimental calibration.
- Keep force residual1e-4 N, positiveJ, fullCCD, boundaries, dissipation and exact
  material/friction histories. Failed trials never commit histories.
- Baseline recovery is secant then same-increment affine,500 per attempt;
  affine has no40-stagnation early stop. Original minimum increment0.000625 mm.
  Do not confuse this with failed fixed-step candidate controllers.

## Current operational state

User explicitly requested stopping computation and removing frozen source copies.
Supervisor and accelerated57 solver were stopped2026-10-22 23:19:55 JST.
No active queue; do not automatically resume. Accelerated57 did NOT complete2 mm.
Material batches2048/PARDISO4, IPC2/BLAS1 previously saved27.21% over three
conditional increments. Their old evidence directory was retired; retain this
as historical summary, not a full-path speed guarantee or new completion claim.
Current code may perform fresh baseline audits without requiring old source trees.

## Data and interpretation

- Same rubber across angles; no angle-specific material, effective-width or force scaling.
- Force inputs: docs/1008n0a2__Fz__curve_1.csv,1008n33a1__Fz__curve_2.csv,
  1014n57a1__Fz__curve_0.csv. Preserve units, zeros and trial identity;57 starts0.51 N.
  Do not use old V01 force channels.
- Later experimental data include fracture.0-degree~1 mm estimate refers to a1,
  not current a2;33~1.2--1.3 mm is approximate;57 onset unknown.
- Optical flow is auxiliary: valid=1 and reference_frame_id=0 for initial comparisons.
  Calibration, first contact and force-video synchronization are not assumed.
- Distinguish contact reference-material offset, current displacement and accumulated slip.
  Missing/ambiguous contact stays missing/an interval.
- Prior numerical failures do not prove no root or material-family insufficiency.
  Retained findings/contact limitations are in docs/research_history.md.

## Repository maintenance and validation

Keep core code, reusable project regressions, original papers and one references/README.md,
reproductions, experimental inputs, three complete baselines and paper2023 results.
Remove obsolete runners, one-off tests, duplicate logs and failed temporary outputs.
Do not preserve repeated frozen code trees; record versions/parameters without pretending
retired source copies remain available. Preserve accurate histories and check fingerprints.
Before recursive deletion verify absolute paths remain in the repository and check live dependencies.
Status/logs/analysis use HTML plus JSON/CSV. Long runs require explicit scope and budgets;
check startup once then leave detached. Do not start new loading merely to clean files.
Run relevant reusable tests after code cleanup. No automatic retries or old queues.
Only code, text docs and approved configs go to Git; PDFs, media, raw data, outputs and runtimes stay local.
