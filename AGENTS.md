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

LATEST user request supersedes pause for NEW57-priority candidate studies:
current code published to main91702c2, research/joint-history-sensitivity-20261022
local+remote branch deleted after remote-main hash verified. Old common2mm stays
paused; no history is transplanted to new material. New3 bounded queues launched
2026-10-23 15:56:50 JST, outputs/angle57_priority_20261023:
reference57 Ge=.14/Gv1.24/tau1.7 worker61656;
balanced57 Ge=.20/Gv.90/tau1.7 worker38688;
fast57 Ge=.14/Gv1.24/tau.85 worker40796.
EACH serial57/0/33, all independent REST->1mm histories, .01 physical increment,
5400s solve+1800s setup/audit/case, up to3 workers in parallel. SameK/mu/v/blade5um/
body/contact/force/J/CCD/dissipation/history acceptance, original500/40->affine500/
no40 and cutbacks. NEW performance batch0/PARDISO4/IPC2/BLAS1 avoids known
batch/native metric replay discrepancy without weakening gates or altering law.
Primary selection57 .1-.5 RMSE/shape, .5-1 separate; small-angle worsening allowed
and quantified, NO angle-specific material. No automatic2mm extension/retry.
Old57 original-blade1mm17.279N vs zeroed experimental17.18 motivates reference;
not a new fine-blade result. Balanced candidate interpolates branch partition;
fast candidate isolates half tau, not a claimed improvement. No new physical
model terms justified yet. Physical source code unchanged from published91702c2;
198 reusable regressions pass. Current only docs/run-config changes afterwards.
Status and plan: outputs/angle57_priority_20261023/index.html. ONE startup
15:58:50 JST confirms all3 workers/solvers exact commands/creation/executables,
stderr0; each20 source/6 input hashes match.57 prefixes at observation ONLY:
reference/balanced.02mm3states,fast.03mm4states; batch0/material parameters match.
No final path audit/completion claimed. Leave detached, no duplicate/monitoring.

PRIOR paused common-candidate review (not current active-run authorization):
LATEST2026-10-23 user requests PAUSE and analyze. Stopped common2mm supervisor
32224 then33 solver59640 at12:37:42 JST, verified no remaining loading process.
NO automatic resume/restart/extensions.33 last accepted1.28mm, trial1.29 unsaved.
0 completed2mm/201states with full native audit;203 files verified.33 paused129
states, fresh native full-prefix audit NOW passes,131 files verified. Standard
33 audit.json published, result remains paused/completedFalse, not2mm success.
57 loaded2mm/201states but full history NOT qualified: fresh postprocessing
checks find ONLY1.26mm failing material metric reproduction (2 symmetric entries).
Continuous qualified prefix1.25mm. All other200 state audits individually pass;
this does NOT certify the entire path past1.26. Exact dual-history/dt/dissipation
chains pass. First failed state's batch replay exact; whole-array max metric
change2.8372e-11,stress1.0516e-11MPa; native residual2.13824e-5N,Fz identical.
Local return-map residual8.5567e-12<original1e-11. Keep original gates; no fix or
promotion yet. Diagnosis: batch stopping-scale reproducibility vs metric gate,
not mechanical failure at1.26 or explanation of multi-N experimental bias.
Same Ge=Gv=.345/tau1.7/K/mu/v/blade5um/batch2048/PARDISO4/IPC2/BLAS1;
ALL actual physical increments.01, no cutbacks.57 .5->2 report2689.5s vs0
2716.969s; shallow expensive windows do not characterize the later whole path.
At1mm FE0/33/57=17.3447/16.9728/13.4191N; first-force-zero experiments
18.29/16.45/17.18N.57 persistent early/mid-depth deficit remains,0/33 close.
Later experiment includes fracture; do not fit intact late response blindly.
57 experiment subtracts.51N,0/33 already0; original CSVs retained, no depth shift.
20 sources/164 inputs match. Review used saved-field audits/plots ONLY; no new
FE loading, code/parameters/solver changes or relaxed acceptance. Evidence:
outputs/common_candidate_20261023/common2mm/pause_review/index.html and root HTML.
Next prospective priority: material integration batch consistency under original
criteria, then common G0/q/tau response fit in qualified early windows. This
analysis does NOT authorize new runs. Keep paused until explicit user resume.

2026-10-23 common.5 queue ENDED05:19:23 JST, all57/0/33 completed.5mm,
51 states EACH, fresh full-chain native audits passed. User-requested result
review verified159 case files plus20 sources/25 inputs, zero mismatches; no
matching loading processes. Ge=Gv=.345/tau1.7/fixedK/mu/v, blade5um,
physical.01 increments throughout; NO cutbacks.57 resumes its OWN audited.19,
0/33 start at rest. Same batch2048/PARDISO4/IPC2/BLAS1 and original500/40->
affine500/no40, force/J/CCD/boundaries/dissipation/history unchanged.
Current.19-.5 window57=1699.375s, prior windows2209.188s SEPARATE;
0/33 own-path reports650.797/554.656s.57 .19/.29 require482/424 iterations,
but .3-.5 costs351.423s/290 reported iterations, no rejected predictors.
Do not claim a universal speed fix or extrapolate worst-point cost to2mm.
Actual.5 Fz0/33/57=7.868814/7.927619/5.796138N, experiments8.27/7.61/8.48.
Common.1-.5 RMSE=.224350/.270487/2.319894N; pooled1.354671N.0/33 both
secants within20%;57 first secant36.71% LOW, second7.81% LOW. At.5,
F33-F0 FE+.058805 vs experiment-.660N; F57-F0 FE-2.072675 vs+.210N.
Numerical short-window goal met; common fit NOT met, candidate not promoted.
Do not infer whole Maxwell family insufficiency or introduce angle-only fits.
No new runs/extensions authorized by this status review.
Evidence/status: outputs/common_candidate_20261023/index.html and common05.
Retained original geometry baseline still usesGe=.14/Gv1.24 and has3x2mm.

Earlier same-history fine.16->.17 original affine500/no40 control passes364
iterations; secant500/no40 also passes348 to same root.40 residual stagnation
is not no-root proof. Gradient/history globalizers failed and were removed.
Original join quadrature weight~302x neighbor; optional5um collinear blade
subdivision keeps canonical polygon/nose/body.Independent.0-.16 attempt time
saves23.4%, max force change.9%, NOT full-path equivalent acceleration.
Coarse physical.02 control remains NOT qualified: strict metric reproducibility
fails.10 from batch stopping-scale discrepancy; never promote/relax gates.
Failure snapshots use failed_previous, not rolled-back previous.198 reusable
regressions passed before launch; default equations match5039e54 at real field.
No frozen source copies; necessary exact histories/failure evidence retained.

User explicitly requested stopping computation and removing frozen source copies.
Supervisor and accelerated57 solver were stopped2026-10-22 23:19:55 JST.
That old queue must not resume. Accelerated57 did NOT complete2 mm.
Material batches2048/PARDISO4, IPC2/BLAS1 previously saved27.21% over three
conditional increments. Their old evidence directory was retired; retain this
as historical summary, not a full-path speed guarantee or new completion claim.
Current code may perform fresh baseline audits without requiring old source trees.

## Data and interpretation

- Same rubber across angles; no angle-specific material, effective-width or force scaling.
- Force inputs: docs/1008n0a2__Fz__curve_1.csv,1008n33a1__Fz__curve_2.csv,
  1014n57a1__Fz__curve_0.csv. Preserve raw files/units/trial identity. Comparison
  targets subtract initial Fz;57 raw starts.51N, derived target starts0 by current user request.
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
