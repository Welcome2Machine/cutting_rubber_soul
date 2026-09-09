# Repository task requirements

Read `docs/task_requirements.md` before changing the model. The experimental
target is `docs/image.png` and the experiment/material/interface sections of
`docs/current_model_spec.md`; implementation details elsewhere are not ground truth.

- Predict crack nucleation location and time without prescribing a crack under
  the blade. Full-domain stress/strain screening is an intermediate diagnostic,
  not the completed cutting/fracture model.
- Use verified research to justify changes; record assumptions and evidence in
  `docs/modeling_roadmap.md` and `references/README.md`.
- Promptly remove demonstrably redundant or obsolete files and repair their
  references. Preserve user-supplied images, papers, data, and active run files.
- Launch full cases as detached processes with persistent stdout/stderr logs,
  config, source fingerprints, accepted states, and completion/failure status.
  Check startup once. Do not continuously monitor or wait for a full case unless
  the user explicitly asks. Short bounded validation remains appropriate.
- Never report numerical completion or experimental validation merely because
  a process started, an optimizer returned success, or unit tests passed.
- Do not change the 3 mm target or material parameters to hide solver failures.
- Publish code and documentation only. PDFs, media, numerical/experimental data,
  fixtures, digitized curves and video-derived observations stay local. Summarize
  useful papers in original Markdown with sources and applicability limits.
- Public main has independent history. Audit all reachable commits using
  `python tools/check_publish.py --ref main`; push only main:main. Never push the
  local legacy master, tags, --all or --mirror, which may disclose local data.
- Missing local fixtures must be reported as skipped tests, not passed tests.
