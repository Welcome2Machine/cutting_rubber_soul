# Repository task requirements

Read `docs/task_requirements.md` before changing the model. The experimental
target is `docs/image.png` and the experiment/material/interface sections of
`docs/current_model_spec.md`; implementation details elsewhere are not ground truth.

- Current research branch: `feature/czm-cutting`. The user now permits a prescribed
  potential fracture path that is initially fully bonded. Predict onset and growth
  conditional on that path; do not insert an initially open crack or force the
  crack tip to follow the blade. Full-domain screening remains a diagnostic.
- Read `docs/server_handoff.md` before server work. CZM is a planned implementation,
  not an existing cutting solver. This handoff/push task must not launch tests or
  simulations locally; server execution is a separate subsequent task.
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
- Public main has independent history. The 2026-09-28 user instruction explicitly
  authorizes creating and pushing `feature/czm-cutting` from that clean lineage;
  publish only `feature/czm-cutting:feature/czm-cutting` for this handoff, leaving
  remote main unchanged. Audit all reachable commits of both main and the exact
  branch with `python tools/check_publish.py --ref REF` before pushing. Future
  main releases remain main:main. Never push/merge the local legacy master, tags,
  --all or --mirror, which may disclose local data.
- Missing local fixtures must be reported as skipped tests, not passed tests.
