# Integration workspace

Read README.md, docs/interface_contract.md and docs/git_workflow.md before changing interfaces.
The original directories outside this workspace are read-only sources; work in projects/ copies.
The user corrected the input: softmodeling is the FE project. The old 0915ForceProcessing is excluded.

- projects/softmodeling retains its own AGENTS.md and numerical invariants. This integration request
  explicitly enables the video comparison layer; earlier local statements deferring matching are historical.
- Preserve numerical algorithms and original inputs. Integration code belongs to softcut_solution/.
- No FE loading, resumed queue or GPU tracking on UI startup. Explicit user actions use separate outputs,
  recorded interpreters, budgets and source hashes. Existing-result export is postprocessing.
- Conditional video scale, clock, reference, material identity and FE qualification must remain explicit.
  Do not infer fracture from a cutoff, promote missing/invalid samples, or fit registration to FE error.
- Use subprocess argument lists and copied project working directories; no shell concatenation.
- Read-only source snapshots, generated runs, weights and raw data stay out of Git.
- Run python -m pytest -q and python tools/check_publish.py --staged before integration commits.
  Subproject changes also require their applicable reusable regressions.
