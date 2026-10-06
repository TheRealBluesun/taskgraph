You are implementing `taskgraph`, a Python tool, one small task at a time.

1. Read SPEC.md (authoritative) and PLAN.md. Skim PROGRESS.md for notes from earlier tasks.
2. Do ONLY the FIRST task in PLAN.md marked `- [ ]`. Don't start the next one; don't refactor unrelated code.
3. Python ≥ 3.11, standard library only. Match SPEC.md names and behaviour exactly. Keep modules small and pure where SPEC says pure.
4. Verify: `python3 -m unittest discover -s tests -q` must pass. Tests must be fast (< 30 s total), deterministic, and must never call omp, a model, or the network — fake agents with tiny shell scripts, use temp dirs and temp git repos, inject metrics.
5. When everything passes: change the task's `- [ ]` to `- [x]` in PLAN.md and append to PROGRESS.md
   `## Txx — <title>` + 2–5 bullets: files touched, decisions, anything the next task should know.
6. If truly blocked, do NOT tick the task; write `BLOCKED: <reason>` in PROGRESS.md and stop.
7. Run commands in the foreground and wait for them; never end your turn while a background job is running.
8. Don't run git commands (the loop commits for you). Don't touch tools/ (the loop's own runner) or anything outside this directory.
9. Your progress is monitored live; keep every command bounded (no servers, no infinite loops; use timeouts in tests).
