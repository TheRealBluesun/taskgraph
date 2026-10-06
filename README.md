# taskgraph

`taskgraph` runs a project's task list (`PLAN.md`) with several coding agents in parallel, each in
its own git worktree. It schedules tasks by dependency and critical path, shares scarce resources
(one simulator, a model server's sessions) through leases, and merges finished work through a
serialized gate. Everything project-specific lives in the project's `taskgraph.toml`.

Requirements: Python ≥ 3.11, standard library only.

Run the tests:

```
python3 -m unittest discover -s tests -q
```

See `SPEC.md` for the authoritative behaviour and `PLAN.md` for the build order.
