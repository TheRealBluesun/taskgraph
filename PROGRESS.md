## T01 — Skeleton

- Files: `taskgraph/__init__.py` (`__version__ = "0.1.0"`), `taskgraph/cli.py`, `bin/taskgraph`, `tests/test_smoke.py`, `README.md`.
- `cli.build_parser()` defines all SPEC §10 subcommands (`run`, `status`, `stats`, `lease`, `leases`, `stop`, `retry`) plus `models` (SPEC §4); `SUBCOMMANDS: dict[name, help]` drives creation, so later tasks just replace the `func` default per subcommand.
- Subcommand flags are already in place: `run --project/--max-agents/--dry-run`, `stats --since` (default `6h`), `stop --agents`, `lease <resource> --task ID -- cmd…` (REMAINDER), `retry <id>`.
- Every handler is `_not_implemented(args)`: prints `taskgraph <cmd>: not implemented` to stderr, returns 1. No command → help on stdout, rc 2. Next tasks should set `func` on their parser (or dispatch in `main`) instead of editing `_not_implemented`.
- `bin/taskgraph` prepends the repo root to `sys.path` so the script runs uninstalled from a checkout; tests never need `pip install`.

## T02 — Config loader

- Files: `taskgraph/config.py`, `tests/test_config.py` (22 tests total, 0.2 s).
- API: `load(path) -> Config` + `ConfigError`; dataclasses `Config` (`root` property = config's dir, `model(name)` lookup), `AgentConfig`, `ModelConfig`. `load` accepts `str | Path`.
- Defaults where SPEC §1's example has a key but does not require it: `links=()`, `resources={}`, `agent.overlay="omp-agent.yml"`, `agent.stall_secs=480.0` (stored as float), `agent.retries=2`, `agent.deny=()`, `model.max_agents=sessions`, `model.fallback=None`.
- Validation: required root keys `plan/prompt/gate/worktrees/main`, required `[agent].command`, non-empty `[[models]]` with unique names and `sessions >= 1`; errors are `"<path>: <key> must be …"` naming the key (nested keys prefixed `[agent] ` / `[[models]][i] `). Extra keys are ignored. `max_agents < sessions`, capacity `< 1`, duplicate model names, and a `fallback` that names no configured model are errors — all of these would otherwise silently disable the over-session pool rule or the quota fallback (§4/§6).
- Next tasks: `stats.buckets` (SPEC §9) is *not* parsed yet — T16 must add it; T10 uses `model.fallback`; T11/T13 resolve `plan`/`prompt`/`worktrees` against `config.root`.
