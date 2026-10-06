## T01 — Skeleton

- Files: `taskgraph/__init__.py` (`__version__ = "0.1.0"`), `taskgraph/cli.py`, `bin/taskgraph`, `tests/test_smoke.py`, `README.md`.
- `cli.build_parser()` defines all SPEC §10 subcommands (`run`, `status`, `stats`, `lease`, `leases`, `stop`, `retry`) plus `models` (SPEC §4); `SUBCOMMANDS: dict[name, help]` drives creation, so later tasks just replace the `func` default per subcommand.
- Subcommand flags are already in place: `run --project/--max-agents/--dry-run`, `stats --since` (default `6h`), `stop --agents`, `lease <resource> --task ID -- cmd…` (REMAINDER), `retry <id>`.
- Every handler is `_not_implemented(args)`: prints `taskgraph <cmd>: not implemented` to stderr, returns 1. No command → help on stdout, rc 2. Next tasks should set `func` on their parser (or dispatch in `main`) instead of editing `_not_implemented`.
- `bin/taskgraph` prepends the repo root to `sys.path` so the script runs uninstalled from a checkout; tests never need `pip install`.
