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

## T03 — Plan parser

- Files: `taskgraph/plan.py`, `tests/test_plan.py` (16 tests; suite now 38 tests, 0.07 s).
- API: `parse(text) -> dict[id, Task]` (dict insertion order = plan order) and `mark_done(text, tid) -> str`; `PlanError` on unknown id. `Task` = frozen dataclass `id, text, done, deps, res`.
- Line pattern `^[ \t]*[-*][ \t]+\[[ xX]\][ \t]+<id>(<rest>)?[ \t]*$` with id `[A-Z]+[0-9]+[a-z]?`; every other line (including lowercase ids, prose, unchecked non-list lines) is ignored. `[deps: A, B]` / `[res: A, B]` are stripped from `text` (tags are metadata, not prompt content), split on commas, blanks dropped. `res` is a tuple for symmetry with `deps` even though SPEC calls it display-only (nothing consumes it yet; T13/T15 may display it).
- `mark_done` edits only the matched line by replacing the marker char in place (preserves spacing/CRLF/no-trailing-newline); already-done → unchanged; no line for id → `PlanError`.
- Next tasks: T04 uses `parse` output + dict order for "plan order"; unknown/cross-dependency ids are left as-is here, ordering decides they count as done.

## T04 — Ordering

- Files: `taskgraph/order.py`, `tests/test_order.py` (15 new tests; suite now 53 tests, 0.07 s).
- API: `runnable(tasks, running=(), blocked=(), started=None) -> list[Task]` and `critical_paths(tasks) -> dict[id, chain]` (exposed so tests can assert chains directly).
- `started` is `Callable[[Task], int]`, lower = further along (0 notes+worktree, 1 worktree, 2 nothing); omitted → every task equal. Sort key is `(-chain, started_rank, plan_index)`, so plan order (dict insertion order) is only the final tie-break.
- Chain = `1 + max(chain(pending dependents))`; done dependents are skipped, and a DFS back-edge (cycle) contributes 0 so `critical_paths` always terminates. Verified A1-last-in-plan wins over leaves, and A↔B cycles yield an empty `runnable`.
- Next tasks: T05 metrics and T06 pool are independent. T13 (`run --dry-run`) should call `runnable` with a `started` callable built from worktree existence + `progress/<id>.md`; `blocked` ids come from scheduler state.

## T05 — Metrics parsing

- Files: `taskgraph/metrics.py`, `tests/test_metrics.py` (11 new tests; suite now 64 tests, 0.08 s).
- API: `parse(text) -> Metrics` (frozen dataclass `running`, `waiting`, `generation_tokens`, all float, 0.0 defaults) and `sample(url, timeout=4) -> Metrics | None`; metric-name constants `RUNNING`/`WAITING`/`GENERATION_TOKENS`.
- `parse` sums every series of the exact metric names (per-engine label sets), so `vllm:generation_tokens_total_created` and unrelated counters (e.g. `prompt_tokens_total`) do not leak in. Skips comments, blank lines, non-finite values (`NaN`/`Inf`) and unparseable lines rather than poisoning sums.
- `sample` uses `urllib.request.urlopen` and returns `None` on `OSError`/`ValueError`/`http.client.HTTPException` (covers `URLError`/`HTTPError`); a skipped sample is normal per SPEC §4.
- No network in tests: `sample` is tested through `file://` URLs. Verified separately over loopback HTTP (200 → parsed) and a refused port (`None`) during development.
- Next: T06 `choose_model` consumes `Metrics.running`/`.waiting` (and `assigned`, `now`, `last_extra`); T17's `models` table needs consecutive `generation_tokens` samples bucketed by `running`.

## T06 — Model pool policy

- Files: `taskgraph/pool.py`, `tests/test_pool.py` (23 new tests; suite now 87 tests, 0.09 s).
- API: `choose_model(models, assigned, load, now, last_extra) -> name | None` (exact SPEC §4 signature), plus `Sample(at, metrics)`, `token_rates(samples) -> dict[bucket, list[float]]` and constants `MIN_SAMPLES=9`, `EXTRA_SPACING=180.0`, `MIN_BUCKET_SAMPLES=20`.
- Two passes: every model's free *base* slot (`assigned < sessions`, config order) is offered before any over-session slot, so a later model's spare base slot beats an earlier model's extra slot. Over-session needs: `assigned < max_agents`, a `metrics` URL, ≥9 samples, mean `running` < `sessions-0.5` (strict), max `waiting` == 0, `now - last_extra[name] >= 180` (boundary allowed), and no throughput veto. Grant mutates `last_extra[name] = now` in place, after deciding.
- `load` is `Mapping[name, Sequence[Sample]]` — samples carry a timestamp, because the veto needs `generation_tokens_total` deltas. `Sample.at` must share the clock with `now`/`last_extra`. `token_rates` sorts by `at`, files each interval's tokens/s under the *later* sample's `round(running)` bucket, and skips dt≤0 and counter resets. Veto fires when both buckets (sessions, sessions+1) have ≥20 rates and mean(sessions+1) < mean(sessions).
- Both the slack gate and the veto read the same `load[name]` window: the scheduler must keep a window long enough to hold ≥20 rates in a busy bucket while still averaging below `sessions-0.5`, or the veto never fires (test builds a 180-sample history: bursts at concurrency 1 and 2 then an idle tail).
- Next: T17 can reuse `token_rates` to print the per-bucket tokens/s table (peak bucket) for `taskgraph models`; T13 wires `assigned` from running records and records `Sample`s into scheduler state. No CLI change in T06.
