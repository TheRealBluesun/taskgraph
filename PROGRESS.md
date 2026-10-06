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

## T07 — Leases

- Files: `taskgraph/lease.py` (the semaphore, 385 lines), `taskgraph/cli.py` (the `lease` wrapper + the `leases` table), `tests/test_lease.py` + `tests/test_lease_cli.py` + `tests/leasehelpers.py` (shared temp-state fixtures; suite 122 tests, ~3.4 s). The smoke test's stub loop now skips `lease`/`leases`. SPEC §0's ~400-line cap forced the split: slot state/staleness/waiting stay in `lease.py`; `run_lease`/`forward_signals`/`format_age`/`format_holders`/`TIMEOUT_EXIT` live in `cli.py` (the command layer).
- API: `lease.state_root/leases_root/resource_dir/audit_path`; `lease.Slot` (`n/pid/cmd/cwd/task/since/path`, `name`, `age`); `lease.slot_files/read_slots/remove_stale/try_acquire/release/holders`; `lease.process_command/looks_like_lease_command/pid_is_live_lease`; `lease.wait_for_slot/wait_message`; `lease.LeaseError`; `cli.run_lease`, `cli.forward_signals`, `cli.format_age/format_holders`, `cli.find_config`. Slot JSON is exactly `{pid,cmd,cwd,task,since}` with `cmd` = the wrapped command (display) and `pid` = the holding `taskgraph lease` process.
- Decisions: a *filename* reserves its index — an unreadable or half-written `slot-0.json` still blocks slot 0 (conservative, never double-claim), while staleness parses the file: `ps -o lstart=,command= -p <pid>` must name a `taskgraph … lease` process. Every read path takes an injectable `alive=` seam (default = the real ps check) so tests can hold slots in-process; `release` re-reads the file and skips unlink if the pid changed (a slot removed as stale and re-claimed is not deleted).
- Decisions: the leased command starts in its **own session** (`start_new_session=True`) and SIGTERM/SIGINT is forwarded to its process *group*. Observed live: with signal-to-pid only, `sh -c 'echo …; sleep 30'` left `sleep` running and the wrapper exited 143 while the grandchild held the machine; group signalling empties the tree (regression test `test_forwarded_signal_stops_the_whole_leased_command_tree`). Exit codes: child's (signal *n* → `128+n`), 127 unrunnable, 124 wait timeout via the internal `timeout=` param, 130 on Ctrl-C while waiting. First "waiting for <res> (held by <task> for N s)" line prints immediately, then every 30 s (both to stderr; `run(out=…)` for tests).
- Decisions: `lease`'s command uses `nargs="+"` instead of `REMAINDER`, because `lease sim --task F03 -- cmd` put `--task F03` *into* the command with REMAINDER; both option orders now parse. Capacity comes from the nearest `taskgraph.toml` walking up from cwd (`cli.find_config`, new public helper) — rc 2 with a clear message when no config is found or the resource is not in `[resources]`. `leases` lists live holders only (stale cleanup stays with waiters) as RESOURCE/SLOT/PID/TASK/AGE/COMMAND.
- Next: T08 reads `lease.LEASE_ENV` (`TASKGRAPH_LEASE`) for shims; T13/T15 should reuse `lease.holders()` + `cli.format_holders` for `status` and must **not** call `remove_stale`/`try_acquire` with a fake `alive` in production code. `taskgraph stop --agents` (T14) kills agent process groups: a lease wrapper in that group forwards to the leased command's own group, so the build tree stops and the slot is released before the wrapper dies.

## T08 — Deny shims

- Files: `taskgraph/shims.py` (182 lines), `tests/test_shims.py` (21 tests; suite 143 tests, ~3 s). No CLI change — SPEC §10 has no shims subcommand; T10 calls the module.
- API: `shim_dir(root=None)`, `create_shims(commands, resource=None, root=None, path=None) -> {cmd: Path}` (state root via `lease.state_root`, i.e. `$TASKGRAPH_STATE/shims`), `real_command(cmd, shims=None, path=None)`, `path_with_shims(base=None, root=None)`, `shim_script(cmd, real, resource)`, `denial_message(cmd, resource)`, `ShimsError`, `DENIED_EXIT=64`, `MARKER`, `LEASE_ENV` (re-export).
- Shim script: `[ -z "${TASKGRAPH_LEASE:-}" ]` → single-quoted `echo <message> >&2; exit 64`, else `exec '<abs realpath>' "$@"`. Empty `TASKGRAPH_LEASE` counts as unset; the message is `_sh_quote`d, so a resource name containing `'`/`$` cannot inject shell (tested with `resource="it's a gpu"`).
- Decisions: deny entries must be bare command names (`[A-Za-z0-9][A-Za-z0-9._+-]*`); a path/metacharacter entry raises `ShimsError` (it could never shadow a bare name, and interpolating it would be an injection). A command that resolves nowhere outside the shim dir is skipped. `real_command` searches PATH with the shim dir removed and returns `os.path.realpath`, so an old shim never wins and the exec target is absolute + symlink-resolved (macOS `/var` → `/private/var`; tests compare realpaths). Scripts are written atomically: `mkstemp` in the shim dir → `chmod 0755` → `os.replace`.
- Beyond the letter of SPEC: `create_shims` also **prunes** generated shims no longer in `agent.deny` (only files whose first two lines contain `MARKER`; foreign files in the shim dir untouched, crashed `.tmp` leftovers do get pruned), so deleting a deny entry really re-enables the command. Idempotent; safe to re-run on every config reload.
- Flake fix in T07's tests (found by running the full suite repeatedly; not related to the new module): `test_lease_cli.CliTest.test_capacity_from_config_limits_concurrent_leases` (0.4 s) and `test_lease.ContentionTest.test_second_holder_waits_for_capacity_one` (0.6 s) slept a fixed time before reading a freshly spawned wrapper's stderr, so under load they asserted against `''`. Both now `select.select([proc.stderr], [], [], 0)` until the waiter actually writes; `time` imports dropped where unused. 14 consecutive full-suite runs OK afterwards.
- Next: T10 must export `shims.path_with_shims()` into the agent env (SPEC §5: the PATH export must reach child shells) and call `shims.create_shims(cfg.agent.deny, resource=…)` before launching. The module depends on lease only for `LEASE_ENV` + `state_root`, so T07b's flock rewrite of `lease.py` does not touch it.

## T06b — Pool review fixes

- Files: `taskgraph/pool.py`, `taskgraph/config.py`, `tests/test_pool.py`, `tests/test_config.py` (suite now 153 tests, ~3.1 s).
- `choose_model` is now **one pass in config order**: `assigned < sessions` → base slot, else the over-session gates for the *same* model, so an earlier model's measured slack beats a later model's free base slot (`test_earlier_model_slack_beats_later_base_slot`); `last_extra[name] = now` recorded only on that over-session branch. New `test_assigned_above_max_agents_gets_nothing` covers a limit lowered at runtime.
- Signature is now `choose_model(models, assigned, recent, history, now, last_extra)`: `recent` drives the slack gate (≥`MIN_SAMPLES`, mean `running` < `sessions-0.5`, `waiting == 0`), `history` the throughput veto. New `recent_samples(samples, now, window=RECENT_WINDOW=180.0)` (future samples dropped, boundary inclusive); tests prove idle-recent/busy-history grants, busy-recent blocks, and a burst still vetoes from `history` after the recent window went quiet. T13 must keep full history per model and pass `recent_samples(history, now)` as `recent`.
- Config §1: `sessions`/`max_agents` minimums are 0, so `sessions = max_agents = 0` disables a model (SPEC §1/§4); `max_agents >= sessions` still enforced and `[resources]` stays ≥ 1. A zero-capacity model is never chosen (base `assigned < 0` false, over-session `assigned >= 0` true).
- Next: T07b (flock leases) is independent; T17's `models` table should reuse `token_rates` over the scheduler's *full history*, not the recent window.
