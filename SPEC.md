# taskgraph — SPEC (authoritative)

taskgraph runs a project's task list with parallel coding agents (omp) in git worktrees: it schedules tasks
by dependency and critical path, shares scarce resources (one iOS simulator, a model server's sessions) through
leases, merges finished work through a serialized gate, and survives its own restarts without killing agents.
It is project-agnostic: everything project-specific lives in the project's `taskgraph.toml`.

It replaces `Elixir/tools/graph.py`, `ralph-run-omp.sh` and dev.sh's ad-hoc simulator lock. Every rule below
comes from a real failure in that system (noted as *why:*).

## 0. Ground rules for the implementation

- Python ≥ 3.11, **standard library only** (tomllib, subprocess, json, threading, argparse, urllib, unittest).
- Package `taskgraph/` (importable), entry point `bin/taskgraph` (a 3-line script calling `taskgraph.cli:main`).
- Tests: `python3 -m unittest discover -s tests -q` must pass. Tests never call omp, a model or the network:
  agents are faked with small shell scripts, git repos are created in temp dirs, metrics are injected.
- Pure logic (parsing, ordering, pool policy, stats) lives in modules without I/O so it is unit-testable.
- No module over ~400 lines. Clear names, docstrings on public functions, comments only for non-obvious *why*.

## 1. Project config — `taskgraph.toml` (in the project root)

```toml
plan = "PLAN.md"                 # task list (format §2)
prompt = "PROMPT.md"             # general agent instructions, prepended with the task header (§6)
gate = "./dev.sh build && ./dev.sh test"   # must pass in the rebased worktree before a merge
worktrees = "../elixir-wt"       # worktree base dir (relative to project root)
main = "main"                    # integration branch
links = [".env"]                 # files symlinked into each new worktree (secrets, never copied)

[resources]                      # name = capacity (concurrent leases)
simulator = 1

[agent]
command = "omp -p --mode json --auto-approve --no-session --model {model} --thinking high --max-time 120m --config {overlay} @{prompt_file}"
overlay = "omp-agent.yml"        # taskgraph ships a default (auto-background off, §6); path relative to taskgraph's share dir unless absolute
stall_secs = 480                 # no trace growth for this long = stalled → kill (resume via retry)
retries = 2                      # resumes after an agent exits without finishing, before blocking
deny = ["xcodebuild", "xcrun"]   # commands agents may run only inside a lease (§5)

[[models]]
name = "local-vllm-27b/Qwen3.8-27B"
sessions = 3                     # concurrent requests the server handles well
max_agents = 5                   # agents allowed when metrics show slack (§4)
metrics = "http://10.10.10.15:8002/metrics"   # vLLM Prometheus endpoint (optional)

[[models]]
name = "local-vllm/Qwen3.8-Flash-Next"
sessions = 1
```

`taskgraph.config.load(path) -> Config` (dataclasses), validating required keys with clear error messages.

## 2. Plan format

One task per line; other lines are ignored:

```
- [ ] F03 Top strip per SPEC … [deps: F01, F02] [res: mac]
- [x] LX01 Linux portability … [res: mac]
```

- id: `[A-Z]+[0-9]+[a-z]?`; done = `[x]`.
- `[deps: A, B]` — ids that must be done first (unknown ids count as done).
- `[res: …]` — informational tag kept for display (resource *leases* are taken at run time, §5).
- `taskgraph.plan.parse(text) -> dict[id, Task]`, `mark_done(text, id) -> text` (only that line changes).

## 3. Ordering (pure: `taskgraph.order`)

`runnable(tasks, running, blocked) -> list[Task]`: not done, not running, not blocked, all deps done; sorted by

1. **critical path**, descending: `chain(t) = 1 + max(chain(c) for pending dependents c)` (0 dependents → 1).
   *why:* LX02 (unblocks the test-speed fix) waited behind leaf tasks.
2. **started first**: a worktree with notes (`progress/<id>.md`) before a worktree without, before no worktree.
   *why:* restarts otherwise gave nearly finished tasks' slots to fresh ones. (Passed in as a callable so the
   module stays pure.)
3. plan order.

Cycles must not recurse forever (treat a back-edge as chain 0).

## 4. Model pool (pure: `taskgraph.pool`)

`[[models]]` order is **priority**: fastest/best first (e.g. a local GPU), auxiliary paid models last, so an
auxiliary agent starts only when every earlier model is full and runnable work remains. The scheduler re-reads
the `[[models]]` section every tick when the file's mtime changes (log `pool …`), so an operator can retune
("stop using .10 today", "allow 4 auxiliary agents") without a restart; agents already running keep their model,
and a model over its new limit simply gets no new agents. `sessions = max_agents = 0` disables a model.

**Job classes (later task).** Capacity is not one number: a model's KV cache may fit only one *long-context*
coding agent but still have room for short jobs. Example (user, 2026-10-05): the .10 Flash server takes ONE
long-context agent, plus at most one periodic short vision job (e.g. a screenshot critique), and nothing more. So
`[[models]]` gets an optional `[models.classes]` table, e.g. `{ agent = 1, side = 1 }`; plan tasks default to class
`agent`; side jobs (`taskgraph side <class> -- <cmd>`, e.g. a critic run) lease a model's `side` capacity and
are preferred onto a model whose agent is currently blocked on a tool. Until that task lands, `sessions`/`max_agents`
count `agent`-class only.

Each running agent is assigned one model. `choose_model(models, assigned, load, now, last_extra) -> name | None`:

- If a model has `assigned < sessions` → it is free (first in config order wins).
- Else if `assigned < max_agents`, the model has metrics, at least 9 samples exist (≈3 min at the 20 s tick),
  the mean of `running` < `sessions − 0.5`, max `waiting` == 0, and no over-session agent started on it in the
  last 180 s → free (record `last_extra[name] = now`).
- Else not free. *why:* agents spend ~half their time blocked on tools (simulator, builds), leaving GPUs idle
  while every slot is "taken"; vLLM's own counters show the real slack.

`taskgraph.metrics.sample(url, timeout=4) -> (running, waiting)` parses vLLM's Prometheus text (sum of
`vllm:num_requests_running{…}` and `vllm:num_requests_waiting{…}` series); network errors → `None` (sample skipped).

**More concurrency is not always more throughput.** On the 27B server (speculative decoding + pipeline
parallel over two GPUs) total generation fell as concurrency rose: 1 request 81 tok/s, 2 → 69, 3 → 49. So
`sessions` must be the concurrency where *total* throughput peaks, not the server's request limit.
`taskgraph.metrics` also samples the counter `vllm:generation_tokens_total` (sum of series); `taskgraph models`
prints, per model, total tokens/s bucketed by the `running` value at sample time (from the scheduler's recorded
samples), and marks the peak bucket — the operator sets `sessions` from it. The pool never starts an
over-session agent on a model whose recorded data show the bucket `sessions+1` with lower total throughput than
bucket `sessions` (once each bucket has ≥ 20 samples).

## 5. Resource leases (`taskgraph.lease`)

A counting semaphore per resource, shared by all processes on the machine, replacing dev.sh's lock dir.

- State dir: `$TASKGRAPH_STATE/leases/<resource>/`; one file per slot `slot-<n>` (n < capacity), created once and
  never deleted. **A slot is held iff some process holds `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on it.** The lease
  process acquires the lock, then truncates and writes `{pid, cmd, cwd, task, since}` JSON into the file (for
  status/audit only — never used to decide ownership), keeps the fd open for the whole lease, and releases by
  closing it. When a holder dies by any means (exit, crash, SIGKILL) the kernel drops the lock at once.
  *why:* ownership by pid needed stale detection (dead pid? reused pid? which command?), and every variant
  raced: two waiters removing the same stale slot, a failed `ps` marking a live holder dead, an agent's own shell
  matching the command pattern, and an agent that copied the lock code into its long-lived shell deadlocked
  everything. With flock there is no stale state to detect; only a process actually holding the lock owns a
  slot. Waiters simply try each slot's lock in turn every 2 s.
- CLI: `taskgraph lease <resource> [--task ID] -- <cmd…>`: wait for a slot (poll 2 s; print one line
  "waiting for <resource> (held by <task> for 73 s)" every 30 s), run cmd as a child with
  `TASKGRAPH_LEASE=<resource>` in its env, release the slot in `finally` and on SIGTERM/SIGINT (forward the signal to
  the child, wait, then release). Exit code = child's.
- Audit: append `time acquire|release|stale-removed resource slot pid task cwd` lines to `leases/audit.log`.
- `taskgraph leases` prints current holders with ages.

**Deny shims** (`taskgraph.shims`): taskgraph creates `$TASKGRAPH_STATE/shims/<cmd>` for each `agent.deny` entry:
a sh script that execs the real binary (absolute path resolved at creation, skipping the shim dir) only when
`TASKGRAPH_LEASE` is set, else prints "`<cmd>` is disabled for agents: run it through the project's lease
wrapper (e.g. ./dev.sh) — it waits for the shared <resource>" and exits 64. Agents get the shim dir first on PATH
(exported — *why:* a `local PATH` in the runner did not reach child shells). Projects then call
`taskgraph lease simulator -- xcodebuild …` inside their dev.sh.

Infrastructure lives **outside worktrees** (taskgraph's install + `$TASKGRAPH_STATE`), so agents can't rewrite
it and fixes reach every agent at once. *why:* per-worktree copies conflicted on rebase and one agent reverted
its lock code.

## 6. Agent runner (`taskgraph.agent`)

`start(task, model, worktree, cfg, state) -> AgentRecord` launches the agent **detached** (`start_new_session=True`,
stdin `/dev/null`, stdout+stderr to `<worktree>/logs/<id>-<HHMMSS>.log`) and returns `{id, pid, pgid, model,
worktree, log, started, retries}`. The prompt file `<worktree>/.taskgraph-prompt.md` (git-excluded via
`.git/info/exclude`) = task header + project prompt:

```
YOUR TASK IS **<id>** (do only this one): <id> <text>
You are one of several agents working in parallel, each in its own git worktree. Therefore:
- Do NOT edit <plan> and do not run git commands.
- Write your notes (2–5 bullets) to progress/<id>.md.
- When — and only when — the task is complete and verified, create the empty file progress/<id>.done.
- Run long commands in the foreground and wait for them; never end your turn while a background job is running.
- Never modify or work around the shared-resource tooling (leases, shims); if a command waits for a lease, wait.
```

A resumed worktree (exists already) prepends: "NOTE: this worktree already contains PARTIAL work on this task from
an earlier, interrupted agent (see `git status` / `git diff`). Review it and continue from it; do not start over."

Default overlay `share/omp-agent.yml` disables `bash.autoBackground` and `eval.autoBackground`
(*why:* an agent's tour got auto-backgrounded, it ended its turn, and omp exited mid-task).

`poll(record) -> "running" | "exited"`, plus watchdogs evaluated each tick:
- **stall**: log size unchanged for `stall_secs` → kill the process group (SIGTERM, then SIGKILL after 10 s) — EXCEPT while a `taskgraph lease` process whose cwd is inside the agent's worktree is alive (waiting for or holding a resource): omp does not stream a running command's output into its trace, so an agent queued for the simulator is silent by design (*why:* three healthy queued agents were killed as "stalled", 2026-10-05). Each lease process extends the deadline; a lease held longer than `agent.max_lease_secs` (default 1800) is reported as an `anomaly` event instead.
- **startup hang**: only while the trace has no `"type":"tool_execution_start"` yet, and only a line beginning
  `Still starting after` within the first 20 lines (*why:* an agent reading the runner's source put that phrase
  into its trace and the old check killed it).
- **loop**: the last `[runner] loop_window` tool calls (default 20; `0` disables) all identical — same tool
  name and same arguments; the per-call id is ignored — → kill and resume with reason `looping` (*why:* a
  degenerate loop keeps the trace growing, so the stall watchdog alone never catches it: Elixir's D5 issued the
  same `ls` ~12,000 times in 2 h while every tick saw fresh trace, 2026-10-06).
- **quota**: trace contains a rate-limit/quota error and the model has a `fallback` model configured → restart
  on the fallback (counts as a retry).

## 7. Scheduler state & restarts (`taskgraph.state`)

`$TASKGRAPH_STATE/projects/<hash-of-project-root>/state.json` holds running AgentRecords, blocked tasks (with
reasons), retry counts, pool `last_extra` and recent load samples. Written atomically (temp + rename) after every
change. On start, the scheduler loads it and **re-adopts** every record whose pid is alive and whose process
start time matches the record; dead ones go through the normal "exited" path (retry/merge).
*why:* each of five scheduler restarts in one evening killed every agent mid-task.

Single instance per project: `state/lock` holds the scheduler pid; refuse to start if that pid is alive.
`taskgraph stop` stops the scheduler only (agents keep running); `taskgraph stop --agents` also stops agents.

**Upgrade on free** (`taskgraph.upgrade`, pure `pick_upgrade`): when the top-priority model has a free `agent`
slot and no runnable task is waiting for a slot, restart one running agent from a lower tier there through the
normal resume path (same worktree, resume note). The task must be on the critical path (§3), have started at
most `[scheduler] upgrade_window` ago (default `"20m"`; `0` disables), and be neither merging, already upgraded
once nor `forced` onto its model (a quota `fallback` must not go back to the model that rate-limited it). At
most one upgrade per tick and per task (no ping-pong); log `upgrade <id> <from> -> <to>`.
*why:* a task that started when only the slow auxiliary tier had room kept it for hours while the local GPU idled.

## 8. Merge queue (`taskgraph.merge`)

When an agent exits with `progress/<id>.done`, the task enters a FIFO merge queue processed by **one worker
thread** (the scheduling loop keeps running — *why:* a 3–5 min gate inside the loop froze all scheduling):

1. In the worktree: `git add -A -- . ':!logs'`, commit `"<id>: work (taskgraph)"`.
2. `git rebase <main>`; on conflict → `git rebase --abort`, block with the conflicting paths listed.
3. Run `gate` in the worktree (output to `logs/gate.log`); fail → block with the last 20 lines summarized.
4. In the project root: `git merge --ff-only task/<id>`; if main moved, go back to step 2 (max 3 times).
5. Tick the plan line (`mark_done`), append `## <id>` + the notes file to `PROGRESS.md` if it exists, commit
   both with the configured trailer lines, remove the worktree and branch.

**Commit guard** (before step 2): block the merge — reason listing the offending paths — when the task commit adds
more than `merge.max_files` (200) files, any single file over `merge.max_file_mb` (5 MB), or any path under a
build/output directory (`.build/`, `build/`, `DerivedData/`, `node_modules/`, `__pycache__/`, `target/`, `dist/`,
plus `merge.deny_paths` from the toml). *why:* a task once committed 5,833 SwiftPM build files because its
output dir was missing from .gitignore; only a failed fast-forward stopped it reaching main.

An agent that exits **without** `.done` is resumed (same worktree, resume note) up to `agent.retries` times,
then blocked.

## 9. Events, status, stats

- Every state change appends one line to `state/events.log`:
  `HH:MM:SS start|adopt|retry|merged|blocked|stall|idle|anomaly <id> <detail>` — designed for `tail -F | grep`.
- `taskgraph status`: table of running (id, model, age, last trace activity, current tool), merge queue,
  blocked (reason), next runnable (in order), lease holders, model load (mean running/waiting).
- `taskgraph stats [--since 6h]`: per-task and total wall-time split from the omp JSON traces — model time
  (sum of assistant `message_end` durations) vs tool time (gap to the next message), tool time bucketed by the
  first tool call's command via project-configurable regexes (`[stats.buckets]` in the toml; default buckets:
  `lease-wait`, `build`, `test`, `wait` (omp's wait tool), `other`). *why:* "where does the time go" decided
  every improvement; it should be one command.
- **idle watch** (in the scheduling loop): when a model with metrics shows `running == 0` for ≥ 3 min while it
  has assigned agents, log `idle <model> <diagnosis>` where diagnosis lists, per agent on that model, its last
  tool call and how long ago, plus lease holders and waiters; flag `anomaly` when a lease is held > 8 min or by
  a non-lease process.

## 10. CLI summary

```
taskgraph run [--project DIR] [--max-agents N] [--dry-run]   # start/resume the scheduler (foreground)
taskgraph status | stats | leases | stop [--agents]
taskgraph lease <resource> [--task ID] -- <cmd…>
taskgraph retry <id>        # unblock a blocked task (resumes its worktree)
```

`--dry-run` prints the ordered runnable list and the model each would get, then exits.


## 11. Work and workers: the request router (user design, 2026-10-05)

Work and workers are separate pools. **Work** = fungible units with priorities (plan tasks → agents, ordered by
§3). **Workers** = model backends with capacity (e.g. .10 Flash × 1 long-context request, .15 27B × 2,
DeepSeek × N overflow). An agent is NOT bound to a model: every agent talks to one OpenAI-compatible endpoint,
`taskgraph router`, and each **generation request** is dispatched to the best worker with a free slot:

1. **affinity** — the worker that served this agent's previous request, if it has a free slot (keeps the
   server's prefix cache warm; a cold 60k-token prefill costs 30–60 s). Agent identity = `X-Taskgraph-Agent`
   header if present, else a hash of the request's first system + first user message.
2. else the **highest-priority** worker (config order) with a free slot;
3. else wait in a FIFO queue (by task priority, then arrival) until any worker frees.

Per worker: `upstream` URL, `model` (the router rewrites the request's `model` field), optional `api_key_env`
(Authorization header injected from that env var — never logged), `concurrency` (slots), `max_context` (requests
whose estimated prompt tokens — chars/3.5 — exceed it skip that worker), and `overflow = true` for paid workers,
which also cap total spend via `max_requests_per_hour`. Request-level capacity is what protects a server (a
second long-context request on .10 thrashes its KV cache); task-level agent counts only need to be high enough
that workers stay busy (the scheduler raises the agent count while the router's queue is empty and any worker is
idle, lowers it while requests wait > 30 s).

`GET /router/stats`: per worker in_flight/queued/admitted/busy-seconds/utilization (last 10 min), affinity hit
rate, per-agent worker switches. Model switching mid-task is allowed (all workers are capable coders); a task may
pin a worker class with `[pin: local]` in the plan if switching hurts it.
