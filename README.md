# taskgraph

`taskgraph` runs a project's task list (`PLAN.md`) with several coding agents in parallel, each in
its own git worktree. It schedules tasks by dependency and critical path, shares scarce resources
(one simulator, a model server's sessions) through leases, merges finished work through a serialized
gate, and survives its own restarts without killing agents. Everything project-specific lives in the
project's `taskgraph.toml`.

Requirements: Python ≥ 3.11, standard library only. Run from a checkout with `bin/taskgraph …`
(no install needed).

## Usage

Put a `taskgraph.toml` in the project root (see [`examples/elixir/taskgraph.toml`](examples/elixir/taskgraph.toml)
for a fully commented one) and a task list like:

```
- [ ] F03 Top strip per SPEC … [deps: F01, F02] [res: mac]
```

Then:

```sh
bin/taskgraph run                  # start the scheduler in the foreground (Ctrl-C / SIGTERM stops it,
                                   # leaving agents running — restart with `run` and they are adopted)
bin/taskgraph run --dry-run        # print the runnable tasks in order and the model each would get
bin/taskgraph run --max-agents 4   # cap concurrent agents
bin/taskgraph status               # running agents, merge queue, blocked tasks, leases, model load
bin/taskgraph stats --since 6h     # per task: model time vs tool time, bucketed (build/test/lease-wait)
bin/taskgraph models               # tokens/s per model by concurrency (pick `sessions` from this)
bin/taskgraph leases               # who holds each shared resource right now
bin/taskgraph side side -- <cmd>   # lease a model's `side` job-class capacity, run <cmd> with TASKGRAPH_MODEL
bin/taskgraph router               # serve the OpenAI-compatible request router (SPEC §11)
bin/taskgraph stop [--agents]      # stop the scheduler (--agents: kill its agents too)
bin/taskgraph retry <id>           # unblock a blocked task and resume its worktree
```

Every scheduler is single-instance per project (`stop` before starting another). Scheduler state and
resource leases live under `$TASKGRAPH_STATE` (default `~/.taskgraph`), outside the worktrees, so agents
cannot rewrite the machinery; each agent's trace is `<worktree>/logs/<id>-<HHMMSS>.log`.

### Shared resources

A resource is a counting semaphore (`[resources] simulator = 1`) shared by every process on the
machine, held with `flock` — a holder that dies by any means releases it at once. Projects run
commands through it:

```sh
taskgraph lease simulator --task "$TASK_ID" -- xcodebuild …
```

`[agent] deny = ["xcodebuild", "xcrun"]` installs PATH shims for those commands: an agent that tries
them directly is told to go through the project's wrapper (`./dev.sh`), which takes the lease. See
[`examples/elixir/dev-sh-lease.md`](examples/elixir/dev-sh-lease.md) for a worked migration of a
project's ad-hoc lock dir.

### The request router

An agent is not bound to a model: point its OpenAI client at one endpoint,
`taskgraph router` (default `http://127.0.0.1:8080/v1`), and every *request* is dispatched to the best
worker with a free slot — affinity to the server that served this agent last (keeps the prefix cache
warm), else `[[workers]]` order, else a FIFO wait (by `X-Taskgraph-Priority`, then arrival).

```toml
[[workers]]
name = "flash"                    # free-form; defaults to `model`
upstream = "http://10.10.10.10:8000"
model = "local-vllm-flash/Qwen3.8-Flash-Next"   # the request's `model` field is rewritten to this
concurrency = 1                   # request slots: the second long-context prompt would thrash the cache
max_context = 70000               # prompts estimated above this (chars/3.5) skip this worker

[[workers]]
name = "deepseek"
upstream = "https://api.deepseek.com"
model = "deepseek-v4-flash"
concurrency = 4
api_key_env = "DEEPSEEK_API_KEY"  # Authorization: Bearer $DEEPSEEK_API_KEY — never logged
overflow = true                   # paid: only used while every local worker is full
max_requests_per_hour = 60        # required for an overflow worker: the spend cap
```

`GET /router/stats` reports per worker in-flight/queued/admitted/done/errors/busy-seconds/utilization
(last 10 minutes), the affinity hit rate, and each agent's worker switches. `--queue-timeout SECS`
makes a request that waited too long answer 503 instead of waiting forever (default: wait); `--port 0`
picks a free port.

The router is a proxy, so it is also hardened like one: only `/v1/…` paths are forwarded (never a
`//host`, a control character or a `..` segment that could rewrite the upstream target), redirects
are not followed (an injected api key must not travel to another host), request bodies are capped at
8 MB, `messages` must be a list, and an unreachable worker fails over to the next candidate. Other
`/v1/…` methods (e.g. `GET /v1/models`) pass through to the highest-priority worker without taking a
generation slot, and a queued request whose client disconnected gives up its place.

### How a task finishes

The agent writes `progress/<id>.done` when its task is verified; a single merge worker then commits
the worktree, rebases on `main`, runs the configured `gate`, fast-forwards `main`, ticks the plan line
and appends the task's notes to `PROGRESS.md`. Conflicts, a failing gate, or a huge commit block the
task (with the reason in `taskgraph status`) instead of merging it; `taskgraph retry <id>` resumes it
from the same worktree. An agent that exits without `.done` is resumed up to `[agent] retries` times.

When `[merge] resolver_prompt` names a prompt file, a rebase conflict is instead handed to one short
resolver agent in the mid-rebase worktree — its prompt lists each conflicted hunk with both sides and
the two commits' messages, and the rule "keep both intents" — which may only `git add` and
`git rebase --continue`, after which the normal build and gate decide. A resolver or gate failure still
blocks the task, and every successful resolution is logged as `resolved <id> by agent (<n> files)`.

When the top-priority model has a free agent slot and nothing is queued for it, a young agent running on
a lower tier (started at most `[scheduler] upgrade_window` ago, default `"20m"`; `0` disables) is restarted
there through the same resume path — critical path first, at most one per tick and per task. Each move is
logged as `upgrade <id> <from> -> <to>`.

An agent that stops making progress is killed and resumed through the same path: no trace growth for
`[agent] stall_secs` (unless a `taskgraph lease` wrapper in its worktree explains the silence), omp never
starting (`Still starting after`), or a degenerate loop — its last `[runner] loop_window` tool calls
(default 20) identical, i.e. the same name and arguments, as when a task issues the same `ls` thousands of
times and the growing trace keeps the stall watchdog quiet. `loop_window = 0` disables the loop check.

## Tests

```
python3 -m unittest discover -s tests -q
```

Tests never call omp, a model or the network: agents are faked with small shell scripts, git repos are
created in temp dirs, metrics are injected, and the router is exercised against loopback fake
upstreams.

See `SPEC.md` for the authoritative behaviour and `PLAN.md` for the build order.
