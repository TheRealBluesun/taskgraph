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

### How a task finishes

The agent writes `progress/<id>.done` when its task is verified; a single merge worker then commits
the worktree, rebases on `main`, runs the configured `gate`, fast-forwards `main`, ticks the plan line
and appends the task's notes to `PROGRESS.md`. Conflicts, a failing gate, or a huge commit block the
task (with the reason in `taskgraph status`) instead of merging it; `taskgraph retry <id>` resumes it
from the same worktree. An agent that exits without `.done` is resumed up to `[agent] retries` times.

## Tests

```
python3 -m unittest discover -s tests -q
```

Tests never call omp, a model or the network: agents are faked with small shell scripts, git repos are
created in temp dirs, and metrics are injected.

See `SPEC.md` for the authoritative behaviour and `PLAN.md` for the build order.
