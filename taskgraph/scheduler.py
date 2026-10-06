"""The scheduler loop: fill model slots, watch agents, hand merges off (SPEC §7–§9).

One foreground process per project. Each tick it

* re-reads the project config when the file changed, so an operator can retune
  the model pool without a restart (SPEC §4);
* records one metrics sample per model that has an endpoint;
* drains finished merges and applies their outcome to state;
* watches every running agent — exit, stall, startup hang, quota error;
* starts the next runnable tasks the model pool admits (SPEC §3/§4), after
  handing already-finished work to the single merge worker (SPEC §8).

Everything durable lives in ``state.json`` (SPEC §7), so a restart re-adopts
running agents by ``(pid, start time)`` instead of killing them. Every state
change is appended to ``events.log`` (SPEC §9). Only the merge worker thread
touches git; this loop never blocks on a gate.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from . import agent, assign, events, idle, merge, metrics, order, plan, pool, state, worktree
from .config import Config, ConfigError
from .config import load as config_load

#: Seconds between ticks (SPEC §4's 20 s), overridable via :data:`TICK_ENV`.
TICK_SECS = 20.0

#: Environment override for the tick period, used by tests and quick runs.
TICK_ENV = "TASKGRAPH_TICK"


def env_tick_secs(env: Mapping[str, str] | None = None) -> float:
    """Return ``$TASKGRAPH_TICK`` seconds, else :data:`TICK_SECS`; bad values fall back."""
    raw = (os.environ if env is None else env).get(TICK_ENV)
    if raw is None:
        return TICK_SECS
    try:
        value = float(raw)
    except ValueError:
        return TICK_SECS
    return value if value > 0 else TICK_SECS


class Scheduler:
    """The per-project scheduling loop (SPEC §7–§9).

    ``state_root`` overrides ``$TASKGRAPH_STATE`` (tests); ``now``, ``sample``
    and ``kill_timeout`` are injectable so a tick can be driven deterministically.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        max_agents: int | None = None,
        tick_secs: float = TICK_SECS,
        state_root: Path | str | None = None,
        now: Callable[[], float] = time.time,
        sample: Callable[[str], metrics.Metrics | None] = metrics.sample,
        kill_timeout: float = agent.STALL_KILL_TIMEOUT,
        out: Any = None,
    ):
        self.cfg = cfg
        self.project = Path(cfg.root)
        self.max_agents = max_agents
        self.tick_secs = tick_secs
        self.state_root = state_root
        self._now = now
        self._sample_fn = sample
        self._kill_timeout = kill_timeout
        self.out = sys.stderr if out is None else out

        self.state_path = state.state_path(self.project, state_root)
        self.events_path = state.project_dir(self.project, state_root) / events.EVENTS_LOG
        self.state = state.load(self.state_path)
        self.history: dict[str, list[pool.Sample]] = assign.history_from_state(self.state.samples)

        self.running: dict[str, agent.AgentRecord] = {}
        self.merging: set[str] = set()
        self.forced: dict[str, str] = {}
        self._lease_anomalies: set[int] = set()
        self.idle = idle.IdleWatch()
        self._results: list[merge.MergeResult] = []
        self._result_lock = threading.Lock()
        self.queue = merge.MergeQueue(cfg, on_result=self._merge_result)

        self._lock: state.Lock | None = None
        self._started = False
        self._closed = False
        self._stopping = False
        self._stop = threading.Event()
        self._signals: dict[int, Any] = {}
        self._config_mtime = _mtime(cfg.path)

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Claim the project lock (raising :class:`state.LockError`) and adopt live agents."""
        if self._started:
            return
        self._started = True
        self._lock = state.claim_lock(self.project, self.state_root)
        self._adopt()

    def run(self, *, max_ticks: int | None = None) -> int:
        """Run until stopped; rc 2 if the lock is held or state is unusable (``max_ticks``: tests)."""
        try:
            self.start()
        except (state.LockError, state.StateError) as exc:
            print(f"taskgraph run: {exc}", file=sys.stderr)
            self.close()
            return 2
        self._install_signals()
        try:
            ticks = 0
            while not self._stopping:
                self.tick()
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                if self._stop.wait(self.tick_secs):
                    break
        except KeyboardInterrupt:  # pragma: no cover - defensive
            self._stopping = True
        finally:
            self._restore_signals()
            self.close()
        return 0

    def close(self) -> None:
        """Stop the merge worker and release the lock. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self.queue is not None:
            self.queue.stop()
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _install_signals(self) -> None:
        """Turn SIGTERM/SIGINT into a graceful loop exit (agents keep running)."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._signals[sig] = signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not the main thread / unsupported
                pass

    def _restore_signals(self) -> None:
        for sig, handler in self._signals.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, TypeError):  # pragma: no cover
                pass
        self._signals.clear()

    def _on_signal(self, signum: int, frame: Any) -> None:
        self._stopping = True
        self._stop.set()

    # ---------------------------------------------------------------------- tick

    def tick(self) -> None:
        """Advance the scheduler by one step and persist any state change."""
        now = self._now()
        self._reload()
        self._consume_intents()
        self._record_metrics(now)
        self._drain()
        self._watch(now)
        self._idle_watch(now)
        if not self._stopping:
            self._fill(now)
        self._save()

    def _reload(self) -> None:
        """Re-read the config when its file changed (SPEC §4 retuning)."""
        mtime = _mtime(self.cfg.path)
        if mtime is None or mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        try:
            new = config_load(self.cfg.path)
        except ConfigError as exc:
            self._event("anomaly", "-", f"config reload failed: {exc}")
            return
        if new.models != self.cfg.models:
            summary = ", ".join(f"{m.name}={m.sessions}/{m.max_agents}" for m in new.models)
            self._event("pool", "-", f"models {summary or 'none'}")
        self.cfg = new

    def _consume_intents(self) -> None:
        """Apply ``taskgraph retry`` requests written by another process (SPEC §10).

        Requests are files under the project's ``pending/`` directory rather than
        ``state.json`` edits, so this loop's own saves cannot lose one.  Applying
        a request also forgets the task's record: the operator asked for a new
        attempt, so a dead agent must not go down the "exited" path and re-block
        the task in the same tick.
        """
        for tid in state.take_intents(self.project, self.state_root):
            record = self.running.pop(tid, None)
            if record is not None:
                agent.kill(record, timeout=self._kill_timeout)
            self.state.blocked.pop(tid, None)
            self.state.retries.pop(tid, None)
            self.forced.pop(tid, None)
            self._event("retry", tid, "retry requested by the operator")

    def _record_metrics(self, now: float) -> None:
        """Record one metrics sample per model that has an endpoint (SPEC §4)."""
        for model in self.cfg.models:
            if not model.metrics:
                continue
            scraped = self._sample_fn(model.metrics)
            if scraped is not None:
                self.history.setdefault(model.name, []).append(pool.Sample(now, scraped))

    def _drain(self) -> None:
        """Apply finished merge results gathered by the worker thread."""
        with self._result_lock:
            results, self._results = self._results, []
        for result in results:
            self.merging.discard(result.id)
            if result.ok:
                self.state.blocked.pop(result.id, None)
                self.state.retries.pop(result.id, None)
                self._event("merged", result.id, f"{result.attempts} attempt(s)")
            else:
                self.state.blocked[result.id] = result.reason
                self._event("blocked", result.id, result.reason)

    def _merge_result(self, result: merge.MergeResult) -> None:
        """Merge-worker callback: queue the result for the next tick."""
        with self._result_lock:
            self._results.append(result)

    def _watch(self, now: float) -> None:
        """Inspect every running agent: exit, startup hang, stall, quota."""
        for record in list(self.running.values()):
            if agent.poll(record) == "exited":
                self._on_exit(record)
                continue
            if agent.startup_hang(record.log):
                self._event("stall", record.id, "startup hang")
                self._requeue(record, "startup hang")
                continue
            if agent.stalled(record.log, now, self.cfg.agent.stall_secs):
                hold = agent.oldest_lease(record, root=self.state_root)
                if hold is None:
                    self._event(
                        "stall", record.id, f"no trace growth for {self.cfg.agent.stall_secs:g}s"
                    )
                    self._requeue(record, "stalled")
                else:
                    # A queued/holding lease process keeps the agent silent by
                    # design (SPEC §6): never stall-kill it. A lease that lives
                    # past agent.max_lease_secs is the anomaly instead.
                    age = hold.age(now)
                    if age > self.cfg.agent.max_lease_secs and hold.pid not in self._lease_anomalies:
                        self._lease_anomalies.add(hold.pid)
                        self._event(
                            "anomaly",
                            record.id,
                            f"{hold.resource} lease held {age:.0f}s (pid {hold.pid})",
                        )
                continue
            fallback = agent.fallback_model(record, self.cfg)
            if fallback is not None:
                self._requeue(record, f"quota error; restart on {fallback.name}", force=fallback.name)

    def _idle_watch(self, now: float) -> None:
        """Report idle models and anomalous lease holders (SPEC §9)."""
        self.idle.watch(
            self.cfg.models,
            self.running.values(),
            self.history,
            now,
            self.state_root,
            self._event,
        )

    def _on_exit(self, record: agent.AgentRecord) -> None:
        """An agent process ended: merge when done, else retry or block (SPEC §8)."""
        if worktree.done_path(self.cfg, record.id).is_file():
            self._enqueue(record.id)
            self.running.pop(record.id, None)
            return
        self._requeue(record, "exited without progress/<id>.done")

    def _requeue(self, record: agent.AgentRecord, detail: str, *, force: str | None = None) -> None:
        """Kill/forget ``record`` and either schedule a retry or block the task."""
        self.running.pop(record.id, None)
        agent.kill(record, timeout=self._kill_timeout)
        tid = record.id
        if record.retries >= self.cfg.agent.retries:
            reason = f"{detail}; giving up after {record.retries} retries"
            self.state.blocked[tid] = reason
            self._event("blocked", tid, reason)
            return
        self.state.retries[tid] = record.retries + 1
        if force is not None:
            self.forced[tid] = force
        self._event("retry", tid, f"{detail} (attempt {record.retries + 1}/{self.cfg.agent.retries})")

    def _enqueue(self, tid: str) -> None:
        """Hand ``tid`` to the merge worker once (idempotent)."""
        if tid in self.merging:
            return
        self.merging.add(tid)
        self.queue.enqueue(tid)

    def _fill(self, now: float) -> None:
        """Start runnable tasks while the model pool admits another agent."""
        text = self._plan_text()
        if text is None:
            return
        ready = order.runnable(
            plan.parse(text),
            set(self.running) | self.merging,
            set(self.state.blocked),
            lambda task: worktree.started_rank(self.cfg, task.id),
        )
        assigned = Counter(record.model for record in self.running.values())
        for task in ready:
            if self._stopping:
                break
            if self.max_agents is not None and len(self.running) >= self.max_agents:
                break
            if worktree.done_path(self.cfg, task.id).is_file():
                self._enqueue(task.id)  # finished but never merged (a crash)
                continue
            forced = self.forced.get(task.id)
            if forced is not None and self.cfg.model(forced) is None:
                # The fallback vanished in a config reload: blocking beats a
                # task that can never be picked again.
                self._block(task.id, f"model {forced} is not configured")
                continue
            name = self._pick(task.id, assigned, now)
            if name is None:
                if task.id in self.forced:
                    continue  # this one wants a specific model; try the rest
                break  # the pool is full for everyone
            self._start(task, name, assigned, now)

    def _pick(self, tid: str, assigned: Mapping[str, int], now: float) -> str | None:
        """Return the model the pool gives task ``tid``, or ``None`` if full."""
        models = self.cfg.models
        forced = self.forced.get(tid)
        if forced is not None:
            models = tuple(model for model in models if model.name == forced)
        return assign.choose(models, assigned, self.history, now, self.state.last_extra)

    def _start(self, task: plan.Task, name: str, assigned: Counter, now: float) -> None:
        """Create the worktree if needed and launch the task's agent."""
        model = self.cfg.model(name)
        if model is None:
            self._block(task.id, f"model {name} is not configured")
            return
        # A worktree that is already there holds an interrupted agent's partial
        # work — after a retry, a `taskgraph retry`, or a `stop --agents` — so
        # this agent must be told to continue rather than start over (SPEC §6).
        resume = worktree.is_resume(self.cfg, task.id)
        try:
            if not resume:
                worktree.create(self.cfg, task.id)
            record = agent.start(
                task,
                model,
                worktree.path(self.cfg, task.id),
                self.cfg,
                self.state,
                resume=resume,
                root=self.state_root,
                now=now,
            )
        except (worktree.WorktreeError, agent.AgentError, OSError) as exc:
            self._block(task.id, f"cannot start agent: {exc}")
            return
        self.running[task.id] = record
        self.forced.pop(task.id, None)
        assigned[name] += 1
        suffix = " (resume)" if resume else ""
        self._event("start", task.id, f"{name} pid {record.pid}{suffix}")

    def _block(self, tid: str, reason: str) -> None:
        self.forced.pop(tid, None)
        self.state.blocked[tid] = reason
        self._event("blocked", tid, reason)

    def _plan_text(self) -> str | None:
        try:
            return (self.project / self.cfg.plan).read_text(encoding="utf-8")
        except OSError as exc:
            self._event("anomaly", "-", f"cannot read plan: {exc.strerror or exc}")
            return None

    # ------------------------------------------------------------------ adoption

    def _adopt(self) -> None:
        """Keep live records running; send dead ones through the exit path (SPEC §7)."""
        for data in self.state.agents:
            try:
                record = agent.AgentRecord.from_dict(data)
            except agent.AgentError:
                continue
            if state.start_time_matches(record.pid, record.started):
                self.running[record.id] = record
                self._event("adopt", record.id, f"pid {record.pid} on {record.model}")
            else:
                self._on_exit(record)
        self._save()

    # --------------------------------------------------------------- persistence

    def _save(self) -> None:
        """Write the current records and metric history to ``state.json`` (SPEC §7)."""
        self.state.agents = [record.as_dict() for record in self.running.values()]
        self.state.samples = {
            name: [sample.to_dict() for sample in samples]
            for name, samples in self.history.items()
        }
        try:
            state.save(self.state, self.state_path)
        except OSError as exc:
            self._event("anomaly", "-", f"cannot save state: {exc.strerror or exc}")

    def _event(self, kind: str, tid: str, detail: str = "") -> None:
        """Append one events-log line (SPEC §9) and echo it to the console."""
        line = events.append(self.events_path, kind, tid, detail, self._now())
        print(line, file=self.out, flush=True)


def _mtime(path: Path | str) -> float | None:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None
