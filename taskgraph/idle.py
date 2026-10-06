"""Idle watch and lease anomalies (SPEC §9).

The scheduling loop calls :meth:`IdleWatch.watch` once per tick with the live
agents, the recorded metrics history and an event callback, and this module
decides what to report:

* a model that has agents assigned but reports ``running == 0`` for at least
  :data:`IDLE_SECS` is idle — the server accepts no requests while agents are
  waiting on it — and one ``idle`` event lists, per agent on that model, its
  last tool call and how long ago, plus the lease holders and waiters;
* a lease held longer than :data:`IDLE_LEASE_SECS`, or held by a process that
  never announced itself as a ``taskgraph lease`` wrapper, is an ``anomaly``.

Both are reported **once per episode**: the zero-running clock and the reported
holders are dropped as soon as the observation changes, so a model that
recovers and idles again is reported again.  The zero clock only ever counts
fresh samples recorded by this process for this tick (``sample.at == now``); a
skipped scrape or a restarted scheduler simply re-arms it, so a stale
``state.json`` can never claim a model has been idle for hours.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

from . import agent, lease, leaseprocs, pool, status, trace
from .config import ModelConfig

#: A model with assigned agents that reports ``running == 0`` this long is idle.
IDLE_SECS = 180.0

#: A lease held longer than this is an anomaly (SPEC §9).
IDLE_LEASE_SECS = 480.0

#: ``event(kind, id, detail)``; the scheduler passes its own event writer.
Event = Callable[[str, str, str], None]


class IdleWatch:
    """Cross-tick state: the zero-running clock, logged episodes, reported holders."""

    def __init__(self) -> None:
        self._since: dict[str, float] = {}
        self._logged: set[str] = set()
        self._reported: set[tuple[str, int, int]] = set()

    def watch(
        self,
        models: Sequence[ModelConfig],
        running: Iterable[agent.AgentRecord],
        history: Mapping[str, Sequence[pool.Sample]],
        now: float,
        state_root: str | None,
        event: Event,
    ) -> None:
        """Report idle models and anomalous lease holders for this tick (SPEC §9)."""
        records = sorted(running, key=lambda record: record.id)
        assigned = Counter(record.model for record in records)
        holders = lease.holders(state_root)
        procs = leaseprocs.active_processes(state_root)
        for model in models:
            samples = history.get(model.name)
            latest = samples[-1] if samples else None
            fresh = latest is not None and latest.at == now
            if assigned.get(model.name, 0) and fresh and latest.metrics.running == 0:
                since = self._since.setdefault(model.name, now)
                if now - since >= IDLE_SECS and model.name not in self._logged:
                    self._logged.add(model.name)
                    event("idle", model.name, diagnosis(model.name, records, holders, procs, now))
                continue
            # Load, assignment or the observation itself changed: re-arm.
            self._since.pop(model.name, None)
            self._logged.discard(model.name)
        self._holders(holders, procs, now, event)

    def _holders(
        self,
        holders: Sequence[lease.Slot],
        procs: Sequence[leaseprocs.LeaseProcess],
        now: float,
        event: Event,
    ) -> None:
        """Flag each anomalous holder once per holding (SPEC §9)."""
        announced = {proc.pid for proc in procs}
        self._reported &= {(slot.resource, slot.n, slot.pid) for slot in holders}
        for slot in holders:
            key = (slot.resource, slot.n, slot.pid)
            if key in self._reported:
                continue
            if slot.pid not in announced:
                detail = f"{slot.name} held by a non-lease process (pid {slot.pid})"
                if slot.since > 0:
                    detail += f" for {status.format_age(slot.age(now))}"
            elif slot.age(now) > IDLE_LEASE_SECS:
                detail = f"{slot.name} held {status.format_age(slot.age(now))} by {slot.task or '-'}"
            else:
                continue
            self._reported.add(key)
            event("anomaly", slot.resource, detail)


def diagnosis(
    model: str,
    records: Sequence[agent.AgentRecord],
    holders: Sequence[lease.Slot],
    procs: Sequence[leaseprocs.LeaseProcess],
    now: float,
) -> str:
    """The why for an idle ``model``: its agents, lease holders and waiters (SPEC §9)."""
    held = {slot.pid for slot in holders}
    agents = [agent_note(record, now) for record in records if record.model == model]
    leases = [holder_note(slot, now) for slot in holders]
    waiters = [
        f"{proc.resource} pid {proc.pid} {status.format_age(proc.age(now))}"
        for proc in procs
        if proc.pid not in held
    ]
    return "; ".join(
        (
            "agents " + (", ".join(agents) or "none"),
            "holders " + (", ".join(leases) or "none"),
            "waiters " + (", ".join(waiters) or "none"),
        )
    )


def agent_note(record: agent.AgentRecord, now: float) -> str:
    """``<id> <last tool> <how long ago>`` for one assigned agent (SPEC §9)."""
    seen = agent.last_activity(record.log)
    ago = "no trace" if seen <= 0.0 else f"{status.format_age(max(0.0, now - seen))} ago"
    return f"{record.id} {trace.last_tool(record.log) or '?'} {ago}"


def holder_note(slot: lease.Slot, now: float) -> str:
    """``<resource>/<slot> task <id> <age>`` for one lease holder (SPEC §9)."""
    who = f"task {slot.task}" if slot.task else f"pid {slot.pid}"
    note = f"{slot.resource}/{slot.name} {who}"
    return note if slot.since <= 0 else f"{note} {status.format_age(slot.age(now))}"
