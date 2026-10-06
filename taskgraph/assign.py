"""Turn the saved pool state into task→model assignments (SPEC §3, §4, §10).

The scheduler calls :func:`choose` on every free slot; ``taskgraph run
--dry-run`` calls :func:`dry_run`, which lists the runnable tasks in scheduling
order with the model each would get. Both read the same recorded metric history
and ``last_extra`` map, so the preview matches what the loop would do.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import agent, order, plan, pool, state, worktree
from .config import Config


def history_from_state(
    samples: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[pool.Sample]]:
    """Rebuild the per-model sample history stored in ``state.samples``.

    Malformed rows are dropped rather than failing a scheduler start.
    """
    history: dict[str, list[pool.Sample]] = {}
    for name, rows in samples.items():
        parsed = [
            sample for sample in (pool.Sample.from_dict(row) for row in rows) if sample is not None
        ]
        history[str(name)] = parsed
    return history


def choose(
    models: Sequence[Any],
    assigned: Mapping[str, int],
    history: Mapping[str, Sequence[pool.Sample]],
    now: float,
    last_extra: Mapping[str, float],
) -> str | None:
    """Pick the model for the next agent, deriving the slack window from history.

    Thin wrapper over :func:`taskgraph.pool.choose_model`: the recent slack
    window is recomputed from the full history each time, so a caller only keeps
    one buffer per model. ``last_extra`` is mutated in place on an over-session
    grant (SPEC §4).
    """
    recent = {name: pool.recent_samples(rows, now) for name, rows in history.items()}
    return pool.choose_model(models, assigned, recent, history, now, last_extra)


def dry_run(
    cfg: Config,
    *,
    max_agents: int | None = None,
    state_root: Path | str | None = None,
    now: float | None = None,
) -> str:
    """Return the ordered runnable tasks and the model each would get (SPEC §10).

    Greedy simulation against the saved state: every runnable task is listed in
    scheduling order with its model, or ``-`` when no slot is free. Already
    running or blocked tasks are skipped, exactly as the loop skips them.
    """
    now = time.time() if now is None else now
    try:
        text = (Path(cfg.root) / cfg.plan).read_text(encoding="utf-8")
    except OSError as exc:
        return f"cannot read plan: {exc.strerror or exc}"
    try:
        saved = state.load(state.state_path(cfg.root, state_root))
    except state.StateError as exc:
        return f"cannot read state: {exc}"

    records = []
    for data in saved.agents:
        try:
            records.append(agent.AgentRecord.from_dict(data))
        except agent.AgentError:
            continue
    assigned = Counter(record.model for record in records)
    history = history_from_state(saved.samples)
    ready = order.runnable(
        plan.parse(text),
        {record.id for record in records},
        set(saved.blocked),
        lambda task: worktree.started_rank(cfg, task.id),
    )
    lines: list[str] = []
    count = len(records)
    for task in ready:
        name = None
        if max_agents is None or count < max_agents:
            name = choose(cfg.models, assigned, history, now, saved.last_extra)
        if name is None:
            lines.append(f"{task.id}\t-")
            continue
        assigned[name] += 1
        count += 1
        lines.append(f"{task.id}\t{name}")
    return "\n".join(lines) if lines else "no runnable tasks"
