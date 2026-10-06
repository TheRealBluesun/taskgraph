"""Serialized merge queue for finished tasks (SPEC §8).

A finished agent leaves a ``progress/<id>.done`` file in its worktree.  The
scheduler hands that task to a :class:`MergeQueue`, whose single worker thread
runs the five steps of SPEC §8 in order:

1. stage and commit everything the agent left in the worktree;
2. rebase the ``task/<id>`` branch onto the integration branch (a conflict
   blocks the task with the conflicting paths);
3. run the configured ``gate`` in the worktree (a failure blocks with the tail
   of its output);
4. fast-forward the integration branch to ``task/<id>`` — if somebody else
   merged first, go back to step 2, at most three times;
5. tick the plan, append the notes to ``PROGRESS.md``, commit both, and remove
   the worktree and branch.

Merging is serialized *and* off the scheduling loop: a several-minute gate must
not freeze scheduling (SPEC §8).  The module only invokes ``git`` and the gate
shell command — no omp, no network.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import guard
from . import plan as plan_mod
from . import worktree
from .config import Config

#: Commit message of the agent's work commit (SPEC §8 step 1).
WORK_COMMIT = "{id}: work (taskgraph)"

#: Commit message of the plan/notes commit (SPEC §8 step 5).
META_COMMIT = "{id}: merge (taskgraph)"

#: SPEC §8 step 4: how many times "main moved" may send us back to the rebase.
MAX_ATTEMPTS = 3

#: Gate output goes to ``logs/gate.log`` in the worktree (SPEC §8 step 3).
GATE_LOG = "logs/gate.log"

#: How many trailing gate-log lines a blocked reason summarizes (SPEC §8).
GATE_TAIL_LINES = 20

#: Name of the merge section appended to the project's progress file.
PROGRESS_FILE = "PROGRESS.md"


class MergeError(Exception):
    """A merge step could not run at all (missing worktree, git missing…)."""


@dataclass(frozen=True)
class MergeResult:
    """Outcome of one task's merge attempt (SPEC §8).

    ``ok`` is True once the work is on the integration branch and the plan is
    ticked.  ``reason`` explains a blocked task (conflicting paths, gate output,
    repeated "main moved"); ``attempts`` counts rebase/gate/merge cycles.
    """

    id: str
    ok: bool
    reason: str = ""
    attempts: int = 1


def merge(cfg: Config, tid: str) -> MergeResult:
    """Merge task ``tid``'s worktree into the integration branch (SPEC §8).

    Runs in the calling thread; :class:`MergeQueue` supplies the single worker.
    Returns a blocked :class:`MergeResult` for the expected failures (a failed
    work commit, the commit guard, conflict, gate failure, main moving
    repeatedly) and keeps the worktree so ``taskgraph retry`` can resume it.
    Raises :class:`MergeError` only when a step cannot run at all.
    """
    worktree_path = worktree.path(cfg, tid)
    if not worktree_path.is_dir():
        raise MergeError(f"{worktree_path}: no worktree for {tid} to merge")

    blocked = _commit_work(cfg, tid, worktree_path)
    if blocked is not None:
        return blocked
    blocked = _guard(cfg, tid, worktree_path)
    if blocked is not None:
        return blocked

    attempts = 0
    while True:
        attempts += 1
        result = _rebase(cfg, tid, worktree_path, attempts)
        if result is not None:
            return result
        result = _gate(cfg, tid, worktree_path, attempts)
        if result is not None:
            return result
        ff = _run(cfg, "merge", "--ff-only", worktree.branch(tid))
        if ff.returncode == 0:
            break
        if attempts >= MAX_ATTEMPTS:
            detail = _tail(ff.stderr or ff.stdout, 1) or f"exit {ff.returncode}"
            return MergeResult(
                tid, False, f"main moved {attempts} times; {detail}", attempts
            )
        # Somebody merged in the window between our rebase and this fast-forward:
        # run the rebase/gate cycle again against the new integration branch.

    _finish(cfg, tid, worktree_path)
    worktree.remove(cfg, tid)
    return MergeResult(tid, True, attempts=attempts)


# --------------------------------------------------------------------------- steps


def _commit_work(cfg: Config, tid: str, worktree_path: Path) -> MergeResult | None:
    """Step 1: stage everything and commit it; a blocked result on git failure.

    ``logs`` is kept out via the clone's local exclude file, not a ``:!logs``
    pathspec: naming a path git ignores makes ``git add`` exit non-zero ("The
    following paths are ignored…"), which silently skipped every commit in the
    predecessor tool.  A git failure is reported, not raised, so the worktree
    survives for ``taskgraph retry``.
    """
    worktree.exclude(cfg, worktree_path, "logs/")
    added = _run(cfg, "add", "-A", cwd=worktree_path)
    if added.returncode != 0:
        detail = _tail(added.stderr or added.stdout, 1)
        return MergeResult(tid, False, f"git add -A failed: {detail}")
    if not _staged(cfg, worktree_path):
        return None
    commit = _run(cfg, "commit", "-q", "-m", WORK_COMMIT.format(id=tid), cwd=worktree_path)
    if commit.returncode != 0:
        detail = _tail(commit.stderr or commit.stdout, 1)
        return MergeResult(tid, False, f"git commit failed: {detail}")
    return None


def _guard(cfg: Config, tid: str, worktree_path: Path) -> MergeResult | None:
    """Block a task branch that is a build dump (SPEC §8 commit guard).

    Inspects everything the branch adds over ``cfg.main``, not just the newest
    commit: a blocked task keeps its earlier commit, so a retry that adds
    nothing must still be checked against the whole branch.
    """
    added, sizes = _changed(cfg, worktree_path)
    issues = guard.problems(
        added,
        sizes,
        max_files=cfg.merge.max_files,
        max_file_mb=cfg.merge.max_file_mb,
        deny_dirs=cfg.merge.deny_dirs,
    )
    return None if not issues else MergeResult(
        tid, False, "commit guard: " + "; ".join(issues)
    )


def _changed(cfg: Config, worktree_path: Path) -> tuple[list[str], dict[str, int]]:
    """Return the added paths and present-path byte sizes over ``cfg.main``.

    The work commit was just made from this worktree, so on-disk sizes match
    it; a path that vanished (deleted, broken link) is skipped, never guessed.
    """
    result = _run(
        cfg, "diff", "--name-status", "--no-renames", f"{cfg.main}...HEAD", cwd=worktree_path
    )
    if result.returncode != 0:
        return [], {}
    added: list[str] = []
    sizes: dict[str, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0].strip():
            continue
        status, path = fields[0][:1], fields[-1]
        if status not in ("A", "M", "R", "C", "T"):
            continue  # a deletion is not a size or path offence
        if status == "A":
            added.append(path)
        try:
            sizes[path] = os.lstat(worktree_path / path).st_size
        except OSError:
            continue
    return added, sizes


def _rebase(cfg: Config, tid: str, worktree_path: Path, attempts: int) -> MergeResult | None:
    """Step 2: rebase onto ``cfg.main``; return a blocked result on conflict."""
    result = _run(cfg, "rebase", cfg.main, cwd=worktree_path)
    if result.returncode == 0:
        return None
    conflicts = _conflicts(cfg, worktree_path)
    _run(cfg, "rebase", "--abort", cwd=worktree_path)
    where = ", ".join(conflicts) if conflicts else "unknown files"
    return MergeResult(tid, False, f"conflict in {where}", attempts)


def _gate(cfg: Config, tid: str, worktree_path: Path, attempts: int) -> MergeResult | None:
    """Step 3: run the gate in the worktree; return a blocked result on failure."""
    log = worktree_path / GATE_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cfg.gate,
                shell=True,
                cwd=worktree_path,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except OSError as exc:
        raise MergeError(f"gate failed to run: {exc}") from exc
    if result.returncode == 0:
        return None
    try:
        output = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
    summary = "; ".join(_tail(output, GATE_TAIL_LINES).splitlines())
    return MergeResult(tid, False, f"gate failed ({result.returncode}): {summary}", attempts)


def _finish(cfg: Config, tid: str, worktree_path: Path) -> None:
    """Step 5: tick the plan, append the notes, commit both (SPEC §8)."""
    changed = _tick_plan(cfg, tid)
    changed += _append_notes(cfg, tid)
    if not changed:
        return
    _git(cfg, "add", "--", *changed, cwd=cfg.root)
    message = META_COMMIT.format(id=tid)
    if cfg.trailer:
        message += "\n\n" + "\n".join(cfg.trailer)
    _git(cfg, "commit", "-q", "-m", message, cwd=cfg.root)


def _tick_plan(cfg: Config, tid: str) -> list[str]:
    """Tick ``tid``'s plan line; return the plan path if it changed."""
    path = Path(cfg.root) / cfg.plan
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MergeError(f"{path}: cannot read plan: {exc.strerror or exc}") from exc
    try:
        updated = plan_mod.mark_done(text, tid)
    except plan_mod.PlanError as exc:
        raise MergeError(str(exc)) from exc
    if updated == text:
        return []
    path.write_text(updated, encoding="utf-8")
    return [cfg.plan]


def _append_notes(cfg: Config, tid: str) -> list[str]:
    """Append ``## <id>`` + the notes file to PROGRESS.md; return its path.

    No notes file means no progress section — a task may legitimately finish
    without writing any (SPEC §6 asks for 2–5 bullets, it does not require it).
    """
    notes = worktree.notes_path(cfg, tid)
    if not notes.is_file():
        return []
    try:
        body = notes.read_text(encoding="utf-8")
    except OSError as exc:
        raise MergeError(f"{notes}: cannot read notes: {exc.strerror or exc}") from exc
    path = Path(cfg.root) / PROGRESS_FILE
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise MergeError(f"{path}: cannot read progress file: {exc.strerror or exc}") from exc
    if existing and not existing.endswith("\n"):
        existing += "\n"
    if existing and not existing.endswith("\n\n"):
        existing += "\n"
    section = f"## {tid}\n\n{body}"
    if not section.endswith("\n"):
        section += "\n"
    path.write_text(existing + section, encoding="utf-8")
    return [PROGRESS_FILE]


# --------------------------------------------------------------------------- queue


class MergeQueue:
    """FIFO queue of finished tasks, drained by one worker thread (SPEC §8).

    The scheduler keeps running while a gate executes; it hands a task id to
    :meth:`enqueue` and learns the outcome from ``on_result`` (called in the
    worker thread) or :meth:`results`.  A merge that raises is reported as a
    blocked :class:`MergeResult`, so one bad task can never kill the worker.
    """

    def __init__(self, cfg: Config, on_result: Callable[[MergeResult], None] | None = None):
        self._cfg = cfg
        self._on_result = on_result
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._results: list[MergeResult] = []
        self._stopped = False
        self._worker = threading.Thread(target=self._run, name="taskgraph-merge", daemon=True)
        self._worker.start()

    def enqueue(self, tid: str) -> None:
        """Add ``tid`` to the back of the queue (FIFO order is the merge order)."""
        with self._lock:
            self._pending.append(tid)
        self._queue.put(tid)

    def pending(self) -> tuple[str, ...]:
        """Return the ids queued but not yet finished, in arrival order."""
        with self._lock:
            return tuple(self._pending)

    def results(self) -> tuple[MergeResult, ...]:
        """Return the results of every finished task, in merge order."""
        with self._lock:
            return tuple(self._results)

    def join(self) -> None:
        """Block until every enqueued task has been processed."""
        self._queue.join()

    def stop(self) -> None:
        """Drain nothing further and wait for the worker to exit."""
        if self._stopped:
            return
        self._stopped = True
        self._queue.put(None)
        self._worker.join()

    def _run(self) -> None:
        while True:
            tid = self._queue.get()
            try:
                if tid is None:
                    return
                with self._lock:
                    self._pending.remove(tid)
                try:
                    result = merge(self._cfg, tid)
                except Exception as exc:  # never let the worker die on one task
                    result = MergeResult(tid, False, f"merge error: {exc}")
                with self._lock:
                    self._results.append(result)
                if self._on_result is not None:
                    try:
                        self._on_result(result)
                    except Exception:
                        pass
            finally:
                self._queue.task_done()


# --------------------------------------------------------------------------- git


def _run(cfg: Config, *args: str, cwd: Path | str | None = None) -> subprocess.CompletedProcess:
    """Run ``git args`` in ``cwd`` (default: project root) without raising."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=os.fspath(cwd or cfg.root),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MergeError("git: executable not found") from exc


def _git(cfg: Config, *args: str, cwd: Path | str | None = None) -> subprocess.CompletedProcess:
    """Run ``git args``, raising :class:`MergeError` with git's message on failure."""
    result = _run(cfg, *args, cwd=cwd)
    if result.returncode != 0:
        raise MergeError(f"git {' '.join(args)}: {_tail(result.stderr or result.stdout, 1)}")
    return result


def _staged(cfg: Config, cwd: Path) -> bool:
    """Return whether the index has any change to commit."""
    return _run(cfg, "diff", "--cached", "--quiet", cwd=cwd).returncode != 0


def _conflicts(cfg: Config, cwd: Path) -> list[str]:
    """Return the unmerged paths left by a failed rebase."""
    result = _run(cfg, "diff", "--name-only", "--diff-filter=U", cwd=cwd)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _tail(text: str, lines: int) -> str:
    """Return the last ``lines`` non-empty lines of ``text`` joined by newlines."""
    kept = [line for line in (text or "").splitlines() if line.strip()]
    return "\n".join(kept[-lines:])
