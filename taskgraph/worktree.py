"""Git worktrees, one per task (SPEC §1, §5, §8).

Each task runs in its own ``git worktree`` on a ``task/<id>`` branch cut from
the integration branch (``main``), so parallel agents never see each other's
checkout.  The configured ``links`` (secrets such as ``.env``) are *symlinked*
into every new worktree — never copied — so a rotated secret is picked up at
once and no secret value ever reaches a commit.

The worktree base lives outside the project (SPEC §1), which keeps it out of
the repository.  This module is the single owner of the layout
``<root>/<worktrees>/<id>`` and of the ``task/<id>`` branch, so the merge queue
(SPEC §8) can remove exactly what the scheduler created.  Only ``git`` is
invoked here: no omp, no network.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import Config

#: Branch prefix for a task, e.g. ``task/T01`` (SPEC §8).
BRANCH_PREFIX = "task/"

#: Directory inside a worktree where an agent writes its notes (SPEC §3, §6).
PROGRESS_DIRNAME = "progress"


class WorktreeError(Exception):
    """A worktree cannot be created or removed (git failed, path clashes)."""


def branch(tid: str) -> str:
    """Return the branch name for task ``tid`` (``task/<id>``, SPEC §8)."""
    return f"{BRANCH_PREFIX}{tid}"


def base_dir(cfg: Config) -> Path:
    """Return the resolved directory that holds every task worktree."""
    return Path(Path(cfg.root) / cfg.worktrees).resolve()


def path(cfg: Config, tid: str) -> Path:
    """Return the worktree directory of task ``tid`` (it need not exist)."""
    return base_dir(cfg) / tid


def exists(cfg: Config, tid: str) -> bool:
    """Return whether task ``tid``'s worktree directory exists."""
    return path(cfg, tid).is_dir()


def is_resume(cfg: Config, tid: str) -> bool:
    """Return whether an agent for ``tid`` would resume partial work (SPEC §6).

    A worktree that already exists is the only source of partial work: it was
    left by an interrupted agent, so the next agent gets the resume note.
    """
    return exists(cfg, tid)


def notes_path(cfg: Config, tid: str) -> Path:
    """Return the path of ``tid``'s notes file (``progress/<id>.md``, SPEC §6)."""
    return path(cfg, tid) / PROGRESS_DIRNAME / f"{tid}.md"


def started_rank(cfg: Config, tid: str) -> int:
    """Rank how far along ``tid`` already is, for SPEC §3's "started first".

    ``0`` = worktree with notes, ``1`` = worktree without notes, ``2`` = no
    worktree.  Lower sorts first, so a nearly finished task keeps its slot
    across a scheduler restart.
    """
    if not exists(cfg, tid):
        return 2
    return 0 if notes_path(cfg, tid).is_file() else 1


def create(cfg: Config, tid: str) -> Path:
    """Create ``tid``'s worktree on ``task/<id>`` from ``cfg.main`` (SPEC §1).

    The branch starts at the integration branch; ``cfg.links`` are symlinked in
    afterwards.  Raises :class:`WorktreeError` if the worktree (or branch)
    already exists, if ``cfg.main`` is unknown, or if ``git`` fails.  On a
    linking failure the half-made worktree is removed again, so a caller either
    gets a complete worktree or nothing.
    """
    worktree = path(cfg, tid)
    if worktree.exists():
        raise WorktreeError(f"{worktree}: worktree for {tid} already exists")
    try:
        worktree.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorktreeError(
            f"{worktree.parent}: cannot create worktree base: {exc.strerror or exc}"
        ) from exc
    _git(cfg, "worktree", "add", "-b", branch(tid), os.fspath(worktree), cfg.main)
    try:
        link_files(cfg, worktree)
    except WorktreeError:
        try:
            remove(cfg, tid)
        except WorktreeError:
            pass
        raise
    return worktree


def link_files(cfg: Config, worktree: Path | str) -> list[Path]:
    """Symlink each ``cfg.links`` entry into ``worktree`` (SPEC §5).

    Returns the links created.  A source missing from the project is skipped —
    an optional secret must not block a worktree.  A path already present in the
    fresh checkout is left alone: a tracked file must never be replaced by a
    symlink to the project's working copy.
    """
    created: list[Path] = []
    for rel in cfg.links:
        source = Path(cfg.root) / rel
        target = Path(worktree) / rel
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.abspath(source), target)
        except OSError as exc:
            raise WorktreeError(f"{target}: cannot link {rel}: {exc.strerror or exc}") from exc
        exclude(cfg, worktree, rel)
        created.append(target)
    return created


def exclude(cfg: Config, worktree: Path | str, pattern: str) -> bool:
    """Add ``pattern`` to the clone's local exclude file; True if newly added.

    The merge step stages everything with ``git add -A`` (SPEC §8), so a
    symlinked secret that the project did not gitignore would land in the merge
    commit (and a later merge into the project root could refuse to overwrite
    the real file).  ``.git/info/exclude`` is local to the clone, never
    committed, and is the same place SPEC §6 keeps the prompt file.  A worktree
    without git metadata, or an unwritable file, is not an error here.
    """
    result = _run(cfg, "rev-parse", "--git-path", "info/exclude", cwd=worktree)
    if result.returncode != 0:
        return False
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = Path(worktree) / path
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if pattern in existing.splitlines():
            return False
        text = existing if not existing or existing.endswith("\n") else existing + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + pattern + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def remove(cfg: Config, tid: str) -> None:
    """Remove ``tid``'s worktree and ``task/<id>`` branch (SPEC §8), if present.

    Idempotent: a worktree that was never created, or whose directory a user
    already deleted, is not an error — merge cleanup can always run.  The
    directory is only removed through ``git worktree remove`` and never by
    deleting an arbitrary path, so a wrong ``worktrees`` setting cannot destroy
    unrelated data.
    """
    worktree = path(cfg, tid)
    if _registered(cfg, worktree):
        _git(cfg, "worktree", "remove", "--force", os.fspath(worktree))
    if _branch_exists(cfg, tid):
        _git(cfg, "branch", "-D", branch(tid))


def _registered(cfg: Config, worktree: Path) -> bool:
    """Return whether ``worktree`` is a worktree git has registered."""
    listing = _git(cfg, "worktree", "list", "--porcelain").stdout
    wanted = os.path.realpath(worktree)
    return any(
        os.path.realpath(line[len("worktree ") :].strip()) == wanted
        for line in listing.splitlines()
        if line.startswith("worktree ")
    )


def _branch_exists(cfg: Config, tid: str) -> bool:
    """Return whether the ``task/<id>`` branch exists."""
    result = _run(cfg, "show-ref", "--verify", "--quiet", f"refs/heads/{branch(tid)}")
    return result.returncode == 0


def _run(cfg: Config, *args: str, cwd: Path | str | None = None) -> subprocess.CompletedProcess:
    """Run ``git args`` in the project root and return the completed process."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=os.fspath(cwd or cfg.root),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise WorktreeError("git: executable not found") from exc


def _git(cfg: Config, *args: str, cwd: Path | str | None = None) -> subprocess.CompletedProcess:
    """Run ``git args``, raising :class:`WorktreeError` with git's message on failure."""
    result = _run(cfg, *args, cwd=cwd)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit {result.returncode}"
        raise WorktreeError(f"git {' '.join(args)}: {reason}")
    return result
