"""Thin wrappers around the ``git`` command line (SPEC §0, §8).

The merge queue shells out to ``git`` for every step; this module keeps the
subprocess plumbing in one place: run, capture text output, never raise for a
non-zero exit unless the caller asked for a command that had to succeed, and
summarize the tail of an error for a blocked reason.  It knows nothing about
tasks, branches or merge policy.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GitError(Exception):
    """``git`` is missing, or a command that had to succeed did not."""


def run(cwd: Path | str, *args: str) -> subprocess.CompletedProcess:
    """Run ``git args`` in ``cwd``, capturing text output.

    A non-zero exit is returned, not raised; a missing ``git`` raises
    :class:`GitError`.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=os.fspath(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git: executable not found") from exc


def git(cwd: Path | str, *args: str) -> subprocess.CompletedProcess:
    """Run ``git args``, raising :class:`GitError` with git's message on failure."""
    result = run(cwd, *args)
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {tail(result.stderr or result.stdout, 1)}")
    return result


def staged(cwd: Path | str) -> bool:
    """Return whether the index has any change to commit."""
    return run(cwd, "diff", "--cached", "--quiet").returncode != 0


def conflicts(cwd: Path | str) -> list[str]:
    """Return the unmerged paths left by a failed rebase."""
    result = run(cwd, "diff", "--name-only", "--diff-filter=U")
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed(base: str, cwd: Path | str) -> tuple[list[str], dict[str, int]]:
    """Return the added paths and present-path byte sizes of ``cwd`` over ``base``.

    The work commit was just made from this worktree, so on-disk sizes match
    it; a path that vanished (deleted, broken link) is skipped, never guessed.
    """
    result = run(cwd, "diff", "--name-status", "--no-renames", f"{base}...HEAD")
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
            sizes[path] = os.lstat(os.path.join(cwd, path)).st_size
        except OSError:
            continue
    return added, sizes


def tail(text: str, lines: int) -> str:
    """Return the last ``lines`` non-empty lines of ``text`` joined by newlines."""
    kept = [line for line in (text or "").splitlines() if line.strip()]
    return "\n".join(kept[-lines:])
