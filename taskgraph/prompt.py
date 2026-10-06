"""Build each agent's prompt file and command line (SPEC §6).

Every agent gets ``<worktree>/.taskgraph-prompt.md``: a fixed header naming the
task — plus a resume warning when the worktree already holds partial work — and
the project's own prompt file.  The file is git-excluded through
``.git/info/exclude``, because the merge step commits everything else the agent
leaves behind.

Only text manipulation and small file writes live here; launching and watching
the agent process is :mod:`taskgraph.agent`.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from . import worktree as worktree_mod

PROMPT_NAME = ".taskgraph-prompt.md"
SHARE_DIRNAME = "share"

# Prepended to the prompt when an earlier agent for this task was interrupted.
RESUME_NOTE = (
    "NOTE: this worktree already contains PARTIAL work on this task from an "
    "earlier, interrupted agent (see `git status` / `git diff`). Review it and "
    "continue from it; do not start over."
)

# The task header, verbatim from SPEC §6; `{plan}` is the project's plan file.
_HEADER = """\
YOUR TASK IS **{id}** (do only this one): {id} {text}
You are one of several agents working in parallel, each in its own git worktree. Therefore:
- Do NOT edit {plan} and do not run git commands.
- Write your notes (2–5 bullets) to progress/{id}.md.
- When — and only when — the task is complete and verified, create the empty file progress/{id}.done.
- Run long commands in the foreground and wait for them; never end your turn while a background job is running.
- Never modify or work around the shared-resource tooling (leases, shims); if a command waits for a lease, wait."""


class PromptError(Exception):
    """The agent command or one of the prompt files cannot be used as written."""


def share_dir() -> Path:
    """Return taskgraph's shipped-files directory (holds the default overlay)."""
    return Path(__file__).resolve().parent.parent / SHARE_DIRNAME


def overlay_path(overlay: str, share: Path | str | None = None) -> Path:
    """Resolve an ``[agent] overlay``: absolute as-is, else under the share dir."""
    path = Path(overlay)
    if path.is_absolute():
        return path
    return (Path(share) if share is not None else share_dir()) / path


def build_command(
    command: str, *, model: str, overlay: Path | str, prompt_file: Path | str
) -> list[str]:
    """Expand the ``agent.command`` template and split it into argv.

    The three placeholders SPEC §1 shows are ``{model}``, ``{overlay}`` and
    ``{prompt_file}``; the command is split with :mod:`shlex` (no shell), so a
    path containing spaces survives.
    """
    try:
        expanded = command.format(
            model=model, overlay=os.fspath(overlay), prompt_file=os.fspath(prompt_file)
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise PromptError(f"agent.command: cannot expand {exc}") from exc
    argv = shlex.split(expanded)
    if not argv:
        raise PromptError("agent.command is empty")
    return argv


def prompt_text(task: Any, plan: str, project_prompt: str, *, resume: bool = False) -> str:
    """Return the full text written for ``task`` (SPEC §6).

    ``plan`` is the plan file name to tell the agent not to edit; ``task`` only
    needs ``.id`` (``str``) and ``.text``.
    """
    header = _HEADER.format(id=task.id, text=task.text, plan=plan)
    parts = [header, project_prompt.strip()] if project_prompt.strip() else [header]
    text = "\n\n".join(parts) + "\n"
    if resume:
        return RESUME_NOTE + "\n\n" + text
    return text


def project_prompt(cfg: Any) -> str:
    """Read the project's prompt file (``cfg.prompt`` relative to ``cfg.root``)."""
    path = Path(cfg.root) / cfg.prompt
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"{path}: cannot read project prompt: {exc.strerror or exc}") from exc


def write_prompt(
    worktree: Path | str,
    task: Any,
    cfg: Any,
    *,
    resume: bool = False,
    project: str | None = None,
) -> Path:
    """Write ``<worktree>/.taskgraph-prompt.md`` and git-exclude it.

    ``project`` overrides reading ``cfg.prompt`` (tests, or a caller that already
    has the text); returns the prompt file path.
    """
    worktree = Path(worktree)
    if project is None:
        project = project_prompt(cfg)
    path = worktree / PROMPT_NAME
    try:
        path.write_text(prompt_text(task, cfg.plan, project, resume=resume), encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"{path}: cannot write prompt file: {exc.strerror or exc}") from exc
    exclude(path.name, worktree)
    return path


def exclude(pattern: str, worktree: Path | str) -> bool:
    """Add ``pattern`` to the exclude file git reads for ``worktree`` (SPEC §6).

    ``git rev-parse --git-path info/exclude`` names the file git actually reads
    (for a linked worktree: the clone's shared ``.git/info/exclude``); writing a
    per-worktree ``info/exclude`` would be ignored, so the prompt file would be
    staged by the merge step's ``git add -A``.  A worktree without git metadata
    (or an unwritable file) is not an error here.
    """
    path = worktree_mod.exclude_file(worktree)
    return False if path is None else worktree_mod.add_exclude(path, pattern)
