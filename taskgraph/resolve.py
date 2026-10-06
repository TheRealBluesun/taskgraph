"""Resolve a rebase conflict with a short agent (T21, SPEC §8).

When ``git rebase <main>`` stops on a conflict the merge step normally aborts
it and blocks the task.  With ``[merge] resolver_prompt`` configured, taskgraph
instead keeps the rebase in progress and runs one short agent *in the worktree
that is mid-rebase*: the prompt names each conflicted hunk with both sides plus
the two commits' messages, and the agent may only stage the resolved files and
finish the rebase (``git add`` / ``git rebase --continue``).  The merge step
then builds and runs the gate as usual; a resolver that fails, or a gate that
fails, aborts the rebase and blocks the task exactly as before.

The module only writes the prompt file, launches the configured agent command
and inspects git state — it never picks a model itself (the scheduler passes a
:class:`Resolver` whose ``pick_model`` consults the shared pool) and never
touches the network.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import prompt as prompt_mod

#: Prompt file written into the worktree for the resolver agent (git-excluded).
PROMPT_NAME = ".taskgraph-resolve.md"

#: A "short" resolver: give up after this long and block the task.
RESOLVE_TIMEOUT = 900.0

#: How long a killed resolver's process group gets to die on SIGTERM.
SIGKILL_GRACE = 5.0

#: The one rule every resolution follows (T21).
RULE = (
    "Keep both intents: the newer task's behaviour, and main's API/structure "
    "changes (renames, new parameters, design tokens)."
)

#: What the resolver is allowed to run; anything else is out of bounds.
INSTRUCTIONS = """\
Resolve every conflict below, then finish the rebase yourself. Run only:
  git add -- <resolved files>      # stage the files you resolved
  git rebase --continue
Do not run `git rebase --abort` or `git rebase --skip`, do not create a commit
by hand, and do not edit anything unrelated to the conflicts."""

_CONFLICT_START = "<<<<<<<"
_CONFLICT_END = ">>>>>>>"


@dataclass(frozen=True)
class ResolveResult:
    """Outcome of one resolver run (T21).

    ``ok`` means the rebase finished and the task's work is applied on top of
    ``main``.  ``model`` is the model the resolver ran on; ``files`` are the
    conflicted paths it resolved; ``reason`` explains a failure.
    """

    ok: bool
    model: str = ""
    files: tuple[str, ...] = ()
    reason: str = ""


def conflict_hunks(text: str) -> list[str]:
    """Return every conflict region of ``text``, markers included (pure).

    A region starts at a ``<<<<<<<`` line, spans the ``=======`` divider and
    both sides, and ends at the matching ``>>>>>>>`` line.  An unterminated
    region is reported as-is (the agent still needs to see it).
    """
    hunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not current and line.startswith(_CONFLICT_START):
            current = [line]
        elif current:
            current.append(line)
            if line.startswith(_CONFLICT_END):
                hunks.append("\n".join(current))
                current = []
    if current:
        hunks.append("\n".join(current))
    return hunks


def dossier(
    main: str, worktree: Path | str, conflicts: list[str] | tuple[str, ...]
) -> str:
    """Describe every conflicted file, both sides and both commit messages.

    Reads the mid-rebase worktree only: each file's conflict regions verbatim,
    then the commit being rebased onto (``HEAD``, i.e. ``main``'s tip) and the
    task commit being applied (``REBASE_HEAD``).
    """
    worktree = Path(worktree)
    sections = ["## Conflicted files"]
    for index, path in enumerate(conflicts, start=1):
        sections.append(f"\n### {index}. {path}")
        hunks = _file_hunks(worktree / path)
        if hunks:
            sections.append("```\n" + "\n```\n```\n".join(hunks) + "\n```")
        else:
            sections.append(
                "(no conflict markers — one side added or deleted the file; "
                "decide with `git status`/`git show :2:`/`git show :3:`)"
            )
    if not conflicts:
        sections.append("\n(git reported no unmerged paths)")

    onto = _commit_message(worktree, "HEAD")
    applied = _commit_message(worktree, "REBASE_HEAD")
    sections.append("\n## Commits being combined")
    sections.append(f"\n- {main} (the commit you rebase onto, current HEAD):\n  {onto}")
    sections.append(f"- the task's commit being applied:\n  {applied}")
    return "\n".join(sections)


def prompt_text(tid: str, main: str, base: str, dossier_text: str) -> str:
    """Assemble the resolver prompt: instructions, project text, then dossier."""
    header = (
        f"You are resolving the conflicts from `git rebase {main}` for task {tid}.\n"
        f"{RULE}\n\n{INSTRUCTIONS}"
    )
    parts = [header]
    if base.strip():
        parts.append(base.strip())
    parts.append(dossier_text)
    return "\n\n".join(parts) + "\n"


@dataclass
class Resolver:
    """Run the conflict-resolving agent for the merge step (T21).

    ``pick_model`` returns the model to run on (the scheduler wires it to the
    shared pool); ``None`` means no model is free, and the merge blocks.  The
    resolver is synchronous — the merge worker has nothing else to do — and
    kills its process group if the agent overruns :data:`RESOLVE_TIMEOUT`.
    """

    pick_model: Callable[[], str | None]
    timeout: float = RESOLVE_TIMEOUT

    def resolve(
        self, cfg: object, tid: str, worktree: Path | str, conflicts: list[str]
    ) -> ResolveResult:
        """Resolve ``tid``'s conflict in ``worktree``; never raises for a failure."""
        worktree = Path(worktree)
        if not cfg.merge.resolver_prompt:
            return ResolveResult(False, reason="no [merge] resolver_prompt is configured")
        try:
            base = _read_prompt(cfg)
        except OSError as exc:
            return ResolveResult(False, reason=f"resolver prompt: {exc.strerror or exc}")

        model = self.pick_model()
        if not model:
            return ResolveResult(False, reason="no model is free for the resolver")

        text = prompt_text(tid, cfg.main, base, dossier(cfg.main, worktree, conflicts))
        try:
            prompt_file = _write_prompt(worktree, text)
        except OSError as exc:
            return ResolveResult(False, model=model, reason=f"cannot write prompt: {exc}")

        overlay = prompt_mod.overlay_path(cfg.agent.overlay)
        if "{overlay}" in cfg.agent.command and not overlay.is_file():
            return ResolveResult(
                False, model=model, reason=f"agent overlay {overlay} does not exist"
            )
        try:
            argv = prompt_mod.build_command(
                cfg.agent.command, model=model, overlay=overlay, prompt_file=prompt_file
            )
        except prompt_mod.PromptError as exc:
            return ResolveResult(False, model=model, reason=str(exc))

        rc, timed_out, error = self._run(worktree, tid, argv)
        if error:
            return ResolveResult(False, model=model, reason=error)
        ok, reason = finished(cfg, worktree)
        if not ok:
            detail = f" (timed out after {self.timeout:g}s)" if timed_out else f" (exit {rc})"
            return ResolveResult(False, model=model, reason=reason + detail)
        return ResolveResult(True, model=model, files=tuple(conflicts))

    def _run(self, worktree: Path, tid: str, argv: list[str]) -> tuple[int, bool, str]:
        """Run the agent in ``worktree``; return ``(exit code, timed out, error)``.

        Output goes to ``logs/resolve-<id>.log``.  ``GIT_EDITOR`` is forced
        non-interactive so ``git rebase --continue`` cannot wait on an editor.
        """
        log = worktree / "logs" / f"resolve-{tid}.log"
        env = dict(os.environ)
        env.setdefault("GIT_EDITOR", "true")
        env.setdefault("GIT_SEQUENCE_EDITOR", "true")
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
        except OSError as exc:
            return 127, False, f"cannot open {log}: {exc.strerror or exc}"
        try:
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=os.fspath(worktree),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                return 127, False, f"cannot run resolver: {exc}"
            try:
                return proc.wait(timeout=self.timeout), False, ""
            except subprocess.TimeoutExpired:
                _kill_group(proc.pid)
                proc.wait()
                return -1, True, ""
        finally:
            handle.close()


def finished(cfg: object, worktree: Path | str) -> tuple[bool, str]:
    """Whether the resolver left the worktree rebased onto ``main`` (pure git).

    All three must hold: no rebase still in progress, no unmerged paths, and
    ``main`` an ancestor of ``HEAD`` — the last catches a resolver that
    aborted or skipped the rebase, which would otherwise silently drop the
    task's work.
    """
    worktree = Path(worktree)
    stopped = _rebase_in_progress(worktree)
    if stopped:
        return False, f"rebase is still in progress ({stopped})"
    unmerged = _unmerged(worktree)
    if unmerged:
        return False, "unresolved conflicts: " + ", ".join(unmerged)
    if _run_git(worktree, "merge-base", "--is-ancestor", cfg.main, "HEAD").returncode != 0:
        return False, f"the rebase was not applied on top of {cfg.main}"
    return True, ""


# --------------------------------------------------------------------------- git


def _read_prompt(cfg: object) -> str:
    path = Path(cfg.root) / cfg.merge.resolver_prompt
    return path.read_text(encoding="utf-8")


def _write_prompt(worktree: Path, text: str) -> Path:
    path = worktree / PROMPT_NAME
    path.write_text(text, encoding="utf-8")
    prompt_mod.exclude(PROMPT_NAME, worktree)
    return path


def _file_hunks(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return conflict_hunks(text)


def _commit_message(worktree: Path, rev: str) -> str:
    result = _run_git(worktree, "show", "-s", "--format=%B", rev)
    if result.returncode != 0:
        return "(unknown)"
    return " ".join(result.stdout.split()) or "(empty message)"


def _rebase_in_progress(worktree: Path) -> str:
    """Return the name of the rebase state directory git kept, or ``""``.

    ``REBASE_HEAD`` outlives a finished rebase, so it cannot tell a stopped one
    apart; git's own ``rebase-merge``/``rebase-apply`` directory can.
    """
    for name in ("rebase-merge", "rebase-apply"):
        path = _git_path(worktree, name)
        if path is not None and path.exists():
            return name
    return ""


def _git_path(worktree: Path, name: str) -> Path | None:
    result = _run_git(worktree, "rev-parse", "--git-path", name)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else worktree / path


def _unmerged(worktree: Path) -> list[str]:
    result = _run_git(worktree, "diff", "--name-only", "--diff-filter=U")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _run_git(worktree: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=os.fspath(worktree),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return subprocess.CompletedProcess(["git", *args], 127, "", "git: executable not found")


def _kill_group(pgid: int) -> None:
    """SIGTERM a resolver's process group, SIGKILL it after :data:`SIGKILL_GRACE`."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + SIGKILL_GRACE
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
