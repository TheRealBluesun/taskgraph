"""Command-line interface for taskgraph.

Every subcommand from SPEC §10 parses its arguments here; each later task
replaces one "not implemented" stub with the real handler.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Sequence, TextIO

from . import lease
from .config import ConfigError, load

CONFIG_NAME = "taskgraph.toml"
TIMEOUT_EXIT = 124

# Subcommands and their one-line purpose, in SPEC §10 order (plus `models`, SPEC §4).
SUBCOMMANDS: dict[str, str] = {
    "run": "start/resume the scheduler (foreground)",
    "status": "show running agents, merge queue, blocked tasks and leases",
    "stats": "per-task wall-time split from the omp JSON traces",
    "models": "total tokens/s per model, bucketed by concurrency",
    "leases": "print current lease holders with ages",
    "stop": "stop the scheduler (--agents also stops agents)",
    "lease": "wait for a resource lease, then run a command",
    "retry": "unblock a blocked task (resumes its worktree)",
}


def _not_implemented(args: argparse.Namespace) -> int:
    """Stub handler: report the subcommand is not implemented yet."""
    print(f"taskgraph {args.command}: not implemented", file=sys.stderr)
    return 1


def find_config(start: Path | str | None = None) -> Path | None:
    """Return the nearest ``taskgraph.toml`` at or above ``start`` (default cwd)."""
    directory = Path(start if start is not None else Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        config = candidate / CONFIG_NAME
        if config.is_file():
            return config
    return None


def format_age(seconds: float) -> str:
    """Render a duration the way ``taskgraph leases`` does: ``59s``, ``2m05s``."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_holders(slots: Sequence[lease.Slot], now: float | None = None) -> str:
    """Render live holders as an aligned table for ``taskgraph leases``."""
    if not slots:
        return "no leases held"
    header = ("RESOURCE", "SLOT", "PID", "TASK", "AGE", "COMMAND")
    rows = [
        [
            slot.resource,
            slot.name,
            str(slot.pid),
            slot.task or "-",
            format_age(slot.age(now)),
            slot.cmd,
        ]
        for slot in slots
    ]
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row[:-1], widths[:-1])) + "  " + row[-1]
        for row in (header, *rows)
    ).rstrip()


@contextlib.contextmanager
def forward_signals(proc: subprocess.Popen) -> Iterator[None]:
    """Forward SIGTERM/SIGINT to ``proc``'s process group while waiting.

    The leased command is started in its own session, so signalling only its
    pid would leave ``sh -c 'build; test'``-style grandchildren running — and a
    killed build that keeps going keeps the resource busy.  The wrapper keeps
    waiting for the child instead of dying first, so the slot is only released
    after the command it guards has actually stopped (SPEC §5).  Silently does
    nothing off the main thread, where ``signal`` forbids handlers.
    """

    def handler(signum: int, _frame: object) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signum)
        except OSError:
            try:
                proc.send_signal(signum)
            except OSError:
                pass

    saved: dict[int, object] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            saved[sig] = signal.signal(sig, handler)
        except (ValueError, OSError, RuntimeError):
            pass
    try:
        yield
    finally:
        for sig, previous in saved.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, RuntimeError):
                pass


def run_lease(
    resource: str,
    command: Sequence[str],
    *,
    capacity: int,
    root: Path | str | None = None,
    task: str | None = None,
    poll: float = lease.POLL_SECS,
    message_secs: float | None = lease.MESSAGE_SECS,
    out: TextIO | None = None,
    timeout: float | None = None,
) -> int:
    """Wait for a lease, run ``command`` under it, and return its exit status.

    ``command`` sees ``TASKGRAPH_LEASE=<resource>`` in its environment, which
    the deny shims (SPEC §5) check.  Returns 124 when ``timeout`` expires while
    waiting and 127 when the command cannot be executed; a child killed by
    signal *n* reports the shell's ``128 + n``.
    """
    stream = sys.stderr if out is None else out
    base = Path(root) if root is not None else lease.state_root()
    try:
        slot = lease.wait_for_slot(
            resource,
            capacity,
            root=base,
            task=task,
            cmd=command,
            poll=poll,
            message_secs=message_secs,
            out=stream,
            timeout=timeout,
        )
    except KeyboardInterrupt:
        print(f"taskgraph lease: interrupted while waiting for {resource}", file=stream, flush=True)
        return 130
    if slot is None:
        print(f"taskgraph lease: timed out waiting for {resource}", file=stream, flush=True)
        return TIMEOUT_EXIT
    env = dict(os.environ)
    env[lease.LEASE_ENV] = resource
    try:
        try:
            # Own session: its pid is a process-group id, so a forwarded signal
            # reaches the whole leased command tree (see forward_signals).
            proc = subprocess.Popen(list(command), env=env, start_new_session=True)
        except OSError as exc:
            print(f"taskgraph lease: cannot run {command[0]!r}: {exc}", file=stream, flush=True)
            return 127
        with forward_signals(proc):
            code = proc.wait()
    finally:
        lease.release(slot, root=base)
    return code if code >= 0 else 128 - code


def _cmd_lease(args: argparse.Namespace) -> int:
    """``taskgraph lease``: wait for a resource slot, then run the command (SPEC §5)."""
    if not args.cmd:
        print("taskgraph lease: no command given", file=sys.stderr)
        return 2
    config_path = find_config()
    if config_path is None:
        print(
            f"taskgraph lease: no {CONFIG_NAME} found in this directory or its parents",
            file=sys.stderr,
        )
        return 2
    try:
        config = load(config_path)
    except ConfigError as exc:
        print(f"taskgraph lease: {exc}", file=sys.stderr)
        return 2
    capacity = config.resources.get(args.resource)
    if capacity is None:
        print(
            f"taskgraph lease: resource '{args.resource}' is not in [resources] of {config_path}",
            file=sys.stderr,
        )
        return 2
    return run_lease(args.resource, args.cmd, capacity=capacity, task=args.task)


def _cmd_leases(args: argparse.Namespace) -> int:
    """``taskgraph leases``: print current holders and their ages (SPEC §5)."""
    print(format_holders(lease.holders()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all taskgraph subcommands."""
    parser = argparse.ArgumentParser(
        prog="taskgraph",
        description="Run a project's task list with parallel coding agents in git worktrees.",
    )
    parser.add_argument("--version", action="version", version="taskgraph 0.1.0")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name in SUBCOMMANDS:
        sp = sub.add_parser(name, help=SUBCOMMANDS[name])
        sp.set_defaults(command=name, func=_not_implemented)
    sub.choices["lease"].set_defaults(func=_cmd_lease)
    sub.choices["leases"].set_defaults(func=_cmd_leases)

    run = sub.choices["run"]
    run.add_argument("--project", metavar="DIR", help="project root (default: cwd)")
    run.add_argument("--max-agents", type=int, metavar="N", help="cap on concurrent agents")
    run.add_argument("--dry-run", action="store_true", help="print the plan and exit")

    sub.choices["stats"].add_argument("--since", default="6h", metavar="WINDOW")

    sub.choices["stop"].add_argument("--agents", action="store_true", help="also stop agents")

    lease_parser = sub.choices["lease"]
    lease_parser.add_argument("resource", help="resource name, e.g. simulator")
    lease_parser.add_argument("--task", metavar="ID", help="task id recorded in the slot file")
    lease_parser.add_argument(
        "cmd", nargs="+", help="command to run, after -- (e.g. -- xcodebuild -scheme App)"
    )

    sub.choices["retry"].add_argument("id", metavar="ID", help="blocked task id")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the chosen subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
