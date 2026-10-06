"""Command-line interface for taskgraph.

Every subcommand from SPEC §10 parses its arguments here; each later task
replaces one "not implemented" stub with the real handler.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, TextIO

from . import (
    assign,
    control,
    lease,
    leaseprocs,
    models,
    proxyserver,
    router,
    scheduler,
    side,
    state,
    stats,
    status,
    worktree,
)
from .config import Config, ConfigError, load
from .procsig import (
    GROUP_GRACE,
    SignalForwarder,
    block_terminating_signals,
    ignore_sigpipe,
    pending_terminating_signal,
    restore_mask,
    swallow_signal,
    terminate_group,
)

CONFIG_NAME = "taskgraph.toml"
TIMEOUT_EXIT = lease.TIMEOUT_EXIT

# Subcommands and their one-line purpose, in SPEC §10 order (plus `models`, SPEC §4).
SUBCOMMANDS: dict[str, str] = {
    "run": "start/resume the scheduler (foreground)",
    "status": "show running agents, merge queue, blocked tasks and leases",
    "stats": "per-task wall-time split from the omp JSON traces",
    "models": "total tokens/s per model, bucketed by concurrency",
    "leases": "print current lease holders with ages",
    "stop": "stop the scheduler (--agents also stops agents)",
    "lease": "wait for a resource lease, then run a command",
    "side": "lease a model's job-class capacity, then run a command",
    "router": "serve the OpenAI-compatible request router (SPEC §11)",
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
    group_grace: float = GROUP_GRACE,
) -> int:
    """Wait for a lease, run ``command`` under it, and return its exit status.

    ``command`` sees ``TASKGRAPH_LEASE=<resource>`` in its environment, which
    the deny shims (SPEC §5) check.  Returns 124 when ``timeout`` expires while
    waiting and 127 when the command cannot be executed; a child killed by
    signal *n* reports the shell's ``128 + n``.

    Forwarding handlers are installed before the slot is acquired and the
    terminating signals stay blocked while waiting, so the wrapper can never die
    between acquiring the slot and attaching the child.  The mask is restored
    before the child is spawned (a fork would inherit it).  After the direct
    child exits its group is emptied (SIGTERM, then SIGKILL after
    ``group_grace``) before the slot is released.
    """
    stream = sys.stderr if out is None else out
    base = Path(root) if root is not None else lease.state_root()
    ignore_sigpipe()
    forwarder = SignalForwarder()
    forwarder.install()
    previous = block_terminating_signals()
    blocked = previous is not None
    announcement: Path | None = None
    try:
        # Announce this wrapper before it may start waiting: while an agent's
        # leased command runs, omp streams nothing into its trace, so the
        # scheduler's stall watchdog must be able to see the wrapper (SPEC §6).
        announcement = leaseprocs.register_process(resource, root=base, cmd=command)
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
                abort=pending_terminating_signal if blocked else None,
            )
        except lease.WaitInterrupted as exc:
            swallow_signal(exc.signum, previous)
            print(
                f"taskgraph lease: interrupted while waiting for {resource}",
                file=stream,
                flush=True,
            )
            return 128 + exc.signum
        except KeyboardInterrupt:
            print(
                f"taskgraph lease: interrupted while waiting for {resource}",
                file=stream,
                flush=True,
            )
            return 130
        if slot is None:
            print(f"taskgraph lease: timed out waiting for {resource}", file=stream, flush=True)
            return TIMEOUT_EXIT
        restore_mask(previous)
        env = dict(os.environ)
        env[lease.LEASE_ENV] = resource
        try:
            try:
                # Own session: its pid is a process-group id, so a forwarded
                # signal reaches the whole leased command tree.
                proc = subprocess.Popen(list(command), env=env, start_new_session=True)
            except OSError as exc:
                print(
                    f"taskgraph lease: cannot run {command[0]!r}: {exc}", file=stream, flush=True
                )
                return 127
            forwarder.attach(proc)
            code = proc.wait()
            terminate_group(proc.pid, grace=group_grace)
        finally:
            lease.release(slot, root=base)
    finally:
        if announcement is not None:
            leaseprocs.unregister_process(announcement)
        forwarder.restore()
        restore_mask(previous)
    return code if code >= 0 else 128 - code


def _cmd_run(args: argparse.Namespace) -> int:
    """``taskgraph run``: start/resume the scheduler, or print the plan (SPEC §10)."""
    if args.project:
        config_path = Path(args.project) / CONFIG_NAME
        if not config_path.is_file():
            print(f"taskgraph run: no {CONFIG_NAME} in {args.project}", file=sys.stderr)
            return 2
    else:
        config_path = find_config()
        if config_path is None:
            print(
                f"taskgraph run: no {CONFIG_NAME} found in this directory or its parents",
                file=sys.stderr,
            )
            return 2
    try:
        config = load(config_path)
    except ConfigError as exc:
        print(f"taskgraph run: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(assign.dry_run(config, max_agents=args.max_agents))
        return 0
    return scheduler.Scheduler(
        config, max_agents=args.max_agents, tick_secs=scheduler.env_tick_secs()
    ).run()


def _project_config(command: str) -> Config | None:
    """Load the nearest project config for ``command``; ``None`` after reporting why not."""
    config_path = find_config()
    if config_path is None:
        print(
            f"taskgraph {command}: no {CONFIG_NAME} found in this directory or its parents",
            file=sys.stderr,
        )
        return None
    try:
        return load(config_path)
    except ConfigError as exc:
        print(f"taskgraph {command}: {exc}", file=sys.stderr)
        return None


def _cmd_lease(args: argparse.Namespace) -> int:
    """``taskgraph lease``: wait for a resource slot, then run the command (SPEC §5)."""
    if not args.cmd:
        print("taskgraph lease: no command given", file=sys.stderr)
        return 2
    config = _project_config("lease")
    if config is None:
        return 2
    capacity = config.resources.get(args.resource)
    if capacity is None:
        print(
            f"taskgraph lease: resource '{args.resource}' is not in [resources] of {config.path}",
            file=sys.stderr,
        )
        return 2
    return run_lease(args.resource, args.cmd, capacity=capacity, task=args.task)


def _cmd_side(args: argparse.Namespace) -> int:
    """``taskgraph side``: lease a model's job-class capacity (SPEC §4)."""
    if not args.cmd:
        print("taskgraph side: no command given", file=sys.stderr)
        return 2
    config = _project_config("side")
    if config is None:
        return 2
    try:
        return side.run_side(args.cls, args.cmd, config, task=args.task)
    except side.SideError as exc:
        print(f"taskgraph side: {exc}", file=sys.stderr)
        return 2


def _cmd_stats(args: argparse.Namespace) -> int:
    """``taskgraph stats``: split each task's wall time from its trace (SPEC §9)."""
    config = _project_config("stats")
    if config is None:
        return 2
    try:
        window = stats.parse_window(args.since)
    except ValueError as exc:
        print(f"taskgraph stats: {exc}", file=sys.stderr)
        return 2
    print(stats.render(stats.collect(config, since=window)))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """``taskgraph models``: throughput per model by concurrency (SPEC §4)."""
    config = _project_config("models")
    if config is None:
        return 2
    try:
        table = models.collect(config)
    except state.StateError as exc:
        print(f"taskgraph models: {exc}", file=sys.stderr)
        return 2
    print(models.render(table))
    return 0


def _cmd_leases(args: argparse.Namespace) -> int:
    """``taskgraph leases``: print current holders and their ages (SPEC §5)."""
    print(status.format_holders(lease.holders()))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """``taskgraph status``: running agents, queue, blocked, leases, load (SPEC §9)."""
    config = _project_config("status")
    if config is None:
        return 2
    try:
        snapshot = status.collect(config)
    except state.StateError as exc:
        print(f"taskgraph status: {exc}", file=sys.stderr)
        return 2
    print(status.render(snapshot))
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    """``taskgraph stop``: stop the scheduler; ``--agents`` stops agents too (SPEC §7)."""
    config = _project_config("stop")
    if config is None:
        return 2
    try:
        result = control.stop(config.root, agents=args.agents)
    except control.ControlError as exc:
        print(f"taskgraph stop: {exc}", file=sys.stderr)
        return 1
    if result.pid is None:
        print(f"taskgraph stop: no scheduler running for {config.root}", file=sys.stderr)
    else:
        how = "killed" if result.forced else "stopped"
        print(f"taskgraph stop: {how} scheduler pid {result.pid}")
    if args.agents:
        if result.agents:
            print(f"taskgraph stop: stopped agents {', '.join(result.agents)}")
        else:
            print("taskgraph stop: no agents running", file=sys.stderr)
    return 0 if result.stopped_anything() else 1


def _cmd_router(args: argparse.Namespace) -> int:
    """``taskgraph router``: serve the OpenAI-compatible request router (SPEC §11)."""
    if args.project:
        config_path = Path(args.project) / CONFIG_NAME
        if not config_path.is_file():
            print(f"taskgraph router: no {CONFIG_NAME} in {args.project}", file=sys.stderr)
            return 2
    else:
        config_path = find_config()
        if config_path is None:
            print(
                f"taskgraph router: no {CONFIG_NAME} found in this directory or its parents",
                file=sys.stderr,
            )
            return 2
    try:
        config = load(config_path)
    except ConfigError as exc:
        print(f"taskgraph router: {exc}", file=sys.stderr)
        return 2
    return proxyserver.run_router(
        config,
        host=args.host,
        port=args.port,
        queue_timeout=args.queue_timeout,
    )


def _cmd_retry(args: argparse.Namespace) -> int:
    """``taskgraph retry <id>``: unblock a task and resume its worktree (SPEC §10)."""
    config = _project_config("retry")
    if config is None:
        return 2
    try:
        result = control.retry(config, args.id)
    except control.ControlError as exc:
        print(f"taskgraph retry: {exc}", file=sys.stderr)
        return 1
    where = ""
    if worktree.is_resume(config, result.id):
        where = f"; {worktree.path(config, result.id)} will be resumed"
    if result.was_blocked:
        print(f"taskgraph retry: queued a retry for {result.id} (was blocked: {result.reason}){where}")
    else:
        print(f"taskgraph retry: queued a retry for {result.id} (it was not blocked){where}")
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
    sub.choices["run"].set_defaults(func=_cmd_run)
    sub.choices["status"].set_defaults(func=_cmd_status)
    sub.choices["stats"].set_defaults(func=_cmd_stats)
    sub.choices["models"].set_defaults(func=_cmd_models)
    sub.choices["lease"].set_defaults(func=_cmd_lease)
    sub.choices["side"].set_defaults(func=_cmd_side)
    sub.choices["router"].set_defaults(func=_cmd_router)
    sub.choices["leases"].set_defaults(func=_cmd_leases)
    sub.choices["stop"].set_defaults(func=_cmd_stop)
    sub.choices["retry"].set_defaults(func=_cmd_retry)

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

    side_parser = sub.choices["side"]
    side_parser.add_argument("cls", metavar="CLASS", help="job class, e.g. side")
    side_parser.add_argument("--task", metavar="ID", help="task id recorded in the slot file")
    side_parser.add_argument(
        "cmd", nargs="+", help="command to run, after -- (gets TASKGRAPH_MODEL)"
    )

    router_parser = sub.choices["router"]
    router_parser.add_argument("--project", metavar="DIR", help="project root (default: cwd)")
    router_parser.add_argument("--host", default=proxyserver.DEFAULT_HOST, help="listen address")
    router_parser.add_argument(
        "--port", type=int, default=proxyserver.DEFAULT_PORT, help="listen port (0 picks a free one)"
    )
    router_parser.add_argument(
        "--queue-timeout",
        type=float,
        default=0.0,
        metavar="SECS",
        help="give up on a request that waited this long (0 waits forever)",
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
