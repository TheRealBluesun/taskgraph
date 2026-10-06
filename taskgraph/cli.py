"""Command-line interface for taskgraph.

Skeleton: every subcommand from SPEC §10 parses its arguments and reports that
the work behind it is not implemented yet. Each later task replaces one stub.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

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

    run = sub.choices["run"]
    run.add_argument("--project", metavar="DIR", help="project root (default: cwd)")
    run.add_argument("--max-agents", type=int, metavar="N", help="cap on concurrent agents")
    run.add_argument("--dry-run", action="store_true", help="print the plan and exit")

    sub.choices["stats"].add_argument("--since", default="6h", metavar="WINDOW")

    sub.choices["stop"].add_argument("--agents", action="store_true", help="also stop agents")

    lease = sub.choices["lease"]
    lease.add_argument("resource", help="resource name, e.g. simulator")
    lease.add_argument("--task", metavar="ID", help="task id recorded in the slot file")
    lease.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (after --)")

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
