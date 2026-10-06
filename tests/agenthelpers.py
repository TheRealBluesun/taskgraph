"""Shared fixtures for the prompt/agent tests (SPEC §6).

Agents are faked with tiny shell scripts: the runner only ever launches
``cfg.agent.command``, so a script that writes a trace, sleeps and exits stands
in for omp without touching a model or the network.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from taskgraph import agent
from taskgraph.config import AgentConfig, Config, ModelConfig
from taskgraph.plan import Task
from taskgraph.state import State

PROJECT_PROMPT = "PROJECT RULES\nNever break the build.\n"


def write_executable(path: Path, text: str) -> Path:
    """Write ``text`` (a full script) to ``path`` and mark it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def make_config(
    root: Path,
    command: str,
    *,
    plan: str = "PLAN.md",
    prompt: str = "PROMPT.md",
    project_prompt: str = PROJECT_PROMPT,
    overlay: str = "omp-agent.yml",
    deny: tuple[str, ...] = (),
    stall_secs: float = 480.0,
    retries: int = 2,
    models: tuple[ModelConfig, ...] | None = None,
) -> Config:
    """Build a :class:`Config` whose prompt file exists under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / prompt).write_text(project_prompt, encoding="utf-8")
    return Config(
        path=root / "taskgraph.toml",
        plan=plan,
        prompt=prompt,
        gate="true",
        worktrees="../wt",
        main="main",
        links=(),
        resources={},
        agent=AgentConfig(
            command=command, overlay=overlay, stall_secs=stall_secs, retries=retries, deny=deny
        ),
        models=models or (ModelConfig(name="m1", sessions=1, max_agents=2),),
    )


def make_task(tid: str = "T01", text: str = "do the thing") -> Task:
    """Build a plan task without parsing a plan file."""
    return Task(id=tid, text=text)


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.02):
    """Poll ``predicate`` until it is truthy; raise :class:`AssertionError` on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        time.sleep(interval)


class AgentTestCase(unittest.TestCase):
    """A temp project, worktree and state; every started agent is killed on cleanup."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.worktree = self.base / "wt"
        self.worktree.mkdir(parents=True)
        self.state_root = self.base / "state"
        self.state = State()

    def script(self, body: str, name: str = "fake-agent.sh") -> Path:
        """Write an executable shell script and return its path."""
        return write_executable(self.base / name, "#!/bin/sh\n" + body + "\n")

    def config(self, command: str, **kwargs) -> Config:
        return make_config(self.root, command, **kwargs)

    def launch(self, command: str, *, task: Task | None = None, cfg: Config | None = None, **kwargs) -> agent.AgentRecord:
        """Start a fake agent and arrange for its process group to be cleaned up."""
        cfg = cfg if cfg is not None else self.config(command)
        model = kwargs.pop("model", cfg.models[0])
        record = agent.start(
            task or make_task(),
            model,
            self.worktree,
            cfg,
            kwargs.pop("state", self.state),
            root=self.state_root,
            **kwargs,
        )
        self.addCleanup(self.stop_record, record)
        return record

    def stop_record(self, record: agent.AgentRecord) -> None:
        """Kill an agent's process group if it is somehow still running."""
        try:
            if agent.poll(record) == "running":
                agent.kill(record, timeout=1.0)
        except OSError:  # pragma: no cover - cleanup must never mask a failure
            pass

    def log_text(self, record: agent.AgentRecord) -> str:
        """Return the trace text written so far."""
        try:
            return Path(record.log).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def wait_log(self, record: agent.AgentRecord, needle: str, timeout: float = 5.0) -> str:
        """Wait until ``needle`` appears in the agent's trace; return the trace."""
        return wait_for(lambda: self.log_text(record) if needle in self.log_text(record) else None, timeout)


__all__ = [
    "AgentTestCase",
    "PROJECT_PROMPT",
    "make_config",
    "make_task",
    "wait_for",
    "write_executable",
]
