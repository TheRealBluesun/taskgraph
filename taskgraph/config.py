"""Load and validate a project's ``taskgraph.toml`` (SPEC §1).

Everything project-specific (paths, gate command, resources, agent command,
models) comes from that one file. Parsing is pure: no I/O beyond reading the
file, so the resulting :class:`Config` is easy to build in tests.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Defaults for keys SPEC §1 shows in the example but does not require.
DEFAULT_OVERLAY = "omp-agent.yml"
DEFAULT_STALL_SECS = 480.0
DEFAULT_RETRIES = 2

_REQUIRED_ROOT_KEYS = ("plan", "prompt", "gate", "worktrees", "main")

# Distinguishes "key absent, caller supplied a default" from "key absent, required".
_MISSING = object()


class ConfigError(Exception):
    """The config file is unreadable, malformed, or missing/invalid a key."""


@dataclass(frozen=True)
class ModelConfig:
    """One ``[[models]]`` entry: a model server agents can be assigned to."""

    name: str
    sessions: int
    max_agents: int
    metrics: str | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    """The ``[agent]`` table: how each coding agent is launched and watched."""

    command: str
    overlay: str = DEFAULT_OVERLAY
    stall_secs: float = DEFAULT_STALL_SECS
    retries: int = DEFAULT_RETRIES
    deny: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    """A validated ``taskgraph.toml``."""

    path: Path
    plan: str
    prompt: str
    gate: str
    worktrees: str
    main: str
    links: tuple[str, ...]
    resources: Mapping[str, int]
    agent: AgentConfig
    models: tuple[ModelConfig, ...]

    @property
    def root(self) -> Path:
        """Project root: the directory containing the config file."""
        return self.path.parent

    def model(self, name: str) -> ModelConfig | None:
        """Return the model called ``name``, or ``None`` if it is not configured."""
        for model in self.models:
            if model.name == name:
                return model
        return None


def load(path: str | Path) -> Config:
    """Read and validate the config file at ``path``.

    Raises :class:`ConfigError` naming the offending key and file.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config: {exc.strerror or exc}") from exc
    try:
        data: Any = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path}: config is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    return _build(data, path)


def _build(data: Mapping[str, Any], path: Path) -> Config:
    """Turn a parsed TOML mapping into a validated :class:`Config`."""
    for key in _REQUIRED_ROOT_KEYS:
        if key not in data:
            raise ConfigError(f"{path}: missing required key '{key}'")
    return Config(
        path=path,
        plan=_string(data, "plan", path, where=""),
        prompt=_string(data, "prompt", path, where=""),
        gate=_string(data, "gate", path, where=""),
        worktrees=_string(data, "worktrees", path, where=""),
        main=_string(data, "main", path, where=""),
        links=_string_list(data, "links", path, where="", default=()),
        resources=_resources(data.get("resources"), path),
        agent=_agent(data.get("agent"), path),
        models=_models(data.get("models"), path),
    )


def _resources(raw: Any, path: Path) -> Mapping[str, int]:
    """Validate the optional ``[resources]`` table (name = capacity ≥ 1)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [resources] must be a table")
    resources: dict[str, int] = {}
    for name, capacity in raw.items():
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ConfigError(
                f"{path}: [resources] '{name}' must be a positive integer, not {capacity!r}"
            )
        resources[name] = capacity
    return resources


def _agent(raw: Any, path: Path) -> AgentConfig:
    """Validate the required ``[agent]`` table."""
    if raw is None:
        raise ConfigError(f"{path}: missing required table [agent]")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [agent] must be a table")
    command = _string(raw, "command", path, where="[agent] ")
    overlay = _string(raw, "overlay", path, where="[agent] ", default=DEFAULT_OVERLAY)
    stall_secs = _number(raw, "stall_secs", path, where="[agent] ", default=DEFAULT_STALL_SECS)
    if stall_secs <= 0:
        raise ConfigError(f"{path}: [agent] 'stall_secs' must be > 0, not {stall_secs!r}")
    retries = _int(raw, "retries", path, where="[agent] ", default=DEFAULT_RETRIES, minimum=0)
    return AgentConfig(
        command=command,
        overlay=overlay,
        stall_secs=float(stall_secs),
        retries=retries,
        deny=_string_list(raw, "deny", path, where="[agent] ", default=()),
    )


def _models(raw: Any, path: Path) -> tuple[ModelConfig, ...]:
    """Validate the required, non-empty ``[[models]]`` array of tables."""
    if raw is None:
        raise ConfigError(f"{path}: missing required [[models]] entries")
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: [[models]] must be an array of tables")
    if not raw:
        raise ConfigError(f"{path}: [[models]] must list at least one model")

    models: list[ModelConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        where = f"[[models]][{index}] "
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: {where}must be a table")
        name = _string(entry, "name", path, where=where)
        if name in seen:
            raise ConfigError(f"{path}: {where}duplicate model name '{name}'")
        seen.add(name)
        sessions = _int(entry, "sessions", path, where=where, minimum=0)
        max_agents = _int(entry, "max_agents", path, where=where, default=sessions, minimum=0)
        if max_agents < sessions:
            raise ConfigError(
                f"{path}: {where}'max_agents' ({max_agents}) must be >= 'sessions' ({sessions})"
            )
        models.append(
            ModelConfig(
                name=name,
                sessions=sessions,
                max_agents=max_agents,
                metrics=_string(entry, "metrics", path, where=where, default=None),
                fallback=_string(entry, "fallback", path, where=where, default=None),
            )
        )

    names = {model.name for model in models}
    for model in models:
        if model.fallback is not None and model.fallback not in names:
            raise ConfigError(
                f"{path}: model '{model.name}' has unknown fallback '{model.fallback}'"
            )
    return tuple(models)


def _string(
    table: Mapping[str, Any],
    key: str,
    path: Path,
    *,
    where: str,
    default: Any = _MISSING,
) -> Any:
    """Return a non-empty string value, or ``default`` when the key is absent."""
    if key not in table:
        if default is not _MISSING:
            return default
        raise ConfigError(f"{path}: missing required key '{where}{key}'")
    value = table[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: '{where}{key}' must be a non-empty string, not {value!r}")
    return value


def _int(
    table: Mapping[str, Any],
    key: str,
    path: Path,
    *,
    where: str,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    """Return an integer value (bools rejected), validating ``minimum``."""
    if key not in table:
        if default is not None:
            return default
        raise ConfigError(f"{path}: missing required key '{where}{key}'")
    value = table[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path}: '{where}{key}' must be an integer, not {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}: '{where}{key}' must be >= {minimum}, not {value!r}")
    return value


def _number(
    table: Mapping[str, Any],
    key: str,
    path: Path,
    *,
    where: str,
    default: float,
) -> float:
    """Return an int-or-float value, or ``default`` when the key is absent."""
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path}: '{where}{key}' must be a number, not {value!r}")
    return float(value)


def _string_list(
    table: Mapping[str, Any],
    key: str,
    path: Path,
    *,
    where: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Return a tuple of non-empty strings, or ``default`` when the key is absent."""
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError(f"{path}: '{where}{key}' must be an array of non-empty strings, not {value!r}")
    return tuple(value)
