import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(\?)?\}$")
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "mcp_servers.yaml"
# yaml 里写 `command: python` 时，实际启动 stdio MCP 子进程会走 PATH 上第一个 python，
# 这在 Windows 下常常落到系统全局 Python，导致子进程看不到 backend/.venv 里的 mcp 包。
# 这里把非绝对路径的 python/python3 都改写成当前后端进程的 sys.executable，
# 让 MCP server 一定用同一个 venv 起。
_INTERPRETER_ALIASES = {"python", "python3", "python.exe", "python3.exe"}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    include_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MCPConfig:
    enabled: bool
    tool_name_prefix: bool
    servers: dict[str, MCPServerConfig]

    def server_connections(self) -> dict[str, dict[str, Any]]:
        connections: dict[str, dict[str, Any]] = {}
        for name, server in self.servers.items():
            entry: dict[str, Any] = {"transport": server.transport}
            if server.command:
                entry["command"] = server.command
            if server.args:
                entry["args"] = list(server.args)
            if server.url:
                entry["url"] = server.url
            if server.env:
                entry["env"] = dict(server.env)
            connections[name] = entry
        return connections


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def _expand_env_value(key: str, value: Any) -> str | None:
    """展开 yaml 里的 ${VAR} / ${VAR?} 占位。

    ${VAR}   —— 必填；env 缺失时抛错（fail-fast，避免半启动状态）。
    ${VAR?}  —— 可选；env 缺失时返回 None，调用方会把它从传给子进程的 env 里剔除。
    其它字符串原样返回。
    """
    if value is None:
        return None
    text = str(value)
    match = _ENV_PATTERN.match(text)
    if not match:
        return text
    env_name = match.group(1)
    optional = match.group(2) == "?"
    resolved = os.getenv(env_name)
    if not resolved:
        if optional:
            return None
        raise ValueError(f"MCP enabled server requires env {env_name} for {key}")
    return resolved


def _resolve_command(command: str | None) -> str | None:
    if command is None:
        return None
    text = str(command).strip()
    if not text:
        return text
    # 已经是绝对路径 / 相对路径（带斜杠），或与 sys.executable 相同——原样保留
    if os.path.sep in text or "/" in text:
        return text
    if text.lower() in _INTERPRETER_ALIASES:
        return sys.executable
    return text


def load_mcp_config(path: Path | None = None) -> MCPConfig:
    enabled = _env_bool("MCP_ENABLED", "false")
    config_path = path or _DEFAULT_CONFIG_PATH
    if not enabled:
        return MCPConfig(enabled=False, tool_name_prefix=True, servers={})
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tool_name_prefix = bool(raw.get("tool_name_prefix", True))
    servers: dict[str, MCPServerConfig] = {}

    for name, data in (raw.get("servers") or {}).items():
        data = data or {}
        if not bool(data.get("enabled", False)):
            continue
        env = {}
        for env_key, env_val in (data.get("env") or {}).items():
            resolved = _expand_env_value(env_key, env_val)
            if resolved is not None:
                env[env_key] = resolved
        server = MCPServerConfig(
            name=name,
            transport=str(data.get("transport", "stdio")),
            command=_resolve_command(data.get("command")),
            args=list(data.get("args") or []),
            url=data.get("url"),
            env=env,
            include_tools=list(data.get("include_tools") or []),
        )
        if server.transport == "stdio" and not server.command:
            raise ValueError(f"MCP server {name} uses stdio but has no command")
        servers[name] = server

    return MCPConfig(enabled=True, tool_name_prefix=tool_name_prefix, servers=servers)
