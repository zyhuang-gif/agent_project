import os
import sys
from pathlib import Path

import pytest

from app.mcp.config_loader import load_mcp_config


def test_load_config_expands_env_and_filters_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("GMAIL_SENDER_EMAIL", "sandbox@example.com")
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
tool_name_prefix: true
servers:
  gmail:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.gmail.server"]
    include_tools: ["send_email"]
    env:
      GMAIL_SENDER_EMAIL: "${GMAIL_SENDER_EMAIL}"
  feishu:
    enabled: false
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.feishu.server"]
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)

    assert cfg.enabled is True
    assert cfg.tool_name_prefix is True
    assert list(cfg.servers) == ["gmail"]
    assert cfg.servers["gmail"].env["GMAIL_SENDER_EMAIL"] == "sandbox@example.com"
    assert cfg.server_connections()["gmail"]["transport"] == "stdio"
    assert cfg.server_connections()["gmail"]["args"] == ["-m", "mcp_servers.gmail.server"]


def test_config_disabled_by_env_returns_no_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "false")
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
tool_name_prefix: true
servers:
  gmail:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.gmail.server"]
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)

    assert cfg.enabled is False
    assert cfg.servers == {}
    assert cfg.server_connections() == {}


def test_enabled_server_missing_env_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
servers:
  gmail:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.gmail.server"]
    env:
      GMAIL_TOKEN_PATH: "${GMAIL_TOKEN_PATH}"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="GMAIL_TOKEN_PATH"):
        load_mcp_config(cfg_file)


def test_stdio_command_python_is_rewritten_to_current_interpreter(tmp_path, monkeypatch):
    """command: python 必须被解析为当前后端进程的 sys.executable，
    否则 stdio 子进程会用系统全局 Python 起，看不到 backend/.venv 里的 mcp 包。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
tool_name_prefix: true
servers:
  gmail:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.gmail.server"]
  contacts:
    enabled: true
    transport: stdio
    command: python3
    args: ["-m", "mcp_servers.contacts.server"]
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)

    assert cfg.servers["gmail"].command == sys.executable
    assert cfg.servers["contacts"].command == sys.executable
    connections = cfg.server_connections()
    assert connections["gmail"]["command"] == sys.executable
    assert connections["contacts"]["command"] == sys.executable


def test_stdio_absolute_command_is_kept_as_is(tmp_path, monkeypatch):
    """已经写死绝对路径的 command 不应被覆盖（比如用户就是要用系统 Python）。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    # yaml 里避免反斜杠转义问题，统一走 forward slash
    custom_path = (Path(sys.executable).parent / "custom_python.exe").as_posix()
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        f"""
servers:
  gmail:
    enabled: true
    transport: stdio
    command: "{custom_path}"
    args: ["-m", "mcp_servers.gmail.server"]
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)
    assert cfg.servers["gmail"].command == custom_path
