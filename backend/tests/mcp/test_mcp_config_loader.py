import os
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
