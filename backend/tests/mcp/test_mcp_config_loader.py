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


def test_optional_env_missing_is_dropped(tmp_path, monkeypatch):
    """${VAR?} 语法：env 缺失时不当报错，只是从传给子进程的 env 里剔除。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("GMAIL_SENDER_EMAIL", "sandbox@example.com")
    monkeypatch.delenv("GMAIL_HTTPS_PROXY", raising=False)
    monkeypatch.delenv("GMAIL_HTTP_PROXY", raising=False)
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
      GMAIL_SENDER_EMAIL: "${GMAIL_SENDER_EMAIL}"
      HTTPS_PROXY: "${GMAIL_HTTPS_PROXY?}"
      HTTP_PROXY: "${GMAIL_HTTP_PROXY?}"
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)
    env = cfg.servers["gmail"].env
    assert env["GMAIL_SENDER_EMAIL"] == "sandbox@example.com"
    assert "HTTPS_PROXY" not in env
    assert "HTTP_PROXY" not in env


def test_optional_env_present_is_passed(tmp_path, monkeypatch):
    """${VAR?} 语法：env 设了就正常展开传给子进程。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("GMAIL_HTTPS_PROXY", "http://127.0.0.1:7897")
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
      HTTPS_PROXY: "${GMAIL_HTTPS_PROXY?}"
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)
    assert cfg.servers["gmail"].env["HTTPS_PROXY"] == "http://127.0.0.1:7897"


# ── P2：feishu server 配置 ────────────────────────────────────────


def test_feishu_enabled_with_include_tools(tmp_path, monkeypatch):
    """启用 feishu 后，include_tools 里 4 个 P2 工具正确解析，env 正常展开。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("LARK_APP_ID", "cli_test_app")
    monkeypatch.setenv("LARK_APP_SECRET", "secret_test")
    monkeypatch.setenv("LARK_DOMAIN", "https://open.feishu.cn")
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
tool_name_prefix: true
servers:
  feishu:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.feishu.server"]
    include_tools:
      - send_message_to_user
      - send_message_to_group
      - send_file_to_user
      - send_file_to_group
    env:
      LARK_APP_ID: "${LARK_APP_ID}"
      LARK_APP_SECRET: "${LARK_APP_SECRET}"
      LARK_DOMAIN: "${LARK_DOMAIN?}"
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)

    assert list(cfg.servers) == ["feishu"]
    server = cfg.servers["feishu"]
    assert server.include_tools == [
        "send_message_to_user",
        "send_message_to_group",
        "send_file_to_user",
        "send_file_to_group",
    ]
    assert server.env["LARK_APP_ID"] == "cli_test_app"
    assert server.env["LARK_APP_SECRET"] == "secret_test"
    assert server.env["LARK_DOMAIN"] == "https://open.feishu.cn"


def test_feishu_missing_required_env_fails_fast(tmp_path, monkeypatch):
    """LARK_APP_ID 是必填 ${VAR}——启用但未提供时 fail-fast，避免半启动。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
servers:
  feishu:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.feishu.server"]
    env:
      LARK_APP_ID: "${LARK_APP_ID}"
      LARK_APP_SECRET: "${LARK_APP_SECRET}"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="LARK_APP_ID"):
        load_mcp_config(cfg_file)


def test_feishu_disabled_when_flag_off(tmp_path, monkeypatch):
    """feishu.enabled=false 时不解析 env，也不出现在 servers 里，
    这样默认部署（无 LARK 凭据）也能启动。"""
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.setenv("GMAIL_SENDER_EMAIL", "sandbox@example.com")
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
      GMAIL_SENDER_EMAIL: "${GMAIL_SENDER_EMAIL}"
  feishu:
    enabled: false
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.feishu.server"]
    env:
      LARK_APP_ID: "${LARK_APP_ID}"
""",
        encoding="utf-8",
    )

    cfg = load_mcp_config(cfg_file)
    assert list(cfg.servers) == ["gmail"]
