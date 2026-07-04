import pytest


@pytest.mark.asyncio
async def test_mcp_manager_filters_prefixed_tools(monkeypatch):
    from app.mcp.client import MCPClientManager
    from app.mcp.config_loader import MCPConfig, MCPServerConfig

    class _Tool:
        def __init__(self, name):
            self.name = name

    class _Client:
        def __init__(self, connections, **kwargs):
            self.connections = connections
            self.kwargs = kwargs

        async def get_tools(self):
            return [
                _Tool("gmail_send_email"),
                _Tool("contacts_resolve"),
                _Tool("contacts_list_all"),
            ]

    monkeypatch.setattr("app.mcp.client.MultiServerMCPClient", _Client)

    cfg = MCPConfig(
        enabled=True,
        tool_name_prefix=True,
        servers={
            "gmail": MCPServerConfig(
                name="gmail",
                transport="stdio",
                command="python",
                args=["-m", "mcp_servers.gmail.server"],
                include_tools=["send_email"],
            ),
            "contacts": MCPServerConfig(
                name="contacts",
                transport="stdio",
                command="python",
                args=["-m", "mcp_servers.contacts.server"],
                include_tools=["resolve"],
            ),
        },
    )
    monkeypatch.setattr("app.mcp.client.load_mcp_config", lambda: cfg)

    manager = MCPClientManager()
    await manager.start()

    assert [tool.name for tool in manager.get_tools()] == [
        "gmail_send_email",
        "contacts_resolve",
    ]
    assert [tool.name for tool in manager.get_tools(["gmail_send_email"])] == [
        "gmail_send_email"
    ]
    assert manager.get_tools(["missing"]) == []


@pytest.mark.asyncio
async def test_mcp_manager_loads_prefixed_feishu_tools(monkeypatch):
    """P2：feishu server enabled + include_tools 生效后，可对 send 节点提供 4 个前缀化工具。"""
    from app.mcp.client import MCPClientManager
    from app.mcp.config_loader import MCPConfig, MCPServerConfig

    class _Tool:
        def __init__(self, name):
            self.name = name

    class _Client:
        def __init__(self, connections, **kwargs):
            self.connections = connections
            self.kwargs = kwargs

        async def get_tools(self):
            # 假设子进程真的暴露了 send_* 与 upload_file，adapters 加前缀后：
            return [
                _Tool("feishu_send_message_to_user"),
                _Tool("feishu_send_message_to_group"),
                _Tool("feishu_send_file_to_user"),
                _Tool("feishu_send_file_to_group"),
                _Tool("feishu_upload_file"),  # 不在 include_tools 中，应被过滤
            ]

    monkeypatch.setattr("app.mcp.client.MultiServerMCPClient", _Client)

    cfg = MCPConfig(
        enabled=True,
        tool_name_prefix=True,
        servers={
            "feishu": MCPServerConfig(
                name="feishu",
                transport="stdio",
                command="python",
                args=["-m", "mcp_servers.feishu.server"],
                include_tools=[
                    "send_message_to_user",
                    "send_message_to_group",
                    "send_file_to_user",
                    "send_file_to_group",
                ],
                env={"LARK_APP_ID": "cli_x", "LARK_APP_SECRET": "y"},
            ),
        },
    )
    monkeypatch.setattr("app.mcp.client.load_mcp_config", lambda: cfg)

    manager = MCPClientManager()
    await manager.start()

    names = [t.name for t in manager.get_tools()]
    assert set(names) == {
        "feishu_send_message_to_user",
        "feishu_send_message_to_group",
        "feishu_send_file_to_user",
        "feishu_send_file_to_group",
    }
    # upload_file 不在白名单里
    assert "feishu_upload_file" not in names
    # send 节点常用取法
    assert [t.name for t in manager.get_tools(["feishu_send_message_to_user"])] == [
        "feishu_send_message_to_user"
    ]
