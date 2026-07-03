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
