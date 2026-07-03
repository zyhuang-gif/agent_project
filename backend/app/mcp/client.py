"""MCP 客户端管理器：应用启动时扫描配置、通过 langchain-mcp-adapters 拉一次
工具清单并按 include_tools 白名单过滤，供 send 节点按名字获取。

设计取舍（见 spec §3）：
- adapters.MultiServerMCPClient.get_tools() 会在每次 tool 调用时按 connection 拉起一次
  MCP session，因此这里 start() 主要是配置校验 + 工具发现，shutdown() 只需要清理内部
  状态，不必维护长驻的 stdio 子进程。
"""
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.logger_handler import logger
from app.mcp.config_loader import MCPConfig, load_mcp_config
from app.mcp.send_guard import SendGuardInterceptor


class MCPClientManager:
    def __init__(self):
        self._config: MCPConfig | None = None
        self._client = None
        self._tools_by_name: dict[str, object] = {}
        self._started = False

    async def start(self):
        self._config = load_mcp_config()
        self._tools_by_name = {}
        self._started = True
        if not self._config.enabled:
            logger.info("[MCP] disabled by MCP_ENABLED")
            return

        self._client = MultiServerMCPClient(
            self._config.server_connections(),
            tool_interceptors=[SendGuardInterceptor()],
            tool_name_prefix=self._config.tool_name_prefix,
            handle_tool_errors=True,
        )
        try:
            tools = await self._client.get_tools()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[MCP] tool discovery failed: {exc}", exc_info=True)
            tools = []
        self._tools_by_name = {
            tool.name: tool
            for tool in tools
            if self._included_tool_name(tool.name)
        }
        logger.info(f"[MCP] loaded tools: {list(self._tools_by_name)}")

    async def shutdown(self):
        # adapters 层的 tool 每次调用都会新建 session；这里不需要显式关闭长驻子进程。
        self._tools_by_name = {}
        self._client = None
        self._started = False

    def get_tools(self, names: list[str] | None = None) -> list[object]:
        if names is None:
            return list(self._tools_by_name.values())
        return [self._tools_by_name[name] for name in names if name in self._tools_by_name]

    def _included_tool_name(self, tool_name: str) -> bool:
        if not self._config:
            return False
        for server_name, server in self._config.servers.items():
            if not server.include_tools:
                continue
            expected = {
                f"{server_name}_{name}" if self._config.tool_name_prefix else name
                for name in server.include_tools
            }
            if tool_name in expected:
                return True
        return False


mcp_manager = MCPClientManager()
