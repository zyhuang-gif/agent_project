"""MCP tool 调用的请求级上下文（user_id / session_id）。

send_guard interceptor 需要知道触发本次发送的用户与会话，但 MCP tool 是通过
create_react_agent → LangChain BaseTool → adapters 层调用的，没法直接把这些信息
作为参数塞进去。用 ContextVar 在 send 节点入口处 set，在 interceptor 里读，
async-safe 且不会串会话。
"""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional


_mcp_context: ContextVar[dict] = ContextVar("mcp_context", default={})


def current_mcp_context() -> dict:
    return dict(_mcp_context.get() or {})


@asynccontextmanager
async def mcp_call_context(user_id: Optional[str], session_id: Optional[str]):
    token = _mcp_context.set({"user_id": user_id, "session_id": session_id})
    try:
        yield
    finally:
        _mcp_context.reset(token)
