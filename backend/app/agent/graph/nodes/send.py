"""send 节点：把 finalize 生成的 final_answer 通过 Gmail MCP 工具发出。

流程（spec §3.2）：
  1. 无 send_intent 直接返回空 dict（等价于跳过）。
  2. resolve_send_targets 做确定性 preflight：收件人/通道任何不确定 → 短路，
     返回 send_report + trace，不调 Gmail 工具。
  3. build_send_payload 生成专用 subject/body，不直接把 final_answer 塞进 Gmail。
  4. 局部 create_react_agent 只暴露 gmail_send_email；ContextVar 挂 user/session
     以便 send_guard interceptor 审计。
"""
from langgraph.prebuilt import create_react_agent

from app.agent.graph._stream import safe_get_stream_writer
from app.agent.graph.state import AgentState
from app.mcp.client import mcp_manager
from app.mcp.context import mcp_call_context
from app.mcp.payload import build_send_payload
from app.mcp.preflight import resolve_send_targets
from app.utils.factory import get_chat_model


async def send_node(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    if not plan.get("send_intent"):
        return {}

    writer = safe_get_stream_writer()
    writer({
        "kind": "step",
        "id": "send_preparing",
        "status": "running",
        "level": "info",
        "detail": "正在准备发送",
        "title": "准备发送",
    })

    preflight = await resolve_send_targets(
        recipients=plan.get("recipients") or [],
        channels=plan.get("channels") or [],
        identity=state.get("identity"),
    )
    if not preflight.can_send:
        writer({
            "kind": "step",
            "id": "send_preparing",
            "status": "done",
            "level": "warning",
            "detail": preflight.user_message,
            "title": "未发送",
        })
        return {
            "send_report": preflight.user_message,
            "trace": [
                {"agent": "send", "status": "skipped", "output": preflight.reason}
            ],
        }

    payload = await build_send_payload(
        source_answer=state.get("final_answer") or "",
        original_query=state.get("query") or "",
        recipients=preflight.recipients,
    )

    tools = mcp_manager.get_tools(names=["gmail_send_email"])
    if not tools:
        report = "发送能力当前不可用：Gmail MCP 工具未加载。"
        writer({
            "kind": "step",
            "id": "send_preparing",
            "status": "done",
            "level": "warning",
            "detail": report,
            "title": "未发送",
        })
        return {
            "send_payload": payload,
            "send_report": report,
            "trace": [
                {"agent": "send", "status": "skipped", "output": "gmail_tool_unavailable"}
            ],
        }

    agent = create_react_agent(model=get_chat_model("send"), tools=tools)
    task = (
        "使用下面的 send_payload 发送 Gmail。"
        "不要添加附件，不要改写收件人，不要重新生成正文，不要调用未提供的工具。\n"
        f"收件人邮箱：{preflight.gmail_to}\n"
        f"subject：{payload.subject}\n"
        f"body：{payload.body_text}"
    )

    identity = state.get("identity")
    user_id = getattr(identity, "user_id", None)
    async with mcp_call_context(user_id=user_id, session_id=state.get("session_id")):
        result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})

    report = result["messages"][-1].content
    writer({
        "kind": "step",
        "id": "send_done",
        "status": "done",
        "level": "success",
        "detail": report[:120],
        "title": "发送完成",
    })
    return {
        "send_payload": payload,
        "send_report": report,
        "trace": [{"agent": "send", "status": "done", "output": report[:200]}],
    }
