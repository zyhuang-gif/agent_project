"""send 节点：把 finalize 生成的 final_answer 通过 Gmail MCP 工具发出。

流程（spec §3.2）：
  1. 无 send_intent 直接返回空 dict（等价于跳过）。
  2. resolve_send_targets 做确定性 preflight：收件人/通道任何不确定 → 短路，
     返回 send_report + trace，不调 Gmail 工具。
  3. build_send_payload 生成专用 subject/body，不直接把 final_answer 塞进 Gmail。
  4. 后端确定性地调用 gmail_send_email 一次；不再走 create_react_agent 让 LLM 决定，
     确保 P0 “一次 send_node 最多一次真实 gmail_send_email 调用”。
"""
import json

from app.agent.graph._stream import safe_get_stream_writer
from app.agent.graph.state import AgentState
from app.mcp.client import mcp_manager
from app.mcp.context import mcp_call_context
from app.mcp.payload import build_send_payload
from app.mcp.preflight import resolve_send_targets


def _extract_tool_error(result) -> dict | None:
    """从 tool.ainvoke 返回结构里抽出错误 payload；没有错则返回 None。"""
    # LangChain BaseTool.ainvoke 通常返回 str/dict；有些实现会返回 tuple(content, artifact)。
    payload = result
    if isinstance(result, tuple) and len(result) >= 1:
        payload = result[0]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {"raw": payload}
    if isinstance(payload, dict):
        if payload.get("error"):
            return payload
        if not payload.get("message_id"):
            return {"error": "GMAIL_SEND_FAILED", "channel": "gmail", "raw": payload}
    return None


def _message_id(result) -> str | None:
    payload = result
    if isinstance(result, tuple) and len(result) >= 1:
        payload = result[0]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    if isinstance(payload, dict):
        return payload.get("message_id")
    return None


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

    gmail_tool = tools[0]
    tool_args = {
        "to": list(preflight.gmail_to),
        "subject": payload.subject,
        "body": payload.body_text,
    }

    identity = state.get("identity")
    user_id = getattr(identity, "user_id", None)
    async with mcp_call_context(user_id=user_id, session_id=state.get("session_id")):
        try:
            result = await gmail_tool.ainvoke(tool_args)
        except Exception as error:  # noqa: BLE001
            report = f"发送失败：{error}"
            writer({
                "kind": "step",
                "id": "send_done",
                "status": "done",
                "level": "warning",
                "detail": report[:120],
                "title": "发送失败",
            })
            return {
                "send_payload": payload,
                "send_report": report,
                "trace": [
                    {"agent": "send", "status": "failed", "output": str(error)[:200]}
                ],
            }

    error_payload = _extract_tool_error(result)
    if error_payload is not None:
        detail = error_payload.get("message") or error_payload.get("error") or "Gmail 发送失败"
        report = f"发送失败：{detail}"
        writer({
            "kind": "step",
            "id": "send_done",
            "status": "done",
            "level": "warning",
            "detail": report[:120],
            "title": "发送失败",
        })
        return {
            "send_payload": payload,
            "send_report": report,
            "trace": [
                {"agent": "send", "status": "failed",
                 "output": json.dumps(error_payload, ensure_ascii=False)[:200]}
            ],
        }

    message_id = _message_id(result)
    recipient_summary = "、".join(r.name for r in preflight.recipients) or "已配置联系人"
    report = f"已通过 Gmail 发送给{recipient_summary}。message_id={message_id}"
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
