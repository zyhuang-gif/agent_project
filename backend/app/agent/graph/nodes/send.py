"""send 节点：把 finalize 生成的 final_answer 通过 Gmail MCP 工具发出。

流程（spec §3.2）：
  1. 无 send_intent 直接返回空 dict（等价于跳过）。
  2. resolve_send_targets 做确定性 preflight：收件人/通道任何不确定 → 短路，
     返回 send_report + trace，不调 Gmail 工具。
  3. build_send_payload 生成专用 subject/body，不直接把 final_answer 塞进 Gmail。
  4. 后端确定性地调用 gmail_send_email 一次；不再走 create_react_agent 让 LLM 决定，
     确保 P0 "一次 send_node 最多一次真实 gmail_send_email 调用"。
"""
import json
from typing import Any

from app.agent.graph._stream import safe_get_stream_writer
from app.agent.graph.state import AgentState
from app.mcp.artifacts import ArtifactError, write_markdown_artifact
from app.mcp.client import mcp_manager
from app.mcp.context import mcp_call_context
from app.mcp.payload import build_send_payload
from app.mcp.preflight import resolve_send_targets


def _extract_dict_payloads(result: Any) -> list[dict]:
    """把 tool.ainvoke 的返回值 **展平** 成一组结构化 dict payload。

    langchain-mcp-adapters 的 StructuredTool(response_format="content_and_artifact")
    真实返回是 list[dict]，形如：
        [{"type":"text","text":"{\"message_id\":\"...\"}"}]
    错误也是同样结构：
        [{"type":"text","text":"{\"error\":\"RECIPIENT_NOT_ALLOWED\",...}"}]
    另外 StructuredTool 也可能返回 (content, artifact) tuple，或直接 dict / JSON string。
    这里统一处理，让 error / message_id 的判定不受返回形态影响。
    """
    payloads: list[dict] = []

    def _walk(node: Any) -> None:
        if node is None:
            return
        # dict 直接算一份 payload
        if isinstance(node, dict):
            # MCP content block：{"type":"text","text":"...json..."}
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                _walk(node["text"])
                return
            payloads.append(node)
            return
        # (content, artifact)：两个位置都要看
        if isinstance(node, tuple):
            for item in node:
                _walk(item)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        # ToolMessage / 其它带 content 属性的对象
        content_attr = getattr(node, "content", None)
        if content_attr is not None and content_attr is not node:
            _walk(content_attr)
            return
        if isinstance(node, str):
            text = node.strip()
            if not text:
                return
            try:
                parsed = json.loads(text)
            except Exception:
                # 不是 JSON，就不当结构化 payload 处理——不视为成功也不视为失败。
                return
            _walk(parsed)
            return
        # 其它类型（数字、bool 等）忽略

    _walk(result)
    return payloads


def _extract_tool_error(result: Any) -> dict | None:
    """从 tool.ainvoke 返回结构里抽出错误 payload；没有错则返回 None。

    判定规则：
      - 任一 payload 含 "error" → 该 payload 就是错误
      - 找不到含 message_id 的 payload → 视为失败（避免静默假成功）
      - 有 message_id 且无 error → 返回 None（真正成功）
    """
    payloads = _extract_dict_payloads(result)
    for data in payloads:
        if data.get("error"):
            return data
    if not any(data.get("message_id") for data in payloads):
        return {"error": "GMAIL_SEND_FAILED", "channel": "gmail",
                "raw": payloads or repr(result)[:200]}
    return None


def _message_id(result: Any) -> str | None:
    for data in _extract_dict_payloads(result):
        if data.get("message_id"):
            return str(data["message_id"])
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

    # P1：在真正调 Gmail 之前生成 Markdown artifact。任何校验/写盘异常都必须
    # short-circuit——绝不能带着"半成品附件"去调 Gmail。
    try:
        artifact = write_markdown_artifact(
            source_answer=state.get("final_answer") or "",
            base_name=payload.subject or "artifact",
        )
    except ArtifactError as err:
        report = f"发送失败：{err}"
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
                {"agent": "send", "status": "failed", "output": str(err)[:200]}
            ],
        }

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
            "artifact_path": artifact.path,
            "artifact_name": artifact.name,
            "artifact_mime": artifact.mime,
            "trace": [
                {"agent": "send", "status": "skipped", "output": "gmail_tool_unavailable"}
            ],
        }

    gmail_tool = tools[0]
    tool_args = {
        "to": list(preflight.gmail_to),
        "subject": payload.subject,
        "body": payload.body_text,
        "attachments": [artifact.path],
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
                "artifact_path": artifact.path,
                "artifact_name": artifact.name,
                "artifact_mime": artifact.mime,
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
            "artifact_path": artifact.path,
            "artifact_name": artifact.name,
            "artifact_mime": artifact.mime,
            "trace": [
                {"agent": "send", "status": "failed",
                 "output": json.dumps(error_payload, ensure_ascii=False)[:200]}
            ],
        }

    message_id = _message_id(result)
    recipient_summary = "、".join(r.name for r in preflight.recipients) or "已配置联系人"
    report = (
        f"已通过 Gmail 发送给{recipient_summary}，附件：{artifact.name}。"
        f"message_id={message_id}"
    )
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
        "artifact_path": artifact.path,
        "artifact_name": artifact.name,
        "artifact_mime": artifact.mime,
        "trace": [{"agent": "send", "status": "done", "output": report[:200]}],
    }
