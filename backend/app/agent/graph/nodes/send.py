"""send 节点：把 finalize 生成的 final_answer 通过多通道 MCP 工具发出。

流程（spec §3.2、P2 扩展）：
  1. 无 send_intent 直接返回空 dict。
  2. resolve_send_targets 做确定性 preflight：任何目标不明确 / 通道不匹配 →
     短路，不调任何工具（不做部分发送）。
  3. build_send_payload 生成专用 subject/body_text。
  4. 生成 Markdown artifact（P1 附件），失败短路。
  5. 构造串行 send plan：Gmail 一次全部收件人（沿用 P1 附件）；Feishu 每 target
     两次调用——先文本、再文件（引用 artifact）。
  6. 依次执行；每步独立 outcome。任一失败即整体 trace failed，send_report 逐条
     列出成功/失败。绝不因中间失败而"假成功"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from app.agent.graph._stream import safe_get_stream_writer
from app.agent.graph.state import AgentState
from app.mcp.artifacts import Artifact, ArtifactError, write_markdown_artifact
from app.mcp.client import mcp_manager
from app.mcp.context import mcp_call_context
from app.mcp.payload import SendPayload, build_send_payload
from app.mcp.preflight import PreflightResult, ResolvedRecipient, resolve_send_targets


# ── Send plan item ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SendItem:
    channel: str          # "gmail" | "feishu"
    tool_name: str        # 已加前缀的 LangChain 工具名，如 "gmail_send_email"
    target_label: str     # 人名 / 群名列表（"张三"、"张三、李四"、"运营组"）
    kind: str             # "email" | "feishu_text" | "feishu_file"
    args: dict[str, Any]


@dataclass
class _Outcome:
    item: _SendItem
    ok: bool
    message_id: Optional[str] = None
    file_key: Optional[str] = None
    error: Optional[str] = None
    error_payload: Optional[dict] = None


# ── 结果解析（复用 P1 里的 helpers 逻辑，含扩展 file_key） ─────────────────


def _extract_dict_payloads(result: Any) -> list[dict]:
    """把 tool.ainvoke 的返回值展平成一组结构化 dict payload。

    支持形态：dict、JSON 字符串、list content blocks、(content, artifact) tuple、
    带 .content 属性的 ToolMessage。
    """
    payloads: list[dict] = []

    def _walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                _walk(node["text"])
                return
            payloads.append(node)
            return
        if isinstance(node, tuple):
            for item in node:
                _walk(item)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
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
                return
            _walk(parsed)
            return

    _walk(result)
    return payloads


def _extract_tool_error(result: Any) -> Optional[dict]:
    payloads = _extract_dict_payloads(result)
    for data in payloads:
        if data.get("error"):
            return data
    if not any(data.get("message_id") for data in payloads):
        return {
            "error": "SEND_FAILED",
            "raw": payloads or repr(result)[:200],
        }
    return None


def _message_id(result: Any) -> Optional[str]:
    for data in _extract_dict_payloads(result):
        if data.get("message_id"):
            return str(data["message_id"])
    return None


def _file_key(result: Any) -> Optional[str]:
    for data in _extract_dict_payloads(result):
        if data.get("file_key"):
            return str(data["file_key"])
    return None


# ── send plan 构造 ────────────────────────────────────────────────────────


def _build_send_plan(
    preflight: PreflightResult,
    payload: SendPayload,
    artifact: Artifact,
) -> list[_SendItem]:
    items: list[_SendItem] = []
    channels = preflight.channels or ["gmail"]

    if "gmail" in channels and preflight.gmail_to:
        gmail_label = "、".join(
            r.name for r in preflight.recipients if r.email
        ) or "已配置联系人"
        items.append(_SendItem(
            channel="gmail",
            tool_name="gmail_send_email",
            target_label=gmail_label,
            kind="email",
            args={
                "to": list(preflight.gmail_to),
                "subject": payload.subject,
                "body": payload.body_text,
                "attachments": [artifact.path],
            },
        ))

    if "feishu" in channels:
        for user in preflight.feishu_users:
            items.append(_SendItem(
                channel="feishu",
                tool_name="feishu_send_message_to_user",
                target_label=user.name,
                kind="feishu_text",
                args={
                    "open_id": user.feishu_open_id,
                    "content": payload.body_text,
                    "msg_type": "text",
                },
            ))
            items.append(_SendItem(
                channel="feishu",
                tool_name="feishu_send_file_to_user",
                target_label=user.name,
                kind="feishu_file",
                args={
                    "open_id": user.feishu_open_id,
                    "file_path": artifact.path,
                    "file_name": artifact.name,
                },
            ))
        for group in preflight.feishu_groups:
            items.append(_SendItem(
                channel="feishu",
                tool_name="feishu_send_message_to_group",
                target_label=group.name,
                kind="feishu_text",
                args={
                    "chat_id": group.chat_id,
                    "content": payload.body_text,
                    "msg_type": "text",
                },
            ))
            items.append(_SendItem(
                channel="feishu",
                tool_name="feishu_send_file_to_group",
                target_label=group.name,
                kind="feishu_file",
                args={
                    "chat_id": group.chat_id,
                    "file_path": artifact.path,
                    "file_name": artifact.name,
                },
            ))
    return items


async def _run_send_item(item: _SendItem) -> _Outcome:
    tools = mcp_manager.get_tools(names=[item.tool_name])
    if not tools:
        return _Outcome(
            item=item,
            ok=False,
            error=f"{item.tool_name} 未加载",
            error_payload={"error": "TOOL_UNAVAILABLE", "tool": item.tool_name},
        )
    tool = tools[0]
    try:
        result = await tool.ainvoke(item.args)
    except Exception as error:  # noqa: BLE001
        return _Outcome(item=item, ok=False, error=str(error)[:200])
    error_payload = _extract_tool_error(result)
    if error_payload is not None:
        detail = (
            error_payload.get("message")
            or error_payload.get("error")
            or "发送失败"
        )
        return _Outcome(
            item=item,
            ok=False,
            error=str(detail)[:200],
            error_payload=error_payload,
        )
    return _Outcome(
        item=item,
        ok=True,
        message_id=_message_id(result),
        file_key=_file_key(result) if item.kind == "feishu_file" else None,
    )


# ── send_report / trace 组装 ──────────────────────────────────────────────


def _channel_label(channel: str) -> str:
    return "Gmail" if channel == "gmail" else "飞书"


def _format_success_line(outcome: _Outcome, artifact: Artifact) -> str:
    item = outcome.item
    label = _channel_label(item.channel)
    if item.kind == "email":
        return (
            f"{label} 已发送给{item.target_label}，附件：{artifact.name}。"
            f"message_id={outcome.message_id}"
        )
    if item.kind == "feishu_text":
        prefix = "飞书私聊" if item.tool_name.endswith("_to_user") else "飞书群"
        return f"{prefix} 已发送给{item.target_label}。message_id={outcome.message_id}"
    if item.kind == "feishu_file":
        prefix = "飞书私聊" if item.tool_name.endswith("_to_user") else "飞书群"
        return (
            f"{prefix} 附件已发送给{item.target_label}（{artifact.name}）。"
            f"message_id={outcome.message_id}, file_key={outcome.file_key}"
        )
    return f"{label} 已发送给{item.target_label}。message_id={outcome.message_id}"


def _format_failure_line(outcome: _Outcome) -> str:
    item = outcome.item
    label = _channel_label(item.channel)
    detail = outcome.error or "未知错误"
    if item.kind == "feishu_text":
        prefix = "飞书私聊" if item.tool_name.endswith("_to_user") else "飞书群"
        return f"{prefix} 发送失败（{item.target_label}）：{detail}"
    if item.kind == "feishu_file":
        prefix = "飞书私聊" if item.tool_name.endswith("_to_user") else "飞书群"
        return f"{prefix} 附件发送失败（{item.target_label}）：{detail}"
    return f"{label} 发送失败（{item.target_label}）：{detail}"


def _format_send_report(outcomes: list[_Outcome], artifact: Artifact) -> str:
    if not outcomes:
        return "未执行任何发送。"
    lines: list[str] = []
    for o in outcomes:
        if o.ok:
            lines.append(_format_success_line(o, artifact))
        else:
            lines.append(_format_failure_line(o))
    return "\n".join(lines)


def _format_trace_output(outcomes: list[_Outcome]) -> str:
    parts: list[str] = []
    for o in outcomes:
        if o.ok:
            parts.append(f"{o.item.channel}:{o.item.kind}:{o.message_id or ''}")
        elif o.error_payload:
            parts.append(json.dumps(o.error_payload, ensure_ascii=False))
        else:
            parts.append(f"{o.item.channel}:{o.item.kind}:err:{o.error or ''}")
    return "|".join(parts)[:200]


# ── send_node 主入口 ─────────────────────────────────────────────────────


def _artifact_state_update(artifact: Artifact | None) -> dict:
    if artifact is None:
        return {}
    return {
        "artifact_path": artifact.path,
        "artifact_name": artifact.name,
        "artifact_mime": artifact.mime,
    }


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

    # P1：真正调外部工具前生成 Markdown artifact；校验/写盘异常立刻短路。
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

    plan_items = _build_send_plan(preflight, payload, artifact)
    if not plan_items:
        # preflight 通过但没有可执行项——比如 gmail 通道命中但 gmail_to 为空。
        # 保守起见按未发送处理。
        report = "发送失败：没有可执行的发送项。"
        writer({
            "kind": "step",
            "id": "send_done",
            "status": "done",
            "level": "warning",
            "detail": report,
            "title": "发送失败",
        })
        return {
            "send_payload": payload,
            "send_report": report,
            **_artifact_state_update(artifact),
            "trace": [
                {"agent": "send", "status": "failed", "output": "no_send_items"}
            ],
        }

    identity = state.get("identity")
    user_id = getattr(identity, "user_id", None)

    outcomes: list[_Outcome] = []
    async with mcp_call_context(user_id=user_id, session_id=state.get("session_id")):
        for item in plan_items:
            outcomes.append(await _run_send_item(item))

    all_ok = all(o.ok for o in outcomes)
    report = _format_send_report(outcomes, artifact)
    trace_output = _format_trace_output(outcomes)

    writer({
        "kind": "step",
        "id": "send_done",
        "status": "done",
        "level": "success" if all_ok else "warning",
        "detail": report[:120],
        "title": "发送完成" if all_ok else "发送失败",
    })

    return {
        "send_payload": payload,
        "send_report": report,
        **_artifact_state_update(artifact),
        "trace": [
            {
                "agent": "send",
                "status": "done" if all_ok else "failed",
                "output": trace_output,
            }
        ],
    }
