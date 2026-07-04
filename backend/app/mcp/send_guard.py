"""发送前置校验 + 成功审计的 MCP ToolCallInterceptor。

护栏原则（对应 spec §6、P2 白名单扩展、P3a 审计上库）：
1. 只拦 gmail send_email 与 feishu send_message_to_*/send_file_to_*；
   其它工具直通 handler。
2. 在 handler 之前做白名单校验：
   - gmail: to/cc/bcc 每个地址必须存在于 active contact.email；
   - feishu send_message_to_user / send_file_to_user: open_id 必须存在于
     active contact.feishu_open_id；
   - feishu send_message_to_group / send_file_to_group: chat_id 必须存在于
     active feishu_groups.chat_id。
   任一不通过 → 返回结构化 RECIPIENT_NOT_ALLOWED，不真的调外部 API。
3. handler 返回后只在真的成功（无 isError、无 error 字段、且拿到 message_id）
   时才委托 send_audit_service.record_send_success 落库；否则把 result 视为
   失败（必要时改写 isError=True），不写审计，防止认证/API 失败被误记为成功。
   文件类工具成功审计里额外附带 file_key。
"""
import json
from pathlib import Path
from typing import Awaitable, Callable, Optional

from mcp.types import CallToolResult, TextContent

from app.mcp.context import current_mcp_context
from app.services.contact_service import contact_service, normalize_email
from app.services.send_audit_service import send_audit_service


# ── 拦截目标 ────────────────────────────────────────────────────────────────
GMAIL_TOOLS: frozenset[str] = frozenset({"send_email"})
FEISHU_USER_TOOLS: frozenset[str] = frozenset({"send_message_to_user", "send_file_to_user"})
FEISHU_GROUP_TOOLS: frozenset[str] = frozenset({"send_message_to_group", "send_file_to_group"})
FEISHU_FILE_TOOLS: frozenset[str] = frozenset({"send_file_to_user", "send_file_to_group"})
FEISHU_TOOLS: frozenset[str] = FEISHU_USER_TOOLS | FEISHU_GROUP_TOOLS


def _error_result(payload: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        isError=True,
    )


def _all_gmail_recipients(args: dict) -> list[str]:
    recipients: list[str] = []
    for key in ("to", "cc", "bcc"):
        for email in args.get(key) or []:
            normalized = normalize_email(email)
            if normalized:
                recipients.append(normalized)
    return recipients


def _content_payloads(result: CallToolResult) -> list[dict]:
    payloads: list[dict] = []
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        payloads.append(structured)
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", "")
        try:
            data = json.loads(text)
        except Exception:
            continue
        if isinstance(data, dict):
            payloads.append(data)
    return payloads


def _extract_message_id(payloads: list[dict]) -> Optional[str]:
    for data in payloads:
        if data.get("message_id"):
            return str(data["message_id"])
    return None


def _extract_file_key(payloads: list[dict]) -> Optional[str]:
    for data in payloads:
        if data.get("file_key"):
            return str(data["file_key"])
    return None


def _has_error_marker(payloads: list[dict]) -> bool:
    return any("error" in data for data in payloads)


def _mark_error_or_fallback(result: CallToolResult, payloads: list[dict]) -> CallToolResult:
    """把非成功结果强制标为 isError=True；失败时降级为新的 error result。"""
    try:
        result.isError = True
        return result
    except Exception:
        return _error_result(
            payloads[0] if payloads else {"error": "SEND_FAILED"}
        )


def _feishu_preview(args: dict, tool_name: str) -> str:
    """审计里用的显示标题——文件类给文件名，消息类给正文前 120 字。"""
    if tool_name in FEISHU_FILE_TOOLS:
        raw = args.get("file_name") or ""
        if not raw:
            raw = Path(args.get("file_path") or "").name
        return raw[:120]
    return (args.get("content") or "")[:120]


class SendGuardInterceptor:
    async def __call__(
        self,
        request,
        handler: Callable[[object], Awaitable[CallToolResult]],
    ) -> CallToolResult:
        server = getattr(request, "server_name", "")
        name = getattr(request, "name", "")
        if server == "gmail" and name in GMAIL_TOOLS:
            return await self._guard_gmail(request, handler)
        if server == "feishu" and name in FEISHU_TOOLS:
            return await self._guard_feishu(request, handler, name)
        return await handler(request)

    async def _guard_gmail(
        self,
        request,
        handler: Callable[[object], Awaitable[CallToolResult]],
    ) -> CallToolResult:
        args = request.args or {}
        recipients = _all_gmail_recipients(args)
        for email in recipients:
            if not await contact_service.email_allowed(email):
                return _error_result({
                    "error": "RECIPIENT_NOT_ALLOWED",
                    "channel": "gmail",
                    "recipient": email,
                })

        result = await handler(request)

        payloads = _content_payloads(result)
        is_error = bool(getattr(result, "isError", False))
        has_error = _has_error_marker(payloads)
        message_id = _extract_message_id(payloads)

        if is_error or has_error or not message_id:
            if not is_error:
                return _mark_error_or_fallback(result, payloads)
            return result

        ctx = current_mcp_context()
        # Gmail 单次调用可发多个 to/cc/bcc → audit 单行，多收件人用逗号连接，
        # 便于 recipient LIKE 查询；request_payload_preview 里也留一份完整 args。
        await send_audit_service.record_send_success(
            channel="gmail",
            recipient_kind="email",
            recipient=",".join(recipients),
            subject=(args.get("subject") or "")[:255],
            message_id=message_id,
            tool_name=getattr(request, "name", ""),
            user_id=ctx.get("user_id"),
            session_id=ctx.get("session_id"),
            request_payload=args,
        )
        return result

    async def _guard_feishu(
        self,
        request,
        handler: Callable[[object], Awaitable[CallToolResult]],
        tool_name: str,
    ) -> CallToolResult:
        args = request.args or {}
        if tool_name in FEISHU_USER_TOOLS:
            recipient = (args.get("open_id") or "").strip()
            recipient_kind = "open_id"
            allowed = await contact_service.feishu_open_id_allowed(recipient) if recipient else False
        else:
            recipient = (args.get("chat_id") or "").strip()
            recipient_kind = "chat_id"
            allowed = await contact_service.chat_id_allowed(recipient) if recipient else False

        if not allowed:
            return _error_result({
                "error": "RECIPIENT_NOT_ALLOWED",
                "channel": "feishu",
                "recipient": recipient,
                "kind": recipient_kind,
            })

        result = await handler(request)

        payloads = _content_payloads(result)
        is_error = bool(getattr(result, "isError", False))
        has_error = _has_error_marker(payloads)
        message_id = _extract_message_id(payloads)

        if is_error or has_error or not message_id:
            if not is_error:
                return _mark_error_or_fallback(result, payloads)
            return result

        ctx = current_mcp_context()
        preview = _feishu_preview(args, tool_name)
        file_key = _extract_file_key(payloads) if tool_name in FEISHU_FILE_TOOLS else None
        await send_audit_service.record_send_success(
            channel="feishu",
            recipient_kind=recipient_kind,
            recipient=recipient,
            subject=preview or None,
            artifact_name=(args.get("file_name") or None) if tool_name in FEISHU_FILE_TOOLS else None,
            artifact_path=(args.get("file_path") or None) if tool_name in FEISHU_FILE_TOOLS else None,
            message_id=message_id,
            file_key=file_key,
            tool_name=tool_name,
            user_id=ctx.get("user_id"),
            session_id=ctx.get("session_id"),
            request_payload=args,
        )
        return result
