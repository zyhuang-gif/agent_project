"""Gmail 发送前置校验 + 成功审计的 MCP ToolCallInterceptor。

护栏原则（对应 spec §6）：
1. 只拦截 (server_name="gmail", tool="send_email")；其它 tool 直通 handler。
2. 在 handler 之前对 to/cc/bcc 每个地址查白名单，任一不通过就返回结构化
   RECIPIENT_NOT_ALLOWED，不真的调 Gmail API。
3. handler 返回后只在真的成功（无 isError、无 error 字段、且拿到 message_id）时
   才写 log/sends.log；否则把 result 视为失败（必要时改写为 isError=True），
   不写审计，防止认证/API 失败被误记为成功。
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from mcp.types import CallToolResult, TextContent

from app.mcp.context import current_mcp_context
from app.services.contact_service import contact_service, normalize_email


def _audit_path() -> Path:
    return Path(__file__).resolve().parents[3] / "log" / "sends.log"


def _error_result(payload: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        isError=True,
    )


def _all_recipients(args: dict) -> list[str]:
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


def _extract_message_id(payloads: list[dict]) -> str | None:
    for data in payloads:
        if data.get("message_id"):
            return str(data["message_id"])
    return None


def _has_error_marker(payloads: list[dict]) -> bool:
    return any("error" in data for data in payloads)


class SendGuardInterceptor:
    async def __call__(
        self,
        request,
        handler: Callable[[object], Awaitable[CallToolResult]],
    ) -> CallToolResult:
        if request.server_name != "gmail" or request.name != "send_email":
            return await handler(request)

        recipients = _all_recipients(request.args)
        for email in recipients:
            if not await contact_service.email_allowed(email):
                return _error_result({
                    "error": "RECIPIENT_NOT_ALLOWED",
                    "channel": "gmail",
                    "recipient": email,
                })

        result = await handler(request)

        # 只要 handler 显式报错、payload 里带 error、或者根本没拿到 message_id，
        # 都视为失败：不写审计，同时把结果强制标为 isError=True 让上游看得见。
        payloads = _content_payloads(result)
        is_error = bool(getattr(result, "isError", False))
        has_error = _has_error_marker(payloads)
        message_id = _extract_message_id(payloads)

        if is_error or has_error or not message_id:
            if not is_error:
                try:
                    result.isError = True
                except Exception:
                    return _error_result(
                        payloads[0] if payloads else {"error": "GMAIL_SEND_FAILED", "channel": "gmail"}
                    )
            return result

        ctx = current_mcp_context()
        _audit_path().parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user_id": ctx.get("user_id"),
            "session_id": ctx.get("session_id"),
            "channel": "gmail",
            "recipient": recipients,
            "subject_or_preview": (request.args.get("subject") or "")[:120],
            "message_id": message_id,
        }
        with _audit_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result
