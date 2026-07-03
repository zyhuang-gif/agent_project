"""Gmail 发送前置校验 + 成功审计的 MCP ToolCallInterceptor。

护栏原则（对应 spec §6）：
1. 只拦截 (server_name="gmail", tool="send_email")；其它 tool 直通 handler。
2. 在 handler 之前对 to/cc/bcc 每个地址查白名单，任一不通过就返回结构化
   RECIPIENT_NOT_ALLOWED，不真的调 Gmail API。
3. handler 成功后向 log/sends.log 追加一条 JSONL 审计记录。
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


def _message_id(result: CallToolResult) -> str | None:
    structured = getattr(result, "structuredContent", None) or {}
    if isinstance(structured, dict) and structured.get("message_id"):
        return str(structured["message_id"])
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", "")
        try:
            data = json.loads(text)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("message_id"):
            return str(data["message_id"])
    return None


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
        if not getattr(result, "isError", False):
            ctx = current_mcp_context()
            _audit_path().parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "user_id": ctx.get("user_id"),
                "session_id": ctx.get("session_id"),
                "channel": "gmail",
                "recipient": recipients,
                "subject_or_preview": (request.args.get("subject") or "")[:120],
                "message_id": _message_id(result),
            }
            with _audit_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result
