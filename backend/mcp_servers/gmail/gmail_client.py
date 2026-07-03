import base64
import os
from email.message import EmailMessage
from typing import Optional

from googleapiclient.errors import HttpError

from mcp_servers.gmail.auth import get_gmail_service


def _sender() -> str:
    sender = os.getenv("GMAIL_SENDER_EMAIL")
    if not sender:
        raise RuntimeError("GMAIL_SENDER_EMAIL is required")
    return sender.strip()


def build_raw_message(
    sender: str,
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
) -> str:
    # 说明：P0 不把 Bcc 头写入 MIME，避免密送地址随 raw 泄漏到收件人的邮件源。
    # Bcc 参数保留但当前不投递；如需真正密送 Bcc，请在 P1 补充 SMTP-level 分发。
    msg = EmailMessage()
    msg.set_content(body or "", subtype="plain", charset="utf-8")
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject or "来自企业知识库 Agent 的消息"
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")


async def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
) -> dict[str, str]:
    try:
        service = await get_gmail_service()
        raw = build_raw_message(_sender(), to, subject, body, cc=cc, bcc=bcc)
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"message_id": sent["id"]}
    except HttpError as error:
        return {
            "error": "GMAIL_API_ERROR",
            "code": str(getattr(error, "status_code", "")),
            "message": str(error),
        }
    except Exception as error:  # noqa: BLE001
        return {"error": "AUTH_FAILED", "channel": "gmail", "message": str(error)}
