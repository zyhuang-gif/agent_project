import base64
import os
from email.message import EmailMessage

from googleapiclient.errors import HttpError

from mcp_servers.gmail.auth import get_gmail_service


class GmailSendError(RuntimeError):
    """Gmail 发送失败（认证 or API）——抛出后由 MCP 层转成 isError CallToolResult，
    确保 send_guard 不会把失败当成功审计。"""

    def __init__(self, error: str, message: str, code: str = ""):
        super().__init__(f"{error}: {message}")
        self.error = error
        self.message = message
        self.code = code

    def to_dict(self) -> dict[str, str]:
        payload = {"error": self.error, "channel": "gmail", "message": self.message}
        if self.code:
            payload["code"] = self.code
        return payload


def _sender() -> str:
    sender = os.getenv("GMAIL_SENDER_EMAIL")
    if not sender:
        raise RuntimeError("GMAIL_SENDER_EMAIL is required")
    return sender.strip()


def build_raw_message(sender: str, to: list[str], subject: str, body: str) -> str:
    # P0 只支持 to 主收件人；cc/bcc 由 P1/P2 附件与飞书通道时再考虑，避免 "参数接受但不投递"
    # 造成假成功。
    msg = EmailMessage()
    msg.set_content(body or "", subtype="plain", charset="utf-8")
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject or "来自企业知识库 Agent 的消息"
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")


async def send_email(to: list[str], subject: str, body: str) -> dict[str, str]:
    """真调 Gmail API。失败一律抛 GmailSendError，让 MCP 层把它变成 isError 结果。"""
    try:
        service = await get_gmail_service()
        raw = build_raw_message(_sender(), to, subject, body)
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except HttpError as error:
        raise GmailSendError(
            "GMAIL_API_ERROR",
            str(error),
            code=str(getattr(error, "status_code", "")),
        ) from error
    except GmailSendError:
        raise
    except Exception as error:  # noqa: BLE001
        raise GmailSendError("AUTH_FAILED", str(error)) from error

    message_id = sent.get("id")
    if not message_id:
        raise GmailSendError("GMAIL_API_ERROR", "Gmail API returned no message id")
    return {"message_id": message_id}
