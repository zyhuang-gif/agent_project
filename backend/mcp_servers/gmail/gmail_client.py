import base64
import mimetypes
import os
from email.message import EmailMessage
from pathlib import Path

from googleapiclient.errors import HttpError

from mcp_servers.gmail.auth import get_gmail_service


class GmailSendError(RuntimeError):
    """Gmail 发送失败（认证 / API / 附件校验）——抛出后由 MCP 层转成 isError CallToolResult，
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


# 附件安全边界（P1）：Markdown-only，单文件 5 MiB，必须落在 artifact_root() 里
_ALLOWED_ATTACHMENT_EXT = {".md"}
_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024


def _sender() -> str:
    sender = os.getenv("GMAIL_SENDER_EMAIL")
    if not sender:
        raise RuntimeError("GMAIL_SENDER_EMAIL is required")
    return sender.strip()


def _artifact_root() -> Path:
    """与 app.mcp.artifacts.artifact_root 保持一致：优先 ARTIFACT_ROOT，
    否则回退到 backend/runtime/artifacts。gmail_client.py → parents[2] = backend/。"""
    override = os.getenv("ARTIFACT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "runtime" / "artifacts").resolve()


def _validate_attachment(raw_path: str) -> Path:
    if not raw_path:
        raise GmailSendError("ATTACHMENT_INVALID", "empty attachment path")
    path = Path(raw_path).expanduser().resolve()
    root = _artifact_root()
    if not path.exists() or not path.is_file():
        raise GmailSendError("ATTACHMENT_INVALID", f"attachment not found: {raw_path}")
    if not path.is_relative_to(root):
        raise GmailSendError(
            "ATTACHMENT_INVALID", f"attachment outside artifact root: {raw_path}"
        )
    if path.suffix.lower() not in _ALLOWED_ATTACHMENT_EXT:
        raise GmailSendError(
            "ATTACHMENT_INVALID", f"unsupported attachment extension: {path.suffix}"
        )
    if path.stat().st_size > _MAX_ATTACHMENT_BYTES:
        raise GmailSendError(
            "ATTACHMENT_INVALID",
            f"attachment too large: {path.stat().st_size} bytes",
        )
    return path


def build_raw_message(
    sender: str,
    to: list[str],
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> str:
    # P0 只支持 to 主收件人；cc/bcc 由 P1/P2 附件与飞书通道时再考虑，避免 "参数接受但不投递"
    # 造成假成功。P1 附件仍走 to，只在 body 之外挂 Markdown。
    msg = EmailMessage()
    msg.set_content(body or "", subtype="plain", charset="utf-8")
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject or "来自企业知识库 Agent 的消息"

    for raw_path in attachments or []:
        path = _validate_attachment(raw_path)
        data = path.read_bytes()
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed:
            maintype, subtype = guessed.split("/", 1)
        else:
            # .md 在部分环境里没被 mimetypes 认出；强制回退到 text/markdown。
            maintype, subtype = "text", "markdown"
        msg.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")


async def send_email(
    to: list[str],
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> dict[str, str]:
    """真调 Gmail API。失败一律抛 GmailSendError，让 MCP 层把它变成 isError 结果。

    附件校验（缺失 / 越权路径 / 扩展名 / 大小）必须在真正调用 Gmail API 之前失败，
    避免把 "本地校验错" 当成 "远端 API 错" 计费和审计。
    """
    for raw_path in attachments or []:
        _validate_attachment(raw_path)

    try:
        service = await get_gmail_service()
        raw = build_raw_message(_sender(), to, subject, body, attachments=attachments)
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
