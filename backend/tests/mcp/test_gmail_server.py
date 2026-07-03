import base64
from email import message_from_bytes
from email.header import decode_header, make_header

import pytest

from mcp_servers.gmail.gmail_client import build_raw_message, send_email


def _hv(msg, key):
    return str(make_header(decode_header(msg[key])))


def test_build_raw_message_sets_headers_and_body():
    raw = build_raw_message(
        sender="sandbox@example.com",
        to=["a@example.com", "b@example.com"],
        subject="测试主题",
        body="正文内容",
        cc=["c@example.com"],
        bcc=["d@example.com"],
    )
    msg = message_from_bytes(base64.urlsafe_b64decode(raw.encode("utf-8")))

    assert msg["From"] == "sandbox@example.com"
    assert msg["To"] == "a@example.com, b@example.com"
    assert msg["Cc"] == "c@example.com"
    assert _hv(msg, "Subject") == "测试主题"
    payload = msg.get_payload(decode=True).decode("utf-8")
    assert payload.rstrip("\r\n") == "正文内容"
    assert "d@example.com" not in msg.as_string()


@pytest.mark.asyncio
async def test_send_email_calls_gmail_api(monkeypatch):
    calls = {}

    class _Send:
        def execute(self):
            return {"id": "msg-123"}

    class _Messages:
        def send(self, userId, body):
            calls["userId"] = userId
            calls["body"] = body
            return _Send()

    class _Users:
        def messages(self):
            return _Messages()

    class _Service:
        def users(self):
            return _Users()

    async def _fake_service():
        return _Service()

    import mcp_servers.gmail.gmail_client as gm
    monkeypatch.setenv("GMAIL_SENDER_EMAIL", "sandbox@example.com")
    monkeypatch.setattr(gm, "get_gmail_service", _fake_service)

    result = await send_email(["a@example.com"], "subject", "body")

    assert result == {"message_id": "msg-123"}
    assert calls["userId"] == "me"
    assert "raw" in calls["body"]
