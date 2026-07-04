import base64
from email import message_from_bytes
from email.header import decode_header, make_header
from pathlib import Path

import pytest

from mcp_servers.gmail.gmail_client import GmailSendError, build_raw_message, send_email


def _hv(msg, key):
    value = msg[key]
    if value is None:
        return None
    return str(make_header(decode_header(value)))


def test_build_raw_message_sets_headers_and_body():
    raw = build_raw_message(
        sender="sandbox@example.com",
        to=["a@example.com", "b@example.com"],
        subject="测试主题",
        body="正文内容",
    )
    msg = message_from_bytes(base64.urlsafe_b64decode(raw.encode("utf-8")))

    assert msg["From"] == "sandbox@example.com"
    assert msg["To"] == "a@example.com, b@example.com"
    # P0 不再暴露 cc/bcc，MIME 里也不应出现
    assert msg["Cc"] is None
    assert msg["Bcc"] is None
    assert _hv(msg, "Subject") == "测试主题"
    payload = msg.get_payload(decode=True).decode("utf-8")
    assert payload.rstrip("\r\n") == "正文内容"


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


@pytest.mark.asyncio
async def test_send_email_auth_failure_raises(monkeypatch):
    """凭据/服务初始化失败必须抛 GmailSendError，避免 send_guard 把假成功写审计。"""
    import mcp_servers.gmail.gmail_client as gm

    async def _boom():
        raise RuntimeError("token file missing")

    monkeypatch.setenv("GMAIL_SENDER_EMAIL", "sandbox@example.com")
    monkeypatch.setattr(gm, "get_gmail_service", _boom)

    with pytest.raises(GmailSendError) as ex_info:
        await send_email(["a@example.com"], "s", "b")

    assert ex_info.value.error == "AUTH_FAILED"
    assert ex_info.value.to_dict()["channel"] == "gmail"


@pytest.mark.asyncio
async def test_send_email_missing_message_id_raises(monkeypatch):
    """Gmail API 拿到空 message id 也算失败，不允许静默返回。"""
    class _Send:
        def execute(self):
            return {}

    class _Messages:
        def send(self, userId, body):
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

    with pytest.raises(GmailSendError) as ex_info:
        await send_email(["a@example.com"], "s", "b")

    assert ex_info.value.error == "GMAIL_API_ERROR"


# ── P1 附件：build_raw_message / send_email 附件校验 ─────────────────────────


def test_build_raw_message_with_attachment(tmp_path, monkeypatch):
    """正文 + Markdown 附件同时出现在 MIME 里。"""
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    # 使用真正的 artifact 生成器 —— 附件必须来自受控目录，保持防线一致。
    from app.mcp.artifacts import write_markdown_artifact

    art = write_markdown_artifact(
        source_answer="# 内容",
        base_name="报告",
        root=tmp_path,
    )
    raw = build_raw_message(
        sender="sandbox@example.com",
        to=["a@example.com"],
        subject="主题",
        body="body",
        attachments=[art.path],
    )
    msg = message_from_bytes(base64.urlsafe_b64decode(raw.encode("utf-8")))
    assert msg.is_multipart()
    parts = list(msg.walk())
    text_parts = [p for p in parts if p.get_content_type() == "text/plain"]
    assert text_parts and text_parts[0].get_payload(decode=True).decode("utf-8").rstrip() == "body"
    md_parts = [p for p in parts if p.get_content_type() == "text/markdown"]
    assert md_parts, "附件必须以 text/markdown 出现"
    filename = md_parts[0].get_filename()
    assert filename and filename.endswith(".md")
    assert md_parts[0].get_payload(decode=True).decode("utf-8") == "# 内容"


def test_build_raw_message_rejects_missing_attachment(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    with pytest.raises(GmailSendError) as ex:
        build_raw_message(
            sender="sandbox@example.com",
            to=["a@example.com"],
            subject="s",
            body="b",
            attachments=[str(tmp_path / "send" / "does-not-exist.md")],
        )
    assert ex.value.error == "ATTACHMENT_INVALID"


def test_build_raw_message_rejects_attachment_outside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "runtime" / "artifacts"))
    outside = tmp_path / "outside.md"
    outside.write_text("hello", encoding="utf-8")
    with pytest.raises(GmailSendError) as ex:
        build_raw_message(
            sender="sandbox@example.com",
            to=["a@example.com"],
            subject="s",
            body="b",
            attachments=[str(outside)],
        )
    assert ex.value.error == "ATTACHMENT_INVALID"


def test_build_raw_message_rejects_disallowed_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    send_dir = tmp_path / "send"
    send_dir.mkdir(parents=True, exist_ok=True)
    bad = send_dir / "bad.exe"
    bad.write_bytes(b"x")
    with pytest.raises(GmailSendError) as ex:
        build_raw_message(
            sender="sandbox@example.com",
            to=["a@example.com"],
            subject="s",
            body="b",
            attachments=[str(bad)],
        )
    assert ex.value.error == "ATTACHMENT_INVALID"


def test_build_raw_message_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    send_dir = tmp_path / "send"
    send_dir.mkdir(parents=True, exist_ok=True)
    big = send_dir / "big.md"
    big.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(GmailSendError) as ex:
        build_raw_message(
            sender="sandbox@example.com",
            to=["a@example.com"],
            subject="s",
            body="b",
            attachments=[str(big)],
        )
    assert ex.value.error == "ATTACHMENT_INVALID"


@pytest.mark.asyncio
async def test_send_email_with_invalid_attachment_does_not_call_gmail(monkeypatch, tmp_path):
    """附件校验失败必须在真实 Gmail API 调用前失败。"""
    from mcp_servers.gmail import gmail_client as gm

    called = {"n": 0}

    class _Service:
        def users(self):
            called["n"] += 1
            raise AssertionError("Gmail API should NOT be called on invalid attachment")

    async def _fake_service():
        return _Service()

    monkeypatch.setenv("GMAIL_SENDER_EMAIL", "sandbox@example.com")
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(gm, "get_gmail_service", _fake_service)

    with pytest.raises(GmailSendError) as ex:
        await send_email(
            ["a@example.com"],
            "s",
            "b",
            attachments=[str(tmp_path / "send" / "missing.md")],
        )
    assert ex.value.error == "ATTACHMENT_INVALID"
    assert called["n"] == 0
