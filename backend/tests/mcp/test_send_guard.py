import json
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from app.mcp.context import mcp_call_context
from app.mcp.send_guard import SendGuardInterceptor


class _Req:
    name = "send_email"
    server_name = "gmail"
    args = {"to": [" allowed@example.com "], "subject": "S", "body": "B"}


class _AllowService:
    async def email_allowed(self, email):
        return email == "allowed@example.com"


@pytest.mark.asyncio
async def test_send_guard_blocks_unlisted_recipient(monkeypatch, tmp_path):
    class _Service:
        async def email_allowed(self, email):
            return False

    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _Service())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    called = {"handler": False}

    async def handler(request):
        called["handler"] = True
        return CallToolResult(content=[TextContent(type="text", text="sent")])

    result = await SendGuardInterceptor()(_Req(), handler)

    assert called["handler"] is False
    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert payload["error"] == "RECIPIENT_NOT_ALLOWED"
    assert not (tmp_path / "sends.log").exists()


@pytest.mark.asyncio
async def test_send_guard_allows_and_audits(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"message_id":"msg-1"}')],
            structuredContent={"message_id": "msg-1"},
        )

    async with mcp_call_context(user_id="user-1", session_id="sess-1"):
        result = await SendGuardInterceptor()(_Req(), handler)

    assert result.isError is False
    lines = (tmp_path / "sends.log").read_text(encoding="utf-8").splitlines()
    audit = json.loads(lines[0])
    assert audit["user_id"] == "user-1"
    assert audit["session_id"] == "sess-1"
    assert audit["recipient"] == ["allowed@example.com"]
    assert audit["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_send_guard_does_not_audit_when_handler_returns_error(monkeypatch, tmp_path):
    """handler 通过 isError=True 报错时，绝不能写 sends.log。"""
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"error":"AUTH_FAILED","channel":"gmail"}')],
            isError=True,
        )

    result = await SendGuardInterceptor()(_Req(), handler)

    assert result.isError is True
    assert not (tmp_path / "sends.log").exists()


@pytest.mark.asyncio
async def test_send_guard_treats_error_payload_as_failure(monkeypatch, tmp_path):
    """即使 handler 忘了设 isError，只要 payload 里有 error 或缺 message_id，也不写审计。"""
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        # 没有设 isError，但 payload 明确写了 error；send_guard 也应视为失败
        return CallToolResult(
            content=[TextContent(type="text", text='{"error":"GMAIL_API_ERROR","message":"500"}')],
        )

    result = await SendGuardInterceptor()(_Req(), handler)

    # send_guard 会把 result 改写为 isError=True 或至少不写审计
    assert not (tmp_path / "sends.log").exists()
    assert getattr(result, "isError", False) is True


@pytest.mark.asyncio
async def test_send_guard_does_not_audit_when_message_id_missing(monkeypatch, tmp_path):
    """handler 意外没带 message_id 也不能被当成功审计。"""
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    result = await SendGuardInterceptor()(_Req(), handler)

    assert not (tmp_path / "sends.log").exists()
    assert getattr(result, "isError", False) is True
