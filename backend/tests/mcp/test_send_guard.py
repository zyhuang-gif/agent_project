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


@pytest.mark.asyncio
async def test_send_guard_allows_and_audits(monkeypatch, tmp_path):
    class _Service:
        async def email_allowed(self, email):
            return email == "allowed@example.com"

    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _Service())
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
