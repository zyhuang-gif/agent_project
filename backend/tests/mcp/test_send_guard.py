import json
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from app.mcp.context import mcp_call_context
from app.mcp.send_guard import SendGuardInterceptor


# ── Gmail 分支（P0/P1 回归） ─────────────────────────────────────────


class _Req:
    name = "send_email"
    server_name = "gmail"
    args = {"to": [" allowed@example.com "], "subject": "S", "body": "B"}


class _AllowService:
    async def email_allowed(self, email):
        return email == "allowed@example.com"

    async def feishu_open_id_allowed(self, open_id):
        return open_id == "ou_ok"

    async def chat_id_allowed(self, chat_id):
        return chat_id == "oc_ok"


class _DenyService:
    async def email_allowed(self, email):
        return False

    async def feishu_open_id_allowed(self, open_id):
        return False

    async def chat_id_allowed(self, chat_id):
        return False


@pytest.mark.asyncio
async def test_send_guard_blocks_unlisted_recipient(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _DenyService())
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
    assert payload["channel"] == "gmail"
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
    assert audit["channel"] == "gmail"
    assert audit["recipient"] == ["allowed@example.com"]
    assert audit["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_send_guard_does_not_audit_when_handler_returns_error(monkeypatch, tmp_path):
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
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"error":"GMAIL_API_ERROR","message":"500"}')],
        )

    result = await SendGuardInterceptor()(_Req(), handler)

    assert not (tmp_path / "sends.log").exists()
    assert getattr(result, "isError", False) is True


@pytest.mark.asyncio
async def test_send_guard_does_not_audit_when_message_id_missing(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    result = await SendGuardInterceptor()(_Req(), handler)

    assert not (tmp_path / "sends.log").exists()
    assert getattr(result, "isError", False) is True


# ── Feishu 分支（P2 新增） ──────────────────────────────────────────


class _FeishuUserReq:
    name = "send_message_to_user"
    server_name = "feishu"
    args = {"open_id": " ou_ok ", "content": "hi", "msg_type": "text"}


class _FeishuUserFileReq:
    name = "send_file_to_user"
    server_name = "feishu"
    args = {"open_id": "ou_ok", "file_path": "/tmp/x.md", "file_name": "x.md"}


class _FeishuGroupReq:
    name = "send_message_to_group"
    server_name = "feishu"
    args = {"chat_id": "oc_ok", "content": "hi", "msg_type": "text"}


class _FeishuGroupFileReq:
    name = "send_file_to_group"
    server_name = "feishu"
    args = {"chat_id": "oc_ok", "file_path": "/tmp/x.md", "file_name": "x.md"}


class _FeishuUnknownReq:
    """未白名单的 open_id/chat_id 请求。"""

    name = "send_message_to_user"
    server_name = "feishu"
    args = {"open_id": "ou_UNKNOWN", "content": "hi"}


@pytest.mark.asyncio
async def test_send_guard_blocks_unknown_feishu_open_id(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())  # only ou_ok allowed
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    called = {"handler": False}

    async def handler(request):
        called["handler"] = True
        return CallToolResult(content=[TextContent(type="text", text="")])

    result = await SendGuardInterceptor()(_FeishuUnknownReq(), handler)

    assert called["handler"] is False
    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert payload["error"] == "RECIPIENT_NOT_ALLOWED"
    assert payload["channel"] == "feishu"
    assert payload["kind"] == "open_id"
    assert not (tmp_path / "sends.log").exists()


@pytest.mark.asyncio
async def test_send_guard_blocks_unknown_feishu_chat_id(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())  # only oc_ok allowed
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    class _Req:
        name = "send_message_to_group"
        server_name = "feishu"
        args = {"chat_id": "oc_UNKNOWN", "content": "hi"}

    async def handler(request):
        raise AssertionError("handler must not be called")

    result = await SendGuardInterceptor()(_Req(), handler)
    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert payload["error"] == "RECIPIENT_NOT_ALLOWED"
    assert payload["kind"] == "chat_id"


@pytest.mark.asyncio
async def test_send_guard_allows_feishu_user_message_and_audits(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"message_id":"om_1"}')],
        )

    async with mcp_call_context(user_id="user-9", session_id="sess-9"):
        result = await SendGuardInterceptor()(_FeishuUserReq(), handler)

    assert result.isError is False
    lines = (tmp_path / "sends.log").read_text(encoding="utf-8").splitlines()
    audit = json.loads(lines[0])
    assert audit["channel"] == "feishu"
    assert audit["recipient"] == "ou_ok"
    assert audit["message_id"] == "om_1"
    assert audit["subject_or_preview"] == "hi"
    assert audit["user_id"] == "user-9"
    assert audit["session_id"] == "sess-9"
    # 文本工具不写 file_key
    assert "file_key" not in audit


@pytest.mark.asyncio
async def test_send_guard_allows_feishu_group_message_and_audits(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"message_id":"om_g"}')],
        )

    result = await SendGuardInterceptor()(_FeishuGroupReq(), handler)

    assert result.isError is False
    audit = json.loads((tmp_path / "sends.log").read_text(encoding="utf-8").splitlines()[0])
    assert audit["channel"] == "feishu"
    assert audit["recipient"] == "oc_ok"
    assert audit["message_id"] == "om_g"


@pytest.mark.asyncio
async def test_send_guard_feishu_file_success_audit_includes_file_key(monkeypatch, tmp_path):
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(
                type="text",
                text='{"message_id":"om_f","file_key":"file_v3_abc"}',
            )],
        )

    result = await SendGuardInterceptor()(_FeishuUserFileReq(), handler)

    assert result.isError is False
    audit = json.loads((tmp_path / "sends.log").read_text(encoding="utf-8").splitlines()[0])
    assert audit["channel"] == "feishu"
    assert audit["recipient"] == "ou_ok"
    assert audit["message_id"] == "om_f"
    assert audit["file_key"] == "file_v3_abc"
    # 文件工具的 preview 用 file_name
    assert audit["subject_or_preview"] == "x.md"


@pytest.mark.asyncio
async def test_send_guard_feishu_failure_no_audit(monkeypatch, tmp_path):
    """飞书 handler 报错 → 不写审计。"""
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"error":"LARK_API_ERROR","code":"230034"}')],
            isError=True,
        )

    result = await SendGuardInterceptor()(_FeishuUserReq(), handler)

    assert result.isError is True
    assert not (tmp_path / "sends.log").exists()


@pytest.mark.asyncio
async def test_send_guard_feishu_missing_message_id_no_audit(monkeypatch, tmp_path):
    """handler 忘了 message_id → 不能被当成功审计。"""
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _AllowService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    async def handler(request):
        return CallToolResult(
            content=[TextContent(type="text", text='{"file_key":"x"}')],
        )

    result = await SendGuardInterceptor()(_FeishuUserFileReq(), handler)

    assert result.isError is True
    assert not (tmp_path / "sends.log").exists()


@pytest.mark.asyncio
async def test_send_guard_passthrough_unknown_server(monkeypatch, tmp_path):
    """非 gmail/feishu 的 server 直通 handler，不校验白名单。"""
    import app.mcp.send_guard as sg
    monkeypatch.setattr(sg, "contact_service", _DenyService())
    monkeypatch.setattr(sg, "_audit_path", lambda: tmp_path / "sends.log")

    class _Req:
        name = "resolve"
        server_name = "contacts"
        args = {"query": "张三"}

    async def handler(request):
        return CallToolResult(content=[TextContent(type="text", text="{}")])

    result = await SendGuardInterceptor()(_Req(), handler)
    # contacts 不受 send_guard 影响；不写审计（本来也没 message_id 触发审计）
    assert result.isError is False
    assert not (tmp_path / "sends.log").exists()
