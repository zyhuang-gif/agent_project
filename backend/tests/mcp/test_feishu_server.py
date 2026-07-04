"""Feishu / Lark MCP server 单测——mock httpx，无网络调用。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── 通用 fake httpx client ────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        # httpx.Response.text 通常存在，仅 _parse_response 里 non-JSON 分支会读
        self.text = text if text else (json.dumps(payload or {}, ensure_ascii=False))

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class _FakeClient:
    """支持 async context manager + async post(); 通过 responder 回调决定响应。"""

    def __init__(self, calls: list[dict], responder):
        self._calls = calls
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, json=None, data=None, files=None, headers=None):
        call = {
            "url": url,
            "json": json,
            "data": data,
            "files": files,
            "headers": headers or {},
        }
        self._calls.append(call)
        response = self._responder(call)
        if response is None:
            raise AssertionError(f"responder returned None for {url}")
        return response


def _install_fake_client(monkeypatch, responder) -> list[dict]:
    """替换 lark_client / token_cache 里的 _http_client 工厂。"""
    calls: list[dict] = []

    def _factory():
        return _FakeClient(calls, responder)

    monkeypatch.setattr("mcp_servers.feishu.lark_client._http_client", _factory)
    monkeypatch.setattr("mcp_servers.feishu.token_cache._http_client", _factory)
    return calls


@pytest.fixture(autouse=True)
def _env_and_cache(monkeypatch):
    """每个测试独立的空 token 缓存 + 必备 env。"""
    monkeypatch.setenv("LARK_APP_ID", "cli_test_app")
    monkeypatch.setenv("LARK_APP_SECRET", "test_secret")
    monkeypatch.delenv("LARK_DOMAIN", raising=False)

    from mcp_servers.feishu import token_cache
    token_cache._reset_for_test()
    yield
    token_cache._reset_for_test()


def _token_response(token: str = "t-abc", expire: int = 7200):
    return _FakeResponse(200, {
        "code": 0, "msg": "ok",
        "tenant_access_token": token, "expire": expire,
    })


# ── token cache ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_cache_fetches_and_caches(monkeypatch):
    def _responder(call):
        assert call["url"].endswith("/open-apis/auth/v3/tenant_access_token/internal")
        assert call["json"] == {"app_id": "cli_test_app", "app_secret": "test_secret"}
        return _token_response(token="t-1", expire=7200)

    calls = _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.token_cache import get_tenant_access_token

    tok1 = await get_tenant_access_token()
    tok2 = await get_tenant_access_token()

    assert tok1 == "t-1"
    assert tok2 == "t-1"
    # 第二次命中缓存，无 HTTP
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_token_cache_refreshes_when_within_refresh_buffer(monkeypatch):
    """expires_at - buffer 内视为需要刷新。"""
    counter = {"n": 0}

    def _responder(call):
        counter["n"] += 1
        return _token_response(token=f"t-{counter['n']}", expire=7200)

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu import token_cache

    # 第一次拿到新 token
    tok1 = await token_cache.get_tenant_access_token()
    # 手动把 _expires_at 拉到几乎等于 now，进入 refresh buffer
    import time as _time
    token_cache._cache._expires_at = _time.time() + 60
    tok2 = await token_cache.get_tenant_access_token()

    assert tok1 == "t-1"
    assert tok2 == "t-2"


@pytest.mark.asyncio
async def test_token_cache_raises_when_credentials_missing(monkeypatch):
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)

    from mcp_servers.feishu.lark_client import LarkAPIError

    # 使得网络层永远不会被触发
    _install_fake_client(monkeypatch, lambda call: _FakeResponse(200, {"code": 0}))

    from mcp_servers.feishu import lark_client

    with pytest.raises(LarkAPIError) as ex:
        await lark_client.send_text_message("open_id", "ou_a", "hi")

    assert ex.value.error == "AUTH_FAILED"


# ── send_text_message ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_text_message_success(monkeypatch):
    def _responder(call):
        if "tenant_access_token" in call["url"]:
            return _token_response()
        assert call["url"].endswith("/open-apis/im/v1/messages?receive_id_type=open_id")
        assert call["headers"]["Authorization"] == "Bearer t-abc"
        assert call["headers"]["Content-Type"].startswith("application/json")
        # content 必须是 JSON 字符串（不是 dict）
        assert call["json"]["msg_type"] == "text"
        content = call["json"]["content"]
        assert isinstance(content, str)
        assert json.loads(content) == {"text": "hello"}
        return _FakeResponse(200, {
            "code": 0, "msg": "success",
            "data": {"message_id": "om_ok"},
        })

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import send_text_message
    result = await send_text_message("open_id", "ou_a", "hello")
    assert result == {"message_id": "om_ok"}


@pytest.mark.asyncio
async def test_send_text_message_chat_id_url(monkeypatch):
    """chat_id 分支：URL query 里 receive_id_type=chat_id。"""

    def _responder(call):
        if "tenant_access_token" in call["url"]:
            return _token_response()
        assert "receive_id_type=chat_id" in call["url"]
        return _FakeResponse(200, {"code": 0, "data": {"message_id": "om_g"}})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import send_text_message
    result = await send_text_message("chat_id", "oc_1", "hi")
    assert result == {"message_id": "om_g"}


@pytest.mark.asyncio
async def test_send_text_message_api_error_raises(monkeypatch):
    def _responder(call):
        if "tenant_access_token" in call["url"]:
            return _token_response()
        return _FakeResponse(400, {"code": 230034, "msg": "invalid receive_id"})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import LarkAPIError, send_text_message

    with pytest.raises(LarkAPIError) as ex:
        await send_text_message("open_id", "ou_bad", "hi")

    assert ex.value.error == "LARK_API_ERROR"
    assert ex.value.code == "230034"
    assert ex.value.to_dict()["channel"] == "feishu"


@pytest.mark.asyncio
async def test_send_text_message_response_missing_message_id_raises(monkeypatch):
    def _responder(call):
        if "tenant_access_token" in call["url"]:
            return _token_response()
        return _FakeResponse(200, {"code": 0, "data": {}})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import LarkAPIError, send_text_message

    with pytest.raises(LarkAPIError) as ex:
        await send_text_message("open_id", "ou_a", "hi")
    assert ex.value.error == "LARK_API_ERROR"


# ── upload_file + send_file_message + end-to-end file tool ─────────


def _make_artifact(tmp_path, name: str = "report.md", content: str = "# report") -> Path:
    """在 artifact_root/send/ 下写一份 .md，返回绝对路径。"""
    send_dir = tmp_path / "send"
    send_dir.mkdir(parents=True, exist_ok=True)
    path = send_dir / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_upload_file_success(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    artifact = _make_artifact(tmp_path, "r.md", "# report")

    def _responder(call):
        if "tenant_access_token" in call["url"]:
            return _token_response()
        assert call["url"].endswith("/open-apis/im/v1/files")
        assert call["headers"]["Authorization"] == "Bearer t-abc"
        # multipart 字段
        assert call["data"]["file_type"] == "stream"
        assert call["data"]["file_name"] == "r.md"
        assert "file" in call["files"]
        return _FakeResponse(200, {"code": 0, "data": {"file_key": "file_v3_abc"}})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import upload_file
    result = await upload_file(str(artifact))
    assert result == {"file_key": "file_v3_abc"}


@pytest.mark.asyncio
async def test_upload_file_rejects_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    from mcp_servers.feishu.lark_client import LarkAPIError, upload_file

    _install_fake_client(monkeypatch, lambda call: _FakeResponse(200, {"code": 0}))

    with pytest.raises(LarkAPIError) as ex:
        await upload_file(str(tmp_path / "send" / "nope.md"))
    assert ex.value.error == "FILE_INVALID"


@pytest.mark.asyncio
async def test_upload_file_rejects_outside_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    outside = tmp_path / "loose.md"
    outside.write_text("x", encoding="utf-8")
    _install_fake_client(monkeypatch, lambda call: _FakeResponse(200, {"code": 0}))

    from mcp_servers.feishu.lark_client import LarkAPIError, upload_file
    with pytest.raises(LarkAPIError) as ex:
        await upload_file(str(outside))
    assert ex.value.error == "FILE_INVALID"


@pytest.mark.asyncio
async def test_upload_file_rejects_wrong_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    send_dir = tmp_path / "send"
    send_dir.mkdir(parents=True, exist_ok=True)
    bad = send_dir / "bad.exe"
    bad.write_bytes(b"x")
    _install_fake_client(monkeypatch, lambda call: _FakeResponse(200, {"code": 0}))
    from mcp_servers.feishu.lark_client import LarkAPIError, upload_file
    with pytest.raises(LarkAPIError):
        await upload_file(str(bad))


@pytest.mark.asyncio
async def test_upload_file_rejects_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    send_dir = tmp_path / "send"
    send_dir.mkdir(parents=True, exist_ok=True)
    empty = send_dir / "empty.md"
    empty.write_bytes(b"")
    _install_fake_client(monkeypatch, lambda call: _FakeResponse(200, {"code": 0}))
    from mcp_servers.feishu.lark_client import LarkAPIError, upload_file
    with pytest.raises(LarkAPIError):
        await upload_file(str(empty))


@pytest.mark.asyncio
async def test_send_file_to_user_end_to_end(monkeypatch, tmp_path):
    """server 工具 send_file_to_user：先 upload 拿 file_key，再 send_file_message。"""
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    artifact = _make_artifact(tmp_path, "r.md", "# r")

    def _responder(call):
        url = call["url"]
        if "tenant_access_token" in url:
            return _token_response()
        if url.endswith("/open-apis/im/v1/files"):
            return _FakeResponse(200, {"code": 0, "data": {"file_key": "file_v3_up"}})
        if "im/v1/messages" in url and "receive_id_type=open_id" in url:
            # msg_type=file, content 引用 file_key
            body = call["json"]
            assert body["msg_type"] == "file"
            assert json.loads(body["content"]) == {"file_key": "file_v3_up"}
            return _FakeResponse(200, {"code": 0, "data": {"message_id": "om_f"}})
        raise AssertionError(f"unexpected url {url}")

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.server import send_file_to_user
    result = await send_file_to_user(
        open_id="ou_a", file_path=str(artifact), file_name="r.md"
    )
    assert result == {"message_id": "om_f", "file_key": "file_v3_up"}


# ── invalid token retry ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_token_retries_once_and_succeeds(monkeypatch):
    """第一次消息发送遇 99991663 → 清缓存重取 token → 第二次成功。"""
    state = {"token_fetches": 0, "msg_calls": 0}

    def _responder(call):
        url = call["url"]
        if "tenant_access_token" in url:
            state["token_fetches"] += 1
            return _token_response(token=f"t-{state['token_fetches']}")
        state["msg_calls"] += 1
        if state["msg_calls"] == 1:
            return _FakeResponse(200, {"code": 99991663, "msg": "invalid tenant_access_token"})
        return _FakeResponse(200, {"code": 0, "data": {"message_id": "om_ok"}})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import send_text_message
    result = await send_text_message("open_id", "ou_a", "hi")

    assert result == {"message_id": "om_ok"}
    # 至少发生两次消息调用（第一次失败 + 第二次成功），且 token 被重取
    assert state["msg_calls"] == 2
    assert state["token_fetches"] >= 2


@pytest.mark.asyncio
async def test_invalid_token_retry_still_fails_raises(monkeypatch):
    """两次都返回 invalid token 就抛。"""
    counter = {"n": 0}

    def _responder(call):
        if "tenant_access_token" in call["url"]:
            counter["n"] += 1
            return _token_response(token=f"t-{counter['n']}")
        return _FakeResponse(200, {"code": 99991661, "msg": "invalid access_token"})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.lark_client import LarkAPIError, send_text_message
    with pytest.raises(LarkAPIError) as ex:
        await send_text_message("open_id", "ou_a", "hi")
    assert ex.value.code == "99991661"


# ── server-tool 层的 unsupported msg_type / LarkAPIError → RuntimeError ─


@pytest.mark.asyncio
async def test_server_tool_rejects_unsupported_msg_type(monkeypatch):
    from mcp_servers.feishu.server import send_message_to_user
    with pytest.raises(RuntimeError) as ex:
        await send_message_to_user(open_id="ou_a", content="hi", msg_type="post")
    payload = ex.value.args[0]
    assert payload["error"] == "UNSUPPORTED_MSG_TYPE"


@pytest.mark.asyncio
async def test_server_tool_wraps_lark_api_error(monkeypatch):
    def _responder(call):
        if "tenant_access_token" in call["url"]:
            return _token_response()
        return _FakeResponse(200, {"code": 230034, "msg": "invalid receive_id"})

    _install_fake_client(monkeypatch, _responder)

    from mcp_servers.feishu.server import send_message_to_user
    with pytest.raises(RuntimeError) as ex:
        await send_message_to_user(open_id="ou_x", content="hi")
    payload = ex.value.args[0]
    assert payload["error"] == "LARK_API_ERROR"
    assert payload["channel"] == "feishu"
    assert payload["code"] == "230034"
