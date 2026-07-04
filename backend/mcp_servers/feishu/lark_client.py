"""飞书 Open Platform HTTP 客户端。

只封装 P2 需要的三个动作：
- 发送文本消息（open_id / chat_id）
- 发送文件消息（引用 file_key）
- 上传文件拿 file_key

设计取舍：
- httpx.AsyncClient 通过 `_http_client()` 工厂返回，测试可以整体 monkeypatch。
- token 缓存放在 token_cache.py；本模块只在拿到 token 后拼请求，不管缓存。
- 官方 `code != 0` 一律抛 `LarkAPIError`，由 server.py 转成 `RuntimeError(dict)`，
  再走 send_guard interceptor 的错误分支（不写审计）。
- token 无效错误码（99991661/99991663）在这里内部重试一次；仍失败才抛。
"""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from mcp_servers.feishu.token_cache import (
    TokenFetchError,
    get_tenant_access_token,
    invalidate_token,
)


class LarkAPIError(RuntimeError):
    """飞书 API 失败统一封装——保证 send_guard 不会把假成功写审计。"""

    def __init__(self, error: str, message: str, code: str = "", data: dict | None = None):
        super().__init__(f"{error}: {message}")
        self.error = error
        self.message = message
        self.code = code
        self.data = data or {}

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {
            "error": self.error,
            "channel": "feishu",
            "message": self.message,
        }
        if self.code:
            payload["code"] = str(self.code)
        return payload


# 官方"token 无效/过期"错误码集合（错误码总表）。命中则清缓存并重试一次。
_TOKEN_INVALID_CODES = {"99991661", "99991663"}

# 飞书官方文档：单文件上限 30MB。我们进一步在 artifact 层强制 5MB Markdown。
_MAX_UPLOAD_BYTES = 30 * 1024 * 1024
_ALLOWED_FILE_EXT = {".md"}


def _base_url() -> str:
    return (os.getenv("LARK_DOMAIN") or "https://open.feishu.cn").rstrip("/")


def _artifact_root() -> Path:
    """与 app.mcp.artifacts.artifact_root / gmail_client._artifact_root 保持一致。
    lark_client.py → parents[2] = backend/。"""
    override = os.getenv("ARTIFACT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "runtime" / "artifacts").resolve()


def _validate_file(raw_path: str) -> Path:
    if not raw_path:
        raise LarkAPIError("FILE_INVALID", "empty file path")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise LarkAPIError("FILE_INVALID", f"file not found: {raw_path}")
    root = _artifact_root()
    if not path.is_relative_to(root):
        raise LarkAPIError("FILE_INVALID", f"file outside artifact root: {raw_path}")
    if path.suffix.lower() not in _ALLOWED_FILE_EXT:
        raise LarkAPIError("FILE_INVALID", f"unsupported file extension: {path.suffix}")
    size = path.stat().st_size
    if size == 0:
        raise LarkAPIError("FILE_INVALID", f"empty file: {raw_path}")
    if size > _MAX_UPLOAD_BYTES:
        raise LarkAPIError("FILE_INVALID", f"file too large: {size} bytes")
    return path


def _http_client() -> httpx.AsyncClient:
    """测试用 monkeypatch 替换整个函数即可注入 fake client。"""
    return httpx.AsyncClient(timeout=30.0)


def _parse_response(resp: httpx.Response) -> dict[str, Any]:
    try:
        payload = resp.json()
    except Exception as error:  # noqa: BLE001
        raise LarkAPIError(
            "LARK_API_ERROR",
            f"non-JSON HTTP {resp.status_code}: {resp.text[:120] if hasattr(resp, 'text') else ''}",
            code=str(resp.status_code),
        ) from error
    if resp.status_code >= 400 and (payload.get("code") in (0, None)):
        # HTTP 层错误但业务码没写——统一按 API_ERROR 处理
        raise LarkAPIError(
            "LARK_API_ERROR",
            payload.get("msg") or f"HTTP {resp.status_code}",
            code=str(resp.status_code),
            data=payload,
        )
    code = payload.get("code")
    if code not in (0, None):
        raise LarkAPIError(
            "LARK_API_ERROR",
            payload.get("msg") or "feishu api error",
            code=str(code),
            data=payload,
        )
    return payload


async def _do_json_request(url: str, json_body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    async with _http_client() as client:
        resp = await client.post(url, json=json_body, headers=headers)
    return _parse_response(resp)


async def _do_multipart_request(
    url: str, data: dict[str, str], files: dict[str, tuple], headers: dict[str, str]
) -> dict[str, Any]:
    async with _http_client() as client:
        resp = await client.post(url, data=data, files=files, headers=headers)
    return _parse_response(resp)


async def _wrap_token_errors(coro_factory):
    """token 拉取失败 (缺 env / 网络错) 也归到 LarkAPIError，保持 server.py 层统一。"""
    try:
        return await coro_factory()
    except TokenFetchError as error:
        raise LarkAPIError("AUTH_FAILED", error.message, code=error.code) from error


async def _post_json_with_token(url: str, body: dict[str, Any], *, retry_on_invalid: bool = True) -> dict[str, Any]:
    async def _once() -> dict[str, Any]:
        tat = await get_tenant_access_token()
        headers = {
            "Authorization": f"Bearer {tat}",
            "Content-Type": "application/json; charset=utf-8",
        }
        return await _do_json_request(url, body, headers)

    try:
        return await _wrap_token_errors(_once)
    except LarkAPIError as error:
        if retry_on_invalid and error.code in _TOKEN_INVALID_CODES:
            invalidate_token()
            return await _wrap_token_errors(_once)
        raise


async def _post_multipart_with_token(
    url: str,
    data: dict[str, str],
    files: dict[str, tuple],
    *,
    retry_on_invalid: bool = True,
) -> dict[str, Any]:
    async def _once() -> dict[str, Any]:
        tat = await get_tenant_access_token()
        headers = {"Authorization": f"Bearer {tat}"}
        return await _do_multipart_request(url, data, files, headers)

    try:
        return await _wrap_token_errors(_once)
    except LarkAPIError as error:
        if retry_on_invalid and error.code in _TOKEN_INVALID_CODES:
            invalidate_token()
            return await _wrap_token_errors(_once)
        raise


def _messages_url(receive_id_type: str) -> str:
    return f"{_base_url()}/open-apis/im/v1/messages?receive_id_type={receive_id_type}"


def _files_url() -> str:
    return f"{_base_url()}/open-apis/im/v1/files"


async def send_text_message(receive_id_type: str, receive_id: str, text: str) -> dict[str, str]:
    """POST /im/v1/messages msg_type=text.

    receive_id_type: "open_id" 或 "chat_id"。
    """
    if receive_id_type not in ("open_id", "chat_id"):
        raise LarkAPIError("PARAM_INVALID", f"unsupported receive_id_type: {receive_id_type}")
    if not receive_id:
        raise LarkAPIError("PARAM_INVALID", "empty receive_id")
    body = {
        "receive_id": receive_id,
        "msg_type": "text",
        # 官方要求 content 是 JSON 字符串——不是嵌套对象。
        "content": json.dumps({"text": text or ""}, ensure_ascii=False),
    }
    payload = await _post_json_with_token(_messages_url(receive_id_type), body)
    message_id = ((payload.get("data") or {}).get("message_id"))
    if not message_id:
        raise LarkAPIError(
            "LARK_API_ERROR",
            "response missing message_id",
            code=str(payload.get("code", "")),
            data=payload,
        )
    return {"message_id": message_id}


async def send_file_message(receive_id_type: str, receive_id: str, file_key: str) -> dict[str, str]:
    """POST /im/v1/messages msg_type=file，content 引用上传得到的 file_key。"""
    if receive_id_type not in ("open_id", "chat_id"):
        raise LarkAPIError("PARAM_INVALID", f"unsupported receive_id_type: {receive_id_type}")
    if not receive_id or not file_key:
        raise LarkAPIError("PARAM_INVALID", "empty receive_id or file_key")
    body = {
        "receive_id": receive_id,
        "msg_type": "file",
        "content": json.dumps({"file_key": file_key}),
    }
    payload = await _post_json_with_token(_messages_url(receive_id_type), body)
    message_id = ((payload.get("data") or {}).get("message_id"))
    if not message_id:
        raise LarkAPIError(
            "LARK_API_ERROR",
            "response missing message_id",
            code=str(payload.get("code", "")),
            data=payload,
        )
    return {"message_id": message_id}


async def upload_file(file_path: str, file_name: str | None = None) -> dict[str, str]:
    """POST /im/v1/files 上传 Markdown artifact，返回 file_key。

    严格校验路径：必须在 ARTIFACT_ROOT 内、后缀 .md、大小 > 0 且 <= 30MB。
    """
    path = _validate_file(file_path)
    file_bytes = path.read_bytes()
    fname = (file_name or path.name).strip() or path.name
    mime = mimetypes.guess_type(fname)[0] or "text/markdown"
    # 官方 file_type 枚举里 .md 落在 "stream"（其它未列出的通用类型）。
    data = {"file_type": "stream", "file_name": fname}
    files = {"file": (fname, file_bytes, mime)}
    payload = await _post_multipart_with_token(_files_url(), data, files)
    file_key = (payload.get("data") or {}).get("file_key")
    if not file_key:
        raise LarkAPIError(
            "LARK_API_ERROR",
            "response missing file_key",
            code=str(payload.get("code", "")),
            data=payload,
        )
    return {"file_key": file_key}
