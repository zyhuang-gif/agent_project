"""飞书 tenant_access_token 缓存（进程内、非持久化）。

设计（spec §7.1）：
- 单进程内维护一份 token + expires_at；过期前 _REFRESH_BUFFER_SECONDS
  内主动重新获取，避免踩过期边界。
- 遇 "invalid / expired token" 类错误码时，lark_client 会调用
  `invalidate()` 清缓存并让本次调用内部重试一次；仍失败才向上抛。
- 无 Redis / 无跨机共享——飞书官方 tat 也允许多份并发生效，本地缓存
  失效不会影响其它进程。
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

import httpx


_REFRESH_BUFFER_SECONDS = 5 * 60  # 5 min


class TokenFetchError(RuntimeError):
    """tenant_access_token 获取失败——包裹为 LarkAPIError 由 lark_client 处理。"""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.message = message
        self.code = code


class TokenCache:
    """线程/协程安全的 tenant_access_token 缓存。"""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._token = None
        self._expires_at = 0.0

    def snapshot(self) -> tuple[Optional[str], float]:
        return self._token, self._expires_at

    async def get(self) -> str:
        # 快路径：缓存有效
        now = time.time()
        if self._token and self._expires_at - _REFRESH_BUFFER_SECONDS > now:
            return self._token

        async with self._lock:
            # 双检；另一协程可能已经刷新过
            now = time.time()
            if self._token and self._expires_at - _REFRESH_BUFFER_SECONDS > now:
                return self._token

            token, expire_seconds = await _fetch_tenant_access_token()
            self._token = token
            self._expires_at = time.time() + max(int(expire_seconds), 60)
            return self._token


_cache = TokenCache()


def _base_url() -> str:
    domain = (os.getenv("LARK_DOMAIN") or "https://open.feishu.cn").rstrip("/")
    return domain


def _http_client() -> httpx.AsyncClient:
    """独立的 client 工厂——测试用 monkeypatch 替换整个函数即可。"""
    return httpx.AsyncClient(timeout=30.0)


async def _fetch_tenant_access_token() -> tuple[str, int]:
    app_id = os.getenv("LARK_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET")
    if not app_id or not app_secret:
        raise TokenFetchError("LARK_APP_ID and LARK_APP_SECRET are required")
    url = f"{_base_url()}/open-apis/auth/v3/tenant_access_token/internal"
    async with _http_client() as client:
        resp = await client.post(url, json={"app_id": app_id, "app_secret": app_secret})
    try:
        payload = resp.json()
    except Exception as error:  # noqa: BLE001
        raise TokenFetchError(f"token endpoint returned non-JSON: {resp.status_code}") from error
    code = payload.get("code")
    if code not in (0, None):
        raise TokenFetchError(
            payload.get("msg") or "tenant_access_token fetch failed",
            code=str(code),
        )
    token = payload.get("tenant_access_token")
    expire = payload.get("expire") or 7200
    if not token:
        raise TokenFetchError("tenant_access_token missing in response")
    return token, int(expire)


async def get_tenant_access_token() -> str:
    return await _cache.get()


def invalidate_token() -> None:
    _cache.invalidate()


def _reset_for_test() -> None:
    """测试用：显式重置全局缓存。"""
    _cache.invalidate()
