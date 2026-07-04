"""Gmail auth 里代理注入逻辑的单元测试。

我们不能真调 Google，但可以验证：
1. 没设 HTTPS_PROXY/HTTP_PROXY 时 _proxy_info() 返回 None。
2. 设了 HTTPS_PROXY 时能构造 httplib2.ProxyInfo，主机/端口正确。
3. HTTP_PROXY 是 HTTPS_PROXY 的降级默认。
"""
import httplib2
import pytest


def test_proxy_info_returns_none_when_env_unset(monkeypatch):
    from mcp_servers.gmail.auth import _proxy_info

    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    assert _proxy_info() is None


def test_proxy_info_uses_https_proxy(monkeypatch):
    from mcp_servers.gmail.auth import _proxy_info

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    info = _proxy_info()
    assert isinstance(info, httplib2.ProxyInfo)
    assert info.proxy_host == "127.0.0.1"
    assert info.proxy_port == 7897
    # httplib2 内部 HTTP CONNECT 隧道类型编码为 3
    assert info.proxy_type == 3


def test_proxy_info_falls_back_to_http_proxy(monkeypatch):
    from mcp_servers.gmail.auth import _proxy_info

    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080")

    info = _proxy_info()
    assert info.proxy_host == "proxy.example.com"
    assert info.proxy_port == 8080
