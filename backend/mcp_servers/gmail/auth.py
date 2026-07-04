import os
from pathlib import Path
from urllib.parse import urlparse

import httplib2
import google_auth_httplib2
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _required_path(env_name: str) -> Path:
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is required for Gmail MCP server")
    return Path(value)


def _proxy_info():
    """从 HTTPS_PROXY / HTTP_PROXY 构造 httplib2 ProxyInfo。
    httplib2 不认环境变量，必须显式传 ProxyInfo。
    httplib2 内部把 HTTP 代理编码为 3；不用 httplib2.socks 常量，
    避免可选依赖 pysocks 未装时 httplib2.socks 是 None。"""
    proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if not proxy_url:
        return None
    p = urlparse(proxy_url)
    PROXY_TYPE_HTTP = 3
    return httplib2.ProxyInfo(
        proxy_type=PROXY_TYPE_HTTP,
        proxy_host=p.hostname,
        proxy_port=p.port or 80,
    )


def load_credentials() -> Credentials:
    credentials_path = _required_path("GMAIL_CREDENTIALS_PATH")
    token_path = _required_path("GMAIL_TOKEN_PATH")
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


async def get_gmail_service():
    creds = load_credentials()
    proxy_info = _proxy_info()
    if proxy_info is not None:
        http = google_auth_httplib2.AuthorizedHttp(
            creds, http=httplib2.Http(proxy_info=proxy_info)
        )
        return build("gmail", "v1", http=http)
    return build("gmail", "v1", credentials=creds)