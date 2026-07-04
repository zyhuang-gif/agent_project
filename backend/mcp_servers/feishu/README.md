# Feishu / Lark MCP Server (P2)

P2 exposes exactly four tools to the LangChain agent (prefixed by
`tool_name_prefix=true`):

- `feishu_send_message_to_user(open_id, content, msg_type="text")`
- `feishu_send_message_to_group(chat_id, content, msg_type="text")`
- `feishu_send_file_to_user(open_id, file_path, file_name?)`
- `feishu_send_file_to_group(chat_id, file_path, file_name?)`

The file tools are atomic: `upload_file` → `send_file_message`. The upload
step is never exposed as its own tool, so an agent can never send a file it
did not first upload through this server.

## Environment

| Name | Purpose |
|------|---------|
| `LARK_APP_ID` | Self-built app credential ID |
| `LARK_APP_SECRET` | Self-built app credential secret |
| `LARK_DOMAIN` | Base URL, defaults to `https://open.feishu.cn` |
| `ARTIFACT_ROOT` | Optional; must equal the backend's artifact root so file uploads share the same controlled directory |

## Permissions (self-built app scopes)

Grant the following scopes on the Feishu developer console:

- `im:message` (send messages)
- `im:message:send_as_bot` (some tenants require this in addition)
- `im:resource` (upload files)

The bot must be **added to the target group** before `send_message_to_group`
/ `send_file_to_group` will succeed.

## Recipient whitelist

All four tools go through `SendGuardInterceptor` in `app/mcp/send_guard.py`:

- `open_id` must match an **active** `Contact.feishu_open_id`.
- `chat_id` must match an **active** `FeishuGroup.chat_id`.
- Any mismatch returns `{"error": "RECIPIENT_NOT_ALLOWED", "channel": "feishu"}`
  **before** any Feishu API call is made.

Successful calls append a JSON line to `backend/log/sends.log` with
`channel="feishu"` and (for file tools) the returned `file_key`.

## File safety

`upload_file` validates the local path **before** hitting the network:

- Must be an absolute path resolved inside `ARTIFACT_ROOT` (defaults to
  `backend/runtime/artifacts`).
- Extension must be `.md` (matches P1 artifact policy).
- Size must be > 0 and ≤ 30 MiB (Feishu limit).
- Missing / oversize / wrong-extension all raise
  `LarkAPIError("FILE_INVALID", ...)`, which surfaces as `isError=True` at the
  MCP layer so `send_guard` never audits it as a success.

## tenant_access_token cache

Per Feishu's docs, `tenant_access_token` is valid for up to 2h. The server
keeps a per-process cache (`token_cache.py`) and refreshes 5 min before
expiry.

If the server receives a token-invalid error (`code=99991661` or `99991663`),
the cache is cleared and the request is retried once internally. A second
failure surfaces as `AUTH_FAILED` / `LARK_API_ERROR` to the caller — no
silent retries beyond that.

No Redis dependency. If the process restarts, the cache is naturally rebuilt.

## Smoke test

```powershell
$env:MCP_ENABLED = "true"
$env:LARK_APP_ID = "cli_xxx"
$env:LARK_APP_SECRET = "yyy"
$env:LARK_DOMAIN = "https://open.feishu.cn"

# 1. Seed contacts.feishu_open_id and feishu_groups.chat_id via the CRUD APIs.
# 2. Enable feishu in backend/app/config/mcp_servers.yaml (enabled: true).
# 3. Ask the assistant: "把刚才那份报告发给张三，同时发到运营群。"
# 4. Verify:
#    - the Feishu user + group both receive the text message and Markdown file
#    - backend/log/sends.log contains two entries with channel="feishu"
#      (one per target × tool call).
```

**Not in scope for P2**: post / interactive messages, PDF / Docx artifacts,
per-user OAuth, Feishu contact-book sync, group management, message editing.
