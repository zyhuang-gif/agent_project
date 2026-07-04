# Gmail MCP Server

P1 exposes one tool only: `send_email(to, subject, body, attachments?)`.

Setup:

1. Create a Google Cloud OAuth client and enable Gmail API.
2. Save OAuth client JSON to `backend/secrets/gmail_credentials.json`.
3. Run a first local authorization to create `backend/secrets/gmail_token.json`.
4. Set `GMAIL_CREDENTIALS_PATH`, `GMAIL_TOKEN_PATH`, and `GMAIL_SENDER_EMAIL`.

Scope: `https://www.googleapis.com/auth/gmail.send`.

## P1: Markdown attachments

`send_email` accepts an optional `attachments: list[str]` of absolute paths.
Every path is validated **before** any Gmail API call:

- Must live under `ARTIFACT_ROOT` (defaults to `backend/runtime/artifacts`).
- Only `.md` is allowed; a single file must be ≤ 5 MiB.
- Missing / oversize / wrong-extension / outside-root all raise
  `GmailSendError("ATTACHMENT_INVALID", ...)`, which `send_guard` still
  reports as `isError=True` and therefore never audits as a success.

No Gmail read, delete, label, filter, or non-Markdown attachment capability is
exposed. PDF / Docx / 飞书 / per-user OAuth remain out of scope for P1.
