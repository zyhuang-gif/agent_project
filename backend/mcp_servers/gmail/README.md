# Gmail MCP Server

P0 exposes one tool only: `send_email`.

Setup:

1. Create a Google Cloud OAuth client and enable Gmail API.
2. Save OAuth client JSON to `backend/secrets/gmail_credentials.json`.
3. Run a first local authorization to create `backend/secrets/gmail_token.json`.
4. Set `GMAIL_CREDENTIALS_PATH`, `GMAIL_TOKEN_PATH`, and `GMAIL_SENDER_EMAIL`.

Scope: `https://www.googleapis.com/auth/gmail.send`.

No Gmail read, delete, label, filter, or attachment capability is exposed in P0.
