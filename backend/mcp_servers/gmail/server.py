from mcp.server.fastmcp import FastMCP

from mcp_servers.gmail.gmail_client import GmailSendError, send_email as gmail_send


mcp = FastMCP("gmail")


@mcp.tool()
async def send_email(
    to: list[str],
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> dict[str, str]:
    """Send a Gmail message to already-approved recipients (with optional
    Markdown attachments introduced in P1).

    Failure (auth / Gmail API / no message id / invalid attachment) raises
    so FastMCP returns isError=True and the send_guard interceptor skips
    the success audit log.
    """
    try:
        return await gmail_send(to=to, subject=subject, body=body, attachments=attachments)
    except GmailSendError as error:
        raise RuntimeError(error.to_dict()) from error


if __name__ == "__main__":
    mcp.run()
