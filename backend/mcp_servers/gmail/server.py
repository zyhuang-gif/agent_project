from mcp.server.fastmcp import FastMCP

from mcp_servers.gmail.gmail_client import GmailSendError, send_email as gmail_send


mcp = FastMCP("gmail")


@mcp.tool()
async def send_email(to: list[str], subject: str, body: str) -> dict[str, str]:
    """Send a plain-text Gmail message to already-approved recipients.

    Failure (auth / Gmail API / no message id) raises so FastMCP returns
    isError=True and the send_guard interceptor skips the success audit log.
    """
    try:
        return await gmail_send(to=to, subject=subject, body=body)
    except GmailSendError as error:
        # FastMCP will convert this exception into a CallToolResult with
        # isError=True; the interceptor uses that to gate the audit write.
        raise RuntimeError(error.to_dict()) from error


if __name__ == "__main__":
    mcp.run()
