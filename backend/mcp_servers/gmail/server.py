from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_servers.gmail.gmail_client import send_email as gmail_send


mcp = FastMCP("gmail")


@mcp.tool()
async def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
) -> dict[str, str]:
    """Send a plain-text Gmail message to already-approved recipients."""
    return await gmail_send(to=to, subject=subject, body=body, cc=cc, bcc=bcc)


if __name__ == "__main__":
    mcp.run()
