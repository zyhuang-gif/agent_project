from mcp.server.fastmcp import FastMCP

from app.services.contact_service import contact_service


mcp = FastMCP("contacts")


def _match_to_dict(match) -> dict:
    return {
        "kind": match.kind,
        "name": match.name,
        "match_type": match.match_type,
        "email": match.email,
        "feishu_open_id": match.feishu_open_id,
        "feishu_user_id": match.feishu_user_id,
        "chat_id": match.chat_id,
    }


@mcp.tool()
async def resolve(query: str) -> dict:
    """Resolve a natural-language person or group name to send targets."""
    result = await contact_service.resolve(query)
    return {
        "matches": [_match_to_dict(match) for match in result.matches],
        "ambiguous": result.ambiguous,
    }


@mcp.tool()
async def list_all() -> dict:
    """List contacts for local smoke/debug scripts."""
    rows = await contact_service.list_contacts()
    return {
        "contacts": [
            {
                "name": row.name,
                "email": row.email,
                "alias": row.alias,
                "is_active": row.is_active,
            }
            for row in rows
        ]
    }


if __name__ == "__main__":
    mcp.run()
