import pytest

from app.services.contact_service import ContactMatch, ContactResolveResult


@pytest.mark.asyncio
async def test_contacts_resolve_tool(monkeypatch):
    import mcp_servers.contacts.server as srv

    class _Service:
        async def resolve(self, query):
            assert query == "张三"
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name="张三",
                        match_type="exact",
                        email="a@example.com",
                    )
                ],
                ambiguous=False,
            )

    monkeypatch.setattr(srv, "contact_service", _Service())
    result = await srv.resolve("张三")

    assert result == {
        "matches": [
            {
                "kind": "person",
                "name": "张三",
                "match_type": "exact",
                "email": "a@example.com",
                "feishu_open_id": None,
                "feishu_user_id": None,
                "chat_id": None,
            }
        ],
        "ambiguous": False,
    }
