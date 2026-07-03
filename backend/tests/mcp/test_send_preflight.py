import pytest

from app.services.contact_service import ContactMatch, ContactResolveResult
from app.mcp.preflight import resolve_send_targets


@pytest.mark.asyncio
async def test_preflight_allows_exact_gmail_person(monkeypatch):
    class _Service:
        async def resolve(self, query):
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name=query,
                        match_type="exact",
                        email="a@example.com",
                    )
                ],
                ambiguous=False,
            )

    monkeypatch.setattr("app.mcp.preflight.contact_service", _Service())

    result = await resolve_send_targets(["张三"], [])

    assert result.can_send is True
    assert result.gmail_to == ["a@example.com"]
    assert result.user_message == ""


@pytest.mark.asyncio
async def test_preflight_rejects_fuzzy_match(monkeypatch):
    class _Service:
        async def resolve(self, query):
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name="张三",
                        match_type="fuzzy",
                        email="a@example.com",
                    )
                ],
                ambiguous=True,
            )

    monkeypatch.setattr("app.mcp.preflight.contact_service", _Service())

    result = await resolve_send_targets(["张"], [])

    assert result.can_send is False
    assert "需要确认" in result.user_message


@pytest.mark.asyncio
async def test_preflight_rejects_group_without_partial_send(monkeypatch):
    class _Service:
        async def resolve(self, query):
            if query == "运营组":
                return ContactResolveResult(
                    matches=[
                        ContactMatch(
                            kind="group",
                            name="运营组",
                            match_type="exact",
                            chat_id="oc_1",
                        )
                    ],
                    ambiguous=False,
                )
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name=query,
                        match_type="exact",
                        email="a@example.com",
                    )
                ],
                ambiguous=False,
            )

    monkeypatch.setattr("app.mcp.preflight.contact_service", _Service())

    result = await resolve_send_targets(["张三", "运营组"], [])

    assert result.can_send is False
    assert result.gmail_to == []
    assert "P0 暂不支持群聊" in result.user_message


@pytest.mark.asyncio
async def test_preflight_rejects_explicit_feishu_channel():
    result = await resolve_send_targets(["张三"], ["feishu"])

    assert result.can_send is False
    assert "飞书发送暂未启用" in result.user_message
