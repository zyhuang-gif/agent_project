import pytest

from app.services.contact_service import ContactMatch, ContactResolveResult
from app.mcp.preflight import resolve_send_targets


# ── 辅助：可注入的 fake contact_service ────────────────────────────────


def _service_with(mapping):
    """按 query → ContactResolveResult 映射构造 stub。缺失 query 返回 no-match。"""

    class _Service:
        async def resolve(self, query):
            if query in mapping:
                return mapping[query]
            return ContactResolveResult(matches=[], ambiguous=False)

    return _Service()


def _person(name, *, email=None, feishu_open_id=None, match_type="exact"):
    return ContactResolveResult(
        matches=[
            ContactMatch(
                kind="person",
                name=name,
                match_type=match_type,
                email=email,
                feishu_open_id=feishu_open_id,
            )
        ],
        ambiguous=False,
    )


def _group(name, *, chat_id=None, match_type="exact"):
    return ContactResolveResult(
        matches=[
            ContactMatch(kind="group", name=name, match_type=match_type, chat_id=chat_id)
        ],
        ambiguous=False,
    )


# ── P0/P1 回归 ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preflight_allows_exact_gmail_person(monkeypatch):
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({"张三": _person("张三", email="a@example.com")}),
    )
    result = await resolve_send_targets(["张三"], [])

    assert result.can_send is True
    assert result.gmail_to == ["a@example.com"]
    assert result.channels == ["gmail"]
    assert result.user_message == ""


@pytest.mark.asyncio
async def test_preflight_rejects_fuzzy_match(monkeypatch):
    class _Service:
        async def resolve(self, query):
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person", name="张三", match_type="fuzzy",
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
async def test_preflight_rejects_group_when_gmail_selected(monkeypatch):
    """channels=[] 默认为 gmail；gmail 通道下遇到群聊 → fail-closed（不部分发送）。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "张三": _person("张三", email="a@example.com"),
            "运营组": _group("运营组", chat_id="oc_1"),
        }),
    )
    result = await resolve_send_targets(["张三", "运营组"], [])

    assert result.can_send is False
    assert result.gmail_to == []
    assert "Gmail 不支持群聊" in result.user_message
    assert result.reason == "gmail_group_unsupported"


@pytest.mark.asyncio
async def test_preflight_rejects_unsupported_channel(monkeypatch):
    """未知通道 (如 slack) → 立即 fail-closed，不查 DB。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({}),
    )
    result = await resolve_send_targets(["张三"], ["slack"])

    assert result.can_send is False
    assert "暂不支持的通道" in result.user_message
    assert result.reason == "unsupported_channel"


# ── P2：单独走 feishu 通道 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_preflight_feishu_user_exact_match(monkeypatch):
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({"张三": _person("张三", feishu_open_id="ou_zs")}),
    )
    result = await resolve_send_targets(["张三"], ["feishu"])

    assert result.can_send is True
    assert result.channels == ["feishu"]
    # feishu-only 时不需要 gmail email
    assert result.gmail_to == []
    assert len(result.feishu_users) == 1
    assert result.feishu_users[0].feishu_open_id == "ou_zs"
    assert result.feishu_users[0].name == "张三"
    assert not result.feishu_groups


@pytest.mark.asyncio
async def test_preflight_feishu_user_alias_match(monkeypatch):
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "老张": _person("张三", feishu_open_id="ou_zs", match_type="alias"),
        }),
    )
    result = await resolve_send_targets(["老张"], ["feishu"])

    assert result.can_send is True
    assert result.feishu_users[0].feishu_open_id == "ou_zs"
    assert result.feishu_users[0].name == "张三"


@pytest.mark.asyncio
async def test_preflight_feishu_group_exact_match(monkeypatch):
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({"运营组": _group("运营组", chat_id="oc_op")}),
    )
    result = await resolve_send_targets(["运营组"], ["feishu"])

    assert result.can_send is True
    assert result.gmail_to == []
    assert len(result.feishu_groups) == 1
    assert result.feishu_groups[0].chat_id == "oc_op"


@pytest.mark.asyncio
async def test_preflight_feishu_missing_open_id_fails_closed(monkeypatch):
    """feishu 通道下 person 缺 open_id → 不发。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({"张三": _person("张三", email="a@example.com", feishu_open_id=None)}),
    )
    result = await resolve_send_targets(["张三"], ["feishu"])

    assert result.can_send is False
    assert "飞书 open_id" in result.user_message
    assert result.reason == "missing_open_id"


@pytest.mark.asyncio
async def test_preflight_feishu_missing_chat_id_fails_closed(monkeypatch):
    """群命中但 chat_id 空 → 不发。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({"运营组": _group("运营组", chat_id=None)}),
    )
    result = await resolve_send_targets(["运营组"], ["feishu"])

    assert result.can_send is False
    assert result.reason == "missing_chat_id"


@pytest.mark.asyncio
async def test_preflight_feishu_group_needs_feishu_channel(monkeypatch):
    """群命中但通道里没 feishu（只有 gmail）→ 不部分发送。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({"运营组": _group("运营组", chat_id="oc_op")}),
    )
    result = await resolve_send_targets(["运营组"], ["gmail"])

    assert result.can_send is False
    assert result.reason == "gmail_group_unsupported"


# ── P2：混合通道 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preflight_mixed_channel_person_success(monkeypatch):
    """gmail + feishu 都选中；person 必须同时有 email 和 open_id。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "张三": _person("张三", email="a@example.com", feishu_open_id="ou_zs"),
        }),
    )
    result = await resolve_send_targets(["张三"], ["gmail", "feishu"])

    assert result.can_send is True
    assert result.channels == ["gmail", "feishu"]
    assert result.gmail_to == ["a@example.com"]
    assert len(result.feishu_users) == 1
    assert result.feishu_users[0].feishu_open_id == "ou_zs"


@pytest.mark.asyncio
async def test_preflight_mixed_channel_missing_email_fails_closed(monkeypatch):
    """混合通道下缺 email 也整体拒绝——不做部分发送。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "张三": _person("张三", email=None, feishu_open_id="ou_zs"),
        }),
    )
    result = await resolve_send_targets(["张三"], ["gmail", "feishu"])

    assert result.can_send is False
    assert result.reason == "missing_email"


@pytest.mark.asyncio
async def test_preflight_mixed_channel_missing_open_id_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "张三": _person("张三", email="a@example.com", feishu_open_id=None),
        }),
    )
    result = await resolve_send_targets(["张三"], ["gmail", "feishu"])

    assert result.can_send is False
    assert result.reason == "missing_open_id"


@pytest.mark.asyncio
async def test_preflight_mixed_channel_with_group_and_person(monkeypatch):
    """混合通道 + 群：gmail 不支持群 → 整体拒绝。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "张三": _person("张三", email="a@example.com", feishu_open_id="ou_zs"),
            "运营组": _group("运营组", chat_id="oc_op"),
        }),
    )
    result = await resolve_send_targets(["张三", "运营组"], ["gmail", "feishu"])

    assert result.can_send is False
    assert result.reason == "gmail_group_unsupported"


@pytest.mark.asyncio
async def test_preflight_feishu_only_person_plus_group(monkeypatch):
    """channels=['feishu'] 下 person + group 同发成功。"""
    monkeypatch.setattr(
        "app.mcp.preflight.contact_service",
        _service_with({
            "张三": _person("张三", feishu_open_id="ou_zs"),
            "运营组": _group("运营组", chat_id="oc_op"),
        }),
    )
    result = await resolve_send_targets(["张三", "运营组"], ["feishu"])

    assert result.can_send is True
    assert len(result.feishu_users) == 1
    assert len(result.feishu_groups) == 1
