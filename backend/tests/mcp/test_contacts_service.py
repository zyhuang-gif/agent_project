import importlib

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.chat_history import Base
from app.models.contact import Contact, FeishuGroup
from app.services.contact_service import ContactService

contact_mod = importlib.import_module("app.services.contact_service")


@pytest_asyncio.fixture
async def contacts_db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(contact_mod, "AsyncSessionLocal", Session)
    async with Session() as db:
        db.add(Contact(name="张三", alias="老张,zhangsan", email=" ZhangSan@Example.COM ", is_active=True))
        db.add(Contact(name="张四", alias="", email="zhangsi@example.com", is_active=True))
        db.add(Contact(name="禁用用户", alias="disabled", email="disabled@example.com", is_active=False))
        db.add(FeishuGroup(name="运营组", chat_id="oc_123", is_active=True))
        await db.commit()
    yield Session
    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_exact_name_person(contacts_db):
    service = ContactService()
    result = await service.resolve("张三")

    assert result.ambiguous is False
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.kind == "person"
    assert match.name == "张三"
    assert match.email == "zhangsan@example.com"
    assert match.match_type == "exact"


@pytest.mark.asyncio
async def test_resolve_alias_person(contacts_db):
    result = await ContactService().resolve("老张")

    assert len(result.matches) == 1
    assert result.matches[0].match_type == "alias"
    assert result.matches[0].email == "zhangsan@example.com"


@pytest.mark.asyncio
async def test_resolve_fuzzy_is_ambiguous_for_auto_send(contacts_db):
    result = await ContactService().resolve("张")

    assert result.ambiguous is True
    assert {m.name for m in result.matches} == {"张三", "张四"}
    assert {m.match_type for m in result.matches} == {"fuzzy"}


@pytest.mark.asyncio
async def test_resolve_group(contacts_db):
    result = await ContactService().resolve("运营组")

    assert len(result.matches) == 1
    assert result.matches[0].kind == "group"
    assert result.matches[0].chat_id == "oc_123"


@pytest.mark.asyncio
async def test_inactive_contacts_are_not_resolved_or_allowed(contacts_db):
    service = ContactService()

    assert (await service.resolve("禁用用户")).matches == []
    assert await service.email_allowed("disabled@example.com") is False
    assert await service.email_allowed(" zhangsan@example.com ") is True
