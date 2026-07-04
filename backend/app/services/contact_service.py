from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select

from app.db.db_config import AsyncSessionLocal
from app.models.contact import Contact, FeishuGroup


def normalize_email(email: str | None) -> str | None:
    if not email:
        return None
    return email.strip().lower() or None


def _split_alias(alias: str | None) -> set[str]:
    return {part.strip() for part in (alias or "").split(",") if part.strip()}


@dataclass(frozen=True)
class ContactMatch:
    kind: str
    name: str
    match_type: str
    email: Optional[str] = None
    feishu_open_id: Optional[str] = None
    feishu_user_id: Optional[str] = None
    chat_id: Optional[str] = None


@dataclass(frozen=True)
class ContactResolveResult:
    matches: list[ContactMatch]
    ambiguous: bool = False


class ContactService:
    """联系人与飞书群解析。P0 send 节点只信任 exact/alias 命中的 person。"""

    async def resolve(self, query: str) -> ContactResolveResult:
        q = (query or "").strip()
        if not q:
            return ContactResolveResult(matches=[])

        async with AsyncSessionLocal() as db:
            exact_people = (
                await db.execute(
                    select(Contact).where(Contact.is_active.is_(True), Contact.name == q)
                )
            ).scalars().all()
            if exact_people:
                return ContactResolveResult(
                    [self._person_match(row, "exact") for row in exact_people]
                )

            people = (
                await db.execute(select(Contact).where(Contact.is_active.is_(True)))
            ).scalars().all()
            alias_matches = [row for row in people if q in _split_alias(row.alias)]
            if alias_matches:
                return ContactResolveResult(
                    [self._person_match(row, "alias") for row in alias_matches]
                )

            exact_groups = (
                await db.execute(
                    select(FeishuGroup).where(
                        FeishuGroup.is_active.is_(True), FeishuGroup.name == q
                    )
                )
            ).scalars().all()
            if exact_groups:
                return ContactResolveResult(
                    [self._group_match(row, "exact") for row in exact_groups]
                )

            fuzzy_people = (
                await db.execute(
                    select(Contact).where(
                        Contact.is_active.is_(True), Contact.name.like(f"%{q}%")
                    )
                )
            ).scalars().all()
            matches = [self._person_match(row, "fuzzy") for row in fuzzy_people]
            return ContactResolveResult(matches=matches, ambiguous=bool(matches))

    async def email_allowed(self, email: str) -> bool:
        normalized = normalize_email(email)
        if not normalized:
            return False
        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(select(Contact).where(Contact.is_active.is_(True)))
            ).scalars().all()
            return any(normalize_email(row.email) == normalized for row in rows)

    async def feishu_open_id_allowed(self, open_id: str) -> bool:
        """P2 白名单：open_id 必须存在于 active contacts.feishu_open_id。
        大小写与空白 sensitive——飞书 open_id 是大小写敏感的不透明字符串。"""
        oid = (open_id or "").strip()
        if not oid:
            return False
        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(
                    select(Contact).where(
                        Contact.is_active.is_(True),
                        Contact.feishu_open_id == oid,
                    )
                )
            ).scalars().all()
            return bool(rows)

    async def chat_id_allowed(self, chat_id: str) -> bool:
        """P2 白名单：chat_id 必须存在于 active feishu_groups.chat_id。"""
        cid = (chat_id or "").strip()
        if not cid:
            return False
        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(
                    select(FeishuGroup).where(
                        FeishuGroup.is_active.is_(True),
                        FeishuGroup.chat_id == cid,
                    )
                )
            ).scalars().all()
            return bool(rows)

    async def create_contact(self, **values) -> Contact:
        values["email"] = normalize_email(values.get("email"))
        async with AsyncSessionLocal() as db:
            row = Contact(**values)
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row

    async def list_contacts(self) -> list[Contact]:
        async with AsyncSessionLocal() as db:
            return (await db.execute(select(Contact).order_by(Contact.name))).scalars().all()

    @staticmethod
    def _person_match(row: Contact, match_type: str) -> ContactMatch:
        return ContactMatch(
            kind="person",
            name=row.name,
            match_type=match_type,
            email=normalize_email(row.email),
            feishu_open_id=row.feishu_open_id,
            feishu_user_id=row.feishu_user_id,
        )

    @staticmethod
    def _group_match(row: FeishuGroup, match_type: str) -> ContactMatch:
        return ContactMatch(
            kind="group",
            name=row.name,
            match_type=match_type,
            chat_id=row.chat_id,
        )


contact_service = ContactService()
