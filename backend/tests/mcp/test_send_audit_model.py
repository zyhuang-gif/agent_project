"""SendAudit ORM 表结构、索引与最小 CRUD 冒烟（P3a）。"""
import pytest
import pytest_asyncio
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.chat_history import Base
from app.models.send_audit import SendAudit


@pytest_asyncio.fixture
async def audit_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield engine, Session
    await engine.dispose()


@pytest.mark.asyncio
async def test_send_audit_table_has_expected_columns(audit_db):
    engine, _Session = audit_db

    def _columns(sync_conn):
        return {c["name"] for c in inspect(sync_conn).get_columns("send_audit")}

    async with engine.begin() as conn:
        cols = await conn.run_sync(_columns)

    required = {
        "id", "created_at", "user_id", "session_id",
        "channel", "recipient_kind", "recipient", "recipient_name",
        "subject", "artifact_name", "artifact_path",
        "message_id", "file_key", "tool_name",
        "request_payload_preview", "status",
    }
    missing = required - cols
    assert not missing, f"send_audit 缺列：{missing}"


@pytest.mark.asyncio
async def test_send_audit_has_required_indexes(audit_db):
    engine, _Session = audit_db

    def _index_columns(sync_conn):
        idx = inspect(sync_conn).get_indexes("send_audit")
        return {tuple(i["column_names"]) for i in idx}

    async with engine.begin() as conn:
        idx_cols = await conn.run_sync(_index_columns)

    # 关键列必须有单列索引
    for anchor in (("channel",), ("recipient",), ("message_id",), ("created_at",), ("session_id",)):
        assert anchor in idx_cols, f"缺 {anchor} 索引"


@pytest.mark.asyncio
async def test_send_audit_insert_and_query_round_trip(audit_db):
    _engine, Session = audit_db

    async with Session() as db:
        db.add(SendAudit(
            channel="gmail",
            recipient_kind="email",
            recipient="a@example.com",
            recipient_name="张三",
            subject="报告",
            artifact_name="report.md",
            artifact_path="/tmp/report.md",
            message_id="msg-1",
            tool_name="send_email",
            user_id="u1",
            session_id="s1",
            request_payload_preview='{"to":["a@example.com"]}',
        ))
        await db.commit()

    async with Session() as db:
        rows = (await db.execute(select(SendAudit))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.channel == "gmail"
    assert row.recipient == "a@example.com"
    assert row.recipient_name == "张三"
    assert row.message_id == "msg-1"
    assert row.tool_name == "send_email"
    assert row.status == "success"
    assert row.created_at is not None


@pytest.mark.asyncio
async def test_send_audit_supports_feishu_file_row_with_file_key(audit_db):
    _engine, Session = audit_db

    async with Session() as db:
        db.add(SendAudit(
            channel="feishu",
            recipient_kind="open_id",
            recipient="ou_ok",
            recipient_name="张三",
            artifact_name="report.md",
            artifact_path="/tmp/report.md",
            message_id="om_file",
            file_key="file_v3_abc",
            tool_name="send_file_to_user",
        ))
        await db.commit()

    async with Session() as db:
        rows = (await db.execute(select(SendAudit))).scalars().all()
    assert len(rows) == 1
    assert rows[0].file_key == "file_v3_abc"
    assert rows[0].recipient_kind == "open_id"


def test_init_db_module_imports_send_audit():
    """init_db 必须显式 import app.models.send_audit，否则 Base.metadata.create_all
    在生产库上不会建表。"""
    import inspect as py_inspect

    from app.db import db_config

    src = py_inspect.getsource(db_config.init_db)
    assert "send_audit" in src, "init_db 里必须显式 import send_audit 模型"
