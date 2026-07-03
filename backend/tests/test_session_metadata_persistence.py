"""验证 ChatMessage.metadata_ 在 add_message / get_session 之间正确读写,
保证前端切换会话回切时能恢复 citations 与 agent steps。"""
import importlib

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.chat_history import Base, ChatMessage, ChatSession
from app.services.database_session_manager import DatabaseSessionManager

# `app.services.__init__.py` 里 `from .database_session_manager import database_session_manager`
# 把模块内同名全局变量(初值 None)挂到了 services 的 attribute 上,因此
# `import app.services.database_session_manager as mod` 会拿到 None。改走 sys.modules / importlib。
_dsm_module = importlib.import_module("app.services.database_session_manager")


@pytest_asyncio.fixture
async def sm_db(monkeypatch):
    """SQLite 内存库替换 database_session_manager 的 AsyncSessionLocal。"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(_dsm_module, "AsyncSessionLocal", Session)
    yield Session
    await engine.dispose()


@pytest.mark.asyncio
async def test_add_message_persists_citations_and_steps(sm_db):
    sm = DatabaseSessionManager()
    sid, uid = "sess-meta-1", "user-1"
    citations = [{"filename": "doc.md", "score": 0.91, "chunk_preview": "片段", "kb_id": "kb1"}]
    steps = [
        {"id": "task_understood", "title": "理解", "status": "done", "level": "success"},
        {"id": "tool_rag_summary_tools", "title": "检索", "status": "done", "level": "success", "detail": "已检索 2 个文档"},
        {"id": "answer_generated", "title": "生成", "status": "done", "level": "success"},
    ]
    await sm.add_message(sid, uid, "问题?", "回答。", citations=citations, steps=steps, send_report="已发送")

    async with sm_db() as db:
        rows = (await db.run_sync(
            lambda s: s.query(ChatMessage).filter(ChatMessage.session_id == sid).order_by(ChatMessage.created_at).all()
        ))
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[0].metadata_ is None
    assert rows[1].metadata_ == {"citations": citations, "steps": steps, "send_report": "已发送"}


@pytest.mark.asyncio
async def test_get_session_restores_messages_with_metadata(sm_db):
    sm = DatabaseSessionManager()
    sid, uid = "sess-meta-2", "user-2"
    citations = [{"filename": "a.txt", "score": 0.8, "chunk_preview": "x", "kb_id": None}]
    steps = [{"id": "tool_rag_summary_tools", "title": "检索", "status": "done", "level": "success"}]
    await sm.add_message(sid, uid, "Q1", "A1", citations=citations, steps=steps, send_report="已发送")
    await sm.add_message(sid, uid, "Q2", "A2", citations=[], steps=[])

    data = await sm.get_session(sid, uid)

    assert data["history"] == [("Q1", "A1"), ("Q2", "A2")]
    msgs = data["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert "citations" not in msgs[0] and "steps" not in msgs[0]
    assert msgs[1]["citations"] == citations
    assert msgs[1]["steps"] == steps
    assert msgs[1]["send_report"] == "已发送"
    assert msgs[3]["citations"] == [] and msgs[3]["steps"] == []
    assert msgs[3]["send_report"] is None


@pytest.mark.asyncio
async def test_get_session_handles_legacy_null_metadata(sm_db):
    """旧数据 metadata_ IS NULL 时,messages 字段应回落到空数组,不抛异常。"""
    sid, uid = "sess-legacy", "user-legacy"
    async with sm_db() as db:
        db.add(ChatSession(id=sid, user_id=uid, title="legacy"))
        await db.commit()
        db.add(ChatMessage(session_id=sid, role="user", content="老问题"))
        db.add(ChatMessage(session_id=sid, role="assistant", content="老回答"))
        await db.commit()

    sm = DatabaseSessionManager()
    data = await sm.get_session(sid, uid)
    assert data["history"] == [("老问题", "老回答")]
    msgs = data["messages"]
    assert msgs[0] == {"role": "user", "content": "老问题"}
    assert msgs[1] == {
        "role": "assistant",
        "content": "老回答",
        "citations": [],
        "steps": [],
        "send_report": None,
    }


@pytest.mark.asyncio
async def test_add_message_without_metadata_keeps_metadata_null(sm_db):
    """不传 citations / steps 时,assistant 消息的 metadata_ 应保持 NULL,与历史行为一致。"""
    sm = DatabaseSessionManager()
    sid, uid = "sess-no-meta", "user-3"
    await sm.add_message(sid, uid, "q", "a")

    async with sm_db() as db:
        rows = (await db.run_sync(
            lambda s: s.query(ChatMessage).filter(ChatMessage.session_id == sid).order_by(ChatMessage.created_at).all()
        ))
    assert all(r.metadata_ is None for r in rows)
